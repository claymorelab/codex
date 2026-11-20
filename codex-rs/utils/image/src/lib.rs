use std::io::Cursor;
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::Arc;
use std::sync::LazyLock;

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use codex_utils_cache::BlockingLruCache;
use codex_utils_cache::sha1_digest;
use image::ColorType;
use image::DynamicImage;
use image::GenericImageView;
use image::ImageDecoder;
use image::ImageEncoder;
use image::ImageFormat;
use image::ImageReader;
use image::codecs::jpeg::JpegEncoder;
use image::codecs::png::PngEncoder;
use image::codecs::webp::WebPEncoder;
use image::imageops::FilterType;
use image::metadata::Orientation;
use tracing::debug;
use tracing::warn;

const DATA_URL_PREFIX: &str = "data:";
pub const PROMPT_IMAGE_PATCH_SIZE: u32 = 32;
/// Maximum width or height used when resizing images before uploading.
pub const MAX_DIMENSION: u32 = 2048;
/// Maximum accepted byte length for prompt image input representations.
///
/// This is a high sanity guard against pathological inputs, not a protocol
/// requirement or target upload size.
pub const MAX_PROMPT_IMAGE_INPUT_BYTES: usize = 1024 * 1024 * 1024;
const MAX_IMAGE_CACHE_BYTES: usize = 64 * 1024 * 1024;

pub mod error;

pub use crate::error::ImageProcessingError;

#[derive(Debug, Clone)]
pub struct EncodedImage {
    pub bytes: Arc<[u8]>,
    pub mime: String,
    pub source_width: u32,
    pub source_height: u32,
    pub width: u32,
    pub height: u32,
}

impl EncodedImage {
    pub fn into_data_url(self) -> String {
        data_url_from_bytes(&self.mime, &self.bytes)
    }
}

/// Wraps image bytes in a data URL without decoding or validating them.
pub fn data_url_from_bytes(mime: &str, bytes: &[u8]) -> String {
    let encoded = BASE64_STANDARD.encode(bytes);
    format!("data:{mime};base64,{encoded}")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PromptImageMode {
    ResizeToFit,
    Original,
    ResizeWithLimits(PromptImageResizeLimits),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PromptImageTreatment {
    Message,
    ToolOutput,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PromptImageResizeLimits {
    pub max_dimension: u32,
    pub max_patches: usize,
}

impl PromptImageMode {
    /// Resize policy for high-detail prompt images.
    pub const HIGH_DETAIL: Self = Self::ResizeWithLimits(PromptImageResizeLimits {
        max_dimension: 2048,
        max_patches: 2_500,
    });
    /// Resize policy for original-detail prompt images.
    pub const ORIGINAL_DETAIL: Self = Self::ResizeWithLimits(PromptImageResizeLimits {
        max_dimension: 6000,
        max_patches: 10_000,
    });
}

#[derive(Clone)]
struct ImageMetadata {
    icc_profile: Option<Vec<u8>>,
    exif: Option<Vec<u8>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ImageCacheKey {
    digest: [u8; 20],
    mode: PromptImageMode,
    treatment: PromptImageTreatment,
}

type ImageCache = BlockingLruCache<ImageCacheKey, EncodedImage>;

static IMAGE_CACHE: LazyLock<ImageCache> =
    LazyLock::new(|| BlockingLruCache::new(NonZeroUsize::new(32).unwrap_or(NonZeroUsize::MIN)));

pub fn load_for_prompt_bytes(
    path: &Path,
    file_bytes: Vec<u8>,
    mode: PromptImageMode,
) -> Result<EncodedImage, ImageProcessingError> {
    load_for_prompt_bytes_with_treatment(path, file_bytes, mode, PromptImageTreatment::Message)
}

fn load_for_prompt_bytes_with_treatment(
    path: &Path,
    file_bytes: Vec<u8>,
    mode: PromptImageMode,
    treatment: PromptImageTreatment,
) -> Result<EncodedImage, ImageProcessingError> {
    let path_buf = path.to_path_buf();

    let key = ImageCacheKey {
        digest: sha1_digest(&file_bytes),
        mode,
        treatment,
    };

    if let Some(image) = IMAGE_CACHE.get(&key) {
        return Ok(image);
    }

    let image = load_for_prompt_bytes_uncached(&path_buf, file_bytes, mode, treatment)?;
    cache_image(&IMAGE_CACHE, key, image.clone(), MAX_IMAGE_CACHE_BYTES);
    Ok(image)
}

fn load_for_prompt_bytes_uncached(
    path: &Path,
    file_bytes: Vec<u8>,
    mode: PromptImageMode,
    treatment: PromptImageTreatment,
) -> Result<EncodedImage, ImageProcessingError> {
    let path_buf = path.to_path_buf();
    (move || {
        let guessed_format = image::guess_format(&file_bytes)
            .map_err(|source| ImageProcessingError::decode_error(&path_buf, source))?;
        let format = match guessed_format {
            ImageFormat::Png => Some(ImageFormat::Png),
            ImageFormat::Jpeg => Some(ImageFormat::Jpeg),
            ImageFormat::Gif => Some(ImageFormat::Gif),
            ImageFormat::WebP => Some(ImageFormat::WebP),
            _ => None,
        };

        let mut decoder = ImageReader::with_format(Cursor::new(&file_bytes), guessed_format)
            .into_decoder()
            .map_err(|source| ImageProcessingError::decode_error(&path_buf, source))?;
        // Preserve the metadata most important for rendering prompt images faithfully: the color
        // profile and EXIF data, including orientation. Other format-specific metadata is
        // intentionally not copied.
        let metadata = ImageMetadata {
            // Only RGB profiles are safe across every re-encoding path. For example, JPEG decoding
            // can convert CMYK/YCCK pixels to RGB while retaining the source profile; copying it
            // would mislabel the output. Bytes 16..20 are the ICC data color space signature.
            icc_profile: decoder
                .icc_profile()
                .ok()
                .flatten()
                .filter(|profile| profile.get(16..20) == Some(b"RGB ")),
            exif: decoder.exif_metadata().ok().flatten(),
        };
        let dynamic = DynamicImage::from_decoder(decoder)
            .map_err(|source| ImageProcessingError::decode_error(&path_buf, source))?;

        let (width, height) = dynamic.dimensions();

        let target_dimensions = match mode {
            PromptImageMode::ResizeToFit if width > MAX_DIMENSION || height > MAX_DIMENSION => {
                let resized = dynamic.resize(MAX_DIMENSION, MAX_DIMENSION, FilterType::Triangle);
                Some((resized.width(), resized.height(), resized))
            }
            PromptImageMode::ResizeWithLimits(limits) => {
                let (target_width, target_height) =
                    prompt_image_output_dimensions_for_limits(width, height, limits);
                if (target_width, target_height) == (width, height) {
                    None
                } else {
                    let resized =
                        dynamic.resize_exact(target_width, target_height, FilterType::Triangle);
                    Some((target_width, target_height, resized))
                }
            }
            PromptImageMode::ResizeToFit | PromptImageMode::Original => None,
        };

        let cap_metadata = metadata.clone();
        let encoded = if let Some((prepared_width, prepared_height, resized)) = &target_dimensions {
            let target_format = format
                .filter(|format| can_preserve_source_bytes(*format))
                .unwrap_or(ImageFormat::Png);
            let (bytes, output_format) = encode_image(resized, target_format, metadata)?;
            let mime = format_to_mime(output_format);
            EncodedImage {
                bytes: bytes.into(),
                mime,
                source_width: width,
                source_height: height,
                width: *prepared_width,
                height: *prepared_height,
            }
        } else {
            if let Some(format) = format.filter(|format| can_preserve_source_bytes(*format)) {
                let mime = format_to_mime(format);
                EncodedImage {
                    bytes: file_bytes.into(),
                    mime,
                    source_width: width,
                    source_height: height,
                    width,
                    height,
                }
            } else {
                let (bytes, output_format) = encode_image(&dynamic, ImageFormat::Png, metadata)?;
                let mime = format_to_mime(output_format);
                EncodedImage {
                    bytes: bytes.into(),
                    mime,
                    source_width: width,
                    source_height: height,
                    width,
                    height,
                }
            }
        };

        match treatment {
            PromptImageTreatment::Message => {
                cap_like_original(encoded, &dynamic, format, mode, cap_metadata)
            }
            PromptImageTreatment::ToolOutput => cap_tool_output_bytes(
                encoded,
                target_dimensions
                    .as_ref()
                    .map_or(&dynamic, |(_, _, resized)| resized),
                cap_metadata,
            ),
        }
    })()
}

/// The original Grok Bot sends no image above 1 MB, nor above 1024 px on its long side
/// (1280 for WebP): its image-resizing middleware shrinks by 0.8 until both hold. Codex
/// keeps high-detail images up to 2048 px and any size, so screenshot-heavy threads grow
/// by megabytes per image. Apply the original's caps to message images in the
/// high-detail and resize-to-fit modes.
const ORIGINAL_MAX_IMAGE_DIMENSION: u32 = 1024;
const ORIGINAL_MAX_WEBP_IMAGE_DIMENSION: u32 = 1280;
const ORIGINAL_MAX_IMAGE_BYTES: usize = 1024 * 1024;

fn cap_like_original(
    encoded: EncodedImage,
    source: &DynamicImage,
    format: Option<ImageFormat>,
    mode: PromptImageMode,
    metadata: ImageMetadata,
) -> Result<EncodedImage, ImageProcessingError> {
    if !matches!(mode, PromptImageMode::ResizeToFit) && mode != PromptImageMode::HIGH_DETAIL {
        return Ok(encoded);
    }
    let max_dimension = if format == Some(ImageFormat::WebP) {
        ORIGINAL_MAX_WEBP_IMAGE_DIMENSION
    } else {
        ORIGINAL_MAX_IMAGE_DIMENSION
    };
    if encoded.width.max(encoded.height) <= max_dimension
        && encoded.bytes.len() <= ORIGINAL_MAX_IMAGE_BYTES
    {
        return Ok(encoded);
    }
    let scale = (f64::from(max_dimension) / f64::from(encoded.width.max(encoded.height))).min(1.0);
    let mut width = ((f64::from(encoded.width) * scale).round() as u32).max(1);
    let mut height = ((f64::from(encoded.height) * scale).round() as u32).max(1);
    let target_format = format
        .filter(|format| can_preserve_source_bytes(*format))
        .unwrap_or(ImageFormat::Png);
    loop {
        let resized = source.resize_exact(width, height, FilterType::Triangle);
        let (bytes, output_format) = encode_image(&resized, target_format, metadata.clone())?;
        if bytes.len() <= ORIGINAL_MAX_IMAGE_BYTES || width <= 1 || height <= 1 {
            return Ok(EncodedImage {
                bytes: bytes.into(),
                mime: format_to_mime(output_format),
                source_width: encoded.source_width,
                source_height: encoded.source_height,
                width,
                height,
            });
        }
        width = ((f64::from(width) * 0.8).round() as u32).max(1);
        height = ((f64::from(height) * 0.8).round() as u32).max(1);
    }
}

fn cap_tool_output_bytes(
    mut encoded: EncodedImage,
    image: &DynamicImage,
    mut metadata: ImageMetadata,
) -> Result<EncodedImage, ImageProcessingError> {
    if encoded.bytes.len() <= ORIGINAL_MAX_IMAGE_BYTES {
        return Ok(encoded);
    }

    // JPEG stores EXIF in one APP1 segment. Retain the orientation if the full EXIF won't fit.
    const JPEG_MAX_EXIF_BYTES: usize = u16::MAX as usize - 2 - 6;
    if let Some(exif) = &metadata.exif
        && exif.len() > JPEG_MAX_EXIF_BYTES
    {
        metadata.exif = Orientation::from_exif_chunk(exif).map(|orientation| {
            // Little-endian TIFF with one IFD0 entry (Orientation, SHORT, count 1).
            let mut tiff = *b"II\x2a\0\x08\0\0\0\
                             \x01\0\x12\x01\x03\0\x01\0\0\0\0\0\0\0\
                             \0\0\0\0";
            tiff[18] = orientation.to_exif();
            tiff.to_vec()
        });
    }
    // JPEG ICC profiles can span up to 255 APP2 segments.
    const JPEG_MAX_ICC_BYTES: usize = 255 * (u16::MAX as usize - 2 - 14);
    metadata.icc_profile = metadata
        .icc_profile
        .filter(|profile| profile.len() <= JPEG_MAX_ICC_BYTES);

    let mut rgb = image.to_rgb8();
    if image.color().has_alpha() {
        for (pixel, rgba) in rgb.pixels_mut().zip(image.to_rgba8().pixels()) {
            let alpha = u16::from(rgba[3]);
            for channel in &mut pixel.0 {
                *channel = ((u16::from(*channel) * alpha + 255 * (255 - alpha) + 127) / 255) as u8;
            }
        }
    }

    let mut bytes = Vec::new();
    let mut quality = 85;
    for step in [85, 70, 50] {
        quality = step;
        bytes.clear();
        let mut encoder = JpegEncoder::new_with_quality(&mut bytes, quality);
        apply_image_metadata(
            &mut encoder,
            metadata.icc_profile.clone(),
            metadata.exif.clone(),
            ImageFormat::Jpeg,
        )?;
        encoder
            .encode_image(&rgb)
            .map_err(|source| ImageProcessingError::Encode {
                format: ImageFormat::Jpeg,
                source,
            })?;
        if bytes.len() <= ORIGINAL_MAX_IMAGE_BYTES {
            break;
        }
    }

    let (width, height) = (encoded.width, encoded.height);
    let size_bytes = bytes.len();
    if size_bytes > ORIGINAL_MAX_IMAGE_BYTES {
        warn!(
            width,
            height,
            quality,
            size_bytes,
            "tool output image still exceeds 1 MiB after JPEG re-encoding"
        );
    } else {
        debug!(
            width,
            height, quality, size_bytes, "re-encoded tool output image as JPEG"
        );
    }
    encoded.bytes = bytes.into();
    encoded.mime = format_to_mime(ImageFormat::Jpeg);
    Ok(encoded)
}

fn cache_image(cache: &ImageCache, key: ImageCacheKey, image: EncodedImage, byte_capacity: usize) {
    if image.bytes.len() > byte_capacity {
        return;
    }

    cache.with_mut(|cache| {
        cache.put(key, image);
        let mut cached_bytes = cache
            .iter()
            .map(|(_, image)| image.bytes.len())
            .sum::<usize>();
        while cached_bytes > byte_capacity {
            let Some((_, evicted)) = cache.pop_lru() else {
                break;
            };
            cached_bytes -= evicted.bytes.len();
        }
    });
}

pub fn load_data_url_for_prompt(
    image_url: &str,
    mode: PromptImageMode,
    treatment: PromptImageTreatment,
) -> Result<EncodedImage, ImageProcessingError> {
    load_data_url_for_prompt_with(image_url, |path, bytes| {
        load_for_prompt_bytes_with_treatment(path, bytes, mode, treatment)
    })
}

pub fn load_data_url_for_prompt_uncached(
    image_url: &str,
    mode: PromptImageMode,
) -> Result<EncodedImage, ImageProcessingError> {
    load_data_url_for_prompt_with(image_url, |path, bytes| {
        load_for_prompt_bytes_uncached(path, bytes, mode, PromptImageTreatment::Message)
    })
}

fn load_data_url_for_prompt_with(
    image_url: &str,
    load: impl FnOnce(&Path, Vec<u8>) -> Result<EncodedImage, ImageProcessingError>,
) -> Result<EncodedImage, ImageProcessingError> {
    let rest = image_url
        .get(..DATA_URL_PREFIX.len())
        .filter(|prefix| prefix.eq_ignore_ascii_case(DATA_URL_PREFIX))
        .and_then(|_| image_url.get(DATA_URL_PREFIX.len()..))
        .ok_or_else(|| ImageProcessingError::InvalidDataUrl {
            reason: "missing data: prefix".to_string(),
        })?;
    let (metadata, encoded) =
        rest.split_once(',')
            .ok_or_else(|| ImageProcessingError::InvalidDataUrl {
                reason: "missing comma separator".to_string(),
            })?;
    if !metadata
        .split(';')
        .any(|part| part.eq_ignore_ascii_case("base64"))
    {
        return Err(ImageProcessingError::InvalidDataUrl {
            reason: "only base64 data URLs are supported".to_string(),
        });
    }

    if encoded.len() > MAX_PROMPT_IMAGE_INPUT_BYTES {
        return Err(ImageProcessingError::ImageTooLarge {
            representation: "base64 payload",
            size: encoded.len(),
            max: MAX_PROMPT_IMAGE_INPUT_BYTES,
        });
    }
    let file_bytes =
        BASE64_STANDARD
            .decode(encoded)
            .map_err(|source| ImageProcessingError::InvalidDataUrl {
                reason: format!("invalid base64 payload: {source}"),
            })?;
    if file_bytes.len() > MAX_PROMPT_IMAGE_INPUT_BYTES {
        return Err(ImageProcessingError::ImageTooLarge {
            representation: "decoded input",
            size: file_bytes.len(),
            max: MAX_PROMPT_IMAGE_INPUT_BYTES,
        });
    }

    load(Path::new("<data-url-image>"), file_bytes)
}

fn prompt_image_output_dimensions_for_limits(
    width: u32,
    height: u32,
    limits: PromptImageResizeLimits,
) -> (u32, u32) {
    let width = width.max(1);
    let height = height.max(1);
    if prompt_image_dimensions_fit(width, height, limits) {
        return (width, height);
    }

    let max_dimension_scale =
        (f64::from(limits.max_dimension) / f64::from(width.max(height))).min(1.0);
    let width = ((f64::from(width) * max_dimension_scale).round() as u32).max(1);
    let height = ((f64::from(height) * max_dimension_scale).round() as u32).max(1);
    if prompt_image_dimensions_fit(width, height, limits) {
        return (width, height);
    }

    let width_f64 = f64::from(width);
    let height_f64 = f64::from(height);
    let patch_size = f64::from(PROMPT_IMAGE_PATCH_SIZE);
    let mut scale =
        (patch_size * patch_size * limits.max_patches as f64 / width_f64 / height_f64).sqrt();
    // Match Responses patch-budget math: shrink by area, then round the scaled
    // patch grid down so integer output dimensions remain within the budget.
    let scaled_patches_wide = width_f64 * scale / patch_size;
    let scaled_patches_high = height_f64 * scale / patch_size;
    scale *= (scaled_patches_wide.floor() / scaled_patches_wide)
        .min(scaled_patches_high.floor() / scaled_patches_high);

    (
        ((width_f64 * scale).floor() as u32).max(1),
        ((height_f64 * scale).floor() as u32).max(1),
    )
}

fn prompt_image_dimensions_fit(width: u32, height: u32, limits: PromptImageResizeLimits) -> bool {
    let patches_wide = width.div_ceil(PROMPT_IMAGE_PATCH_SIZE);
    let patches_high = height.div_ceil(PROMPT_IMAGE_PATCH_SIZE);
    let patch_count = u64::from(patches_wide) * u64::from(patches_high);
    width <= limits.max_dimension
        && height <= limits.max_dimension
        && patch_count <= limits.max_patches as u64
}

fn can_preserve_source_bytes(format: ImageFormat) -> bool {
    // Public API docs explicitly call out non-animated GIF support only.
    // Preserve byte-for-byte only for formats we can safely pass through.
    matches!(
        format,
        ImageFormat::Png | ImageFormat::Jpeg | ImageFormat::WebP
    )
}

fn encode_image(
    image: &DynamicImage,
    preferred_format: ImageFormat,
    metadata: ImageMetadata,
) -> Result<(Vec<u8>, ImageFormat), ImageProcessingError> {
    let target_format = match preferred_format {
        ImageFormat::Jpeg => ImageFormat::Jpeg,
        ImageFormat::WebP => ImageFormat::WebP,
        _ => ImageFormat::Png,
    };

    let mut buffer = Vec::new();
    let ImageMetadata { icc_profile, exif } = metadata;

    match target_format {
        ImageFormat::Png => {
            let rgba = image.to_rgba8();
            let mut encoder = PngEncoder::new(&mut buffer);
            apply_image_metadata(&mut encoder, icc_profile, exif, target_format)?;
            encoder
                .write_image(
                    rgba.as_raw(),
                    image.width(),
                    image.height(),
                    ColorType::Rgba8.into(),
                )
                .map_err(|source| ImageProcessingError::Encode {
                    format: target_format,
                    source,
                })?;
        }
        ImageFormat::Jpeg => {
            let mut encoder = JpegEncoder::new_with_quality(&mut buffer, 85);
            apply_image_metadata(&mut encoder, icc_profile, exif, target_format)?;
            encoder
                .encode_image(image)
                .map_err(|source| ImageProcessingError::Encode {
                    format: target_format,
                    source,
                })?;
        }
        ImageFormat::WebP => {
            let rgba = image.to_rgba8();
            let mut encoder = WebPEncoder::new_lossless(&mut buffer);
            apply_image_metadata(&mut encoder, icc_profile, exif, target_format)?;
            encoder
                .write_image(
                    rgba.as_raw(),
                    image.width(),
                    image.height(),
                    ColorType::Rgba8.into(),
                )
                .map_err(|source| ImageProcessingError::Encode {
                    format: target_format,
                    source,
                })?;
        }
        _ => unreachable!("unsupported target_format should have been handled earlier"),
    }

    Ok((buffer, target_format))
}

fn apply_image_metadata(
    encoder: &mut impl ImageEncoder,
    icc_profile: Option<Vec<u8>>,
    exif: Option<Vec<u8>>,
    format: ImageFormat,
) -> Result<(), ImageProcessingError> {
    if let Some(icc_profile) = icc_profile {
        encoder
            .set_icc_profile(icc_profile)
            .map_err(|source| ImageProcessingError::Encode {
                format,
                source: image::ImageError::Unsupported(source),
            })?;
    }
    if let Some(exif) = exif {
        encoder
            .set_exif_metadata(exif)
            .map_err(|source| ImageProcessingError::Encode {
                format,
                source: image::ImageError::Unsupported(source),
            })?;
    }
    Ok(())
}

fn format_to_mime(format: ImageFormat) -> String {
    match format {
        ImageFormat::Jpeg => "image/jpeg".to_string(),
        ImageFormat::Gif => "image/gif".to_string(),
        ImageFormat::WebP => "image/webp".to_string(),
        _ => "image/png".to_string(),
    }
}

#[cfg(test)]
#[path = "image_tests.rs"]
mod tests;
