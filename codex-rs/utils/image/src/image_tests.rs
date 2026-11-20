use std::io::Cursor;

use super::*;
use image::GenericImageView;
use image::ImageBuffer;
use image::ImageDecoder;
use image::Rgba;
use image::metadata::Orientation;

const TEST_RGB_ICC_PROFILE: &[u8] = b"0123456789abcdefRGB ";
const TEST_CMYK_ICC_PROFILE: &[u8] = b"0123456789abcdefCMYK";
const ROTATE_90_EXIF: &[u8] = &[
    0x49, 0x49, 0x2a, 0x00, 0x08, 0x00, 0x00, 0x00, 0x01, 0x00, 0x12, 0x01, 0x03, 0x00, 0x01, 0x00,
    0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

fn image_bytes(image: &ImageBuffer<Rgba<u8>, Vec<u8>>, format: ImageFormat) -> Vec<u8> {
    let mut encoded = Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(image.clone())
        .write_to(&mut encoded, format)
        .expect("encode image to bytes");
    encoded.into_inner()
}

fn image_bytes_with_metadata(
    image: &ImageBuffer<Rgba<u8>, Vec<u8>>,
    format: ImageFormat,
    icc_profile: &[u8],
) -> Vec<u8> {
    let mut encoded = Vec::new();
    match format {
        ImageFormat::Png => {
            let mut encoder = PngEncoder::new(&mut encoded);
            encoder
                .set_icc_profile(icc_profile.to_vec())
                .expect("set PNG ICC profile");
            encoder
                .set_exif_metadata(ROTATE_90_EXIF.to_vec())
                .expect("set PNG EXIF metadata");
            encoder
                .write_image(
                    image.as_raw(),
                    image.width(),
                    image.height(),
                    ColorType::Rgba8.into(),
                )
                .expect("encode PNG with metadata");
        }
        ImageFormat::Jpeg => {
            let mut encoder = JpegEncoder::new_with_quality(&mut encoded, 90);
            encoder
                .set_icc_profile(icc_profile.to_vec())
                .expect("set JPEG ICC profile");
            encoder
                .set_exif_metadata(ROTATE_90_EXIF.to_vec())
                .expect("set JPEG EXIF metadata");
            encoder
                .encode_image(&DynamicImage::ImageRgba8(image.clone()))
                .expect("encode JPEG with metadata");
        }
        ImageFormat::WebP => {
            let mut encoder = WebPEncoder::new_lossless(&mut encoded);
            encoder
                .set_icc_profile(icc_profile.to_vec())
                .expect("set WebP ICC profile");
            encoder
                .set_exif_metadata(ROTATE_90_EXIF.to_vec())
                .expect("set WebP EXIF metadata");
            encoder
                .write_image(
                    image.as_raw(),
                    image.width(),
                    image.height(),
                    ColorType::Rgba8.into(),
                )
                .expect("encode WebP with metadata");
        }
        _ => panic!("unsupported test format"),
    }
    encoded
}

#[tokio::test(flavor = "multi_thread")]
async fn returns_original_image_when_within_bounds() {
    for (format, mime) in [
        (ImageFormat::Png, "image/png"),
        (ImageFormat::WebP, "image/webp"),
    ] {
        let image = ImageBuffer::from_pixel(64, 32, Rgba([10u8, 20, 30, 255]));
        let original_bytes = image_bytes(&image, format);

        let encoded = load_for_prompt_bytes(
            Path::new("in-memory-image"),
            original_bytes.clone(),
            PromptImageMode::ResizeToFit,
        )
        .expect("process image");

        assert_eq!(encoded.width, 64);
        assert_eq!(encoded.height, 32);
        assert_eq!(encoded.mime, mime);
        assert_eq!(encoded.bytes.as_ref(), original_bytes);
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn downscales_large_image() {
    for (format, mime) in [
        (ImageFormat::Png, "image/png"),
        (ImageFormat::WebP, "image/webp"),
    ] {
        let image = ImageBuffer::from_pixel(4096, 2048, Rgba([200u8, 10, 10, 255]));
        let original_bytes = image_bytes(&image, format);

        let processed = load_for_prompt_bytes(
            Path::new("in-memory-image"),
            original_bytes,
            PromptImageMode::ResizeToFit,
        )
        .expect("process image");

        assert!(processed.width <= MAX_DIMENSION);
        assert!(processed.height <= MAX_DIMENSION);
        assert_eq!(processed.mime, mime);

        let detected_format =
            image::guess_format(&processed.bytes).expect("detect resized output format");
        assert_eq!(detected_format, format);

        let loaded =
            image::load_from_memory(&processed.bytes).expect("read resized bytes back into image");
        assert_eq!(loaded.dimensions(), (processed.width, processed.height));
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn downscales_tall_image_to_fit_square_bounds() {
    let image = ImageBuffer::from_pixel(1024, 4096, Rgba([200u8, 10, 10, 255]));
    let original_bytes = image_bytes(&image, ImageFormat::Png);

    let processed = load_for_prompt_bytes(
        Path::new("in-memory-image"),
        original_bytes,
        PromptImageMode::ResizeToFit,
    )
    .expect("process image");

    // The original Grok Bot's 1024 px cap applies after Codex's own resize.
    assert_eq!(processed.width, 256);
    assert_eq!(processed.height, 1024);
    assert_eq!(processed.mime, "image/png");
}

#[tokio::test(flavor = "multi_thread")]
async fn resizing_preserves_supported_metadata() {
    for format in [ImageFormat::Png, ImageFormat::Jpeg, ImageFormat::WebP] {
        let image = ImageBuffer::from_pixel(2050, 2, Rgba([200u8, 10, 10, 255]));
        let original_bytes = image_bytes_with_metadata(&image, format, TEST_RGB_ICC_PROFILE);

        let processed = load_for_prompt_bytes(
            Path::new("in-memory-image"),
            original_bytes,
            PromptImageMode::ResizeToFit,
        )
        .expect("process image");

        // The original Grok Bot caps the long side at 1024 px, or 1280 px for WebP.
        let expected = if format == ImageFormat::WebP {
            (1280, 1)
        } else {
            (1024, 1)
        };
        assert_eq!((processed.width, processed.height), expected);

        let mut decoder = ImageReader::with_format(Cursor::new(&processed.bytes), format)
            .into_decoder()
            .expect("create decoder");
        assert_eq!(
            (
                decoder.dimensions(),
                decoder.orientation().expect("read orientation"),
                decoder.icc_profile().expect("read ICC profile"),
                decoder.exif_metadata().expect("read EXIF metadata"),
            ),
            (
                expected,
                Orientation::Rotate90,
                Some(TEST_RGB_ICC_PROFILE.to_vec()),
                Some(ROTATE_90_EXIF.to_vec()),
            )
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn resizing_drops_non_rgb_icc_profile() {
    let image = ImageBuffer::from_pixel(2050, 2, Rgba([200u8, 10, 10, 255]));
    let original_bytes =
        image_bytes_with_metadata(&image, ImageFormat::Jpeg, TEST_CMYK_ICC_PROFILE);

    let processed = load_for_prompt_bytes(
        Path::new("in-memory-image"),
        original_bytes,
        PromptImageMode::ResizeToFit,
    )
    .expect("process image");

    let mut decoder = ImageReader::with_format(Cursor::new(&processed.bytes), ImageFormat::Jpeg)
        .into_decoder()
        .expect("create decoder");
    assert_eq!(
        (
            decoder.icc_profile().expect("read ICC profile"),
            decoder.exif_metadata().expect("read EXIF metadata"),
        ),
        (None, Some(ROTATE_90_EXIF.to_vec()))
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn preserves_large_image_in_original_mode() {
    let image = ImageBuffer::from_pixel(4096, 2048, Rgba([180u8, 30, 30, 255]));
    let original_bytes = image_bytes(&image, ImageFormat::Png);

    let processed = load_for_prompt_bytes(
        Path::new("in-memory-image"),
        original_bytes.clone(),
        PromptImageMode::Original,
    )
    .expect("process image");

    assert_eq!(processed.width, 4096);
    assert_eq!(processed.height, 2048);
    assert_eq!(processed.mime, "image/png");
    assert_eq!(processed.bytes.as_ref(), original_bytes);
}

#[tokio::test(flavor = "current_thread")]
async fn data_url_processing_preserves_supported_source_bytes() {
    let image = ImageBuffer::from_pixel(64, 32, Rgba([10u8, 20, 30, 255]));
    let original_bytes = image_bytes(&image, ImageFormat::Png);
    let image_url = data_url_from_bytes("image/png", &original_bytes)
        .replacen("data:", "DATA:", 1)
        .replacen(";base64,", ";BASE64,", 1);

    let processed = load_data_url_for_prompt_uncached(&image_url, PromptImageMode::ResizeToFit)
        .expect("process data URL image");

    assert_eq!(processed.width, 64);
    assert_eq!(processed.height, 32);
    assert_eq!(processed.mime, "image/png");
    assert_eq!(processed.bytes.as_ref(), original_bytes);
}

#[tokio::test(flavor = "multi_thread")]
async fn data_url_processing_converts_gif_to_png() {
    let image = ImageBuffer::from_pixel(64, 32, Rgba([10u8, 20, 30, 255]));
    let gif_bytes = image_bytes(&image, ImageFormat::Gif);
    let image_url = data_url_from_bytes("image/gif", &gif_bytes);

    let processed = load_data_url_for_prompt(
        &image_url,
        PromptImageMode::ResizeToFit,
        PromptImageTreatment::Message,
    )
    .expect("process GIF data URL");

    assert_eq!(processed.mime, "image/png");
    assert_eq!(
        image::guess_format(&processed.bytes).expect("detect processed format"),
        ImageFormat::Png
    );
}

#[test]
fn data_url_processing_rejects_malformed_input() {
    for image_url in [
        "image/png;base64,AAAA",
        "data:image/png;base64",
        "data:image/png,AAAA",
        "data:image/png;base64,not base64",
    ] {
        assert!(matches!(
            load_data_url_for_prompt(
                image_url,
                PromptImageMode::ResizeToFit,
                PromptImageTreatment::Message,
            ),
            Err(ImageProcessingError::InvalidDataUrl { .. })
        ));
    }
}

/// The shared detail modes apply the policies used by external byte-oriented callers.
#[tokio::test(flavor = "multi_thread")]
async fn detail_modes_apply_expected_budgets() {
    for (mode, input_dimensions, expected_dimensions) in [
        (PromptImageMode::HIGH_DETAIL, (2048, 2048), (1024, 1024)),
        (PromptImageMode::ORIGINAL_DETAIL, (6401, 100), (6000, 94)),
    ] {
        let image = ImageBuffer::from_pixel(
            input_dimensions.0,
            input_dimensions.1,
            Rgba([200u8, 10, 10, 255]),
        );
        let original_bytes = image_bytes(&image, ImageFormat::Png);
        let processed = load_for_prompt_bytes(Path::new("in-memory-image"), original_bytes, mode)
            .expect("process image with detail mode");

        assert_eq!(
            (
                processed.source_width,
                processed.source_height,
                processed.width,
                processed.height,
            ),
            (
                input_dimensions.0,
                input_dimensions.1,
                expected_dimensions.0,
                expected_dimensions.1,
            )
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn fails_cleanly_for_invalid_images() {
    let err = load_for_prompt_bytes(
        Path::new("in-memory-image"),
        b"not an image".to_vec(),
        PromptImageMode::ResizeToFit,
    )
    .expect_err("invalid image should fail");
    assert!(matches!(
        err,
        ImageProcessingError::Decode { .. } | ImageProcessingError::UnsupportedImageFormat { .. }
    ));
}

#[tokio::test(flavor = "multi_thread")]
async fn reprocesses_updated_file_contents() {
    IMAGE_CACHE.clear();

    let first_image = ImageBuffer::from_pixel(32, 16, Rgba([20u8, 120, 220, 255]));
    let first_bytes = image_bytes(&first_image, ImageFormat::Png);

    let first = load_for_prompt_bytes(
        Path::new("in-memory-image"),
        first_bytes,
        PromptImageMode::ResizeToFit,
    )
    .expect("process first image");

    let second_image = ImageBuffer::from_pixel(96, 48, Rgba([50u8, 60, 70, 255]));
    let second_bytes = image_bytes(&second_image, ImageFormat::Png);

    let second = load_for_prompt_bytes(
        Path::new("in-memory-image"),
        second_bytes,
        PromptImageMode::ResizeToFit,
    )
    .expect("process updated image");

    assert_eq!(first.width, 32);
    assert_eq!(first.height, 16);
    assert_eq!(second.width, 96);
    assert_eq!(second.height, 48);
    assert_ne!(second.bytes, first.bytes);
}

#[tokio::test(flavor = "multi_thread")]
async fn bounds_cache_by_encoded_byte_size() {
    let cache = ImageCache::new(NonZeroUsize::new(4).expect("non-zero cache capacity"));
    let key = |digest_byte| ImageCacheKey {
        digest: [digest_byte; 20],
        mode: PromptImageMode::Original,
        treatment: PromptImageTreatment::Message,
    };
    let image = |size| EncodedImage {
        bytes: vec![0; size].into(),
        mime: "image/png".to_string(),
        source_width: 1,
        source_height: 1,
        width: 1,
        height: 1,
    };

    cache_image(&cache, key(1), image(3), /*byte_capacity*/ 5);
    cache_image(&cache, key(2), image(3), /*byte_capacity*/ 5);
    cache_image(&cache, key(3), image(6), /*byte_capacity*/ 5);

    assert!(cache.get(&key(1)).is_none());
    assert!(cache.get(&key(2)).is_some());
    assert!(cache.get(&key(3)).is_none());
}

fn noise_image(width: u32, height: u32) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
    let mut state = 0x2545_f491_u32;
    ImageBuffer::from_fn(width, height, |_, _| {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let [r, g, b, _] = state.to_le_bytes();
        Rgba([r, g, b, 255])
    })
}

/// Like the original Grok Bot, a message image above 1 MiB shrinks by 0.8 until it fits.
#[tokio::test(flavor = "multi_thread")]
async fn high_detail_images_stay_within_one_megabyte() {
    let image = noise_image(1000, 1000);
    let original_bytes = image_bytes(&image, ImageFormat::Png);
    assert!(original_bytes.len() > 1024 * 1024);

    let processed = load_for_prompt_bytes(
        Path::new("in-memory-image"),
        original_bytes,
        PromptImageMode::HIGH_DETAIL,
    )
    .expect("process image");

    assert!(processed.bytes.len() <= 1024 * 1024);
    assert!(processed.width < 1000 && processed.width == processed.height);
    assert_eq!(processed.mime, "image/png");
}

#[test]
fn tool_output_and_message_images_use_separate_cache_entries() {
    use PromptImageTreatment::Message;
    use PromptImageTreatment::ToolOutput;

    for (red, order) in [(10, [Message, ToolOutput]), (20, [ToolOutput, Message])] {
        let image = ImageBuffer::from_pixel(1430, 805, Rgba([red, 20, 30, 255]));
        let original_bytes = image_bytes(&image, ImageFormat::Png);
        assert!(original_bytes.len() <= ORIGINAL_MAX_IMAGE_BYTES);
        let url = data_url_from_bytes("image/png", &original_bytes);

        for treatment in order.into_iter().chain(order) {
            let processed = load_data_url_for_prompt(&url, PromptImageMode::HIGH_DETAIL, treatment)
                .expect("process PNG");
            let decoded = image::load_from_memory(&processed.bytes).expect("decode processed PNG");
            assert_eq!((processed.width, processed.height), decoded.dimensions());
            assert_eq!(processed.mime, "image/png");
            match treatment {
                Message => assert_eq!(decoded.dimensions(), (1024, 576)),
                ToolOutput => {
                    assert_eq!(decoded.dimensions(), (1430, 805));
                    assert_eq!(processed.bytes.as_ref(), original_bytes);
                }
            }
        }
    }
}

#[tracing_test::traced_test]
#[test]
fn large_tool_output_png_uses_first_jpeg_quality_that_fits() {
    let image = noise_image(1430, 805);
    let original_bytes = image_bytes(&image, ImageFormat::Png);
    assert!(original_bytes.len() > ORIGINAL_MAX_IMAGE_BYTES);
    let url = data_url_from_bytes("image/png", &original_bytes);

    let processed = load_data_url_for_prompt(
        &url,
        PromptImageMode::HIGH_DETAIL,
        PromptImageTreatment::ToolOutput,
    )
    .expect("process tool image");
    assert_eq!(processed.mime, "image/jpeg");
    assert_eq!(
        image::guess_format(&processed.bytes).unwrap(),
        ImageFormat::Jpeg
    );
    assert_eq!(
        image::load_from_memory(&processed.bytes)
            .unwrap()
            .dimensions(),
        (1430, 805)
    );
    assert!(processed.bytes.len() <= ORIGINAL_MAX_IMAGE_BYTES);

    let rgb = DynamicImage::ImageRgba8(image).into_rgb8();
    for quality in [85, 70, 50] {
        let mut candidate = Vec::new();
        JpegEncoder::new_with_quality(&mut candidate, quality)
            .encode_image(&rgb)
            .expect("encode reference JPEG");
        if candidate.len() <= ORIGINAL_MAX_IMAGE_BYTES {
            assert!(quality < 85, "fixture must exercise JPEG quality fallback");
            assert_eq!(processed.bytes.as_ref(), candidate);
            assert!(logs_contain(&format!("quality={quality}")));
            return;
        }
    }
    panic!("fixture must fit at one of the JPEG quality steps");
}

#[test]
fn large_tool_output_png_flattens_alpha_onto_white() {
    let mut image = noise_image(1000, 1000);
    for (x, _, pixel) in image.enumerate_pixels_mut() {
        if x < 200 {
            *pixel = Rgba([255, 0, 0, 0]);
        } else if x < 400 {
            *pixel = Rgba([0, 0, 0, 128]);
        }
    }
    let mut exif = ROTATE_90_EXIF.to_vec();
    exif.resize(66_000, 0);
    let mut original_bytes = Vec::new();
    let mut encoder = PngEncoder::new(&mut original_bytes);
    encoder.set_exif_metadata(exif).expect("set long PNG EXIF");
    encoder
        .write_image(image.as_raw(), 1000, 1000, ColorType::Rgba8.into())
        .expect("encode transparent PNG");
    assert!(original_bytes.len() > ORIGINAL_MAX_IMAGE_BYTES);
    let url = data_url_from_bytes("image/png", &original_bytes);

    let processed = load_data_url_for_prompt(
        &url,
        PromptImageMode::ORIGINAL_DETAIL,
        PromptImageTreatment::ToolOutput,
    )
    .expect("process tool image");
    assert_eq!(processed.mime, "image/jpeg");
    let decoded = image::load_from_memory(&processed.bytes).expect("decode JPEG");
    assert_eq!(decoded.dimensions(), (1000, 1000));
    assert_eq!(decoded.get_pixel(64, 64), Rgba([255, 255, 255, 255]));
    assert_eq!(decoded.get_pixel(256, 64), Rgba([127, 127, 127, 255]));
    let mut decoder = ImageReader::with_format(Cursor::new(&processed.bytes), ImageFormat::Jpeg)
        .into_decoder()
        .expect("create JPEG decoder");
    assert_eq!(
        decoder.orientation().expect("read EXIF orientation"),
        Orientation::Rotate90
    );
}

#[tracing_test::traced_test]
#[test]
fn tool_output_jpeg_over_budget_keeps_dimensions_and_warns_once() {
    let image = noise_image(2048, 1248);
    let original_bytes = image_bytes(&image, ImageFormat::Png);
    let url = data_url_from_bytes("image/png", &original_bytes);

    let processed = load_data_url_for_prompt(
        &url,
        PromptImageMode::HIGH_DETAIL,
        PromptImageTreatment::ToolOutput,
    )
    .expect("process tool image");
    assert_eq!(processed.mime, "image/jpeg");
    assert_eq!(
        image::load_from_memory(&processed.bytes)
            .unwrap()
            .dimensions(),
        (2048, 1248)
    );
    assert!(processed.bytes.len() > ORIGINAL_MAX_IMAGE_BYTES);
    logs_assert(|lines| {
        let warnings: Vec<_> = lines
            .iter()
            .filter(|line| line.contains("tool output image still exceeds 1 MiB"))
            .collect();
        if warnings.len() == 1 && warnings[0].contains("WARN") && warnings[0].contains("quality=50")
        {
            Ok(())
        } else {
            Err(format!(
                "expected one warning at quality 50, got {warnings:?}"
            ))
        }
    });
}
