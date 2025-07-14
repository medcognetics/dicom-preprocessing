use std::io::{Read, Seek};
use tiff::decoder::{Decoder, DecodingResult};

use snafu::ResultExt;

use crate::errors::{tiff::SeekToFrameSnafu, TiffError};
use crate::metadata::{Dimensions, FrameCount};
use ndarray::{s, Array, Array4};

use crate::color::DicomColorType;
use image::DynamicImage;

pub trait LoadFromTiff<T: Clone + num::Zero> {
    fn to_vec(decoded: DecodingResult) -> Result<Vec<T>, TiffError>;

    fn decode_frames<R: Read + Seek>(
        decoder: &mut Decoder<R>,
        frames: impl Iterator<Item = usize>,
    ) -> Result<Array4<T>, TiffError> {
        let frames = frames.into_iter().collect::<Vec<_>>();
        let num_frames = frames.len();

        // Determine dimensions of the loaded result, accounting for the selected frames subset
        // TODO: Support channel-first
        let dimensions = Dimensions::try_from(&mut *decoder)?;
        let decoded_dimensions = dimensions.with_num_frames(frames.len());

        // Validate that the requested frames are within the range of the TIFF file
        let max_frame = *frames.iter().max().unwrap();
        if max_frame >= dimensions.num_frames {
            return Err(TiffError::InvalidFrameIndex {
                frame: max_frame,
                num_frames: dimensions.num_frames,
            });
        }

        // If only one frame is requested, we can avoid the overhead of pre-allocating the array
        // and just use the original buffer.
        if num_frames == 1 {
            let frame = frames[0];
            decoder.seek_to_image(frame).context(SeekToFrameSnafu {
                frame,
                num_frames: dimensions.num_frames,
            })?;
            let image = decoder.read_image()?;
            let image = Self::to_vec(image)?;
            return Ok(Array::from_shape_vec(decoded_dimensions.shape(), image).unwrap());
        }

        // Pre-allocate contiguous array and fill it with the decoded frames
        let mut array = Array4::<T>::zeros(decoded_dimensions.shape());
        for (i, frame) in frames.into_iter().enumerate() {
            decoder.seek_to_image(frame).context(SeekToFrameSnafu {
                frame,
                num_frames: dimensions.num_frames,
            })?;
            // TODO: It would be nice to decode directly into the array, without the intermediate vector.
            // However, read_image() has a complex implementation so it is non-trivial to reimplement.
            // Maybe revisit this in the future.
            let image = decoder.read_image()?;
            let image = Self::to_vec(image)?;
            let mut slice = array.slice_mut(s![i, .., .., ..]);
            let new = Array::from_shape_vec(decoded_dimensions.frame_shape(), image).unwrap();
            slice.assign(&new);
        }

        Ok(array)
    }

    fn decode<R: Read + Seek>(decoder: &mut Decoder<R>) -> Result<Array4<T>, TiffError> {
        let frame_count = FrameCount::try_from(&mut *decoder)?;
        let frames = 0..(frame_count.into());
        Self::decode_frames(decoder, frames)
    }
}

impl LoadFromTiff<u8> for Array4<u8> {
    fn to_vec(decoded: DecodingResult) -> Result<Vec<u8>, TiffError> {
        match decoded {
            DecodingResult::U8(image) => Ok(image),
            DecodingResult::U16(image) => Ok(image
                .iter()
                .map(|&x| (x as u8) * (u8::MAX / u16::MAX as u8))
                .collect()),
            _ => Err(TiffError::UnsupportedDataType { data_type: decoded }),
        }
    }
}

impl LoadFromTiff<u16> for Array4<u16> {
    fn to_vec(decoded: DecodingResult) -> Result<Vec<u16>, TiffError> {
        match decoded {
            DecodingResult::U8(image) => Ok(image
                .iter()
                .map(|&x| (x as u16) * (u16::MAX / u8::MAX as u16))
                .collect()),
            DecodingResult::U16(image) => Ok(image),
            _ => Err(TiffError::UnsupportedDataType { data_type: decoded }),
        }
    }
}

impl LoadFromTiff<f32> for Array4<f32> {
    fn to_vec(decoded: DecodingResult) -> Result<Vec<f32>, TiffError> {
        match decoded {
            DecodingResult::U8(image) => Ok(image
                .into_iter()
                .map(|x| x as f32 / u8::MAX as f32)
                .collect()),
            DecodingResult::U16(image) => Ok(image
                .into_iter()
                .map(|x| x as f32 / u16::MAX as f32)
                .collect()),
            _ => Err(TiffError::UnsupportedDataType { data_type: decoded }),
        }
    }
}

/// Convert frame data (RGB or RGBA) to RGB format for image creation.
/// If alpha is present, it is ignored.
fn frame_to_rgb_data<T: Copy>(frame: ndarray::ArrayView3<T>) -> Vec<T> {
    match frame.shape() {
        [height, width, 3 | 4] => (0..*height)
            .flat_map(|y| {
                (0..*width)
                    .flat_map(move |x| [frame[[y, x, 0]], frame[[y, x, 1]], frame[[y, x, 2]]])
            })
            .collect(),
        _ => panic!("Expected 3 or 4 channels for RGB conversion"),
    }
}

/// Convert array frames to DynamicImages using a conversion function
fn convert_frames_to_images<T, F>(
    array: &ndarray::Array4<T>,
    convert_frame: F,
) -> Result<Vec<DynamicImage>, TiffError>
where
    T: Copy,
    F: Fn(ndarray::ArrayView3<T>) -> Result<DynamicImage, TiffError>,
{
    (0..array.shape()[0])
        .map(|i| {
            let frame = array.slice(ndarray::s![i, .., .., ..]);
            convert_frame(frame)
        })
        .collect()
}

/// Load frames from a TIFF decoder as DynamicImages.
///
/// This function handles different color types (Gray8, Gray16, RGB8) and converts
/// the specified frames of the TIFF to the appropriate DynamicImage variants.
pub fn load_frames_as_dynamic_images<R: Read + Seek>(
    decoder: &mut Decoder<R>,
    color_type: &DicomColorType,
    frames: impl Iterator<Item = usize>,
) -> Result<Vec<DynamicImage>, TiffError> {
    let frames_vec: Vec<usize> = frames.collect();

    match color_type {
        DicomColorType::Gray8(_) => {
            let array = Array4::<u8>::decode_frames(decoder, frames_vec.iter().cloned())?;
            convert_frames_to_images(&array, |frame| {
                match frame.shape() {
                    [height, width, 1] => {
                        // Grayscale image
                        let data: Vec<u8> = frame.iter().cloned().collect();
                        let image_buffer =
                            image::ImageBuffer::from_raw(*width as u32, *height as u32, data)
                                .ok_or(TiffError::DynamicImageError {
                                    color_type: image::ColorType::L8,
                                })?;
                        Ok(DynamicImage::ImageLuma8(image_buffer))
                    }
                    [height, width, 3] => {
                        // RGB image
                        let rgb_data = frame_to_rgb_data(frame);
                        let image_buffer =
                            image::ImageBuffer::from_raw(*width as u32, *height as u32, rgb_data)
                                .ok_or(TiffError::DynamicImageError {
                                color_type: image::ColorType::Rgb8,
                            })?;
                        Ok(DynamicImage::ImageRgb8(image_buffer))
                    }
                    _ => Err(TiffError::UnsupportedDataType {
                        data_type: DecodingResult::U8(vec![]),
                    }),
                }
            })
        }
        DicomColorType::Gray16(_) => {
            let array = Array4::<u16>::decode_frames(decoder, frames_vec.iter().cloned())?;
            convert_frames_to_images(&array, |frame| {
                match frame.shape() {
                    [height, width, 1] => {
                        // Grayscale 16-bit image
                        let data: Vec<u16> = frame.iter().cloned().collect();
                        let image_buffer =
                            image::ImageBuffer::from_raw(*width as u32, *height as u32, data)
                                .ok_or(TiffError::DynamicImageError {
                                    color_type: image::ColorType::L16,
                                })?;
                        Ok(DynamicImage::ImageLuma16(image_buffer))
                    }
                    _ => Err(TiffError::UnsupportedDataType {
                        data_type: DecodingResult::U16(vec![]),
                    }),
                }
            })
        }
        DicomColorType::RGB8(_) => {
            let array = Array4::<u8>::decode_frames(decoder, frames_vec.iter().cloned())?;
            convert_frames_to_images(&array, |frame| {
                match frame.shape() {
                    [height, width, 3] => {
                        // RGB image
                        let rgb_data = frame_to_rgb_data(frame);
                        let image_buffer =
                            image::ImageBuffer::from_raw(*width as u32, *height as u32, rgb_data)
                                .ok_or(TiffError::DynamicImageError {
                                color_type: image::ColorType::Rgb8,
                            })?;
                        Ok(DynamicImage::ImageRgb8(image_buffer))
                    }
                    _ => Err(TiffError::UnsupportedDataType {
                        data_type: DecodingResult::U8(vec![]),
                    }),
                }
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dicom::object::open_file;
    use dicom::pixeldata::ConvertOptions;
    use std::fs::File;
    use std::io::BufReader;
    use tempfile;

    use crate::color::DicomColorType;
    use crate::preprocess::Preprocessor;
    use crate::save::TiffSaver;
    use crate::transform::resize::FilterType;
    use crate::transform::PaddingDirection;
    use crate::volume::{KeepVolume, VolumeHandler};
    use image::DynamicImage;
    use image::ImageBuffer;
    use image::Luma;
    use image::Pixel;
    use rstest::rstest;
    use tiff::encoder::compression::{Compressor, Uncompressed};

    const NUM_CHANNELS_MONO: usize = 1;
    const NUM_CHANNELS_RGB: usize = 3;

    #[rstest]
    #[case("pydicom/CT_small.dcm", 16, false, false)]
    #[case("pydicom/MR_small.dcm", 16, false, false)]
    #[case("pydicom/JPEG2000_UNC.dcm", 16, false, false)]
    #[case("pydicom/JPGLosslessP14SV1_1s_1f_8b.dcm", 16, false, false)] // Gray8 -> u16 array
    #[case("pydicom/JPGLosslessP14SV1_1s_1f_8b.dcm", 8, false, false)] // Gray8 -> u8 array
    #[case("pydicom/SC_rgb.dcm", 8, true, false)] // RGB8 -> u8 array
    #[case("pydicom/emri_small.dcm", 16, false, false)] // multi frame
    #[case("pydicom/CT_small.dcm", 16, false, true)] // float32
    fn test_load_preprocessed(
        #[case] dicom_file_path: &str,
        #[case] bits: usize,
        #[case] rgb: bool,
        #[case] fp32: bool,
    ) {
        let config = Preprocessor {
            crop: true,
            size: Some((64, 64)),
            filter: FilterType::Triangle,
            padding_direction: PaddingDirection::default(),
            crop_max: false,
            volume_handler: VolumeHandler::Keep(KeepVolume),
            use_components: true,
            use_padding: true,
            border_frac: None,
            target_frames: 1,
            convert_options: ConvertOptions::default(),
        };

        let dicom_file = open_file(dicom_test_files::path(dicom_file_path).unwrap()).unwrap();
        let color_type = DicomColorType::try_from(&dicom_file).unwrap();

        // Run preprocessing
        let (images, metadata) = config.prepare_image(&dicom_file, false).unwrap();

        // Save to TIFF
        let saver = TiffSaver::new(Compressor::Uncompressed(Uncompressed), color_type);
        let tmpdir = tempfile::tempdir().unwrap();
        let path = tmpdir.path().join("test.tiff");
        let mut encoder = saver.open_tiff(path.clone()).unwrap();
        images
            .clone()
            .into_iter()
            .try_for_each(|image| saver.save(&mut encoder, &image, &metadata))
            .unwrap();
        drop(encoder);

        // Load from TIFF
        let file = BufReader::new(File::open(&path).unwrap());
        let mut decoder = Decoder::new(file).unwrap();
        match (bits, rgb, fp32) {
            (16, false, false) => exec_test_luma16(images, &mut decoder),
            (8, false, false) => exec_test_luma8(images, &mut decoder),
            (8, true, false) => exec_test_rgb8(images, &mut decoder),
            (_, false, true) => exec_test_float32(images, &mut decoder),
            _ => unreachable!(),
        }
    }

    fn exec_test_luma16(images: Vec<DynamicImage>, decoder: &mut Decoder<BufReader<File>>) {
        let array = Array4::<u16>::decode(&mut *decoder).unwrap();
        assert_eq!(array.shape()[0], images.len());
        for i in 0..images.len() {
            let actual_frame = array.slice(s![i, .., .., ..]);
            let expected_frame = images[i].clone().into_luma16();
            assert_eq!(actual_frame.shape()[0], expected_frame.height() as usize);
            assert_eq!(actual_frame.shape()[1], expected_frame.width() as usize);
            assert_eq!(actual_frame.shape()[2], NUM_CHANNELS_MONO);
            for (x, y, pixel) in expected_frame.enumerate_pixels() {
                let expected_value = pixel.channels()[0];
                let actual_value = actual_frame.get((y as usize, x as usize, 0_usize)).unwrap();
                assert_eq!(*actual_value, expected_value, "at ({x}, {y})");
            }
        }
    }

    fn exec_test_luma8(images: Vec<DynamicImage>, decoder: &mut Decoder<BufReader<File>>) {
        let array = Array4::<u8>::decode(&mut *decoder).unwrap();
        assert_eq!(array.shape()[0], images.len());
        for i in 0..images.len() {
            let actual_frame = array.slice(s![i, .., .., ..]);
            let expected_frame = images[i].clone().into_luma8();
            assert_eq!(actual_frame.shape()[0], expected_frame.height() as usize);
            assert_eq!(actual_frame.shape()[1], expected_frame.width() as usize);
            assert_eq!(actual_frame.shape()[2], NUM_CHANNELS_MONO);
            for (x, y, pixel) in expected_frame.enumerate_pixels() {
                let expected_value = pixel.channels()[0];
                let actual_value = actual_frame.get((y as usize, x as usize, 0_usize)).unwrap();
                assert_eq!(*actual_value, expected_value, "at ({x}, {y})");
            }
        }
    }

    fn exec_test_rgb8(images: Vec<DynamicImage>, decoder: &mut Decoder<BufReader<File>>) {
        let array = Array4::<u8>::decode(&mut *decoder).unwrap();
        assert_eq!(array.shape()[0], images.len());
        for i in 0..images.len() {
            let actual_frame = array.slice(s![i, .., .., ..]);
            let expected_frame = images[i].clone().into_rgb8();
            assert_eq!(actual_frame.shape()[0], expected_frame.height() as usize);
            assert_eq!(actual_frame.shape()[1], expected_frame.width() as usize);
            assert_eq!(actual_frame.shape()[2], NUM_CHANNELS_RGB);
            for (x, y, pixel) in expected_frame.enumerate_pixels() {
                for i in 0..NUM_CHANNELS_RGB {
                    let expected_value = pixel.channels()[i];
                    let actual_value = actual_frame.get((y as usize, x as usize, i)).unwrap();
                    assert_eq!(*actual_value, expected_value, "at ({x}, {y}, {i})");
                }
            }
        }
    }

    fn exec_test_float32(images: Vec<DynamicImage>, decoder: &mut Decoder<BufReader<File>>) {
        let array = Array4::<f32>::decode(&mut *decoder).unwrap();
        assert_eq!(array.shape()[0], images.len());
        for i in 0..images.len() {
            let actual_frame = array.slice(s![i, .., .., ..]);
            let expected_frame = images[i].clone().into_luma16();
            // Convert u16 pixels to normalized f32 values
            let pixels: Vec<f32> = expected_frame
                .pixels()
                .map(|p| (p.0[0] as f32) / (u16::MAX as f32))
                .collect();
            let expected_frame: ImageBuffer<Luma<f32>, Vec<f32>> =
                ImageBuffer::from_raw(expected_frame.width(), expected_frame.height(), pixels)
                    .unwrap();
            assert_eq!(actual_frame.shape()[0], expected_frame.height() as usize);
            assert_eq!(actual_frame.shape()[1], expected_frame.width() as usize);
            assert_eq!(actual_frame.shape()[2], NUM_CHANNELS_MONO);
            for (x, y, pixel) in expected_frame.enumerate_pixels() {
                let expected_value = pixel.channels()[0];
                let actual_value = actual_frame.get((y as usize, x as usize, 0_usize)).unwrap();
                assert_eq!(*actual_value, expected_value, "at ({x}, {y})");
            }
        }
    }

    #[test]
    fn test_load_frame_subset() {
        let config = Preprocessor {
            crop: true,
            size: Some((64, 64)),
            filter: FilterType::Triangle,
            padding_direction: PaddingDirection::default(),
            crop_max: false,
            volume_handler: VolumeHandler::Keep(KeepVolume),
            use_components: true,
            use_padding: true,
            border_frac: None,
            target_frames: 1,
            convert_options: ConvertOptions::default(),
        };

        let dicom_file_path = "pydicom/emri_small.dcm";
        let dicom_file = open_file(dicom_test_files::path(dicom_file_path).unwrap()).unwrap();
        let color_type = DicomColorType::try_from(&dicom_file).unwrap();

        // Run preprocessing
        let (images, metadata) = config.prepare_image(&dicom_file, false).unwrap();

        // Save to TIFF
        let saver = TiffSaver::new(Compressor::Uncompressed(Uncompressed), color_type);
        let tmpdir = tempfile::tempdir().unwrap();
        let path = tmpdir.path().join("test.tiff");
        let mut encoder = saver.open_tiff(path.clone()).unwrap();
        images
            .clone()
            .into_iter()
            .try_for_each(|image| saver.save(&mut encoder, &image, &metadata))
            .unwrap();
        drop(encoder);

        // Load from TIFF
        let file = BufReader::new(File::open(&path).unwrap());
        let mut decoder = Decoder::new(file).unwrap();

        // Choose frame N/2 to sample and put it in a vector
        let frame: usize = metadata.num_frames.into();
        let frame = frame / 2;
        let frames = vec![frame];

        let array = Array4::<u16>::decode_frames(&mut decoder, frames.into_iter()).unwrap();
        assert_eq!(array.shape()[0], 1);
        let actual_frame = array.slice(s![0, .., .., ..]);
        let expected_frame = images[frame].clone().into_luma16();
        assert_eq!(actual_frame.shape()[0], expected_frame.height() as usize);
        assert_eq!(actual_frame.shape()[1], expected_frame.width() as usize);
        assert_eq!(actual_frame.shape()[2], NUM_CHANNELS_MONO);
        for (x, y, pixel) in expected_frame.enumerate_pixels() {
            let expected_value = pixel.channels()[0];
            let actual_value = actual_frame.get((y as usize, x as usize, 0_usize)).unwrap();
            assert_eq!(*actual_value, expected_value, "at ({x}, {y})");
        }
    }

    #[rstest]
    #[case("pydicom/CT_small.dcm", false)]
    #[case("pydicom/MR_small.dcm", false)]
    #[case("pydicom/JPGLosslessP14SV1_1s_1f_8b.dcm", false)] // Gray8
    #[case("pydicom/SC_rgb.dcm", true)] // RGB8
    #[case("pydicom/emri_small.dcm", false)] // multi-frame
    fn test_load_frames_as_dynamic_images(#[case] dicom_file_path: &str, #[case] is_rgb: bool) {
        let config = Preprocessor {
            crop: false,
            size: None,
            filter: FilterType::Triangle,
            padding_direction: PaddingDirection::default(),
            crop_max: false,
            volume_handler: VolumeHandler::Keep(KeepVolume),
            use_components: true,
            use_padding: false,
            border_frac: None,
            target_frames: 1,
            convert_options: ConvertOptions::default(),
        };

        let dicom_file = open_file(dicom_test_files::path(dicom_file_path).unwrap()).unwrap();
        let color_type = DicomColorType::try_from(&dicom_file).unwrap();

        // Run preprocessing to get expected images
        let (images, metadata) = config.prepare_image(&dicom_file, false).unwrap();

        // Save to TIFF
        let saver = TiffSaver::new(Compressor::Uncompressed(Uncompressed), color_type.clone());
        let tmpdir = tempfile::tempdir().unwrap();
        let path = tmpdir.path().join("test.tiff");
        let mut encoder = saver.open_tiff(path.clone()).unwrap();
        images
            .iter()
            .try_for_each(|image| saver.save(&mut encoder, image, &metadata))
            .unwrap();
        drop(encoder);

        // Load using our new function
        let file = BufReader::new(File::open(&path).unwrap());
        let mut decoder = Decoder::new(file).unwrap();
        let dynamic_images =
            super::load_frames_as_dynamic_images(&mut decoder, &color_type, std::iter::once(0))
                .unwrap();
        let dynamic_image = &dynamic_images[0];

        // Compare with the first expected image
        let expected_image = &images[0];
        assert_eq!(dynamic_image.width(), expected_image.width());
        assert_eq!(dynamic_image.height(), expected_image.height());

        if is_rgb {
            let actual_rgb = dynamic_image.clone().into_rgb8();
            let expected_rgb = expected_image.clone().into_rgb8();
            assert_eq!(actual_rgb.dimensions(), expected_rgb.dimensions());

            for (actual_pixel, expected_pixel) in actual_rgb.pixels().zip(expected_rgb.pixels()) {
                assert_eq!(actual_pixel, expected_pixel);
            }
        } else {
            match &color_type {
                DicomColorType::Gray8(_) => {
                    let actual_luma = dynamic_image.clone().into_luma8();
                    let expected_luma = expected_image.clone().into_luma8();
                    assert_eq!(actual_luma.dimensions(), expected_luma.dimensions());

                    for (actual_pixel, expected_pixel) in
                        actual_luma.pixels().zip(expected_luma.pixels())
                    {
                        assert_eq!(actual_pixel, expected_pixel);
                    }
                }
                DicomColorType::Gray16(_) => {
                    let actual_luma = dynamic_image.clone().into_luma16();
                    let expected_luma = expected_image.clone().into_luma16();
                    assert_eq!(actual_luma.dimensions(), expected_luma.dimensions());

                    for (actual_pixel, expected_pixel) in
                        actual_luma.pixels().zip(expected_luma.pixels())
                    {
                        assert_eq!(actual_pixel, expected_pixel);
                    }
                }
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn test_load_first_frame_with_single_frame_tiff() {
        // Create a simple single-frame TIFF for testing
        use crate::metadata::PreprocessingMetadata;
        use crate::FrameCount;

        let tmpdir = tempfile::tempdir().unwrap();
        let path = tmpdir.path().join("single_frame.tiff");

        // Create test data
        let metadata = PreprocessingMetadata {
            crop: None,
            resize: None,
            padding: None,
            resolution: None,
            num_frames: FrameCount::from(1_u16),
        };
        let color_type = DicomColorType::Gray8(tiff::encoder::colortype::Gray8);

        // Save test TIFF
        let saver = TiffSaver::new(Compressor::Uncompressed(Uncompressed), color_type.clone());
        let mut encoder = saver.open_tiff(&path).unwrap();
        let test_image = DynamicImage::ImageLuma8(
            image::ImageBuffer::from_raw(32, 32, vec![128; 32 * 32]).unwrap(),
        );
        saver.save(&mut encoder, &test_image, &metadata).unwrap();
        drop(encoder);

        // Load using our function
        let file = BufReader::new(File::open(&path).unwrap());
        let mut decoder = Decoder::new(file).unwrap();
        let loaded_images =
            super::load_frames_as_dynamic_images(&mut decoder, &color_type, std::iter::once(0))
                .unwrap();
        let loaded_image = &loaded_images[0];

        // Verify
        assert_eq!(loaded_image.width(), 32);
        assert_eq!(loaded_image.height(), 32);
        match loaded_image {
            DynamicImage::ImageLuma8(img) => {
                assert_eq!(img.get_pixel(0, 0).0[0], 128);
            }
            _ => panic!("Expected Luma8 image"),
        }
    }

    #[test]
    fn test_load_multiple_frames() {
        // Create a multi-frame TIFF and test loading specific frames
        use crate::metadata::PreprocessingMetadata;
        use crate::FrameCount;

        let tmpdir = tempfile::tempdir().unwrap();
        let path = tmpdir.path().join("multi_frame.tiff");

        let metadata = PreprocessingMetadata {
            crop: None,
            resize: None,
            padding: None,
            resolution: None,
            num_frames: FrameCount::from(3_u16),
        };
        let color_type = DicomColorType::Gray8(tiff::encoder::colortype::Gray8);

        // Create test images with different pixel values
        let test_images = vec![
            DynamicImage::ImageLuma8(
                image::ImageBuffer::from_raw(16, 16, vec![100; 16 * 16]).unwrap(),
            ),
            DynamicImage::ImageLuma8(
                image::ImageBuffer::from_raw(16, 16, vec![150; 16 * 16]).unwrap(),
            ),
            DynamicImage::ImageLuma8(
                image::ImageBuffer::from_raw(16, 16, vec![200; 16 * 16]).unwrap(),
            ),
        ];

        // Save multi-frame TIFF
        let saver = TiffSaver::new(Compressor::Uncompressed(Uncompressed), color_type.clone());
        let mut encoder = saver.open_tiff(&path).unwrap();
        for image in &test_images {
            saver.save(&mut encoder, image, &metadata).unwrap();
        }
        drop(encoder);

        // Test loading specific frames
        let file = BufReader::new(File::open(&path).unwrap());
        let mut decoder = Decoder::new(file).unwrap();

        // Load frames 0 and 2
        let loaded_images =
            super::load_frames_as_dynamic_images(&mut decoder, &color_type, vec![0, 2].into_iter())
                .unwrap();

        assert_eq!(loaded_images.len(), 2);

        // Check first loaded frame (frame 0)
        match &loaded_images[0] {
            DynamicImage::ImageLuma8(img) => {
                assert_eq!(img.get_pixel(0, 0).0[0], 100);
            }
            _ => panic!("Expected Luma8 image"),
        }

        // Check second loaded frame (frame 2)
        match &loaded_images[1] {
            DynamicImage::ImageLuma8(img) => {
                assert_eq!(img.get_pixel(0, 0).0[0], 200);
            }
            _ => panic!("Expected Luma8 image"),
        }
    }
}
