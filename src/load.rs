use std::fs::File;
use std::io::{BufReader, Read, Seek};
use std::path::PathBuf;
use tiff::decoder::{Decoder, DecodingResult};

use snafu::{ResultExt, Snafu};

use crate::color::DicomColorType;
use crate::errors::{tiff::SeekToFrameSnafu, DicomError, TiffError};
use crate::metadata::FrameCount;
use ndarray::{s, Array, Array4};

/// The order of the channels in the loaded image
#[derive(Clone, Copy, Debug)]
pub enum ChannelOrder {
    NHWC,
    NCHW,
}

#[derive(Clone, Debug)]
struct Dimensions {
    width: usize,
    height: usize,
    num_frames: usize,
    color_type: DicomColorType,
}

impl Dimensions {
    pub fn shape(&self, channel_order: &ChannelOrder) -> (usize, usize, usize, usize) {
        match channel_order {
            ChannelOrder::NHWC => (
                self.num_frames,
                self.height,
                self.width,
                self.color_type.channels(),
            ),
            ChannelOrder::NCHW => (
                self.num_frames,
                self.color_type.channels(),
                self.height,
                self.width,
            ),
        }
    }

    pub fn frame_shape(&self, channel_order: &ChannelOrder) -> (usize, usize, usize) {
        match channel_order {
            ChannelOrder::NHWC => (self.height, self.width, self.color_type.channels()),
            ChannelOrder::NCHW => (self.color_type.channels(), self.height, self.width),
        }
    }

    pub fn with_num_frames(self, num_frames: usize) -> Self {
        Dimensions { num_frames, ..self }
    }
}

impl<R: Read + Seek> TryFrom<&mut Decoder<R>> for Dimensions {
    type Error = TiffError;
    fn try_from(decoder: &mut Decoder<R>) -> Result<Self, Self::Error> {
        let (width, height) = decoder.dimensions()?;
        let color_type = decoder.colortype()?;
        let color_type = DicomColorType::try_from(color_type)?;
        let num_frames: u16 = FrameCount::try_from(decoder)?.into();
        Ok(Self {
            width: width as usize,
            height: height as usize,
            num_frames: num_frames as usize,
            color_type,
        })
    }
}

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
        let channel_order = ChannelOrder::NHWC;
        let decoded_dimensions = dimensions.clone().with_num_frames(frames.len());

        // Validate that the requested frames are within the range of the TIFF file
        let max_frame = frames.iter().max().unwrap().clone();
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
            return Ok(
                Array::from_shape_vec(decoded_dimensions.shape(&channel_order), image).unwrap(),
            );
        }

        // Pre-allocate contiguous array and fill it with the decoded frames
        let mut array = Array4::<T>::zeros(decoded_dimensions.shape(&channel_order));
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
            let new = Array::from_shape_vec(decoded_dimensions.frame_shape(&channel_order), image)
                .unwrap();
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

#[cfg(test)]
mod tests {
    use super::*;
    use dicom::object::open_file;
    use tempfile;

    use crate::preprocess::Preprocessor;
    use crate::save::TiffSaver;
    use crate::transform::PaddingDirection;
    use crate::volume::{KeepVolume, VolumeHandler};
    use image::imageops::FilterType;
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
        };

        let dicom_file = open_file(&dicom_test_files::path(dicom_file_path).unwrap()).unwrap();
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
                let actual_value = actual_frame
                    .get((y as usize, x as usize, 0 as usize))
                    .unwrap();
                assert_eq!(*actual_value, expected_value, "at ({}, {})", x, y);
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
                let actual_value = actual_frame
                    .get((y as usize, x as usize, 0 as usize))
                    .unwrap();
                assert_eq!(*actual_value, expected_value, "at ({}, {})", x, y);
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
                    let actual_value = actual_frame
                        .get((y as usize, x as usize, i as usize))
                        .unwrap();
                    assert_eq!(*actual_value, expected_value, "at ({}, {}, {})", x, y, i);
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
                let actual_value = actual_frame
                    .get((y as usize, x as usize, 0 as usize))
                    .unwrap();
                assert_eq!(*actual_value, expected_value, "at ({}, {})", x, y);
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
        };

        let dicom_file_path = "pydicom/emri_small.dcm";
        let dicom_file = open_file(&dicom_test_files::path(dicom_file_path).unwrap()).unwrap();
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
            let actual_value = actual_frame
                .get((y as usize, x as usize, 0 as usize))
                .unwrap();
            assert_eq!(*actual_value, expected_value, "at ({}, {})", x, y);
        }
    }
}
