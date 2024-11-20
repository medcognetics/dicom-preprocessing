use std::fs::File;
use std::io::{BufReader, Read, Seek};
use std::path::PathBuf;
use tiff::decoder::{Decoder, DecodingResult};
use tiff::TiffError;

use snafu::{ResultExt, Snafu};

use crate::color::{ColorError, DicomColorType};
use crate::metadata::FrameCount;
use ndarray::{s, Array, Array4};

#[derive(Debug, Snafu)]
pub enum LoadError {
    ParseColorType {
        #[snafu(source(from(TiffError, Box::new)))]
        source: Box<TiffError>,
    },
    #[snafu(display("could not open TIFF file {}", path.display()))]
    OpenTiff {
        #[snafu(source(from(std::io::Error, Box::new)))]
        source: Box<std::io::Error>,
        path: PathBuf,
    },
    #[snafu(display("could not read TIFF file {}", path.display()))]
    ReadTiff {
        #[snafu(source(from(TiffError, Box::new)))]
        source: Box<TiffError>,
        path: PathBuf,
    },
    DecodeImage {
        #[snafu(source(from(TiffError, Box::new)))]
        source: Box<TiffError>,
    },
    DimensionsError {
        #[snafu(source(from(TiffError, Box::new)))]
        source: Box<TiffError>,
    },
    UnsupportedColorType {
        #[snafu(source(from(ColorError, Box::new)))]
        source: Box<ColorError>,
    },
    #[snafu(display("could not convert color type from {:?} to {}", input, target))]
    ConvertColorType {
        input: DecodingResult,
        target: &'static str,
    },
    InvalidFrameIndex {
        frame: usize,
        num_frames: usize,
    },
    SeekToFrame {
        #[snafu(source(from(TiffError, Box::new)))]
        source: Box<TiffError>,
        frame: usize,
        num_frames: usize,
    },
}

/// The order of the channels in the loaded image
pub enum ChannelOrder {
    NHWC,
    NCHW,
}

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
    type Error = LoadError;
    fn try_from(decoder: &mut Decoder<R>) -> Result<Self, Self::Error> {
        let (width, height) = decoder.dimensions().context(DimensionsSnafu)?;
        let color_type = decoder.colortype().context(ParseColorTypeSnafu)?;
        let color_type = DicomColorType::try_from(color_type).context(UnsupportedColorTypeSnafu)?;
        let num_frames: u16 = FrameCount::try_from(decoder)
            .context(DimensionsSnafu)?
            .into();
        Ok(Self {
            width: width as usize,
            height: height as usize,
            num_frames: num_frames as usize,
            color_type,
        })
    }
}

pub trait LoadFromTiff<T: Clone + num::Zero> {
    fn to_vec(decoded: DecodingResult) -> Result<Vec<T>, LoadError>;

    /// Open the TIFF file, producing a decoder
    fn open(path: PathBuf) -> Result<Decoder<BufReader<File>>, LoadError> {
        let file = BufReader::new(File::open(&path).context(OpenTiffSnafu { path: path.clone() })?);
        Decoder::new(file).context(ReadTiffSnafu { path })
    }

    fn decode_frames<R: Read + Seek>(
        decoder: &mut Decoder<R>,
        frames: impl Iterator<Item = usize>,
    ) -> Result<Array4<T>, LoadError> {
        let frames = frames.into_iter().collect::<Vec<_>>();

        // Determine dimensions of the loaded result, accounting for the selected frames subset
        // TODO: Support channel-first
        let dimensions = Dimensions::try_from(&mut *decoder)?.with_num_frames(frames.len());
        let channel_order = ChannelOrder::NHWC;

        // Validate that the requested frames are within the range of the TIFF file
        let max_frame = frames.iter().max().unwrap().clone();
        if max_frame >= dimensions.num_frames {
            return Err(LoadError::InvalidFrameIndex {
                frame: max_frame,
                num_frames: dimensions.num_frames,
            });
        }

        // Pre-allocate contiguous array and fill it with the decoded frames
        let mut array = Array4::<T>::zeros(dimensions.shape(&channel_order));
        for frame in frames {
            decoder.seek_to_image(frame).context(SeekToFrameSnafu {
                frame,
                num_frames: dimensions.num_frames,
            })?;
            // TODO: It would be nice to decode directly into the array, without the intermediate vector.
            // However, read_image() has a complex implementation so it is non-trivial to reimplement.
            // Maybe revisit this in the future.
            let image = decoder.read_image().context(DecodeImageSnafu)?;
            let image = Self::to_vec(image)?;
            let mut slice = array.slice_mut(s![frame, .., .., ..]);
            let new = Array::from_shape_vec(dimensions.frame_shape(&channel_order), image).unwrap();
            slice.assign(&new);
        }

        Ok(array)
    }

    fn decode<R: Read + Seek>(decoder: &mut Decoder<R>) -> Result<Array4<T>, LoadError> {
        let frame_count = FrameCount::try_from(&mut *decoder).context(DimensionsSnafu)?;
        let frames = 0..(frame_count.into());
        Self::decode_frames(decoder, frames)
    }
}

impl LoadFromTiff<u8> for Array4<u8> {
    fn to_vec(decoded: DecodingResult) -> Result<Vec<u8>, LoadError> {
        match decoded {
            DecodingResult::U8(image) => Ok(image),
            DecodingResult::U16(image) => Ok(image
                .iter()
                .map(|&x| (x as u8) * (u8::MAX / u16::MAX as u8))
                .collect()),
            _ => Err(LoadError::ConvertColorType {
                input: decoded,
                target: "u8",
            }),
        }
    }
}

impl LoadFromTiff<u16> for Array4<u16> {
    fn to_vec(decoded: DecodingResult) -> Result<Vec<u16>, LoadError> {
        match decoded {
            DecodingResult::U8(image) => Ok(image
                .iter()
                .map(|&x| (x as u16) * (u16::MAX / u8::MAX as u16))
                .collect()),
            DecodingResult::U16(image) => Ok(image),
            _ => Err(LoadError::ConvertColorType {
                input: decoded,
                target: "u16",
            }),
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
    use image::Pixel;
    use rstest::rstest;
    use tiff::encoder::compression::{Compressor, Uncompressed};

    #[rstest]
    #[case("pydicom/CT_small.dcm", 16, false)]
    #[case("pydicom/MR_small.dcm", 16, false)]
    #[case("pydicom/JPEG2000_UNC.dcm", 16, false)]
    #[case("pydicom/JPGLosslessP14SV1_1s_1f_8b.dcm", 16, false)] // Gray8 -> u16 array
    #[case("pydicom/JPGLosslessP14SV1_1s_1f_8b.dcm", 8, false)] // Gray8 -> u8 array
    #[case("pydicom/SC_rgb.dcm", 8, true)] // RGB8 -> u8 array
    fn test_load_preprocessed(
        #[case] dicom_file_path: &str,
        #[case] bits: usize,
        #[case] rgb: bool,
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
        assert_eq!(images.len(), 1);

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
        match (bits, rgb) {
            (16, false) => exec_test_luma16(images, &mut decoder),
            (8, false) => exec_test_luma8(images, &mut decoder),
            (8, true) => exec_test_rgb8(images, &mut decoder),
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
            assert_eq!(actual_frame.shape()[2], 1);
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
            assert_eq!(actual_frame.shape()[2], 1);
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
            assert_eq!(actual_frame.shape()[2], 3);
            for (x, y, pixel) in expected_frame.enumerate_pixels() {
                for i in 0..3 {
                    let expected_value = pixel.channels()[i];
                    let actual_value = actual_frame
                        .get((y as usize, x as usize, i as usize))
                        .unwrap();
                    assert_eq!(*actual_value, expected_value, "at ({}, {}, {})", x, y, i);
                }
            }
        }
    }
}
