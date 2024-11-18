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
    CastPropertyValue {
        name: &'static str,
        #[snafu(source(from(dicom::core::value::CastValueError, Box::new)))]
        source: Box<dicom::core::value::CastValueError>,
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
    WriteToTiff {
        #[snafu(source(from(TiffError, Box::new)))]
        source: Box<TiffError>,
    },
    ConvertImageToBytes,
    WriteTags {
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
}

/// The order of the channels in the loaded image
pub enum ChannelOrder {
    First,
    Last,
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
            ChannelOrder::First => (
                self.num_frames,
                self.height,
                self.width,
                self.color_type.channels(),
            ),
            ChannelOrder::Last => (
                self.num_frames,
                self.color_type.channels(),
                self.height,
                self.width,
            ),
        }
    }

    pub fn frame_shape(&self, channel_order: &ChannelOrder) -> (usize, usize, usize) {
        match channel_order {
            ChannelOrder::First => (self.height, self.width, self.color_type.channels()),
            ChannelOrder::Last => (self.color_type.channels(), self.height, self.width),
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
        // Determine dimensions of the loaded result, accounting for the selected frames subset
        let frames = frames.collect::<Vec<_>>();
        let dimensions = Dimensions::try_from(&mut *decoder)?.with_num_frames(frames.len());
        let channel_order = ChannelOrder::First;

        let mut array = Array4::<T>::zeros(dimensions.shape(&channel_order));

        let mut frame = 0;
        loop {
            let image = decoder.read_image().context(DecodeImageSnafu)?;
            let image = Self::to_vec(image)?;
            let mut slice = array.slice_mut(s![frame, .., .., ..]);
            let new = Array::from_shape_vec(dimensions.frame_shape(&channel_order), image).unwrap();
            slice.assign(&new);

            frame += 1;
            if !decoder.more_images() {
                break;
            }
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
            _ => panic!("Unsupported color type"),
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
            _ => panic!("Unsupported color type"),
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
    use image::Pixel;
    use rstest::rstest;
    use tiff::encoder::compression::{Compressor, Uncompressed};

    #[rstest]
    #[case("pydicom/CT_small.dcm")]
    #[case("pydicom/MR_small.dcm")]
    #[case("pydicom/JPEG2000_UNC.dcm")]
    #[case("pydicom/JPGLosslessP14SV1_1s_1f_8b.dcm")] // Gray8
    fn test_load_preprocessed(#[case] dicom_file_path: &str) {
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
        println!("color_type: {:?}", color_type);

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
        let array = Array4::<u16>::decode(&mut decoder).unwrap();

        // Compare with DynamicImage still in memory
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
}
