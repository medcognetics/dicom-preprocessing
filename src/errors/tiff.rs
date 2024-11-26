use dicom::core::value::Value;
use dicom::core::value::{CastValueError, ConvertValueError};
use dicom::object::ReadError;
use dicom::pixeldata::PhotometricInterpretation;
use image::{ColorType as ImageColorType, DynamicImage};
pub use snafu::{Snafu, Whatever};
use std::path::PathBuf;
use tiff::decoder::DecodingResult;
use tiff::ColorType;
use tiff::TiffError as BaseTiffError;

#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum TiffError {
    #[snafu(display("IO error on TIFF file {}", path.display()))]
    IOError {
        #[snafu(source(from(std::io::Error, Box::new)))]
        source: Box<std::io::Error>,
        path: PathBuf,
    },

    #[snafu(display("error reading TIFF file {}", path.display()))]
    ReadError {
        path: PathBuf,
        #[snafu(source(from(BaseTiffError, Box::new)))]
        source: Box<BaseTiffError>,
    },

    #[snafu(display("error writing TIFF file {}", path.display()))]
    WriteError {
        path: PathBuf,
        #[snafu(source(from(BaseTiffError, Box::new)))]
        source: Box<BaseTiffError>,
    },

    #[snafu(display("missing TIFF tag: {}", name))]
    MissingPropertyError { name: &'static str },

    #[snafu(display("invalid frame index: {} of {}", frame, num_frames))]
    InvalidFrameIndex { frame: usize, num_frames: usize },

    #[snafu(display("error seeking to frame {} of {}", frame, num_frames))]
    SeekToFrameError {
        #[snafu(source(from(BaseTiffError, Box::new)))]
        source: Box<BaseTiffError>,
        frame: usize,
        num_frames: usize,
    },

    #[snafu(display("unsupported color type: {:?}", color_type))]
    UnsupportedColorType { color_type: ColorType },

    #[snafu(display("unsupported data type: {:?}", data_type))]
    UnsupportedDataType { data_type: DecodingResult },

    #[snafu(display("tag {} has invalid length: {} (expected {})", name, actual, expected))]
    CardinalityError {
        name: &'static str,
        actual: usize,
        expected: usize,
    },

    #[snafu(display("error converting DynamicImage to bytes: {:?}", color_type))]
    DynamicImageError { color_type: ImageColorType },

    #[snafu(display("error processing TIFF file: {:?}", source))]
    GeneralTiffError {
        #[snafu(source(from(BaseTiffError, Box::new)))]
        source: Box<BaseTiffError>,
    },

    #[snafu(display("{}", message))]
    Other { message: String },
}

impl From<Whatever> for TiffError {
    fn from(source: Whatever) -> Self {
        Self::Other {
            message: source.to_string(),
        }
    }
}

impl From<BaseTiffError> for TiffError {
    fn from(source: BaseTiffError) -> Self {
        Self::GeneralTiffError {
            source: Box::new(source),
        }
    }
}
