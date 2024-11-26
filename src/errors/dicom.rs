use dicom::core::value::Value;
use dicom::core::value::{CastValueError, ConvertValueError};
use dicom::object::ReadError;
use dicom::pixeldata::PhotometricInterpretation;
pub use snafu::{Snafu, Whatever};
use std::path::PathBuf;
use std::sync::Arc;
use tiff::ColorType;
use tiff::TiffError;

#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum DicomError {
    #[snafu(display("error reading DICOM file: {:?}", source))]
    ReadError {
        #[snafu(source(from(ReadError, Box::new)))]
        source: Box<ReadError>,
    },

    #[snafu(display("missing DICOM property: {}", name))]
    MissingPropertyError { name: &'static str },

    #[snafu(display("unable to cast DICOM property value '{}': {:?}", name, source))]
    CastValueError {
        name: &'static str,
        #[snafu(source(from(CastValueError, Box::new)))]
        source: Box<CastValueError>,
    },

    #[snafu(display("unable to convert DICOM property value '{}': {:?}", name, source))]
    ConvertValueError {
        name: &'static str,
        #[snafu(source(from(ConvertValueError, Box::new)))]
        source: Box<ConvertValueError>,
    },

    #[snafu(display("invalid DICOM property value '{}': {}", name, value))]
    InvalidValueError { name: &'static str, value: String },

    #[snafu(display("error processing DICOM pixel data: {:?}", source))]
    PixelDataError {
        #[snafu(source(from(dicom::pixeldata::Error, Box::new)))]
        source: Box<dicom::pixeldata::Error>,
    },

    #[snafu(display(
        "Unsupported photometric interpretation: BitsAllocated={}, PhotometricInterpretation={}",
        bits_allocated,
        photometric_interpretation
    ))]
    UnsupportedPhotometricInterpretation {
        bits_allocated: u16,
        photometric_interpretation: PhotometricInterpretation,
    },

    #[snafu(display("error parsing float: {:?}", source))]
    ParseFloatError {
        #[snafu(source(from(std::num::ParseFloatError, Box::new)))]
        source: Box<std::num::ParseFloatError>,
    },

    #[snafu(display(
        "frame index out of bounds: start={}, end={}, number_of_frames={}",
        start,
        end,
        number_of_frames
    ))]
    FrameIndexError {
        start: usize,
        end: usize,
        number_of_frames: usize,
    },

    #[snafu(display("{}", message))]
    Other { message: String },
}

impl From<Whatever> for DicomError {
    fn from(source: Whatever) -> Self {
        Self::Other {
            message: source.to_string(),
        }
    }
}
