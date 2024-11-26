use dicom::core::value::Value;
pub use snafu::Snafu;
use dicom::pixeldata::PhotometricInterpretation;
use dicom::core::value::{CastValueError, ConvertValueError};
use std::path::PathBuf;
use dicom::object::ReadError;
use tiff::TiffError;
use tiff::ColorType;
use std::sync::Arc;

#[derive(Debug, Snafu)]
#[snafu(display("invalid volume range: start={}, end={}, total={}", start, end, total))]
pub struct RangeError {
    start: u32,
    end: u32,
    total: u32,
}


#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
/// Errors that can occur when extracting information from a DICOM file
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
    InvalidValueError {
        name: &'static str,
        value: String,
    },
    #[snafu(display("error processing DICOM pixel data: {:?}", source))]
    PixelDataError {
        #[snafu(source(from(dicom::pixeldata::Error, Box::new)))]
        source: Box<dicom::pixeldata::Error>,
    },
    #[snafu(display(
        "Unsupported photometric interpretation: {}, {}",
        bits_allocated,
        photometric_interpretation
    ))]
    UnsupportedPhotometricInterpretation {
        bits_allocated: u16,
        photometric_interpretation: PhotometricInterpretation,
    },
}
