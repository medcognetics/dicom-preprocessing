use crate::errors::{dicom::CastValueSnafu, DicomError, TiffError};
use dicom::dictionary_std::tags;
use dicom::object::{FileDicomObject, InMemDicomObject};
use dicom::pixeldata::PhotometricInterpretation;
use snafu::ResultExt;
use std::io::{Read, Seek};
use tiff::decoder::Decoder;
use tiff::encoder::colortype::{Gray16, Gray8, RGB8};
use tiff::ColorType;

/// The color types we expect to encounter in DICOM files and support processing of
pub enum DicomColorType {
    Gray8(Gray8),
    Gray16(Gray16),
    RGB8(RGB8),
}

impl DicomColorType {
    /// Infer color type from BitsAllocated and PhotometricInterpretation
    pub fn try_new(
        bits_allocated: u16,
        photometric_interpretation: PhotometricInterpretation,
    ) -> Result<Self, DicomError> {
        match (bits_allocated, photometric_interpretation) {
            // Monochrome 16-bit
            (16, PhotometricInterpretation::Monochrome1)
            | (16, PhotometricInterpretation::Monochrome2) => Ok(DicomColorType::Gray16(Gray16)),
            // Monochrome 8-bit
            (8, PhotometricInterpretation::Monochrome1)
            | (8, PhotometricInterpretation::Monochrome2) => Ok(DicomColorType::Gray8(Gray8)),
            // RGB 8-bit
            (8, PhotometricInterpretation::Rgb) => Ok(DicomColorType::RGB8(RGB8)),
            // Unsupported
            (bits_allocated, photometric_interpretation) => {
                Err(DicomError::UnsupportedPhotometricInterpretation {
                    bits_allocated,
                    photometric_interpretation,
                })
            }
        }
    }

    pub fn channels(&self) -> usize {
        match self {
            DicomColorType::Gray8(_) => 1,
            DicomColorType::Gray16(_) => 1,
            DicomColorType::RGB8(_) => 3,
        }
    }
}

impl TryFrom<&FileDicomObject<InMemDicomObject>> for DicomColorType {
    type Error = DicomError;

    /// Read the BitsAllocated and PhotometricInterpretation tags from the DICOM file
    /// and infer the appropriate color type.
    fn try_from(file: &FileDicomObject<InMemDicomObject>) -> Result<Self, Self::Error> {
        let bits_allocated = file
            .get(tags::BITS_ALLOCATED)
            .ok_or(DicomError::MissingPropertyError {
                name: "Bits Allocated",
            })?
            .value()
            .uint16()
            .context(CastValueSnafu {
                name: "Bits Allocated",
            })?;
        let photometric_interpretation = file
            .get(tags::PHOTOMETRIC_INTERPRETATION)
            .ok_or(DicomError::MissingPropertyError {
                name: "Photometric Interpretation",
            })?
            .value()
            .string()
            .context(CastValueSnafu {
                name: "Photometric Interpretation",
            })?;
        let photometric_interpretation =
            PhotometricInterpretation::from(photometric_interpretation.trim());

        DicomColorType::try_new(bits_allocated, photometric_interpretation)
    }
}

// NOTE: There is a trait ColorType in the encoder module, and an enum ColorType at the top level.
impl From<DicomColorType> for ColorType {
    fn from(color_type: DicomColorType) -> Self {
        match color_type {
            DicomColorType::Gray8(Gray8) => ColorType::Gray(8),
            DicomColorType::Gray16(Gray16) => ColorType::Gray(16),
            DicomColorType::RGB8(RGB8) => ColorType::RGB(8),
        }
    }
}

impl TryFrom<ColorType> for DicomColorType {
    type Error = TiffError;
    fn try_from(color_type: ColorType) -> Result<Self, Self::Error> {
        match color_type {
            ColorType::Gray(8) => Ok(DicomColorType::Gray8(Gray8)),
            ColorType::Gray(16) => Ok(DicomColorType::Gray16(Gray16)),
            ColorType::RGB(8) => Ok(DicomColorType::RGB8(RGB8)),
            _ => Err(TiffError::UnsupportedColorType { color_type }),
        }
    }
}

impl<R: Read + Seek> TryFrom<&mut Decoder<R>> for DicomColorType {
    type Error = TiffError;
    fn try_from(decoder: &mut Decoder<R>) -> Result<Self, Self::Error> {
        let color_type = decoder.colortype()?;
        Self::try_from(color_type)
    }
}

impl Clone for DicomColorType {
    fn clone(&self) -> Self {
        match self {
            DicomColorType::Gray8(Gray8) => DicomColorType::Gray8(Gray8),
            DicomColorType::Gray16(Gray16) => DicomColorType::Gray16(Gray16),
            DicomColorType::RGB8(RGB8) => DicomColorType::RGB8(RGB8),
        }
    }
}

impl std::fmt::Debug for DicomColorType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DicomColorType::Gray8(Gray8) => write!(f, "Gray8"),
            DicomColorType::Gray16(Gray16) => write!(f, "Gray16"),
            DicomColorType::RGB8(RGB8) => write!(f, "RGB8"),
        }
    }
}
