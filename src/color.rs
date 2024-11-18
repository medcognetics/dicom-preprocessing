use dicom::dictionary_std::tags;
use dicom::object::{FileDicomObject, InMemDicomObject};
use dicom::pixeldata::PhotometricInterpretation;
use snafu::{ResultExt, Snafu};
use std::io::{Read, Seek};
use tiff::decoder::Decoder;
use tiff::encoder::colortype::{Gray16, Gray8, RGB8};
use tiff::ColorType;
use tiff::TiffError;

#[derive(Debug, Snafu)]
pub enum ColorError {
    MissingProperty {
        name: &'static str,
    },
    CastPropertyValue {
        name: &'static str,
        #[snafu(source(from(dicom::core::value::CastValueError, Box::new)))]
        source: Box<dicom::core::value::CastValueError>,
    },
    UnsupportedPhotometricInterpretation {
        bits_allocated: u16,
        photometric_interpretation: PhotometricInterpretation,
    },
    UnsupportedColorType {
        color_type: ColorType,
    },
    ParseFromTiff {
        #[snafu(source(from(TiffError, Box::new)))]
        source: Box<TiffError>,
    },
}

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
    ) -> Result<Self, ColorError> {
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
                Err(ColorError::UnsupportedPhotometricInterpretation {
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
    type Error = ColorError;

    /// Read the BitsAllocated and PhotometricInterpretation tags from the DICOM file
    /// and infer the appropriate color type.
    fn try_from(file: &FileDicomObject<InMemDicomObject>) -> Result<Self, Self::Error> {
        let bits_allocated = file
            .get(tags::BITS_ALLOCATED)
            .ok_or(ColorError::MissingProperty {
                name: "Bits Allocated",
            })?
            .value()
            .uint16()
            .context(CastPropertyValueSnafu {
                name: "Bits Allocated",
            })?;
        let photometric_interpretation = file
            .get(tags::PHOTOMETRIC_INTERPRETATION)
            .ok_or(ColorError::MissingProperty {
                name: "Photometric Interpretation",
            })?
            .value()
            .string()
            .context(CastPropertyValueSnafu {
                name: "Photometric Interpretation",
            })?;
        let photometric_interpretation =
            PhotometricInterpretation::from(photometric_interpretation.trim());

        DicomColorType::try_new(bits_allocated, photometric_interpretation)
    }
}

// NOTE: There is a trait ColorType in the encoder module, and an enum ColorType at the top level.
impl Into<ColorType> for DicomColorType {
    fn into(self) -> ColorType {
        match self {
            DicomColorType::Gray8(Gray8) => ColorType::Gray(8),
            DicomColorType::Gray16(Gray16) => ColorType::Gray(16),
            DicomColorType::RGB8(RGB8) => ColorType::RGB(8),
        }
    }
}

impl TryFrom<ColorType> for DicomColorType {
    type Error = ColorError;
    fn try_from(color_type: ColorType) -> Result<Self, Self::Error> {
        match color_type {
            ColorType::Gray(8) => Ok(DicomColorType::Gray8(Gray8)),
            ColorType::Gray(16) => Ok(DicomColorType::Gray16(Gray16)),
            ColorType::RGB(8) => Ok(DicomColorType::RGB8(RGB8)),
            _ => Err(ColorError::UnsupportedColorType { color_type }),
        }
    }
}

impl<R: Read + Seek> TryFrom<&mut Decoder<R>> for DicomColorType {
    type Error = ColorError;
    fn try_from(decoder: &mut Decoder<R>) -> Result<Self, Self::Error> {
        let color_type = decoder.colortype().context(ParseFromTiffSnafu)?;
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
