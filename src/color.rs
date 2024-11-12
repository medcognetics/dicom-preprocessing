use dicom::dictionary_std::tags;
use dicom::object::{FileDicomObject, InMemDicomObject};
use dicom::pixeldata::PhotometricInterpretation;
use snafu::{ResultExt, Snafu};
use tiff::encoder::colortype::{Gray16, RGB8};

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
}

/// The color types we expect to encounter in DICOM files and support processing of
pub enum DicomColorType {
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
            (16, PhotometricInterpretation::Monochrome1)
            | (16, PhotometricInterpretation::Monochrome2) => Ok(DicomColorType::Gray16(Gray16)),
            (8, PhotometricInterpretation::Rgb) => Ok(DicomColorType::RGB8(RGB8)),
            (bits_allocated, photometric_interpretation) => {
                Err(ColorError::UnsupportedPhotometricInterpretation {
                    bits_allocated,
                    photometric_interpretation,
                })
            }
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
