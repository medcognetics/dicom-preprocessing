use std::io::{Read, Seek, Write};
use tiff::decoder::Decoder;
use tiff::encoder::colortype::ColorType;
use tiff::encoder::compression::Compression;
use tiff::encoder::{ImageEncoder, Rational, TiffKind};
use tiff::tags::ResolutionUnit;
use tiff::TiffError;

use dicom::dictionary_std::tags;
use dicom::object::{FileDicomObject, InMemDicomObject};
use snafu::{ResultExt, Snafu};

use crate::metadata::WriteTags;

const MM_PER_CM: f32 = 10.0;
const MM_PER_IN: f32 = 25.4;

#[derive(Debug, Snafu)]
pub enum ResolutionError {
    MissingProperty {
        name: &'static str,
    },
    InvalidPropertyValue {
        name: &'static str,
        #[snafu(source(from(dicom::core::value::ConvertValueError, Box::new)))]
        source: Box<dicom::core::value::ConvertValueError>,
    },
    ParsePixelSpacing {
        #[snafu(source(from(std::num::ParseFloatError, Box::new)))]
        source: Box<std::num::ParseFloatError>,
    },
    WriteTags {
        name: &'static str,
        #[snafu(source(from(TiffError, Box::new)))]
        source: Box<TiffError>,
    },
    InvalidResolutionUnit {
        #[snafu(source(from(TiffError, Box::new)))]
        source: Box<TiffError>,
    },
    CastResolutionUnit {
        val: u16,
    },
    InvalidResolution {
        #[snafu(source(from(TiffError, Box::new)))]
        source: Box<TiffError>,
    },
}

#[derive(Debug, PartialEq)]
pub struct Resolution {
    pub pixels_per_mm_x: f32,
    pub pixels_per_mm_y: f32,
}

impl<T> TryFrom<&mut Decoder<T>> for Resolution
where
    T: Read + Seek,
{
    type Error = ResolutionError;

    // Extract resolution metadata from a TIFF decoder
    fn try_from(decoder: &mut Decoder<T>) -> Result<Self, ResolutionError> {
        // Parse resolution unit
        let unit = decoder
            .get_tag(tiff::tags::Tag::ResolutionUnit)
            .map_err(|_| ResolutionError::MissingProperty {
                name: "Resolution Unit",
            })?
            .into_u16()
            .map_err(|s| ResolutionError::InvalidResolutionUnit {
                source: Box::new(s),
            })?;
        let unit = ResolutionUnit::from_u16(unit)
            .ok_or(ResolutionError::CastResolutionUnit { val: unit })?;

        // Parse x resolution
        let x_resolution = decoder
            .get_tag_u32_vec(tiff::tags::Tag::XResolution)
            .map_err(|s| ResolutionError::InvalidResolution {
                source: Box::new(s),
            })?;
        let x_resolution = x_resolution[0] as f32 / x_resolution[1] as f32;

        // Parse y resolution
        let y_resolution = decoder
            .get_tag_u32_vec(tiff::tags::Tag::YResolution)
            .map_err(|s| ResolutionError::InvalidResolution {
                source: Box::new(s),
            })?;
        let y_resolution = y_resolution[0] as f32 / y_resolution[1] as f32;

        // Convert to pixels per mm
        let (x_resolution, y_resolution) = match unit {
            ResolutionUnit::Centimeter => (x_resolution / MM_PER_CM, y_resolution / MM_PER_CM),
            ResolutionUnit::Inch => (x_resolution / MM_PER_IN, y_resolution / MM_PER_IN),
            _ => (x_resolution, y_resolution),
        };

        Ok(Resolution {
            pixels_per_mm_x: x_resolution,
            pixels_per_mm_y: y_resolution,
        })
    }
}

impl Resolution {
    pub fn scale(&self, value: f32) -> Self {
        Resolution {
            pixels_per_mm_x: self.pixels_per_mm_x * value,
            pixels_per_mm_y: self.pixels_per_mm_y * value,
        }
    }
}

impl WriteTags for Resolution {
    fn write_tags<W, C, K, D>(&self, tiff: &mut ImageEncoder<W, C, K, D>) -> Result<(), TiffError>
    where
        W: Write + Seek,
        C: ColorType,
        K: TiffKind,
        D: Compression,
    {
        tiff.x_resolution(Rational {
            n: (self.pixels_per_mm_x * MM_PER_CM) as u32,
            d: 1,
        });
        tiff.y_resolution(Rational {
            n: (self.pixels_per_mm_y * MM_PER_CM) as u32,
            d: 1,
        });
        tiff.resolution_unit(tiff::tags::ResolutionUnit::Centimeter);
        Ok(())
    }
}

impl From<Resolution> for (f32, f32) {
    fn from(resolution: Resolution) -> Self {
        (resolution.pixels_per_mm_x, resolution.pixels_per_mm_y)
    }
}

impl From<(f32, f32)> for Resolution {
    fn from((pixels_per_mm_x, pixels_per_mm_y): (f32, f32)) -> Self {
        Resolution {
            pixels_per_mm_x,
            pixels_per_mm_y,
        }
    }
}

impl TryFrom<&FileDicomObject<InMemDicomObject>> for Resolution {
    type Error = ResolutionError;

    fn try_from(file: &FileDicomObject<InMemDicomObject>) -> Result<Self, Self::Error> {
        // Read the spacing, first from the Pixel Spacing tag, then from the Imager Pixel Spacing tag.
        let spacing = file
            .get(tags::PIXEL_SPACING)
            .or_else(|| file.get(tags::IMAGER_PIXEL_SPACING))
            .ok_or(ResolutionError::MissingProperty {
                name: "Pixel Spacing",
            })?
            .value()
            .to_str()
            .context(InvalidPropertyValueSnafu {
                name: "Pixel Spacing",
            })?;

        // Parse spacing into x and y. First value is row spacing (y)
        let mut spacing_iter = spacing.split('\\');
        let pixel_spacing_mm_y = spacing_iter
            .next()
            .ok_or(ResolutionError::MissingProperty {
                name: "Pixel Spacing",
            })?
            .parse::<f32>()
            .context(ParsePixelSpacingSnafu)?;

        let pixel_spacing_mm_x = spacing_iter
            .next()
            .ok_or(ResolutionError::MissingProperty {
                name: "Pixel Spacing",
            })?
            .parse::<f32>()
            .context(ParsePixelSpacingSnafu)?;

        // Convert to pixels per mm
        Ok((1.0 / pixel_spacing_mm_x, 1.0 / pixel_spacing_mm_y).into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dicom::object::open_file;
    use rstest::rstest;
    use std::fs::File;

    use tempfile::tempdir;
    use tiff::decoder::Decoder as TiffDecoder;
    use tiff::encoder::TiffEncoder;

    #[rstest]
    #[case("pydicom/CT_small.dcm", Resolution { pixels_per_mm_x: 1.5117888, pixels_per_mm_y: 1.5117888 })]
    fn test_try_from(#[case] dicom_path: &str, #[case] expected: Resolution) {
        let dicom_file = dicom_test_files::path(dicom_path).unwrap();
        let dicom_file = open_file(dicom_file).unwrap();
        let result = Resolution::try_from(&dicom_file).unwrap();
        assert_eq!(result, expected);
    }

    #[rstest]
    #[case(
        Resolution { pixels_per_mm_x: 1.5117888, pixels_per_mm_y: 1.5117888 },
        1.5,
        1.5,
    )]
    fn test_write_tags(
        #[case] resolution: Resolution,
        #[case] x_resolution: f32,
        #[case] y_resolution: f32,
    ) {
        // Prepare the TIFF
        let temp_dir = tempdir().unwrap();
        let temp_file_path = temp_dir.path().join("temp.tif");
        let mut tiff = TiffEncoder::new(File::create(temp_file_path.clone()).unwrap()).unwrap();
        let mut img = tiff
            .new_image::<tiff::encoder::colortype::Gray16>(1, 1)
            .unwrap();

        // Write the tags
        resolution.write_tags(&mut img).unwrap();

        // Write some dummy image data
        let data: Vec<u16> = vec![0; 2];
        img.write_data(data.as_slice()).unwrap();

        // Read the TIFF back
        let mut tiff = TiffDecoder::new(File::open(temp_file_path).unwrap()).unwrap();
        let actual = Resolution::try_from(&mut tiff).unwrap();
        assert_eq!(x_resolution, actual.pixels_per_mm_x);
        assert_eq!(y_resolution, actual.pixels_per_mm_y);
    }

    #[rstest]
    #[case(Resolution { pixels_per_mm_x: 1.0, pixels_per_mm_y: 1.0 }, 2.0, Resolution { pixels_per_mm_x: 2.0, pixels_per_mm_y: 2.0 })]
    #[case(Resolution { pixels_per_mm_x: 1.5, pixels_per_mm_y: 1.5 }, 0.5, Resolution { pixels_per_mm_x: 0.75, pixels_per_mm_y: 0.75 })]
    #[case(Resolution { pixels_per_mm_x: 2.0, pixels_per_mm_y: 2.0 }, 1.0, Resolution { pixels_per_mm_x: 2.0, pixels_per_mm_y: 2.0 })]
    fn test_scale(
        #[case] resolution: Resolution,
        #[case] scale_factor: f32,
        #[case] expected: Resolution,
    ) {
        let result = resolution.scale(scale_factor);
        assert_eq!(result, expected);
    }
}
