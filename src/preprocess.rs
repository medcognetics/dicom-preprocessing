use clap::builder::styling::Color;
use clap::ValueEnum;
use dicom::core::prelude::*;
use image::{DynamicImage, GenericImageView};
use log::error;
use std::ffi::CString;
use std::fs::File;
use std::io::{Seek, Write};
use std::{borrow::Cow, path::PathBuf, str::FromStr};
use tiff::encoder::colortype::ColorType;
use tiff::encoder::compression::{Compression, CompressionAlgorithm, Compressor};
use tiff::encoder::{ImageEncoder, Rational, TiffEncoder, TiffKind};
use tiff::tags::CompressionMethod;
use tiff::tags::Tag;
use tiff::TiffError;

use clap::error::ErrorKind;
use clap::Parser;
use dicom::core::prelude::*;
use dicom::dictionary_std::{tags, uids};
use dicom::object::{open_file, FileDicomObject, InMemDicomObject, ReadError};
use dicom::pixeldata::{ConvertOptions, DecodedPixelData, PixelDecoder};
use image::imageops::FilterType;
use snafu::{OptionExt, Report, ResultExt, Snafu, Whatever};

use crate::crop::Crop;
use crate::pad::{Padding, PaddingDirection};
use crate::resize::{DisplayFilterType, Resize};
use crate::traits::{Transform, WriteTags};

const VERSION: &str = concat!("dicom-preprocessing==", env!("CARGO_PKG_VERSION"), "\0");

#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(display("could not read DICOM file {}", path.display()))]
    ReadFile {
        #[snafu(source(from(ReadError, Box::new)))]
        source: Box<ReadError>,
        path: PathBuf,
    },
    /// failed to decode pixel data
    DecodePixelData {
        #[snafu(source(from(dicom::pixeldata::Error, Box::new)))]
        source: Box<dicom::pixeldata::Error>,
    },
    /// missing offset table entry for frame #{frame_number}
    MissingOffsetEntry {
        frame_number: u32,
    },
    /// missing key property {name}
    MissingProperty {
        name: &'static str,
    },
    /// property {name} contains an invalid value
    InvalidPropertyValue {
        name: &'static str,
        #[snafu(source(from(dicom::core::value::ConvertValueError, Box::new)))]
        source: Box<dicom::core::value::ConvertValueError>,
    },
    /// pixel data of frame #{frame_number} is out of bounds
    FrameOutOfBounds {
        frame_number: u32,
    },
    ConvertImage {
        #[snafu(source(from(dicom::pixeldata::Error, Box::new)))]
        source: Box<dicom::pixeldata::Error>,
    },
    #[snafu(display("could not open TIFF file {}", path.display()))]
    OpenTiff {
        #[snafu(source(from(TiffError, Box::new)))]
        source: Box<TiffError>,
        path: PathBuf,
    },
    SaveData {
        source: std::io::Error,
    },
    UnexpectedPixelData,
    /// failed to parse pixel spacing value
    ParsePixelSpacingError {
        #[snafu(source(from(std::num::ParseFloatError, Box::new)))]
        source: Box<std::num::ParseFloatError>,
    },
    WriteTags {
        name: &'static str,
        #[snafu(source(from(TiffError, Box::new)))]
        source: Box<TiffError>,
    },
}

pub struct Resolution {
    pub pixels_per_mm_x: f32,
    pub pixels_per_mm_y: f32,
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
            n: (self.pixels_per_mm_x * 10.0) as u32,
            d: 1,
        });
        tiff.y_resolution(Rational {
            n: (self.pixels_per_mm_y * 10.0) as u32,
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

impl TryFrom<&FileDicomObject<InMemDicomObject>> for Resolution {
    type Error = Error;

    // Extract
    fn try_from(file: &FileDicomObject<InMemDicomObject>) -> Result<Self, Self::Error> {
        // Read the spacing
        let spacing = file
            .get(tags::PIXEL_SPACING)
            .or_else(|| file.get(tags::IMAGER_PIXEL_SPACING))
            .ok_or(Error::MissingProperty {
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
            .ok_or(Error::MissingProperty {
                name: "Pixel Spacing",
            })?
            .parse::<f32>()
            .map_err(|e| Error::ParsePixelSpacingError {
                source: Box::new(e),
            })?;
        let pixel_spacing_mm_x = spacing_iter
            .next()
            .ok_or(Error::MissingProperty {
                name: "Pixel Spacing",
            })?
            .parse::<f32>()
            .map_err(|e| Error::ParsePixelSpacingError {
                source: Box::new(e),
            })?;

        // Convert to pixels per mm
        Ok(Resolution {
            pixels_per_mm_x: 1.0 / pixel_spacing_mm_x,
            pixels_per_mm_y: 1.0 / pixel_spacing_mm_y,
        })
    }
}

pub struct PreprocessingOptions {
    crop: bool,
    size: Option<(u32, u32)>,
    filter: FilterType,
    padding_direction: PaddingDirection,
    compression: Compressor,
}

pub trait Preprocessing {
    fn preprocess(
        &self,
        output: PathBuf,
        crop: bool,
        size: Option<(u32, u32)>,
        filter: FilterType,
        padding_direction: PaddingDirection,
        compression: impl Compression,
    ) -> Result<(), Error>;
}

pub fn preprocess<C>(
    file: &FileDicomObject<InMemDicomObject>,
    output: PathBuf,
    crop: bool,
    size: Option<(u32, u32)>,
    filter: FilterType,
    padding_direction: PaddingDirection,
    compression: C,
) -> Result<(), Error>
where
    C: Compression + Copy,
{
    // Decode the pixel data and extract the dimensions
    let decoded = file.decode_pixel_data().context(DecodePixelDataSnafu)?;
    let mut rows = decoded.rows();
    let mut columns = decoded.columns();
    let number_of_frames = decoded.number_of_frames();

    // Try to determine the resolution from pixel spacing attributes
    let resolution = Resolution::try_from(file).ok();

    // Read each frame and scan the frames to determine the crop
    let (image_data, crop_config) = match number_of_frames {
        1 => {
            let img = decoded.to_dynamic_image(0).context(DecodePixelDataSnafu)?;
            let crop_config = match crop {
                true => Some(Crop::from(&img)),
                false => None,
            };
            (vec![img].into_iter(), crop_config)
        }
        _ => {
            let mut image_data = Vec::with_capacity(number_of_frames as usize);
            for frame_number in 0..number_of_frames {
                image_data.push(
                    decoded
                        .to_dynamic_image(frame_number)
                        .context(DecodePixelDataSnafu)?,
                );
            }
            let crop_config = match crop {
                true => Some(Crop::from(&image_data.iter().collect::<Vec<_>>()[..])),
                false => None,
            };
            (image_data.into_iter(), crop_config)
        }
    };

    // Apply crop
    let image_data = image_data
        .map(|img| match &crop_config {
            Some(config) => config.apply(&img),
            None => img,
        })
        .collect::<Vec<_>>();

    // Determine and apply resize
    let resize_config = match size {
        Some((target_width, target_height)) => {
            let first_image = image_data.first().unwrap();
            let config = Resize::new(&first_image, target_width, target_height, filter);
            Some(config)
        }
        None => None,
    };
    let image_data = image_data
        .into_iter()
        .map(|img| match &resize_config {
            Some(config) => config.apply(&img),
            None => img,
        })
        .collect::<Vec<_>>();

    // Determine and apply padding
    let padding_config = match size {
        Some((target_width, target_height)) => {
            let first_image = image_data.first().unwrap();
            let config = Padding::new(&first_image, target_width, target_height, padding_direction);
            Some(config)
        }
        None => None,
    };
    let image_data = image_data
        .into_iter()
        .map(|img| match &padding_config {
            Some(config) => config.apply(&img),
            None => img,
        })
        .collect::<Vec<_>>();

    // Open the TIFF file
    let mut tiff_encoder = TiffEncoder::new(File::create(&output).context(SaveDataSnafu)?)
        .context(OpenTiffSnafu {
            path: output.clone(),
        })?;

    // Preprocess each frame
    for img in image_data.iter() {
        // Create the TIFF encoder for single frame
        let (columns, rows) = img.dimensions();
        let mut tiff = tiff_encoder
            .new_image_with_compression::<tiff::encoder::colortype::Gray16, _>(
                columns,
                rows,
                compression,
            )
            .context(OpenTiffSnafu {
                path: output.clone(),
            })?;

        // Write some tags
        tiff.encoder()
            .write_tag(Tag::Software, VERSION.as_bytes())
            .context(WriteTagsSnafu { name: "Software" })?;

        // Write transform related tags
        if let Some(resolution) = &resolution {
            resolution
                .write_tags(&mut tiff)
                .context(WriteTagsSnafu { name: "Resolution" })?;
        }
        if let Some(crop_config) = &crop_config {
            crop_config
                .write_tags(&mut tiff)
                .context(WriteTagsSnafu { name: "Crop" })?;
        }
        if let Some(resize_config) = &resize_config {
            resize_config
                .write_tags(&mut tiff)
                .context(WriteTagsSnafu { name: "Resize" })?;
        }
        if let Some(padding_config) = &padding_config {
            padding_config
                .write_tags(&mut tiff)
                .context(WriteTagsSnafu { name: "Padding" })?;
        }

        // Write the image data
        let bytes = img.as_luma16().unwrap();
        tiff.write_data(bytes).context(OpenTiffSnafu {
            path: output.clone(),
        })?
    }
    Ok(())
}
