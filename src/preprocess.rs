use image::DynamicImage;
use image::GenericImageView;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use tiff::encoder::colortype::ColorType;
use tiff::encoder::compression::{Compression, Compressor, Packbits};
use tiff::encoder::TiffEncoder;
use tiff::TiffError;

use dicom::dictionary_std::tags;
use dicom::object::{FileDicomObject, InMemDicomObject, ReadError};
use dicom::pixeldata::PhotometricInterpretation;
use image::imageops::FilterType;
use snafu::{ResultExt, Snafu};
use tiff::encoder::colortype::{Gray16, RGB8};
use tiff::encoder::compression::{Lzw, Uncompressed};

use crate::color::{ColorError, DicomColorType};
use crate::metadata::{PreprocessingMetadata, Resolution};
use crate::transform::volume::VolumeError;
use crate::transform::{
    Crop, HandleVolume, Padding, PaddingDirection, Resize, Transform, VolumeHandler,
};

#[derive(Debug, Snafu)]
pub enum PreprocessError {
    DecodePixelData {
        #[snafu(source(from(VolumeError, Box::new)))]
        source: Box<VolumeError>,
    },
}

// Responsible for preprocessing image data before saving
pub struct Preprocessor {
    pub crop: bool,
    pub size: Option<(u32, u32)>,
    pub filter: FilterType,
    pub padding_direction: PaddingDirection,
    pub crop_max: bool,
    pub volume_handler: VolumeHandler,
}

impl Default for Preprocessor {
    fn default() -> Self {
        Preprocessor {
            crop: true,
            size: None,
            filter: FilterType::Triangle,
            padding_direction: PaddingDirection::Zero,
            crop_max: true,
            volume_handler: VolumeHandler::default(),
        }
    }
}

impl Preprocessor {
    fn get_crop(&self, images: &Vec<DynamicImage>) -> Option<Crop> {
        match self.crop {
            true => Some(Crop::new_from_images(
                &images.iter().collect::<Vec<_>>(),
                self.crop_max,
            )),
            false => None,
        }
    }

    fn get_resize(&self, images: &Vec<DynamicImage>) -> Option<Resize> {
        match self.size {
            Some((target_width, target_height)) => {
                let first_image = images.first().unwrap();
                let config = Resize::new(&first_image, target_width, target_height, self.filter);
                Some(config)
            }
            None => None,
        }
    }

    fn get_padding(&self, images: &Vec<DynamicImage>) -> Option<Padding> {
        match self.size {
            Some((target_width, target_height)) => {
                let first_image = images.first().unwrap();
                let config = Padding::new(
                    &first_image,
                    target_width,
                    target_height,
                    self.padding_direction,
                );
                Some(config)
            }
            None => None,
        }
    }

    // Decodes the pixel data and applies transformations
    pub fn prepare_image(
        &self,
        file: &FileDicomObject<InMemDicomObject>,
    ) -> Result<(Vec<DynamicImage>, PreprocessingMetadata), PreprocessError> {
        // Decode the pixel data, applying volume handling
        let image_data = self
            .volume_handler
            .decode_volume(file)
            .context(DecodePixelDataSnafu)?;

        // Try to determine the resolution from pixel spacing attributes
        let resolution = Resolution::try_from(file).ok();

        // Determine and apply crop
        let crop_config = self.get_crop(&image_data);
        let image_data = match &crop_config {
            Some(config) => config.apply_iter(image_data.into_iter()).collect(),
            None => image_data,
        };

        // Determine and apply resize, ensuring we also update the resolution
        let resize_config = self.get_resize(&image_data);
        let image_data = match &resize_config {
            Some(config) => config.apply_iter(image_data.into_iter()).collect(),
            None => image_data,
        };

        // Update the resolution if we resized
        let resolution = match (resolution, &resize_config) {
            (Some(res), Some(config)) => Some(config.apply(&res)),
            _ => None,
        };

        // Determine and apply padding
        let padding_config = self.get_padding(&image_data);
        let image_data = match &padding_config {
            Some(config) => config.apply_iter(image_data.into_iter()).collect(),
            None => image_data,
        };

        Ok((
            image_data,
            PreprocessingMetadata {
                crop: crop_config,
                resize: resize_config,
                padding: padding_config,
                resolution,
            },
        ))
    }
}
