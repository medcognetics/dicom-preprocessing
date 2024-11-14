use image::DynamicImage;

use dicom::object::{FileDicomObject, InMemDicomObject};
use image::imageops::FilterType;
use rayon::{ThreadPoolBuildError, ThreadPoolBuilder};
use snafu::{ResultExt, Snafu};
use std::num::NonZero;
use std::thread::available_parallelism;

use crate::metadata::{PreprocessingMetadata, Resolution};
use crate::transform::volume::{get_number_of_frames, VolumeError};
use crate::transform::{
    Crop, HandleVolume, Padding, PaddingDirection, Resize, Transform, VolumeHandler,
};

#[derive(Debug, Snafu)]
pub enum PreprocessError {
    DecodePixelData {
        #[snafu(source(from(VolumeError, Box::new)))]
        source: Box<VolumeError>,
    },
    BuildThreadPool {
        #[snafu(source(from(ThreadPoolBuildError, Box::new)))]
        source: Box<ThreadPoolBuildError>,
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

    /// Decodes the pixel data and applies transformations
    pub fn prepare_image(
        &self,
        file: &FileDicomObject<InMemDicomObject>,
        parallelism: usize,
    ) -> Result<(Vec<DynamicImage>, PreprocessingMetadata), PreprocessError> {
        // Determine the number of threads to use for parallel processing across multiple frames.
        // This is the minimum of the number of frames, the number of threads requested, and the
        // number of threads available.
        let num_frames = get_number_of_frames(file).context(DecodePixelDataSnafu)?;
        let max_parallelism = available_parallelism()
            .unwrap_or(NonZero::new(1).unwrap())
            .get();
        let parallelism = parallelism.min(max_parallelism).min(num_frames as usize);

        // Run decoding and volume handling
        let image_data = match parallelism {
            0 | 1 => self.volume_handler.decode_volume(file),
            p => ThreadPoolBuilder::new()
                .num_threads(p)
                .build()
                .context(BuildThreadPoolSnafu)?
                .install(|| self.volume_handler.par_decode_volume(file)),
        }
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

        let num_frames = image_data.len().into();

        Ok((
            image_data,
            PreprocessingMetadata {
                crop: crop_config,
                resize: resize_config,
                padding: padding_config,
                resolution,
                num_frames,
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dicom::object::open_file;

    use crate::volume::{CentralSlice, KeepVolume, VolumeHandler};
    use image::GenericImageView;
    use rstest::rstest;

    #[rstest]
    #[case(
        "pydicom/CT_small.dcm", 
        Preprocessor {
            crop: true,
            size: Some((64, 64)),
            filter: FilterType::Nearest,
            padding_direction: PaddingDirection::default(),
            crop_max: false,
            volume_handler: VolumeHandler::Keep(KeepVolume),
        }
    )]
    #[case(
        "pydicom/MR_small.dcm", 
        Preprocessor {
            crop: false,
            size: None,
            filter: FilterType::Nearest,
            padding_direction: PaddingDirection::default(),
            crop_max: false,
            volume_handler: VolumeHandler::CentralSlice(CentralSlice),
        }
    )]
    #[case(
        "pydicom/JPEG2000_UNC.dcm",
        Preprocessor {
            crop: false,
            size: None,
            filter: FilterType::Nearest,
            padding_direction: PaddingDirection::default(),
            crop_max: false,
            volume_handler: VolumeHandler::CentralSlice(CentralSlice),
        }
    )]
    #[case(
        "pydicom/US1_J2KI.dcm",
        Preprocessor {
            crop: false,
            size: None,
            filter: FilterType::Nearest,
            padding_direction: PaddingDirection::default(),
            crop_max: false,
            volume_handler: VolumeHandler::CentralSlice(CentralSlice),
        }
    )]
    #[case(
        "pydicom/JPGLosslessP14SV1_1s_1f_8b.dcm",
        Preprocessor {
            crop: false,
            size: None,
            filter: FilterType::Nearest,
            padding_direction: PaddingDirection::default(),
            crop_max: false,
            volume_handler: VolumeHandler::CentralSlice(CentralSlice),
        }
    )]
    fn test_preprocess(#[case] dicom_file_path: &str, #[case] config: Preprocessor) {
        let dicom_file = open_file(&dicom_test_files::path(dicom_file_path).unwrap()).unwrap();

        // Run preprocessing
        let (images, _) = config.prepare_image(&dicom_file, 1).unwrap();
        assert_eq!(images.len(), 1);

        // Check the image size
        if let Some((exp_width, exp_height)) = config.size {
            for image in images.iter() {
                let (width, height) = image.dimensions();
                assert_eq!(width, exp_width);
                assert_eq!(height, exp_height);
            }
        }
    }
}
