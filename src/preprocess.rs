use dicom::core::header::HasLength;
use image::DynamicImage;

use dicom::dictionary_std::tags;
use dicom::object::{FileDicomObject, InMemDicomObject};

use crate::errors::DicomError;
use crate::metadata::{PreprocessingMetadata, Resolution};
use crate::transform::resize;
use crate::transform::{
    Crop, HandleVolume, Padding, PaddingDirection, Resize, Transform, VolumeHandler,
};

// Responsible for preprocessing image data before saving
#[derive(Debug, Clone, Copy)]
pub struct Preprocessor {
    pub crop: bool,
    pub size: Option<(u32, u32)>,
    pub filter: resize::FilterType,
    pub padding_direction: PaddingDirection,
    pub crop_max: bool,
    pub volume_handler: VolumeHandler,
    pub use_components: bool,
    pub use_padding: bool,
    pub border_frac: Option<f32>,
}

impl Default for Preprocessor {
    fn default() -> Self {
        Preprocessor {
            crop: true,
            size: None,
            filter: resize::FilterType::Triangle,
            padding_direction: PaddingDirection::Zero,
            crop_max: true,
            volume_handler: VolumeHandler::default(),
            use_components: true,
            use_padding: true,
            border_frac: None,
        }
    }
}

impl Preprocessor {
    fn get_crop(&self, images: &[DynamicImage]) -> Option<Crop> {
        match self.crop {
            true => Some(Crop::new_from_images(
                &images.iter().collect::<Vec<_>>(),
                self.crop_max,
                self.use_components,
                self.border_frac,
            )),
            false => None,
        }
    }

    fn get_resize(&self, images: &[DynamicImage]) -> Option<Resize> {
        match self.size {
            Some((target_width, target_height)) => {
                let first_image = images.first().unwrap();
                let config = Resize::new(first_image, target_width, target_height, self.filter);
                Some(config)
            }
            None => None,
        }
    }

    fn get_padding(&self, images: &[DynamicImage]) -> Option<Padding> {
        match (self.use_padding, self.size) {
            (true, Some((target_width, target_height))) => {
                let first_image = images.first().unwrap();
                let config = Padding::new(
                    first_image,
                    target_width,
                    target_height,
                    self.padding_direction,
                );
                Some(config)
            }
            _ => None,
        }
    }

    /// Decoding will error if the VOILUT tag is present but empty.
    /// This function removes the tag if it is empty.
    fn sanitize_voi_lut_function(
        dcm: &mut FileDicomObject<InMemDicomObject>,
    ) -> &FileDicomObject<InMemDicomObject> {
        let voi_lut_function = dcm.get(tags::VOILUT_FUNCTION);
        if let Some(voi_lut_function) = voi_lut_function {
            if voi_lut_function.is_empty() {
                dcm.remove_element(tags::VOILUT_FUNCTION);
            }
        }
        dcm
    }

    /// Performs sanitization of the DICOM object to ensure decoding will succeed.
    pub fn sanitize_dicom(dcm: &mut FileDicomObject<InMemDicomObject>) {
        Self::sanitize_voi_lut_function(dcm);
    }

    /// Decodes the pixel data and applies transformations.
    /// When `parallel` is true, the pixel data is decoded in parallel using rayon
    pub fn prepare_image(
        &self,
        file: &FileDicomObject<InMemDicomObject>,
        parallel: bool,
    ) -> Result<(Vec<DynamicImage>, PreprocessingMetadata), DicomError> {
        // Run decoding and volume handling
        let image_data = match parallel {
            false => self.volume_handler.decode_volume(file),
            true => self.volume_handler.par_decode_volume(file),
        }?;

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

// Add this extension method for testing
impl Preprocessor {
    #[cfg(test)]
    fn prepare_image_for_test(
        &self,
        images: &Vec<DynamicImage>,
    ) -> Result<(Vec<DynamicImage>, PreprocessingMetadata), DicomError> {
        // Try to determine the resolution (none for test images)
        let resolution = None;

        // Determine and apply crop
        let crop_config = self.get_crop(images);
        let image_data = match &crop_config {
            Some(config) => config.apply_iter(images.clone().into_iter()).collect(),
            None => images.clone(),
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
    use dicom::core::{DataElement, PrimitiveValue, VR};
    use dicom::dictionary_std::tags;

    use dicom::object::open_file;

    use crate::volume::{CentralSlice, KeepVolume, VolumeHandler};
    use image::{GenericImageView, RgbaImage};
    use rstest::rstest;

    #[rstest]
    #[case(
        "pydicom/CT_small.dcm", 
        Preprocessor {
            crop: true,
            size: Some((64, 64)),
            filter: resize::FilterType::Nearest,
            padding_direction: PaddingDirection::default(),
            crop_max: false,
            volume_handler: VolumeHandler::Keep(KeepVolume),
            use_components: true,
            use_padding: true,
            border_frac: None,
        },
        false
    )]
    #[case(
        "pydicom/MR_small.dcm", 
        Preprocessor {
            crop: false,
            size: None,
            filter: resize::FilterType::Nearest,
            padding_direction: PaddingDirection::default(),
            crop_max: false,
            volume_handler: VolumeHandler::CentralSlice(CentralSlice),
            use_components: true,
            use_padding: true,
            border_frac: None,
        },
        true
    )]
    #[case(
        "pydicom/JPEG2000_UNC.dcm",
        Preprocessor {
            crop: false,
            size: None,
            filter: resize::FilterType::Nearest,
            padding_direction: PaddingDirection::default(),
            crop_max: false,
            volume_handler: VolumeHandler::CentralSlice(CentralSlice),
            use_components: true,
            use_padding: true,
            border_frac: None,
        },
        false
    )]
    #[case(
        "pydicom/US1_J2KI.dcm",
        Preprocessor {
            crop: false,
            size: None,
            filter: resize::FilterType::Nearest,
            padding_direction: PaddingDirection::default(),
            crop_max: false,
            volume_handler: VolumeHandler::CentralSlice(CentralSlice),
            use_components: true,
            use_padding: true,
            border_frac: None,
        },
        false
    )]
    #[case(
        "pydicom/JPGLosslessP14SV1_1s_1f_8b.dcm",
        Preprocessor {
            crop: false,
            size: None,
            filter: resize::FilterType::Nearest,
            padding_direction: PaddingDirection::default(),
            crop_max: false,
            volume_handler: VolumeHandler::CentralSlice(CentralSlice),
            use_components: true,
            use_padding: true,
            border_frac: None,
        },
        false
    )]
    #[case(
        "pydicom/SC_rgb.dcm",
        Preprocessor {
            crop: false,
            size: None,
            filter: resize::FilterType::Nearest,
            padding_direction: PaddingDirection::default(),
            crop_max: false,
            volume_handler: VolumeHandler::CentralSlice(CentralSlice),
            use_components: true,
            use_padding: true,
            border_frac: None,
        },
        false
    )]
    fn test_preprocess(
        #[case] dicom_file_path: &str,
        #[case] config: Preprocessor,
        #[case] parallel: bool,
    ) {
        let dicom_file = open_file(dicom_test_files::path(dicom_file_path).unwrap()).unwrap();

        // Run preprocessing
        let (images, _) = config.prepare_image(&dicom_file, parallel).unwrap();
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

    #[rstest]
    #[case(DataElement::new(tags::VOILUT_FUNCTION, VR::LO, PrimitiveValue::Empty))]
    fn test_sanitize_dicom(#[case] elem: DataElement<InMemDicomObject>) {
        let mut dicom_file =
            open_file(dicom_test_files::path("pydicom/CT_small.dcm").unwrap()).unwrap();
        dicom_file.put_element(elem);
        assert!(dicom_file.get(tags::VOILUT_FUNCTION).is_some());
        Preprocessor::sanitize_dicom(&mut dicom_file);
        assert!(dicom_file.get(tags::VOILUT_FUNCTION).is_none());
    }

    #[rstest]
    #[case(
        vec![
            vec![1, 1],
            vec![1, 1],
        ],
        (4, 4),
        true,
        Some(Padding { left: 1, top: 1, right: 1, bottom: 1 })
    )]
    #[case(
        vec![
            vec![1, 1],
            vec![1, 1],
        ],
        (4, 4),
        false,
        None
    )]
    fn test_padding_enabled(
        #[case] pixels: Vec<Vec<u8>>,
        #[case] (target_width, target_height): (u32, u32),
        #[case] use_padding: bool,
        #[case] expected_padding: Option<Padding>,
    ) {
        // Create image from pixels
        let width = pixels[0].len() as u32;
        let height = pixels.len() as u32;
        let mut img = RgbaImage::new(width, height);
        for (y, row) in pixels.iter().enumerate() {
            for (x, &value) in row.iter().enumerate() {
                img.put_pixel(x as u32, y as u32, image::Rgba([value, value, value, 255]));
            }
        }
        let dynamic_image = DynamicImage::ImageRgba8(img);

        let preprocessor = Preprocessor {
            crop: false,
            size: Some((target_width, target_height)),
            filter: resize::FilterType::Nearest,
            padding_direction: PaddingDirection::Center,
            crop_max: false,
            volume_handler: VolumeHandler::default(),
            use_components: true,
            use_padding,
            border_frac: None,
        };

        let padding = preprocessor.get_padding(&[dynamic_image]);
        assert_eq!(padding, expected_padding);
    }

    #[rstest]
    #[case(
        vec![
            vec![0, 0, 0, 0, 0, 0],
            vec![0, 0, 1, 1, 0, 0],
            vec![0, 0, 1, 1, 0, 0],
            vec![0, 0, 1, 1, 0, 0],
            vec![0, 0, 1, 1, 0, 0],
            vec![0, 0, 0, 0, 0, 0],
        ],
        Some(0.2),
        (1, 0, 4, 5)  // Actual behavior with border exclusion
    )]
    #[case(
        vec![
            vec![0, 0, 0, 0, 0, 0],
            vec![0, 0, 1, 1, 0, 0],
            vec![0, 0, 1, 1, 0, 0],
            vec![0, 0, 1, 1, 0, 0],
            vec![0, 0, 1, 1, 0, 0],
            vec![0, 0, 0, 0, 0, 0],
        ],
        None,
        (2, 1, 2, 4)  // Without border exclusion, should crop to content
    )]
    fn test_preprocessor_border_frac(
        #[case] pixels: Vec<Vec<u8>>,
        #[case] border_frac: Option<f32>,
        #[case] expected_crop: (u32, u32, u32, u32),
    ) {
        // Create image from pixels
        let width = pixels[0].len() as u32;
        let height = pixels.len() as u32;
        let mut img = RgbaImage::new(width, height);
        for (y, row) in pixels.iter().enumerate() {
            for (x, &value) in row.iter().enumerate() {
                img.put_pixel(x as u32, y as u32, image::Rgba([value, value, value, 255]));
            }
        }
        let dynamic_image = DynamicImage::ImageRgba8(img);

        let preprocessor = Preprocessor {
            crop: true,
            size: None,
            filter: resize::FilterType::Nearest,
            padding_direction: PaddingDirection::Center,
            crop_max: false,
            volume_handler: VolumeHandler::default(),
            use_components: true,
            use_padding: true,
            border_frac,
        };

        let (processed_images, metadata) = preprocessor
            .prepare_image_for_test(&vec![dynamic_image])
            .unwrap();

        assert_eq!(processed_images.len(), 1);

        if let Some(crop) = metadata.crop {
            assert_eq!(
                (crop.left, crop.top, crop.width, crop.height),
                expected_crop
            );
        } else {
            panic!("Crop should be Some");
        }
    }
}
