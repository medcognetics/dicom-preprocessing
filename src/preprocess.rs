use dicom::core::header::HasLength;
use image::DynamicImage;

use dicom::dictionary_std::tags;
use dicom::object::{FileDicomObject, InMemDicomObject};
use image::imageops::FilterType;

use crate::errors::DicomError;
use crate::metadata::{PreprocessingMetadata, Resolution};
use crate::transform::{
    Crop, HandleVolume, Padding, PaddingDirection, Resize, Transform, VolumeHandler,
};

// Responsible for preprocessing image data before saving
#[derive(Debug, Clone, Copy)]
pub struct Preprocessor {
    pub crop: bool,
    pub size: Option<(u32, u32)>,
    pub filter: FilterType,
    pub padding_direction: PaddingDirection,
    pub crop_max: bool,
    pub volume_handler: VolumeHandler,
    pub use_components: bool,
    pub use_padding: bool,
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
            use_components: true,
            use_padding: true,
        }
    }
}

impl Preprocessor {
    fn get_crop(&self, images: &Vec<DynamicImage>) -> Option<Crop> {
        match self.crop {
            true => Some(Crop::new_from_images(
                &images.iter().collect::<Vec<_>>(),
                self.crop_max,
                self.use_components,
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
        match (self.use_padding, self.size) {
            (true, Some((target_width, target_height))) => {
                let first_image = images.first().unwrap();
                let config = Padding::new(
                    &first_image,
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
            filter: FilterType::Nearest,
            padding_direction: PaddingDirection::default(),
            crop_max: false,
            volume_handler: VolumeHandler::Keep(KeepVolume),
            use_components: true,
            use_padding: true,
        },
        false
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
            use_components: true,
            use_padding: true,
        },
        true
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
            use_components: true,
            use_padding: true,
        },
        false
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
            use_components: true,
            use_padding: true,
        },
        false
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
            use_components: true,
            use_padding: true,
        },
        false
    )]
    #[case(
        "pydicom/SC_rgb.dcm",
        Preprocessor {
            crop: false,
            size: None,
            filter: FilterType::Nearest,
            padding_direction: PaddingDirection::default(),
            crop_max: false,
            volume_handler: VolumeHandler::CentralSlice(CentralSlice),
            use_components: true,
            use_padding: true,
        },
        false
    )]
    fn test_preprocess(
        #[case] dicom_file_path: &str,
        #[case] config: Preprocessor,
        #[case] parallel: bool,
    ) {
        let dicom_file = open_file(&dicom_test_files::path(dicom_file_path).unwrap()).unwrap();

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
            open_file(&dicom_test_files::path("pydicom/CT_small.dcm").unwrap()).unwrap();
        dicom_file.put_element(elem);
        assert_eq!(dicom_file.get(tags::VOILUT_FUNCTION).is_some(), true);
        Preprocessor::sanitize_dicom(&mut dicom_file);
        assert_eq!(dicom_file.get(tags::VOILUT_FUNCTION).is_none(), true);
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
            filter: FilterType::Nearest,
            padding_direction: PaddingDirection::Center,
            crop_max: false,
            volume_handler: VolumeHandler::default(),
            use_components: true,
            use_padding,
        };

        let padding = preprocessor.get_padding(&vec![dynamic_image]);
        assert_eq!(padding, expected_padding);
    }
}
