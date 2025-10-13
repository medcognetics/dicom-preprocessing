use dicom::core::header::HasLength;
use image::{DynamicImage, GenericImageView};

use dicom::dictionary_std::tags;
use dicom::object::{FileDicomObject, InMemDicomObject};
use dicom::pixeldata::ConvertOptions;

use crate::errors::DicomError;
use crate::metadata::{PreprocessingMetadata, Resolution};
use crate::transform::resize;
use crate::transform::{
    Crop, HandleVolume, Padding, PaddingDirection, Resize, Transform, VolumeHandler,
};

/// Configuration for spacing-based resizing
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpacingConfig {
    /// Target pixel spacing in mm for x dimension
    pub spacing_mm_x: f32,
    /// Target pixel spacing in mm for y dimension
    pub spacing_mm_y: f32,
    /// Target spacing in mm for z dimension (between frames)
    pub spacing_mm_z: Option<f32>,
}

impl SpacingConfig {
    pub fn new(spacing_mm_x: f32, spacing_mm_y: f32) -> Self {
        Self {
            spacing_mm_x,
            spacing_mm_y,
            spacing_mm_z: None,
        }
    }

    pub fn new_3d(spacing_mm_x: f32, spacing_mm_y: f32, spacing_mm_z: f32) -> Self {
        Self {
            spacing_mm_x,
            spacing_mm_y,
            spacing_mm_z: Some(spacing_mm_z),
        }
    }

    pub fn with_spacing_z(mut self, spacing_mm_z: f32) -> Self {
        self.spacing_mm_z = Some(spacing_mm_z);
        self
    }
}

// Responsible for preprocessing image data before saving
#[derive(Debug, Clone)]
pub struct Preprocessor {
    pub crop: bool,
    pub size: Option<(u32, u32)>,
    pub spacing: Option<SpacingConfig>,
    pub filter: resize::FilterType,
    pub padding_direction: PaddingDirection,
    pub crop_max: bool,
    pub volume_handler: VolumeHandler,
    pub use_components: bool,
    pub use_padding: bool,
    pub border_frac: Option<f32>,
    pub target_frames: u32,
    pub convert_options: ConvertOptions,
}

impl Default for Preprocessor {
    fn default() -> Self {
        Preprocessor {
            crop: true,
            size: None,
            spacing: None,
            filter: resize::FilterType::Triangle,
            padding_direction: PaddingDirection::Zero,
            crop_max: true,
            volume_handler: VolumeHandler::default(),
            use_components: true,
            use_padding: true,
            border_frac: None,
            target_frames: 32,
            convert_options: ConvertOptions::default(),
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

    fn get_resize(
        &self,
        images: &[DynamicImage],
        resolution: &Option<Resolution>,
    ) -> Option<Resize> {
        // Compute target size from either fixed size or spacing
        let target_size = match (self.size, self.spacing, resolution) {
            // Fixed size takes precedence
            (Some((w, h)), _, _) => Some((w, h)),
            // Spacing-based sizing requires resolution info
            (None, Some(spacing_config), Some(res)) => {
                let first_image = images.first().unwrap();
                let (current_width, current_height) = first_image.dimensions();

                // Target resolution = 1 / target_spacing_mm
                let target_pixels_per_mm_x = 1.0 / spacing_config.spacing_mm_x;
                let target_pixels_per_mm_y = 1.0 / spacing_config.spacing_mm_y;

                // Scale factor = target_resolution / current_resolution
                let scale_x = target_pixels_per_mm_x / res.pixels_per_mm_x;
                let scale_y = target_pixels_per_mm_y / res.pixels_per_mm_y;

                // New dimensions
                let target_width = (current_width as f32 * scale_x).round() as u32;
                let target_height = (current_height as f32 * scale_y).round() as u32;

                Some((target_width, target_height))
            }
            _ => None,
        };

        target_size.map(|(target_width, target_height)| {
            let first_image = images.first().unwrap();
            Resize::new(first_image, target_width, target_height, self.filter)
        })
    }

    fn get_padding(
        &self,
        images: &[DynamicImage],
        _resize_config: &Option<Resize>,
    ) -> Option<Padding> {
        // Use self.size directly - if resize happened, the images are already at the correct size
        let target_size = self.size;

        match (self.use_padding, target_size) {
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

    /// Compute target frame count based on spacing configuration
    /// TODO: This will be used in future for z-spacing based frame interpolation
    #[allow(dead_code)]
    fn get_target_frames(&self, resolution: &Option<Resolution>) -> Option<u32> {
        match (self.spacing, resolution) {
            (Some(spacing_config), Some(res))
                if spacing_config.spacing_mm_z.is_some() && res.frames_per_mm.is_some() =>
            {
                let target_spacing_mm_z = spacing_config.spacing_mm_z.unwrap();
                let target_frames_per_mm = 1.0 / target_spacing_mm_z;
                let current_frames_per_mm = res.frames_per_mm.unwrap();

                // Scale factor for frames
                let scale_z = target_frames_per_mm / current_frames_per_mm;

                // Compute target frame count (assumes we know current frame count from volume handler)
                Some((self.target_frames as f32 * scale_z).round() as u32)
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
            false => self
                .volume_handler
                .decode_volume_with_options(file, &self.convert_options),
            true => self
                .volume_handler
                .par_decode_volume_with_options(file, &self.convert_options),
        }?;

        // Try to determine the resolution from pixel spacing attributes
        let mut resolution = Resolution::try_from(file).ok();

        // Determine and apply crop
        let crop_config = self.get_crop(&image_data);
        let image_data = match &crop_config {
            Some(config) => config.apply_iter(image_data.into_iter()).collect(),
            None => image_data,
        };

        // Determine and apply resize, ensuring we also update the resolution
        let resize_config = self.get_resize(&image_data, &resolution);
        let image_data = match &resize_config {
            Some(config) => config.apply_iter(image_data.into_iter()).collect(),
            None => image_data,
        };

        // Update the resolution if we resized
        if let (Some(res), Some(config)) = (&resolution, &resize_config) {
            resolution = Some(config.apply(res));
        }

        // Determine and apply padding
        let padding_config = self.get_padding(&image_data, &resize_config);
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

    /// Processes multiple DICOM files with a common crop bound.
    /// This is useful for CT scans where each slice is stored in a separate DICOM file.
    /// The crop bounds are determined across all slices and applied consistently.
    /// Output order matches input order.
    pub fn prepare_images_batch(
        &self,
        files: &[FileDicomObject<InMemDicomObject>],
        parallel: bool,
    ) -> Result<(Vec<Vec<DynamicImage>>, PreprocessingMetadata), DicomError> {
        if files.is_empty() {
            return Err(DicomError::Other {
                message: "Cannot process empty batch of files".to_string(),
            });
        }

        // Decode all files and collect their images
        let mut all_decoded_images: Vec<Vec<DynamicImage>> = Vec::with_capacity(files.len());
        for file in files {
            let images = match parallel {
                false => self
                    .volume_handler
                    .decode_volume_with_options(file, &self.convert_options),
                true => self
                    .volume_handler
                    .par_decode_volume_with_options(file, &self.convert_options),
            }?;
            all_decoded_images.push(images);
        }

        // Try to determine the resolution from the first file's pixel spacing attributes
        let mut resolution = Resolution::try_from(&files[0]).ok();

        // Flatten all images to determine common crop bounds
        let all_images_flat: Vec<DynamicImage> = all_decoded_images
            .iter()
            .flat_map(|images| images.iter().cloned())
            .collect();

        // Determine common crop bounds across all images
        let crop_config = self.get_crop(&all_images_flat);

        // Apply crop to each batch of images
        let mut cropped_batches: Vec<Vec<DynamicImage>> = Vec::with_capacity(files.len());
        for images in all_decoded_images {
            let cropped = match &crop_config {
                Some(config) => config.apply_iter(images.into_iter()).collect(),
                None => images,
            };
            cropped_batches.push(cropped);
        }

        // For resize and padding, use the first image from the first batch as reference
        let first_image = &cropped_batches[0][0];
        let resize_config = self.get_resize(&[first_image.clone()], &resolution);

        // Apply resize to each batch
        let mut resized_batches: Vec<Vec<DynamicImage>> = Vec::with_capacity(files.len());
        for images in cropped_batches {
            let resized = match &resize_config {
                Some(config) => config.apply_iter(images.into_iter()).collect(),
                None => images,
            };
            resized_batches.push(resized);
        }

        // Update the resolution if we resized
        if let (Some(res), Some(config)) = (&resolution, &resize_config) {
            resolution = Some(config.apply(res));
        }

        // Determine padding from the first image
        let first_image = &resized_batches[0][0];
        let padding_config = self.get_padding(&[first_image.clone()], &resize_config);

        // Apply padding to each batch
        let mut padded_batches: Vec<Vec<DynamicImage>> = Vec::with_capacity(files.len());
        for images in resized_batches {
            let padded = match &padding_config {
                Some(config) => config.apply_iter(images.into_iter()).collect(),
                None => images,
            };
            padded_batches.push(padded);
        }

        // Calculate total frames
        let num_frames = padded_batches.iter().map(|b| b.len()).sum::<usize>().into();

        Ok((
            padded_batches,
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
        images: &[DynamicImage],
    ) -> Result<(Vec<DynamicImage>, PreprocessingMetadata), DicomError> {
        // Try to determine the resolution (none for test images)
        let resolution = None;

        // Determine and apply crop
        let crop_config = self.get_crop(images);
        let image_data = match &crop_config {
            Some(config) => config.apply_iter(images.iter().cloned()).collect(),
            None => images.to_vec(),
        };

        // Determine and apply resize, ensuring we also update the resolution
        let resize_config = self.get_resize(&image_data, &resolution);
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
        let padding_config = self.get_padding(&image_data, &resize_config);
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
    use dicom::pixeldata::WindowLevel;

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
            spacing: None,
            filter: resize::FilterType::Nearest,
            padding_direction: PaddingDirection::default(),
            crop_max: false,
            volume_handler: VolumeHandler::Keep(KeepVolume),
            use_components: true,
            use_padding: true,
            border_frac: None,
            target_frames: 32,
            convert_options: ConvertOptions::default(),
        },
        false
    )]
    #[case(
        "pydicom/MR_small.dcm", 
        Preprocessor {
            crop: false,
            size: None,
            spacing: None,
            filter: resize::FilterType::Nearest,
            padding_direction: PaddingDirection::default(),
            crop_max: false,
            volume_handler: VolumeHandler::CentralSlice(CentralSlice),
            use_components: true,
            use_padding: true,
            border_frac: None,
            target_frames: 32,
            convert_options: ConvertOptions::default(),
        },
        true
    )]
    #[case(
        "pydicom/JPEG2000_UNC.dcm",
        Preprocessor {
            crop: false,
            size: None,
            spacing: None,
            filter: resize::FilterType::Nearest,
            padding_direction: PaddingDirection::default(),
            crop_max: false,
            volume_handler: VolumeHandler::CentralSlice(CentralSlice),
            use_components: true,
            use_padding: true,
            border_frac: None,
            target_frames: 32,
            convert_options: ConvertOptions::default(),
        },
        false
    )]
    #[case(
        "pydicom/US1_J2KI.dcm",
        Preprocessor {
            crop: false,
            size: None,
            spacing: None,
            filter: resize::FilterType::Nearest,
            padding_direction: PaddingDirection::default(),
            crop_max: false,
            volume_handler: VolumeHandler::CentralSlice(CentralSlice),
            use_components: true,
            use_padding: true,
            border_frac: None,
            target_frames: 32,
            convert_options: ConvertOptions::default(),
        },
        false
    )]
    #[case(
        "pydicom/JPGLosslessP14SV1_1s_1f_8b.dcm",
        Preprocessor {
            crop: false,
            size: None,
            spacing: None,
            filter: resize::FilterType::Nearest,
            padding_direction: PaddingDirection::default(),
            crop_max: false,
            volume_handler: VolumeHandler::CentralSlice(CentralSlice),
            use_components: true,
            use_padding: true,
            border_frac: None,
            target_frames: 32,
            convert_options: ConvertOptions::default(),
        },
        false
    )]
    #[case(
        "pydicom/SC_rgb.dcm",
        Preprocessor {
            crop: false,
            size: None,
            spacing: None,
            filter: resize::FilterType::Nearest,
            padding_direction: PaddingDirection::default(),
            crop_max: false,
            volume_handler: VolumeHandler::CentralSlice(CentralSlice),
            use_components: true,
            use_padding: true,
            border_frac: None,
            target_frames: 32,
            convert_options: ConvertOptions::default(),
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
            spacing: None,
            filter: resize::FilterType::Nearest,
            padding_direction: PaddingDirection::Center,
            crop_max: false,
            volume_handler: VolumeHandler::default(),
            use_components: true,
            use_padding,
            border_frac: None,
            target_frames: 32,
            convert_options: ConvertOptions::default(),
        };

        let resize_config = None;
        let padding = preprocessor.get_padding(&[dynamic_image], &resize_config);
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
            spacing: None,
            filter: resize::FilterType::Nearest,
            padding_direction: PaddingDirection::Center,
            crop_max: false,
            volume_handler: VolumeHandler::default(),
            use_components: true,
            use_padding: true,
            border_frac,
            target_frames: 32,
            convert_options: ConvertOptions::default(),
        };

        let (processed_images, metadata) = preprocessor
            .prepare_image_for_test(&[dynamic_image])
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

    #[rstest]
    #[case(
        "pydicom/CT_small.dcm",
        ConvertOptions::default(),
        ConvertOptions::default().with_voi_lut(dicom::pixeldata::VoiLutOption::Custom(WindowLevel { center: 0.0, width: 1.0 }))
    )]
    fn test_different_convert_options_produce_different_outputs(
        #[case] dicom_file_path: &str,
        #[case] convert_options_1: ConvertOptions,
        #[case] convert_options_2: ConvertOptions,
    ) {
        let dicom_file = open_file(dicom_test_files::path(dicom_file_path).unwrap()).unwrap();

        // Create two preprocessors with different convert options
        let preprocessor_1 = Preprocessor {
            crop: false,
            size: None,
            spacing: None,
            filter: resize::FilterType::Nearest,
            padding_direction: PaddingDirection::default(),
            crop_max: false,
            volume_handler: VolumeHandler::CentralSlice(CentralSlice),
            use_components: true,
            use_padding: true,
            border_frac: None,
            target_frames: 32,
            convert_options: convert_options_1,
        };

        let preprocessor_2 = Preprocessor {
            crop: false,
            size: None,
            spacing: None,
            filter: resize::FilterType::Nearest,
            padding_direction: PaddingDirection::default(),
            crop_max: false,
            volume_handler: VolumeHandler::CentralSlice(CentralSlice),
            use_components: true,
            use_padding: true,
            border_frac: None,
            target_frames: 32,
            convert_options: convert_options_2,
        };

        // Process the same DICOM file with both preprocessors
        let (images_1, _) = preprocessor_1.prepare_image(&dicom_file, false).unwrap();
        let (images_2, _) = preprocessor_2.prepare_image(&dicom_file, false).unwrap();

        // Both should produce the same number of images
        assert_eq!(images_1.len(), images_2.len());
        assert_eq!(images_1.len(), 1);

        // Compare pixel data to ensure they're different
        let img_1 = &images_1[0];
        let img_2 = &images_2[0];

        // Images should have the same dimensions
        assert_eq!(img_1.dimensions(), img_2.dimensions());

        // Find at least one pixel that differs between the two images
        let (width, height) = img_1.dimensions();
        let mut found_difference = false;

        for y in 0..height {
            for x in 0..width {
                if img_1.get_pixel(x, y) != img_2.get_pixel(x, y) {
                    found_difference = true;
                    break;
                }
            }
            if found_difference {
                break;
            }
        }

        assert!(
            found_difference,
            "Images should be different when using different convert_options"
        );
    }

    #[rstest]
    #[case("pydicom/CT_small.dcm", 1.0, 1.0)]
    #[case("pydicom/CT_small.dcm", 0.5, 0.5)]
    #[case("pydicom/CT_small.dcm", 2.0, 2.0)]
    fn test_spacing_based_resize(
        #[case] dicom_file_path: &str,
        #[case] target_spacing_x: f32,
        #[case] target_spacing_y: f32,
    ) {
        let dicom_file = open_file(dicom_test_files::path(dicom_file_path).unwrap()).unwrap();

        // Get native resolution
        let native_resolution = Resolution::try_from(&dicom_file).unwrap();
        let native_spacing_x = 1.0 / native_resolution.pixels_per_mm_x;
        let native_spacing_y = 1.0 / native_resolution.pixels_per_mm_y;

        // Get native dimensions
        let native_width = dicom_file
            .get(tags::COLUMNS)
            .unwrap()
            .to_int::<u32>()
            .unwrap();
        let native_height = dicom_file.get(tags::ROWS).unwrap().to_int::<u32>().unwrap();

        // Create preprocessor with spacing config
        let spacing_config = SpacingConfig::new(target_spacing_x, target_spacing_y);
        let preprocessor = Preprocessor {
            crop: false,
            size: None,
            spacing: Some(spacing_config),
            filter: resize::FilterType::Nearest,
            padding_direction: PaddingDirection::default(),
            crop_max: false,
            volume_handler: VolumeHandler::CentralSlice(CentralSlice),
            use_components: true,
            use_padding: false,
            border_frac: None,
            target_frames: 32,
            convert_options: ConvertOptions::default(),
        };

        // Process the DICOM file
        let (images, metadata) = preprocessor.prepare_image(&dicom_file, false).unwrap();
        assert_eq!(images.len(), 1);

        // Check that the output dimensions match expected scaling
        let (output_width, output_height) = images[0].dimensions();
        let expected_width =
            (native_width as f32 * (native_spacing_x / target_spacing_x)).round() as u32;
        let expected_height =
            (native_height as f32 * (native_spacing_y / target_spacing_y)).round() as u32;

        assert_eq!(
            output_width, expected_width,
            "Width should match expected based on spacing"
        );
        assert_eq!(
            output_height, expected_height,
            "Height should match expected based on spacing"
        );

        // Verify resolution metadata was updated correctly
        let output_resolution = metadata.resolution.unwrap();
        let output_spacing_x = 1.0 / output_resolution.pixels_per_mm_x;
        let output_spacing_y = 1.0 / output_resolution.pixels_per_mm_y;

        // Allow small floating point error (tolerance is higher for larger spacing values)
        let tolerance = target_spacing_x.max(target_spacing_y) * 0.02; // 2% tolerance
        assert!((output_spacing_x - target_spacing_x).abs() < tolerance,
            "Output spacing X should be close to target. Got {output_spacing_x} expected {target_spacing_x}");
        assert!((output_spacing_y - target_spacing_y).abs() < tolerance,
            "Output spacing Y should be close to target. Got {output_spacing_y} expected {target_spacing_y}");
    }

    #[rstest]
    #[case("pydicom/CT_small.dcm")]
    fn test_spacing_without_resolution_does_nothing(#[case] dicom_file_path: &str) {
        let dicom_file = open_file(dicom_test_files::path(dicom_file_path).unwrap()).unwrap();

        // Get native dimensions
        let native_width = dicom_file
            .get(tags::COLUMNS)
            .unwrap()
            .to_int::<u32>()
            .unwrap();
        let native_height = dicom_file.get(tags::ROWS).unwrap().to_int::<u32>().unwrap();

        // Create preprocessor with spacing but remove resolution from the file
        let spacing_config = SpacingConfig::new(1.0, 1.0);
        let preprocessor = Preprocessor {
            crop: false,
            size: None,
            spacing: Some(spacing_config),
            filter: resize::FilterType::Nearest,
            padding_direction: PaddingDirection::default(),
            crop_max: false,
            volume_handler: VolumeHandler::CentralSlice(CentralSlice),
            use_components: true,
            use_padding: false,
            border_frac: None,
            target_frames: 32,
            convert_options: ConvertOptions::default(),
        };

        // Create a copy without pixel spacing
        let mut file_without_spacing = dicom_file.clone();
        file_without_spacing.remove_element(tags::PIXEL_SPACING);
        file_without_spacing.remove_element(tags::IMAGER_PIXEL_SPACING);

        // Process - should keep original dimensions since no resolution info available
        let (images, _) = preprocessor
            .prepare_image(&file_without_spacing, false)
            .unwrap();

        let (output_width, output_height) = images[0].dimensions();
        assert_eq!(
            output_width, native_width,
            "Width should remain unchanged without resolution info"
        );
        assert_eq!(
            output_height, native_height,
            "Height should remain unchanged without resolution info"
        );
    }
}
