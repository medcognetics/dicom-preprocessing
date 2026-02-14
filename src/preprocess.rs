use dicom::core::header::HasLength;
use image::{DynamicImage, GenericImageView};

use dicom::dictionary_std::tags;
use dicom::object::{FileDicomObject, InMemDicomObject};
use dicom::pixeldata::ConvertOptions;
use rayon::prelude::*;

use crate::errors::DicomError;
use crate::metadata::{FrameCount, PreprocessingMetadata, Resolution};
use crate::transform::resize;
use crate::transform::{
    Crop, HandleVolume, KeepVolume, Padding, PaddingDirection, Resize, Transform, VolumeHandler,
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
    const PARALLEL_TRANSFORM_MIN_FRAMES: usize = 2;

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

    /// Compute target frame count based on spacing configuration and current frame count
    fn compute_target_frame_count(
        &self,
        current_frame_count: usize,
        resolution: &Option<Resolution>,
    ) -> Option<u32> {
        match (self.spacing, resolution) {
            (Some(spacing_config), Some(res))
                if spacing_config.spacing_mm_z.is_some() && res.frames_per_mm.is_some() =>
            {
                let target_spacing_mm_z = spacing_config.spacing_mm_z.unwrap();
                let target_frames_per_mm = 1.0 / target_spacing_mm_z;
                let current_frames_per_mm = res.frames_per_mm.unwrap();

                // Scale factor for frames
                let scale_z = target_frames_per_mm / current_frames_per_mm;

                // Compute target frame count based on actual current frame count
                Some((current_frame_count as f32 * scale_z).round() as u32)
            }
            _ => None,
        }
    }

    /// Apply z-direction interpolation to resample frames based on spacing
    fn interpolate_z_spacing(
        &self,
        images: Vec<DynamicImage>,
        target_frames: u32,
    ) -> Vec<DynamicImage> {
        use crate::volume::InterpolateVolume;

        if images.len() <= 1 || target_frames <= 1 || images.len() == target_frames as usize {
            return images;
        }

        InterpolateVolume::interpolate_frames(&images, target_frames)
    }

    fn apply_transform_to_frames<T>(
        &self,
        images: Vec<DynamicImage>,
        config: &Option<T>,
        parallel: bool,
    ) -> Vec<DynamicImage>
    where
        T: Transform<DynamicImage> + Sync,
    {
        match config {
            Some(config) if parallel && images.len() >= Self::PARALLEL_TRANSFORM_MIN_FRAMES => {
                images
                    .into_par_iter()
                    .map(|image| config.apply(&image))
                    .collect()
            }
            Some(config) => config.apply_iter(images.into_iter()).collect(),
            None => images,
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

    fn decode_with_single_frame_guard(
        &self,
        file: &FileDicomObject<InMemDicomObject>,
        parallel: bool,
    ) -> Result<Vec<DynamicImage>, DicomError> {
        let frame_count: u32 = FrameCount::try_from(file)?.into();

        if frame_count == 1 {
            let keep_handler = KeepVolume;
            return if parallel {
                keep_handler.par_decode_volume_with_options(file, &self.convert_options)
            } else {
                keep_handler.decode_volume_with_options(file, &self.convert_options)
            };
        }

        if parallel {
            self.volume_handler
                .par_decode_volume_with_options(file, &self.convert_options)
        } else {
            self.volume_handler
                .decode_volume_with_options(file, &self.convert_options)
        }
    }

    /// Decodes the pixel data and applies transformations.
    /// When `parallel` is true, the pixel data is decoded in parallel using rayon
    pub fn prepare_image(
        &self,
        file: &FileDicomObject<InMemDicomObject>,
        parallel: bool,
    ) -> Result<(Vec<DynamicImage>, PreprocessingMetadata), DicomError> {
        // Run decoding and volume handling
        let mut image_data = self.decode_with_single_frame_guard(file, parallel)?;

        // Try to determine the resolution from pixel spacing attributes
        let mut resolution = Resolution::try_from(file).ok();

        // Apply z-spacing interpolation if needed
        // First check if there's a spacing-based target, otherwise use VolumeHandler target
        let target_frames = self
            .compute_target_frame_count(image_data.len(), &resolution)
            .or_else(|| self.volume_handler.get_target_frames());

        if let Some(target_frames) = target_frames {
            image_data = self.interpolate_z_spacing(image_data, target_frames);

            // Update the z-resolution after interpolation
            if let Some(ref mut res) = resolution {
                if let Some(spacing_config) = self.spacing {
                    if let Some(target_spacing_mm_z) = spacing_config.spacing_mm_z {
                        res.frames_per_mm = Some(1.0 / target_spacing_mm_z);
                    }
                }
            }
        }

        // Determine and apply crop
        let crop_config = self.get_crop(&image_data);
        let image_data = self.apply_transform_to_frames(image_data, &crop_config, parallel);

        // Determine and apply resize, ensuring we also update the resolution
        let resize_config = self.get_resize(&image_data, &resolution);
        let image_data = self.apply_transform_to_frames(image_data, &resize_config, parallel);

        // Update the resolution if we resized
        if let (Some(res), Some(config)) = (&resolution, &resize_config) {
            resolution = Some(config.apply(res));
        }

        // Determine and apply padding
        let padding_config = self.get_padding(&image_data, &resize_config);
        let image_data = self.apply_transform_to_frames(image_data, &padding_config, parallel);

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

        // Decode all files and collect their images into a single volume
        // For CT scans, each file is typically a single 2D slice
        let mut combined_volume: Vec<DynamicImage> = Vec::new();
        for file in files {
            let images = self.decode_with_single_frame_guard(file, parallel)?;

            // Add all frames from this file to the combined volume
            combined_volume.extend(images);
        }

        // Try to determine the resolution from the first file's pixel spacing attributes
        let mut resolution = Resolution::try_from(&files[0]).ok();

        // Apply z-spacing interpolation to the entire combined volume
        // First check if there's a spacing-based target, otherwise use VolumeHandler target
        let target_frames = self
            .compute_target_frame_count(combined_volume.len(), &resolution)
            .or_else(|| self.volume_handler.get_target_frames());

        if let Some(target_frames) = target_frames {
            combined_volume = self.interpolate_z_spacing(combined_volume, target_frames);

            // Update the z-resolution after interpolation
            if let Some(ref mut res) = resolution {
                if let Some(spacing_config) = self.spacing {
                    if let Some(target_spacing_mm_z) = spacing_config.spacing_mm_z {
                        res.frames_per_mm = Some(1.0 / target_spacing_mm_z);
                    }
                }
            }
        }

        // Use the combined volume for determining common crop bounds
        let all_images_flat = combined_volume;

        // Determine common crop bounds across all images in the volume
        let crop_config = self.get_crop(&all_images_flat);

        // Apply crop to the entire volume
        let cropped_volume =
            self.apply_transform_to_frames(all_images_flat, &crop_config, parallel);

        // Determine resize from the first image in the volume
        let resize_config = self.get_resize(&cropped_volume, &resolution);

        // Apply resize to the entire volume
        let resized_volume =
            self.apply_transform_to_frames(cropped_volume, &resize_config, parallel);

        // Update the resolution if we resized
        if let (Some(res), Some(config)) = (&resolution, &resize_config) {
            resolution = Some(config.apply(res));
        }

        // Determine padding from the first image
        let padding_config = self.get_padding(&resized_volume, &resize_config);

        // Apply padding to the entire volume
        let padded_volume: Vec<DynamicImage> =
            self.apply_transform_to_frames(resized_volume, &padding_config, parallel);

        let num_frames = padded_volume.len().into();

        // Split the volume back into individual slices
        // After z-interpolation, the number of output slices may differ from input
        // Each output slice is a single frame
        let result_batches: Vec<Vec<DynamicImage>> =
            padded_volume.into_iter().map(|image| vec![image]).collect();

        Ok((
            result_batches,
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
    use crate::volume::InterpolateVolume;
    use dicom::core::{DataElement, PrimitiveValue, VR};
    use dicom::dictionary_std::tags;
    use dicom::pixeldata::WindowLevel;

    use dicom::object::open_file;

    use crate::metadata::FrameCount;
    use crate::volume::{CentralSlice, KeepVolume, LaplacianMip, VolumeHandler};
    use image::{GenericImageView, RgbaImage};
    use rstest::rstest;

    fn assert_images_equal(expected: &[DynamicImage], actual: &[DynamicImage]) {
        assert_eq!(
            expected.len(),
            actual.len(),
            "Frame count mismatch: expected {}, got {}",
            expected.len(),
            actual.len()
        );

        for (frame_idx, (expected_frame, actual_frame)) in expected.iter().zip(actual).enumerate() {
            assert_eq!(
                expected_frame.dimensions(),
                actual_frame.dimensions(),
                "Dimensions mismatch on frame {frame_idx}"
            );

            let (width, height) = expected_frame.dimensions();
            for y in 0..height {
                for x in 0..width {
                    assert_eq!(
                        expected_frame.get_pixel(x, y),
                        actual_frame.get_pixel(x, y),
                        "Pixel mismatch on frame {frame_idx} at ({x}, {y})"
                    );
                }
            }
        }
    }

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

    #[rstest]
    #[case("pydicom/emri_small.dcm", 16)] // Downsample to 16 frames
    #[case("pydicom/emri_small.dcm", 20)] // Upsample to 20 frames
    fn test_interpolate_volume_handler(#[case] dicom_file_path: &str, #[case] target_frames: u32) {
        let dicom_file = open_file(dicom_test_files::path(dicom_file_path).unwrap()).unwrap();

        // Get native frame count
        let native_frame_count: u32 = FrameCount::try_from(&dicom_file).unwrap().into();

        // Create preprocessor with Interpolate volume handler
        let preprocessor = Preprocessor {
            crop: false,
            size: None,
            spacing: None,
            filter: resize::FilterType::Nearest,
            padding_direction: PaddingDirection::default(),
            crop_max: false,
            volume_handler: VolumeHandler::Interpolate(InterpolateVolume::new(target_frames)),
            use_components: true,
            use_padding: false,
            border_frac: None,
            target_frames,
            convert_options: ConvertOptions::default(),
        };

        // Process the DICOM file
        let (images, _metadata) = preprocessor.prepare_image(&dicom_file, false).unwrap();

        // Check that the output frame count matches target_frames
        assert_eq!(
            images.len(),
            target_frames as usize,
            "Frame count should match target_frames. Native frames: {native_frame_count}, Target: {target_frames}"
        );
    }

    fn laplacian_mip_preprocessor() -> Preprocessor {
        Preprocessor {
            crop: false,
            size: None,
            spacing: None,
            filter: resize::FilterType::Nearest,
            padding_direction: PaddingDirection::default(),
            crop_max: false,
            volume_handler: VolumeHandler::LaplacianMip(LaplacianMip::default()),
            use_components: true,
            use_padding: false,
            border_frac: None,
            target_frames: 32,
            convert_options: ConvertOptions::default(),
        }
    }

    fn malformed_zero_frame_dicom() -> FileDicomObject<InMemDicomObject> {
        let mut dicom_file =
            open_file(dicom_test_files::path("pydicom/CT_small.dcm").unwrap()).unwrap();
        dicom_file.put_element(DataElement::new(
            tags::NUMBER_OF_FRAMES,
            VR::IS,
            PrimitiveValue::from("0"),
        ));
        dicom_file
    }

    #[test]
    fn test_zero_frame_input_does_not_bypass_laplacian_mip() {
        let dicom_file = malformed_zero_frame_dicom();
        let preprocessor = laplacian_mip_preprocessor();

        let err = preprocessor
            .decode_with_single_frame_guard(&dicom_file, false)
            .unwrap_err();
        match err {
            DicomError::LaplacianMipInsufficientFrames {
                number_of_frames,
                skip_start,
                skip_end,
            } => {
                assert_eq!(number_of_frames, 0);
                assert_eq!(skip_start, 5);
                assert_eq!(skip_end, 5);
            }
            other => panic!("Expected LaplacianMipInsufficientFrames, got {other}"),
        }
    }

    #[test]
    fn test_single_frame_input_bypasses_laplacian_mip_prepare_image() {
        let dicom_file =
            open_file(dicom_test_files::path("pydicom/CT_small.dcm").unwrap()).unwrap();
        let preprocessor = laplacian_mip_preprocessor();

        let (images, metadata) = preprocessor.prepare_image(&dicom_file, false).unwrap();

        assert_eq!(images.len(), 1);
        let metadata_frames: usize = metadata.num_frames.into();
        assert_eq!(metadata_frames, 1);
    }

    #[test]
    fn test_single_frame_input_bypasses_laplacian_mip_prepare_image_parallel() {
        let dicom_file =
            open_file(dicom_test_files::path("pydicom/CT_small.dcm").unwrap()).unwrap();
        let preprocessor = laplacian_mip_preprocessor();

        let (images, metadata) = preprocessor.prepare_image(&dicom_file, true).unwrap();

        assert_eq!(images.len(), 1);
        let metadata_frames: usize = metadata.num_frames.into();
        assert_eq!(metadata_frames, 1);
    }

    #[test]
    fn test_single_frame_batch_bypasses_laplacian_mip() {
        const NUM_INPUT_FILES: usize = 3;

        let files: Vec<FileDicomObject<InMemDicomObject>> = (0..NUM_INPUT_FILES)
            .map(|_| open_file(dicom_test_files::path("pydicom/CT_small.dcm").unwrap()).unwrap())
            .collect();
        let preprocessor = laplacian_mip_preprocessor();

        let (image_batches, metadata) = preprocessor.prepare_images_batch(&files, false).unwrap();

        assert_eq!(image_batches.len(), NUM_INPUT_FILES);
        let total_frames: usize = image_batches.iter().map(Vec::len).sum();
        assert_eq!(total_frames, NUM_INPUT_FILES);
        let metadata_frames: usize = metadata.num_frames.into();
        assert_eq!(metadata_frames, NUM_INPUT_FILES);
    }

    #[rstest]
    #[case("pydicom/emri_small.dcm", 10.0)] // Upsample: larger spacing = fewer frames
    #[case("pydicom/emri_small.dcm", 2.5)] // Downsample: smaller spacing = more frames
    fn test_spacing_based_z_resize(#[case] dicom_file_path: &str, #[case] target_spacing_z: f32) {
        let mut dicom_file = open_file(dicom_test_files::path(dicom_file_path).unwrap()).unwrap();

        // Get native frame count
        let native_frame_count: u32 = FrameCount::try_from(&dicom_file).unwrap().into();

        // Clear any existing spacing metadata
        dicom_file.remove_element(tags::PIXEL_SPACING);
        dicom_file.remove_element(tags::IMAGER_PIXEL_SPACING);
        dicom_file.remove_element(tags::SLICE_THICKNESS);
        dicom_file.remove_element(tags::SPACING_BETWEEN_SLICES);

        // Manually add pixel spacing and slice thickness for testing
        // emri_small.dcm has 10 frames, let's assume 5mm spacing between slices
        let native_spacing_z = 5.0; // mm
        dicom_file.put_element(DataElement::new(
            tags::PIXEL_SPACING,
            VR::DS,
            PrimitiveValue::from("0.5\\0.5"),
        ));
        dicom_file.put_element(DataElement::new(
            tags::SLICE_THICKNESS,
            VR::DS,
            PrimitiveValue::from(native_spacing_z.to_string()),
        ));

        // Create preprocessor with 3D spacing config and Keep volume handler
        let spacing_config = SpacingConfig::new(0.5, 0.5).with_spacing_z(target_spacing_z);
        let preprocessor = Preprocessor {
            crop: false,
            size: None,
            spacing: Some(spacing_config),
            filter: resize::FilterType::Nearest,
            padding_direction: PaddingDirection::default(),
            crop_max: false,
            volume_handler: VolumeHandler::Keep(KeepVolume),
            use_components: true,
            use_padding: false,
            border_frac: None,
            target_frames: 32,
            convert_options: ConvertOptions::default(),
        };

        // Process the DICOM file
        let (images, metadata) = preprocessor.prepare_image(&dicom_file, false).unwrap();

        // Check that the output frame count matches expected scaling
        let expected_frame_count =
            (native_frame_count as f32 * (native_spacing_z / target_spacing_z)).round() as usize;

        assert_eq!(
            images.len(),
            expected_frame_count,
            "Frame count should match expected based on z-spacing. Native frames: {native_frame_count}, Native spacing: {native_spacing_z}, Target spacing: {target_spacing_z}, Expected frames: {expected_frame_count}"
        );

        // Verify resolution metadata was updated correctly
        let output_resolution = metadata.resolution.unwrap();
        let output_spacing_z = output_resolution
            .frames_per_mm
            .map(|f| 1.0 / f)
            .expect("Should have output z-spacing");

        // Allow small floating point error
        let tolerance = target_spacing_z * 0.05; // 5% tolerance
        assert!(
            (output_spacing_z - target_spacing_z).abs() < tolerance,
            "Output spacing Z should be close to target. Got {output_spacing_z} expected {target_spacing_z}"
        );
    }

    #[rstest]
    #[case(32)] // Test with 32 target frames
    #[case(16)] // Test with 16 target frames
    fn test_interpolate_volume_handler_batch(#[case] target_frames: u32) {
        use crate::volume::InterpolateVolume;

        // Open multiple CT slices - pydicom has CT_small.dcm which is a single slice
        // We'll simulate a batch by loading the same file multiple times
        let dicom_file_1 =
            open_file(dicom_test_files::path("pydicom/CT_small.dcm").unwrap()).unwrap();
        let dicom_file_2 =
            open_file(dicom_test_files::path("pydicom/CT_small.dcm").unwrap()).unwrap();
        let dicom_file_3 =
            open_file(dicom_test_files::path("pydicom/CT_small.dcm").unwrap()).unwrap();
        let dicom_file_4 =
            open_file(dicom_test_files::path("pydicom/CT_small.dcm").unwrap()).unwrap();
        let dicom_file_5 =
            open_file(dicom_test_files::path("pydicom/CT_small.dcm").unwrap()).unwrap();

        let files = vec![
            dicom_file_1,
            dicom_file_2,
            dicom_file_3,
            dicom_file_4,
            dicom_file_5,
        ];

        // Create preprocessor with Interpolate volume handler
        let preprocessor = Preprocessor {
            crop: false,
            size: None,
            spacing: None,
            filter: resize::FilterType::Nearest,
            padding_direction: PaddingDirection::default(),
            crop_max: false,
            volume_handler: VolumeHandler::Interpolate(InterpolateVolume::new(target_frames)),
            use_components: true,
            use_padding: false,
            border_frac: None,
            target_frames,
            convert_options: ConvertOptions::default(),
        };

        // Process the batch
        let (image_batches, metadata) = preprocessor.prepare_images_batch(&files, false).unwrap();

        // The total number of output frames should match target_frames
        let total_frames: usize = image_batches.iter().map(|batch| batch.len()).sum();
        assert_eq!(
            total_frames,
            target_frames as usize,
            "Total frame count should match target_frames after interpolation. Got {total_frames}, expected {target_frames}"
        );

        // Verify metadata reports the correct number of frames
        let metadata_frames: usize = metadata.num_frames.into();
        assert_eq!(
            metadata_frames, target_frames as usize,
            "Metadata should report correct frame count"
        );
    }

    #[test]
    fn test_prepare_image_parallel_transforms_match_serial() {
        let dicom_file =
            open_file(dicom_test_files::path("pydicom/emri_small.dcm").unwrap()).unwrap();
        let preprocessor = Preprocessor {
            crop: true,
            size: Some((96, 96)),
            spacing: None,
            filter: resize::FilterType::CatmullRom,
            padding_direction: PaddingDirection::Center,
            crop_max: false,
            volume_handler: VolumeHandler::Keep(KeepVolume),
            use_components: true,
            use_padding: true,
            border_frac: Some(0.05),
            target_frames: 32,
            convert_options: ConvertOptions::default(),
        };

        let (serial_images, serial_metadata) =
            preprocessor.prepare_image(&dicom_file, false).unwrap();
        let (parallel_images, parallel_metadata) =
            preprocessor.prepare_image(&dicom_file, true).unwrap();

        assert_eq!(serial_metadata, parallel_metadata);
        assert_images_equal(&serial_images, &parallel_images);
    }

    #[test]
    fn test_prepare_images_batch_parallel_transforms_match_serial() {
        let dicom_file_1 =
            open_file(dicom_test_files::path("pydicom/CT_small.dcm").unwrap()).unwrap();
        let dicom_file_2 =
            open_file(dicom_test_files::path("pydicom/CT_small.dcm").unwrap()).unwrap();
        let dicom_file_3 =
            open_file(dicom_test_files::path("pydicom/CT_small.dcm").unwrap()).unwrap();
        let files = vec![dicom_file_1, dicom_file_2, dicom_file_3];

        let preprocessor = Preprocessor {
            crop: true,
            size: Some((96, 96)),
            spacing: None,
            filter: resize::FilterType::Triangle,
            padding_direction: PaddingDirection::Center,
            crop_max: false,
            volume_handler: VolumeHandler::Keep(KeepVolume),
            use_components: true,
            use_padding: true,
            border_frac: None,
            target_frames: 32,
            convert_options: ConvertOptions::default(),
        };

        let (serial_batches, serial_metadata) =
            preprocessor.prepare_images_batch(&files, false).unwrap();
        let (parallel_batches, parallel_metadata) =
            preprocessor.prepare_images_batch(&files, true).unwrap();

        let serial_images: Vec<DynamicImage> = serial_batches.into_iter().flatten().collect();
        let parallel_images: Vec<DynamicImage> = parallel_batches.into_iter().flatten().collect();

        assert_eq!(serial_metadata, parallel_metadata);
        assert_images_equal(&serial_images, &parallel_images);
    }
}
