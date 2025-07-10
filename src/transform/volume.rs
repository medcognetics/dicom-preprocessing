use crate::errors::{dicom::PixelDataSnafu, DicomError};
use crate::metadata::preprocessing::FrameCount;
use dicom::object::{FileDicomObject, InMemDicomObject};
use dicom::pixeldata::{ConvertOptions, PixelDecoder};
use image::DynamicImage;
use image::Pixel;
use image::{GenericImage, GenericImageView};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use snafu::ResultExt;
use std::cmp::{max, min};
use std::fmt;

pub const DEFAULT_INTERPOLATE_TARGET_FRAMES: u32 = 32;

#[derive(Debug, Clone, Copy)]
pub enum VolumeHandler {
    Keep(KeepVolume),
    CentralSlice(CentralSlice),
    MaxIntensity(MaxIntensity),
    Interpolate(InterpolateVolume),
}

impl Default for VolumeHandler {
    fn default() -> Self {
        VolumeHandler::Keep(KeepVolume)
    }
}

#[derive(Default, Debug, Clone, Copy, clap::ValueEnum)]
pub enum DisplayVolumeHandler {
    #[default]
    Keep,
    CentralSlice,
    MaxIntensity,
    Interpolate,
}

impl From<DisplayVolumeHandler> for VolumeHandler {
    fn from(handler: DisplayVolumeHandler) -> Self {
        match handler {
            DisplayVolumeHandler::Keep => VolumeHandler::Keep(KeepVolume),
            DisplayVolumeHandler::CentralSlice => VolumeHandler::CentralSlice(CentralSlice),
            DisplayVolumeHandler::MaxIntensity => {
                VolumeHandler::MaxIntensity(MaxIntensity::default())
            }
            DisplayVolumeHandler::Interpolate => {
                VolumeHandler::Interpolate(InterpolateVolume::default())
            }
        }
    }
}

impl fmt::Display for DisplayVolumeHandler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let filter_str = match self {
            DisplayVolumeHandler::Keep => "keep",
            DisplayVolumeHandler::CentralSlice => "central-slice",
            DisplayVolumeHandler::MaxIntensity => "max-intensity",
            DisplayVolumeHandler::Interpolate => "interpolate",
        };
        write!(f, "{filter_str}")
    }
}

pub trait HandleVolume {
    /// Decode and handle the volume frame by frame with custom conversion options
    fn decode_volume_with_options(
        &self,
        file: &FileDicomObject<InMemDicomObject>,
        options: &ConvertOptions,
    ) -> Result<Vec<DynamicImage>, DicomError>;

    /// Decode each frame in parallel and handle the volume with custom conversion options
    fn par_decode_volume_with_options(
        &self,
        file: &FileDicomObject<InMemDicomObject>,
        options: &ConvertOptions,
    ) -> Result<Vec<DynamicImage>, DicomError>;

    /// Decode and handle the volume frame by frame with default options
    fn decode_volume(
        &self,
        file: &FileDicomObject<InMemDicomObject>,
    ) -> Result<Vec<DynamicImage>, DicomError> {
        self.decode_volume_with_options(file, &ConvertOptions::default())
    }

    /// Decode each frame in parallel and handle the volume with default options
    fn par_decode_volume(
        &self,
        file: &FileDicomObject<InMemDicomObject>,
    ) -> Result<Vec<DynamicImage>, DicomError> {
        self.par_decode_volume_with_options(file, &ConvertOptions::default())
    }
}

impl HandleVolume for VolumeHandler {
    fn decode_volume_with_options(
        &self,
        file: &FileDicomObject<InMemDicomObject>,
        options: &ConvertOptions,
    ) -> Result<Vec<DynamicImage>, DicomError> {
        match self {
            VolumeHandler::Keep(handler) => handler.decode_volume_with_options(file, options),
            VolumeHandler::CentralSlice(handler) => {
                handler.decode_volume_with_options(file, options)
            }
            VolumeHandler::MaxIntensity(handler) => {
                handler.decode_volume_with_options(file, options)
            }
            VolumeHandler::Interpolate(handler) => {
                handler.decode_volume_with_options(file, options)
            }
        }
    }

    fn par_decode_volume_with_options(
        &self,
        file: &FileDicomObject<InMemDicomObject>,
        options: &ConvertOptions,
    ) -> Result<Vec<DynamicImage>, DicomError> {
        match self {
            VolumeHandler::Keep(handler) => handler.par_decode_volume_with_options(file, options),
            VolumeHandler::CentralSlice(handler) => {
                handler.par_decode_volume_with_options(file, options)
            }
            VolumeHandler::MaxIntensity(handler) => {
                handler.par_decode_volume_with_options(file, options)
            }
            VolumeHandler::Interpolate(handler) => {
                handler.par_decode_volume_with_options(file, options)
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
/// Keep all frames
pub struct KeepVolume;

impl HandleVolume for KeepVolume {
    fn decode_volume_with_options(
        &self,
        file: &FileDicomObject<InMemDicomObject>,
        options: &ConvertOptions,
    ) -> Result<Vec<DynamicImage>, DicomError> {
        let number_of_frames: u32 = FrameCount::try_from(file)?.into();
        let mut image_data = Vec::with_capacity(number_of_frames as usize);
        for frame_number in 0..number_of_frames {
            let decoded = file
                .decode_pixel_data_frame(frame_number)
                .context(PixelDataSnafu)?;
            image_data.push(
                decoded
                    .to_dynamic_image_with_options(0, options)
                    .context(PixelDataSnafu)?,
            );
        }
        Ok(image_data)
    }

    fn par_decode_volume_with_options(
        &self,
        file: &FileDicomObject<InMemDicomObject>,
        options: &ConvertOptions,
    ) -> Result<Vec<DynamicImage>, DicomError> {
        let number_of_frames: u32 = FrameCount::try_from(file)?.into();
        let result = (0..number_of_frames)
            .into_par_iter()
            .map(|frame| {
                let result = file
                    .decode_pixel_data_frame(frame)
                    .context(PixelDataSnafu)?
                    .to_dynamic_image_with_options(0, options)
                    .context(PixelDataSnafu)?;
                Ok::<DynamicImage, DicomError>(result)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(result)
    }
}

#[derive(Debug, Clone, Copy)]
/// Keep only the central frame
pub struct CentralSlice;

impl HandleVolume for CentralSlice {
    fn decode_volume_with_options(
        &self,
        file: &FileDicomObject<InMemDicomObject>,
        options: &ConvertOptions,
    ) -> Result<Vec<DynamicImage>, DicomError> {
        let number_of_frames: u32 = FrameCount::try_from(file)?.into();
        let central_frame = number_of_frames / 2;
        let decoded = file
            .decode_pixel_data_frame(central_frame)
            .context(PixelDataSnafu)?;
        let image = decoded
            .to_dynamic_image_with_options(0, options)
            .context(PixelDataSnafu)?;
        Ok(vec![image])
    }

    fn par_decode_volume_with_options(
        &self,
        file: &FileDicomObject<InMemDicomObject>,
        options: &ConvertOptions,
    ) -> Result<Vec<DynamicImage>, DicomError> {
        // Since there is only one frame, we can just decode it serially
        self.decode_volume_with_options(file, options)
    }
}

#[derive(Debug, Clone, Copy, Default)]
/// Reduce the volume by taking the maximum intensity of each pixel across all frames
pub struct MaxIntensity {
    skip_start: u32,
    skip_end: u32,
}

impl MaxIntensity {
    /// Create a new `MaxIntensity` handler
    ///
    /// # Arguments
    ///
    /// * `skip_start` - The number of frames to skip at the start
    /// * `skip_end` - The number of frames to skip at the end
    ///
    /// # Returns
    pub fn new(skip_start: u32, skip_end: u32) -> Self {
        Self {
            skip_start,
            skip_end,
        }
    }
}

impl MaxIntensity {
    fn reduce(current: DynamicImage, new: DynamicImage) -> DynamicImage {
        let mut current = current;
        let (width, height) = current.dimensions();
        for x in 0..width {
            for y in 0..height {
                let mut current_pixel = current.get_pixel(x, y);
                let new_pixel = new.get_pixel(x, y);
                current_pixel.apply2(&new_pixel, max);
                current.put_pixel(x, y, current_pixel);
            }
        }
        current
    }
}

impl HandleVolume for MaxIntensity {
    fn decode_volume_with_options(
        &self,
        file: &FileDicomObject<InMemDicomObject>,
        options: &ConvertOptions,
    ) -> Result<Vec<DynamicImage>, DicomError> {
        let number_of_frames: u32 = FrameCount::try_from(file)?.into();
        let start = min(number_of_frames, self.skip_start);
        let end = max(0, number_of_frames as i64 - self.skip_end as i64) as u32;

        // Validate the start/end relative to the number of frames
        if start >= end || start >= number_of_frames {
            return Err(DicomError::FrameIndexError {
                start: start as usize,
                end: end as usize,
                number_of_frames: number_of_frames as usize,
            });
        }

        let decoded = file
            .decode_pixel_data_frame(start)
            .context(PixelDataSnafu)?;

        let mut image = decoded
            .to_dynamic_image_with_options(0, options)
            .context(PixelDataSnafu)?;
        for frame_number in (start + 1)..end {
            let decoded = file
                .decode_pixel_data_frame(frame_number)
                .context(PixelDataSnafu)?;
            let frame = decoded
                .to_dynamic_image_with_options(0, options)
                .context(PixelDataSnafu)?;
            image = Self::reduce(image, frame);
        }
        Ok(vec![image])
    }

    fn par_decode_volume_with_options(
        &self,
        file: &FileDicomObject<InMemDicomObject>,
        options: &ConvertOptions,
    ) -> Result<Vec<DynamicImage>, DicomError> {
        let number_of_frames: u32 = FrameCount::try_from(file)?.into();
        let start = min(number_of_frames, self.skip_start);
        let end = max(0, number_of_frames as i64 - self.skip_end as i64) as u32;

        // Validate the start/end relative to the number of frames
        if start >= end || start >= number_of_frames {
            return Err(DicomError::FrameIndexError {
                start: start as usize,
                end: end as usize,
                number_of_frames: number_of_frames as usize,
            });
        }

        let image = (start..end)
            .into_par_iter()
            .map(|frame_number| {
                let frame = file
                    .decode_pixel_data_frame(frame_number)
                    .context(PixelDataSnafu)?
                    .to_dynamic_image_with_options(0, options)
                    .context(PixelDataSnafu)?;
                Ok::<DynamicImage, DicomError>(frame)
            })
            .try_reduce_with(|image, frame| Ok(Self::reduce(image, frame)));

        if let Some(image) = image {
            Ok(vec![image?])
        } else {
            Err(DicomError::FrameIndexError {
                start: start as usize,
                end: end as usize,
                number_of_frames: number_of_frames as usize,
            })
        }
    }
}

#[derive(Debug, Clone, Copy)]
/// Interpolate between frames using linear interpolation
pub struct InterpolateVolume {
    target_frames: u32,
}

impl Default for InterpolateVolume {
    fn default() -> Self {
        Self {
            target_frames: DEFAULT_INTERPOLATE_TARGET_FRAMES,
        }
    }
}

impl InterpolateVolume {
    /// Create a new `InterpolateVolume` handler
    ///
    /// # Arguments
    ///
    /// * `target_frames` - The number of frames to interpolate to
    ///
    pub fn new(target_frames: u32) -> Self {
        Self { target_frames }
    }

    /// Interpolate between frames using linear interpolation
    fn interpolate_frames(frames: &[DynamicImage], target_frames: u32) -> Vec<DynamicImage> {
        if frames.len() <= 1 {
            return frames.to_vec();
        }

        let mut result = Vec::with_capacity(target_frames as usize);
        let (width, height) = frames[0].dimensions();

        for i in 0..target_frames {
            let t = i as f32 / (target_frames - 1) as f32;
            let frame_idx = t * (frames.len() - 1) as f32;
            let frame_idx_floor = frame_idx.floor() as usize;
            let frame_idx_ceil = frame_idx.ceil() as usize;
            let alpha = frame_idx - frame_idx_floor as f32;

            // Create a new image with the same color type as the input
            let mut interpolated = frames[0].clone();

            for x in 0..width {
                for y in 0..height {
                    let pixel1 = frames[frame_idx_floor].get_pixel(x, y);
                    let pixel2 = frames[frame_idx_ceil].get_pixel(x, y);

                    // Interpolate each channel independently
                    let interpolated_pixel = pixel1.map2(&pixel2, |p1, p2| {
                        let p1 = p1 as f32;
                        let p2 = p2 as f32;
                        (p1 * (1.0 - alpha) + p2 * alpha) as u8
                    });

                    interpolated.put_pixel(x, y, interpolated_pixel);
                }
            }

            result.push(interpolated);
        }

        result
    }
}

impl HandleVolume for InterpolateVolume {
    fn decode_volume_with_options(
        &self,
        file: &FileDicomObject<InMemDicomObject>,
        options: &ConvertOptions,
    ) -> Result<Vec<DynamicImage>, DicomError> {
        let number_of_frames: u32 = FrameCount::try_from(file)?.into();
        let mut frames = Vec::with_capacity(number_of_frames as usize);

        for frame_number in 0..number_of_frames {
            let decoded = file
                .decode_pixel_data_frame(frame_number)
                .context(PixelDataSnafu)?;
            frames.push(
                decoded
                    .to_dynamic_image_with_options(0, options)
                    .context(PixelDataSnafu)?,
            );
        }

        Ok(Self::interpolate_frames(&frames, self.target_frames))
    }

    fn par_decode_volume_with_options(
        &self,
        file: &FileDicomObject<InMemDicomObject>,
        options: &ConvertOptions,
    ) -> Result<Vec<DynamicImage>, DicomError> {
        let number_of_frames: u32 = FrameCount::try_from(file)?.into();
        let frames = (0..number_of_frames)
            .into_par_iter()
            .map(|frame| {
                let result = file
                    .decode_pixel_data_frame(frame)
                    .context(PixelDataSnafu)?
                    .to_dynamic_image_with_options(0, options)
                    .context(PixelDataSnafu)?;
                Ok::<DynamicImage, DicomError>(result)
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self::interpolate_frames(&frames, self.target_frames))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use dicom::object::open_file;
    use dicom::pixeldata::VoiLutOption;
    use image::{ImageBuffer, Rgb};
    use rstest::rstest;

    #[rstest]
    #[case("pydicom/CT_small.dcm", VolumeHandler::Keep(KeepVolume), 1)]
    #[case("pydicom/CT_small.dcm", VolumeHandler::CentralSlice(CentralSlice), 1)]
    #[case("pydicom/CT_small.dcm", VolumeHandler::MaxIntensity(MaxIntensity { skip_start: 0, skip_end: 0 }), 1)]
    #[case("pydicom/CT_small.dcm", VolumeHandler::Interpolate(InterpolateVolume { target_frames: 32 }), 1)]
    fn test_decode_volume(
        #[case] dicom_file_path: &str,
        #[case] volume_handler: VolumeHandler,
        #[case] expected_number_of_frames: u32,
        #[values(true, false)] use_parallel: bool,
    ) {
        let dicom = dicom_test_files::path(dicom_file_path).unwrap();
        let dicom = open_file(&dicom).unwrap();
        let images = if use_parallel {
            volume_handler.par_decode_volume(&dicom).unwrap()
        } else {
            volume_handler.decode_volume(&dicom).unwrap()
        };
        assert_eq!(images.len() as u32, expected_number_of_frames);
    }

    #[rstest]
    #[case("pydicom/CT_small.dcm", VolumeHandler::Keep(KeepVolume), 1)]
    #[case("pydicom/CT_small.dcm", VolumeHandler::CentralSlice(CentralSlice), 1)]
    #[case("pydicom/CT_small.dcm", VolumeHandler::MaxIntensity(MaxIntensity { skip_start: 0, skip_end: 0 }), 1)]
    #[case("pydicom/CT_small.dcm", VolumeHandler::Interpolate(InterpolateVolume { target_frames: 32 }), 1)]
    fn test_decode_volume_with_options(
        #[case] dicom_file_path: &str,
        #[case] volume_handler: VolumeHandler,
        #[case] expected_number_of_frames: u32,
    ) {
        let dicom = dicom_test_files::path(dicom_file_path).unwrap();
        let dicom = open_file(&dicom).unwrap();
        let mut options = ConvertOptions::default();
        options.voi_lut = VoiLutOption::Normalize;
        let images = volume_handler
            .decode_volume_with_options(&dicom, &options)
            .unwrap();
        assert_eq!(images.len() as u32, expected_number_of_frames);
    }

    #[rstest]
    #[case("pydicom/CT_small.dcm", VolumeHandler::Keep(KeepVolume), 1)]
    #[case("pydicom/CT_small.dcm", VolumeHandler::CentralSlice(CentralSlice), 1)]
    #[case("pydicom/CT_small.dcm", VolumeHandler::MaxIntensity(MaxIntensity { skip_start: 0, skip_end: 0 }), 1)]
    #[case("pydicom/CT_small.dcm", VolumeHandler::Interpolate(InterpolateVolume { target_frames: 32 }), 1)]
    fn test_par_decode_volume_with_options(
        #[case] dicom_file_path: &str,
        #[case] volume_handler: VolumeHandler,
        #[case] expected_number_of_frames: u32,
    ) {
        let dicom = dicom_test_files::path(dicom_file_path).unwrap();
        let dicom = open_file(&dicom).unwrap();
        let mut options = ConvertOptions::default();
        options.voi_lut = VoiLutOption::Normalize;
        let images = volume_handler
            .par_decode_volume_with_options(&dicom, &options)
            .unwrap();
        assert_eq!(images.len() as u32, expected_number_of_frames);
    }

    #[test]
    fn test_interpolate_frames() {
        // Create two test images with different pixel values
        let width = 2;
        let height = 2;
        let interpolated_frames = 3;

        // First image: all pixels are 100
        let mut img1 = ImageBuffer::new(width, height);
        for x in 0..width {
            for y in 0..height {
                img1.put_pixel(x, y, Rgb([100, 100, 100]));
            }
        }

        // Second image: all pixels are 200
        let mut img2 = ImageBuffer::new(width, height);
        for x in 0..width {
            for y in 0..height {
                img2.put_pixel(x, y, Rgb([200, 200, 200]));
            }
        }

        let frames = vec![DynamicImage::ImageRgb8(img1), DynamicImage::ImageRgb8(img2)];
        let interpolated = InterpolateVolume::interpolate_frames(&frames, interpolated_frames);

        // Should have 3 frames
        assert_eq!(interpolated.len(), interpolated_frames as usize);

        // First frame should be close to 100
        let first_frame = interpolated[0].as_rgb8().unwrap();
        const TOLERANCE: u8 = 5;
        for x in 0..width {
            for y in 0..height {
                let pixel = first_frame.get_pixel(x, y);
                assert!(pixel[0] >= 100 - TOLERANCE && pixel[0] <= 100 + TOLERANCE);
            }
        }

        // Middle frame should be close to 150
        let middle_frame = interpolated[1].as_rgb8().unwrap();
        for x in 0..width {
            for y in 0..height {
                let pixel = middle_frame.get_pixel(x, y);
                assert!(pixel[0] >= 150 - TOLERANCE && pixel[0] <= 150 + TOLERANCE);
            }
        }

        // Last frame should be close to 200
        let last_frame = interpolated[2].as_rgb8().unwrap();
        for x in 0..width {
            for y in 0..height {
                let pixel = last_frame.get_pixel(x, y);
                assert!(pixel[0] >= 200 - TOLERANCE && pixel[0] <= 200 + TOLERANCE);
            }
        }
    }

    #[test]
    fn test_interpolate_single_frame() {
        // Test interpolation with a single input frame
        let width = 2;
        let height = 2;
        let mut img = ImageBuffer::new(width, height);
        for x in 0..width {
            for y in 0..height {
                img.put_pixel(x, y, Rgb([100, 100, 100]));
            }
        }

        let frames = vec![DynamicImage::ImageRgb8(img)];
        let interpolated = InterpolateVolume::interpolate_frames(&frames, 3);

        // Should have 1 frame (same as input)
        assert_eq!(interpolated.len(), 1);

        // Frame should be identical to the input
        let frame = interpolated[0].as_rgb8().unwrap();
        for x in 0..width {
            for y in 0..height {
                let pixel = frame.get_pixel(x, y);
                assert_eq!(pixel[0], 100);
            }
        }
    }
}
