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

#[derive(Debug, Clone)]
pub enum VolumeHandler {
    Keep(KeepVolume),
    CentralSlice(CentralSlice),
    MaxIntensity(MaxIntensity),
    Interpolate(InterpolateVolume),
    AverageIntensity(AverageIntensity),
    GaussianWeighted(GaussianWeighted),
    SoftMip(SoftMip),
    LaplacianMip(LaplacianMip),
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
    AverageIntensity,
    GaussianWeighted,
    SoftMip,
    LaplacianMip,
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
            DisplayVolumeHandler::AverageIntensity => {
                VolumeHandler::AverageIntensity(AverageIntensity::default())
            }
            DisplayVolumeHandler::GaussianWeighted => {
                VolumeHandler::GaussianWeighted(GaussianWeighted::default())
            }
            DisplayVolumeHandler::SoftMip => VolumeHandler::SoftMip(SoftMip::default()),
            DisplayVolumeHandler::LaplacianMip => {
                VolumeHandler::LaplacianMip(LaplacianMip::default())
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
            DisplayVolumeHandler::AverageIntensity => "average-intensity",
            DisplayVolumeHandler::GaussianWeighted => "gaussian-weighted",
            DisplayVolumeHandler::SoftMip => "soft-mip",
            DisplayVolumeHandler::LaplacianMip => "laplacian-mip",
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
            VolumeHandler::AverageIntensity(handler) => {
                handler.decode_volume_with_options(file, options)
            }
            VolumeHandler::GaussianWeighted(handler) => {
                handler.decode_volume_with_options(file, options)
            }
            VolumeHandler::SoftMip(handler) => handler.decode_volume_with_options(file, options),
            VolumeHandler::LaplacianMip(handler) => {
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
            VolumeHandler::AverageIntensity(handler) => {
                handler.par_decode_volume_with_options(file, options)
            }
            VolumeHandler::GaussianWeighted(handler) => {
                handler.par_decode_volume_with_options(file, options)
            }
            VolumeHandler::SoftMip(handler) => {
                handler.par_decode_volume_with_options(file, options)
            }
            VolumeHandler::LaplacianMip(handler) => {
                handler.par_decode_volume_with_options(file, options)
            }
        }
    }
}

impl VolumeHandler {
    /// Get the target frames from Interpolate handler, if applicable
    pub fn get_target_frames(&self) -> Option<u32> {
        match self {
            VolumeHandler::Interpolate(handler) => Some(handler.target_frames),
            _ => None,
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
    pub(crate) target_frames: u32,
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
    pub fn interpolate_frames(frames: &[DynamicImage], target_frames: u32) -> Vec<DynamicImage> {
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

#[derive(Debug, Clone, Copy, Default)]
/// Reduce the volume by taking the average intensity of each pixel across all frames.
/// This is the Average Intensity Projection (AIP) method.
pub struct AverageIntensity {
    skip_start: u32,
    skip_end: u32,
}

impl AverageIntensity {
    /// Create a new `AverageIntensity` handler
    ///
    /// # Arguments
    ///
    /// * `skip_start` - The number of frames to skip at the start
    /// * `skip_end` - The number of frames to skip at the end
    pub fn new(skip_start: u32, skip_end: u32) -> Self {
        Self {
            skip_start,
            skip_end,
        }
    }

    /// Compute the average of pixel values across frames
    fn reduce_average(frames: &[DynamicImage]) -> DynamicImage {
        if frames.is_empty() {
            panic!("Cannot reduce empty frames");
        }
        if frames.len() == 1 {
            return frames[0].clone();
        }

        let (width, height) = frames[0].dimensions();
        let mut result = frames[0].clone();
        let n = frames.len() as f32;

        for x in 0..width {
            for y in 0..height {
                // Sum all pixel values
                let mut sum = [0.0f32; 4];
                for frame in frames {
                    let pixel = frame.get_pixel(x, y);
                    for (i, channel) in pixel.channels().iter().enumerate() {
                        sum[i] += *channel as f32;
                    }
                }
                // Compute average
                let mut avg_pixel = result.get_pixel(x, y);
                for (i, channel) in avg_pixel.channels_mut().iter_mut().enumerate() {
                    *channel = (sum[i] / n).round() as u8;
                }
                result.put_pixel(x, y, avg_pixel);
            }
        }
        result
    }
}

impl HandleVolume for AverageIntensity {
    fn decode_volume_with_options(
        &self,
        file: &FileDicomObject<InMemDicomObject>,
        options: &ConvertOptions,
    ) -> Result<Vec<DynamicImage>, DicomError> {
        let number_of_frames: u32 = FrameCount::try_from(file)?.into();
        let start = min(number_of_frames, self.skip_start);
        let end = max(0, number_of_frames as i64 - self.skip_end as i64) as u32;

        if start >= end || start >= number_of_frames {
            return Err(DicomError::FrameIndexError {
                start: start as usize,
                end: end as usize,
                number_of_frames: number_of_frames as usize,
            });
        }

        let mut frames = Vec::with_capacity((end - start) as usize);
        for frame_number in start..end {
            let decoded = file
                .decode_pixel_data_frame(frame_number)
                .context(PixelDataSnafu)?;
            frames.push(
                decoded
                    .to_dynamic_image_with_options(0, options)
                    .context(PixelDataSnafu)?,
            );
        }

        Ok(vec![Self::reduce_average(&frames)])
    }

    fn par_decode_volume_with_options(
        &self,
        file: &FileDicomObject<InMemDicomObject>,
        options: &ConvertOptions,
    ) -> Result<Vec<DynamicImage>, DicomError> {
        let number_of_frames: u32 = FrameCount::try_from(file)?.into();
        let start = min(number_of_frames, self.skip_start);
        let end = max(0, number_of_frames as i64 - self.skip_end as i64) as u32;

        if start >= end || start >= number_of_frames {
            return Err(DicomError::FrameIndexError {
                start: start as usize,
                end: end as usize,
                number_of_frames: number_of_frames as usize,
            });
        }

        let frames = (start..end)
            .into_par_iter()
            .map(|frame_number| {
                let frame = file
                    .decode_pixel_data_frame(frame_number)
                    .context(PixelDataSnafu)?
                    .to_dynamic_image_with_options(0, options)
                    .context(PixelDataSnafu)?;
                Ok::<DynamicImage, DicomError>(frame)
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(vec![Self::reduce_average(&frames)])
    }
}

#[derive(Debug, Clone, Copy)]
/// Reduce the volume using Gaussian-weighted averaging centered on the middle slice.
/// Central slices contribute more than edge slices.
pub struct GaussianWeighted {
    /// Sigma as a fraction of the total number of frames (default 0.25)
    sigma_fraction: f32,
    skip_start: u32,
    skip_end: u32,
}

impl Default for GaussianWeighted {
    fn default() -> Self {
        Self {
            sigma_fraction: 0.25,
            skip_start: 0,
            skip_end: 0,
        }
    }
}

impl GaussianWeighted {
    /// Create a new `GaussianWeighted` handler
    ///
    /// # Arguments
    ///
    /// * `sigma_fraction` - Sigma as a fraction of the total number of frames
    /// * `skip_start` - The number of frames to skip at the start
    /// * `skip_end` - The number of frames to skip at the end
    pub fn new(sigma_fraction: f32, skip_start: u32, skip_end: u32) -> Self {
        Self {
            sigma_fraction,
            skip_start,
            skip_end,
        }
    }

    /// Compute Gaussian weight for a given frame index
    fn gaussian_weight(index: f32, center: f32, sigma: f32) -> f32 {
        let diff = index - center;
        (-0.5 * (diff / sigma).powi(2)).exp()
    }

    /// Compute Gaussian-weighted average of frames
    fn reduce_gaussian(frames: &[DynamicImage], sigma_fraction: f32) -> DynamicImage {
        if frames.is_empty() {
            panic!("Cannot reduce empty frames");
        }
        if frames.len() == 1 {
            return frames[0].clone();
        }

        let n = frames.len();
        let center = (n - 1) as f32 / 2.0;
        let sigma = n as f32 * sigma_fraction;

        // Compute weights
        let weights: Vec<f32> = (0..n)
            .map(|i| Self::gaussian_weight(i as f32, center, sigma))
            .collect();
        let weight_sum: f32 = weights.iter().sum();

        let (width, height) = frames[0].dimensions();
        let mut result = frames[0].clone();

        for x in 0..width {
            for y in 0..height {
                let mut weighted_sum = [0.0f32; 4];
                for (i, frame) in frames.iter().enumerate() {
                    let pixel = frame.get_pixel(x, y);
                    for (j, channel) in pixel.channels().iter().enumerate() {
                        weighted_sum[j] += *channel as f32 * weights[i];
                    }
                }
                // Normalize by weight sum
                let mut avg_pixel = result.get_pixel(x, y);
                for (j, channel) in avg_pixel.channels_mut().iter_mut().enumerate() {
                    *channel = (weighted_sum[j] / weight_sum).round() as u8;
                }
                result.put_pixel(x, y, avg_pixel);
            }
        }
        result
    }
}

impl HandleVolume for GaussianWeighted {
    fn decode_volume_with_options(
        &self,
        file: &FileDicomObject<InMemDicomObject>,
        options: &ConvertOptions,
    ) -> Result<Vec<DynamicImage>, DicomError> {
        let number_of_frames: u32 = FrameCount::try_from(file)?.into();
        let start = min(number_of_frames, self.skip_start);
        let end = max(0, number_of_frames as i64 - self.skip_end as i64) as u32;

        if start >= end || start >= number_of_frames {
            return Err(DicomError::FrameIndexError {
                start: start as usize,
                end: end as usize,
                number_of_frames: number_of_frames as usize,
            });
        }

        let mut frames = Vec::with_capacity((end - start) as usize);
        for frame_number in start..end {
            let decoded = file
                .decode_pixel_data_frame(frame_number)
                .context(PixelDataSnafu)?;
            frames.push(
                decoded
                    .to_dynamic_image_with_options(0, options)
                    .context(PixelDataSnafu)?,
            );
        }

        Ok(vec![Self::reduce_gaussian(&frames, self.sigma_fraction)])
    }

    fn par_decode_volume_with_options(
        &self,
        file: &FileDicomObject<InMemDicomObject>,
        options: &ConvertOptions,
    ) -> Result<Vec<DynamicImage>, DicomError> {
        let number_of_frames: u32 = FrameCount::try_from(file)?.into();
        let start = min(number_of_frames, self.skip_start);
        let end = max(0, number_of_frames as i64 - self.skip_end as i64) as u32;

        if start >= end || start >= number_of_frames {
            return Err(DicomError::FrameIndexError {
                start: start as usize,
                end: end as usize,
                number_of_frames: number_of_frames as usize,
            });
        }

        let frames = (start..end)
            .into_par_iter()
            .map(|frame_number| {
                let frame = file
                    .decode_pixel_data_frame(frame_number)
                    .context(PixelDataSnafu)?
                    .to_dynamic_image_with_options(0, options)
                    .context(PixelDataSnafu)?;
                Ok::<DynamicImage, DicomError>(frame)
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(vec![Self::reduce_gaussian(&frames, self.sigma_fraction)])
    }
}

#[derive(Debug, Clone, Copy)]
/// SoftMIP: A hybrid MIP/AIP approach that computes weighted average of sorted voxel stack.
/// By adjusting the blend_factor, can smoothly transition between pure AIP (0.0) and pure MIP (1.0).
pub struct SoftMip {
    /// Blend factor between AIP (0.0) and MIP (1.0). Default is 0.7 for MIP-like behavior with less noise.
    blend_factor: f32,
    skip_start: u32,
    skip_end: u32,
}

impl Default for SoftMip {
    fn default() -> Self {
        Self {
            blend_factor: 0.7,
            skip_start: 0,
            skip_end: 0,
        }
    }
}

impl SoftMip {
    /// Create a new `SoftMip` handler
    ///
    /// # Arguments
    ///
    /// * `blend_factor` - Blend factor between AIP (0.0) and MIP (1.0)
    /// * `skip_start` - The number of frames to skip at the start
    /// * `skip_end` - The number of frames to skip at the end
    pub fn new(blend_factor: f32, skip_start: u32, skip_end: u32) -> Self {
        Self {
            blend_factor: blend_factor.clamp(0.0, 1.0),
            skip_start,
            skip_end,
        }
    }

    /// Compute SoftMIP reduction: weighted average of sorted voxel stack.
    /// Higher weights are applied to brighter (higher sorted position) values.
    fn reduce_softmip(frames: &[DynamicImage], blend_factor: f32) -> DynamicImage {
        if frames.is_empty() {
            panic!("Cannot reduce empty frames");
        }
        if frames.len() == 1 {
            return frames[0].clone();
        }

        let n = frames.len();
        let (width, height) = frames[0].dimensions();
        let mut result = frames[0].clone();

        // Precompute position-dependent weights
        // Weight increases with position in sorted array (brighter values get more weight)
        // blend_factor controls how steep the weighting is
        let weights: Vec<f32> = (0..n)
            .map(|i| {
                let position = i as f32 / (n - 1) as f32; // 0.0 to 1.0
                                                          // Exponential weighting: at blend_factor=0, all weights equal; at blend_factor=1, only max matters
                (position * blend_factor * 4.0).exp()
            })
            .collect();
        let weight_sum: f32 = weights.iter().sum();

        for x in 0..width {
            for y in 0..height {
                // Collect all channel values across frames
                let num_channels = frames[0].get_pixel(x, y).channels().len();
                let mut channel_values: Vec<Vec<u8>> = vec![Vec::with_capacity(n); num_channels];

                for frame in frames {
                    let pixel = frame.get_pixel(x, y);
                    for (c, channel) in pixel.channels().iter().enumerate() {
                        channel_values[c].push(*channel);
                    }
                }

                // Sort each channel and compute weighted average
                let mut avg_pixel = result.get_pixel(x, y);
                for (c, channel) in avg_pixel.channels_mut().iter_mut().enumerate() {
                    let mut values = channel_values[c].clone();
                    values.sort_unstable();

                    // Weighted average of sorted values
                    let weighted_sum: f32 = values
                        .iter()
                        .zip(weights.iter())
                        .map(|(&v, &w)| v as f32 * w)
                        .sum();
                    *channel = (weighted_sum / weight_sum).round() as u8;
                }
                result.put_pixel(x, y, avg_pixel);
            }
        }
        result
    }
}

impl HandleVolume for SoftMip {
    fn decode_volume_with_options(
        &self,
        file: &FileDicomObject<InMemDicomObject>,
        options: &ConvertOptions,
    ) -> Result<Vec<DynamicImage>, DicomError> {
        let number_of_frames: u32 = FrameCount::try_from(file)?.into();
        let start = min(number_of_frames, self.skip_start);
        let end = max(0, number_of_frames as i64 - self.skip_end as i64) as u32;

        if start >= end || start >= number_of_frames {
            return Err(DicomError::FrameIndexError {
                start: start as usize,
                end: end as usize,
                number_of_frames: number_of_frames as usize,
            });
        }

        let mut frames = Vec::with_capacity((end - start) as usize);
        for frame_number in start..end {
            let decoded = file
                .decode_pixel_data_frame(frame_number)
                .context(PixelDataSnafu)?;
            frames.push(
                decoded
                    .to_dynamic_image_with_options(0, options)
                    .context(PixelDataSnafu)?,
            );
        }

        Ok(vec![Self::reduce_softmip(&frames, self.blend_factor)])
    }

    fn par_decode_volume_with_options(
        &self,
        file: &FileDicomObject<InMemDicomObject>,
        options: &ConvertOptions,
    ) -> Result<Vec<DynamicImage>, DicomError> {
        let number_of_frames: u32 = FrameCount::try_from(file)?.into();
        let start = min(number_of_frames, self.skip_start);
        let end = max(0, number_of_frames as i64 - self.skip_end as i64) as u32;

        if start >= end || start >= number_of_frames {
            return Err(DicomError::FrameIndexError {
                start: start as usize,
                end: end as usize,
                number_of_frames: number_of_frames as usize,
            });
        }

        let frames = (start..end)
            .into_par_iter()
            .map(|frame_number| {
                let frame = file
                    .decode_pixel_data_frame(frame_number)
                    .context(PixelDataSnafu)?
                    .to_dynamic_image_with_options(0, options)
                    .context(PixelDataSnafu)?;
                Ok::<DynamicImage, DicomError>(frame)
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(vec![Self::reduce_softmip(&frames, self.blend_factor)])
    }
}

/// Laplacian Pyramid MIP Fusion - the most sophisticated DBT projection method.
///
/// Based on "Synthesizing Mammogram from DBT" (PMC6438841).
///
/// Algorithm:
/// 1. Build Gaussian/Laplacian pyramids for each DBT slice
/// 2. Apply bilateral filtering on high-frequency (Laplacian) bands
/// 3. Compute MIP across slices for each Laplacian pyramid level
/// 4. Fuse high-frequency MIP with central slice using nonlinear combination
/// 5. Reconstruct the final image

/// Fusion parameters (α, β, p) for a pyramid level
/// Formula: r(k) = α * Expand(g_{k+1}) + β * (Expand(g_{k+1}))^p * (L_k + ML_k)
#[derive(Debug, Clone, Copy)]
pub struct FusionParams {
    pub alpha: f32,
    pub beta: f32,
    pub p: f32,
}

impl FusionParams {
    pub const fn new(alpha: f32, beta: f32, p: f32) -> Self {
        Self { alpha, beta, p }
    }
}

/// Default fusion parameters from the paper (Table 1) for 7 pyramid levels
/// Level indices are 0-based (level 0 = finest detail, level 6 = coarsest)
pub const DEFAULT_FUSION_PARAMS: [FusionParams; 7] = [
    FusionParams::new(1.0, 2.0, 1.0), // Level 0 (paper level 1) - fine detail
    FusionParams::new(1.0, 2.0, 1.0), // Level 1 (paper level 2) - fine detail
    FusionParams::new(1.0, 2.0, 1.0), // Level 2 (paper level 3) - fine detail
    FusionParams::new(1.0, 1.0, 2.0), // Level 3 (paper level 4) - middle
    FusionParams::new(1.0, 1.0, 2.0), // Level 4 (paper level 5) - middle
    FusionParams::new(0.4, 1.0, 2.0), // Level 5 (paper level 6) - coarse
    FusionParams::new(0.2, 1.0, 2.0), // Level 6 (paper level 7) - coarsest
];

#[derive(Debug, Clone)]
pub struct LaplacianMip {
    /// Number of pyramid levels (default 7, matching paper)
    pub num_levels: usize,
    /// Bilateral filter spatial sigma (default 3.0, matching paper)
    pub bilateral_sigma_s: f32,
    /// Bilateral filter range sigma - static fallback value (default 0.015)
    /// Literature recommends 0.005-0.02 for calcification preservation (PMC4277560)
    /// Used when adaptive_sigma_r is false, or as minimum when adaptive
    pub bilateral_sigma_r_frac: f32,
    /// Use adaptive sigma_r estimation based on local variance (default true, matching paper)
    pub adaptive_sigma_r: bool,
    /// Patch size for local variance estimation (default 16)
    pub variance_patch_size: usize,
    /// Fusion parameters per level (α, β, p). If fewer than num_levels, last entry is repeated.
    pub fusion_params: Vec<FusionParams>,
    /// Skip bilateral filtering at first N pyramid levels (default 0)
    /// Set to 3 to preserve calcifications at fine detail levels (0-2)
    /// Fine levels contain high-frequency details like calcifications
    pub skip_bilateral_levels: usize,
    /// First pyramid level to include MIP Laplacian contribution (default 0 = all levels)
    /// Set to 3 to exclude MIP at fine levels (0-2), using only central Laplacian there.
    /// Higher values produce smoother output closer to vendor synthesized mammograms.
    /// Level 0 = finest detail (calcifications), Level 6 = coarsest (overall brightness)
    pub mip_levels_start: usize,
    /// Weight for MIP Laplacian contribution (default 1.0)
    /// Values > 1.0 amplify MIP features (calcifications), < 1.0 reduce them
    /// Formula becomes: L_central + mip_weight * L_mip
    pub mip_weight: f32,
    /// Use MIP Gaussian (instead of central) as reconstruction base at fine levels (default false)
    /// When true, fine levels (0 to mip_gaussian_levels-1) use MIP Gaussian, preserving
    /// bright calcification centers that may be out of focus in central slice
    pub use_mip_gaussian_base: bool,
    /// Number of fine levels to use MIP Gaussian base (default 3 = levels 0, 1, 2)
    /// Only used when use_mip_gaussian_base is true
    pub mip_gaussian_levels: usize,
    /// Skip frames at start (default 5 to exclude noisy edge frames)
    skip_start: u32,
    /// Skip frames at end (default 5 to exclude noisy edge frames)
    skip_end: u32,
}

impl Default for LaplacianMip {
    fn default() -> Self {
        Self {
            num_levels: 7,
            bilateral_sigma_s: 3.0,
            bilateral_sigma_r_frac: 0.015, // Literature recommends 0.005-0.02 (PMC4277560)
            adaptive_sigma_r: false,       // Static value works better for FFDM-like output
            variance_patch_size: 16,
            fusion_params: DEFAULT_FUSION_PARAMS.to_vec(),
            skip_bilateral_levels: 0, // 0 = filter all levels; set to 3 for calc-friendly mode
            mip_levels_start: 0,      // 0 = include MIP at all levels
            mip_weight: 1.0,          // 1.0 = equal weight; >1 amplifies calcifications
            use_mip_gaussian_base: false, // Use central gaussian by default
            mip_gaussian_levels: 3,   // When enabled, use MIP gaussian at levels 0-2
            skip_start: 5,
            skip_end: 5,
        }
    }
}

impl LaplacianMip {
    /// Create a new LaplacianMip handler
    pub fn new(num_levels: usize, skip_start: u32, skip_end: u32) -> Self {
        Self {
            num_levels,
            skip_start,
            skip_end,
            ..Default::default()
        }
    }

    /// Set bilateral filter range sigma (lower = sharper, preserves small features like calcifications)
    pub fn with_sigma_r(mut self, sigma_r: f32) -> Self {
        self.bilateral_sigma_r_frac = sigma_r;
        self
    }

    /// Set bilateral filter spatial sigma
    pub fn with_sigma_s(mut self, sigma_s: f32) -> Self {
        self.bilateral_sigma_s = sigma_s;
        self
    }

    /// Set beta for fine detail levels (0-2). Higher values enhance calcifications more.
    pub fn with_beta_fine(mut self, beta: f32) -> Self {
        for i in 0..3.min(self.fusion_params.len()) {
            self.fusion_params[i].beta = beta;
        }
        self
    }

    /// Set beta for middle levels (3-4)
    pub fn with_beta_mid(mut self, beta: f32) -> Self {
        for i in 3..5.min(self.fusion_params.len()) {
            self.fusion_params[i].beta = beta;
        }
        self
    }

    /// Set number of fine pyramid levels to skip bilateral filtering
    /// Levels 0-2 contain high-frequency details like calcifications
    /// Set to 3 to preserve calcifications (skip filtering at levels 0, 1, 2)
    pub fn with_skip_bilateral_levels(mut self, levels: usize) -> Self {
        self.skip_bilateral_levels = levels;
        self
    }

    /// Set first pyramid level to include MIP Laplacian contribution
    /// Levels below this use only central Laplacian (smoother, closer to vendor synth)
    /// Set to 3 to exclude MIP at fine levels (0-2), including MIP only at levels 3-6
    /// Higher values = smoother output, lower values = more MIP-like sharpness
    pub fn with_mip_levels_start(mut self, level: usize) -> Self {
        self.mip_levels_start = level;
        self
    }

    /// Set weight for MIP Laplacian contribution
    /// Values > 1.0 amplify MIP features (calcifications), < 1.0 reduce them
    /// Default is 1.0 (equal weight with central Laplacian)
    pub fn with_mip_weight(mut self, weight: f32) -> Self {
        self.mip_weight = weight;
        self
    }

    /// Enable using MIP Gaussian as reconstruction base at fine levels
    /// This preserves bright calcification centers that may be out of focus in central slice
    pub fn with_mip_gaussian_base(mut self, enable: bool) -> Self {
        self.use_mip_gaussian_base = enable;
        self
    }

    /// Set number of fine levels to use MIP Gaussian base (default 3 = levels 0, 1, 2)
    /// Only applies when use_mip_gaussian_base is true
    pub fn with_mip_gaussian_levels(mut self, levels: usize) -> Self {
        self.mip_gaussian_levels = levels;
        self
    }

    /// Get enhancement parameters for a pyramid level
    /// Returns FusionParams (alpha, beta, p) for fusion formula:
    /// r(k) = α * Expand(g_{k+1}) + β * (Expand(g_{k+1}))^p * (L_k + ML_k)
    fn get_level_params(&self, level: usize) -> FusionParams {
        if level < self.fusion_params.len() {
            self.fusion_params[level]
        } else if let Some(last) = self.fusion_params.last() {
            // If we have more levels than params, repeat the last one
            *last
        } else {
            // Fallback to paper's coarsest level params
            FusionParams::new(0.2, 1.0, 2.0)
        }
    }

    /// Estimate noise level using local variance in patches centered on tissue
    /// Returns the minimum local standard deviation from tissue regions,
    /// which approximates the noise floor (homogeneous tissue has variance dominated by noise)
    fn estimate_sigma_r(
        data: &[f32],
        width: usize,
        height: usize,
        patch_size: usize,
        min_sigma: f32,
    ) -> f32 {
        // Find a seed pixel that's definitely tissue (intensity > 0.5)
        // This avoids background which is typically exactly 0
        let mut seed_x = width / 2;
        let mut seed_y = height / 2;
        let mut found_seed = false;

        // Search for a bright pixel, starting from center and spiraling out
        'outer: for radius in 0..((width.max(height) / 2) as i32) {
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    if dx.abs() != radius && dy.abs() != radius {
                        continue; // Only check perimeter of current radius
                    }
                    let x = (width as i32 / 2 + dx).clamp(0, width as i32 - 1) as usize;
                    let y = (height as i32 / 2 + dy).clamp(0, height as i32 - 1) as usize;
                    if data[y * width + x] > 0.3 {
                        seed_x = x;
                        seed_y = y;
                        found_seed = true;
                        break 'outer;
                    }
                }
            }
        }

        if !found_seed {
            tracing::debug!(
                "No tissue seed found (no pixel > 0.3), using min_sigma={:.4}",
                min_sigma
            );
            return min_sigma;
        }

        // Define sampling region centered on seed (e.g., 512x512 around seed)
        let sample_radius = 256usize;
        let x_start = seed_x.saturating_sub(sample_radius);
        let x_end = (seed_x + sample_radius).min(width);
        let y_start = seed_y.saturating_sub(sample_radius);
        let y_end = (seed_y + sample_radius).min(height);

        let region_w = x_end - x_start;
        let region_h = y_end - y_start;
        let patches_x = region_w / patch_size;
        let patches_y = region_h / patch_size;

        if patches_x == 0 || patches_y == 0 {
            return min_sigma;
        }

        let mut min_std = f32::MAX;
        let mut valid_patches = 0usize;

        // Sample patches within the tissue-centered region
        for py in 0..patches_y {
            for px in 0..patches_x {
                let x0 = x_start + px * patch_size;
                let y0 = y_start + py * patch_size;

                // Compute mean of this patch
                let mut sum = 0.0f32;
                let mut count = 0usize;
                for y in y0..(y0 + patch_size).min(y_end) {
                    for x in x0..(x0 + patch_size).min(x_end) {
                        sum += data[y * width + x];
                        count += 1;
                    }
                }
                if count == 0 {
                    continue;
                }
                let mean = sum / count as f32;

                // Skip patches that are background (near zero)
                if mean < 0.05 {
                    continue;
                }

                // Compute variance
                let mut var_sum = 0.0f32;
                for y in y0..(y0 + patch_size).min(y_end) {
                    for x in x0..(x0 + patch_size).min(x_end) {
                        let diff = data[y * width + x] - mean;
                        var_sum += diff * diff;
                    }
                }
                let std = (var_sum / count as f32).sqrt();

                // Track minimum (most homogeneous tissue patch = noise floor estimate)
                if std < min_std && std > 0.0 {
                    min_std = std;
                }
                valid_patches += 1;
            }
        }

        // If no valid patches found, fall back to minimum
        if valid_patches == 0 || min_std == f32::MAX {
            tracing::debug!(
                "No valid tissue patches in region around ({}, {}), using min_sigma={:.4}",
                seed_x,
                seed_y,
                min_sigma
            );
            return min_sigma;
        }

        // Use minimum std as noise estimate, but enforce a floor
        // Multiply by a factor to account for the bilateral filter's behavior
        // (sigma_r should be ~2-3x the noise level for good edge preservation)
        let estimated = min_std * 2.5;
        tracing::debug!(
            "Local variance: seed=({}, {}), valid_patches={}, min_std={:.6}, estimate={:.6}",
            seed_x,
            seed_y,
            valid_patches,
            min_std,
            estimated.max(min_sigma)
        );
        estimated.max(min_sigma)
    }

    /// Convert DynamicImage to grayscale f32 buffer (normalized to 0-1 range)
    fn to_grayscale_f32(img: &DynamicImage) -> Vec<f32> {
        // Use 16-bit to preserve full dynamic range
        let gray = img.to_luma16();
        gray.pixels().map(|p| p.0[0] as f32 / 65535.0).collect()
    }

    /// Convert f32 buffer (0-1 range) back to DynamicImage (16-bit grayscale)
    fn from_grayscale_f32(data: &[f32], width: u32, height: u32) -> DynamicImage {
        let mut img = image::ImageBuffer::<image::Luma<u16>, Vec<u16>>::new(width, height);
        for (i, pixel) in img.pixels_mut().enumerate() {
            let val = (data[i].clamp(0.0, 1.0) * 65535.0) as u16;
            pixel.0[0] = val;
        }
        DynamicImage::ImageLuma16(img)
    }

    /// Gaussian blur using separable 5x5 kernel [1,4,6,4,1]/16
    fn gaussian_blur(data: &[f32], width: usize, height: usize) -> Vec<f32> {
        let kernel = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];

        // Horizontal pass
        let mut temp = vec![0.0f32; width * height];
        for y in 0..height {
            for x in 0..width {
                let mut sum = 0.0;
                for (ki, &kv) in kernel.iter().enumerate() {
                    let xi = (x as i32 + ki as i32 - 2).clamp(0, width as i32 - 1) as usize;
                    sum += data[y * width + xi] * kv;
                }
                temp[y * width + x] = sum;
            }
        }

        // Vertical pass
        let mut result = vec![0.0f32; width * height];
        for y in 0..height {
            for x in 0..width {
                let mut sum = 0.0;
                for (ki, &kv) in kernel.iter().enumerate() {
                    let yi = (y as i32 + ki as i32 - 2).clamp(0, height as i32 - 1) as usize;
                    sum += temp[yi * width + x] * kv;
                }
                result[y * width + x] = sum;
            }
        }

        result
    }

    /// Downsample by factor of 2 (after blur)
    fn downsample(data: &[f32], width: usize, height: usize) -> (Vec<f32>, usize, usize) {
        let new_width = (width + 1) / 2;
        let new_height = (height + 1) / 2;
        let mut result = vec![0.0f32; new_width * new_height];

        for y in 0..new_height {
            for x in 0..new_width {
                result[y * new_width + x] = data[(y * 2) * width + (x * 2)];
            }
        }

        (result, new_width, new_height)
    }

    /// Upsample by factor of 2 using bilinear interpolation
    fn upsample(
        data: &[f32],
        width: usize,
        height: usize,
        target_w: usize,
        target_h: usize,
    ) -> Vec<f32> {
        let mut result = vec![0.0f32; target_w * target_h];

        for y in 0..target_h {
            for x in 0..target_w {
                // Map to source coordinates
                let src_x = x as f32 * (width - 1) as f32 / (target_w - 1).max(1) as f32;
                let src_y = y as f32 * (height - 1) as f32 / (target_h - 1).max(1) as f32;

                let x0 = src_x.floor() as usize;
                let y0 = src_y.floor() as usize;
                let x1 = (x0 + 1).min(width - 1);
                let y1 = (y0 + 1).min(height - 1);

                let fx = src_x - x0 as f32;
                let fy = src_y - y0 as f32;

                let v00 = data[y0 * width + x0];
                let v10 = data[y0 * width + x1];
                let v01 = data[y1 * width + x0];
                let v11 = data[y1 * width + x1];

                result[y * target_w + x] = v00 * (1.0 - fx) * (1.0 - fy)
                    + v10 * fx * (1.0 - fy)
                    + v01 * (1.0 - fx) * fy
                    + v11 * fx * fy;
            }
        }

        result
    }

    /// Build Gaussian pyramid (returns list of (data, width, height) at each level)
    fn build_gaussian_pyramid(
        data: &[f32],
        width: usize,
        height: usize,
        num_levels: usize,
    ) -> Vec<(Vec<f32>, usize, usize)> {
        let mut pyramid = Vec::with_capacity(num_levels);
        let mut current = data.to_vec();
        let mut w = width;
        let mut h = height;

        for _ in 0..num_levels {
            pyramid.push((current.clone(), w, h));
            if w <= 2 || h <= 2 {
                break;
            }
            let blurred = Self::gaussian_blur(&current, w, h);
            let (downsampled, new_w, new_h) = Self::downsample(&blurred, w, h);
            current = downsampled;
            w = new_w;
            h = new_h;
        }

        pyramid
    }

    /// Build Laplacian pyramid from Gaussian pyramid
    /// Laplacian[k] = Gaussian[k] - Expand(Gaussian[k+1])
    fn build_laplacian_pyramid(
        gaussian: &[(Vec<f32>, usize, usize)],
    ) -> Vec<(Vec<f32>, usize, usize)> {
        let num_levels = gaussian.len();
        let mut laplacian = Vec::with_capacity(num_levels);

        for k in 0..num_levels - 1 {
            let (g_k, w_k, h_k) = &gaussian[k];
            let (g_k1, w_k1, h_k1) = &gaussian[k + 1];

            // Expand g_{k+1} to size of g_k
            let expanded = Self::upsample(g_k1, *w_k1, *h_k1, *w_k, *h_k);

            // Laplacian = g_k - expanded
            let lap: Vec<f32> = g_k
                .iter()
                .zip(expanded.iter())
                .map(|(a, b)| a - b)
                .collect();

            laplacian.push((lap, *w_k, *h_k));
        }

        // Lowest frequency band is the coarsest Gaussian level
        let (g_last, w_last, h_last) = &gaussian[num_levels - 1];
        laplacian.push((g_last.clone(), *w_last, *h_last));

        laplacian
    }

    /// Simple bilateral filter approximation for edge-preserving smoothing
    fn bilateral_filter(
        data: &[f32],
        width: usize,
        height: usize,
        sigma_s: f32,
        sigma_r: f32,
    ) -> Vec<f32> {
        let radius = (sigma_s * 2.0).ceil() as i32;
        let mut result = vec![0.0f32; width * height];

        for y in 0..height {
            for x in 0..width {
                let center_val = data[y * width + x];
                let mut sum = 0.0f32;
                let mut weight_sum = 0.0f32;

                for dy in -radius..=radius {
                    for dx in -radius..=radius {
                        let nx = (x as i32 + dx).clamp(0, width as i32 - 1) as usize;
                        let ny = (y as i32 + dy).clamp(0, height as i32 - 1) as usize;

                        let neighbor_val = data[ny * width + nx];

                        // Spatial weight (Gaussian)
                        let spatial_dist = (dx * dx + dy * dy) as f32;
                        let spatial_weight = (-spatial_dist / (2.0 * sigma_s * sigma_s)).exp();

                        // Range weight (Gaussian on intensity difference)
                        let range_dist = (neighbor_val - center_val).powi(2);
                        let range_weight = (-range_dist / (2.0 * sigma_r * sigma_r)).exp();

                        let weight = spatial_weight * range_weight;
                        sum += neighbor_val * weight;
                        weight_sum += weight;
                    }
                }

                result[y * width + x] = if weight_sum > 0.0 {
                    sum / weight_sum
                } else {
                    center_val
                };
            }
        }

        result
    }

    /// Reconstruct image from fused pyramid using nonlinear combination
    /// Paper formula: r(k) = α * Expand(g_{k+1}) + β * (Expand(g_{k+1}))^p * (L_k + ML_k)
    fn reconstruct_with_fusion(
        &self,
        central_laplacian: &[(Vec<f32>, usize, usize)],
        mip_laplacian: &[(Vec<f32>, usize, usize)],
        central_gaussian: &[(Vec<f32>, usize, usize)],
        mip_gaussian: &[(Vec<f32>, usize, usize)],
        num_levels: usize,
    ) -> Vec<f32> {
        // Start from coarsest level (this IS g_n, the coarsest Gaussian)
        let (mut result, mut _w, mut _h) = central_gaussian.last().unwrap().clone();

        // Reconstruct from coarse to fine
        for level in (0..num_levels - 1).rev() {
            let (l_central, w_l, h_l) = &central_laplacian[level];
            let (l_mip, _, _) = &mip_laplacian[level];

            // Select Gaussian base: use MIP Gaussian at fine levels if enabled
            // This preserves bright calcification centers that may be out of focus in central
            let use_mip_base = self.use_mip_gaussian_base && level < self.mip_gaussian_levels;
            let (g_k1, w_k1, h_k1) = if use_mip_base {
                &mip_gaussian[level + 1]
            } else {
                &central_gaussian[level + 1]
            };

            // Get fusion parameters for this level
            let params = self.get_level_params(level);

            // Expand(g_{k+1}) - upsample the Gaussian at level k+1
            let g_expanded = Self::upsample(g_k1, *w_k1, *h_k1, *w_l, *h_l);

            // Apply fusion formula:
            // r(k) = α * Expand(g_{k+1}) + β * (Expand(g_{k+1}))^p * (L_k + ML_k)
            // When level < mip_levels_start, exclude MIP Laplacian for smoother output
            let include_mip = level >= self.mip_levels_start;

            result = Vec::with_capacity(w_l * h_l);
            for i in 0..(w_l * h_l) {
                let g_exp = g_expanded[i];
                let l_k = l_central[i];

                // Combine central Laplacian with weighted MIP Laplacian (if enabled for this level)
                // mip_weight > 1.0 amplifies calcifications
                let combined_lap = if include_mip {
                    l_k + self.mip_weight * l_mip[i]
                } else {
                    l_k // Central only at excluded levels
                };

                // Apply nonlinear fusion (g_exp is always positive for image data)
                let fused =
                    params.alpha * g_exp + params.beta * g_exp.powf(params.p) * combined_lap;
                result.push(fused);
            }

            _w = *w_l;
            _h = *h_l;
        }

        result
    }

    /// Compute simple MIP across frames
    fn compute_mip(frames: &[Vec<f32>], width: usize, height: usize) -> Vec<f32> {
        let mut mip = vec![f32::MIN; width * height];
        for frame in frames {
            for (i, &val) in frame.iter().enumerate() {
                if val > mip[i] {
                    mip[i] = val;
                }
            }
        }
        mip
    }

    /// Main projection algorithm - matches paper's approach:
    /// 1. Compute MIP across all DBT slices
    /// 2. Build Laplacian pyramid for central slice (L_k)
    /// 3. Build Laplacian pyramid for MIP result (ML_k)
    /// 4. Fuse: r(k) = α*Expand(g_{k+1}) + β*(Expand(g_{k+1}))^p*(L_k + ML_k)
    fn project_laplacian_mip(&self, frames: &[DynamicImage]) -> DynamicImage {
        if frames.is_empty() {
            panic!("Cannot project empty frames");
        }
        if frames.len() == 1 {
            return frames[0].clone();
        }

        let (width, height) = frames[0].dimensions();
        let w = width as usize;
        let h = height as usize;

        // Determine actual number of levels based on image size (paper uses 7)
        let max_levels = ((w.min(h) as f32).log2().floor() as usize).max(2);
        let num_levels = self.num_levels.min(max_levels);

        // Convert all frames to f32 grayscale (normalized 0-1)
        let gray_frames: Vec<Vec<f32>> = frames.iter().map(|f| Self::to_grayscale_f32(f)).collect();

        // Get central frame (projection view - PV)
        let central_idx = frames.len() / 2;
        let central_frame = &gray_frames[central_idx];

        // Step 1: Compute MIP across all frames
        let mip_frame = Self::compute_mip(&gray_frames, w, h);

        // Step 2: Apply bilateral filter to reduce noise (optional)
        // When skip_bilateral_levels > 0, skip filtering to preserve calcifications
        let apply_bilateral = self.skip_bilateral_levels == 0;

        let (central_filtered, mip_filtered) = if apply_bilateral {
            // Estimate sigma_r adaptively based on local variance, or use static value
            let sigma_r_central = if self.adaptive_sigma_r {
                let estimated = Self::estimate_sigma_r(
                    central_frame,
                    w,
                    h,
                    self.variance_patch_size,
                    self.bilateral_sigma_r_frac,
                );
                tracing::debug!(
                    "Adaptive sigma_r (central): {:.4} (min: {:.4})",
                    estimated,
                    self.bilateral_sigma_r_frac
                );
                estimated
            } else {
                self.bilateral_sigma_r_frac
            };

            let sigma_r_mip = if self.adaptive_sigma_r {
                let estimated = Self::estimate_sigma_r(
                    &mip_frame,
                    w,
                    h,
                    self.variance_patch_size,
                    self.bilateral_sigma_r_frac,
                );
                tracing::debug!(
                    "Adaptive sigma_r (MIP): {:.4} (min: {:.4})",
                    estimated,
                    self.bilateral_sigma_r_frac
                );
                estimated
            } else {
                self.bilateral_sigma_r_frac
            };

            (
                Self::bilateral_filter(
                    central_frame,
                    w,
                    h,
                    self.bilateral_sigma_s,
                    sigma_r_central,
                ),
                Self::bilateral_filter(&mip_frame, w, h, self.bilateral_sigma_s, sigma_r_mip),
            )
        } else {
            tracing::debug!(
                "Skipping bilateral filter (skip_bilateral_levels={})",
                self.skip_bilateral_levels
            );
            (central_frame.clone(), mip_frame.clone())
        };

        // Step 3: Build Gaussian and Laplacian pyramids for both
        let central_gaussian = Self::build_gaussian_pyramid(&central_filtered, w, h, num_levels);
        let mip_gaussian = Self::build_gaussian_pyramid(&mip_filtered, w, h, num_levels);

        let central_laplacian = Self::build_laplacian_pyramid(&central_gaussian);
        let mip_laplacian = Self::build_laplacian_pyramid(&mip_gaussian);

        // Step 4: Fuse using paper's formula
        let reconstructed = self.reconstruct_with_fusion(
            &central_laplacian,
            &mip_laplacian,
            &central_gaussian,
            &mip_gaussian,
            num_levels,
        );

        // Convert back to image
        Self::from_grayscale_f32(&reconstructed, width, height)
    }
}

impl HandleVolume for LaplacianMip {
    fn decode_volume_with_options(
        &self,
        file: &FileDicomObject<InMemDicomObject>,
        options: &ConvertOptions,
    ) -> Result<Vec<DynamicImage>, DicomError> {
        let number_of_frames: u32 = FrameCount::try_from(file)?.into();
        let start = min(number_of_frames, self.skip_start);
        let end = max(0, number_of_frames as i64 - self.skip_end as i64) as u32;

        if start >= end || start >= number_of_frames {
            return Err(DicomError::FrameIndexError {
                start: start as usize,
                end: end as usize,
                number_of_frames: number_of_frames as usize,
            });
        }

        let mut frames = Vec::with_capacity((end - start) as usize);
        for frame_number in start..end {
            let decoded = file
                .decode_pixel_data_frame(frame_number)
                .context(PixelDataSnafu)?;
            frames.push(
                decoded
                    .to_dynamic_image_with_options(0, options)
                    .context(PixelDataSnafu)?,
            );
        }

        Ok(vec![self.project_laplacian_mip(&frames)])
    }

    fn par_decode_volume_with_options(
        &self,
        file: &FileDicomObject<InMemDicomObject>,
        options: &ConvertOptions,
    ) -> Result<Vec<DynamicImage>, DicomError> {
        let number_of_frames: u32 = FrameCount::try_from(file)?.into();
        let start = min(number_of_frames, self.skip_start);
        let end = max(0, number_of_frames as i64 - self.skip_end as i64) as u32;

        if start >= end || start >= number_of_frames {
            return Err(DicomError::FrameIndexError {
                start: start as usize,
                end: end as usize,
                number_of_frames: number_of_frames as usize,
            });
        }

        // Parallel frame decoding
        let frames = (start..end)
            .into_par_iter()
            .map(|frame_number| {
                let frame = file
                    .decode_pixel_data_frame(frame_number)
                    .context(PixelDataSnafu)?
                    .to_dynamic_image_with_options(0, options)
                    .context(PixelDataSnafu)?;
                Ok::<DynamicImage, DicomError>(frame)
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Projection is done serially as it's already computationally intensive
        Ok(vec![self.project_laplacian_mip(&frames)])
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
