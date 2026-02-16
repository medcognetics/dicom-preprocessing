//! # Volume Handlers for DBT and Multi-frame DICOM
//!
//! This module provides handlers for processing multi-frame DICOM volumes,
//! particularly Digital Breast Tomosynthesis (DBT) data.
//!
//! ## LaplacianMip - Synthesized Mammography from DBT
//!
//! The [`LaplacianMip`] handler implements Laplacian pyramid + MIP fusion for generating
//! 2D mammogram-like images from 3D digital breast tomosynthesis volumes.
//!
//! ### Reference
//!
//! Garrett JW, et al. "Synthesizing Mammogram from Digital Breast Tomosynthesis
//! Using Laplacian Pyramid Decomposition and Maximum Intensity Projection."
//! Proc SPIE Int Soc Opt Eng. 2018;10577:105770N.
//! <https://pmc.ncbi.nlm.nih.gov/articles/PMC6438841/>
//!
//! ### Algorithm Overview
//!
//! 1. Compute MIP across all DBT slices
//! 2. Build Laplacian pyramids for central slice and MIP
//! 3. Apply bilateral filtering for edge-preserving noise reduction
//! 4. Fuse pyramids using level-specific parameters from paper Table 1:
//!    `r(k) = α·Expand(g_{k+1}) + β·(Expand(g_{k+1}))^p · (L_k + w·ML_k)`
//!
//! ### Key Parameters
//!
//! - `mip_weight`: Weight for MIP Laplacian contribution (default 1.5).
//!   Higher values preserve calcifications better.
//! - `skip_start`/`skip_end`: Trim noisy edge frames (default 5).

use crate::errors::{dicom::PixelDataSnafu, DicomError};
use crate::metadata::preprocessing::FrameCount;
use dicom::dictionary_std::tags;
use dicom::object::{FileDicomObject, InMemDicomObject};
use dicom::pixeldata::{ConvertOptions, PixelDecoder};
use image::DynamicImage;
use image::Pixel;
use image::{GenericImage, GenericImageView};
use rayon::prelude::*;
use snafu::ResultExt;
use std::cmp::{max, min};
use std::fmt;

pub const DEFAULT_INTERPOLATE_TARGET_FRAMES: u32 = 32;
const PARALLEL_MIN_PIXEL_COUNT: usize = 4_096;
const PARALLEL_MIN_FRAME_COUNT: usize = 2;
const PARALLEL_MIN_TARGET_FRAMES: usize = 2;
const BILATERAL_RADIUS_SIGMA_MULTIPLIER: f32 = 2.0;
const BILATERAL_RANGE_LUT_BINS: usize = 2_048;
const NORMALIZED_INTENSITY_SQUARED_MAX: f32 = 1.0;

#[derive(Debug, Clone)]
pub enum VolumeHandler {
    Keep(KeepVolume),
    CentralSlice(CentralSlice),
    MaxIntensity(MaxIntensity),
    Interpolate(InterpolateVolume),
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

fn has_malformed_basic_offset_table(
    file: &FileDicomObject<InMemDicomObject>,
) -> Result<bool, DicomError> {
    let number_of_frames: u32 = FrameCount::try_from(file)?.into();
    let Some(pixel_data) = file.get(tags::PIXEL_DATA) else {
        return Ok(false);
    };
    let value = pixel_data.value();
    let Some(offset_table) = value.offset_table() else {
        return Ok(false);
    };
    let Some(fragments) = value.fragments() else {
        return Ok(false);
    };

    if offset_table.is_empty() || number_of_frames <= 1 {
        return Ok(false);
    }

    let number_of_frames = number_of_frames as usize;
    if fragments.len() == 1 || fragments.len() == number_of_frames {
        // The decoder does not consult BOT when there is one fragment per frame.
        return Ok(false);
    }

    let offset_count_matches_frames = offset_table.len() == number_of_frames;
    let starts_at_zero = offset_table.first().copied() == Some(0);
    let strictly_increasing = offset_table.windows(2).all(|window| window[0] < window[1]);
    let encoded_stream_len = fragments.iter().fold(0usize, |acc, fragment| {
        acc.saturating_add(fragment.len().saturating_add(8))
    });
    let offsets_in_bounds = offset_table
        .iter()
        .all(|&offset| (offset as usize) < encoded_stream_len);

    Ok(
        !(offset_count_matches_frames
            && starts_at_zero
            && strictly_increasing
            && offsets_in_bounds),
    )
}

fn decode_all_frames_with_options(
    file: &FileDicomObject<InMemDicomObject>,
    options: &ConvertOptions,
) -> Result<Vec<DynamicImage>, DicomError> {
    let number_of_frames: u32 = FrameCount::try_from(file)?.into();
    let decoded = file.decode_pixel_data().context(PixelDataSnafu)?;
    let mut image_data = Vec::with_capacity(number_of_frames as usize);
    for frame_number in 0..number_of_frames {
        image_data.push(
            decoded
                .to_dynamic_image_with_options(frame_number, options)
                .context(PixelDataSnafu)?,
        );
    }
    Ok(image_data)
}

fn maybe_decode_all_frames_with_malformed_bot_fallback(
    file: &FileDicomObject<InMemDicomObject>,
    options: &ConvertOptions,
) -> Result<Option<Vec<DynamicImage>>, DicomError> {
    if has_malformed_basic_offset_table(file)? {
        return Ok(Some(decode_all_frames_with_options(file, options)?));
    }
    Ok(None)
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
        if let Some(images) = maybe_decode_all_frames_with_malformed_bot_fallback(file, options)? {
            return Ok(images);
        }

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
        if let Some(images) = maybe_decode_all_frames_with_malformed_bot_fallback(file, options)? {
            return Ok(images);
        }

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
        if let Some(frames) = maybe_decode_all_frames_with_malformed_bot_fallback(file, options)? {
            return Ok(vec![frames[central_frame as usize].clone()]);
        }

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

        if let Some(frames) = maybe_decode_all_frames_with_malformed_bot_fallback(file, options)? {
            let frame_count = (end - start) as usize;
            let mut frame_iter = frames.into_iter().skip(start as usize).take(frame_count);
            let mut image = frame_iter.next().ok_or(DicomError::FrameIndexError {
                start: start as usize,
                end: end as usize,
                number_of_frames: number_of_frames as usize,
            })?;
            for frame in frame_iter {
                image = Self::reduce(image, frame);
            }
            return Ok(vec![image]);
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

        if let Some(frames) = maybe_decode_all_frames_with_malformed_bot_fallback(file, options)? {
            let frame_count = (end - start) as usize;
            let mut frame_iter = frames.into_iter().skip(start as usize).take(frame_count);
            let mut image = frame_iter.next().ok_or(DicomError::FrameIndexError {
                start: start as usize,
                end: end as usize,
                number_of_frames: number_of_frames as usize,
            })?;
            for frame in frame_iter {
                image = Self::reduce(image, frame);
            }
            return Ok(vec![image]);
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

    fn should_parallelize_interpolation(
        input_frames: usize,
        target_frames: usize,
        pixel_count: usize,
    ) -> bool {
        input_frames >= PARALLEL_MIN_FRAME_COUNT
            && target_frames >= PARALLEL_MIN_TARGET_FRAMES
            && pixel_count >= PARALLEL_MIN_PIXEL_COUNT
    }

    fn interpolate_single_frame(
        frames: &[DynamicImage],
        target_frames: u32,
        output_frame_idx: u32,
    ) -> DynamicImage {
        let t = output_frame_idx as f32 / (target_frames - 1) as f32;
        let frame_idx = t * (frames.len() - 1) as f32;
        let frame_idx_floor = frame_idx.floor() as usize;
        let frame_idx_ceil = frame_idx.ceil() as usize;
        let alpha = frame_idx - frame_idx_floor as f32;
        let (width, height) = frames[0].dimensions();

        // Create a new image with the same color type as the input.
        let mut interpolated = frames[0].clone();
        for x in 0..width {
            for y in 0..height {
                let pixel1 = frames[frame_idx_floor].get_pixel(x, y);
                let pixel2 = frames[frame_idx_ceil].get_pixel(x, y);
                let interpolated_pixel = pixel1.map2(&pixel2, |p1, p2| {
                    let p1 = p1 as f32;
                    let p2 = p2 as f32;
                    (p1 * (1.0 - alpha) + p2 * alpha) as u8
                });
                interpolated.put_pixel(x, y, interpolated_pixel);
            }
        }
        interpolated
    }

    fn interpolate_frames_serial(frames: &[DynamicImage], target_frames: u32) -> Vec<DynamicImage> {
        let mut result = Vec::with_capacity(target_frames as usize);
        for i in 0..target_frames {
            result.push(Self::interpolate_single_frame(frames, target_frames, i));
        }
        result
    }

    fn interpolate_frames_parallel(
        frames: &[DynamicImage],
        target_frames: u32,
    ) -> Vec<DynamicImage> {
        (0..target_frames)
            .into_par_iter()
            .map(|i| Self::interpolate_single_frame(frames, target_frames, i))
            .collect()
    }

    /// Interpolate between frames using linear interpolation
    pub fn interpolate_frames(frames: &[DynamicImage], target_frames: u32) -> Vec<DynamicImage> {
        if frames.is_empty() {
            return vec![];
        }
        if frames.len() == 1 {
            return frames.to_vec();
        }
        if target_frames <= 1 {
            return vec![frames[0].clone()];
        }

        let (width, height) = frames[0].dimensions();
        let pixel_count = (width as usize).saturating_mul(height as usize);
        if Self::should_parallelize_interpolation(frames.len(), target_frames as usize, pixel_count)
        {
            return Self::interpolate_frames_parallel(frames, target_frames);
        }

        Self::interpolate_frames_serial(frames, target_frames)
    }
}

impl HandleVolume for InterpolateVolume {
    fn decode_volume_with_options(
        &self,
        file: &FileDicomObject<InMemDicomObject>,
        options: &ConvertOptions,
    ) -> Result<Vec<DynamicImage>, DicomError> {
        if let Some(frames) = maybe_decode_all_frames_with_malformed_bot_fallback(file, options)? {
            return Ok(Self::interpolate_frames(&frames, self.target_frames));
        }

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
        if let Some(frames) = maybe_decode_all_frames_with_malformed_bot_fallback(file, options)? {
            return Ok(Self::interpolate_frames(&frames, self.target_frames));
        }

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

/// How to compute the central frame for Laplacian pyramid fusion
#[derive(Debug, Clone, Copy, Default)]
pub enum ProjectionMode {
    /// Use the central slice
    CentralSlice,
    /// Sum all slices along z-axis (parallel beam approximation)
    #[default]
    ParallelBeam,
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
///
/// ## Fusion parameters (α, β, p) for a pyramid level
///
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
    num_levels: usize,
    /// Bilateral filter spatial sigma (default 3.0, matching paper)
    bilateral_sigma_s: f32,
    /// Bilateral filter range sigma (default 0.015)
    /// Literature recommends 0.005-0.02 for calcification preservation (PMC4277560)
    bilateral_sigma_r_frac: f32,
    /// Fusion parameters per level (α, β, p). If fewer than num_levels, last entry is repeated.
    fusion_params: Vec<FusionParams>,
    /// Weight for MIP Laplacian contribution (default 1.5)
    /// Values > 1.0 amplify MIP features (calcifications), < 1.0 reduce them
    /// Formula becomes: L_central + mip_weight * L_mip
    pub mip_weight: f32,
    /// Skip frames at start (default 5 to exclude noisy edge frames)
    pub skip_start: u32,
    /// Skip frames at end (default 5 to exclude noisy edge frames)
    pub skip_end: u32,
    /// How to compute the central frame for fusion
    pub projection_mode: ProjectionMode,
}

impl Default for LaplacianMip {
    fn default() -> Self {
        Self {
            num_levels: 7,
            bilateral_sigma_s: 3.0,
            bilateral_sigma_r_frac: 0.015,
            fusion_params: DEFAULT_FUSION_PARAMS.to_vec(),
            mip_weight: 1.5,
            skip_start: 5,
            skip_end: 5,
            projection_mode: ProjectionMode::default(),
        }
    }
}

impl LaplacianMip {
    /// Create a new LaplacianMip handler with custom frame skip parameters
    pub fn new(skip_start: u32, skip_end: u32) -> Self {
        Self {
            skip_start,
            skip_end,
            ..Default::default()
        }
    }

    /// Set weight for MIP Laplacian contribution
    /// Values > 1.0 amplify MIP features (calcifications), < 1.0 reduce them
    /// Default is 1.5
    pub fn with_mip_weight(mut self, weight: f32) -> Self {
        self.mip_weight = weight;
        self
    }

    /// Set the projection mode for computing the central frame
    pub fn with_projection_mode(mut self, mode: ProjectionMode) -> Self {
        self.projection_mode = mode;
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

    fn should_parallelize_pixels(pixel_count: usize) -> bool {
        pixel_count >= PARALLEL_MIN_PIXEL_COUNT
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
        let pixel_count = width.saturating_mul(height);
        let use_parallel = Self::should_parallelize_pixels(pixel_count);

        // Horizontal pass
        let mut temp = vec![0.0f32; width * height];
        if use_parallel {
            temp.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
                for (x, out) in row.iter_mut().enumerate() {
                    let mut sum = 0.0;
                    for (ki, &kv) in kernel.iter().enumerate() {
                        let xi = (x as i32 + ki as i32 - 2).clamp(0, width as i32 - 1) as usize;
                        sum += data[y * width + xi] * kv;
                    }
                    *out = sum;
                }
            });
        } else {
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
        }

        // Vertical pass
        let mut result = vec![0.0f32; width * height];
        if use_parallel {
            result
                .par_chunks_mut(width)
                .enumerate()
                .for_each(|(y, row)| {
                    for (x, out) in row.iter_mut().enumerate() {
                        let mut sum = 0.0;
                        for (ki, &kv) in kernel.iter().enumerate() {
                            let yi =
                                (y as i32 + ki as i32 - 2).clamp(0, height as i32 - 1) as usize;
                            sum += temp[yi * width + x] * kv;
                        }
                        *out = sum;
                    }
                });
        } else {
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
        }

        result
    }

    /// Downsample by factor of 2 (after blur)
    fn downsample(data: &[f32], width: usize, height: usize) -> (Vec<f32>, usize, usize) {
        let new_width = width.div_ceil(2);
        let new_height = height.div_ceil(2);
        let mut result = vec![0.0f32; new_width * new_height];
        let use_parallel = Self::should_parallelize_pixels(new_width.saturating_mul(new_height));

        if use_parallel {
            result
                .par_chunks_mut(new_width)
                .enumerate()
                .for_each(|(y, row)| {
                    for (x, out) in row.iter_mut().enumerate() {
                        *out = data[(y * 2) * width + (x * 2)];
                    }
                });
        } else {
            for y in 0..new_height {
                for x in 0..new_width {
                    result[y * new_width + x] = data[(y * 2) * width + (x * 2)];
                }
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
        let use_parallel = Self::should_parallelize_pixels(target_w.saturating_mul(target_h));

        if use_parallel {
            result
                .par_chunks_mut(target_w)
                .enumerate()
                .for_each(|(y, row)| {
                    for (x, out) in row.iter_mut().enumerate() {
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

                        *out = v00 * (1.0 - fx) * (1.0 - fy)
                            + v10 * fx * (1.0 - fy)
                            + v01 * (1.0 - fx) * fy
                            + v11 * fx * fy;
                    }
                });
        } else {
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
            let mut lap = vec![0.0f32; g_k.len()];
            if Self::should_parallelize_pixels(g_k.len()) {
                lap.par_iter_mut()
                    .enumerate()
                    .for_each(|(i, out)| *out = g_k[i] - expanded[i]);
            } else {
                for i in 0..g_k.len() {
                    lap[i] = g_k[i] - expanded[i];
                }
            }

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
        if width == 0 || height == 0 {
            return Vec::new();
        }

        let sigma_s = sigma_s.max(f32::EPSILON);
        let sigma_r = sigma_r.max(f32::EPSILON);
        let radius = (sigma_s * BILATERAL_RADIUS_SIGMA_MULTIPLIER).ceil() as i32;
        let mut result = vec![0.0f32; width * height];
        let use_parallel = Self::should_parallelize_pixels(width.saturating_mul(height));
        let width_i32 = width as i32;
        let height_i32 = height as i32;

        let spatial_divisor = 2.0 * sigma_s * sigma_s;
        let mut spatial_kernel = Vec::with_capacity(((radius * 2 + 1) as usize).saturating_pow(2));
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let spatial_dist = (dx * dx + dy * dy) as f32;
                let spatial_weight = (-spatial_dist / spatial_divisor).exp();
                spatial_kernel.push((dx, dy, spatial_weight));
            }
        }

        let range_divisor = 2.0 * sigma_r * sigma_r;
        let max_lut_index = BILATERAL_RANGE_LUT_BINS - 1;
        let range_lut_scale = max_lut_index as f32 / NORMALIZED_INTENSITY_SQUARED_MAX;
        let range_weight_lut: Vec<f32> = (0..BILATERAL_RANGE_LUT_BINS)
            .map(|index| {
                let range_dist = index as f32 / range_lut_scale;
                (-range_dist / range_divisor).exp()
            })
            .collect();

        let apply_row = |y: usize, row: &mut [f32]| {
            let y_offset = y * width;
            for (x, out) in row.iter_mut().enumerate() {
                let center_val = data[y_offset + x];
                let mut sum = 0.0f32;
                let mut weight_sum = 0.0f32;

                for &(dx, dy, spatial_weight) in &spatial_kernel {
                    let nx = (x as i32 + dx).clamp(0, width_i32 - 1) as usize;
                    let ny = (y as i32 + dy).clamp(0, height_i32 - 1) as usize;
                    let neighbor_val = data[ny * width + nx];
                    let range_dist = (neighbor_val - center_val)
                        .powi(2)
                        .min(NORMALIZED_INTENSITY_SQUARED_MAX);
                    let lut_index = (range_dist * range_lut_scale).round() as usize;
                    let range_weight = range_weight_lut[lut_index.min(max_lut_index)];
                    let weight = spatial_weight * range_weight;
                    sum += neighbor_val * weight;
                    weight_sum += weight;
                }

                *out = if weight_sum > 0.0 {
                    sum / weight_sum
                } else {
                    center_val
                };
            }
        };

        if use_parallel {
            result
                .par_chunks_mut(width)
                .enumerate()
                .for_each(|(y, row)| apply_row(y, row));
        } else {
            for (y, row) in result.chunks_mut(width).enumerate() {
                apply_row(y, row);
            }
        }

        result
    }

    #[cfg(test)]
    fn bilateral_filter_reference(
        data: &[f32],
        width: usize,
        height: usize,
        sigma_s: f32,
        sigma_r: f32,
    ) -> Vec<f32> {
        let radius = (sigma_s * BILATERAL_RADIUS_SIGMA_MULTIPLIER).ceil() as i32;
        let mut result = vec![0.0f32; width * height];
        let use_parallel = Self::should_parallelize_pixels(width.saturating_mul(height));

        let apply_row = |y: usize, row: &mut [f32]| {
            for (x, out) in row.iter_mut().enumerate() {
                let center_val = data[y * width + x];
                let mut sum = 0.0f32;
                let mut weight_sum = 0.0f32;

                for dy in -radius..=radius {
                    for dx in -radius..=radius {
                        let nx = (x as i32 + dx).clamp(0, width as i32 - 1) as usize;
                        let ny = (y as i32 + dy).clamp(0, height as i32 - 1) as usize;
                        let neighbor_val = data[ny * width + nx];
                        let spatial_dist = (dx * dx + dy * dy) as f32;
                        let spatial_weight = (-spatial_dist / (2.0 * sigma_s * sigma_s)).exp();
                        let range_dist = (neighbor_val - center_val).powi(2);
                        let range_weight = (-range_dist / (2.0 * sigma_r * sigma_r)).exp();
                        let weight = spatial_weight * range_weight;
                        sum += neighbor_val * weight;
                        weight_sum += weight;
                    }
                }

                *out = if weight_sum > 0.0 {
                    sum / weight_sum
                } else {
                    center_val
                };
            }
        };

        if use_parallel {
            result
                .par_chunks_mut(width)
                .enumerate()
                .for_each(|(y, row)| apply_row(y, row));
        } else {
            for (y, row) in result.chunks_mut(width).enumerate() {
                apply_row(y, row);
            }
        }

        result
    }

    /// Reconstruct image from fused pyramid using nonlinear combination
    /// Paper formula: r(k) = α * Expand(g_{k+1}) + β * (Expand(g_{k+1}))^p * (L_k + w*ML_k)
    fn reconstruct_with_fusion(
        &self,
        central_laplacian: &[(Vec<f32>, usize, usize)],
        mip_laplacian: &[(Vec<f32>, usize, usize)],
        central_gaussian: &[(Vec<f32>, usize, usize)],
        num_levels: usize,
    ) -> Vec<f32> {
        // Start from coarsest level (this IS g_n, the coarsest Gaussian)
        let (mut result, mut _w, mut _h) = central_gaussian.last().unwrap().clone();

        // Reconstruct from coarse to fine
        for level in (0..num_levels - 1).rev() {
            let (l_central, w_l, h_l) = &central_laplacian[level];
            let (l_mip, _, _) = &mip_laplacian[level];
            let (g_k1, w_k1, h_k1) = &central_gaussian[level + 1];

            // Get fusion parameters for this level
            let params = self.get_level_params(level);

            // Expand(g_{k+1}) - upsample the Gaussian at level k+1
            let g_expanded = Self::upsample(g_k1, *w_k1, *h_k1, *w_l, *h_l);

            // Apply fusion formula:
            // r(k) = α * Expand(g_{k+1}) + β * (Expand(g_{k+1}))^p * (L_k + w*ML_k)
            result = vec![0.0f32; w_l * h_l];
            if Self::should_parallelize_pixels(w_l * h_l) {
                result.par_iter_mut().enumerate().for_each(|(i, out)| {
                    let g_exp = g_expanded[i];
                    let l_k = l_central[i];
                    let combined_lap = l_k + self.mip_weight * l_mip[i];
                    *out = params.alpha * g_exp + params.beta * g_exp.powf(params.p) * combined_lap;
                });
            } else {
                for i in 0..(w_l * h_l) {
                    let g_exp = g_expanded[i];
                    let l_k = l_central[i];
                    let combined_lap = l_k + self.mip_weight * l_mip[i];
                    result[i] =
                        params.alpha * g_exp + params.beta * g_exp.powf(params.p) * combined_lap;
                }
            }

            _w = *w_l;
            _h = *h_l;
        }

        result
    }

    /// Compute simple MIP across frames
    fn compute_mip_serial(frames: &[Vec<f32>], pixel_count: usize) -> Vec<f32> {
        let mut mip = vec![f32::MIN; pixel_count];
        for frame in frames {
            for (i, &val) in frame.iter().enumerate() {
                if val > mip[i] {
                    mip[i] = val;
                }
            }
        }
        mip
    }

    fn compute_mip(frames: &[Vec<f32>], width: usize, height: usize) -> Vec<f32> {
        let pixel_count = width.saturating_mul(height);
        if !Self::should_parallelize_pixels(pixel_count) {
            return Self::compute_mip_serial(frames, pixel_count);
        }
        let mut mip = vec![f32::MIN; pixel_count];
        mip.par_iter_mut().enumerate().for_each(|(i, out)| {
            let mut max_value = f32::MIN;
            for frame in frames {
                let value = frame[i];
                if value > max_value {
                    max_value = value;
                }
            }
            *out = max_value;
        });
        mip
    }

    /// Parallel-beam forward projection: sum all slices along z-axis, then normalize to [0, 1]
    fn compute_parallel_projection_serial(
        gray_frames: &[Vec<f32>],
        pixel_count: usize,
    ) -> Vec<f32> {
        let mut projection = vec![0.0f32; pixel_count];
        for slice in gray_frames {
            for (p, &s) in projection.iter_mut().zip(slice.iter()) {
                *p += s;
            }
        }
        projection
    }

    fn compute_parallel_projection(gray_frames: &[Vec<f32>], w: usize, h: usize) -> Vec<f32> {
        let pixel_count = w.saturating_mul(h);
        let mut projection = if Self::should_parallelize_pixels(pixel_count) {
            let mut output = vec![0.0f32; pixel_count];
            output.par_iter_mut().enumerate().for_each(|(i, out)| {
                let mut sum = 0.0f32;
                for slice in gray_frames {
                    sum += slice[i];
                }
                *out = sum;
            });
            output
        } else {
            Self::compute_parallel_projection_serial(gray_frames, pixel_count)
        };

        if Self::should_parallelize_pixels(pixel_count) {
            let max_val = projection.par_iter().cloned().reduce(|| 0.0f32, f32::max);
            if max_val > 0.0 {
                projection.par_iter_mut().for_each(|p| *p /= max_val);
            }
        } else {
            let max_val = projection.iter().cloned().fold(0.0f32, f32::max);
            if max_val > 0.0 {
                for p in &mut projection {
                    *p /= max_val;
                }
            }
        }
        projection
    }

    #[cfg(test)]
    fn compute_parallel_projection_serial_normalized(
        gray_frames: &[Vec<f32>],
        pixel_count: usize,
    ) -> Vec<f32> {
        let mut projection = Self::compute_parallel_projection_serial(gray_frames, pixel_count);
        let max_val = projection.iter().cloned().fold(0.0f32, f32::max);
        if max_val > 0.0 {
            for p in &mut projection {
                *p /= max_val;
            }
        }
        projection
    }

    /// Project a stack of frames into a single 2D image using Laplacian pyramid + MIP fusion.
    ///
    /// This method is DICOM-independent and operates purely on in-memory images.
    /// It implements the algorithm from Garrett et al. (2018):
    /// 1. Compute MIP across all slices
    /// 2. Build Laplacian pyramid for central slice (L_k)
    /// 3. Build Laplacian pyramid for MIP result (ML_k)
    /// 4. Fuse: r(k) = α·Expand(g_{k+1}) + β·(Expand(g_{k+1}))^p·(L_k + w·ML_k)
    ///
    /// # Errors
    ///
    /// Returns [`DicomError::LaplacianMipEmptyInput`] if `frames` is empty.
    pub fn project_laplacian_mip(
        &self,
        frames: &[DynamicImage],
    ) -> Result<DynamicImage, DicomError> {
        if frames.is_empty() {
            return Err(DicomError::LaplacianMipEmptyInput);
        }
        if frames.len() == 1 {
            return Ok(frames[0].clone());
        }

        let (width, height) = frames[0].dimensions();
        let w = width as usize;
        let h = height as usize;

        // Determine actual number of levels based on image size (paper uses 7)
        let max_levels = ((w.min(h) as f32).log2().floor() as usize).max(2);
        let num_levels = self.num_levels.min(max_levels);

        // Convert all frames to f32 grayscale (normalized 0-1)
        let pixel_count = w.saturating_mul(h);
        let gray_frames: Vec<Vec<f32>> = if Self::should_parallelize_pixels(pixel_count) {
            frames.par_iter().map(Self::to_grayscale_f32).collect()
        } else {
            frames.iter().map(Self::to_grayscale_f32).collect()
        };

        // Get central frame based on projection mode
        let central_frame_owned: Vec<f32>;
        let central_frame: &[f32] = match self.projection_mode {
            ProjectionMode::CentralSlice => {
                let central_idx = gray_frames.len() / 2;
                &gray_frames[central_idx]
            }
            ProjectionMode::ParallelBeam => {
                central_frame_owned = Self::compute_parallel_projection(&gray_frames, w, h);
                &central_frame_owned
            }
        };

        // Step 1: Compute MIP across all frames
        let mip_frame = Self::compute_mip(&gray_frames, w, h);

        // Step 2: Apply bilateral filter to reduce noise
        let central_filtered = Self::bilateral_filter(
            central_frame,
            w,
            h,
            self.bilateral_sigma_s,
            self.bilateral_sigma_r_frac,
        );
        let mip_filtered = Self::bilateral_filter(
            &mip_frame,
            w,
            h,
            self.bilateral_sigma_s,
            self.bilateral_sigma_r_frac,
        );

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
            num_levels,
        );

        // Convert back to image
        Ok(Self::from_grayscale_f32(&reconstructed, width, height))
    }

    fn validate_trimmed_frame_range(
        &self,
        number_of_frames: u32,
    ) -> Result<(u32, u32), DicomError> {
        let start = min(number_of_frames, self.skip_start);
        let end = max(0, number_of_frames as i64 - self.skip_end as i64) as u32;

        if start >= end || start >= number_of_frames {
            return Err(DicomError::LaplacianMipInsufficientFrames {
                number_of_frames: number_of_frames as usize,
                skip_start: self.skip_start as usize,
                skip_end: self.skip_end as usize,
            });
        }

        Ok((start, end))
    }
}

impl HandleVolume for LaplacianMip {
    fn decode_volume_with_options(
        &self,
        file: &FileDicomObject<InMemDicomObject>,
        options: &ConvertOptions,
    ) -> Result<Vec<DynamicImage>, DicomError> {
        let number_of_frames: u32 = FrameCount::try_from(file)?.into();
        let (start, end) = self.validate_trimmed_frame_range(number_of_frames)?;

        if let Some(frames) = maybe_decode_all_frames_with_malformed_bot_fallback(file, options)? {
            let trimmed_frames = frames[start as usize..end as usize].to_vec();
            return Ok(vec![self.project_laplacian_mip(&trimmed_frames)?]);
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

        Ok(vec![self.project_laplacian_mip(&frames)?])
    }

    fn par_decode_volume_with_options(
        &self,
        file: &FileDicomObject<InMemDicomObject>,
        options: &ConvertOptions,
    ) -> Result<Vec<DynamicImage>, DicomError> {
        let number_of_frames: u32 = FrameCount::try_from(file)?.into();
        let (start, end) = self.validate_trimmed_frame_range(number_of_frames)?;

        if let Some(frames) = maybe_decode_all_frames_with_malformed_bot_fallback(file, options)? {
            let trimmed_frames = frames[start as usize..end as usize].to_vec();
            return Ok(vec![self.project_laplacian_mip(&trimmed_frames)?]);
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
        Ok(vec![self.project_laplacian_mip(&frames)?])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use dicom::core::{DataElement, PrimitiveValue, VR};
    use dicom::dictionary_std::tags;
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

    fn split_fragment_even(fragment: &[u8]) -> (Vec<u8>, Vec<u8>) {
        let midpoint = (fragment.len() / 2) & !1;
        let midpoint = midpoint.clamp(2, fragment.len().saturating_sub(2));
        (fragment[..midpoint].to_vec(), fragment[midpoint..].to_vec())
    }

    fn rewrite_to_two_frame_split_fragment_jpeg(
        mut dicom: FileDicomObject<InMemDicomObject>,
        offset_table: Vec<u32>,
    ) -> FileDicomObject<InMemDicomObject> {
        let pixel_data = dicom.get(tags::PIXEL_DATA).unwrap();
        let fragments = pixel_data.value().fragments().unwrap();

        // Use the first two source JPEG frame fragments and split each into two fragments
        // so the decoder is forced to use BOT-derived frame boundaries.
        let (f0a, f0b) = split_fragment_even(&fragments[0]);
        let (f1a, f1b) = split_fragment_even(&fragments[1]);
        let rewritten_fragments = vec![f0a, f0b, f1a, f1b];

        dicom.put_element(DataElement::new(
            tags::NUMBER_OF_FRAMES,
            VR::IS,
            PrimitiveValue::from("2"),
        ));
        let updated = dicom.update_value(tags::PIXEL_DATA, move |value| {
            let offset_table_mut = value.offset_table_mut().expect("pixel sequence BOT");
            *offset_table_mut = offset_table.clone().into();

            let fragments_mut = value.fragments_mut().expect("pixel sequence fragments");
            *fragments_mut = rewritten_fragments.clone().into();
        });
        assert!(updated, "expected pixel data to be present");
        dicom
    }

    #[test]
    fn test_keep_volume_handles_malformed_bot_without_frame_mixup() {
        let source = dicom_test_files::path("pydicom/color3d_jpeg_baseline.dcm").unwrap();
        let base_dicom = open_file(&source).unwrap();
        let pixel_data = base_dicom.get(tags::PIXEL_DATA).unwrap();
        let fragments = pixel_data.value().fragments().unwrap();
        let (f0a, f0b) = split_fragment_even(&fragments[0]);
        let valid_second_offset = (f0a.len() + 8 + f0b.len() + 8) as u32;

        // Control object with valid BOT.
        let valid_dicom = rewrite_to_two_frame_split_fragment_jpeg(
            open_file(&source).unwrap(),
            vec![0, valid_second_offset],
        );
        let valid_images = KeepVolume
            .decode_volume_with_options(&valid_dicom, &ConvertOptions::default())
            .unwrap();
        assert_eq!(valid_images.len(), 2);

        // Malformed BOT object where frame 1 offset incorrectly points to 0.
        let malformed_dicom =
            rewrite_to_two_frame_split_fragment_jpeg(open_file(&source).unwrap(), vec![0, 0]);
        let malformed_images = KeepVolume
            .decode_volume_with_options(&malformed_dicom, &ConvertOptions::default())
            .unwrap();
        assert_eq!(malformed_images.len(), 2);

        assert_eq!(malformed_images[0], valid_images[0]);
        assert_eq!(malformed_images[1], valid_images[1]);
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

    #[test]
    fn test_interpolate_target_single_frame_returns_first_input_frame() {
        let width = 4;
        let height = 4;
        let mut img1 = ImageBuffer::new(width, height);
        let mut img2 = ImageBuffer::new(width, height);
        for x in 0..width {
            for y in 0..height {
                img1.put_pixel(x, y, Rgb([50, 50, 50]));
                img2.put_pixel(x, y, Rgb([200, 200, 200]));
            }
        }

        let frames = vec![DynamicImage::ImageRgb8(img1), DynamicImage::ImageRgb8(img2)];
        let interpolated = InterpolateVolume::interpolate_frames(&frames, 1);

        assert_eq!(interpolated.len(), 1);
        assert_eq!(
            interpolated[0].as_rgb8().unwrap(),
            frames[0].as_rgb8().unwrap()
        );
    }

    #[test]
    fn test_interpolate_parallel_matches_serial() {
        let width = 64;
        let height = 64;
        let target_frames = 6;
        let source_frames = 5;

        let frames: Vec<DynamicImage> = (0..source_frames)
            .map(|frame_idx| {
                let buffer = ImageBuffer::from_fn(width, height, |x, y| {
                    let value = ((x + y + frame_idx) % 255) as u8;
                    Rgb([value, value, value])
                });
                DynamicImage::ImageRgb8(buffer)
            })
            .collect();

        let serial = InterpolateVolume::interpolate_frames_serial(&frames, target_frames);
        let parallel = InterpolateVolume::interpolate_frames_parallel(&frames, target_frames);
        assert_eq!(serial, parallel);
    }

    // --- Parallel beam projection tests ---

    #[test]
    fn test_parallel_projection_uniform() {
        let w = 4;
        let h = 4;
        let slice = vec![0.5f32; w * h];
        let frames = vec![slice.clone(), slice.clone(), slice];
        let proj = LaplacianMip::compute_parallel_projection(&frames, w, h);
        for &v in &proj {
            assert!((v - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_parallel_projection_sum() {
        let w = 4;
        let h = 4;
        let frame_a = vec![0.25f32; w * h];
        let frame_b = vec![0.75f32; w * h];
        let frames = vec![frame_a, frame_b];
        let proj = LaplacianMip::compute_parallel_projection(&frames, w, h);
        for &v in &proj {
            assert!((v - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_parallel_projection_single_slice() {
        let w = 4;
        let h = 4;
        let mut slice = vec![0.0f32; w * h];
        slice[0] = 1.0;
        slice[5] = 0.5;
        let frames = vec![slice];
        let proj = LaplacianMip::compute_parallel_projection(&frames, w, h);
        assert!((proj[0] - 1.0).abs() < 1e-6);
        assert!((proj[5] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_parallel_projection_all_zeros() {
        let w = 4;
        let h = 4;
        let frames = vec![vec![0.0f32; w * h]; 3];
        let proj = LaplacianMip::compute_parallel_projection(&frames, w, h);
        for &v in &proj {
            assert_eq!(v, 0.0);
        }
    }

    #[rstest]
    #[case(3)]
    #[case(10)]
    #[case(1)]
    fn test_parallel_projection_normalization(#[case] num_slices: usize) {
        let w = 8;
        let h = 8;
        let frames: Vec<Vec<f32>> = (0..num_slices)
            .map(|k| vec![(k as f32 + 1.0) / num_slices as f32; w * h])
            .collect();

        let proj = LaplacianMip::compute_parallel_projection(&frames, w, h);
        let max_val = proj.iter().cloned().fold(0.0f32, f32::max);
        let min_val = proj.iter().cloned().fold(f32::MAX, f32::min);
        assert!(min_val >= 0.0);
        assert!(max_val <= 1.0 + 1e-6);
        assert!(
            (max_val - 1.0).abs() < 1e-6,
            "Max should be normalized to 1.0"
        );
    }

    #[test]
    fn test_compute_mip_matches_serial_reference() {
        let w = 80usize;
        let h = 80usize;
        let frames: Vec<Vec<f32>> = (0..6usize)
            .map(|frame_idx| {
                (0..(w * h))
                    .map(|pixel_idx| ((pixel_idx + frame_idx) % 257) as f32 / 256.0)
                    .collect()
            })
            .collect();

        let expected = LaplacianMip::compute_mip_serial(&frames, w * h);
        let actual = LaplacianMip::compute_mip(&frames, w, h);
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_parallel_projection_matches_serial_reference() {
        let w = 80usize;
        let h = 80usize;
        let frames: Vec<Vec<f32>> = (0..6usize)
            .map(|frame_idx| {
                (0..(w * h))
                    .map(|pixel_idx| ((pixel_idx + frame_idx) % 103) as f32 / 102.0)
                    .collect()
            })
            .collect();

        let expected = LaplacianMip::compute_parallel_projection_serial_normalized(&frames, w * h);
        let actual = LaplacianMip::compute_parallel_projection(&frames, w, h);
        assert_eq!(expected, actual);
    }

    const BILATERAL_TEST_WIDTH: usize = 64;
    const BILATERAL_TEST_HEIGHT: usize = 64;
    const BILATERAL_SIGMA_S: f32 = 3.0;
    const BILATERAL_SIGMA_R: f32 = 0.015;
    const MAX_NORMALIZED_BILATERAL_MAE: f32 = 0.01;

    fn normalized_mae(expected: &[f32], actual: &[f32]) -> f32 {
        expected
            .iter()
            .zip(actual.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .sum::<f32>()
            / expected.len() as f32
    }

    fn assert_bilateral_matches_reference(input: Vec<f32>, width: usize, height: usize) {
        let expected = LaplacianMip::bilateral_filter_reference(
            &input,
            width,
            height,
            BILATERAL_SIGMA_S,
            BILATERAL_SIGMA_R,
        );
        let actual = LaplacianMip::bilateral_filter(
            &input,
            width,
            height,
            BILATERAL_SIGMA_S,
            BILATERAL_SIGMA_R,
        );
        assert_eq!(expected.len(), actual.len());
        assert!(actual.iter().all(|value| value.is_finite()));
        let mae = normalized_mae(&expected, &actual);
        assert!(
            mae <= MAX_NORMALIZED_BILATERAL_MAE,
            "Expected normalized MAE <= {}, got {}",
            MAX_NORMALIZED_BILATERAL_MAE,
            mae
        );
    }

    #[test]
    fn test_bilateral_filter_matches_reference_flat_image() {
        const FLAT_VALUE: f32 = 0.4;
        let input = vec![FLAT_VALUE; BILATERAL_TEST_WIDTH * BILATERAL_TEST_HEIGHT];
        assert_bilateral_matches_reference(input, BILATERAL_TEST_WIDTH, BILATERAL_TEST_HEIGHT);
    }

    #[test]
    fn test_bilateral_filter_matches_reference_gradient_image() {
        let denominator = (BILATERAL_TEST_WIDTH * BILATERAL_TEST_HEIGHT - 1) as f32;
        let input: Vec<f32> = (0..(BILATERAL_TEST_WIDTH * BILATERAL_TEST_HEIGHT))
            .map(|idx| idx as f32 / denominator)
            .collect();
        assert_bilateral_matches_reference(input, BILATERAL_TEST_WIDTH, BILATERAL_TEST_HEIGHT);
    }

    #[test]
    fn test_bilateral_filter_matches_reference_high_contrast_edge() {
        const LEFT_HALF_VALUE: f32 = 0.2;
        const RIGHT_HALF_VALUE: f32 = 0.9;
        let input: Vec<f32> = (0..BILATERAL_TEST_HEIGHT)
            .flat_map(|_| {
                (0..BILATERAL_TEST_WIDTH).map(|x| {
                    if x < BILATERAL_TEST_WIDTH / 2 {
                        LEFT_HALF_VALUE
                    } else {
                        RIGHT_HALF_VALUE
                    }
                })
            })
            .collect();
        assert_bilateral_matches_reference(input, BILATERAL_TEST_WIDTH, BILATERAL_TEST_HEIGHT);
    }

    // --- LaplacianMip::project_laplacian_mip standalone tests ---

    #[test]
    fn test_project_laplacian_mip_synthetic() {
        let width = 16u32;
        let height = 16u32;
        let mip = LaplacianMip::default();

        // Build 4 synthetic Luma16 frames with increasing intensity
        let frames: Vec<DynamicImage> = (0..4)
            .map(|i| {
                let val = (i + 1) * 16000;
                let buf = ImageBuffer::from_fn(width, height, |_, _| image::Luma([val]));
                DynamicImage::ImageLuma16(buf)
            })
            .collect();

        let result = mip.project_laplacian_mip(&frames).unwrap();
        let (rw, rh) = result.dimensions();
        assert_eq!(rw, width);
        assert_eq!(rh, height);
    }

    #[test]
    fn test_project_laplacian_mip_two_frames() {
        let width = 16u32;
        let height = 16u32;
        let mip = LaplacianMip::default();

        let frames: Vec<DynamicImage> = (0..2)
            .map(|i| {
                let val = (i + 1) * 30000;
                let buf = ImageBuffer::from_fn(width, height, |_, _| image::Luma([val]));
                DynamicImage::ImageLuma16(buf)
            })
            .collect();

        let result = mip.project_laplacian_mip(&frames).unwrap();
        let (rw, rh) = result.dimensions();
        assert_eq!(rw, width);
        assert_eq!(rh, height);
    }

    #[test]
    fn test_project_laplacian_mip_single_frame() {
        let width = 16u32;
        let height = 16u32;
        let mip = LaplacianMip::default();

        let buf = ImageBuffer::from_fn(width, height, |_, _| image::Luma([42000u16]));
        let frames = vec![DynamicImage::ImageLuma16(buf)];

        let result = mip.project_laplacian_mip(&frames).unwrap();
        let (rw, rh) = result.dimensions();
        assert_eq!(rw, width);
        assert_eq!(rh, height);
    }

    #[test]
    fn test_project_laplacian_mip_empty_input_returns_error() {
        let mip = LaplacianMip::default();
        let frames: Vec<DynamicImage> = vec![];
        let err = mip.project_laplacian_mip(&frames).unwrap_err();
        assert_eq!(
            err.to_string(),
            "cannot project laplacian mip from empty input frames"
        );

        match err {
            DicomError::LaplacianMipEmptyInput => {}
            other => panic!("Expected LaplacianMipEmptyInput, got {other}"),
        }
    }

    #[test]
    fn test_laplacian_mip_insufficient_frames_error_serial() {
        let dicom = dicom_test_files::path("pydicom/emri_small.dcm").unwrap();
        let dicom = open_file(&dicom).unwrap();
        let number_of_frames: u32 = FrameCount::try_from(&dicom).unwrap().into();
        let skip_start = number_of_frames;
        let skip_end = 0u32;

        let handler = LaplacianMip::new(skip_start, skip_end);
        let err = handler
            .decode_volume_with_options(&dicom, &ConvertOptions::default())
            .unwrap_err();
        let expected_message = format!(
            "number of frames {} is insufficient for laplacian mip with skip {}-{}",
            number_of_frames, skip_start, skip_end
        );
        assert_eq!(err.to_string(), expected_message);

        match err {
            DicomError::LaplacianMipInsufficientFrames {
                number_of_frames: actual_number_of_frames,
                skip_start: actual_skip_start,
                skip_end: actual_skip_end,
            } => {
                assert_eq!(actual_number_of_frames, number_of_frames as usize);
                assert_eq!(actual_skip_start, skip_start as usize);
                assert_eq!(actual_skip_end, skip_end as usize);
            }
            other => panic!("Expected LaplacianMipInsufficientFrames, got {other}"),
        }
    }

    #[test]
    fn test_laplacian_mip_insufficient_frames_error_parallel() {
        let dicom = dicom_test_files::path("pydicom/emri_small.dcm").unwrap();
        let dicom = open_file(&dicom).unwrap();
        let number_of_frames: u32 = FrameCount::try_from(&dicom).unwrap().into();
        let skip_start = number_of_frames;
        let skip_end = 0u32;

        let handler = LaplacianMip::new(skip_start, skip_end);
        let err = handler
            .par_decode_volume_with_options(&dicom, &ConvertOptions::default())
            .unwrap_err();
        let expected_message = format!(
            "number of frames {} is insufficient for laplacian mip with skip {}-{}",
            number_of_frames, skip_start, skip_end
        );
        assert_eq!(err.to_string(), expected_message);

        match err {
            DicomError::LaplacianMipInsufficientFrames {
                number_of_frames: actual_number_of_frames,
                skip_start: actual_skip_start,
                skip_end: actual_skip_end,
            } => {
                assert_eq!(actual_number_of_frames, number_of_frames as usize);
                assert_eq!(actual_skip_start, skip_start as usize);
                assert_eq!(actual_skip_end, skip_end as usize);
            }
            other => panic!("Expected LaplacianMipInsufficientFrames, got {other}"),
        }
    }
}
