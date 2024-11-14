use dicom::dictionary_std::tags;
use dicom::object::{FileDicomObject, InMemDicomObject};
use dicom::pixeldata::PixelDecoder;
use image::DynamicImage;
use image::Pixel;
use image::{GenericImage, GenericImageView};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use snafu::{ResultExt, Snafu};
use std::cmp::{max, min};
use std::fmt;

#[derive(Debug, Snafu)]
pub enum VolumeError {
    MissingProperty {
        name: &'static str,
    },
    InvalidPropertyValue {
        name: &'static str,
        #[snafu(source(from(dicom::core::value::ConvertValueError, Box::new)))]
        source: Box<dicom::core::value::ConvertValueError>,
    },
    InvalidNumberOfFrames {
        value: i32,
    },
    CastPropertyValue {
        name: &'static str,
        #[snafu(source(from(dicom::core::value::CastValueError, Box::new)))]
        source: Box<dicom::core::value::CastValueError>,
    },
    DecodePixelData {
        #[snafu(source(from(dicom::pixeldata::Error, Box::new)))]
        source: Box<dicom::pixeldata::Error>,
    },
    #[snafu(display(
        "Invalid volume range: start={}, end={}, number_of_frames={}",
        start,
        end,
        number_of_frames
    ))]
    InvalidVolumeRange {
        start: u32,
        end: u32,
        number_of_frames: u32,
    },
}

pub enum VolumeHandler {
    Keep(KeepVolume),
    CentralSlice(CentralSlice),
    MaxIntensity(MaxIntensity),
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
}

impl From<DisplayVolumeHandler> for VolumeHandler {
    fn from(handler: DisplayVolumeHandler) -> Self {
        match handler {
            DisplayVolumeHandler::Keep => VolumeHandler::Keep(KeepVolume),
            DisplayVolumeHandler::CentralSlice => VolumeHandler::CentralSlice(CentralSlice),
            DisplayVolumeHandler::MaxIntensity => VolumeHandler::MaxIntensity(MaxIntensity {
                skip_start: 0,
                skip_end: 0,
            }),
        }
    }
}

impl fmt::Display for DisplayVolumeHandler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let filter_str = match self {
            DisplayVolumeHandler::Keep => "keep",
            DisplayVolumeHandler::CentralSlice => "central-slice",
            DisplayVolumeHandler::MaxIntensity => "max-intensity",
        };
        write!(f, "{}", filter_str)
    }
}

pub fn get_number_of_frames(file: &FileDicomObject<InMemDicomObject>) -> Result<u32, VolumeError> {
    let number_of_frames = file.get(tags::NUMBER_OF_FRAMES);

    let number_of_frames = match number_of_frames {
        Some(elem) => elem.to_int::<i32>().context(InvalidPropertyValueSnafu {
            name: "Number of Frames",
        })?,
        _ => 1,
    };

    match number_of_frames >= 1 {
        true => Ok(number_of_frames as u32),
        false => Err(VolumeError::InvalidNumberOfFrames {
            value: number_of_frames,
        }),
    }
}

pub trait HandleVolume {
    /// Decode and handle the volume frame by frame
    fn decode_volume(
        &self,
        file: &FileDicomObject<InMemDicomObject>,
    ) -> Result<Vec<DynamicImage>, VolumeError>;

    /// Decode each frame in parallel and handle the volume
    fn par_decode_volume(
        &self,
        file: &FileDicomObject<InMemDicomObject>,
    ) -> Result<Vec<DynamicImage>, VolumeError>;
}

impl HandleVolume for VolumeHandler {
    fn decode_volume(
        &self,
        file: &FileDicomObject<InMemDicomObject>,
    ) -> Result<Vec<DynamicImage>, VolumeError> {
        match self {
            VolumeHandler::Keep(handler) => handler.decode_volume(file),
            VolumeHandler::CentralSlice(handler) => handler.decode_volume(file),
            VolumeHandler::MaxIntensity(handler) => handler.decode_volume(file),
        }
    }

    fn par_decode_volume(
        &self,
        file: &FileDicomObject<InMemDicomObject>,
    ) -> Result<Vec<DynamicImage>, VolumeError> {
        match self {
            VolumeHandler::Keep(handler) => handler.par_decode_volume(file),
            VolumeHandler::CentralSlice(handler) => handler.par_decode_volume(file),
            VolumeHandler::MaxIntensity(handler) => handler.par_decode_volume(file),
        }
    }
}

pub struct KeepVolume;

impl HandleVolume for KeepVolume {
    fn decode_volume(
        &self,
        file: &FileDicomObject<InMemDicomObject>,
    ) -> Result<Vec<DynamicImage>, VolumeError> {
        let number_of_frames = get_number_of_frames(file)?;
        let mut image_data = Vec::with_capacity(number_of_frames as usize);
        for frame_number in 0..number_of_frames {
            let decoded = file
                .decode_pixel_data_frame(frame_number)
                .context(DecodePixelDataSnafu)?;
            image_data.push(decoded.to_dynamic_image(0).context(DecodePixelDataSnafu)?);
        }
        Ok(image_data)
    }

    fn par_decode_volume(
        &self,
        file: &FileDicomObject<InMemDicomObject>,
    ) -> Result<Vec<DynamicImage>, VolumeError> {
        let number_of_frames = get_number_of_frames(file)?;
        let result = (0..number_of_frames)
            .into_par_iter()
            .map(|frame| {
                let result = file
                    .decode_pixel_data_frame(frame)
                    .context(DecodePixelDataSnafu)?
                    .to_dynamic_image(0)
                    .context(DecodePixelDataSnafu)?;
                Ok(result)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(result)
    }
}

pub struct CentralSlice;

impl HandleVolume for CentralSlice {
    fn decode_volume(
        &self,
        file: &FileDicomObject<InMemDicomObject>,
    ) -> Result<Vec<DynamicImage>, VolumeError> {
        let number_of_frames = get_number_of_frames(file)?;
        let central_frame = number_of_frames / 2;
        let decoded = file
            .decode_pixel_data_frame(central_frame)
            .context(DecodePixelDataSnafu)?;
        let image = decoded.to_dynamic_image(0).context(DecodePixelDataSnafu)?;
        Ok(vec![image])
    }

    fn par_decode_volume(
        &self,
        file: &FileDicomObject<InMemDicomObject>,
    ) -> Result<Vec<DynamicImage>, VolumeError> {
        // Since there is only one frame, we can just decode it serially
        self.decode_volume(file)
    }
}

pub struct MaxIntensity {
    skip_start: u32,
    skip_end: u32,
}

impl MaxIntensity {
    fn reduce(current: DynamicImage, new: DynamicImage) -> DynamicImage {
        let mut current = current;
        let (width, height) = current.dimensions();
        for x in 0..width {
            for y in 0..height {
                let mut current_pixel = current.get_pixel(x, y);
                let new_pixel = new.get_pixel(x, y);
                current_pixel.apply2(&new_pixel, |p1, p2| max(p1, p2));
                current.put_pixel(x, y, current_pixel);
            }
        }
        current
    }
}

impl HandleVolume for MaxIntensity {
    fn decode_volume(
        &self,
        file: &FileDicomObject<InMemDicomObject>,
    ) -> Result<Vec<DynamicImage>, VolumeError> {
        let number_of_frames = get_number_of_frames(file)?;
        let start = min(number_of_frames, self.skip_start);
        let end = max(0, number_of_frames - self.skip_end);

        // Validate the start/end relative to the number of frames
        if start >= end || start >= number_of_frames || end <= 0 {
            return Err(VolumeError::InvalidVolumeRange {
                start,
                end,
                number_of_frames,
            });
        }

        let decoded = file
            .decode_pixel_data_frame(start)
            .context(DecodePixelDataSnafu)?;

        let mut image = decoded.to_dynamic_image(0).context(DecodePixelDataSnafu)?;
        for frame_number in (start + 1)..end {
            let decoded = file
                .decode_pixel_data_frame(frame_number)
                .context(DecodePixelDataSnafu)?;
            let frame = decoded.to_dynamic_image(0).context(DecodePixelDataSnafu)?;
            image = Self::reduce(image, frame);
        }
        Ok(vec![image])
    }

    fn par_decode_volume(
        &self,
        file: &FileDicomObject<InMemDicomObject>,
    ) -> Result<Vec<DynamicImage>, VolumeError> {
        let number_of_frames = get_number_of_frames(file)?;
        let start = min(number_of_frames, self.skip_start);
        let end = max(0, number_of_frames - self.skip_end);

        // Validate the start/end relative to the number of frames
        if start >= end || start >= number_of_frames || end <= 0 {
            return Err(VolumeError::InvalidVolumeRange {
                start,
                end,
                number_of_frames,
            });
        }

        let image = (start..end)
            .into_par_iter()
            .map(|frame_number| {
                let frame = file
                    .decode_pixel_data_frame(frame_number)
                    .context(DecodePixelDataSnafu)?
                    .to_dynamic_image(0)
                    .context(DecodePixelDataSnafu)?;
                Ok::<DynamicImage, VolumeError>(frame)
            })
            .try_reduce_with(|image, frame| Ok(Self::reduce(image, frame)));

        if let Some(image) = image {
            Ok(vec![image?])
        } else {
            Err(VolumeError::InvalidVolumeRange {
                start,
                end,
                number_of_frames,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use dicom::object::open_file;
    use rstest::rstest;

    #[rstest]
    #[case("pydicom/CT_small.dcm", VolumeHandler::Keep(KeepVolume), 1)]
    #[case("pydicom/CT_small.dcm", VolumeHandler::CentralSlice(CentralSlice), 1)]
    #[case("pydicom/CT_small.dcm", VolumeHandler::MaxIntensity(MaxIntensity { skip_start: 0, skip_end: 0 }), 1)]
    fn test_decode_volume(
        #[case] dicom_file_path: &str,
        #[case] volume_handler: VolumeHandler,
        #[case] expected_number_of_frames: u32,
    ) {
        let dicom = dicom_test_files::path(dicom_file_path).unwrap();
        let dicom = open_file(&dicom).unwrap();
        let images = volume_handler.decode_volume(&dicom).unwrap();
        assert_eq!(images.len() as u32, expected_number_of_frames);
    }

    #[rstest]
    #[case("pydicom/CT_small.dcm", VolumeHandler::Keep(KeepVolume), 1)]
    #[case("pydicom/CT_small.dcm", VolumeHandler::CentralSlice(CentralSlice), 1)]
    #[case("pydicom/CT_small.dcm", VolumeHandler::MaxIntensity(MaxIntensity { skip_start: 0, skip_end: 0 }), 1)]
    fn test_par_decode_volume(
        #[case] dicom_file_path: &str,
        #[case] volume_handler: VolumeHandler,
        #[case] expected_number_of_frames: u32,
    ) {
        let dicom = dicom_test_files::path(dicom_file_path).unwrap();
        let dicom = open_file(&dicom).unwrap();
        let images = volume_handler.par_decode_volume(&dicom).unwrap();
        assert_eq!(images.len() as u32, expected_number_of_frames);
    }
}
