use dicom::core::header::HasLength;
use dicom::dictionary_std::tags;
use dicom::object::{FileDicomObject, InMemDicomObject};
use snafu::ResultExt;
use std::io::{Read, Seek, Write};
use tiff::decoder::Decoder;
use tiff::encoder::colortype::ColorType;
use tiff::encoder::compression::Compression;
use tiff::encoder::{ImageEncoder, TiffKind};
use tiff::tags::Tag;

use crate::dicom::ConvertValueSnafu;
use crate::errors::{DicomError, TiffError};
use crate::metadata::{Resolution, WriteTags};
use crate::transform::{Coord, Crop, InvertibleTransform, Padding, Resize, Transform};

const VERSION: &str = concat!("dicom-preprocessing==", env!("CARGO_PKG_VERSION"), "\0");

#[derive(Debug, PartialEq)]
pub struct Version(String);

impl Version {
    const TAG: Tag = Tag::Software;
}

impl Default for Version {
    fn default() -> Self {
        Self(VERSION.to_string())
    }
}

impl WriteTags for Version {
    fn write_tags<W, C, K, D>(&self, tiff: &mut ImageEncoder<W, C, K, D>) -> Result<(), TiffError>
    where
        W: Write + Seek,
        C: ColorType,
        K: TiffKind,
        D: Compression,
    {
        tiff.encoder().write_tag(Self::TAG, self.0.as_bytes())?;
        Ok(())
    }
}

impl From<String> for Version {
    fn from(version: String) -> Self {
        Self(version)
    }
}

impl From<Version> for String {
    fn from(version: Version) -> Self {
        version.0
    }
}

impl<T> TryFrom<&mut Decoder<T>> for Version
where
    T: Read + Seek,
{
    type Error = TiffError;

    fn try_from(decoder: &mut Decoder<T>) -> Result<Self, Self::Error> {
        let software = decoder.get_tag(Self::TAG)?.into_string()?;
        Ok(Version(software))
    }
}

#[derive(Debug, PartialEq)]
pub struct FrameCount(u16);

impl FrameCount {
    // PageNumber
    const TAG: Tag = Tag::Unknown(297);
}

impl From<FrameCount> for u16 {
    fn from(frame_count: FrameCount) -> Self {
        frame_count.0
    }
}

impl From<u16> for FrameCount {
    fn from(num_frames: u16) -> Self {
        Self(num_frames)
    }
}

impl From<usize> for FrameCount {
    fn from(num_frames: usize) -> Self {
        Self(num_frames as u16)
    }
}

impl From<FrameCount> for usize {
    fn from(frame_count: FrameCount) -> Self {
        frame_count.0 as usize
    }
}

impl From<u32> for FrameCount {
    fn from(num_frames: u32) -> Self {
        Self(num_frames as u16)
    }
}

impl From<FrameCount> for u32 {
    fn from(frame_count: FrameCount) -> Self {
        frame_count.0 as u32
    }
}

impl<T> TryFrom<&mut Decoder<T>> for FrameCount
where
    T: Read + Seek,
{
    type Error = TiffError;

    fn try_from(decoder: &mut Decoder<T>) -> Result<Self, Self::Error> {
        // First try to parse the image description as a tuple of u16 (page, total).
        // The first value is ignored, and filled with 0 to satisfy the TIFF spec.
        let page_info = decoder
            .get_tag(Self::TAG)
            .map_or(None, |tag| tag.into_u16_vec().ok());
        if let Some(page_info) = page_info {
            if let [_, total] = page_info.as_slice() {
                return Ok(FrameCount(*total));
            }
        }

        // Otherwise, we scan the file for the number of frames
        // This implementation should avoid having to read image data between frames
        let mut num_frames = 0;
        while decoder.seek_to_image(num_frames).is_ok() {
            num_frames += 1;
        }
        Ok(num_frames.into())
    }
}

impl TryFrom<&FileDicomObject<InMemDicomObject>> for FrameCount {
    type Error = DicomError;
    fn try_from(dcm: &FileDicomObject<InMemDicomObject>) -> Result<Self, Self::Error> {
        let number_of_frames = dcm.get(tags::NUMBER_OF_FRAMES);

        let number_of_frames = match number_of_frames {
            Some(elem) if !elem.is_empty() => elem.to_int::<u16>().context(ConvertValueSnafu {
                name: "Number of Frames",
            })?,
            _ => 1,
        };

        Ok(number_of_frames.into())
    }
}

impl IntoIterator for FrameCount {
    type Item = u16;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    /// Returns an iterator over the frame indices
    fn into_iter(self) -> Self::IntoIter {
        (0..self.0).collect::<Vec<_>>().into_iter()
    }
}

impl WriteTags for FrameCount {
    fn write_tags<W, C, K, D>(&self, tiff: &mut ImageEncoder<W, C, K, D>) -> Result<(), TiffError>
    where
        W: Write + Seek,
        C: ColorType,
        K: TiffKind,
        D: Compression,
    {
        let page_info = vec![0, self.0];
        tiff.encoder().write_tag(Self::TAG, page_info.as_slice())?;
        Ok(())
    }
}

/// Tracks all of the preprocessing metadata and augmentations
#[derive(Debug, PartialEq)]
pub struct PreprocessingMetadata {
    pub crop: Option<Crop>,
    pub resize: Option<Resize>,
    pub padding: Option<Padding>,
    pub resolution: Option<Resolution>,
    pub num_frames: FrameCount,
}

impl Transform<Coord> for PreprocessingMetadata {
    fn apply(&self, coord: &Coord) -> Coord {
        // Forward application order is crop, resize, padding
        let coord = self
            .crop
            .as_ref()
            .map(|crop| crop.apply(coord))
            .unwrap_or(*coord);
        let coord = self
            .resize
            .as_ref()
            .map(|resize| resize.apply(&coord))
            .unwrap_or(coord);
        let coord = self
            .padding
            .as_ref()
            .map(|padding| padding.apply(&coord))
            .unwrap_or(coord);
        coord
    }
}

impl InvertibleTransform<Coord> for PreprocessingMetadata {
    fn invert(&self, coord: &Coord) -> Coord {
        // Invert application order is padding, resize, crop
        let coord = self
            .padding
            .as_ref()
            .map(|padding| padding.invert(coord))
            .unwrap_or(*coord);
        let coord = self
            .resize
            .as_ref()
            .map(|resize| resize.invert(&coord))
            .unwrap_or(coord);
        let coord = self
            .crop
            .as_ref()
            .map(|crop| crop.invert(&coord))
            .unwrap_or(coord);
        coord
    }
}

impl WriteTags for PreprocessingMetadata {
    /// Writes TIFF tags for the respective transforms, along with version and resolution metadata
    fn write_tags<W, C, K, D>(&self, tiff: &mut ImageEncoder<W, C, K, D>) -> Result<(), TiffError>
    where
        W: Write + Seek,
        C: ColorType,
        K: TiffKind,
        D: Compression,
    {
        // Write the resolution tag
        if let Some(resolution) = &self.resolution {
            resolution.write_tags(tiff)?;
        }
        // Write metadata software tag
        Version::default().write_tags(tiff)?;
        // Write the frame count into ImageDescription
        self.num_frames.write_tags(tiff)?;

        // Write transform related tags
        if let Some(crop_config) = &self.crop {
            crop_config.write_tags(tiff)?;
        }
        if let Some(resize_config) = &self.resize {
            resize_config.write_tags(tiff)?;
        }
        if let Some(padding_config) = &self.padding {
            padding_config.write_tags(tiff)?;
        }
        Ok(())
    }
}

impl<T> TryFrom<&mut Decoder<T>> for PreprocessingMetadata
where
    T: Read + Seek,
{
    type Error = TiffError;

    /// Read the preprocessing metadata from the TIFF file
    fn try_from(decoder: &mut Decoder<T>) -> Result<Self, Self::Error> {
        // NOTE: We don't distinguish between an unexpected error and an expected missing/malformed tag.
        // The TIFF could be using these tags for another purpose, e.g. if it was created by a different software.
        let crop = Crop::try_from(&mut *decoder).ok();
        let resize = Resize::try_from(&mut *decoder).ok();
        let padding = Padding::try_from(&mut *decoder).ok();
        let resolution = Resolution::try_from(&mut *decoder).ok();

        // This has a fallback to scanning the file, so it should never fail
        let num_frames = FrameCount::try_from(&mut *decoder)?;

        Ok(Self {
            crop,
            resize,
            padding,
            resolution,
            num_frames,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::resize::FilterType;
    use crate::transform::Coord;
    use rstest::rstest;
    use std::fs::File;
    use tempfile::tempdir;
    use tiff::encoder::TiffEncoder;

    #[rstest]
    #[case(
        PreprocessingMetadata {
            crop: Some(Crop { left: 0, top: 0, width: 1, height: 1 }),
            resize: Some(Resize { scale_x: 2.0, scale_y: 2.0, filter: FilterType::Nearest }),
            padding: Some(Padding { left: 0, top: 0, right: 1, bottom: 1 }),
            resolution: Some(Resolution { pixels_per_mm_x: 1.0, pixels_per_mm_y: 1.0 }),
            num_frames: FrameCount(1),
        }
    )]
    fn test_write_tags(#[case] metadata: PreprocessingMetadata) {
        // Prepare the TIFF
        let temp_dir = tempdir().unwrap();
        let temp_file_path = temp_dir.path().join("temp.tif");
        let mut tiff = TiffEncoder::new(File::create(temp_file_path.clone()).unwrap()).unwrap();
        let mut img = tiff
            .new_image::<tiff::encoder::colortype::Gray16>(1, 1)
            .unwrap();

        // Write the tags
        metadata.write_tags(&mut img).unwrap();

        // Write some dummy image data
        let data: Vec<u16> = vec![0; 2];
        img.write_data(data.as_slice()).unwrap();

        // Read the TIFF back
        let mut tiff = Decoder::new(File::open(temp_file_path).unwrap()).unwrap();
        let actual = PreprocessingMetadata::try_from(&mut tiff).unwrap();
        assert_eq!(metadata, actual);
    }

    #[test]
    fn test_frame_count_fallback() {
        // Create a multi-frame TIFF without explicit frame count tag
        let temp_dir = tempdir().unwrap();
        let temp_file_path = temp_dir.path().join("temp.tif");
        let mut tiff = TiffEncoder::new(File::create(temp_file_path.clone()).unwrap()).unwrap();

        // Write 3 frames
        for _ in 0..3 {
            let img = tiff
                .new_image::<tiff::encoder::colortype::Gray16>(1, 1)
                .unwrap();
            let data: Vec<u16> = vec![0; 1];
            img.write_data(data.as_slice()).unwrap();
        }

        // Read back and verify frame count
        let mut tiff = Decoder::new(File::open(temp_file_path).unwrap()).unwrap();
        let frame_count = FrameCount::try_from(&mut tiff).unwrap();
        assert_eq!(frame_count, FrameCount(3));
    }

    #[rstest]
    #[case(Coord::new(0, 0), Coord::new(0, 0))] // Unchanged
    #[case(Coord::new(1, 1), Coord::new(2, 2))] // Scaled 2x
    #[case(Coord::new(7, 7), Coord::new(6, 6))] // Out of bounds
    fn test_apply_coord_upper_left(#[case] input: Coord, #[case] expected: Coord) {
        // Assume image is 8x8 pixels
        // We crop to the top left quadrant and then scale that by 2x
        // Then we pad the right and bottom by 1 pixel each
        let metadata = PreprocessingMetadata {
            crop: Some(Crop {
                left: 0,
                top: 0,
                width: 4,
                height: 4,
            }),
            resize: Some(Resize {
                scale_x: 2.0,
                scale_y: 2.0,
                filter: FilterType::Nearest,
            }),
            padding: Some(Padding {
                left: 0,
                top: 0,
                right: 1,
                bottom: 1,
            }),
            resolution: Some(Resolution {
                pixels_per_mm_x: 1.0,
                pixels_per_mm_y: 1.0,
            }),
            num_frames: FrameCount(1),
        };

        let result = metadata.apply(&input);
        assert_eq!(result, expected);
    }

    #[rstest]
    #[case(Coord::new(0, 0), Coord::new(1, 1))] // Clipped to upper left corner, plus 1 padding px
    #[case(Coord::new(4, 4), Coord::new(1, 1))] // At upper left corner of crop, plus 1 padding px
    #[case(Coord::new(7, 7), Coord::new(7, 7))] // (3, 3), scaled 2x, plus 1 padding px
    fn test_apply_coord_lower_right(#[case] input: Coord, #[case] expected: Coord) {
        // Assume image is 8x8 pixels
        // We crop to the bottom right quadrant and then scale that by 2x
        // Then we pad the top and left by 1 pixel each
        let metadata = PreprocessingMetadata {
            crop: Some(Crop {
                left: 4,
                top: 4,
                width: 4,
                height: 4,
            }),
            resize: Some(Resize {
                scale_x: 2.0,
                scale_y: 2.0,
                filter: FilterType::Nearest,
            }),
            padding: Some(Padding {
                left: 1,
                top: 1,
                right: 0,
                bottom: 0,
            }),
            resolution: Some(Resolution {
                pixels_per_mm_x: 1.0,
                pixels_per_mm_y: 1.0,
            }),
            num_frames: FrameCount(1),
        };

        let result = metadata.apply(&input);
        assert_eq!(result, expected);
    }
}
