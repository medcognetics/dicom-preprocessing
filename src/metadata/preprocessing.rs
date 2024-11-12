use std::io::{Read, Seek, Write};
use tiff::decoder::Decoder;
use tiff::encoder::colortype::ColorType;
use tiff::encoder::compression::Compression;
use tiff::encoder::{ImageEncoder, TiffKind};
use tiff::tags::Tag;
use tiff::TiffError;

use crate::metadata::{Resolution, WriteTags};
use crate::transform::{Crop, Padding, Resize};

const VERSION: &str = concat!("dicom-preprocessing==", env!("CARGO_PKG_VERSION"), "\0");

/// Tracks all of the preprocessing metadata and augmentations
#[derive(Debug, PartialEq)]
pub struct PreprocessingMetadata {
    pub crop: Option<Crop>,
    pub resize: Option<Resize>,
    pub padding: Option<Padding>,
    pub resolution: Option<Resolution>,
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
        tiff.encoder()
            .write_tag(Tag::Software, VERSION.as_bytes())?;

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

impl<T> From<&mut Decoder<T>> for PreprocessingMetadata
where
    T: Read + Seek,
{
    /// Read the preprocessing metadata from the TIFF file
    fn from(decoder: &mut Decoder<T>) -> Self {
        // NOTE: We don't distinguish between an unexpected error and an expected missing/malformed tag.
        // The TIFF could be using these tags for another purpose, e.g. if it was created by a different software.
        let crop = Crop::try_from(&mut *decoder).ok();
        let resize = Resize::try_from(&mut *decoder).ok();
        let padding = Padding::try_from(&mut *decoder).ok();
        let resolution = Resolution::try_from(&mut *decoder).ok();
        Self {
            crop,
            resize,
            padding,
            resolution,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::imageops::FilterType;

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
        let actual = PreprocessingMetadata::from(&mut tiff);
        assert_eq!(metadata, actual);
    }
}
