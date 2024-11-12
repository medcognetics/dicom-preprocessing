use std::io::{Seek, Write};
use tiff::encoder::colortype::ColorType;
use tiff::encoder::compression::Compression;
use tiff::encoder::{ImageEncoder, TiffKind};
use tiff::tags::Tag;
use tiff::TiffError;

use crate::metadata::{Resolution, WriteTags};
use crate::transform::{Crop, Padding, Resize};

const VERSION: &str = concat!("dicom-preprocessing==", env!("CARGO_PKG_VERSION"), "\0");

/// Tracks all of the preprocessing metadata and augmentations
pub struct PreprocessingMetadata {
    pub crop: Option<Crop>,
    pub resize: Option<Resize>,
    pub padding: Option<Padding>,
    pub resolution: Option<Resolution>,
}

impl WriteTags for PreprocessingMetadata {
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

