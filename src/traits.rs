use clap::ValueEnum;
use image::imageops::FilterType;
use image::{DynamicImage, GenericImage, GenericImageView, Pixel, Rgba};
use std::fmt;
use std::io::{Seek, Write};
use tiff::encoder::colortype::ColorType;
use tiff::encoder::compression::Compression;
use tiff::encoder::ImageEncoder;
use tiff::encoder::TiffKind;
use tiff::tags::Tag;
use tiff::TiffError;

pub trait Transform {
    fn apply(&self, image: &DynamicImage) -> DynamicImage;

    fn apply_iter(
        &self,
        images: impl Iterator<Item = DynamicImage>,
    ) -> impl Iterator<Item = DynamicImage> {
        images.map(|img| self.apply(&img))
    }
}

pub trait WriteTags {
    /// Write tags describing the transform to a TIFF encoder.
    fn write_tags<W, C, K, D>(&self, tiff: &mut ImageEncoder<W, C, K, D>) -> Result<(), TiffError>
    where
        W: Write + Seek,
        C: ColorType,
        K: TiffKind,
        D: Compression;
}
