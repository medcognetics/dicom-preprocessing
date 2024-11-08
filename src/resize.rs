use crate::traits::{Transform, WriteTags};
use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView};
use std::fmt;
use std::io::{Seek, Write};
use tiff::encoder::colortype::ColorType;
use tiff::encoder::compression::Compression;
use tiff::encoder::ImageEncoder;
use tiff::encoder::TiffKind;
use tiff::tags::Tag;
use tiff::TiffError;

const DEFAULT_SCALE: u16 = 50718;

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum DisplayFilterType {
    #[default]
    Triangle,
    Nearest,
    CatmullRom,
    Gaussian,
    Lanczos3,
}

impl fmt::Display for DisplayFilterType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let filter_str = match self {
            DisplayFilterType::Triangle => "triangle",
            DisplayFilterType::Nearest => "nearest",
            DisplayFilterType::CatmullRom => "catmull-rom",
            DisplayFilterType::Gaussian => "gaussian",
            DisplayFilterType::Lanczos3 => "lanczos3",
        };
        write!(f, "{}", filter_str)
    }
}

impl From<FilterType> for DisplayFilterType {
    fn from(filter: FilterType) -> Self {
        match filter {
            FilterType::Nearest => DisplayFilterType::Nearest,
            FilterType::Triangle => DisplayFilterType::Triangle,
            FilterType::CatmullRom => DisplayFilterType::CatmullRom,
            FilterType::Gaussian => DisplayFilterType::Gaussian,
            FilterType::Lanczos3 => DisplayFilterType::Lanczos3,
        }
    }
}

impl Into<FilterType> for DisplayFilterType {
    fn into(self) -> FilterType {
        match self {
            DisplayFilterType::Nearest => FilterType::Nearest,
            DisplayFilterType::Triangle => FilterType::Triangle,
            DisplayFilterType::CatmullRom => FilterType::CatmullRom,
            DisplayFilterType::Gaussian => FilterType::Gaussian,
            DisplayFilterType::Lanczos3 => FilterType::Lanczos3,
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct Resize {
    pub scale_x: f32,
    pub scale_y: f32,
    pub filter: FilterType,
}

impl Resize {
    pub fn new(
        image: &DynamicImage,
        target_width: u32,
        target_height: u32,
        filter: FilterType,
    ) -> Self {
        // Determine scale factors
        let (width, height) = image.dimensions();
        let scale_x = target_width as f32 / width as f32;
        let scale_y = target_height as f32 / height as f32;

        // Preserve aspect ratio by choosing the smaller scale factor
        let scale = scale_x.min(scale_y);
        let scale_x = scale;
        let scale_y = scale;

        Resize {
            scale_x,
            scale_y,
            filter,
        }
    }
}

impl Transform for Resize {
    fn apply(&self, image: &DynamicImage) -> DynamicImage {
        let (width, height) = image.dimensions();
        let target_width = (width as f32 * self.scale_x) as u32;
        let target_height = (height as f32 * self.scale_y) as u32;
        image.resize(target_width, target_height, self.filter)
    }
}

impl WriteTags for Resize {
    fn write_tags<W, C, K, D>(&self, tiff: &mut ImageEncoder<W, C, K, D>) -> Result<(), TiffError>
    where
        W: Write + Seek,
        C: ColorType,
        K: TiffKind,
        D: Compression,
    {
        let tag = Tag::Unknown(DEFAULT_SCALE);
        let scale = vec![self.scale_x, self.scale_y];
        tiff.encoder().write_tag(tag, scale.as_slice())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, RgbaImage};
    use rstest::rstest;

    #[rstest]
    #[case(
        vec![
            vec![1, 1, 2, 2],
            vec![1, 1, 2, 2],
            vec![3, 3, 4, 4],
            vec![3, 3, 4, 4],
        ],
        (2, 2),
        vec![
            vec![1, 2],
            vec![3, 4],
        ],
    )]
    fn test_resize(
        #[case] pixels: Vec<Vec<u8>>,
        #[case] target_size: (u32, u32),
        #[case] expected_pixels: Vec<Vec<u8>>,
    ) {
        // Create a new image from the pixel data
        let width = pixels[0].len() as u32;
        let height = pixels.len() as u32;
        let mut img = RgbaImage::new(width, height);
        for (y, row) in pixels.iter().enumerate() {
            for (x, &value) in row.iter().enumerate() {
                img.put_pixel(x as u32, y as u32, image::Rgba([value, value, value, 255]));
            }
        }
        let dynamic_image = DynamicImage::ImageRgba8(img);

        // Create an expected image
        let mut expected_img = RgbaImage::new(target_size.0, target_size.1);
        for (y, row) in expected_pixels.iter().enumerate() {
            for (x, &value) in row.iter().enumerate() {
                expected_img.put_pixel(x as u32, y as u32, image::Rgba([value, value, value, 255]));
            }
        }
        let expected_dynamic_image = DynamicImage::ImageRgba8(expected_img);

        let resize = Resize::new(
            &dynamic_image,
            target_size.0,
            target_size.1,
            FilterType::Nearest,
        );
        assert_eq!(resize.apply(&dynamic_image), expected_dynamic_image);
    }
}
