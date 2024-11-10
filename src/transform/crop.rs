use image::{DynamicImage, GenericImageView, Pixel};
use std::io::{Seek, Write};
use tiff::encoder::colortype::ColorType;
use tiff::encoder::compression::Compression;
use tiff::encoder::ImageEncoder;
use tiff::encoder::TiffKind;
use tiff::tags::Tag;
use tiff::TiffError;

use crate::metadata::WriteTags;
use crate::transform::Transform;

pub const DEFAULT_CROP_ORIGIN: u16 = 50719;
pub const DEFAULT_CROP_SIZE: u16 = 50720;
const DEFAULT_CHECK_MAX: bool = false;

#[derive(Debug, PartialEq, Eq)]
pub struct Crop {
    pub left: u32,
    pub top: u32,
    pub width: u32,
    pub height: u32,
}

impl Crop {
    pub fn new(image: &DynamicImage, check_max: bool) -> Self {
        let (width, height) = image.dimensions();
        let mut left = 0;
        let mut top = 0;
        let mut right = width - 1;
        let mut bottom = height - 1;

        // Find the left boundary
        for x in 0..width {
            if (0..height).any(|y| is_uncroppable_pixel(x, y, image, check_max)) {
                left = x;
                break;
            }
        }

        // Find the right boundary
        for x in (0..width).rev() {
            if (0..height).any(|y| is_uncroppable_pixel(x, y, image, check_max)) {
                right = x;
                break;
            }
        }

        // Find the top boundary
        for y in 0..height {
            if (0..width).any(|x| is_uncroppable_pixel(x, y, image, check_max)) {
                top = y;
                break;
            }
        }

        // Find the bottom boundary
        for y in (0..height).rev() {
            if (0..width).any(|x| is_uncroppable_pixel(x, y, image, check_max)) {
                bottom = y;
                break;
            }
        }

        let width = right - left + 1;
        let height = bottom - top + 1;
        Crop {
            left,
            top,
            width,
            height,
        }
    }

    pub fn new_from_images(images: &[&DynamicImage], check_max: bool) -> Self {
        images
            .iter()
            .map(|&image| Crop::new(image, check_max))
            .reduce(|a, b| a.union(&b))
            .unwrap()
    }

    pub fn xyxy(&self) -> (u32, u32, u32, u32) {
        (
            self.left,
            self.top,
            self.left + self.width,
            self.top + self.height,
        )
    }

    pub fn union(&self, other: &Crop) -> Crop {
        let (x1, y1, x2, y2) = self.xyxy();
        let (ox1, oy1, ox2, oy2) = other.xyxy();

        let left = x1.min(ox1);
        let top = y1.min(oy1);
        let width = x2.max(ox2) - left;
        let height = y2.max(oy2) - top;

        Crop {
            left,
            top,
            width,
            height,
        }
    }

    pub fn intersection(&self, other: &Crop) -> Crop {
        let (x1, y1, x2, y2) = self.xyxy();
        let (ox1, oy1, ox2, oy2) = other.xyxy();

        let left = x1.max(ox1);
        let top = y1.max(oy1);
        let width = x2.min(ox2) - left;
        let height = y2.min(oy2) - top;

        Crop {
            left,
            top,
            width,
            height,
        }
    }
}

fn is_uncroppable_pixel(x: u32, y: u32, image: &DynamicImage, check_max: bool) -> bool {
    let croppable = match image {
        DynamicImage::ImageLuma8(image) => {
            let pixel = image.get_pixel(x, y).channels()[0];
            pixel == 0 || (check_max && pixel == u8::max_value())
        }
        DynamicImage::ImageLumaA8(image) => {
            let pixel = image.get_pixel(x, y).channels()[0];
            pixel == 0 || (check_max && pixel == u8::max_value())
        }
        DynamicImage::ImageLuma16(image) => {
            let pixel = image.get_pixel(x, y).channels()[0];
            pixel == 0 || (check_max && pixel == u16::max_value())
        }
        DynamicImage::ImageLumaA16(image) => {
            let pixel = image.get_pixel(x, y).channels()[0];
            pixel == 0 || (check_max && pixel == u16::max_value())
        }
        DynamicImage::ImageRgb8(image) => {
            let pixel = image.get_pixel(x, y).to_luma().channels()[0];
            pixel == 0 || (check_max && pixel == u8::max_value())
        }
        DynamicImage::ImageRgba8(image) => {
            let pixel = image.get_pixel(x, y).to_luma().channels()[0];
            pixel == 0 || (check_max && pixel == u8::max_value())
        }
        DynamicImage::ImageRgb16(image) => {
            let pixel = image.get_pixel(x, y).to_luma().channels()[0];
            pixel == 0 || (check_max && pixel == u16::max_value())
        }
        DynamicImage::ImageRgba16(image) => {
            let pixel = image.get_pixel(x, y).to_luma().channels()[0];
            pixel == 0 || (check_max && pixel == u16::max_value())
        }
        // On fallback we can only check for zero. Only floating point types should hit this
        // branch.
        _ => {
            let pixel = image.get_pixel(x, y).to_luma().channels()[0];
            pixel == 0
        }
    };
    !croppable
}

impl From<&DynamicImage> for Crop {
    fn from(image: &DynamicImage) -> Self {
        Crop::new(image, DEFAULT_CHECK_MAX)
    }
}

impl From<&[&DynamicImage]> for Crop {
    fn from(images: &[&DynamicImage]) -> Self {
        images
            .iter()
            .map(|&image| Crop::from(image))
            .reduce(|a, b| a.union(&b))
            .unwrap()
    }
}

impl Transform<DynamicImage> for Crop {
    fn apply(&self, image: &DynamicImage) -> DynamicImage {
        image.crop_imm(self.left, self.top, self.width, self.height)
    }
}

impl WriteTags for Crop {
    fn write_tags<W, C, K, D>(&self, tiff: &mut ImageEncoder<W, C, K, D>) -> Result<(), TiffError>
    where
        W: Write + Seek,
        C: ColorType,
        K: TiffKind,
        D: Compression,
    {
        let origin = vec![self.left + self.width / 2, self.top + self.height / 2];
        let size = vec![self.width, self.height];

        let origin_tag = Tag::Unknown(DEFAULT_CROP_ORIGIN);
        let size_tag = Tag::Unknown(DEFAULT_CROP_SIZE);

        tiff.encoder().write_tag(origin_tag, origin.as_slice())?;
        tiff.encoder().write_tag(size_tag, size.as_slice())?;
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
            vec![0, 0, 0, 0],
            vec![0, 1, 1, 0],
            vec![0, 1, 1, 0],
            vec![0, 0, 0, 0],
        ],
        false,
        (1, 1, 2, 2)
    )]
    #[case(
        vec![
            vec![0, 0, 0, 0],
            vec![0, 0, 0, 0],
            vec![0, 0, 0, 0],
            vec![0, 0, 0, 0],
        ],
        false,
        (0, 0, 4, 4)
    )]
    #[case(
        vec![
            vec![1, 1, 1, 1],
            vec![1, 1, 1, 1],
            vec![1, 1, 1, 1],
            vec![1, 1, 1, 1],
        ],
        false,
        (0, 0, 4, 4)
    )]
    #[case(
        vec![
            vec![255, 0, 0, 0],
            vec![0, 1, 1, 0],
            vec![0, 1, 1, 0],
            vec![0, 0, 0, 255],
        ],
        true,
        (1, 1, 2, 2)
    )]
    fn test_find_non_zero_boundaries(
        #[case] pixels: Vec<Vec<u8>>,
        #[case] crop_max: bool,
        #[case] expected_crop: (u32, u32, u32, u32),
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

        let crop = Crop::new(&dynamic_image, crop_max);
        let expected_crop = Crop {
            left: expected_crop.0,
            top: expected_crop.1,
            width: expected_crop.2,
            height: expected_crop.3,
        };
        assert_eq!(crop, expected_crop);
    }

    #[rstest]
    #[case(Crop { left: 1, top: 1, width: 2, height: 2 }, (1, 1, 3, 3))]
    #[case(Crop { left: 0, top: 0, width: 4, height: 4 }, (0, 0, 4, 4))]
    #[case(Crop { left: 2, top: 2, width: 1, height: 1 }, (2, 2, 3, 3))]
    fn test_xyxy(#[case] crop: Crop, #[case] expected: (u32, u32, u32, u32)) {
        let result = crop.xyxy();
        assert_eq!(result, expected);
    }

    #[rstest]
    #[case(
        Crop { left: 1, top: 1, width: 2, height: 2 },
        Crop { left: 2, top: 2, width: 2, height: 2 },
        Crop { left: 1, top: 1, width: 3, height: 3 }
    )]
    #[case(
        Crop { left: 0, top: 0, width: 4, height: 4 },
        Crop { left: 1, top: 1, width: 2, height: 2 },
        Crop { left: 0, top: 0, width: 4, height: 4 }
    )]
    #[case(
        Crop { left: 1, top: 1, width: 2, height: 2 },
        Crop { left: 1, top: 1, width: 2, height: 2 },
        Crop { left: 1, top: 1, width: 2, height: 2 }
    )]
    fn test_union(#[case] crop1: Crop, #[case] crop2: Crop, #[case] expected: Crop) {
        let result = crop1.union(&crop2);
        assert_eq!(result, expected);
    }

    #[rstest]
    #[case(
        Crop { left: 1, top: 1, width: 2, height: 2 },
        Crop { left: 2, top: 2, width: 2, height: 2 },
        Crop { left: 2, top: 2, width: 1, height: 1 }
    )]
    #[case(
        Crop { left: 0, top: 0, width: 4, height: 4 },
        Crop { left: 1, top: 1, width: 2, height: 2 },
        Crop { left: 1, top: 1, width: 2, height: 2 }
    )]
    #[case(
        Crop { left: 1, top: 1, width: 2, height: 2 },
        Crop { left: 1, top: 1, width: 2, height: 2 },
        Crop { left: 1, top: 1, width: 2, height: 2 }
    )]
    fn test_intersection(#[case] crop1: Crop, #[case] crop2: Crop, #[case] expected: Crop) {
        let result = crop1.intersection(&crop2);
        assert_eq!(result, expected);
    }
}
