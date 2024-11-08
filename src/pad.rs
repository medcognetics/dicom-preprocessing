use crate::traits::{Transform, WriteTags};
use image::{DynamicImage, GenericImage, GenericImageView, Pixel};
use std::fmt;
use std::io::{Seek, Write};
use tiff::encoder::colortype::ColorType;
use tiff::encoder::compression::Compression;
use tiff::encoder::ImageEncoder;
use tiff::encoder::TiffKind;
use tiff::tags::Tag;
use tiff::TiffError;

const ACTIVE_AREA: u16 = 50829;

#[derive(Clone, Debug, clap::ValueEnum, Default)]
pub enum PaddingDirection {
    #[default]
    Zero,
    TopLeft,
    BottomRight,
    Center,
}

impl fmt::Display for PaddingDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let direction_str = match self {
            PaddingDirection::TopLeft => "top-left",
            PaddingDirection::BottomRight => "bottom-right",
            PaddingDirection::Center => "center",
            PaddingDirection::Zero => "zero",
        };
        write!(f, "{}", direction_str)
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct Padding {
    pub left: u32,
    pub top: u32,
    pub right: u32,
    pub bottom: u32,
}

impl Padding {
    pub fn is_zero(&self) -> bool {
        self.left == 0 && self.top == 0 && self.right == 0 && self.bottom == 0
    }

    pub fn new(
        image: &DynamicImage,
        target_width: u32,
        target_height: u32,
        direction: PaddingDirection,
    ) -> Self {
        let (width, height) = image.dimensions();
        let padding_width = target_width - width;
        let padding_height = target_height - height;

        let is_zero = |x, y| image.get_pixel(x, y).to_luma().channels()[0] == 0;

        match direction {
            PaddingDirection::TopLeft => Padding {
                left: padding_width,
                top: padding_height,
                right: 0,
                bottom: 0,
            },
            PaddingDirection::BottomRight => Padding {
                left: 0,
                top: 0,
                right: padding_width,
                bottom: padding_height,
            },
            PaddingDirection::Center => Padding {
                left: padding_width / 2,
                top: padding_height / 2,
                right: padding_width / 2,
                bottom: padding_height / 2,
            },
            PaddingDirection::Zero => {
                // Count the number of zero pixels on each side
                let count_left = (0..height).map(|y| is_zero(0, y)).filter(|&v| v).count();
                let count_right = (0..height)
                    .map(|y| is_zero(width - 1, y))
                    .filter(|&v| v)
                    .count();
                let count_top = (0..width).map(|x| is_zero(x, 0)).filter(|&v| v).count();
                let count_bottom = (0..width)
                    .map(|x| is_zero(x, height - 1))
                    .filter(|&v| v)
                    .count();

                // Place all padding on the side with the most zero pixels
                let (left, right) = match count_left < count_right {
                    true => (0, padding_width),
                    false => (padding_width, 0),
                };
                let (top, bottom) = match count_top < count_bottom {
                    true => (0, padding_height),
                    false => (padding_height, 0),
                };

                Padding {
                    left,
                    top,
                    right,
                    bottom,
                }
            }
        }
    }
}

impl Transform for Padding {
    fn apply(&self, image: &DynamicImage) -> DynamicImage {
        let (width, height) = image.dimensions();
        let mut padded_image = DynamicImage::new(
            width + self.left + self.right,
            height + self.top + self.bottom,
            image.color(),
        );

        for y in 0..height {
            for x in 0..width {
                let pixel = image.get_pixel(x, y);
                padded_image.put_pixel(x + self.left, y + self.top, pixel);
            }
        }
        padded_image
    }
}

impl WriteTags for Padding {
    fn write_tags<W, C, K, D>(&self, tiff: &mut ImageEncoder<W, C, K, D>) -> Result<(), TiffError>
    where
        W: Write + Seek,
        C: ColorType,
        K: TiffKind,
        D: Compression,
    {
        let tag = Tag::Unknown(ACTIVE_AREA);
        let active_area = vec![self.left, self.top, self.right, self.bottom];
        tiff.encoder().write_tag(tag, active_area.as_slice())?;
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
        (6, 6),
        PaddingDirection::Center,
        Padding { left: 1, top: 1, right: 1, bottom: 1 }
    )]
    #[case(
        vec![
            vec![0, 0, 0, 0],
            vec![0, 1, 1, 0],
            vec![0, 1, 1, 0],
            vec![0, 0, 0, 0],
        ],
        (6, 6),
        PaddingDirection::TopLeft,
        Padding { left: 2, top: 2, right: 0, bottom: 0 }
    )]
    #[case(
        vec![
            vec![1, 1, 0, 0],
            vec![1, 1, 0, 0],
            vec![1, 1, 0, 0],
            vec![1, 0, 0, 0],
        ],
        (6, 8),
        PaddingDirection::Zero,
        Padding { left: 0, top: 0, right: 2, bottom: 4 }
    )]
    #[case(
        vec![
            vec![0, 0, 0, 1],
            vec![0, 0, 1, 1],
            vec![0, 0, 1, 1],
            vec![0, 0, 1, 1],
        ],
        (6, 8),
        PaddingDirection::Zero,
        Padding { left: 2, top: 4, right: 0, bottom: 0 }
    )]
    fn test_compute_required_padding(
        #[case] pixels: Vec<Vec<u8>>,
        #[case] (target_width, target_height): (u32, u32),
        #[case] direction: PaddingDirection,
        #[case] expected_padding: Padding,
    ) {
        // Create image from pixels
        let width = pixels[0].len() as u32;
        let height = pixels.len() as u32;
        let mut img = RgbaImage::new(width, height);
        for (y, row) in pixels.iter().enumerate() {
            for (x, &value) in row.iter().enumerate() {
                img.put_pixel(x as u32, y as u32, image::Rgba([value, value, value, 255]));
            }
        }
        let dynamic_image = DynamicImage::ImageRgba8(img);

        let padding = Padding::new(&dynamic_image, target_width, target_height, direction);
        assert_eq!(padding, expected_padding);
    }

    #[rstest]
    #[case(
        vec![
            vec![0, 0, 0, 0],
            vec![0, 1, 1, 0],
            vec![0, 1, 1, 0],
            vec![0, 0, 0, 0],
        ],
        (6, 6),
        PaddingDirection::Center,
        vec![
            vec![0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0],
            vec![0, 0, 1, 1, 0, 0],
            vec![0, 0, 1, 1, 0, 0],
            vec![0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0],
        ],
    )]
    #[case(
        vec![
            vec![1, 1, 0],
            vec![1, 1, 0],
            vec![0, 0, 0],
        ],
        (3, 4),
        PaddingDirection::TopLeft,
        vec![
            vec![0, 0, 0],
            vec![1, 1, 0],
            vec![1, 1, 0],
            vec![0, 0, 0],
        ],
    )]
    #[case(
        vec![
            vec![1, 1, 0],
            vec![1, 1, 0],
            vec![0, 0, 0],
        ],
        (4, 3),
        PaddingDirection::Zero,
        vec![
            vec![1, 1, 0, 0],
            vec![1, 1, 0, 0],
            vec![0, 0, 0, 0],
        ],
    )]
    fn test_pad(
        #[case] pixels: Vec<Vec<u8>>,
        #[case] (target_width, target_height): (u32, u32),
        #[case] direction: PaddingDirection,
        #[case] expected_pixels: Vec<Vec<u8>>,
    ) {
        // Create image from pixels
        let width = pixels[0].len() as u32;
        let height = pixels.len() as u32;
        let mut img = RgbaImage::new(width, height);
        for (y, row) in pixels.iter().enumerate() {
            for (x, &value) in row.iter().enumerate() {
                img.put_pixel(x as u32, y as u32, image::Rgba([value, value, value, 255]));
            }
        }
        let dynamic_image = DynamicImage::ImageRgba8(img);

        let padding = Padding::new(&dynamic_image, target_width, target_height, direction);
        let padded_image = padding.apply(&dynamic_image);
        let padded_pixels: Vec<Vec<u8>> = (0..target_height)
            .map(|y| {
                (0..target_width)
                    .map(|x| padded_image.get_pixel(x, y).0[0])
                    .collect()
            })
            .collect();

        assert_eq!(padded_pixels, expected_pixels);
    }
}
