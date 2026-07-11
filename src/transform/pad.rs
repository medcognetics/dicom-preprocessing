use image::{DynamicImage, GenericImage, GenericImageView, ImageBuffer, Luma, Pixel};
use std::fmt;
use std::io::{Read, Seek, Write};
use tiff::decoder::Decoder;
use tiff::encoder::colortype::ColorType;
use tiff::encoder::ImageEncoder;
use tiff::encoder::TiffKind;
use tiff::tags::Tag;

use crate::errors::tiff::TiffError;
use crate::metadata::WriteTags;
use crate::transform::{Coord, InvertibleTransform, Transform};

pub const ACTIVE_AREA: u16 = 50829;

type Luma16Image = ImageBuffer<Luma<u16>, Vec<u16>>;

#[derive(Clone, Debug, clap::ValueEnum, Default, Copy)]
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
        write!(f, "{direction_str}")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Padding {
    pub left: u32,
    pub top: u32,
    pub right: u32,
    pub bottom: u32,
}

impl Padding {
    const TAG_CARDINALITY: usize = 4;

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
            PaddingDirection::Center => {
                let left = padding_width / 2;
                let top = padding_height / 2;
                Padding {
                    left,
                    top,
                    right: padding_width - left,
                    bottom: padding_height - top,
                }
            }
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

    fn apply_luma16(&self, image: &Luma16Image) -> DynamicImage {
        let (source_width_px, source_height_px) = image.dimensions();
        let destination_width_px = source_width_px + self.left + self.right;
        let destination_height_px = source_height_px + self.top + self.bottom;
        let mut padded_image = Luma16Image::new(destination_width_px, destination_height_px);

        let source_width = source_width_px as usize;
        let source_height = source_height_px as usize;
        let destination_width = destination_width_px as usize;
        let left_offset = self.left as usize;
        let top_offset = self.top as usize;
        let source = image.as_flat_samples().samples;

        {
            let destination = padded_image.as_flat_samples_mut().samples;

            for row in 0..source_height {
                let source_start = row * source_width;
                let source_end = source_start + source_width;
                let destination_start = (row + top_offset) * destination_width + left_offset;
                let destination_end = destination_start + source_width;
                destination[destination_start..destination_end]
                    .copy_from_slice(&source[source_start..source_end]);
            }
        }

        DynamicImage::ImageLuma16(padded_image)
    }
}

impl Transform<DynamicImage> for Padding {
    fn apply(&self, image: &DynamicImage) -> DynamicImage {
        if let DynamicImage::ImageLuma16(image) = image {
            return self.apply_luma16(image);
        }

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

impl Transform<Coord> for Padding {
    fn apply(&self, coord: &Coord) -> Coord {
        let (x, y): (u32, u32) = coord.into();
        Coord::new(x + self.left, y + self.top)
    }
}

impl InvertibleTransform<Coord> for Padding {
    fn invert(&self, coord: &Coord) -> Coord {
        let (x, y): (u32, u32) = coord.into();
        let (left, top) = (self.left, self.top);
        let new_x = x.saturating_sub(left);
        let new_y = y.saturating_sub(top);
        Coord::new(new_x, new_y)
    }
}

impl WriteTags for Padding {
    fn write_tags<W, C, K>(&self, tiff: &mut ImageEncoder<W, C, K>) -> Result<(), TiffError>
    where
        W: Write + Seek,
        C: ColorType,
        K: TiffKind,
    {
        let tag = Tag::Unknown(ACTIVE_AREA);
        let active_area = vec![self.left, self.top, self.right, self.bottom];
        tiff.encoder().write_tag(tag, active_area.as_slice())?;
        Ok(())
    }
}

impl<T> TryFrom<&mut Decoder<T>> for Padding
where
    T: Read + Seek,
{
    type Error = TiffError;

    /// Read the padding metadata from a TIFF file
    fn try_from(decoder: &mut Decoder<T>) -> Result<Self, Self::Error> {
        let active_area = decoder.get_tag_u32_vec(Tag::Unknown(ACTIVE_AREA))?;
        Padding::try_from(active_area)
    }
}

impl From<Padding> for (u32, u32, u32, u32) {
    fn from(padding: Padding) -> Self {
        (padding.left, padding.top, padding.right, padding.bottom)
    }
}

impl From<(u32, u32, u32, u32)> for Padding {
    fn from((left, top, right, bottom): (u32, u32, u32, u32)) -> Self {
        Padding {
            left,
            top,
            right,
            bottom,
        }
    }
}

impl TryFrom<Vec<u32>> for Padding {
    type Error = TiffError;
    fn try_from(vec: Vec<u32>) -> Result<Self, Self::Error> {
        if vec.len() != Self::TAG_CARDINALITY {
            return Err(TiffError::CardinalityError {
                name: "ActiveArea",
                actual: vec.len(),
                expected: Self::TAG_CARDINALITY,
            });
        }
        Ok((vec[0], vec[1], vec[2], vec[3]).into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, ImageBuffer, Luma, RgbaImage};
    use rstest::rstest;
    use std::fs::File;
    use tempfile::tempdir;
    use tiff::decoder::Decoder;
    use tiff::encoder::TiffEncoder;

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
            vec![1, 1],
            vec![1, 1],
        ],
        (5, 7),
        PaddingDirection::Center,
        Padding { left: 1, top: 2, right: 2, bottom: 3 }
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

    #[test]
    fn test_pad_luma16_fast_path() {
        const WIDTH: u32 = 3;
        const HEIGHT: u32 = 2;
        const PADDING: Padding = Padding {
            left: 1,
            top: 2,
            right: 2,
            bottom: 1,
        };
        const OUTPUT_WIDTH: u32 = WIDTH + PADDING.left + PADDING.right;
        const OUTPUT_HEIGHT: u32 = HEIGHT + PADDING.top + PADDING.bottom;

        fn pixel_value(x: u32, y: u32) -> u16 {
            (y * WIDTH + x + 1) as u16
        }

        let image = ImageBuffer::from_fn(WIDTH, HEIGHT, |x, y| Luma([pixel_value(x, y)]));
        let dynamic_image = DynamicImage::ImageLuma16(image);

        let padded_image = PADDING.apply(&dynamic_image);
        let padded = padded_image
            .as_luma16()
            .expect("Padding Luma16 input should preserve Luma16 color type");
        assert_eq!(padded.dimensions(), (OUTPUT_WIDTH, OUTPUT_HEIGHT));

        let mut expected = vec![0_u16; (OUTPUT_WIDTH * OUTPUT_HEIGHT) as usize];
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let destination_x = x + PADDING.left;
                let destination_y = y + PADDING.top;
                let destination_index = (destination_y * OUTPUT_WIDTH + destination_x) as usize;
                expected[destination_index] = pixel_value(x, y);
            }
        }

        assert_eq!(padded.as_raw().as_slice(), expected.as_slice());
    }

    #[test]
    fn center_padding_with_odd_differences_reaches_target_dimensions() {
        const SOURCE_WIDTH: u32 = 2;
        const SOURCE_HEIGHT: u32 = 2;
        const TARGET_WIDTH: u32 = 5;
        const TARGET_HEIGHT: u32 = 7;
        let image = DynamicImage::ImageLuma16(ImageBuffer::from_pixel(
            SOURCE_WIDTH,
            SOURCE_HEIGHT,
            Luma([7]),
        ));

        let padding = Padding::new(
            &image,
            TARGET_WIDTH,
            TARGET_HEIGHT,
            PaddingDirection::Center,
        );
        let padded = padding.apply(&image);

        assert_eq!(
            padding,
            Padding {
                left: 1,
                top: 2,
                right: 2,
                bottom: 3,
            }
        );
        assert_eq!(padded.dimensions(), (TARGET_WIDTH, TARGET_HEIGHT));
        assert_eq!(padded.as_luma16().unwrap().get_pixel(1, 2), &Luma([7]));
        assert_eq!(padding.left + padding.right, TARGET_WIDTH - SOURCE_WIDTH);
        assert_eq!(padding.top + padding.bottom, TARGET_HEIGHT - SOURCE_HEIGHT);
    }

    #[rstest]
    #[case(Padding { left: 1, top: 1, right: 2, bottom: 2 })]
    #[case(Padding { left: 1, top: 2, right: 2, bottom: 3 })]
    fn test_write_tags(#[case] padding: Padding) {
        // Prepare the TIFF
        let temp_dir = tempdir().unwrap();
        let temp_file_path = temp_dir.path().join("temp.tif");
        let mut tiff = TiffEncoder::new(File::create(temp_file_path.clone()).unwrap()).unwrap();
        let mut img = tiff
            .new_image::<tiff::encoder::colortype::Gray16>(1, 1)
            .unwrap();

        // Write the tags
        padding.write_tags(&mut img).unwrap();

        // Write some dummy image data
        let data: Vec<u16> = vec![0; 2];
        img.write_data(data.as_slice()).unwrap();

        // Read the TIFF back
        let mut tiff = Decoder::new(File::open(temp_file_path).unwrap()).unwrap();
        let actual = Padding::try_from(&mut tiff).unwrap();
        assert_eq!(padding, actual);
    }

    #[rstest]
    #[case(
        Padding { left: 1, top: 1, right: 2, bottom: 2 },
        Coord::new(0, 0),
        Coord::new(1, 1)
    )]
    #[case(
        Padding { left: 2, top: 3, right: 1, bottom: 1 },
        Coord::new(5, 5),
        Coord::new(7, 8)
    )]
    fn test_apply_coord(#[case] padding: Padding, #[case] coord: Coord, #[case] expected: Coord) {
        let result = padding.apply(&coord);
        assert_eq!(result, expected);
    }

    #[rstest]
    #[case(
        Padding { left: 1, top: 1, right: 2, bottom: 2 },
        Coord::new(1, 1),
        Coord::new(0, 0)
    )]
    #[case(
        Padding { left: 2, top: 3, right: 1, bottom: 1 },
        Coord::new(7, 8),
        Coord::new(5, 5)
    )]
    #[case(
        Padding { left: 2, top: 2, right: 1, bottom: 1 },
        Coord::new(1, 1),
        Coord::new(0, 0)
    )]
    fn test_invert_coord(#[case] padding: Padding, #[case] coord: Coord, #[case] expected: Coord) {
        let result = padding.invert(&coord);
        assert_eq!(result, expected);
    }
}
