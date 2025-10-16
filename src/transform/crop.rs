use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView, Luma, Pixel};
use imageproc::contrast::{threshold, ThresholdType};
use imageproc::region_labelling::{connected_components, Connectivity};
use itertools::Itertools;
use std::io::{Read, Seek, Write};
use tiff::decoder::Decoder;
use tiff::encoder::colortype::ColorType;
use tiff::encoder::ImageEncoder;
use tiff::encoder::TiffKind;
use tiff::tags::Tag;

use crate::errors::tiff::TiffError;
use crate::metadata::WriteTags;
use crate::transform::{Coord, InvertibleTransform, Transform};

pub const DEFAULT_CROP_ORIGIN: u16 = 50719;
pub const DEFAULT_CROP_SIZE: u16 = 50720;
const DEFAULT_CHECK_MAX: bool = false;
const DEFAULT_RESIZE_SIZE: u32 = 512;
const NONZERO_THRESHOLD: u8 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Crop {
    pub left: u32,
    pub top: u32,
    pub width: u32,
    pub height: u32,
}

impl Crop {
    const TAG_CARDINALITY: usize = 2;

    pub fn new(image: &DynamicImage, check_max: bool, border_frac: Option<f32>) -> Self {
        let (width, height) = image.dimensions();

        let (start_w, start_h, end_w, end_h) = if let Some(border_frac) = border_frac {
            (
                (width as f32 * border_frac) as u32,
                (height as f32 * border_frac) as u32,
                (width as f32 * (1.0 - border_frac)) as u32,
                (height as f32 * (1.0 - border_frac)) as u32,
            )
        } else {
            (0, 0, width, height)
        };

        let left = (start_w..end_w)
            .find(|&x| (start_h..end_h).any(|y| is_uncroppable_pixel(x, y, image, check_max)))
            .unwrap_or(0);

        let right = (start_w..end_w)
            .rev()
            .find(|&x| (start_h..end_h).any(|y| is_uncroppable_pixel(x, y, image, check_max)))
            .unwrap_or(width - 1);

        let top = (start_h..end_h)
            .find(|&y| (start_w..end_w).any(|x| is_uncroppable_pixel(x, y, image, check_max)))
            .unwrap_or(0);

        let bottom = (start_h..end_h)
            .rev()
            .find(|&y| (start_w..end_w).any(|x| is_uncroppable_pixel(x, y, image, check_max)))
            .unwrap_or(height - 1);

        // Expand the crop by the border fraction if provided
        let (left, right, top, bottom) = if let Some(border_frac) = border_frac {
            let offset_w = (width as f32 * border_frac) as u32;
            let offset_h = (height as f32 * border_frac) as u32;

            let left = (left as i64 - offset_w as i64).max(0) as u32;
            let right = (right + offset_w).min(width - 1);
            let top = (top as i64 - offset_h as i64).max(0) as u32;
            let bottom = (bottom + offset_h).min(height - 1);
            (left, right, top, bottom)
        } else {
            (left, right, top, bottom)
        };

        let width = right - left + 1;
        let height = bottom - top + 1;
        Crop {
            left,
            top,
            width,
            height,
        }
    }

    pub fn new_from_components(
        image: &DynamicImage,
        check_max: bool,
        border_frac: Option<f32>,
    ) -> Self {
        // First generate a baseline crop
        let image = image.clone();
        let crop = Crop::new(&image, check_max, border_frac);
        let thumbnail = image.crop_imm(crop.left, crop.top, crop.width, crop.height);

        // Resize the image to smaller size for fast computation of crop boundaries
        let thumbnail = if thumbnail.width() > DEFAULT_RESIZE_SIZE
            || thumbnail.height() > DEFAULT_RESIZE_SIZE
        {
            thumbnail.resize(
                DEFAULT_RESIZE_SIZE,
                DEFAULT_RESIZE_SIZE,
                FilterType::Nearest,
            )
        } else {
            thumbnail
        };

        // Threshold the image to find the background
        // NOTE: This assumes that the background is black
        let thumbnail = thumbnail.into_luma8();
        let thumbnail = threshold(&thumbnail, NONZERO_THRESHOLD, ThresholdType::Binary);

        // Compute connected components
        let bg = Luma([0]);
        let components = connected_components(&thumbnail, Connectivity::Four, bg);

        // Find the largest connected component
        let (max_component, _) = components
            .iter()
            .filter(|&&c| c > 0)
            .counts()
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .unwrap_or((&0, 0));

        // Compute crop bounds from the largest connected component
        let (left, right, top, bottom) = components
            .enumerate_pixels()
            .filter(|&(_, _, c)| c[0] == *max_component)
            .fold(
                (thumbnail.width() - 1, 0, thumbnail.height() - 1, 0),
                |(min_x, max_x, min_y, max_y), (x, y, _)| {
                    (min_x.min(x), max_x.max(x), min_y.min(y), max_y.max(y))
                },
            );

        // Determine scale factor between resized and original cropped image
        let orig_width = crop.width;
        let orig_height = crop.height;
        let scale_w = orig_width as f32 / thumbnail.width() as f32;
        let scale_h = orig_height as f32 / thumbnail.height() as f32;

        // Scale crop coordinates to the original image size
        let left = (left as f32 * scale_w).round() as u32;
        let top = (top as f32 * scale_h).round() as u32;
        let right = (right as f32 * scale_w).round() as u32;
        let bottom = (bottom as f32 * scale_h).round() as u32;

        // Offset crop coordinates to the original image
        let left = left + crop.left;
        let top = top + crop.top;
        let right = right + crop.left;
        let bottom = bottom + crop.top;

        let width = right - left + 1;
        let height = bottom - top + 1;
        Crop {
            left,
            top,
            width,
            height,
        }
    }

    pub fn new_from_images(
        images: &[&DynamicImage],
        check_max: bool,
        use_components: bool,
        border_frac: Option<f32>,
    ) -> Self {
        images
            .iter()
            .map(|&image| {
                if use_components {
                    Crop::new_from_components(image, check_max, border_frac)
                } else {
                    Crop::new(image, check_max, border_frac)
                }
            })
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
    let (max_value, pixel_value) = match image {
        DynamicImage::ImageLuma8(image) => {
            (u8::MAX as u16, image.get_pixel(x, y).channels()[0] as u16)
        }
        DynamicImage::ImageLumaA8(image) => {
            (u8::MAX as u16, image.get_pixel(x, y).channels()[1] as u16)
        }
        DynamicImage::ImageLuma16(image) => (u16::MAX, image.get_pixel(x, y).channels()[0]),
        DynamicImage::ImageLumaA16(image) => (u16::MAX, image.get_pixel(x, y).channels()[1]),
        DynamicImage::ImageRgb8(image) => {
            (u8::MAX as u16, image.get_pixel(x, y).channels()[0] as u16)
        }
        DynamicImage::ImageRgba8(image) => (
            u8::MAX as u16,
            image.get_pixel(x, y).to_luma().channels()[0] as u16,
        ),
        DynamicImage::ImageRgb16(image) => (u16::MAX, image.get_pixel(x, y).channels()[0]),
        DynamicImage::ImageRgba16(image) => (u16::MAX, image.get_pixel(x, y).channels()[1]),
        _ => (
            u16::MAX,
            image.get_pixel(x, y).to_luma().channels()[0] as u16,
        ),
    };
    pixel_value != 0 && !(check_max && pixel_value == max_value)
}

impl From<&DynamicImage> for Crop {
    fn from(image: &DynamicImage) -> Self {
        Crop::new_from_components(image, DEFAULT_CHECK_MAX, None)
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

impl Transform<Coord> for Crop {
    fn apply(&self, coord: &Coord) -> Coord {
        let (x, y): (u32, u32) = coord.into();
        let (left, top, _, _) = self.xyxy();
        let new_x = if x > left {
            (x - left).min(self.width - 1)
        } else {
            0
        };
        let new_y = if y > top {
            (y - top).min(self.height - 1)
        } else {
            0
        };
        Coord::new(new_x, new_y)
    }
}

impl InvertibleTransform<Coord> for Crop {
    fn invert(&self, coord: &Coord) -> Coord {
        let (x, y): (u32, u32) = coord.into();
        let (left, top, _, _) = self.xyxy();
        let new_x = x + left;
        let new_y = y + top;
        Coord::new(new_x, new_y)
    }
}

impl WriteTags for Crop {
    fn write_tags<W, C, K>(&self, tiff: &mut ImageEncoder<W, C, K>) -> Result<(), TiffError>
    where
        W: Write + Seek,
        C: ColorType,
        K: TiffKind,
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

impl<T> TryFrom<&mut Decoder<T>> for Crop
where
    T: Read + Seek,
{
    type Error = TiffError;

    /// Read the crop metadata from a TIFF file
    fn try_from(decoder: &mut Decoder<T>) -> Result<Self, Self::Error> {
        // Read and parse crop origin
        let origin = decoder.get_tag_u32_vec(Tag::Unknown(DEFAULT_CROP_ORIGIN))?;
        if origin.len() != 2 {
            return Err(TiffError::CardinalityError {
                name: "DefaultCropOrigin",
                actual: origin.len(),
                expected: Self::TAG_CARDINALITY,
            });
        }
        let (origin_x, origin_y) = (origin[0], origin[1]);

        // Read and parse crop size
        let size = decoder.get_tag_u32_vec(Tag::Unknown(DEFAULT_CROP_SIZE))?;
        if size.len() != Self::TAG_CARDINALITY {
            return Err(TiffError::CardinalityError {
                name: "DefaultCropSize",
                actual: size.len(),
                expected: Self::TAG_CARDINALITY,
            });
        }
        let (width, height) = (size[0], size[1]);

        // Build final result
        let top = origin_y - height / 2;
        let left = origin_x - width / 2;
        Ok((left, top, width, height).into())
    }
}

impl From<Crop> for (u32, u32, u32, u32) {
    fn from(crop: Crop) -> Self {
        (crop.left, crop.top, crop.width, crop.height)
    }
}

impl From<(u32, u32, u32, u32)> for Crop {
    fn from((left, top, width, height): (u32, u32, u32, u32)) -> Self {
        Crop {
            left,
            top,
            width,
            height,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, RgbaImage};
    use rstest::rstest;
    use std::fs::File;
    use tempfile::tempdir;
    use tiff::decoder::Decoder as TiffDecoder;
    use tiff::encoder::TiffEncoder;

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

        let crop = Crop::new(&dynamic_image, crop_max, None);
        let expected_crop = Crop {
            left: expected_crop.0,
            top: expected_crop.1,
            width: expected_crop.2,
            height: expected_crop.3,
        };
        assert_eq!(crop, expected_crop);
    }

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
    fn test_find_non_zero_boundaries_components(
        #[case] pixels: Vec<Vec<u8>>,
        #[case] check_max: bool,
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

        let crop = Crop::new_from_components(&dynamic_image, check_max, None);
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

    #[rstest]
    #[case(Crop { left: 1, top: 1, width: 2, height: 2 })]
    fn test_write_tags(#[case] crop: Crop) {
        // Prepare the TIFF
        let temp_dir = tempdir().unwrap();
        let temp_file_path = temp_dir.path().join("temp.tif");
        let mut tiff = TiffEncoder::new(File::create(temp_file_path.clone()).unwrap()).unwrap();
        let mut img = tiff
            .new_image::<tiff::encoder::colortype::Gray16>(1, 1)
            .unwrap();

        // Write the tags
        crop.write_tags(&mut img).unwrap();

        // Write some dummy image data
        let data: Vec<u16> = vec![0; 2];
        img.write_data(data.as_slice()).unwrap();

        // Read the TIFF back
        let mut tiff = TiffDecoder::new(File::open(temp_file_path).unwrap()).unwrap();
        let actual = Crop::try_from(&mut tiff).unwrap();
        assert_eq!(crop, actual);
    }

    #[rstest]
    #[case(
        Crop { left: 1, top: 1, width: 2, height: 2 },
        Coord::new(3, 3),
        Coord::new(1, 1)
    )]
    #[case(
        Crop { left: 1, top: 1, width: 2, height: 2 },
        Coord::new(0, 0),
        Coord::new(0, 0)
    )]
    #[case(
        Crop { left: 1, top: 1, width: 2, height: 2 },
        Coord::new(4, 4),
        Coord::new(1, 1)
    )]
    fn test_apply_coord(#[case] crop: Crop, #[case] coord: Coord, #[case] expected: Coord) {
        let result = crop.apply(&coord);
        assert_eq!(result, expected);
    }

    #[rstest]
    #[case(
        vec![
            vec![0, 0, 0, 0, 0, 0],
            vec![0, 0, 1, 1, 0, 0],
            vec![0, 0, 1, 1, 0, 0],
            vec![0, 0, 1, 1, 0, 0],
            vec![0, 0, 1, 1, 0, 0],
            vec![0, 0, 0, 0, 0, 0],
        ],
        0.2,
        false,
        (1, 0, 4, 5)  // Actual behavior with border exclusion
    )]
    #[case(
        vec![
            vec![1, 1, 1, 1, 1, 1],
            vec![1, 0, 0, 0, 0, 1],
            vec![1, 0, 1, 1, 0, 1],
            vec![1, 0, 1, 1, 0, 1],
            vec![1, 0, 0, 0, 0, 1],
            vec![1, 1, 1, 1, 1, 1],
        ],
        0.1,
        false,
        (0, 0, 5, 5)  // Actual behavior with border exclusion
    )]
    fn test_border_exclusion(
        #[case] pixels: Vec<Vec<u8>>,
        #[case] border_frac: f32,
        #[case] check_max: bool,
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

        let crop = Crop::new(&dynamic_image, check_max, Some(border_frac));
        let expected_crop = Crop {
            left: expected_crop.0,
            top: expected_crop.1,
            width: expected_crop.2,
            height: expected_crop.3,
        };
        assert_eq!(crop, expected_crop);
    }

    #[rstest]
    #[case(
        Crop { left: 1, top: 1, width: 2, height: 2 },
        Coord::new(1, 1),
        Coord::new(2, 2)
    )]
    #[case(
        Crop { left: 1, top: 1, width: 2, height: 2 },
        Coord::new(0, 0),
        Coord::new(1, 1)
    )]
    #[case(
        Crop { left: 2, top: 2, width: 3, height: 3 },
        Coord::new(1, 1),
        Coord::new(3, 3)
    )]
    fn test_invert_coord(#[case] crop: Crop, #[case] coord: Coord, #[case] expected: Coord) {
        let result = crop.invert(&coord);
        assert_eq!(result, expected);
    }

    #[rstest]
    #[case(
        vec![
            vec![0, 0, 0, 0, 0, 0],
            vec![0, 0, 1, 1, 0, 0],
            vec![0, 0, 1, 1, 0, 0],
            vec![0, 0, 1, 1, 0, 0],
            vec![0, 0, 1, 1, 0, 0],
            vec![0, 0, 0, 0, 0, 0],
        ],
        Some(0.2),
        false,
        (1, 0, 4, 5)  // Actual behavior with border exclusion
    )]
    #[case(
        vec![
            vec![1, 1, 1, 1, 1, 1],
            vec![1, 0, 0, 0, 0, 1],
            vec![1, 0, 1, 1, 0, 1],
            vec![1, 0, 1, 1, 0, 1],
            vec![1, 0, 0, 0, 0, 1],
            vec![1, 1, 1, 1, 1, 1],
        ],
        None,
        false,
        (0, 0, 6, 6)  // Without border exclusion, should include the whole image
    )]
    fn test_new_from_images_with_border(
        #[case] pixels: Vec<Vec<u8>>,
        #[case] border_frac: Option<f32>,
        #[case] check_max: bool,
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

        let crop = Crop::new_from_images(&[&dynamic_image], check_max, true, border_frac);
        let expected_crop = Crop {
            left: expected_crop.0,
            top: expected_crop.1,
            width: expected_crop.2,
            height: expected_crop.3,
        };
        assert_eq!(crop, expected_crop);
    }
}
