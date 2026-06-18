use image::DynamicImage;
use std::io::{Read, Seek, Write};
use tiff::decoder::Decoder;
use tiff::encoder::colortype::ColorType;
use tiff::encoder::{ImageEncoder, TiffKind};
use tiff::tags::Tag;

use crate::errors::tiff::TiffError;
use crate::metadata::WriteTags;
use crate::transform::{Coord, InvertibleTransform, Transform};

/// TIFF tag for preprocessing flips, encoded as
/// `[width, height, horizontal, vertical]`.
pub const PREPROCESSING_FLIP: u16 = 65000;

/// A horizontal and/or vertical flip in the original image coordinate space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Flip {
    /// Width of the coordinate space before the flip.
    pub width: u32,
    /// Height of the coordinate space before the flip.
    pub height: u32,
    /// Whether x coordinates were flipped.
    pub horizontal: bool,
    /// Whether y coordinates were flipped.
    pub vertical: bool,
}

impl Flip {
    const TAG_CARDINALITY: usize = 4;

    pub fn new(width: u32, height: u32, horizontal: bool, vertical: bool) -> Self {
        Self {
            width,
            height,
            horizontal,
            vertical,
        }
    }

    pub fn horizontal_from_image(image: &DynamicImage) -> Self {
        Self::new(image.width(), image.height(), true, false)
    }

    pub fn vertical_from_image(image: &DynamicImage) -> Self {
        Self::new(image.width(), image.height(), false, true)
    }

    pub fn both_from_image(image: &DynamicImage) -> Self {
        Self::new(image.width(), image.height(), true, true)
    }

    fn encoded_flags(&self) -> [u32; 2] {
        [u32::from(self.horizontal), u32::from(self.vertical)]
    }
}

impl Transform<DynamicImage> for Flip {
    fn apply(&self, image: &DynamicImage) -> DynamicImage {
        match (self.horizontal, self.vertical) {
            (true, true) => image.rotate180(),
            (true, false) => image.fliph(),
            (false, true) => image.flipv(),
            (false, false) => image.clone(),
        }
    }
}

impl Transform<Coord> for Flip {
    fn apply(&self, coord: &Coord) -> Coord {
        let (x, y): (u32, u32) = coord.into();
        let max_x = self.width.saturating_sub(1);
        let max_y = self.height.saturating_sub(1);
        let x = if self.horizontal {
            max_x.saturating_sub(x.min(max_x))
        } else {
            x
        };
        let y = if self.vertical {
            max_y.saturating_sub(y.min(max_y))
        } else {
            y
        };
        Coord::new(x, y)
    }
}

impl InvertibleTransform<Coord> for Flip {
    fn invert(&self, coord: &Coord) -> Coord {
        self.apply(coord)
    }
}

impl WriteTags for Flip {
    fn write_tags<W, C, K>(&self, tiff: &mut ImageEncoder<W, C, K>) -> Result<(), TiffError>
    where
        W: Write + Seek,
        C: ColorType,
        K: TiffKind,
    {
        let [horizontal, vertical] = self.encoded_flags();
        let values = vec![self.width, self.height, horizontal, vertical];
        tiff.encoder()
            .write_tag(Tag::Unknown(PREPROCESSING_FLIP), values.as_slice())?;
        Ok(())
    }
}

impl<T> TryFrom<&mut Decoder<T>> for Flip
where
    T: Read + Seek,
{
    type Error = TiffError;

    fn try_from(decoder: &mut Decoder<T>) -> Result<Self, Self::Error> {
        let values = decoder.get_tag_u32_vec(Tag::Unknown(PREPROCESSING_FLIP))?;
        Self::try_from(values)
    }
}

impl TryFrom<Vec<u32>> for Flip {
    type Error = TiffError;

    fn try_from(values: Vec<u32>) -> Result<Self, Self::Error> {
        if values.len() != Self::TAG_CARDINALITY {
            return Err(TiffError::CardinalityError {
                name: "PreprocessingFlip",
                actual: values.len(),
                expected: Self::TAG_CARDINALITY,
            });
        }
        let horizontal = bool_flag(values[2], "PreprocessingFlipHorizontal")?;
        let vertical = bool_flag(values[3], "PreprocessingFlipVertical")?;
        Ok(Self::new(values[0], values[1], horizontal, vertical))
    }
}

fn bool_flag(value: u32, name: &'static str) -> Result<bool, TiffError> {
    match value {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(TiffError::InvalidPropertyError { name }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, GenericImageView, ImageBuffer, Rgba};
    use rstest::rstest;
    use std::fs::File;
    use tempfile::tempdir;
    use tiff::decoder::Decoder as TiffDecoder;
    use tiff::encoder::TiffEncoder;

    #[rstest]
    #[case(
        Flip::new(100, 80, true, false),
        Coord::new(10, 30),
        Coord::new(89, 30)
    )]
    #[case(
        Flip::new(100, 80, false, true),
        Coord::new(10, 30),
        Coord::new(10, 49)
    )]
    #[case(Flip::new(100, 80, true, true), Coord::new(10, 30), Coord::new(89, 49))]
    #[case(Flip::new(100, 80, true, true), Coord::new(20, 40), Coord::new(79, 39))]
    fn test_apply_coord(#[case] flip: Flip, #[case] coord: Coord, #[case] expected: Coord) {
        assert_eq!(flip.apply(&coord), expected);
    }

    #[test]
    fn test_apply_image_both_axes() {
        let image = DynamicImage::ImageRgba8(
            ImageBuffer::from_raw(
                2,
                2,
                vec![1, 0, 0, 255, 2, 0, 0, 255, 3, 0, 0, 255, 4, 0, 0, 255],
            )
            .unwrap(),
        );
        let flipped = Flip::both_from_image(&image).apply(&image);
        assert_eq!(flipped.get_pixel(0, 0), Rgba([4, 0, 0, 255]));
        assert_eq!(flipped.get_pixel(1, 1), Rgba([1, 0, 0, 255]));
    }

    #[test]
    fn test_write_tags() {
        let flip = Flip::new(100, 80, true, false);
        let temp_dir = tempdir().unwrap();
        let temp_file_path = temp_dir.path().join("temp.tif");
        let mut tiff = TiffEncoder::new(File::create(temp_file_path.clone()).unwrap()).unwrap();
        let mut img = tiff
            .new_image::<tiff::encoder::colortype::Gray16>(1, 1)
            .unwrap();

        flip.write_tags(&mut img).unwrap();
        img.write_data(&[0_u16]).unwrap();

        let mut tiff = TiffDecoder::new(File::open(temp_file_path).unwrap()).unwrap();
        let actual = Flip::try_from(&mut tiff).unwrap();
        assert_eq!(flip, actual);
    }
}
