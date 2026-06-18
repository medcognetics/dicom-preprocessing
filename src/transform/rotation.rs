use image::DynamicImage;
use std::io::{Read, Seek, Write};
use tiff::decoder::Decoder;
use tiff::encoder::colortype::ColorType;
use tiff::encoder::{ImageEncoder, TiffKind};
use tiff::tags::Tag;

use crate::errors::tiff::TiffError;
use crate::metadata::WriteTags;
use crate::transform::{Coord, InvertibleTransform, Transform};

pub const PREPROCESSING_ROTATION_180: u16 = 65000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rotation180 {
    pub width: u32,
    pub height: u32,
}

impl Rotation180 {
    const TAG_CARDINALITY: usize = 2;

    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    pub fn from_image(image: &DynamicImage) -> Self {
        Self::new(image.width(), image.height())
    }
}

impl Transform<DynamicImage> for Rotation180 {
    fn apply(&self, image: &DynamicImage) -> DynamicImage {
        image.rotate180()
    }
}

impl Transform<Coord> for Rotation180 {
    fn apply(&self, coord: &Coord) -> Coord {
        let (x, y): (u32, u32) = coord.into();
        let max_x = self.width.saturating_sub(1);
        let max_y = self.height.saturating_sub(1);
        Coord::new(
            max_x.saturating_sub(x.min(max_x)),
            max_y.saturating_sub(y.min(max_y)),
        )
    }
}

impl InvertibleTransform<Coord> for Rotation180 {
    fn invert(&self, coord: &Coord) -> Coord {
        self.apply(coord)
    }
}

impl WriteTags for Rotation180 {
    fn write_tags<W, C, K>(&self, tiff: &mut ImageEncoder<W, C, K>) -> Result<(), TiffError>
    where
        W: Write + Seek,
        C: ColorType,
        K: TiffKind,
    {
        let dimensions = vec![self.width, self.height];
        tiff.encoder().write_tag(
            Tag::Unknown(PREPROCESSING_ROTATION_180),
            dimensions.as_slice(),
        )?;
        Ok(())
    }
}

impl<T> TryFrom<&mut Decoder<T>> for Rotation180
where
    T: Read + Seek,
{
    type Error = TiffError;

    fn try_from(decoder: &mut Decoder<T>) -> Result<Self, Self::Error> {
        let dimensions = decoder.get_tag_u32_vec(Tag::Unknown(PREPROCESSING_ROTATION_180))?;
        Self::try_from(dimensions)
    }
}

impl TryFrom<Vec<u32>> for Rotation180 {
    type Error = TiffError;

    fn try_from(dimensions: Vec<u32>) -> Result<Self, Self::Error> {
        if dimensions.len() != Self::TAG_CARDINALITY {
            return Err(TiffError::CardinalityError {
                name: "PreprocessingRotation180",
                actual: dimensions.len(),
                expected: Self::TAG_CARDINALITY,
            });
        }
        Ok(Self::new(dimensions[0], dimensions[1]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, GenericImageView, ImageBuffer, Rgba};
    use std::fs::File;
    use tempfile::tempdir;
    use tiff::decoder::Decoder as TiffDecoder;
    use tiff::encoder::TiffEncoder;

    #[test]
    fn test_apply_coord() {
        let rotation = Rotation180::new(100, 80);
        assert_eq!(rotation.apply(&Coord::new(10, 30)), Coord::new(89, 49));
        assert_eq!(rotation.apply(&Coord::new(20, 40)), Coord::new(79, 39));
    }

    #[test]
    fn test_apply_image() {
        let image = DynamicImage::ImageRgba8(
            ImageBuffer::from_raw(
                2,
                2,
                vec![1, 0, 0, 255, 2, 0, 0, 255, 3, 0, 0, 255, 4, 0, 0, 255],
            )
            .unwrap(),
        );
        let rotated = Rotation180::from_image(&image).apply(&image);
        assert_eq!(rotated.get_pixel(0, 0), Rgba([4, 0, 0, 255]));
        assert_eq!(rotated.get_pixel(1, 1), Rgba([1, 0, 0, 255]));
    }

    #[test]
    fn test_write_tags() {
        let rotation = Rotation180::new(100, 80);
        let temp_dir = tempdir().unwrap();
        let temp_file_path = temp_dir.path().join("temp.tif");
        let mut tiff = TiffEncoder::new(File::create(temp_file_path.clone()).unwrap()).unwrap();
        let mut img = tiff
            .new_image::<tiff::encoder::colortype::Gray16>(1, 1)
            .unwrap();

        rotation.write_tags(&mut img).unwrap();
        img.write_data(&[0_u16]).unwrap();

        let mut tiff = TiffDecoder::new(File::open(temp_file_path).unwrap()).unwrap();
        let actual = Rotation180::try_from(&mut tiff).unwrap();
        assert_eq!(rotation, actual);
    }
}
