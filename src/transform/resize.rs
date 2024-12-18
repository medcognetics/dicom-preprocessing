use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView};
use std::fmt;
use std::io::{Read, Seek, Write};
use tiff::decoder::Decoder;
use tiff::encoder::colortype::ColorType;
use tiff::encoder::compression::Compression;
use tiff::encoder::ImageEncoder;
use tiff::encoder::TiffKind;
use tiff::tags::Tag;

use crate::errors::tiff::TiffError;
use crate::metadata::{Resolution, WriteTags};
use crate::transform::Transform;

pub const DEFAULT_SCALE: u16 = 50718;

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

impl From<DisplayFilterType> for FilterType {
    fn from(filter: DisplayFilterType) -> Self {
        match filter {
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
    const TAG_CARDINALITY: usize = 2;
    const DEFAULT_FILTER: FilterType = FilterType::Nearest;

    pub fn new(
        image: &DynamicImage,
        target_width: u32,
        target_height: u32,
        filter: FilterType,
    ) -> Self {
        // Determine scale factors
        let (width, height) = image.dimensions();
        let scale = (target_width as f32 / width as f32).min(target_height as f32 / height as f32);

        Resize {
            scale_x: scale,
            scale_y: scale,
            filter,
        }
    }
}

impl Transform<DynamicImage> for Resize {
    fn apply(&self, image: &DynamicImage) -> DynamicImage {
        let (width, height) = image.dimensions();
        let target_width = (width as f32 * self.scale_x) as u32;
        let target_height = (height as f32 * self.scale_y) as u32;
        image.resize(target_width, target_height, self.filter)
    }
}

impl Transform<Resolution> for Resize {
    fn apply(&self, resolution: &Resolution) -> Resolution {
        assert_eq!(
            self.scale_x, self.scale_y,
            "Expected scale_x and scale_y to be equal: {} {}",
            self.scale_x, self.scale_y
        );
        let scale = self.scale_x;
        resolution.scale(scale)
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

impl<T> TryFrom<&mut Decoder<T>> for Resize
where
    T: Read + Seek,
{
    type Error = TiffError;

    /// Read the resize metadata from a TIFF file
    fn try_from(decoder: &mut Decoder<T>) -> Result<Self, Self::Error> {
        let scale = decoder.get_tag_f32_vec(Tag::Unknown(DEFAULT_SCALE))?;
        Resize::try_from(scale)
    }
}

impl From<Resize> for (f32, f32) {
    fn from(resize: Resize) -> Self {
        (resize.scale_x, resize.scale_y)
    }
}

impl From<(f32, f32)> for Resize {
    fn from((scale_x, scale_y): (f32, f32)) -> Self {
        Resize {
            scale_x,
            scale_y,
            filter: Self::DEFAULT_FILTER,
        }
    }
}

impl TryFrom<Vec<f32>> for Resize {
    type Error = TiffError;
    fn try_from(vec: Vec<f32>) -> Result<Self, Self::Error> {
        if vec.len() != Self::TAG_CARDINALITY {
            return Err(TiffError::CardinalityError {
                name: "DefaultScale",
                actual: vec.len(),
                expected: Self::TAG_CARDINALITY,
            });
        }
        Ok((vec[0], vec[1]).into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, RgbaImage};
    use rstest::rstest;
    use std::fs::File;
    use tempfile::tempdir;
    use tiff::encoder::TiffEncoder;

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

    #[rstest]
    #[case(Resize { scale_x: 2.0, scale_y: 2.0, filter: FilterType::Nearest })]
    fn test_write_tags(#[case] resize: Resize) {
        // Prepare the TIFF
        let temp_dir = tempdir().unwrap();
        let temp_file_path = temp_dir.path().join("temp.tif");
        let mut tiff = TiffEncoder::new(File::create(temp_file_path.clone()).unwrap()).unwrap();
        let mut img = tiff
            .new_image::<tiff::encoder::colortype::Gray16>(1, 1)
            .unwrap();

        // Write the tags
        resize.write_tags(&mut img).unwrap();

        // Write some dummy image data
        let data: Vec<u16> = vec![0; 2];
        img.write_data(data.as_slice()).unwrap();

        // Read the TIFF back
        let mut tiff = Decoder::new(File::open(temp_file_path).unwrap()).unwrap();
        let actual = Resize::try_from(&mut tiff).unwrap();
        assert_eq!(resize, actual);
    }
}
