use image::imageops;
use image::{DynamicImage, GenericImage, GenericImageView, Pixel};
use std::fmt;
use std::io::{Read, Seek, Write};
use tiff::decoder::Decoder;
use tiff::encoder::colortype::ColorType;
use tiff::encoder::ImageEncoder;
use tiff::encoder::TiffKind;
use tiff::tags::Tag;

use crate::errors::tiff::TiffError;
use crate::metadata::{Resolution, WriteTags};
use crate::transform::{Coord, InvertibleTransform, Transform};

pub const DEFAULT_SCALE: u16 = 50718;

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum FilterType {
    #[default]
    Triangle,
    Nearest,
    CatmullRom,
    Gaussian,
    Lanczos3,
    MaxPool,
}

impl fmt::Display for FilterType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let filter_str = match self {
            FilterType::Triangle => "triangle",
            FilterType::Nearest => "nearest",
            FilterType::CatmullRom => "catmull-rom",
            FilterType::Gaussian => "gaussian",
            FilterType::Lanczos3 => "lanczos3",
            FilterType::MaxPool => "maxpool",
        };
        write!(f, "{filter_str}")
    }
}

impl From<FilterType> for imageops::FilterType {
    fn from(filter: FilterType) -> Self {
        match filter {
            FilterType::Nearest => imageops::FilterType::Nearest,
            FilterType::Triangle => imageops::FilterType::Triangle,
            FilterType::CatmullRom => imageops::FilterType::CatmullRom,
            FilterType::Gaussian => imageops::FilterType::Gaussian,
            FilterType::Lanczos3 => imageops::FilterType::Lanczos3,
            FilterType::MaxPool => imageops::FilterType::Nearest, // Default for image crate compatibility
        }
    }
}

impl From<imageops::FilterType> for FilterType {
    fn from(filter: imageops::FilterType) -> Self {
        match filter {
            imageops::FilterType::Nearest => FilterType::Nearest,
            imageops::FilterType::Triangle => FilterType::Triangle,
            imageops::FilterType::CatmullRom => FilterType::CatmullRom,
            imageops::FilterType::Gaussian => FilterType::Gaussian,
            imageops::FilterType::Lanczos3 => FilterType::Lanczos3,
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

        // Special handling for MaxPool
        if self.filter == FilterType::MaxPool {
            let mut output = DynamicImage::new(target_width, target_height, image.color());

            for y in 0..target_height {
                for x in 0..target_width {
                    let start_x = (x * width / target_width).min(width - 1);
                    let start_y = (y * height / target_height).min(height - 1);
                    let end_x = ((x + 1) * width / target_width).min(width);
                    let end_y = ((y + 1) * height / target_height).min(height);

                    let mut max_pixel = image.get_pixel(start_x, start_y);

                    for ky in start_y..end_y {
                        for kx in start_x..end_x {
                            let pixel = image.get_pixel(kx, ky);
                            max_pixel.apply2(&pixel, |a, b| a.max(b));
                        }
                    }

                    output.put_pixel(x, y, max_pixel);
                }
            }
            output
        } else {
            image.resize(target_width, target_height, self.filter.into())
        }
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

impl InvertibleTransform<Resolution> for Resize {
    fn invert(&self, resolution: &Resolution) -> Resolution {
        let scale = 1.0 / self.scale_x;
        resolution.scale(scale)
    }
}

impl Transform<Coord> for Resize {
    fn apply(&self, coord: &Coord) -> Coord {
        let (x, y): (u32, u32) = coord.into();
        let new_x = (x as f32 * self.scale_x) as u32;
        let new_y = (y as f32 * self.scale_y) as u32;
        Coord::new(new_x, new_y)
    }
}

impl InvertibleTransform<Coord> for Resize {
    fn invert(&self, coord: &Coord) -> Coord {
        let (x, y): (u32, u32) = coord.into();
        let new_x = (x as f32 / self.scale_x) as u32;
        let new_y = (y as f32 / self.scale_y) as u32;
        Coord::new(new_x, new_y)
    }
}

impl WriteTags for Resize {
    fn write_tags<W, C, K>(&self, tiff: &mut ImageEncoder<W, C, K>) -> Result<(), TiffError>
    where
        W: Write + Seek,
        C: ColorType,
        K: TiffKind,
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
        FilterType::Nearest,
        vec![
            vec![1, 2],
            vec![3, 4],
        ],
    )]
    #[case(
        vec![
            vec![1, 2, 3, 4],
            vec![5, 6, 7, 8],
            vec![9, 10, 11, 12],
            vec![13, 14, 15, 16],
        ],
        (2, 2),
        FilterType::MaxPool,
        vec![
            vec![6, 8],
            vec![14, 16],
        ],
    )]
    fn test_resize(
        #[case] pixels: Vec<Vec<u8>>,
        #[case] target_size: (u32, u32),
        #[case] filter: FilterType,
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

        let resize = Resize::new(&dynamic_image, target_size.0, target_size.1, filter);
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

    #[rstest]
    #[case(
        Resize { scale_x: 2.0, scale_y: 2.0, filter: FilterType::Nearest },
        Coord::new(5, 5),
        Coord::new(10, 10)
    )]
    #[case(
        Resize { scale_x: 0.5, scale_y: 0.5, filter: FilterType::Nearest },
        Coord::new(10, 10),
        Coord::new(5, 5)
    )]
    #[case(
        Resize { scale_x: 1.5, scale_y: 1.5, filter: FilterType::Nearest },
        Coord::new(4, 6),
        Coord::new(6, 9)
    )]
    fn test_apply_coord(#[case] resize: Resize, #[case] coord: Coord, #[case] expected: Coord) {
        let result = resize.apply(&coord);
        assert_eq!(result, expected);
    }

    #[rstest]
    #[case(
        Resize { scale_x: 2.0, scale_y: 2.0, filter: FilterType::Nearest },
        Coord::new(10, 10),
        Coord::new(5, 5)
    )]
    #[case(
        Resize { scale_x: 0.5, scale_y: 0.5, filter: FilterType::Nearest },
        Coord::new(5, 5),
        Coord::new(10, 10)
    )]
    #[case(
        Resize { scale_x: 1.5, scale_y: 1.5, filter: FilterType::Nearest },
        Coord::new(6, 9),
        Coord::new(4, 6)
    )]
    fn test_invert_coord(#[case] resize: Resize, #[case] coord: Coord, #[case] expected: Coord) {
        let result = resize.invert(&coord);
        assert_eq!(result, expected);
    }

    #[rstest]
    #[case(
        Resize { scale_x: 2.0, scale_y: 2.0, filter: FilterType::Nearest },
        Resolution::new(300.0, 300.0),
        Resolution::new(600.0, 600.0)
    )]
    #[case(
        Resize { scale_x: 0.5, scale_y: 0.5, filter: FilterType::Nearest },
        Resolution::new(300.0, 300.0),
        Resolution::new(150.0, 150.0)
    )]
    #[case(
        Resize { scale_x: 1.5, scale_y: 1.5, filter: FilterType::Nearest },
        Resolution::new(200.0, 200.0),
        Resolution::new(300.0, 300.0)
    )]
    fn test_apply_resolution(
        #[case] resize: Resize,
        #[case] resolution: Resolution,
        #[case] expected: Resolution,
    ) {
        let result = resize.apply(&resolution);
        assert_eq!(result, expected);
    }

    #[rstest]
    #[case(
        Resize { scale_x: 2.0, scale_y: 2.0, filter: FilterType::Nearest },
        Resolution::new(600.0, 600.0),
        Resolution::new(300.0, 300.0)
    )]
    #[case(
        Resize { scale_x: 0.5, scale_y: 0.5, filter: FilterType::Nearest },
        Resolution::new(150.0, 150.0),
        Resolution::new(300.0, 300.0)
    )]
    #[case(
        Resize { scale_x: 1.5, scale_y: 1.5, filter: FilterType::Nearest },
        Resolution::new(300.0, 300.0),
        Resolution::new(200.0, 200.0)
    )]
    fn test_invert_resolution(
        #[case] resize: Resize,
        #[case] resolution: Resolution,
        #[case] expected: Resolution,
    ) {
        let result = resize.invert(&resolution);
        assert_eq!(result, expected);
    }
}
