use crate::metadata::FrameCount;
use dicom::dictionary_std::tags;
use dicom::object::{FileDicomObject, InMemDicomObject};
use image::DynamicImage;
use image::GenericImageView;
use num::PrimInt;
use snafu::{ResultExt, Snafu};
use std::io::{Read, Seek};
use tiff::decoder::Decoder;

use crate::color::DicomColorType;
use crate::errors::{dicom::ConvertValueSnafu, DicomError, TiffError};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum DimensionOrder {
    #[default]
    NHWC,
    NCHW,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dimensions {
    pub width: usize,
    pub height: usize,
    pub num_frames: usize,
    pub channels: usize,
    pub order: DimensionOrder,
}

impl Dimensions {
    pub fn new(width: usize, height: usize, num_frames: usize, channels: usize) -> Self {
        Self {
            width,
            height,
            num_frames,
            channels,
            order: DimensionOrder::default(),
        }
    }

    pub fn with_num_frames(self, num_frames: usize) -> Self {
        Self { num_frames, ..self }
    }

    pub fn with_order(self, order: DimensionOrder) -> Self {
        Self { order, ..self }
    }

    pub fn shape(&self) -> (usize, usize, usize, usize) {
        match self.order {
            DimensionOrder::NHWC => (self.num_frames, self.height, self.width, self.channels),
            DimensionOrder::NCHW => (self.num_frames, self.channels, self.height, self.width),
        }
    }

    pub fn frame_shape(&self) -> (usize, usize, usize) {
        match self.order {
            DimensionOrder::NHWC => (self.height, self.width, self.channels),
            DimensionOrder::NCHW => (self.channels, self.height, self.width),
        }
    }

    pub fn numel(&self) -> usize {
        self.num_frames * self.height * self.width * self.channels
    }

    pub fn size<T: PrimInt>(&self) -> usize {
        self.numel() * std::mem::size_of::<T>()
    }
}

impl<T: PrimInt> From<(T, T, T, T)> for Dimensions {
    /// Returns the shape of the entire volume
    fn from((num_frames, height, width, channels): (T, T, T, T)) -> Self {
        Self {
            num_frames: num_frames.to_usize().unwrap(),
            height: height.to_usize().unwrap(),
            width: width.to_usize().unwrap(),
            channels: channels.to_usize().unwrap(),
            order: DimensionOrder::default(),
        }
    }
}

impl<R: Read + Seek> TryFrom<&mut Decoder<R>> for Dimensions {
    type Error = TiffError;
    /// Extract dimensions from a TIFF decoder. Note that if the TIFF was not preprocessed,
    /// this will require traversing the IFD chain to get the number of frames.
    fn try_from(decoder: &mut Decoder<R>) -> Result<Self, Self::Error> {
        let (width, height) = decoder.dimensions()?;
        let color_type = decoder.colortype()?;
        let color_type = DicomColorType::try_from(color_type)?;
        let num_frames: u16 = FrameCount::try_from(decoder)?.into();
        Ok(Self {
            width: width as usize,
            height: height as usize,
            num_frames: num_frames as usize,
            channels: color_type.channels(),
            order: DimensionOrder::default(),
        })
    }
}

impl TryFrom<&FileDicomObject<InMemDicomObject>> for Dimensions {
    type Error = DicomError;
    fn try_from(dcm: &FileDicomObject<InMemDicomObject>) -> Result<Self, Self::Error> {
        let num_frames = FrameCount::try_from(dcm)?.into();
        let rows = dcm
            .get(tags::ROWS)
            .ok_or(DicomError::MissingPropertyError { name: "Rows" })?
            .value()
            .to_int::<i32>()
            .context(ConvertValueSnafu { name: "Rows" })? as usize;
        let columns = dcm
            .get(tags::COLUMNS)
            .ok_or(DicomError::MissingPropertyError { name: "Columns" })?
            .value()
            .to_int::<i32>()
            .context(ConvertValueSnafu { name: "Columns" })? as usize;
        let color_type = DicomColorType::try_from(dcm)?;
        Ok(Self {
            num_frames,
            height: rows,
            width: columns,
            channels: color_type.channels(),
            order: DimensionOrder::default(),
        })
    }
}

impl From<&DynamicImage> for Dimensions {
    fn from(img: &DynamicImage) -> Self {
        let (width, height) = img.dimensions();
        let channels = img.color().channel_count();
        Self {
            num_frames: 1,
            height: height as usize,
            width: width as usize,
            channels: channels as usize,
            order: DimensionOrder::default(),
        }
    }
}

#[derive(Debug, Snafu)]
pub enum DynamicImageError {
    #[snafu(display("No images provided"))]
    NoImages,
    #[snafu(display("Dimension mismatch: first {expected:?}, other {actual:?}"))]
    DimensionMismatch {
        expected: Dimensions,
        actual: Dimensions,
    },
}

impl TryFrom<&Vec<DynamicImage>> for Dimensions {
    type Error = DynamicImageError;

    /// Get dimensions from first image and verify all images have same dimensions
    fn try_from(imgs: &Vec<DynamicImage>) -> Result<Self, Self::Error> {
        let dims: Dimensions = imgs.first().ok_or(DynamicImageError::NoImages)?.into();
        for img in imgs {
            let other_dims: Dimensions = img.into();
            if other_dims != dims {
                return Err(DynamicImageError::DimensionMismatch {
                    expected: dims,
                    actual: other_dims,
                });
            }
        }
        Ok(dims.with_num_frames(imgs.len()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[test]
    fn test_dimension_order_default() {
        assert_eq!(DimensionOrder::default(), DimensionOrder::NHWC);
    }

    #[test]
    fn test_with_num_frames() {
        let dims = Dimensions::new(3, 4, 1, 2);
        let new_dims = dims.with_num_frames(5);
        assert_eq!(new_dims.num_frames, 5);
        assert_eq!(new_dims.width, dims.width);
        assert_eq!(new_dims.height, dims.height);
        assert_eq!(new_dims.channels, dims.channels);
        assert_eq!(new_dims.order, dims.order);
    }

    #[rstest]
    #[case(1, 2, 3, 4, DimensionOrder::NHWC, (1, 2, 3, 4))]
    #[case(1, 2, 3, 4, DimensionOrder::NCHW, (1, 4, 2, 3))]
    fn test_shape(
        #[case] n: usize,
        #[case] h: usize,
        #[case] w: usize,
        #[case] c: usize,
        #[case] order: DimensionOrder,
        #[case] expected: (usize, usize, usize, usize),
    ) {
        let dims = Dimensions::new(w, h, n, c).with_order(order);
        assert_eq!(dims.shape(), expected);
    }

    #[rstest]
    #[case(1, 2, 3, 4, DimensionOrder::NHWC, (2, 3, 4))]
    #[case(1, 2, 3, 4, DimensionOrder::NCHW, (4, 2, 3))]
    fn test_frame_shape(
        #[case] n: usize,
        #[case] h: usize,
        #[case] w: usize,
        #[case] c: usize,
        #[case] order: DimensionOrder,
        #[case] expected: (usize, usize, usize),
    ) {
        let dims = Dimensions::new(w, h, n, c).with_order(order);
        assert_eq!(dims.frame_shape(), expected);
    }

    #[rstest]
    #[case(2, 3, 4, 5, 120)]
    #[case(1, 1, 1, 1, 1)]
    fn test_numel(
        #[case] n: usize,
        #[case] h: usize,
        #[case] w: usize,
        #[case] c: usize,
        #[case] expected: usize,
    ) {
        let dims = Dimensions::new(w, h, n, c);
        assert_eq!(dims.numel(), expected);
    }

    #[rstest]
    #[case(2, 3, 4, 5, 480)] // 120 elements * 4 bytes for i32
    fn test_size(
        #[case] n: usize,
        #[case] h: usize,
        #[case] w: usize,
        #[case] c: usize,
        #[case] expected: usize,
    ) {
        let dims = Dimensions::new(w, h, n, c);
        assert_eq!(dims.size::<i32>(), expected);
    }

    #[test]
    fn test_from_tuple() {
        let dims = Dimensions::from((1_i32, 2_i32, 3_i32, 4_i32));
        assert_eq!(dims.num_frames, 1);
        assert_eq!(dims.height, 2);
        assert_eq!(dims.width, 3);
        assert_eq!(dims.channels, 4);
        assert_eq!(dims.order, DimensionOrder::default());
    }

    #[test]
    fn test_dynamic_image_vec_empty() {
        let imgs: Vec<DynamicImage> = vec![];
        assert!(matches!(
            Dimensions::try_from(&imgs),
            Err(DynamicImageError::NoImages)
        ));
    }

    #[test]
    fn test_dynamic_image_vec_mismatch() {
        use image::{ImageBuffer, Rgb};

        let img1 = DynamicImage::ImageRgb8(ImageBuffer::<Rgb<u8>, Vec<u8>>::new(10, 20));
        let img2 = DynamicImage::ImageRgb8(ImageBuffer::<Rgb<u8>, Vec<u8>>::new(20, 10));
        let imgs = vec![img1, img2];

        assert!(matches!(
            Dimensions::try_from(&imgs),
            Err(DynamicImageError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_dynamic_image_vec_success() {
        use image::{ImageBuffer, Rgb};

        let img1 = DynamicImage::ImageRgb8(ImageBuffer::<Rgb<u8>, Vec<u8>>::new(10, 20));
        let img2 = DynamicImage::ImageRgb8(ImageBuffer::<Rgb<u8>, Vec<u8>>::new(10, 20));
        let imgs = vec![img1, img2];

        let dims = Dimensions::try_from(&imgs).unwrap();
        assert_eq!(dims.width, 10);
        assert_eq!(dims.height, 20);
        assert_eq!(dims.channels, 3);
        assert_eq!(dims.num_frames, 2);
    }
}
