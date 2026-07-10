use crate::{Dimensions, LoadFromTiff, TiffError};
use image::DynamicImage;
use ndarray::Array4;
use num::Zero;
use tiff::decoder::DecodingResult;

#[derive(Debug, thiserror::Error)]
pub enum ImageArrayError {
    #[error("cannot convert an empty image stack to an array")]
    Empty,

    #[error("image frame {frame} has dimensions {actual:?}, expected {expected:?}")]
    DimensionMismatch {
        frame: usize,
        expected: Dimensions,
        actual: Dimensions,
    },

    #[error("image frame {frame} has unsupported color type {color_type:?}")]
    UnsupportedColorType {
        frame: usize,
        color_type: image::ColorType,
    },

    #[error("image frame {frame} uses a different sample type from the first frame")]
    MixedSampleTypes { frame: usize },

    #[error("failed to convert image samples: {0}")]
    SampleConversion(#[from] TiffError),

    #[error("failed to construct image array: {0}")]
    Shape(#[from] ndarray::ShapeError),
}

enum ImageSamples {
    U8(Vec<u8>),
    U16(Vec<u16>),
}

/// Convert a homogeneous image stack into an NHWC array without TIFF serialization.
///
/// Supported image variants are `Luma8`, `Luma16`, and `Rgb8`. Numeric conversion
/// delegates to [`LoadFromTiff`] so direct arrays have the same dtype semantics as
/// explicit TIFF loading.
pub fn images_to_array<T>(images: Vec<DynamicImage>) -> Result<Array4<T>, ImageArrayError>
where
    T: Clone + Zero,
    Array4<T>: LoadFromTiff<T>,
{
    let first = images.first().ok_or(ImageArrayError::Empty)?;
    let frame_dimensions = Dimensions::from(first);
    for (frame, image) in images.iter().enumerate().skip(1) {
        let actual = Dimensions::from(image);
        if actual != frame_dimensions {
            return Err(ImageArrayError::DimensionMismatch {
                frame,
                expected: frame_dimensions,
                actual,
            });
        }
    }

    let dimensions = frame_dimensions.with_num_frames(images.len());
    let mut samples = match first {
        DynamicImage::ImageLuma8(_) | DynamicImage::ImageRgb8(_) => {
            ImageSamples::U8(Vec::with_capacity(dimensions.numel()))
        }
        DynamicImage::ImageLuma16(_) => ImageSamples::U16(Vec::with_capacity(dimensions.numel())),
        _ => {
            return Err(ImageArrayError::UnsupportedColorType {
                frame: 0,
                color_type: first.color(),
            })
        }
    };

    for (frame, image) in images.into_iter().enumerate() {
        match (&mut samples, image) {
            (ImageSamples::U8(samples), DynamicImage::ImageLuma8(image)) => {
                samples.extend(image.into_raw());
            }
            (ImageSamples::U8(samples), DynamicImage::ImageRgb8(image)) => {
                samples.extend(image.into_raw());
            }
            (ImageSamples::U16(samples), DynamicImage::ImageLuma16(image)) => {
                samples.extend(image.into_raw());
            }
            (_, image @ (DynamicImage::ImageLuma8(_) | DynamicImage::ImageRgb8(_)))
            | (_, image @ DynamicImage::ImageLuma16(_)) => {
                let _ = image;
                return Err(ImageArrayError::MixedSampleTypes { frame });
            }
            (_, image) => {
                return Err(ImageArrayError::UnsupportedColorType {
                    frame,
                    color_type: image.color(),
                })
            }
        }
    }

    let decoded = match samples {
        ImageSamples::U8(samples) => DecodingResult::U8(samples),
        ImageSamples::U16(samples) => DecodingResult::U16(samples),
    };
    let samples = Array4::<T>::to_vec(decoded)?;
    Ok(Array4::from_shape_vec(dimensions.shape(), samples)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DicomColorType, FrameCount, PreprocessingMetadata, TiffSaver};
    use image::{ImageBuffer, Luma, Rgb, Rgba};
    use std::fmt::Debug;
    use std::io::{Cursor, Seek, SeekFrom};
    use tiff::decoder::Decoder;
    use tiff::encoder::compression::{Compressor, Uncompressed};
    use tiff::encoder::TiffEncoder;

    fn metadata(frame_count: usize) -> PreprocessingMetadata {
        PreprocessingMetadata {
            flip: None,
            crop: None,
            resize: None,
            padding: None,
            resolution: None,
            num_frames: FrameCount::from(frame_count),
        }
    }

    fn tiff_round_trip<T>(images: &[DynamicImage], color_type: DicomColorType) -> Array4<T>
    where
        T: Clone + Zero,
        Array4<T>: LoadFromTiff<T>,
    {
        let saver = TiffSaver::new(Compressor::Uncompressed(Uncompressed), color_type);
        let mut buffer = Cursor::new(Vec::new());
        {
            let mut encoder = TiffEncoder::new(&mut buffer).unwrap();
            let metadata = metadata(images.len());
            for image in images {
                saver.save(&mut encoder, image, &metadata).unwrap();
            }
        }
        buffer.seek(SeekFrom::Start(0)).unwrap();
        let mut decoder = Decoder::new(buffer).unwrap();
        Array4::<T>::decode(&mut decoder).unwrap()
    }

    fn assert_tiff_parity<T>(images: &[DynamicImage], color_type: DicomColorType)
    where
        T: Clone + Zero + PartialEq + Debug,
        Array4<T>: LoadFromTiff<T>,
    {
        let expected = tiff_round_trip::<T>(images, color_type);
        let actual = images_to_array::<T>(images.to_vec()).unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn direct_arrays_match_tiff_round_trip_for_all_supported_types() {
        let luma8 = (0..2)
            .map(|frame| {
                DynamicImage::ImageLuma8(ImageBuffer::from_fn(3, 2, |x, y| {
                    Luma([(x + y * 3 + frame * 11) as u8])
                }))
            })
            .collect::<Vec<_>>();
        let luma16 = (0..2)
            .map(|frame| {
                DynamicImage::ImageLuma16(ImageBuffer::from_fn(3, 2, |x, y| {
                    Luma([((x + y * 3 + frame * 11) * 257) as u16])
                }))
            })
            .collect::<Vec<_>>();
        let rgb8 = (0..2)
            .map(|frame| {
                DynamicImage::ImageRgb8(ImageBuffer::from_fn(3, 2, |x, y| {
                    let value = (x + y * 3 + frame * 11) as u8;
                    Rgb([value, value.wrapping_add(1), value.wrapping_add(2)])
                }))
            })
            .collect::<Vec<_>>();

        assert_tiff_parity::<u8>(
            &luma8,
            DicomColorType::Gray8(tiff::encoder::colortype::Gray8),
        );
        assert_tiff_parity::<u16>(
            &luma8,
            DicomColorType::Gray8(tiff::encoder::colortype::Gray8),
        );
        assert_tiff_parity::<f32>(
            &luma8,
            DicomColorType::Gray8(tiff::encoder::colortype::Gray8),
        );
        assert_tiff_parity::<u8>(
            &luma16,
            DicomColorType::Gray16(tiff::encoder::colortype::Gray16),
        );
        assert_tiff_parity::<u16>(
            &luma16,
            DicomColorType::Gray16(tiff::encoder::colortype::Gray16),
        );
        assert_tiff_parity::<f32>(
            &luma16,
            DicomColorType::Gray16(tiff::encoder::colortype::Gray16),
        );
        assert_tiff_parity::<u8>(&rgb8, DicomColorType::RGB8(tiff::encoder::colortype::RGB8));
        assert_tiff_parity::<u16>(&rgb8, DicomColorType::RGB8(tiff::encoder::colortype::RGB8));
        assert_tiff_parity::<f32>(&rgb8, DicomColorType::RGB8(tiff::encoder::colortype::RGB8));
    }

    #[test]
    fn direct_array_rejects_invalid_stacks() {
        assert!(matches!(
            images_to_array::<u8>(vec![]),
            Err(ImageArrayError::Empty)
        ));

        let mismatched = vec![
            DynamicImage::ImageLuma8(ImageBuffer::new(2, 2)),
            DynamicImage::ImageLuma8(ImageBuffer::new(3, 2)),
        ];
        assert!(matches!(
            images_to_array::<u8>(mismatched),
            Err(ImageArrayError::DimensionMismatch { frame: 1, .. })
        ));

        let mixed_sample_types = vec![
            DynamicImage::ImageLuma8(ImageBuffer::new(2, 2)),
            DynamicImage::ImageLuma16(ImageBuffer::new(2, 2)),
        ];
        assert!(matches!(
            images_to_array::<u8>(mixed_sample_types),
            Err(ImageArrayError::MixedSampleTypes { frame: 1 })
        ));

        let unsupported = vec![DynamicImage::ImageRgba8(ImageBuffer::from_pixel(
            2,
            2,
            Rgba([0, 0, 0, u8::MAX]),
        ))];
        assert!(matches!(
            images_to_array::<u8>(unsupported),
            Err(ImageArrayError::UnsupportedColorType { frame: 0, .. })
        ));
    }
}
