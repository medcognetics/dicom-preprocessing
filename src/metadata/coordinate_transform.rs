use snafu::Snafu;

use crate::metadata::PreprocessingMetadata;
use crate::transform::Flip;

/// A two-dimensional image size in pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PixelDimensions {
    pub width: u32,
    pub height: u32,
}

impl PixelDimensions {
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }
}

/// A half-open rectangle in pixel-edge coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PixelRect {
    pub left: u32,
    pub top: u32,
    pub width: u32,
    pub height: u32,
}

impl PixelRect {
    pub const fn new(left: u32, top: u32, width: u32, height: u32) -> Self {
        Self {
            left,
            top,
            width,
            height,
        }
    }
}

/// A serializable affine coordinate transform for one rendered image.
///
/// Matrices use Canvas order `[a, b, c, d, e, f]`, where
/// `x' = a*x + c*y + e` and `y' = b*x + d*y + f`. Integer coordinates
/// identify pixel centers. Resize transforms use center-aligned sampling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoordinateTransform {
    pub source_dimensions: PixelDimensions,
    pub display_dimensions: PixelDimensions,
    pub source_to_display: [f64; 6],
    pub display_to_source: [f64; 6],
    pub valid_source_rect: PixelRect,
    pub valid_display_rect: PixelRect,
}

impl CoordinateTransform {
    pub fn identity(width: u32, height: u32) -> Result<Self, CoordinateTransformError> {
        let dimensions = PixelDimensions::new(width, height);
        validate_dimensions(dimensions)?;
        Ok(Self {
            source_dimensions: dimensions,
            display_dimensions: dimensions,
            source_to_display: Affine2d::IDENTITY.0,
            display_to_source: Affine2d::IDENTITY.0,
            valid_source_rect: PixelRect::new(0, 0, width, height),
            valid_display_rect: PixelRect::new(0, 0, width, height),
        })
    }

    pub fn from_preprocessing(
        source_dimensions: PixelDimensions,
        metadata: &PreprocessingMetadata,
    ) -> Result<Self, CoordinateTransformError> {
        validate_dimensions(source_dimensions)?;

        let mut source_to_display = Affine2d::IDENTITY;
        if let Some(flip) = metadata.flip {
            if flip.width != source_dimensions.width || flip.height != source_dimensions.height {
                return InvalidFlipDimensionsSnafu {
                    source_width: source_dimensions.width,
                    source_height: source_dimensions.height,
                    flip_width: flip.width,
                    flip_height: flip.height,
                }
                .fail();
            }
            source_to_display = source_to_display.then(Affine2d::flip(
                source_dimensions,
                flip.horizontal,
                flip.vertical,
            ));
        }

        let (content_source_rect, valid_source_rect) = if let Some(crop) = metadata.crop {
            validate_crop(
                source_dimensions,
                crop.left,
                crop.top,
                crop.width,
                crop.height,
            )?;
            source_to_display = source_to_display.then(Affine2d::translation(
                -f64::from(crop.left),
                -f64::from(crop.top),
            ));
            let source_left = if metadata.flip.is_some_and(|flip| flip.horizontal) {
                source_dimensions.width - crop.left - crop.width
            } else {
                crop.left
            };
            let source_top = if metadata.flip.is_some_and(|flip| flip.vertical) {
                source_dimensions.height - crop.top - crop.height
            } else {
                crop.top
            };
            (
                PixelDimensions::new(crop.width, crop.height),
                PixelRect::new(source_left, source_top, crop.width, crop.height),
            )
        } else {
            (
                source_dimensions,
                PixelRect::new(0, 0, source_dimensions.width, source_dimensions.height),
            )
        };

        let resized_dimensions = if let Some(resize) = metadata.resize {
            validate_scale(resize.scale_x, resize.scale_y)?;
            let width = scaled_dimension(content_source_rect.width, resize.scale_x)?;
            let height = scaled_dimension(content_source_rect.height, resize.scale_y)?;
            let scale_x = f64::from(width) / f64::from(content_source_rect.width);
            let scale_y = f64::from(height) / f64::from(content_source_rect.height);
            source_to_display = source_to_display.then(Affine2d::centered_scale(scale_x, scale_y));
            PixelDimensions::new(width, height)
        } else {
            content_source_rect
        };

        let (display_dimensions, valid_display_rect) = if let Some(padding) = metadata.padding {
            source_to_display = source_to_display.then(Affine2d::translation(
                f64::from(padding.left),
                f64::from(padding.top),
            ));
            let width = resized_dimensions
                .width
                .checked_add(padding.left)
                .and_then(|width| width.checked_add(padding.right))
                .ok_or(CoordinateTransformError::DimensionOverflow)?;
            let height = resized_dimensions
                .height
                .checked_add(padding.top)
                .and_then(|height| height.checked_add(padding.bottom))
                .ok_or(CoordinateTransformError::DimensionOverflow)?;
            (
                PixelDimensions::new(width, height),
                PixelRect::new(
                    padding.left,
                    padding.top,
                    resized_dimensions.width,
                    resized_dimensions.height,
                ),
            )
        } else {
            (
                resized_dimensions,
                PixelRect::new(0, 0, resized_dimensions.width, resized_dimensions.height),
            )
        };

        let display_to_source = source_to_display
            .inverse()
            .ok_or(CoordinateTransformError::NonInvertible)?;
        Ok(Self {
            source_dimensions,
            display_dimensions,
            source_to_display: source_to_display.0,
            display_to_source: display_to_source.0,
            valid_source_rect,
            valid_display_rect,
        })
    }

    pub fn from_flip(
        source_dimensions: PixelDimensions,
        flip: Option<Flip>,
    ) -> Result<Self, CoordinateTransformError> {
        Self::from_preprocessing(
            source_dimensions,
            &PreprocessingMetadata {
                flip,
                crop: None,
                resize: None,
                padding: None,
                resolution: None,
                num_frames: 1_u16.into(),
            },
        )
    }

    pub fn map_source_to_display(&self, x: f64, y: f64) -> (f64, f64) {
        Affine2d(self.source_to_display).apply(x, y)
    }

    pub fn map_display_to_source(&self, x: f64, y: f64) -> (f64, f64) {
        Affine2d(self.display_to_source).apply(x, y)
    }
}

#[derive(Debug, Snafu, PartialEq)]
pub enum CoordinateTransformError {
    #[snafu(display("source dimensions must be non-zero, received {width}x{height}"))]
    InvalidSourceDimensions { width: u32, height: u32 },

    #[snafu(display(
        "flip dimensions {flip_width}x{flip_height} do not match source dimensions {source_width}x{source_height}"
    ))]
    InvalidFlipDimensions {
        source_width: u32,
        source_height: u32,
        flip_width: u32,
        flip_height: u32,
    },

    #[snafu(display(
        "crop rectangle ({left}, {top}, {width}, {height}) is outside source dimensions {source_width}x{source_height}"
    ))]
    InvalidCrop {
        left: u32,
        top: u32,
        width: u32,
        height: u32,
        source_width: u32,
        source_height: u32,
    },

    #[snafu(display("resize scales must be finite and positive, received {scale_x}x{scale_y}"))]
    InvalidResizeScale { scale_x: f32, scale_y: f32 },

    #[snafu(display("coordinate transform dimensions overflow u32"))]
    DimensionOverflow,

    #[snafu(display("coordinate transform is not invertible"))]
    NonInvertible,
}

#[derive(Debug, Clone, Copy)]
struct Affine2d([f64; 6]);

impl Affine2d {
    const IDENTITY: Self = Self([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);

    fn translation(x: f64, y: f64) -> Self {
        Self([1.0, 0.0, 0.0, 1.0, x, y])
    }

    fn centered_scale(x: f64, y: f64) -> Self {
        Self([x, 0.0, 0.0, y, (x - 1.0) / 2.0, (y - 1.0) / 2.0])
    }

    fn flip(dimensions: PixelDimensions, horizontal: bool, vertical: bool) -> Self {
        Self([
            if horizontal { -1.0 } else { 1.0 },
            0.0,
            0.0,
            if vertical { -1.0 } else { 1.0 },
            if horizontal {
                f64::from(dimensions.width - 1)
            } else {
                0.0
            },
            if vertical {
                f64::from(dimensions.height - 1)
            } else {
                0.0
            },
        ])
    }

    /// Compose `next` after `self`.
    fn then(self, next: Self) -> Self {
        let [a, b, c, d, e, f] = self.0;
        let [next_a, next_b, next_c, next_d, next_e, next_f] = next.0;
        Self([
            next_a * a + next_c * b,
            next_b * a + next_d * b,
            next_a * c + next_c * d,
            next_b * c + next_d * d,
            next_a * e + next_c * f + next_e,
            next_b * e + next_d * f + next_f,
        ])
    }

    fn inverse(self) -> Option<Self> {
        let [a, b, c, d, e, f] = self.0;
        let determinant = a * d - b * c;
        if !determinant.is_finite() || determinant == 0.0 {
            return None;
        }
        Some(
            Self([
                d / determinant,
                -b / determinant,
                -c / determinant,
                a / determinant,
                (c * f - d * e) / determinant,
                (b * e - a * f) / determinant,
            ])
            .normalize_zero(),
        )
    }

    fn apply(self, x: f64, y: f64) -> (f64, f64) {
        let [a, b, c, d, e, f] = self.0;
        (a * x + c * y + e, b * x + d * y + f)
    }

    fn normalize_zero(mut self) -> Self {
        for value in &mut self.0 {
            if *value == 0.0 {
                *value = 0.0;
            }
        }
        self
    }
}

fn validate_dimensions(dimensions: PixelDimensions) -> Result<(), CoordinateTransformError> {
    if dimensions.width == 0 || dimensions.height == 0 {
        return InvalidSourceDimensionsSnafu {
            width: dimensions.width,
            height: dimensions.height,
        }
        .fail();
    }
    Ok(())
}

fn validate_crop(
    source: PixelDimensions,
    left: u32,
    top: u32,
    width: u32,
    height: u32,
) -> Result<(), CoordinateTransformError> {
    let valid = width > 0
        && height > 0
        && left
            .checked_add(width)
            .is_some_and(|right| right <= source.width)
        && top
            .checked_add(height)
            .is_some_and(|bottom| bottom <= source.height);
    if !valid {
        return InvalidCropSnafu {
            left,
            top,
            width,
            height,
            source_width: source.width,
            source_height: source.height,
        }
        .fail();
    }
    Ok(())
}

fn validate_scale(scale_x: f32, scale_y: f32) -> Result<(), CoordinateTransformError> {
    if !scale_x.is_finite() || !scale_y.is_finite() || scale_x <= 0.0 || scale_y <= 0.0 {
        return InvalidResizeScaleSnafu { scale_x, scale_y }.fail();
    }
    Ok(())
}

fn scaled_dimension(source: u32, scale: f32) -> Result<u32, CoordinateTransformError> {
    let scaled = (source as f64 * f64::from(scale)).round();
    if !(1.0..=f64::from(u32::MAX)).contains(&scaled) {
        return DimensionOverflowSnafu.fail();
    }
    Ok(scaled as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::{Crop, FilterType, Flip, Padding, Resize};
    use rstest::rstest;

    const WIDTH: u32 = 100;
    const HEIGHT: u32 = 80;

    fn metadata_with_flip(flip: Option<Flip>) -> PreprocessingMetadata {
        PreprocessingMetadata {
            flip,
            crop: None,
            resize: None,
            padding: None,
            resolution: None,
            num_frames: 1_u16.into(),
        }
    }

    fn assert_point_close(actual: (f64, f64), expected: (f64, f64)) {
        const TOLERANCE: f64 = 1e-9;
        assert!((actual.0 - expected.0).abs() < TOLERANCE);
        assert!((actual.1 - expected.1).abs() < TOLERANCE);
    }

    #[rstest]
    #[case(false, false, (0.0, 0.0), (0.0, 0.0))]
    #[case(true, false, (0.0, 0.0), (99.0, 0.0))]
    #[case(false, true, (0.0, 0.0), (0.0, 79.0))]
    #[case(true, true, (99.0, 79.0), (0.0, 0.0))]
    fn flips_map_integer_pixel_centers_exactly(
        #[case] horizontal: bool,
        #[case] vertical: bool,
        #[case] source: (f64, f64),
        #[case] expected: (f64, f64),
    ) {
        let metadata = metadata_with_flip(
            (horizontal || vertical).then(|| Flip::new(WIDTH, HEIGHT, horizontal, vertical)),
        );
        let transform =
            CoordinateTransform::from_preprocessing(PixelDimensions::new(WIDTH, HEIGHT), &metadata)
                .unwrap();

        assert_point_close(
            transform.map_source_to_display(source.0, source.1),
            expected,
        );
        assert_point_close(
            transform.map_display_to_source(expected.0, expected.1),
            source,
        );
    }

    #[test]
    fn composed_transform_round_trips_and_reports_valid_bounds() {
        let metadata = PreprocessingMetadata {
            flip: Some(Flip::new(WIDTH, HEIGHT, true, false)),
            crop: Some(Crop {
                left: 70,
                top: 20,
                width: 20,
                height: 20,
            }),
            resize: Some(Resize {
                scale_x: 2.0,
                scale_y: 2.0,
                filter: FilterType::Nearest,
            }),
            padding: Some(Padding {
                left: 5,
                top: 7,
                right: 1,
                bottom: 1,
            }),
            resolution: None,
            num_frames: 1_u16.into(),
        };

        let transform =
            CoordinateTransform::from_preprocessing(PixelDimensions::new(WIDTH, HEIGHT), &metadata)
                .unwrap();

        assert_eq!(transform.display_dimensions, PixelDimensions::new(46, 48));
        assert_eq!(transform.valid_source_rect, PixelRect::new(10, 20, 20, 20));
        assert_eq!(transform.valid_display_rect, PixelRect::new(5, 7, 40, 40));
        let display = transform.map_source_to_display(10.0, 20.0);
        assert_point_close(display, (43.5, 7.5));
        assert_point_close(
            transform.map_display_to_source(display.0, display.1),
            (10.0, 20.0),
        );
    }
}
