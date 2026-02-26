use crate::errors::tiff::TiffError;
use std::io::{Seek, Write};
use tiff::encoder::colortype::ColorType;
use tiff::encoder::ImageEncoder;
use tiff::encoder::TiffKind;

pub mod resolution;
pub use resolution::*;

pub mod preprocessing;
pub use preprocessing::*;

pub mod dimensions;
pub use dimensions::*;

pub mod bot;
pub use bot::*;

pub trait WriteTags {
    /// Write tags describing the transform to a TIFF encoder.
    fn write_tags<W, C, K>(&self, tiff: &mut ImageEncoder<W, C, K>) -> Result<(), TiffError>
    where
        W: Write + Seek,
        C: ColorType,
        K: TiffKind;
}
