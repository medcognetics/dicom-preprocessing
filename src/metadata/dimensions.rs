use crate::metadata::FrameCount;
use dicom::dictionary_std::tags;
use dicom::object::{FileDicomObject, InMemDicomObject};
use image::DynamicImage;
use ndarray::{Dim, Shape};
use num::PrimInt;
use snafu::{ResultExt, Snafu};
use std::io::{Read, Seek};
use tiff::decoder::Decoder;

use crate::color::DicomColorType;
use crate::errors::{
    dicom::{CastValueSnafu, ConvertValueSnafu, MissingPropertySnafu},
    DicomError, TiffError,
};

#[derive(Clone, Copy, Debug)]
pub enum ChannelOrder {
    NHWC,
    NCHW,
}

#[derive(Clone, Copy, Debug)]
pub struct Dimensions {
    width: usize,
    height: usize,
    num_frames: usize,
    channels: usize,
    channel_order: ChannelOrder,
}

impl Dimensions {
    fn new(
        width: usize,
        height: usize,
        num_frames: usize,
        channels: usize,
        channel_order: ChannelOrder,
    ) -> Self {
        Self {
            width,
            height,
            num_frames,
            channels,
            channel_order,
        }
    }

    fn with_num_frames(self, num_frames: usize) -> Self {
        Self { num_frames, ..self }
    }

    fn with_channel_order(self, channel_order: ChannelOrder) -> Self {
        Self {
            channel_order,
            ..self
        }
    }
}

impl<T: PrimInt> Into<(T, T, T, T)> for Dimensions {
    /// Returns the shape of the entire volume
    fn into(self) -> (T, T, T, T) {
        match self.channel_order {
            ChannelOrder::NHWC => (
                T::from(self.num_frames).unwrap(),
                T::from(self.height).unwrap(),
                T::from(self.width).unwrap(),
                T::from(self.channels).unwrap(),
            ),
            ChannelOrder::NCHW => (
                T::from(self.num_frames).unwrap(),
                T::from(self.channels).unwrap(),
                T::from(self.height).unwrap(),
                T::from(self.width).unwrap(),
            ),
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
            channel_order: ChannelOrder::NHWC,
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
            channel_order: ChannelOrder::NHWC,
        })
    }
}

impl TryFrom<&DynamicImage> for Dimensions {
    type Error = LoadError;
    fn try_from(dcm: &FileDicomObject<InMemDicomObject>) -> Result<Self, Self::Error> {
        let num_frames = FrameCount::try_from(dcm).context(VolumeSnafu)?.into();
        let rows = dcm
            .get(tags::ROWS)
            .ok_or(LoadError::MissingProperty { name: "Rows" })?
            .value()
            .to_int::<i32>()
            .context(InvalidPropertyValueSnafu { name: "Rows" })? as usize;
        let columns = dcm
            .get(tags::COLUMNS)
            .ok_or(LoadError::MissingProperty { name: "Columns" })?
            .value()
            .to_int::<i32>()
            .context(InvalidPropertyValueSnafu { name: "Columns" })? as usize;
        let color_type = DicomColorType::try_from(dcm).context(ColorTypeSnafu)?;
        Ok(Self {
            num_frames,
            height: rows,
            width: columns,
            channels: color_type.channels(),
            channel_order: ChannelOrder::NHWC,
        })
    }
}
