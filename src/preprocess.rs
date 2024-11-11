use image::DynamicImage;
use image::GenericImageView;
use std::fs::File;
use std::io::{Seek, Write};
use std::path::PathBuf;
use tiff::encoder::colortype::ColorType;
use tiff::encoder::compression::{Compression, Compressor};
use tiff::encoder::{ImageEncoder, TiffEncoder, TiffKind};
use tiff::tags::Tag;
use tiff::TiffError;

use dicom::dictionary_std::tags;
use dicom::object::{FileDicomObject, InMemDicomObject, ReadError};
use dicom::pixeldata::PhotometricInterpretation;
use image::imageops::FilterType;
use snafu::{ResultExt, Snafu};
use tiff::encoder::colortype::{Gray16, RGB8};
use tiff::encoder::compression::{Lzw, Packbits, Uncompressed};

use crate::metadata::{Resolution, WriteTags};
use crate::transform::volume::VolumeError;
use crate::transform::{
    Crop, HandleVolume, Padding, PaddingDirection, Resize, Transform, VolumeHandler,
};

const VERSION: &str = concat!("dicom-preprocessing==", env!("CARGO_PKG_VERSION"), "\0");

#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(display("could not read DICOM file {}", path.display()))]
    ReadFile {
        #[snafu(source(from(ReadError, Box::new)))]
        source: Box<ReadError>,
        path: PathBuf,
    },
    /// failed to decode pixel data
    DecodePixelData {
        #[snafu(source(from(VolumeError, Box::new)))]
        source: Box<VolumeError>,
    },
    /// missing offset table entry for frame #{frame_number}
    MissingOffsetEntry {
        frame_number: u32,
    },
    /// missing key property {name}
    MissingProperty {
        name: &'static str,
    },
    /// property {name} contains an invalid value
    InvalidPropertyValue {
        name: &'static str,
        #[snafu(source(from(dicom::core::value::ConvertValueError, Box::new)))]
        source: Box<dicom::core::value::ConvertValueError>,
    },
    CastPropertyValue {
        name: &'static str,
        #[snafu(source(from(dicom::core::value::CastValueError, Box::new)))]
        source: Box<dicom::core::value::CastValueError>,
    },
    /// pixel data of frame #{frame_number} is out of bounds
    FrameOutOfBounds {
        frame_number: u32,
    },
    ConvertImage {
        #[snafu(source(from(dicom::pixeldata::Error, Box::new)))]
        source: Box<dicom::pixeldata::Error>,
    },
    #[snafu(display("could not open TIFF file {}", path.display()))]
    OpenTiff {
        #[snafu(source(from(TiffError, Box::new)))]
        source: Box<TiffError>,
        path: PathBuf,
    },
    WriteToTiff {
        #[snafu(source(from(TiffError, Box::new)))]
        source: Box<TiffError>,
    },
    ConvertImageToBytes,
    SaveData {
        source: std::io::Error,
    },
    UnexpectedPixelData,
    WriteTags {
        #[snafu(source(from(TiffError, Box::new)))]
        source: Box<TiffError>,
    },
    #[snafu(display("unsupported photometric interpretation {photometric_interpretation} for {bits_allocated} bits allocated"))]
    UnsupportedPhotometricInterpretation {
        bits_allocated: u16,
        photometric_interpretation: PhotometricInterpretation,
    },
}

pub struct PreprocessingMetadata {
    crop: Option<Crop>,
    resize: Option<Resize>,
    padding: Option<Padding>,
    resolution: Option<Resolution>,
}

impl WriteTags for PreprocessingMetadata {
    fn write_tags<W, C, K, D>(&self, tiff: &mut ImageEncoder<W, C, K, D>) -> Result<(), TiffError>
    where
        W: Write + Seek,
        C: ColorType,
        K: TiffKind,
        D: Compression,
    {
        // Write the resolution tag
        if let Some(resolution) = &self.resolution {
            resolution.write_tags(tiff)?;
        }
        // Write metadata software tag
        tiff.encoder()
            .write_tag(Tag::Software, VERSION.as_bytes())?;

        // Write transform related tags
        if let Some(resolution) = &self.resolution {
            resolution.write_tags(tiff)?;
        }
        if let Some(crop_config) = &self.crop {
            crop_config.write_tags(tiff)?;
        }
        if let Some(resize_config) = &self.resize {
            resize_config.write_tags(tiff)?;
        }
        if let Some(padding_config) = &self.padding {
            padding_config.write_tags(tiff)?;
        }
        Ok(())
    }
}

pub struct Preprocessor {
    pub crop: bool,
    pub size: Option<(u32, u32)>,
    pub filter: FilterType,
    pub padding_direction: PaddingDirection,
    pub compressor: Compressor,
    pub crop_max: bool,
    pub volume_handler: VolumeHandler,
}

impl Preprocessor {
    pub fn preprocess(
        &self,
        file: &FileDicomObject<InMemDicomObject>,
        output: PathBuf,
    ) -> Result<(), Error> {
        let bits_allocated = file
            .get(tags::BITS_ALLOCATED)
            .ok_or(Error::MissingProperty {
                name: "Bits Allocated",
            })?
            .value()
            .uint16()
            .context(CastPropertyValueSnafu {
                name: "Bits Allocated",
            })?;
        let photometric_interpretation = file
            .get(tags::PHOTOMETRIC_INTERPRETATION)
            .ok_or(Error::MissingProperty {
                name: "Photometric Interpretation",
            })?
            .value()
            .string()
            .context(CastPropertyValueSnafu {
                name: "Photometric Interpretation",
            })?;
        let photometric_interpretation =
            PhotometricInterpretation::from(photometric_interpretation.trim());

        let (frames, metadata) = self.prepare_image(file)?;

        // Use the metadata to determine the correct implementation of Preprocess
        match (bits_allocated, photometric_interpretation, &self.compressor) {
            (16, PhotometricInterpretation::Monochrome1, Compressor::Uncompressed(_)) => {
                <Preprocessor as Preprocess<Gray16, Uncompressed>>::preprocess_frames(
                    self, frames, metadata, output,
                )
            }
            (16, PhotometricInterpretation::Monochrome2, Compressor::Uncompressed(_)) => {
                <Preprocessor as Preprocess<Gray16, Uncompressed>>::preprocess_frames(
                    self, frames, metadata, output,
                )
            }
            (16, PhotometricInterpretation::Monochrome1, Compressor::Packbits(_)) => {
                <Preprocessor as Preprocess<Gray16, Packbits>>::preprocess_frames(
                    self, frames, metadata, output,
                )
            }
            (16, PhotometricInterpretation::Monochrome2, Compressor::Packbits(_)) => {
                <Preprocessor as Preprocess<Gray16, Packbits>>::preprocess_frames(
                    self, frames, metadata, output,
                )
            }
            (16, PhotometricInterpretation::Monochrome1, Compressor::Lzw(_)) => {
                <Preprocessor as Preprocess<Gray16, Lzw>>::preprocess_frames(
                    self, frames, metadata, output,
                )
            }
            (16, PhotometricInterpretation::Monochrome2, Compressor::Lzw(_)) => {
                <Preprocessor as Preprocess<Gray16, Lzw>>::preprocess_frames(
                    self, frames, metadata, output,
                )
            }
            (8, PhotometricInterpretation::Rgb, Compressor::Uncompressed(_)) => {
                <Preprocessor as Preprocess<RGB8, Uncompressed>>::preprocess_frames(
                    self, frames, metadata, output,
                )
            }
            (8, PhotometricInterpretation::Rgb, Compressor::Packbits(_)) => {
                <Preprocessor as Preprocess<RGB8, Packbits>>::preprocess_frames(
                    self, frames, metadata, output,
                )
            }
            (8, PhotometricInterpretation::Rgb, Compressor::Lzw(_)) => {
                <Preprocessor as Preprocess<RGB8, Lzw>>::preprocess_frames(
                    self, frames, metadata, output,
                )
            }
            (bits_allocated, photometric_interpretation, _) => {
                Err(Error::UnsupportedPhotometricInterpretation {
                    bits_allocated,
                    photometric_interpretation,
                })
            }
        }
    }

    // Decodes the pixel data and applies transformations
    fn prepare_image(
        &self,
        file: &FileDicomObject<InMemDicomObject>,
    ) -> Result<(Vec<DynamicImage>, PreprocessingMetadata), Error> {
        // Decode the pixel data, applying volume handling
        let image_data = self
            .volume_handler
            .decode_volume(file)
            .context(DecodePixelDataSnafu)?;

        // Try to determine the resolution from pixel spacing attributes
        let resolution = Resolution::try_from(file).ok();

        // Determine and apply crop
        let crop_config = match self.crop {
            true => Some(Crop::new_from_images(
                &image_data.iter().collect::<Vec<_>>(),
                self.crop_max,
            )),
            false => None,
        };
        let image_data = image_data
            .into_iter()
            .map(|img| match &crop_config {
                Some(config) => config.apply(&img),
                None => img,
            })
            .collect::<Vec<_>>();

        // Determine and apply resize, ensuring we also update the resolution
        let resize_config = match self.size {
            Some((target_width, target_height)) => {
                let first_image = image_data.first().unwrap();
                let config = Resize::new(&first_image, target_width, target_height, self.filter);
                Some(config)
            }
            None => None,
        };
        let image_data = image_data
            .into_iter()
            .map(|img| match &resize_config {
                Some(config) => config.apply(&img),
                None => img,
            })
            .collect::<Vec<_>>();
        let resolution = match (resolution, &resize_config) {
            (Some(res), Some(config)) => Some(config.apply(&res)),
            _ => None,
        };

        // Determine and apply padding
        let padding_config = match self.size {
            Some((target_width, target_height)) => {
                let first_image = image_data.first().unwrap();
                let config = Padding::new(
                    &first_image,
                    target_width,
                    target_height,
                    self.padding_direction,
                );
                Some(config)
            }
            None => None,
        };
        let image_data = image_data
            .into_iter()
            .map(|img| match &padding_config {
                Some(config) => config.apply(&img),
                None => img,
            })
            .collect::<Vec<_>>();

        Ok((
            image_data,
            PreprocessingMetadata {
                crop: crop_config,
                resize: resize_config,
                padding: padding_config,
                resolution,
            },
        ))
    }
}

trait Preprocess<C, D>
where
    C: ColorType,
    D: Compression,
{
    fn preprocess_frame(
        &self,
        encoder: &mut TiffEncoder<File>,
        image: &DynamicImage,
        metadata: &PreprocessingMetadata,
    ) -> Result<(), Error>;

    fn preprocess_frames(
        &self,
        frames: Vec<DynamicImage>,
        metadata: PreprocessingMetadata,
        output: PathBuf,
    ) -> Result<(), Error> {
        // Open the TIFF file
        let mut tiff_encoder = TiffEncoder::new(File::create(&output).context(SaveDataSnafu)?)
            .context(OpenTiffSnafu {
                path: output.clone(),
            })?;

        for img in frames.iter() {
            self.preprocess_frame(&mut tiff_encoder, img, &metadata)?;
        }
        Ok(())
    }
}

macro_rules! impl_preprocess_frame {
    ($color_type:ty, $compression:ty, $as_fn:ident, $error_variant:ident) => {
        impl Preprocess<$color_type, $compression> for Preprocessor {
            fn preprocess_frame(
                &self,
                encoder: &mut TiffEncoder<File>,
                image: &DynamicImage,
                metadata: &PreprocessingMetadata,
            ) -> Result<(), Error> {
                let (columns, rows) = image.dimensions();
                let mut tiff = encoder
                    .new_image_with_compression::<$color_type, _>(
                        columns,
                        rows,
                        <$compression>::default(),
                    )
                    .context(WriteToTiffSnafu)?;

                metadata
                    .write_tags(&mut tiff)
                    .map_err(|e| Error::WriteTags {
                        source: Box::new(e),
                    })?;
                let bytes = image.$as_fn().ok_or(Error::$error_variant)?;
                tiff.write_data(bytes).context(WriteToTiffSnafu)?;
                Ok(())
            }
        }
    };
}

// Implementations for Gray16
impl_preprocess_frame!(Gray16, Uncompressed, as_luma16, ConvertImageToBytes);
impl_preprocess_frame!(Gray16, Packbits, as_luma16, ConvertImageToBytes);
impl_preprocess_frame!(Gray16, Lzw, as_luma16, ConvertImageToBytes);

// Implementations for RGB8
impl_preprocess_frame!(RGB8, Uncompressed, as_rgb8, ConvertImageToBytes);
impl_preprocess_frame!(RGB8, Packbits, as_rgb8, ConvertImageToBytes);
impl_preprocess_frame!(RGB8, Lzw, as_rgb8, ConvertImageToBytes);
