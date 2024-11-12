use image::DynamicImage;
use image::GenericImageView;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use tiff::encoder::colortype::ColorType;
use tiff::encoder::compression::{Compression, Compressor, Deflate, Packbits};
use tiff::encoder::TiffEncoder;
use tiff::TiffError;

use dicom::dictionary_std::tags;
use dicom::object::{FileDicomObject, InMemDicomObject, ReadError};
use dicom::pixeldata::PhotometricInterpretation;
use image::imageops::FilterType;
use snafu::{ResultExt, Snafu};
use tiff::encoder::colortype::{Gray16, RGB8};
use tiff::encoder::compression::{Lzw, Uncompressed};

use crate::color::{ColorError, DicomColorType};
use crate::metadata::{PreprocessingMetadata, Resolution, WriteTags};
use crate::transform::volume::VolumeError;
use crate::transform::{
    Crop, HandleVolume, Padding, PaddingDirection, Resize, Transform, VolumeHandler,
};

#[derive(Debug, Snafu)]
pub enum SaveError {
    MissingProperty {
        name: &'static str,
    },
    CastPropertyValue {
        name: &'static str,
        #[snafu(source(from(dicom::core::value::CastValueError, Box::new)))]
        source: Box<dicom::core::value::CastValueError>,
    },
    #[snafu(display("could not open TIFF file {}", path.display()))]
    CreateFile {
        #[snafu(source(from(std::io::Error, Box::new)))]
        source: Box<std::io::Error>,
        path: PathBuf,
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
    WriteTags {
        #[snafu(source(from(TiffError, Box::new)))]
        source: Box<TiffError>,
    },
}

// Trait for saving a DynamicImage to a TIFF via a TiffEncoder under a given color type and compression
pub trait SaveToTiff<C, D>
where
    C: ColorType,
    D: Compression + Clone,
{
    fn save(
        &self,
        encoder: &mut TiffEncoder<File>,
        image: &DynamicImage,
        metadata: &PreprocessingMetadata,
        compression: D,
    ) -> Result<(), SaveError>;

    fn save_all(
        &self,
        frames: Vec<DynamicImage>,
        metadata: PreprocessingMetadata,
        output: PathBuf,
        compression: D,
    ) -> Result<(), SaveError> {
        // Open the TIFF file
        let file = File::create(&output).context(CreateFileSnafu {
            path: output.clone(),
        })?;
        let mut tiff_encoder = TiffEncoder::new(file).context(OpenTiffSnafu {
            path: output.clone(),
        })?;

        for img in frames.iter() {
            self.save(&mut tiff_encoder, img, &metadata, compression.clone())?;
        }
        Ok(())
    }
}

pub struct TiffSaver {
    compressor: Compressor,
    color: DicomColorType,
}

/// Implement supported combinations of color type and compression
macro_rules! impl_save_frame {
    ($color_type:ty, $compression:ty, $as_fn:ident) => {
        impl SaveToTiff<$color_type, $compression> for TiffSaver {
            fn save(
                &self,
                encoder: &mut TiffEncoder<File>,
                image: &DynamicImage,
                metadata: &PreprocessingMetadata,
                compression: $compression,
            ) -> Result<(), SaveError> {
                let (columns, rows) = image.dimensions();
                let mut tiff = encoder
                    .new_image_with_compression::<$color_type, _>(columns, rows, compression)
                    .context(WriteToTiffSnafu)?;

                metadata
                    .write_tags(&mut tiff)
                    .map_err(|e| SaveError::WriteTags {
                        source: Box::new(e),
                    })?;
                let bytes = image.$as_fn().ok_or(SaveError::ConvertImageToBytes)?;
                tiff.write_data(bytes).context(WriteToTiffSnafu)?;
                Ok(())
            }
        }
    };
}

// Implementations for Gray16
impl_save_frame!(Gray16, Uncompressed, as_luma16);
impl_save_frame!(Gray16, Packbits, as_luma16);
impl_save_frame!(Gray16, Lzw, as_luma16);
impl_save_frame!(Gray16, Deflate, as_luma16);

// Implementations for RGB8
impl_save_frame!(RGB8, Uncompressed, as_rgb8);
impl_save_frame!(RGB8, Packbits, as_rgb8);
impl_save_frame!(RGB8, Lzw, as_rgb8);
impl_save_frame!(RGB8, Deflate, as_rgb8);

impl TiffSaver {
    pub fn new(compressor: Compressor, color: DicomColorType) -> Self {
        Self { compressor, color }
    }

    pub fn save(
        &self,
        encoder: &mut TiffEncoder<File>,
        image: &DynamicImage,
        metadata: &PreprocessingMetadata,
    ) -> Result<(), SaveError> {
        match (&self.color, &self.compressor) {
            (DicomColorType::Gray16(_), Compressor::Uncompressed(c)) => {
                <TiffSaver as SaveToTiff<Gray16, Uncompressed>>::save(
                    self, encoder, image, metadata, *c,
                )
            }
            (DicomColorType::Gray16(_), Compressor::Packbits(c)) => {
                <TiffSaver as SaveToTiff<Gray16, Packbits>>::save(
                    self, encoder, image, metadata, *c,
                )
            }
            (DicomColorType::Gray16(_), Compressor::Lzw(c)) => {
                <TiffSaver as SaveToTiff<Gray16, Lzw>>::save(self, encoder, image, metadata, *c)
            }
            (DicomColorType::Gray16(_), Compressor::Deflate(c)) => {
                <TiffSaver as SaveToTiff<Gray16, Deflate>>::save(self, encoder, image, metadata, *c)
            }
            (DicomColorType::RGB8(_), Compressor::Uncompressed(c)) => {
                <TiffSaver as SaveToTiff<RGB8, Uncompressed>>::save(
                    self, encoder, image, metadata, *c,
                )
            }
            (DicomColorType::RGB8(_), Compressor::Packbits(c)) => {
                <TiffSaver as SaveToTiff<RGB8, Packbits>>::save(self, encoder, image, metadata, *c)
            }
            (DicomColorType::RGB8(_), Compressor::Lzw(c)) => {
                <TiffSaver as SaveToTiff<RGB8, Lzw>>::save(self, encoder, image, metadata, *c)
            }
            (DicomColorType::RGB8(_), Compressor::Deflate(c)) => {
                <TiffSaver as SaveToTiff<RGB8, Deflate>>::save(self, encoder, image, metadata, *c)
            }
        }
    }

    pub fn save_all(
        &self,
        frames: Vec<DynamicImage>,
        metadata: PreprocessingMetadata,
        output: &PathBuf,
    ) -> Result<(), SaveError> {
        // Open the TIFF file
        let file = File::create(&output).context(CreateFileSnafu {
            path: output.clone(),
        })?;
        let mut tiff_encoder = TiffEncoder::new(file).context(OpenTiffSnafu {
            path: output.clone(),
        })?;

        for img in frames.iter() {
            self.save(&mut tiff_encoder, img, &metadata)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{SaveToTiff, TiffSaver};
    use crate::color::DicomColorType;
    use crate::metadata::PreprocessingMetadata;
    use crate::DisplayFilterType;
    use crate::PaddingDirection;
    use dicom::dictionary_std::tags;
    use dicom::object::open_file;
    use dicom_pixeldata::PixelDecoder;
    use rstest::rstest;
    use std::fs::File;
    use std::io::BufReader;
    use tiff::decoder::Decoder;
    use tiff::encoder::compression::{Compressor, Uncompressed};

    use tiff::tags::ResolutionUnit;
    use tiff::tags::Tag;

    #[rstest]
    #[case("pydicom/CT_small.dcm", Compressor::Uncompressed(Uncompressed))]
    fn test_save(#[case] dicom_file: &str, #[case] compressor: Compressor) {
        // Open the DICOM file
        let dicom_file = dicom_test_files::path(dicom_file).unwrap();
        let dicom_file = open_file(&dicom_file).unwrap();

        // Decode DICOM to DynamicImage
        let image = dicom_file
            .decode_pixel_data()
            .unwrap()
            .to_dynamic_image(0)
            .unwrap();

        // Determine the color type and create a saver with the given compressor
        let color = DicomColorType::try_from(&dicom_file).unwrap();
        let saver = TiffSaver::new(compressor, color);

        // Create a temp directory for the TIFF output
        let temp_dir = tempfile::tempdir().unwrap();
        let temp_path = temp_dir.path().join("output.tiff");

        // Create dummy metadata
        let metadata = PreprocessingMetadata {
            crop: None,
            resize: None,
            padding: None,
            resolution: None,
        };

        saver.save_all(vec![image], metadata, &temp_path).unwrap();

        // Check the output file exists
        assert!(temp_path.exists());
    }
}
