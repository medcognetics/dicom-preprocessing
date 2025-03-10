use image::DynamicImage;
use image::GenericImageView;
use std::fs::File;
use std::io::{BufWriter, Seek, Write};
use std::path::Path;
use tiff::encoder::colortype::ColorType;
use tiff::encoder::compression::{Compression, Compressor, Deflate, Packbits};
use tiff::encoder::TiffEncoder;

use snafu::ResultExt;
use tiff::encoder::colortype::{Gray16, Gray8, RGB8};
use tiff::encoder::compression::{Lzw, Uncompressed};

use crate::color::DicomColorType;
use crate::errors::{
    tiff::{IOSnafu, WriteSnafu},
    TiffError,
};
use crate::metadata::{PreprocessingMetadata, WriteTags};

// Trait for saving a DynamicImage to a TIFF via a TiffEncoder under a given color type and compression
pub trait SaveToTiff<C, D>
where
    C: ColorType,
    D: Compression + Clone,
{
    fn save<T: Write + Seek>(
        &self,
        encoder: &mut TiffEncoder<T>,
        image: &DynamicImage,
        metadata: &PreprocessingMetadata,
        compression: D,
    ) -> Result<(), TiffError>;
}

pub struct TiffSaver {
    compressor: Compressor,
    color: DicomColorType,
}

/// Implement supported combinations of color type and compression
macro_rules! impl_save_frame {
    ($color_type:ty, $compression:ty, $as_fn:ident) => {
        impl SaveToTiff<$color_type, $compression> for TiffSaver {
            fn save<T: Write + Seek>(
                &self,
                encoder: &mut TiffEncoder<T>,
                image: &DynamicImage,
                metadata: &PreprocessingMetadata,
                compression: $compression,
            ) -> Result<(), TiffError> {
                let (columns, rows) = image.dimensions();
                let mut tiff = encoder.new_image_with_compression::<$color_type, _>(
                    columns,
                    rows,
                    compression,
                )?;

                metadata.write_tags(&mut tiff)?;
                let bytes = image.$as_fn().ok_or(TiffError::DynamicImageError {
                    color_type: image.color(),
                })?;
                tiff.write_data(bytes)?;
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

// Implementations for Gray8
impl_save_frame!(Gray8, Uncompressed, as_luma8);
impl_save_frame!(Gray8, Packbits, as_luma8);
impl_save_frame!(Gray8, Lzw, as_luma8);
impl_save_frame!(Gray8, Deflate, as_luma8);

impl TiffSaver {
    pub fn new(compressor: Compressor, color: DicomColorType) -> Self {
        Self { compressor, color }
    }

    pub fn open_tiff<P: AsRef<Path>>(
        &self,
        output: P,
    ) -> Result<TiffEncoder<BufWriter<File>>, TiffError> {
        let output = output.as_ref();
        let file = File::create(&output).context(IOSnafu { path: output })?;
        let file = BufWriter::new(file);
        TiffEncoder::new(file).context(WriteSnafu { path: output })
    }

    pub fn save<T: Write + Seek>(
        &self,
        encoder: &mut TiffEncoder<T>,
        image: &DynamicImage,
        metadata: &PreprocessingMetadata,
    ) -> Result<(), TiffError> {
        macro_rules! save_with {
            ($color_type:ty, $compression:ty, $compressor:expr) => {
                <TiffSaver as SaveToTiff<$color_type, $compression>>::save(
                    self,
                    encoder,
                    image,
                    metadata,
                    $compressor,
                )
            };
        }

        match (&self.color, &self.compressor) {
            // Monochrome 16-bit
            (DicomColorType::Gray16(_), Compressor::Uncompressed(c)) => {
                save_with!(Gray16, Uncompressed, *c)
            }
            (DicomColorType::Gray16(_), Compressor::Packbits(c)) => {
                save_with!(Gray16, Packbits, *c)
            }
            (DicomColorType::Gray16(_), Compressor::Lzw(c)) => save_with!(Gray16, Lzw, *c),
            (DicomColorType::Gray16(_), Compressor::Deflate(c)) => save_with!(Gray16, Deflate, *c),
            // RGB 8-bit
            (DicomColorType::RGB8(_), Compressor::Uncompressed(c)) => {
                save_with!(RGB8, Uncompressed, *c)
            }
            (DicomColorType::RGB8(_), Compressor::Packbits(c)) => save_with!(RGB8, Packbits, *c),
            (DicomColorType::RGB8(_), Compressor::Lzw(c)) => save_with!(RGB8, Lzw, *c),
            (DicomColorType::RGB8(_), Compressor::Deflate(c)) => save_with!(RGB8, Deflate, *c),
            // Monochrome 8-bit
            (DicomColorType::Gray8(_), Compressor::Uncompressed(c)) => {
                save_with!(Gray8, Uncompressed, *c)
            }
            (DicomColorType::Gray8(_), Compressor::Packbits(c)) => save_with!(Gray8, Packbits, *c),
            (DicomColorType::Gray8(_), Compressor::Lzw(c)) => save_with!(Gray8, Lzw, *c),
            (DicomColorType::Gray8(_), Compressor::Deflate(c)) => save_with!(Gray8, Deflate, *c),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::TiffSaver;
    use crate::color::DicomColorType;
    use crate::metadata::PreprocessingMetadata;

    use dicom::object::open_file;
    use dicom_pixeldata::PixelDecoder;
    use rstest::rstest;

    use tiff::encoder::compression::{Compressor, Uncompressed};

    #[rstest]
    #[case("pydicom/CT_small.dcm", Compressor::Uncompressed(Uncompressed))]
    #[case("pydicom/MR_small.dcm", Compressor::Uncompressed(Uncompressed))]
    #[case("pydicom/JPEG2000_UNC.dcm", Compressor::Uncompressed(Uncompressed))]
    // 8-bit monochrome
    #[case(
        "pydicom/JPGLosslessP14SV1_1s_1f_8b.dcm",
        Compressor::Uncompressed(Uncompressed)
    )]
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
            num_frames: (1 as u16).into(),
        };

        let mut encoder = saver.open_tiff(temp_path.clone()).unwrap();
        let image_vec = vec![image];
        image_vec
            .into_iter()
            .try_for_each(|image| saver.save(&mut encoder, &image, &metadata))
            .unwrap();

        // Check the output file exists
        assert!(temp_path.exists());
    }
}
