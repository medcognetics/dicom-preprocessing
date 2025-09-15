use image::DynamicImage;
use image::GenericImageView;
use std::fs::File;
use std::io::{BufWriter, Seek, Write};
use std::path::Path;
use tiff::encoder::colortype::ColorType;
use tiff::encoder::compression::Compressor;
use tiff::encoder::Compression;
use tiff::encoder::TiffEncoder;

use snafu::ResultExt;
use tiff::encoder::colortype::{Gray16, Gray8, RGB8};
// Removed unused compression imports

use crate::color::DicomColorType;
use crate::errors::{
    tiff::{IOSnafu, WriteSnafu},
    TiffError,
};
use crate::metadata::{PreprocessingMetadata, WriteTags};

// Helper function to convert Compressor to Compression
fn compressor_to_compression(compressor: &Compressor) -> Compression {
    match compressor {
        Compressor::Uncompressed(_) => Compression::Uncompressed,
        Compressor::Lzw(_) => Compression::Lzw,
        Compressor::Deflate(_) => Compression::Deflate(tiff::encoder::DeflateLevel::default()),
        Compressor::Packbits(_) => Compression::Packbits,
        _ => Compression::Uncompressed, // Default fallback for non-exhaustive enum
    }
}

// Trait for saving a DynamicImage to a TIFF via a TiffEncoder under a given color type
pub trait SaveToTiff<C>
where
    C: ColorType,
{
    fn save<T: Write + Seek>(
        &self,
        encoder: &mut TiffEncoder<T>,
        image: &DynamicImage,
        metadata: &PreprocessingMetadata,
    ) -> Result<(), TiffError>;
}

pub struct TiffSaver {
    compressor: Compressor,
    color: DicomColorType,
}

/// Implement supported combinations of color type
macro_rules! impl_save_frame {
    ($color_type:ty, $as_fn:ident) => {
        impl SaveToTiff<$color_type> for TiffSaver {
            fn save<T: Write + Seek>(
                &self,
                encoder: &mut TiffEncoder<T>,
                image: &DynamicImage,
                metadata: &PreprocessingMetadata,
            ) -> Result<(), TiffError> {
                let (columns, rows) = image.dimensions();
                let mut tiff = encoder.new_image::<$color_type>(columns, rows)?;

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

// Implementations for each color type
impl_save_frame!(Gray16, as_luma16);
impl_save_frame!(RGB8, as_rgb8);
impl_save_frame!(Gray8, as_luma8);

impl TiffSaver {
    pub fn new(compressor: Compressor, color: DicomColorType) -> Self {
        Self { compressor, color }
    }

    pub fn open_tiff<P: AsRef<Path>>(
        &self,
        output: P,
    ) -> Result<TiffEncoder<BufWriter<File>>, TiffError> {
        let output = output.as_ref();
        let file = File::create(output).context(IOSnafu { path: output })?;
        let file = BufWriter::new(file);
        let encoder = TiffEncoder::new(file).context(WriteSnafu { path: output })?;
        Ok(encoder.with_compression(compressor_to_compression(&self.compressor)))
    }

    pub fn save<T: Write + Seek>(
        &self,
        encoder: &mut TiffEncoder<T>,
        image: &DynamicImage,
        metadata: &PreprocessingMetadata,
    ) -> Result<(), TiffError> {
        match &self.color {
            DicomColorType::Gray16(_) => {
                <TiffSaver as SaveToTiff<Gray16>>::save(self, encoder, image, metadata)
            }
            DicomColorType::RGB8(_) => {
                <TiffSaver as SaveToTiff<RGB8>>::save(self, encoder, image, metadata)
            }
            DicomColorType::Gray8(_) => {
                <TiffSaver as SaveToTiff<Gray8>>::save(self, encoder, image, metadata)
            }
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
            num_frames: 1_u16.into(),
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
