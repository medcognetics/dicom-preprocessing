use image::DynamicImage;
use image::GenericImageView;
use std::fs::File;
use std::io::{Read, Seek, BufReader};
use tiff::decoder::{Decoder, DecodingResult};
use std::path::PathBuf;
use tiff::encoder::colortype::ColorType;
use tiff::encoder::compression::{Compression, Compressor, Deflate, Packbits};
use tiff::encoder::TiffEncoder;
use tiff::TiffError;

use snafu::{ResultExt, Snafu};
use tiff::encoder::colortype::{Gray16, RGB8};
use tiff::encoder::compression::{Lzw, Uncompressed};

use crate::color::DicomColorType;
use crate::metadata::{PreprocessingMetadata, WriteTags};
use ndarray::{Array, Array4};

#[derive(Debug, Snafu)]
pub enum LoadError {
    ParseColorType {
        #[snafu(source(from(TiffError, Box::new)))]
        source: Box<TiffError>,
    },
    CastPropertyValue {
        name: &'static str,
        #[snafu(source(from(dicom::core::value::CastValueError, Box::new)))]
        source: Box<dicom::core::value::CastValueError>,
    },
    #[snafu(display("could not open TIFF file {}", path.display()))]
    OpenTiff {
        #[snafu(source(from(std::io::Error, Box::new)))]
        source: Box<std::io::Error>,
        path: PathBuf,
    },
    #[snafu(display("could not read TIFF file {}", path.display()))]
    ReadTiff {
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

pub enum ChannelOrder {
    First,
    Last,
}


struct LoadGray8;
struct LoadGray16;
struct LoadRgb8;
struct LoadNormalizedF32;


trait Load<T> 
where T: Clone + num::Zero
{
    const CHANNEL_ORDER: ChannelOrder;
    const CHANNELS: usize;

    fn load(path: PathBuf) -> Result<Array4<T>, LoadError> {
        let file = BufReader::new(File::open(path).context(OpenTiffSnafu { path })?);
        let mut decoder = Decoder::new(file).unwrap();

        // Determine height, width, number of frames, and color type
        let (width, height) = decoder.dimensions().unwrap();
        let color_type = decoder.colortype().unwrap();
        let mut frames: usize = 0;
        while decoder.mo() {
            frames += 1;
            decoder.next_image().unwrap();
        }

        // Pre-allocate array
        let mut image_data = Array4::<T>::zeros((frames, height as usize, width as usize, Self::CHANNELS));

        // Reset decoder to first frame
        let mut decoder = Decoder::new(file).unwrap();




        let foo = decoder.read_image().unwrap();
        let foo = DynamicImage::from_decoder(decoder).unwrap();

        foo


    }


}



impl Load<f32> for LoadNormalizedF32 {
    fn load(path: PathBuf) -> Result<Array4<f32>, LoadError> {
        todo!()
    }
}