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

/// The order of the channels in the loaded image
pub enum ChannelOrder {
    First,
    Last,
}

struct Dimensions {
    width: usize,
    height: usize,
    num_frames: usize,
}

impl<R: Read+Seek> TryFrom<&mut Decoder<R>> for Dimensions {
    type Error = TiffError;
    fn try_from(decoder: &mut Decoder<R>) -> Result<Self, Self::Error> {
        let (width, height) = decoder.dimensions()?;
        //let num_frames = decoder.get_tag(tags::NUMBER_OF_FRAMES)?;
        let num_frames = 2;
        Ok(Self { width: width as usize, height: height as usize, num_frames: num_frames as usize })
    }
}


trait LoadFromTiff<T: Clone + num::Zero> {
    const CHANNEL_ORDER: ChannelOrder;
    const CHANNELS: usize;

    fn open(path: PathBuf) -> Result<Decoder<BufReader<File>>, LoadError> {
        let file = BufReader::new(File::open(&path).context(OpenTiffSnafu { path: path.clone() })?);
        Decoder::new(file).context(ReadTiffSnafu { path })
    }

    fn preallocate(decoder: &mut Decoder<BufReader<File>>) -> Result<Array4<T>, TiffError> {
        let dimensions = Dimensions::try_from(decoder)?;
        let array = Array4::<T>::zeros((
            dimensions.num_frames,
            dimensions.height,
            dimensions.width,
            Self::CHANNELS));
        Array::from
        Ok(array)
    }

    fn load(path: PathBuf, channel_order: ChannelOrder) -> Result<Array4<T>, LoadError>;
}


struct LoadGray8;

impl LoadFromTiff<u8> for LoadGray8 {
    fn load(path: PathBuf, channel_order: ChannelOrder) -> Result<Array4<u8>, LoadError> {
        let decoder = LoadFromTiff::open(path)?;
        decoder.read_image()
        let mut array = LoadFromTiff::preallocate(&mut decoder).context(ReadTiffSnafu { path })?;

        let num_frames = decoder.get_tag()
        let (width, height) = decoder.dimensions().unwrap();

        // Pre-allocate array
        let mut image_data = Array4::<u8>::zeros((frames, height as usize, width as usize, Self::CHANNELS));

        decoder.read_image()

        let image = decoder.read_image().context(ReadTiffSnafu { path })?;
        let buffer = image.as_buffer(0);



        Ok(image.to_luma8().into_raw())
    }
}


struct LoadGray16;
struct LoadRgb8;
struct LoadNormalizedF32;

/*
    fn load(path: PathBuf) -> Result<Array4<T>, LoadError> {

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
*/