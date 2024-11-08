use std::path::PathBuf;

use clap::error::ErrorKind;
use clap::Parser;
use dicom::object::open_file;

use dicom_preprocessing::pad::PaddingDirection;
use dicom_preprocessing::preprocess::preprocess;
use dicom_preprocessing::resize::DisplayFilterType;

#[derive(Parser, Debug)]
#[command(author = "Scott Chase Waggener", version = "0.1.0", about = "Preprocess DICOM files", long_about = None)]
struct Args {
    #[arg(help = "Source DICOM file")]
    source: PathBuf,

    #[arg(help = "Output TIFF file")]
    output: PathBuf,

    #[arg(
        help = "Crop the image",
        long = "crop",
        short = 'c',
        default_value_t = false
    )]
    crop: bool,

    #[arg(
        help = "Target size",
        long = "size",
        short = 's',
        value_parser = clap::builder::ValueParser::new(|s: &str| {
            let parts: Vec<&str> = s.split(',').collect();
            if parts.len() == 2 {
                let width = parts[0].parse::<u32>().map_err(|_| clap::Error::raw(ErrorKind::InvalidValue, "Invalid width"))?;
                let height = parts[1].parse::<u32>().map_err(|_| clap::Error::raw(ErrorKind::InvalidValue, "Invalid height"))?;
                Ok((width, height))
            } else {
                Err(clap::Error::raw(ErrorKind::InvalidValue, "Size must be in the format width,height"))
            }
        })
    )]
    size: Option<(u32, u32)>,

    #[arg(
        help = "Filter type",
        long = "filter",
        short = 'f',
        value_parser = clap::value_parser!(DisplayFilterType),
        default_value_t = DisplayFilterType::default(),
    )]
    filter: DisplayFilterType,

    #[arg(
        help = "Padding direction",
        long = "padding",
        short = 'p',
        value_parser = clap::value_parser!(PaddingDirection),
        default_value_t = PaddingDirection::default(),
    )]
    padding_direction: PaddingDirection,
}

fn main() {
    env_logger::init();
    let args = Args::parse();
    let file = open_file(&args.source).unwrap();
    preprocess(
        &file,
        args.output,
        args.crop,
        args.size,
        args.filter.into(),
        args.padding_direction,
        // TODO: Make this configurable
        tiff::encoder::compression::Packbits,
    )
    .unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_main_output() {
        // Capture the output of the main function
        let output = std::panic::catch_unwind(|| {
            main();
        });

        // Ensure the main function runs without panicking
        assert!(output.is_ok());
    }
}
