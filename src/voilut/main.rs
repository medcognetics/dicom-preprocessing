use dicom::pixeldata::PixelDecoder;
use dicom_preprocessing::errors::dicom::{PixelDataSnafu, ReadSnafu};
use dicom_preprocessing::errors::DicomError;
use snafu::ResultExt;

use std::path::PathBuf;
use tracing::error;

use clap::Parser;
use dicom::object::open_file;

use snafu::{Report, Snafu};

#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(display("Invalid source path: {}", path.display()))]
    InvalidSourcePath { path: PathBuf },

    #[snafu(display("error processing DICOM pixel data: {:?}", source))]
    DicomError {
        #[snafu(source(from(DicomError, Box::new)))]
        source: Box<DicomError>,
    },
}

#[derive(Parser, Debug)]
#[command(author = "Scott Chase Waggener", version = env!("CARGO_PKG_VERSION"), about = "Print information about the VOILUT transform in a DICOM", long_about = None)]
struct Args {
    #[arg(
        help = "Source path. Can be a DICOM file, directory, or a text file with DICOM file paths"
    )]
    source: PathBuf,
}

fn main() {
    let args = Args::parse();
    run(args).unwrap_or_else(|e| {
        error!("{}", Report::from_error(e));
        std::process::exit(-1);
    });
}

fn run(args: Args) -> Result<(), Error> {
    if !args.source.is_file() {
        return Err(Error::InvalidSourcePath {
            path: args.source.to_path_buf(),
        });
    }

    let file = open_file(&args.source)
        .context(ReadSnafu)
        .context(DicomSnafu)?;
    let decoded = file
        .decode_pixel_data_frame(0)
        .context(PixelDataSnafu)
        .context(DicomSnafu)?;

    println!("VOILUT summary for {}", args.source.display());
    match decoded.voi_lut_function() {
        Ok(Some(voi_lut_function)) => {
            println!("VOI LUT Function: {:?}", voi_lut_function);
        }
        _ => {
            println!("No valid VOI LUT Function found");
        }
    }
    match decoded.rescale() {
        Ok(rescale) => {
            println!("Rescale: {:?}", rescale);
        }
        _ => {
            println!("No valid Rescale found");
        }
    }
    match decoded.window() {
        Ok(Some(window)) => {
            println!("Window: {:?}", window);
        }
        _ => {
            println!("No valid Window found");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case("pydicom/CT_small.dcm")]
    fn test_voilut(#[case] dicom_file_path: &str) {
        let path = dicom_test_files::path(dicom_file_path).unwrap();
        let args = Args { source: path };

        // Run main and verify it succeeds
        let result = run(args);
        assert!(result.is_ok());
    }
}
