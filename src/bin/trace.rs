use arrow::array::{Int64Array, StringArray, UInt32Array};
use arrow::error::ArrowError;
use arrow::record_batch::RecordBatch;
use clap::Parser;
use indicatif::ProgressFinish;
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::errors::ParquetError;
use parquet::file::properties::WriterProperties;
use std::fs::File;
use std::io::BufWriter;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use tracing::{error, Level};

use indicatif::{ProgressBar, ProgressStyle};
use snafu::{Report, ResultExt, Snafu, Whatever};

use dicom_preprocessing::file::Inode;
use dicom_preprocessing::manifest::{get_manifest_with_progress, ManifestEntry};

#[derive(Debug, Snafu)]
enum Error {
    #[snafu(display("Invalid source path: {}", path.display()))]
    InvalidSourcePath { path: PathBuf },

    #[snafu(display("No sources found in source path: {}", path.display()))]
    NoSources { path: PathBuf },

    #[snafu(display("Error reading inode: {:?}", source))]
    ReadInode {
        #[snafu(source(from(std::io::Error, Box::new)))]
        source: Box<std::io::Error>,
    },

    #[snafu(display("Invalid output path: {}", path.display()))]
    InvalidOutputPath { path: PathBuf },

    #[snafu(display("Arrow error: {:?}", source))]
    ArrowError {
        #[snafu(source(from(ArrowError, Box::new)))]
        source: Box<ArrowError>,
    },

    #[snafu(display("Parquet error: {:?}", source))]
    ParquetError {
        #[snafu(source(from(ParquetError, Box::new)))]
        source: Box<ParquetError>,
    },
}


#[derive(Parser, Debug)]
#[command(author = "Scott Chase Waggener", version = env!("CARGO_PKG_VERSION"), about = "Visualize traces on preprocessed TIFF files", long_about = None)]
struct Args {
    #[arg(help = "Source TIFF or directory of TIFFs")]
    source: PathBuf,

    #[arg(help = "CSV or Parquet file with sop_instance_uid, x_min, x_max, y_min, y_max")]
    traces: PathBuf,

    #[arg(help = "Output file or directory")]
    output: PathBuf,
}

fn main() {
    let args = Args::parse();

    tracing::subscriber::set_global_default(
        tracing_subscriber::FmtSubscriber::builder()
            .with_max_level(Level::ERROR)
            .finish(),
    )
    .whatever_context("Could not set up global logging subscriber")
    .unwrap_or_else(|e: Whatever| {
        eprintln!("[ERROR] {}", Report::from_error(e));
    });

    run(args).unwrap_or_else(|e| {
        error!("{}", Report::from_error(e));
        std::process::exit(-1);
    });
}

