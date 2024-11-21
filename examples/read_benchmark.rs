use clap::Parser;
use indicatif::{ProgressBar, ProgressFinish, ProgressStyle};
use ndarray::Array4;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use rust_search::SearchBuilder;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use tiff::decoder::Decoder;
use tiff::TiffError;

use dicom_preprocessing::load::LoadFromTiff;
use dicom_preprocessing::metadata::PreprocessingMetadata;

#[derive(Parser, Debug)]
#[command(
    author = "Scott Chase Waggener",
    about = "Benchmark reading TIFF files"
)]
struct Args {
    #[arg(help = "Directory containing TIFF files to read")]
    source: PathBuf,

    #[arg(
        help = "Number of times to read a random TIFF file",
        short = 'i',
        long = "iterations",
        default_value_t = 1000
    )]
    iterations: u32,
}

fn find_tiff_files(dir: &PathBuf) -> impl Iterator<Item = PathBuf> {
    // Set up spinner, iterating may files may take some time
    let spinner = ProgressBar::new_spinner();
    spinner.set_message("Searching for TIFF files");
    spinner.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.blue} {msg}")
            .unwrap(),
    );

    // Yield from the search
    SearchBuilder::default()
        .location(dir)
        .build()
        .inspect(move |_| spinner.tick())
        .map(PathBuf::from)
        .filter(|file| file.extension().unwrap_or_default() == "tiff")
}

fn open_tiff(path: &PathBuf) -> Result<Vec<usize>, TiffError> {
    // Open the file and build a decoder
    let file = File::open(path)?;
    let mut decoder = Decoder::new(BufReader::new(file))?;

    // Read metadata injected by the preprocessor
    let metadata = PreprocessingMetadata::try_from(&mut decoder)?;

    // Choose a random frame
    let mut rng = rand::thread_rng();
    let frame = metadata
        .num_frames
        .into_iter()
        .map(|f| f as usize)
        .collect::<Vec<_>>()
        .choose(&mut rng)
        .unwrap()
        .clone();
    let frames = vec![frame];

    let array = Array4::<u16>::decode_frames(&mut decoder, frames.into_iter()).unwrap();

    Ok(array.shape().to_owned())
}

fn open_random_tiff(paths: &[PathBuf]) -> Result<Vec<usize>, TiffError> {
    let mut rng = rand::thread_rng();
    let choice = paths.choose(&mut rng).unwrap().clone();
    open_tiff(&choice)
}

fn main() {
    let args = Args::parse();
    let source = args.source;
    let tiff_files = find_tiff_files(&source).collect::<Vec<_>>();
    println!("Found {} TIFF files", tiff_files.len());
    if tiff_files.is_empty() {
        println!("No TIFF files found in {}", source.display());
        return;
    }

    let pb = ProgressBar::new(args.iterations as u64).with_finish(ProgressFinish::AndLeave);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{msg} {spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta} @ {per_sec})",
            )
            .unwrap(),
    );
    pb.set_message("Reading random TIFF files");

    let size: usize = (0..args.iterations)
        .into_par_iter()
        .map(|_| {
            let result = open_random_tiff(&tiff_files).unwrap();
            pb.inc(1);
            result.into_iter().reduce(|a, b| a * b).unwrap()
        })
        .sum();

    pb.finish();
    println!("Total pixels read: {}", size);
}
