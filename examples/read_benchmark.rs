use clap::Parser;
use indicatif::{ProgressBar, ProgressFinish, ProgressStyle};
use ndarray::Array4;
use rand::prelude::*;
use rand::SeedableRng;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use rust_search::SearchBuilder;
use snafu::ResultExt;
use std::fs::File;
use std::io::BufReader;
use std::os::unix::fs::MetadataExt;
use std::path::PathBuf;
use std::thread::available_parallelism;
use tiff::decoder::Decoder;

use dicom_preprocessing::errors::{tiff::IOSnafu, TiffError};
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

    #[arg(
        help = "Thread pool size",
        short = 't',
        long = "threads",
        default_value = None
    )]
    threads: Option<usize>,

    #[arg(help = "Random seed", short = 's', long = "seed", default_value_t = 0)]
    seed: u64,

    #[arg(help = "Sort by inode number", long = "inode", default_value_t = false)]
    inode: bool,
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

/// Sort files by inode number to optimize for sequential reads
fn sort_by_inode(files: &mut Vec<PathBuf>) {
    files.sort_by_key(|path| {
        std::fs::metadata(path).map(|m| m.ino()).unwrap_or(u64::MAX) // Fall back to end of list if metadata read fails
    });
}

fn open_tiff(path: &PathBuf, seed: u64) -> Result<Vec<usize>, TiffError> {
    // Open the file and build a decoder
    let file = File::open(path).context(IOSnafu { path: path.clone() })?;
    let mut decoder = Decoder::new(BufReader::new(file))?;

    // Read metadata injected by the preprocessor
    let metadata = PreprocessingMetadata::try_from(&mut decoder)?;

    // Choose a random frame
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let frame = *metadata
        .num_frames
        .into_iter()
        .map(|f| f as usize)
        .collect::<Vec<_>>()
        .as_slice()
        .choose(&mut rng)
        .unwrap();
    let frames = vec![frame];

    let array = Array4::<u16>::decode_frames(&mut decoder, frames.into_iter()).unwrap();

    Ok(array.shape().to_owned())
}

fn open_random_tiff(paths: &[PathBuf], seed: u64) -> Result<Vec<usize>, TiffError> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let choice = paths.choose(&mut rng).unwrap().clone();
    open_tiff(&choice, seed)
}

fn main() {
    let args = Args::parse();

    // Set up thread pool
    ThreadPoolBuilder::new()
        .num_threads(
            args.threads
                .unwrap_or(available_parallelism().unwrap().into()),
        )
        .build_global()
        .unwrap();

    // Find TIFF files
    let source = args.source;
    let mut tiff_files = find_tiff_files(&source).collect::<Vec<_>>();
    if args.inode {
        sort_by_inode(&mut tiff_files);
    } else {
        tiff_files.sort();
    }
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
        .map(|v| {
            let seed = v as u64 + args.seed;
            let result = if args.inode {
                let path = tiff_files[v as usize / args.iterations as usize].clone();
                open_tiff(&path, seed).unwrap()
            } else {
                open_random_tiff(&tiff_files, seed).unwrap()
            };
            pb.inc(1);
            result.into_iter().reduce(|a, b| a * b).unwrap()
        })
        .sum();

    pb.finish();
    println!("Total pixels read: {size}");
}
