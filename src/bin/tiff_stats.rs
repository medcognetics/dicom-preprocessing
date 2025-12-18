use clap::Parser;
use dicom_preprocessing::file::{default_bar, InodeSort, TiffFileOperations};
use indicatif::ParallelProgressIterator;
use rayon::prelude::*;
use snafu::{Report, ResultExt, Snafu, Whatever};
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use std::sync::Mutex;
use tiff::decoder::{Decoder, DecodingResult};
use tracing::{error, info, Level};

#[derive(Debug, Snafu)]
enum Error {
    #[snafu(display("Invalid source path: {}", path.display()))]
    InvalidSourcePath { path: PathBuf },

    #[snafu(display("No TIFF files found in source path: {}", path.display()))]
    NoSources { path: PathBuf },

    #[snafu(display("Inconsistent channel count: expected {}, found {}", expected, found))]
    InconsistentChannels { expected: usize, found: usize },

    #[snafu(display("IO error: {:?}", source))]
    IO {
        #[snafu(source(from(std::io::Error, Box::new)))]
        source: Box<std::io::Error>,
    },

    #[snafu(display("Error reading TIFF: {:?}", source))]
    TiffRead {
        #[snafu(source(from(tiff::TiffError, Box::new)))]
        source: Box<tiff::TiffError>,
    },

    #[snafu(display("Unsupported TIFF format"))]
    UnsupportedFormat,
}

#[derive(Parser, Debug)]
#[command(
    author = "Scott Chase Waggener",
    version = env!("CARGO_PKG_VERSION"),
    about = "Compute per-channel pixel statistics (mean/std) for TIFF files",
    long_about = None
)]
struct Args {
    #[arg(help = "Directory containing TIFF files")]
    source: PathBuf,

    #[arg(
        help = "Normalize pixel values to [0, 1] range based on input dtype maximum",
        long = "normalize",
        short = 'n',
        default_value = "false"
    )]
    normalize: bool,

    #[arg(
        help = "Enable verbose logging",
        long = "verbose",
        short = 'v',
        default_value = "false"
    )]
    verbose: bool,
}

/// Welford's online algorithm state for computing mean and variance
#[derive(Debug, Clone)]
struct WelfordState {
    count: u64,
    mean: Vec<f64>,
    m2: Vec<f64>,
}

impl WelfordState {
    fn new(num_channels: usize) -> Self {
        Self {
            count: 0,
            mean: vec![0.0; num_channels],
            m2: vec![0.0; num_channels],
        }
    }

    fn num_channels(&self) -> usize {
        self.mean.len()
    }

    /// Update the state with a new pixel value for each channel
    fn update(&mut self, values: &[f64]) {
        assert_eq!(values.len(), self.mean.len());
        self.count += 1;
        let n = self.count as f64;

        for (i, &value) in values.iter().enumerate() {
            let delta = value - self.mean[i];
            self.mean[i] += delta / n;
            let delta2 = value - self.mean[i];
            self.m2[i] += delta * delta2;
        }
    }

    /// Merge another WelfordState into this one
    fn merge(&mut self, other: &WelfordState) {
        assert_eq!(self.mean.len(), other.mean.len());

        if other.count == 0 {
            return;
        }

        let n_a = self.count as f64;
        let n_b = other.count as f64;
        let n = n_a + n_b;

        for i in 0..self.mean.len() {
            let delta = other.mean[i] - self.mean[i];
            self.mean[i] = (n_a * self.mean[i] + n_b * other.mean[i]) / n;
            self.m2[i] += other.m2[i] + delta * delta * n_a * n_b / n;
        }

        self.count += other.count;
    }

    /// Compute the standard deviation
    fn std(&self) -> Vec<f64> {
        if self.count < 2 {
            return vec![0.0; self.mean.len()];
        }

        self.m2
            .iter()
            .map(|&m2| (m2 / (self.count - 1) as f64).sqrt())
            .collect()
    }
}

fn main() {
    let args = Args::parse();

    let level = if args.verbose {
        Level::DEBUG
    } else {
        Level::ERROR
    };
    tracing::subscriber::set_global_default(
        tracing_subscriber::FmtSubscriber::builder()
            .with_max_level(level)
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

fn validate_args(args: &Args) -> Result<(), Error> {
    if !args.source.is_dir() {
        return Err(Error::InvalidSourcePath {
            path: args.source.clone(),
        });
    }
    Ok(())
}

fn load_source_files(source_path: &PathBuf) -> Result<Vec<PathBuf>, Error> {
    let source_files = source_path
        .find_tiffs()
        .map_err(|_| Error::InvalidSourcePath {
            path: source_path.clone(),
        })?
        .collect::<Vec<_>>();

    let source_files = source_files
        .into_iter()
        .sorted_by_inode_with_progress()
        .collect::<Vec<_>>();

    if source_files.is_empty() {
        return Err(Error::NoSources {
            path: source_path.clone(),
        });
    }

    info!("Found {} TIFF files", source_files.len());
    Ok(source_files)
}

/// Process a single TIFF file and accumulate statistics
fn process_tiff(
    path: &PathBuf,
    expected_channels: Option<usize>,
    normalize: bool,
) -> Result<WelfordState, Error> {
    let file = File::open(path).context(IOSnafu)?;
    let reader = BufReader::new(file);
    let mut decoder = Decoder::new(reader).context(TiffReadSnafu)?;

    // Determine number of channels from the first frame
    let colortype = decoder.colortype().context(TiffReadSnafu)?;
    let num_channels = match colortype {
        tiff::ColorType::Gray(_) => 1,
        tiff::ColorType::RGB(_) => 3,
        tiff::ColorType::RGBA(_) => 4,
        tiff::ColorType::CMYK(_) => 4,
        _ => return Err(Error::UnsupportedFormat),
    };

    // Validate channel count consistency
    if let Some(expected) = expected_channels {
        if num_channels != expected {
            return Err(Error::InconsistentChannels {
                expected,
                found: num_channels,
            });
        }
    }

    let mut state = WelfordState::new(num_channels);

    // Process all frames in the TIFF
    loop {
        // Read the current frame
        let result = decoder.read_image().context(TiffReadSnafu)?;

        // Process pixels based on bit depth
        match result {
            DecodingResult::U8(data) => {
                process_pixels_u8(&data, num_channels, normalize, &mut state);
            }
            DecodingResult::U16(data) => {
                process_pixels_u16(&data, num_channels, normalize, &mut state);
            }
            DecodingResult::U32(data) => {
                process_pixels_u32(&data, num_channels, normalize, &mut state);
            }
            DecodingResult::U64(data) => {
                process_pixels_u64(&data, num_channels, normalize, &mut state);
            }
            DecodingResult::F32(data) => {
                process_pixels_f32(&data, num_channels, normalize, &mut state);
            }
            DecodingResult::F64(data) => {
                process_pixels_f64(&data, num_channels, normalize, &mut state);
            }
            DecodingResult::I8(data) => {
                process_pixels_i8(&data, num_channels, normalize, &mut state);
            }
            DecodingResult::I16(data) => {
                process_pixels_i16(&data, num_channels, normalize, &mut state);
            }
            DecodingResult::I32(data) => {
                process_pixels_i32(&data, num_channels, normalize, &mut state);
            }
            DecodingResult::I64(data) => {
                process_pixels_i64(&data, num_channels, normalize, &mut state);
            }
            DecodingResult::F16(data) => {
                process_pixels_f16(&data, num_channels, normalize, &mut state);
            }
        }

        // Try to move to next frame
        if !decoder.more_images() {
            break;
        }
        decoder.next_image().context(TiffReadSnafu)?;
    }

    Ok(state)
}

fn process_pixels_u8(data: &[u8], num_channels: usize, normalize: bool, state: &mut WelfordState) {
    if normalize {
        for chunk in data.chunks_exact(num_channels) {
            let values: Vec<f64> = chunk
                .iter()
                .map(|&x| (x as f64) / (u8::MAX as f64))
                .collect();
            state.update(&values);
        }
    } else {
        for chunk in data.chunks_exact(num_channels) {
            let values: Vec<f64> = chunk.iter().map(|&x| x as f64).collect();
            state.update(&values);
        }
    }
}

fn process_pixels_u16(
    data: &[u16],
    num_channels: usize,
    normalize: bool,
    state: &mut WelfordState,
) {
    if normalize {
        for chunk in data.chunks_exact(num_channels) {
            let values: Vec<f64> = chunk
                .iter()
                .map(|&x| (x as f64) / (u16::MAX as f64))
                .collect();
            state.update(&values);
        }
    } else {
        for chunk in data.chunks_exact(num_channels) {
            let values: Vec<f64> = chunk.iter().map(|&x| x as f64).collect();
            state.update(&values);
        }
    }
}

fn process_pixels_u32(
    data: &[u32],
    num_channels: usize,
    normalize: bool,
    state: &mut WelfordState,
) {
    if normalize {
        for chunk in data.chunks_exact(num_channels) {
            let values: Vec<f64> = chunk
                .iter()
                .map(|&x| (x as f64) / (u32::MAX as f64))
                .collect();
            state.update(&values);
        }
    } else {
        for chunk in data.chunks_exact(num_channels) {
            let values: Vec<f64> = chunk.iter().map(|&x| x as f64).collect();
            state.update(&values);
        }
    }
}

fn process_pixels_u64(
    data: &[u64],
    num_channels: usize,
    normalize: bool,
    state: &mut WelfordState,
) {
    if normalize {
        for chunk in data.chunks_exact(num_channels) {
            let values: Vec<f64> = chunk
                .iter()
                .map(|&x| (x as f64) / (u64::MAX as f64))
                .collect();
            state.update(&values);
        }
    } else {
        for chunk in data.chunks_exact(num_channels) {
            let values: Vec<f64> = chunk.iter().map(|&x| x as f64).collect();
            state.update(&values);
        }
    }
}

fn process_pixels_i8(data: &[i8], num_channels: usize, normalize: bool, state: &mut WelfordState) {
    if normalize {
        for chunk in data.chunks_exact(num_channels) {
            let values: Vec<f64> = chunk
                .iter()
                .map(|&x| (x as f64) / (i8::MAX as f64))
                .collect();
            state.update(&values);
        }
    } else {
        for chunk in data.chunks_exact(num_channels) {
            let values: Vec<f64> = chunk.iter().map(|&x| x as f64).collect();
            state.update(&values);
        }
    }
}

fn process_pixels_i16(
    data: &[i16],
    num_channels: usize,
    normalize: bool,
    state: &mut WelfordState,
) {
    if normalize {
        for chunk in data.chunks_exact(num_channels) {
            let values: Vec<f64> = chunk
                .iter()
                .map(|&x| (x as f64) / (i16::MAX as f64))
                .collect();
            state.update(&values);
        }
    } else {
        for chunk in data.chunks_exact(num_channels) {
            let values: Vec<f64> = chunk.iter().map(|&x| x as f64).collect();
            state.update(&values);
        }
    }
}

fn process_pixels_i32(
    data: &[i32],
    num_channels: usize,
    normalize: bool,
    state: &mut WelfordState,
) {
    if normalize {
        for chunk in data.chunks_exact(num_channels) {
            let values: Vec<f64> = chunk
                .iter()
                .map(|&x| (x as f64) / (i32::MAX as f64))
                .collect();
            state.update(&values);
        }
    } else {
        for chunk in data.chunks_exact(num_channels) {
            let values: Vec<f64> = chunk.iter().map(|&x| x as f64).collect();
            state.update(&values);
        }
    }
}

fn process_pixels_i64(
    data: &[i64],
    num_channels: usize,
    normalize: bool,
    state: &mut WelfordState,
) {
    if normalize {
        for chunk in data.chunks_exact(num_channels) {
            let values: Vec<f64> = chunk
                .iter()
                .map(|&x| (x as f64) / (i64::MAX as f64))
                .collect();
            state.update(&values);
        }
    } else {
        for chunk in data.chunks_exact(num_channels) {
            let values: Vec<f64> = chunk.iter().map(|&x| x as f64).collect();
            state.update(&values);
        }
    }
}

fn process_pixels_f32(
    data: &[f32],
    num_channels: usize,
    _normalize: bool,
    state: &mut WelfordState,
) {
    // F32 is already in floating point, normalization doesn't apply the same way
    // Assume values are already in appropriate range
    for chunk in data.chunks_exact(num_channels) {
        let values: Vec<f64> = chunk.iter().map(|&x| x as f64).collect();
        state.update(&values);
    }
}

fn process_pixels_f64(
    data: &[f64],
    num_channels: usize,
    _normalize: bool,
    state: &mut WelfordState,
) {
    // F64 is already in floating point, normalization doesn't apply the same way
    // Assume values are already in appropriate range
    for chunk in data.chunks_exact(num_channels) {
        state.update(chunk);
    }
}

fn process_pixels_f16(
    data: &[half::f16],
    num_channels: usize,
    _normalize: bool,
    state: &mut WelfordState,
) {
    // F16 is already in floating point, normalization doesn't apply the same way
    // Assume values are already in appropriate range
    for chunk in data.chunks_exact(num_channels) {
        let values: Vec<f64> = chunk.iter().map(|&x| f64::from(x)).collect();
        state.update(&values);
    }
}

fn compute_stats(source_files: Vec<PathBuf>, normalize: bool) -> Result<WelfordState, Error> {
    let pb = default_bar(source_files.len() as u64);
    pb.set_message("Computing statistics");

    // Process first file to determine number of channels
    let first_state = process_tiff(&source_files[0], None, normalize)?;
    let num_channels = first_state.num_channels();

    // Accumulator for merging results from parallel processing
    let global_state = Mutex::new(first_state);

    // Process remaining files in parallel
    let results: Result<Vec<_>, Error> = source_files[1..]
        .into_par_iter()
        .progress_with(pb)
        .map(|path| process_tiff(path, Some(num_channels), normalize))
        .collect();

    let states = results?;

    // Merge all states
    let mut final_state = global_state.into_inner().unwrap();
    for state in states {
        final_state.merge(&state);
    }

    Ok(final_state)
}

fn print_results(state: &WelfordState, normalize: bool) {
    let std = state.std();

    println!("\nPixel Statistics:");
    println!("Total pixels processed: {}", state.count);
    println!("Number of channels: {}", state.num_channels());
    if normalize {
        println!("Values normalized to [0, 1] range");
    }
    println!();

    for (i, (&mean, &std)) in state.mean.iter().zip(std.iter()).enumerate() {
        println!("Channel {i}: mean = {mean:.4}, std = {std:.4}");
    }
}

fn run(args: Args) -> Result<(), Error> {
    validate_args(&args)?;
    let source_files = load_source_files(&args.source)?;
    let state = compute_stats(source_files, args.normalize)?;
    print_results(&state, args.normalize);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use dicom_preprocessing::color::DicomColorType;
    use dicom_preprocessing::metadata::PreprocessingMetadata;
    use dicom_preprocessing::save::TiffSaver;
    use dicom_preprocessing::FrameCount;
    use image::DynamicImage;
    use ndarray::Array4;
    use std::fs;
    use tempfile::TempDir;
    use tiff::encoder::compression::{Compressor, Uncompressed};

    const TOLERANCE: f64 = 1e-2;

    fn create_test_tiff_gray(
        path: &PathBuf,
        width: u32,
        height: u32,
        num_frames: usize,
        pixel_value: u8,
    ) {
        let array = Array4::<u8>::from_elem(
            (num_frames, height as usize, width as usize, 1),
            pixel_value,
        );
        let metadata = PreprocessingMetadata {
            crop: None,
            resize: None,
            padding: None,
            resolution: None,
            num_frames: FrameCount::from(num_frames as u16),
        };
        let saver = TiffSaver::new(
            Compressor::Uncompressed(Uncompressed),
            DicomColorType::Gray8(tiff::encoder::colortype::Gray8),
        );

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }

        let mut encoder = saver.open_tiff(path).unwrap();

        for frame_idx in 0..num_frames {
            let frame_data = array.slice(ndarray::s![frame_idx, .., .., ..]);
            let dynamic_image = DynamicImage::ImageLuma8(
                image::ImageBuffer::from_raw(width, height, frame_data.iter().cloned().collect())
                    .unwrap(),
            );
            saver.save(&mut encoder, &dynamic_image, &metadata).unwrap();
        }
    }

    #[test]
    fn test_welford_state_basic() {
        let mut state = WelfordState::new(1);

        // Add values: 1, 2, 3, 4, 5
        for i in 1..=5 {
            state.update(&[i as f64]);
        }

        // Expected mean: 3.0
        // Expected variance: 2.5
        // Expected std: sqrt(2.5) ≈ 1.58
        assert!((state.mean[0] - 3.0).abs() < TOLERANCE);
        let std = state.std();
        assert!((std[0] - 1.58113883).abs() < TOLERANCE);
    }

    #[test]
    fn test_welford_state_merge() {
        let mut state1 = WelfordState::new(1);
        let mut state2 = WelfordState::new(1);

        // State 1: values 1, 2, 3
        for i in 1..=3 {
            state1.update(&[i as f64]);
        }

        // State 2: values 4, 5
        for i in 4..=5 {
            state2.update(&[i as f64]);
        }

        // Merge
        state1.merge(&state2);

        // Should have same result as processing all together
        assert_eq!(state1.count, 5);
        assert!((state1.mean[0] - 3.0).abs() < TOLERANCE);
        let std = state1.std();
        assert!((std[0] - 1.58113883).abs() < TOLERANCE);
    }

    #[test]
    fn test_welford_state_multi_channel() {
        let mut state = WelfordState::new(3);

        // Add RGB values
        state.update(&[100.0, 150.0, 200.0]);
        state.update(&[110.0, 160.0, 210.0]);
        state.update(&[90.0, 140.0, 190.0]);

        // Expected means: 100, 150, 200
        assert!((state.mean[0] - 100.0).abs() < TOLERANCE);
        assert!((state.mean[1] - 150.0).abs() < TOLERANCE);
        assert!((state.mean[2] - 200.0).abs() < TOLERANCE);

        let std = state.std();
        // Expected std: 10.0 for each channel
        assert!((std[0] - 10.0).abs() < TOLERANCE);
        assert!((std[1] - 10.0).abs() < TOLERANCE);
        assert!((std[2] - 10.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_single_tiff_constant_value() {
        let tmp_dir = TempDir::new().unwrap();
        let source_dir = tmp_dir.path().join("source");
        fs::create_dir(&source_dir).unwrap();

        let tiff_path = source_dir.join("test.tiff");
        create_test_tiff_gray(&tiff_path, 4, 4, 1, 128);

        let args = Args {
            source: source_dir,
            normalize: false,
            verbose: false,
        };

        let source_files = load_source_files(&args.source).unwrap();
        let state = compute_stats(source_files, false).unwrap();

        // All pixels are 128, so mean should be 128 and std should be 0
        assert_eq!(state.num_channels(), 1);
        assert!((state.mean[0] - 128.0).abs() < TOLERANCE);
        let std = state.std();
        assert!(std[0] < TOLERANCE);
    }

    #[test]
    fn test_multiple_tiffs() {
        let tmp_dir = TempDir::new().unwrap();
        let source_dir = tmp_dir.path().join("source");
        fs::create_dir(&source_dir).unwrap();

        // Create TIFFs with different values
        create_test_tiff_gray(&source_dir.join("test1.tiff"), 2, 2, 1, 100);
        create_test_tiff_gray(&source_dir.join("test2.tiff"), 2, 2, 1, 200);

        let args = Args {
            source: source_dir,
            normalize: false,
            verbose: false,
        };

        let source_files = load_source_files(&args.source).unwrap();
        let state = compute_stats(source_files, false).unwrap();

        // Mean should be 150
        assert_eq!(state.num_channels(), 1);
        assert!((state.mean[0] - 150.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_multi_frame_tiff() {
        let tmp_dir = TempDir::new().unwrap();
        let source_dir = tmp_dir.path().join("source");
        fs::create_dir(&source_dir).unwrap();

        let tiff_path = source_dir.join("test.tiff");
        create_test_tiff_gray(&tiff_path, 4, 4, 5, 100);

        let args = Args {
            source: source_dir,
            normalize: false,
            verbose: false,
        };

        let source_files = load_source_files(&args.source).unwrap();
        let state = compute_stats(source_files, false).unwrap();

        // All pixels in all frames are 100
        assert_eq!(state.count, 4 * 4 * 5);
        assert!((state.mean[0] - 100.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_inconsistent_channels() {
        let tmp_dir = TempDir::new().unwrap();
        let source_dir = tmp_dir.path().join("source");
        fs::create_dir(&source_dir).unwrap();

        // Create a grayscale TIFF
        create_test_tiff_gray(&source_dir.join("test1.tiff"), 4, 4, 1, 100);

        // Create an RGB TIFF manually
        let rgb_path = source_dir.join("test2.tiff");
        let rgb_image = DynamicImage::ImageRgb8(image::RgbImage::new(4, 4));
        rgb_image.save(&rgb_path).unwrap();

        let args = Args {
            source: source_dir,
            normalize: false,
            verbose: false,
        };

        let source_files = load_source_files(&args.source).unwrap();
        let result = compute_stats(source_files, false);

        // Should fail due to inconsistent channels
        assert!(matches!(result, Err(Error::InconsistentChannels { .. })));
    }

    #[test]
    fn test_invalid_source_path() {
        let tmp_dir = TempDir::new().unwrap();
        let args = Args {
            source: tmp_dir.path().join("nonexistent"),
            normalize: false,
            verbose: false,
        };
        assert!(matches!(run(args), Err(Error::InvalidSourcePath { .. })));
    }

    #[test]
    fn test_no_sources() {
        let tmp_dir = TempDir::new().unwrap();
        let source_dir = tmp_dir.path().join("source");
        fs::create_dir(&source_dir).unwrap();

        let args = Args {
            source: source_dir,
            normalize: false,
            verbose: false,
        };
        assert!(matches!(run(args), Err(Error::NoSources { .. })));
    }

    #[test]
    fn test_normalize_u8() {
        let tmp_dir = TempDir::new().unwrap();
        let source_dir = tmp_dir.path().join("source");
        fs::create_dir(&source_dir).unwrap();

        // Create TIFF with u8 values: 0, 127, 255
        create_test_tiff_gray(&source_dir.join("test1.tiff"), 2, 2, 1, 0);
        create_test_tiff_gray(&source_dir.join("test2.tiff"), 2, 2, 1, 127);
        create_test_tiff_gray(&source_dir.join("test3.tiff"), 2, 2, 1, 255);

        let source_files = load_source_files(&source_dir).unwrap();
        let state = compute_stats(source_files, true).unwrap();

        // With normalization, mean should be (0 + 127/255 + 1) / 3 ≈ 0.498
        // Expected: (0 + 0.498039 + 1.0) / 3 = 0.499346
        assert_eq!(state.num_channels(), 1);
        assert!((state.mean[0] - 0.499346).abs() < TOLERANCE);
    }

    #[test]
    fn test_normalize_vs_raw() {
        let tmp_dir = TempDir::new().unwrap();
        let source_dir = tmp_dir.path().join("source");
        fs::create_dir(&source_dir).unwrap();

        let tiff_path = source_dir.join("test.tiff");
        create_test_tiff_gray(&tiff_path, 4, 4, 1, 128);

        let source_files = load_source_files(&source_dir).unwrap();

        // Without normalization
        let state_raw = compute_stats(source_files.clone(), false).unwrap();
        assert!((state_raw.mean[0] - 128.0).abs() < TOLERANCE);

        // With normalization (128 / 255 ≈ 0.502)
        let state_norm = compute_stats(source_files, true).unwrap();
        assert!((state_norm.mean[0] - (128.0 / 255.0)).abs() < TOLERANCE);
    }
}
