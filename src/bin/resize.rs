use clap::Parser;
use dicom_preprocessing::color::DicomColorType;
use dicom_preprocessing::errors::TiffError;
use dicom_preprocessing::file::{default_bar, InodeSort, TiffFileOperations};
use dicom_preprocessing::load::load_frames_as_dynamic_images;
use dicom_preprocessing::metadata::PreprocessingMetadata;
use dicom_preprocessing::save::TiffSaver;
use dicom_preprocessing::transform::resize::FilterType;
use dicom_preprocessing::transform::Transform;
use image::GenericImageView;
use indicatif::ParallelProgressIterator;
use rayon::prelude::*;
use snafu::{Report, ResultExt, Snafu, Whatever};
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use tiff::decoder::Decoder;
use tiff::encoder::compression::{Compressor, Uncompressed};
use tracing::{error, info, Level};

#[derive(Debug, Snafu)]
enum Error {
    #[snafu(display("Invalid source path: {}", path.display()))]
    InvalidSourcePath { path: PathBuf },

    #[snafu(display("No TIFF files found in source path: {}", path.display()))]
    NoSources { path: PathBuf },

    #[snafu(display("Invalid output path: {}", path.display()))]
    InvalidOutputPath { path: PathBuf },

    #[snafu(display("Invalid scale factor: {}", scale))]
    InvalidScaleFactor { scale: f32 },

    #[snafu(display("IO error: {:?}", source))]
    IO {
        #[snafu(source(from(std::io::Error, Box::new)))]
        source: Box<std::io::Error>,
    },

    #[snafu(display("Error reading TIFF: {:?}", source))]
    TiffRead {
        #[snafu(source(from(TiffError, Box::new)))]
        source: Box<TiffError>,
    },

    #[snafu(display("Error writing TIFF: {:?}", source))]
    TiffWrite {
        #[snafu(source(from(TiffError, Box::new)))]
        source: Box<TiffError>,
    },
}

#[derive(Parser, Debug)]
#[command(
    author = "Scott Chase Waggener",
    version = env!("CARGO_PKG_VERSION"),
    about = "Resize preprocessed TIFF files by a scale factor",
    long_about = None
)]
struct Args {
    #[arg(help = "Directory containing TIFF files to resize")]
    source: PathBuf,

    #[arg(help = "Scale factor for resizing (e.g., 0.5 for half size, 2.0 for double size)")]
    scale: f32,

    #[arg(help = "Output directory for resized TIFFs")]
    output: PathBuf,

    #[arg(
        help = "Filter type for resizing",
        long = "filter",
        short = 'f',
        default_value = "triangle"
    )]
    filter: FilterType,

    #[arg(
        help = "Enable verbose logging",
        long = "verbose",
        short = 'v',
        default_value = "false"
    )]
    verbose: bool,
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

    if args.scale <= 0.0 {
        return Err(Error::InvalidScaleFactor { scale: args.scale });
    }

    if !args.output.exists() {
        std::fs::create_dir_all(&args.output).context(IOSnafu)?;
    } else if !args.output.is_dir() {
        return Err(Error::InvalidOutputPath {
            path: args.output.clone(),
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

fn resize_tiff(
    input_path: &Path,
    output_path: &Path,
    scale: f32,
    filter: FilterType,
) -> Result<(), Error> {
    // Load TIFF metadata and frames
    let file = File::open(input_path).context(IOSnafu)?;
    let reader = BufReader::new(file);
    let mut decoder = Decoder::new(reader)
        .map_err(TiffError::from)
        .context(TiffReadSnafu)?;

    // Get metadata from input TIFF
    let original_metadata = PreprocessingMetadata::try_from(&mut decoder).context(TiffReadSnafu)?;
    let color_type = DicomColorType::try_from(&mut decoder).context(TiffReadSnafu)?;

    // Extract num_frames, consuming the frame count
    let num_frames: usize = original_metadata.num_frames.into();

    // Load all frames as DynamicImages
    let images = load_frames_as_dynamic_images(&mut decoder, &color_type, 0..num_frames)
        .context(TiffReadSnafu)?;

    // Create new metadata with updated information
    let mut metadata = PreprocessingMetadata {
        crop: original_metadata.crop,
        resize: original_metadata.resize,
        padding: original_metadata.padding,
        resolution: original_metadata.resolution,
        num_frames: num_frames.into(),
    };

    // Create resize transform
    let first_image = images.first().unwrap();
    let (width, height) = first_image.dimensions();
    let target_width = (width as f32 * scale).round() as u32;
    let target_height = (height as f32 * scale).round() as u32;

    let resize_transform = dicom_preprocessing::transform::resize::Resize::new(
        first_image,
        target_width,
        target_height,
        filter,
    );

    // Apply resize to all frames
    let resized_images: Vec<_> = images
        .iter()
        .map(|img| resize_transform.apply(img))
        .collect();

    // Update metadata with new resize info
    let new_resize = Some(dicom_preprocessing::transform::resize::Resize {
        scale_x: scale,
        scale_y: scale,
        filter,
    });

    // Compose with existing resize if present
    metadata.resize = match metadata.resize {
        Some(existing_resize) => {
            let combined_scale_x = existing_resize.scale_x * scale;
            let combined_scale_y = existing_resize.scale_y * scale;
            Some(dicom_preprocessing::transform::resize::Resize {
                scale_x: combined_scale_x,
                scale_y: combined_scale_y,
                filter,
            })
        }
        None => new_resize,
    };

    // Update resolution if present
    if let (Some(resolution), Some(resize)) = (&metadata.resolution, &metadata.resize) {
        metadata.resolution = Some(resize.apply(resolution));
    }

    // Update padding if present - scale padding values to account for resized coordinate space
    if let Some(padding) = metadata.padding {
        metadata.padding = Some(dicom_preprocessing::transform::pad::Padding {
            left: (padding.left as f32 * scale).round() as u32,
            top: (padding.top as f32 * scale).round() as u32,
            right: (padding.right as f32 * scale).round() as u32,
            bottom: (padding.bottom as f32 * scale).round() as u32,
        });
    }

    // Create output directory if it doesn't exist
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent).context(IOSnafu)?;
    }

    // Save resized TIFF
    let saver = TiffSaver::new(Compressor::Uncompressed(Uncompressed), color_type);
    let mut encoder = saver.open_tiff(output_path).context(TiffWriteSnafu)?;

    for image in resized_images.iter() {
        saver
            .save(&mut encoder, image, &metadata)
            .context(TiffWriteSnafu)?;
    }

    Ok(())
}

fn compute_output_path(input_path: &Path, source_dir: &Path, output_dir: &Path) -> PathBuf {
    let relative_path = input_path
        .strip_prefix(source_dir)
        .expect("Input path should be within source directory");
    output_dir.join(relative_path)
}

fn resize_all_tiffs(
    source_files: Vec<PathBuf>,
    source_dir: &Path,
    output_dir: &Path,
    scale: f32,
    filter: FilterType,
) -> Result<usize, Error> {
    let pb = default_bar(source_files.len() as u64);
    pb.set_message("Resizing TIFFs");

    let num_files = source_files.len();

    let results: Result<Vec<_>, Error> = source_files
        .into_par_iter()
        .progress_with(pb)
        .map(|input_path| {
            let output_path = compute_output_path(&input_path, source_dir, output_dir);
            resize_tiff(&input_path, &output_path, scale, filter)?;
            Ok(())
        })
        .collect();

    results?;
    Ok(num_files)
}

fn run(args: Args) -> Result<usize, Error> {
    validate_args(&args)?;
    let source_files = load_source_files(&args.source)?;
    resize_all_tiffs(
        source_files,
        &args.source,
        &args.output,
        args.scale,
        args.filter,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use dicom_preprocessing::FrameCount;
    use image::DynamicImage;
    use ndarray::Array4;
    use rstest::rstest;
    use std::fs;
    use tempfile::TempDir;

    const TOLERANCE: f32 = 1e-6;

    fn create_test_tiff(path: &Path, width: u32, height: u32, num_frames: usize) -> PathBuf {
        let array = Array4::<u8>::zeros((num_frames, height as usize, width as usize, 1));
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

        path.to_path_buf()
    }

    #[rstest]
    #[case(0.5, FilterType::Triangle)]
    #[case(2.0, FilterType::Nearest)]
    #[case(1.5, FilterType::Lanczos3)]
    fn test_resize_single_frame(#[case] scale: f32, #[case] filter: FilterType) {
        let tmp_dir = TempDir::new().unwrap();
        let source_dir = tmp_dir.path().join("source");
        let output_dir = tmp_dir.path().join("output");
        fs::create_dir(&source_dir).unwrap();
        fs::create_dir(&output_dir).unwrap();

        let input_path = source_dir.join("test.tiff");
        create_test_tiff(&input_path, 64, 64, 1);

        let args = Args {
            source: source_dir,
            output: output_dir.clone(),
            scale,
            filter,
            verbose: false,
        };

        let num_files = run(args).unwrap();
        assert_eq!(num_files, 1);

        // Verify output file exists
        let output_path = output_dir.join("test.tiff");
        assert!(output_path.exists());

        // Verify dimensions are scaled
        let file = File::open(output_path).unwrap();
        let reader = BufReader::new(file);
        let mut decoder = Decoder::new(reader).unwrap();
        let (width, height) = decoder.dimensions().unwrap();
        let expected_width = (64.0 * scale).round() as u32;
        let expected_height = (64.0 * scale).round() as u32;
        assert_eq!(width, expected_width);
        assert_eq!(height, expected_height);

        // Verify metadata contains resize info
        let metadata = PreprocessingMetadata::try_from(&mut decoder).unwrap();
        assert!(metadata.resize.is_some());
        let resize = metadata.resize.unwrap();
        assert!((resize.scale_x - scale).abs() < TOLERANCE);
        assert!((resize.scale_y - scale).abs() < TOLERANCE);
    }

    #[rstest]
    #[case(3)]
    #[case(5)]
    #[case(10)]
    fn test_resize_multi_frame(#[case] num_frames: usize) {
        let tmp_dir = TempDir::new().unwrap();
        let source_dir = tmp_dir.path().join("source");
        let output_dir = tmp_dir.path().join("output");
        fs::create_dir(&source_dir).unwrap();
        fs::create_dir(&output_dir).unwrap();

        let input_path = source_dir.join("test.tiff");
        create_test_tiff(&input_path, 64, 64, num_frames);

        let scale = 0.5;
        let args = Args {
            source: source_dir,
            output: output_dir.clone(),
            scale,
            filter: FilterType::Triangle,
            verbose: false,
        };

        let num_files = run(args).unwrap();
        assert_eq!(num_files, 1);

        // Verify output file exists
        let output_path = output_dir.join("test.tiff");
        assert!(output_path.exists());

        // Verify frame count is preserved
        let file = File::open(output_path).unwrap();
        let reader = BufReader::new(file);
        let mut decoder = Decoder::new(reader).unwrap();
        let metadata = PreprocessingMetadata::try_from(&mut decoder).unwrap();
        let frame_count: u16 = metadata.num_frames.into();
        assert_eq!(frame_count as usize, num_frames);

        // Verify dimensions of first frame
        let (width, height) = decoder.dimensions().unwrap();
        assert_eq!(width, 32);
        assert_eq!(height, 32);
    }

    #[test]
    fn test_preserve_directory_structure() {
        let tmp_dir = TempDir::new().unwrap();
        let source_dir = tmp_dir.path().join("source");
        let output_dir = tmp_dir.path().join("output");
        fs::create_dir(&source_dir).unwrap();
        fs::create_dir(&output_dir).unwrap();

        // Create nested directory structure
        let nested_path = source_dir.join("study1").join("series1");
        fs::create_dir_all(&nested_path).unwrap();
        let input_path = nested_path.join("test.tiff");
        create_test_tiff(&input_path, 64, 64, 1);

        let args = Args {
            source: source_dir,
            output: output_dir.clone(),
            scale: 0.5,
            filter: FilterType::Triangle,
            verbose: false,
        };

        run(args).unwrap();

        // Verify output preserves directory structure
        let output_path = output_dir.join("study1").join("series1").join("test.tiff");
        assert!(output_path.exists());
    }

    #[test]
    fn test_multiple_files() {
        let tmp_dir = TempDir::new().unwrap();
        let source_dir = tmp_dir.path().join("source");
        let output_dir = tmp_dir.path().join("output");
        fs::create_dir(&source_dir).unwrap();
        fs::create_dir(&output_dir).unwrap();

        // Create multiple TIFFs
        create_test_tiff(&source_dir.join("test1.tiff"), 64, 64, 1);
        create_test_tiff(&source_dir.join("test2.tiff"), 128, 128, 2);
        create_test_tiff(&source_dir.join("test3.tiff"), 256, 256, 3);

        let args = Args {
            source: source_dir,
            output: output_dir.clone(),
            scale: 0.5,
            filter: FilterType::Triangle,
            verbose: false,
        };

        let num_files = run(args).unwrap();
        assert_eq!(num_files, 3);

        // Verify all output files exist with correct dimensions
        let expected = vec![
            ("test1.tiff", 32, 32, 1),
            ("test2.tiff", 64, 64, 2),
            ("test3.tiff", 128, 128, 3),
        ];

        for (filename, expected_width, expected_height, expected_frames) in expected {
            let output_path = output_dir.join(filename);
            assert!(output_path.exists());

            let file = File::open(output_path).unwrap();
            let reader = BufReader::new(file);
            let mut decoder = Decoder::new(reader).unwrap();
            let (width, height) = decoder.dimensions().unwrap();
            assert_eq!(width, expected_width);
            assert_eq!(height, expected_height);

            let metadata = PreprocessingMetadata::try_from(&mut decoder).unwrap();
            let frame_count: u16 = metadata.num_frames.into();
            assert_eq!(frame_count as usize, expected_frames);
        }
    }

    #[test]
    fn test_invalid_source_path() {
        let tmp_dir = TempDir::new().unwrap();
        let args = Args {
            source: tmp_dir.path().join("nonexistent"),
            output: tmp_dir.path().join("output"),
            scale: 0.5,
            filter: FilterType::Triangle,
            verbose: false,
        };
        assert!(matches!(run(args), Err(Error::InvalidSourcePath { .. })));
    }

    #[test]
    fn test_invalid_scale_factor() {
        let tmp_dir = TempDir::new().unwrap();
        let source_dir = tmp_dir.path().join("source");
        fs::create_dir(&source_dir).unwrap();

        let args = Args {
            source: source_dir,
            output: tmp_dir.path().join("output"),
            scale: -0.5,
            filter: FilterType::Triangle,
            verbose: false,
        };
        assert!(matches!(run(args), Err(Error::InvalidScaleFactor { .. })));

        let args = Args {
            source: tmp_dir.path().join("source"),
            output: tmp_dir.path().join("output"),
            scale: 0.0,
            filter: FilterType::Triangle,
            verbose: false,
        };
        assert!(matches!(run(args), Err(Error::InvalidScaleFactor { .. })));
    }

    #[test]
    fn test_no_sources() {
        let tmp_dir = TempDir::new().unwrap();
        let source_dir = tmp_dir.path().join("source");
        fs::create_dir(&source_dir).unwrap();

        let args = Args {
            source: source_dir,
            output: tmp_dir.path().join("output"),
            scale: 0.5,
            filter: FilterType::Triangle,
            verbose: false,
        };
        assert!(matches!(run(args), Err(Error::NoSources { .. })));
    }

    #[test]
    fn test_compose_resize() {
        let tmp_dir = TempDir::new().unwrap();
        let source_dir = tmp_dir.path().join("source");
        let output_dir = tmp_dir.path().join("output");
        fs::create_dir(&source_dir).unwrap();
        fs::create_dir(&output_dir).unwrap();

        // Create a TIFF with existing resize metadata
        let input_path = source_dir.join("test.tiff");
        let initial_resize = dicom_preprocessing::transform::resize::Resize {
            scale_x: 2.0,
            scale_y: 2.0,
            filter: FilterType::Nearest,
        };
        let metadata = PreprocessingMetadata {
            crop: None,
            resize: Some(initial_resize),
            padding: None,
            resolution: None,
            num_frames: FrameCount::from(1_u16),
        };

        let array = Array4::<u8>::zeros((1, 64, 64, 1));
        let saver = TiffSaver::new(
            Compressor::Uncompressed(Uncompressed),
            DicomColorType::Gray8(tiff::encoder::colortype::Gray8),
        );
        {
            let mut encoder = saver.open_tiff(&input_path).unwrap();
            let dynamic_image = DynamicImage::ImageLuma8(
                image::ImageBuffer::from_raw(64, 64, array.into_raw_vec_and_offset().0).unwrap(),
            );
            saver.save(&mut encoder, &dynamic_image, &metadata).unwrap();
        }

        // Apply another resize
        let new_scale = 0.5;
        let args = Args {
            source: source_dir,
            output: output_dir.clone(),
            scale: new_scale,
            filter: FilterType::Triangle,
            verbose: false,
        };

        run(args).unwrap();

        // Verify composed resize metadata
        let output_path = output_dir.join("test.tiff");
        let file = File::open(output_path).unwrap();
        let reader = BufReader::new(file);
        let mut decoder = Decoder::new(reader).unwrap();
        let metadata = PreprocessingMetadata::try_from(&mut decoder).unwrap();

        assert!(metadata.resize.is_some());
        let resize = metadata.resize.unwrap();
        // Should be 2.0 * 0.5 = 1.0
        assert!((resize.scale_x - 1.0).abs() < TOLERANCE);
        assert!((resize.scale_y - 1.0).abs() < TOLERANCE);
    }

    #[rstest]
    #[case(0.5)]
    #[case(2.0)]
    #[case(0.25)]
    fn test_resize_with_padding(#[case] scale: f32) {
        let tmp_dir = TempDir::new().unwrap();
        let source_dir = tmp_dir.path().join("source");
        let output_dir = tmp_dir.path().join("output");
        fs::create_dir(&source_dir).unwrap();
        fs::create_dir(&output_dir).unwrap();

        // Create a TIFF with padding metadata
        let input_path = source_dir.join("test.tiff");
        let initial_padding = dicom_preprocessing::transform::pad::Padding {
            left: 10,
            top: 20,
            right: 30,
            bottom: 40,
        };
        let metadata = PreprocessingMetadata {
            crop: None,
            resize: None,
            padding: Some(initial_padding),
            resolution: None,
            num_frames: FrameCount::from(1_u16),
        };

        let array = Array4::<u8>::zeros((1, 64, 64, 1));
        let saver = TiffSaver::new(
            Compressor::Uncompressed(Uncompressed),
            DicomColorType::Gray8(tiff::encoder::colortype::Gray8),
        );
        {
            let mut encoder = saver.open_tiff(&input_path).unwrap();
            let dynamic_image = DynamicImage::ImageLuma8(
                image::ImageBuffer::from_raw(64, 64, array.into_raw_vec_and_offset().0).unwrap(),
            );
            saver.save(&mut encoder, &dynamic_image, &metadata).unwrap();
        }

        // Apply resize
        let args = Args {
            source: source_dir,
            output: output_dir.clone(),
            scale,
            filter: FilterType::Triangle,
            verbose: false,
        };

        run(args).unwrap();

        // Verify padding metadata is scaled
        let output_path = output_dir.join("test.tiff");
        let file = File::open(output_path).unwrap();
        let reader = BufReader::new(file);
        let mut decoder = Decoder::new(reader).unwrap();
        let metadata = PreprocessingMetadata::try_from(&mut decoder).unwrap();

        assert!(metadata.padding.is_some());
        let padding = metadata.padding.unwrap();

        // Padding values should be scaled by the scale factor
        let expected_left = (initial_padding.left as f32 * scale).round() as u32;
        let expected_top = (initial_padding.top as f32 * scale).round() as u32;
        let expected_right = (initial_padding.right as f32 * scale).round() as u32;
        let expected_bottom = (initial_padding.bottom as f32 * scale).round() as u32;

        assert_eq!(padding.left, expected_left);
        assert_eq!(padding.top, expected_top);
        assert_eq!(padding.right, expected_right);
        assert_eq!(padding.bottom, expected_bottom);
    }
}
