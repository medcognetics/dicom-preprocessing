use arrow::array::StringArray;
use arrow::error::ArrowError;
use clap::Parser;
use csv::Reader as CsvReader;
use dicom_preprocessing::color::DicomColorType;
use dicom_preprocessing::errors::TiffError;
use dicom_preprocessing::file::default_bar;
use dicom_preprocessing::metadata::PreprocessingMetadata;
use dicom_preprocessing::save::TiffSaver;
use dicom_preprocessing::transform::{FilterType, Resize, Transform};
use image::DynamicImage;
use image::ImageReader;
use indicatif::ParallelProgressIterator;
use parquet::arrow::arrow_reader::ParquetRecordBatchReader;
use parquet::errors::ParquetError;
use rayon::prelude::*;
use snafu::{Report, ResultExt, Snafu, Whatever};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::path::PathBuf;
use tiff::decoder::Decoder;
use tiff::encoder::colortype::{Gray8, RGB8};
use tiff::encoder::compression::{Compressor, Uncompressed};
use tracing::{error, Level};

const BATCH_SIZE: usize = 128;

#[derive(Debug, Snafu)]
enum Error {
    #[snafu(display("Invalid manifest path: {}", path.display()))]
    InvalidManifestPath { path: PathBuf },

    #[snafu(display("Invalid manifest format: {}", path.display()))]
    InvalidManifestFormat { path: PathBuf },

    #[snafu(display("Invalid mask directory: {}", path.display()))]
    InvalidMaskDirectory { path: PathBuf },

    #[snafu(display("No entries found in manifest: {}", path.display()))]
    NoManifestEntries { path: PathBuf },

    #[snafu(display("IO error: {:?}", source))]
    IO {
        #[snafu(source(from(std::io::Error, Box::new)))]
        source: Box<std::io::Error>,
    },

    #[snafu(display("Error reading CSV: {:?}", source))]
    Csv {
        #[snafu(source(from(csv::Error, Box::new)))]
        source: Box<csv::Error>,
    },

    #[snafu(display("Arrow error: {:?}", source))]
    Arrow {
        #[snafu(source(from(ArrowError, Box::new)))]
        source: Box<ArrowError>,
    },

    #[snafu(display("Parquet error: {:?}", source))]
    Parquet {
        #[snafu(source(from(ParquetError, Box::new)))]
        source: Box<ParquetError>,
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

    #[snafu(display("Error reading image: {:?}", source))]
    ImageRead {
        #[snafu(source(from(image::ImageError, Box::new)))]
        source: Box<image::ImageError>,
    },
}

/// Entry from the manifest file
#[derive(Debug, Clone)]
struct ManifestEntry {
    sop_instance_uid: String,
    path: PathBuf,
}

#[derive(Parser, Debug)]
#[command(author = "Scott Chase Waggener", version = env!("CARGO_PKG_VERSION"), about = "Apply preprocessing transforms to masks", long_about = None)]
struct Args {
    #[arg(help = "Manifest file (CSV or Parquet) with preprocessed image paths")]
    manifest: PathBuf,

    #[arg(help = "Directory containing masks (named {sop_instance_uid}.{tiff/png})")]
    masks: PathBuf,

    #[arg(help = "Output directory for preprocessed masks")]
    output: PathBuf,

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

fn validate_paths(args: &Args) -> Result<(), Error> {
    if !args.masks.is_dir() {
        return Err(Error::InvalidMaskDirectory {
            path: args.masks.clone(),
        });
    }
    if !args.output.is_dir() {
        std::fs::create_dir_all(&args.output).context(IOSnafu)?;
    }
    Ok(())
}

fn load_manifest_csv(path: &Path) -> Result<Vec<ManifestEntry>, Error> {
    let mut reader = CsvReader::from_path(path).context(CsvSnafu)?;
    let mut entries = Vec::new();

    for result in reader.deserialize() {
        let record: HashMap<String, String> = result.context(CsvSnafu)?;
        let sop_instance_uid = record
            .get("sop_instance_uid")
            .ok_or(Error::InvalidManifestFormat {
                path: path.to_path_buf(),
            })?
            .clone();
        let file_path = record.get("path").ok_or(Error::InvalidManifestFormat {
            path: path.to_path_buf(),
        })?;
        entries.push(ManifestEntry {
            sop_instance_uid,
            path: PathBuf::from(file_path),
        });
    }

    Ok(entries)
}

fn load_manifest_parquet(path: &Path) -> Result<Vec<ManifestEntry>, Error> {
    let file = File::open(path).context(IOSnafu)?;
    let reader = ParquetRecordBatchReader::try_new(file, BATCH_SIZE).context(ParquetSnafu)?;
    let mut entries = Vec::new();

    for result in reader {
        let batch = result.context(ArrowSnafu)?;
        let sop_array = batch
            .column_by_name("sop_instance_uid")
            .ok_or(Error::InvalidManifestFormat {
                path: path.to_path_buf(),
            })?
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or(Error::InvalidManifestFormat {
                path: path.to_path_buf(),
            })?;
        let path_array = batch
            .column_by_name("path")
            .ok_or(Error::InvalidManifestFormat {
                path: path.to_path_buf(),
            })?
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or(Error::InvalidManifestFormat {
                path: path.to_path_buf(),
            })?;

        for i in 0..batch.num_rows() {
            entries.push(ManifestEntry {
                sop_instance_uid: sop_array.value(i).to_string(),
                path: PathBuf::from(path_array.value(i)),
            });
        }
    }

    Ok(entries)
}

fn load_manifest(path: &Path) -> Result<Vec<ManifestEntry>, Error> {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some("csv") => load_manifest_csv(path),
        Some("parquet") => load_manifest_parquet(path),
        _ => Err(Error::InvalidManifestFormat {
            path: path.to_path_buf(),
        }),
    }
}

/// Find mask file for a given SOP Instance UID in the mask directory
fn find_mask(mask_dir: &Path, sop_instance_uid: &str) -> Option<PathBuf> {
    for ext in ["tiff", "tif", "png", "TIFF", "TIF", "PNG"] {
        let path = mask_dir.join(format!("{sop_instance_uid}.{ext}"));
        if path.exists() {
            return Some(path);
        }
    }
    None
}

/// Load mask image from file (supports PNG and TIFF)
fn load_mask(path: &Path) -> Result<DynamicImage, Error> {
    let reader = ImageReader::open(path)
        .context(IOSnafu)?
        .with_guessed_format()
        .context(IOSnafu)?;
    reader.decode().context(ImageReadSnafu)
}

/// Apply preprocessing transforms to a mask image
fn apply_transforms(image: &DynamicImage, metadata: &PreprocessingMetadata) -> DynamicImage {
    // Apply crop
    let image = metadata
        .crop
        .as_ref()
        .map(|crop| crop.apply(image))
        .unwrap_or_else(|| image.clone());

    // Apply resize with nearest neighbor interpolation
    let image = metadata
        .resize
        .as_ref()
        .map(|resize| {
            // Create a new resize with nearest neighbor filter
            let nearest_resize = Resize {
                scale_x: resize.scale_x,
                scale_y: resize.scale_y,
                filter: FilterType::Nearest,
            };
            nearest_resize.apply(&image)
        })
        .unwrap_or(image);

    // Apply padding
    metadata
        .padding
        .as_ref()
        .map(|padding| padding.apply(&image))
        .unwrap_or(image)
}

/// Process a single mask file
fn process_mask(
    entry: &ManifestEntry,
    source_dir: &Path,
    mask_dir: &Path,
    output_dir: &Path,
) -> Result<Option<PathBuf>, Error> {
    // Find the corresponding mask
    let mask_path = match find_mask(mask_dir, &entry.sop_instance_uid) {
        Some(path) => path,
        None => {
            tracing::debug!(
                "No mask found for SOP Instance UID: {}",
                entry.sop_instance_uid
            );
            return Ok(None);
        }
    };

    // Load the preprocessed TIFF to get metadata
    let source_path = source_dir.join(&entry.path);
    let file = File::open(&source_path).context(IOSnafu)?;
    let reader = BufReader::new(file);
    let mut decoder = Decoder::new(reader)
        .map_err(TiffError::from)
        .context(TiffReadSnafu)?;
    let metadata = PreprocessingMetadata::try_from(&mut decoder).context(TiffReadSnafu)?;

    // Load and transform the mask
    let mask = load_mask(&mask_path)?;
    let transformed = apply_transforms(&mask, &metadata);

    // Determine output path (preserve directory structure from manifest path)
    let output_path = output_dir.join(entry.path.with_extension("tiff"));
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent).context(IOSnafu)?;
    }

    // Determine color type based on input mask
    let color = match transformed.color() {
        image::ColorType::L8 | image::ColorType::La8 => DicomColorType::Gray8(Gray8),
        image::ColorType::Rgb8 | image::ColorType::Rgba8 => DicomColorType::RGB8(RGB8),
        image::ColorType::L16 | image::ColorType::La16 => DicomColorType::Gray8(Gray8),
        image::ColorType::Rgb16 | image::ColorType::Rgba16 => DicomColorType::RGB8(RGB8),
        _ => DicomColorType::Gray8(Gray8),
    };

    // Convert to appropriate format
    let transformed = match &color {
        DicomColorType::Gray8(_) => DynamicImage::ImageLuma8(transformed.into_luma8()),
        DicomColorType::RGB8(_) => DynamicImage::ImageRgb8(transformed.into_rgb8()),
        DicomColorType::Gray16(_) => DynamicImage::ImageLuma8(transformed.into_luma8()),
    };

    // Save the transformed mask
    let saver = TiffSaver::new(Compressor::Uncompressed(Uncompressed), color);
    let mut encoder = saver.open_tiff(&output_path).context(TiffWriteSnafu)?;
    saver
        .save(&mut encoder, &transformed, &metadata)
        .context(TiffWriteSnafu)?;

    tracing::debug!(
        "Processed mask for {} -> {}",
        entry.sop_instance_uid,
        output_path.display()
    );

    Ok(Some(output_path))
}

fn run(args: Args) -> Result<usize, Error> {
    validate_paths(&args)?;

    // Load manifest
    if !args.manifest.is_file() {
        return Err(Error::InvalidManifestPath {
            path: args.manifest.clone(),
        });
    }
    let entries = load_manifest(&args.manifest)?;
    if entries.is_empty() {
        return Err(Error::NoManifestEntries {
            path: args.manifest.clone(),
        });
    }

    tracing::info!("Loaded {} entries from manifest", entries.len());

    // Determine source directory from manifest path (assuming manifest is in or relative to source)
    let source_dir = args
        .manifest
        .parent()
        .unwrap_or(Path::new("."))
        .to_path_buf();

    // Process masks in parallel
    let pb = default_bar(entries.len() as u64);
    pb.set_message("Processing masks");

    let results: Vec<_> = entries
        .par_iter()
        .progress_with(pb)
        .map(|entry| process_mask(entry, &source_dir, &args.masks, &args.output))
        .collect::<Result<Vec<_>, _>>()?;

    let processed_count = results.iter().filter(|r| r.is_some()).count();
    println!(
        "Processed {} masks out of {} manifest entries",
        processed_count,
        entries.len()
    );

    Ok(processed_count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dicom_preprocessing::metadata::WriteTags;
    use dicom_preprocessing::transform::{Crop, Padding};
    use dicom_preprocessing::FrameCount;
    use image::{GrayImage, Luma, Rgb, RgbImage};
    use rstest::rstest;
    use std::fs;
    use tempfile::TempDir;
    use tiff::encoder::TiffEncoder;

    /// Create a test preprocessed TIFF with metadata
    fn create_preprocessed_tiff(
        path: &Path,
        width: u32,
        height: u32,
        crop: Option<Crop>,
        resize: Option<Resize>,
        padding: Option<Padding>,
    ) {
        let metadata = PreprocessingMetadata {
            crop,
            resize,
            padding,
            resolution: None,
            num_frames: FrameCount::from(1_u16),
        };

        let file = File::create(path).unwrap();
        let mut encoder = TiffEncoder::new(file).unwrap();
        let mut img = encoder
            .new_image::<tiff::encoder::colortype::Gray8>(width, height)
            .unwrap();

        metadata.write_tags(&mut img).unwrap();
        let data: Vec<u8> = vec![0; (width * height) as usize];
        img.write_data(&data).unwrap();
    }

    /// Create a test grayscale mask PNG
    fn create_gray_mask_png(
        path: &Path,
        width: u32,
        height: u32,
        pattern: impl Fn(u32, u32) -> u8,
    ) {
        let mut img = GrayImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                img.put_pixel(x, y, Luma([pattern(x, y)]));
            }
        }
        img.save(path).unwrap();
    }

    /// Create a test RGB mask PNG
    fn create_rgb_mask_png(
        path: &Path,
        width: u32,
        height: u32,
        pattern: impl Fn(u32, u32) -> [u8; 3],
    ) {
        let mut img = RgbImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let [r, g, b] = pattern(x, y);
                img.put_pixel(x, y, Rgb([r, g, b]));
            }
        }
        img.save(path).unwrap();
    }

    /// Create a test grayscale mask TIFF
    fn create_gray_mask_tiff(
        path: &Path,
        width: u32,
        height: u32,
        pattern: impl Fn(u32, u32) -> u8,
    ) {
        let mut img = GrayImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                img.put_pixel(x, y, Luma([pattern(x, y)]));
            }
        }
        img.save(path).unwrap();
    }

    /// Create a manifest CSV
    fn create_manifest_csv(path: &Path, entries: &[(&str, &str)]) {
        let mut writer = csv::Writer::from_path(path).unwrap();
        writer
            .write_record([
                "sop_instance_uid",
                "path",
                "study_instance_uid",
                "series_instance_uid",
            ])
            .unwrap();
        for (sop_uid, file_path) in entries {
            writer
                .write_record([*sop_uid, *file_path, "study1", "series1"])
                .unwrap();
        }
        writer.flush().unwrap();
    }

    #[test]
    fn test_find_mask_png() {
        let tmp_dir = TempDir::new().unwrap();
        let mask_path = tmp_dir.path().join("test_sop.png");
        fs::write(&mask_path, b"dummy").unwrap();

        let found = find_mask(tmp_dir.path(), "test_sop");
        assert!(found.is_some());
        assert_eq!(found.unwrap(), mask_path);
    }

    #[test]
    fn test_find_mask_tiff() {
        let tmp_dir = TempDir::new().unwrap();
        let mask_path = tmp_dir.path().join("test_sop.tiff");
        fs::write(&mask_path, b"dummy").unwrap();

        let found = find_mask(tmp_dir.path(), "test_sop");
        assert!(found.is_some());
        assert_eq!(found.unwrap(), mask_path);
    }

    #[test]
    fn test_find_mask_not_found() {
        let tmp_dir = TempDir::new().unwrap();
        let found = find_mask(tmp_dir.path(), "nonexistent");
        assert!(found.is_none());
    }

    #[rstest]
    #[case(None, None, None, 100, 100)] // No transforms
    #[case(
        Some(Crop { left: 10, top: 10, width: 80, height: 80 }),
        None,
        None,
        80,
        80
    )] // Crop only
    #[case(
        None,
        Some(Resize { scale_x: 0.5, scale_y: 0.5, filter: FilterType::Triangle }),
        None,
        50,
        50
    )] // Resize only
    #[case(
        None,
        None,
        Some(Padding { left: 10, top: 10, right: 10, bottom: 10 }),
        120,
        120
    )] // Padding only
    fn test_apply_transforms_dimensions(
        #[case] crop: Option<Crop>,
        #[case] resize: Option<Resize>,
        #[case] padding: Option<Padding>,
        #[case] expected_width: u32,
        #[case] expected_height: u32,
    ) {
        let img = DynamicImage::new_luma8(100, 100);
        let metadata = PreprocessingMetadata {
            crop,
            resize,
            padding,
            resolution: None,
            num_frames: FrameCount::from(1_u16),
        };

        let result = apply_transforms(&img, &metadata);
        assert_eq!(result.width(), expected_width);
        assert_eq!(result.height(), expected_height);
    }

    #[test]
    fn test_apply_transforms_combined() {
        // 100x100 image -> crop to 80x80 -> resize by 0.5 to 40x40 -> pad by 10 to 60x60
        let img = DynamicImage::new_luma8(100, 100);
        let metadata = PreprocessingMetadata {
            crop: Some(Crop {
                left: 10,
                top: 10,
                width: 80,
                height: 80,
            }),
            resize: Some(Resize {
                scale_x: 0.5,
                scale_y: 0.5,
                filter: FilterType::Triangle,
            }),
            padding: Some(Padding {
                left: 10,
                top: 10,
                right: 10,
                bottom: 10,
            }),
            resolution: None,
            num_frames: FrameCount::from(1_u16),
        };

        let result = apply_transforms(&img, &metadata);
        assert_eq!(result.width(), 60);
        assert_eq!(result.height(), 60);
    }

    #[test]
    fn test_apply_transforms_uses_nearest_neighbor() {
        // Create an image with a checkerboard pattern to verify nearest neighbor
        let mut img = GrayImage::new(4, 4);
        for y in 0..4 {
            for x in 0..4 {
                let value = if (x + y) % 2 == 0 { 255 } else { 0 };
                img.put_pixel(x, y, Luma([value]));
            }
        }
        let img = DynamicImage::ImageLuma8(img);

        // Scale up by 2x - nearest neighbor should preserve sharp edges
        let metadata = PreprocessingMetadata {
            crop: None,
            resize: Some(Resize {
                scale_x: 2.0,
                scale_y: 2.0,
                filter: FilterType::Triangle, // This should be overridden to Nearest
            }),
            padding: None,
            resolution: None,
            num_frames: FrameCount::from(1_u16),
        };

        let result = apply_transforms(&img, &metadata);
        let result = result.into_luma8();

        // Check that we get a scaled checkerboard (each original pixel becomes 2x2)
        assert_eq!(result.width(), 8);
        assert_eq!(result.height(), 8);

        // Verify the pattern is preserved (not interpolated)
        // Original pixel (0,0) = 255, so pixels (0,0), (0,1), (1,0), (1,1) should all be 255
        assert_eq!(result.get_pixel(0, 0)[0], 255);
        assert_eq!(result.get_pixel(1, 0)[0], 255);
        assert_eq!(result.get_pixel(0, 1)[0], 255);
        assert_eq!(result.get_pixel(1, 1)[0], 255);

        // Original pixel (1,0) = 0, so pixels (2,0), (2,1), (3,0), (3,1) should all be 0
        assert_eq!(result.get_pixel(2, 0)[0], 0);
        assert_eq!(result.get_pixel(3, 0)[0], 0);
        assert_eq!(result.get_pixel(2, 1)[0], 0);
        assert_eq!(result.get_pixel(3, 1)[0], 0);
    }

    #[test]
    fn test_load_manifest_csv() {
        let tmp_dir = TempDir::new().unwrap();
        let manifest_path = tmp_dir.path().join("manifest.csv");

        create_manifest_csv(
            &manifest_path,
            &[
                ("sop1", "study1/series1/sop1.tiff"),
                ("sop2", "study1/series1/sop2.tiff"),
            ],
        );

        let entries = load_manifest(&manifest_path).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].sop_instance_uid, "sop1");
        assert_eq!(entries[0].path, PathBuf::from("study1/series1/sop1.tiff"));
        assert_eq!(entries[1].sop_instance_uid, "sop2");
    }

    #[test]
    fn test_process_mask_grayscale_png() {
        let tmp_dir = TempDir::new().unwrap();

        // Create source directory structure
        let source_dir = tmp_dir.path().join("source");
        let study_series = source_dir.join("study1").join("series1");
        fs::create_dir_all(&study_series).unwrap();

        // Create preprocessed TIFF
        create_preprocessed_tiff(
            &study_series.join("sop1.tiff"),
            50,
            50,
            Some(Crop {
                left: 10,
                top: 10,
                width: 80,
                height: 80,
            }),
            Some(Resize {
                scale_x: 0.625,
                scale_y: 0.625,
                filter: FilterType::Triangle,
            }),
            None,
        );

        // Create mask directory
        let mask_dir = tmp_dir.path().join("masks");
        fs::create_dir(&mask_dir).unwrap();

        // Create mask (100x100 with pattern)
        create_gray_mask_png(&mask_dir.join("sop1.png"), 100, 100, |x, y| {
            ((x + y) % 256) as u8
        });

        // Create output directory
        let output_dir = tmp_dir.path().join("output");
        fs::create_dir(&output_dir).unwrap();

        // Process
        let entry = ManifestEntry {
            sop_instance_uid: "sop1".to_string(),
            path: PathBuf::from("study1/series1/sop1.tiff"),
        };

        let result = process_mask(&entry, &source_dir, &mask_dir, &output_dir).unwrap();
        assert!(result.is_some());

        let output_path = result.unwrap();
        assert!(output_path.exists());

        // Verify output dimensions
        let output = ImageReader::open(&output_path).unwrap().decode().unwrap();
        assert_eq!(output.width(), 50);
        assert_eq!(output.height(), 50);
    }

    #[test]
    fn test_process_mask_rgb_png() {
        let tmp_dir = TempDir::new().unwrap();

        // Create source directory structure
        let source_dir = tmp_dir.path().join("source");
        let study_series = source_dir.join("study1").join("series1");
        fs::create_dir_all(&study_series).unwrap();

        // Create preprocessed TIFF (no transforms)
        create_preprocessed_tiff(&study_series.join("sop1.tiff"), 100, 100, None, None, None);

        // Create mask directory
        let mask_dir = tmp_dir.path().join("masks");
        fs::create_dir(&mask_dir).unwrap();

        // Create RGB mask
        create_rgb_mask_png(&mask_dir.join("sop1.png"), 100, 100, |x, y| {
            [(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8]
        });

        // Create output directory
        let output_dir = tmp_dir.path().join("output");
        fs::create_dir(&output_dir).unwrap();

        // Process
        let entry = ManifestEntry {
            sop_instance_uid: "sop1".to_string(),
            path: PathBuf::from("study1/series1/sop1.tiff"),
        };

        let result = process_mask(&entry, &source_dir, &mask_dir, &output_dir).unwrap();
        assert!(result.is_some());

        let output_path = result.unwrap();
        let output = ImageReader::open(&output_path).unwrap().decode().unwrap();
        assert_eq!(output.width(), 100);
        assert_eq!(output.height(), 100);
        // Verify it's RGB
        assert!(matches!(output.color(), image::ColorType::Rgb8));
    }

    #[test]
    fn test_process_mask_tiff_input() {
        let tmp_dir = TempDir::new().unwrap();

        // Create source directory structure
        let source_dir = tmp_dir.path().join("source");
        let study_series = source_dir.join("study1").join("series1");
        fs::create_dir_all(&study_series).unwrap();

        // Create preprocessed TIFF
        create_preprocessed_tiff(&study_series.join("sop1.tiff"), 100, 100, None, None, None);

        // Create mask directory
        let mask_dir = tmp_dir.path().join("masks");
        fs::create_dir(&mask_dir).unwrap();

        // Create TIFF mask
        create_gray_mask_tiff(&mask_dir.join("sop1.tiff"), 100, 100, |x, _| {
            (x % 256) as u8
        });

        // Create output directory
        let output_dir = tmp_dir.path().join("output");
        fs::create_dir(&output_dir).unwrap();

        // Process
        let entry = ManifestEntry {
            sop_instance_uid: "sop1".to_string(),
            path: PathBuf::from("study1/series1/sop1.tiff"),
        };

        let result = process_mask(&entry, &source_dir, &mask_dir, &output_dir).unwrap();
        assert!(result.is_some());
    }

    #[test]
    fn test_process_mask_not_found() {
        let tmp_dir = TempDir::new().unwrap();

        // Create source directory structure
        let source_dir = tmp_dir.path().join("source");
        let study_series = source_dir.join("study1").join("series1");
        fs::create_dir_all(&study_series).unwrap();

        // Create preprocessed TIFF
        create_preprocessed_tiff(&study_series.join("sop1.tiff"), 100, 100, None, None, None);

        // Create empty mask directory
        let mask_dir = tmp_dir.path().join("masks");
        fs::create_dir(&mask_dir).unwrap();

        // Create output directory
        let output_dir = tmp_dir.path().join("output");
        fs::create_dir(&output_dir).unwrap();

        // Process - should return None since mask doesn't exist
        let entry = ManifestEntry {
            sop_instance_uid: "sop1".to_string(),
            path: PathBuf::from("study1/series1/sop1.tiff"),
        };

        let result = process_mask(&entry, &source_dir, &mask_dir, &output_dir).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_end_to_end_with_csv_manifest() {
        let tmp_dir = TempDir::new().unwrap();

        // Create source directory structure
        let source_dir = tmp_dir.path().join("source");
        let study_series = source_dir.join("study1").join("series1");
        fs::create_dir_all(&study_series).unwrap();

        // Create preprocessed TIFFs
        create_preprocessed_tiff(
            &study_series.join("sop1.tiff"),
            50,
            50,
            Some(Crop {
                left: 0,
                top: 0,
                width: 100,
                height: 100,
            }),
            Some(Resize {
                scale_x: 0.5,
                scale_y: 0.5,
                filter: FilterType::Triangle,
            }),
            None,
        );
        create_preprocessed_tiff(&study_series.join("sop2.tiff"), 100, 100, None, None, None);

        // Create manifest
        let manifest_path = source_dir.join("manifest.csv");
        create_manifest_csv(
            &manifest_path,
            &[
                ("sop1", "study1/series1/sop1.tiff"),
                ("sop2", "study1/series1/sop2.tiff"),
            ],
        );

        // Create mask directory with only one mask
        let mask_dir = tmp_dir.path().join("masks");
        fs::create_dir(&mask_dir).unwrap();
        create_gray_mask_png(&mask_dir.join("sop1.png"), 100, 100, |_, _| 128);

        // Create output directory
        let output_dir = tmp_dir.path().join("output");
        fs::create_dir(&output_dir).unwrap();

        // Run
        let args = Args {
            manifest: manifest_path,
            masks: mask_dir,
            output: output_dir.clone(),
            verbose: true,
        };

        let count = run(args).unwrap();
        assert_eq!(count, 1); // Only sop1 has a mask

        // Verify output
        let output_path = output_dir.join("study1/series1/sop1.tiff");
        assert!(output_path.exists());
    }

    #[test]
    fn test_crop_transform_preserves_content() {
        // Create a mask with a specific pattern
        let mut img = GrayImage::new(100, 100);
        // Mark center region with 255
        for y in 25..75 {
            for x in 25..75 {
                img.put_pixel(x, y, Luma([255]));
            }
        }
        let img = DynamicImage::ImageLuma8(img);

        // Crop to center 50x50
        let metadata = PreprocessingMetadata {
            crop: Some(Crop {
                left: 25,
                top: 25,
                width: 50,
                height: 50,
            }),
            resize: None,
            padding: None,
            resolution: None,
            num_frames: FrameCount::from(1_u16),
        };

        let result = apply_transforms(&img, &metadata);
        let result = result.into_luma8();

        // All pixels in result should be 255
        for y in 0..50 {
            for x in 0..50 {
                assert_eq!(
                    result.get_pixel(x, y)[0],
                    255,
                    "Pixel at ({x}, {y}) should be 255"
                );
            }
        }
    }

    #[test]
    fn test_padding_fills_with_black() {
        let img = DynamicImage::new_luma8(50, 50);
        let img = {
            let mut gray = img.into_luma8();
            for y in 0..50 {
                for x in 0..50 {
                    gray.put_pixel(x, y, Luma([255]));
                }
            }
            DynamicImage::ImageLuma8(gray)
        };

        let metadata = PreprocessingMetadata {
            crop: None,
            resize: None,
            padding: Some(Padding {
                left: 10,
                top: 10,
                right: 10,
                bottom: 10,
            }),
            resolution: None,
            num_frames: FrameCount::from(1_u16),
        };

        let result = apply_transforms(&img, &metadata);
        let result = result.into_luma8();

        assert_eq!(result.width(), 70);
        assert_eq!(result.height(), 70);

        // Check padding area is black (0)
        for y in 0..10 {
            for x in 0..70 {
                assert_eq!(result.get_pixel(x, y)[0], 0, "Top padding should be black");
            }
        }

        // Check original content is preserved
        for y in 10..60 {
            for x in 10..60 {
                assert_eq!(
                    result.get_pixel(x, y)[0],
                    255,
                    "Original content should be preserved"
                );
            }
        }
    }

    #[test]
    fn test_run_skips_missing_masks_gracefully() {
        let tmp_dir = TempDir::new().unwrap();

        // Create source directory structure
        let source_dir = tmp_dir.path().join("source");
        let study_series = source_dir.join("study1").join("series1");
        fs::create_dir_all(&study_series).unwrap();

        // Create preprocessed TIFFs for multiple entries
        for sop in ["sop1", "sop2", "sop3", "sop4"] {
            create_preprocessed_tiff(
                &study_series.join(format!("{sop}.tiff")),
                100,
                100,
                None,
                None,
                None,
            );
        }

        // Create manifest with all 4 entries
        let manifest_path = source_dir.join("manifest.csv");
        create_manifest_csv(
            &manifest_path,
            &[
                ("sop1", "study1/series1/sop1.tiff"),
                ("sop2", "study1/series1/sop2.tiff"),
                ("sop3", "study1/series1/sop3.tiff"),
                ("sop4", "study1/series1/sop4.tiff"),
            ],
        );

        // Create mask directory with only 2 masks (sop1 and sop3)
        let mask_dir = tmp_dir.path().join("masks");
        fs::create_dir(&mask_dir).unwrap();
        create_gray_mask_png(&mask_dir.join("sop1.png"), 100, 100, |_, _| 128);
        create_gray_mask_png(&mask_dir.join("sop3.png"), 100, 100, |_, _| 64);

        // Create output directory
        let output_dir = tmp_dir.path().join("output");
        fs::create_dir(&output_dir).unwrap();

        // Run - should succeed without errors despite missing masks
        let args = Args {
            manifest: manifest_path,
            masks: mask_dir,
            output: output_dir.clone(),
            verbose: false,
        };

        let count = run(args).unwrap();

        // Only 2 masks existed, so only 2 should be processed
        assert_eq!(count, 2);

        // Verify only the entries with masks have output files
        assert!(output_dir.join("study1/series1/sop1.tiff").exists());
        assert!(!output_dir.join("study1/series1/sop2.tiff").exists());
        assert!(output_dir.join("study1/series1/sop3.tiff").exists());
        assert!(!output_dir.join("study1/series1/sop4.tiff").exists());
    }
}
