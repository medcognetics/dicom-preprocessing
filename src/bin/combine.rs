use arrow::array::{Array, Int32Array, Int64Array, StringArray};
use arrow::datatypes::DataType;
use arrow::error::ArrowError;

use clap::Parser;
use csv::Reader as CsvReader;
use dicom_preprocessing::color::DicomColorType;
use dicom_preprocessing::errors::TiffError;
use dicom_preprocessing::file::default_bar;
use dicom_preprocessing::file::{InodeSort, TiffFileOperations};
use dicom_preprocessing::load::load_frames_as_dynamic_images;
use dicom_preprocessing::metadata::PreprocessingMetadata;
use dicom_preprocessing::save::TiffSaver;
use indicatif::ParallelProgressIterator;

use parquet::arrow::arrow_reader::ParquetRecordBatchReader;
use rayon::prelude::*;
use snafu::{Report, ResultExt, Snafu, Whatever};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::num::ParseIntError;
use std::path::{Path, PathBuf};
use tiff::decoder::Decoder;
use tiff::encoder::compression::{Compressor, Uncompressed};
use tracing::{error, info, Level};

const BATCH_SIZE: usize = 128;

#[derive(Debug, Snafu)]
enum Error {
    #[snafu(display("Invalid source path: {}", path.display()))]
    InvalidSourcePath { path: PathBuf },

    #[snafu(display("No TIFF files found in source path: {}", path.display()))]
    NoSources { path: PathBuf },

    #[snafu(display("Invalid metadata format: {}", path.display()))]
    InvalidMetadataFormat { path: PathBuf },

    #[snafu(display("Invalid output path: {}", path.display()))]
    InvalidOutputPath { path: PathBuf },

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

    #[snafu(display("Error parsing integer: {:?}", source))]
    ParseInt {
        #[snafu(source(from(ParseIntError, Box::new)))]
        source: Box<ParseIntError>,
    },

    #[snafu(display("Arrow error: {:?}", source))]
    Arrow {
        #[snafu(source(from(ArrowError, Box::new)))]
        source: Box<ArrowError>,
    },

    #[snafu(display("Parquet error: {:?}", source))]
    Parquet {
        #[snafu(source(from(parquet::errors::ParquetError, Box::new)))]
        source: Box<parquet::errors::ParquetError>,
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

#[derive(Debug, Clone)]
struct FrameMetadata {
    series_instance_uid: String,
    sop_instance_uid: String,
    instance_number: i32,
}

impl FrameMetadata {
    pub fn new(
        series_instance_uid: String,
        sop_instance_uid: String,
        instance_number: i32,
    ) -> Self {
        Self {
            series_instance_uid,
            sop_instance_uid,
            instance_number,
        }
    }

    #[allow(dead_code)]
    pub fn series_instance_uid(&self) -> &str {
        &self.series_instance_uid
    }

    pub fn sop_instance_uid(&self) -> &str {
        &self.sop_instance_uid
    }

    pub fn instance_number(&self) -> i32 {
        self.instance_number
    }
}

#[derive(Parser, Debug)]
#[command(
    author = "Scott Chase Waggener", 
    version = env!("CARGO_PKG_VERSION"), 
    about = "Combine single-frame TIFF files into multi-frame TIFFs", 
    long_about = None
)]
struct Args {
    #[arg(
        help = "Directory containing TIFF files with structure study_instance_uid/series_instance_uid/sop_instance_uid.tiff"
    )]
    source: PathBuf,

    #[arg(
        help = "CSV or Parquet file with series_instance_uid, sop_instance_uid, and instance_number columns"
    )]
    metadata: PathBuf,

    #[arg(help = "Output directory for combined multi-frame TIFFs")]
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
    if !args.source.is_dir() {
        return Err(Error::InvalidSourcePath {
            path: args.source.clone(),
        });
    }

    if !args.metadata.is_file() {
        return Err(Error::InvalidMetadataFormat {
            path: args.metadata.clone(),
        });
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

fn load_metadata_csv(path: &PathBuf) -> Result<HashMap<String, Vec<FrameMetadata>>, Error> {
    let mut reader = CsvReader::from_path(path).context(CsvSnafu)?;
    let mut metadata_map = HashMap::new();

    for result in reader.deserialize() {
        let record: HashMap<String, String> = result.context(CsvSnafu)?;
        let series_instance_uid = record["series_instance_uid"].clone();
        let sop_instance_uid = record["sop_instance_uid"].clone();
        let instance_number = record["instance_number"]
            .parse::<i32>()
            .context(ParseIntSnafu)?;

        let frame_metadata = FrameMetadata::new(
            series_instance_uid.clone(),
            sop_instance_uid,
            instance_number,
        );

        metadata_map
            .entry(series_instance_uid)
            .or_insert_with(Vec::new)
            .push(frame_metadata);
    }

    Ok(metadata_map)
}

fn load_metadata_parquet(path: &PathBuf) -> Result<HashMap<String, Vec<FrameMetadata>>, Error> {
    let file = File::open(path).context(IOSnafu)?;
    let reader = ParquetRecordBatchReader::try_new(file, BATCH_SIZE).context(ParquetSnafu)?;
    let mut metadata_map = HashMap::new();

    for result in reader {
        let batch = result.context(ArrowSnafu)?;
        let series_array = batch
            .column_by_name("series_instance_uid")
            .expect("Failed to get series_instance_uid column")
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("Failed to downcast series_instance_uid to StringArray");
        let sop_array = batch
            .column_by_name("sop_instance_uid")
            .expect("Failed to get sop_instance_uid column")
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("Failed to downcast sop_instance_uid to StringArray");
        let instance_column = batch
            .column_by_name("instance_number")
            .expect("Failed to get instance_number column");

        for i in 0..batch.num_rows() {
            let series_instance_uid = series_array.value(i).to_string();
            let sop_instance_uid = sop_array.value(i).to_string();

            // Handle different integer types for instance_number
            let instance_number = match instance_column.data_type() {
                DataType::Int32 => {
                    let array = instance_column
                        .as_any()
                        .downcast_ref::<Int32Array>()
                        .expect("Failed to downcast instance_number to Int32Array");
                    array.value(i)
                }
                DataType::Int64 => {
                    let array = instance_column
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .expect("Failed to downcast instance_number to Int64Array");
                    array.value(i) as i32
                }
                _ => panic!(
                    "Unsupported data type for instance_number: {:?}",
                    instance_column.data_type()
                ),
            };

            let frame_metadata = FrameMetadata::new(
                series_instance_uid.clone(),
                sop_instance_uid,
                instance_number,
            );

            metadata_map
                .entry(series_instance_uid)
                .or_insert_with(Vec::new)
                .push(frame_metadata);
        }
    }

    Ok(metadata_map)
}

fn load_metadata(path: &PathBuf) -> Result<HashMap<String, Vec<FrameMetadata>>, Error> {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some("csv") => load_metadata_csv(path),
        Some("parquet") => load_metadata_parquet(path),
        _ => Err(Error::InvalidMetadataFormat { path: path.clone() }),
    }
}

fn extract_path_components(path: &Path) -> Option<(String, String, String)> {
    // Extract study_instance_uid/series_instance_uid/sop_instance_uid from path
    let sop_instance_uid = path.file_stem()?.to_str()?.to_string();
    let series_instance_uid = path.parent()?.file_name()?.to_str()?.to_string();
    let study_instance_uid = path.parent()?.parent()?.file_name()?.to_str()?.to_string();

    Some((study_instance_uid, series_instance_uid, sop_instance_uid))
}

fn combine_series(
    source_files: Vec<PathBuf>,
    metadata_map: &HashMap<String, Vec<FrameMetadata>>,
    output_path: &Path,
) -> Result<(usize, usize), Error> {
    // Group files by series
    let mut series_files: HashMap<String, Vec<PathBuf>> = HashMap::new();
    for file_path in source_files {
        if let Some((_, series_instance_uid, _)) = extract_path_components(&file_path) {
            series_files
                .entry(series_instance_uid)
                .or_default()
                .push(file_path);
        }
    }

    let pb = default_bar(series_files.len() as u64);
    pb.set_message("Combining series into multi-frame TIFFs");

    let results: Result<Vec<_>, Error> = series_files
        .into_par_iter()
        .progress_with(pb)
        .map(|(series_uid, files)| {
            combine_single_series(&series_uid, files, metadata_map, output_path)
        })
        .collect();

    let results = results?;
    let num_series = results.len();
    let num_files: usize = results.iter().sum();

    println!(
        "Combined {} series with {} total files",
        num_series, num_files
    );

    Ok((num_series, num_files))
}

fn combine_single_series(
    series_uid: &str,
    files: Vec<PathBuf>,
    metadata_map: &HashMap<String, Vec<FrameMetadata>>,
    output_path: &Path,
) -> Result<usize, Error> {
    // Get metadata for this series
    let frame_metadata = match metadata_map.get(series_uid) {
        Some(metadata) => metadata,
        None => {
            tracing::debug!("No metadata found for series {}", series_uid);
            return Ok(0);
        }
    };

    // Create a mapping from sop_instance_uid to instance_number
    let mut sop_to_instance: HashMap<String, i32> = HashMap::new();
    for metadata in frame_metadata {
        sop_to_instance.insert(
            metadata.sop_instance_uid().to_string(),
            metadata.instance_number(),
        );
    }

    // Filter and sort files by instance number
    let mut ordered_files: Vec<(PathBuf, i32)> = Vec::new();
    for file_path in files {
        if let Some((_, _, sop_instance_uid)) = extract_path_components(&file_path) {
            if let Some(&instance_number) = sop_to_instance.get(&sop_instance_uid) {
                ordered_files.push((file_path, instance_number));
            }
        }
    }

    if ordered_files.is_empty() {
        tracing::debug!("No matching files found for series {}", series_uid);
        return Ok(0);
    }

    // Sort by instance number
    ordered_files.sort_by_key(|(_, instance_number)| *instance_number);

    // Load all frames
    let mut images = Vec::new();
    let mut reference_metadata = None;
    let mut color_type = None;
    let mut study_uid = None;

    for (file_path, _) in &ordered_files {
        // Extract study UID from the first file
        if study_uid.is_none() {
            if let Some((study_instance_uid, _, _)) = extract_path_components(file_path) {
                study_uid = Some(study_instance_uid);
            }
        }

        let file = File::open(file_path).context(IOSnafu)?;
        let reader = BufReader::new(file);
        let mut decoder = Decoder::new(reader)
            .map_err(TiffError::from)
            .context(TiffReadSnafu)?;

        // Read metadata from the first frame
        if reference_metadata.is_none() {
            reference_metadata =
                Some(PreprocessingMetadata::try_from(&mut decoder).context(TiffReadSnafu)?);

            // Determine color type from first frame
            let dicom_color_type = DicomColorType::try_from(&mut decoder).context(TiffReadSnafu)?;
            color_type = Some(dicom_color_type);
        }

        // Load the first frame as a DynamicImage using the reusable function
        let dynamic_images = load_frames_as_dynamic_images(
            &mut decoder,
            color_type.as_ref().unwrap(),
            std::iter::once(0),
        )
        .context(TiffReadSnafu)?;
        let dynamic_image = dynamic_images.into_iter().next().unwrap();

        images.push(dynamic_image);
    }

    // Create output path
    let study_uid = study_uid.unwrap_or_else(|| "unknown_study".to_string());

    // Use the first frame's SOP instance UID as the filename
    let first_sop_uid = if let Some((_, _, sop_uid)) = extract_path_components(&ordered_files[0].0)
    {
        sop_uid
    } else {
        format!("{}_combined", series_uid)
    };

    let output_file = output_path
        .join(&study_uid)
        .join(series_uid)
        .join(format!("{}.tiff", first_sop_uid));

    // Create parent directory if it doesn't exist
    if let Some(parent) = output_file.parent() {
        std::fs::create_dir_all(parent).context(IOSnafu)?;
    }

    // Update metadata to reflect the number of frames
    let mut metadata = reference_metadata.unwrap();
    metadata.num_frames = (images.len() as u16).into();

    // Save multi-frame TIFF
    let color_type = color_type.unwrap();
    let saver = TiffSaver::new(Compressor::Uncompressed(Uncompressed), color_type);
    let mut encoder = saver.open_tiff(&output_file).context(TiffWriteSnafu)?;

    for image in images.iter() {
        saver
            .save(&mut encoder, image, &metadata)
            .context(TiffWriteSnafu)?;
    }

    Ok(ordered_files.len())
}

fn run(args: Args) -> Result<(usize, usize), Error> {
    validate_paths(&args)?;
    let source_files = load_source_files(&args.source)?;
    let metadata_map = load_metadata(&args.metadata)?;
    combine_series(source_files, &metadata_map, &args.output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::record_batch::RecordBatch;
    use dicom_preprocessing::FrameCount;
    use image::DynamicImage;
    use ndarray::Array4;
    use rstest::rstest;
    use std::fs;
    use tempfile::TempDir;

    fn create_test_tiff(
        tmp_dir: &Path,
        study_uid: &str,
        series_uid: &str,
        sop_uid: &str,
        width: u32,
        height: u32,
    ) -> PathBuf {
        let array = Array4::<u8>::zeros((1, height as usize, width as usize, 1));
        let metadata = PreprocessingMetadata {
            crop: None,
            resize: None,
            padding: None,
            resolution: None,
            num_frames: FrameCount::from(1_u16),
        };
        let saver = TiffSaver::new(
            Compressor::Uncompressed(Uncompressed),
            DicomColorType::Gray8(tiff::encoder::colortype::Gray8),
        );
        let path = tmp_dir
            .join(study_uid)
            .join(series_uid)
            .join(format!("{}.tiff", sop_uid));

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }

        let mut encoder = saver.open_tiff(&path).unwrap();
        let dynamic_image = DynamicImage::ImageLuma8(
            image::ImageBuffer::from_raw(
                array.shape()[2] as u32,
                array.shape()[1] as u32,
                array.into_raw_vec_and_offset().0,
            )
            .unwrap(),
        );

        saver.save(&mut encoder, &dynamic_image, &metadata).unwrap();
        path
    }

    fn create_test_metadata_csv(tmp_dir: &Path) -> PathBuf {
        let path = tmp_dir.join("metadata.csv");
        let mut writer = csv::Writer::from_path(&path).unwrap();
        writer
            .write_record(["series_instance_uid", "sop_instance_uid", "instance_number"])
            .unwrap();
        writer.write_record(["series1", "slice1", "1"]).unwrap();
        writer.write_record(["series1", "slice2", "2"]).unwrap();
        writer.write_record(["series1", "slice3", "3"]).unwrap();
        writer.flush().unwrap();
        path
    }

    fn create_test_metadata_parquet(tmp_dir: &Path) -> PathBuf {
        let path = tmp_dir.join("metadata.parquet");
        let schema = std::sync::Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new(
                "series_instance_uid",
                arrow::datatypes::DataType::Utf8,
                false,
            ),
            arrow::datatypes::Field::new(
                "sop_instance_uid",
                arrow::datatypes::DataType::Utf8,
                false,
            ),
            arrow::datatypes::Field::new(
                "instance_number",
                arrow::datatypes::DataType::Int64,
                false,
            ),
        ]));

        let series_array = StringArray::from(vec!["series1", "series1", "series1"]);
        let sop_array = StringArray::from(vec!["slice1", "slice2", "slice3"]);
        let instance_array = Int64Array::from(vec![1, 2, 3]);

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                std::sync::Arc::new(series_array),
                std::sync::Arc::new(sop_array),
                std::sync::Arc::new(instance_array),
            ],
        )
        .unwrap();

        let file = File::create(&path).unwrap();
        let props = parquet::file::properties::WriterProperties::builder().build();
        let mut writer =
            parquet::arrow::arrow_writer::ArrowWriter::try_new(file, schema, Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        path
    }

    #[rstest]
    #[case("csv")]
    #[case("parquet")]
    fn test_combine_series(#[case] format: &str) {
        let tmp_dir = TempDir::new().unwrap();
        let source_dir = tmp_dir.path().join("source");
        fs::create_dir(&source_dir).unwrap();

        // Create test TIFFs for a single series with 3 slices
        create_test_tiff(&source_dir, "study1", "series1", "slice1", 64, 64);
        create_test_tiff(&source_dir, "study1", "series1", "slice2", 64, 64);
        create_test_tiff(&source_dir, "study1", "series1", "slice3", 64, 64);

        // Create test metadata file
        let metadata_path = match format {
            "csv" => create_test_metadata_csv(tmp_dir.path()),
            "parquet" => create_test_metadata_parquet(tmp_dir.path()),
            _ => panic!("Unsupported format"),
        };

        // Create output directory
        let output_dir = tmp_dir.path().join("output");
        fs::create_dir(&output_dir).unwrap();

        // Run the CLI
        let args = Args {
            source: source_dir,
            metadata: metadata_path,
            output: output_dir.clone(),
            verbose: true,
        };
        let (num_series, num_files) = run(args).unwrap();

        // Verify results
        assert_eq!(num_series, 1);
        assert_eq!(num_files, 3);

        // Verify output file exists
        let output_file = output_dir
            .join("study1")
            .join("series1")
            .join("slice1.tiff");
        assert!(output_file.exists());

        // Verify it's a multi-frame TIFF with 3 frames
        let file = File::open(output_file).unwrap();
        let reader = BufReader::new(file);
        let mut decoder = Decoder::new(reader).unwrap();
        let metadata = PreprocessingMetadata::try_from(&mut decoder).unwrap();
        let num_frames: u16 = metadata.num_frames.into();
        assert_eq!(num_frames, 3);

        // Verify dimensions are preserved
        let (width, height) = decoder.dimensions().unwrap();
        assert_eq!(width, 64);
        assert_eq!(height, 64);
    }

    #[test]
    fn test_invalid_source_path() {
        let tmp_dir = TempDir::new().unwrap();
        let args = Args {
            source: tmp_dir.path().join("nonexistent"),
            metadata: tmp_dir.path().join("metadata.csv"),
            output: tmp_dir.path().join("output"),
            verbose: true,
        };
        assert!(matches!(run(args), Err(Error::InvalidSourcePath { .. })));
    }

    #[test]
    fn test_invalid_metadata_format() {
        let tmp_dir = TempDir::new().unwrap();
        let source_dir = tmp_dir.path().join("source");
        fs::create_dir(&source_dir).unwrap();

        let args = Args {
            source: source_dir,
            metadata: tmp_dir.path().join("metadata.txt"),
            output: tmp_dir.path().join("output"),
            verbose: true,
        };
        assert!(matches!(
            run(args),
            Err(Error::InvalidMetadataFormat { .. })
        ));
    }

    #[test]
    fn test_no_sources() {
        let tmp_dir = TempDir::new().unwrap();
        let source_dir = tmp_dir.path().join("source");
        fs::create_dir(&source_dir).unwrap();

        let args = Args {
            source: source_dir,
            metadata: create_test_metadata_csv(tmp_dir.path()),
            output: tmp_dir.path().join("output"),
            verbose: true,
        };
        assert!(matches!(run(args), Err(Error::NoSources { .. })));
    }

    #[test]
    fn test_extract_path_components() {
        let path = Path::new("study1/series1/slice1.tiff");
        let (study, series, sop) = extract_path_components(path).unwrap();
        assert_eq!(study, "study1");
        assert_eq!(series, "series1");
        assert_eq!(sop, "slice1");
    }

    #[test]
    fn test_multiple_series() {
        let tmp_dir = TempDir::new().unwrap();
        let source_dir = tmp_dir.path().join("source");
        fs::create_dir(&source_dir).unwrap();

        // Create test TIFFs for two series
        create_test_tiff(&source_dir, "study1", "series1", "slice1", 64, 64);
        create_test_tiff(&source_dir, "study1", "series1", "slice2", 64, 64);
        create_test_tiff(&source_dir, "study1", "series2", "slice1", 64, 64);

        // Create metadata for both series
        let metadata_path = tmp_dir.path().join("metadata.csv");
        let mut writer = csv::Writer::from_path(&metadata_path).unwrap();
        writer
            .write_record(["series_instance_uid", "sop_instance_uid", "instance_number"])
            .unwrap();
        writer.write_record(["series1", "slice1", "1"]).unwrap();
        writer.write_record(["series1", "slice2", "2"]).unwrap();
        writer.write_record(["series2", "slice1", "1"]).unwrap();
        writer.flush().unwrap();

        let output_dir = tmp_dir.path().join("output");
        fs::create_dir(&output_dir).unwrap();

        let args = Args {
            source: source_dir,
            metadata: metadata_path,
            output: output_dir.clone(),
            verbose: true,
        };
        let (num_series, num_files) = run(args).unwrap();

        assert_eq!(num_series, 2);
        assert_eq!(num_files, 3);

        // Verify both output files exist
        assert!(output_dir
            .join("study1")
            .join("series1")
            .join("slice1.tiff")
            .exists());
        assert!(output_dir
            .join("study1")
            .join("series2")
            .join("slice1.tiff")
            .exists());
    }
}
