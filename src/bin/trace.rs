use arrow::array::{Int64Array, StringArray};
use arrow::error::ArrowError;
use arrow::record_batch::RecordBatch;
use clap::Parser;
use csv::Reader as CsvReader;
use dicom_preprocessing::color::DicomColorType;
use dicom_preprocessing::errors::TiffError;
use dicom_preprocessing::file::default_bar;
use dicom_preprocessing::file::{InodeSort, TiffFileOperations};
use dicom_preprocessing::load::LoadFromTiff;
use dicom_preprocessing::metadata::PreprocessingMetadata;
use dicom_preprocessing::save::TiffSaver;
use dicom_preprocessing::transform::{Coord, Transform};
use image::DynamicImage;
use indicatif::ParallelProgressIterator;
use indicatif::ProgressBar;
use ndarray::Array4;
use parquet::arrow::arrow_reader::ParquetRecordBatchReader;
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::errors::ParquetError;
use parquet::file::properties::WriterProperties;
use rayon::prelude::*;
use snafu::{Report, ResultExt, Snafu, Whatever};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::io::BufWriter;
use std::io::Write;
use std::num::ParseIntError;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use tiff::decoder::Decoder;
use tiff::encoder::colortype::RGB8;
use tiff::encoder::compression::{Compressor, Uncompressed};
use tracing::{error, Level};

const BATCH_SIZE: usize = 128;
const TRACE_COLOR: [u8; 3] = [255, 0, 0];
const BORDER: usize = 2;

#[derive(Debug, Snafu)]
enum Error {
    #[snafu(display("Invalid source path: {}", path.display()))]
    InvalidSourcePath { path: PathBuf },

    #[snafu(display("No sources found in source path: {}", path.display()))]
    NoSources { path: PathBuf },

    #[snafu(display("Invalid traces format: {}", path.display()))]
    InvalidTracesFormat { path: PathBuf },

    #[snafu(display("Invalid output extension for {}, supported extensions: {}", path.display(), supported.join(", ")))]
    InvalidOutputExtension {
        path: PathBuf,
        supported: Vec<&'static str>,
    },

    #[snafu(display("Invalid preview path: {}", path.display()))]
    InvalidPreviewPath { path: PathBuf },

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
}

#[derive(Debug, Clone, Copy)]
enum OutputFormat {
    Csv,
    Parquet,
}

impl OutputFormat {
    fn from_extension(path: &Path) -> Result<Self, Error> {
        match path.extension().and_then(|ext| ext.to_str()) {
            Some("csv") => Ok(OutputFormat::Csv),
            Some("parquet") => Ok(OutputFormat::Parquet),
            _ => Err(Error::InvalidOutputExtension {
                path: path.to_path_buf(),
                supported: vec!["csv", "parquet"],
            }),
        }
    }
}

#[derive(Debug)]
struct Trace {
    sop_instance_uid: String,
    hash: i64,
    x_min: u32,
    x_max: u32,
    y_min: u32,
    y_max: u32,
}

impl Trace {
    pub fn new(
        sop_instance_uid: String,
        hash: i64,
        x_min: u32,
        x_max: u32,
        y_min: u32,
        y_max: u32,
    ) -> Self {
        Self {
            sop_instance_uid,
            hash,
            x_min,
            x_max,
            y_min,
            y_max,
        }
    }

    pub fn sop_instance_uid(&self) -> &str {
        &self.sop_instance_uid
    }

    pub fn hash(&self) -> i64 {
        self.hash
    }

    pub fn is_valid(&self) -> bool {
        self.x_min < self.x_max && self.y_min < self.y_max
    }

    pub fn x_min(&self) -> u32 {
        self.x_min
    }

    pub fn x_max(&self) -> u32 {
        self.x_max
    }

    pub fn y_min(&self) -> u32 {
        self.y_min
    }

    pub fn y_max(&self) -> u32 {
        self.y_max
    }
}

impl From<Trace> for (Coord, Coord) {
    fn from(trace: Trace) -> Self {
        (
            Coord::new(trace.x_min, trace.y_min),
            Coord::new(trace.x_max, trace.y_max),
        )
    }
}

impl From<&Trace> for (Coord, Coord) {
    fn from(trace: &Trace) -> Self {
        (
            Coord::new(trace.x_min, trace.y_min),
            Coord::new(trace.x_max, trace.y_max),
        )
    }
}

impl<T: From<u32>> From<Trace> for (T, T, T, T) {
    fn from(trace: Trace) -> Self {
        (
            trace.x_min.into(),
            trace.y_min.into(),
            trace.x_max.into(),
            trace.y_max.into(),
        )
    }
}

impl<T: From<u32>> From<&Trace> for (T, T, T, T) {
    fn from(trace: &Trace) -> Self {
        (
            trace.x_min.into(),
            trace.y_min.into(),
            trace.x_max.into(),
            trace.y_max.into(),
        )
    }
}

#[derive(Parser, Debug)]
#[command(author = "Scott Chase Waggener", version = env!("CARGO_PKG_VERSION"), about = "Preprocess traces into a preprocessed image coordinate system", long_about = None)]
struct Args {
    #[arg(help = "Source TIFF, directory of TIFFs, or .txt file with TIFF paths")]
    source: PathBuf,

    #[arg(
        help = "CSV or Parquet file with sop_instance_uid, trace_hash, x_min, x_max, y_min, y_max"
    )]
    traces: PathBuf,

    #[arg(help = "Output file or directory")]
    output: PathBuf,

    #[arg(
        help = "Optional directory to save previews to",
        long = "preview",
        short = 'p',
        default_value = None,
    )]
    preview: Option<PathBuf>,

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

fn validate_paths(args: &Args) -> Result<Option<&PathBuf>, Error> {
    if let Some(preview) = &args.preview {
        if !preview.is_dir() {
            return Err(Error::InvalidPreviewPath {
                path: preview.clone(),
            });
        }
    }
    Ok(args.preview.as_ref())
}

fn load_source_files(source_path: &PathBuf) -> Result<Vec<PathBuf>, Error> {
    let source = if source_path.is_dir() {
        source_path
            .find_tiffs()
            .map_err(|_| Error::InvalidSourcePath {
                path: source_path.to_path_buf(),
            })?
            .collect::<Vec<_>>()
    } else if source_path.is_file() && source_path.extension().unwrap() == "txt" {
        source_path
            .iter()
            .read_tiff_paths_with_bar()
            .map_err(|_| Error::InvalidSourcePath {
                path: source_path.to_path_buf(),
            })?
            .collect::<Vec<_>>()
    } else if source_path.is_file() {
        vec![source_path.clone()]
    } else {
        return Err(Error::InvalidSourcePath {
            path: source_path.to_path_buf(),
        });
    };

    let source = source
        .into_iter()
        .sorted_by_inode_with_progress()
        .collect::<Vec<_>>();

    if source.is_empty() {
        return Err(Error::NoSources {
            path: source_path.clone(),
        });
    }

    tracing::info!("Number of sources found: {}", source.len());
    Ok(source)
}

fn load_traces_csv(path: &PathBuf) -> Result<HashMap<String, Vec<Trace>>, Error> {
    let mut reader = CsvReader::from_path(path).context(CsvSnafu)?;
    let mut traces = HashMap::new();
    for result in reader.deserialize() {
        let record: HashMap<String, String> = result.context(CsvSnafu)?;
        let sop_instance_uid = record["sop_instance_uid"].clone();
        let hash = record["trace_hash"].parse::<i64>().context(ParseIntSnafu)?;
        let x_min = record["x_min"].parse::<u32>().context(ParseIntSnafu)?;
        let x_max = record["x_max"].parse::<u32>().context(ParseIntSnafu)?;
        let y_min = record["y_min"].parse::<u32>().context(ParseIntSnafu)?;
        let y_max = record["y_max"].parse::<u32>().context(ParseIntSnafu)?;
        traces
            .entry(sop_instance_uid.clone())
            .or_insert_with(Vec::new)
            .push(Trace::new(
                sop_instance_uid,
                hash,
                x_min,
                x_max,
                y_min,
                y_max,
            ));
    }
    Ok(traces)
}

fn load_traces_parquet(path: &PathBuf) -> Result<HashMap<String, Vec<Trace>>, Error> {
    let file = File::open(path).context(IOSnafu)?;
    let reader = ParquetRecordBatchReader::try_new(file, BATCH_SIZE).context(ParquetSnafu)?;
    let mut traces = HashMap::new();

    for result in reader {
        let batch = result.context(ArrowSnafu)?;
        let sop_array = batch
            .column_by_name("sop_instance_uid")
            .expect("Failed to get sop_instance_uid column")
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("Failed to downcast sop_instance_uid to StringArray");
        let hash_array = batch
            .column_by_name("trace_hash")
            .expect("Failed to get trace_hash column")
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("Failed to downcast trace_hash to Int64Array");
        let x_min_array = batch
            .column_by_name("x_min")
            .expect("Failed to get x_min column")
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("Failed to downcast x_min to Int64Array");
        let x_max_array = batch
            .column_by_name("x_max")
            .expect("Failed to get x_max column")
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("Failed to downcast x_max to Int64Array");
        let y_min_array = batch
            .column_by_name("y_min")
            .expect("Failed to get y_min column")
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("Failed to downcast y_min to Int64Array");
        let y_max_array = batch
            .column_by_name("y_max")
            .expect("Failed to get y_max column")
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("Failed to downcast y_max to Int64Array");

        for i in 0..batch.num_rows() {
            let sop_instance_uid = sop_array.value(i).to_string();
            let hash = hash_array.value(i);
            let x_min = x_min_array.value(i) as u32;
            let x_max = x_max_array.value(i) as u32;
            let y_min = y_min_array.value(i) as u32;
            let y_max = y_max_array.value(i) as u32;
            traces
                .entry(sop_instance_uid.clone())
                .or_insert_with(Vec::new)
                .push(Trace::new(
                    sop_instance_uid,
                    hash,
                    x_min,
                    x_max,
                    y_min,
                    y_max,
                ));
        }
    }
    Ok(traces)
}

fn load_traces(path: &PathBuf) -> Result<HashMap<String, Vec<Trace>>, Error> {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some("csv") => load_traces_csv(path),
        Some("parquet") => load_traces_parquet(path),
        _ => Err(Error::InvalidTracesFormat { path: path.clone() }),
    }
}

fn process_traces(
    source_files: Vec<PathBuf>,
    trace_metadata: &HashMap<String, Vec<Trace>>,
    preview: Option<&PathBuf>,
) -> Result<Vec<(String, Vec<Trace>)>, Error> {
    let pb = default_bar(source_files.len() as u64);
    pb.set_message("Processing traces");
    let results = source_files
        .into_par_iter()
        .progress_with(pb)
        .map(|s| process(&s, trace_metadata, preview))
        .collect::<Result<Vec<_>, Error>>()?
        .into_iter()
        .filter(|(_, traces)| !traces.is_empty())
        .collect::<Vec<_>>();

    tracing::info!(
        "Found {} traces",
        results
            .iter()
            .map(|(_, traces)| traces.len())
            .sum::<usize>()
    );
    Ok(results)
}

fn write_output(
    results: &Vec<(String, Vec<Trace>)>,
    output_path: &Path,
) -> Result<(usize, usize), Error> {
    let output_format =
        OutputFormat::from_extension(output_path).map_err(|_| Error::InvalidOutputExtension {
            path: output_path.to_path_buf(),
            supported: vec!["csv", "parquet"],
        })?;

    let pb = default_bar(results.len() as u64);
    pb.set_message("Writing outputs");

    match output_format {
        OutputFormat::Csv => write_traces_csv(results, output_path, &pb)?,
        OutputFormat::Parquet => write_traces_parquet(results, output_path, &pb)?,
    }
    pb.finish();

    let num_images = results.len();
    let num_traces = results
        .iter()
        .map(|(_, traces)| traces.len())
        .sum::<usize>();
    println!("Wrote {num_traces} traces across {num_images} images");
    Ok((num_images, num_traces))
}

fn run(args: Args) -> Result<(usize, usize), Error> {
    let preview = validate_paths(&args)?;
    let source_files = load_source_files(&args.source)?;
    let traces = load_traces(&args.traces)?;
    let results = process_traces(source_files, &traces, preview)?;
    write_output(&results, &args.output)
}

fn process(
    source: &PathBuf,
    trace_metadata: &HashMap<String, Vec<Trace>>,
    output: Option<&PathBuf>,
) -> Result<(String, Vec<Trace>), Error> {
    // Open TIFF
    let file = File::open(source).context(IOSnafu)?;
    let reader = BufReader::new(file);
    let mut decoder = Decoder::new(reader)
        .map_err(TiffError::from)
        .context(TiffReadSnafu)?;
    let (width, height) = decoder
        .dimensions()
        .map_err(TiffError::from)
        .context(TiffReadSnafu)?;

    // Read metadata and find corresponding traces
    let sop_instance_uid = source
        .file_stem()
        .unwrap_or_default()
        .to_str()
        .unwrap_or_default();
    let metadata = PreprocessingMetadata::try_from(&mut decoder).context(TiffReadSnafu)?;
    let traces = match trace_metadata.get(sop_instance_uid) {
        Some(traces) => traces,
        None => {
            tracing::debug!(
                "sop_instance_uid {} not found in trace metadata",
                sop_instance_uid
            );
            return Ok((sop_instance_uid.to_string(), vec![]));
        }
    };

    // Adjust traces
    let traces = traces
        .iter()
        .map(|trace| {
            tracing::debug!("Processing trace: {:?}", trace);
            let (min, max): (Coord, Coord) = trace.into();
            let min = metadata.apply(&min);
            let max = metadata.apply(&max);
            let (x_min, y_min): (u32, u32) = min.into();
            let (x_max, y_max): (u32, u32) = max.into();
            let processed_trace = Trace::new(
                trace.sop_instance_uid().to_string(),
                trace.hash(),
                x_min,
                x_max.min(width - 1),
                y_min,
                y_max.min(height - 1),
            );
            tracing::debug!("Processed trace: {:?}", processed_trace);
            processed_trace
        })
        .filter(|trace| trace.is_valid())
        .collect::<Vec<_>>();

    if traces.is_empty() {
        tracing::debug!("No traces found for {}", sop_instance_uid);
        return Ok((sop_instance_uid.to_string(), vec![]));
    }

    // If an output path is provided, save the TIFF with traces overlaid
    if let Some(output) = output {
        // Determine output filename/path
        let study_instance_uid = source
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .file_name()
            .unwrap()
            .to_str()
            .unwrap_or_default();
        let series_instance_uid = source
            .parent()
            .unwrap()
            .file_name()
            .unwrap()
            .to_str()
            .unwrap_or_default();
        let path = output.join(study_instance_uid).join(series_instance_uid);
        let filename = format!("{sop_instance_uid}.tiff");
        let path = path.join(filename);

        // Create parent directory if it doesn't exist
        if !path.parent().unwrap().is_dir() {
            std::fs::create_dir_all(path.parent().unwrap()).context(IOSnafu)?;
        }

        let array = Array4::<u8>::decode(&mut decoder).context(TiffReadSnafu)?;

        // Reduce dimension N using a max
        let array_max = array.map_axis(ndarray::Axis(0), |view| view.fold(0, |a, &b| a.max(b)));

        // Expand dimension C to 3 when C=1
        let mut array_expanded = if array_max.shape()[2] == 1 {
            array_max
                .broadcast((array_max.shape()[0], array_max.shape()[1], 3))
                .unwrap()
                .to_owned()
        } else {
            array_max
        };

        // Overlay traces
        for trace in traces.iter() {
            let (x_min, y_min, x_max, y_max): (u32, u32, u32, u32) = trace.into();
            for x in x_min..x_max {
                for y in y_min..y_max {
                    if x < x_min + BORDER as u32
                        || x >= x_max - BORDER as u32
                        || y < y_min + BORDER as u32
                        || y >= y_max - BORDER as u32
                    {
                        array_expanded[[y as usize, x as usize, 0]] = TRACE_COLOR[0];
                        array_expanded[[y as usize, x as usize, 1]] = TRACE_COLOR[1];
                        array_expanded[[y as usize, x as usize, 2]] = TRACE_COLOR[2];
                    }
                }
            }
        }

        // Convert the array to a DynamicImage
        let dynamic_image = DynamicImage::ImageRgb8(
            image::ImageBuffer::from_raw(
                array_expanded.shape()[1] as u32,
                array_expanded.shape()[0] as u32,
                array_expanded.into_raw_vec_and_offset().0,
            )
            .ok_or(TiffError::DynamicImageError {
                color_type: image::ColorType::Rgb8,
            })
            .context(TiffWriteSnafu)?,
        );

        let saver = TiffSaver::new(
            Compressor::Uncompressed(Uncompressed),
            DicomColorType::RGB8(RGB8),
        );
        let mut encoder = saver.open_tiff(&path).context(TiffWriteSnafu)?;
        saver
            .save(&mut encoder, &dynamic_image, &metadata)
            .context(TiffWriteSnafu)?;
    }

    Ok((sop_instance_uid.to_string(), traces))
}

fn write_traces_csv(
    entries: &Vec<(String, Vec<Trace>)>,
    dest: &Path,
    pb: &ProgressBar,
) -> Result<(), Error> {
    let mut csv_file = BufWriter::new(File::create(dest).context(IOSnafu)?);
    csv_file
        .write_all(b"sop_instance_uid,trace_hash,x_min,x_max,y_min,y_max\n")
        .context(IOSnafu)?;
    for (sop_instance_uid, traces) in entries {
        for trace in traces {
            let (x_min, y_min, x_max, y_max): (u32, u32, u32, u32) = trace.into();
            writeln!(
                csv_file,
                "{},{},{},{},{},{}",
                sop_instance_uid,
                trace.hash(),
                x_min,
                x_max,
                y_min,
                y_max,
            )
            .context(IOSnafu)?;
        }
        pb.inc(1);
    }
    Ok(())
}

fn write_traces_parquet(
    entries: &Vec<(String, Vec<Trace>)>,
    dest: &Path,
    pb: &ProgressBar,
) -> Result<(), Error> {
    // Create the schema
    let schema = arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("sop_instance_uid", arrow::datatypes::DataType::Utf8, false),
        arrow::datatypes::Field::new("trace_hash", arrow::datatypes::DataType::Int64, false),
        arrow::datatypes::Field::new("x_min", arrow::datatypes::DataType::Int64, false),
        arrow::datatypes::Field::new("x_max", arrow::datatypes::DataType::Int64, false),
        arrow::datatypes::Field::new("y_min", arrow::datatypes::DataType::Int64, false),
        arrow::datatypes::Field::new("y_max", arrow::datatypes::DataType::Int64, false),
    ]);

    let schema_arc = Arc::new(schema);

    // Create the file and writer
    let file = File::create(dest).context(IOSnafu)?;
    let props = WriterProperties::builder().build();
    let mut writer =
        ArrowWriter::try_new(file, schema_arc.clone(), Some(props)).context(ParquetSnafu)?;

    for (sop_instance_uid, traces) in entries {
        let sop_instance_uid_array =
            StringArray::from(vec![sop_instance_uid.clone(); traces.len()]);
        let hash_array = Int64Array::from(traces.iter().map(|t| t.hash()).collect::<Vec<_>>());
        let x_min_array =
            Int64Array::from(traces.iter().map(|t| t.x_min() as i64).collect::<Vec<_>>());
        let x_max_array =
            Int64Array::from(traces.iter().map(|t| t.x_max() as i64).collect::<Vec<_>>());
        let y_min_array =
            Int64Array::from(traces.iter().map(|t| t.y_min() as i64).collect::<Vec<_>>());
        let y_max_array =
            Int64Array::from(traces.iter().map(|t| t.y_max() as i64).collect::<Vec<_>>());

        let batch = RecordBatch::try_new(
            schema_arc.clone(),
            vec![
                std::sync::Arc::new(sop_instance_uid_array),
                std::sync::Arc::new(hash_array),
                std::sync::Arc::new(x_min_array),
                std::sync::Arc::new(x_max_array),
                std::sync::Arc::new(y_min_array),
                std::sync::Arc::new(y_max_array),
            ],
        )
        .context(ArrowSnafu)?;
        writer.write(&batch).context(ParquetSnafu)?;
    }

    writer.close().context(ParquetSnafu)?;
    pb.inc(entries.len() as u64);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use dicom_preprocessing::FrameCount;
    use ndarray::Array4;
    use rstest::rstest;
    use std::fs;
    use std::path::PathBuf;
    use tempfile::TempDir;

    fn create_test_tiff(
        tmp_dir: &Path,
        sop_instance_uid: &str,
        study_instance_uid: &str,
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
            .join(study_instance_uid)
            .join("series")
            .join(format!("{sop_instance_uid}.tiff"));
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

    fn create_test_traces_csv(tmp_dir: &Path) -> PathBuf {
        let path = tmp_dir.join("traces.csv");
        let mut writer = csv::Writer::from_path(&path).unwrap();
        writer
            .write_record([
                "sop_instance_uid",
                "trace_hash",
                "x_min",
                "x_max",
                "y_min",
                "y_max",
            ])
            .unwrap();
        writer
            .write_record(["test1", "1", "10", "20", "30", "40"])
            .unwrap();
        writer
            .write_record(["test2", "2", "15", "25", "35", "45"])
            .unwrap();
        writer.flush().unwrap();
        path
    }

    fn create_test_traces_parquet(tmp_dir: &Path) -> PathBuf {
        let path = tmp_dir.join("traces.parquet");
        let schema = Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new(
                "sop_instance_uid",
                arrow::datatypes::DataType::Utf8,
                false,
            ),
            arrow::datatypes::Field::new("trace_hash", arrow::datatypes::DataType::Int64, false),
            arrow::datatypes::Field::new("x_min", arrow::datatypes::DataType::Int64, false),
            arrow::datatypes::Field::new("x_max", arrow::datatypes::DataType::Int64, false),
            arrow::datatypes::Field::new("y_min", arrow::datatypes::DataType::Int64, false),
            arrow::datatypes::Field::new("y_max", arrow::datatypes::DataType::Int64, false),
        ]));

        let sop_array = StringArray::from(vec!["test1", "test2"]);
        let hash_array = Int64Array::from(vec![1, 2]);
        let x_min_array = Int64Array::from(vec![10, 15]);
        let x_max_array = Int64Array::from(vec![20, 25]);
        let y_min_array = Int64Array::from(vec![30, 35]);
        let y_max_array = Int64Array::from(vec![40, 45]);

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(sop_array),
                Arc::new(hash_array),
                Arc::new(x_min_array),
                Arc::new(x_max_array),
                Arc::new(y_min_array),
                Arc::new(y_max_array),
            ],
        )
        .unwrap();

        let file = File::create(&path).unwrap();
        let props = WriterProperties::builder().build();
        let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        path
    }

    #[rstest]
    #[case("csv")]
    #[case("parquet")]
    fn test_trace_processing(#[case] format: &str) {
        let tmp_dir = TempDir::new().unwrap();
        let source_dir = tmp_dir.path().join("source");
        fs::create_dir(&source_dir).unwrap();

        // Create test TIFFs
        create_test_tiff(&source_dir, "test1", "study1", 100, 100);
        create_test_tiff(&source_dir, "test2", "study1", 100, 100);

        // Create test traces file
        let traces_path = match format {
            "csv" => create_test_traces_csv(tmp_dir.path()),
            "parquet" => create_test_traces_parquet(tmp_dir.path()),
            _ => panic!("Unsupported format"),
        };

        // Create output directory
        let output_dir = tmp_dir.path().join("output");
        fs::create_dir(&output_dir).unwrap();
        let output_path = output_dir.join(format!("output.{format}"));

        // Create preview directory
        let preview_dir = tmp_dir.path().join("preview");
        fs::create_dir(&preview_dir).unwrap();

        // Run the CLI
        let args = Args {
            source: source_dir,
            traces: traces_path,
            output: output_path.clone(),
            preview: Some(preview_dir.clone()),
            verbose: true,
        };
        run(args).unwrap();

        // Verify output file exists
        assert!(output_path.exists());

        // Verify preview files were created
        assert!(preview_dir
            .join("study1")
            .join("series")
            .join("test1.tiff")
            .exists());
        assert!(preview_dir
            .join("study1")
            .join("series")
            .join("test2.tiff")
            .exists());

        // Verify output contents
        match format {
            "csv" => {
                let mut reader = csv::Reader::from_path(output_path).unwrap();
                let records: Vec<_> = reader.records().collect();
                assert_eq!(records.len(), 2);

                // Verify the contents of each record (order is not guaranteed)
                let mut found_test1 = false;
                let mut found_test2 = false;
                for record in records.iter() {
                    let record = record.as_ref().unwrap();
                    match record.get(0).unwrap() {
                        "test1" => {
                            assert_eq!(record.get(1).unwrap(), "1"); // trace_hash
                            assert_eq!(record.get(2).unwrap(), "10"); // x_min
                            assert_eq!(record.get(3).unwrap(), "20"); // x_max
                            assert_eq!(record.get(4).unwrap(), "30"); // y_min
                            assert_eq!(record.get(5).unwrap(), "40"); // y_max
                            found_test1 = true;
                        }
                        "test2" => {
                            assert_eq!(record.get(1).unwrap(), "2");
                            assert_eq!(record.get(2).unwrap(), "15");
                            assert_eq!(record.get(3).unwrap(), "25");
                            assert_eq!(record.get(4).unwrap(), "35");
                            assert_eq!(record.get(5).unwrap(), "45");
                            found_test2 = true;
                        }
                        _ => panic!("Unexpected record"),
                    }
                }
                assert!(
                    found_test1 && found_test2,
                    "Not all expected records were found"
                );
            }
            "parquet" => {
                let file = File::open(output_path).unwrap();
                let reader = ParquetRecordBatchReader::try_new(file, 1024).unwrap();
                let batches: Vec<_> = reader.collect();
                assert_eq!(batches.len(), 1);
                let batch = batches[0].as_ref().unwrap();
                assert_eq!(batch.num_rows(), 2);

                // Verify the contents of the batch
                let sop_array = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap();
                let hash_array = batch
                    .column(1)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap();
                let x_min_array = batch
                    .column(2)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap();
                let x_max_array = batch
                    .column(3)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap();
                let y_min_array = batch
                    .column(4)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap();
                let y_max_array = batch
                    .column(5)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap();

                // Check both rows are present (order is not guaranteed)
                let mut found_test1 = false;
                let mut found_test2 = false;
                for i in 0..2 {
                    match sop_array.value(i) {
                        "test1" => {
                            assert_eq!(hash_array.value(i), 1);
                            assert_eq!(x_min_array.value(i), 10);
                            assert_eq!(x_max_array.value(i), 20);
                            assert_eq!(y_min_array.value(i), 30);
                            assert_eq!(y_max_array.value(i), 40);
                            found_test1 = true;
                        }
                        "test2" => {
                            assert_eq!(hash_array.value(i), 2);
                            assert_eq!(x_min_array.value(i), 15);
                            assert_eq!(x_max_array.value(i), 25);
                            assert_eq!(y_min_array.value(i), 35);
                            assert_eq!(y_max_array.value(i), 45);
                            found_test2 = true;
                        }
                        _ => panic!("Unexpected SOP value"),
                    }
                }
                assert!(
                    found_test1 && found_test2,
                    "Both test rows should be present"
                );
            }
            _ => panic!("Unsupported format"),
        }
    }

    #[test]
    fn test_invalid_source_path() {
        let tmp_dir = TempDir::new().unwrap();
        let args = Args {
            source: tmp_dir.path().join("nonexistent"),
            traces: tmp_dir.path().join("traces.csv"),
            output: tmp_dir.path().join("output.csv"),
            preview: None,
            verbose: true,
        };
        assert!(matches!(run(args), Err(Error::InvalidSourcePath { .. })));
    }

    #[test]
    fn test_invalid_traces_format() {
        let tmp_dir = TempDir::new().unwrap();
        let source_dir = tmp_dir.path().join("source");
        fs::create_dir(&source_dir).unwrap();
        create_test_tiff(&source_dir, "test1", "study1", 100, 100);

        let args = Args {
            source: source_dir,
            traces: tmp_dir.path().join("traces.txt"),
            output: tmp_dir.path().join("output.csv"),
            preview: None,
            verbose: true,
        };
        assert!(matches!(run(args), Err(Error::InvalidTracesFormat { .. })));
    }

    #[test]
    fn test_invalid_output_extension() {
        let tmp_dir = TempDir::new().unwrap();
        let source_dir = tmp_dir.path().join("source");
        fs::create_dir(&source_dir).unwrap();
        create_test_tiff(&source_dir, "test1", "study1", 100, 100);

        let args = Args {
            source: source_dir,
            traces: create_test_traces_csv(tmp_dir.path()),
            output: tmp_dir.path().join("output.txt"),
            preview: None,
            verbose: true,
        };
        assert!(matches!(
            run(args),
            Err(Error::InvalidOutputExtension { .. })
        ));
    }

    #[test]
    fn test_invalid_preview_path() {
        let tmp_dir = TempDir::new().unwrap();
        let source_dir = tmp_dir.path().join("source");
        fs::create_dir(&source_dir).unwrap();
        create_test_tiff(&source_dir, "test1", "study1", 100, 100);

        // Create a file instead of a directory for preview path
        let preview_path = tmp_dir.path().join("preview");
        fs::write(&preview_path, "").unwrap();

        let args = Args {
            source: source_dir,
            traces: create_test_traces_csv(tmp_dir.path()),
            output: tmp_dir.path().join("output.csv"),
            preview: Some(preview_path),
            verbose: true,
        };
        assert!(matches!(run(args), Err(Error::InvalidPreviewPath { .. })));
    }
}
