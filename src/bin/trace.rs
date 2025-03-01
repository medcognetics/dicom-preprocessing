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
    IOError {
        #[snafu(source(from(std::io::Error, Box::new)))]
        source: Box<std::io::Error>,
    },

    #[snafu(display("Error reading CSV: {:?}", source))]
    CSVError {
        #[snafu(source(from(csv::Error, Box::new)))]
        source: Box<csv::Error>,
    },

    #[snafu(display("Error parsing integer: {:?}", source))]
    ParseIntError {
        #[snafu(source(from(ParseIntError, Box::new)))]
        source: Box<ParseIntError>,
    },

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

    #[snafu(display("Error reading TIFF: {:?}", source))]
    TiffReadError {
        #[snafu(source(from(TiffError, Box::new)))]
        source: Box<TiffError>,
    },

    #[snafu(display("Error writing TIFF: {:?}", source))]
    TiffWriteError {
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

impl Into<(Coord, Coord)> for Trace {
    fn into(self) -> (Coord, Coord) {
        (
            Coord::new(self.x_min, self.y_min),
            Coord::new(self.x_max, self.y_max),
        )
    }
}

impl Into<(Coord, Coord)> for &Trace {
    fn into(self) -> (Coord, Coord) {
        (
            Coord::new(self.x_min, self.y_min),
            Coord::new(self.x_max, self.y_max),
        )
    }
}

impl<T: From<u32>> Into<(T, T, T, T)> for Trace {
    fn into(self) -> (T, T, T, T) {
        (
            self.x_min.into(),
            self.y_min.into(),
            self.x_max.into(),
            self.y_max.into(),
        )
    }
}

impl<T: From<u32>> Into<(T, T, T, T)> for &Trace {
    fn into(self) -> (T, T, T, T) {
        (
            self.x_min.into(),
            self.y_min.into(),
            self.x_max.into(),
            self.y_max.into(),
        )
    }
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

    #[arg(
        help = "Directory to save previews to",
        long = "preview",
        short = 'p',
        default_value = None,
    )]
    preview: Option<PathBuf>,
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

fn run(args: Args) -> Result<(), Error> {
    // Validate that the preview directory exists
    let preview = match &args.preview {
        Some(preview) if !preview.is_dir() => {
            return Err(Error::InvalidPreviewPath {
                path: preview.clone(),
            })
        }
        Some(preview) => Some(preview),
        None => None,
    };

    // Parse the sources
    let source = if args.source.is_dir() {
        args.source
            .find_tiffs()
            .map_err(|_| Error::InvalidSourcePath {
                path: args.source.to_path_buf(),
            })?
            .collect::<Vec<_>>()
    } else if args.source.is_file() && args.source.extension().unwrap() == "txt" {
        args.source
            .iter()
            .read_tiff_paths_with_bar()
            .map_err(|_| Error::InvalidSourcePath {
                path: args.source.to_path_buf(),
            })?
            .collect::<Vec<_>>()
    } else {
        vec![args.source.clone()]
    };
    let source = source
        .into_iter()
        .sorted_by_inode_with_progress()
        .collect::<Vec<_>>();

    tracing::info!("Number of sources found: {}", source.len());
    if source.len() == 0 {
        return Err(Error::NoSources { path: args.source });
    }

    // Read the source into a hashmap of sop_instance_uid to list of traces
    let traces = match args.traces.extension().and_then(|ext| ext.to_str()) {
        Some("csv") => {
            let mut reader = CsvReader::from_path(&args.traces).context(CSVSnafu)?;
            let mut traces = HashMap::new();
            for result in reader.deserialize() {
                let record: HashMap<String, String> = result.context(CSVSnafu)?;
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
            traces
        }
        Some("parquet") => {
            let file = File::open(&args.traces).context(IOSnafu)?;
            let mut reader =
                ParquetRecordBatchReader::try_new(file, BATCH_SIZE).context(ParquetSnafu)?;
            let mut traces = HashMap::new();
            while let Some(result) = reader.next() {
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
                    let hash = hash_array.value(i) as i64;
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
            traces
        }
        _ => return Err(Error::InvalidTracesFormat { path: args.traces }),
    };
    tracing::info!("Loaded {} traces", traces.len());

    // Process in parallel
    let pb = default_bar(source.len() as u64);
    pb.set_message("Processing traces");
    let results = source
        .into_par_iter()
        .progress_with(pb)
        .map(|s| process(&s, &traces, preview))
        .collect::<Result<Vec<_>, Error>>()?
        .into_iter()
        .filter(|(_, traces)| !traces.is_empty())
        .collect::<Vec<_>>();

    // Write the results to the output path
    let output_format =
        OutputFormat::from_extension(&args.output).map_err(|_| Error::InvalidOutputExtension {
            path: args.output.clone(),
            supported: vec!["csv", "parquet"],
        })?;

    let pb = default_bar(results.len() as u64);
    pb.set_message("Writing outputs");
    match output_format {
        OutputFormat::Csv => write_traces_csv(&results, &args.output, &pb)?,
        OutputFormat::Parquet => write_traces_parquet(&results, &args.output, &pb)?,
    }
    Ok(())
}

fn process(
    source: &PathBuf,
    trace_metadata: &HashMap<String, Vec<Trace>>,
    output: Option<&PathBuf>,
) -> Result<(String, Vec<Trace>), Error> {
    // Open TIFF
    let file = File::open(&source).context(IOSnafu)?;
    let reader = BufReader::new(file);
    let mut decoder = Decoder::new(reader)
        .map_err(|e| TiffError::from(e))
        .context(TiffReadSnafu)?;

    // Read metadata find corresponding traces
    let sop_instance_uid = source
        .file_stem()
        .unwrap_or_default()
        .to_str()
        .unwrap_or_default();
    let metadata = PreprocessingMetadata::try_from(&mut decoder).context(TiffReadSnafu)?;
    let traces = match trace_metadata.get(sop_instance_uid) {
        Some(traces) => traces,
        None => return Ok((sop_instance_uid.to_string(), vec![])),
    };

    // Adjust traces
    let traces = traces
        .into_iter()
        .map(|trace| {
            let (min, max): (Coord, Coord) = trace.into();
            let min = metadata.apply(&min);
            let max = metadata.apply(&max);
            let (x_min, y_min): (u32, u32) = min.into();
            let (x_max, y_max): (u32, u32) = max.into();
            Trace::new(
                trace.sop_instance_uid().to_string(),
                trace.hash(),
                x_min,
                x_max,
                y_min,
                y_max,
            )
        })
        .filter(|trace| trace.is_valid())
        .collect::<Vec<_>>();

    if traces.is_empty() {
        return Ok((sop_instance_uid.to_string(), vec![]));
    }

    // If an output path is provided, save the TIFF with traces overlaid
    if let Some(output) = output {
        // Determine output filename/path
        let study_instance_uid = source
            .parent()
            .unwrap()
            .file_name()
            .unwrap()
            .to_str()
            .unwrap_or_default();
        let path = output.join(study_instance_uid);
        let filename = format!("{}.tiff", sop_instance_uid);
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
            .map_err(|e| TiffError::from(e))
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
        .write_all(b"sop_instance_uid,x_min,x_max,y_min,y_max\n")
        .context(IOSnafu)?;
    for (sop_instance_uid, traces) in entries {
        for trace in traces {
            let (x_min, y_min, x_max, y_max): (u32, u32, u32, u32) = trace.into();
            writeln!(
                csv_file,
                "{},{},{},{},{}",
                sop_instance_uid, x_min, x_max, y_min, y_max,
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
