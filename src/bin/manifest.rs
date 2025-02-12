use arrow::array::{Int64Array, StringArray, UInt32Array};
use arrow::error::ArrowError;
use arrow::record_batch::RecordBatch;
use clap::{Parser, ValueEnum};
use indicatif::ProgressFinish;
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::errors::ParquetError;
use parquet::file::properties::WriterProperties;
use std::fs::File;
use std::io::BufWriter;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::str::FromStr;
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

    #[snafu(display("Error creating manifest: {:?}", source))]
    CreateManifest {
        #[snafu(source(from(std::io::Error, Box::new)))]
        source: Box<std::io::Error>,
    },

    #[snafu(display("Error writing manifest: {:?}", source))]
    WriteManifest {
        #[snafu(source(from(std::io::Error, Box::new)))]
        source: Box<std::io::Error>,
    },

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

#[derive(Debug, Clone, Copy, ValueEnum)]
enum OutputFormat {
    Csv,
    Parquet,
}

impl FromStr for OutputFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "csv" => Ok(OutputFormat::Csv),
            "parquet" => Ok(OutputFormat::Parquet),
            _ => Err(format!("Unknown format: {}", s)),
        }
    }
}

#[derive(Parser, Debug)]
#[command(author = "Scott Chase Waggener", version = env!("CARGO_PKG_VERSION"), about = "Create a manifest of preprocessed TIFF files", long_about = None)]
struct Args {
    #[arg(help = "Source directory")]
    source: PathBuf,

    #[arg(help = "Output filepath")]
    output: PathBuf,

    #[arg(help = "Output format (csv or parquet)", default_value = "csv")]
    format: OutputFormat,
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

fn write_manifest_csv(
    entries: &[ManifestEntry],
    source: &Path,
    dest: &Path,
    pb: &ProgressBar,
) -> Result<(), Error> {
    let mut csv_file = BufWriter::new(File::create(dest).context(CreateManifestSnafu)?);
    csv_file
        .write_all(
            b"study_instance_uid,sop_instance_uid,path,inode,width,height,channels,num_frames\n",
        )
        .context(WriteManifestSnafu)?;
    for entry in entries {
        let path = entry.relative_path(source);
        let dims = entry.dimensions();
        writeln!(
            csv_file,
            "{},{},{},{},{},{},{},{}",
            entry.study_instance_uid(),
            entry.sop_instance_uid(),
            path.display(),
            entry.inode().context(ReadInodeSnafu)?,
            dims.map(|d| d.width).unwrap_or(0),
            dims.map(|d| d.height).unwrap_or(0),
            dims.map(|d| d.channels).unwrap_or(0),
            dims.map(|d| d.num_frames).unwrap_or(0),
        )
        .context(WriteManifestSnafu)?;
        pb.inc(1);
    }
    Ok(())
}

fn write_manifest_parquet(
    entries: &[ManifestEntry],
    source: &Path,
    dest: &Path,
    pb: &ProgressBar,
) -> Result<(), Error> {
    let study_uids: Vec<_> = entries.iter().map(|e| e.study_instance_uid()).collect();
    let sop_uids: Vec<_> = entries.iter().map(|e| e.sop_instance_uid()).collect();
    let paths: Vec<_> = entries
        .iter()
        .map(|e| e.relative_path(source).display().to_string())
        .collect();
    let inodes: Vec<i64> = entries
        .iter()
        .map(|e| Ok(i64::try_from(e.inode().context(ReadInodeSnafu)?).unwrap()))
        .collect::<Result<_, Error>>()?;
    let dims: Vec<_> = entries.iter().map(|e| e.dimensions()).collect();

    let study_uid_array = StringArray::from(study_uids);
    let sop_uid_array = StringArray::from(sop_uids);
    let path_array = StringArray::from(paths);
    let inode_array = Int64Array::from(inodes);
    let width_array = UInt32Array::from(
        dims.iter()
            .map(|d| d.map(|d| d.width).unwrap_or(0) as u32)
            .collect::<Vec<_>>(),
    );
    let height_array = UInt32Array::from(
        dims.iter()
            .map(|d| d.map(|d| d.height).unwrap_or(0) as u32)
            .collect::<Vec<_>>(),
    );
    let channels_array = UInt32Array::from(
        dims.iter()
            .map(|d| d.map(|d| d.channels).unwrap_or(0) as u32)
            .collect::<Vec<_>>(),
    );
    let frames_array = UInt32Array::from(
        dims.iter()
            .map(|d| d.map(|d| d.num_frames).unwrap_or(0) as u32)
            .collect::<Vec<_>>(),
    );

    let schema = arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new(
            "study_instance_uid",
            arrow::datatypes::DataType::Utf8,
            false,
        ),
        arrow::datatypes::Field::new("sop_instance_uid", arrow::datatypes::DataType::Utf8, false),
        arrow::datatypes::Field::new("path", arrow::datatypes::DataType::Utf8, false),
        arrow::datatypes::Field::new("inode", arrow::datatypes::DataType::Int64, false),
        arrow::datatypes::Field::new("width", arrow::datatypes::DataType::UInt32, false),
        arrow::datatypes::Field::new("height", arrow::datatypes::DataType::UInt32, false),
        arrow::datatypes::Field::new("channels", arrow::datatypes::DataType::UInt32, false),
        arrow::datatypes::Field::new("num_frames", arrow::datatypes::DataType::UInt32, false),
    ]);

    let batch = RecordBatch::try_new(
        std::sync::Arc::new(schema),
        vec![
            std::sync::Arc::new(study_uid_array),
            std::sync::Arc::new(sop_uid_array),
            std::sync::Arc::new(path_array),
            std::sync::Arc::new(inode_array),
            std::sync::Arc::new(width_array),
            std::sync::Arc::new(height_array),
            std::sync::Arc::new(channels_array),
            std::sync::Arc::new(frames_array),
        ],
    )
    .context(ArrowSnafu)?;

    let file = File::create(dest).context(CreateManifestSnafu)?;
    let props = WriterProperties::builder().build();
    let mut writer =
        ArrowWriter::try_new(file, batch.schema(), Some(props)).context(ParquetSnafu)?;

    writer.write(&batch).context(ParquetSnafu)?;
    writer.close().context(ParquetSnafu)?;
    pb.inc(entries.len() as u64);

    Ok(())
}

fn run(args: Args) -> Result<(), Error> {
    // Parse the source and dest
    let source = if args.source.is_dir() {
        Ok(args.source)
    } else {
        Err(Error::InvalidSourcePath {
            path: args.source.to_path_buf(),
        })
    }?;
    let dest = if args.output.is_dir() {
        Err(Error::InvalidOutputPath {
            path: args.output.to_path_buf(),
        })
    } else {
        Ok(args.output)
    }?;

    // Read the files and create a manifest
    let entries = get_manifest_with_progress(&source).context(CreateManifestSnafu)?;
    if entries.is_empty() {
        return Err(Error::NoSources {
            path: source.to_path_buf(),
        });
    }
    tracing::info!("Number of entries found: {}", entries.len());

    // Create progress bar
    let pb = ProgressBar::new(entries.len() as u64).with_finish(ProgressFinish::AndLeave);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{msg} {spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta} @ {per_sec})",
            )
            .unwrap(),
    );
    pb.set_message("Writing manifest");

    match args.format {
        OutputFormat::Csv => write_manifest_csv(&entries, &source, &dest, &pb),
        OutputFormat::Parquet => write_manifest_parquet(&entries, &source, &dest, &pb),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;
    use std::fs::{self, File};
    use std::io::Read;
    use std::path::Path;
    use tempfile::TempDir;

    type IOResult<T> = Result<T, std::io::Error>;

    fn setup_test_dir() -> IOResult<(TempDir, Vec<PathBuf>)> {
        let temp_dir = TempDir::new()?;

        // Create study directories
        let study1_dir = temp_dir.path().join("study1");
        let study2_dir = temp_dir.path().join("study2");
        fs::create_dir(&study1_dir)?;
        fs::create_dir(&study2_dir)?;

        // Create test files
        let mut paths = Vec::new();
        paths.push(create_test_tiff(&study1_dir, "image1.tiff")?);
        paths.push(create_test_tiff(&study1_dir, "image2.tiff")?);
        paths.push(create_test_tiff(&study2_dir, "image3.tiff")?);

        Ok((temp_dir, paths))
    }

    fn create_test_tiff(dir: &Path, filename: &str) -> IOResult<PathBuf> {
        let path = dir.join(filename);
        fs::write(&path, b"dummy tiff content")?;
        Ok(path)
    }

    #[rstest]
    fn test_manifest_generation() -> Result<(), Error> {
        let (temp_dir, _) = setup_test_dir().unwrap();
        let output_file = temp_dir.path().join("manifest.csv");

        let args = Args {
            source: temp_dir.path().to_path_buf(),
            output: output_file.clone(),
            format: OutputFormat::Csv,
        };

        run(args)?;

        // Verify manifest file exists and has expected content
        assert!(output_file.exists());

        let mut contents = String::new();
        File::open(&output_file)
            .unwrap()
            .read_to_string(&mut contents)
            .unwrap();

        // Check header
        assert!(contents.starts_with(
            "study_instance_uid,sop_instance_uid,path,inode,width,height,channels,num_frames\n"
        ));

        // Check we have 3 data rows (one per file)
        let num_lines = contents.lines().count();
        assert_eq!(num_lines, 4); // Header + 3 data rows

        Ok(())
    }

    #[rstest]
    fn test_manifest_generation_parquet() -> Result<(), Error> {
        let (temp_dir, _) = setup_test_dir().unwrap();
        let output_file = temp_dir.path().join("manifest.parquet");

        let args = Args {
            source: temp_dir.path().to_path_buf(),
            output: output_file.clone(),
            format: OutputFormat::Parquet,
        };

        run(args)?;

        // Verify manifest file exists
        assert!(output_file.exists());

        // Read parquet file to verify content
        let file = File::open(&output_file).unwrap();
        let reader =
            parquet::arrow::arrow_reader::ParquetRecordBatchReader::try_new(file, 1024).unwrap();
        let batch = reader.into_iter().next().unwrap().unwrap();

        // Check schema fields
        let schema = batch.schema();
        let expected_fields = vec![
            "study_instance_uid",
            "sop_instance_uid",
            "path",
            "inode",
            "width",
            "height",
            "channels",
            "num_frames",
        ];
        for field in expected_fields {
            assert!(schema.field_with_name(field).is_ok());
        }

        // Check we have 3 rows (one per file)
        assert_eq!(batch.num_rows(), 3);

        Ok(())
    }

    #[rstest]
    fn test_empty_directory() {
        let temp_dir = TempDir::new().unwrap();
        let output_file = temp_dir.path().join("manifest.csv");

        let args = Args {
            source: temp_dir.path().to_path_buf(),
            output: output_file,
            format: OutputFormat::Csv,
        };

        let result = run(args);
        assert!(matches!(result, Err(Error::NoSources { .. })));
    }

    #[rstest]
    fn test_invalid_output_directory() {
        let (temp_dir, _) = setup_test_dir().unwrap();
        let output_dir = temp_dir.path().join("output");
        fs::create_dir(&output_dir).unwrap();

        let args = Args {
            source: temp_dir.path().to_path_buf(),
            output: output_dir,
            format: OutputFormat::Csv,
        };

        let result = run(args);
        assert!(matches!(result, Err(Error::InvalidOutputPath { .. })));
    }
}
