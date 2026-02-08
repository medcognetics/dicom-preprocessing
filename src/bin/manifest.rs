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
use std::path::Component;
use std::path::Path;
use std::path::PathBuf;
use tracing::{error, Level};

use indicatif::{ProgressBar, ProgressStyle};
use snafu::{Report, ResultExt, Snafu, Whatever};

use dicom_preprocessing::file::Inode;
use dicom_preprocessing::manifest::{get_manifest_with_progress, ManifestEntry};

const DEFAULT_OUTPUT_FILENAME: &str = "manifest.parquet";

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
    Arrow {
        #[snafu(source(from(ArrowError, Box::new)))]
        source: Box<ArrowError>,
    },

    #[snafu(display("Parquet error: {:?}", source))]
    Parquet {
        #[snafu(source(from(ParquetError, Box::new)))]
        source: Box<ParquetError>,
    },

    #[snafu(display("Invalid output extension for {}, supported extensions: {}", path.display(), supported.join(", ")))]
    InvalidOutputExtension {
        path: PathBuf,
        supported: Vec<&'static str>,
    },

    #[snafu(display("Could not resolve path {} relative to manifest directory {}", entry_path.display(), manifest_dir.display()))]
    InvalidPathRelation {
        entry_path: PathBuf,
        manifest_dir: PathBuf,
    },

    #[snafu(display("Could not determine current working directory: {:?}", source))]
    CurrentDirectory {
        #[snafu(source(from(std::io::Error, Box::new)))]
        source: Box<std::io::Error>,
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

#[derive(Parser, Debug)]
#[command(author = "Scott Chase Waggener", version = env!("CARGO_PKG_VERSION"), about = "Create a manifest of preprocessed TIFF files", long_about = None)]
struct Args {
    #[arg(help = "Source directory")]
    source: PathBuf,

    #[arg(help = format!("Output filepath, extension determines format: .csv or .parquet (default: <source>/{DEFAULT_OUTPUT_FILENAME})"))]
    output: Option<PathBuf>,
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
    manifest_dir: &Path,
    dest: &Path,
    pb: &ProgressBar,
) -> Result<(), Error> {
    let mut csv_file = BufWriter::new(File::create(dest).context(CreateManifestSnafu)?);
    csv_file
        .write_all(
            b"study_instance_uid,series_instance_uid,sop_instance_uid,path,inode,width,height,channels,num_frames\n",
        )
        .context(WriteManifestSnafu)?;
    for entry in entries {
        let path = path_relative_to_manifest(entry.path(), manifest_dir)?;
        let dims = entry.dimensions();
        writeln!(
            csv_file,
            "{},{},{},{},{},{},{},{},{}",
            entry.study_instance_uid(),
            entry.series_instance_uid(),
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
    manifest_dir: &Path,
    dest: &Path,
    pb: &ProgressBar,
) -> Result<(), Error> {
    let study_uids: Vec<_> = entries.iter().map(|e| e.study_instance_uid()).collect();
    let series_uids: Vec<_> = entries.iter().map(|e| e.series_instance_uid()).collect();
    let sop_uids: Vec<_> = entries.iter().map(|e| e.sop_instance_uid()).collect();
    let paths: Vec<_> = entries
        .iter()
        .map(|e| path_relative_to_manifest(e.path(), manifest_dir).map(|p| p.display().to_string()))
        .collect::<Result<_, Error>>()?;
    let inodes: Vec<i64> = entries
        .iter()
        .map(|e| Ok(i64::try_from(e.inode().context(ReadInodeSnafu)?).unwrap()))
        .collect::<Result<_, Error>>()?;
    let dims: Vec<_> = entries.iter().map(|e| e.dimensions()).collect();

    let study_uid_array = StringArray::from(study_uids);
    let series_uid_array = StringArray::from(series_uids);
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
        arrow::datatypes::Field::new(
            "series_instance_uid",
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
            std::sync::Arc::new(series_uid_array),
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
    let dest = match args.output {
        Some(output) if output.is_dir() => Err(Error::InvalidOutputPath { path: output }),
        Some(output) => Ok(output),
        None => Ok(source.join(DEFAULT_OUTPUT_FILENAME)),
    }?;
    let manifest_dir = dest
        .parent()
        .ok_or_else(|| Error::InvalidOutputPath { path: dest.clone() })?;

    let format = OutputFormat::from_extension(&dest)?;

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

    match format {
        OutputFormat::Csv => write_manifest_csv(&entries, manifest_dir, &dest, &pb),
        OutputFormat::Parquet => write_manifest_parquet(&entries, manifest_dir, &dest, &pb),
    }
}

fn path_relative_to_manifest(entry_path: &Path, manifest_dir: &Path) -> Result<PathBuf, Error> {
    let manifest_dir = if manifest_dir.as_os_str().is_empty() {
        Path::new(".")
    } else {
        manifest_dir
    };

    if let Ok(path) = entry_path.strip_prefix(manifest_dir) {
        return Ok(path.to_path_buf());
    }

    let cwd = std::env::current_dir().context(CurrentDirectorySnafu)?;
    let entry_path = if entry_path.is_absolute() {
        entry_path.to_path_buf()
    } else {
        cwd.join(entry_path)
    };
    let manifest_dir = if manifest_dir.is_absolute() {
        manifest_dir.to_path_buf()
    } else {
        cwd.join(manifest_dir)
    };

    if let Ok(path) = entry_path.strip_prefix(&manifest_dir) {
        return Ok(path.to_path_buf());
    }

    let entry_components = entry_path.components().collect::<Vec<_>>();
    let manifest_components = manifest_dir.components().collect::<Vec<_>>();

    let mut shared_prefix_len = 0;
    while shared_prefix_len < entry_components.len()
        && shared_prefix_len < manifest_components.len()
        && entry_components[shared_prefix_len] == manifest_components[shared_prefix_len]
    {
        shared_prefix_len += 1;
    }

    if !roots_match(entry_components.first(), manifest_components.first()) {
        return Err(Error::InvalidPathRelation {
            entry_path: entry_path.to_path_buf(),
            manifest_dir: manifest_dir.to_path_buf(),
        });
    }

    let mut relative_path = PathBuf::new();
    for component in &manifest_components[shared_prefix_len..] {
        if matches!(component, Component::Normal(_)) {
            relative_path.push("..");
        }
    }
    for component in &entry_components[shared_prefix_len..] {
        relative_path.push(component.as_os_str());
    }

    if relative_path.as_os_str().is_empty() {
        Ok(PathBuf::from("."))
    } else {
        Ok(relative_path)
    }
}

fn roots_match(entry_root: Option<&Component<'_>>, manifest_root: Option<&Component<'_>>) -> bool {
    match (entry_root, manifest_root) {
        (Some(Component::Prefix(e)), Some(Component::Prefix(m))) => e == m,
        (Some(Component::RootDir), Some(Component::RootDir)) => true,
        (Some(Component::CurDir), Some(Component::CurDir)) => true,
        (Some(Component::Normal(_)), Some(Component::Normal(_))) => true,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::StringArray;
    use rstest::rstest;
    use std::collections::HashSet;
    use std::fs::{self, File};
    use std::io::Read;
    use std::path::Path;
    use tempfile::TempDir;

    type IOResult<T> = Result<T, std::io::Error>;

    const NUM_FILES: usize = 3;

    fn setup_test_dir() -> IOResult<(TempDir, Vec<PathBuf>)> {
        let temp_dir = TempDir::new()?;

        // Create study and series directories
        let study1_series1_dir = temp_dir.path().join("study1").join("series1");
        fs::create_dir_all(&study1_series1_dir)?;

        // Create test files
        let mut paths = Vec::new();
        for i in 0..NUM_FILES {
            paths.push(create_test_tiff(
                &study1_series1_dir,
                &format!("image{i}.tiff"),
            )?);
        }

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
            output: Some(output_file.clone()),
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
            "study_instance_uid,series_instance_uid,sop_instance_uid,path,inode,width,height,channels,num_frames\n"
        ));

        // Check we have 3 data rows (one per file)
        let num_lines = contents.lines().count();
        assert_eq!(num_lines, NUM_FILES + 1); // Header + NUM_FILES data rows
        let expected_path = format!(
            "study1{}series1{}image0.tiff",
            std::path::MAIN_SEPARATOR,
            std::path::MAIN_SEPARATOR
        );
        assert!(contents.contains(&expected_path));

        Ok(())
    }

    #[rstest]
    fn test_manifest_generation_parquet() -> Result<(), Error> {
        let (temp_dir, _) = setup_test_dir().unwrap();
        let output_file = temp_dir.path().join("manifest.parquet");

        let args = Args {
            source: temp_dir.path().to_path_buf(),
            output: Some(output_file.clone()),
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
            "series_instance_uid",
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

        // Check we have NUM_FILES rows (one per file)
        assert_eq!(batch.num_rows(), NUM_FILES);

        Ok(())
    }

    #[rstest]
    fn test_manifest_paths_relative_to_output_manifest_directory() -> Result<(), Error> {
        let temp_dir = TempDir::new().unwrap();
        let images_dir = temp_dir.path().join("images");
        let study_series_dir = images_dir.join("study1").join("series1");
        fs::create_dir_all(&study_series_dir).unwrap();
        create_test_tiff(&study_series_dir, "image0.tiff").unwrap();

        let output_file = temp_dir.path().join("manifest.csv");
        run(Args {
            source: images_dir.clone(),
            output: Some(output_file.clone()),
        })?;

        let mut contents = String::new();
        File::open(&output_file)
            .unwrap()
            .read_to_string(&mut contents)
            .unwrap();
        let expected_path = format!(
            "images{}study1{}series1{}image0.tiff",
            std::path::MAIN_SEPARATOR,
            std::path::MAIN_SEPARATOR,
            std::path::MAIN_SEPARATOR
        );
        assert!(contents
            .lines()
            .skip(1)
            .all(|line| line.contains(&expected_path)));

        Ok(())
    }

    #[rstest]
    fn test_manifest_only_uses_source_subtree() -> Result<(), Error> {
        let temp_dir = TempDir::new().unwrap();
        let images_dir = temp_dir
            .path()
            .join("images")
            .join("study1")
            .join("series1");
        let masks_dir = temp_dir.path().join("masks").join("studyX").join("seriesX");
        fs::create_dir_all(&images_dir).unwrap();
        fs::create_dir_all(&masks_dir).unwrap();
        create_test_tiff(&images_dir, "image0.tiff").unwrap();
        create_test_tiff(&masks_dir, "mask0.tiff").unwrap();

        let output_file = temp_dir.path().join("manifest.parquet");
        run(Args {
            source: temp_dir.path().join("images"),
            output: Some(output_file.clone()),
        })?;

        let file = File::open(&output_file).unwrap();
        let reader =
            parquet::arrow::arrow_reader::ParquetRecordBatchReader::try_new(file, 1024).unwrap();
        let paths = reader
            .into_iter()
            .flat_map(|batch| {
                let batch = batch.unwrap();
                let paths = batch
                    .column_by_name("path")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap()
                    .iter()
                    .map(|p| p.unwrap().to_string())
                    .collect::<Vec<_>>();
                paths
            })
            .collect::<HashSet<_>>();

        assert_eq!(paths.len(), 1);
        assert!(paths.contains(&format!(
            "images{}study1{}series1{}image0.tiff",
            std::path::MAIN_SEPARATOR,
            std::path::MAIN_SEPARATOR,
            std::path::MAIN_SEPARATOR
        )));

        Ok(())
    }

    #[rstest]
    fn test_empty_directory() {
        let temp_dir = TempDir::new().unwrap();
        let output_file = temp_dir.path().join("manifest.csv");

        let args = Args {
            source: temp_dir.path().to_path_buf(),
            output: Some(output_file),
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
            output: Some(output_dir),
        };

        let result = run(args);
        assert!(matches!(result, Err(Error::InvalidOutputPath { .. })));
    }

    #[rstest]
    fn test_default_output_path() -> Result<(), Error> {
        let (temp_dir, _) = setup_test_dir().unwrap();

        let args = Args {
            source: temp_dir.path().to_path_buf(),
            output: None,
        };

        run(args)?;

        let expected_output = temp_dir.path().join(DEFAULT_OUTPUT_FILENAME);
        assert!(expected_output.exists());
        Ok(())
    }
}
