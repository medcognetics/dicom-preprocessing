use std::io::Write;
use std::path::PathBuf;

use clap::Parser;
use indicatif::ProgressFinish;
use std::fs::File;
use std::io::BufWriter;
use tracing::{error, Level};

use indicatif::{ProgressBar, ProgressStyle};
use snafu::{Report, ResultExt, Snafu, Whatever};

use dicom_preprocessing::file::Inode;
use dicom_preprocessing::manifest::get_manifest_with_progress;

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
}

#[derive(Parser, Debug)]
#[command(author = "Scott Chase Waggener", version = env!("CARGO_PKG_VERSION"), about = "Create a manifest of preprocessed TIFF files", long_about = None)]
struct Args {
    #[arg(help = "Source directory")]
    source: PathBuf,

    #[arg(help = "Output filepath")]
    output: PathBuf,
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

    // Write each entry to CSV file
    let mut csv_file = BufWriter::new(File::create(dest).context(CreateManifestSnafu)?);
    csv_file
        .write_all(b"study_instance_uid,sop_instance_uid,path,inode\n")
        .context(WriteManifestSnafu)?;
    for entry in entries {
        let path = entry.relative_path(&source);
        writeln!(
            csv_file,
            "{},{},{},{}",
            entry.study_instance_uid(),
            entry.sop_instance_uid(),
            path.display(),
            entry.inode().context(ReadInodeSnafu)?,
        )
        .context(WriteManifestSnafu)?;
    }

    Ok(())
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
        assert!(contents.starts_with("study_instance_uid,sop_instance_uid,path,inode\n"));

        // Check we have 3 data rows (one per file)
        let num_lines = contents.lines().count();
        assert_eq!(num_lines, 4); // Header + 3 data rows

        Ok(())
    }

    #[rstest]
    fn test_empty_directory() {
        let temp_dir = TempDir::new().unwrap();
        let output_file = temp_dir.path().join("manifest.csv");

        let args = Args {
            source: temp_dir.path().to_path_buf(),
            output: output_file,
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
        };

        let result = run(args);
        assert!(matches!(result, Err(Error::InvalidOutputPath { .. })));
    }
}
