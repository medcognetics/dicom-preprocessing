use std::path::PathBuf;

use clap::error::ErrorKind;
use clap::Parser;
use dicom::dictionary_std::tags;
use dicom::object::open_file;
use dicom::object::ReadError;
use rayon::prelude::*;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use tracing::{error, Level};

use dicom_preprocessing::pad::PaddingDirection;
use dicom_preprocessing::preprocess::preprocess;
use dicom_preprocessing::resize::DisplayFilterType;
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use rust_search::SearchBuilder;
use snafu::{OptionExt, Report, ResultExt, Snafu, Whatever};
use std::path::Path;

use dicom_preprocessing::preprocess::Error as PreprocessingError;

#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(display("Invalid source path: {}", path.display()))]
    InvalidSourcePath { path: PathBuf },
    #[snafu(display("No sources found in source path: {}", path.display()))]
    NoSources { path: PathBuf },
    #[snafu(display("Invalid output path: {}", path.display()))]
    InvalidOutputPath { path: PathBuf },
    #[snafu(display("Failed to read DICOM file: {}", path.display()))]
    DicomRead {
        path: PathBuf,
        #[snafu(source(from(ReadError, Box::new)))]
        source: Box<ReadError>,
    },
    /// missing key property {name}
    MissingProperty { name: &'static str },
    /// property {name} contains an invalid value
    InvalidPropertyValue {
        name: &'static str,
        #[snafu(source(from(dicom::core::value::ConvertValueError, Box::new)))]
        source: Box<dicom::core::value::ConvertValueError>,
    },
    Preprocessing {
        #[snafu(source(from(PreprocessingError, Box::new)))]
        source: Box<PreprocessingError>,
    },
    CreateDir {
        path: PathBuf,
        #[snafu(source(from(std::io::Error, Box::new)))]
        source: Box<std::io::Error>,
    },
}

fn is_dicom_file(path: &Path) -> bool {
    let has_extension = path.extension().is_some();
    let ext_is_dicom = path
        .extension()
        .map(|ext| ext.to_ascii_lowercase())
        .map(|ext| ext == "dcm" || ext == "dicom")
        .unwrap_or(false);

    // Try to check only the extension if possible
    if has_extension && ext_is_dicom {
        return path.is_file();
    } else if has_extension {
        return false;
    }

    if path.is_dir() {
        return false;
    }

    // Extensionless file, we must check the DICM prefix
    const DICM_PREFIX: &[u8; 4] = b"DICM";
    let has_prefix = File::open(path)
        .and_then(|mut file| {
            file.seek(SeekFrom::Start(128))?;
            let mut buffer = [0; 4];
            file.read_exact(&mut buffer)?;
            Ok(buffer)
        })
        .map_or(false, |buffer| &buffer == DICM_PREFIX);
    has_prefix
}

fn find_dicom_files(dir: &PathBuf) -> impl Iterator<Item = PathBuf> {
    // Set up spinner, iterating may files may take some time
    let spinner = ProgressBar::new_spinner();
    spinner.set_message("Searching for DICOM files");
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
        .filter(move |file| is_dicom_file(file) && file.is_file())
}

#[derive(Parser, Debug)]
#[command(author = "Scott Chase Waggener", version = "0.1.0", about = "Preprocess DICOM files", long_about = None)]
struct Args {
    #[arg(
        help = "Source path. Can be a DICOM file, directory, or a text file with DICOM file paths"
    )]
    source: PathBuf,

    #[arg(
        help = "Output path. Can be a directory (for multiple files) or a file (for a single file)"
    )]
    output: PathBuf,

    #[arg(
        help = "Crop the image",
        long = "crop",
        short = 'c',
        default_value_t = false
    )]
    crop: bool,

    #[arg(
        help = "Target size (width,height)",
        long = "size",
        short = 's',
        value_parser = clap::builder::ValueParser::new(|s: &str| {
            let parts: Vec<&str> = s.split(',').collect();
            if parts.len() == 2 {
                let width = parts[0].parse::<u32>().map_err(|_| clap::Error::raw(ErrorKind::InvalidValue, "Invalid width"))?;
                let height = parts[1].parse::<u32>().map_err(|_| clap::Error::raw(ErrorKind::InvalidValue, "Invalid height"))?;
                Ok((width, height))
            } else {
                Err(clap::Error::raw(ErrorKind::InvalidValue, "Size must be in the format width,height"))
            }
        })
    )]
    size: Option<(u32, u32)>,

    #[arg(
        help = "Filter type",
        long = "filter",
        short = 'f',
        value_parser = clap::value_parser!(DisplayFilterType),
        default_value_t = DisplayFilterType::default(),
    )]
    filter: DisplayFilterType,

    #[arg(
        help = "Padding direction",
        long = "padding",
        short = 'p',
        value_parser = clap::value_parser!(PaddingDirection),
        default_value_t = PaddingDirection::default(),
    )]
    padding_direction: PaddingDirection,
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

fn process(
    source: &PathBuf,
    dest: &PathBuf,
    crop: bool,
    size: Option<(u32, u32)>,
    filter: DisplayFilterType,
    padding_direction: PaddingDirection,
) -> Result<(), Error> {
    let file = open_file(&source).context(DicomReadSnafu { path: source })?;

    let dest = if dest.is_dir() {
        // Build filepath of form dest/study_instance_uid/sop_instance_uid.tiff
        let study_instance_uid = file
            .get(tags::STUDY_INSTANCE_UID)
            .context(MissingPropertySnafu {
                name: "Study Instance UID",
            })?
            .value()
            .to_str()
            .context(InvalidPropertyValueSnafu {
                name: "Study Instance UID",
            })?
            .into_owned();
        let sop_instance_uid = file
            .get(tags::SOP_INSTANCE_UID)
            .context(MissingPropertySnafu {
                name: "SOP Instance UID",
            })?
            .value()
            .to_str()
            .context(InvalidPropertyValueSnafu {
                name: "SOP Instance UID",
            })?
            .into_owned();
        let filepath = dest
            .join(study_instance_uid)
            .join(sop_instance_uid)
            .with_extension("tiff");

        // Create the parents
        let parent = filepath.parent().unwrap();
        std::fs::create_dir_all(parent).context(CreateDirSnafu { path: parent })?;
        filepath
    } else {
        dest.clone()
    };

    tracing::info!("Processing {} -> {}", source.display(), dest.display());
    preprocess(
        &file,
        dest,
        crop,
        size,
        filter.into(),
        padding_direction,
        // TODO: Make this configurable
        tiff::encoder::compression::Packbits,
    )
    .context(PreprocessingSnafu)?;
    Ok(())
}

fn run(args: Args) -> Result<(), Error> {
    // Parse the sources
    let source = if args.source.is_dir() {
        find_dicom_files(&args.source).collect()
    } else if args.source.is_file() && args.source.extension().unwrap() == "txt" {
        std::fs::read_to_string(&args.source)
            .map_err(|_| Error::InvalidSourcePath { path: args.source })?
            .lines()
            .map(PathBuf::from)
            .filter(|path| is_dicom_file(path))
            .collect()
    } else {
        vec![args.source]
    };

    tracing::info!("Number of sources found: {}", source.len());

    // Validate the output path
    let dest = match (source.len(), args.output.is_dir()) {
        // No sources
        (0, _) => NoSourcesSnafu { path: args.output }.fail(),
        // Single source
        (1, _) => Ok(args.output),
        // Multiple sources, target not a directory. Cannot continue.
        (_, false) => InvalidOutputPathSnafu { path: args.output }.fail(),
        // Multiple sources, target is a directory
        _ => Ok(args.output),
    }?;

    // Create progress bar
    let pb = ProgressBar::new(source.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{msg} {spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta} @ {per_sec})",
            )
            .unwrap(),
    );
    pb.set_message("Preprocessing DICOM files");

    // Process each file in parallel
    source
        .into_par_iter()
        .progress_with(pb)
        .try_for_each(|file| {
            process(
                &file,
                &dest,
                args.crop,
                args.size,
                args.filter,
                args.padding_direction,
            )
        })?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_main_output() {
        // Capture the output of the main function
        let output = std::panic::catch_unwind(|| {
            main();
        });

        // Ensure the main function runs without panicking
        assert!(output.is_ok());
    }
}
