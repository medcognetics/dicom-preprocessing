use std::path::PathBuf;

use clap::error::ErrorKind;
use clap::Parser;
use dicom::dictionary_std::tags;
use dicom::object::open_file;
use dicom::object::{FileDicomObject, InMemDicomObject};
use dicom_preprocessing::DicomColorType;
use indicatif::ProgressFinish;
use rayon::prelude::*;
use std::fmt;
use tiff::encoder::compression::Compressor;
use tracing::{error, Level};

use dicom_preprocessing::pad::PaddingDirection;
use dicom_preprocessing::preprocess::Preprocessor;
use dicom_preprocessing::resize::DisplayFilterType;
use indicatif::{ProgressBar, ProgressStyle};
use snafu::{OptionExt, Report, ResultExt, Snafu, Whatever};
use std::num::NonZero;
use std::path::Path;
use std::thread::available_parallelism;

use dicom_preprocessing::errors::{
    dicom::{ConvertValueSnafu, MissingPropertySnafu, ReadSnafu},
    DicomError, TiffError,
};
use dicom_preprocessing::file::{DicomFileOperations, InodeSort};
use dicom_preprocessing::save::TiffSaver;
use dicom_preprocessing::transform::volume::DisplayVolumeHandler;

#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(display("Invalid source path: {}", path.display()))]
    InvalidSourcePath { path: PathBuf },

    #[snafu(display("No sources found in source path: {}", path.display()))]
    NoSources { path: PathBuf },

    #[snafu(display("Invalid output path: {}", path.display()))]
    InvalidOutputPath { path: PathBuf },

    #[snafu(display("Failed to create directory: {}", path.display()))]
    CreateDir {
        path: PathBuf,
        #[snafu(source(from(std::io::Error, Box::new)))]
        source: Box<std::io::Error>,
    },

    #[snafu(display("DICOM error on {}: {}", path.display(), source))]
    DicomError {
        path: PathBuf,
        #[snafu(source(from(DicomError, Box::new)))]
        source: Box<DicomError>,
    },

    #[snafu(display("TIFF error on {}: {}", path.display(), source))]
    TiffError {
        path: PathBuf,
        #[snafu(source(from(TiffError, Box::new)))]
        source: Box<TiffError>,
    },
}

#[derive(Debug, Clone, clap::ValueEnum, Default)]
enum SupportedCompressor {
    #[default]
    Packbits,
    Lzw,
    Uncompressed,
}

impl From<SupportedCompressor> for Compressor {
    fn from(value: SupportedCompressor) -> Self {
        match value {
            SupportedCompressor::Packbits => {
                Compressor::Packbits(tiff::encoder::compression::Packbits)
            }
            SupportedCompressor::Lzw => Compressor::Lzw(tiff::encoder::compression::Lzw),
            SupportedCompressor::Uncompressed => {
                Compressor::Uncompressed(tiff::encoder::compression::Uncompressed)
            }
        }
    }
}

impl fmt::Display for SupportedCompressor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let direction_str = match self {
            SupportedCompressor::Packbits => "packbits",
            SupportedCompressor::Lzw => "lzw",
            SupportedCompressor::Uncompressed => "none",
        };
        write!(f, "{}", direction_str)
    }
}

#[derive(Parser, Debug)]
#[command(author = "Scott Chase Waggener", version = env!("CARGO_PKG_VERSION"), about = "Preprocess DICOM files into (multi-frame) TIFFs", long_about = None)]
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
        help = "Crop the image. Pixels with value equal to zero are cropped away.",
        long = "crop",
        short = 'c',
        default_value_t = false
    )]
    crop: bool,

    #[arg(
        help = "Also include pixels with value equal to the data type's maximum value in the crop calculation",
        long = "crop-max",
        short = 'm',
        default_value_t = false
    )]
    crop_max: bool,

    #[arg(
        help = "Do not use connected components for the crop calculation",
        long = "no-components",
        short = 'n',
        default_value_t = false
    )]
    no_components: bool,

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

    #[arg(
        help = "Compression type",
        long = "compressor",
        short = 'z',
        value_parser = clap::value_parser!(SupportedCompressor),
        default_value_t = SupportedCompressor::default(),
    )]
    compressor: SupportedCompressor,

    #[arg(
        help = "How to handle volumes",
        long = "volume-handler",
        short = 'v',
        value_parser = clap::value_parser!(DisplayVolumeHandler),
        default_value_t = DisplayVolumeHandler::default(),
    )]
    volume_handler: DisplayVolumeHandler,

    #[arg(
        help = "Fail on input paths that are not DICOM files, or if any file processing fails",
        long = "strict",
        default_value_t = false
    )]
    strict: bool,
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

fn get_output_path<P: AsRef<Path>>(
    file: &FileDicomObject<InMemDicomObject>,
    dest: P,
) -> Result<PathBuf, DicomError> {
    // Build filepath of form dest/study_instance_uid/sop_instance_uid.tiff
    let study_instance_uid = file
        .get(tags::STUDY_INSTANCE_UID)
        .context(MissingPropertySnafu {
            name: "Study Instance UID",
        })?
        .value()
        .to_str()
        .context(ConvertValueSnafu {
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
        .context(ConvertValueSnafu {
            name: "SOP Instance UID",
        })?
        .into_owned();
    let filename = format!("{}.tiff", sop_instance_uid);
    let dest = dest.as_ref();
    Ok(dest.join(study_instance_uid).join(filename))
}

fn process<P: AsRef<Path>>(
    source: P,
    dest: P,
    preprocessor: &Preprocessor,
    compressor: SupportedCompressor,
    parallelism: bool,
) -> Result<(), Error> {
    let source = source.as_ref();
    let dest = dest.as_ref();

    let mut file = open_file(&source)
        .context(ReadSnafu)
        .context(DicomSnafu { path: source })?;

    let dest = if dest.is_dir() {
        let filepath = get_output_path(&file, dest).context(DicomSnafu { path: source })?;

        // Create the parents
        let parent = filepath.parent().unwrap();
        std::fs::create_dir_all(parent).context(CreateDirSnafu {
            path: parent.to_path_buf(),
        })?;
        filepath
    } else {
        dest.to_path_buf()
    };
    let dest = dest.as_path();

    tracing::info!("Processing {} -> {}", source.display(), dest.display());
    Preprocessor::sanitize_dicom(&mut file);
    let (images, metadata) =
        preprocessor
            .prepare_image(&file, parallelism)
            .context(DicomSnafu {
                path: source.to_path_buf(),
            })?;
    let color_type = DicomColorType::try_from(&file).context(DicomSnafu { path: source })?;

    let saver = TiffSaver::new(compressor.into(), color_type);
    let mut encoder = saver.open_tiff(dest).unwrap();
    images
        .into_iter()
        .try_for_each(|image| saver.save(&mut encoder, &image, &metadata))
        .context(TiffSnafu { path: dest })?;

    Ok(())
}

/// Determines the number of threads to use for parallel processing for multi-frame DICOM files.
/// Frame parallelism will only be used when the number of inputs is small. Otherwise, it as assumed
/// that file parallelism is desired and only a single thread is used for a given file.
fn determine_parallelism(num_inputs: usize) -> bool {
    let max_parallelism = available_parallelism()
        .unwrap_or(NonZero::new(1).unwrap())
        .get();
    (max_parallelism / num_inputs).max(1) > 1
}

fn run(args: Args) -> Result<(), Error> {
    // Parse the sources
    let source = if args.source.is_dir() {
        args.source
            .find_dicoms()
            .map_err(|_| Error::InvalidSourcePath {
                path: args.source.to_path_buf(),
            })?
            .collect::<Vec<_>>()
    } else if args.source.is_file() && args.source.extension().unwrap() == "txt" {
        args.source
            .read_dicom_paths_with_bar()
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

    // Validate the output path
    let dest = match (source.len(), args.output.is_dir()) {
        // No sources
        (0, _) => NoSourcesSnafu { path: args.source }.fail(),
        // Single source
        (1, _) => Ok(args.output),
        // Multiple sources, target not a directory. Cannot continue.
        (_, false) => InvalidOutputPathSnafu { path: args.output }.fail(),
        // Multiple sources, target is a directory
        _ => Ok(args.output),
    }?;

    // Build the preprocessor and compressor
    let preprocessor = Preprocessor {
        crop: args.crop,
        size: args.size,
        filter: args.filter.into(),
        padding_direction: args.padding_direction,
        crop_max: args.crop_max,
        volume_handler: args.volume_handler.into(),
        use_components: !args.no_components,
    };
    let compressor = args.compressor;

    // Create progress bar
    let pb = ProgressBar::new(source.len() as u64).with_finish(ProgressFinish::AndLeave);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{msg} {spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta} @ {per_sec})",
            )
            .unwrap(),
    );
    pb.set_message("Preprocessing DICOM files");

    // Define function to process each file in parallel
    let parallelism = determine_parallelism(source.len());
    let par_func = |file: PathBuf| {
        let result = process(&file, &dest, &preprocessor, compressor.clone(), parallelism);
        pb.inc(1);
        match result {
            Ok(result) => Ok(result),
            Err(e) => {
                error!(
                    "Error processing file {}: {}",
                    file.display(),
                    Report::from_error(&e)
                );
                Err(e)
            }
        }
    };

    // Run processing in parallel
    if args.strict {
        // In strict mode, abort on first error
        source.into_par_iter().try_for_each(par_func)?;
    } else {
        // In non-strict mode, only log errors and continue
        source.into_par_iter().map(par_func).collect::<Vec<_>>();
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{run, Args};
    use crate::DisplayFilterType;
    use crate::PaddingDirection;
    use crate::SupportedCompressor;
    use dicom::dictionary_std::tags;
    use dicom::object::open_file;
    use dicom_preprocessing::crop::{DEFAULT_CROP_ORIGIN, DEFAULT_CROP_SIZE};
    use dicom_preprocessing::pad::ACTIVE_AREA;
    use dicom_preprocessing::resize::DEFAULT_SCALE;

    use dicom_preprocessing::DisplayVolumeHandler;
    use rstest::rstest;
    use std::fs::File;
    use std::io::BufReader;
    use tiff::decoder::Decoder;

    use tiff::tags::ResolutionUnit;
    use tiff::tags::Tag;

    #[rstest]
    #[case("path")]
    #[case("text")]
    #[case("dir")]
    fn test_main(#[case] input_type: &str) {
        // Get the expected SOPInstanceUID from the DICOM
        let dicom_file_path = dicom_test_files::path("pydicom/CT_small.dcm").unwrap();

        // Create a temp directory and copy the test file to it
        let temp_dir = tempfile::tempdir().unwrap();
        let temp_dicom_path = temp_dir.path().join("CT_small.dcm");
        std::fs::copy(&dicom_file_path, &temp_dicom_path).unwrap();

        // Create a temp directory to hold the output
        let output_dir = tempfile::tempdir().unwrap();

        // Decide the source based on the input type
        let source = match input_type {
            "path" => temp_dicom_path,
            "text" => {
                let paths_file_path = temp_dir.path().join("paths.txt");
                std::fs::write(&paths_file_path, temp_dicom_path.to_str().unwrap()).unwrap();
                paths_file_path
            }
            "dir" => temp_dir.path().to_path_buf(),
            _ => unreachable!(),
        };

        // Run the main function
        let args = Args {
            source: source,
            output: output_dir.path().to_path_buf(),
            crop: true,
            size: Some((64, 64)),
            filter: DisplayFilterType::default(),
            padding_direction: PaddingDirection::default(),
            strict: true,
            compressor: SupportedCompressor::default(),
            crop_max: false,
            no_components: false,
            volume_handler: DisplayVolumeHandler::default(),
        };
        run(args).unwrap();

        // Get the StudyInstanceUID and SOPInstanceUID from the DICOM
        let dicom_file = open_file(&dicom_file_path).unwrap();
        let study_instance_uid = dicom_file
            .get(tags::STUDY_INSTANCE_UID)
            .unwrap()
            .value()
            .to_str()
            .unwrap()
            .into_owned();
        let sop_instance_uid = dicom_file
            .get(tags::SOP_INSTANCE_UID)
            .unwrap()
            .value()
            .to_str()
            .unwrap()
            .into_owned();

        // Build the expected output file path
        let filename = format!("{}.tiff", sop_instance_uid);
        let output_file_path = output_dir.path().join(study_instance_uid).join(filename);

        // Open the output file as a TIFF and check the dimensions
        let mut tiff_decoder =
            Decoder::new(BufReader::new(File::open(output_file_path).unwrap())).unwrap();
        let (width, height) = tiff_decoder.dimensions().unwrap();
        assert_eq!(width, 64);
        assert_eq!(height, 64);

        // Check the augmentation tags
        let area = tiff_decoder
            .get_tag(Tag::Unknown(ACTIVE_AREA))
            .unwrap()
            .into_u32_vec()
            .unwrap();
        assert_eq!(area, &[0, 0, 0, 0]);

        let origin = tiff_decoder
            .get_tag(Tag::Unknown(DEFAULT_CROP_ORIGIN))
            .unwrap()
            .into_u32_vec()
            .unwrap();
        assert_eq!(origin, &[64, 64]);

        let size = tiff_decoder
            .get_tag(Tag::Unknown(DEFAULT_CROP_SIZE))
            .unwrap()
            .into_u32_vec()
            .unwrap();
        assert_eq!(size, &[128, 128]);

        let scale = tiff_decoder
            .get_tag(Tag::Unknown(DEFAULT_SCALE))
            .unwrap()
            .into_f32_vec()
            .unwrap();
        assert_eq!(scale, &[0.5, 0.5]);

        // Check the extra tags
        let resolution_unit = ResolutionUnit::from_u16(
            tiff_decoder
                .get_tag(Tag::ResolutionUnit)
                .unwrap()
                .into_u16()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(resolution_unit, ResolutionUnit::Centimeter);

        let resolution_x = tiff_decoder
            .get_tag(Tag::XResolution)
            .unwrap()
            .into_u32_vec()
            .unwrap();
        assert_eq!(resolution_x, &[7, 1]);

        let resolution_y = tiff_decoder
            .get_tag(Tag::YResolution)
            .unwrap()
            .into_u32_vec()
            .unwrap();
        assert_eq!(resolution_y, &[7, 1]);
    }
}
