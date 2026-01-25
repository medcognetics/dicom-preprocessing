use std::path::PathBuf;

use clap::error::ErrorKind;
use clap::Parser;
use dicom::dictionary_std::tags;
use dicom::object::open_file;
use dicom::object::{FileDicomObject, InMemDicomObject};
use dicom::pixeldata::{ConvertOptions, VoiLutOption, WindowLevel};
use dicom_preprocessing::DicomColorType;
use indicatif::ProgressFinish;
use rayon::prelude::*;
use std::fmt;
use tiff::encoder::compression::Compressor;
use tracing::{error, Level};

use dicom_preprocessing::pad::PaddingDirection;
use dicom_preprocessing::preprocess::Preprocessor;
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
use dicom_preprocessing::transform::resize::FilterType;
use dicom_preprocessing::transform::volume::{
    AverageIntensity, CentralSlice, DisplayVolumeHandler, GaussianWeighted, InterpolateVolume,
    KeepVolume, LaplacianMip, MaxIntensity, SoftMip, VolumeHandler,
    DEFAULT_INTERPOLATE_TARGET_FRAMES,
};

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
        write!(f, "{direction_str}")
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
        help = "Border fraction to exclude from crop calculation and grow final crop by",
        long = "border-frac",
        short = 'b',
        value_parser = clap::builder::ValueParser::new(|s: &str| {
            let value = s.parse::<f32>().map_err(|_| clap::Error::raw(ErrorKind::InvalidValue, "Invalid border fraction"))?;
            if value < 0.0 || value > 0.5 {
                Err(clap::Error::raw(ErrorKind::InvalidValue, "Border fraction must be between 0.0 and 0.5"))
            } else {
                Ok(value)
            }
        })
    )]
    border_frac: Option<f32>,

    #[arg(
        help = "Target size (width,height)",
        long = "size",
        short = 's',
        conflicts_with = "spacing",
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
        help = "Target pixel/voxel spacing in mm (x,y or x,y,z)",
        long = "spacing",
        conflicts_with = "size",
        value_parser = clap::builder::ValueParser::new(|s: &str| {
            let parts: Vec<&str> = s.split(',').collect();
            if parts.len() == 2 {
                let x = parts[0].parse::<f32>().map_err(|_| clap::Error::raw(ErrorKind::InvalidValue, "Invalid x spacing"))?;
                let y = parts[1].parse::<f32>().map_err(|_| clap::Error::raw(ErrorKind::InvalidValue, "Invalid y spacing"))?;
                Ok((x, y, None))
            } else if parts.len() == 3 {
                let x = parts[0].parse::<f32>().map_err(|_| clap::Error::raw(ErrorKind::InvalidValue, "Invalid x spacing"))?;
                let y = parts[1].parse::<f32>().map_err(|_| clap::Error::raw(ErrorKind::InvalidValue, "Invalid y spacing"))?;
                let z = parts[2].parse::<f32>().map_err(|_| clap::Error::raw(ErrorKind::InvalidValue, "Invalid z spacing"))?;
                Ok((x, y, Some(z)))
            } else {
                Err(clap::Error::raw(ErrorKind::InvalidValue, "Spacing must be in the format x,y or x,y,z"))
            }
        })
    )]
    spacing: Option<(f32, f32, Option<f32>)>,

    #[arg(
        help = "Filter type",
        long = "filter",
        short = 'f',
        value_parser = clap::value_parser!(FilterType),
        default_value_t = FilterType::default(),
    )]
    filter: FilterType,

    #[arg(
        help = "Padding direction",
        long = "padding",
        short = 'p',
        value_parser = clap::value_parser!(PaddingDirection),
        default_value_t = PaddingDirection::default(),
    )]
    padding_direction: PaddingDirection,

    #[arg(help = "Disable padding", long = "no-padding", default_value_t = false)]
    no_padding: bool,

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
        help = "Target number of frames when using interpolation",
        long = "target-frames",
        short = 't',
        default_value_t = DEFAULT_INTERPOLATE_TARGET_FRAMES,
        requires = "volume_handler",
    )]
    target_frames: u32,

    // LaplacianMip-specific options
    #[arg(
        help = "LaplacianMip: bilateral filter range sigma (lower = sharper, preserves calcifications). Literature recommends 0.005-0.02",
        long = "laplacian-sigma-r",
        default_value_t = 0.015
    )]
    laplacian_sigma_r: f32,

    #[arg(
        help = "LaplacianMip: bilateral filter spatial sigma",
        long = "laplacian-sigma-s",
        default_value_t = 3.0
    )]
    laplacian_sigma_s: f32,

    #[arg(
        help = "LaplacianMip: detail enhancement beta for fine levels 0-2 (higher = more calcification enhancement)",
        long = "laplacian-beta-fine",
        default_value_t = 2.0
    )]
    laplacian_beta_fine: f32,

    #[arg(
        help = "LaplacianMip: detail enhancement beta for middle levels 3-4",
        long = "laplacian-beta-mid",
        default_value_t = 1.0
    )]
    laplacian_beta_mid: f32,

    #[arg(
        help = "LaplacianMip: number of pyramid levels",
        long = "laplacian-levels",
        default_value_t = 7
    )]
    laplacian_levels: usize,

    #[arg(
        help = "LaplacianMip: frames to skip at start of volume",
        long = "laplacian-skip-start",
        default_value_t = 5
    )]
    laplacian_skip_start: u32,

    #[arg(
        help = "LaplacianMip: frames to skip at end of volume",
        long = "laplacian-skip-end",
        default_value_t = 5
    )]
    laplacian_skip_end: u32,

    #[arg(
        help = "LaplacianMip: skip bilateral filtering at first N pyramid levels (0=filter all, 3=calc-friendly mode)",
        long = "laplacian-skip-bilateral-levels",
        default_value_t = 0
    )]
    laplacian_skip_bilateral_levels: usize,

    #[arg(
        help = "LaplacianMip: first pyramid level to include MIP Laplacian (0=all levels, 3=exclude fine levels 0-2)",
        long = "laplacian-mip-levels-start",
        default_value_t = 0
    )]
    laplacian_mip_levels_start: usize,

    #[arg(
        help = "LaplacianMip: weight for MIP Laplacian contribution (1.0=equal, >1=amplify calcifications)",
        long = "laplacian-mip-weight",
        default_value_t = 1.0
    )]
    laplacian_mip_weight: f32,

    #[arg(
        help = "LaplacianMip: use MIP Gaussian as reconstruction base at fine levels (preserves calcification centers)",
        long = "laplacian-use-mip-gaussian",
        default_value_t = false
    )]
    laplacian_use_mip_gaussian: bool,

    #[arg(
        help = "LaplacianMip: number of fine levels to use MIP Gaussian base (default 3 = levels 0-2)",
        long = "laplacian-mip-gaussian-levels",
        default_value_t = 3
    )]
    laplacian_mip_gaussian_levels: usize,

    #[arg(
        help = "Fail on input paths that are not DICOM files, or if any file processing fails",
        long = "strict",
        default_value_t = false
    )]
    strict: bool,

    #[arg(
        help = "Window center and width",
        long = "window",
        short = 'w',
        allow_negative_numbers = true,
        allow_hyphen_values = true,
        value_parser = clap::builder::ValueParser::new(|s: &str| {
            let parts: Vec<&str> = s.split(',').collect();
            if parts.len() == 2 {
                let center = parts[0].parse::<f64>().map_err(|_| clap::Error::raw(ErrorKind::InvalidValue, "Invalid center"))?;
                let width = parts[1].parse::<f64>().map_err(|_| clap::Error::raw(ErrorKind::InvalidValue, "Invalid width"))?;
                Ok((center, width))
            } else {
                Err(clap::Error::raw(ErrorKind::InvalidValue, "Size must be in the format width,height"))
            }
        })
    )]
    window: Option<(f64, f64)>,
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
    // Build filepath of form dest/study_instance_uid/series_instance_uid/sop_instance_uid.tiff
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
    let series_instance_uid = file
        .get(tags::SERIES_INSTANCE_UID)
        .map(|element| element.value().to_str().ok().map(|s| s.into_owned()))
        .unwrap_or(None)
        .unwrap_or_else(|| "series".to_string());
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
    let filename = format!("{sop_instance_uid}.tiff");
    let dest = dest.as_ref();
    Ok(dest
        .join(study_instance_uid)
        .join(series_instance_uid)
        .join(filename))
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

    let mut file = open_file(source)
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
    let convert_options = match args.window {
        Some((center, width)) => ConvertOptions::default()
            .with_voi_lut(VoiLutOption::Custom(WindowLevel { width, center })),
        None => ConvertOptions::default(),
    };

    // Convert spacing argument to SpacingConfig
    let spacing_config = args.spacing.map(|(x, y, z)| {
        let mut config = dicom_preprocessing::preprocess::SpacingConfig::new(x, y);
        if let Some(z_val) = z {
            config = config.with_spacing_z(z_val);
        }
        config
    });

    let preprocessor = Preprocessor {
        crop: args.crop,
        size: args.size,
        spacing: spacing_config,
        filter: args.filter,
        padding_direction: args.padding_direction,
        crop_max: args.crop_max,
        volume_handler: match args.volume_handler {
            DisplayVolumeHandler::Interpolate => {
                VolumeHandler::Interpolate(InterpolateVolume::new(args.target_frames))
            }
            DisplayVolumeHandler::Keep => VolumeHandler::Keep(KeepVolume),
            DisplayVolumeHandler::CentralSlice => VolumeHandler::CentralSlice(CentralSlice),
            DisplayVolumeHandler::MaxIntensity => {
                VolumeHandler::MaxIntensity(MaxIntensity::default())
            }
            DisplayVolumeHandler::AverageIntensity => {
                VolumeHandler::AverageIntensity(AverageIntensity::default())
            }
            DisplayVolumeHandler::GaussianWeighted => {
                VolumeHandler::GaussianWeighted(GaussianWeighted::default())
            }
            DisplayVolumeHandler::SoftMip => VolumeHandler::SoftMip(SoftMip::default()),
            DisplayVolumeHandler::LaplacianMip => VolumeHandler::LaplacianMip(
                LaplacianMip::new(
                    args.laplacian_levels,
                    args.laplacian_skip_start,
                    args.laplacian_skip_end,
                )
                .with_sigma_r(args.laplacian_sigma_r)
                .with_sigma_s(args.laplacian_sigma_s)
                .with_beta_fine(args.laplacian_beta_fine)
                .with_beta_mid(args.laplacian_beta_mid)
                .with_skip_bilateral_levels(args.laplacian_skip_bilateral_levels)
                .with_mip_levels_start(args.laplacian_mip_levels_start)
                .with_mip_weight(args.laplacian_mip_weight)
                .with_mip_gaussian_base(args.laplacian_use_mip_gaussian)
                .with_mip_gaussian_levels(args.laplacian_mip_gaussian_levels),
            ),
        },
        use_components: !args.no_components,
        use_padding: !args.no_padding,
        border_frac: args.border_frac,
        target_frames: args.target_frames,
        convert_options,
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
    use crate::FilterType;
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
            source,
            output: output_dir.path().to_path_buf(),
            crop: true,
            size: Some((64, 64)),
            spacing: None,
            filter: FilterType::default(),
            padding_direction: PaddingDirection::default(),
            strict: true,
            compressor: SupportedCompressor::default(),
            crop_max: false,
            no_components: false,
            volume_handler: DisplayVolumeHandler::default(),
            no_padding: false,
            border_frac: None,
            target_frames: 32,
            window: None,
            // LaplacianMip defaults
            laplacian_sigma_r: 0.015,
            laplacian_sigma_s: 3.0,
            laplacian_beta_fine: 2.0,
            laplacian_beta_mid: 1.0,
            laplacian_levels: 7,
            laplacian_skip_start: 5,
            laplacian_skip_end: 5,
            laplacian_skip_bilateral_levels: 0,
            laplacian_mip_levels_start: 0,
            laplacian_mip_weight: 1.0,
            laplacian_use_mip_gaussian: false,
            laplacian_mip_gaussian_levels: 3,
        };
        run(args).unwrap();

        // Get the StudyInstanceUID, SeriesInstanceUID, and SOPInstanceUID from the DICOM
        let dicom_file = open_file(&dicom_file_path).unwrap();
        let study_instance_uid = dicom_file
            .get(tags::STUDY_INSTANCE_UID)
            .unwrap()
            .value()
            .to_str()
            .unwrap()
            .into_owned();
        let series_instance_uid = dicom_file
            .get(tags::SERIES_INSTANCE_UID)
            .map(|element| element.value().to_str().ok().map(|s| s.into_owned()))
            .unwrap_or(None)
            .unwrap_or_else(|| "series".to_string());
        let sop_instance_uid = dicom_file
            .get(tags::SOP_INSTANCE_UID)
            .unwrap()
            .value()
            .to_str()
            .unwrap()
            .into_owned();

        // Build the expected output file path
        let filename = format!("{sop_instance_uid}.tiff");
        let output_file_path = output_dir
            .path()
            .join(study_instance_uid)
            .join(series_instance_uid)
            .join(filename);

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
