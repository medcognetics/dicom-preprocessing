use arrow::array::StringArray;
use arrow::error::ArrowError;
use clap::ArgAction;
use clap::Parser;
use csv::Reader as CsvReader;
use dicom_preprocessing::color::DicomColorType;
use dicom_preprocessing::errors::TiffError;
use dicom_preprocessing::file::default_bar;
use dicom_preprocessing::metadata::PreprocessingMetadata;
use dicom_preprocessing::save::TiffSaver;
use dicom_preprocessing::transform::{FilterType, Resize, Transform};
use image::DynamicImage;
use image::{GenericImage, GenericImageView, ImageReader, Rgba};
use indicatif::ParallelProgressIterator;
use parquet::arrow::arrow_reader::ParquetRecordBatchReader;
use parquet::errors::ParquetError;
use rayon::prelude::*;
use serde_json::json;
use snafu::{Report, ResultExt, Snafu, Whatever};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::path::PathBuf;
use tiff::decoder::Decoder;
use tiff::encoder::colortype::{Gray16, Gray8, RGB8};
use tiff::encoder::compression::Compressor;
use tracing::{error, Level};

const BATCH_SIZE: usize = 128;

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
        let compressor_str = match self {
            SupportedCompressor::Packbits => "packbits",
            SupportedCompressor::Lzw => "lzw",
            SupportedCompressor::Uncompressed => "none",
        };
        write!(f, "{compressor_str}")
    }
}

#[derive(Debug, Clone, Copy, clap::ValueEnum, Default, PartialEq, Eq)]
enum OutputDtype {
    #[default]
    U8,
    U16,
}

impl OutputDtype {
    fn max_label(self) -> u16 {
        match self {
            OutputDtype::U8 => u8::MAX as u16,
            OutputDtype::U16 => u16::MAX,
        }
    }
}

impl fmt::Display for OutputDtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let dtype = match self {
            OutputDtype::U8 => "u8",
            OutputDtype::U16 => "u16",
        };
        write!(f, "{dtype}")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ClassMapEntry {
    rgb: [u8; 3],
    label: u16,
}

type ClassMap = HashMap<[u8; 3], u16>;

#[derive(Debug, Clone, Copy, clap::ValueEnum, Default, PartialEq, Eq)]
enum ReportFormat {
    #[default]
    Text,
    Json,
}

impl fmt::Display for ReportFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let format = match self {
            ReportFormat::Text => "text",
            ReportFormat::Json => "json",
        };
        write!(f, "{format}")
    }
}

#[derive(Debug, Default, Clone)]
struct ProcessResult {
    output_path: Option<PathBuf>,
    class_histogram: BTreeMap<u16, u64>,
    has_class_histogram: bool,
}

#[derive(Debug, Clone)]
struct RunSummary {
    total_entries: usize,
    processed_masks: usize,
    missing_masks: usize,
    outputs_with_class_histogram: usize,
    outputs_without_class_histogram: usize,
    class_histogram: BTreeMap<u16, u64>,
    total_class_pixels: u64,
}

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

    #[snafu(display("IO error on path '{}': {:?}", path.display(), source))]
    IO {
        path: PathBuf,
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

    #[snafu(display("Invalid class map path: {}", path.display()))]
    InvalidClassMapPath { path: PathBuf },

    #[snafu(display("Invalid class map format in {}: {}", path.display(), reason))]
    InvalidClassMapFormat { path: PathBuf, reason: String },

    #[snafu(display(
        "Mapped label {} exceeds max {} for output dtype {} ({})",
        label,
        max,
        dtype,
        context
    ))]
    LabelOutOfRange {
        context: String,
        label: u16,
        max: u16,
        dtype: OutputDtype,
    },

    #[snafu(display("Encountered unmapped RGB value ({},{},{}) in {}", r, g, b, path.display()))]
    UnmappedColor { path: PathBuf, r: u8, g: u8, b: u8 },

    #[snafu(display("Grayscale value {} exceeds max {} for output dtype {} in {}", value, max, dtype, path.display()))]
    GrayscaleValueOutOfRange {
        path: PathBuf,
        value: u16,
        max: u16,
        dtype: OutputDtype,
    },

    #[snafu(display("Invalid --map-unmapped-to-fill usage: {}", reason))]
    InvalidMapUnmappedToFillUsage { reason: String },

    #[snafu(display("Fill value ({},{},{}) is not present in the class map", r, g, b))]
    FillValueNotMapped { r: u8, g: u8, b: u8 },
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
        help = "Completion summary output format",
        long = "format",
        value_parser = clap::value_parser!(ReportFormat),
        default_value_t = ReportFormat::default(),
    )]
    format: ReportFormat,

    #[arg(
        help = "CSV file with class mapping columns r,g,b,label",
        long = "class-map"
    )]
    class_map: Option<PathBuf>,

    #[arg(
        help = "Inline class mapping entry in the format R,G,B=LABEL (repeatable, overrides duplicate --class-map entries)",
        long = "map",
        value_parser = parse_map_entry,
        action = ArgAction::Append
    )]
    map: Vec<ClassMapEntry>,

    #[arg(
        help = "Output dtype for mapped labels",
        long = "output-dtype",
        value_parser = clap::value_parser!(OutputDtype),
        default_value_t = OutputDtype::default(),
    )]
    output_dtype: OutputDtype,

    #[arg(
        help = "When class mapping is enabled, map unmapped RGB values to the mapped label of --fill and emit warnings instead of failing",
        long = "map-unmapped-to-fill",
        requires = "fill",
        default_value_t = false
    )]
    map_unmapped_to_fill: bool,

    #[arg(
        help = "Padding fill value in source mask space. Applied before class mapping; for mapped RGB masks this should be the original RGB color (e.g., '255,0,0')",
        long = "fill",
        short = 'f',
        value_parser = parse_fill_value,
        default_value = None
    )]
    fill: Option<Rgba<u8>>,

    #[arg(
        help = "Compression type",
        long = "compressor",
        short = 'z',
        value_parser = clap::value_parser!(SupportedCompressor),
        default_value_t = SupportedCompressor::default(),
    )]
    compressor: SupportedCompressor,

    #[arg(
        help = "Enable verbose logging",
        long = "verbose",
        short = 'v',
        default_value = "false"
    )]
    verbose: bool,
}

/// Parse fill value from string (either "R,G,B" or single grayscale value)
fn parse_fill_value(s: &str) -> Result<Rgba<u8>, String> {
    let parts: Vec<&str> = s.split(',').collect();
    match parts.len() {
        1 => {
            // Single grayscale value
            let val = parts[0]
                .trim()
                .parse::<u8>()
                .map_err(|e| format!("Invalid grayscale value: {}", e))?;
            Ok(Rgba([val, val, val, 255]))
        }
        3 => {
            // RGB triplet
            let r = parts[0]
                .trim()
                .parse::<u8>()
                .map_err(|e| format!("Invalid R value: {}", e))?;
            let g = parts[1]
                .trim()
                .parse::<u8>()
                .map_err(|e| format!("Invalid G value: {}", e))?;
            let b = parts[2]
                .trim()
                .parse::<u8>()
                .map_err(|e| format!("Invalid B value: {}", e))?;
            Ok(Rgba([r, g, b, 255]))
        }
        _ => Err(format!(
            "Fill value must be either a single grayscale value or R,G,B triplet, got: {}",
            s
        )),
    }
}

fn parse_map_entry(s: &str) -> Result<ClassMapEntry, String> {
    let (rgb, label) = s
        .split_once('=')
        .ok_or_else(|| format!("Mapping must be in format R,G,B=LABEL, got: {s}"))?;
    let rgb_parts: Vec<&str> = rgb.split(',').collect();
    if rgb_parts.len() != 3 {
        return Err(format!(
            "Mapping must include three RGB components, got: {s}"
        ));
    }

    let r = rgb_parts[0]
        .trim()
        .parse::<u8>()
        .map_err(|e| format!("Invalid R value in mapping '{s}': {e}"))?;
    let g = rgb_parts[1]
        .trim()
        .parse::<u8>()
        .map_err(|e| format!("Invalid G value in mapping '{s}': {e}"))?;
    let b = rgb_parts[2]
        .trim()
        .parse::<u8>()
        .map_err(|e| format!("Invalid B value in mapping '{s}': {e}"))?;
    let label = label
        .trim()
        .parse::<u16>()
        .map_err(|e| format!("Invalid label value in mapping '{s}': {e}"))?;

    Ok(ClassMapEntry {
        rgb: [r, g, b],
        label,
    })
}

fn parse_class_map_csv(path: &Path) -> Result<Vec<ClassMapEntry>, Error> {
    let mut reader = CsvReader::from_path(path).context(CsvSnafu)?;
    let headers = reader.headers().context(CsvSnafu)?.clone();
    let col = |name: &str| -> Result<usize, Error> {
        headers
            .iter()
            .position(|h| h.trim() == name)
            .ok_or_else(|| Error::InvalidClassMapFormat {
                path: path.to_path_buf(),
                reason: format!("missing required '{name}' column"),
            })
    };

    let r_idx = col("r")?;
    let g_idx = col("g")?;
    let b_idx = col("b")?;
    let label_idx = col("label")?;

    let mut entries = Vec::new();
    for (row, record) in reader.records().enumerate() {
        let record = record.context(CsvSnafu)?;
        let get = |idx: usize, name: &str| -> Result<&str, Error> {
            record
                .get(idx)
                .map(str::trim)
                .ok_or_else(|| Error::InvalidClassMapFormat {
                    path: path.to_path_buf(),
                    reason: format!("row {} missing '{name}' value", row + 2),
                })
        };

        let r = get(r_idx, "r")?
            .parse::<u8>()
            .map_err(|e| Error::InvalidClassMapFormat {
                path: path.to_path_buf(),
                reason: format!("row {} invalid r value: {}", row + 2, e),
            })?;
        let g = get(g_idx, "g")?
            .parse::<u8>()
            .map_err(|e| Error::InvalidClassMapFormat {
                path: path.to_path_buf(),
                reason: format!("row {} invalid g value: {}", row + 2, e),
            })?;
        let b = get(b_idx, "b")?
            .parse::<u8>()
            .map_err(|e| Error::InvalidClassMapFormat {
                path: path.to_path_buf(),
                reason: format!("row {} invalid b value: {}", row + 2, e),
            })?;
        let label =
            get(label_idx, "label")?
                .parse::<u16>()
                .map_err(|e| Error::InvalidClassMapFormat {
                    path: path.to_path_buf(),
                    reason: format!("row {} invalid label value: {}", row + 2, e),
                })?;

        entries.push(ClassMapEntry {
            rgb: [r, g, b],
            label,
        });
    }

    Ok(entries)
}

fn validate_label_range(label: u16, output_dtype: OutputDtype, source: &str) -> Result<(), Error> {
    let max = output_dtype.max_label();
    if label > max {
        return Err(Error::LabelOutOfRange {
            context: source.to_string(),
            label,
            max,
            dtype: output_dtype,
        });
    }
    Ok(())
}

fn build_class_map(
    class_map_path: Option<&Path>,
    inline_map_entries: &[ClassMapEntry],
    output_dtype: OutputDtype,
) -> Result<Option<ClassMap>, Error> {
    if class_map_path.is_none() && inline_map_entries.is_empty() {
        return Ok(None);
    }

    let mut class_map = ClassMap::new();

    if let Some(path) = class_map_path {
        if !path.is_file() {
            return Err(Error::InvalidClassMapPath {
                path: path.to_path_buf(),
            });
        }
        for entry in parse_class_map_csv(path)? {
            validate_label_range(entry.label, output_dtype, &path.display().to_string())?;
            class_map.insert(entry.rgb, entry.label);
        }
    }

    for entry in inline_map_entries {
        validate_label_range(entry.label, output_dtype, "--map")?;
        class_map.insert(entry.rgb, entry.label);
    }

    if class_map.is_empty() {
        let path = class_map_path
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("--map"));
        return Err(Error::InvalidClassMapFormat {
            path,
            reason: "no class mapping entries were provided".to_string(),
        });
    }

    Ok(Some(class_map))
}

fn class_percent(count: u64, total: u64) -> f64 {
    if total == 0 {
        0.0
    } else {
        (count as f64 / total as f64) * 100.0
    }
}

fn collect_class_histogram(image: &DynamicImage) -> Option<BTreeMap<u16, u64>> {
    if let Some(gray) = image.as_luma8() {
        let mut histogram = BTreeMap::new();
        for &value in gray.as_raw() {
            *histogram.entry(value as u16).or_insert(0) += 1;
        }
        return Some(histogram);
    }

    if let Some(gray) = image.as_luma16() {
        let mut histogram = BTreeMap::new();
        for &value in gray.as_raw() {
            *histogram.entry(value).or_insert(0) += 1;
        }
        return Some(histogram);
    }

    None
}

fn merge_class_histograms(target: &mut BTreeMap<u16, u64>, source: &BTreeMap<u16, u64>) {
    for (&label, &count) in source {
        *target.entry(label).or_insert(0) += count;
    }
}

fn build_run_summary(results: &[ProcessResult], total_entries: usize) -> RunSummary {
    let processed_masks = results.iter().filter(|r| r.output_path.is_some()).count();
    let missing_masks = total_entries.saturating_sub(processed_masks);
    let outputs_with_class_histogram = results
        .iter()
        .filter(|r| r.output_path.is_some() && r.has_class_histogram)
        .count();
    let outputs_without_class_histogram =
        processed_masks.saturating_sub(outputs_with_class_histogram);

    let mut class_histogram = BTreeMap::new();
    for result in results.iter().filter(|r| r.output_path.is_some()) {
        merge_class_histograms(&mut class_histogram, &result.class_histogram);
    }
    let total_class_pixels = class_histogram.values().sum();

    RunSummary {
        total_entries,
        processed_masks,
        missing_masks,
        outputs_with_class_histogram,
        outputs_without_class_histogram,
        class_histogram,
        total_class_pixels,
    }
}

fn format_count(value: u64) -> String {
    let s = value.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, ch) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out.chars().rev().collect()
}

fn render_text_summary(summary: &RunSummary) -> String {
    let mut lines = vec![
        "Mask Preprocess Summary".to_string(),
        "=======================".to_string(),
        String::new(),
        format!(
            "  {:<30} {}",
            "manifest entries:",
            format_count(summary.total_entries as u64)
        ),
        format!(
            "  {:<30} {}",
            "masks processed:",
            format_count(summary.processed_masks as u64)
        ),
        format!(
            "  {:<30} {}",
            "masks missing:",
            format_count(summary.missing_masks as u64)
        ),
        format!(
            "  {:<30} {}",
            "outputs with class histogram:",
            format_count(summary.outputs_with_class_histogram as u64)
        ),
        format!(
            "  {:<30} {}",
            "outputs without class histogram:",
            format_count(summary.outputs_without_class_histogram as u64)
        ),
    ];

    if summary.class_histogram.is_empty() {
        lines.push(String::new());
        lines.push("Observed Classes".to_string());
        lines.push("  no grayscale outputs available for class statistics".to_string());
        return lines.join("\n");
    }

    let class_width = summary
        .class_histogram
        .keys()
        .map(|c| c.to_string().len())
        .max()
        .unwrap_or(1)
        .max("class".len());
    let count_width = summary
        .class_histogram
        .values()
        .map(|v| format_count(*v).len())
        .max()
        .unwrap_or(1)
        .max("pixels".len());

    lines.push(String::new());
    lines.push("Observed Classes".to_string());
    lines.push(format!(
        "  {:>class_width$}  {:>count_width$}  {:>7}",
        "class",
        "pixels",
        "percent",
        class_width = class_width,
        count_width = count_width
    ));
    for (label, count) in &summary.class_histogram {
        let percent = class_percent(*count, summary.total_class_pixels);
        lines.push(format!(
            "  {:>class_width$}  {:>count_width$}  {:>6.2}%",
            label,
            format_count(*count),
            percent,
            class_width = class_width,
            count_width = count_width
        ));
    }

    lines.join("\n")
}

fn render_json_summary(summary: &RunSummary) -> String {
    let class_breakdown = summary
        .class_histogram
        .iter()
        .map(|(label, count)| {
            let percent = class_percent(*count, summary.total_class_pixels);
            json!({
                "class": *label,
                "pixels": *count,
                "percent": percent,
            })
        })
        .collect::<Vec<_>>();

    serde_json::to_string_pretty(&json!({
        "total_entries": summary.total_entries,
        "processed_masks": summary.processed_masks,
        "missing_masks": summary.missing_masks,
        "outputs_with_class_histogram": summary.outputs_with_class_histogram,
        "outputs_without_class_histogram": summary.outputs_without_class_histogram,
        "total_class_pixels": summary.total_class_pixels,
        "class_breakdown": class_breakdown,
    }))
    .expect("summary JSON serialization should always succeed")
}

fn emit_summary(summary: &RunSummary, format: ReportFormat) {
    match format {
        ReportFormat::Text => println!("{}", render_text_summary(summary)),
        ReportFormat::Json => println!("{}", render_json_summary(summary)),
    }
}

fn main() {
    let args = Args::parse();

    let level = if args.verbose {
        Level::DEBUG
    } else {
        Level::WARN
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
    if let Some(class_map) = &args.class_map {
        if !class_map.is_file() {
            return Err(Error::InvalidClassMapPath {
                path: class_map.clone(),
            });
        }
    }
    if !args.output.is_dir() {
        std::fs::create_dir_all(&args.output).with_context(|_| IOSnafu {
            path: args.output.clone(),
        })?;
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
    let file = File::open(path).with_context(|_| IOSnafu {
        path: path.to_path_buf(),
    })?;
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
        .with_context(|_| IOSnafu {
            path: path.to_path_buf(),
        })?
        .with_guessed_format()
        .with_context(|_| IOSnafu {
            path: path.to_path_buf(),
        })?;
    reader.decode().context(ImageReadSnafu)
}

/// Apply preprocessing transforms to a mask image
fn apply_transforms(
    image: &DynamicImage,
    metadata: &PreprocessingMetadata,
    fill_value: Option<Rgba<u8>>,
) -> DynamicImage {
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

    // Apply padding with custom fill value
    metadata
        .padding
        .as_ref()
        .map(|padding| apply_padding_with_fill(&image, padding, fill_value))
        .unwrap_or(image)
}

/// Apply padding with a custom fill value
fn apply_padding_with_fill(
    image: &DynamicImage,
    padding: &dicom_preprocessing::Padding,
    fill_value: Option<Rgba<u8>>,
) -> DynamicImage {
    use image::{Luma, Rgb};

    let (width, height) = image.dimensions();
    let mut padded_image = DynamicImage::new(
        width + padding.left + padding.right,
        height + padding.top + padding.bottom,
        image.color(),
    );

    // Fill with custom color if specified, converting to match image type
    if let Some(fill) = fill_value {
        for y in 0..padded_image.height() {
            for x in 0..padded_image.width() {
                // Convert fill value to match the image color type
                match &mut padded_image {
                    DynamicImage::ImageLuma8(img) => {
                        // For grayscale, use the R channel value
                        img.put_pixel(x, y, Luma([fill[0]]));
                    }
                    DynamicImage::ImageLuma16(img) => {
                        let val = (fill[0] as u16) << 8 | fill[0] as u16;
                        img.put_pixel(x, y, Luma([val]));
                    }
                    DynamicImage::ImageRgb8(img) => {
                        img.put_pixel(x, y, Rgb([fill[0], fill[1], fill[2]]));
                    }
                    DynamicImage::ImageRgb16(img) => {
                        let r = (fill[0] as u16) << 8 | fill[0] as u16;
                        let g = (fill[1] as u16) << 8 | fill[1] as u16;
                        let b = (fill[2] as u16) << 8 | fill[2] as u16;
                        img.put_pixel(x, y, Rgb([r, g, b]));
                    }
                    _ => {
                        // For other types, use the generic put_pixel
                        padded_image.put_pixel(x, y, fill);
                    }
                }
            }
        }
    }

    // Copy original image content
    for y in 0..height {
        for x in 0..width {
            let pixel = image.get_pixel(x, y);
            padded_image.put_pixel(x + padding.left, y + padding.top, pixel);
        }
    }

    padded_image
}

fn to_luma16_from_u8(image: image::GrayImage) -> image::ImageBuffer<image::Luma<u16>, Vec<u16>> {
    let (width, height) = image.dimensions();
    let data: Vec<u16> = image.into_raw().into_iter().map(u16::from).collect();
    image::ImageBuffer::from_raw(width, height, data).expect("shape should be valid")
}

fn to_luma8_checked(
    image: image::ImageBuffer<image::Luma<u16>, Vec<u16>>,
    output_dtype: OutputDtype,
    mask_path: &Path,
) -> Result<image::GrayImage, Error> {
    let (width, height) = image.dimensions();
    let mut data = Vec::with_capacity((width * height) as usize);
    let max = output_dtype.max_label();

    for value in image.into_raw() {
        if value > max {
            return Err(Error::GrayscaleValueOutOfRange {
                path: mask_path.to_path_buf(),
                value,
                max,
                dtype: output_dtype,
            });
        }
        data.push(value as u8);
    }

    Ok(image::ImageBuffer::from_raw(width, height, data).expect("shape should be valid"))
}

fn resolve_rgb_label(
    class_map: &ClassMap,
    rgb: [u8; 3],
    mask_path: &Path,
    unmapped_fallback_label: Option<u16>,
    warned_unmapped_values: &mut HashSet<[u8; 3]>,
) -> Result<u16, Error> {
    if let Some(label) = class_map.get(&rgb).copied() {
        return Ok(label);
    }

    let [r, g, b] = rgb;
    if let Some(label) = unmapped_fallback_label {
        if warned_unmapped_values.insert(rgb) {
            tracing::warn!(
                "Encountered unmapped RGB value ({},{},{}) in {}",
                r,
                g,
                b,
                mask_path.display()
            );
        }
        Ok(label)
    } else {
        Err(Error::UnmappedColor {
            path: mask_path.to_path_buf(),
            r,
            g,
            b,
        })
    }
}

fn map_rgb_to_labels(
    image: DynamicImage,
    class_map: &ClassMap,
    output_dtype: OutputDtype,
    mask_path: &Path,
    unmapped_fallback_label: Option<u16>,
) -> Result<DynamicImage, Error> {
    let rgb = image.into_rgb8();
    let (width, height) = rgb.dimensions();
    let mut warned_unmapped_values = HashSet::new();
    let labels = rgb
        .pixels()
        .map(|pixel| {
            resolve_rgb_label(
                class_map,
                pixel.0,
                mask_path,
                unmapped_fallback_label,
                &mut warned_unmapped_values,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    match output_dtype {
        OutputDtype::U8 => {
            let max = output_dtype.max_label();
            let mut labels_u8 = Vec::with_capacity((width * height) as usize);
            for label in labels {
                if label > max {
                    return Err(Error::LabelOutOfRange {
                        context: mask_path.display().to_string(),
                        label,
                        max,
                        dtype: output_dtype,
                    });
                }
                labels_u8.push(label as u8);
            }
            let image = image::ImageBuffer::from_raw(width, height, labels_u8)
                .expect("shape should be valid");
            Ok(DynamicImage::ImageLuma8(image))
        }
        OutputDtype::U16 => {
            let image =
                image::ImageBuffer::from_raw(width, height, labels).expect("shape should be valid");
            Ok(DynamicImage::ImageLuma16(image))
        }
    }
}

fn convert_grayscale_to_labels(
    image: DynamicImage,
    output_dtype: OutputDtype,
    mask_path: &Path,
) -> Result<DynamicImage, Error> {
    match output_dtype {
        OutputDtype::U8 => match image.color() {
            image::ColorType::L8 | image::ColorType::La8 => {
                Ok(DynamicImage::ImageLuma8(image.into_luma8()))
            }
            image::ColorType::L16 | image::ColorType::La16 => {
                let luma16 = image.into_luma16();
                let luma8 = to_luma8_checked(luma16, output_dtype, mask_path)?;
                Ok(DynamicImage::ImageLuma8(luma8))
            }
            _ => Ok(DynamicImage::ImageLuma8(image.into_luma8())),
        },
        OutputDtype::U16 => match image.color() {
            image::ColorType::L16 | image::ColorType::La16 => {
                Ok(DynamicImage::ImageLuma16(image.into_luma16()))
            }
            image::ColorType::L8 | image::ColorType::La8 => {
                let luma8 = image.into_luma8();
                Ok(DynamicImage::ImageLuma16(to_luma16_from_u8(luma8)))
            }
            _ => {
                let luma8 = image.into_luma8();
                Ok(DynamicImage::ImageLuma16(to_luma16_from_u8(luma8)))
            }
        },
    }
}

fn get_output_image_and_color(
    transformed: DynamicImage,
    class_map: Option<&ClassMap>,
    output_dtype: OutputDtype,
    mask_path: &Path,
    unmapped_fallback_label: Option<u16>,
) -> Result<(DynamicImage, DicomColorType), Error> {
    if let Some(class_map) = class_map {
        let transformed = match transformed.color() {
            image::ColorType::Rgb8
            | image::ColorType::Rgba8
            | image::ColorType::Rgb16
            | image::ColorType::Rgba16 => map_rgb_to_labels(
                transformed,
                class_map,
                output_dtype,
                mask_path,
                unmapped_fallback_label,
            )?,
            _ => convert_grayscale_to_labels(transformed, output_dtype, mask_path)?,
        };
        let color = match output_dtype {
            OutputDtype::U8 => DicomColorType::Gray8(Gray8),
            OutputDtype::U16 => DicomColorType::Gray16(Gray16),
        };
        return Ok((transformed, color));
    }

    // Preserve existing behavior when class mapping is disabled.
    let color = match transformed.color() {
        image::ColorType::L8 | image::ColorType::La8 => DicomColorType::Gray8(Gray8),
        image::ColorType::Rgb8 | image::ColorType::Rgba8 => DicomColorType::RGB8(RGB8),
        image::ColorType::L16 | image::ColorType::La16 => DicomColorType::Gray8(Gray8),
        image::ColorType::Rgb16 | image::ColorType::Rgba16 => DicomColorType::RGB8(RGB8),
        _ => DicomColorType::Gray8(Gray8),
    };

    let transformed = match &color {
        DicomColorType::Gray8(_) => DynamicImage::ImageLuma8(transformed.into_luma8()),
        DicomColorType::RGB8(_) => DynamicImage::ImageRgb8(transformed.into_rgb8()),
        DicomColorType::Gray16(_) => DynamicImage::ImageLuma8(transformed.into_luma8()),
    };
    Ok((transformed, color))
}

/// Process a single mask file
fn process_mask(
    entry: &ManifestEntry,
    source_dir: &Path,
    mask_dir: &Path,
    output_dir: &Path,
    class_map: Option<&ClassMap>,
    output_dtype: OutputDtype,
    unmapped_fallback_label: Option<u16>,
    fill_value: Option<Rgba<u8>>,
    compressor: SupportedCompressor,
) -> Result<ProcessResult, Error> {
    // Find the corresponding mask
    let mask_path = match find_mask(mask_dir, &entry.sop_instance_uid) {
        Some(path) => path,
        None => {
            tracing::debug!(
                "No mask found for SOP Instance UID: {}",
                entry.sop_instance_uid
            );
            return Ok(ProcessResult::default());
        }
    };

    // Load the preprocessed TIFF to get metadata
    let source_path = source_dir.join(&entry.path);
    let file = File::open(&source_path).with_context(|_| IOSnafu {
        path: source_path.clone(),
    })?;
    let reader = BufReader::new(file);
    let mut decoder = Decoder::new(reader)
        .map_err(TiffError::from)
        .context(TiffReadSnafu)?;
    let metadata = PreprocessingMetadata::try_from(&mut decoder).context(TiffReadSnafu)?;

    // Load and transform the mask
    let mask = load_mask(&mask_path)?;
    let transformed = apply_transforms(&mask, &metadata, fill_value);

    // Determine output path (preserve directory structure from manifest path)
    let output_path = output_dir.join(entry.path.with_extension("tiff"));
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent).with_context(|_| IOSnafu {
            path: parent.to_path_buf(),
        })?;
    }

    let (transformed, color) = get_output_image_and_color(
        transformed,
        class_map,
        output_dtype,
        &mask_path,
        unmapped_fallback_label,
    )?;
    let class_histogram = collect_class_histogram(&transformed);
    let has_class_histogram = class_histogram.is_some();

    // Save the transformed mask
    let saver = TiffSaver::new(compressor.into(), color);
    let mut encoder = saver.open_tiff(&output_path).context(TiffWriteSnafu)?;
    saver
        .save(&mut encoder, &transformed, &metadata)
        .context(TiffWriteSnafu)?;

    tracing::debug!(
        "Processed mask for {} -> {}",
        entry.sop_instance_uid,
        output_path.display()
    );

    Ok(ProcessResult {
        output_path: Some(output_path),
        class_histogram: class_histogram.unwrap_or_default(),
        has_class_histogram,
    })
}

fn resolve_unmapped_fallback_label(
    args: &Args,
    class_map: Option<&ClassMap>,
) -> Result<Option<u16>, Error> {
    if !args.map_unmapped_to_fill {
        return Ok(None);
    }

    let class_map = class_map.ok_or_else(|| Error::InvalidMapUnmappedToFillUsage {
        reason: "class mapping is not configured (set --class-map and/or --map)".to_string(),
    })?;
    let fill = args
        .fill
        .ok_or_else(|| Error::InvalidMapUnmappedToFillUsage {
            reason: "--fill is required when --map-unmapped-to-fill is set".to_string(),
        })?;
    let fill_rgb = [fill[0], fill[1], fill[2]];

    class_map
        .get(&fill_rgb)
        .copied()
        .ok_or_else(|| Error::FillValueNotMapped {
            r: fill_rgb[0],
            g: fill_rgb[1],
            b: fill_rgb[2],
        })
        .map(Some)
}

fn run(args: Args) -> Result<usize, Error> {
    validate_paths(&args)?;

    let class_map = build_class_map(args.class_map.as_deref(), &args.map, args.output_dtype)?;
    if let Some(class_map) = &class_map {
        tracing::info!("Loaded {} class mappings", class_map.len());
    }
    let unmapped_fallback_label = resolve_unmapped_fallback_label(&args, class_map.as_ref())?;

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
    let results: Vec<_> = match args.format {
        ReportFormat::Text => {
            let pb = default_bar(entries.len() as u64);
            pb.set_message("Processing masks");
            entries
                .par_iter()
                .progress_with(pb)
                .map(|entry| {
                    process_mask(
                        entry,
                        &source_dir,
                        &args.masks,
                        &args.output,
                        class_map.as_ref(),
                        args.output_dtype,
                        unmapped_fallback_label,
                        args.fill,
                        args.compressor.clone(),
                    )
                })
                .collect::<Result<Vec<_>, _>>()?
        }
        ReportFormat::Json => entries
            .par_iter()
            .map(|entry| {
                process_mask(
                    entry,
                    &source_dir,
                    &args.masks,
                    &args.output,
                    class_map.as_ref(),
                    args.output_dtype,
                    unmapped_fallback_label,
                    args.fill,
                    args.compressor.clone(),
                )
            })
            .collect::<Result<Vec<_>, _>>()?,
    };

    let summary = build_run_summary(&results, entries.len());
    emit_summary(&summary, args.format);

    Ok(summary.processed_masks)
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

        let result = apply_transforms(&img, &metadata, None);
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

        let result = apply_transforms(&img, &metadata, None);
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

        let result = apply_transforms(&img, &metadata, None);
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

        let result = process_mask(
            &entry,
            &source_dir,
            &mask_dir,
            &output_dir,
            None,
            OutputDtype::default(),
            None,
            None,
            SupportedCompressor::default(),
        )
        .unwrap();
        let output_path = result.output_path.expect("expected output path");
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

        let result = process_mask(
            &entry,
            &source_dir,
            &mask_dir,
            &output_dir,
            None,
            OutputDtype::default(),
            None,
            None,
            SupportedCompressor::default(),
        )
        .unwrap();
        let output_path = result.output_path.expect("expected output path");
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

        let result = process_mask(
            &entry,
            &source_dir,
            &mask_dir,
            &output_dir,
            None,
            OutputDtype::default(),
            None,
            None,
            SupportedCompressor::default(),
        )
        .unwrap();
        assert!(result.output_path.is_some());
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

        let result = process_mask(
            &entry,
            &source_dir,
            &mask_dir,
            &output_dir,
            None,
            OutputDtype::default(),
            None,
            None,
            SupportedCompressor::default(),
        )
        .unwrap();
        assert!(result.output_path.is_none());
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
            format: ReportFormat::Text,
            class_map: None,
            map: vec![],
            output_dtype: OutputDtype::default(),
            map_unmapped_to_fill: false,
            fill: None,
            compressor: SupportedCompressor::default(),
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

        let result = apply_transforms(&img, &metadata, None);
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

        let result = apply_transforms(&img, &metadata, None);
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
    fn test_padding_with_custom_fill_value() {
        let img = DynamicImage::new_luma8(50, 50);
        let img = {
            let mut gray = img.into_luma8();
            for y in 0..50 {
                for x in 0..50 {
                    gray.put_pixel(x, y, Luma([128]));
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

        // Test with red fill (255, 0, 0) - should be converted to grayscale
        let fill_value = Some(Rgba([255, 0, 0, 255]));
        let result = apply_transforms(&img, &metadata, fill_value);
        let result = result.into_luma8();

        assert_eq!(result.width(), 70);
        assert_eq!(result.height(), 70);

        // Check padding area has the fill value (red as grayscale = 255)
        assert_eq!(
            result.get_pixel(0, 0)[0],
            255,
            "Top-left padding should be red (255)"
        );
        assert_eq!(
            result.get_pixel(5, 5)[0],
            255,
            "Padding area should be red (255)"
        );
        assert_eq!(
            result.get_pixel(69, 69)[0],
            255,
            "Bottom-right padding should be red (255)"
        );

        // Check original content is preserved
        assert_eq!(
            result.get_pixel(10, 10)[0],
            128,
            "Original content should be preserved"
        );
        assert_eq!(
            result.get_pixel(30, 30)[0],
            128,
            "Original content should be preserved"
        );
    }

    #[test]
    fn test_padding_with_grayscale_fill_value() {
        let img = DynamicImage::new_luma8(50, 50);
        let img = {
            let mut gray = img.into_luma8();
            for y in 0..50 {
                for x in 0..50 {
                    gray.put_pixel(x, y, Luma([200]));
                }
            }
            DynamicImage::ImageLuma8(gray)
        };

        let metadata = PreprocessingMetadata {
            crop: None,
            resize: None,
            padding: Some(Padding {
                left: 5,
                top: 5,
                right: 5,
                bottom: 5,
            }),
            resolution: None,
            num_frames: FrameCount::from(1_u16),
        };

        // Test with grayscale fill value (100, 100, 100)
        let fill_value = Some(Rgba([100, 100, 100, 255]));
        let result = apply_transforms(&img, &metadata, fill_value);
        let result = result.into_luma8();

        // Check padding area has the fill value
        assert_eq!(
            result.get_pixel(0, 0)[0],
            100,
            "Padding should have fill value"
        );

        // Check original content is preserved
        assert_eq!(
            result.get_pixel(5, 5)[0],
            200,
            "Original content should be preserved"
        );
    }

    #[test]
    fn test_parse_fill_value() {
        // Test single grayscale value
        let gray = parse_fill_value("128").unwrap();
        assert_eq!(gray, Rgba([128, 128, 128, 255]));

        // Test RGB triplet
        let red = parse_fill_value("255,0,0").unwrap();
        assert_eq!(red, Rgba([255, 0, 0, 255]));

        let white = parse_fill_value("255, 255, 255").unwrap();
        assert_eq!(white, Rgba([255, 255, 255, 255]));

        // Test errors
        assert!(parse_fill_value("invalid").is_err());
        assert!(parse_fill_value("256").is_err());
        assert!(parse_fill_value("1,2").is_err());
        assert!(parse_fill_value("1,2,3,4").is_err());
    }

    #[test]
    fn test_parse_map_entry() {
        let entry = parse_map_entry("255,0,7=42").unwrap();
        assert_eq!(entry.rgb, [255, 0, 7]);
        assert_eq!(entry.label, 42);

        assert!(parse_map_entry("255,0=1").is_err());
        assert!(parse_map_entry("x,0,0=1").is_err());
        assert!(parse_map_entry("1,2,3").is_err());
    }

    #[test]
    fn test_build_class_map_cli_overrides_csv() {
        let tmp_dir = TempDir::new().unwrap();
        let class_map_path = tmp_dir.path().join("class_map.csv");
        fs::write(&class_map_path, "r,g,b,label\n255,0,0,1\n0,255,0,2\n").unwrap();

        let inline = vec![ClassMapEntry {
            rgb: [255, 0, 0],
            label: 9,
        }];
        let class_map = build_class_map(Some(&class_map_path), &inline, OutputDtype::U8)
            .unwrap()
            .unwrap();

        assert_eq!(class_map.get(&[255, 0, 0]), Some(&9));
        assert_eq!(class_map.get(&[0, 255, 0]), Some(&2));
    }

    #[test]
    fn test_build_class_map_rejects_out_of_range_label_for_u8() {
        let inline = vec![ClassMapEntry {
            rgb: [1, 2, 3],
            label: 300,
        }];
        let err = build_class_map(None, &inline, OutputDtype::U8).unwrap_err();
        assert!(matches!(err, Error::LabelOutOfRange { .. }));
    }

    #[test]
    fn test_parse_class_map_csv_requires_columns() {
        let tmp_dir = TempDir::new().unwrap();
        let class_map_path = tmp_dir.path().join("class_map.csv");
        fs::write(&class_map_path, "r,g,b\n255,0,0\n").unwrap();

        let err = parse_class_map_csv(&class_map_path).unwrap_err();
        assert!(matches!(err, Error::InvalidClassMapFormat { .. }));
    }

    #[test]
    fn test_collect_class_histogram_luma8() {
        let mut img = GrayImage::new(2, 2);
        img.put_pixel(0, 0, Luma([0]));
        img.put_pixel(1, 0, Luma([1]));
        img.put_pixel(0, 1, Luma([1]));
        img.put_pixel(1, 1, Luma([2]));
        let image = DynamicImage::ImageLuma8(img);

        let histogram = collect_class_histogram(&image).unwrap();
        assert_eq!(histogram.get(&0), Some(&1));
        assert_eq!(histogram.get(&1), Some(&2));
        assert_eq!(histogram.get(&2), Some(&1));
    }

    #[test]
    fn test_render_summaries_with_class_breakdown() {
        let mut histogram = BTreeMap::new();
        histogram.insert(0, 90);
        histogram.insert(1, 10);
        let summary = RunSummary {
            total_entries: 3,
            processed_masks: 2,
            missing_masks: 1,
            outputs_with_class_histogram: 2,
            outputs_without_class_histogram: 0,
            class_histogram: histogram,
            total_class_pixels: 100,
        };

        let text = render_text_summary(&summary);
        assert!(text.contains("Observed Classes"));
        assert!(text.contains("0"));
        assert!(text.contains("90.00%"));

        let json = render_json_summary(&summary);
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["processed_masks"], 2);
        assert_eq!(parsed["class_breakdown"][0]["class"], 0);
        assert_eq!(parsed["class_breakdown"][0]["pixels"], 90);
    }

    #[test]
    fn test_process_mask_rgb_png_with_class_map_u8() {
        let tmp_dir = TempDir::new().unwrap();

        let source_dir = tmp_dir.path().join("source");
        let study_series = source_dir.join("study1").join("series1");
        fs::create_dir_all(&study_series).unwrap();
        create_preprocessed_tiff(&study_series.join("sop1.tiff"), 100, 100, None, None, None);

        let mask_dir = tmp_dir.path().join("masks");
        fs::create_dir(&mask_dir).unwrap();
        create_rgb_mask_png(&mask_dir.join("sop1.png"), 100, 100, |x, _| {
            if x < 50 {
                [255, 0, 0]
            } else {
                [0, 255, 0]
            }
        });

        let output_dir = tmp_dir.path().join("output");
        fs::create_dir(&output_dir).unwrap();

        let entry = ManifestEntry {
            sop_instance_uid: "sop1".to_string(),
            path: PathBuf::from("study1/series1/sop1.tiff"),
        };
        let class_map = ClassMap::from([([255, 0, 0], 1), ([0, 255, 0], 2)]);

        let output_path = process_mask(
            &entry,
            &source_dir,
            &mask_dir,
            &output_dir,
            Some(&class_map),
            OutputDtype::U8,
            None,
            None,
            SupportedCompressor::default(),
        )
        .unwrap()
        .output_path
        .expect("expected output path");

        let output = ImageReader::open(&output_path).unwrap().decode().unwrap();
        assert!(matches!(output.color(), image::ColorType::L8));
        let output = output.into_luma8();
        assert_eq!(output.get_pixel(10, 10)[0], 1);
        assert_eq!(output.get_pixel(90, 10)[0], 2);
    }

    #[test]
    fn test_process_mask_rgb_png_with_class_map_u16() {
        let tmp_dir = TempDir::new().unwrap();

        let source_dir = tmp_dir.path().join("source");
        let study_series = source_dir.join("study1").join("series1");
        fs::create_dir_all(&study_series).unwrap();
        create_preprocessed_tiff(&study_series.join("sop1.tiff"), 100, 100, None, None, None);

        let mask_dir = tmp_dir.path().join("masks");
        fs::create_dir(&mask_dir).unwrap();
        create_rgb_mask_png(&mask_dir.join("sop1.png"), 100, 100, |_, _| [255, 0, 0]);

        let output_dir = tmp_dir.path().join("output");
        fs::create_dir(&output_dir).unwrap();

        let entry = ManifestEntry {
            sop_instance_uid: "sop1".to_string(),
            path: PathBuf::from("study1/series1/sop1.tiff"),
        };
        let class_map = ClassMap::from([([255, 0, 0], 512)]);

        let output_path = process_mask(
            &entry,
            &source_dir,
            &mask_dir,
            &output_dir,
            Some(&class_map),
            OutputDtype::U16,
            None,
            None,
            SupportedCompressor::default(),
        )
        .unwrap()
        .output_path
        .expect("expected output path");

        let output = ImageReader::open(&output_path).unwrap().decode().unwrap();
        assert!(matches!(output.color(), image::ColorType::L16));
        let output = output.into_luma16();
        assert_eq!(output.get_pixel(10, 10)[0], 512);
    }

    #[test]
    fn test_process_mask_fails_on_unmapped_color() {
        let tmp_dir = TempDir::new().unwrap();

        let source_dir = tmp_dir.path().join("source");
        let study_series = source_dir.join("study1").join("series1");
        fs::create_dir_all(&study_series).unwrap();
        create_preprocessed_tiff(&study_series.join("sop1.tiff"), 100, 100, None, None, None);

        let mask_dir = tmp_dir.path().join("masks");
        fs::create_dir(&mask_dir).unwrap();
        create_rgb_mask_png(&mask_dir.join("sop1.png"), 100, 100, |x, _| {
            if x < 50 {
                [255, 0, 0]
            } else {
                [0, 255, 0]
            }
        });

        let output_dir = tmp_dir.path().join("output");
        fs::create_dir(&output_dir).unwrap();

        let entry = ManifestEntry {
            sop_instance_uid: "sop1".to_string(),
            path: PathBuf::from("study1/series1/sop1.tiff"),
        };
        let class_map = ClassMap::from([([255, 0, 0], 1)]);

        let err = process_mask(
            &entry,
            &source_dir,
            &mask_dir,
            &output_dir,
            Some(&class_map),
            OutputDtype::U8,
            None,
            None,
            SupportedCompressor::default(),
        )
        .unwrap_err();

        assert!(matches!(err, Error::UnmappedColor { .. }));
    }

    #[test]
    fn test_process_mask_maps_unmapped_to_fill_label_when_enabled() {
        let tmp_dir = TempDir::new().unwrap();

        let source_dir = tmp_dir.path().join("source");
        let study_series = source_dir.join("study1").join("series1");
        fs::create_dir_all(&study_series).unwrap();
        create_preprocessed_tiff(&study_series.join("sop1.tiff"), 100, 100, None, None, None);

        let mask_dir = tmp_dir.path().join("masks");
        fs::create_dir(&mask_dir).unwrap();
        create_rgb_mask_png(&mask_dir.join("sop1.png"), 100, 100, |x, _| {
            if x < 50 {
                [255, 0, 0]
            } else {
                [0, 255, 0]
            }
        });

        let output_dir = tmp_dir.path().join("output");
        fs::create_dir(&output_dir).unwrap();

        let entry = ManifestEntry {
            sop_instance_uid: "sop1".to_string(),
            path: PathBuf::from("study1/series1/sop1.tiff"),
        };
        let class_map = ClassMap::from([([255, 0, 0], 7)]);

        let output_path = process_mask(
            &entry,
            &source_dir,
            &mask_dir,
            &output_dir,
            Some(&class_map),
            OutputDtype::U8,
            Some(7),
            Some(Rgba([255, 0, 0, 255])),
            SupportedCompressor::default(),
        )
        .unwrap()
        .output_path
        .expect("expected output path");

        let output = ImageReader::open(&output_path).unwrap().decode().unwrap();
        let output = output.into_luma8();
        assert_eq!(output.get_pixel(10, 10)[0], 7);
        assert_eq!(output.get_pixel(90, 10)[0], 7);
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
            format: ReportFormat::Text,
            class_map: None,
            map: vec![],
            output_dtype: OutputDtype::default(),
            map_unmapped_to_fill: false,
            fill: None,
            compressor: SupportedCompressor::default(),
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
