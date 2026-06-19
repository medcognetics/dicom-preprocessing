use std::io::{IsTerminal, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use clap::{Parser, ValueEnum};
use dicom_preprocessing::validation::{
    validate_path, CheckStatus, DecodeValidation, Severity, ValidationReport,
    ValidationRuntimeError, ValidationStatus,
};
use snafu::Snafu;

const TOOL_NAME: &str = "dicom-validate";
const ABSENT: &str = "absent";
const UNAVAILABLE: &str = "unavailable";

#[derive(Debug, Snafu)]
enum Error {
    #[snafu(display("{}", source))]
    Validation { source: ValidationRuntimeError },

    #[snafu(display("failed to write output: {}", source))]
    WriteOutput { source: std::io::Error },

    #[snafu(display("failed to serialize JSON output: {}", source))]
    SerializeJson { source: serde_json::Error },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum OutputFormat {
    Text,
    Json,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ColorMode {
    Auto,
    Always,
    Never,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum DecodeMode {
    None,
    Frame,
}

impl From<DecodeMode> for DecodeValidation {
    fn from(value: DecodeMode) -> Self {
        match value {
            DecodeMode::None => DecodeValidation::None,
            DecodeMode::Frame => DecodeValidation::Frame,
        }
    }
}

#[derive(Parser, Debug)]
#[command(
    author = "Scott Chase Waggener",
    version = env!("CARGO_PKG_VERSION"),
    about = "Validate whether a DICOM file has preprocessing-ready image metadata",
    long_about = None
)]
struct Args {
    #[arg(help = "DICOM file to validate")]
    source: PathBuf,

    #[arg(long = "format", value_enum, default_value_t = OutputFormat::Text)]
    format: OutputFormat,

    #[arg(long = "color", value_enum, default_value_t = ColorMode::Auto)]
    color: ColorMode,

    #[arg(long = "no-color", default_value_t = false)]
    no_color: bool,

    #[arg(
        long = "quiet",
        short = 'q',
        default_value_t = false,
        conflicts_with = "verbose"
    )]
    quiet: bool,

    #[arg(
        long = "verbose",
        short = 'v',
        default_value_t = false,
        conflicts_with = "quiet"
    )]
    verbose: bool,

    #[arg(long = "decode", value_enum, default_value_t = DecodeMode::Frame)]
    decode: DecodeMode,
}

fn main() {
    let args = Args::parse();
    let mut stdout = std::io::stdout().lock();
    let mut stderr = std::io::stderr().lock();
    let code = execute(args, &mut stdout, &mut stderr);
    std::process::exit(code);
}

fn execute(args: Args, stdout: &mut impl Write, stderr: &mut impl Write) -> i32 {
    match run(args, stdout) {
        Ok(code) => code,
        Err(error) => {
            let _ = writeln!(stderr, "{TOOL_NAME} failed: {error}");
            2
        }
    }
}

fn run(args: Args, stdout: &mut impl Write) -> Result<i32, Error> {
    let start = Instant::now();
    let report = validate_path(&args.source, args.decode.into())
        .map_err(|source| Error::Validation { source })?;
    let duration = start.elapsed();

    match args.format {
        OutputFormat::Json => {
            serde_json::to_writer_pretty(&mut *stdout, &report)
                .map_err(|source| Error::SerializeJson { source })?;
            writeln!(stdout).map_err(|source| Error::WriteOutput { source })?;
        }
        OutputFormat::Text => {
            let styles = Styles::new(resolve_color(
                args.format,
                args.color,
                args.no_color,
                std::io::stdout().is_terminal(),
            ));
            render_text_report(stdout, &report, duration, &styles, args.quiet, args.verbose)
                .map_err(|source| Error::WriteOutput { source })?;
        }
    }

    Ok(if report.is_valid() { 0 } else { 1 })
}

fn resolve_color(
    format: OutputFormat,
    color: ColorMode,
    no_color: bool,
    stdout_is_terminal: bool,
) -> bool {
    if no_color || format != OutputFormat::Text {
        return false;
    }

    match color {
        ColorMode::Always => true,
        ColorMode::Never => false,
        ColorMode::Auto => stdout_is_terminal,
    }
}

struct Styles {
    enabled: bool,
}

impl Styles {
    fn new(enabled: bool) -> Self {
        Self { enabled }
    }

    fn status(&self, status: ValidationStatus) -> String {
        match status {
            ValidationStatus::Pass => self.paint("PASS", "1;32"),
            ValidationStatus::Fail => self.paint("FAIL", "1;31"),
        }
    }

    fn section(&self, text: &str) -> String {
        self.paint(text, "1")
    }

    fn check_status(&self, status: CheckStatus) -> String {
        match status {
            CheckStatus::Pass => self.paint("PASS", "1;32"),
            CheckStatus::Fail => self.paint("FAIL", "1;31"),
            CheckStatus::Warn => self.paint("WARN", "1;33"),
            CheckStatus::Info => self.paint("INFO", "36"),
            CheckStatus::Skip => self.paint("SKIP", "2"),
        }
    }

    fn paint(&self, text: &str, code: &str) -> String {
        if self.enabled {
            format!("\x1b[{code}m{text}\x1b[0m")
        } else {
            text.to_string()
        }
    }
}

fn render_text_report(
    writer: &mut impl Write,
    report: &ValidationReport,
    duration: Duration,
    styles: &Styles,
    quiet: bool,
    verbose: bool,
) -> std::io::Result<()> {
    writeln!(
        writer,
        "{}  {TOOL_NAME}  {}  ({})",
        styles.status(report.status),
        report.file.path,
        format_duration(duration)
    )?;
    writeln!(writer)?;

    write_section(
        writer,
        &styles.section("Summary"),
        &[
            ("valid", report.summary.valid.to_string()),
            ("errors", report.summary.error_count.to_string()),
            ("warnings", report.summary.warning_count.to_string()),
            ("info", report.summary.info_count.to_string()),
            ("decode", report.summary.decode_mode.clone()),
        ],
    )?;

    if quiet {
        return Ok(());
    }

    writeln!(writer)?;
    write_section(
        writer,
        &styles.section("Image"),
        &[
            ("rows", optional_display(report.image.rows)),
            ("columns", optional_display(report.image.columns)),
            ("frames", optional_display(report.image.number_of_frames)),
            (
                "frames_source",
                report.image.number_of_frames_source.clone(),
            ),
            (
                "samples_per_pixel",
                optional_display(report.image.samples_per_pixel),
            ),
            (
                "photometric_interpretation",
                optional_string_display(report.image.photometric_interpretation.as_deref()),
            ),
            (
                "planar_configuration",
                optional_display(report.image.planar_configuration),
            ),
            (
                "pixel_representation",
                optional_display(report.image.pixel_representation),
            ),
        ],
    )?;

    writeln!(writer)?;
    write_section(
        writer,
        &styles.section("Pixel Format"),
        &[
            (
                "bits_allocated",
                optional_display(report.pixel_format.bits_allocated),
            ),
            (
                "bits_stored",
                optional_display(report.pixel_format.bits_stored),
            ),
            ("high_bit", optional_display(report.pixel_format.high_bit)),
            (
                "output_color_type",
                optional_string_display(report.pixel_format.output_color_type.as_deref()),
            ),
        ],
    )?;

    writeln!(writer)?;
    write_section(
        writer,
        &styles.section("Preprocessing Diagnostics"),
        &[
            ("transfer_syntax", transfer_syntax_display(report)),
            ("resolution", resolution_display(report)),
            ("window", window_display(report)),
            ("voi_lut", voi_lut_display(report)),
            ("voi_lut_function", voi_lut_function_display(report)),
            ("directory_output", directory_output_display(report)),
        ],
    )?;

    writeln!(writer)?;
    write_checks(writer, styles, report, verbose)?;

    if verbose {
        writeln!(writer)?;
        write_section(
            writer,
            &styles.section("Verbose"),
            &[
                (
                    "study_instance_uid",
                    optional_string_display(
                        report
                            .preprocessing
                            .output_naming
                            .study_instance_uid
                            .as_deref(),
                    ),
                ),
                (
                    "series_instance_uid",
                    optional_string_display(
                        report
                            .preprocessing
                            .output_naming
                            .series_instance_uid
                            .as_deref(),
                    ),
                ),
                (
                    "sop_instance_uid",
                    optional_string_display(
                        report
                            .preprocessing
                            .output_naming
                            .sop_instance_uid
                            .as_deref(),
                    ),
                ),
                ("decode_smoke_test", decode_smoke_display(report)),
            ],
        )?;
    }

    Ok(())
}

fn transfer_syntax_display(report: &ValidationReport) -> String {
    format!(
        "{} ({})",
        report.file.transfer_syntax_uid, report.file.transfer_syntax_name
    )
}

fn resolution_display(report: &ValidationReport) -> String {
    let resolution = &report.preprocessing.resolution;
    if !resolution.available {
        return UNAVAILABLE.to_string();
    }

    format!(
        "x={:.6} px/mm y={:.6} px/mm z={}",
        resolution.pixels_per_mm_x.unwrap_or_default(),
        resolution.pixels_per_mm_y.unwrap_or_default(),
        resolution
            .frames_per_mm
            .map(|value| format!("{value:.6} frames/mm"))
            .unwrap_or_else(|| UNAVAILABLE.to_string())
    )
}

fn window_display(report: &ValidationReport) -> String {
    match (
        report.preprocessing.window_center.as_ref(),
        report.preprocessing.window_width.as_ref(),
    ) {
        (Some(center), Some(width)) => format!("center={center:?} width={width:?}"),
        _ => ABSENT.to_string(),
    }
}

fn voi_lut_display(report: &ValidationReport) -> String {
    if report.preprocessing.voi_lut_sequence_present {
        let item_count = report
            .preprocessing
            .voi_lut_sequence_items
            .map(|value| value.to_string())
            .unwrap_or_else(|| "unknown".to_string());
        format!("sequence present items={item_count}")
    } else {
        "sequence absent".to_string()
    }
}

fn voi_lut_function_display(report: &ValidationReport) -> String {
    report
        .preprocessing
        .voi_lut_function
        .as_ref()
        .map(|value| format!("{value:?}"))
        .unwrap_or_else(|| ABSENT.to_string())
}

fn directory_output_display(report: &ValidationReport) -> String {
    if report.preprocessing.output_naming.directory_output_ready {
        "ready".to_string()
    } else {
        "missing StudyInstanceUID or SOPInstanceUID".to_string()
    }
}

fn decode_smoke_display(report: &ValidationReport) -> String {
    report
        .decode_smoke_test
        .as_ref()
        .map(|smoke| {
            format!(
                "attempted={} status={:?} frame={:?} image={:?} expected_frame_bytes={:?}",
                smoke.attempted,
                smoke.status,
                smoke.frame_index,
                smoke.image_color_type,
                smoke.expected_frame_bytes
            )
        })
        .unwrap_or_else(|| "none".to_string())
}

fn write_section(
    writer: &mut impl Write,
    title: &str,
    values: &[(&str, String)],
) -> std::io::Result<()> {
    writeln!(writer, "{title}")?;
    let width = values
        .iter()
        .map(|(label, _)| label.len())
        .max()
        .unwrap_or(0);
    for (label, value) in values {
        writeln!(writer, "  {label:<width$}:  {value}", width = width)?;
    }
    Ok(())
}

fn write_checks(
    writer: &mut impl Write,
    styles: &Styles,
    report: &ValidationReport,
    verbose: bool,
) -> std::io::Result<()> {
    writeln!(writer, "{}", styles.section("Checks"))?;
    for check in &report.checks {
        if !verbose && check.severity == Severity::Info && check.status != CheckStatus::Skip {
            continue;
        }
        let status = styles.check_status(check.status);
        let context = match (&check.tag_name, &check.tag, &check.value) {
            (Some(name), Some(tag), Some(value)) => format!(" ({name} {tag}: {value})"),
            (Some(name), Some(tag), None) => format!(" ({name} {tag})"),
            (_, _, Some(value)) => format!(" ({value})"),
            _ => String::new(),
        };
        writeln!(writer, "  {status}  {}{}", check.message, context)?;
    }
    Ok(())
}

fn optional_display<T: std::fmt::Display>(value: Option<T>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| UNAVAILABLE.to_string())
}

fn optional_string_display(value: Option<&str>) -> String {
    value.unwrap_or(UNAVAILABLE).to_string()
}

fn format_duration(duration: Duration) -> String {
    if duration.as_secs() >= 1 {
        format!("{:.2}s", duration.as_secs_f64())
    } else {
        format!("{}ms", duration.as_millis())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dicom::dictionary_std::tags;
    use dicom::object::open_file;

    fn fixture_path() -> PathBuf {
        dicom_test_files::path("pydicom/CT_small.dcm").unwrap()
    }

    fn args(format: OutputFormat) -> Args {
        Args {
            source: fixture_path(),
            format,
            color: ColorMode::Never,
            no_color: false,
            quiet: false,
            verbose: false,
            decode: DecodeMode::Frame,
        }
    }

    #[test]
    fn json_output_is_parseable_and_uncolored() {
        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
        let mut args = args(OutputFormat::Json);
        args.color = ColorMode::Always;

        let code = execute(args, &mut stdout, &mut stderr);
        assert_eq!(code, 0);
        assert!(stderr.is_empty());
        assert!(!String::from_utf8_lossy(&stdout).contains("\x1b["));

        let value: serde_json::Value = serde_json::from_slice(&stdout).unwrap();
        assert_eq!(value["status"], "pass");
        assert_eq!(value["summary"]["valid"], true);
        assert_eq!(value["file"]["transfer_syntax_uid"], "1.2.840.10008.1.2.1");
        assert_eq!(
            value["file"]["transfer_syntax_name"],
            "Explicit VR Little Endian"
        );
    }

    #[test]
    fn check_statuses_are_colorized_when_color_is_enabled() {
        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
        let mut args = args(OutputFormat::Text);
        args.color = ColorMode::Always;

        let code = execute(args, &mut stdout, &mut stderr);
        assert_eq!(code, 0);
        assert!(stderr.is_empty());

        let output = String::from_utf8(stdout).unwrap();
        assert!(output.contains("\n  \x1b[1;32mPASS\x1b[0m  "));
        assert!(output.contains("\n  \x1b[1;33mWARN\x1b[0m  "));
        assert!(output.contains("transfer_syntax"));
        assert!(output.contains("1.2.840.10008.1.2.1 (Explicit VR Little Endian)"));
    }

    #[test]
    fn quiet_text_output_omits_detail_sections() {
        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
        let mut args = args(OutputFormat::Text);
        args.quiet = true;

        let code = execute(args, &mut stdout, &mut stderr);
        assert_eq!(code, 0);
        assert!(stderr.is_empty());
        let output = String::from_utf8(stdout).unwrap();
        assert!(output.starts_with("PASS  dicom-validate"));
        assert!(output.contains("Summary"));
        assert!(!output.contains("Pixel Format"));
    }

    #[test]
    fn verbose_text_output_includes_verbose_section() {
        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
        let mut args = args(OutputFormat::Text);
        args.verbose = true;

        let code = execute(args, &mut stdout, &mut stderr);
        assert_eq!(code, 0);
        assert!(stderr.is_empty());
        let output = String::from_utf8(stdout).unwrap();
        assert!(output.contains("Verbose"));
        assert!(output.contains("expected_frame_bytes"));
    }

    #[test]
    fn no_color_disables_ansi_even_when_always_requested() {
        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
        let mut args = args(OutputFormat::Text);
        args.color = ColorMode::Always;
        args.no_color = true;

        let code = execute(args, &mut stdout, &mut stderr);
        assert_eq!(code, 0);
        assert!(stderr.is_empty());
        assert!(!String::from_utf8_lossy(&stdout).contains("\x1b["));
    }

    #[test]
    fn invalid_path_returns_runtime_error_code() {
        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
        let mut args = args(OutputFormat::Text);
        args.source = PathBuf::from("/does/not/exist.dcm");

        let code = execute(args, &mut stdout, &mut stderr);
        assert_eq!(code, 2);
        assert!(stdout.is_empty());
        assert!(String::from_utf8_lossy(&stderr).contains("dicom-validate failed:"));
    }

    #[test]
    fn validation_failure_returns_validation_error_code() {
        let temp_dir = tempfile::tempdir().unwrap();
        let invalid_path = temp_dir.path().join("missing-high-bit.dcm");
        let mut dicom = open_file(fixture_path()).unwrap();
        assert!(dicom.remove_element(tags::HIGH_BIT));
        dicom.write_to_file(&invalid_path).unwrap();

        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
        let mut args = args(OutputFormat::Text);
        args.source = invalid_path;

        let code = execute(args, &mut stdout, &mut stderr);
        assert_eq!(code, 1);
        assert!(stderr.is_empty());
        assert!(String::from_utf8_lossy(&stdout).starts_with("FAIL  dicom-validate"));
    }
}
