use clap::Parser;
use dicom::object::open_file;
use dicom_preprocessing::metadata::{summarize_basic_offset_table, BotSummary};
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug, thiserror::Error)]
enum Error {
    #[error("unable to read DICOM file {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: Box<dicom::object::ReadError>,
    },

    #[error("unable to summarize BOT: {source}")]
    Summarize {
        #[from]
        source: dicom_preprocessing::errors::DicomError,
    },
}

#[derive(Debug, Clone, Copy, clap::ValueEnum, PartialEq, Eq)]
enum OutputFormat {
    Text,
    Json,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum, PartialEq, Eq)]
enum ColorMode {
    Auto,
    Always,
    Never,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Status {
    Ok,
    Warn,
}

impl Status {
    fn as_word(self) -> &'static str {
        match self {
            Status::Ok => "OK",
            Status::Warn => "WARN",
        }
    }
}

#[derive(Parser, Debug)]
#[command(
    author = "Scott Chase Waggener",
    version = env!("CARGO_PKG_VERSION"),
    about = "Inspect DICOM Basic Offset Table (BOT) health"
)]
struct Args {
    #[arg(help = "Path to a DICOM file")]
    dicom: PathBuf,

    #[arg(
        long = "format",
        value_enum,
        default_value_t = OutputFormat::Text,
        help = "Output format"
    )]
    format: OutputFormat,

    #[arg(
        long = "color",
        value_enum,
        default_value_t = ColorMode::Auto,
        help = "Color mode for text output"
    )]
    color: ColorMode,

    #[arg(
        long = "no-color",
        default_value_t = false,
        help = "Disable color output (alias for --color never)"
    )]
    no_color: bool,

    #[arg(
        long = "quiet",
        default_value_t = false,
        conflicts_with = "verbose",
        help = "Print header and summary only"
    )]
    quiet: bool,

    #[arg(
        long = "verbose",
        short = 'v',
        default_value_t = false,
        conflicts_with = "quiet",
        help = "Print additional BOT detail"
    )]
    verbose: bool,
}

fn resolve_color_enabled(args: &Args) -> bool {
    if args.no_color {
        return false;
    }
    if args.format != OutputFormat::Text {
        return false;
    }
    match args.color {
        ColorMode::Always => true,
        ColorMode::Never => false,
        ColorMode::Auto => std::io::stdout().is_terminal(),
    }
}

fn style_status(status: Status, color_enabled: bool) -> String {
    let word = status.as_word();
    if !color_enabled {
        return word.to_string();
    }
    match status {
        Status::Ok => format!("\x1b[1;32m{word}\x1b[0m"),
        Status::Warn => format!("\x1b[1;33m{word}\x1b[0m"),
    }
}

fn pad_rows(rows: &[(&str, String)]) -> Vec<String> {
    let width = rows.iter().map(|(label, _)| label.len()).max().unwrap_or(0);
    rows.iter()
        .map(|(label, value)| format!("  {:width$}: {}", label, value, width = width))
        .collect()
}

fn format_duration(duration: std::time::Duration) -> String {
    let millis = duration.as_millis();
    if millis < 1_000 {
        format!("{millis}ms")
    } else {
        format!("{:.2}s", duration.as_secs_f64())
    }
}

fn bool_label(value: bool) -> &'static str {
    if value {
        "true"
    } else {
        "false"
    }
}

fn status_for_summary(summary: &BotSummary) -> Status {
    if summary.needs_correction {
        Status::Warn
    } else {
        Status::Ok
    }
}

fn summarize_rows(summary: &BotSummary) -> Vec<(&'static str, String)> {
    vec![
        (
            "Encapsulated Pixel Data",
            bool_label(summary.file_is_encapsulated).to_string(),
        ),
        ("Number of Frames", format!("{}", summary.number_of_frames)),
        ("Fragment Count", format!("{}", summary.fragment_count)),
        ("BOT Offset Entries", format!("{}", summary.offset_count)),
        (
            "BOT Required for Frame Decode",
            bool_label(summary.bot_required_for_decode_frame).to_string(),
        ),
        (
            "Decode Path At Risk",
            bool_label(summary.decode_frame_path_at_risk).to_string(),
        ),
        (
            "Needs Correction",
            bool_label(summary.needs_correction).to_string(),
        ),
    ]
}

fn checks_rows(summary: &BotSummary) -> Vec<(&'static str, String)> {
    vec![
        (
            "Offset Count Matches Frames",
            bool_label(summary.offset_count_matches_frames).to_string(),
        ),
        (
            "First Offset Is Zero",
            bool_label(summary.starts_at_zero).to_string(),
        ),
        (
            "Offsets Strictly Increasing",
            bool_label(summary.strictly_increasing).to_string(),
        ),
        (
            "Offsets In Bounds",
            bool_label(summary.offsets_in_bounds).to_string(),
        ),
        (
            "Encoded Stream Bytes",
            format!("{}", summary.encoded_stream_len),
        ),
    ]
}

fn render_text(
    path: &Path,
    summary: &BotSummary,
    status: Status,
    duration: std::time::Duration,
    args: &Args,
) {
    let color_enabled = resolve_color_enabled(args);
    let status_word = style_status(status, color_enabled);
    println!(
        "{}  dicom-bot  {}  ({})",
        status_word,
        path.display(),
        format_duration(duration)
    );
    println!();

    println!("Summary");
    for line in pad_rows(&summarize_rows(summary)) {
        println!("{line}");
    }

    if args.quiet {
        return;
    }

    if summary.file_is_encapsulated {
        println!();
        println!("Checks");
        for line in pad_rows(&checks_rows(summary)) {
            println!("{line}");
        }

        let preview = if summary.offsets_preview.is_empty() {
            "[]".to_string()
        } else {
            let entries = summary
                .offsets_preview
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            format!("[{entries}]")
        };

        println!();
        println!("Offsets");
        let mut rows = vec![("Preview", preview)];
        if args.verbose {
            rows.push((
                "Preview Count",
                format!("{}", summary.offsets_preview.len()),
            ));
        }
        for line in pad_rows(&rows) {
            println!("{line}");
        }
    } else {
        println!();
        println!("Notes");
        for line in pad_rows(&[(
            "BOT",
            "Not applicable (pixel data is not encapsulated)".to_string(),
        )]) {
            println!("{line}");
        }
    }
}

fn json_escape(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

fn render_json(path: &Path, summary: &BotSummary, status: Status) {
    let preview = summary
        .offsets_preview
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(", ");

    println!("{{");
    println!(
        "  \"path\": \"{}\",",
        json_escape(&path.display().to_string())
    );
    println!("  \"status\": \"{}\",", status.as_word());
    println!(
        "  \"encapsulated_pixel_data\": {},",
        bool_label(summary.file_is_encapsulated)
    );
    println!("  \"number_of_frames\": {},", summary.number_of_frames);
    println!("  \"fragment_count\": {},", summary.fragment_count);
    println!("  \"bot_offset_entries\": {},", summary.offset_count);
    println!(
        "  \"bot_required_for_frame_decode\": {},",
        bool_label(summary.bot_required_for_decode_frame)
    );
    println!(
        "  \"decode_frame_path_at_risk\": {},",
        bool_label(summary.decode_frame_path_at_risk)
    );
    println!("  \"checks\": {{");
    println!(
        "    \"offset_count_matches_frames\": {},",
        bool_label(summary.offset_count_matches_frames)
    );
    println!(
        "    \"first_offset_is_zero\": {},",
        bool_label(summary.starts_at_zero)
    );
    println!(
        "    \"offsets_strictly_increasing\": {},",
        bool_label(summary.strictly_increasing)
    );
    println!(
        "    \"offsets_in_bounds\": {}",
        bool_label(summary.offsets_in_bounds)
    );
    println!("  }},");
    println!(
        "  \"encoded_stream_bytes\": {},",
        summary.encoded_stream_len
    );
    println!("  \"offsets_preview\": [{}],", preview);
    println!(
        "  \"needs_correction\": {}",
        bool_label(summary.needs_correction)
    );
    println!("}}");
}

fn run(args: &Args) -> Result<i32, Error> {
    let started = Instant::now();
    let file = open_file(&args.dicom).map_err(|source| Error::Read {
        path: args.dicom.clone(),
        source: Box::new(source),
    })?;
    let summary = summarize_basic_offset_table(&file)?;
    let status = status_for_summary(&summary);

    match args.format {
        OutputFormat::Text => render_text(&args.dicom, &summary, status, started.elapsed(), args),
        OutputFormat::Json => render_json(&args.dicom, &summary, status),
    }

    Ok(match status {
        Status::Ok => 0,
        Status::Warn => 1,
    })
}

fn main() {
    let args = Args::parse();
    let exit_code = match run(&args) {
        Ok(code) => code,
        Err(error) => {
            eprintln!("dicom-bot failed: {error}");
            2
        }
    };
    std::process::exit(exit_code);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_escape_escapes_quotes_and_backslashes() {
        let escaped = json_escape("a\"b\\c");
        assert_eq!(escaped, "a\\\"b\\\\c");
    }

    #[test]
    fn status_reports_warn_when_correction_is_needed() {
        let summary = BotSummary {
            file_is_encapsulated: true,
            number_of_frames: 2,
            fragment_count: 4,
            offset_count: 2,
            offset_count_matches_frames: true,
            starts_at_zero: true,
            strictly_increasing: false,
            offsets_in_bounds: true,
            bot_required_for_decode_frame: true,
            decode_frame_path_at_risk: true,
            encoded_stream_len: 100,
            offsets_preview: vec![0, 0],
            needs_correction: true,
        };
        assert_eq!(status_for_summary(&summary), Status::Warn);
    }
}
