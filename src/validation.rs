use std::path::{Path, PathBuf};

use dicom::core::header::HasLength;
use dicom::core::{DataElement, DicomValue, Tag};
use dicom::dictionary_std::tags;
use dicom::object::{open_file, FileDicomObject, InMemDicomObject, ReadError};
use dicom::pixeldata::PixelDecoder;
use dicom::transfer_syntax::{TransferSyntaxIndex, TransferSyntaxRegistry};
use image::ColorType;
use serde::Serialize;
use snafu::{ResultExt, Snafu};

use crate::color::DicomColorType;
use crate::metadata::Resolution;
use crate::preprocess::Preprocessor;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeValidation {
    None,
    Frame,
}

#[derive(Debug, Snafu)]
pub enum ValidationRuntimeError {
    #[snafu(display("Invalid source path: {}", path.display()))]
    InvalidSourcePath { path: PathBuf },

    #[snafu(display("error reading DICOM file {}: {:?}", path.display(), source))]
    ReadDicom {
        path: PathBuf,
        #[snafu(source(from(ReadError, Box::new)))]
        source: Box<ReadError>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ValidationStatus {
    Pass,
    Fail,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum CheckStatus {
    Pass,
    Fail,
    Warn,
    Info,
    Skip,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Critical,
    Warning,
    Info,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MessageKind {
    Error,
    Warning,
    Info,
}

struct CheckRecord<'a> {
    name: &'a str,
    status: CheckStatus,
    severity: Severity,
    message: String,
    tag: Option<Tag>,
    tag_name: Option<&'a str>,
    value: Option<String>,
}

impl<'a> CheckRecord<'a> {
    fn critical(
        name: &'a str,
        status: CheckStatus,
        message: String,
        value: Option<String>,
    ) -> Self {
        Self {
            name,
            status,
            severity: Severity::Critical,
            message,
            tag: None,
            tag_name: None,
            value,
        }
    }

    fn with_tag(mut self, tag: Tag, tag_name: &'a str) -> Self {
        self.tag = Some(tag);
        self.tag_name = Some(tag_name);
        self
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ValidationMessage {
    pub code: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tag: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tag_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ValidationCheck {
    pub name: String,
    pub status: CheckStatus,
    pub severity: Severity,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tag: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tag_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ValidationSummary {
    pub valid: bool,
    pub error_count: usize,
    pub warning_count: usize,
    pub info_count: usize,
    pub decode_mode: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct FileMetaReport {
    pub path: String,
    pub transfer_syntax_uid: String,
    pub transfer_syntax_name: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ImageReport {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rows: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub columns: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub number_of_frames: Option<u32>,
    pub number_of_frames_source: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub samples_per_pixel: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub photometric_interpretation: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub planar_configuration: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pixel_representation: Option<u16>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PixelFormatReport {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bits_allocated: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bits_stored: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub high_bit: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_color_type: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResolutionReport {
    pub available: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pixels_per_mm_x: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pixels_per_mm_y: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frames_per_mm: Option<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OutputNamingReport {
    pub directory_output_ready: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub study_instance_uid: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub series_instance_uid: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sop_instance_uid: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PreprocessingDiagnostics {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rescale_slope: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rescale_intercept: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub window_center: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub window_width: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voi_lut_function: Option<Vec<String>>,
    pub voi_lut_sequence_present: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voi_lut_sequence_items: Option<usize>,
    pub resolution: ResolutionReport,
    pub output_naming: OutputNamingReport,
}

#[derive(Debug, Clone, Serialize)]
pub struct DecodeSmokeTestReport {
    pub attempted: bool,
    pub status: CheckStatus,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frame_index: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_color_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_frame_bytes: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ValidationReport {
    pub status: ValidationStatus,
    pub summary: ValidationSummary,
    pub file: FileMetaReport,
    pub image: ImageReport,
    pub pixel_format: PixelFormatReport,
    pub preprocessing: PreprocessingDiagnostics,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode_smoke_test: Option<DecodeSmokeTestReport>,
    pub errors: Vec<ValidationMessage>,
    pub warnings: Vec<ValidationMessage>,
    pub info: Vec<ValidationMessage>,
    pub checks: Vec<ValidationCheck>,
}

impl ValidationReport {
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }

    fn new(
        path: &Path,
        transfer_syntax_uid: String,
        transfer_syntax_name: String,
        decode_mode: DecodeValidation,
    ) -> Self {
        Self {
            status: ValidationStatus::Pass,
            summary: ValidationSummary {
                valid: true,
                error_count: 0,
                warning_count: 0,
                info_count: 0,
                decode_mode: decode_mode.as_str().to_string(),
            },
            file: FileMetaReport {
                path: path.display().to_string(),
                transfer_syntax_uid,
                transfer_syntax_name,
            },
            image: ImageReport {
                rows: None,
                columns: None,
                number_of_frames: None,
                number_of_frames_source: "unknown".to_string(),
                samples_per_pixel: None,
                photometric_interpretation: None,
                planar_configuration: None,
                pixel_representation: None,
            },
            pixel_format: PixelFormatReport {
                bits_allocated: None,
                bits_stored: None,
                high_bit: None,
                output_color_type: None,
            },
            preprocessing: PreprocessingDiagnostics {
                rescale_slope: None,
                rescale_intercept: None,
                window_center: None,
                window_width: None,
                voi_lut_function: None,
                voi_lut_sequence_present: false,
                voi_lut_sequence_items: None,
                resolution: ResolutionReport {
                    available: false,
                    pixels_per_mm_x: None,
                    pixels_per_mm_y: None,
                    frames_per_mm: None,
                },
                output_naming: OutputNamingReport {
                    directory_output_ready: false,
                    study_instance_uid: None,
                    series_instance_uid: None,
                    sop_instance_uid: None,
                },
            },
            decode_smoke_test: None,
            errors: Vec::new(),
            warnings: Vec::new(),
            info: Vec::new(),
            checks: Vec::new(),
        }
    }

    fn finalize(&mut self) {
        self.status = if self.errors.is_empty() {
            ValidationStatus::Pass
        } else {
            ValidationStatus::Fail
        };
        self.summary.valid = self.errors.is_empty();
        self.summary.error_count = self.errors.len();
        self.summary.warning_count = self.warnings.len();
        self.summary.info_count = self.info.len();
    }

    fn record_error(
        &mut self,
        code: &str,
        message: String,
        tag: Tag,
        tag_name: &str,
        value: Option<String>,
    ) {
        self.record_message_and_check(
            MessageKind::Error,
            code,
            CheckRecord::critical(tag_name, CheckStatus::Fail, message, value)
                .with_tag(tag, tag_name),
        );
    }

    fn record_general_error(
        &mut self,
        code: &str,
        name: &str,
        message: String,
        value: Option<String>,
    ) {
        self.record_message_and_check(
            MessageKind::Error,
            code,
            CheckRecord::critical(name, CheckStatus::Fail, message, value),
        );
    }

    fn record_warning(
        &mut self,
        code: &str,
        name: &str,
        message: String,
        tag: Option<Tag>,
        value: Option<String>,
    ) {
        self.record_message_and_check(
            MessageKind::Warning,
            code,
            CheckRecord {
                name,
                status: CheckStatus::Warn,
                severity: Severity::Warning,
                message,
                tag,
                tag_name: Some(name),
                value,
            },
        );
    }

    fn record_info(
        &mut self,
        code: &str,
        name: &str,
        message: String,
        tag: Option<Tag>,
        value: Option<String>,
    ) {
        self.record_message_and_check(
            MessageKind::Info,
            code,
            CheckRecord {
                name,
                status: CheckStatus::Info,
                severity: Severity::Info,
                message,
                tag,
                tag_name: Some(name),
                value,
            },
        );
    }

    fn record_pass(&mut self, name: &str, message: String, tag: Tag, value: Option<String>) {
        self.record_check(
            CheckRecord::critical(name, CheckStatus::Pass, message, value).with_tag(tag, name),
        );
    }

    fn record_general_pass(&mut self, name: &str, message: String, value: Option<String>) {
        self.record_check(CheckRecord::critical(
            name,
            CheckStatus::Pass,
            message,
            value,
        ));
    }

    fn record_skip(&mut self, name: &str, message: String) {
        self.record_check(CheckRecord {
            name,
            status: CheckStatus::Skip,
            severity: Severity::Info,
            message,
            tag: None,
            tag_name: None,
            value: None,
        });
    }

    fn record_message_and_check(&mut self, kind: MessageKind, code: &str, check: CheckRecord<'_>) {
        let validation_message = ValidationMessage {
            code: code.to_string(),
            message: check.message.clone(),
            tag: check.tag.map(format_tag),
            tag_name: check.tag_name.map(ToOwned::to_owned),
            value: check.value.clone(),
        };
        match kind {
            MessageKind::Error => self.errors.push(validation_message),
            MessageKind::Warning => self.warnings.push(validation_message),
            MessageKind::Info => self.info.push(validation_message),
        }
        self.record_check(check);
    }

    fn record_check(&mut self, check: CheckRecord<'_>) {
        self.checks.push(ValidationCheck {
            name: check.name.to_string(),
            status: check.status,
            severity: check.severity,
            message: check.message,
            tag: check.tag.map(format_tag),
            tag_name: check.tag_name.map(ToOwned::to_owned),
            value: check.value,
        });
    }
}

impl DecodeValidation {
    fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Frame => "frame",
        }
    }
}

pub fn validate_path(
    path: &Path,
    decode_mode: DecodeValidation,
) -> Result<ValidationReport, ValidationRuntimeError> {
    if !path.is_file() {
        return Err(ValidationRuntimeError::InvalidSourcePath {
            path: path.to_path_buf(),
        });
    }

    let mut file = open_file(path).context(ReadDicomSnafu { path })?;
    Ok(validate_object(path, &mut file, decode_mode))
}

pub fn validate_object(
    path: &Path,
    file: &mut FileDicomObject<InMemDicomObject>,
    decode_mode: DecodeValidation,
) -> ValidationReport {
    let transfer_syntax_uid = file
        .meta()
        .transfer_syntax
        .trim_matches(|c: char| c.is_whitespace() || c == '\0')
        .to_string();
    let transfer_syntax_name = TransferSyntaxRegistry
        .get(&transfer_syntax_uid)
        .map(|syntax| syntax.name().to_string())
        .unwrap_or_else(|| "unknown transfer syntax".to_string());
    let mut report =
        ValidationReport::new(path, transfer_syntax_uid, transfer_syntax_name, decode_mode);

    let rows = required_u16(&mut report, file, tags::ROWS, "Rows", "missing_rows");
    let columns = required_u16(
        &mut report,
        file,
        tags::COLUMNS,
        "Columns",
        "missing_columns",
    );
    let samples_per_pixel = required_u16(
        &mut report,
        file,
        tags::SAMPLES_PER_PIXEL,
        "SamplesPerPixel",
        "missing_samples_per_pixel",
    );
    let photometric_interpretation = required_string(
        &mut report,
        file,
        tags::PHOTOMETRIC_INTERPRETATION,
        "PhotometricInterpretation",
        "missing_photometric_interpretation",
    );
    let bits_allocated = required_u16(
        &mut report,
        file,
        tags::BITS_ALLOCATED,
        "BitsAllocated",
        "missing_bits_allocated",
    );
    let bits_stored = required_u16(
        &mut report,
        file,
        tags::BITS_STORED,
        "BitsStored",
        "missing_bits_stored",
    );
    let high_bit = required_u16(
        &mut report,
        file,
        tags::HIGH_BIT,
        "HighBit",
        "missing_high_bit",
    );
    let pixel_representation = required_u16(
        &mut report,
        file,
        tags::PIXEL_REPRESENTATION,
        "PixelRepresentation",
        "missing_pixel_representation",
    );

    report.image.rows = rows;
    report.image.columns = columns;
    report.image.samples_per_pixel = samples_per_pixel;
    report.image.photometric_interpretation = photometric_interpretation.clone();
    report.image.pixel_representation = pixel_representation;
    report.pixel_format.bits_allocated = bits_allocated;
    report.pixel_format.bits_stored = bits_stored;
    report.pixel_format.high_bit = high_bit;

    validate_pixel_data(&mut report, file);
    validate_number_of_frames(&mut report, file);
    validate_planar_configuration(&mut report, file);
    validate_pixel_representation(&mut report, pixel_representation);
    validate_bit_relationships(&mut report, bits_allocated, bits_stored, high_bit);
    validate_output_color_type(&mut report, file);
    collect_optional_diagnostics(&mut report, file);

    if decode_mode == DecodeValidation::Frame {
        validate_frame_decode(
            &mut report,
            file,
            bits_allocated,
            samples_per_pixel,
            rows,
            columns,
        );
    } else {
        report.decode_smoke_test = Some(DecodeSmokeTestReport {
            attempted: false,
            status: CheckStatus::Skip,
            message: "frame decode smoke test disabled by --decode none".to_string(),
            frame_index: None,
            image_color_type: None,
            expected_frame_bytes: None,
        });
        report.record_skip(
            "Frame decode smoke test",
            "frame decode smoke test disabled by --decode none".to_string(),
        );
    }

    report.finalize();
    report
}

fn required_u16(
    report: &mut ValidationReport,
    file: &FileDicomObject<InMemDicomObject>,
    tag: Tag,
    name: &'static str,
    missing_code: &'static str,
) -> Option<u16> {
    match file.get(tag) {
        Some(element) if !element.is_empty() => match element.value().uint16() {
            Ok(value) => {
                report.record_pass(
                    name,
                    format!("{name} is present"),
                    tag,
                    Some(value.to_string()),
                );
                Some(value)
            }
            Err(source) => {
                report.record_error(
                    "invalid_u16_value",
                    format!("{name} is present but cannot be read as u16: {source:?}"),
                    tag,
                    name,
                    element_value(element),
                );
                None
            }
        },
        Some(_) => {
            report.record_error(
                "empty_required_tag",
                format!("{name} is present but empty"),
                tag,
                name,
                None,
            );
            None
        }
        None => {
            report.record_error(
                missing_code,
                format!("{name} is required for preprocessing"),
                tag,
                name,
                None,
            );
            None
        }
    }
}

fn required_string(
    report: &mut ValidationReport,
    file: &FileDicomObject<InMemDicomObject>,
    tag: Tag,
    name: &'static str,
    missing_code: &'static str,
) -> Option<String> {
    match file.get(tag) {
        Some(element) if !element.is_empty() => match element.value().string() {
            Ok(value) => {
                let value = value.trim_matches(|c: char| c.is_whitespace() || c == '\0');
                if value.is_empty() {
                    report.record_error(
                        "empty_required_tag",
                        format!("{name} is present but empty"),
                        tag,
                        name,
                        None,
                    );
                    None
                } else {
                    report.record_pass(
                        name,
                        format!("{name} is present"),
                        tag,
                        Some(value.to_string()),
                    );
                    Some(value.to_string())
                }
            }
            Err(source) => {
                report.record_error(
                    "invalid_string_value",
                    format!("{name} is present but cannot be read as a string: {source:?}"),
                    tag,
                    name,
                    element_value(element),
                );
                None
            }
        },
        Some(_) => {
            report.record_error(
                "empty_required_tag",
                format!("{name} is present but empty"),
                tag,
                name,
                None,
            );
            None
        }
        None => {
            report.record_error(
                missing_code,
                format!("{name} is required for preprocessing"),
                tag,
                name,
                None,
            );
            None
        }
    }
}

fn validate_pixel_data(report: &mut ValidationReport, file: &FileDicomObject<InMemDicomObject>) {
    match file.get(tags::PIXEL_DATA) {
        Some(element) => match element.value() {
            DicomValue::Primitive(value) => {
                let byte_len = value.to_bytes().len();
                if byte_len == 0 {
                    report.record_error(
                        "empty_pixel_data",
                        "PixelData is present but empty".to_string(),
                        tags::PIXEL_DATA,
                        "PixelData",
                        Some("0 bytes".to_string()),
                    );
                } else {
                    report.record_pass(
                        "PixelData",
                        "PixelData is present".to_string(),
                        tags::PIXEL_DATA,
                        Some(format!("{byte_len} bytes")),
                    );
                }
            }
            DicomValue::PixelSequence(sequence) => {
                report.record_pass(
                    "PixelData",
                    "encapsulated PixelData sequence is present".to_string(),
                    tags::PIXEL_DATA,
                    Some(format!("{} fragments", sequence.fragments().len())),
                );
            }
            DicomValue::Sequence(_) => {
                report.record_error(
                    "invalid_pixel_data_value",
                    "PixelData is not a primitive value or pixel sequence".to_string(),
                    tags::PIXEL_DATA,
                    "PixelData",
                    None,
                );
            }
        },
        None => report.record_error(
            "missing_pixel_data",
            "PixelData is required for preprocessing".to_string(),
            tags::PIXEL_DATA,
            "PixelData",
            None,
        ),
    }
}

fn validate_number_of_frames(
    report: &mut ValidationReport,
    file: &FileDicomObject<InMemDicomObject>,
) {
    match file.get(tags::NUMBER_OF_FRAMES) {
        Some(element) if !element.is_empty() => {
            match element.to_int::<i32>() {
                Ok(value) if value > 0 => {
                    report.image.number_of_frames = Some(value as u32);
                    report.image.number_of_frames_source = "explicit".to_string();
                    report.record_pass(
                        "NumberOfFrames",
                        "NumberOfFrames is present and positive".to_string(),
                        tags::NUMBER_OF_FRAMES,
                        Some(value.to_string()),
                    );
                }
                Ok(value) => {
                    report.image.number_of_frames_source = "invalid".to_string();
                    report.record_error(
                        "invalid_number_of_frames",
                        "NumberOfFrames must be a positive integer when present".to_string(),
                        tags::NUMBER_OF_FRAMES,
                        "NumberOfFrames",
                        Some(value.to_string()),
                    );
                }
                Err(source) => {
                    report.image.number_of_frames_source = "invalid".to_string();
                    report.record_error(
                    "invalid_number_of_frames",
                    format!("NumberOfFrames is present but cannot be read as an integer: {source:?}"),
                    tags::NUMBER_OF_FRAMES,
                    "NumberOfFrames",
                    element_value(element),
                );
                }
            }
        }
        _ => {
            report.image.number_of_frames = Some(1);
            report.image.number_of_frames_source = "default".to_string();
            report.record_info(
                "default_number_of_frames",
                "NumberOfFrames",
                "NumberOfFrames is absent or empty; preprocessing will treat this as a single-frame image".to_string(),
                Some(tags::NUMBER_OF_FRAMES),
                Some("1".to_string()),
            );
        }
    }
}

fn validate_planar_configuration(
    report: &mut ValidationReport,
    file: &FileDicomObject<InMemDicomObject>,
) {
    match file.get(tags::PLANAR_CONFIGURATION) {
        Some(element) if !element.is_empty() => match element.to_int::<i32>() {
            Ok(value @ 0..=1) => {
                report.image.planar_configuration = Some(value as u16);
                report.record_pass(
                    "PlanarConfiguration",
                    "PlanarConfiguration is valid".to_string(),
                    tags::PLANAR_CONFIGURATION,
                    Some(value.to_string()),
                );
            }
            Ok(value) => {
                report.image.planar_configuration = Some(value as u16);
                report.record_error(
                    "invalid_planar_configuration",
                    "PlanarConfiguration must be 0 or 1 when present".to_string(),
                    tags::PLANAR_CONFIGURATION,
                    "PlanarConfiguration",
                    Some(value.to_string()),
                );
            }
            Err(source) => report.record_error(
                "invalid_planar_configuration",
                format!("PlanarConfiguration cannot be read as an integer: {source:?}"),
                tags::PLANAR_CONFIGURATION,
                "PlanarConfiguration",
                element_value(element),
            ),
        },
        _ => {
            report.image.planar_configuration = Some(0);
            report.record_info(
                "default_planar_configuration",
                "PlanarConfiguration",
                "PlanarConfiguration is absent or empty; dicom-pixeldata defaults to 0".to_string(),
                Some(tags::PLANAR_CONFIGURATION),
                Some("0".to_string()),
            );
        }
    }
}

fn validate_pixel_representation(report: &mut ValidationReport, pixel_representation: Option<u16>) {
    if let Some(value) = pixel_representation {
        if value <= 1 {
            report.record_general_pass(
                "PixelRepresentation value",
                "PixelRepresentation is supported".to_string(),
                Some(value.to_string()),
            );
        } else {
            report.record_error(
                "invalid_pixel_representation",
                "PixelRepresentation must be 0 for unsigned or 1 for signed".to_string(),
                tags::PIXEL_REPRESENTATION,
                "PixelRepresentation",
                Some(value.to_string()),
            );
        }
    }
}

fn validate_bit_relationships(
    report: &mut ValidationReport,
    bits_allocated: Option<u16>,
    bits_stored: Option<u16>,
    high_bit: Option<u16>,
) {
    if let Some(bits_allocated) = bits_allocated {
        if bits_allocated == 8 || bits_allocated == 16 {
            report.record_general_pass(
                "BitsAllocated support",
                "BitsAllocated is supported by this preprocessing output path".to_string(),
                Some(bits_allocated.to_string()),
            );
        } else {
            report.record_error(
                "unsupported_bits_allocated",
                "BitsAllocated must be 8 or 16 for this preprocessing output path".to_string(),
                tags::BITS_ALLOCATED,
                "BitsAllocated",
                Some(bits_allocated.to_string()),
            );
        }
    }

    if let (Some(bits_stored), Some(bits_allocated)) = (bits_stored, bits_allocated) {
        if (1..=bits_allocated).contains(&bits_stored) {
            report.record_general_pass(
                "BitsStored relationship",
                "BitsStored is within the allocated bit depth".to_string(),
                Some(format!("{bits_stored}/{bits_allocated}")),
            );
        } else {
            report.record_error(
                "invalid_bits_stored",
                "BitsStored must be in the range 1..=BitsAllocated".to_string(),
                tags::BITS_STORED,
                "BitsStored",
                Some(bits_stored.to_string()),
            );
        }
    }

    if let (Some(high_bit), Some(bits_stored), Some(bits_allocated)) =
        (high_bit, bits_stored, bits_allocated)
    {
        let expected = bits_stored.saturating_sub(1);
        if bits_stored > 0 && high_bit == expected && high_bit < bits_allocated {
            report.record_general_pass(
                "HighBit relationship",
                "HighBit equals BitsStored - 1 and is less than BitsAllocated".to_string(),
                Some(high_bit.to_string()),
            );
        } else {
            report.record_error(
                "invalid_high_bit",
                format!(
                    "HighBit must equal BitsStored - 1 ({expected}) and be less than BitsAllocated"
                ),
                tags::HIGH_BIT,
                "HighBit",
                Some(high_bit.to_string()),
            );
        }
    }
}

fn validate_output_color_type(
    report: &mut ValidationReport,
    file: &FileDicomObject<InMemDicomObject>,
) {
    match DicomColorType::try_from(file) {
        Ok(color_type) => {
            report.pixel_format.output_color_type = Some(format!("{color_type:?}"));
            report.record_general_pass(
                "Output color type",
                "DICOM pixel format maps to a supported TIFF output color type".to_string(),
                Some(format!("{color_type:?}")),
            );
        }
        Err(source) => report.record_general_error(
            "unsupported_output_color_type",
            "Output color type",
            format!("DICOM pixel format is not supported by the TIFF save path: {source}"),
            None,
        ),
    }
}

fn collect_optional_diagnostics(
    report: &mut ValidationReport,
    file: &FileDicomObject<InMemDicomObject>,
) {
    report.preprocessing.rescale_slope = optional_float_values(file, tags::RESCALE_SLOPE);
    report.preprocessing.rescale_intercept = optional_float_values(file, tags::RESCALE_INTERCEPT);
    report.preprocessing.window_center = optional_float_values(file, tags::WINDOW_CENTER);
    report.preprocessing.window_width = optional_float_values(file, tags::WINDOW_WIDTH);
    report.preprocessing.voi_lut_function = optional_string_values(file, tags::VOILUT_FUNCTION);

    let rescale_slope = report
        .preprocessing
        .rescale_slope
        .as_ref()
        .map(|values| format!("{values:?}"));
    let rescale_intercept = report
        .preprocessing
        .rescale_intercept
        .as_ref()
        .map(|values| format!("{values:?}"));
    let window_center = report
        .preprocessing
        .window_center
        .as_ref()
        .map(|values| format!("{values:?}"));
    let window_width = report
        .preprocessing
        .window_width
        .as_ref()
        .map(|values| format!("{values:?}"));
    let voi_lut_function = report
        .preprocessing
        .voi_lut_function
        .as_ref()
        .map(|values| format!("{values:?}"));

    diagnostic_for_optional_values(
        report,
        "RescaleSlope",
        tags::RESCALE_SLOPE,
        rescale_slope,
        "RescaleSlope is absent; dicom-pixeldata defaults to slope 1.0",
    );
    diagnostic_for_optional_values(
        report,
        "RescaleIntercept",
        tags::RESCALE_INTERCEPT,
        rescale_intercept,
        "RescaleIntercept is absent; dicom-pixeldata defaults to intercept 0.0",
    );
    diagnostic_for_optional_values(
        report,
        "WindowCenter",
        tags::WINDOW_CENTER,
        window_center,
        "WindowCenter is absent; display normalization will use other VOI data or pixel range normalization",
    );
    diagnostic_for_optional_values(
        report,
        "WindowWidth",
        tags::WINDOW_WIDTH,
        window_width,
        "WindowWidth is absent; display normalization will use other VOI data or pixel range normalization",
    );

    match file.get(tags::VOILUT_FUNCTION) {
        Some(element) if element.is_empty() => report.record_warning(
            "empty_voi_lut_function",
            "VOILUTFunction",
            "VOILUTFunction is present but empty; preprocessing sanitizes this before decode"
                .to_string(),
            Some(tags::VOILUT_FUNCTION),
            None,
        ),
        Some(_) => report.record_info(
            "voi_lut_function_present",
            "VOILUTFunction",
            "VOILUTFunction is present".to_string(),
            Some(tags::VOILUT_FUNCTION),
            voi_lut_function,
        ),
        None => report.record_info(
            "voi_lut_function_absent",
            "VOILUTFunction",
            "VOILUTFunction is absent; dicom-pixeldata defaults to LINEAR when needed".to_string(),
            Some(tags::VOILUT_FUNCTION),
            None,
        ),
    }

    match file.get(tags::VOILUT_SEQUENCE) {
        Some(element) => {
            let item_count = element.items().map(|items| items.len());
            report.preprocessing.voi_lut_sequence_present = true;
            report.preprocessing.voi_lut_sequence_items = item_count;
            report.record_info(
                "voi_lut_sequence_present",
                "VOILUTSequence",
                "VOILUTSequence is present".to_string(),
                Some(tags::VOILUT_SEQUENCE),
                item_count.map(|value| value.to_string()),
            );
        }
        None => report.record_info(
            "voi_lut_sequence_absent",
            "VOILUTSequence",
            "VOILUTSequence is absent; this is not required for preprocessing".to_string(),
            Some(tags::VOILUT_SEQUENCE),
            None,
        ),
    }

    collect_resolution_diagnostics(report, file);
    collect_output_naming_diagnostics(report, file);
}

fn diagnostic_for_optional_values(
    report: &mut ValidationReport,
    name: &str,
    tag: Tag,
    value: Option<String>,
    absent_message: &str,
) {
    match value {
        Some(value) => report.record_info(
            "optional_tag_present",
            name,
            format!("{name} is present"),
            Some(tag),
            Some(value),
        ),
        None => report.record_warning(
            "optional_tag_absent",
            name,
            absent_message.to_string(),
            Some(tag),
            None,
        ),
    }
}

fn collect_resolution_diagnostics(
    report: &mut ValidationReport,
    file: &FileDicomObject<InMemDicomObject>,
) {
    match Resolution::try_from(file) {
        Ok(resolution) => {
            report.preprocessing.resolution = ResolutionReport {
                available: true,
                pixels_per_mm_x: Some(resolution.pixels_per_mm_x),
                pixels_per_mm_y: Some(resolution.pixels_per_mm_y),
                frames_per_mm: resolution.frames_per_mm,
            };
            report.record_info(
                "resolution_available",
                "Pixel spacing",
                "pixel spacing metadata is available for spacing-based preprocessing".to_string(),
                Some(tags::PIXEL_SPACING),
                Some(format!(
                    "x={:.6}, y={:.6}, z={}",
                    resolution.pixels_per_mm_x,
                    resolution.pixels_per_mm_y,
                    resolution
                        .frames_per_mm
                        .map(|value| format!("{value:.6}"))
                        .unwrap_or_else(|| "unavailable".to_string())
                )),
            );
        }
        Err(_) => report.record_warning(
            "resolution_unavailable",
            "Pixel spacing",
            "pixel spacing metadata is unavailable; spacing-based resize metadata cannot be derived".to_string(),
            Some(tags::PIXEL_SPACING),
            None,
        ),
    }
}

fn collect_output_naming_diagnostics(
    report: &mut ValidationReport,
    file: &FileDicomObject<InMemDicomObject>,
) {
    let study_instance_uid = optional_single_string(file, tags::STUDY_INSTANCE_UID);
    let series_instance_uid = optional_single_string(file, tags::SERIES_INSTANCE_UID);
    let sop_instance_uid = optional_single_string(file, tags::SOP_INSTANCE_UID);
    let directory_output_ready = study_instance_uid.is_some() && sop_instance_uid.is_some();

    report.preprocessing.output_naming = OutputNamingReport {
        directory_output_ready,
        study_instance_uid,
        series_instance_uid: series_instance_uid.or_else(|| Some("series".to_string())),
        sop_instance_uid,
    };

    if report.preprocessing.output_naming.directory_output_ready {
        report.record_info(
            "directory_output_ready",
            "Directory output naming",
            "StudyInstanceUID and SOPInstanceUID are present for directory output naming"
                .to_string(),
            None,
            None,
        );
    } else {
        report.record_warning(
            "directory_output_not_ready",
            "Directory output naming",
            "StudyInstanceUID or SOPInstanceUID is absent; directory-output preprocessing would fail, but explicit file output may still work".to_string(),
            None,
            None,
        );
    }
}

fn validate_frame_decode(
    report: &mut ValidationReport,
    file: &mut FileDicomObject<InMemDicomObject>,
    bits_allocated: Option<u16>,
    samples_per_pixel: Option<u16>,
    rows: Option<u16>,
    columns: Option<u16>,
) {
    let expected_frame_bytes = match (bits_allocated, samples_per_pixel, rows, columns) {
        (Some(bits_allocated), Some(samples_per_pixel), Some(rows), Some(columns)) => {
            let bytes_per_sample = usize::from(bits_allocated).div_ceil(8);
            Some(
                usize::from(rows)
                    * usize::from(columns)
                    * usize::from(samples_per_pixel)
                    * bytes_per_sample,
            )
        }
        _ => None,
    };

    if !report.errors.is_empty() {
        let message = "frame decode smoke test skipped because critical metadata errors were found"
            .to_string();
        report.decode_smoke_test = Some(DecodeSmokeTestReport {
            attempted: false,
            status: CheckStatus::Skip,
            message: message.clone(),
            frame_index: Some(0),
            image_color_type: None,
            expected_frame_bytes,
        });
        report.record_skip("Frame decode smoke test", message);
        return;
    }

    Preprocessor::sanitize_dicom(file);

    match file
        .decode_pixel_data_frame(0)
        .and_then(|decoded| decoded.to_dynamic_image(0))
    {
        Ok(image) => {
            let color_type = image.color();
            report.decode_smoke_test = Some(DecodeSmokeTestReport {
                attempted: true,
                status: CheckStatus::Pass,
                message: "frame 0 decoded and converted to an image".to_string(),
                frame_index: Some(0),
                image_color_type: Some(format_image_color_type(color_type)),
                expected_frame_bytes,
            });
            report.record_general_pass(
                "Frame decode smoke test",
                "frame 0 decoded and converted to an image".to_string(),
                Some(format_image_color_type(color_type)),
            );
        }
        Err(source) => {
            let message = format!("frame 0 decode/image conversion failed: {source:?}");
            report.decode_smoke_test = Some(DecodeSmokeTestReport {
                attempted: true,
                status: CheckStatus::Fail,
                message: message.clone(),
                frame_index: Some(0),
                image_color_type: None,
                expected_frame_bytes,
            });
            report.record_general_error(
                "frame_decode_failed",
                "Frame decode smoke test",
                message,
                None,
            );
        }
    }
}

fn optional_float_values(file: &FileDicomObject<InMemDicomObject>, tag: Tag) -> Option<Vec<f64>> {
    let element = file.get(tag)?;
    if element.is_empty() {
        return None;
    }
    element
        .value()
        .to_str()
        .ok()
        .and_then(|value| {
            let values: Option<Vec<f64>> = value
                .split('\\')
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(|value| value.parse::<f64>().ok())
                .collect();
            values.filter(|values| !values.is_empty())
        })
        .or_else(|| element.to_float64().ok().map(|value| vec![value]))
}

fn optional_string_values(
    file: &FileDicomObject<InMemDicomObject>,
    tag: Tag,
) -> Option<Vec<String>> {
    let element = file.get(tag)?;
    if element.is_empty() {
        return None;
    }
    element.value().to_str().ok().and_then(|value| {
        let values: Vec<String> = value
            .split('\\')
            .map(|value| value.trim_matches(|c: char| c.is_whitespace() || c == '\0'))
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)
            .collect();
        (!values.is_empty()).then_some(values)
    })
}

fn optional_single_string(file: &FileDicomObject<InMemDicomObject>, tag: Tag) -> Option<String> {
    optional_string_values(file, tag).and_then(|values| values.into_iter().next())
}

fn element_value(element: &DataElement<InMemDicomObject>) -> Option<String> {
    element
        .value()
        .to_str()
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn format_tag(tag: Tag) -> String {
    format!("({:04X},{:04X})", tag.0, tag.1)
}

fn format_image_color_type(color_type: ColorType) -> String {
    match color_type {
        ColorType::L8 => "L8".to_string(),
        ColorType::La8 => "La8".to_string(),
        ColorType::Rgb8 => "Rgb8".to_string(),
        ColorType::Rgba8 => "Rgba8".to_string(),
        ColorType::L16 => "L16".to_string(),
        ColorType::La16 => "La16".to_string(),
        ColorType::Rgb16 => "Rgb16".to_string(),
        ColorType::Rgba16 => "Rgba16".to_string(),
        ColorType::Rgb32F => "Rgb32F".to_string(),
        ColorType::Rgba32F => "Rgba32F".to_string(),
        other => format!("{other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use dicom::core::{DataElement, PrimitiveValue, VR};
    use dicom::object::open_file;
    use rstest::rstest;

    fn fixture() -> FileDicomObject<InMemDicomObject> {
        open_file(dicom_test_files::path("pydicom/CT_small.dcm").unwrap()).unwrap()
    }

    fn validation_codes(report: &ValidationReport) -> Vec<&str> {
        report
            .errors
            .iter()
            .map(|error| error.code.as_str())
            .collect()
    }

    #[rstest]
    #[case(tags::HIGH_BIT, "missing_high_bit")]
    #[case(tags::BITS_STORED, "missing_bits_stored")]
    #[case(tags::BITS_ALLOCATED, "missing_bits_allocated")]
    fn missing_critical_bit_tags_fail_decoder_and_validator(
        #[case] tag: Tag,
        #[case] expected_code: &str,
    ) {
        let mut dicom = fixture();
        assert!(dicom.remove_element(tag));
        assert!(dicom.decode_pixel_data().is_err());

        let report = validate_object(
            Path::new("missing-tag.dcm"),
            &mut dicom,
            DecodeValidation::Frame,
        );
        assert!(!report.is_valid());
        assert!(validation_codes(&report).contains(&expected_code));
    }

    #[test]
    fn valid_fixture_passes_with_frame_decode() {
        let mut dicom = fixture();
        let report = validate_object(Path::new("valid.dcm"), &mut dicom, DecodeValidation::Frame);
        assert!(report.is_valid(), "{:?}", report.errors);
        assert_eq!(report.status, ValidationStatus::Pass);
        assert_eq!(
            report
                .decode_smoke_test
                .as_ref()
                .expect("smoke test report")
                .status,
            CheckStatus::Pass
        );
    }

    #[test]
    fn unsupported_output_color_type_is_critical() {
        let mut dicom = fixture();
        dicom.put_element(DataElement::new(
            tags::PHOTOMETRIC_INTERPRETATION,
            VR::CS,
            PrimitiveValue::from("RGB"),
        ));

        let report = validate_object(
            Path::new("unsupported-color.dcm"),
            &mut dicom,
            DecodeValidation::Frame,
        );
        assert!(!report.is_valid());
        assert!(validation_codes(&report).contains(&"unsupported_output_color_type"));
    }

    #[test]
    fn invalid_bits_stored_relationship_is_critical() {
        let mut dicom = fixture();
        dicom.put_element(DataElement::new(
            tags::BITS_STORED,
            VR::US,
            PrimitiveValue::from(17_u16),
        ));

        let report = validate_object(
            Path::new("bad-bits-stored.dcm"),
            &mut dicom,
            DecodeValidation::Frame,
        );
        assert!(!report.is_valid());
        assert!(validation_codes(&report).contains(&"invalid_bits_stored"));
    }

    #[test]
    fn invalid_high_bit_relationship_is_critical() {
        let mut dicom = fixture();
        dicom.put_element(DataElement::new(
            tags::HIGH_BIT,
            VR::US,
            PrimitiveValue::from(14_u16),
        ));

        let report = validate_object(
            Path::new("bad-high-bit.dcm"),
            &mut dicom,
            DecodeValidation::Frame,
        );
        assert!(!report.is_valid());
        assert!(validation_codes(&report).contains(&"invalid_high_bit"));
    }

    #[test]
    fn missing_optional_display_metadata_warns_without_invalidating() {
        let mut dicom = fixture();
        dicom.remove_element(tags::WINDOW_CENTER);
        dicom.remove_element(tags::WINDOW_WIDTH);
        dicom.remove_element(tags::VOILUT_FUNCTION);
        dicom.remove_element(tags::VOILUT_SEQUENCE);

        let report = validate_object(
            Path::new("optional-missing.dcm"),
            &mut dicom,
            DecodeValidation::Frame,
        );
        assert!(report.is_valid(), "{:?}", report.errors);
        assert!(report
            .warnings
            .iter()
            .any(|warning| warning.tag_name.as_deref() == Some("WindowCenter")));
    }

    #[test]
    fn empty_voi_lut_function_is_sanitized_for_decode() {
        let mut dicom = fixture();
        dicom.put_element(DataElement::new(
            tags::VOILUT_FUNCTION,
            VR::LO,
            PrimitiveValue::Empty,
        ));

        let report = validate_object(
            Path::new("empty-voi.dcm"),
            &mut dicom,
            DecodeValidation::Frame,
        );
        assert!(report.is_valid(), "{:?}", report.errors);
        assert!(report
            .warnings
            .iter()
            .any(|warning| warning.code == "empty_voi_lut_function"));
        assert_eq!(
            report
                .decode_smoke_test
                .as_ref()
                .expect("smoke test report")
                .status,
            CheckStatus::Pass
        );
    }

    #[test]
    fn invalid_number_of_frames_is_critical() {
        let mut dicom = fixture();
        dicom.put_element(DataElement::new(
            tags::NUMBER_OF_FRAMES,
            VR::IS,
            PrimitiveValue::from("0"),
        ));

        let report = validate_object(
            Path::new("bad-frames.dcm"),
            &mut dicom,
            DecodeValidation::Frame,
        );
        assert!(!report.is_valid());
        assert!(validation_codes(&report).contains(&"invalid_number_of_frames"));
    }
}
