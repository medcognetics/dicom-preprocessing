//! Built-in, offline runtime self-test of DICOM parsing, pixel decoding, and preprocessing.
//!
//! [`verify_runtime`] decodes small embedded synthetic DICOM files for every enabled transfer
//! syntax and checks the results against expectations from independent encoders. Nothing runs
//! until it is called. The report omits timestamps so repeated runs compare equal.
mod builtin;
#[cfg(test)]
mod tests;

use std::io::Cursor;

use dicom::core::Tag;
use dicom::object::{from_reader, FileDicomObject, InMemDicomObject};
use serde::{Deserialize, Serialize};

use crate::{ViewerDicom, VolumeHandler};

/// Version of the report shape.
const SCHEMA_VERSION: u32 = 1;
/// Version of the embedded corpus and check semantics.
const SUITE_VERSION: &str = "1";

/// A primitive tag, or a sequence item followed by another path component.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TagPathComponent {
    group: u16,
    element: u16,
    #[serde(default)]
    item: Option<usize>,
}

/// Expected primitive values use DICOM's canonical string conversion, in value order.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TagExpectation {
    path: Vec<TagPathComponent>,
    vr: String,
    values: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum SampleType {
    U8,
    I8,
    U16,
    I16,
}

/// Samples follow the decoded raw frame layout, including planar configuration.
#[derive(Clone, Debug)]
struct FrameExpectation {
    width: u32,
    height: u32,
    samples_per_pixel: u16,
    planar_configuration: u16,
    sample_type: SampleType,
    values: Vec<i32>,
    absolute_tolerance: u32,
}

/// One embedded fixture and its expectations.
#[derive(Clone, Debug)]
struct VerificationCase {
    id: String,
    dicom_bytes: &'static [u8],
    transfer_syntax_uid: String,
    tags: Vec<TagExpectation>,
    frames: Vec<FrameExpectation>,
}

/// One named check within a case.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct VerificationCheck {
    pub name: String,
    pub passed: bool,
    pub diagnostic: Option<String>,
}

impl VerificationCheck {
    fn new(name: impl Into<String>, passed: bool) -> Self {
        Self {
            name: name.into(),
            passed,
            diagnostic: None,
        }
    }

    fn failure(name: impl Into<String>, diagnostic: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            passed: false,
            diagnostic: Some(diagnostic.into()),
        }
    }
}

/// The result of one embedded fixture (`source` "embedded_fixture") or built-in check
/// (`source` "builtin").
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct VerificationCaseResult {
    pub id: String,
    pub source: String,
    pub transfer_syntax_uid: Option<String>,
    pub passed: bool,
    pub checks: Vec<VerificationCheck>,
}

/// Coverage of one transfer syntax with a registered decoder: every required case must pass.
/// A decoder without a fixture has no required cases and fails.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct CodecCoverageResult {
    pub transfer_syntax_uid: String,
    pub required_cases: Vec<String>,
    pub passed: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct VerificationReport {
    pub schema_version: u32,
    pub suite_version: String,
    pub library_version: String,
    pub passed: bool,
    pub cases: Vec<VerificationCaseResult>,
    pub codecs: Vec<CodecCoverageResult>,
}

/// Runs the built-in suite. Failures are report data, not errors.
pub fn verify_runtime() -> VerificationReport {
    let (cases, codecs) = builtin::run();
    VerificationReport {
        schema_version: SCHEMA_VERSION,
        suite_version: SUITE_VERSION.into(),
        library_version: env!("CARGO_PKG_VERSION").into(),
        passed: cases.iter().all(|case| case.passed) && codecs.iter().all(|codec| codec.passed),
        cases,
        codecs,
    }
}

fn result(
    id: &str,
    source: &str,
    uid: Option<&str>,
    checks: Vec<VerificationCheck>,
) -> VerificationCaseResult {
    VerificationCaseResult {
        id: id.into(),
        source: source.into(),
        transfer_syntax_uid: uid.map(Into::into),
        passed: !checks.is_empty() && checks.iter().all(|check| check.passed),
        checks,
    }
}

fn check_tags(object: &InMemDicomObject, expected: &[TagExpectation]) -> bool {
    expected.iter().all(|expected| {
        let mut object = object;
        for component in &expected.path {
            let Ok(element) = object.element(Tag(component.group, component.element)) else {
                return false;
            };
            if let Some(index) = component.item {
                let Some(item) = element.items().and_then(|items| items.get(index)) else {
                    return false;
                };
                object = item;
            } else {
                return element.vr().to_string() == expected.vr
                    && element
                        .to_multi_str()
                        .is_ok_and(|values| values.as_ref() == expected.values);
            }
        }
        false
    })
}

/// Checks one load of a fixture and returns the checks with the raw decoded frames.
fn snapshot(
    file: FileDicomObject<InMemDicomObject>,
    case: &VerificationCase,
) -> Result<(Vec<VerificationCheck>, Vec<Vec<u8>>), ()> {
    let mut checks = vec![
        VerificationCheck::new(
            "transfer_syntax",
            file.meta().transfer_syntax.trim_end_matches('\0') == case.transfer_syntax_uid,
        ),
        VerificationCheck::new("tags", check_tags(&file, &case.tags)),
    ];
    let viewer = ViewerDicom::from_object(file, VolumeHandler::keep()).map_err(|_| ())?;
    let count = viewer
        .file()
        .element(dicom::dictionary_std::tags::NUMBER_OF_FRAMES)
        .ok()
        .map(|element| element.to_int::<u32>())
        .transpose()
        .map_err(|_| ())?
        .unwrap_or(1);
    checks.push(VerificationCheck::new(
        "frame_count",
        count as usize == case.frames.len(),
    ));
    let mut raw = Vec::new();
    for (index, expected) in case.frames.iter().enumerate() {
        let frame =
            VolumeHandler::decode_stored_frame_raw(viewer.file(), index as u32).map_err(|_| ())?;
        let sample_type = match (frame.bits_allocated, frame.pixel_representation_signed) {
            (8, false) => SampleType::U8,
            (8, true) => SampleType::I8,
            (16, false) => SampleType::U16,
            (16, true) => SampleType::I16,
            _ => return Err(()),
        };
        let values: Vec<i32> = match sample_type {
            SampleType::U8 => frame.data.iter().map(|&v| i32::from(v)).collect(),
            SampleType::I8 => frame.data.iter().map(|&v| i32::from(v as i8)).collect(),
            SampleType::U16 => frame
                .data
                .chunks_exact(2)
                .map(|v| i32::from(u16::from_ne_bytes([v[0], v[1]])))
                .collect(),
            SampleType::I16 => frame
                .data
                .chunks_exact(2)
                .map(|v| i32::from(i16::from_ne_bytes([v[0], v[1]])))
                .collect(),
        };
        let layout = frame.width == expected.width
            && frame.height == expected.height
            && frame.samples_per_pixel == expected.samples_per_pixel
            && frame.planar_configuration as u16 == expected.planar_configuration
            && sample_type == expected.sample_type
            && frame.data.len() == expected.values.len() * usize::from(frame.bits_allocated / 8);
        checks.push(VerificationCheck::new(
            format!("frame_{index}_layout"),
            layout,
        ));
        checks.push(VerificationCheck::new(
            format!("frame_{index}_pixels"),
            values.len() == expected.values.len()
                && values
                    .iter()
                    .zip(&expected.values)
                    .all(|(&actual, &expected_value)| {
                        actual.abs_diff(expected_value) <= expected.absolute_tolerance
                    }),
        ));
        raw.push(frame.data);
    }
    Ok((checks, raw))
}

/// Loads a fixture twice from its bytes, checks the first load, and requires the second to
/// produce identical checks and samples.
fn run_case(case: &VerificationCase) -> VerificationCaseResult {
    let load = || -> Result<_, &'static str> {
        let file = from_reader(Cursor::new(case.dicom_bytes)).map_err(|_| "DICOM parse failed")?;
        snapshot(file, case).map_err(|()| "pixel decoding or frame metadata failed")
    };
    let checks = match (load(), load()) {
        (Ok(first), Ok(second)) => {
            let repeated = first == second;
            let mut checks = first.0;
            checks.push(VerificationCheck::new("repeat", repeated));
            checks
        }
        (Err(diagnostic), _) | (_, Err(diagnostic)) => {
            vec![VerificationCheck::failure("load", diagnostic)]
        }
    };
    result(
        &case.id,
        "embedded_fixture",
        Some(&case.transfer_syntax_uid),
        checks,
    )
}
