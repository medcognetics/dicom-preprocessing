//! Offline, deterministic installation checks. No test runs until explicitly requested.
mod builtin;
#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, BTreeSet};
use std::io::Cursor;

use dicom::core::{Tag, VR};
use dicom::object::{from_reader, open_file, FileDicomObject, InMemDicomObject};
use serde::{Deserialize, Serialize};

use crate::{ViewerDicom, VolumeHandler};

/// A primitive tag, or a sequence item followed by another path component.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TagPathComponent {
    pub group: u16,
    pub element: u16,
    #[serde(default)]
    pub item: Option<usize>,
}

/// Expected primitive values use DICOM's canonical string conversion, in value order.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TagExpectation {
    pub path: Vec<TagPathComponent>,
    pub vr: String,
    pub values: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SampleType {
    U8,
    I8,
    U16,
    I16,
}

/// Samples follow the decoded raw frame layout (including planar configuration).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrameExpectation {
    pub width: u32,
    pub height: u32,
    pub samples_per_pixel: u16,
    pub planar_configuration: u16,
    pub sample_type: SampleType,
    pub values: Vec<i32>,
    #[serde(default)]
    pub absolute_tolerance: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VerificationCase {
    pub id: String,
    pub dicom_bytes: Vec<u8>,
    pub transfer_syntax_uid: String,
    pub tags: Vec<TagExpectation>,
    pub frames: Vec<FrameExpectation>,
}

/// Each UID maps to one or more required fixture or custom-test IDs.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CodecDeclaration {
    pub id: String,
    pub required_cases: BTreeMap<String, Vec<String>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VerificationCheck {
    pub name: String,
    pub passed: bool,
    #[serde(default)]
    pub diagnostic: Option<String>,
}

impl VerificationCheck {
    pub fn new(name: impl Into<String>, passed: bool) -> Self {
        Self {
            name: name.into(),
            passed,
            diagnostic: None,
        }
    }

    pub fn failure(name: impl Into<String>, diagnostic: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            passed: false,
            diagnostic: Some(diagnostic.into()),
        }
    }
}

/// Callbacks run synchronously, once, in declaration order. They must not panic.
pub trait VerificationTest {
    fn id(&self) -> &str;
    fn run(&self) -> Result<Vec<VerificationCheck>, String>;
}

#[derive(Default)]
pub struct VerificationOptions<'a> {
    pub cases: Vec<VerificationCase>,
    pub codecs: Vec<CodecDeclaration>,
    pub tests: Vec<&'a dyn VerificationTest>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationPath {
    SharedLibrary,
    CallerIntegration,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationCaseResult {
    pub id: String,
    pub source: String,
    pub path: VerificationPath,
    pub transfer_syntax_uid: Option<String>,
    pub passed: bool,
    pub checks: Vec<VerificationCheck>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CodecCoverageResult {
    pub id: String,
    pub transfer_syntax_uid: String,
    pub required_cases: Vec<String>,
    /// True only when a successful shared-library fixture tests this exact UID.
    pub shared_library_verified: bool,
    pub passed: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationReport {
    pub schema_version: u32,
    pub suite_version: String,
    pub library_version: String,
    pub passed: bool,
    pub cases: Vec<VerificationCaseResult>,
    pub codecs: Vec<CodecCoverageResult>,
}

#[derive(Clone, Debug, thiserror::Error)]
#[error("invalid verification configuration: {0}")]
pub struct VerificationConfigurationError(pub String);

/// Run the built-in suite. Failures are data, not exceptions.
pub fn verify_runtime() -> VerificationReport {
    verify_runtime_with(&VerificationOptions::default()).expect("built-in configuration is valid")
}

/// Validate all declarations before running any checks or callbacks.
pub fn verify_runtime_with(
    options: &VerificationOptions<'_>,
) -> Result<VerificationReport, VerificationConfigurationError> {
    let ids = options
        .tests
        .iter()
        .map(|test| test.id().to_owned())
        .collect();
    let suite = PreparedVerification::new(options.cases.clone(), options.codecs.clone(), ids)?;
    let mut run = suite.run_shared();
    for test in &options.tests {
        run.record_custom(test.run());
    }
    Ok(run.finish())
}

/// Two-stage runner for bindings: release interpreter locks only during `run_shared`.
/// Custom test IDs are validated before any shared checks execute.
pub struct PreparedVerification {
    cases: Vec<VerificationCase>,
    codecs: Vec<CodecDeclaration>,
    test_ids: Vec<String>,
}

fn config(message: &str) -> VerificationConfigurationError {
    VerificationConfigurationError(message.into())
}
fn valid_id(id: &str) -> bool {
    id.len() <= 128
        && id
            .split_once('/')
            .is_some_and(|(namespace, name)| !namespace.is_empty() && !name.is_empty())
        && id
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"/_-.".contains(&c))
}
fn valid_uid(uid: &str) -> bool {
    !uid.is_empty()
        && uid.len() <= 64
        && uid
            .split('.')
            .all(|part| !part.is_empty() && part.bytes().all(|c| c.is_ascii_digit()))
}

impl PreparedVerification {
    pub fn new(
        cases: Vec<VerificationCase>,
        codecs: Vec<CodecDeclaration>,
        test_ids: Vec<String>,
    ) -> Result<Self, VerificationConfigurationError> {
        let mut ids = BTreeSet::new();
        for id in cases
            .iter()
            .map(|case| &case.id)
            .chain(&test_ids)
            .chain(codecs.iter().map(|codec| &codec.id))
        {
            if !valid_id(id) || id.starts_with("builtin/") || !ids.insert(id.clone()) {
                return Err(config(
                    "IDs must be unique, namespaced, and outside builtin/",
                ));
            }
        }
        for case in &cases {
            validate_case(case)?;
        }
        let builtin_ids = builtin::case_ids();
        for codec in &codecs {
            if codec.required_cases.is_empty() {
                return Err(config("codec coverage must not be empty"));
            }
            for (uid, required) in &codec.required_cases {
                if !valid_uid(uid) || required.is_empty() {
                    return Err(config("codec coverage requires a valid UID and test IDs"));
                }
                let mut seen = BTreeSet::new();
                for id in required {
                    if !seen.insert(id) {
                        return Err(config("duplicate coverage reference"));
                    }
                    if let Some(case) = cases.iter().find(|case| &case.id == id) {
                        if &case.transfer_syntax_uid != uid {
                            return Err(config(
                                "fixture coverage UID does not match its expectation",
                            ));
                        }
                    } else if !test_ids.contains(id)
                        && !builtin_ids
                            .iter()
                            .any(|(builtin_id, syntax)| builtin_id == id && syntax == uid)
                    {
                        return Err(config("unresolved codec coverage reference"));
                    }
                }
            }
        }
        Ok(Self {
            cases,
            codecs,
            test_ids,
        })
    }

    pub fn run_shared(self) -> VerificationRun {
        let (mut cases, mut codecs) = builtin::run();
        cases.extend(
            self.cases
                .iter()
                .map(|case| run_case(case, "extension_fixture")),
        );
        // Newly linked production decoders require shared-path fixtures. A callback
        // cannot satisfy the built-in decoder inventory on its own.
        for codec in &mut codecs {
            for (uid, required) in &mut codec.required_cases {
                if required.is_empty() {
                    required.extend(
                        self.cases
                            .iter()
                            .filter(|case| &case.transfer_syntax_uid == uid)
                            .map(|case| case.id.clone()),
                    );
                }
            }
        }
        codecs.extend(self.codecs);
        VerificationRun {
            cases,
            codecs,
            test_ids: self.test_ids,
            completed: 0,
        }
    }
}

/// Accumulates binding callbacks in their previously validated order.
pub struct VerificationRun {
    cases: Vec<VerificationCaseResult>,
    codecs: Vec<CodecDeclaration>,
    test_ids: Vec<String>,
    completed: usize,
}

impl VerificationRun {
    pub fn record_custom(&mut self, outcome: Result<Vec<VerificationCheck>, String>) {
        let Some(id) = self.test_ids.get(self.completed).cloned() else {
            self.cases.push(result(
                "builtin/unexpected-callback".into(),
                "custom",
                VerificationPath::CallerIntegration,
                None,
                vec![VerificationCheck::failure(
                    "callback_count",
                    "more callbacks than declared",
                )],
            ));
            return;
        };
        self.completed += 1;
        let mut checks = match outcome {
            Ok(checks) => checks,
            Err(_) => vec![VerificationCheck::failure(
                "callback",
                "custom callback failed",
            )],
        };
        let mut names = BTreeSet::new();
        if checks.is_empty()
            || checks.iter().any(|check| {
                check.name.is_empty() || check.name.len() > 128 || !names.insert(check.name.clone())
            })
        {
            checks = vec![VerificationCheck::failure(
                "callback_result",
                "expected nonempty checks with unique names of at most 128 bytes",
            )];
        }
        for check in &mut checks {
            check.diagnostic = check
                .diagnostic
                .take()
                .map(|text| text.chars().filter(|c| !c.is_control()).take(256).collect());
        }
        self.cases.push(result(
            id,
            "custom",
            VerificationPath::CallerIntegration,
            None,
            checks,
        ));
    }

    pub fn finish(mut self) -> VerificationReport {
        while self.completed < self.test_ids.len() {
            self.record_custom(Err("callback not executed".into()));
        }
        let codecs = self
            .codecs
            .iter()
            .flat_map(|codec| {
                codec.required_cases.iter().map(|(uid, ids)| {
                    let matches: Vec<_> = ids
                        .iter()
                        .filter_map(|id| self.cases.iter().find(|case| &case.id == id))
                        .collect();
                    CodecCoverageResult {
                        id: codec.id.clone(),
                        transfer_syntax_uid: uid.clone(),
                        required_cases: ids.clone(),
                        passed: !ids.is_empty()
                            && matches.len() == ids.len()
                            && matches.iter().all(|case| case.passed),
                        shared_library_verified: matches.iter().any(|case| {
                            case.passed
                                && case.path == VerificationPath::SharedLibrary
                                && case.transfer_syntax_uid.as_ref() == Some(uid)
                        }),
                    }
                })
            })
            .collect::<Vec<_>>();
        VerificationReport {
            schema_version: 1,
            suite_version: "2".into(),
            library_version: env!("CARGO_PKG_VERSION").into(),
            passed: self.cases.iter().all(|case| case.passed)
                && codecs.iter().all(|codec| codec.passed),
            cases: self.cases,
            codecs,
        }
    }
}

fn validate_case(case: &VerificationCase) -> Result<(), VerificationConfigurationError> {
    if !valid_uid(&case.transfer_syntax_uid)
        || case.dicom_bytes.is_empty()
        || case.frames.is_empty()
        || case.tags.is_empty()
    {
        return Err(config(
            "fixtures require bytes, a valid UID, tags, and frames",
        ));
    }
    for tag in &case.tags {
        if tag.path.is_empty()
            || tag.values.is_empty()
            || tag.vr.parse::<VR>().is_err()
            || tag
                .path
                .last()
                .is_some_and(|component| component.item.is_some())
            || tag.path[..tag.path.len() - 1]
                .iter()
                .any(|component| component.item.is_none())
        {
            return Err(config("invalid tag expectation or sequence path"));
        }
    }
    for frame in &case.frames {
        let length = (frame.width as usize)
            .checked_mul(frame.height as usize)
            .and_then(|n| n.checked_mul(frame.samples_per_pixel as usize));
        let (min, max) = match frame.sample_type {
            SampleType::U8 => (0, 255),
            SampleType::I8 => (-128, 127),
            SampleType::U16 => (0, 65535),
            SampleType::I16 => (-32768, 32767),
        };
        if frame.width == 0
            || frame.height == 0
            || !matches!(frame.samples_per_pixel, 1 | 3)
            || frame.planar_configuration > 1
            || (frame.samples_per_pixel == 1 && frame.planar_configuration != 0)
            || length != Some(frame.values.len())
            || frame.values.iter().any(|&v| v < min || v > max)
            || frame.absolute_tolerance > (max - min) as u32
        {
            return Err(config(
                "invalid frame dimensions, layout, sample values, or tolerance",
            ));
        }
    }
    Ok(())
}

fn result(
    id: String,
    source: &str,
    path: VerificationPath,
    uid: Option<String>,
    checks: Vec<VerificationCheck>,
) -> VerificationCaseResult {
    VerificationCaseResult {
        id,
        source: source.into(),
        path,
        transfer_syntax_uid: uid,
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

fn run_case(case: &VerificationCase, source: &str) -> VerificationCaseResult {
    let run = || -> Result<Vec<VerificationCheck>, &'static str> {
        let mut first = None;
        let mut checks = Vec::new();
        let directory = tempfile::tempdir().map_err(|_| "temporary storage unavailable")?;
        let path = directory.path().join("fixture.dcm");
        std::fs::write(&path, &case.dicom_bytes).map_err(|_| "fixture write failed")?;
        for iteration in 0..3 {
            let file = if iteration == 1 {
                open_file(&path).map_err(|_| "file parse failed")?
            } else {
                from_reader(Cursor::new(&case.dicom_bytes)).map_err(|_| "byte parse failed")?
            };
            let current =
                snapshot(file, case).map_err(|_| "pixel decoding or frame metadata failed")?;
            if let Some(first) = &first {
                checks.push(VerificationCheck::new(
                    format!("repeat_{iteration}"),
                    first == &current,
                ));
            } else {
                checks.extend(current.0.clone());
                first = Some(current);
            }
        }
        directory
            .close()
            .map_err(|_| "temporary storage cleanup failed")?;
        Ok(checks)
    };
    let checks =
        run().unwrap_or_else(|diagnostic| vec![VerificationCheck::failure("load", diagnostic)]);
    result(
        case.id.clone(),
        source,
        VerificationPath::SharedLibrary,
        Some(case.transfer_syntax_uid.clone()),
        checks,
    )
}
