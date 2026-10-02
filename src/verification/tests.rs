use super::builtin::{self, FIXTURES};
use super::*;

/// Upper bound on fixture and manifest bytes embedded in the binary.
const EMBEDDED_BUDGET: usize = 48 * 1024;

#[test]
fn builtin_suite_passes_and_is_repeatable() {
    let first = verify_runtime();
    // Fix failures in the decoders; never change independent pixel oracles to match them.
    let failures: Vec<_> = first
        .cases
        .iter()
        .filter(|case| !case.passed)
        .map(|case| case.id.as_str())
        .collect();
    assert_eq!(failures, Vec::<&str>::new());
    assert!(first.codecs.iter().all(|codec| codec.passed));
    assert!(first.passed);
    assert_eq!(first, verify_runtime());
}

#[test]
fn concurrent_runs_agree() {
    let reports: Vec<_> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..3).map(|_| scope.spawn(verify_runtime)).collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    assert!(reports.windows(2).all(|pair| pair[0] == pair[1]));
}

fn case(id: &str) -> VerificationCase {
    builtin::static_cases()
        .into_iter()
        .find(|case| case.id == id)
        .unwrap()
}

fn failed_checks(result: &VerificationCaseResult) -> Vec<&str> {
    result
        .checks
        .iter()
        .filter(|check| !check.passed)
        .map(|check| check.name.as_str())
        .collect()
}

#[test]
fn wrong_pixels_and_tags_fail() {
    let mut case = case("builtin/static-explicit");
    case.frames[0].values[0] += 1;
    case.tags[0].values[0] = "WRONG".into();
    let result = run_case(&case);
    assert!(!result.passed);
    assert_eq!(failed_checks(&result), ["tags", "frame_0_pixels"]);
}

#[test]
fn wrong_layout_and_frame_count_fail() {
    let mut case = case("builtin/static-multiframe-native");
    case.frames.pop();
    case.frames[0].sample_type = SampleType::I8;
    let result = run_case(&case);
    assert!(failed_checks(&result).contains(&"frame_count"));
    assert!(failed_checks(&result).contains(&"frame_0_layout"));
}

#[test]
fn malformed_bytes_fail() {
    let mut case = case("builtin/static-explicit");
    case.dicom_bytes = b"not a DICOM file";
    let result = run_case(&case);
    assert!(!result.passed);
    assert_eq!(failed_checks(&result), ["load"]);
}

#[test]
fn every_embedded_fixture_has_a_valid_case() {
    let manifest = builtin::manifest();
    assert_eq!(manifest.schema_version, 1);
    let mut files: Vec<_> = manifest
        .cases
        .iter()
        .map(|case| case.file.as_str())
        .collect();
    files.sort_unstable();
    let mut embedded: Vec<_> = FIXTURES.iter().map(|(name, _)| *name).collect();
    embedded.sort_unstable();
    assert_eq!(files, embedded, "manifest cases and embedded files differ");
    let mut ids: Vec<_> = manifest.cases.iter().map(|case| case.id.as_str()).collect();
    ids.dedup();
    assert_eq!(ids.len(), manifest.cases.len(), "duplicate case IDs");
    for name in manifest
        .cases
        .iter()
        .flat_map(|case| &case.frames)
        .map(|f| &f.pixels)
    {
        assert!(manifest.pixels.contains_key(name), "unknown pixels {name}");
    }
    for case in builtin::static_cases() {
        for frame in &case.frames {
            let length = frame.width * frame.height * u32::from(frame.samples_per_pixel);
            assert_eq!(frame.values.len(), length as usize, "{}", case.id);
        }
    }
}

#[test]
fn a_decoder_without_a_fixture_fails_coverage() {
    let report = verify_runtime();
    let mut decodable: Vec<_> = report
        .codecs
        .iter()
        .map(|codec| codec.transfer_syntax_uid.clone())
        .collect();
    decodable.push("1.2.3.4".into());
    let coverage = builtin::coverage(&report.cases, decodable);
    let missing = coverage
        .iter()
        .find(|codec| codec.transfer_syntax_uid == "1.2.3.4")
        .unwrap();
    assert!(!missing.passed);
    assert!(missing.required_cases.is_empty());
}

#[test]
fn embedded_corpus_stays_within_budget() {
    let embedded = builtin::embedded_len();
    assert!(
        embedded <= EMBEDDED_BUDGET,
        "embedded verification data is {embedded} bytes, over the {EMBEDDED_BUDGET}-byte budget"
    );
}
