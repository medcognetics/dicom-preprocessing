use super::*;

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
    assert!(first.passed);
    assert!(first.codecs.iter().all(|codec| codec.passed));
    assert_eq!(first, verify_runtime());
}

#[test]
fn bad_pixels_and_tags_fail() {
    let mut case = builtin::static_cases().remove(0);
    case.id = "example/bad".into();
    case.frames[0].values[0] += 1;
    case.tags[0].values[0] = "WRONG".into();
    let result = run_case(&case, "extension_fixture");
    assert!(!result.passed);
    assert!(result
        .checks
        .iter()
        .any(|check| check.name == "tags" && !check.passed));
    assert!(result
        .checks
        .iter()
        .any(|check| check.name == "frame_0_pixels" && !check.passed));
}

fn extension() -> VerificationCase {
    let mut case = builtin::static_cases().remove(0);
    case.id = "example/fixture".into();
    case
}

struct Test<'a> {
    id: &'a str,
    calls: &'a std::cell::Cell<usize>,
    outcome: Result<Vec<VerificationCheck>, String>,
}
impl VerificationTest for Test<'_> {
    fn id(&self) -> &str {
        self.id
    }
    fn run(&self) -> Result<Vec<VerificationCheck>, String> {
        self.calls.set(self.calls.get() + 1);
        self.outcome.clone()
    }
}

#[test]
fn invalid_configuration_precedes_callbacks() {
    let calls = std::cell::Cell::new(0);
    let test = Test {
        id: "example/test",
        calls: &calls,
        outcome: Ok(vec![VerificationCheck::new("check", true)]),
    };
    for id in ["builtin/override", "no_namespace", "example/test"] {
        let mut case = extension();
        case.id = id.into();
        let options = VerificationOptions {
            cases: vec![case],
            tests: vec![&test],
            ..Default::default()
        };
        assert!(verify_runtime_with(&options).is_err());
    }
    assert_eq!(calls.get(), 0);
}

#[test]
fn malformed_expectations_are_configuration_errors() {
    let mut bad = Vec::new();
    let mut case = extension();
    case.frames[0].values.pop();
    bad.push(case);
    let mut case = extension();
    case.frames[0].width = 0;
    bad.push(case);
    let mut case = extension();
    case.frames[0].values[0] = -1;
    bad.push(case);
    let mut case = extension();
    case.frames[0].absolute_tolerance = u32::MAX;
    bad.push(case);
    let mut case = extension();
    case.tags[0].path.clear();
    bad.push(case);
    let mut case = extension();
    case.tags[0].vr = "invalid".into();
    bad.push(case);
    let mut case = extension();
    case.tags[0].path[0].item = Some(0);
    bad.push(case);
    for case in bad {
        assert!(PreparedVerification::new(vec![case], vec![], vec![]).is_err());
    }
}

#[test]
fn wrong_frames_and_malformed_input_fail() {
    let mut case = builtin::generated("builtin/generated-native", "1.2.840.10008.1.2.1").unwrap();
    case.frames.swap(0, 1);
    assert!(!run_case(&case, "extension_fixture").passed);
    case.dicom_bytes = b"not a DICOM".to_vec();
    assert!(!run_case(&case, "extension_fixture").passed);
    assert!(builtin::generated("builtin/generated-jpeg", "1.2.3").is_err());
}

#[test]
fn custom_results_are_bounded_and_fail_closed() {
    let suite = PreparedVerification::new(
        vec![],
        vec![],
        vec![
            "app/empty".into(),
            "app/error".into(),
            "app/long".into(),
            "app/not-called".into(),
        ],
    )
    .unwrap();
    let mut run = suite.run_shared();
    run.record_custom(Ok(vec![]));
    run.record_custom(Err("sensitive path must not appear".into()));
    run.record_custom(Ok(vec![VerificationCheck::failure(
        "check",
        "x".repeat(1000),
    )]));
    let report = run.finish();
    assert!(!report.passed);
    let custom: Vec<_> = report
        .cases
        .iter()
        .filter(|case| case.path == VerificationPath::CallerIntegration)
        .collect();
    assert_eq!(custom.len(), 4);
    assert!(custom.iter().all(|case| !case.passed));
    assert_eq!(custom[2].checks[0].diagnostic.as_ref().unwrap().len(), 256);
    assert!(!serde_json::to_string(&report)
        .unwrap()
        .contains("sensitive"));
}

#[test]
fn coverage_distinguishes_shared_library_from_custom_integration() {
    let fixture = extension();
    let codecs = vec![CodecDeclaration {
        id: "app/codecs".into(),
        required_cases: BTreeMap::from([
            (
                fixture.transfer_syntax_uid.clone(),
                vec![fixture.id.clone()],
            ),
            ("1.2.3".into(), vec!["app/custom".into()]),
        ]),
    }];
    let mut run = PreparedVerification::new(vec![fixture], codecs, vec!["app/custom".into()])
        .unwrap()
        .run_shared();
    run.record_custom(Ok(vec![VerificationCheck::new("decode", true)]));
    let report = run.finish();
    let coverage: Vec<_> = report
        .codecs
        .iter()
        .filter(|codec| codec.id == "app/codecs")
        .collect();
    assert!(coverage.iter().all(|codec| codec.passed));
    assert!(
        !coverage
            .iter()
            .find(|codec| codec.transfer_syntax_uid == "1.2.3")
            .unwrap()
            .shared_library_verified
    );
    assert!(
        coverage
            .iter()
            .find(|codec| codec.transfer_syntax_uid != "1.2.3")
            .unwrap()
            .shared_library_verified
    );
}

#[test]
fn coverage_references_are_checked_before_execution() {
    for required in [
        vec![],
        vec!["app/missing".into()],
        vec!["builtin/static-explicit".into()],
    ] {
        let codec = CodecDeclaration {
            id: "app/codec".into(),
            required_cases: BTreeMap::from([("1.2.3".into(), required)]),
        };
        assert!(PreparedVerification::new(vec![], vec![codec], vec![]).is_err());
    }
}

#[test]
fn missing_builtin_coverage_fails() {
    let run = VerificationRun {
        cases: vec![],
        codecs: vec![CodecDeclaration {
            id: "builtin/missing".into(),
            required_cases: BTreeMap::from([("1.2.3".into(), vec![])]),
        }],
        test_ids: vec![],
        completed: 0,
    };
    assert!(!run.finish().passed);
}

#[test]
fn static_manifest_is_valid_and_complete() {
    let cases = builtin::static_cases();
    for case in &cases {
        validate_case(case).unwrap();
    }
    let ids = cases.iter().map(|case| &case.id).collect::<BTreeSet<_>>();
    assert_eq!(cases.len(), ids.len());
    let report = verify_runtime();
    assert!(report
        .codecs
        .iter()
        .all(|codec| !codec.required_cases.is_empty()));
}

#[test]
fn independent_concurrent_calls_agree() {
    let expected = verify_runtime();
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..3).map(|_| scope.spawn(verify_runtime)).collect();
        for handle in handles {
            assert_eq!(expected, handle.join().unwrap());
        }
    });
}

#[test]
fn custom_tests_run_once_in_order() {
    let calls = std::cell::Cell::new(0);
    let first = Test {
        id: "app/first",
        calls: &calls,
        outcome: Err("failed".into()),
    };
    let second = Test {
        id: "app/second",
        calls: &calls,
        outcome: Ok(vec![VerificationCheck::new("check", true)]),
    };
    let report = verify_runtime_with(&VerificationOptions {
        tests: vec![&first, &second],
        ..Default::default()
    })
    .unwrap();
    assert_eq!(calls.get(), 2);
    assert_eq!(report.cases[report.cases.len() - 2].id, "app/first");
    assert_eq!(report.cases.last().unwrap().id, "app/second");
    assert!(report.cases.last().unwrap().passed);
    assert!(!report.passed);
}
