import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast, get_type_hints

import pytest

import dicom_preprocessing as dp

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "verification"


def fixture_case() -> dp.VerificationCase:
    return {
        "id": "example/fixture",
        "dicom_bytes": (FIXTURES / "explicit.dcm").read_bytes(),
        "transfer_syntax_uid": "1.2.840.10008.1.2.1",
        "tags": [{"path": [{"group": 0x28, "element": 0x10}], "vr": "US", "values": ["8"]}],
        "frames": [
            {
                "width": 8,
                "height": 8,
                "samples_per_pixel": 1,
                "planar_configuration": 0,
                "sample_type": "u8",
                "values": [i * 3 for i in range(64)],
            }
        ],
    }


def test_default_report_is_repeatable_and_offline(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    first = dp.verify_runtime()
    assert first == dp.verify_runtime()
    assert first["schema_version"] == 1
    assert first["library_version"] == dp.__version__
    assert len(first["cases"]) >= 25
    assert all(codec["required_cases"] for codec in first["codecs"])
    assert first["passed"] == all(case["passed"] for case in first["cases"])
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert all(report == first for report in pool.map(lambda _: dp.verify_runtime(), range(2)))
    assert list(tmp_path.iterdir()) == []


def test_fixture_and_callback_coverage():
    calls = []

    def custom() -> list[dp.VerificationCheck]:
        calls.append(1)
        return [{"name": "decode", "passed": True}]

    case = fixture_case()
    report = dp.verify_runtime(
        cases=[case],
        tests=[{"id": "example/custom", "run": custom}],
        codecs=[
            {
                "id": "example/codecs",
                "required_cases": {case["transfer_syntax_uid"]: [case["id"]], "1.2.3": ["example/custom"]},
            }
        ],
    )
    assert calls == [1]
    assert all(case["passed"] for case in report["cases"][-2:])
    coverage = {codec["transfer_syntax_uid"]: codec for codec in report["codecs"] if codec["id"] == "example/codecs"}
    assert coverage[case["transfer_syntax_uid"]]["shared_library_verified"]
    assert coverage["1.2.3"]["passed"]
    assert not coverage["1.2.3"]["shared_library_verified"]


@pytest.mark.parametrize("id", ["builtin/override", "bad", "example/callback"])
def test_bad_ids_fail_before_callbacks(id):
    case = fixture_case()
    case["id"] = id
    calls = []
    with pytest.raises(ValueError):
        dp.verify_runtime(cases=[case], tests=[{"id": "example/callback", "run": cast(Any, lambda: calls.append(1))}])
    assert calls == []


def test_bad_expectations_and_references():
    case = fixture_case()
    case["frames"][0]["values"].pop()
    with pytest.raises(ValueError):
        dp.verify_runtime(cases=[case])
    with pytest.raises(ValueError):
        dp.verify_runtime(codecs=[{"id": "example/codec", "required_cases": {"1.2.3": ["example/missing"]}}])


def test_corrupt_bytes_and_wrong_pixels_fail():
    for corrupt in (False, True):
        case = fixture_case()
        if corrupt:
            case["dicom_bytes"] = b"invalid DICOM"
        else:
            case["frames"][0]["values"][0] = 1
        report = dp.verify_runtime(cases=[case])
        assert not report["passed"]
        assert not report["cases"][-1]["passed"]


def test_callbacks_fail_closed_and_continue():
    calls = []

    def raises():
        calls.append("error")
        raise RuntimeError("sensitive detail")

    def passes() -> list[dp.VerificationCheck]:
        calls.append("pass")
        return [{"name": "check", "passed": True}]

    report = dp.verify_runtime(
        tests=[
            {"id": "example/empty", "run": lambda: []},
            {"id": "example/error", "run": raises},
            {"id": "example/pass", "run": passes},
            {"id": "example/malformed", "run": cast(Any, lambda: [{"name": "x", "passed": "yes"}])},
        ]
    )
    assert calls == ["error", "pass"]
    assert [case["passed"] for case in report["cases"][-4:]] == [False, False, True, False]
    assert "sensitive detail" not in str(report)


@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit])
def test_interruption_propagates(exception):
    def interrupt():
        raise exception()

    with pytest.raises(exception):
        dp.verify_runtime(tests=[{"id": "example/interrupt", "run": interrupt}])


def test_temporary_storage_failure_is_reported_and_cleanup_is_complete(tmp_path):
    program = "import json, dicom_preprocessing as dp; print(json.dumps(dp.verify_runtime()))"
    for temporary_root in (tmp_path, tmp_path / "missing"):
        result = subprocess.run(
            [sys.executable, "-c", program],
            check=True,
            text=True,
            capture_output=True,
            cwd=tmp_path,
            env={**os.environ, "TMPDIR": str(temporary_root)},
        )
        report = json.loads(result.stdout)
        if temporary_root != tmp_path:
            assert not report["passed"]
            assert report["cases"][0]["checks"][0]["diagnostic"] == "temporary storage unavailable"
            assert str(temporary_root) not in result.stdout
        assert list(tmp_path.iterdir()) == []


def test_runtime_typing_exports_exist():
    # Evaluate annotations eagerly, as Python 3.10 does at function definition.
    assert get_type_hints(fixture_case)["return"] is dp.VerificationCase
    assert "dicom_bytes" in get_type_hints(dp.VerificationCase)
    assert "run" in get_type_hints(dp.VerificationCustomTest)
    assert dp.VerificationCheck(name="check", passed=True) == {"name": "check", "passed": True}
