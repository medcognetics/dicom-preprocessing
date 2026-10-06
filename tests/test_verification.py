import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import get_type_hints

import dicom_preprocessing as dp


def test_default_report_passes_and_is_repeatable(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    first = dp.verify_runtime()
    assert first == dp.verify_runtime()
    assert first["schema_version"] == 1
    assert first["library_version"] == dp.__version__
    assert len(first["cases"]) >= 25
    assert first["passed"], [case["id"] for case in first["cases"] if not case["passed"]]
    assert all(codec["passed"] and codec["required_cases"] for codec in first["codecs"])
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert all(report == first for report in pool.map(lambda _: dp.verify_runtime(), range(2)))
    assert list(tmp_path.iterdir()) == []


def test_report_runs_offline_from_an_installed_package(tmp_path):
    program = "import json, dicom_preprocessing as dp; print(json.dumps(dp.verify_runtime()['passed']))"
    environment = {**os.environ, "TMPDIR": str(tmp_path / "missing")}
    result = subprocess.run(
        [sys.executable, "-c", program], cwd=tmp_path, env=environment, capture_output=True, text=True, check=True
    )
    # The suite needs no temporary storage, so a missing TMPDIR does not matter.
    assert json.loads(result.stdout) is True


def test_runtime_typing_exports_exist():
    for name in ["VerificationCheck", "VerificationCaseResult", "VerificationCodecResult", "VerificationReport"]:
        assert name in getattr(dp, "__all__")
        assert get_type_hints(getattr(dp, name))
    assert set(get_type_hints(dp.VerificationReport)) == {
        "schema_version",
        "suite_version",
        "library_version",
        "passed",
        "cases",
        "codecs",
    }
