import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPOSITORY_ROOT = Path(__file__).parents[1]
DEPENDENCY_HEALTH_PATH = REPOSITORY_ROOT / "scripts" / "ci" / "dependency_health.py"
SCANNER_VERSIONS = {
    "Cargo Audit": "cargo-audit 0.22.1",
    "uv": "uv 0.11.18",
    "npm": "11.0.0",
    "Rust": "rustc 1.97.1",
    "zizmor": "zizmor 1.27.0",
}


def load_dependency_health() -> ModuleType:
    spec = importlib.util.spec_from_file_location("dependency_health", DEPENDENCY_HEALTH_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_clean_security_reports_have_no_findings() -> None:
    dependency_health = load_dependency_health()

    cargo = dependency_health.parse_cargo_audit(
        json.dumps({"database": {"last-commit": "abc"}, "vulnerabilities": {"list": []}, "warnings": {}})
    )
    uv = dependency_health.parse_uv_audit(
        json.dumps({"summary": {"audited_packages": 4}, "vulnerabilities": [], "adverse_statuses": []})
    )
    npm = dependency_health.parse_npm_audit(json.dumps({"vulnerabilities": {}}))
    zizmor = dependency_health.parse_zizmor("[]")

    assert cargo.security_findings == ()
    assert uv.security_findings == ()
    assert npm.security_findings == ()
    assert zizmor.security_findings == ()
    assert cargo.metadata["database_revision"] == "abc"


def test_clean_security_audit_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    dependency_health = load_dependency_health()
    results = {
        "cargo audit": dependency_health.CommandResult(
            "cargo audit",
            ("cargo", "audit"),
            0,
            json.dumps(
                {
                    "database": {"last-commit": "abc"},
                    "vulnerabilities": {"list": []},
                    "warnings": {},
                }
            ),
            "",
        ),
        "uv audit 3.10": dependency_health.CommandResult(
            "uv audit 3.10",
            ("uv", "audit"),
            0,
            json.dumps({"summary": {}, "vulnerabilities": [], "adverse_statuses": []}),
            "",
        ),
        "uv audit 3.14": dependency_health.CommandResult(
            "uv audit 3.14",
            ("uv", "audit"),
            0,
            json.dumps({"summary": {}, "vulnerabilities": [], "adverse_statuses": []}),
            "",
        ),
        "npm audit": dependency_health.CommandResult(
            "npm audit", ("npm", "audit"), 0, json.dumps({"vulnerabilities": {}}), ""
        ),
        "zizmor": dependency_health.CommandResult("zizmor", ("zizmor",), 0, "[]", ""),
    }
    summaries: list[str] = []
    monkeypatch.setattr(dependency_health, "scanner_versions", lambda: SCANNER_VERSIONS)
    monkeypatch.setattr(dependency_health, "run_command", lambda name, _command: results[name])
    monkeypatch.setattr(dependency_health, "write_summary", summaries.append)

    assert dependency_health.run_security() == 0
    assert "## Unsuppressed findings\n\n- None" in summaries[0]


def test_security_parsers_return_actionable_finding_identifiers() -> None:
    dependency_health = load_dependency_health()

    cargo = dependency_health.parse_cargo_audit(
        json.dumps(
            {
                "database": {},
                "vulnerabilities": {
                    "list": [{"advisory": {"id": "RUSTSEC-1"}, "package": {"name": "crate", "version": "1"}}]
                },
                "warnings": {
                    "unsound": [{"advisory": {"id": "RUSTSEC-2"}, "package": {"name": "unsafe", "version": "2"}}]
                },
            }
        )
    )
    uv = dependency_health.parse_uv_audit(
        json.dumps(
            {
                "summary": {},
                "vulnerabilities": [{"id": "PYSEC-1", "dependency": {"name": "pillow", "version": "1"}}],
                "adverse_statuses": [],
            }
        )
    )
    npm = dependency_health.parse_npm_audit(
        json.dumps({"vulnerabilities": {"pkg": {"severity": "high", "via": ["GHSA-1"]}}})
    )
    zizmor = dependency_health.parse_zizmor(json.dumps([{"ident": "unpinned-uses", "ignored": False}]))

    assert cargo.security_findings == ("RUSTSEC-1: crate 1", "RUSTSEC-2: unsafe 2")
    assert uv.security_findings == ("PYSEC-1: pillow 1",)
    assert npm.security_findings == ("pkg: high",)
    assert zizmor.security_findings == ("unpinned-uses",)


def test_security_audit_fails_on_findings(monkeypatch: pytest.MonkeyPatch) -> None:
    dependency_health = load_dependency_health()
    results = {
        "cargo audit": dependency_health.CommandResult(
            "cargo audit",
            ("cargo", "audit"),
            1,
            json.dumps(
                {
                    "database": {"last-commit": "abc"},
                    "vulnerabilities": {
                        "list": [
                            {
                                "advisory": {"id": "RUSTSEC-1"},
                                "package": {"name": "crate", "version": "1"},
                            }
                        ]
                    },
                    "warnings": {},
                }
            ),
            "",
        ),
        "uv audit 3.10": dependency_health.CommandResult(
            "uv audit 3.10",
            ("uv", "audit"),
            0,
            json.dumps({"summary": {}, "vulnerabilities": [], "adverse_statuses": []}),
            "",
        ),
        "uv audit 3.14": dependency_health.CommandResult(
            "uv audit 3.14",
            ("uv", "audit"),
            0,
            json.dumps({"summary": {}, "vulnerabilities": [], "adverse_statuses": []}),
            "",
        ),
        "npm audit": dependency_health.CommandResult(
            "npm audit", ("npm", "audit"), 0, json.dumps({"vulnerabilities": {}}), ""
        ),
        "zizmor": dependency_health.CommandResult("zizmor", ("zizmor",), 0, "[]", ""),
    }
    summaries: list[str] = []
    monkeypatch.setattr(dependency_health, "scanner_versions", lambda: SCANNER_VERSIONS)
    monkeypatch.setattr(dependency_health, "run_command", lambda name, _command: results[name])
    monkeypatch.setattr(dependency_health, "write_summary", summaries.append)

    assert dependency_health.run_security() == 1
    assert "RUSTSEC-1: crate 1" in summaries[0]


def test_deprecation_findings_are_reported_separately() -> None:
    dependency_health = load_dependency_health()

    cargo = dependency_health.parse_cargo_audit(
        json.dumps(
            {
                "database": {},
                "vulnerabilities": {"list": []},
                "warnings": {
                    "unmaintained": [{"advisory": {"id": "RUSTSEC-3"}, "package": {"name": "old", "version": "3"}}],
                    "yanked": [{"package": {"name": "gone", "version": "4"}}],
                },
            }
        )
    )
    uv = dependency_health.parse_uv_audit(
        json.dumps(
            {
                "summary": {},
                "vulnerabilities": [],
                "adverse_statuses": [{"id": "withdrawn", "dependency": {"name": "oldpy", "version": "5"}}],
            }
        )
    )
    npm = dependency_health.parse_npm_list(
        json.dumps({"dependencies": {"oldjs": {"version": "6", "deprecated": "use newjs"}}})
    )

    assert cargo.deprecations == ("RUSTSEC-3: old 3", "yanked: gone 4")
    assert uv.deprecations == ("withdrawn: oldpy 5",)
    assert npm.deprecations == ("oldjs 6: use newjs",)


def test_deprecation_audit_reports_findings_without_failing(monkeypatch: pytest.MonkeyPatch) -> None:
    dependency_health = load_dependency_health()
    results = {
        "cargo audit": dependency_health.CommandResult(
            "cargo audit",
            ("cargo", "audit"),
            1,
            json.dumps(
                {
                    "database": {"last-commit": "abc"},
                    "vulnerabilities": {"list": []},
                    "warnings": {
                        "unmaintained": [
                            {
                                "advisory": {"id": "RUSTSEC-3"},
                                "package": {"name": "encoding", "version": "0.2.33"},
                            }
                        ]
                    },
                }
            ),
            "",
        ),
        "cargo future incompatibility": dependency_health.CommandResult(
            "cargo future incompatibility",
            ("cargo", "check"),
            0,
            "0 dependencies had future-incompatible warnings",
            "",
        ),
        "uv audit 3.10": dependency_health.CommandResult(
            "uv audit 3.10",
            ("uv", "audit"),
            0,
            json.dumps({"summary": {}, "vulnerabilities": [], "adverse_statuses": []}),
            "",
        ),
        "uv audit 3.14": dependency_health.CommandResult(
            "uv audit 3.14",
            ("uv", "audit"),
            0,
            json.dumps({"summary": {}, "vulnerabilities": [], "adverse_statuses": []}),
            "",
        ),
        "npm ls": dependency_health.CommandResult(
            "npm ls",
            ("npm", "ls"),
            0,
            json.dumps({"dependencies": {"oldjs": {"version": "1", "deprecated": "replace it"}}}),
            "",
        ),
    }
    summaries: list[str] = []
    monkeypatch.setattr(dependency_health, "scanner_versions", lambda: SCANNER_VERSIONS)
    monkeypatch.setattr(dependency_health, "run_command", lambda name, _command: results[name])
    monkeypatch.setattr(dependency_health, "write_summary", summaries.append)

    assert dependency_health.run_deprecation() == 0
    assert "RustSec advisory database revision: `abc`" in summaries[0]
    assert "RUSTSEC-3: encoding 0.2.33" in summaries[0]
    assert "oldjs 1: replace it" in summaries[0]


def test_malformed_scanner_output_is_incomplete() -> None:
    dependency_health = load_dependency_health()

    with pytest.raises(dependency_health.AuditIncomplete, match="cargo-audit returned invalid JSON"):
        dependency_health.parse_cargo_audit("not-json")


def test_unexpected_tool_exit_is_incomplete() -> None:
    dependency_health = load_dependency_health()

    result = dependency_health.CommandResult(
        name="npm audit",
        command=("npm", "audit"),
        returncode=2,
        stdout="{}",
        stderr="network failure",
    )

    with pytest.raises(dependency_health.AuditIncomplete, match="npm audit failed with exit code 2"):
        dependency_health.validate_command_result(result, finding_exit_codes={1})


def test_missing_scanner_is_incomplete(monkeypatch: pytest.MonkeyPatch) -> None:
    dependency_health = load_dependency_health()

    def missing_scanner(*_args: object, **_kwargs: object) -> None:
        raise FileNotFoundError("scanner not found")

    monkeypatch.setattr(dependency_health.subprocess, "run", missing_scanner)

    with pytest.raises(dependency_health.AuditIncomplete, match="cargo-audit could not start: scanner not found"):
        dependency_health.run_command("cargo-audit", ("cargo", "audit"))


def test_finding_exit_without_parseable_findings_is_incomplete() -> None:
    dependency_health = load_dependency_health()

    result = dependency_health.CommandResult(
        name="uv audit",
        command=("uv", "audit"),
        returncode=1,
        stdout="{}",
        stderr="",
    )

    with pytest.raises(dependency_health.AuditIncomplete, match="reported findings but none were parsed"):
        dependency_health.validate_command_result(result, finding_exit_codes={1}, finding_count=0)
