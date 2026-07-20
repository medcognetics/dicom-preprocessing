#!/usr/bin/env python3
"""Run reproducible security and deprecation checks for repository dependencies."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CARGO_AUDIT_VERSION = "0.22.1"
UV_VERSION = "0.11.18"
ZIZMOR_VERSION = "1.27.0"
PYTHON_VERSIONS = ("3.10", "3.14")
ZIZMOR_FINDING_EXIT_CODES = {11, 12, 13, 14}
RUNTIME_STATUS = (
    "Python 3.10 receives security fixes through October 2026; choose a new minimum before EOL.",
    "Node 22 is supported through 2027-04-30, Node 24 through 2028-04-30, and Node 26 through 2029-04-30.",
    "Rust 1.89.0 is the declared minimum supported Rust version; CI also tests the current pinned toolchain.",
)


class AuditIncomplete(RuntimeError):
    """Raised when a scanner cannot produce an interpretable complete report."""


@dataclass(frozen=True)
class CommandResult:
    """Captured scanner process result."""

    name: str
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class AuditReport:
    """Normalized scanner findings."""

    security_findings: tuple[str, ...] = ()
    deprecations: tuple[str, ...] = ()
    metadata: dict[str, str | int] = field(default_factory=dict)


def parse_json(name: str, output: str) -> Any:
    """Parse scanner JSON or classify the scan as incomplete."""

    try:
        return json.loads(output)
    except json.JSONDecodeError as error:
        raise AuditIncomplete(f"{name} returned invalid JSON: {error}") from error


def advisory_finding(item: dict[str, Any], fallback: str) -> str:
    """Format a Rust advisory entry as a stable, actionable identifier."""

    advisory = item.get("advisory") or {}
    package = item.get("package") or {}
    identifier = advisory.get("id") or fallback
    package_name = package.get("name") or advisory.get("package") or "unknown-package"
    version = package.get("version") or "unknown-version"
    return f"{identifier}: {package_name} {version}"


def parse_cargo_audit(output: str) -> AuditReport:
    """Normalize Cargo Audit vulnerabilities and informational warnings."""

    payload = parse_json("cargo-audit", output)
    if not isinstance(payload, dict):
        raise AuditIncomplete("cargo-audit returned a non-object JSON report")

    vulnerabilities = payload.get("vulnerabilities") or {}
    warnings = payload.get("warnings") or {}
    if not isinstance(vulnerabilities, dict) or not isinstance(warnings, dict):
        raise AuditIncomplete("cargo-audit report has invalid findings sections")

    security_items = list(vulnerabilities.get("list") or []) + list(warnings.get("unsound") or [])
    security_findings = tuple(advisory_finding(item, "RustSec advisory") for item in security_items)

    deprecations: list[str] = []
    for item in warnings.get("unmaintained") or []:
        deprecations.append(advisory_finding(item, "unmaintained"))
    for item in warnings.get("yanked") or []:
        package = item.get("package") or item
        deprecations.append(
            f"yanked: {package.get('name', 'unknown-package')} {package.get('version', 'unknown-version')}"
        )

    database = payload.get("database") or {}
    lockfile = payload.get("lockfile") or {}
    metadata: dict[str, str | int] = {}
    if isinstance(database, dict) and database.get("last-commit"):
        metadata["database_revision"] = str(database["last-commit"])
    if isinstance(lockfile, dict) and isinstance(lockfile.get("dependency-count"), int):
        metadata["dependency_count"] = lockfile["dependency-count"]

    return AuditReport(security_findings, tuple(deprecations), metadata)


def parse_uv_audit(output: str) -> AuditReport:
    """Normalize uv vulnerability and adverse-status reports."""

    payload = parse_json("uv audit", output)
    if not isinstance(payload, dict):
        raise AuditIncomplete("uv audit returned a non-object JSON report")

    security_findings = tuple(
        f"{item.get('id', 'Python advisory')}: "
        f"{(item.get('dependency') or {}).get('name', 'unknown-package')} "
        f"{(item.get('dependency') or {}).get('version', 'unknown-version')}"
        for item in payload.get("vulnerabilities") or []
    )
    deprecations = tuple(
        f"{item.get('id') or item.get('status') or 'adverse'}: "
        f"{(item.get('dependency') or {}).get('name', 'unknown-package')} "
        f"{(item.get('dependency') or {}).get('version', 'unknown-version')}"
        for item in payload.get("adverse_statuses") or []
    )
    summary = payload.get("summary") or {}
    metadata = {
        "audited_packages": int(summary.get("audited_packages", 0)),
        "adverse_statuses": int(summary.get("adverse_statuses", len(deprecations))),
    }
    return AuditReport(security_findings, deprecations, metadata)


def parse_npm_audit(output: str) -> AuditReport:
    """Normalize npm audit findings at the configured threshold."""

    payload = parse_json("npm audit", output)
    if not isinstance(payload, dict):
        raise AuditIncomplete("npm audit returned a non-object JSON report")
    vulnerabilities = payload.get("vulnerabilities") or {}
    if not isinstance(vulnerabilities, dict):
        raise AuditIncomplete("npm audit report has an invalid vulnerabilities section")
    findings = tuple(
        f"{package}: {(details or {}).get('severity', 'unknown severity')}"
        for package, details in sorted(vulnerabilities.items())
    )
    return AuditReport(security_findings=findings)


def parse_npm_list(output: str) -> AuditReport:
    """Collect npm deprecation fields from the full installed dependency graph."""

    payload = parse_json("npm ls", output)
    if not isinstance(payload, dict):
        raise AuditIncomplete("npm ls returned a non-object JSON report")

    deprecations: set[str] = set()

    def visit(dependencies: object) -> None:
        if not isinstance(dependencies, dict):
            return
        for package, details in dependencies.items():
            if not isinstance(details, dict):
                continue
            if details.get("deprecated"):
                deprecations.add(f"{package} {details.get('version', 'unknown-version')}: {details['deprecated']}")
            visit(details.get("dependencies"))

    visit(payload.get("dependencies"))
    return AuditReport(deprecations=tuple(sorted(deprecations)))


def parse_zizmor(output: str) -> AuditReport:
    """Normalize unignored zizmor workflow-security findings."""

    payload = parse_json("zizmor", output)
    if not isinstance(payload, list):
        raise AuditIncomplete("zizmor returned a non-list JSON report")
    findings = tuple(str(item.get("ident", "zizmor finding")) for item in payload if not item.get("ignored", False))
    return AuditReport(security_findings=findings)


def validate_command_result(
    result: CommandResult,
    finding_exit_codes: set[int] | None = None,
    finding_count: int | None = None,
) -> None:
    """Separate scanner finding exits from incomplete scanner execution."""

    finding_exit_codes = finding_exit_codes or set()
    if result.returncode == 0:
        return
    if result.returncode not in finding_exit_codes:
        detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic output"
        raise AuditIncomplete(f"{result.name} failed with exit code {result.returncode}: {detail}")
    if finding_count == 0:
        raise AuditIncomplete(f"{result.name} reported findings but none were parsed")


def run_command(name: str, command: tuple[str, ...]) -> CommandResult:
    """Run a scanner without allowing shell interpretation."""

    try:
        process = subprocess.run(command, check=False, capture_output=True, text=True)
    except OSError as error:
        raise AuditIncomplete(f"{name} could not start: {error}") from error
    return CommandResult(name, command, process.returncode, process.stdout, process.stderr)


def command_version(name: str, command: tuple[str, ...]) -> str:
    """Return a scanner version or mark the report incomplete."""

    result = run_command(name, command)
    validate_command_result(result)
    output = result.stdout.strip() or result.stderr.strip()
    if not output:
        raise AuditIncomplete(f"{name} returned an empty version")
    return output.splitlines()[0]


def uv_audit_command(python_version: str) -> tuple[str, ...]:
    """Construct the pinned, JSON-producing uv audit command."""

    return (
        "uv",
        "audit",
        "--locked",
        "--python-version",
        python_version,
        "--output-format",
        "json",
        "--preview-features",
        "audit",
        "--preview-features",
        "json-output",
    )


def scanner_versions() -> dict[str, str]:
    """Capture the exact scanner versions used by the report."""

    return {
        "Cargo Audit": command_version("cargo-audit version", ("cargo", "audit", "--version")),
        "uv": command_version("uv version", ("uv", "--version")),
        "npm": command_version("npm version", ("npm", "--version")),
        "Rust": command_version("rustc version", ("rustc", "--version")),
        "zizmor": command_version(
            "zizmor version", ("uvx", "--from", f"zizmor=={ZIZMOR_VERSION}", "zizmor", "--version")
        ),
    }


def format_findings(findings: list[str]) -> str:
    """Render findings for a GitHub job summary."""

    if not findings:
        return "- None\n"
    return "".join(f"- {finding}\n" for finding in findings)


def provenance_table(versions: dict[str, str], commands: list[CommandResult]) -> str:
    """Render scanner provenance and audited inputs."""

    inputs = {
        "cargo audit": "Cargo.lock; all locked workspace dependencies",
        "uv audit 3.10": "uv.lock; build and default development groups for Python 3.10",
        "uv audit 3.14": "uv.lock; build and default development groups for Python 3.14",
        "npm audit": "package-lock.json; production, development, optional, and peer dependencies",
        "npm ls": "installed package-lock.json dependency graph",
        "zizmor": ".github/workflows; ignored findings remain visible to the parser",
        "cargo future incompatibility": "Cargo.lock and all workspace features",
    }
    version_lookup = {
        "cargo audit": versions["Cargo Audit"],
        "uv audit 3.10": versions["uv"],
        "uv audit 3.14": versions["uv"],
        "npm audit": versions["npm"],
        "npm ls": versions["npm"],
        "zizmor": versions["zizmor"],
        "cargo future incompatibility": versions["Rust"],
    }
    rows = ["| Scanner | Version | Command | Audited input |\n", "| --- | --- | --- | --- |\n"]
    for result in commands:
        command = shlex.join(result.command).replace("|", "\\|")
        rows.append(f"| {result.name} | {version_lookup[result.name]} | `{command}` | {inputs[result.name]} |\n")
    return "".join(rows)


def write_summary(contents: str) -> None:
    """Write to the Actions job summary, or stdout during local validation."""

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with Path(summary_path).open("a", encoding="utf-8") as summary:
            summary.write(contents)
    else:
        print(contents)


def run_security() -> int:
    """Run security scanners and fail on any unignored finding."""

    versions = scanner_versions()
    commands: list[CommandResult] = []
    findings: list[str] = []

    cargo_result = run_command("cargo audit", ("cargo", "audit", "--json"))
    commands.append(cargo_result)
    cargo_report = parse_cargo_audit(cargo_result.stdout)
    validate_command_result(cargo_result, {1}, len(cargo_report.security_findings))
    findings.extend(cargo_report.security_findings)

    for python_version in PYTHON_VERSIONS:
        uv_result = run_command(f"uv audit {python_version}", uv_audit_command(python_version))
        commands.append(uv_result)
        uv_report = parse_uv_audit(uv_result.stdout)
        validate_command_result(uv_result, {1}, len(uv_report.security_findings) + len(uv_report.deprecations))
        findings.extend(uv_report.security_findings)

    npm_result = run_command("npm audit", ("npm", "audit", "--audit-level=low", "--json"))
    commands.append(npm_result)
    npm_report = parse_npm_audit(npm_result.stdout)
    validate_command_result(npm_result, {1}, len(npm_report.security_findings))
    findings.extend(npm_report.security_findings)

    zizmor_result = run_command(
        "zizmor",
        (
            "uvx",
            "--from",
            f"zizmor=={ZIZMOR_VERSION}",
            "zizmor",
            "--persona",
            "auditor",
            "--strict-collection",
            "--format",
            "json-v1",
            ".github/workflows",
        ),
    )
    commands.append(zizmor_result)
    zizmor_report = parse_zizmor(zizmor_result.stdout)
    validate_command_result(zizmor_result, ZIZMOR_FINDING_EXIT_CODES, len(zizmor_report.security_findings))
    findings.extend(zizmor_report.security_findings)

    database_revision = cargo_report.metadata.get("database_revision", "unknown")
    summary = (
        "# Dependency security audit\n\n"
        f"- Check date: {datetime.now(timezone.utc).date().isoformat()} UTC\n"
        f"- RustSec advisory database revision: `{database_revision}`\n"
        "- Exit policy: scanner/tool/network/parse errors are incomplete; unignored findings fail this job.\n\n"
        "## Scanner provenance\n\n"
        f"{provenance_table(versions, commands)}\n"
        "## Unsuppressed findings\n\n"
        f"{format_findings(findings)}"
    )
    write_summary(summary)
    return 1 if findings else 0


def future_incompatibility_findings(result: CommandResult) -> tuple[str, ...]:
    """Extract compiler future-incompatibility warnings without treating normal output as findings."""

    output = f"{result.stdout}\n{result.stderr}".lower()
    if "future-incompat" not in output or "0 dependencies had future-incompatible warnings" in output:
        return ()
    return ("Cargo reported future-incompatible dependency warnings; inspect the command output.",)


def run_deprecation() -> int:
    """Report deprecations while failing only when a report is incomplete."""

    versions = scanner_versions()
    commands: list[CommandResult] = []
    findings: list[str] = []

    cargo_result = run_command("cargo audit", ("cargo", "audit", "--json"))
    commands.append(cargo_result)
    cargo_report = parse_cargo_audit(cargo_result.stdout)
    validate_command_result(
        cargo_result,
        {1},
        len(cargo_report.security_findings) + len(cargo_report.deprecations),
    )
    findings.extend(cargo_report.deprecations)

    future_result = run_command(
        "cargo future incompatibility",
        ("cargo", "check", "--locked", "--workspace", "--all-features", "--future-incompat-report"),
    )
    commands.append(future_result)
    validate_command_result(future_result)
    findings.extend(future_incompatibility_findings(future_result))

    for python_version in PYTHON_VERSIONS:
        uv_result = run_command(f"uv audit {python_version}", uv_audit_command(python_version))
        commands.append(uv_result)
        uv_report = parse_uv_audit(uv_result.stdout)
        validate_command_result(uv_result, {1}, len(uv_report.security_findings) + len(uv_report.deprecations))
        findings.extend(uv_report.deprecations)

    npm_result = run_command("npm ls", ("npm", "ls", "--all", "--json", "--long"))
    commands.append(npm_result)
    npm_report = parse_npm_list(npm_result.stdout)
    validate_command_result(npm_result)
    findings.extend(npm_report.deprecations)

    database_revision = cargo_report.metadata.get("database_revision", "unknown")
    summary = (
        "# Dependency deprecation report\n\n"
        f"- Check date: {datetime.now(timezone.utc).date().isoformat()} UTC\n"
        f"- RustSec advisory database revision: `{database_revision}`\n"
        "- Exit policy: findings are report-only; command, network, and parse failures fail this job.\n\n"
        "## Scanner provenance\n\n"
        f"{provenance_table(versions, commands)}\n"
        "## Deprecation and compatibility findings\n\n"
        f"{format_findings(findings)}\n"
        "## Runtime support status\n\n"
        f"{format_findings(list(RUNTIME_STATUS))}"
    )
    write_summary(summary)
    return 0


def parse_args() -> argparse.Namespace:
    """Parse the report mode."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("security", "deprecation"))
    return parser.parse_args()


def main() -> int:
    """Run the selected report and preserve incomplete-scan failures."""

    args = parse_args()
    try:
        if args.mode == "security":
            return run_security()
        return run_deprecation()
    except AuditIncomplete as error:
        write_summary(f"# Dependency health incomplete\n\n- {error}\n")
        print(error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
