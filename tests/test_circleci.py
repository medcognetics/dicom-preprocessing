import json
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[1]
CIRCLECI_CONFIG_PATH = REPOSITORY_ROOT / ".circleci" / "config.yml"
NODE_PACKAGE_PATH = REPOSITORY_ROOT / "bindings" / "node" / "package.json"
QUALITY_JOBS = {"rust_quality", "python_quality", "node_quality"}
WORKFLOW_JOBS = (
    "rust_quality",
    "python_quality",
    "rust_tests",
    "python_tests",
    "node_quality",
    "node_tests",
    "windows_node_tests",
)
VERSION_TAG_FILTER = "/^v[0-9]+\\.[0-9]+\\.[0-9]+$/"


def job_definition(config: str, job_name: str) -> str:
    lines = config.splitlines()
    job_marker = f"    {job_name}:"

    for job_index, line in enumerate(lines):
        if line != job_marker:
            continue

        job_lines = [line]
        for job_line in lines[job_index + 1 :]:
            job_indent = len(job_line) - len(job_line.lstrip())
            if job_line.strip() and job_indent <= 4:
                break
            job_lines.append(job_line)
        return "\n".join(job_lines)

    raise AssertionError(f"Job definition not found: {job_name}")


def workflow_job_definition(config: str, job_name: str) -> str:
    lines = config.splitlines()
    job_markers = {f"            - {job_name}", f"            - {job_name}:"}

    for job_index, line in enumerate(lines):
        if line not in job_markers:
            continue

        job_indent = len(line) - len(line.lstrip())
        job_lines = [line]
        for job_line in lines[job_index + 1 :]:
            definition_indent = len(job_line) - len(job_line.lstrip())
            if job_line.strip() and definition_indent <= job_indent:
                break
            job_lines.append(job_line)
        return "\n".join(job_lines)

    raise AssertionError(f"Workflow job not found: {job_name}")


def workflow_job_requirements(config: str, job_name: str) -> set[str]:
    lines = workflow_job_definition(config, job_name).splitlines()

    for requires_index, line in enumerate(lines):
        if line.strip() != "requires:":
            continue

        requires_indent = len(line) - len(line.lstrip())
        requirements = set()
        for dependency_line in lines[requires_index + 1 :]:
            dependency_indent = len(dependency_line) - len(dependency_line.lstrip())
            if dependency_line.strip() and dependency_indent <= requires_indent:
                break
            if dependency_line.strip().startswith("- "):
                requirements.add(dependency_line.strip().removeprefix("- "))
        return requirements

    return set()


def test_windows_node_tests_require_quality_jobs() -> None:
    config = CIRCLECI_CONFIG_PATH.read_text()

    assert workflow_job_requirements(config, "windows_node_tests") == QUALITY_JOBS


def test_node_validation_uses_debug_builds() -> None:
    package = json.loads(NODE_PACKAGE_PATH.read_text())

    assert package["scripts"]["typecheck"] == "tsc --noEmit"
    assert package["scripts"]["test"] == "npm run build:debug && npm run typecheck && node --test test/*.test.mjs"


def test_node_quality_does_not_install_rust() -> None:
    config = CIRCLECI_CONFIG_PATH.read_text()

    assert "Install Rust" not in job_definition(config, "node_quality")


def test_windows_build_mode_depends_on_exact_version_tag() -> None:
    config = CIRCLECI_CONFIG_PATH.read_text()
    windows_job = job_definition(config, "windows_node_tests")

    assert "cargo test -p dicom-preprocessing --lib file::tests::test_inode_sort" in windows_job
    assert f"$env:CIRCLE_TAG -match '{VERSION_TAG_FILTER[1:-1]}'" in windows_job
    assert '"build"' in windows_job
    assert '"build:debug"' in windows_job
    assert "name: Install Node dependencies" in windows_job
    assert "name: Test Windows file identifiers" in windows_job
    assert "name: Build Node bindings" in windows_job


def test_all_workflow_jobs_accept_exact_version_tags() -> None:
    config = CIRCLECI_CONFIG_PATH.read_text()

    assert VERSION_TAG_FILTER in config
    for job_name in WORKFLOW_JOBS:
        filter_alias = "&version_tags" if job_name == "rust_quality" else "*version_tags"
        assert f"filters: {filter_alias}" in workflow_job_definition(config, job_name)


def test_ci_avoids_cross_executor_build_artifacts() -> None:
    config = CIRCLECI_CONFIG_PATH.read_text()

    assert "persist_to_workspace" not in config
    assert "attach_workspace" not in config
    assert "v3-rust-tests" not in config
    assert "~/.rustup" not in config
    assert "bindings/node/node_modules" not in config

    assert "v4-rust-deps-" in job_definition(config, "rust_tests")
    assert "v4-python-rust-deps-" in job_definition(config, "python_tests")
    assert "v4-node-rust-deps-" in job_definition(config, "node_tests")
    assert "v4-windows-rust-deps-" in job_definition(config, "windows_node_tests")

    for job_name in ("rust_tests", "python_tests", "node_tests", "windows_node_tests"):
        assert "./target" not in job_definition(config, job_name)
