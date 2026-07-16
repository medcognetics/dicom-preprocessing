import json
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[1]
CIRCLECI_CONFIG_PATH = REPOSITORY_ROOT / ".circleci" / "config.yml"
NODE_PACKAGE_PATH = REPOSITORY_ROOT / "package.json"
QUALITY_JOBS = {"rust_quality", "python_quality", "node_quality"}
RUNTIME_JOBS = {
    "rust_tests",
    "python_tests",
    "node_tests",
    "windows_node_tests",
    "macos_arm64_node_tests",
    "macos_x64_node_tests",
}
WORKFLOW_JOBS = (
    "rust_quality",
    "python_quality",
    "rust_tests",
    "python_tests",
    "node_quality",
    "node_tests",
    "windows_node_tests",
    "macos_arm64_node_tests",
    "macos_x64_node_tests",
)
VERSION_TAG_FILTER = "/^v[0-9]+\\.[0-9]+\\.[0-9]+$/"
WINDOWS_INODE_TEST_COMMAND = "cargo test -p dicom-preprocessing --lib file::tests::test_inode_sort"
WINDOWS_GIT_INSTALL_NO_OUTPUT_TIMEOUT = "30m"


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


def test_runtime_jobs_require_quality_jobs() -> None:
    config = CIRCLECI_CONFIG_PATH.read_text()

    for job_name in RUNTIME_JOBS:
        assert workflow_job_requirements(config, job_name) == QUALITY_JOBS


def test_node_validation_uses_debug_builds() -> None:
    package = json.loads(NODE_PACKAGE_PATH.read_text())

    assert package["scripts"]["typecheck"] == "tsc --noEmit --project bindings/node/tsconfig.json"
    assert package["scripts"]["test"] == (
        "npm run build:debug && npm run typecheck && node --test bindings/node/test/api.test.mjs"
    )


def test_node_quality_does_not_install_rust() -> None:
    config = CIRCLECI_CONFIG_PATH.read_text()

    assert "Install Rust" not in job_definition(config, "node_quality")


def test_windows_runs_commit_pinned_git_install_contract() -> None:
    config = CIRCLECI_CONFIG_PATH.read_text()
    windows_job = job_definition(config, "windows_node_tests")

    assert WINDOWS_INODE_TEST_COMMAND in windows_job
    assert "name: Install Node dependencies" in windows_job
    assert "name: Test Windows file identifiers" in windows_job
    assert "name: Test commit-pinned Git installation" in windows_job
    assert "npm ci --ignore-scripts" in windows_job
    assert "npm run test:git-install" in windows_job
    assert f"no_output_timeout: {WINDOWS_GIT_INSTALL_NO_OUTPUT_TIMEOUT}" in windows_job
    assert "--prefix bindings/node" not in windows_job


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
    assert "bindings/node/package-lock.json" not in config

    assert "v4-rust-deps-" in job_definition(config, "rust_tests")
    assert "v4-python-rust-deps-" in job_definition(config, "python_tests")
    assert "v4-node-rust-deps-" in job_definition(config, "node_tests")
    assert "v4-windows-rust-deps-" in job_definition(config, "windows_node_tests")

    for job_name in ("rust_tests", "python_tests", "node_tests", "windows_node_tests"):
        assert "./target" not in job_definition(config, job_name)
