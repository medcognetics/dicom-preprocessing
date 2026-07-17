import json
import re
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[1]
CIRCLECI_CONFIG_PATH = REPOSITORY_ROOT / ".circleci" / "config.yml"
GITHUB_ACTIONS_CONFIG_PATH = REPOSITORY_ROOT / ".github" / "workflows" / "linux-ci.yml"
NODE_PACKAGE_PATH = REPOSITORY_ROOT / "package.json"

CROSS_PLATFORM_JOBS = (
    "windows_node_tests",
    "macos_arm64_node_tests",
    "macos_x64_node_tests",
)
LINUX_CIRCLECI_JOBS = (
    "rust_quality",
    "python_quality",
    "node_quality",
    "rust_tests",
    "python_tests",
    "node_tests",
)
GITHUB_ACTIONS_JOBS = ("rust", "python", "node")
VERSION_TAG_FILTER = "/^v[0-9]+\\.[0-9]+\\.[0-9]+$/"
GITHUB_VERSION_TAG_FILTER = '"v[0-9]+.[0-9]+.[0-9]+"'
NIGHTLY_CRON = '"17 6 * * *"'
WINDOWS_INODE_TEST_COMMAND = "cargo test -p dicom-preprocessing --lib file::tests::test_inode_sort"
WINDOWS_GIT_INSTALL_NO_OUTPUT_TIMEOUT = "30m"
SHA_PINNED_ACTION_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+@[0-9a-f]{40}$")


def mapping_definition(config: str, key: str, indentation: int) -> str:
    lines = config.splitlines()
    marker = f"{' ' * indentation}{key}:"

    for definition_index, line in enumerate(lines):
        if line != marker:
            continue

        definition_lines = [line]
        for definition_line in lines[definition_index + 1 :]:
            definition_indentation = len(definition_line) - len(definition_line.lstrip())
            if definition_line.strip() and definition_indentation <= indentation:
                break
            definition_lines.append(definition_line)
        return "\n".join(definition_lines)

    raise AssertionError(f"Mapping definition not found: {key}")


def circleci_job_definition(config: str, job_name: str) -> str:
    return mapping_definition(config, job_name, 4)


def circleci_workflow_definition(config: str, workflow_name: str) -> str:
    workflows = config.split("\nworkflows:\n", maxsplit=1)[1]
    return mapping_definition(workflows, workflow_name, 4)


def github_job_definition(config: str, job_name: str) -> str:
    jobs = config.split("\njobs:\n", maxsplit=1)[1]
    return mapping_definition(jobs, job_name, 2)


def test_linux_workflow_uses_expected_triggers() -> None:
    config = GITHUB_ACTIONS_CONFIG_PATH.read_text()

    assert "  pull_request:\n    branches: [master]" in config
    assert "  push:\n    branches: [master]" in config
    assert f"      - {GITHUB_VERSION_TAG_FILTER}" in config
    assert "  workflow_dispatch:" in config
    assert f"    - cron: {NIGHTLY_CRON}" in config
    assert "GIT_INSTALL_SHA: ${{ github.event.pull_request.head.sha || github.sha }}" in config


def test_linux_jobs_use_beryl_and_skip_fork_pull_requests() -> None:
    config = GITHUB_ACTIONS_CONFIG_PATH.read_text()

    for job_name in GITHUB_ACTIONS_JOBS:
        job = github_job_definition(config, job_name)
        assert "runs-on: [self-hosted, linux, x64, beryl]" in job
        assert "github.event_name != 'pull_request'" in job
        assert "github.event.pull_request.head.repo.full_name == github.repository" in job


def test_linux_jobs_test_pull_request_merge_result() -> None:
    config = GITHUB_ACTIONS_CONFIG_PATH.read_text()

    assert "CI_SHA:" not in config
    for job_name in GITHUB_ACTIONS_JOBS:
        assert "ref:" not in github_job_definition(config, job_name)

    node_job = github_job_definition(config, "node")
    assert "DICOM_PREPROCESSING_GIT_SHA: ${{ env.GIT_INSTALL_SHA }}" in node_job


def test_linux_jobs_consolidate_quality_and_runtime_checks_by_language() -> None:
    config = GITHUB_ACTIONS_CONFIG_PATH.read_text()

    rust_job = github_job_definition(config, "rust")
    assert "cargo fmt -- --check" in rust_job
    assert "cargo clippy --workspace --all-features -- -D warnings" in rust_job
    assert "cargo test --workspace --all-features" in rust_job

    python_job = github_job_definition(config, "python")
    assert 'python-version: "3.13"' in python_job
    assert "make init-no-project" in python_job
    assert "make quality-python" in python_job
    assert "make test-python-ci" in python_job

    node_job = github_job_definition(config, "node")
    assert 'node-version: "24.13.0"' in node_job
    assert "make quality-node" in node_job
    assert "make test-node" in node_job
    assert "DICOM_PREPROCESSING_GIT_URL" in node_job
    assert "DICOM_PREPROCESSING_GIT_SHA" in node_job


def test_linux_workflow_minimizes_permissions_and_pins_actions() -> None:
    config = GITHUB_ACTIONS_CONFIG_PATH.read_text()

    assert "permissions:\n  contents: read" in config
    assert "cancel-in-progress: ${{ github.event_name == 'pull_request' }}" in config

    action_references = [line.strip().removeprefix("- uses: ") for line in config.splitlines() if "- uses: " in line]
    assert action_references
    assert all(SHA_PINNED_ACTION_PATTERN.fullmatch(reference) for reference in action_references)


def test_circleci_only_defines_cross_platform_jobs() -> None:
    config = CIRCLECI_CONFIG_PATH.read_text()

    for job_name in CROSS_PLATFORM_JOBS:
        assert circleci_job_definition(config, job_name)
    for job_name in LINUX_CIRCLECI_JOBS:
        assert f"    {job_name}:" not in config


def test_circleci_cross_platform_jobs_are_opt_in_and_parallel() -> None:
    config = CIRCLECI_CONFIG_PATH.read_text()

    assert "run_cross_platform:\n        type: boolean\n        default: false" in config
    workflow = circleci_workflow_definition(config, "cross_platform")
    assert "when: << pipeline.parameters.run_cross_platform >>" in workflow
    for job_name in CROSS_PLATFORM_JOBS:
        assert f"            - {job_name}" in workflow
    assert "requires:" not in workflow


def test_circleci_cross_platform_jobs_run_for_exact_version_tags() -> None:
    config = CIRCLECI_CONFIG_PATH.read_text()
    workflow = circleci_workflow_definition(config, "release_cross_platform")

    assert VERSION_TAG_FILTER in workflow
    assert "branches:\n                        ignore: /.*/" in workflow
    for job_name in CROSS_PLATFORM_JOBS:
        assert f"            - {job_name}:" in workflow


def test_node_validation_uses_debug_builds() -> None:
    package = json.loads(NODE_PACKAGE_PATH.read_text())

    assert package["scripts"]["typecheck"] == "tsc --noEmit --project bindings/node/tsconfig.json"
    assert package["scripts"]["test"] == (
        "npm run build:debug && npm run typecheck && node --test bindings/node/test/api.test.mjs"
    )


def test_windows_runs_commit_pinned_git_install_contract() -> None:
    config = CIRCLECI_CONFIG_PATH.read_text()
    windows_job = circleci_job_definition(config, "windows_node_tests")

    assert WINDOWS_INODE_TEST_COMMAND in windows_job
    assert "name: Install Node dependencies" in windows_job
    assert "name: Test Windows file identifiers" in windows_job
    assert "name: Test commit-pinned Git installation" in windows_job
    assert "npm ci --ignore-scripts" in windows_job
    assert "npm run test:git-install" in windows_job
    assert f"no_output_timeout: {WINDOWS_GIT_INSTALL_NO_OUTPUT_TIMEOUT}" in windows_job
    assert "--prefix bindings/node" not in windows_job


def test_macos_jobs_validate_native_architectures_and_git_install() -> None:
    config = CIRCLECI_CONFIG_PATH.read_text()
    arm64_job = circleci_job_definition(config, "macos_arm64_node_tests")
    x64_job = circleci_job_definition(config, "macos_x64_node_tests")

    assert 'test "$(node -p process.arch)" = arm64' in arm64_job
    assert "npm run test:git-install" in arm64_job
    assert "macos/install-rosetta" in x64_job
    assert 'test "$(node -p process.arch)" = x64' in x64_job
    assert "npm run test:git-install" in x64_job


def test_circleci_avoids_cross_executor_build_artifacts() -> None:
    config = CIRCLECI_CONFIG_PATH.read_text()

    assert "persist_to_workspace" not in config
    assert "attach_workspace" not in config
    assert "~/.rustup" not in config
    assert "bindings/node/node_modules" not in config
    assert "bindings/node/package-lock.json" not in config
    assert "v4-windows-rust-deps-" in circleci_job_definition(config, "windows_node_tests")
    assert "./target" not in circleci_job_definition(config, "windows_node_tests")
