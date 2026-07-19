import json
import re
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[1]
CIRCLECI_CONFIG_PATH = REPOSITORY_ROOT / ".circleci" / "config.yml"
GITHUB_ACTIONS_CONFIG_PATH = REPOSITORY_ROOT / ".github" / "workflows" / "linux-ci.yml"
NIGHTLY_BUILD_CONFIG_PATH = REPOSITORY_ROOT / ".github" / "workflows" / "nightly-build.yml"
WEEKLY_CROSS_PLATFORM_CONFIG_PATH = REPOSITORY_ROOT / ".github" / "workflows" / "weekly-cross-platform.yml"
MAKEFILE_PATH = REPOSITORY_ROOT / "Makefile"
NODE_PACKAGE_PATH = REPOSITORY_ROOT / "package.json"

CIRCLECI_JOBS = ("macos_x64_node_tests",)
MIGRATED_CIRCLECI_JOBS = (
    "windows_node_tests",
    "macos_arm64_node_tests",
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
GITHUB_CROSS_PLATFORM_JOBS = ("windows", "macos_arm64")
VERSION_TAG_FILTER = "/^v[0-9]+\\.[0-9]+\\.[0-9]+$/"
GITHUB_VERSION_TAG_FILTER = '"v[0-9]+.[0-9]+.[0-9]+"'
NIGHTLY_CRON = '"17 6 * * *"'
WEEKLY_CRON = '"17 5 * * 0"'
WINDOWS_INODE_TEST_COMMAND = "cargo test -p dicom-preprocessing --lib file::tests::test_inode_sort"
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
    assert "  schedule:" not in config


def test_nightly_build_uses_expected_triggers() -> None:
    config = NIGHTLY_BUILD_CONFIG_PATH.read_text()

    assert f"    - cron: {NIGHTLY_CRON}" in config
    assert "  workflow_dispatch:" in config


def test_weekly_cross_platform_uses_expected_triggers() -> None:
    config = WEEKLY_CROSS_PLATFORM_CONFIG_PATH.read_text()

    assert f"    - cron: {WEEKLY_CRON}" in config
    assert "  workflow_dispatch:" in config
    assert f"      - {GITHUB_VERSION_TAG_FILTER}" in config
    assert "  pull_request:" not in config
    assert "    branches:" not in config


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
    assert "GIT_INSTALL_SHA:" not in config
    for job_name in GITHUB_ACTIONS_JOBS:
        assert "ref:" not in github_job_definition(config, job_name)


def test_python_and_node_require_rust() -> None:
    config = GITHUB_ACTIONS_CONFIG_PATH.read_text()

    assert "needs:" not in github_job_definition(config, "rust")
    for job_name in ("python", "node"):
        assert "needs: rust" in github_job_definition(config, job_name)


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
    assert "make test-node-direct" in node_job
    assert "make test-node-git-install" not in node_job
    assert "DICOM_PREPROCESSING_GIT_URL" not in node_job
    assert "DICOM_PREPROCESSING_GIT_SHA" not in node_job


def test_nightly_build_creates_and_verifies_all_artifacts() -> None:
    config = NIGHTLY_BUILD_CONFIG_PATH.read_text()
    build_job = github_job_definition(config, "build")

    assert "runs-on: [self-hosted, linux, x64, beryl]" in build_job
    assert 'python-version: "3.13"' in build_job
    assert 'node-version: "24.13.0"' in build_job
    assert "make build" in build_job
    assert "make test-build" in build_job
    assert "DICOM_PREPROCESSING_GIT_URL: git+https://github.com/${{ github.repository }}.git" in build_job
    assert "DICOM_PREPROCESSING_GIT_SHA: ${{ github.sha }}" in build_job
    assert "path: dist/" in build_job
    assert "if-no-files-found: error" in build_job


def test_weekly_cross_platform_uses_public_github_runners() -> None:
    config = WEEKLY_CROSS_PLATFORM_CONFIG_PATH.read_text()

    assert "runs-on: windows-2022" in github_job_definition(config, "windows")
    assert "runs-on: macos-15" in github_job_definition(config, "macos_arm64")
    for job_name in GITHUB_CROSS_PLATFORM_JOBS:
        assert "self-hosted" not in github_job_definition(config, job_name)


def test_weekly_cross_platform_validates_native_git_installs() -> None:
    config = WEEKLY_CROSS_PLATFORM_CONFIG_PATH.read_text()

    for job_name in GITHUB_CROSS_PLATFORM_JOBS:
        job = github_job_definition(config, job_name)
        assert 'node-version: "24.13.0"' in job
        assert "npm ci --ignore-scripts" in job
        assert "DICOM_PREPROCESSING_GIT_URL: git+https://github.com/${{ github.repository }}.git" in job
        assert "DICOM_PREPROCESSING_GIT_SHA: ${{ github.sha }}" in job
        assert "npm run test:git-install" in job

    windows_job = github_job_definition(config, "windows")
    assert WINDOWS_INODE_TEST_COMMAND in windows_job
    assert "rustc -vV" in windows_job

    macos_job = github_job_definition(config, "macos_arm64")
    assert 'test "$(node -p process.arch)" = arm64' in macos_job


def test_makefile_builds_and_verifies_distributable_artifacts() -> None:
    config = MAKEFILE_PATH.read_text()

    assert "build: build-rust build-python build-node-package" in config
    assert "test-node: test-node-direct test-node-git-install" in config
    assert "test-build: test-rust-artifacts test-python-wheel test-node-package-install test-node-git-install" in config
    assert "RUST_RELEASE_DIR=target/$(RUST_TARGET)/release" in config
    assert "RUST_PACKAGE=dicom-preprocessing-cli-$(RUST_TARGET).tar.gz" in config
    assert "PYTHON_BUILD_VERSION?=3.13" in config
    assert "--interpreter $(PYTHON_BUILD_INTERPRETER)" in config
    assert "rm -f $(PYTHON_ARTIFACT_DIR)/*.whl" in config
    assert "cargo build --locked --workspace --release --target $(RUST_TARGET)" in config
    assert "tar -czf $(RUST_ARTIFACT_DIR)/$(RUST_PACKAGE)" in config
    assert 'tar -xzf "$$archive" -C "$$test_env"' in config
    assert (
        "$(MATURIN) build --locked $(MATURIN_FEATURES) --release --target $(RUST_TARGET) "
        "--interpreter $(PYTHON_BUILD_INTERPRETER) --out $(PYTHON_ARTIFACT_DIR)"
    ) in config
    assert "maturin" in config
    assert "$(NPM) pack --ignore-scripts" in config


def test_rust_tests_link_against_setup_python_tool_cache() -> None:
    config = GITHUB_ACTIONS_CONFIG_PATH.read_text()
    rust_job = github_job_definition(config, "rust")

    assert 'export LIBRARY_PATH="${pythonLocation}/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"' in rust_job


def test_linux_workflow_minimizes_permissions_and_pins_actions() -> None:
    configs = (
        GITHUB_ACTIONS_CONFIG_PATH.read_text(),
        NIGHTLY_BUILD_CONFIG_PATH.read_text(),
        WEEKLY_CROSS_PLATFORM_CONFIG_PATH.read_text(),
    )

    assert "permissions:\n  contents: read" in configs[0]
    assert "cancel-in-progress: ${{ github.event_name == 'pull_request' }}" in configs[0]

    action_references = [
        line.strip().removeprefix("- uses: ")
        for config in configs
        for line in config.splitlines()
        if "- uses: " in line
    ]
    assert action_references
    assert all(SHA_PINNED_ACTION_PATTERN.fullmatch(reference) for reference in action_references)


def test_circleci_only_defines_macos_x64_job() -> None:
    config = CIRCLECI_CONFIG_PATH.read_text()

    for job_name in CIRCLECI_JOBS:
        assert circleci_job_definition(config, job_name)
    for job_name in MIGRATED_CIRCLECI_JOBS:
        assert f"    {job_name}:" not in config
    for job_name in LINUX_CIRCLECI_JOBS:
        assert f"    {job_name}:" not in config
    assert "circleci/windows" not in config


def test_circleci_macos_x64_job_is_opt_in() -> None:
    config = CIRCLECI_CONFIG_PATH.read_text()

    assert "run_cross_platform:\n        type: boolean\n        default: false" in config
    workflow = circleci_workflow_definition(config, "cross_platform")
    assert "when: << pipeline.parameters.run_cross_platform >>" in workflow
    for job_name in CIRCLECI_JOBS:
        assert f"            - {job_name}" in workflow
    assert "requires:" not in workflow


def test_circleci_macos_x64_job_runs_for_exact_version_tags() -> None:
    config = CIRCLECI_CONFIG_PATH.read_text()
    workflow = circleci_workflow_definition(config, "release_cross_platform")

    assert VERSION_TAG_FILTER in workflow
    assert "branches:\n                        ignore: /.*/" in workflow
    for job_name in CIRCLECI_JOBS:
        assert f"            - {job_name}:" in workflow


def test_node_validation_uses_debug_builds() -> None:
    package = json.loads(NODE_PACKAGE_PATH.read_text())

    assert package["scripts"]["typecheck"] == "tsc --noEmit --project bindings/node/tsconfig.json"
    assert package["scripts"]["test"] == (
        "npm run build:debug && npm run typecheck && node --test bindings/node/test/api.test.mjs"
    )


def test_circleci_macos_x64_validates_native_architecture_and_git_install() -> None:
    config = CIRCLECI_CONFIG_PATH.read_text()
    x64_job = circleci_job_definition(config, "macos_x64_node_tests")

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
