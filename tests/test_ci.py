import json
import re
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[1]
WORKFLOW_DIRECTORY = REPOSITORY_ROOT / ".github" / "workflows"
LINUX_CI_PATH = WORKFLOW_DIRECTORY / "linux-ci.yml"
NIGHTLY_BUILD_PATH = WORKFLOW_DIRECTORY / "nightly-build.yml"
CROSS_PLATFORM_PATH = WORKFLOW_DIRECTORY / "weekly-cross-platform.yml"
DEPENDENCY_HEALTH_PATH = WORKFLOW_DIRECTORY / "dependency-health.yml"
ACTIONLINT_CONFIG_PATH = REPOSITORY_ROOT / ".github" / "actionlint.yaml"
CARGO_AUDIT_CONFIG_PATH = REPOSITORY_ROOT / ".cargo" / "audit.toml"
CIRCLECI_CONFIG_PATH = REPOSITORY_ROOT / ".circleci" / "config.yml"
MAKEFILE_PATH = REPOSITORY_ROOT / "Makefile"
NODE_PACKAGE_PATH = REPOSITORY_ROOT / "package.json"
NODE_LOADER_PATH = REPOSITORY_ROOT / "bindings" / "node" / "index.js"
NODE_GIT_INSTALL_PATH = REPOSITORY_ROOT / "bindings" / "node" / "test" / "git-install.mjs"

LINUX_JOBS = ("rust", "python", "node")
ALL_LINUX_JOBS = (*LINUX_JOBS, "minimum_versions")
CROSS_PLATFORM_JOBS = ("windows", "macos_arm64")
ALL_WORKFLOW_PATHS = (LINUX_CI_PATH, NIGHTLY_BUILD_PATH, CROSS_PLATFORM_PATH, DEPENDENCY_HEALTH_PATH)
VERSION_TAG_FILTER = '"v[0-9]+.[0-9]+.[0-9]+"'
NIGHTLY_CRON = '"17 6 * * *"'
CROSS_PLATFORM_CRON = '"17 5 * * 0"'
DEPENDENCY_HEALTH_CRON = '"17 7 * * 1"'
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


def github_job_definition(config: str, job_name: str) -> str:
    jobs = config.split("\njobs:\n", maxsplit=1)[1]
    return mapping_definition(jobs, job_name, 2)


def action_references(config: str) -> list[str]:
    return [
        line.strip().removeprefix("- uses: ").split(" #", maxsplit=1)[0]
        for line in config.splitlines()
        if line.strip().startswith("- uses: ")
    ]


def test_linux_workflow_uses_expected_triggers_and_concurrency() -> None:
    config = LINUX_CI_PATH.read_text()

    assert "  pull_request:\n    branches: [master]" in config
    assert "  push:\n    branches: [master]" in config
    assert f"      - {VERSION_TAG_FILTER}" in config
    assert "  workflow_dispatch:" in config
    assert "  schedule:" not in config
    assert "group: linux-ci-${{ github.event.pull_request.number || github.ref }}" in config
    assert "cancel-in-progress: ${{ github.event_name == 'pull_request' }}" in config


def test_linux_jobs_route_forks_away_from_ephemeral_beryl() -> None:
    config = LINUX_CI_PATH.read_text()

    for job_name in LINUX_JOBS:
        job = github_job_definition(config, job_name)
        assert "fromJSON" in job
        assert "github.event.pull_request.head.repo.full_name == github.repository" in job
        assert '["self-hosted","linux","x64","beryl"]' in job
        assert '["ubuntu-24.04"]' in job
        assert "zizmor: ignore[self-hosted-runner]" in job
        assert "one-job ephemeral" in job
        assert "timeout-minutes: 20" in job

    minimum_job = github_job_definition(config, "minimum_versions")
    assert "runs-on: ubuntu-24.04" in minimum_job
    assert "self-hosted" not in minimum_job
    assert "timeout-minutes: 30" in minimum_job


def test_linux_jobs_preserve_names_and_rust_gate() -> None:
    config = LINUX_CI_PATH.read_text()

    assert "name: Linux / Rust" in github_job_definition(config, "rust")
    assert "name: Linux / Python" in github_job_definition(config, "python")
    assert "name: Linux / Node" in github_job_definition(config, "node")
    assert "name: Linux / Minimum versions" in github_job_definition(config, "minimum_versions")
    assert "needs: rust" not in github_job_definition(config, "rust")
    for job_name in ("python", "node", "minimum_versions"):
        assert "needs: rust" in github_job_definition(config, job_name)


def test_linux_jobs_cover_current_and_minimum_runtime_boundaries() -> None:
    config = LINUX_CI_PATH.read_text()
    rust_job = github_job_definition(config, "rust")
    python_job = github_job_definition(config, "python")
    node_job = github_job_definition(config, "node")
    minimum_job = github_job_definition(config, "minimum_versions")

    assert "rustup toolchain install 1.97.1" in rust_job
    assert "make quality-rust" in rust_job
    assert "make test-rust" in rust_job

    assert 'python-version: "3.14"' in python_job
    assert 'assert numpy.__version__ == "2.4.6"' in python_job
    assert "make init-no-project" in python_job
    assert "make quality-python" in python_job
    assert "make test-python-ci" in python_job

    assert 'node-version: "24.18.0"' in node_job
    assert 'node-version: "26.5.0"' in node_job
    assert node_job.count("make test-node-direct") == 2
    assert "make quality-node" in node_job

    assert "rustup toolchain install 1.89.0" in minimum_job
    assert 'python-version: "3.10"' in minimum_job
    assert 'assert numpy.__version__ == "2.2.6"' in minimum_job
    assert 'node-version: "22.13.0"' in minimum_job
    assert "make test-rust" in minimum_job
    assert "make test-python-ci" in minimum_job
    assert "make test-node-direct" in minimum_job

    for job_name in ALL_LINUX_JOBS:
        job = github_job_definition(config, job_name)
        assert "test-node-git-install" not in job
        assert "test:git-install" not in job
        assert "rust-cache" not in job
        assert "cache: npm" not in job
        assert "enable-cache: true" not in job
        for setup_node in re.findall(r"uses: actions/setup-node@.*?(?=\n      -|\Z)", job, re.DOTALL):
            assert "package-manager-cache: false" in setup_node


def test_nightly_build_is_ephemeral_release_validation() -> None:
    config = NIGHTLY_BUILD_PATH.read_text()
    job = github_job_definition(config, "build")

    assert f"    - cron: {NIGHTLY_CRON}" in config
    assert f"      - {VERSION_TAG_FILTER}" in config
    assert "  workflow_dispatch:" in config
    assert "name: Nightly / Build and install" in job
    assert "runs-on: [self-hosted, linux, x64, beryl]" in job
    assert "zizmor: ignore[self-hosted-runner]" in job
    assert "timeout-minutes: 60" in job
    assert "rustup toolchain install 1.97.1" in job
    assert 'python-version: "3.14"' in job
    assert 'node-version: "26.5.0"' in job
    assert (
        "ARTIFACT_DIR: ${{ runner.temp }}/dicom-preprocessing-${{ github.run_id }}-${{ github.run_attempt }}/dist"
        in job
    )
    assert (
        "CARGO_TARGET_DIR: ${{ runner.temp }}/dicom-preprocessing-${{ github.run_id }}-${{ github.run_attempt }}/target"
        in job
    )
    assert 'PYTHON_BUILD_VERSION: "3.14"' in job
    assert "make build" in job
    assert "make test-build" in job
    assert "DICOM_PREPROCESSING_GIT_SHA: ${{ github.sha }}" in job
    assert "upload-artifact" not in job
    assert "rust-cache" not in job
    assert "cache: npm" not in job
    assert "enable-cache: true" not in job
    assert "package-manager-cache: false" in job


def test_cross_platform_workflow_is_scheduled_and_native() -> None:
    config = CROSS_PLATFORM_PATH.read_text()

    assert f"    - cron: {CROSS_PLATFORM_CRON}" in config
    assert f"      - {VERSION_TAG_FILTER}" in config
    assert "  workflow_dispatch:" in config
    assert "  pull_request:" not in config
    assert "    branches:" not in config

    windows_job = github_job_definition(config, "windows")
    assert "name: Cross-platform / Windows x64" in windows_job
    assert "runs-on: windows-2022" in windows_job
    assert "timeout-minutes: 75" in windows_job
    assert 'node-version: "26.5.0"' in windows_job
    assert "rustup toolchain install 1.97.1" in windows_job
    assert 'test "$(node -p process.arch)" = x64' in windows_job
    assert WINDOWS_INODE_TEST_COMMAND in windows_job

    macos_job = github_job_definition(config, "macos_arm64")
    assert "name: Cross-platform / macOS arm64" in macos_job
    assert "runs-on: macos-15" in macos_job
    assert "timeout-minutes: 30" in macos_job
    assert 'node-version: "26.5.0"' in macos_job
    assert "rustup toolchain install 1.97.1" in macos_job
    assert 'test "$(node -p process.arch)" = arm64' in macos_job

    for job_name in CROSS_PLATFORM_JOBS:
        job = github_job_definition(config, job_name)
        assert "npm ci --ignore-scripts" in job
        assert "npm run test:git-install" in job
        assert "DICOM_PREPROCESSING_GIT_SHA: ${{ github.sha }}" in job
        assert "cache-targets: false" in job
        assert "cache-bin: false" in job
        assert "self-hosted" not in job


def test_dependency_health_jobs_are_independent_and_read_only() -> None:
    config = DEPENDENCY_HEALTH_PATH.read_text()

    assert f"    - cron: {DEPENDENCY_HEALTH_CRON}" in config
    assert "  workflow_dispatch:" in config
    assert "  pull_request:" not in config
    assert "permissions:\n  contents: read" in config
    assert "runs-on: ubuntu-24.04" in github_job_definition(config, "security_audit")
    assert "runs-on: ubuntu-24.04" in github_job_definition(config, "deprecation_report")
    assert "name: Dependency Health / Security Audit" in github_job_definition(config, "security_audit")
    assert "name: Dependency Health / Deprecation Report" in github_job_definition(config, "deprecation_report")
    for job_name in ("security_audit", "deprecation_report"):
        job = github_job_definition(config, job_name)
        assert "timeout-minutes: 20" in job
        assert "needs:" not in job
        assert "contents: write" not in job
        assert "upload-artifact" not in job
    assert "cargo-audit --version 0.22.1" in config
    assert 'version: "0.11.18"' in config
    assert "zizmor==1.27.0" in config
    assert "scripts/ci/dependency_health.py security" in config
    assert "scripts/ci/dependency_health.py deprecation" in config


def test_workflows_pin_actions_and_disable_checkout_credentials() -> None:
    for workflow_path in ALL_WORKFLOW_PATHS:
        config = workflow_path.read_text()
        references = action_references(config)

        assert "permissions:\n  contents: read" in config
        assert references
        assert all(SHA_PINNED_ACTION_PATTERN.fullmatch(reference) for reference in references)
        assert config.count("persist-credentials: false") == config.count("actions/checkout@")
        assert config.count("clean: true") == config.count("actions/checkout@")


def test_actionlint_knows_custom_runner_label() -> None:
    config = ACTIONLINT_CONFIG_PATH.read_text()

    assert "self-hosted-runner:" in config
    assert "labels:" in config
    assert "- beryl" in config


def test_circleci_is_retired() -> None:
    assert not CIRCLECI_CONFIG_PATH.exists()


def test_cargo_audit_exception_is_narrow_and_documented() -> None:
    config = CARGO_AUDIT_CONFIG_PATH.read_text()

    assert 'ignore = ["RUSTSEC-2026-0151"]' in config
    assert set(re.findall(r"RUSTSEC-\d{4}-\d{4}", config)) == {"RUSTSEC-2026-0151"}
    assert "32-bit" in config
    assert "jxl-grid" in config
    assert "JPEG XL" in config


def test_makefile_exposes_locked_ci_targets() -> None:
    config = MAKEFILE_PATH.read_text()

    assert "UV_SYNC_ALL_GROUPS=$(UV) sync --locked --all-groups" in config
    assert "$(MATURIN): pyproject.toml uv.lock | ensure-uv\n\t$(UV_SYNC_ALL_GROUPS) --no-install-project" in config
    assert "PYTHON_BUILD_VERSION?=3.14" in config
    assert "CARGO_TARGET_DIR?=target" in config
    assert "RUST_RELEASE_DIR=$(CARGO_TARGET_DIR)/$(RUST_TARGET)/release" in config
    assert "quality: quality-rust quality-python quality-node" in config
    assert "quality-rust:" in config
    assert "cargo check --locked --workspace --all-features" in config
    assert "cargo clippy --locked --workspace --all-features --all-targets -- -D warnings" in config
    assert "test: test-rust test-python test-node" in config
    assert "test-rust:" in config
    assert "cargo test --locked --workspace --all-features" in config
    assert "$(UV) venv --python $(PYTHON_BUILD_VERSION)" in config


def test_node_support_contract_matches_ci_platforms() -> None:
    package = json.loads(NODE_PACKAGE_PATH.read_text())

    assert package["engines"]["node"] == "^22.13.0 || ^24.0.0 || ^26.0.0"
    assert package["napi"]["targets"] == [
        "aarch64-apple-darwin",
        "x86_64-pc-windows-msvc",
        "x86_64-unknown-linux-gnu",
    ]
    assert "darwin-x64" not in NODE_LOADER_PATH.read_text()
    assert "darwin:x64" not in NODE_GIT_INSTALL_PATH.read_text()


def test_node_validation_uses_debug_builds() -> None:
    package = json.loads(NODE_PACKAGE_PATH.read_text())

    assert package["scripts"]["typecheck"] == "tsc --noEmit --project bindings/node/tsconfig.json"
    assert package["scripts"]["test"] == (
        "npm run build:debug && npm run typecheck && node --test bindings/node/test/api.test.mjs"
    )
