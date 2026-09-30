import json
import re
from pathlib import Path

import pytest

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
NODE_PACKAGE_LOCK_PATH = REPOSITORY_ROOT / "package-lock.json"
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
MINIMUM_SAFE_JS_YAML_VERSION = (4, 3, 1)
WINDOWS_INODE_TEST_COMMAND = "cargo test -p dicom-preprocessing --lib file::tests::test_inode_sort"
SHA_PINNED_ACTION_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+@[0-9a-f]{40}$")
JOB_KEY_PATTERN = re.compile(r"^  ([a-zA-Z0-9_-]+):$")
PULL_REQUEST_TRIGGER_PATTERN = re.compile(r"^(?:on:.*\bpull_request\b|  (?:- )?pull_request(?::.*)?$)", re.MULTILINE)
PULL_REQUEST_TARGET_PATTERN = re.compile(r"\bpull_request_target\b")
HOSTED_RUNS_ON_PATTERN = re.compile(r"    runs-on: (?:ubuntu|windows|macos)-[0-9a-z.-]+")
TRUSTED_BERYL_RUNS_ON = (
    "    runs-on: >- # zizmor: ignore[self-hosted-runner] one-job ephemeral; fork pull requests use ubuntu-24.04\n"
    "      ${{ fromJSON((github.event_name != 'pull_request' || "
    "github.event.pull_request.head.repo.full_name == github.repository) && "
    '\'["self-hosted","linux","x64","beryl"]\' || \'["ubuntu-24.04"]\') }}'
)
RUSTUP_BOOTSTRAP = "if ! command -v rustup >/dev/null 2>&1; then"


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


def github_job_names(config: str) -> list[str]:
    jobs = config.split("\njobs:\n", maxsplit=1)[1]
    return [match.group(1) for line in jobs.splitlines() if (match := JOB_KEY_PATTERN.fullmatch(line))]


def runs_on_definition(job: str) -> str:
    lines = job.splitlines()
    for index, line in enumerate(lines):
        if not line.startswith("    runs-on:"):
            continue

        definition_lines = [line]
        for continuation in lines[index + 1 :]:
            if continuation.strip() and len(continuation) - len(continuation.lstrip()) <= 4:
                break
            definition_lines.append(continuation)
        return "\n".join(definition_lines).rstrip()

    raise AssertionError("Job has no runs-on definition")


def discovered_workflow_paths() -> list[Path]:
    return sorted((*WORKFLOW_DIRECTORY.glob("*.yml"), *WORKFLOW_DIRECTORY.glob("*.yaml")))


def requires_trusted_beryl_routing(config: str) -> bool:
    return PULL_REQUEST_TRIGGER_PATTERN.search(config) is not None


def pull_request_target_runner_violations(config: str) -> list[str]:
    # TRUSTED_BERYL_RUNS_ON selects beryl for every pull_request_target event, including fork pull requests.
    # Any mention of the trigger fails closed, whatever the trigger syntax.
    if PULL_REQUEST_TARGET_PATTERN.search(config) is None:
        return []
    return [
        job_name
        for job_name in github_job_names(config)
        if not any(
            HOSTED_RUNS_ON_PATTERN.fullmatch(line) for line in github_job_definition(config, job_name).splitlines()
        )
    ]


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


def test_linux_workflow_defines_expected_jobs_and_timeouts() -> None:
    config = LINUX_CI_PATH.read_text()

    assert github_job_names(config) == list(ALL_LINUX_JOBS)
    for job_name in LINUX_JOBS:
        assert "timeout-minutes: 20" in github_job_definition(config, job_name)
    assert "timeout-minutes: 30" in github_job_definition(config, "minimum_versions")


def test_pull_request_workflows_use_hosted_runners_only_for_forks() -> None:
    pull_request_workflow_paths = [
        path for path in discovered_workflow_paths() if requires_trusted_beryl_routing(path.read_text())
    ]

    assert LINUX_CI_PATH in pull_request_workflow_paths
    for workflow_path in pull_request_workflow_paths:
        config = workflow_path.read_text()
        job_names = github_job_names(config)
        assert job_names, workflow_path.name
        for job_name in job_names:
            runs_on = runs_on_definition(github_job_definition(config, job_name))
            assert runs_on == TRUSTED_BERYL_RUNS_ON, f"{workflow_path.name}: {job_name}"


def synthetic_workflow(triggers: str, runs_on: str) -> str:
    return f"name: Synthetic\n\n{triggers}\n\njobs:\n  build:\n{runs_on}\n    steps:\n      - run: true\n"


@pytest.mark.parametrize(
    ("triggers", "expected"),
    [
        pytest.param("on:\n  pull_request:\n    branches: [master]", True, id="pull_request-mapping"),
        pytest.param("on: [push, pull_request]", True, id="pull_request-inline"),
        pytest.param("on:\n  - push\n  - pull_request", True, id="pull_request-list"),
        pytest.param("on:\n  pull_request_target:\n    branches: [master]", False, id="pull_request_target-mapping"),
        pytest.param("on: [push, pull_request_target]", False, id="pull_request_target-inline"),
        pytest.param("on:\n  push:\n    branches: [master]", False, id="push"),
    ],
)
def test_trusted_beryl_routing_applies_only_to_pull_request_trigger(triggers: str, expected: bool) -> None:
    assert requires_trusted_beryl_routing(synthetic_workflow(triggers, TRUSTED_BERYL_RUNS_ON)) is expected


@pytest.mark.parametrize(
    "triggers",
    [
        pytest.param("on:\n  pull_request_target:\n    branches: [master]", id="mapping"),
        pytest.param("on: pull_request_target", id="scalar"),
        pytest.param("on: [push, pull_request_target]", id="inline"),
        pytest.param("on:\n  - push\n  - pull_request_target", id="list"),
        pytest.param("on:\n  pull_request:\n  pull_request_target:", id="with-pull_request"),
    ],
)
@pytest.mark.parametrize(
    ("runs_on", "expected_violations"),
    [
        pytest.param(TRUSTED_BERYL_RUNS_ON, ["build"], id="trusted-beryl-expression"),
        pytest.param("    runs-on: [self-hosted, linux, x64, beryl]", ["build"], id="beryl-labels"),
        pytest.param("    runs-on: ${{ vars.RUNNER }}", ["build"], id="runner-expression"),
        pytest.param("    uses: ./.github/workflows/reusable.yml", ["build"], id="reusable-workflow"),
        pytest.param("    runs-on: ubuntu-24.04", [], id="hosted-label"),
    ],
)
def test_pull_request_target_jobs_require_literal_hosted_runners(
    triggers: str, runs_on: str, expected_violations: list[str]
) -> None:
    assert pull_request_target_runner_violations(synthetic_workflow(triggers, runs_on)) == expected_violations


def test_pull_request_target_runner_check_ignores_other_triggers() -> None:
    config = synthetic_workflow("on:\n  push:\n    branches: [master]", TRUSTED_BERYL_RUNS_ON)

    assert pull_request_target_runner_violations(config) == []


def test_pull_request_target_workflows_never_use_beryl() -> None:
    for workflow_path in discovered_workflow_paths():
        assert pull_request_target_runner_violations(workflow_path.read_text()) == [], workflow_path.name


def test_beryl_jobs_bootstrap_rustup_and_select_linked_python() -> None:
    beryl_jobs = {
        (workflow_path.name, job_name): github_job_definition(config, job_name)
        for workflow_path, config in ((path, path.read_text()) for path in ALL_WORKFLOW_PATHS)
        for job_name in github_job_names(config)
        if "beryl" in runs_on_definition(github_job_definition(config, job_name))
    }

    assert set(beryl_jobs) == {
        *((LINUX_CI_PATH.name, job_name) for job_name in ALL_LINUX_JOBS),
        (NIGHTLY_BUILD_PATH.name, "build"),
    }
    for job in beryl_jobs.values():
        if "rustup toolchain install" in job:
            assert RUSTUP_BOOTSTRAP in job
            assert "https://sh.rustup.rs" in job
            assert job.index(RUSTUP_BOOTSTRAP) < job.index("rustup toolchain install")
        if "${pythonLocation}" in job:
            assert "uses: actions/setup-python@" in job
            assert job.index("uses: actions/setup-python@") < job.index("${pythonLocation}")


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


def test_node_lock_excludes_vulnerable_js_yaml_versions() -> None:
    package_lock = json.loads(NODE_PACKAGE_LOCK_PATH.read_text())
    js_yaml = package_lock["packages"].get("node_modules/js-yaml")
    if js_yaml is None:
        return

    version_without_build_metadata = js_yaml["version"].partition("+")[0]
    version_core, prerelease_marker, _ = version_without_build_metadata.partition("-")
    version = tuple(int(component) for component in version_core.split("."))

    assert version > MINIMUM_SAFE_JS_YAML_VERSION or (version == MINIMUM_SAFE_JS_YAML_VERSION and not prerelease_marker)


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


def test_cargo_audit_has_no_advisory_exceptions() -> None:
    config = CARGO_AUDIT_CONFIG_PATH.read_text()

    assert "ignore =" not in config
    assert not re.findall(r"RUSTSEC-\d{4}-\d{4}", config)
    assert 'informational_warnings = ["unmaintained", "unsound", "notice"]' in config


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
    loader = NODE_LOADER_PATH.read_text()

    assert package["engines"]["node"] == "^22.13.0 || ^24.0.0 || ^26.0.0"
    assert package["napi"]["targets"] == [
        "aarch64-apple-darwin",
        "x86_64-pc-windows-msvc",
        "x86_64-unknown-linux-gnu",
    ]
    assert set(re.findall(r"bindingPackageVersion !== '([^']+)'", loader)) == {package["version"]}
    assert set(re.findall(r"version mismatch, expected (\d+\.\d+\.\d+)", loader)) == {package["version"]}
    assert "darwin-x64" not in loader
    assert "darwin:x64" not in NODE_GIT_INSTALL_PATH.read_text()


def test_node_validation_uses_debug_builds() -> None:
    package = json.loads(NODE_PACKAGE_PATH.read_text())

    assert package["scripts"]["typecheck"] == "tsc --noEmit --project bindings/node/tsconfig.json"
    assert package["scripts"]["test"] == (
        "npm run build:debug && npm run typecheck && node --test bindings/node/test/api.test.mjs"
    )
