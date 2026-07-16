from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[1]
QUALITY_JOBS = {"rust_quality", "python_quality", "node_quality"}
RUNTIME_JOBS = {
    "rust_tests",
    "python_tests",
    "node_tests",
    "windows_node_tests",
    "macos_arm64_node_tests",
    "macos_x64_node_tests",
}
WINDOWS_GIT_INSTALL_NO_OUTPUT_TIMEOUT = "30m"


def workflow_job_requirements(config: str, job_name: str) -> set[str]:
    lines = config.splitlines()
    job_markers = {f"- {job_name}", f"- {job_name}:"}

    for job_index, line in enumerate(lines):
        if line.strip() not in job_markers:
            continue

        job_indent = len(line) - len(line.lstrip())
        for requires_index in range(job_index + 1, len(lines)):
            requires_line = lines[requires_index]
            requires_indent = len(requires_line) - len(requires_line.lstrip())
            if requires_line.strip() and requires_indent <= job_indent:
                return set()
            if requires_line.strip() != "requires:":
                continue

            requirements = set()
            for dependency_line in lines[requires_index + 1 :]:
                dependency_indent = len(dependency_line) - len(dependency_line.lstrip())
                if dependency_line.strip() and dependency_indent <= requires_indent:
                    break
                if dependency_line.strip().startswith("- "):
                    requirements.add(dependency_line.strip().removeprefix("- "))
            return requirements
        return set()

    raise AssertionError(f"Workflow job not found: {job_name}")


def test_runtime_jobs_require_quality_jobs() -> None:
    config = (REPOSITORY_ROOT / ".circleci" / "config.yml").read_text()

    for job_name in RUNTIME_JOBS:
        assert workflow_job_requirements(config, job_name) == QUALITY_JOBS


def test_windows_git_install_allows_silent_source_build() -> None:
    config = (REPOSITORY_ROOT / ".circleci" / "config.yml").read_text()
    run_step_header = f"""                  name: Build and test Node bindings
                  no_output_timeout: {WINDOWS_GIT_INSTALL_NO_OUTPUT_TIMEOUT}
                  command: |"""

    assert run_step_header in config
