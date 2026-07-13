from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[1]
QUALITY_JOBS = {"rust_quality", "python_quality", "node_quality"}


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


def test_windows_node_tests_require_quality_jobs() -> None:
    config = (REPOSITORY_ROOT / ".circleci" / "config.yml").read_text()

    assert workflow_job_requirements(config, "windows_node_tests") == QUALITY_JOBS
