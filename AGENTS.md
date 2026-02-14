# Repository Guidelines

## Project Structure & Module Organization
Core Rust code lives in `src/`.
- CLI entry points: `src/main.rs` (`dicom-preprocess`) and `src/bin/*.rs` (manifest, combine, stats, trace, resize).
- Library modules: `src/transform/`, `src/metadata/`, `src/python/`, `src/errors/`.
- Python typing stub: `dicom_preprocessing.pyi`.

Tests are in `tests/` (pytest for Python bindings). Benchmarks live in `benches/`. Examples are in `examples/`, and documentation images in `docs/`.

## Build, Test, and Development Commands
- `make init`: install `uv` if missing and sync all dependency groups.
- `make init-no-project`: install `uv` if missing and sync dependency groups without installing the local project package.
- `make develop`: build/install the Python extension locally via `maturin --release`.
- `make develop-debug`: build/install the Python extension in debug mode (faster, CI-friendly).
- `make develop-release`: build/install the Python extension in release mode.
- `make quality`: run Rust checks plus Python quality checks (`ruff format --check`, `ruff check`, `basedpyright`).
- `make quality-python`: run Python quality checks only.
- `make style`: apply auto-fixes (`cargo fix`, `clippy --fix`, `cargo fmt`, `ruff check --fix`, `ruff format`).
- `make test`: run Rust tests and Python tests.
- `make test-python`: run only the Python test suite under `tests/`.
- `make test-python-ci`: run Python tests against a debug extension build (CI target).

## CLI Binaries
Run binaries with `cargo run --release --bin <name> -- ...`:
- `dicom-preprocess <SOURCE> <OUTPUT> -s 384,512 -c`: preprocess DICOM to TIFF.
- `dicom-manifest <SOURCE_DIR> [OUTPUT.{csv|parquet}]`: generate TIFF manifest (default `manifest.parquet` in source).
- `dicom-voilut <DICOM_FILE>`: print VOI LUT, rescale, and window metadata.
- `dicom-traces <SOURCE> <TRACES.{csv|parquet}> <OUTPUT> [-p PREVIEW_DIR]`: map traces into preprocessed coordinates.
- `tiff-combine <SOURCE_DIR> <METADATA.{csv|parquet}> <OUTPUT_DIR>`: combine single-frame TIFF slices into multi-frame TIFFs.
- `tiff-stats <SOURCE_DIR> [-n]`: compute per-channel mean/std (`-n` normalizes to `[0,1]`).
- `resize <SOURCE_DIR> <SCALE> <OUTPUT_DIR> [-f lanczos3]`: scale preprocessed TIFF datasets.

## Coding Style & Naming Conventions
Use Rust 2021 defaults and keep code `rustfmt`/`clippy` clean (`-D warnings` in CI). Prefer snake_case for functions/modules and descriptive CLI flag names.

Python style is formatter-driven:
- Ruff is the formatter/import sorter and is configured in `pyproject.toml` (`line-length = 120`, `target-version = "py313"`).
- basedpyright type checking is configured in `pyproject.toml` (`pythonVersion = "3.13"`, `typeCheckingMode = "basic"`).
- Test files use `test_*.py`; test functions use `test_*`.

## CI Quality Pipeline
- Rust quality gate: `rust_quality` job (`cargo fmt --check`, `cargo clippy --all-features -- -D warnings`).
- Python quality gate: `python_quality` job (Python 3.13; runs `make init-no-project` then `make quality-python`).
- Rust runtime tests: `rust_tests` job (`cargo test --all-features`).
- Python runtime tests: `python_tests` job (`make test-python-ci`, which uses a debug extension build).
- Test jobs are gated on both quality jobs passing.

## Testing Guidelines
Add or update tests when behavior changes in preprocessing, manifest generation, TIFF I/O, or Python bindings.
- Run: `make quality` and `make test` before opening a PR.
- Keep fixtures lightweight (`tmp_path`, `pydicom` sample data) and assert shape/dtype/value bounds for image outputs.
- No explicit coverage threshold is enforced; maintain or improve coverage in touched areas.

## Commit & Pull Request Guidelines
Recent commits use concise imperative subjects, often with a PR reference, e.g. `Add DBT single-frame projection methods (#54)`.

For PRs, include:
- What changed and why.
- Validation commands you ran (`make quality`, `make test`).
- Linked issue/PR context.
- Output examples or docs image updates when preprocessing behavior or CLI output changes.
