# Repository Guidelines

## Project Structure & Module Organization
Core Rust code lives in `src/`.
- CLI entry points: `src/main.rs` (`dicom-preprocess`) and `src/bin/*.rs` (manifest, combine, stats, trace, resize).
- Library modules: `src/transform/`, `src/metadata/`, `src/python/`, `src/errors/`.
- Python typing stub: `dicom_preprocessing.pyi`.
- Node package metadata and its lockfile: root `package.json` and `package-lock.json`; NAPI-RS source and generated entry points: `bindings/node/`.

Tests are in `tests/` (pytest for Python bindings). Benchmarks live in `benches/`. Examples are in `examples/`, and documentation images in `docs/`.

## Build, Test, and Development Commands
- `make init`: install `uv` if missing and sync all dependency groups.
- `make init-no-project`: install `uv` if missing and sync dependency groups without installing the local project package.
- `make develop`: build/install the Python extension locally via `maturin --release`.
- `make develop-debug`: build/install the Python extension in debug mode (faster, CI-friendly).
- `make develop-release`: build/install the Python extension in release mode.
- `make init-node`: install locked root Node dependencies without running the package `prepare` build.
- `make build-node`: build the host N-API module in release mode.
- `make quality`: run Rust, Python, and Node quality checks.
- `make quality-python`: run Python quality checks only.
- `make quality-node`: install locked Node dependencies and type-check the generated declarations without compiling the native module.
- `make style`: apply auto-fixes (`cargo fix`, `clippy --fix`, `cargo fmt`, `ruff check --fix`, `ruff format`).
- `make test`: run Rust, Python, and Node tests.
- `make test-python`: run only the Python test suite under `tests/`.
- `make test-python-ci`: run Python tests against a debug extension build (CI target).
- `make test-node`: run Node API tests plus the commit-pinned Git-install contract.
- `make test-node-git-install`: test root package installation in a temporary npm consumer.

## CLI Binaries
Run binaries with `cargo run --release --bin <name> -- ...`:
- `dicom-preprocess <SOURCE> <OUTPUT> -s 384,512 -c`: preprocess DICOM to TIFF.
- `dicom-manifest <SOURCE_DIR> [OUTPUT.{csv|parquet}]`: generate TIFF manifest (default `manifest.parquet` in source).
- `dicom-validate <DICOM_FILE> [--format text|json]`: validate preprocessing-critical DICOM tags and report optional diagnostics.
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
- GitHub Actions owns Linux CI on the self-hosted `beryl` runner. The `Linux / Rust`, `Linux / Python`, and `Linux / Node` jobs combine each language's quality and runtime checks so setup and build caches are reused within the job.
- Linux CI runs for same-repository pull requests targeting `master`, pushes to `master`, exact semantic-version tags, manual dispatches, and nightly at `06:17 UTC`. Pull-request jobs test GitHub's synthetic merge result; the Node Git-install contract separately uses the remotely available pull-request head SHA. Fork pull requests are skipped; move trusted fork commits to a repository branch before running CI.
- The Rust job runs `cargo fmt -- --check`, Clippy with all workspace features and warnings denied, and Rust tests with all workspace features.
- The Python 3.13 job runs `make init-no-project`, `make quality-python`, and `make test-python-ci` with a debug extension build.
- The Node 24.13 job runs `make quality-node` and `make test-node`, including the debug JavaScript API tests and commit-pinned npm Git-install contract.
- CircleCI temporarily owns Windows x64 MSVC, macOS arm64, and Rosetta-backed macOS x64 Node validation. Migrate these jobs to GitHub Actions when suitable runners are available.
- CircleCI cross-platform jobs run automatically only for exact semantic-version tags. For an on-demand run, use **Trigger Pipeline** on the intended branch and set the Boolean pipeline parameter `run_cross_platform` to `true`.
- The required CircleCI schedule trigger is `weekly-cross-platform-master`: run every Sunday at `05:00 UTC` against `master`, with `run_cross_platform=true` and the scheduling system as actor. Ordinary pull requests and branch pushes run no CircleCI jobs.
- CI caches are lockfile-scoped and local to each runner or executor; jobs do not transfer build artifacts through a workspace.

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
