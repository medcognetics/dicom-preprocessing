# Repository Guidelines

## Project Structure & Module Organization
Core Rust code lives in `src/`.
- CLI entry points: `src/main.rs` (`dicom-preprocess`) and `src/bin/*.rs` (manifest, combine, stats, trace, resize).
- Library modules: `src/transform/`, `src/metadata/`, `src/python/`, `src/errors/`.
- Python typing stub: `dicom_preprocessing.pyi`.
- Node package metadata and its lockfile: root `package.json` and `package-lock.json`; NAPI-RS source and generated entry points: `bindings/node/`.
- Dependency-health command and parsers: `scripts/ci/dependency_health.py`.

Tests are in `tests/` (pytest for Python bindings). Benchmarks live in `benches/`. Examples are in `examples/`, and documentation images in `docs/`.

## Build, Test, and Development Commands
- `make init`: install `uv` if missing and sync all dependency groups.
- `make init-no-project`: install `uv` if missing and sync dependency groups without installing the local project package.
- `make develop`: build/install the Python extension locally via `maturin --release`.
- `make develop-debug`: build/install the Python extension in debug mode (faster, CI-friendly).
- `make develop-release`: build/install the Python extension in release mode.
- `make build`: build an archive of release Rust binaries, a Python wheel, and a Node package tarball under `dist/`.
- `make init-node`: install locked root Node dependencies without running the package `prepare` build.
- `make build-node`: build the host N-API module in release mode without regenerating the checked-in supported-platform loader.
- `make quality`: run Rust, Python, and Node quality checks.
- `make quality-python`: run Python quality checks only.
- `make quality-node`: install locked Node dependencies and type-check the generated declarations without compiling the native module.
- `make style`: apply auto-fixes (`cargo fix`, `clippy --fix`, `cargo fmt`, `ruff check --fix`, `ruff format`).
- `make test`: run Rust, Python, and Node tests.
- `make test-python`: run only the Python test suite under `tests/`.
- `make test-python-ci`: run Python tests against a debug extension build (CI target).
- `make test-node-direct`: build the debug Node binding and run its Rust fixture, type, and JavaScript API tests.
- `make test-node`: run direct Node tests plus the commit-pinned Git-install contract.
- `make test-node-git-install`: test root package installation in a temporary npm consumer.
- `make test-build`: verify artifacts previously created by `make build`, including the full Node Git-install contract.

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
- Ruff is the formatter/import sorter and is configured in `pyproject.toml` (`line-length = 120`, `target-version = "py310"`).
- basedpyright type checking is configured in `pyproject.toml` (`pythonVersion = "3.10"`, `typeCheckingMode = "basic"`).
- Test files use `test_*.py`; test functions use `test_*`.

## CI Quality Pipeline
- GitHub Actions owns all CI. CircleCI is retired.
- Linux CI runs for pull requests targeting `master`, pushes to `master`, exact semantic-version tags, and manual dispatches. Pull-request jobs test GitHub's synthetic merge result.
- Same-repository Linux jobs use the `[self-hosted, linux, x64, beryl]` runner labels. `beryl` provisions one Ubuntu x64 CPU node at a time, assigns exactly one job, and destroys the node and local storage afterward. Jobs must install every toolchain and dependency they use and must not rely on prior runner state.
- Fork pull requests use `ubuntu-24.04` instead of `beryl`, preserving the same required-check names without exposing self-hosted infrastructure.
- `Linux / Python`, `Linux / Node`, and the hosted `Linux / Minimum versions` job require `Linux / Rust`. With one `beryl` node, Python and Node queue independently after Rust; the hosted minimum-version job may run concurrently.
- `Linux / Rust` uses Rust 1.97.1 and runs `make quality-rust` and `make test-rust`. `Linux / Python` tests Python 3.14 with NumPy 2.4.6. `Linux / Node` tests Node 24.18 and 26.5, with quality checks on Node 26.5.
- `Linux / Minimum versions` runs on `ubuntu-24.04` and tests Rust 1.89.0, Python 3.10 with NumPy 2.2.6, and Node 22.13. Regular CI uses direct tests and does not run the commit-pinned Git-install contract.
- The `Nightly Build` workflow runs daily at `06:17 UTC`, for exact semantic-version tags, and by manual dispatch. Its `Nightly / Build and install` job uses `beryl`, runs `make build` and `make test-build`, and verifies the full commit-pinned npm `install` and `ci` pathways. Artifacts and the clean job-scoped Cargo target live only under the ephemeral runner's temporary directory, and no output is uploaded.
- The `Cross-platform CI` workflow runs Windows x64 on `windows-2022` and native macOS arm64 on `macos-15` every Sunday at `05:17 UTC`, for exact semantic-version tags, and by manual dispatch. It tests Node 26.5 and the full Git-install contract; Windows also runs the focused file-identifier test.
- The independent `Dependency Health` workflow runs every Monday at `07:17 UTC` and by manual dispatch. `Dependency Health / Security Audit` fails on unsuppressed Rust, Python, Node, or workflow-security findings and on incomplete scans. `Dependency Health / Deprecation Report` reports unmaintained, yanked, deprecated, future-incompatible, and runtime-lifecycle findings without failing on findings; command or parse failures still fail the job.
- Linux and nightly jobs do not use Actions caches. Scheduled hosted cross-platform jobs cache only lockfile-scoped npm and Cargo downloads, never Rust build targets. No workflow uploads retained artifacts.

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
