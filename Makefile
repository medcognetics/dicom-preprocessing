UV=uv
UV_RUN=$(UV) run
UV_NO_PROJECT=$(UV) run --no-project
UV_SYNC_ALL_GROUPS=$(UV) sync --locked --all-groups
VENV=.venv
VENV_BIN=$(VENV)/bin
MATURIN=$(VENV_BIN)/maturin
MATURIN_FEATURES=-F python -F pyo3/extension-module
PYTHON=$(UV_RUN) python
PYTHON_QUALITY_TARGETS=tests examples scripts/ci dicom_preprocessing.pyi
PYTEST_ARGS=-rs ./tests/
NPM=npm
ARTIFACT_DIR?=dist
RUST_ARTIFACT_DIR=$(ARTIFACT_DIR)/rust
PYTHON_ARTIFACT_DIR=$(ARTIFACT_DIR)/python
NODE_ARTIFACT_DIR=$(ARTIFACT_DIR)/node
CARGO_TARGET_DIR?=target
RUST_TARGET?=$(shell rustc -vV | sed -n 's/^host: //p')
RUST_RELEASE_DIR=$(CARGO_TARGET_DIR)/$(RUST_TARGET)/release
RUST_PACKAGE=dicom-preprocessing-cli-$(RUST_TARGET).tar.gz
PYTHON_BUILD_VERSION?=3.14
PYTHON_BUILD_INTERPRETER?=$(shell $(UV) python find $(PYTHON_BUILD_VERSION))
RUST_BINARIES=dicom-preprocess dicom-manifest dicom-voilut dicom-validate dicom-traces tiff-combine tiff-stats resize

.PHONY: init init-no-project init-node ensure-uv develop develop-debug develop-release build build-rust build-python build-node build-node-package quality quality-rust quality-python quality-node style test-rust test-python test-python-ci test-python-pdb test-node test-node-direct test-node-package-install test-node-git-install test-build test-rust-artifacts test-python-wheel test

ensure-uv:
	which $(UV) || curl -LsSf https://astral.sh/uv/install.sh | sh

$(MATURIN): pyproject.toml uv.lock | ensure-uv
	$(UV_SYNC_ALL_GROUPS) --no-install-project

init: ensure-uv
	$(UV_SYNC_ALL_GROUPS)

init-no-project: ensure-uv
	$(UV_SYNC_ALL_GROUPS) --no-install-project

init-node:
	$(NPM) ci --ignore-scripts

develop: develop-release

develop-debug: $(MATURIN)
	$(MATURIN) develop --uv $(MATURIN_FEATURES)

develop-release: $(MATURIN)
	$(MATURIN) develop --uv $(MATURIN_FEATURES) --release

build: build-rust build-python build-node-package

build-rust:
	mkdir -p $(RUST_ARTIFACT_DIR)
	rm -f $(RUST_ARTIFACT_DIR)/*.tar.gz
	cargo build --locked --workspace --release --target $(RUST_TARGET)
	tar -czf $(RUST_ARTIFACT_DIR)/$(RUST_PACKAGE) -C $(RUST_RELEASE_DIR) $(RUST_BINARIES)

build-python: $(MATURIN)
	mkdir -p $(PYTHON_ARTIFACT_DIR)
	rm -f $(PYTHON_ARTIFACT_DIR)/*.whl
	$(MATURIN) build --locked $(MATURIN_FEATURES) --release --target $(RUST_TARGET) --interpreter $(PYTHON_BUILD_INTERPRETER) --out $(PYTHON_ARTIFACT_DIR)

quality: quality-rust quality-python quality-node

quality-rust:
	cargo fmt -- --check
	cargo check --locked --workspace --all-features
	cargo clippy --locked --workspace --all-features --all-targets -- -D warnings

quality-python:
	$(UV_NO_PROJECT) ruff format --check $(PYTHON_QUALITY_TARGETS)
	$(UV_NO_PROJECT) ruff check $(PYTHON_QUALITY_TARGETS)
	$(UV_NO_PROJECT) basedpyright

build-node:
	$(NPM) run build

build-node-package: init-node
	mkdir -p $(NODE_ARTIFACT_DIR)
	$(NPM) run build
	$(NPM) pack --ignore-scripts --pack-destination $(NODE_ARTIFACT_DIR)

quality-node: init-node
	$(NPM) run typecheck

style:
	cargo fix --allow-dirty --all-features
	cargo clippy --all-features --fix --allow-dirty
	cargo fmt
	$(UV_RUN) ruff check --fix $(PYTHON_QUALITY_TARGETS)
	$(UV_RUN) ruff format $(PYTHON_QUALITY_TARGETS)

test-python: develop
	$(PYTHON) -m pytest $(PYTEST_ARGS)

test-python-ci: develop-debug
	$(PYTHON) -m pytest $(PYTEST_ARGS)

test-python-pdb: develop
	$(PYTHON) -m pytest $(PYTEST_ARGS) --pdb

test: test-rust test-python test-node

test-rust:
	cargo test --locked --workspace --all-features

test-node: test-node-direct test-node-git-install

test-node-direct: init-node
	cargo test -p dicom-preprocessing-node ensure_node_fixtures
	DICOM_PREPROCESSING_CT_FIXTURE="$(CURDIR)/target/dicom_test_files/pydicom/CT_small.dcm" \
	DICOM_PREPROCESSING_MULTIFRAME_FIXTURE="$(CURDIR)/target/dicom_test_files/pydicom/emri_small.dcm" \
	DICOM_PREPROCESSING_RGB_FIXTURE="$(CURDIR)/target/dicom_test_files/pydicom/SC_rgb.dcm" \
	$(NPM) test

test-node-package-install:
	@package="$$(find "$(NODE_ARTIFACT_DIR)" -maxdepth 1 -type f -name '*.tgz' -print -quit)"; \
	test -n "$$package"; \
	DICOM_PREPROCESSING_PACKAGE_PATH="$$package" $(NPM) run test:package-install

test-node-git-install:
	$(NPM) run test:git-install

test-build: test-rust-artifacts test-python-wheel test-node-package-install test-node-git-install

test-rust-artifacts:
	@archive="$$(find "$(RUST_ARTIFACT_DIR)" -maxdepth 1 -type f -name '*.tar.gz' -print -quit)"; \
	test -n "$$archive"; \
	test_env="$$(mktemp -d)"; \
	trap 'rm -rf "$$test_env"' EXIT; \
	tar -xzf "$$archive" -C "$$test_env"; \
	for binary in $(RUST_BINARIES); do \
		"$$test_env/$$binary" --help >/dev/null; \
	done

test-python-wheel:
	@wheel="$$(find "$(PYTHON_ARTIFACT_DIR)" -maxdepth 1 -type f -name '*.whl' -print -quit)"; \
	test -n "$$wheel"; \
	test_env="$$(mktemp -d)"; \
	trap 'rm -rf "$$test_env"' EXIT; \
	$(UV) venv --python $(PYTHON_BUILD_VERSION) "$$test_env"; \
	$(UV) pip install --python "$$test_env/bin/python" "$$wheel"; \
	"$$test_env/bin/python" -c 'import dicom_preprocessing'

# Docs image generation recipe.
# NOTE: The lesion crop coordinates were manually determined for the specific source DICOM
# used to generate these images. Coordinates are given in 384x512 space, then scaled 4x to
# 1536x2048 for high-res crops.
# - Lesion 1: center (18, 165) -> (72, 660) at 4x, crop at (0, 532)
# - Lesion 2: center (58, 185) -> (232, 740) at 4x, crop at (104, 612)
.PHONY: docs-images

docs-images:
ifndef SOURCE
	$(error SOURCE is required. Usage: make docs-images SOURCE=/path/to/file.dcm)
endif
	# Full-size images at 384x512
	cargo run --release --bin dicom-preprocess -- $(SOURCE) /tmp/central_slice.tiff -s 384,512 -v central-slice -c -m
	convert /tmp/central_slice.tiff docs/central_slice.png
	cargo run --release --bin dicom-preprocess -- $(SOURCE) /tmp/max_intensity.tiff -s 384,512 -v max-intensity -c -m
	convert /tmp/max_intensity.tiff docs/max_intensity.png
	cargo run --release --bin dicom-preprocess -- $(SOURCE) /tmp/laplacian_mip.tiff -s 384,512 -v laplacian-mip -c -m
	convert /tmp/laplacian_mip.tiff docs/laplacian_mip.png
	# High-res images at 1536x2048 (4x), then crop 256x256 lesion region
	cargo run --release --bin dicom-preprocess -- $(SOURCE) /tmp/central_slice_hires.tiff -s 1536,2048 -v central-slice -c -m
	convert /tmp/central_slice_hires.tiff -crop 256x256+0+532 +repage docs/central_slice_crop.png
	cargo run --release --bin dicom-preprocess -- $(SOURCE) /tmp/max_intensity_hires.tiff -s 1536,2048 -v max-intensity -c -m
	convert /tmp/max_intensity_hires.tiff -crop 256x256+0+532 +repage docs/max_intensity_crop.png
	cargo run --release --bin dicom-preprocess -- $(SOURCE) /tmp/laplacian_mip_hires.tiff -s 1536,2048 -v laplacian-mip -c -m
	convert /tmp/laplacian_mip_hires.tiff -crop 256x256+0+532 +repage docs/laplacian_mip_crop.png
	# Lesion 2 crops (reusing high-res TIFFs)
	convert /tmp/central_slice_hires.tiff -crop 256x256+104+612 +repage docs/central_slice_crop2.png
	convert /tmp/max_intensity_hires.tiff -crop 256x256+104+612 +repage docs/max_intensity_crop2.png
	convert /tmp/laplacian_mip_hires.tiff -crop 256x256+104+612 +repage docs/laplacian_mip_crop2.png
	rm -f /tmp/central_slice.tiff /tmp/max_intensity.tiff /tmp/laplacian_mip.tiff
	rm -f /tmp/central_slice_hires.tiff /tmp/max_intensity_hires.tiff /tmp/laplacian_mip_hires.tiff
