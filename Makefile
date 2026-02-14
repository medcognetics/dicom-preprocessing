UV=uv run
PYTHON=$(UV) python
PYTHON_QUALITY_TARGETS=tests examples dicom_preprocessing.pyi

.PHONY: init develop develop-debug develop-release quality quality-python style test-python test-python-ci test-python-pdb test

init:
	which uv || curl -LsSf https://astral.sh/uv/install.sh | sh
	uv sync --all-groups

develop: develop-release

develop-debug:
	uv run maturin develop --uv -F python

develop-release:
	uv run maturin develop --uv -F python --release

quality:
	cargo fmt -- --check
	cargo check --all-features
	cargo clippy --all-features -- -D warnings
	$(MAKE) quality-python

quality-python:
	$(UV) ruff format --check $(PYTHON_QUALITY_TARGETS)
	$(UV) ruff check $(PYTHON_QUALITY_TARGETS)
	$(UV) basedpyright

style:
	cargo fix --allow-dirty --all-features
	cargo clippy --all-features --fix --allow-dirty
	cargo fmt
	$(UV) ruff check --fix $(PYTHON_QUALITY_TARGETS)
	$(UV) ruff format $(PYTHON_QUALITY_TARGETS)

test-python: develop
	$(PYTHON) -m pytest \
		-rs \
		./tests/

test-python-ci: develop-debug
	$(PYTHON) -m pytest \
		-rs \
		./tests/

test-python-pdb: develop
	$(PYTHON) -m pytest \
		-rs \
		./tests/ \
		--pdb

test:
	cargo test
	$(MAKE) test-python

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
