PYTHON=uv run python
PYTHON_DIRS=tests examples dicom_preprocessing.pyi

.PHONY: init develop quality style test-python test-python-pdb test

init:
	which uv || curl -LsSf https://astral.sh/uv/install.sh | sh
	uv sync --all-groups

develop:
	uv run maturin develop -F python --release

quality:
	cargo fmt -- --check
	cargo check --all-features
	cargo clippy --all-features -- -D warnings
	$(PYTHON) -m black --check $(PYTHON_DIRS)
	$(PYTHON) -m autopep8 -a $(PYTHON_DIRS)

style:
	cargo fix --allow-dirty --all-features
	cargo clippy --all-features --fix --allow-dirty
	cargo fmt
	$(PYTHON) -m autoflake -r -i $(PYTHON_DIRS)
	$(PYTHON) -m isort $(PYTHON_DIRS)
	$(PYTHON) -m autopep8 -a $(PYTHON_DIRS)
	$(PYTHON) -m black $(PYTHON_DIRS)

test-python: develop
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
	# Central slice projection mode (for comparison; default parallel-beam is generated above as laplacian_mip)
	cargo run --release --bin dicom-preprocess -- $(SOURCE) /tmp/laplacian_mip_central.tiff -s 384,512 -v laplacian-mip --projection-mode central-slice -c -m
	convert /tmp/laplacian_mip_central.tiff docs/laplacian_mip_central.png
	cargo run --release --bin dicom-preprocess -- $(SOURCE) /tmp/laplacian_mip_central_hires.tiff -s 1536,2048 -v laplacian-mip --projection-mode central-slice -c -m
	convert /tmp/laplacian_mip_central_hires.tiff -crop 256x256+0+532 +repage docs/laplacian_mip_central_crop.png
	convert /tmp/laplacian_mip_central_hires.tiff -crop 256x256+104+612 +repage docs/laplacian_mip_central_crop2.png
	rm -f /tmp/central_slice.tiff /tmp/max_intensity.tiff /tmp/laplacian_mip.tiff
	rm -f /tmp/central_slice_hires.tiff /tmp/max_intensity_hires.tiff /tmp/laplacian_mip_hires.tiff
	rm -f /tmp/laplacian_mip_central.tiff /tmp/laplacian_mip_central_hires.tiff

# Sweep mip_weight for each projection mode to find optimal defaults.
# Generates montages of full images and lesion crops at different weights.
# Output: /tmp/sweep_<mode>_full.png, /tmp/sweep_<mode>_crops.png, /tmp/sweep_all.png
.PHONY: sweep-mip-weight

SWEEP_WEIGHTS := 0.5 1.0 1.5 2.0 3.0 5.0
SWEEP_MODES := central-slice parallel-beam

sweep-mip-weight:
ifndef SOURCE
	$(error SOURCE is required. Usage: make sweep-mip-weight SOURCE=/path/to/file.dcm)
endif
	@mkdir -p /tmp/sweep
	@for mode in $(SWEEP_MODES); do \
		fulls=""; \
		crops=""; \
		for w in $(SWEEP_WEIGHTS); do \
			echo "Rendering $$mode w=$$w"; \
			cargo run --release --bin dicom-preprocess -- $(SOURCE) /tmp/sweep/$${mode}_w$${w}_full.tiff \
				-s 384,512 -v laplacian-mip --projection-mode $$mode --mip-weight $$w -c -m; \
			convert /tmp/sweep/$${mode}_w$${w}_full.tiff \
				-font Helvetica -pointsize 14 -gravity South -splice 0x20 -annotate +0+4 "w=$$w" \
				/tmp/sweep/$${mode}_w$${w}_full.png; \
			fulls="$$fulls /tmp/sweep/$${mode}_w$${w}_full.png"; \
			cargo run --release --bin dicom-preprocess -- $(SOURCE) /tmp/sweep/$${mode}_w$${w}_hires.tiff \
				-s 1536,2048 -v laplacian-mip --projection-mode $$mode --mip-weight $$w -c -m; \
			convert /tmp/sweep/$${mode}_w$${w}_hires.tiff -crop 256x256+0+532 +repage /tmp/sweep/$${mode}_w$${w}_crop1.png; \
			convert /tmp/sweep/$${mode}_w$${w}_hires.tiff -crop 256x256+104+612 +repage /tmp/sweep/$${mode}_w$${w}_crop2.png; \
			convert /tmp/sweep/$${mode}_w$${w}_crop1.png /tmp/sweep/$${mode}_w$${w}_crop2.png -append \
				-font Helvetica -pointsize 14 -gravity South -splice 0x20 -annotate +0+4 "w=$$w" \
				/tmp/sweep/$${mode}_w$${w}_col.png; \
			crops="$$crops /tmp/sweep/$${mode}_w$${w}_col.png"; \
		done; \
		montage $$fulls -tile 6x1 -geometry +2+2 -label "" /tmp/sweep_$${mode}_full.png; \
		convert /tmp/sweep_$${mode}_full.png -font Helvetica -pointsize 20 \
			-gravity North -splice 0x30 -annotate +0+5 "$$mode (full)" /tmp/sweep_$${mode}_full.png; \
		montage $$crops -tile 6x1 -geometry +2+2 -label "" /tmp/sweep_$${mode}_crops.png; \
		convert /tmp/sweep_$${mode}_crops.png -font Helvetica -pointsize 20 \
			-gravity North -splice 0x30 -annotate +0+5 "$$mode (crops)" /tmp/sweep_$${mode}_crops.png; \
		echo "=> /tmp/sweep_$${mode}_full.png"; \
		echo "=> /tmp/sweep_$${mode}_crops.png"; \
	done
	# Stack all modes vertically: full images then crops for each mode
	convert \
		/tmp/sweep_central-slice_full.png /tmp/sweep_central-slice_crops.png \
		/tmp/sweep_parallel-beam_full.png /tmp/sweep_parallel-beam_crops.png \
		-append /tmp/sweep_all.png
	@echo "=> /tmp/sweep_all.png"
	rm -rf /tmp/sweep