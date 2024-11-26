PYTHON=pdm run python
PYTHON_DIRS=tests examples

init:
	which pdm || pip install --user pdm
	pdm venv create -n dicom-preprocessing
	pdm install -d

develop:
	maturin develop -F python --release

quality:
	$(PYTHON) -m black --check $(PYTHON_DIRS)
	$(PYTHON) -m autopep8 -a $(PYTHON_DIRS)

style:
	cargo fmt
	$(PYTHON) -m autoflake -r -i $(PYTHON_DIRS)
	$(PYTHON) -m isort $(PYTHON_DIRS)
	$(PYTHON) -m autopep8 -a $(PYTHON_DIRS)
	$(PYTHON) -m black $(PYTHON_DIRS)

test: 
	cargo test
	$(PYTHON) -m pytest \
		-rs \
		./tests/