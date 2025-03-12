import shutil
from pathlib import Path

import dicom_preprocessing as dp
import numpy as np
import pydicom
import pytest


def test_preprocessor():
    preprocessor = dp.Preprocessor()
    assert isinstance(repr(preprocessor), str)


@pytest.fixture(params=[Path, str])
def dicom_path(request, tmp_path):
    path = tmp_path / "test.dcm"
    source = pydicom.data.get_testdata_file("CT_small.dcm")
    shutil.copy(source, path)
    if request.param == Path:
        return path
    elif request.param == str:
        return str(path)
    else:
        raise ValueError(f"Invalid parameter: {request.param}")


@pytest.fixture
def dicom_stream(tmp_path):
    path = tmp_path / "test.dcm"
    source = pydicom.data.get_testdata_file("CT_small.dcm")
    shutil.copy(source, path)
    with open(path, "rb") as f:
        return f.read()


def test_preprocess_u8(dicom_path):
    preprocessor = dp.Preprocessor(size=(32, 32))
    result = dp.preprocess_u8(dicom_path, preprocessor)
    assert result.shape == (1, 32, 32, 1)
    assert result.dtype == np.uint8
    assert result.min() >= 0
    assert result.max() <= 255


def test_preprocess_u16(dicom_path):
    preprocessor = dp.Preprocessor(size=(32, 32))
    result = dp.preprocess_u16(dicom_path, preprocessor)
    assert result.shape == (1, 32, 32, 1)
    assert result.dtype == np.uint16
    assert result.min() >= 0
    assert result.max() <= 65535


def test_preprocess_f32(dicom_path):
    preprocessor = dp.Preprocessor(size=(32, 32))
    result = dp.preprocess_f32(dicom_path, preprocessor)
    assert result.shape == (1, 32, 32, 1)
    assert result.dtype == np.float32
    assert result.min() >= 0
    assert result.max() <= 1


def test_preprocess_u8_stream(dicom_stream):
    preprocessor = dp.Preprocessor(size=(32, 32))
    result = dp.preprocess_stream_u8(dicom_stream, preprocessor)
    assert result.shape == (1, 32, 32, 1)
    assert result.dtype == np.uint8
    assert result.min() >= 0
    assert result.max() <= 255


def test_preprocess_u16_stream(dicom_stream):
    preprocessor = dp.Preprocessor(size=(32, 32))
    result = dp.preprocess_stream_u16(dicom_stream, preprocessor)
    assert result.shape == (1, 32, 32, 1)
    assert result.dtype == np.uint16
    assert result.min() >= 0
    assert result.max() <= 65535


def test_preprocess_f32_stream(dicom_stream):
    preprocessor = dp.Preprocessor(size=(32, 32))
    result = dp.preprocess_stream_f32(dicom_stream, preprocessor)
    assert result.shape == (1, 32, 32, 1)
    assert result.dtype == np.float32
    assert result.min() >= 0
    assert result.max() <= 1
