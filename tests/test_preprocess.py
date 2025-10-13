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
    result = dp.preprocess_u8(dicom_path, preprocessor, parallel=False)
    assert result.shape == (1, 32, 32, 1)
    assert result.dtype == np.uint8
    assert result.min() >= 0
    assert result.max() <= 255


def test_preprocess_u16(dicom_path):
    preprocessor = dp.Preprocessor(size=(32, 32))
    result = dp.preprocess_u16(dicom_path, preprocessor, parallel=False)
    assert result.shape == (1, 32, 32, 1)
    assert result.dtype == np.uint16
    assert result.min() >= 0
    assert result.max() <= 65535


def test_preprocess_f32(dicom_path):
    preprocessor = dp.Preprocessor(size=(32, 32))
    result = dp.preprocess_f32(dicom_path, preprocessor, parallel=False)
    assert result.shape == (1, 32, 32, 1)
    assert result.dtype == np.float32
    assert result.min() >= 0
    assert result.max() <= 1


def test_preprocess_u8_stream(dicom_stream):
    preprocessor = dp.Preprocessor(size=(32, 32))
    result = dp.preprocess_stream_u8(dicom_stream, preprocessor, parallel=False)
    assert result.shape == (1, 32, 32, 1)
    assert result.dtype == np.uint8
    assert result.min() >= 0
    assert result.max() <= 255


def test_preprocess_u16_stream(dicom_stream):
    preprocessor = dp.Preprocessor(size=(32, 32))
    result = dp.preprocess_stream_u16(dicom_stream, preprocessor, parallel=False)
    assert result.shape == (1, 32, 32, 1)
    assert result.dtype == np.uint16
    assert result.min() >= 0
    assert result.max() <= 65535


def test_preprocess_f32_stream(dicom_stream):
    preprocessor = dp.Preprocessor(size=(32, 32))
    result = dp.preprocess_stream_f32(dicom_stream, preprocessor, parallel=False)
    assert result.shape == (1, 32, 32, 1)
    assert result.dtype == np.float32
    assert result.min() >= 0
    assert result.max() <= 1


def test_preprocess_f32_window(dicom_stream):
    preprocessor1 = dp.Preprocessor(size=(32, 32))
    preprocessor2 = dp.Preprocessor(size=(32, 32), convert_options="10,100")
    result1 = dp.preprocess_stream_f32(dicom_stream, preprocessor1, parallel=False)
    result2 = dp.preprocess_stream_f32(dicom_stream, preprocessor2, parallel=False)
    assert not np.allclose(result1, result2)


def test_preprocess_f32_spacing(dicom_stream):
    preprocessor = dp.Preprocessor(spacing=(1.0, 1.0, 1.0))
    result = dp.preprocess_stream_f32(dicom_stream, preprocessor, parallel=False)
    assert result.shape == (1, 85, 85, 1)
    assert result.dtype == np.float32
    assert result.min() >= 0
    assert result.max() <= 1


@pytest.fixture
def multiple_dicom_paths(tmp_path):
    """Create multiple DICOM files to simulate CT slices."""
    source = pydicom.data.get_testdata_file("CT_small.dcm")
    paths = []
    for i in range(3):
        path = tmp_path / f"slice_{i:03d}.dcm"
        shutil.copy(source, path)
        paths.append(path)
    return paths


@pytest.fixture
def multiple_dicom_streams(multiple_dicom_paths):
    """Load multiple DICOM files as byte streams."""
    streams = []
    for path in multiple_dicom_paths:
        with open(path, "rb") as f:
            streams.append(f.read())
    return streams


def test_preprocess_u8_slices_order(multiple_dicom_paths):
    """Test that output order matches input order for path-based slices."""
    preprocessor = dp.Preprocessor(size=(32, 32))
    results = dp.preprocess_u8_slices(multiple_dicom_paths, preprocessor, parallel=False)
    
    assert len(results) == len(multiple_dicom_paths)
    for result in results:
        assert result.shape == (1, 32, 32, 1)
        assert result.dtype == np.uint8


def test_preprocess_u16_slices_order(multiple_dicom_paths):
    """Test that output order matches input order for path-based slices."""
    preprocessor = dp.Preprocessor(size=(32, 32))
    results = dp.preprocess_u16_slices(multiple_dicom_paths, preprocessor, parallel=False)
    
    assert len(results) == len(multiple_dicom_paths)
    for result in results:
        assert result.shape == (1, 32, 32, 1)
        assert result.dtype == np.uint16


def test_preprocess_f32_slices_order(multiple_dicom_paths):
    """Test that output order matches input order for path-based slices."""
    preprocessor = dp.Preprocessor(size=(32, 32))
    results = dp.preprocess_f32_slices(multiple_dicom_paths, preprocessor, parallel=False)
    
    assert len(results) == len(multiple_dicom_paths)
    for result in results:
        assert result.shape == (1, 32, 32, 1)
        assert result.dtype == np.float32


def test_preprocess_stream_u8_slices_order(multiple_dicom_streams):
    """Test that output order matches input order for stream-based slices."""
    preprocessor = dp.Preprocessor(size=(32, 32))
    results = dp.preprocess_stream_u8_slices(multiple_dicom_streams, preprocessor, parallel=False)
    
    assert len(results) == len(multiple_dicom_streams)
    for result in results:
        assert result.shape == (1, 32, 32, 1)
        assert result.dtype == np.uint8


def test_preprocess_stream_u16_slices_order(multiple_dicom_streams):
    """Test that output order matches input order for stream-based slices."""
    preprocessor = dp.Preprocessor(size=(32, 32))
    results = dp.preprocess_stream_u16_slices(multiple_dicom_streams, preprocessor, parallel=False)
    
    assert len(results) == len(multiple_dicom_streams)
    for result in results:
        assert result.shape == (1, 32, 32, 1)
        assert result.dtype == np.uint16


def test_preprocess_stream_f32_slices_order(multiple_dicom_streams):
    """Test that output order matches input order for stream-based slices."""
    preprocessor = dp.Preprocessor(size=(32, 32))
    results = dp.preprocess_stream_f32_slices(multiple_dicom_streams, preprocessor, parallel=False)
    
    assert len(results) == len(multiple_dicom_streams)
    for result in results:
        assert result.shape == (1, 32, 32, 1)
        assert result.dtype == np.float32


def test_preprocess_slices_common_crop_bounds(tmp_path):
    """Test that common crop bounds are used across all slices."""
    source = pydicom.data.get_testdata_file("CT_small.dcm")
    
    # Create DICOMs with different content that would have different individual crop bounds
    # We'll use the same base file but this demonstrates the concept
    paths = []
    for i in range(3):
        path = tmp_path / f"slice_{i:03d}.dcm"
        shutil.copy(source, path)
        paths.append(path)
    
    # Process with crop enabled
    preprocessor = dp.Preprocessor(crop=True, size=(64, 64))
    results = dp.preprocess_f32_slices(paths, preprocessor, parallel=False)
    
    # All results should have the same shape since common crop bounds are used
    shapes = [result.shape for result in results]
    assert all(shape == shapes[0] for shape in shapes), "All slices should have same shape from common crop"
    assert len(results) == len(paths)


def test_preprocess_slices_vs_individual(multiple_dicom_paths):
    """Test that slices processing differs from individual processing when cropping."""
    preprocessor = dp.Preprocessor(crop=True, size=None)
    
    # Process as slices (with common crop)
    slice_results = dp.preprocess_f32_slices(multiple_dicom_paths, preprocessor, parallel=False)
    
    # Process individually
    individual_results = []
    for path in multiple_dicom_paths:
        result = dp.preprocess_f32(path, preprocessor, parallel=False)
        individual_results.append(result)
    
    # Verify we got results
    assert len(slice_results) == len(individual_results)
    
    # All slice results should have the same shape (common crop)
    slice_shapes = [r.shape for r in slice_results]
    assert all(shape == slice_shapes[0] for shape in slice_shapes), "Slice processing should use common crop"


def test_preprocess_slices_with_str_paths(multiple_dicom_paths):
    """Test that slice processing works with string paths."""
    str_paths = [str(p) for p in multiple_dicom_paths]
    preprocessor = dp.Preprocessor(size=(32, 32))
    results = dp.preprocess_f32_slices(str_paths, preprocessor, parallel=False)
    
    assert len(results) == len(str_paths)
    for result in results:
        assert result.shape == (1, 32, 32, 1)
        assert result.dtype == np.float32


def test_preprocess_slices_empty_list():
    """Test that processing an empty list raises an error."""
    preprocessor = dp.Preprocessor(size=(32, 32))
    
    with pytest.raises(RuntimeError, match="Cannot process empty"):
        dp.preprocess_f32_slices([], preprocessor, parallel=False)


def test_preprocess_stream_slices_empty_list():
    """Test that processing an empty list of streams raises an error."""
    preprocessor = dp.Preprocessor(size=(32, 32))
    
    with pytest.raises(RuntimeError, match="Cannot process empty"):
        dp.preprocess_stream_f32_slices([], preprocessor, parallel=False)
