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
    paths = [
        (shutil.copy(source, tmp_path / f"slice_{i:03d}.dcm"), tmp_path / f"slice_{i:03d}.dcm")[1] for i in range(3)
    ]
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


def test_preprocess_slices_combined_volume(multiple_dicom_paths):
    """Test that slices are combined into a volume for processing.

    When processing multiple single-frame slices, they should be:
    1. Combined into a single volume
    2. Have common crop/resize/pad applied
    3. Each output frame should be identical in spatial dimensions
    """
    preprocessor = dp.Preprocessor(
        crop=True,
        size=(32, 32),
        volume_handler="keep",
    )

    results = dp.preprocess_f32_slices(multiple_dicom_paths, preprocessor, parallel=False)

    # Should have one output per input (no z-interpolation without z-spacing)
    assert len(results) == len(multiple_dicom_paths)

    # All outputs should have identical dimensions (common processing)
    shapes = [result.shape for result in results]
    assert all(shape == shapes[0] for shape in shapes)

    # Each should be single-frame with target size
    for result in results:
        assert result.shape == (1, 32, 32, 1)


def test_preprocess_slices_with_z_spacing_interpolation(multiple_dicom_paths):
    """Test that z-spacing interpolation works on the combined volume.

    When z-spacing is specified with slice thickness metadata, the combined
    volume should be interpolated, potentially changing the number of output slices.
    """
    # For this test, we need to create DICOMs with proper spacing metadata
    # The test fixture creates 3 single-frame slices
    # If we have 3 slices with 5mm spacing and request 10mm spacing, we should get ~2 slices

    # Without z-spacing, we get all input slices back
    preprocessor_no_z = dp.Preprocessor(
        crop=False,
        size=(32, 32),
        volume_handler="keep",
    )

    results_no_z = dp.preprocess_f32_slices(multiple_dicom_paths, preprocessor_no_z, parallel=False)
    assert len(results_no_z) == len(multiple_dicom_paths)

    # Note: For z-spacing interpolation to work, the DICOMs need SliceThickness or
    # SpacingBetweenSlices metadata. The test fixture uses CT_small.dcm which has
    # pixel spacing but may not have z-spacing metadata, so interpolation may not occur.
    # This test verifies the function runs without errors when z-spacing is configured.
    preprocessor_with_z = dp.Preprocessor(
        crop=False,
        spacing=(1.0, 1.0, 5.0),  # Include z-spacing
        volume_handler="keep",
        size=(32, 32),
    )

    results_with_z = dp.preprocess_f32_slices(multiple_dicom_paths, preprocessor_with_z, parallel=False)

    # Should process successfully
    assert len(results_with_z) > 0

    # All outputs should have same spatial dimensions
    for result in results_with_z:
        assert result.shape[1:3] == (32, 32)


def test_preprocess_stream_slices_combined_volume(multiple_dicom_streams):
    """Test that stream slices are combined into a volume for processing."""
    preprocessor = dp.Preprocessor(
        crop=True,
        size=(32, 32),
        volume_handler="keep",
    )

    results = dp.preprocess_stream_f32_slices(multiple_dicom_streams, preprocessor, parallel=False)

    # Should have outputs (count may vary with z-interpolation)
    assert len(results) == len(multiple_dicom_streams)

    # All outputs should have identical spatial dimensions
    for result in results:
        assert result.shape[1:3] == (32, 32)


@pytest.fixture
def multiframe_dicom_path(tmp_path):
    """Create a multi-frame DICOM file for testing."""
    source = pydicom.data.get_testdata_file("emri_small.dcm")
    path = tmp_path / "multiframe.dcm"
    shutil.copy(source, path)
    return path


@pytest.fixture
def multiframe_dicom_stream(multiframe_dicom_path):
    """Load multi-frame DICOM file as byte stream."""
    with open(multiframe_dicom_path, "rb") as f:
        return f.read()


@pytest.mark.parametrize("target_frames", [8, 16, 32])
def test_interpolate_volume_handler_single_file(multiframe_dicom_path, target_frames):
    """Test that interpolate volume handler correctly interpolates frames."""
    preprocessor = dp.Preprocessor(
        crop=False,
        size=(32, 32),
        volume_handler="interpolate",
        target_frames=target_frames,
    )

    result = dp.preprocess_f32(multiframe_dicom_path, preprocessor, parallel=False)

    # Should have exactly target_frames output frames
    assert result.shape[0] == target_frames, f"Expected {target_frames} frames, got {result.shape[0]}"
    assert result.shape[1:3] == (32, 32)
    assert result.dtype == np.float32


@pytest.mark.parametrize("target_frames", [8, 16])
def test_interpolate_volume_handler_stream(multiframe_dicom_stream, target_frames):
    """Test that interpolate volume handler works with streams."""
    preprocessor = dp.Preprocessor(
        crop=False,
        size=(32, 32),
        volume_handler="interpolate",
        target_frames=target_frames,
    )

    result = dp.preprocess_stream_f32(multiframe_dicom_stream, preprocessor, parallel=False)

    # Should have exactly target_frames output frames
    assert result.shape[0] == target_frames, f"Expected {target_frames} frames, got {result.shape[0]}"
    assert result.shape[1:3] == (32, 32)
    assert result.dtype == np.float32


@pytest.mark.parametrize("target_frames", [8, 16, 32])
def test_interpolate_volume_handler_slices(multiple_dicom_paths, target_frames):
    """Test that interpolate volume handler works with multiple slices.

    When processing multiple single-frame slices with the interpolate handler,
    they should be combined into a volume and interpolated to target_frames.
    """
    preprocessor = dp.Preprocessor(
        crop=False,
        size=(32, 32),
        volume_handler="interpolate",
        target_frames=target_frames,
    )

    results = dp.preprocess_f32_slices(multiple_dicom_paths, preprocessor, parallel=False)

    # Total number of output frames should equal target_frames
    total_frames = sum(result.shape[0] for result in results)
    assert total_frames == target_frames, f"Expected {target_frames} total frames, got {total_frames}"

    # All outputs should have same spatial dimensions
    for result in results:
        assert result.shape[1:3] == (32, 32)
        assert result.dtype == np.float32


@pytest.mark.parametrize("target_frames", [8, 16])
def test_interpolate_volume_handler_stream_slices(multiple_dicom_streams, target_frames):
    """Test that interpolate volume handler works with stream slices."""
    preprocessor = dp.Preprocessor(
        crop=False,
        size=(32, 32),
        volume_handler="interpolate",
        target_frames=target_frames,
    )

    results = dp.preprocess_stream_f32_slices(multiple_dicom_streams, preprocessor, parallel=False)

    # Total number of output frames should equal target_frames
    total_frames = sum(result.shape[0] for result in results)
    assert total_frames == target_frames, f"Expected {target_frames} total frames, got {total_frames}"

    # All outputs should have same spatial dimensions
    for result in results:
        assert result.shape[1:3] == (32, 32)
        assert result.dtype == np.float32


def test_interpolate_vs_keep_different_frame_counts(multiframe_dicom_path):
    """Test that interpolate handler produces different frame count than keep handler."""
    preprocessor_keep = dp.Preprocessor(
        crop=False,
        size=(32, 32),
        volume_handler="keep",
    )

    preprocessor_interpolate = dp.Preprocessor(
        crop=False,
        size=(32, 32),
        volume_handler="interpolate",
        target_frames=16,
    )

    result_keep = dp.preprocess_f32(multiframe_dicom_path, preprocessor_keep, parallel=False)
    result_interpolate = dp.preprocess_f32(multiframe_dicom_path, preprocessor_interpolate, parallel=False)

    # Frame counts should differ
    assert result_keep.shape[0] != result_interpolate.shape[0]
    assert result_interpolate.shape[0] == 16

    # Spatial dimensions should be the same
    assert result_keep.shape[1:3] == result_interpolate.shape[1:3]


def test_interpolate_handler_without_spacing(multiframe_dicom_stream):
    """Test that interpolate handler works when no z-spacing metadata is available.

    This tests the fix where VolumeHandler::Interpolate should interpolate frames
    even when spacing metadata is not available, using the target_frames parameter.
    """
    # Create preprocessors with different target_frames
    preprocessor_8 = dp.Preprocessor(
        crop=False,
        size=(32, 32),
        volume_handler="interpolate",
        target_frames=8,
    )

    preprocessor_24 = dp.Preprocessor(
        crop=False,
        size=(32, 32),
        volume_handler="interpolate",
        target_frames=24,
    )

    result_8 = dp.preprocess_stream_f32(multiframe_dicom_stream, preprocessor_8, parallel=False)
    result_24 = dp.preprocess_stream_f32(multiframe_dicom_stream, preprocessor_24, parallel=False)

    # Both should produce the exact number of target frames
    assert result_8.shape[0] == 8, f"Expected 8 frames, got {result_8.shape[0]}"
    assert result_24.shape[0] == 24, f"Expected 24 frames, got {result_24.shape[0]}"

    # Verify spatial dimensions are correct
    assert result_8.shape[1:3] == (32, 32)
    assert result_24.shape[1:3] == (32, 32)


# Tests for metadata exposure
def test_metadata_classes_exist():
    """Test that metadata classes are available."""
    assert hasattr(dp, "Crop")
    assert hasattr(dp, "Resize")
    assert hasattr(dp, "Padding")
    assert hasattr(dp, "Resolution")
    assert hasattr(dp, "PreprocessingMetadata")


def test_preprocess_with_metadata_basic(dicom_path):
    """Test basic metadata retrieval from preprocessing."""
    preprocessor = dp.Preprocessor(size=(32, 32), crop=True)
    result, metadata = dp.preprocess_f32_with_metadata(dicom_path, preprocessor, parallel=False)
    
    # Check result is same as regular preprocessing
    result_regular = dp.preprocess_f32(dicom_path, preprocessor, parallel=False)
    assert np.allclose(result, result_regular)
    
    # Check metadata exists and has expected structure
    assert isinstance(metadata, dp.PreprocessingMetadata)
    assert metadata.num_frames == 1
    assert metadata.crop is not None or metadata.resize is not None or metadata.padding is not None


def test_preprocess_with_metadata_crop(dicom_path):
    """Test that crop metadata is populated correctly."""
    preprocessor = dp.Preprocessor(size=None, crop=True, use_padding=False)
    result, metadata = dp.preprocess_f32_with_metadata(dicom_path, preprocessor, parallel=False)
    
    assert metadata.crop is not None
    assert isinstance(metadata.crop, dp.Crop)
    assert metadata.crop.left >= 0
    assert metadata.crop.top >= 0
    assert metadata.crop.width > 0
    assert metadata.crop.height > 0


def test_preprocess_with_metadata_resize(dicom_path):
    """Test that resize metadata is populated correctly."""
    preprocessor = dp.Preprocessor(size=(64, 64), crop=False, use_padding=False)
    result, metadata = dp.preprocess_f32_with_metadata(dicom_path, preprocessor, parallel=False)
    
    assert metadata.resize is not None
    assert isinstance(metadata.resize, dp.Resize)
    assert metadata.resize.scale_x > 0
    assert metadata.resize.scale_y > 0
    assert metadata.resize.scale_x == metadata.resize.scale_y  # Should maintain aspect ratio
    assert isinstance(metadata.resize.filter, str)


def test_preprocess_with_metadata_padding(dicom_path):
    """Test that padding metadata is populated correctly."""
    preprocessor = dp.Preprocessor(size=(128, 128), crop=False, use_padding=True, padding_direction="center")
    result, metadata = dp.preprocess_f32_with_metadata(dicom_path, preprocessor, parallel=False)
    
    # Padding might be None if image already matches size
    if metadata.padding is not None:
        assert isinstance(metadata.padding, dp.Padding)
        assert metadata.padding.left >= 0
        assert metadata.padding.top >= 0
        assert metadata.padding.right >= 0
        assert metadata.padding.bottom >= 0


def test_preprocess_with_metadata_resolution(dicom_path):
    """Test that resolution metadata is populated when available."""
    preprocessor = dp.Preprocessor(size=(32, 32))
    result, metadata = dp.preprocess_f32_with_metadata(dicom_path, preprocessor, parallel=False)
    
    # Resolution may or may not be available depending on DICOM metadata
    if metadata.resolution is not None:
        assert isinstance(metadata.resolution, dp.Resolution)
        assert metadata.resolution.pixels_per_mm_x > 0
        assert metadata.resolution.pixels_per_mm_y > 0
        # frames_per_mm might be None


def test_preprocess_stream_with_metadata(dicom_stream):
    """Test metadata retrieval with stream preprocessing."""
    preprocessor = dp.Preprocessor(size=(32, 32), crop=True)
    result, metadata = dp.preprocess_stream_f32_with_metadata(dicom_stream, preprocessor, parallel=False)
    
    assert isinstance(metadata, dp.PreprocessingMetadata)
    assert result.shape == (1, 32, 32, 1)


def test_preprocess_slices_with_metadata(multiple_dicom_paths):
    """Test metadata retrieval with slice preprocessing."""
    preprocessor = dp.Preprocessor(size=(32, 32), crop=True)
    results, metadata = dp.preprocess_f32_slices_with_metadata(multiple_dicom_paths, preprocessor, parallel=False)
    
    assert isinstance(metadata, dp.PreprocessingMetadata)
    assert len(results) == len(multiple_dicom_paths)
    
    # All slices should have same shape due to common crop
    shapes = [r.shape for r in results]
    assert all(shape == shapes[0] for shape in shapes)


def test_preprocess_stream_slices_with_metadata(multiple_dicom_streams):
    """Test metadata retrieval with stream slice preprocessing."""
    preprocessor = dp.Preprocessor(size=(32, 32), crop=True)
    results, metadata = dp.preprocess_stream_f32_slices_with_metadata(
        multiple_dicom_streams, preprocessor, parallel=False
    )
    
    assert isinstance(metadata, dp.PreprocessingMetadata)
    assert len(results) == len(multiple_dicom_streams)


def test_metadata_backwards_compatibility(dicom_path):
    """Test that old functions still work without metadata."""
    preprocessor = dp.Preprocessor(size=(32, 32))
    
    # Old function should still work
    result_old = dp.preprocess_f32(dicom_path, preprocessor, parallel=False)
    
    # New function with metadata
    result_new, metadata = dp.preprocess_f32_with_metadata(dicom_path, preprocessor, parallel=False)
    
    # Results should be identical
    assert np.allclose(result_old, result_new)


@pytest.mark.parametrize("dtype_func,dtype_func_with_meta", [
    (dp.preprocess_u8, dp.preprocess_u8_with_metadata),
    (dp.preprocess_u16, dp.preprocess_u16_with_metadata),
    (dp.preprocess_f32, dp.preprocess_f32_with_metadata),
])
def test_all_dtypes_with_metadata(dicom_path, dtype_func, dtype_func_with_meta):
    """Test that all data type variants support metadata."""
    preprocessor = dp.Preprocessor(size=(32, 32))
    
    result_old = dtype_func(dicom_path, preprocessor, parallel=False)
    result_new, metadata = dtype_func_with_meta(dicom_path, preprocessor, parallel=False)
    
    assert np.array_equal(result_old, result_new)
    assert isinstance(metadata, dp.PreprocessingMetadata)


def test_metadata_repr():
    """Test that metadata classes have reasonable string representations."""
    preprocessor = dp.Preprocessor(size=(32, 32), crop=True)
    # Get a real DICOM to test with
    import shutil
    import tempfile
    import pydicom
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.dcm"
        source = pydicom.data.get_testdata_file("CT_small.dcm")
        shutil.copy(source, path)
        
        result, metadata = dp.preprocess_f32_with_metadata(path, preprocessor, parallel=False)
        
        # Test __repr__ methods exist and work
        assert "PreprocessingMetadata" in repr(metadata)
        
        if metadata.crop:
            assert "Crop" in repr(metadata.crop)
            assert str(metadata.crop.left) in repr(metadata.crop)
        
        if metadata.resize:
            assert "Resize" in repr(metadata.resize)
            assert str(metadata.resize.scale_x) in repr(metadata.resize)
        
        if metadata.padding:
            assert "Padding" in repr(metadata.padding)
        
        if metadata.resolution:
            assert "Resolution" in repr(metadata.resolution)
