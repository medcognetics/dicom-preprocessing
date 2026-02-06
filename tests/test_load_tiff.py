from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dicom_preprocessing import get_frame_count, load_tiff_f32, load_tiff_f32_batched, load_tiff_u8, load_tiff_u16


@pytest.fixture(params=[Path, str])
def path(request, tmp_path):
    path = tmp_path / "test.tiff"
    if request.param == Path:
        return path
    elif request.param == str:
        return str(path)
    else:
        raise ValueError(f"Invalid parameter: {request.param}")


def test_get_frame_count(path):
    N, H, W, C = 3, 10, 10, 1
    array = np.random.randint(0, 255, size=(N, H, W, C), dtype=np.uint8)
    img = Image.fromarray(array[0, ..., 0])
    img.save(path, format="TIFF", save_all=True, append_images=[Image.fromarray(array[i, ..., 0]) for i in range(1, N)])

    count = get_frame_count(path)
    assert count == N


def test_load_tiff_u8_with_frames(path):
    N, H, W, C = 5, 10, 10, 1
    array = np.random.randint(0, 255, size=(N, H, W, C), dtype=np.uint8)
    img = Image.fromarray(array[0, ..., 0])
    img.save(path, format="TIFF", save_all=True, append_images=[Image.fromarray(array[i, ..., 0]) for i in range(1, N)])

    # Test loading specific frames
    frames = [1, 3]  # Load second and fourth frames
    result = load_tiff_u8(path, frames=frames)
    assert result.shape[0] == len(frames)
    assert np.array_equal(result, array[frames])

    # Test loading all frames (None)
    result = load_tiff_u8(path, frames=None)
    assert np.array_equal(result, array)

    # Test different sequence types
    frames_tuple = (1, 3)
    result_tuple = load_tiff_u8(path, frames=frames_tuple)
    assert np.array_equal(result_tuple, array[list(frames_tuple)])

    frames_range = range(2)  # [0, 1]
    result_range = load_tiff_u8(path, frames=frames_range)
    assert np.array_equal(result_range, array[list(frames_range)])


def test_load_tiff_u16_with_frames(path):
    N, H, W, C = 5, 10, 10, 1
    array = np.random.randint(0, 65535, size=(N, H, W, C), dtype=np.uint16)
    img = Image.fromarray(array[0, ..., 0])
    img.save(path, format="TIFF", save_all=True, append_images=[Image.fromarray(array[i, ..., 0]) for i in range(1, N)])

    frames = [1, 3]
    result = load_tiff_u16(path, frames=frames)
    assert np.array_equal(result, array[frames])


def test_load_tiff_f32_with_frames(path):
    N, H, W, C = 5, 10, 10, 1
    array = np.random.randint(0, 65535, size=(N, H, W, C), dtype=np.uint16)
    img = Image.fromarray(array[0, ..., 0])
    img.save(path, format="TIFF", save_all=True, append_images=[Image.fromarray(array[i, ..., 0]) for i in range(1, N)])
    array = array.astype(np.float32) / 65535

    frames = [1, 3]
    result = load_tiff_f32(path, frames=frames)
    assert np.array_equal(result, array[frames])


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("num_paths", [1, 4])
@pytest.mark.parametrize("channels", [1, 3])
def test_load_tiff_f32_batched_with_frames(path, batch_size, num_paths, channels):
    N, H, W, C = 5, 10, 10, channels
    array = np.random.randint(0, 255, size=(N, H, W, C), dtype=np.uint8)
    img = Image.fromarray(array[0, ..., 0] if C == 1 else array[0])
    img.save(
        path,
        format="TIFF",
        save_all=True,
        append_images=[Image.fromarray(array[i, ..., 0] if C == 1 else array[i]) for i in range(1, N)],
    )
    array = array.astype(np.float32) / 255

    frames = [1, 3]
    seen = 0
    for batch in load_tiff_f32_batched([path] * num_paths, batch_size, frames=frames):
        for example in batch:
            assert np.array_equal(example, array[frames])
            seen += 1

    assert seen == num_paths


def test_invalid_frame_indices(path):
    N, H, W, C = 3, 10, 10, 1
    array = np.random.randint(0, 255, size=(N, H, W, C), dtype=np.uint8)
    img = Image.fromarray(array[0, ..., 0])
    img.save(path, format="TIFF", save_all=True, append_images=[Image.fromarray(array[i, ..., 0]) for i in range(1, N)])

    # Test empty frame list
    with pytest.raises(ValueError, match="Frame list cannot be empty"):
        load_tiff_u8(path, frames=[])

    with pytest.raises(ValueError, match="Frame list cannot be empty"):
        load_tiff_u16(path, frames=[])

    with pytest.raises(ValueError, match="Frame list cannot be empty"):
        load_tiff_f32(path, frames=[])

    with pytest.raises(ValueError, match="Frame list cannot be empty"):
        next(load_tiff_f32_batched([path], batch_size=1, frames=[]))

    # Test frame index out of bounds
    expected_msg = f"Frame index {N} is out of bounds for TIFF with {N} frames"
    with pytest.raises(ValueError, match=expected_msg):
        load_tiff_u8(path, frames=[N])  # N is out of bounds

    with pytest.raises(OverflowError):
        load_tiff_u8(path, frames=[-1])  # Negative index

    # Test invalid frame indices with other formats
    with pytest.raises(ValueError, match=expected_msg):
        load_tiff_u16(path, frames=[N])

    with pytest.raises(ValueError, match=expected_msg):
        load_tiff_f32(path, frames=[N])

    with pytest.raises(ValueError, match=expected_msg):
        next(load_tiff_f32_batched([path], batch_size=1, frames=[N]))


def test_load_tiff_u8(path):
    N, H, W, C = 3, 10, 10, 1
    array = np.random.randint(0, 255, size=(N, H, W, C), dtype=np.uint8)
    img = Image.fromarray(array[0, ..., 0])
    img.save(path, format="TIFF", save_all=True, append_images=[Image.fromarray(array[i, ..., 0]) for i in range(1, N)])

    result = load_tiff_u8(path)
    assert np.array_equal(result, array)


def test_load_tiff_u16(path):
    N, H, W, C = 3, 10, 10, 1
    array = np.random.randint(0, 65535, size=(N, H, W, C), dtype=np.uint16)
    img = Image.fromarray(array[0, ..., 0])
    img.save(path, format="TIFF", save_all=True, append_images=[Image.fromarray(array[i, ..., 0]) for i in range(1, N)])

    result = load_tiff_u16(path)
    assert np.array_equal(result, array)


def test_load_tiff_f32(path):
    N, H, W, C = 3, 10, 10, 1
    array = np.random.randint(0, 65535, size=(N, H, W, C), dtype=np.uint16)
    img = Image.fromarray(array[0, ..., 0])
    img.save(path, format="TIFF", save_all=True, append_images=[Image.fromarray(array[i, ..., 0]) for i in range(1, N)])
    array = array.astype(np.float32) / 65535
    result = load_tiff_f32(path)
    assert np.array_equal(result, array)


@pytest.mark.parametrize("batch_size", [1, 2, 3])
@pytest.mark.parametrize("num_paths", [1, 4, 9])
@pytest.mark.parametrize("channels", [1, 3])
def test_load_tiff_f32_batched(path, batch_size, num_paths, channels):
    N, H, W, C = 3, 10, 10, channels
    array = np.random.randint(0, 255, size=(N, H, W, C), dtype=np.uint8)
    img = Image.fromarray(array[0, ..., 0] if C == 1 else array[0])
    img.save(
        path,
        format="TIFF",
        save_all=True,
        append_images=[Image.fromarray(array[i, ..., 0] if C == 1 else array[i]) for i in range(1, N)],
    )
    array = array.astype(np.float32) / 255

    seen = 0
    for batch in load_tiff_f32_batched([path] * num_paths, batch_size):
        for example in batch:
            assert np.array_equal(example, array)
            seen += 1

    assert seen == num_paths


@pytest.mark.parametrize("channels", [1, 3])
def test_load_tiff_f32_batched_deterministic_order(tmp_path, channels):
    N, H, W, C = 3, 10, 10, channels
    num_paths = 10
    arrays = []
    paths = []
    for i in range(num_paths):
        path = tmp_path / f"test_{i}.tiff"
        array = np.random.randint(0, 255, size=(N, H, W, C), dtype=np.uint8)
        img = Image.fromarray(array[0, ..., 0] if C == 1 else array[0])
        img.save(
            path,
            format="TIFF",
            save_all=True,
            append_images=[Image.fromarray(array[i, ..., 0] if C == 1 else array[i]) for i in range(1, N)],
        )
        array = array.astype(np.float32) / 255
        arrays.append(array)
        paths.append(path)

    seen = 0
    for batch in load_tiff_f32_batched(paths, batch_size=1):
        for example in batch:
            assert np.array_equal(example, arrays[seen])
            seen += 1
