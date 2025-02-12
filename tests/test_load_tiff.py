from pathlib import Path

import numpy as np
import pytest
from dicom_preprocessing import load_tiff_f32, load_tiff_f32_batched, load_tiff_u8, load_tiff_u16
from PIL import Image


@pytest.fixture(params=[Path, str])
def path(request, tmp_path):
    path = tmp_path / "test.tiff"
    if request.param == Path:
        return path
    elif request.param == str:
        return str(path)
    else:
        raise ValueError(f"Invalid parameter: {request.param}")


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
def test_load_tiff_f32_batched(path, batch_size, num_paths):
    N, H, W, C = 3, 10, 10, 1
    array = np.random.randint(0, 255, size=(N, H, W, C), dtype=np.uint8)
    img = Image.fromarray(array[0, ..., 0])
    img.save(path, format="TIFF", save_all=True, append_images=[Image.fromarray(array[i, ..., 0]) for i in range(1, N)])
    array = array.astype(np.float32) / 255

    seen = 0
    for batch in load_tiff_f32_batched([path] * num_paths, batch_size):
        for example in batch:
            assert np.array_equal(example, array)
            seen += 1

    assert seen == num_paths


def test_load_tiff_f32_batched_deterministic_order(tmp_path):
    N, H, W, C = 3, 10, 10, 1
    num_paths = 10
    arrays = []
    paths = []
    for i in range(num_paths):
        path = tmp_path / f"test_{i}.tiff"
        array = np.random.randint(0, 255, size=(N, H, W, C), dtype=np.uint8)
        img = Image.fromarray(array[0, ..., 0])
        img.save(
            path, format="TIFF", save_all=True, append_images=[Image.fromarray(array[i, ..., 0]) for i in range(1, N)]
        )
        array = array.astype(np.float32) / 255
        arrays.append(array)
        paths.append(path)

    seen = 0
    for batch in load_tiff_f32_batched(paths, batch_size=1):
        for example in batch:
            assert np.array_equal(example, arrays[seen])
            seen += 1
