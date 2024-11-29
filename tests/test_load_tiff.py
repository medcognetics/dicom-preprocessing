from pathlib import Path

import numpy as np
import pytest
from dicom_preprocessing import load_tiff_f32, load_tiff_u8, load_tiff_u16
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
