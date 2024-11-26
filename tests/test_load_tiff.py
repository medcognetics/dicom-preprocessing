import numpy as np
from dicom_preprocessing import load_tiff_f32, load_tiff_u8, load_tiff_u16
from PIL import Image


def test_load_tiff_u8(tmp_path):
    path = tmp_path / "test.tiff"
    N, H, W, C = 3, 10, 10, 1
    array = np.random.randint(0, 255, size=(N, H, W, C), dtype=np.uint8)
    img = Image.fromarray(array[0, ..., 0])
    img.save(path, format="TIFF", save_all=True, append_images=[Image.fromarray(array[i, ..., 0]) for i in range(1, N)])

    result = load_tiff_u8(str(path))
    assert np.array_equal(result, array)


def test_load_tiff_u16(tmp_path):
    path = tmp_path / "test.tiff"
    N, H, W, C = 3, 10, 10, 1
    array = np.random.randint(0, 65535, size=(N, H, W, C), dtype=np.uint16)
    img = Image.fromarray(array[0, ..., 0])
    img.save(path, format="TIFF", save_all=True, append_images=[Image.fromarray(array[i, ..., 0]) for i in range(1, N)])

    result = load_tiff_u16(str(path))
    assert np.array_equal(result, array)


def test_load_tiff_f32(tmp_path):
    path = tmp_path / "test.tiff"
    N, H, W, C = 3, 10, 10, 1
    array = np.random.randint(0, 65535, size=(N, H, W, C), dtype=np.uint16)
    img = Image.fromarray(array[0, ..., 0])
    img.save(path, format="TIFF", save_all=True, append_images=[Image.fromarray(array[i, ..., 0]) for i in range(1, N)])
    array = array.astype(np.float32) / 65535
    result = load_tiff_f32(str(path))
    assert np.array_equal(result, array)
