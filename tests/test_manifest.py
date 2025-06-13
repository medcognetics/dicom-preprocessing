from itertools import product
from pathlib import Path

import numpy as np
import pytest
from dicom_preprocessing import get_manifest
from PIL import Image


@pytest.fixture(params=[Path, str])
def root(request, tmp_path):
    if request.param == Path:
        return Path(tmp_path)
    elif request.param == str:
        return str(tmp_path)
    else:
        raise ValueError(f"Invalid parameter: {request.param}")


@pytest.fixture
def tiff_files(tmp_path):
    # Create test TIFF files
    study_uids = ("study1", "study2", "study3")
    sop_uids = ("sop1", "sop2", "sop3")
    series_uids = ("series1", "series2", "series3")
    files = []
    for study_uid, series_uid, sop_uid in product(study_uids, series_uids, sop_uids):
        path = tmp_path / study_uid / series_uid / f"{sop_uid}.tiff"
        path.parent.mkdir(parents=True, exist_ok=True)
        # Create a valid TIFF file with basic image data
        img_data = np.zeros((64, 64), dtype=np.uint8)
        img_data[16:48, 16:48] = 255  # Create a white square in center
        img = Image.fromarray(img_data)
        img.save(path, format="TIFF", resolution=(72, 72))
        files.append(path)
    return files


@pytest.mark.parametrize("bar", [True, False])
def test_get_manifest(root, tiff_files, bar):
    files = get_manifest(root, bar)
    assert len(files) == len(tiff_files)
    assert set(m.sop_instance_uid for m in files) == {
        "sop1",
        "sop2",
        "sop3",
    }
    assert set(m.study_instance_uid for m in files) == {
        "study1",
        "study2",
        "study3",
    }
    assert set(m.series_instance_uid for m in files) == {
        "series1",
        "series2",
        "series3",
    }
    assert set(m.path for m in files) == set(tiff_files)
    assert all(isinstance(m.inode, int) for m in files)
    assert all(m.relative_path(root) == m.path.relative_to(root) for m in files)
    assert all(m.dimensions == {"width": 64, "height": 64, "channels": 1, "num_frames": 1} for m in files)
