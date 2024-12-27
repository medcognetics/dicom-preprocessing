from pathlib import Path

import pytest
from dicom_preprocessing import get_manifest


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
    files = []
    for study_uid in study_uids:
        for sop_uid in sop_uids:
            path = tmp_path / study_uid / f"{sop_uid}.tiff"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
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
    assert set(m.path for m in files) == set(tiff_files)
    assert all(isinstance(m.inode, int) for m in files)
    assert all(m.relative_path(root) == m.path.relative_to(root) for m in files)
