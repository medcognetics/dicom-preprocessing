from pathlib import Path

import pytest

from dicom_preprocessing import find_dicom_files, find_tiff_files, inode_sort, read_dicom_paths, read_tiff_paths


@pytest.fixture(params=[Path, str])
def path(request, tmp_path):
    path = tmp_path / "test.txt"
    if request.param == Path:
        return path
    elif request.param == str:
        return str(path)
    else:
        raise ValueError(f"Invalid parameter: {request.param}")


@pytest.fixture
def dicom_dir(tmp_path):
    # Create test DICOM files
    (tmp_path / "test1.dcm").touch()
    (tmp_path / "test2.DCM").touch()
    (tmp_path / "test3.dicom").touch()
    (tmp_path / "test4.DICOM").touch()
    (tmp_path / "not_dicom.txt").touch()
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "test5.dcm").touch()
    return tmp_path


@pytest.fixture
def tiff_dir(tmp_path):
    # Create test TIFF files
    (tmp_path / "test1.tif").touch()
    (tmp_path / "test2.TIF").touch()
    (tmp_path / "test3.tiff").touch()
    (tmp_path / "test4.TIFF").touch()
    (tmp_path / "not_tiff.txt").touch()
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "test5.tiff").touch()
    return tmp_path


@pytest.mark.parametrize("spinner", [True, False])
def test_find_dicom_files(dicom_dir, spinner):
    files = find_dicom_files(dicom_dir, spinner)
    assert len(files) == 5
    assert all(isinstance(p, Path) for p in files)
    extensions = {Path(p).suffix.lower() for p in files}
    assert extensions == {".dcm", ".dicom"}


@pytest.mark.parametrize("spinner", [True, False])
def test_find_tiff_files(tiff_dir, spinner):
    files = find_tiff_files(tiff_dir, spinner)
    assert len(files) == 5
    assert all(isinstance(p, Path) for p in files)
    extensions = {Path(p).suffix.lower() for p in files}
    assert extensions == {".tif", ".tiff"}


@pytest.mark.parametrize("bar", [True, False])
def test_read_dicom_paths(path, bar):
    # Create test file with paths
    paths = [Path(path).parent / f"file{i}.dcm" for i in range(2)]
    for p in paths:
        p.touch()
    with open(path, "w") as f:
        f.write("\n".join([str(p) for p in paths]))
    result = read_dicom_paths(path, bar)
    assert len(result) == 2
    assert all(isinstance(p, Path) for p in result)
    assert result == paths


@pytest.mark.parametrize("bar", [True, False])
def test_read_tiff_paths(path, bar):
    # Create test file with paths
    paths = [Path(path).parent / f"file{i}.tiff" for i in range(2)]
    for p in paths:
        p.touch()
    with open(path, "w") as f:
        f.write("\n".join([str(p) for p in paths]))

    result = read_tiff_paths(path, bar)
    assert len(result) == 2
    assert all(isinstance(p, Path) for p in result)
    assert result == paths


@pytest.mark.parametrize("bar", [True, False])
def test_inode_sort(tmp_path, bar):
    # Create test files
    paths = []
    for i in range(3):
        p = tmp_path / f"file{i}.txt"
        p.touch()
        paths.append(p)

    # Sort paths and verify they're ordered by inode
    sorted_paths = inode_sort(paths, bar)
    inodes = [Path(p).stat().st_ino for p in sorted_paths]
    assert inodes == sorted(inodes)
