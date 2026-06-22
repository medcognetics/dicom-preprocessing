from pathlib import Path
from typing import Any, cast

import pydicom
import pytest
from pydicom.data import get_testdata_file

import dicom_preprocessing as dp

CT_SMALL_DICOM = "CT_small.dcm"


def get_testdata_filepath(name: str) -> str:
    source = get_testdata_file(name)
    if source is None or not isinstance(source, str):
        raise FileNotFoundError(f"Unable to locate pydicom test file: {name}")
    return source


@pytest.fixture(params=[Path, str])
def dicom_path(request, tmp_path):
    source = get_testdata_filepath(CT_SMALL_DICOM)
    path = tmp_path / "test.dcm"
    path.write_bytes(Path(source).read_bytes())

    return request.param(path)


def test_validate_dicom_returns_report_dict(dicom_path):
    report = dp.validate_dicom(dicom_path)

    assert report["status"] == "pass"
    assert report["summary"]["valid"] is True
    assert report["summary"]["decode_mode"] == "frame"
    assert report["file"]["transfer_syntax_uid"] == "1.2.840.10008.1.2.1"
    assert report["file"]["transfer_syntax_name"] == "Explicit VR Little Endian"
    assert report["pixel_format"]["bits_allocated"] == 16
    assert report["pixel_format"]["bits_stored"] == 16
    assert report["pixel_format"]["high_bit"] == 15
    assert report["decode_smoke_test"]["status"] == "pass"


def test_validate_dicom_decode_none_skips_smoke_test(dicom_path):
    report = dp.validate_dicom(dicom_path, decode="none")

    assert report["summary"]["decode_mode"] == "none"
    assert report["decode_smoke_test"]["attempted"] is False
    assert report["decode_smoke_test"]["status"] == "skip"


def test_validate_dicom_validation_failure_returns_invalid_report(tmp_path):
    source = get_testdata_filepath(CT_SMALL_DICOM)
    path = tmp_path / "missing-high-bit.dcm"
    dicom = pydicom.dcmread(source)
    del dicom.HighBit
    dicom.save_as(path)

    report = dp.validate_dicom(path)

    assert report["status"] == "fail"
    assert report["summary"]["valid"] is False
    assert any(error["code"] == "missing_high_bit" for error in report["errors"])


def test_validate_dicom_rejects_invalid_decode_mode(dicom_path):
    with pytest.raises(ValueError, match="Invalid decode mode"):
        dp.validate_dicom(dicom_path, decode=cast(Any, "bad"))


def test_validate_dicom_missing_path_raises_file_not_found(tmp_path):
    missing_path = tmp_path / "missing.dcm"

    with pytest.raises(FileNotFoundError, match="File not found"):
        dp.validate_dicom(missing_path)
