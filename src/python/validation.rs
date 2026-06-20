use crate::python::path::PyPath;
use crate::validation::{validate_path, DecodeValidation, ValidationRuntimeError};
use pyo3::exceptions::{PyFileNotFoundError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::{types::PyAnyMethods, Bound, PyAny, PyResult, Python};

fn parse_decode_mode(decode: &str) -> PyResult<DecodeValidation> {
    if decode.eq_ignore_ascii_case("frame") {
        Ok(DecodeValidation::Frame)
    } else if decode.eq_ignore_ascii_case("none") {
        Ok(DecodeValidation::None)
    } else {
        Err(PyValueError::new_err(format!(
            "Invalid decode mode: {decode}. Expected 'frame' or 'none'"
        )))
    }
}

fn validation_error_to_py(error: ValidationRuntimeError) -> PyErr {
    match error {
        ValidationRuntimeError::InvalidSourcePath { path } => {
            PyFileNotFoundError::new_err(format!("File not found: {}", path.display()))
        }
        error => PyRuntimeError::new_err(error.to_string()),
    }
}

#[pyfunction]
#[pyo3(name = "validate_dicom", signature = (path, decode="frame"))]
pub(crate) fn validate_dicom<'py>(
    py: Python<'py>,
    path: &Bound<'py, PyAny>,
    decode: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let path = path.extract::<PyPath>()?;
    let report =
        validate_path(path.as_ref(), parse_decode_mode(decode)?).map_err(validation_error_to_py)?;
    let report_json = serde_json::to_string(&report).map_err(|error| {
        PyRuntimeError::new_err(format!("Failed to serialize validation report: {error}"))
    })?;
    py.import("json")?.call_method1("loads", (report_json,))
}

pub(crate) fn register_submodule<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(validate_dicom, m)?)?;
    Ok(())
}
