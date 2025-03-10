pub mod manifest;
pub mod path;
pub mod tiff;
pub mod preprocess;

use pyo3::prelude::*;

#[pymodule]
pub fn dicom_preprocessing<'py>(py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    tiff::register_submodule(py, m)?;
    path::register_submodule(py, m)?;
    manifest::register_submodule(py, m)?;
    preprocess::register_submodule(py, m)?;
    Ok(())
}
