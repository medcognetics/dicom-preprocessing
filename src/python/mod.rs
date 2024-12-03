pub mod path;
pub mod tiff;

use pyo3::prelude::*;

#[pymodule]
pub fn dicom_preprocessing<'py>(py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    tiff::register_submodule(py, m)?;
    path::register_submodule(py, m)?;
    Ok(())
}
