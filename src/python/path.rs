use pyo3::{types::PyAnyMethods, Bound, FromPyObject, IntoPy, PyAny, PyResult, Python, ToPyObject};
use std::ops::Deref;
use std::path::Path;
use std::path::PathBuf;

use pyo3::prelude::*;

/// Wrapper to convert between Python Path and Rust PathBuf
pub struct PyPath(PathBuf);

impl PyPath {
    pub fn new<P: AsRef<Path>>(path: P) -> Self {
        PyPath(PathBuf::from(path.as_ref()))
    }
}

impl FromPyObject<'_> for PyPath {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        let path = ob.extract::<PathBuf>()?;
        Ok(PyPath(path))
    }
}

impl ToPyObject for PyPath {
    fn to_object(&self, py: Python<'_>) -> PyObject {
        let pathlib = py
            .import_bound("pathlib")
            .expect("Failed to import pathlib");
        let path_class = pathlib.getattr("Path").expect("Failed to get Path class");
        path_class
            .call1((self.0.to_string_lossy().into_owned(),))
            .expect("Failed to create Path")
            .into()
    }
}

impl IntoPy<PyObject> for PyPath {
    fn into_py(self, py: Python<'_>) -> PyObject {
        self.to_object(py)
    }
}

impl Deref for PyPath {
    type Target = PathBuf;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl AsRef<Path> for PyPath {
    fn as_ref(&self) -> &Path {
        &self.0.as_ref()
    }
}

impl From<PyPath> for PathBuf {
    fn from(path: PyPath) -> Self {
        path.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::Python;

    #[test]
    fn test_pypath_conversions() {
        Python::with_gil(|py| {
            // Create a Python Path object
            let pathlib = py.import_bound("pathlib").unwrap();
            let path_class = pathlib.getattr("Path").unwrap();
            let test_path = path_class.call1(("test/path",)).unwrap();

            // Convert Python Path to PyPath
            let py_path: PyPath = test_path.extract().unwrap();
            assert_eq!(py_path.0.to_str().unwrap(), "test/path");

            // Convert PyPath back to Python Path
            let py_obj = py_path.to_object(py);
            let path_str = py_obj.call_method0(py, "__str__").unwrap();
            assert_eq!(path_str.extract::<String>(py).unwrap(), "test/path");

            // Test From<PyPath> for PathBuf
            let path_buf: PathBuf = py_path.into();
            assert_eq!(path_buf.to_str().unwrap(), "test/path");
        });
    }
}
