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
