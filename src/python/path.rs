use crate::file::{DicomFileOperations, InodeSort, TiffFileOperations};
use pyo3::{
    exceptions::{PyFileNotFoundError, PyNotADirectoryError, PyRuntimeError},
    pymodule,
    types::{PyAnyMethods, PyList, PyListMethods, PyModule},
    Bound, IntoPy, PyAny, PyResult, Python,
};
use pyo3::{FromPyObject, ToPyObject};
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

#[pymodule]
#[pyo3(name = "path")]
pub(crate) fn register_submodule<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    #[pyfn(m)]
    #[pyo3(name = "inode_sort", signature = (paths, bar=false))]
    fn inode_sort<'py>(
        py: Python<'py>,
        paths: Bound<'py, PyList>,
        bar: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mut iter = paths
            .iter()
            .map(|p| p.extract::<PyPath>())
            .collect::<Result<Vec<_>, _>>()?
            .into_iter();
        let result: Vec<_> = match bar {
            true => iter.sorted_by_inode_with_progress().collect(),
            false => iter.sorted_by_inode().collect(),
        };
        Ok(result.into_py(py).into_bound(py))
    }

    #[pyfn(m)]
    #[pyo3(name = "is_dicom_file")]
    fn is_dicom_file<'py>(path: &Bound<'py, PyAny>) -> PyResult<bool> {
        let path = path.extract::<PyPath>()?;
        path.is_dicom_file()
            .map_err(|_| PyRuntimeError::new_err("Failed to check if file is DICOM"))
    }

    #[pyfn(m)]
    #[pyo3(name = "is_tiff_file")]
    fn is_tiff_file<'py>(path: &Bound<'py, PyAny>) -> PyResult<bool> {
        let path = path.extract::<PyPath>()?;
        path.is_tiff_file()
            .map_err(|_| PyRuntimeError::new_err("Failed to check if file is TIFF"))
    }

    #[pyfn(m)]
    #[pyo3(name = "find_dicom_files", signature = (path, spinner=false))]
    fn find_dicom_files<'py>(
        py: Python<'py>,
        path: &Bound<'py, PyAny>,
        spinner: bool,
    ) -> PyResult<Bound<'py, PyList>> {
        let path = path.extract::<PyPath>()?;
        if !path.is_dir() {
            return Err(PyNotADirectoryError::new_err(format!(
                "Not a directory: {}",
                path.display()
            )));
        }
        let result: Vec<PyPath> = match spinner {
            true => path
                .find_dicoms_with_spinner()?
                .map(|p| PyPath::new(p))
                .collect(),
            false => path.find_dicoms()?.map(|p| PyPath::new(p)).collect(),
        };
        let result = PyList::new_bound(py, result);
        Ok(result)
    }

    #[pyfn(m)]
    #[pyo3(name = "find_tiff_files", signature = (path, spinner=false))]
    fn find_tiff_files<'py>(
        py: Python<'py>,
        path: &Bound<'py, PyAny>,
        spinner: bool,
    ) -> PyResult<Bound<'py, PyList>> {
        let path = path.extract::<PyPath>()?;
        if !path.is_dir() {
            return Err(PyNotADirectoryError::new_err(format!(
                "Not a directory: {}",
                path.display()
            )));
        }
        let result: Vec<PyPath> = match spinner {
            true => path
                .find_tiffs_with_spinner()?
                .map(|p| PyPath::new(p))
                .collect(),
            false => path.find_tiffs()?.map(|p| PyPath::new(p)).collect(),
        };
        let result = PyList::new_bound(py, result);
        Ok(result)
    }

    #[pyfn(m)]
    #[pyo3(name = "read_dicom_paths", signature = (path, bar=false))]
    fn read_dicom_paths<'py>(
        py: Python<'py>,
        path: &Bound<'py, PyAny>,
        bar: bool,
    ) -> PyResult<Bound<'py, PyList>> {
        let path = path.extract::<PyPath>()?;
        if !path.is_file() {
            return Err(PyFileNotFoundError::new_err(format!(
                "File not found: {}",
                path.display()
            )));
        }
        let result: Vec<PyPath> = match bar {
            true => path
                .read_dicom_paths_with_bar()?
                .map(|p| PyPath::new(p))
                .collect(),
            false => path.read_dicom_paths()?.map(|p| PyPath::new(p)).collect(),
        };
        let result = PyList::new_bound(py, result);
        Ok(result)
    }

    #[pyfn(m)]
    #[pyo3(name = "read_tiff_paths", signature = (path, bar=false))]
    fn read_tiff_paths<'py>(
        py: Python<'py>,
        path: &Bound<'py, PyAny>,
        bar: bool,
    ) -> PyResult<Bound<'py, PyList>> {
        let path = path.extract::<PyPath>()?;
        if !path.is_file() {
            return Err(PyFileNotFoundError::new_err(format!(
                "File not found: {}",
                path.display()
            )));
        }
        let result: Vec<PyPath> = match bar {
            true => path
                .read_tiff_paths_with_bar()?
                .map(|p| PyPath::new(p))
                .collect(),
            false => path.read_tiff_paths()?.map(|p| PyPath::new(p)).collect(),
        };
        let result = PyList::new_bound(py, result);
        Ok(result)
    }

    Ok(())
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
