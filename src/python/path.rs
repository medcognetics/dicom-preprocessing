use crate::file::{DicomFileOperations, InodeSort, TiffFileOperations};
use pyo3::FromPyObject;
use pyo3::{
    exceptions::{PyFileNotFoundError, PyNotADirectoryError, PyRuntimeError, PyValueError},
    pymodule,
    types::{PyAnyMethods, PyList, PyListMethods, PyModule},
    Bound, IntoPyObject, PyAny, PyResult, Python,
};
use std::ops::Deref;
use std::path::{Path, PathBuf};

use pyo3::prelude::*;

/// A wrapper type that provides bidirectional conversion between Python's pathlib.Path
/// and Rust's PathBuf.
#[derive(Clone, Debug)]
pub struct PyPath(PathBuf);

impl PyPath {
    /// Creates a new PyPath from any type that can be converted to a Path
    pub fn new<P: AsRef<Path>>(path: P) -> Self {
        PyPath(PathBuf::from(path.as_ref()))
    }

    /// Attempts to create a Python Path object from this PyPath
    fn to_py_path(&self, py: Python<'_>) -> PyResult<PyObject> {
        let pathlib = py.import("pathlib")?;
        let path_class = pathlib.getattr("Path")?;
        path_class
            .call1((self.0.to_string_lossy().into_owned(),))
            .map(|p| p.into())
    }
}

impl FromPyObject<'_> for PyPath {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        // First try to get the string representation of the path
        match ob.str()?.extract::<String>() {
            Ok(path_str) => Ok(PyPath(PathBuf::from(path_str))),
            Err(_) => Err(PyValueError::new_err("Could not convert path to string")),
        }
    }
}

impl<'py> IntoPyObject<'py> for PyPath {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let obj = self.to_py_path(py)?;
        Ok(obj.into_bound(py))
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
        self.0.as_ref()
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
    ) -> PyResult<Bound<'py, PyList>> {
        let mut iter = paths
            .iter()
            .map(|p| p.extract::<PyPath>())
            .collect::<Result<Vec<_>, _>>()?
            .into_iter();
        let result: Vec<_> = match bar {
            true => iter.sorted_by_inode_with_progress().collect(),
            false => iter.sorted_by_inode().collect(),
        };
        let py_list = PyList::new(py, result)?;
        Ok(py_list)
    }

    #[pyfn(m)]
    #[pyo3(name = "is_dicom_file")]
    fn is_dicom_file(path: &Bound<'_, PyAny>) -> PyResult<bool> {
        let path = path.extract::<PyPath>()?;
        path.is_dicom_file()
            .map_err(|_| PyRuntimeError::new_err("Failed to check if file is DICOM"))
    }

    #[pyfn(m)]
    #[pyo3(name = "is_tiff_file")]
    fn is_tiff_file(path: &Bound<'_, PyAny>) -> PyResult<bool> {
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
            true => path.find_dicoms_with_spinner()?.map(PyPath::new).collect(),
            false => path.find_dicoms()?.map(PyPath::new).collect(),
        };
        PyList::new(py, result)
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
            true => path.find_tiffs_with_spinner()?.map(PyPath::new).collect(),
            false => path.find_tiffs()?.map(PyPath::new).collect(),
        };
        PyList::new(py, result)
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
            true => path.read_dicom_paths_with_bar()?.map(PyPath::new).collect(),
            false => path.read_dicom_paths()?.map(PyPath::new).collect(),
        };
        PyList::new(py, result)
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
            true => path.read_tiff_paths_with_bar()?.map(PyPath::new).collect(),
            false => path.read_tiff_paths()?.map(PyPath::new).collect(),
        };
        PyList::new(py, result)
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::Python;
    use rstest::rstest;

    #[rstest]
    #[case("test/path")]
    #[case("/absolute/path")]
    #[case("relative/path/with/multiple/segments")]
    fn test_pypath_roundtrip(#[case] path: &str) {
        Python::with_gil(|py| {
            // Python Path -> PyPath
            let pathlib = py.import("pathlib").unwrap();
            let path_class = pathlib.getattr("Path").unwrap();
            let py_path_obj = path_class.call1((path,)).unwrap();
            let py_path: PyPath = py_path_obj.extract().unwrap();
            assert_eq!(py_path.0.to_str().unwrap(), path);

            // PyPath -> Python Path
            let converted_back = py_path.into_pyobject(py).unwrap().unbind();
            let path_str = converted_back.call_method0(py, "__str__").unwrap();
            assert_eq!(path_str.extract::<String>(py).unwrap(), path);
        });
    }

    #[test]
    fn test_pypath_from_string() {
        Python::with_gil(|py| {
            let path = "test/path";
            let py_str = path.into_pyobject(py).unwrap().unbind();
            let py_path: PyPath = py_str.extract(py).unwrap();
            assert_eq!(py_path.0.to_str().unwrap(), path);
        });
    }

    #[test]
    fn test_pathbuf_conversion() {
        let py_path = PyPath::new("test/path");
        let path_buf: PathBuf = py_path.into();
        assert_eq!(path_buf.to_str().unwrap(), "test/path");
    }
}
