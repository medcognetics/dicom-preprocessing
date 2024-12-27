use crate::file::Inode;
use crate::manifest::ManifestEntry;
use crate::python::path::PyPath;
use pyo3::exceptions::PyNotADirectoryError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::{
    pymodule,
    types::{PyAnyMethods, PyModule},
    Bound, PyAny, PyResult, Python,
};
use std::clone::Clone;

use crate::manifest::{get_manifest, get_manifest_with_progress};

#[pyclass]
#[derive(Clone)]
struct PyManifestEntry(ManifestEntry);

#[pymethods]
impl PyManifestEntry {
    #[new]
    fn try_new<'py>(
        path: Bound<'py, PyAny>,
        sop_instance_uid: String,
        study_instance_uid: String,
    ) -> PyResult<Self> {
        let py_path = path.extract::<PyPath>()?;
        let path = py_path.as_path();
        let entry = ManifestEntry::new(path, sop_instance_uid, study_instance_uid);
        Ok(Self(entry))
    }

    #[getter]
    fn sop_instance_uid(&self) -> String {
        self.0.sop_instance_uid().to_string()
    }

    #[getter]
    fn study_instance_uid(&self) -> String {
        self.0.study_instance_uid().to_string()
    }

    #[getter]
    fn path(&self) -> PyPath {
        PyPath::new(self.0.path())
    }

    #[getter]
    fn inode(&self) -> PyResult<u64> {
        let inode = self.0.inode()?;
        Ok(inode)
    }

    fn __repr__(&self) -> String {
        format!(
            "ManifestEntry(path='{}', sop_instance_uid='{}', study_instance_uid='{}')",
            self.0.path().display(),
            self.0.sop_instance_uid(),
            self.0.study_instance_uid()
        )
    }

    fn relative_path<'py>(
        &self,
        py: Python<'py>,
        root: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let root = root.extract::<PyPath>()?;
        let py_root = root.as_path();
        let relpath = self.0.relative_path(py_root);
        let py_path = PyPath::new(relpath).into_py(py);
        Ok(py_path.into_bound(py))
    }
}

impl From<ManifestEntry> for PyManifestEntry {
    fn from(entry: ManifestEntry) -> Self {
        Self(entry)
    }
}

#[pymodule]
#[pyo3(name = "manifest")]
pub(crate) fn register_submodule<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    #[pyfn(m)]
    #[pyo3(name = "get_manifest", signature = (root, bar=false))]
    fn get_manifest<'py>(
        py: Python<'py>,
        root: Bound<'py, PyAny>,
        bar: bool,
    ) -> PyResult<Bound<'py, PyList>> {
        let path = root.extract::<PyPath>()?;
        let path = path.as_path();
        if !path.is_dir() {
            return Err(PyNotADirectoryError::new_err(format!(
                "Not a directory: {}",
                path.display()
            )));
        }

        let result = match bar {
            true => get_manifest_with_progress(path),
            false => crate::manifest::get_manifest(path),
        }?;

        let result = result
            .into_iter()
            .map(|e| PyManifestEntry::from(e))
            .collect::<Vec<_>>();

        let result: Vec<PyObject> = result.into_iter().map(|e| e.into_py(py)).collect();
        let result = PyList::new_bound(py, result);
        Ok(result)
    }

    Ok(())
}
