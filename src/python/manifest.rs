use crate::file::Inode;
use crate::manifest::ManifestEntry;
use crate::python::path::PyPath;
use pyo3::exceptions::PyNotADirectoryError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::wrap_pyfunction;
use pyo3::{
    pymodule,
    types::{PyAnyMethods, PyDict, PyModule},
    Bound, PyAny, PyResult, Python,
};
use std::clone::Clone;

use crate::manifest::get_manifest_with_progress;

#[pyclass]
#[derive(Clone)]
struct PyManifestEntry(ManifestEntry);

#[pymethods]
impl PyManifestEntry {
    #[new]
    fn try_new(
        path: Bound<'_, PyAny>,
        sop_instance_uid: String,
        study_instance_uid: String,
        series_instance_uid: String,
    ) -> PyResult<Self> {
        let py_path = path.extract::<PyPath>()?;
        let path = py_path.as_path();
        let entry = ManifestEntry::new(
            path,
            sop_instance_uid,
            study_instance_uid,
            series_instance_uid,
        );
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
    fn series_instance_uid(&self) -> String {
        self.0.series_instance_uid().to_string()
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

    #[getter]
    fn dimensions<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        if let Some(dims) = self.0.dimensions() {
            let dict = PyDict::new(py);
            dict.set_item("width", dims.width)?;
            dict.set_item("height", dims.height)?;
            dict.set_item("channels", dims.channels)?;
            dict.set_item("num_frames", dims.num_frames)?;
            Ok(Some(dict))
        } else {
            Ok(None)
        }
    }

    fn __repr__(&self) -> String {
        let dims_str = if let Some(dims) = self.0.dimensions() {
            format!(
                ", dimensions=({}x{}x{}x{})",
                dims.num_frames, dims.height, dims.width, dims.channels
            )
        } else {
            String::new()
        };

        format!(
            "ManifestEntry(path='{}', sop_instance_uid='{}', study_instance_uid='{}', series_instance_uid='{}'{})",
            self.0.path().display(),
            self.0.sop_instance_uid(),
            self.0.study_instance_uid(),
            self.0.series_instance_uid(),
            dims_str
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
        let py_path = PyPath::new(relpath).into_pyobject(py)?;
        Ok(py_path)
    }
}

impl From<ManifestEntry> for PyManifestEntry {
    fn from(entry: ManifestEntry) -> Self {
        Self(entry)
    }
}

#[pyfunction]
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
        .map(PyManifestEntry::from)
        .collect::<Vec<_>>();

    let result = result
        .into_iter()
        .map(|e| e.into_pyobject(py))
        .collect::<Result<Vec<_>, _>>()?;
    let result = PyList::new(py, result)?;
    Ok(result)
}

#[pymodule]
#[pyo3(name = "manifest")]
pub(crate) fn register_submodule<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_manifest, m)?)?;
    Ok(())
}
