use crate::load::LoadFromTiff;
use crate::python::path::PyPath;
use ::tiff::decoder::Decoder;
use ndarray::Array4;
use num::Zero;
use numpy::Element;
use numpy::{IntoPyArray, PyArray4};
use pyo3::prelude::*;
use pyo3::{
    exceptions::{PyFileNotFoundError, PyIOError, PyRuntimeError},
    pymodule,
    types::{PyAnyMethods, PyList, PyModule},
    Bound, PyAny, PyResult, Python,
};
use rayon::prelude::*;
use std::clone::Clone;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

#[pymodule]
#[pyo3(name = "tiff")]
pub(crate) fn register_submodule<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    fn load_tiff<T, P>(py: Python<'_>, path: P) -> PyResult<Bound<'_, PyArray4<T>>>
    where
        T: Clone + Zero + Element,
        Array4<T>: LoadFromTiff<T>,
        P: AsRef<Path>,
    {
        let path = path.as_ref();
        if !path.is_file() {
            return Err(PyFileNotFoundError::new_err(format!(
                "File not found: {}",
                path.display()
            )));
        }

        let file = File::open(path).map_err(|_| PyIOError::new_err("Failed to open file"))?;
        let reader = BufReader::new(file);
        let mut decoder = Decoder::new(reader)
            .map_err(|_| PyRuntimeError::new_err("Failed to create decoder"))?;
        let array = Array4::<T>::decode(&mut decoder)
            .map_err(|_| PyRuntimeError::new_err("Failed to decode TIFF"))?;
        Ok(array.into_pyarray(py))
    }

    #[pyfn(m)]
    #[pyo3(name = "load_tiff_u8")]
    fn load_tiff_u8<'py>(
        py: Python<'py>,
        path: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray4<u8>>> {
        let path = path.extract::<PyPath>()?;
        load_tiff::<u8, _>(py, path)
    }

    #[pyfn(m)]
    #[pyo3(name = "load_tiff_u16")]
    fn load_tiff_u16<'py>(
        py: Python<'py>,
        path: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray4<u16>>> {
        let path = path.extract::<PyPath>()?;
        load_tiff::<u16, _>(py, path)
    }

    #[pyfn(m)]
    #[pyo3(name = "load_tiff_f32")]
    fn load_tiff_f32<'py>(
        py: Python<'py>,
        path: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray4<f32>>> {
        let path = path.extract::<PyPath>()?;
        load_tiff::<f32, _>(py, path)
    }

    #[pyfn(m)]
    #[pyo3(name = "load_tiff_f32_batched")]
    fn load_tiff_f32_batched<'py>(
        py: Python<'py>,
        paths: &Bound<'py, PyList>,
        batch_size: usize,
    ) -> PyResult<Py<TiffBatchIterator>> {
        let paths: Vec<PyPath> = paths
            .iter()
            .map(|p| p.extract::<PyPath>())
            .collect::<PyResult<_>>()?;

        let iter = TiffBatchIterator {
            paths,
            batch_size,
            current_idx: 0,
        };

        Py::new(py, iter)
    }

    #[pyclass]
    struct TiffBatchIterator {
        paths: Vec<PyPath>,
        batch_size: usize,
        current_idx: usize,
    }

    #[pymethods]
    impl TiffBatchIterator {
        fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
            slf
        }

        fn __next__<'py>(
            mut slf: PyRefMut<'py, Self>,
            py: Python<'py>,
        ) -> Option<Bound<'py, PyList>> {
            if slf.current_idx >= slf.paths.len() {
                return None;
            }

            let end_idx = (slf.current_idx + slf.batch_size).min(slf.paths.len());
            let batch_paths: Vec<std::path::PathBuf> = slf.paths[slf.current_idx..end_idx]
                .iter()
                .map(|p| p.as_path().to_path_buf())
                .collect();

            let raw_arrays = match batch_paths
                .into_par_iter()
                .map(|path| {
                    let file = File::open(&path).ok()?;
                    let reader = BufReader::new(file);
                    let mut decoder = Decoder::new(reader).ok()?;
                    Array4::<f32>::decode(&mut decoder).ok()
                })
                .collect::<Option<Vec<_>>>()
            {
                Some(arrays) => arrays,
                None => {
                    PyRuntimeError::new_err("Failed to load one or more TIFF files").restore(py);
                    return None;
                }
            };

            let arrays: Vec<_> = raw_arrays
                .into_iter()
                .map(|arr| arr.into_pyarray(py))
                .collect();

            let batch = match PyList::new(py, arrays) {
                Ok(batch) => batch,
                Err(e) => {
                    PyRuntimeError::new_err(format!("Failed to create batch: {}", e)).restore(py);
                    return None;
                }
            };
            slf.current_idx = end_idx;
            Some(batch)
        }
    }

    Ok(())
}
