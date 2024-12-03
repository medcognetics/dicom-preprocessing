use crate::file::{DicomFileOperations, InodeSort, TiffFileOperations};
use crate::load::LoadFromTiff;
use crate::python::path::PyPath;
use ndarray::Array4;
use num::Zero;
use numpy::Element;
use numpy::{IntoPyArray, PyArray4};
use pyo3::{
    exceptions::{PyFileNotFoundError, PyIOError, PyNotADirectoryError, PyRuntimeError},
    pymodule,
    types::{PyAnyMethods, PyList, PyListMethods, PyModule},
    Bound, IntoPy, PyAny, PyResult, Python,
};
use std::clone::Clone;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use tiff::decoder::Decoder;

#[pymodule]
fn dicom_preprocessing<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    fn load_tiff<'py, T, P>(py: Python<'py>, path: P) -> PyResult<Bound<'py, PyArray4<T>>>
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
        Ok(array.into_pyarray_bound(py))
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
