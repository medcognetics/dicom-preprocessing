use crate::file::{DicomFileOperations, InodeSort, TiffFileOperations};
use crate::load::LoadFromTiff;
use ndarray::Array4;
use num::Zero;
use numpy::Element;
use numpy::{IntoPyArray, PyArray4};
use pyo3::{
    exceptions::{PyIOError, PyRuntimeError},
    pymodule,
    types::PyModule,
    Bound, PyResult, Python,
};
use std::clone::Clone;
use std::fs::File;
use std::path::PathBuf;
use tiff::decoder::Decoder;

#[pymodule]
fn dicom_preprocessing<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    fn load_tiff<'py, T>(py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyArray4<T>>>
    where
        T: Clone + Zero + Element,
        Array4<T>: LoadFromTiff<T>,
    {
        let file = File::open(path).map_err(|_| PyIOError::new_err("Failed to open file"))?;
        let mut decoder =
            Decoder::new(file).map_err(|_| PyRuntimeError::new_err("Failed to create decoder"))?;
        let array = Array4::<T>::decode(&mut decoder)
            .map_err(|_| PyRuntimeError::new_err("Failed to decode TIFF"))?;
        Ok(array.into_pyarray_bound(py))
    }

    #[pyfn(m)]
    #[pyo3(name = "load_tiff_u8")]
    fn load_tiff_u8<'py>(py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyArray4<u8>>> {
        load_tiff::<u8>(py, path)
    }

    #[pyfn(m)]
    #[pyo3(name = "load_tiff_u16")]
    fn load_tiff_u16<'py>(py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyArray4<u16>>> {
        load_tiff::<u16>(py, path)
    }

    #[pyfn(m)]
    #[pyo3(name = "load_tiff_f32")]
    fn load_tiff_f32<'py>(py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyArray4<f32>>> {
        load_tiff::<f32>(py, path)
    }

    #[pyfn(m)]
    #[pyo3(name = "inode_sort")]
    fn inode_sort(paths: Vec<PathBuf>) -> PyResult<Vec<PathBuf>> {
        let result = paths.into_iter().sorted_by_inode().collect::<Vec<_>>();
        Ok(result)
    }

    #[pyfn(m)]
    #[pyo3(name = "find_dicom_files", signature = (path, spinner=false))]
    fn find_dicom_files(path: PathBuf, spinner: bool) -> PyResult<Vec<PathBuf>> {
        let result = match spinner {
            true => path.find_dicoms_with_spinner()?.collect(),
            false => path.find_dicoms()?.collect(),
        };
        Ok(result)
    }

    #[pyfn(m)]
    #[pyo3(name = "find_tiff_files", signature = (path, spinner=false))]
    fn find_tiff_files(path: PathBuf, spinner: bool) -> PyResult<Vec<PathBuf>> {
        let result = match spinner {
            true => path.find_tiffs_with_spinner()?.collect(),
            false => path.find_tiffs()?.collect(),
        };
        Ok(result)
    }

    #[pyfn(m)]
    #[pyo3(name = "read_dicom_paths", signature = (path, bar=false))]
    fn read_dicom_paths(path: PathBuf, bar: bool) -> PyResult<Vec<PathBuf>> {
        let result = match bar {
            true => path.read_dicom_paths_with_bar()?.collect(),
            false => path.read_dicom_paths()?.collect(),
        };
        Ok(result)
    }

    #[pyfn(m)]
    #[pyo3(name = "read_tiff_paths", signature = (path, bar=false))]
    fn read_tiff_paths(path: PathBuf, bar: bool) -> PyResult<Vec<PathBuf>> {
        let result = match bar {
            true => path.read_tiff_paths_with_bar()?.collect(),
            false => path.read_tiff_paths()?.collect(),
        };
        Ok(result)
    }

    Ok(())
}
