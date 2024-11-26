use crate::load::LoadFromTiff;
use ndarray::Array4;
use numpy::{IntoPyArray, PyArray4};
use pyo3::{pymodule, types::PyModule, Bound, PyResult, Python};
use std::fs::File;
use tiff::decoder::Decoder;

#[pymodule]
fn dicom_preprocessing<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    #[pyfn(m)]
    #[pyo3(name = "load_tiff_u8")]
    fn load_tiff_u8<'py>(py: Python<'py>, path: &str) -> Bound<'py, PyArray4<u8>> {
        let mut decoder = Decoder::new(File::open(path).unwrap()).unwrap();
        let array = Array4::<u8>::decode(&mut decoder).unwrap();
        array.into_pyarray_bound(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "load_tiff_u16")]
    fn load_tiff_u16<'py>(py: Python<'py>, path: &str) -> Bound<'py, PyArray4<u16>> {
        let mut decoder = Decoder::new(File::open(path).unwrap()).unwrap();
        let array = Array4::<u16>::decode(&mut decoder).unwrap();
        array.into_pyarray_bound(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "load_tiff_f32")]
    fn load_tiff_f32<'py>(py: Python<'py>, path: &str) -> Bound<'py, PyArray4<f32>> {
        let mut decoder = Decoder::new(File::open(path).unwrap()).unwrap();
        let array = Array4::<f32>::decode(&mut decoder).unwrap();
        array.into_pyarray_bound(py)
    }

    Ok(())
}
