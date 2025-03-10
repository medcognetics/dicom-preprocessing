use crate::load::LoadFromTiff;
use crate::python::path::PyPath;
use ::tiff::decoder::Decoder;
use ndarray::Array4;
use num::Zero;
use numpy::Element;
use dicom::object::{open_file, from_reader};
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
use pyo3::types::PyBytes;
use pyo3::buffer::PyBuffer;
use std::io::BufReader;
use std::path::Path;
use crate::preprocess::Preprocessor;
use crate::metadata::preprocessing::PreprocessingMetadata;

#[pymodule]
#[pyo3(name = "preprocess")]
pub(crate) fn register_submodule<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    fn preprocess_stream<'py, T>(
        py: Python<'py>, 
        buffer: &PyBuffer<u8>,
        preprocessor: &Preprocessor,
    ) -> PyResult<Bound<'py, PyArray4<T>>>
    where
        T: Clone + Zero + Element,
        Array4<T>: LoadFromTiff<T>,
    {
        // Check if buffer is readable and contiguous
        if !buffer.is_c_contiguous() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Buffer must be C-contiguous"
            ));
        }

        // Create a slice from the buffer and read it into a DICOM object
        let len = buffer.len_bytes();
        let ptr = buffer.buf_ptr();
        let bytes = unsafe { std::slice::from_raw_parts(ptr as *const u8, len) };
        let dcm = from_reader(bytes).map_err(|_| PyRuntimeError::new_err("Failed to create DICOM object"))?;

        Preprocessor::sanitize_dicom(&mut dcm);
        let (images, metadata) =
            preprocessor
                .prepare_image(&dcm, false)?;
        let color_type = DicomColorType::try_from(&file).context(DicomSnafu { path: source })?;

        let saver = TiffSaver::new(compressor.into(), color_type);
        let mut encoder = saver.open_tiff(dest).unwrap();
        images
            .into_iter()
            .try_for_each(|image| saver.save(&mut encoder, &image, &metadata))
            .context(TiffSnafu { path: dest })?;




        //let file = File::open(path).map_err(|_| PyIOError::new_err("Failed to open file"))?;
        //let reader = BufReader::new(file);
        //let mut decoder = Decoder::new(reader)
        //    .map_err(|_| PyRuntimeError::new_err("Failed to create decoder"))?;
        //let array = Array4::<T>::decode(&mut decoder)
        //    .map_err(|_| PyRuntimeError::new_err("Failed to decode TIFF"))?;
        //Ok(array.into_pyarray_bound(py))
    }

    fn preprocess_file<'py, T, P>(py: Python<'py>, path: P) -> PyResult<Bound<'py, PyArray4<T>>>
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
        let dcm = open_file()


        //let mut decoder = Decoder::new(reader)
        //    .map_err(|_| PyRuntimeError::new_err("Failed to create decoder"))?;
        //let array = Array4::<T>::decode(&mut decoder)
        //    .map_err(|_| PyRuntimeError::new_err("Failed to decode TIFF"))?;
        //Ok(array.into_pyarray_bound(py))
    }

    #[pyfn(m)]
    #[pyo3(name = "preprocess_u8")]
    fn preprocess_u8<'py>(
        py: Python<'py>,
        path: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray4<u8>>> {
        let path = path.extract::<PyPath>()?;
        preprocess_file::<u8, _>(py, path)
    }

    #[pyfn(m)]
    #[pyo3(name = "preprocess_u16")]
    fn preprocess_u16<'py>(
        py: Python<'py>,
        path: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray4<u16>>> {
        let path = path.extract::<PyPath>()?;
        preprocess_file::<u16, _>(py, path)
    }

    #[pyfn(m)]
    #[pyo3(name = "preprocess_f32")]
    fn preprocess_f32<'py>(
        py: Python<'py>,
        path: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray4<f32>>> {
        let path = path.extract::<PyPath>()?;
        preprocess_file::<f32, _>(py, path)
    }

    Ok(())
}
