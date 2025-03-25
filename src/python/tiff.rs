use crate::load::LoadFromTiff;
use crate::metadata::FrameCount;
use crate::python::path::PyPath;
use ::tiff::decoder::Decoder;
use ndarray::Array4;
use num::Zero;
use numpy::Element;
use numpy::{IntoPyArray, PyArray4};
use pyo3::prelude::*;
use pyo3::{
    exceptions::{PyFileNotFoundError, PyIOError, PyRuntimeError, PyValueError},
    pymodule,
    types::{PyAnyMethods, PyList, PyModule, PySequence},
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
    fn get_decoder<P: AsRef<Path>>(path: P) -> PyResult<Decoder<BufReader<File>>> {
        let path = path.as_ref();
        if !path.is_file() {
            return Err(PyFileNotFoundError::new_err(format!(
                "File not found: {}",
                path.display()
            )));
        }

        let file = File::open(path).map_err(|_| PyIOError::new_err("Failed to open file"))?;
        let reader = BufReader::new(file);
        Decoder::new(reader).map_err(|_| PyRuntimeError::new_err("Failed to create decoder"))
    }

    fn load_tiff<'py, T, P>(
        py: Python<'py>,
        path: P,
        frames: Option<&Bound<'py, PySequence>>,
    ) -> PyResult<Bound<'py, PyArray4<T>>>
    where
        T: Clone + Zero + Element,
        Array4<T>: LoadFromTiff<T>,
        P: AsRef<Path>,
    {
        let mut decoder = get_decoder(path)?;

        // Get frame count first to validate indices
        let frame_count = FrameCount::try_from(&mut decoder)
            .map_err(|_| PyRuntimeError::new_err("Failed to get frame count"))?;
        let frame_count: usize = frame_count.into();

        let array = match frames {
            Some(frames) => {
                let frames: Vec<usize> = frames.extract()?;
                if frames.is_empty() {
                    return Err(PyValueError::new_err("Frame list cannot be empty"));
                }
                // Check for out of bounds indices
                if let Some(&max_frame) = frames.iter().max() {
                    if max_frame >= frame_count {
                        return Err(PyValueError::new_err(format!(
                            "Frame index {} is out of bounds for TIFF with {} frames",
                            max_frame, frame_count
                        )));
                    }
                }
                Array4::<T>::decode_frames(&mut decoder, frames.into_iter())
                    .map_err(|_| PyRuntimeError::new_err("Failed to decode TIFF frames"))?
            }
            None => Array4::<T>::decode(&mut decoder)
                .map_err(|_| PyRuntimeError::new_err("Failed to decode TIFF"))?,
        };

        Ok(array.into_pyarray(py))
    }

    #[pyfn(m)]
    #[pyo3(name = "get_frame_count")]
    fn get_frame_count(path: &Bound<'_, PyAny>) -> PyResult<usize> {
        let path = path.extract::<PyPath>()?;
        let mut decoder = get_decoder(path)?;
        let frame_count = FrameCount::try_from(&mut decoder)
            .map_err(|_| PyRuntimeError::new_err("Failed to get frame count"))?;
        Ok(frame_count.into())
    }

    #[pyfn(m)]
    #[pyo3(name = "load_tiff_u8", signature = (path, frames=None))]
    fn load_tiff_u8<'py>(
        py: Python<'py>,
        path: &Bound<'py, PyAny>,
        frames: Option<&Bound<'py, PySequence>>,
    ) -> PyResult<Bound<'py, PyArray4<u8>>> {
        let path = path.extract::<PyPath>()?;
        load_tiff::<u8, _>(py, path, frames)
    }

    #[pyfn(m)]
    #[pyo3(name = "load_tiff_u16", signature = (path, frames=None))]
    fn load_tiff_u16<'py>(
        py: Python<'py>,
        path: &Bound<'py, PyAny>,
        frames: Option<&Bound<'py, PySequence>>,
    ) -> PyResult<Bound<'py, PyArray4<u16>>> {
        let path = path.extract::<PyPath>()?;
        load_tiff::<u16, _>(py, path, frames)
    }

    #[pyfn(m)]
    #[pyo3(name = "load_tiff_f32", signature = (path, frames=None))]
    fn load_tiff_f32<'py>(
        py: Python<'py>,
        path: &Bound<'py, PyAny>,
        frames: Option<&Bound<'py, PySequence>>,
    ) -> PyResult<Bound<'py, PyArray4<f32>>> {
        let path = path.extract::<PyPath>()?;
        load_tiff::<f32, _>(py, path, frames)
    }

    #[pyfn(m)]
    #[pyo3(name = "load_tiff_f32_batched", signature = (paths, batch_size, frames=None))]
    fn load_tiff_f32_batched<'py>(
        py: Python<'py>,
        paths: &Bound<'py, PyList>,
        batch_size: usize,
        frames: Option<&Bound<'py, PySequence>>,
    ) -> PyResult<Py<TiffBatchIterator>> {
        let paths: Vec<PyPath> = paths
            .iter()
            .map(|p| p.extract::<PyPath>())
            .collect::<PyResult<_>>()?;

        // Validate frame indices if provided
        let frames: Option<Vec<usize>> = frames
            .map(|f| {
                let frames: Vec<usize> = f.extract()?;
                if frames.is_empty() {
                    return Err(PyValueError::new_err("Frame list cannot be empty"));
                }

                // Check first file for frame count to validate indices
                if let Some(first_path) = paths.first() {
                    let mut decoder = get_decoder(first_path)?;
                    let frame_count = FrameCount::try_from(&mut decoder)
                        .map_err(|_| PyRuntimeError::new_err("Failed to get frame count"))?;
                    let frame_count: usize = frame_count.into();

                    if let Some(&max_frame) = frames.iter().max() {
                        if max_frame >= frame_count {
                            return Err(PyValueError::new_err(format!(
                                "Frame index {} is out of bounds for TIFF with {} frames",
                                max_frame, frame_count
                            )));
                        }
                    }
                }
                Ok(frames)
            })
            .transpose()?;

        let iter = TiffBatchIterator {
            paths,
            batch_size,
            current_idx: 0,
            frames,
        };

        Py::new(py, iter)
    }

    #[pyclass]
    struct TiffBatchIterator {
        paths: Vec<PyPath>,
        batch_size: usize,
        current_idx: usize,
        frames: Option<Vec<usize>>,
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

            let frames = &slf.frames;

            let raw_arrays = match batch_paths
                .into_par_iter()
                .map(|path| {
                    let file = File::open(&path).ok()?;
                    let reader = BufReader::new(file);
                    let mut decoder = Decoder::new(reader).ok()?;
                    match frames {
                        Some(frames) => {
                            Array4::<f32>::decode_frames(&mut decoder, frames.iter().cloned()).ok()
                        }
                        None => Array4::<f32>::decode(&mut decoder).ok(),
                    }
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
