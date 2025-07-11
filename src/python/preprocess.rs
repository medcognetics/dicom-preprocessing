use crate::color::DicomColorType;
use crate::load::LoadFromTiff;
use crate::preprocess::Preprocessor;
use crate::python::path::PyPath;
use crate::save::TiffSaver;
use crate::transform::resize::FilterType;
use crate::transform::volume::{CentralSlice, InterpolateVolume, KeepVolume, VolumeHandler};
use crate::transform::PaddingDirection;
use crate::volume::DEFAULT_INTERPOLATE_TARGET_FRAMES;
use ::tiff::decoder::Decoder;
use dicom::object::{from_reader, open_file, FileDicomObject, InMemDicomObject};
use dicom::pixeldata::{ConvertOptions, VoiLutOption, WindowLevel};
use ndarray::Array4;
use num::Zero;
use numpy::Element;
use numpy::{IntoPyArray, PyArray4};
use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;
use pyo3::{
    exceptions::{PyFileNotFoundError, PyRuntimeError},
    pymodule,
    types::{PyAnyMethods, PyModule},
    Bound, PyAny, PyResult, Python,
};
use std::clone::Clone;
use std::io::Seek;
use std::path::Path;
use tempfile::spooled_tempfile;
use tiff::encoder::compression::{Compressor, Uncompressed};
use tiff::encoder::TiffEncoder;

// We guess 64MB as enough for most preprocessed images without being burdensome.
const SPOOL_SIZE: usize = 1024 * 1024 * 64;

#[pyclass(name = "Preprocessor")]
#[derive(Clone)]
pub struct PyPreprocessor {
    inner: Preprocessor,
}

#[pymethods]
impl PyPreprocessor {
    #[new]
    #[pyo3(signature = (
        crop=true,
        size=None,
        filter="triangle",
        padding_direction="zero",
        crop_max=true,
        volume_handler="keep",
        use_components=true,
        use_padding=true,
        border_frac=None,
        target_frames=DEFAULT_INTERPOLATE_TARGET_FRAMES,
        convert_options="default",
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        crop: bool,
        size: Option<(u32, u32)>,
        filter: &str,
        padding_direction: &str,
        crop_max: bool,
        volume_handler: &str,
        use_components: bool,
        use_padding: bool,
        border_frac: Option<f32>,
        target_frames: u32,
        convert_options: &str,
    ) -> PyResult<Self> {
        let filter = match filter.to_lowercase().as_str() {
            "nearest" => FilterType::Nearest,
            "triangle" => FilterType::Triangle,
            "catmull" => FilterType::CatmullRom,
            "gaussian" => FilterType::Gaussian,
            "lanczos3" => FilterType::Lanczos3,
            "maxpool" => FilterType::MaxPool,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Invalid filter type: {filter}"
                )))
            }
        };

        let padding_direction = match padding_direction.to_lowercase().as_str() {
            "zero" => PaddingDirection::Zero,
            "top-left" => PaddingDirection::TopLeft,
            "bottom-right" => PaddingDirection::BottomRight,
            "center" => PaddingDirection::Center,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Invalid padding direction: {padding_direction}"
                )))
            }
        };

        let volume_handler = match volume_handler.to_lowercase().as_str() {
            "keep" => VolumeHandler::Keep(KeepVolume),
            "central" => VolumeHandler::CentralSlice(CentralSlice),
            "interpolate" => VolumeHandler::Interpolate(InterpolateVolume::new(target_frames)),
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Invalid volume handler: {volume_handler}"
                )))
            }
        };

        let convert_options = match convert_options.to_lowercase().as_str() {
            "default" => ConvertOptions::default(),
            "normalize" => ConvertOptions::default().with_voi_lut(VoiLutOption::Normalize),
            s if s.contains(',') => {
                let mut parts = s.split(',');
                let (first, second) = (parts.next().unwrap(), parts.next().unwrap());
                let window = WindowLevel {
                    center: first
                        .parse()
                        .unwrap_or_else(|_| panic!("Invalid window center: {first}")),
                    width: second
                        .parse()
                        .unwrap_or_else(|_| panic!("Invalid window width: {second}")),
                };
                let voi_lut = VoiLutOption::Custom(window);
                ConvertOptions::default().with_voi_lut(voi_lut)
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Invalid convert options: {convert_options}"
                )))
            }
        };

        Ok(Self {
            inner: Preprocessor {
                crop,
                size,
                filter,
                padding_direction,
                crop_max,
                volume_handler,
                use_components,
                use_padding,
                border_frac,
                target_frames,
                convert_options,
            },
        })
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "Preprocessor(crop={}, size={:?}, filter={:?}, padding_direction={:?}, crop_max={}, volume_handler={:?}, use_components={}, use_padding={}, border_frac={:?}, target_frames={}, convert_options={:?})",
            self.inner.crop,
            self.inner.size,
            self.inner.filter,
            self.inner.padding_direction,
            self.inner.crop_max,
            self.inner.volume_handler,
            self.inner.use_components,
            self.inner.use_padding,
            self.inner.border_frac,
            self.inner.target_frames,
            self.inner.convert_options,
        ))
    }
}

/*
TODO: We preprocess by saving to a temporary TIFF file in memory, then decoding it back to an array.
It would be faster to directly preprocess the DICOM object without an intermediate TIFF file.
 */
fn preprocess_with_temp_tiff<'py, T>(
    py: Python<'py>,
    preprocessor: &Preprocessor,
    dcm: &FileDicomObject<InMemDicomObject>,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray4<T>>>
where
    T: Clone + Zero + Element,
    Array4<T>: LoadFromTiff<T>,
{
    // Run preprocessing
    let (images, metadata) = preprocessor
        .prepare_image(dcm, parallel)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to prepare image: {e}")))?;

    // Save the images to an in-memory temporary TIFF file
    let color_type = DicomColorType::try_from(dcm)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get color type: {e}")))?;
    let compressor = Compressor::Uncompressed(Uncompressed);
    let saver = TiffSaver::new(compressor, color_type);
    let mut buffer = spooled_tempfile(SPOOL_SIZE);
    let mut encoder = TiffEncoder::new(&mut buffer)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create TIFF encoder: {e}")))?;
    images
        .into_iter()
        .try_for_each(|image| saver.save(&mut encoder, &image, &metadata))
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to save temporary TIFF: {e}")))?;

    // Decode the TIFF file back to an array
    buffer
        .rewind()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to rewind buffer: {e}")))?;
    let mut decoder = Decoder::new(buffer)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create decoder: {e}")))?;
    let array = Array4::<T>::decode(&mut decoder)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to decode TIFF: {e}")))?;
    Ok(array.into_pyarray(py))
}

#[pymodule]
#[pyo3(name = "preprocess")]
pub(crate) fn register_submodule<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    fn preprocess_stream<'py, T>(
        py: Python<'py>,
        buffer: &PyBuffer<u8>,
        preprocessor: &Preprocessor,
        parallel: bool,
    ) -> PyResult<Bound<'py, PyArray4<T>>>
    where
        T: Clone + Zero + Element,
        Array4<T>: LoadFromTiff<T>,
    {
        // Check if buffer is readable and contiguous
        if !buffer.is_c_contiguous() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Buffer must be C-contiguous",
            ));
        }

        // Create a slice from the buffer and read it into a DICOM object
        let len = buffer.len_bytes();
        let ptr = buffer.buf_ptr();
        let bytes = unsafe { std::slice::from_raw_parts(ptr as *const u8, len) };
        let mut dcm = from_reader(bytes)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create DICOM object: {e}")))?;
        Preprocessor::sanitize_dicom(&mut dcm);
        preprocess_with_temp_tiff::<T>(py, preprocessor, &dcm, parallel)
    }

    #[pyfn(m)]
    #[pyo3(name = "preprocess_stream_u8", signature = (buffer, preprocessor, parallel=false))]
    fn preprocess_stream_u8<'py>(
        py: Python<'py>,
        buffer: &Bound<'py, PyAny>,
        preprocessor: &PyPreprocessor,
        parallel: bool,
    ) -> PyResult<Bound<'py, PyArray4<u8>>> {
        let buffer = buffer.extract::<PyBuffer<u8>>()?;
        preprocess_stream::<u8>(py, &buffer, &preprocessor.inner, parallel)
    }

    #[pyfn(m)]
    #[pyo3(name = "preprocess_stream_u16", signature = (buffer, preprocessor, parallel=false))]
    fn preprocess_stream_u16<'py>(
        py: Python<'py>,
        buffer: &Bound<'py, PyAny>,
        preprocessor: &PyPreprocessor,
        parallel: bool,
    ) -> PyResult<Bound<'py, PyArray4<u16>>> {
        let buffer = buffer.extract::<PyBuffer<u8>>()?;
        preprocess_stream::<u16>(py, &buffer, &preprocessor.inner, parallel)
    }

    #[pyfn(m)]
    #[pyo3(name = "preprocess_stream_f32", signature = (buffer, preprocessor, parallel=false))]
    fn preprocess_stream_f32<'py>(
        py: Python<'py>,
        buffer: &Bound<'py, PyAny>,
        preprocessor: &PyPreprocessor,
        parallel: bool,
    ) -> PyResult<Bound<'py, PyArray4<f32>>> {
        let buffer = buffer.extract::<PyBuffer<u8>>()?;
        preprocess_stream::<f32>(py, &buffer, &preprocessor.inner, parallel)
    }

    fn preprocess_file<'py, T, P>(
        py: Python<'py>,
        path: P,
        preprocessor: &Preprocessor,
        parallel: bool,
    ) -> PyResult<Bound<'py, PyArray4<T>>>
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

        let mut dcm = open_file(path)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to open DICOM file: {e}")))?;
        Preprocessor::sanitize_dicom(&mut dcm);
        preprocess_with_temp_tiff::<T>(py, preprocessor, &dcm, parallel)
    }

    #[pyfn(m)]
    #[pyo3(name = "preprocess_u8", signature = (path, preprocessor, parallel=false))]
    fn preprocess_u8<'py>(
        py: Python<'py>,
        path: &Bound<'py, PyAny>,
        preprocessor: &PyPreprocessor,
        parallel: bool,
    ) -> PyResult<Bound<'py, PyArray4<u8>>> {
        let path = path.extract::<PyPath>()?;
        preprocess_file::<u8, _>(py, path, &preprocessor.inner, parallel)
    }

    #[pyfn(m)]
    #[pyo3(name = "preprocess_u16", signature = (path, preprocessor, parallel=false))]
    fn preprocess_u16<'py>(
        py: Python<'py>,
        path: &Bound<'py, PyAny>,
        preprocessor: &PyPreprocessor,
        parallel: bool,
    ) -> PyResult<Bound<'py, PyArray4<u16>>> {
        let path = path.extract::<PyPath>()?;
        preprocess_file::<u16, _>(py, path, &preprocessor.inner, parallel)
    }

    #[pyfn(m)]
    #[pyo3(name = "preprocess_f32", signature = (path, preprocessor, parallel=false))]
    fn preprocess_f32<'py>(
        py: Python<'py>,
        path: &Bound<'py, PyAny>,
        preprocessor: &PyPreprocessor,
        parallel: bool,
    ) -> PyResult<Bound<'py, PyArray4<f32>>> {
        let path = path.extract::<PyPath>()?;
        preprocess_file::<f32, _>(py, path, &preprocessor.inner, parallel)
    }

    m.add_class::<PyPreprocessor>()?;
    Ok(())
}
