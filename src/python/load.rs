use crate::load::LoadFromTiff;
use image::GenericImageView;
use ndarray::Array4;
use dicom::object::open_file;
use std::io::BufReader;
use std::{fs::File, io::Read};
use tiff::decoder::Decoder;
use crate::preprocess::Preprocessor;
use crate::transform::{PaddingDirection, DisplayFilterType, DisplayVolumeHandler, VolumeHandler};
use image::imageops::FilterType;
use std::path::Path;
use numpy::{IntoPyArray, PyArray4};
use pyo3::{pymodule, types::PyModule, Bound, PyResult, Python, pymethods, pyclass};
use pyo3::prelude::*;
use pyo3::exceptions::{PyFileNotFoundError, PyValueError, PyIOError, PyRuntimeError};
use clap::ValueEnum;

const DEFAULT_FILTER: &str = "triangle";
const DEFAULT_PADDING_DIRECTION: &str = "zero";
const DEFAULT_VOLUME_HANDLER: &str = "keep";

#[pyclass]
struct PyPreprocessor {
    preprocessor: Preprocessor,
}

#[pymethods]
impl PyPreprocessor {
    #[pyo3(signature = (crop=None, size=None, filter=None, padding_direction=None, crop_max=None, volume_handler=None))]
    #[new]
    fn new(
        crop: Option<bool>,
        size: Option<(u32, u32)>,
        filter: Option<&str>,
        padding_direction: Option<&str>,
        crop_max: Option<bool>,
        volume_handler: Option<&str>,
    ) -> PyResult<Self> {
        // Parse args
        let crop = crop.unwrap_or(true);
        let filter: FilterType = DisplayFilterType::from_str(filter.unwrap_or("triangle"), true)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .into();
        let padding_direction: PaddingDirection = PaddingDirection::from_str(padding_direction.unwrap_or(DEFAULT_PADDING_DIRECTION), true)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .into();
        let crop_max = crop_max.unwrap_or(true);
        let volume_handler: VolumeHandler = DisplayVolumeHandler::from_str(volume_handler.unwrap_or(DEFAULT_VOLUME_HANDLER), true)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .into();

        let preprocessor = Preprocessor {
            crop,
            size,
            filter,
            padding_direction,
            crop_max,
            volume_handler,
        };
        Ok(PyPreprocessor { preprocessor })
    }

    #[pyo3(signature = (path, dest, parallel=None))]
    fn __call__<'py>(&self, path: &str, dest: &str, parallel: Option<bool>) -> PyResult<Bound<'py, PyArray4<f32>>> {
        let parallel = parallel.unwrap_or(false);
        let path = Path::new(path);
        if !path.is_file() {
            return Err(PyFileNotFoundError::new_err(format!("File not found: {}", path.display())));
        }

        let dcm = open_file(&path).map_err(|e| PyIOError::new_err(e.to_string()))?;
        let (images, metadata) = self.preprocessor.prepare_image(&dcm, parallel).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let num_frames = images.len();
        if num_frames == 0 {
            return Err(PyRuntimeError::new_err("No frames found in the DICOM series"));
        }
        let (width, height) = images[0].dimensions();

        if num_frames == 1 {
            let image = images[0].to_owned().as_luma16();
            let array = Array4::from_shape_vec(image.shape(), image.to_vec()).unwrap();
            return Ok(array.into_pyarray_bound(py));
        }
        
        let array = Array4::<f32>::decode_frames(&mut decoder, 0..num_frames).unwrap();



    }
}


#[pymodule]
fn dicom_preprocessing<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {



    #[pyfn(m)]
    #[pyo3(name = "load_tiff_f32", signature = (path, crop, size, filter, padding_direction, crop_max, volume_handler, parallel))]
    fn preprocess<'py>(
        py: Python<'py>, 
        path: &str,
        crop: bool,
        size: Option<(u32, u32)>,
        filter: &str,
        padding_direction: &str,
        crop_max: bool,
        volume_handler: &str,
        parallel: Option<bool>,
    ) -> PyResult<Bound<'py, PyArray4<f32>>> {
        // Parse args
        let path = Path::new(path);
        if !path.is_file() {
            return Err(PyFileNotFoundError::new_err(format!("File not found: {}", path.display())));
        }
        let filter: FilterType = DisplayFilterType::from_str(filter, true)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .into();
        let padding_direction: PaddingDirection = PaddingDirection::from_str(padding_direction, true)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .into();
        let volume_handler: VolumeHandler = DisplayVolumeHandler::from_str(volume_handler, true)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .into();
        let parallel = parallel.unwrap_or(false);

        let preprocessor = Preprocessor {
            crop,
            size,
            filter,
            padding_direction,
            crop_max,
            volume_handler,
        };
        let dcm = open_file(&path).map_err(|e| PyIOError::new_err(e.to_string()))?;
        let (images, metadata) = preprocessor.prepare_image(&dcm, parallel).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    }

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
