use crate::image_array::images_to_array;
use crate::load::LoadFromTiff;
use crate::metadata::{PreprocessingMetadata, Resolution};
use crate::preprocess::Preprocessor;
use crate::python::path::PyPath;
use crate::transform::resize::FilterType;
use crate::transform::volume::{
    CentralSlice, InterpolateVolume, KeepVolume, LaplacianMip, MaxIntensity, ProjectionMode,
    VolumeHandler,
};
use crate::transform::{Crop, Flip, Padding, PaddingDirection, Resize};
use crate::volume::DEFAULT_INTERPOLATE_TARGET_FRAMES;
use dicom::object::{from_reader, open_file, FileDicomObject, InMemDicomObject};
use dicom::pixeldata::{ConvertOptions, VoiLutOption, WindowLevel};
use ndarray::Array4;
use num::Zero;
use numpy::Element;
use numpy::{IntoPyArray, PyArray4};
use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::{
    exceptions::{PyFileNotFoundError, PyRuntimeError, PyValueError},
    pymodule,
    types::{PyAnyMethods, PyModule},
    Bound, PyAny, PyResult, Python,
};
use std::clone::Clone;
use std::path::Path;

const DEFAULT_LAPLACIAN_SKIP_FRAMES: i64 = 0;
const DEFAULT_LAPLACIAN_MIP_WEIGHT: f64 = 1.5;

#[pyclass(name = "VolumeHandler", frozen, from_py_object)]
#[derive(Debug, Clone)]
pub struct PyVolumeHandler {
    inner: VolumeHandler,
}

#[pymethods]
impl PyVolumeHandler {
    #[staticmethod]
    fn keep() -> Self {
        Self {
            inner: VolumeHandler::Keep(KeepVolume),
        }
    }

    #[staticmethod]
    fn central_slice() -> Self {
        Self {
            inner: VolumeHandler::CentralSlice(CentralSlice),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (skip_start=0, skip_end=0))]
    fn max_intensity(skip_start: i64, skip_end: i64) -> PyResult<Self> {
        Ok(Self {
            inner: VolumeHandler::MaxIntensity(MaxIntensity::new(
                nonnegative_u32("skip_start", skip_start)?,
                nonnegative_u32("skip_end", skip_end)?,
            )),
        })
    }

    #[staticmethod]
    fn interpolate(target_frames: i64) -> PyResult<Self> {
        Ok(Self {
            inner: VolumeHandler::Interpolate(InterpolateVolume::new(positive_u32(
                "target_frames",
                target_frames,
            )?)),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (
        skip_start=DEFAULT_LAPLACIAN_SKIP_FRAMES,
        skip_end=DEFAULT_LAPLACIAN_SKIP_FRAMES,
        mip_weight=DEFAULT_LAPLACIAN_MIP_WEIGHT,
        projection_mode="parallel-beam",
    ))]
    fn laplacian_mip(
        skip_start: i64,
        skip_end: i64,
        mip_weight: f64,
        projection_mode: &str,
    ) -> PyResult<Self> {
        if !mip_weight.is_finite() || mip_weight < 0.0 || mip_weight > f64::from(f32::MAX) {
            return Err(PyValueError::new_err(
                "mip_weight must be a finite non-negative number representable as float32",
            ));
        }
        let projection_mode = parse_projection_mode(projection_mode)?;
        Ok(Self {
            inner: VolumeHandler::LaplacianMip(
                LaplacianMip::new(
                    nonnegative_u32("skip_start", skip_start)?,
                    nonnegative_u32("skip_end", skip_end)?,
                )
                .with_mip_weight(mip_weight as f32)
                .with_projection_mode(projection_mode),
            ),
        })
    }

    fn __repr__(&self) -> String {
        format!("VolumeHandler({:?})", self.inner)
    }
}

fn nonnegative_u32(name: &str, value: i64) -> PyResult<u32> {
    u32::try_from(value).map_err(|_| {
        PyValueError::new_err(format!("{name} must be an integer from 0 to {}", u32::MAX))
    })
}

fn positive_u32(name: &str, value: i64) -> PyResult<u32> {
    let value = nonnegative_u32(name, value)?;
    if value == 0 {
        return Err(PyValueError::new_err(format!(
            "{name} must be greater than zero"
        )));
    }
    Ok(value)
}

fn parse_projection_mode(projection_mode: &str) -> PyResult<ProjectionMode> {
    match projection_mode.to_ascii_lowercase().as_str() {
        "central-slice" => Ok(ProjectionMode::CentralSlice),
        "parallel-beam" => Ok(ProjectionMode::ParallelBeam),
        _ => Err(PyValueError::new_err(format!(
            "Invalid projection mode: {projection_mode}"
        ))),
    }
}

fn parse_volume_handler(
    volume_handler: Option<&Bound<'_, PyAny>>,
    target_frames: u32,
) -> PyResult<VolumeHandler> {
    let Some(volume_handler) = volume_handler else {
        return Ok(VolumeHandler::Keep(KeepVolume));
    };
    if let Ok(name) = volume_handler.extract::<String>() {
        return match name.to_ascii_lowercase().as_str() {
            "keep" => Ok(VolumeHandler::Keep(KeepVolume)),
            "central" | "central-slice" => Ok(VolumeHandler::CentralSlice(CentralSlice)),
            "max-intensity" => Ok(VolumeHandler::MaxIntensity(MaxIntensity::default())),
            "interpolate" => Ok(VolumeHandler::Interpolate(InterpolateVolume::new(
                positive_u32("target_frames", i64::from(target_frames))?,
            ))),
            "laplacian-mip" => Ok(VolumeHandler::LaplacianMip(LaplacianMip::new(0, 0))),
            _ => Err(PyValueError::new_err(format!(
                "Invalid volume handler: {name}"
            ))),
        };
    }
    if let Ok(config) = volume_handler.extract::<PyRef<'_, PyVolumeHandler>>() {
        return Ok(config.inner.clone());
    }
    Err(PyValueError::new_err(
        "volume_handler must be a string or VolumeHandler",
    ))
}

#[pyclass(name = "Preprocessor", from_py_object)]
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
        spacing=None,
        filter="triangle",
        padding_direction="zero",
        crop_max=true,
        volume_handler=None,
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
        spacing: Option<(f32, f32, Option<f32>)>,
        filter: &str,
        padding_direction: &str,
        crop_max: bool,
        volume_handler: Option<&Bound<'_, PyAny>>,
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

        let volume_handler = parse_volume_handler(volume_handler, target_frames)?;

        let convert_options = match convert_options.to_lowercase().as_str() {
            "default" => ConvertOptions::default(),
            "normalize" => ConvertOptions::default().with_voi_lut(VoiLutOption::Normalize),
            s if s.contains(',') => {
                let mut parts = s.split(',');
                let (first, second) = (parts.next().unwrap(), parts.next().unwrap());
                let center = first.parse().map_err(|_| {
                    PyValueError::new_err(format!("Invalid window center: {first}"))
                })?;
                let width = second.parse().map_err(|_| {
                    PyValueError::new_err(format!("Invalid window width: {second}"))
                })?;
                let window = WindowLevel { center, width };
                let voi_lut = VoiLutOption::Custom(window);
                ConvertOptions::default().with_voi_lut(voi_lut)
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Invalid convert options: {convert_options}"
                )))
            }
        };

        let spacing_config = spacing.map(|(x, y, z)| {
            use crate::preprocess::SpacingConfig;
            match z {
                Some(z_val) => SpacingConfig::new_3d(x, y, z_val),
                None => SpacingConfig::new(x, y),
            }
        });

        let inner = Preprocessor {
            crop,
            size,
            spacing: spacing_config,
            filter,
            padding_direction,
            crop_max,
            volume_handler,
            use_components,
            use_padding,
            border_frac,
            target_frames,
            convert_options,
        };
        inner
            .validate()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;

        Ok(Self { inner })
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "Preprocessor(crop={}, size={:?}, spacing={:?}, filter={:?}, padding_direction={:?}, crop_max={}, volume_handler={:?}, use_components={}, use_padding={}, border_frac={:?}, target_frames={}, convert_options={:?})",
            self.inner.crop,
            self.inner.size,
            self.inner.spacing,
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

#[pyclass(name = "Flip", skip_from_py_object)]
#[derive(Clone)]
pub struct PyFlip {
    inner: Flip,
}

#[pymethods]
impl PyFlip {
    #[getter]
    fn width(&self) -> u32 {
        self.inner.width
    }

    #[getter]
    fn height(&self) -> u32 {
        self.inner.height
    }

    #[getter]
    fn horizontal(&self) -> bool {
        self.inner.horizontal
    }

    #[getter]
    fn vertical(&self) -> bool {
        self.inner.vertical
    }

    fn __repr__(&self) -> String {
        format!(
            "Flip(width={}, height={}, horizontal={}, vertical={})",
            self.inner.width, self.inner.height, self.inner.horizontal, self.inner.vertical
        )
    }
}

impl From<Flip> for PyFlip {
    fn from(flip: Flip) -> Self {
        PyFlip { inner: flip }
    }
}

#[pyclass(name = "Crop", skip_from_py_object)]
#[derive(Clone)]
pub struct PyCrop {
    inner: Crop,
}

#[pymethods]
impl PyCrop {
    #[getter]
    fn left(&self) -> u32 {
        self.inner.left
    }

    #[getter]
    fn top(&self) -> u32 {
        self.inner.top
    }

    #[getter]
    fn width(&self) -> u32 {
        self.inner.width
    }

    #[getter]
    fn height(&self) -> u32 {
        self.inner.height
    }

    fn __repr__(&self) -> String {
        format!(
            "Crop(left={}, top={}, width={}, height={})",
            self.inner.left, self.inner.top, self.inner.width, self.inner.height
        )
    }
}

impl From<Crop> for PyCrop {
    fn from(crop: Crop) -> Self {
        PyCrop { inner: crop }
    }
}

#[pyclass(name = "Resize", skip_from_py_object)]
#[derive(Clone)]
pub struct PyResize {
    inner: Resize,
}

#[pymethods]
impl PyResize {
    #[getter]
    fn scale_x(&self) -> f32 {
        self.inner.scale_x
    }

    #[getter]
    fn scale_y(&self) -> f32 {
        self.inner.scale_y
    }

    #[getter]
    fn filter(&self) -> String {
        format!("{:?}", self.inner.filter).to_lowercase()
    }

    fn __repr__(&self) -> String {
        format!(
            "Resize(scale_x={}, scale_y={}, filter={:?})",
            self.inner.scale_x, self.inner.scale_y, self.inner.filter
        )
    }
}

impl From<Resize> for PyResize {
    fn from(resize: Resize) -> Self {
        PyResize { inner: resize }
    }
}

#[pyclass(name = "Padding", skip_from_py_object)]
#[derive(Clone)]
pub struct PyPadding {
    inner: Padding,
}

#[pymethods]
impl PyPadding {
    #[getter]
    fn left(&self) -> u32 {
        self.inner.left
    }

    #[getter]
    fn top(&self) -> u32 {
        self.inner.top
    }

    #[getter]
    fn right(&self) -> u32 {
        self.inner.right
    }

    #[getter]
    fn bottom(&self) -> u32 {
        self.inner.bottom
    }

    fn __repr__(&self) -> String {
        format!(
            "Padding(left={}, top={}, right={}, bottom={})",
            self.inner.left, self.inner.top, self.inner.right, self.inner.bottom
        )
    }
}

impl From<Padding> for PyPadding {
    fn from(padding: Padding) -> Self {
        PyPadding { inner: padding }
    }
}

#[pyclass(name = "Resolution", skip_from_py_object)]
#[derive(Clone)]
pub struct PyResolution {
    inner: Resolution,
}

#[pymethods]
impl PyResolution {
    #[getter]
    fn pixels_per_mm_x(&self) -> f32 {
        self.inner.pixels_per_mm_x
    }

    #[getter]
    fn pixels_per_mm_y(&self) -> f32 {
        self.inner.pixels_per_mm_y
    }

    #[getter]
    fn frames_per_mm(&self) -> Option<f32> {
        self.inner.frames_per_mm
    }

    fn __repr__(&self) -> String {
        format!(
            "Resolution(pixels_per_mm_x={}, pixels_per_mm_y={}, frames_per_mm={:?})",
            self.inner.pixels_per_mm_x, self.inner.pixels_per_mm_y, self.inner.frames_per_mm
        )
    }
}

impl From<Resolution> for PyResolution {
    fn from(resolution: Resolution) -> Self {
        PyResolution { inner: resolution }
    }
}

#[pyclass(name = "PreprocessingMetadata", skip_from_py_object)]
#[derive(Clone)]
pub struct PyPreprocessingMetadata {
    inner: PreprocessingMetadata,
}

#[pymethods]
impl PyPreprocessingMetadata {
    #[getter]
    fn flip(&self) -> Option<PyFlip> {
        self.inner.flip.map(|f| f.into())
    }

    #[getter]
    fn crop(&self) -> Option<PyCrop> {
        self.inner.crop.map(|c| c.into())
    }

    #[getter]
    fn resize(&self) -> Option<PyResize> {
        self.inner.resize.map(|r| r.into())
    }

    #[getter]
    fn padding(&self) -> Option<PyPadding> {
        self.inner.padding.map(|p| p.into())
    }

    #[getter]
    fn resolution(&self) -> Option<PyResolution> {
        self.inner.resolution.map(|r| r.into())
    }

    #[getter]
    fn num_frames(&self) -> u16 {
        self.inner.num_frames.into()
    }

    fn __repr__(&self) -> String {
        format!(
            "PreprocessingMetadata(flip={:?}, crop={:?}, resize={:?}, padding={:?}, resolution={:?}, num_frames={})",
            self.inner.flip.is_some(),
            self.inner.crop.is_some(),
            self.inner.resize.is_some(),
            self.inner.padding.is_some(),
            self.inner.resolution.is_some(),
            u16::from(self.inner.num_frames)
        )
    }
}

impl From<PreprocessingMetadata> for PyPreprocessingMetadata {
    fn from(metadata: PreprocessingMetadata) -> Self {
        PyPreprocessingMetadata { inner: metadata }
    }
}

fn preprocess_to_array<'py, T>(
    py: Python<'py>,
    preprocessor: &Preprocessor,
    dcm: &FileDicomObject<InMemDicomObject>,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray4<T>>>
where
    T: Clone + Zero + Element,
    Array4<T>: LoadFromTiff<T>,
{
    let (array, _metadata) = preprocess_to_array_and_metadata(py, preprocessor, dcm, parallel)?;
    Ok(array)
}

fn preprocess_to_array_and_metadata<'py, T>(
    py: Python<'py>,
    preprocessor: &Preprocessor,
    dcm: &FileDicomObject<InMemDicomObject>,
    parallel: bool,
) -> PyResult<(Bound<'py, PyArray4<T>>, PyPreprocessingMetadata)>
where
    T: Clone + Zero + Element,
    Array4<T>: LoadFromTiff<T>,
{
    let (array, metadata) = py
        .detach(|| {
            let (images, metadata) = preprocessor
                .prepare_image(dcm, parallel)
                .map_err(|error| format!("Failed to prepare image: {error}"))?;
            let array = images_to_array::<T>(images)
                .map_err(|error| format!("Failed to create image array: {error}"))?;
            Ok::<_, String>((array, metadata))
        })
        .map_err(PyRuntimeError::new_err)?;
    Ok((array.into_pyarray(py), metadata.into()))
}

/*
Preprocess multiple DICOM files (slices) with common crop bounds.
Input order is preserved; automatic metadata-based reordering only applies to
single multi-frame DICOM objects.
Returns a 5D array with shape (num_slices, num_frames, height, width, channels).
 */
fn preprocess_slices_to_arrays<'py, T>(
    py: Python<'py>,
    preprocessor: &Preprocessor,
    dcms: &[FileDicomObject<InMemDicomObject>],
    parallel: bool,
) -> PyResult<Vec<Bound<'py, PyArray4<T>>>>
where
    T: Clone + Zero + Element,
    Array4<T>: LoadFromTiff<T>,
{
    let (arrays, _metadata) =
        preprocess_slices_to_arrays_and_metadata(py, preprocessor, dcms, parallel)?;
    Ok(arrays)
}

fn preprocess_slices_to_arrays_and_metadata<'py, T>(
    py: Python<'py>,
    preprocessor: &Preprocessor,
    dcms: &[FileDicomObject<InMemDicomObject>],
    parallel: bool,
) -> PyResult<(Vec<Bound<'py, PyArray4<T>>>, PyPreprocessingMetadata)>
where
    T: Clone + Zero + Element,
    Array4<T>: LoadFromTiff<T>,
{
    if dcms.is_empty() {
        return Err(PyRuntimeError::new_err(
            "Cannot process empty list of DICOMs",
        ));
    }

    let (arrays, metadata) = py
        .detach(|| {
            let (batch_images, metadata) = preprocessor
                .prepare_images_batch(dcms, parallel)
                .map_err(|error| format!("Failed to prepare images batch: {error}"))?;
            let arrays = batch_images
                .into_iter()
                .map(images_to_array::<T>)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| format!("Failed to create image array: {error}"))?;
            Ok::<_, String>((arrays, metadata))
        })
        .map_err(PyRuntimeError::new_err)?;
    let result_arrays = arrays
        .into_iter()
        .map(|array| array.into_pyarray(py))
        .collect();

    Ok((result_arrays, metadata.into()))
}

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
    preprocess_to_array::<T>(py, preprocessor, &dcm, parallel)
}

#[pyfunction]
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

#[pyfunction]
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

#[pyfunction]
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
    preprocess_to_array::<T>(py, preprocessor, &dcm, parallel)
}

#[pyfunction]
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

#[pyfunction]
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

#[pyfunction]
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

fn preprocess_slices<'py, T, P>(
    py: Python<'py>,
    paths: Vec<P>,
    preprocessor: &Preprocessor,
    parallel: bool,
) -> PyResult<Vec<Bound<'py, PyArray4<T>>>>
where
    T: Clone + Zero + Element,
    Array4<T>: LoadFromTiff<T>,
    P: AsRef<Path>,
{
    let mut dcms = Vec::with_capacity(paths.len());
    for path in &paths {
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
        dcms.push(dcm);
    }
    preprocess_slices_to_arrays::<T>(py, preprocessor, &dcms, parallel)
}

#[pyfunction]
#[pyo3(name = "preprocess_u8_slices", signature = (paths, preprocessor, parallel=false))]
fn preprocess_u8_slices<'py>(
    py: Python<'py>,
    paths: Vec<Bound<'py, PyAny>>,
    preprocessor: &PyPreprocessor,
    parallel: bool,
) -> PyResult<Vec<Bound<'py, PyArray4<u8>>>> {
    let paths: Result<Vec<PyPath>, _> = paths.iter().map(|p| p.extract::<PyPath>()).collect();
    let paths = paths?;
    preprocess_slices::<u8, _>(py, paths, &preprocessor.inner, parallel)
}

#[pyfunction]
#[pyo3(name = "preprocess_u16_slices", signature = (paths, preprocessor, parallel=false))]
fn preprocess_u16_slices<'py>(
    py: Python<'py>,
    paths: Vec<Bound<'py, PyAny>>,
    preprocessor: &PyPreprocessor,
    parallel: bool,
) -> PyResult<Vec<Bound<'py, PyArray4<u16>>>> {
    let paths: Result<Vec<PyPath>, _> = paths.iter().map(|p| p.extract::<PyPath>()).collect();
    let paths = paths?;
    preprocess_slices::<u16, _>(py, paths, &preprocessor.inner, parallel)
}

#[pyfunction]
#[pyo3(name = "preprocess_f32_slices", signature = (paths, preprocessor, parallel=false))]
fn preprocess_f32_slices<'py>(
    py: Python<'py>,
    paths: Vec<Bound<'py, PyAny>>,
    preprocessor: &PyPreprocessor,
    parallel: bool,
) -> PyResult<Vec<Bound<'py, PyArray4<f32>>>> {
    let paths: Result<Vec<PyPath>, _> = paths.iter().map(|p| p.extract::<PyPath>()).collect();
    let paths = paths?;
    preprocess_slices::<f32, _>(py, paths, &preprocessor.inner, parallel)
}

fn preprocess_stream_slices<'py, T>(
    py: Python<'py>,
    buffers: Vec<&PyBuffer<u8>>,
    preprocessor: &Preprocessor,
    parallel: bool,
) -> PyResult<Vec<Bound<'py, PyArray4<T>>>>
where
    T: Clone + Zero + Element,
    Array4<T>: LoadFromTiff<T>,
{
    let mut dcms = Vec::with_capacity(buffers.len());
    for buffer in buffers {
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
        dcms.push(dcm);
    }
    preprocess_slices_to_arrays::<T>(py, preprocessor, &dcms, parallel)
}

#[pyfunction]
#[pyo3(name = "preprocess_stream_u8_slices", signature = (buffers, preprocessor, parallel=false))]
fn preprocess_stream_u8_slices<'py>(
    py: Python<'py>,
    buffers: Vec<Bound<'py, PyAny>>,
    preprocessor: &PyPreprocessor,
    parallel: bool,
) -> PyResult<Vec<Bound<'py, PyArray4<u8>>>> {
    let buffers: Result<Vec<PyBuffer<u8>>, _> = buffers
        .iter()
        .map(|b| b.extract::<PyBuffer<u8>>())
        .collect();
    let buffers = buffers?;
    let buffer_refs: Vec<&PyBuffer<u8>> = buffers.iter().collect();
    preprocess_stream_slices::<u8>(py, buffer_refs, &preprocessor.inner, parallel)
}

#[pyfunction]
#[pyo3(name = "preprocess_stream_u16_slices", signature = (buffers, preprocessor, parallel=false))]
fn preprocess_stream_u16_slices<'py>(
    py: Python<'py>,
    buffers: Vec<Bound<'py, PyAny>>,
    preprocessor: &PyPreprocessor,
    parallel: bool,
) -> PyResult<Vec<Bound<'py, PyArray4<u16>>>> {
    let buffers: Result<Vec<PyBuffer<u8>>, _> = buffers
        .iter()
        .map(|b| b.extract::<PyBuffer<u8>>())
        .collect();
    let buffers = buffers?;
    let buffer_refs: Vec<&PyBuffer<u8>> = buffers.iter().collect();
    preprocess_stream_slices::<u16>(py, buffer_refs, &preprocessor.inner, parallel)
}

#[pyfunction]
#[pyo3(name = "preprocess_stream_f32_slices", signature = (buffers, preprocessor, parallel=false))]
fn preprocess_stream_f32_slices<'py>(
    py: Python<'py>,
    buffers: Vec<Bound<'py, PyAny>>,
    preprocessor: &PyPreprocessor,
    parallel: bool,
) -> PyResult<Vec<Bound<'py, PyArray4<f32>>>> {
    let buffers: Result<Vec<PyBuffer<u8>>, _> = buffers
        .iter()
        .map(|b| b.extract::<PyBuffer<u8>>())
        .collect();
    let buffers = buffers?;
    let buffer_refs: Vec<&PyBuffer<u8>> = buffers.iter().collect();
    preprocess_stream_slices::<f32>(py, buffer_refs, &preprocessor.inner, parallel)
}

#[pyfunction]
#[pyo3(name = "preprocess_u8_with_metadata", signature = (path, preprocessor, parallel=false))]
fn preprocess_u8_with_metadata<'py>(
    py: Python<'py>,
    path: &Bound<'py, PyAny>,
    preprocessor: &PyPreprocessor,
    parallel: bool,
) -> PyResult<(Bound<'py, PyArray4<u8>>, PyPreprocessingMetadata)> {
    let path = path.extract::<PyPath>()?;
    let path: &Path = path.as_ref();
    if !path.is_file() {
        return Err(PyFileNotFoundError::new_err(format!(
            "File not found: {}",
            path.display()
        )));
    }
    let mut dcm = open_file(path)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to open DICOM file: {e}")))?;
    Preprocessor::sanitize_dicom(&mut dcm);
    preprocess_to_array_and_metadata::<u8>(py, &preprocessor.inner, &dcm, parallel)
}

#[pyfunction]
#[pyo3(name = "preprocess_u16_with_metadata", signature = (path, preprocessor, parallel=false))]
fn preprocess_u16_with_metadata<'py>(
    py: Python<'py>,
    path: &Bound<'py, PyAny>,
    preprocessor: &PyPreprocessor,
    parallel: bool,
) -> PyResult<(Bound<'py, PyArray4<u16>>, PyPreprocessingMetadata)> {
    let path = path.extract::<PyPath>()?;
    let path: &Path = path.as_ref();
    if !path.is_file() {
        return Err(PyFileNotFoundError::new_err(format!(
            "File not found: {}",
            path.display()
        )));
    }
    let mut dcm = open_file(path)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to open DICOM file: {e}")))?;
    Preprocessor::sanitize_dicom(&mut dcm);
    preprocess_to_array_and_metadata::<u16>(py, &preprocessor.inner, &dcm, parallel)
}

#[pyfunction]
#[pyo3(name = "preprocess_f32_with_metadata", signature = (path, preprocessor, parallel=false))]
fn preprocess_f32_with_metadata<'py>(
    py: Python<'py>,
    path: &Bound<'py, PyAny>,
    preprocessor: &PyPreprocessor,
    parallel: bool,
) -> PyResult<(Bound<'py, PyArray4<f32>>, PyPreprocessingMetadata)> {
    let path = path.extract::<PyPath>()?;
    let path: &Path = path.as_ref();
    if !path.is_file() {
        return Err(PyFileNotFoundError::new_err(format!(
            "File not found: {}",
            path.display()
        )));
    }
    let mut dcm = open_file(path)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to open DICOM file: {e}")))?;
    Preprocessor::sanitize_dicom(&mut dcm);
    preprocess_to_array_and_metadata::<f32>(py, &preprocessor.inner, &dcm, parallel)
}

#[pyfunction]
#[pyo3(name = "preprocess_stream_u8_with_metadata", signature = (buffer, preprocessor, parallel=false))]
fn preprocess_stream_u8_with_metadata<'py>(
    py: Python<'py>,
    buffer: &Bound<'py, PyAny>,
    preprocessor: &PyPreprocessor,
    parallel: bool,
) -> PyResult<(Bound<'py, PyArray4<u8>>, PyPreprocessingMetadata)> {
    let buffer = buffer.extract::<PyBuffer<u8>>()?;
    if !buffer.is_c_contiguous() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Buffer must be C-contiguous",
        ));
    }
    let len = buffer.len_bytes();
    let ptr = buffer.buf_ptr();
    let bytes = unsafe { std::slice::from_raw_parts(ptr as *const u8, len) };
    let mut dcm = from_reader(bytes)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create DICOM object: {e}")))?;
    Preprocessor::sanitize_dicom(&mut dcm);
    preprocess_to_array_and_metadata::<u8>(py, &preprocessor.inner, &dcm, parallel)
}

#[pyfunction]
#[pyo3(name = "preprocess_stream_u16_with_metadata", signature = (buffer, preprocessor, parallel=false))]
fn preprocess_stream_u16_with_metadata<'py>(
    py: Python<'py>,
    buffer: &Bound<'py, PyAny>,
    preprocessor: &PyPreprocessor,
    parallel: bool,
) -> PyResult<(Bound<'py, PyArray4<u16>>, PyPreprocessingMetadata)> {
    let buffer = buffer.extract::<PyBuffer<u8>>()?;
    if !buffer.is_c_contiguous() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Buffer must be C-contiguous",
        ));
    }
    let len = buffer.len_bytes();
    let ptr = buffer.buf_ptr();
    let bytes = unsafe { std::slice::from_raw_parts(ptr as *const u8, len) };
    let mut dcm = from_reader(bytes)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create DICOM object: {e}")))?;
    Preprocessor::sanitize_dicom(&mut dcm);
    preprocess_to_array_and_metadata::<u16>(py, &preprocessor.inner, &dcm, parallel)
}

#[pyfunction]
#[pyo3(name = "preprocess_stream_f32_with_metadata", signature = (buffer, preprocessor, parallel=false))]
fn preprocess_stream_f32_with_metadata<'py>(
    py: Python<'py>,
    buffer: &Bound<'py, PyAny>,
    preprocessor: &PyPreprocessor,
    parallel: bool,
) -> PyResult<(Bound<'py, PyArray4<f32>>, PyPreprocessingMetadata)> {
    let buffer = buffer.extract::<PyBuffer<u8>>()?;
    if !buffer.is_c_contiguous() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Buffer must be C-contiguous",
        ));
    }
    let len = buffer.len_bytes();
    let ptr = buffer.buf_ptr();
    let bytes = unsafe { std::slice::from_raw_parts(ptr as *const u8, len) };
    let mut dcm = from_reader(bytes)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create DICOM object: {e}")))?;
    Preprocessor::sanitize_dicom(&mut dcm);
    preprocess_to_array_and_metadata::<f32>(py, &preprocessor.inner, &dcm, parallel)
}

#[pyfunction]
#[pyo3(name = "preprocess_u8_slices_with_metadata", signature = (paths, preprocessor, parallel=false))]
fn preprocess_u8_slices_with_metadata<'py>(
    py: Python<'py>,
    paths: Vec<Bound<'py, PyAny>>,
    preprocessor: &PyPreprocessor,
    parallel: bool,
) -> PyResult<(Vec<Bound<'py, PyArray4<u8>>>, PyPreprocessingMetadata)> {
    let paths: Result<Vec<PyPath>, _> = paths.iter().map(|p| p.extract::<PyPath>()).collect();
    let paths = paths?;
    let mut dcms = Vec::with_capacity(paths.len());
    for path in &paths {
        let path: &Path = path.as_ref();
        if !path.is_file() {
            return Err(PyFileNotFoundError::new_err(format!(
                "File not found: {}",
                path.display()
            )));
        }
        let mut dcm = open_file(path)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to open DICOM file: {e}")))?;
        Preprocessor::sanitize_dicom(&mut dcm);
        dcms.push(dcm);
    }
    preprocess_slices_to_arrays_and_metadata::<u8>(py, &preprocessor.inner, &dcms, parallel)
}

#[pyfunction]
#[pyo3(name = "preprocess_u16_slices_with_metadata", signature = (paths, preprocessor, parallel=false))]
fn preprocess_u16_slices_with_metadata<'py>(
    py: Python<'py>,
    paths: Vec<Bound<'py, PyAny>>,
    preprocessor: &PyPreprocessor,
    parallel: bool,
) -> PyResult<(Vec<Bound<'py, PyArray4<u16>>>, PyPreprocessingMetadata)> {
    let paths: Result<Vec<PyPath>, _> = paths.iter().map(|p| p.extract::<PyPath>()).collect();
    let paths = paths?;
    let mut dcms = Vec::with_capacity(paths.len());
    for path in &paths {
        let path: &Path = path.as_ref();
        if !path.is_file() {
            return Err(PyFileNotFoundError::new_err(format!(
                "File not found: {}",
                path.display()
            )));
        }
        let mut dcm = open_file(path)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to open DICOM file: {e}")))?;
        Preprocessor::sanitize_dicom(&mut dcm);
        dcms.push(dcm);
    }
    preprocess_slices_to_arrays_and_metadata::<u16>(py, &preprocessor.inner, &dcms, parallel)
}

#[pyfunction]
#[pyo3(name = "preprocess_f32_slices_with_metadata", signature = (paths, preprocessor, parallel=false))]
fn preprocess_f32_slices_with_metadata<'py>(
    py: Python<'py>,
    paths: Vec<Bound<'py, PyAny>>,
    preprocessor: &PyPreprocessor,
    parallel: bool,
) -> PyResult<(Vec<Bound<'py, PyArray4<f32>>>, PyPreprocessingMetadata)> {
    let paths: Result<Vec<PyPath>, _> = paths.iter().map(|p| p.extract::<PyPath>()).collect();
    let paths = paths?;
    let mut dcms = Vec::with_capacity(paths.len());
    for path in &paths {
        let path: &Path = path.as_ref();
        if !path.is_file() {
            return Err(PyFileNotFoundError::new_err(format!(
                "File not found: {}",
                path.display()
            )));
        }
        let mut dcm = open_file(path)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to open DICOM file: {e}")))?;
        Preprocessor::sanitize_dicom(&mut dcm);
        dcms.push(dcm);
    }
    preprocess_slices_to_arrays_and_metadata::<f32>(py, &preprocessor.inner, &dcms, parallel)
}

#[pyfunction]
#[pyo3(
    name = "preprocess_stream_u8_slices_with_metadata",
    signature = (buffers, preprocessor, parallel=false)
)]
fn preprocess_stream_u8_slices_with_metadata<'py>(
    py: Python<'py>,
    buffers: Vec<Bound<'py, PyAny>>,
    preprocessor: &PyPreprocessor,
    parallel: bool,
) -> PyResult<(Vec<Bound<'py, PyArray4<u8>>>, PyPreprocessingMetadata)> {
    let buffers: Result<Vec<PyBuffer<u8>>, _> = buffers
        .iter()
        .map(|b| b.extract::<PyBuffer<u8>>())
        .collect();
    let buffers = buffers?;
    let mut dcms = Vec::with_capacity(buffers.len());
    for buffer in &buffers {
        if !buffer.is_c_contiguous() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Buffer must be C-contiguous",
            ));
        }
        let len = buffer.len_bytes();
        let ptr = buffer.buf_ptr();
        let bytes = unsafe { std::slice::from_raw_parts(ptr as *const u8, len) };
        let mut dcm = from_reader(bytes)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create DICOM object: {e}")))?;
        Preprocessor::sanitize_dicom(&mut dcm);
        dcms.push(dcm);
    }
    preprocess_slices_to_arrays_and_metadata::<u8>(py, &preprocessor.inner, &dcms, parallel)
}

#[pyfunction]
#[pyo3(
    name = "preprocess_stream_u16_slices_with_metadata",
    signature = (buffers, preprocessor, parallel=false)
)]
fn preprocess_stream_u16_slices_with_metadata<'py>(
    py: Python<'py>,
    buffers: Vec<Bound<'py, PyAny>>,
    preprocessor: &PyPreprocessor,
    parallel: bool,
) -> PyResult<(Vec<Bound<'py, PyArray4<u16>>>, PyPreprocessingMetadata)> {
    let buffers: Result<Vec<PyBuffer<u8>>, _> = buffers
        .iter()
        .map(|b| b.extract::<PyBuffer<u8>>())
        .collect();
    let buffers = buffers?;
    let mut dcms = Vec::with_capacity(buffers.len());
    for buffer in &buffers {
        if !buffer.is_c_contiguous() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Buffer must be C-contiguous",
            ));
        }
        let len = buffer.len_bytes();
        let ptr = buffer.buf_ptr();
        let bytes = unsafe { std::slice::from_raw_parts(ptr as *const u8, len) };
        let mut dcm = from_reader(bytes)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create DICOM object: {e}")))?;
        Preprocessor::sanitize_dicom(&mut dcm);
        dcms.push(dcm);
    }
    preprocess_slices_to_arrays_and_metadata::<u16>(py, &preprocessor.inner, &dcms, parallel)
}

#[pyfunction]
#[pyo3(
    name = "preprocess_stream_f32_slices_with_metadata",
    signature = (buffers, preprocessor, parallel=false)
)]
fn preprocess_stream_f32_slices_with_metadata<'py>(
    py: Python<'py>,
    buffers: Vec<Bound<'py, PyAny>>,
    preprocessor: &PyPreprocessor,
    parallel: bool,
) -> PyResult<(Vec<Bound<'py, PyArray4<f32>>>, PyPreprocessingMetadata)> {
    let buffers: Result<Vec<PyBuffer<u8>>, _> = buffers
        .iter()
        .map(|b| b.extract::<PyBuffer<u8>>())
        .collect();
    let buffers = buffers?;
    let mut dcms = Vec::with_capacity(buffers.len());
    for buffer in &buffers {
        if !buffer.is_c_contiguous() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Buffer must be C-contiguous",
            ));
        }
        let len = buffer.len_bytes();
        let ptr = buffer.buf_ptr();
        let bytes = unsafe { std::slice::from_raw_parts(ptr as *const u8, len) };
        let mut dcm = from_reader(bytes)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create DICOM object: {e}")))?;
        Preprocessor::sanitize_dicom(&mut dcm);
        dcms.push(dcm);
    }
    preprocess_slices_to_arrays_and_metadata::<f32>(py, &preprocessor.inner, &dcms, parallel)
}

#[pymodule]
#[pyo3(name = "preprocess")]
pub(crate) fn register_submodule<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(preprocess_stream_u8, m)?)?;
    m.add_function(wrap_pyfunction!(preprocess_stream_u16, m)?)?;
    m.add_function(wrap_pyfunction!(preprocess_stream_f32, m)?)?;
    m.add_function(wrap_pyfunction!(preprocess_u8, m)?)?;
    m.add_function(wrap_pyfunction!(preprocess_u16, m)?)?;
    m.add_function(wrap_pyfunction!(preprocess_f32, m)?)?;
    m.add_function(wrap_pyfunction!(preprocess_u8_slices, m)?)?;
    m.add_function(wrap_pyfunction!(preprocess_u16_slices, m)?)?;
    m.add_function(wrap_pyfunction!(preprocess_f32_slices, m)?)?;
    m.add_function(wrap_pyfunction!(preprocess_stream_u8_slices, m)?)?;
    m.add_function(wrap_pyfunction!(preprocess_stream_u16_slices, m)?)?;
    m.add_function(wrap_pyfunction!(preprocess_stream_f32_slices, m)?)?;
    m.add_function(wrap_pyfunction!(preprocess_u8_with_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(preprocess_u16_with_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(preprocess_f32_with_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(preprocess_stream_u8_with_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(preprocess_stream_u16_with_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(preprocess_stream_f32_with_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(preprocess_u8_slices_with_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(preprocess_u16_slices_with_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(preprocess_f32_slices_with_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(
        preprocess_stream_u8_slices_with_metadata,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        preprocess_stream_u16_slices_with_metadata,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        preprocess_stream_f32_slices_with_metadata,
        m
    )?)?;
    m.add_class::<PyVolumeHandler>()?;
    m.add_class::<PyPreprocessor>()?;
    m.add_class::<PyFlip>()?;
    m.add_class::<PyCrop>()?;
    m.add_class::<PyResize>()?;
    m.add_class::<PyPadding>()?;
    m.add_class::<PyResolution>()?;
    m.add_class::<PyPreprocessingMetadata>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use dicom::object::open_file;

    fn assert_handler_output_parity(python: PyVolumeHandler, rust: VolumeHandler) {
        let file = open_file(dicom_test_files::path("pydicom/emri_small.dcm").unwrap()).unwrap();
        let options = ConvertOptions::default();
        let actual = python
            .inner
            .prepare_volume_with_options(&file, &options, false)
            .unwrap()
            .images;
        let expected = rust
            .prepare_volume_with_options(&file, &options, false)
            .unwrap()
            .images;
        assert_eq!(actual, expected);
    }

    #[test]
    fn typed_volume_handlers_match_direct_rust_outputs() {
        assert_handler_output_parity(PyVolumeHandler::keep(), VolumeHandler::Keep(KeepVolume));
        assert_handler_output_parity(
            PyVolumeHandler::central_slice(),
            VolumeHandler::CentralSlice(CentralSlice),
        );
        assert_handler_output_parity(
            PyVolumeHandler::max_intensity(1, 2).unwrap(),
            VolumeHandler::MaxIntensity(MaxIntensity::new(1, 2)),
        );
        assert_handler_output_parity(
            PyVolumeHandler::interpolate(6).unwrap(),
            VolumeHandler::Interpolate(InterpolateVolume::new(6)),
        );
        assert_handler_output_parity(
            PyVolumeHandler::laplacian_mip(0, 0, 0.75, "central-slice").unwrap(),
            VolumeHandler::LaplacianMip(
                LaplacianMip::new(0, 0)
                    .with_mip_weight(0.75)
                    .with_projection_mode(ProjectionMode::CentralSlice),
            ),
        );
    }

    #[test]
    fn typed_volume_handler_validation_returns_python_errors() {
        assert!(PyVolumeHandler::max_intensity(-1, 0).is_err());
        assert!(PyVolumeHandler::interpolate(0).is_err());
        assert!(PyVolumeHandler::laplacian_mip(0, 0, f64::NAN, "parallel-beam").is_err());
        assert!(PyVolumeHandler::laplacian_mip(0, 0, 1.5, "invalid").is_err());
    }
}
