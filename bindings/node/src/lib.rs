pub mod verification;

use std::io::Cursor;

use dicom::object::from_reader;
use dicom::pixeldata::{ConvertOptions, PlanarConfiguration};
use dicom_preprocessing::{
    metadata::pixel_spacing_mm, CoordinateTransform, DicomError, Flip, FlipOptions,
    FrameOrderStrategy, PixelDimensions, PixelRect, ViewerDicom, ViewerOptions, VolumeFrameSource,
    VolumeHandler,
};
use napi::bindgen_prelude::{Buffer, Object, Uint8Array, Unknown};
use napi::{Error, JsValue, ValueType};
use napi_derive::napi;

const CODE_INVALID_INPUT: &str = "INVALID_INPUT";
const CODE_READ_FILE: &str = "READ_FILE";
const CODE_READ_BYTES: &str = "READ_BYTES";
const CODE_UNSUPPORTED_TRANSFER_SYNTAX: &str = "UNSUPPORTED_TRANSFER_SYNTAX";
const CODE_UNSUPPORTED_IMAGE_LAYOUT: &str = "UNSUPPORTED_IMAGE_LAYOUT";
const CODE_FRAME_INDEX_OUT_OF_RANGE: &str = "FRAME_INDEX_OUT_OF_RANGE";
const CODE_DERIVED_FRAME_NO_RAW_SOURCE: &str = "DERIVED_FRAME_NO_RAW_SOURCE";
const CODE_PIXEL_DECODE: &str = "PIXEL_DECODE";

#[napi(object)]
#[derive(Clone)]
pub struct NodeFrameSource {
    pub kind: String,
    pub stored_frame_index: Option<u32>,
}

#[napi(object)]
#[derive(Clone)]
pub struct NodeFramePlan {
    pub display_frames: Vec<NodeFrameSource>,
    pub stored_frame_order: Vec<u32>,
    pub frame_order_strategy: String,
    pub z_spacing_mm: Option<f64>,
}

#[napi(object)]
#[derive(Clone)]
pub struct NodePixelDimensions {
    pub width: u32,
    pub height: u32,
}

#[napi(object)]
#[derive(Clone)]
pub struct NodePixelRect {
    pub left: u32,
    pub top: u32,
    pub width: u32,
    pub height: u32,
}

#[napi(object)]
#[derive(Clone)]
pub struct NodeCoordinateTransform {
    pub source_dimensions: NodePixelDimensions,
    pub display_dimensions: NodePixelDimensions,
    #[napi(ts_type = "[number, number, number, number, number, number]")]
    pub source_to_display: Vec<f64>,
    #[napi(ts_type = "[number, number, number, number, number, number]")]
    pub display_to_source: Vec<f64>,
    pub valid_source_rect: NodePixelRect,
    pub valid_display_rect: NodePixelRect,
}

#[napi(object)]
pub struct NodeRenderedFrame {
    pub display_frame_index: u32,
    pub source: NodeFrameSource,
    pub coordinate_transform: NodeCoordinateTransform,
    pub data: Buffer,
    pub width: u32,
    pub height: u32,
    pub dtype: String,
    pub samples_per_pixel: u32,
    pub photometric_interpretation: String,
    #[napi(ts_type = "[number, number] | undefined")]
    pub pixel_spacing: Option<Vec<f64>>,
    pub rescale_slope: f64,
    pub rescale_intercept: f64,
    pub window_center: Option<f64>,
    pub window_width: Option<f64>,
}

#[napi]
pub struct PreparedDicom {
    viewer: ViewerDicom,
    frame_plan: NodeFramePlan,
}

#[napi]
impl PreparedDicom {
    #[napi(getter, ts_return_type = "FramePlan")]
    pub fn frame_plan(&self) -> NodeFramePlan {
        self.frame_plan.clone()
    }

    #[napi(ts_args_type = "frameIndex: number", ts_return_type = "RenderedFrame")]
    pub fn render_frame(
        &self,
        frame_index: Unknown<'_>,
    ) -> std::result::Result<NodeRenderedFrame, Error<String>> {
        render_raw_prepared_frame(self, parse_frame_index(frame_index)?)
    }

    #[napi(ts_args_type = "frameIndex: number", ts_return_type = "RenderedFrame")]
    pub fn render_display_frame(
        &self,
        frame_index: Unknown<'_>,
    ) -> std::result::Result<NodeRenderedFrame, Error<String>> {
        render_prepared_display_frame(self, parse_frame_index(frame_index)?)
    }
}

#[napi(
    ts_args_type = "input: DicomInput, options?: PrepareOptions",
    ts_return_type = "PreparedDicom"
)]
pub fn prepare_dicom(
    input: Unknown<'_>,
    options: Option<Unknown<'_>>,
) -> std::result::Result<PreparedDicom, Error<String>> {
    let viewer_options = parse_prepare_options(options)?;
    let viewer = parse_dicom_input(input, viewer_options)?;
    let frame_plan = node_frame_plan(viewer.frame_plan());
    Ok(PreparedDicom { viewer, frame_plan })
}

#[napi(
    ts_args_type = "prepared: PreparedDicom, frameIndex: number",
    ts_return_type = "RenderedFrame"
)]
pub fn render_frame(
    prepared: &PreparedDicom,
    frame_index: Unknown<'_>,
) -> std::result::Result<NodeRenderedFrame, Error<String>> {
    render_raw_prepared_frame(prepared, parse_frame_index(frame_index)?)
}

#[napi(
    ts_args_type = "prepared: PreparedDicom, frameIndex: number",
    ts_return_type = "RenderedFrame"
)]
pub fn render_display_frame(
    prepared: &PreparedDicom,
    frame_index: Unknown<'_>,
) -> std::result::Result<NodeRenderedFrame, Error<String>> {
    render_prepared_display_frame(prepared, parse_frame_index(frame_index)?)
}

fn display_frame_source(
    prepared: &PreparedDicom,
    frame_index: u32,
) -> std::result::Result<VolumeFrameSource, Error<String>> {
    prepared
        .viewer
        .frame_plan()
        .display_frames
        .get(frame_index as usize)
        .copied()
        .ok_or_else(|| {
            js_error(
                CODE_FRAME_INDEX_OUT_OF_RANGE,
                format!("display frame index {frame_index} is out of range"),
            )
        })
}

fn render_raw_prepared_frame(
    prepared: &PreparedDicom,
    frame_index: u32,
) -> std::result::Result<NodeRenderedFrame, Error<String>> {
    let source = display_frame_source(prepared, frame_index)?;
    let raw = prepared
        .viewer
        .decode_raw_display_frame(frame_index as usize)
        .map_err(map_dicom_render_error)?;
    let coordinate_transform = CoordinateTransform::identity(raw.width, raw.height)
        .map_err(map_coordinate_transform_error)?;

    node_rendered_raw_frame(prepared, frame_index, source, raw, coordinate_transform)
}

fn node_rendered_raw_frame(
    prepared: &PreparedDicom,
    frame_index: u32,
    source: VolumeFrameSource,
    raw: dicom_preprocessing::DecodedStoredFrame,
    coordinate_transform: CoordinateTransform,
) -> std::result::Result<NodeRenderedFrame, Error<String>> {
    let dtype = dtype_for_frame(raw.bits_allocated, raw.pixel_representation_signed)?;

    Ok(NodeRenderedFrame {
        display_frame_index: frame_index,
        source: node_frame_source(source),
        coordinate_transform: node_coordinate_transform(coordinate_transform),
        data: raw.data.into(),
        width: raw.width,
        height: raw.height,
        dtype: dtype.to_string(),
        samples_per_pixel: u32::from(raw.samples_per_pixel),
        photometric_interpretation: raw.photometric_interpretation,
        pixel_spacing: node_pixel_spacing(&prepared.viewer),
        rescale_slope: raw.rescale_slope,
        rescale_intercept: raw.rescale_intercept,
        window_center: raw.window_center,
        window_width: raw.window_width,
    })
}

fn render_prepared_display_frame(
    prepared: &PreparedDicom,
    frame_index: u32,
) -> std::result::Result<NodeRenderedFrame, Error<String>> {
    let source = display_frame_source(prepared, frame_index)?;
    match source {
        VolumeFrameSource::StoredFrame { .. } => {
            render_stored_display_frame(prepared, frame_index, source)
        }
        VolumeFrameSource::Derived => render_derived_display_frame(prepared, frame_index, source),
    }
}

fn render_stored_display_frame(
    prepared: &PreparedDicom,
    frame_index: u32,
    source: VolumeFrameSource,
) -> std::result::Result<NodeRenderedFrame, Error<String>> {
    let mut raw = prepared
        .viewer
        .decode_raw_display_frame(frame_index as usize)
        .map_err(map_dicom_render_error)?;
    let flip = prepared.viewer.display_flip(raw.width, raw.height);
    if let Some(flip) = flip {
        raw.data = flip_frame_data(
            raw.data,
            raw.width,
            raw.height,
            raw.samples_per_pixel,
            raw.planar_configuration,
            raw.bits_allocated,
            flip,
        )?;
    }
    let coordinate_transform =
        CoordinateTransform::from_flip(PixelDimensions::new(raw.width, raw.height), flip)
            .map_err(map_coordinate_transform_error)?;
    node_rendered_raw_frame(prepared, frame_index, source, raw, coordinate_transform)
}

fn render_derived_display_frame(
    prepared: &PreparedDicom,
    frame_index: u32,
    source: VolumeFrameSource,
) -> std::result::Result<NodeRenderedFrame, Error<String>> {
    let image = prepared
        .viewer
        .decode_display_frame_with_options(frame_index as usize, &ConvertOptions::default())
        .map_err(map_dicom_render_error)?;
    let width = image.width();
    let height = image.height();
    let coordinate_transform = CoordinateTransform::from_flip(
        PixelDimensions::new(width, height),
        prepared.viewer.display_flip(width, height),
    )
    .map_err(map_coordinate_transform_error)?;
    let color = image.color();
    let (data, dtype, samples_per_pixel, photometric_interpretation) =
        match (color.channel_count(), color.bits_per_pixel()) {
            (1, 8) => (
                image.into_luma8().into_raw(),
                "uint8",
                1_u32,
                "MONOCHROME2".to_string(),
            ),
            (1, 16) => {
                let data = image
                    .into_luma16()
                    .into_raw()
                    .into_iter()
                    .flat_map(u16::to_le_bytes)
                    .collect();
                (data, "uint16", 1_u32, "MONOCHROME2".to_string())
            }
            (3, 24) => (
                image.into_rgb8().into_raw(),
                "uint8",
                3_u32,
                "RGB".to_string(),
            ),
            _ => {
                return Err(js_error(
                    CODE_UNSUPPORTED_IMAGE_LAYOUT,
                    format!("unsupported derived display frame color type: {color:?}"),
                ));
            }
        };

    Ok(NodeRenderedFrame {
        display_frame_index: frame_index,
        source: node_frame_source(source),
        coordinate_transform: node_coordinate_transform(coordinate_transform),
        data: data.into(),
        width,
        height,
        dtype: dtype.to_string(),
        samples_per_pixel,
        photometric_interpretation,
        pixel_spacing: node_pixel_spacing(&prepared.viewer),
        rescale_slope: 1.0,
        rescale_intercept: 0.0,
        window_center: None,
        window_width: None,
    })
}

fn flip_frame_data(
    data: Vec<u8>,
    width: u32,
    height: u32,
    samples_per_pixel: u16,
    planar_configuration: PlanarConfiguration,
    bits_allocated: u16,
    flip: Flip,
) -> std::result::Result<Vec<u8>, Error<String>> {
    const BITS_PER_BYTE: u16 = 8;
    if !bits_allocated.is_multiple_of(BITS_PER_BYTE) {
        return Err(js_error(
            CODE_UNSUPPORTED_IMAGE_LAYOUT,
            format!("cannot flip pixels with BitsAllocated={bits_allocated}"),
        ));
    }
    let bytes_per_sample = usize::from(bits_allocated / BITS_PER_BYTE);
    let samples_per_pixel = usize::from(samples_per_pixel);
    let pixel_count = (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| js_error(CODE_UNSUPPORTED_IMAGE_LAYOUT, "frame size overflow"))?;
    let (plane_count, bytes_per_location) = match planar_configuration {
        PlanarConfiguration::Standard => (
            1,
            samples_per_pixel
                .checked_mul(bytes_per_sample)
                .ok_or_else(|| js_error(CODE_UNSUPPORTED_IMAGE_LAYOUT, "pixel stride overflow"))?,
        ),
        PlanarConfiguration::PixelFirst => (samples_per_pixel, bytes_per_sample),
    };
    let plane_len = pixel_count
        .checked_mul(bytes_per_location)
        .ok_or_else(|| js_error(CODE_UNSUPPORTED_IMAGE_LAYOUT, "plane size overflow"))?;
    let width = width as usize;
    let height = height as usize;
    let expected_len = plane_len
        .checked_mul(plane_count)
        .ok_or_else(|| js_error(CODE_UNSUPPORTED_IMAGE_LAYOUT, "frame size overflow"))?;
    if data.len() != expected_len {
        return Err(js_error(
            CODE_UNSUPPORTED_IMAGE_LAYOUT,
            format!(
                "decoded frame length {} does not match expected length {expected_len}",
                data.len()
            ),
        ));
    }

    let mut output = vec![0_u8; data.len()];
    for plane_index in 0..plane_count {
        let plane_offset = plane_index * plane_len;
        for output_y in 0..height {
            let input_y = if flip.vertical {
                height - 1 - output_y
            } else {
                output_y
            };
            for output_x in 0..width {
                let input_x = if flip.horizontal {
                    width - 1 - output_x
                } else {
                    output_x
                };
                let input_offset = plane_offset + (input_y * width + input_x) * bytes_per_location;
                let output_offset =
                    plane_offset + (output_y * width + output_x) * bytes_per_location;
                output[output_offset..output_offset + bytes_per_location]
                    .copy_from_slice(&data[input_offset..input_offset + bytes_per_location]);
            }
        }
    }
    Ok(output)
}

fn node_coordinate_transform(transform: CoordinateTransform) -> NodeCoordinateTransform {
    NodeCoordinateTransform {
        source_dimensions: node_pixel_dimensions(transform.source_dimensions),
        display_dimensions: node_pixel_dimensions(transform.display_dimensions),
        source_to_display: transform.source_to_display.to_vec(),
        display_to_source: transform.display_to_source.to_vec(),
        valid_source_rect: node_pixel_rect(transform.valid_source_rect),
        valid_display_rect: node_pixel_rect(transform.valid_display_rect),
    }
}

fn node_pixel_dimensions(dimensions: PixelDimensions) -> NodePixelDimensions {
    NodePixelDimensions {
        width: dimensions.width,
        height: dimensions.height,
    }
}

fn node_pixel_rect(rect: PixelRect) -> NodePixelRect {
    NodePixelRect {
        left: rect.left,
        top: rect.top,
        width: rect.width,
        height: rect.height,
    }
}

fn node_pixel_spacing(viewer: &ViewerDicom) -> Option<Vec<f64>> {
    pixel_spacing_mm(viewer.file())
        .map(|[row_mm, column_mm]| vec![f64::from(column_mm), f64::from(row_mm)])
}

fn parse_dicom_input(
    input: Unknown<'_>,
    options: ViewerOptions,
) -> std::result::Result<ViewerDicom, Error<String>> {
    let input = unknown_to_object(input, "input")?;
    let path = input
        .get::<String>("path")
        .map_err(map_napi_invalid_input)?;
    let bytes = input
        .get::<Uint8Array>("bytes")
        .map_err(map_napi_invalid_input)?;

    match (path, bytes) {
        (Some(path), None) => {
            ViewerDicom::open_with_options(path, options).map_err(map_dicom_open_error)
        }
        (None, Some(bytes)) => {
            let data = bytes.as_ref().to_vec();
            let file = from_reader(Cursor::new(data)).map_err(|error| {
                js_error(
                    CODE_READ_BYTES,
                    format!("error reading DICOM bytes: {error}"),
                )
            })?;
            ViewerDicom::from_object_with_options(file, options).map_err(map_dicom_open_error)
        }
        (Some(_), Some(_)) => Err(js_error(
            CODE_INVALID_INPUT,
            "DicomInput must provide exactly one of path or bytes",
        )),
        (None, None) => Err(js_error(
            CODE_INVALID_INPUT,
            "DicomInput must provide either path or bytes",
        )),
    }
}

fn parse_prepare_options(
    options: Option<Unknown<'_>>,
) -> std::result::Result<ViewerOptions, Error<String>> {
    let Some(options) = options else {
        return Ok(ViewerOptions::default());
    };
    let options = unknown_to_object(options, "options")?;
    let handler = options
        .get::<Unknown>("volumeHandler")
        .map_err(map_napi_invalid_input)?;
    let volume_handler = match handler {
        Some(handler) => parse_volume_handler(handler)?,
        None => VolumeHandler::keep(),
    };
    Ok(ViewerOptions::new(
        volume_handler,
        FlipOptions::new(
            optional_bool(&options, "flipHorizontal")?.unwrap_or(false),
            optional_bool(&options, "flipVertical")?.unwrap_or(false),
        ),
    ))
}

fn optional_bool(
    object: &Object<'_>,
    field: &str,
) -> std::result::Result<Option<bool>, Error<String>> {
    object.get::<bool>(field).map_err(map_napi_invalid_input)
}

fn parse_frame_index(frame_index: Unknown<'_>) -> std::result::Result<u32, Error<String>> {
    if frame_index.get_type().map_err(map_napi_invalid_input)? != ValueType::Number {
        return Err(js_error(CODE_INVALID_INPUT, "frameIndex must be a number"));
    }
    let value = frame_index
        .coerce_to_number()
        .and_then(|number| number.get_double())
        .map_err(map_napi_invalid_input)?;
    checked_frame_index(value)
}

fn checked_frame_index(value: f64) -> std::result::Result<u32, Error<String>> {
    if !value.is_finite() || value.fract() != 0.0 || value < 0.0 || value > f64::from(u32::MAX) {
        return Err(js_error(
            CODE_FRAME_INDEX_OUT_OF_RANGE,
            format!("frameIndex must be a non-negative integer <= {}", u32::MAX),
        ));
    }
    Ok(value as u32)
}

fn parse_volume_handler(handler: Unknown<'_>) -> std::result::Result<VolumeHandler, Error<String>> {
    match handler.get_type().map_err(map_napi_invalid_input)? {
        ValueType::String => {
            let name = handler
                .coerce_to_string()
                .and_then(|value| value.into_utf8())
                .and_then(|value| value.into_owned())
                .map_err(map_napi_invalid_input)?;
            volume_handler_from_options(name.as_str(), None, None, None)
        }
        ValueType::Object => {
            let object = handler.coerce_to_object().map_err(map_napi_invalid_input)?;
            let kind = object
                .get::<String>("kind")
                .map_err(map_napi_invalid_input)?
                .ok_or_else(|| js_error(CODE_INVALID_INPUT, "volume handler kind is required"))?;
            volume_handler_from_options(
                kind.as_str(),
                optional_u32(&object, "skipStart")?,
                optional_u32(&object, "skipEnd")?,
                optional_u32(&object, "targetFrames")?,
            )
        }
        _ => Err(js_error(
            CODE_INVALID_INPUT,
            "volumeHandler must be a string or object",
        )),
    }
}

fn volume_handler_from_options(
    kind: &str,
    skip_start: Option<u32>,
    skip_end: Option<u32>,
    target_frames: Option<u32>,
) -> std::result::Result<VolumeHandler, Error<String>> {
    match kind {
        "keep" => Ok(VolumeHandler::keep()),
        "central-slice" => Ok(VolumeHandler::central_slice()),
        "max-intensity" => Ok(VolumeHandler::max_intensity(
            skip_start.unwrap_or(0),
            skip_end.unwrap_or(0),
        )),
        "interpolate" => Ok(VolumeHandler::interpolate(target_frames.ok_or_else(
            || js_error(CODE_INVALID_INPUT, "interpolate targetFrames is required"),
        )?)),
        "laplacian-mip" => Ok(VolumeHandler::laplacian_mip(
            skip_start.unwrap_or(0),
            skip_end.unwrap_or(0),
        )),
        _ => Err(js_error(
            CODE_INVALID_INPUT,
            format!("unsupported volume handler: {kind}"),
        )),
    }
}

fn optional_u32(
    object: &Object<'_>,
    field: &str,
) -> std::result::Result<Option<u32>, Error<String>> {
    let Some(value) = object.get::<f64>(field).map_err(map_napi_invalid_input)? else {
        return Ok(None);
    };
    if !value.is_finite() || value.fract() != 0.0 || value < 0.0 || value > f64::from(u32::MAX) {
        return Err(js_error(
            CODE_INVALID_INPUT,
            format!("{field} must be a non-negative integer"),
        ));
    }
    Ok(Some(value as u32))
}

fn unknown_to_object<'env>(
    value: Unknown<'env>,
    name: &str,
) -> std::result::Result<Object<'env>, Error<String>> {
    if value.get_type().map_err(map_napi_invalid_input)? != ValueType::Object {
        return Err(js_error(
            CODE_INVALID_INPUT,
            format!("{name} must be an object"),
        ));
    }
    value.coerce_to_object().map_err(map_napi_invalid_input)
}

fn dtype_for_frame(
    bits_allocated: u16,
    signed: bool,
) -> std::result::Result<&'static str, Error<String>> {
    match (bits_allocated, signed) {
        (8, false) => Ok("uint8"),
        (8, true) => Ok("int8"),
        (16, false) => Ok("uint16"),
        (16, true) => Ok("int16"),
        _ => Err(js_error(
            CODE_UNSUPPORTED_IMAGE_LAYOUT,
            format!("unsupported BitsAllocated={bits_allocated}"),
        )),
    }
}

fn node_frame_plan(frame_plan: &dicom_preprocessing::VolumeFramePlan) -> NodeFramePlan {
    NodeFramePlan {
        display_frames: frame_plan
            .display_frames
            .iter()
            .copied()
            .map(node_frame_source)
            .collect(),
        stored_frame_order: frame_plan.stored_frame_order.clone(),
        frame_order_strategy: frame_order_strategy_name(frame_plan.frame_order_strategy)
            .to_string(),
        z_spacing_mm: frame_plan.z_spacing_mm.map(f64::from),
    }
}

fn node_frame_source(source: VolumeFrameSource) -> NodeFrameSource {
    match source {
        VolumeFrameSource::StoredFrame { stored_frame_index } => NodeFrameSource {
            kind: "stored".to_string(),
            stored_frame_index: Some(stored_frame_index),
        },
        VolumeFrameSource::Derived => NodeFrameSource {
            kind: "derived".to_string(),
            stored_frame_index: None,
        },
    }
}

fn frame_order_strategy_name(strategy: FrameOrderStrategy) -> &'static str {
    match strategy {
        FrameOrderStrategy::RawPreserved => "raw-preserved",
        FrameOrderStrategy::DimensionIndexValues => "dimension-index-values",
        FrameOrderStrategy::StackPosition => "stack-position",
        FrameOrderStrategy::Geometry => "geometry",
        FrameOrderStrategy::FrameIncrementPointer => "frame-increment-pointer",
        FrameOrderStrategy::SliceLocation => "slice-location",
    }
}

fn map_dicom_open_error(error: DicomError) -> Error<String> {
    match error {
        DicomError::ReadError { .. } => js_error(CODE_READ_FILE, error.to_string()),
        _ => map_dicom_render_error(error),
    }
}

fn map_dicom_render_error(error: DicomError) -> Error<String> {
    match error {
        DicomError::DisplayFrameIndexError { .. } => {
            js_error(CODE_FRAME_INDEX_OUT_OF_RANGE, error.to_string())
        }
        DicomError::DerivedFrameDecodeError { .. } => {
            js_error(CODE_DERIVED_FRAME_NO_RAW_SOURCE, error.to_string())
        }
        DicomError::UnsupportedPhotometricInterpretation { .. }
        | DicomError::UnsupportedMultiVolume { .. }
        | DicomError::MissingPropertyError { .. }
        | DicomError::InvalidValueError { .. } => {
            js_error(CODE_UNSUPPORTED_IMAGE_LAYOUT, error.to_string())
        }
        DicomError::PixelDataError { .. } => {
            let message = error.to_string();
            let lower = message.to_ascii_lowercase();
            if lower.contains("transfer syntax") || lower.contains("unsupported") {
                js_error(CODE_UNSUPPORTED_TRANSFER_SYNTAX, message)
            } else {
                js_error(CODE_PIXEL_DECODE, message)
            }
        }
        DicomError::ReadError { .. } => js_error(CODE_READ_FILE, error.to_string()),
        _ => js_error(CODE_PIXEL_DECODE, error.to_string()),
    }
}

fn map_napi_invalid_input(error: napi::Error) -> Error<String> {
    js_error(CODE_INVALID_INPUT, error.reason)
}

fn map_coordinate_transform_error(error: impl std::fmt::Display) -> Error<String> {
    js_error(CODE_UNSUPPORTED_IMAGE_LAYOUT, error.to_string())
}

fn js_error(code: impl Into<String>, message: impl Into<String>) -> Error<String> {
    Error::new(code.into(), message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use dicom::object::open_file;

    const MULTI_FRAME_FIXTURE: &str = "pydicom/emri_small.dcm";

    #[test]
    fn dtype_maps_supported_native_frames() {
        assert_eq!(dtype_for_frame(8, false).unwrap(), "uint8");
        assert_eq!(dtype_for_frame(8, true).unwrap(), "int8");
        assert_eq!(dtype_for_frame(16, false).unwrap(), "uint16");
        assert_eq!(dtype_for_frame(16, true).unwrap(), "int16");
        assert_eq!(
            dtype_for_frame(12, false).unwrap_err().status,
            CODE_UNSUPPORTED_IMAGE_LAYOUT
        );
    }

    #[test]
    fn frame_source_serializes_exact_and_derived_frames() {
        let stored = node_frame_source(VolumeFrameSource::StoredFrame {
            stored_frame_index: 7,
        });
        assert_eq!(stored.kind, "stored");
        assert_eq!(stored.stored_frame_index, Some(7));

        let derived = node_frame_source(VolumeFrameSource::Derived);
        assert_eq!(derived.kind, "derived");
        assert_eq!(derived.stored_frame_index, None);
    }

    #[test]
    fn frame_plan_serializes_ordering_metadata() {
        let dicom = open_file(dicom_test_files::path("pydicom/CT_small.dcm").unwrap()).unwrap();
        let plan = VolumeHandler::keep().frame_plan(&dicom).unwrap();
        let node_plan = node_frame_plan(&plan);

        assert_eq!(node_plan.display_frames.len(), 1);
        assert_eq!(node_plan.stored_frame_order, vec![0]);
        assert_eq!(node_plan.frame_order_strategy, "raw-preserved");
    }

    #[test]
    fn frame_data_flip_preserves_pixel_bytes_for_each_axis() {
        const WIDTH: u32 = 3;
        const HEIGHT: u32 = 2;
        let source = vec![1, 2, 3, 4, 5, 6];
        for (flip, expected) in [
            (
                Flip::new(WIDTH, HEIGHT, true, false),
                vec![3, 2, 1, 6, 5, 4],
            ),
            (
                Flip::new(WIDTH, HEIGHT, false, true),
                vec![4, 5, 6, 1, 2, 3],
            ),
            (Flip::new(WIDTH, HEIGHT, true, true), vec![6, 5, 4, 3, 2, 1]),
        ] {
            assert_eq!(
                flip_frame_data(
                    source.clone(),
                    WIDTH,
                    HEIGHT,
                    1,
                    PlanarConfiguration::Standard,
                    8,
                    flip,
                )
                .unwrap(),
                expected
            );
        }
    }

    #[test]
    fn frame_data_flip_preserves_planar_color_planes() {
        const WIDTH: u32 = 2;
        const HEIGHT: u32 = 2;
        const SAMPLES_PER_PIXEL: u16 = 3;
        const BITS_ALLOCATED: u16 = 8;
        let source = vec![
            1, 2, 3, 4, // red
            11, 12, 13, 14, // green
            21, 22, 23, 24, // blue
        ];
        for (flip, expected) in [
            (
                Flip::new(WIDTH, HEIGHT, true, false),
                vec![2, 1, 4, 3, 12, 11, 14, 13, 22, 21, 24, 23],
            ),
            (
                Flip::new(WIDTH, HEIGHT, false, true),
                vec![3, 4, 1, 2, 13, 14, 11, 12, 23, 24, 21, 22],
            ),
            (
                Flip::new(WIDTH, HEIGHT, true, true),
                vec![4, 3, 2, 1, 14, 13, 12, 11, 24, 23, 22, 21],
            ),
        ] {
            assert_eq!(
                flip_frame_data(
                    source.clone(),
                    WIDTH,
                    HEIGHT,
                    SAMPLES_PER_PIXEL,
                    PlanarConfiguration::PixelFirst,
                    BITS_ALLOCATED,
                    flip,
                )
                .unwrap(),
                expected
            );
        }
    }

    #[test]
    fn stored_display_frame_returns_applied_horizontal_transform() {
        let dicom = open_file(dicom_test_files::path("pydicom/CT_small.dcm").unwrap()).unwrap();
        let viewer = ViewerDicom::from_object_with_options(
            dicom,
            ViewerOptions::new(VolumeHandler::keep(), FlipOptions::new(true, false)),
        )
        .unwrap();
        let frame_plan = node_frame_plan(viewer.frame_plan());
        let prepared = PreparedDicom { viewer, frame_plan };
        let raw = render_raw_prepared_frame(&prepared, 0).unwrap();
        let source = display_frame_source(&prepared, 0).unwrap();
        let display = render_stored_display_frame(&prepared, 0, source).unwrap();
        let bytes_per_pixel = usize::try_from(raw.samples_per_pixel).unwrap()
            * if raw.dtype.ends_with('8') { 1 } else { 2 };
        let last_pixel_offset = (usize::try_from(raw.width).unwrap() - 1) * bytes_per_pixel;

        assert_eq!(
            display.coordinate_transform.source_to_display,
            vec![-1.0, 0.0, 0.0, 1.0, f64::from(raw.width - 1), 0.0]
        );
        assert_eq!(
            &display.data[..bytes_per_pixel],
            &raw.data[last_pixel_offset..last_pixel_offset + bytes_per_pixel]
        );
        assert_eq!(
            raw.coordinate_transform.source_to_display,
            vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        );
    }

    #[test]
    fn derived_display_frame_returns_applied_horizontal_transform() {
        let dicom = open_file(dicom_test_files::path(MULTI_FRAME_FIXTURE).unwrap()).unwrap();
        let viewer = ViewerDicom::from_object_with_options(
            dicom,
            ViewerOptions::new(
                VolumeHandler::laplacian_mip(0, 0),
                FlipOptions::new(true, false),
            ),
        )
        .unwrap();
        let frame_plan = node_frame_plan(viewer.frame_plan());
        let prepared = PreparedDicom { viewer, frame_plan };
        let source = display_frame_source(&prepared, 0).unwrap();

        let display = render_derived_display_frame(&prepared, 0, source).unwrap();

        assert_eq!(
            display.coordinate_transform.source_to_display,
            vec![-1.0, 0.0, 0.0, 1.0, f64::from(display.width - 1), 0.0]
        );
        assert_eq!(display.coordinate_transform.valid_source_rect.left, 0);
        assert_eq!(display.coordinate_transform.valid_display_rect.left, 0);
    }

    #[test]
    fn checked_frame_index_rejects_lossy_js_numbers() {
        assert_eq!(checked_frame_index(0.0).unwrap(), 0);
        assert_eq!(checked_frame_index(f64::from(u32::MAX)).unwrap(), u32::MAX);

        for value in [
            -1.0,
            0.5,
            f64::NAN,
            f64::INFINITY,
            f64::from(u32::MAX) + 1.0,
        ] {
            assert_eq!(
                checked_frame_index(value).unwrap_err().status,
                CODE_FRAME_INDEX_OUT_OF_RANGE
            );
        }
    }

    #[test]
    fn volume_handler_options_map_to_rust_handlers() {
        assert!(matches!(
            volume_handler_from_options("keep", None, None, None).unwrap(),
            VolumeHandler::Keep(_)
        ));
        assert!(matches!(
            volume_handler_from_options("central-slice", None, None, None).unwrap(),
            VolumeHandler::CentralSlice(_)
        ));
        assert!(matches!(
            volume_handler_from_options("max-intensity", Some(1), Some(2), None).unwrap(),
            VolumeHandler::MaxIntensity(_)
        ));
        assert!(matches!(
            volume_handler_from_options("laplacian-mip", Some(3), Some(4), None).unwrap(),
            VolumeHandler::LaplacianMip(_)
        ));

        let interpolate = volume_handler_from_options("interpolate", None, None, Some(8)).unwrap();
        assert_eq!(interpolate.get_target_frames(), Some(8));
        assert_eq!(
            volume_handler_from_options("interpolate", None, None, None)
                .unwrap_err()
                .status,
            CODE_INVALID_INPUT
        );
        assert_eq!(
            volume_handler_from_options("unknown", None, None, None)
                .unwrap_err()
                .status,
            CODE_INVALID_INPUT
        );
    }

    #[test]
    fn pixel_spacing_metadata_uses_x_y_order() {
        let viewer = ViewerDicom::open(
            dicom_test_files::path("pydicom/CT_small.dcm").unwrap(),
            VolumeHandler::keep(),
        )
        .unwrap();
        let [row_mm, column_mm] = pixel_spacing_mm(viewer.file()).unwrap();
        let spacing = pixel_spacing_mm(viewer.file())
            .map(|[row_mm, column_mm]| vec![f64::from(column_mm), f64::from(row_mm)])
            .unwrap();

        assert_eq!(spacing, vec![f64::from(column_mm), f64::from(row_mm)]);
    }

    #[test]
    fn dicom_error_mapping_uses_stable_codes() {
        assert_eq!(
            map_dicom_render_error(DicomError::DerivedFrameDecodeError {
                display_frame_index: 0,
            })
            .status,
            CODE_DERIVED_FRAME_NO_RAW_SOURCE
        );
        assert_eq!(
            map_dicom_render_error(DicomError::DisplayFrameIndexError {
                display_frame_index: 99,
                number_of_frames: 1,
            })
            .status,
            CODE_FRAME_INDEX_OUT_OF_RANGE
        );
    }

    #[test]
    fn ensure_node_fixtures() {
        dicom_test_files::path("pydicom/CT_small.dcm").unwrap();
        dicom_test_files::path("pydicom/emri_small.dcm").unwrap();
        dicom_test_files::path("pydicom/SC_rgb.dcm").unwrap();
    }
}
