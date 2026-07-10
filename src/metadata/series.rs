use crate::errors::DicomError;
use crate::metadata::{pixel_spacing_mm, FrameCount};
use dicom::dictionary_std::tags;
use dicom::object::{FileDicomObject, InMemDicomObject};
use std::cmp::Ordering;

const VALUE_TOLERANCE: f64 = 1.0e-4;
const SPACING_RELATIVE_TOLERANCE: f64 = 0.05;
const SPACING_ABSOLUTE_TOLERANCE_MM: f64 = 1.0e-3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeriesOrderStrategy {
    Geometry,
    InstanceNumber,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SeriesOrderPlan {
    /// Original input indexes in prepared slice order.
    pub source_indices: Vec<usize>,
    pub strategy: SeriesOrderStrategy,
    /// Center-to-center spacing from geometry, or Spacing Between Slices when
    /// geometry is unavailable. Slice Thickness is never used as spacing.
    pub z_spacing_mm: Option<f32>,
}

#[derive(Debug, Clone, PartialEq)]
struct Compatibility {
    series_instance_uid: String,
    frame_of_reference_uid: String,
    rows: u32,
    columns: u32,
    samples_per_pixel: u16,
    bits_allocated: u16,
    photometric_interpretation: String,
    pixel_spacing_mm: [f32; 2],
}

/// Validate and order a single-frame DICOM series.
///
/// Complete Image Position/Orientation geometry takes precedence. Monotonic input,
/// including descending input, retains its direction; mixed input is sorted by
/// projected position. If geometry is absent from every input, Instance Number is
/// used with the same monotonic-preservation rule. Partial geometry is rejected.
pub fn resolve_series_order(
    files: &[FileDicomObject<InMemDicomObject>],
) -> Result<SeriesOrderPlan, DicomError> {
    if files.is_empty() {
        return Err(invalid(0, "series is empty"));
    }

    let reference = compatibility(&files[0], 0)?;
    for (input_index, file) in files.iter().enumerate() {
        let frame_count: u32 = FrameCount::try_from(file)?.into();
        if frame_count != 1 {
            return Err(invalid(
                input_index,
                format!("expected one frame, found {frame_count}"),
            ));
        }
        let actual = compatibility(file, input_index)?;
        validate_compatibility(&reference, &actual, input_index)?;
    }

    let positions = files
        .iter()
        .map(|file| float_values::<3>(file, tags::IMAGE_POSITION_PATIENT))
        .collect::<Vec<_>>();
    let orientations = files
        .iter()
        .map(|file| float_values::<6>(file, tags::IMAGE_ORIENTATION_PATIENT))
        .collect::<Vec<_>>();
    let geometry_count = positions
        .iter()
        .zip(&orientations)
        .filter(|(position, orientation)| position.is_some() && orientation.is_some())
        .count();

    if geometry_count == files.len() {
        return geometry_order(&positions, &orientations);
    }
    if geometry_count != 0
        || positions.iter().any(Option::is_some)
        || orientations.iter().any(Option::is_some)
    {
        let input_index = positions
            .iter()
            .zip(&orientations)
            .position(|(position, orientation)| position.is_none() || orientation.is_none())
            .unwrap_or(0);
        return Err(invalid(
            input_index,
            "Image Position Patient and Image Orientation Patient must be present on every input",
        ));
    }

    instance_number_order(files)
}

fn compatibility(
    file: &FileDicomObject<InMemDicomObject>,
    input_index: usize,
) -> Result<Compatibility, DicomError> {
    Ok(Compatibility {
        series_instance_uid: string_value(file, tags::SERIES_INSTANCE_UID)
            .ok_or_else(|| invalid(input_index, "missing Series Instance UID"))?,
        frame_of_reference_uid: string_value(file, tags::FRAME_OF_REFERENCE_UID)
            .ok_or_else(|| invalid(input_index, "missing Frame of Reference UID"))?,
        rows: int_value(file, tags::ROWS)
            .ok_or_else(|| invalid(input_index, "missing or invalid Rows"))?,
        columns: int_value(file, tags::COLUMNS)
            .ok_or_else(|| invalid(input_index, "missing or invalid Columns"))?,
        samples_per_pixel: int_value(file, tags::SAMPLES_PER_PIXEL)
            .and_then(|value| u16::try_from(value).ok())
            .ok_or_else(|| invalid(input_index, "missing or invalid Samples Per Pixel"))?,
        bits_allocated: int_value(file, tags::BITS_ALLOCATED)
            .and_then(|value| u16::try_from(value).ok())
            .ok_or_else(|| invalid(input_index, "missing or invalid Bits Allocated"))?,
        photometric_interpretation: string_value(file, tags::PHOTOMETRIC_INTERPRETATION)
            .ok_or_else(|| invalid(input_index, "missing Photometric Interpretation"))?,
        pixel_spacing_mm: pixel_spacing_mm(file)
            .ok_or_else(|| invalid(input_index, "missing or invalid Pixel Spacing"))?,
    })
}

fn validate_compatibility(
    reference: &Compatibility,
    actual: &Compatibility,
    input_index: usize,
) -> Result<(), DicomError> {
    if actual.series_instance_uid != reference.series_instance_uid {
        return Err(invalid(
            input_index,
            "Series Instance UID differs from input 0",
        ));
    }
    if actual.frame_of_reference_uid != reference.frame_of_reference_uid {
        return Err(invalid(
            input_index,
            "Frame of Reference UID differs from input 0",
        ));
    }
    if actual.rows != reference.rows || actual.columns != reference.columns {
        return Err(invalid(input_index, "image dimensions differ from input 0"));
    }
    if actual.samples_per_pixel != reference.samples_per_pixel
        || actual.bits_allocated != reference.bits_allocated
        || actual.photometric_interpretation != reference.photometric_interpretation
    {
        return Err(invalid(input_index, "color type differs from input 0"));
    }
    if actual
        .pixel_spacing_mm
        .iter()
        .zip(reference.pixel_spacing_mm)
        .any(|(actual, expected)| (*actual - expected).abs() > VALUE_TOLERANCE as f32)
    {
        return Err(invalid(input_index, "Pixel Spacing differs from input 0"));
    }
    Ok(())
}

fn geometry_order(
    positions: &[Option<[f64; 3]>],
    orientations: &[Option<[f64; 6]>],
) -> Result<SeriesOrderPlan, DicomError> {
    let reference_orientation = orientations[0].unwrap();
    let reference_normal = normal(reference_orientation, 0)?;
    let mut projections = Vec::with_capacity(positions.len());
    for input_index in 0..positions.len() {
        let orientation = orientations[input_index].unwrap();
        if orientation
            .iter()
            .zip(reference_orientation)
            .any(|(actual, expected)| (*actual - expected).abs() > VALUE_TOLERANCE)
        {
            return Err(invalid(
                input_index,
                "Image Orientation Patient differs from input 0",
            ));
        }
        let projection = dot(positions[input_index].unwrap(), reference_normal);
        projections.push(projection);
    }

    let source_indices = order_indices(&projections, true)?;
    let ordered_positions = source_indices
        .iter()
        .map(|&index| projections[index])
        .collect::<Vec<_>>();
    let z_spacing_mm = validate_spacing(&ordered_positions, &source_indices)?;
    Ok(SeriesOrderPlan {
        source_indices,
        strategy: SeriesOrderStrategy::Geometry,
        z_spacing_mm,
    })
}

fn instance_number_order(
    files: &[FileDicomObject<InMemDicomObject>],
) -> Result<SeriesOrderPlan, DicomError> {
    let instance_numbers = files
        .iter()
        .enumerate()
        .map(|(input_index, file)| {
            file.get(tags::INSTANCE_NUMBER)
                .and_then(|element| element.to_int::<i32>().ok())
                .map(f64::from)
                .ok_or_else(|| {
                    invalid(
                        input_index,
                        "geometry is unavailable and Instance Number is missing or invalid",
                    )
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(SeriesOrderPlan {
        source_indices: order_indices(&instance_numbers, false)?,
        strategy: SeriesOrderStrategy::InstanceNumber,
        z_spacing_mm: spacing_between_slices(files)?,
    })
}

fn spacing_between_slices(
    files: &[FileDicomObject<InMemDicomObject>],
) -> Result<Option<f32>, DicomError> {
    let values = files
        .iter()
        .map(|file| {
            file.get(tags::SPACING_BETWEEN_SLICES)
                .and_then(|element| element.to_float64().ok())
        })
        .collect::<Vec<_>>();
    if values.iter().all(Option::is_none) {
        return Ok(None);
    }
    let reference = values[0].ok_or_else(|| {
        invalid(
            0,
            "Spacing Between Slices must be present on every input when geometry is unavailable",
        )
    })?;
    if reference <= 0.0 {
        return Err(invalid(0, "Spacing Between Slices must be positive"));
    }
    for (input_index, value) in values.iter().enumerate().skip(1) {
        let value = value.ok_or_else(|| {
            invalid(
                input_index,
                "Spacing Between Slices must be present on every input when geometry is unavailable",
            )
        })?;
        if (value - reference).abs() > VALUE_TOLERANCE {
            return Err(invalid(
                input_index,
                "Spacing Between Slices differs from input 0",
            ));
        }
    }
    Ok(Some(reference as f32))
}

fn order_indices(values: &[f64], geometry: bool) -> Result<Vec<usize>, DicomError> {
    for left in 0..values.len() {
        for right in left + 1..values.len() {
            if (values[left] - values[right]).abs() <= VALUE_TOLERANCE {
                let name = if geometry {
                    "projected Image Position Patient"
                } else {
                    "Instance Number"
                };
                return Err(invalid(right, format!("duplicate {name}")));
            }
        }
    }

    let increasing = values.windows(2).all(|pair| pair[1] > pair[0]);
    let decreasing = values.windows(2).all(|pair| pair[1] < pair[0]);
    if increasing || decreasing {
        return Ok((0..values.len()).collect());
    }

    let mut indices = (0..values.len()).collect::<Vec<_>>();
    indices.sort_by(|&left, &right| {
        values[left]
            .partial_cmp(&values[right])
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.cmp(&right))
    });
    Ok(indices)
}

fn validate_spacing(
    ordered_positions: &[f64],
    source_indices: &[usize],
) -> Result<Option<f32>, DicomError> {
    if ordered_positions.len() < 2 {
        return Ok(None);
    }
    let gaps = ordered_positions
        .windows(2)
        .map(|pair| (pair[1] - pair[0]).abs())
        .collect::<Vec<_>>();
    let mut sorted_gaps = gaps.clone();
    sorted_gaps.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    let spacing = sorted_gaps[sorted_gaps.len() / 2];
    let tolerance = (spacing * SPACING_RELATIVE_TOLERANCE).max(SPACING_ABSOLUTE_TOLERANCE_MM);
    if let Some(gap_index) = gaps
        .iter()
        .position(|gap| (*gap - spacing).abs() > tolerance)
    {
        return Err(invalid(
            source_indices[gap_index + 1],
            format!(
                "irregular slice spacing: gap {:.6} mm differs from median {:.6} mm",
                gaps[gap_index], spacing
            ),
        ));
    }
    Ok(Some(spacing as f32))
}

fn normal(orientation: [f64; 6], input_index: usize) -> Result<[f64; 3], DicomError> {
    let row = [orientation[0], orientation[1], orientation[2]];
    let column = [orientation[3], orientation[4], orientation[5]];
    let normal = [
        row[1] * column[2] - row[2] * column[1],
        row[2] * column[0] - row[0] * column[2],
        row[0] * column[1] - row[1] * column[0],
    ];
    let magnitude = dot(normal, normal).sqrt();
    if magnitude <= f64::EPSILON {
        return Err(invalid(
            input_index,
            "Image Orientation Patient has no valid normal",
        ));
    }
    Ok(normal.map(|value| value / magnitude))
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn float_values<const N: usize>(
    file: &FileDicomObject<InMemDicomObject>,
    tag: dicom::core::Tag,
) -> Option<[f64; N]> {
    let values = file.get(tag)?.to_multi_float64().ok()?;
    values.try_into().ok()
}

fn string_value(file: &FileDicomObject<InMemDicomObject>, tag: dicom::core::Tag) -> Option<String> {
    file.get(tag)
        .and_then(|element| element.to_str().ok())
        .map(|value| value.trim().to_string())
}

fn int_value(file: &FileDicomObject<InMemDicomObject>, tag: dicom::core::Tag) -> Option<u32> {
    file.get(tag)?.to_int::<u32>().ok()
}

fn invalid(input_index: usize, reason: impl Into<String>) -> DicomError {
    DicomError::InvalidSeriesInput {
        input_index,
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dicom::core::{DataElement, PrimitiveValue, VR};
    use dicom::object::open_file;

    const AXIAL: [f64; 6] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];

    fn series_file(
        position: Option<[f64; 3]>,
        orientation: Option<[f64; 6]>,
        instance_number: i32,
    ) -> FileDicomObject<InMemDicomObject> {
        let mut file = open_file(dicom_test_files::path("pydicom/CT_small.dcm").unwrap()).unwrap();
        file.put_element(DataElement::new(
            tags::SERIES_INSTANCE_UID,
            VR::UI,
            "1.2.826.0.1.3680043.10.100",
        ));
        file.put_element(DataElement::new(
            tags::FRAME_OF_REFERENCE_UID,
            VR::UI,
            "1.2.826.0.1.3680043.10.200",
        ));
        file.put_element(DataElement::new(
            tags::INSTANCE_NUMBER,
            VR::IS,
            instance_number.to_string(),
        ));
        file.remove_element(tags::SPACING_BETWEEN_SLICES);
        match position {
            Some(position) => {
                file.put_element(DataElement::new(
                    tags::IMAGE_POSITION_PATIENT,
                    VR::DS,
                    PrimitiveValue::from(position),
                ));
            }
            None => {
                file.remove_element(tags::IMAGE_POSITION_PATIENT);
            }
        }
        match orientation {
            Some(orientation) => {
                file.put_element(DataElement::new(
                    tags::IMAGE_ORIENTATION_PATIENT,
                    VR::DS,
                    PrimitiveValue::from(orientation),
                ));
            }
            None => {
                file.remove_element(tags::IMAGE_ORIENTATION_PATIENT);
            }
        }
        file
    }

    #[test]
    fn shuffled_axial_geometry_is_sorted_and_spacing_is_derived() {
        let files = vec![
            series_file(Some([0.0, 0.0, 2.0]), Some(AXIAL), 3),
            series_file(Some([0.0, 0.0, 0.0]), Some(AXIAL), 1),
            series_file(Some([0.0, 0.0, 1.0]), Some(AXIAL), 2),
        ];

        let plan = resolve_series_order(&files).unwrap();

        assert_eq!(plan.strategy, SeriesOrderStrategy::Geometry);
        assert_eq!(plan.source_indices, vec![1, 2, 0]);
        assert_eq!(plan.z_spacing_mm, Some(1.0));
    }

    #[test]
    fn shuffled_oblique_geometry_uses_orientation_normal() {
        let root_half = 0.5_f64.sqrt();
        let orientation = [1.0, 0.0, 0.0, 0.0, root_half, root_half];
        let position = |distance: f64| [0.0, -distance * root_half, distance * root_half];
        let files = vec![
            series_file(Some(position(4.0)), Some(orientation), 3),
            series_file(Some(position(0.0)), Some(orientation), 1),
            series_file(Some(position(2.0)), Some(orientation), 2),
        ];

        let plan = resolve_series_order(&files).unwrap();

        assert_eq!(plan.source_indices, vec![1, 2, 0]);
        assert_eq!(plan.z_spacing_mm, Some(2.0));
    }

    #[test]
    fn monotonic_descending_geometry_preserves_input_direction() {
        let files = vec![
            series_file(Some([0.0, 0.0, 2.0]), Some(AXIAL), 3),
            series_file(Some([0.0, 0.0, 1.0]), Some(AXIAL), 2),
            series_file(Some([0.0, 0.0, 0.0]), Some(AXIAL), 1),
        ];

        let plan = resolve_series_order(&files).unwrap();

        assert_eq!(plan.source_indices, vec![0, 1, 2]);
        assert_eq!(plan.z_spacing_mm, Some(1.0));
    }

    #[test]
    fn geometry_absence_falls_back_to_instance_number() {
        let files = vec![
            series_file(None, None, 3),
            series_file(None, None, 1),
            series_file(None, None, 2),
        ];

        let plan = resolve_series_order(&files).unwrap();

        assert_eq!(plan.strategy, SeriesOrderStrategy::InstanceNumber);
        assert_eq!(plan.source_indices, vec![1, 2, 0]);
        assert_eq!(plan.z_spacing_mm, None);
    }

    #[test]
    fn instance_fallback_uses_spacing_between_slices_but_not_slice_thickness() {
        let mut files = vec![series_file(None, None, 1), series_file(None, None, 2)];
        for file in &mut files {
            file.put_element(DataElement::new(tags::SLICE_THICKNESS, VR::DS, "9.0"));
        }
        assert_eq!(resolve_series_order(&files).unwrap().z_spacing_mm, None);

        for file in &mut files {
            file.put_element(DataElement::new(
                tags::SPACING_BETWEEN_SLICES,
                VR::DS,
                "2.5",
            ));
        }
        assert_eq!(
            resolve_series_order(&files).unwrap().z_spacing_mm,
            Some(2.5)
        );
    }

    #[test]
    fn mixed_series_is_rejected() {
        let mut files = vec![
            series_file(Some([0.0, 0.0, 0.0]), Some(AXIAL), 1),
            series_file(Some([0.0, 0.0, 1.0]), Some(AXIAL), 2),
        ];
        files[1].put_element(DataElement::new(
            tags::SERIES_INSTANCE_UID,
            VR::UI,
            "1.2.826.0.1.3680043.10.999",
        ));

        let error = resolve_series_order(&files).unwrap_err();

        assert!(matches!(
            error,
            DicomError::InvalidSeriesInput { input_index: 1, .. }
        ));
    }

    #[test]
    fn inconsistent_orientation_is_rejected() {
        let coronal = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let files = vec![
            series_file(Some([0.0, 0.0, 0.0]), Some(AXIAL), 1),
            series_file(Some([0.0, 1.0, 0.0]), Some(coronal), 2),
        ];

        let error = resolve_series_order(&files).unwrap_err();

        assert!(error.to_string().contains("Orientation"));
    }

    #[test]
    fn partial_geometry_is_rejected_instead_of_falling_back() {
        let files = vec![
            series_file(Some([0.0, 0.0, 0.0]), Some(AXIAL), 1),
            series_file(None, None, 2),
        ];

        let error = resolve_series_order(&files).unwrap_err();

        assert!(error.to_string().contains("must be present on every input"));
    }

    #[test]
    fn heterogeneous_dimensions_are_rejected() {
        let mut files = vec![
            series_file(Some([0.0, 0.0, 0.0]), Some(AXIAL), 1),
            series_file(Some([0.0, 0.0, 1.0]), Some(AXIAL), 2),
        ];
        files[1].put_element(DataElement::new(
            tags::ROWS,
            VR::US,
            PrimitiveValue::from(64_u16),
        ));

        let error = resolve_series_order(&files).unwrap_err();

        assert!(error.to_string().contains("dimensions differ"));
    }

    #[test]
    fn irregular_geometry_gaps_are_rejected() {
        let files = vec![
            series_file(Some([0.0, 0.0, 0.0]), Some(AXIAL), 1),
            series_file(Some([0.0, 0.0, 1.0]), Some(AXIAL), 2),
            series_file(Some([0.0, 0.0, 4.0]), Some(AXIAL), 3),
        ];

        let error = resolve_series_order(&files).unwrap_err();

        assert!(error.to_string().contains("irregular slice spacing"));
    }
}
