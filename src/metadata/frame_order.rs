use crate::errors::DicomError;
use dicom::core::header::HasLength;
use dicom::core::Tag;
use dicom::dictionary_std::tags;
use dicom::object::InMemDicomObject;
use std::cmp::Ordering;
use std::collections::BTreeSet;

const POSITION_EQUAL_TOLERANCE_MM: f64 = 1.0e-4;
const MIN_SPACING_TOLERANCE_MM: f64 = 1.0e-4;
const ORIENTATION_PARALLEL_DOT_TOLERANCE: f64 = 1.0e-4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameOrderStrategy {
    RawPreserved,
    DimensionIndexValues,
    StackPosition,
    Geometry,
    FrameIncrementPointer,
    SliceLocation,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FrameOrderPlan {
    pub ordered_frame_numbers: Vec<u32>,
    pub strategy: FrameOrderStrategy,
    pub z_spacing_mm: Option<f32>,
}

impl FrameOrderPlan {
    pub fn identity(number_of_frames: u32) -> Self {
        Self {
            ordered_frame_numbers: (0..number_of_frames).collect(),
            strategy: FrameOrderStrategy::RawPreserved,
            z_spacing_mm: None,
        }
    }

    pub fn is_identity(&self) -> bool {
        self.ordered_frame_numbers
            .iter()
            .enumerate()
            .all(|(index, frame_number)| *frame_number == index as u32)
    }
}

#[derive(Debug, Clone, Default)]
struct FrameMetadata {
    frame_number: u32,
    dimension_index_values: Vec<u32>,
    temporal_position_index: Option<u32>,
    stack_id: Option<String>,
    in_stack_position_number: Option<u32>,
    image_position_patient: Option<[f64; 3]>,
    image_orientation_patient: Option<[f64; 6]>,
    slice_location: Option<f64>,
    spacing_between_slices: Option<f64>,
    slice_thickness: Option<f64>,
    sampled: bool,
}

pub fn resolve_frame_order(file: &InMemDicomObject) -> Result<FrameOrderPlan, DicomError> {
    let number_of_frames = number_of_frames(file)?;
    if number_of_frames <= 1 {
        return Ok(FrameOrderPlan::identity(number_of_frames));
    }

    let dimension_organization_type = string_value(file, tags::DIMENSION_ORGANIZATION_TYPE);
    if matches!(dimension_organization_type.as_deref(), Some("3D_TEMPORAL")) {
        return Err(DicomError::UnsupportedMultiVolume {
            reason: "DimensionOrganizationType=3D_TEMPORAL".to_string(),
        });
    }

    let frames = collect_frame_metadata(file, number_of_frames);
    validate_single_volume_context(&frames)?;

    if let Some(plan) = resolve_dimension_index_order(file, &frames)? {
        return Ok(plan);
    }

    if let Some(plan) = resolve_stack_order(&frames)? {
        return Ok(plan);
    }

    if let Some(plan) = resolve_geometry_order(&frames)? {
        return Ok(plan);
    }

    if let Some(plan) = resolve_frame_increment_pointer_order(file, number_of_frames) {
        return Ok(plan);
    }

    if let Some(plan) = resolve_slice_location_order(&frames) {
        return Ok(plan);
    }

    let mut plan = FrameOrderPlan::identity(number_of_frames);
    plan.z_spacing_mm = pixel_measure_spacing(&frames);
    Ok(plan)
}

fn number_of_frames(file: &InMemDicomObject) -> Result<u32, DicomError> {
    match file.get(tags::NUMBER_OF_FRAMES) {
        Some(element) if !element.is_empty() => {
            element
                .to_int::<u32>()
                .map_err(|source| DicomError::ConvertValueError {
                    name: "Number of Frames",
                    source: Box::new(source),
                })
        }
        _ => Ok(1),
    }
}

fn collect_frame_metadata(file: &InMemDicomObject, number_of_frames: u32) -> Vec<FrameMetadata> {
    let shared_fg = sequence_item_in_object(file, tags::SHARED_FUNCTIONAL_GROUPS_SEQUENCE);
    let per_frame_groups = file
        .get(tags::PER_FRAME_FUNCTIONAL_GROUPS_SEQUENCE)
        .and_then(|element| element.items());

    (0..number_of_frames)
        .map(|frame_number| {
            let per_frame_fg = per_frame_groups.and_then(|items| items.get(frame_number as usize));
            let frame_content =
                sequence_item_in_scope(per_frame_fg, shared_fg, tags::FRAME_CONTENT_SEQUENCE);
            let plane_position =
                sequence_item_in_scope(per_frame_fg, shared_fg, tags::PLANE_POSITION_SEQUENCE);
            let plane_orientation =
                sequence_item_in_scope(per_frame_fg, shared_fg, tags::PLANE_ORIENTATION_SEQUENCE);
            let pixel_measures =
                sequence_item_in_scope(per_frame_fg, shared_fg, tags::PIXEL_MEASURES_SEQUENCE);

            FrameMetadata {
                frame_number,
                dimension_index_values: frame_content
                    .and_then(|item| multi_int_value(item, tags::DIMENSION_INDEX_VALUES))
                    .unwrap_or_default(),
                temporal_position_index: frame_content
                    .and_then(|item| int_value(item, tags::TEMPORAL_POSITION_INDEX)),
                stack_id: frame_content.and_then(|item| string_value(item, tags::STACK_ID)),
                in_stack_position_number: frame_content
                    .and_then(|item| int_value(item, tags::IN_STACK_POSITION_NUMBER)),
                image_position_patient: plane_position
                    .and_then(|item| float_triplet(item, tags::IMAGE_POSITION_PATIENT))
                    .or_else(|| float_triplet(file, tags::IMAGE_POSITION_PATIENT)),
                image_orientation_patient: plane_orientation
                    .and_then(|item| float_six(item, tags::IMAGE_ORIENTATION_PATIENT))
                    .or_else(|| float_six(file, tags::IMAGE_ORIENTATION_PATIENT)),
                slice_location: per_frame_fg
                    .and_then(|item| float_value(item, tags::SLICE_LOCATION))
                    .or_else(|| float_value(file, tags::SLICE_LOCATION)),
                spacing_between_slices: pixel_measures
                    .and_then(|item| float_value(item, tags::SPACING_BETWEEN_SLICES))
                    .or_else(|| float_value(file, tags::SPACING_BETWEEN_SLICES)),
                slice_thickness: pixel_measures
                    .and_then(|item| float_value(item, tags::SLICE_THICKNESS))
                    .or_else(|| float_value(file, tags::SLICE_THICKNESS)),
                sampled: is_sampled_frame(file, shared_fg, per_frame_fg),
            }
        })
        .collect()
}

fn validate_single_volume_context(frames: &[FrameMetadata]) -> Result<(), DicomError> {
    let temporal_positions: BTreeSet<u32> = frames
        .iter()
        .filter_map(|frame| frame.temporal_position_index)
        .collect();
    if temporal_positions.len() > 1 {
        return Err(DicomError::UnsupportedMultiVolume {
            reason: format!(
                "multiple TemporalPositionIndex values detected: {:?}",
                temporal_positions
            ),
        });
    }

    let stack_ids: BTreeSet<String> = frames
        .iter()
        .filter_map(|frame| frame.stack_id.clone())
        .collect();
    if stack_ids.len() > 1 {
        return Err(DicomError::UnsupportedMultiVolume {
            reason: format!("multiple StackID values detected: {:?}", stack_ids),
        });
    }

    Ok(())
}

fn resolve_dimension_index_order(
    file: &InMemDicomObject,
    frames: &[FrameMetadata],
) -> Result<Option<FrameOrderPlan>, DicomError> {
    let descriptors = dimension_index_pointers(file);
    if descriptors.is_empty() {
        return Ok(None);
    }

    if frames
        .iter()
        .any(|frame| frame.dimension_index_values.len() != descriptors.len())
    {
        return Ok(None);
    }

    for (index, pointer) in descriptors.iter().enumerate() {
        let distinct_values: BTreeSet<u32> = frames
            .iter()
            .filter_map(|frame| frame.dimension_index_values.get(index).copied())
            .collect();
        if distinct_values.len() <= 1 {
            continue;
        }
        match *pointer {
            tags::IN_STACK_POSITION_NUMBER
            | tags::IMAGE_POSITION_PATIENT
            | tags::SLICE_LOCATION => {}
            tags::TEMPORAL_POSITION_INDEX => {
                return Err(DicomError::UnsupportedMultiVolume {
                    reason: "DimensionIndexSequence varies TemporalPositionIndex".to_string(),
                });
            }
            tags::STACK_ID => {
                return Err(DicomError::UnsupportedMultiVolume {
                    reason: "DimensionIndexSequence varies StackID".to_string(),
                });
            }
            other => {
                return Err(DicomError::UnsupportedMultiVolume {
                    reason: format!(
                        "DimensionIndexSequence varies unsupported pointer ({:04X},{:04X})",
                        other.0, other.1
                    ),
                });
            }
        }
    }

    let order = sorted_frame_numbers_by(
        frames,
        |left, right| {
            left.dimension_index_values
                .cmp(&right.dimension_index_values)
                .then_with(|| left.frame_number.cmp(&right.frame_number))
        },
        true,
    );

    Ok(Some(order_plan_with_metadata_spacing(
        FrameOrderStrategy::DimensionIndexValues,
        order,
        frames,
    )))
}

fn resolve_stack_order(frames: &[FrameMetadata]) -> Result<Option<FrameOrderPlan>, DicomError> {
    let all_have_in_stack = frames
        .iter()
        .all(|frame| frame.in_stack_position_number.is_some());
    if !all_have_in_stack {
        return Ok(None);
    }

    let unique_positions: BTreeSet<u32> = frames
        .iter()
        .filter_map(|frame| frame.in_stack_position_number)
        .collect();
    if unique_positions.len() != frames.len() {
        return Ok(None);
    }

    let order = sorted_frame_numbers_by(
        frames,
        |left, right| {
            left.in_stack_position_number
                .cmp(&right.in_stack_position_number)
                .then_with(|| left.frame_number.cmp(&right.frame_number))
        },
        false,
    );

    Ok(Some(order_plan_with_metadata_spacing(
        FrameOrderStrategy::StackPosition,
        order,
        frames,
    )))
}

fn resolve_geometry_order(frames: &[FrameMetadata]) -> Result<Option<FrameOrderPlan>, DicomError> {
    if frames.iter().any(|frame| frame.sampled) {
        return Ok(None);
    }

    if frames.iter().any(|frame| {
        frame.image_position_patient.is_none() || frame.image_orientation_patient.is_none()
    }) {
        return Ok(None);
    }

    let reference_orientation = frames[0].image_orientation_patient.unwrap();
    let reference_normal = normal_from_orientation(reference_orientation)?;

    let mut projections = Vec::with_capacity(frames.len());
    for frame in frames {
        let normal = normal_from_orientation(frame.image_orientation_patient.unwrap())?;
        let dot = dot_product(reference_normal, normal).abs();
        if (1.0 - dot) > ORIENTATION_PARALLEL_DOT_TOLERANCE {
            return Ok(None);
        }

        let projection = dot_product(frame.image_position_patient.unwrap(), reference_normal);
        projections.push((frame.frame_number, projection));
    }

    let values: Vec<f64> = projections
        .iter()
        .map(|(_, projection)| *projection)
        .collect();
    if !has_unique_values(&values) {
        return Ok(None);
    }

    let direction = monotonic_direction(&values);
    let ordered_frame_numbers: Vec<u32> = match direction {
        MonotonicDirection::Increasing | MonotonicDirection::Decreasing => {
            frames.iter().map(|frame| frame.frame_number).collect()
        }
        MonotonicDirection::Mixed => {
            let mut sorted = projections;
            sorted.sort_by(|left, right| {
                left.1
                    .partial_cmp(&right.1)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| left.0.cmp(&right.0))
            });
            sorted
                .into_iter()
                .map(|(frame_number, _)| frame_number)
                .collect()
        }
    };

    Ok(Some(order_plan_with_metadata_spacing(
        FrameOrderStrategy::Geometry,
        ordered_frame_numbers,
        frames,
    )))
}

fn resolve_frame_increment_pointer_order(
    file: &InMemDicomObject,
    number_of_frames: u32,
) -> Option<FrameOrderPlan> {
    let pointer = file
        .get(tags::FRAME_INCREMENT_POINTER)
        .and_then(|element| element.value().to_tag().ok())?;
    let values = numeric_vector_value(file, pointer)?;
    if values.len() != number_of_frames as usize || !has_unique_values(&values) {
        return None;
    }

    let direction = monotonic_direction(&values);
    let ordered_frame_numbers = match direction {
        MonotonicDirection::Increasing | MonotonicDirection::Decreasing => {
            (0..number_of_frames).collect()
        }
        MonotonicDirection::Mixed => sorted_frame_numbers_for_values(&values),
    };

    Some(FrameOrderPlan {
        z_spacing_mm: spacing_from_ordered_frame_numbers(&values, &ordered_frame_numbers),
        ordered_frame_numbers,
        strategy: FrameOrderStrategy::FrameIncrementPointer,
    })
}

fn resolve_slice_location_order(frames: &[FrameMetadata]) -> Option<FrameOrderPlan> {
    if frames.iter().any(|frame| frame.slice_location.is_none()) {
        return None;
    }

    let slice_locations: Vec<f64> = frames
        .iter()
        .map(|frame| frame.slice_location.unwrap())
        .collect();
    if !has_unique_values(&slice_locations) {
        return None;
    }

    let direction = monotonic_direction(&slice_locations);
    let ordered_frame_numbers = match direction {
        MonotonicDirection::Increasing | MonotonicDirection::Decreasing => {
            frames.iter().map(|frame| frame.frame_number).collect()
        }
        MonotonicDirection::Mixed => sorted_frame_numbers_for_values(&slice_locations),
    };

    Some(FrameOrderPlan {
        z_spacing_mm: spacing_from_ordered_frame_numbers(&slice_locations, &ordered_frame_numbers),
        ordered_frame_numbers,
        strategy: FrameOrderStrategy::SliceLocation,
    })
}

fn order_plan_with_metadata_spacing(
    strategy: FrameOrderStrategy,
    ordered_frame_numbers: Vec<u32>,
    frames: &[FrameMetadata],
) -> FrameOrderPlan {
    FrameOrderPlan {
        z_spacing_mm: derived_spacing_mm(frames, &ordered_frame_numbers),
        ordered_frame_numbers,
        strategy,
    }
}

fn dimension_index_pointers(file: &InMemDicomObject) -> Vec<Tag> {
    file.get(tags::DIMENSION_INDEX_SEQUENCE)
        .and_then(|element| element.items())
        .map(|items| {
            items
                .iter()
                .filter_map(|item| {
                    item.get(tags::DIMENSION_INDEX_POINTER)
                        .and_then(|element| element.value().to_tag().ok())
                })
                .collect()
        })
        .unwrap_or_default()
}

fn spacing_from_geometry(frames: &[FrameMetadata], order: &[u32]) -> Option<f32> {
    if frames.iter().any(|frame| frame.sampled) {
        return None;
    }

    let first_frame = frames.iter().find(|frame| frame.frame_number == order[0])?;
    let normal = normal_from_orientation(first_frame.image_orientation_patient?).ok()?;
    let mut values = Vec::with_capacity(order.len());
    for frame_number in order {
        let frame = frames
            .iter()
            .find(|frame| frame.frame_number == *frame_number)?;
        values.push(dot_product(frame.image_position_patient?, normal));
    }
    spacing_from_values(&values)
}

fn derived_spacing_mm(frames: &[FrameMetadata], order: &[u32]) -> Option<f32> {
    spacing_from_geometry(frames, order).or_else(|| pixel_measure_spacing(frames))
}

fn spacing_from_ordered_frame_numbers(
    values: &[f64],
    ordered_frame_numbers: &[u32],
) -> Option<f32> {
    let ordered_values: Vec<f64> = ordered_frame_numbers
        .iter()
        .map(|frame_number| values.get(*frame_number as usize).copied())
        .collect::<Option<_>>()?;
    spacing_from_values(&ordered_values)
}

fn pixel_measure_spacing(frames: &[FrameMetadata]) -> Option<f32> {
    frames
        .iter()
        .find_map(|frame| frame.spacing_between_slices.or(frame.slice_thickness))
        .map(|value| value as f32)
}

fn sorted_frame_numbers_by(
    frames: &[FrameMetadata],
    compare: impl Fn(&FrameMetadata, &FrameMetadata) -> Ordering,
    preserve_if_monotonic: bool,
) -> Vec<u32> {
    let mut indexed_frames: Vec<&FrameMetadata> = frames.iter().collect();
    indexed_frames.sort_by(|left, right| compare(left, right));

    if preserve_if_monotonic {
        let monotonic = indexed_frames
            .iter()
            .enumerate()
            .all(|(index, frame)| frame.frame_number == index as u32);
        if monotonic {
            return frames.iter().map(|frame| frame.frame_number).collect();
        }
    }

    indexed_frames
        .into_iter()
        .map(|frame| frame.frame_number)
        .collect()
}

fn sorted_frame_numbers_for_values(values: &[f64]) -> Vec<u32> {
    let mut indexed: Vec<(u32, f64)> = values
        .iter()
        .enumerate()
        .map(|(index, value)| (index as u32, *value))
        .collect();
    indexed.sort_by(|left, right| {
        left.1
            .partial_cmp(&right.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.0.cmp(&right.0))
    });
    indexed
        .into_iter()
        .map(|(frame_number, _)| frame_number)
        .collect()
}

fn sequence_item_in_scope<'a>(
    primary: Option<&'a InMemDicomObject>,
    fallback: Option<&'a InMemDicomObject>,
    tag: Tag,
) -> Option<&'a InMemDicomObject> {
    primary
        .and_then(|object| sequence_item_in_object(object, tag))
        .or_else(|| fallback.and_then(|object| sequence_item_in_object(object, tag)))
}

fn sequence_item_in_object(object: &InMemDicomObject, tag: Tag) -> Option<&InMemDicomObject> {
    object.get(tag)?.items()?.first()
}

fn is_sampled_frame(
    file: &InMemDicomObject,
    shared_fg: Option<&InMemDicomObject>,
    per_frame_fg: Option<&InMemDicomObject>,
) -> bool {
    per_frame_fg
        .and_then(volumetric_properties_from_object)
        .or_else(|| shared_fg.and_then(volumetric_properties_from_object))
        .or_else(|| volumetric_properties_from_object(file))
        .is_some_and(|value| value == "SAMPLED")
}

fn volumetric_properties_from_object(object: &InMemDicomObject) -> Option<String> {
    string_value(object, tags::VOLUMETRIC_PROPERTIES).or_else(|| {
        object.iter().find_map(|element| {
            element
                .items()
                .and_then(|items| items.first())
                .and_then(|item| string_value(item, tags::VOLUMETRIC_PROPERTIES))
        })
    })
}

fn int_value(object: &InMemDicomObject, tag: Tag) -> Option<u32> {
    object
        .get(tag)
        .and_then(|element| element.to_int::<u32>().ok())
}

fn multi_int_value(object: &InMemDicomObject, tag: Tag) -> Option<Vec<u32>> {
    object
        .get(tag)
        .and_then(|element| element.to_multi_int::<u32>().ok())
}

fn float_value(object: &InMemDicomObject, tag: Tag) -> Option<f64> {
    object
        .get(tag)
        .and_then(|element| element.to_float64().ok())
}

fn float_triplet(object: &InMemDicomObject, tag: Tag) -> Option<[f64; 3]> {
    let values = object
        .get(tag)
        .and_then(|element| element.to_multi_float64().ok())?;
    Some([*values.first()?, *values.get(1)?, *values.get(2)?])
}

fn float_six(object: &InMemDicomObject, tag: Tag) -> Option<[f64; 6]> {
    let values = object
        .get(tag)
        .and_then(|element| element.to_multi_float64().ok())?;
    Some([
        *values.first()?,
        *values.get(1)?,
        *values.get(2)?,
        *values.get(3)?,
        *values.get(4)?,
        *values.get(5)?,
    ])
}

fn string_value(object: &InMemDicomObject, tag: Tag) -> Option<String> {
    object
        .get(tag)
        .and_then(|element| element.to_str().ok())
        .map(|value| value.into_owned())
}

fn numeric_vector_value(object: &InMemDicomObject, tag: Tag) -> Option<Vec<f64>> {
    object
        .get(tag)
        .and_then(|element| element.to_multi_float64().ok())
        .or_else(|| {
            object.get(tag).and_then(|element| {
                element
                    .to_multi_int::<i32>()
                    .ok()
                    .map(|values| values.into_iter().map(|value| value as f64).collect())
            })
        })
}

fn normal_from_orientation(orientation: [f64; 6]) -> Result<[f64; 3], DicomError> {
    let row = [orientation[0], orientation[1], orientation[2]];
    let column = [orientation[3], orientation[4], orientation[5]];
    let normal = [
        row[1] * column[2] - row[2] * column[1],
        row[2] * column[0] - row[0] * column[2],
        row[0] * column[1] - row[1] * column[0],
    ];
    let magnitude = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
    if magnitude <= f64::EPSILON {
        return Err(DicomError::InvalidValueError {
            name: "Image Orientation Patient",
            value: format!("{orientation:?}"),
        });
    }
    Ok([
        normal[0] / magnitude,
        normal[1] / magnitude,
        normal[2] / magnitude,
    ])
}

fn dot_product(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn has_unique_values(values: &[f64]) -> bool {
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    sorted
        .windows(2)
        .all(|window| (window[1] - window[0]).abs() > POSITION_EQUAL_TOLERANCE_MM)
}

fn spacing_from_values(values: &[f64]) -> Option<f32> {
    if values.len() < 2 {
        return None;
    }

    let mut diffs: Vec<f64> = values
        .windows(2)
        .map(|window| (window[1] - window[0]).abs())
        .filter(|value| *value > MIN_SPACING_TOLERANCE_MM)
        .collect();
    if diffs.is_empty() {
        return None;
    }
    diffs.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    Some(diffs[diffs.len() / 2] as f32)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MonotonicDirection {
    Increasing,
    Decreasing,
    Mixed,
}

fn monotonic_direction(values: &[f64]) -> MonotonicDirection {
    let mut has_positive = false;
    let mut has_negative = false;
    for window in values.windows(2) {
        let delta = window[1] - window[0];
        if delta > POSITION_EQUAL_TOLERANCE_MM {
            has_positive = true;
        } else if delta < -POSITION_EQUAL_TOLERANCE_MM {
            has_negative = true;
        }
    }

    match (has_positive, has_negative) {
        (true, false) => MonotonicDirection::Increasing,
        (false, true) => MonotonicDirection::Decreasing,
        _ => MonotonicDirection::Mixed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dicom::core::value::DataSetSequence;
    use dicom::core::{dicom_value, PrimitiveValue, VR};
    use dicom::object::mem::InMemElement;

    const TEST_AXIAL_ORIENTATION: [f64; 6] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];

    fn seq(tag: Tag, items: Vec<InMemDicomObject>) -> InMemElement {
        InMemElement::new(tag, VR::SQ, DataSetSequence::from(items))
    }

    fn frame_content(
        in_stack_position_number: Option<u32>,
        temporal_position_index: Option<u32>,
        dimension_index_values: Option<Vec<u32>>,
        stack_id: Option<&str>,
    ) -> InMemDicomObject {
        let mut elements = Vec::new();
        if let Some(value) = temporal_position_index {
            elements.push(InMemElement::new(
                tags::TEMPORAL_POSITION_INDEX,
                VR::UL,
                PrimitiveValue::from(value),
            ));
        }
        if let Some(value) = in_stack_position_number {
            elements.push(InMemElement::new(
                tags::IN_STACK_POSITION_NUMBER,
                VR::UL,
                PrimitiveValue::from(value),
            ));
        }
        if let Some(values) = dimension_index_values {
            elements.push(InMemElement::new(
                tags::DIMENSION_INDEX_VALUES,
                VR::UL,
                PrimitiveValue::U32(values.into()),
            ));
        }
        if let Some(value) = stack_id {
            elements.push(InMemElement::new(tags::STACK_ID, VR::SH, value));
        }
        InMemDicomObject::from_element_iter(elements)
    }

    fn plane_position(position: [f64; 3]) -> InMemDicomObject {
        InMemDicomObject::from_element_iter([InMemElement::new(
            tags::IMAGE_POSITION_PATIENT,
            VR::DS,
            PrimitiveValue::from(position),
        )])
    }

    fn plane_orientation(orientation: [f64; 6]) -> InMemDicomObject {
        InMemDicomObject::from_element_iter([InMemElement::new(
            tags::IMAGE_ORIENTATION_PATIENT,
            VR::DS,
            PrimitiveValue::from(orientation),
        )])
    }

    fn pixel_measures(spacing_between_slices: f64) -> InMemDicomObject {
        InMemDicomObject::from_element_iter([InMemElement::new(
            tags::SPACING_BETWEEN_SLICES,
            VR::DS,
            spacing_between_slices.to_string(),
        )])
    }

    fn frame_group(
        in_stack_position_number: Option<u32>,
        position: Option<[f64; 3]>,
        orientation: Option<[f64; 6]>,
        sampled: bool,
        temporal_position_index: Option<u32>,
        dimension_index_values: Option<Vec<u32>>,
        stack_id: Option<&str>,
    ) -> InMemDicomObject {
        let mut elements = vec![seq(
            tags::FRAME_CONTENT_SEQUENCE,
            vec![frame_content(
                in_stack_position_number,
                temporal_position_index,
                dimension_index_values,
                stack_id,
            )],
        )];
        if let Some(position) = position {
            elements.push(seq(
                tags::PLANE_POSITION_SEQUENCE,
                vec![plane_position(position)],
            ));
        }
        if let Some(orientation) = orientation {
            elements.push(seq(
                tags::PLANE_ORIENTATION_SEQUENCE,
                vec![plane_orientation(orientation)],
            ));
        }
        elements.push(seq(
            tags::PIXEL_MEASURES_SEQUENCE,
            vec![pixel_measures(1.5)],
        ));
        if sampled {
            let xray_3d = InMemDicomObject::from_element_iter([InMemElement::new(
                tags::VOLUMETRIC_PROPERTIES,
                VR::CS,
                "SAMPLED",
            )]);
            elements.push(seq(tags::X_RAY3_D_FRAME_TYPE_SEQUENCE, vec![xray_3d]));
        }
        InMemDicomObject::from_element_iter(elements)
    }

    fn test_object(per_frame_groups: Vec<InMemDicomObject>) -> InMemDicomObject {
        let frame_count = per_frame_groups.len() as u32;
        InMemDicomObject::from_element_iter([
            InMemElement::new(tags::NUMBER_OF_FRAMES, VR::IS, frame_count.to_string()),
            seq(tags::PER_FRAME_FUNCTIONAL_GROUPS_SEQUENCE, per_frame_groups),
        ])
    }

    fn geometry_test_object(positions: &[[f64; 3]], sampled: bool) -> InMemDicomObject {
        test_object(
            positions
                .iter()
                .copied()
                .map(|position| {
                    frame_group(
                        None,
                        Some(position),
                        Some(TEST_AXIAL_ORIENTATION),
                        sampled,
                        None,
                        None,
                        None,
                    )
                })
                .collect(),
        )
    }

    fn frame_increment_pointer_test_object(values: [f64; 3]) -> InMemDicomObject {
        InMemDicomObject::from_element_iter([
            InMemElement::new(tags::NUMBER_OF_FRAMES, VR::IS, "3"),
            InMemElement::new(
                tags::FRAME_INCREMENT_POINTER,
                VR::AT,
                dicom_value!(Tags, [tags::FRAME_TIME_VECTOR]),
            ),
            InMemElement::new(
                tags::FRAME_TIME_VECTOR,
                VR::DS,
                PrimitiveValue::from(values),
            ),
        ])
    }

    fn slice_location_group(slice_location: f64) -> InMemDicomObject {
        InMemDicomObject::from_element_iter([
            InMemElement::new(tags::SLICE_LOCATION, VR::DS, slice_location.to_string()),
            seq(tags::PIXEL_MEASURES_SEQUENCE, vec![pixel_measures(1.5)]),
        ])
    }

    #[test]
    fn resolves_dimension_index_values_order() {
        let object = InMemDicomObject::from_element_iter([
            InMemElement::new(tags::NUMBER_OF_FRAMES, VR::IS, "3"),
            seq(
                tags::DIMENSION_INDEX_SEQUENCE,
                vec![InMemDicomObject::from_element_iter([InMemElement::new(
                    tags::DIMENSION_INDEX_POINTER,
                    VR::AT,
                    dicom_value!(Tags, [tags::IN_STACK_POSITION_NUMBER]),
                )])],
            ),
            seq(
                tags::PER_FRAME_FUNCTIONAL_GROUPS_SEQUENCE,
                vec![
                    frame_group(Some(3), None, None, false, None, Some(vec![3]), None),
                    frame_group(Some(1), None, None, false, None, Some(vec![1]), None),
                    frame_group(Some(2), None, None, false, None, Some(vec![2]), None),
                ],
            ),
        ]);

        let plan = resolve_frame_order(&object).unwrap();
        assert_eq!(plan.strategy, FrameOrderStrategy::DimensionIndexValues);
        assert_eq!(plan.ordered_frame_numbers, vec![1, 2, 0]);
    }

    #[test]
    fn resolves_stack_position_order() {
        let object = test_object(vec![
            frame_group(Some(3), None, None, false, None, None, Some("A")),
            frame_group(Some(1), None, None, false, None, None, Some("A")),
            frame_group(Some(2), None, None, false, None, None, Some("A")),
        ]);

        let plan = resolve_frame_order(&object).unwrap();
        assert_eq!(plan.strategy, FrameOrderStrategy::StackPosition);
        assert_eq!(plan.ordered_frame_numbers, vec![1, 2, 0]);
        assert_eq!(plan.z_spacing_mm, Some(1.5));
    }

    #[test]
    fn geometry_order_sorts_mixed_positions_and_extracts_spacing() {
        let object =
            geometry_test_object(&[[0.0, 0.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], false);

        let plan = resolve_frame_order(&object).unwrap();
        assert_eq!(plan.strategy, FrameOrderStrategy::Geometry);
        assert_eq!(plan.ordered_frame_numbers, vec![1, 2, 0]);
        assert_eq!(plan.z_spacing_mm, Some(1.0));
    }

    #[test]
    fn geometry_order_preserves_monotonic_descending_direction() {
        let object =
            geometry_test_object(&[[0.0, 0.0, 2.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]], false);

        let plan = resolve_frame_order(&object).unwrap();
        assert_eq!(plan.strategy, FrameOrderStrategy::Geometry);
        assert_eq!(plan.ordered_frame_numbers, vec![0, 1, 2]);
    }

    #[test]
    fn sampled_frames_do_not_use_geometry_order() {
        let object =
            geometry_test_object(&[[0.0, 0.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], true);

        let plan = resolve_frame_order(&object).unwrap();
        assert_eq!(plan.strategy, FrameOrderStrategy::RawPreserved);
        assert_eq!(plan.ordered_frame_numbers, vec![0, 1, 2]);
    }

    #[test]
    fn frame_increment_pointer_spacing_uses_sorted_order() {
        let object = frame_increment_pointer_test_object([10.0, 0.0, 5.0]);

        let plan = resolve_frame_order(&object).unwrap();
        assert_eq!(plan.strategy, FrameOrderStrategy::FrameIncrementPointer);
        assert_eq!(plan.ordered_frame_numbers, vec![1, 2, 0]);
        assert_eq!(plan.z_spacing_mm, Some(5.0));
    }

    #[test]
    fn slice_location_spacing_uses_sorted_order() {
        let object = test_object(vec![
            slice_location_group(10.0),
            slice_location_group(0.0),
            slice_location_group(5.0),
        ]);

        let plan = resolve_frame_order(&object).unwrap();
        assert_eq!(plan.strategy, FrameOrderStrategy::SliceLocation);
        assert_eq!(plan.ordered_frame_numbers, vec![1, 2, 0]);
        assert_eq!(plan.z_spacing_mm, Some(5.0));
    }

    #[test]
    fn multiple_temporal_positions_are_rejected() {
        let object = test_object(vec![
            frame_group(Some(1), None, None, false, Some(1), None, None),
            frame_group(Some(2), None, None, false, Some(2), None, None),
        ]);

        let error = resolve_frame_order(&object).unwrap_err();
        assert!(matches!(error, DicomError::UnsupportedMultiVolume { .. }));
    }
}
