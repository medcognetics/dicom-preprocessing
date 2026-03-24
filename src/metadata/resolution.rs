use crate::errors::dicom::DicomError;
use crate::errors::tiff::TiffError;
use std::io::{Read, Seek, Write};
use tiff::decoder::Decoder;
use tiff::encoder::colortype::ColorType;
// Removed unused Compression import
use tiff::encoder::{ImageEncoder, Rational, TiffKind};
use tiff::tags::ResolutionUnit;

use dicom::dictionary_std::tags;
use dicom::object::{FileDicomObject, InMemDicomObject};

use crate::metadata::resolve_frame_order;
use crate::metadata::WriteTags;

const MM_PER_CM: f32 = 10.0;
const MM_PER_IN: f32 = 25.4;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Resolution {
    pub pixels_per_mm_x: f32,
    pub pixels_per_mm_y: f32,
    pub frames_per_mm: Option<f32>,
}

impl<T> TryFrom<&mut Decoder<T>> for Resolution
where
    T: Read + Seek,
{
    type Error = TiffError;

    // Extract resolution metadata from a TIFF decoder
    fn try_from(decoder: &mut Decoder<T>) -> Result<Self, TiffError> {
        // Parse resolution unit
        let unit = decoder
            .get_tag(tiff::tags::Tag::ResolutionUnit)
            .map_err(|_| TiffError::MissingPropertyError {
                name: "Resolution Unit",
            })?
            .into_u16()?;
        let unit = ResolutionUnit::from_u16(unit).ok_or(TiffError::Other {
            message: "invalid resolution unit".to_string(),
        })?;

        // Parse x resolution from a rational
        let x_resolution = if let [x_numerator, x_denominator] =
            decoder.get_tag_u32_vec(tiff::tags::Tag::XResolution)?[..]
        {
            x_numerator as f32 / x_denominator as f32
        } else {
            return Err(TiffError::InvalidPropertyError {
                name: "X Resolution",
            });
        };

        // Parse y resolution from a rational
        let y_resolution = if let [y_numerator, y_denominator] =
            decoder.get_tag_u32_vec(tiff::tags::Tag::YResolution)?[..]
        {
            y_numerator as f32 / y_denominator as f32
        } else {
            return Err(TiffError::InvalidPropertyError {
                name: "Y Resolution",
            });
        };

        // Convert to pixels per mm
        let (x_resolution, y_resolution) = match unit {
            ResolutionUnit::Centimeter => (x_resolution / MM_PER_CM, y_resolution / MM_PER_CM),
            ResolutionUnit::Inch => (x_resolution / MM_PER_IN, y_resolution / MM_PER_IN),
            _ => (x_resolution, y_resolution),
        };

        Ok(Resolution {
            pixels_per_mm_x: x_resolution,
            pixels_per_mm_y: y_resolution,
            frames_per_mm: None,
        })
    }
}

impl Resolution {
    pub fn new(pixels_per_mm_x: f32, pixels_per_mm_y: f32) -> Self {
        Resolution {
            pixels_per_mm_x,
            pixels_per_mm_y,
            frames_per_mm: None,
        }
    }

    pub fn new_3d(pixels_per_mm_x: f32, pixels_per_mm_y: f32, frames_per_mm: f32) -> Self {
        Resolution {
            pixels_per_mm_x,
            pixels_per_mm_y,
            frames_per_mm: Some(frames_per_mm),
        }
    }

    pub fn scale(&self, value: f32) -> Self {
        Resolution {
            pixels_per_mm_x: self.pixels_per_mm_x * value,
            pixels_per_mm_y: self.pixels_per_mm_y * value,
            frames_per_mm: self.frames_per_mm.map(|f| f * value),
        }
    }

    pub fn with_frames_per_mm(mut self, frames_per_mm: f32) -> Self {
        self.frames_per_mm = Some(frames_per_mm);
        self
    }
}

impl WriteTags for Resolution {
    fn write_tags<W, C, K>(&self, tiff: &mut ImageEncoder<W, C, K>) -> Result<(), TiffError>
    where
        W: Write + Seek,
        C: ColorType,
        K: TiffKind,
    {
        tiff.x_resolution(Rational {
            n: (self.pixels_per_mm_x * MM_PER_CM) as u32,
            d: 1,
        });
        tiff.y_resolution(Rational {
            n: (self.pixels_per_mm_y * MM_PER_CM) as u32,
            d: 1,
        });
        tiff.resolution_unit(tiff::tags::ResolutionUnit::Centimeter);
        Ok(())
    }
}

impl From<Resolution> for (f32, f32) {
    fn from(resolution: Resolution) -> Self {
        (resolution.pixels_per_mm_x, resolution.pixels_per_mm_y)
    }
}

impl From<(f32, f32)> for Resolution {
    fn from((pixels_per_mm_x, pixels_per_mm_y): (f32, f32)) -> Self {
        Resolution {
            pixels_per_mm_x,
            pixels_per_mm_y,
            frames_per_mm: None,
        }
    }
}

impl From<(f32, f32, f32)> for Resolution {
    fn from((pixels_per_mm_x, pixels_per_mm_y, frames_per_mm): (f32, f32, f32)) -> Self {
        Resolution {
            pixels_per_mm_x,
            pixels_per_mm_y,
            frames_per_mm: Some(frames_per_mm),
        }
    }
}

impl TryFrom<&FileDicomObject<InMemDicomObject>> for Resolution {
    type Error = DicomError;

    fn try_from(file: &FileDicomObject<InMemDicomObject>) -> Result<Self, Self::Error> {
        let [pixel_spacing_mm_y, pixel_spacing_mm_x] =
            pixel_spacing_mm(file).ok_or(DicomError::MissingPropertyError {
                name: "Pixel Spacing",
            })?;

        let frame_spacing_mm = resolve_frame_order(file)
            .ok()
            .and_then(|plan| plan.z_spacing_mm)
            .or_else(|| pixel_measure_spacing_mm(file))
            .or_else(|| top_level_slice_spacing_mm(file));

        // Convert to pixels per mm and frames per mm
        let mut resolution: Resolution =
            (1.0 / pixel_spacing_mm_x, 1.0 / pixel_spacing_mm_y).into();
        if let Some(spacing_mm) = frame_spacing_mm {
            resolution.frames_per_mm = Some(1.0 / spacing_mm);
        }
        Ok(resolution)
    }
}

fn pixel_spacing_mm(file: &FileDicomObject<InMemDicomObject>) -> Option<[f32; 2]> {
    file.get(tags::PIXEL_SPACING)
        .or_else(|| file.get(tags::IMAGER_PIXEL_SPACING))
        .and_then(parse_pair_from_element)
        .or_else(|| pixel_measure_item(file).and_then(pixel_spacing_from_pixel_measure_item))
}

fn pixel_spacing_from_pixel_measure_item(pixel_measures: &InMemDicomObject) -> Option<[f32; 2]> {
    pixel_measures
        .get(tags::PIXEL_SPACING)
        .and_then(parse_pair_from_element)
}

fn pixel_measure_item(file: &FileDicomObject<InMemDicomObject>) -> Option<&InMemDicomObject> {
    first_sequence_item(file, tags::SHARED_FUNCTIONAL_GROUPS_SEQUENCE)
        .and_then(|shared| sequence_item_in_object(shared, tags::PIXEL_MEASURES_SEQUENCE))
        .or_else(|| {
            file.get(tags::PER_FRAME_FUNCTIONAL_GROUPS_SEQUENCE)
                .and_then(|element| element.items())
                .and_then(|items| items.first())
                .and_then(|frame| sequence_item_in_object(frame, tags::PIXEL_MEASURES_SEQUENCE))
        })
}

fn pixel_measure_spacing_mm(file: &FileDicomObject<InMemDicomObject>) -> Option<f32> {
    pixel_measure_item(file).and_then(spacing_from_pixel_measure_item)
}

fn spacing_from_pixel_measure_item(pixel_measures: &InMemDicomObject) -> Option<f32> {
    pixel_measures
        .get(tags::SPACING_BETWEEN_SLICES)
        .or_else(|| pixel_measures.get(tags::SLICE_THICKNESS))
        .and_then(parse_single_float_from_element)
        .map(|value| value as f32)
}

fn top_level_slice_spacing_mm(file: &FileDicomObject<InMemDicomObject>) -> Option<f32> {
    file.get(tags::SPACING_BETWEEN_SLICES)
        .or_else(|| file.get(tags::SLICE_THICKNESS))
        .and_then(parse_single_float_from_element)
        .map(|value| value as f32)
}

fn first_sequence_item(
    object: &InMemDicomObject,
    tag: dicom::core::Tag,
) -> Option<&InMemDicomObject> {
    object.get(tag)?.items()?.first()
}

fn sequence_item_in_object(
    object: &InMemDicomObject,
    tag: dicom::core::Tag,
) -> Option<&InMemDicomObject> {
    object.get(tag)?.items()?.first()
}

fn parse_pair_from_element(
    element: &dicom::core::DataElement<InMemDicomObject>,
) -> Option<[f32; 2]> {
    let raw = element.value().to_str().ok()?;
    let mut parts = raw.split('\\');
    let first = parts.next()?.parse::<f32>().ok()?;
    let second = parts.next()?.parse::<f32>().ok()?;
    Some([first, second])
}

fn parse_single_float_from_element(
    element: &dicom::core::DataElement<InMemDicomObject>,
) -> Option<f64> {
    element
        .value()
        .to_str()
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use dicom::core::value::DataSetSequence;
    use dicom::core::{PrimitiveValue, VR};
    use dicom::dictionary_std::uids;
    use dicom::object::mem::InMemElement;
    use dicom::object::open_file;
    use dicom::object::FileMetaTableBuilder;
    use rstest::rstest;
    use std::fs::File;

    use tempfile::tempdir;
    use tiff::decoder::Decoder as TiffDecoder;
    use tiff::encoder::TiffEncoder;

    fn seq(tag: dicom::core::Tag, items: Vec<InMemDicomObject>) -> InMemElement {
        InMemElement::new(tag, VR::SQ, DataSetSequence::from(items))
    }

    #[rstest]
    #[case("pydicom/CT_small.dcm", Resolution { pixels_per_mm_x: 1.5117888, pixels_per_mm_y: 1.5117888, frames_per_mm: Some(0.2) })]
    fn test_try_from(#[case] dicom_path: &str, #[case] expected: Resolution) {
        let dicom_file = dicom_test_files::path(dicom_path).unwrap();
        let dicom_file = open_file(dicom_file).unwrap();
        let result = Resolution::try_from(&dicom_file).unwrap();
        assert_eq!(result, expected);
    }

    #[rstest]
    #[case(
        Resolution { pixels_per_mm_x: 1.5117888, pixels_per_mm_y: 1.5117888, frames_per_mm: None },
        1.5,
        1.5,
    )]
    fn test_write_tags(
        #[case] resolution: Resolution,
        #[case] x_resolution: f32,
        #[case] y_resolution: f32,
    ) {
        // Prepare the TIFF
        let temp_dir = tempdir().unwrap();
        let temp_file_path = temp_dir.path().join("temp.tif");
        let mut tiff = TiffEncoder::new(File::create(temp_file_path.clone()).unwrap()).unwrap();
        let mut img = tiff
            .new_image::<tiff::encoder::colortype::Gray16>(1, 1)
            .unwrap();

        // Write the tags
        resolution.write_tags(&mut img).unwrap();

        // Write some dummy image data
        let data: Vec<u16> = vec![0; 2];
        img.write_data(data.as_slice()).unwrap();

        // Read the TIFF back
        let mut tiff = TiffDecoder::new(File::open(temp_file_path).unwrap()).unwrap();
        let actual = Resolution::try_from(&mut tiff).unwrap();
        assert_eq!(x_resolution, actual.pixels_per_mm_x);
        assert_eq!(y_resolution, actual.pixels_per_mm_y);
    }

    #[rstest]
    #[case(Resolution { pixels_per_mm_x: 1.0, pixels_per_mm_y: 1.0, frames_per_mm: None }, 2.0, Resolution { pixels_per_mm_x: 2.0, pixels_per_mm_y: 2.0, frames_per_mm: None })]
    #[case(Resolution { pixels_per_mm_x: 1.5, pixels_per_mm_y: 1.5, frames_per_mm: None }, 0.5, Resolution { pixels_per_mm_x: 0.75, pixels_per_mm_y: 0.75, frames_per_mm: None })]
    #[case(Resolution { pixels_per_mm_x: 2.0, pixels_per_mm_y: 2.0, frames_per_mm: Some(1.0) }, 1.0, Resolution { pixels_per_mm_x: 2.0, pixels_per_mm_y: 2.0, frames_per_mm: Some(1.0) })]
    fn test_scale(
        #[case] resolution: Resolution,
        #[case] scale_factor: f32,
        #[case] expected: Resolution,
    ) {
        let result = resolution.scale(scale_factor);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_try_from_shared_pixel_measures_sequence() {
        let orientation = PrimitiveValue::from([1.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let object = InMemDicomObject::from_element_iter([
            InMemElement::new(tags::NUMBER_OF_FRAMES, VR::IS, "3"),
            seq(
                tags::SHARED_FUNCTIONAL_GROUPS_SEQUENCE,
                vec![InMemDicomObject::from_element_iter([seq(
                    tags::PIXEL_MEASURES_SEQUENCE,
                    vec![InMemDicomObject::from_element_iter([
                        InMemElement::new(tags::PIXEL_SPACING, VR::DS, "0.5\\0.25"),
                        InMemElement::new(tags::SPACING_BETWEEN_SLICES, VR::DS, "1.5"),
                    ])],
                )])],
            ),
            seq(
                tags::PER_FRAME_FUNCTIONAL_GROUPS_SEQUENCE,
                vec![
                    InMemDicomObject::from_element_iter([
                        seq(
                            tags::PLANE_POSITION_SEQUENCE,
                            vec![InMemDicomObject::from_element_iter([InMemElement::new(
                                tags::IMAGE_POSITION_PATIENT,
                                VR::DS,
                                PrimitiveValue::from([0.0_f64, 0.0, 0.0]),
                            )])],
                        ),
                        seq(
                            tags::PLANE_ORIENTATION_SEQUENCE,
                            vec![InMemDicomObject::from_element_iter([InMemElement::new(
                                tags::IMAGE_ORIENTATION_PATIENT,
                                VR::DS,
                                orientation.clone(),
                            )])],
                        ),
                    ]),
                    InMemDicomObject::from_element_iter([
                        seq(
                            tags::PLANE_POSITION_SEQUENCE,
                            vec![InMemDicomObject::from_element_iter([InMemElement::new(
                                tags::IMAGE_POSITION_PATIENT,
                                VR::DS,
                                PrimitiveValue::from([0.0_f64, 0.0, 1.5]),
                            )])],
                        ),
                        seq(
                            tags::PLANE_ORIENTATION_SEQUENCE,
                            vec![InMemDicomObject::from_element_iter([InMemElement::new(
                                tags::IMAGE_ORIENTATION_PATIENT,
                                VR::DS,
                                orientation.clone(),
                            )])],
                        ),
                    ]),
                    InMemDicomObject::from_element_iter([
                        seq(
                            tags::PLANE_POSITION_SEQUENCE,
                            vec![InMemDicomObject::from_element_iter([InMemElement::new(
                                tags::IMAGE_POSITION_PATIENT,
                                VR::DS,
                                PrimitiveValue::from([0.0_f64, 0.0, 3.0]),
                            )])],
                        ),
                        seq(
                            tags::PLANE_ORIENTATION_SEQUENCE,
                            vec![InMemDicomObject::from_element_iter([InMemElement::new(
                                tags::IMAGE_ORIENTATION_PATIENT,
                                VR::DS,
                                orientation,
                            )])],
                        ),
                    ]),
                ],
            ),
        ])
        .with_meta(
            FileMetaTableBuilder::new()
                .transfer_syntax(uids::EXPLICIT_VR_LITTLE_ENDIAN)
                .media_storage_sop_class_uid(uids::SECONDARY_CAPTURE_IMAGE_STORAGE)
                .media_storage_sop_instance_uid("1.2.826.0.1.3680043.2.1125.1"),
        )
        .unwrap();

        let result = Resolution::try_from(&object).unwrap();
        assert_eq!(result.pixels_per_mm_x, 4.0);
        assert_eq!(result.pixels_per_mm_y, 2.0);
        assert_eq!(result.frames_per_mm, Some(2.0 / 3.0));
    }
}
