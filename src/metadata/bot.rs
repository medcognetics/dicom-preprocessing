use crate::errors::DicomError;
use crate::metadata::preprocessing::FrameCount;
use dicom::dictionary_std::tags;
use dicom::object::{FileDicomObject, InMemDicomObject};

const OFFSET_ITEM_DELIMITER_BYTES: usize = 8;
const OFFSETS_PREVIEW_LIMIT: usize = 16;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BotSummary {
    pub file_is_encapsulated: bool,
    pub number_of_frames: usize,
    pub fragment_count: usize,
    pub offset_count: usize,
    pub offset_count_matches_frames: bool,
    pub starts_at_zero: bool,
    pub strictly_increasing: bool,
    pub offsets_in_bounds: bool,
    pub bot_required_for_decode_frame: bool,
    pub decode_frame_path_at_risk: bool,
    pub encoded_stream_len: usize,
    pub offsets_preview: Vec<u32>,
    pub needs_correction: bool,
}

impl BotSummary {
    fn non_encapsulated(number_of_frames: usize) -> Self {
        Self {
            file_is_encapsulated: false,
            number_of_frames,
            fragment_count: 0,
            offset_count: 0,
            offset_count_matches_frames: true,
            starts_at_zero: true,
            strictly_increasing: true,
            offsets_in_bounds: true,
            bot_required_for_decode_frame: false,
            decode_frame_path_at_risk: false,
            encoded_stream_len: 0,
            offsets_preview: Vec::new(),
            needs_correction: false,
        }
    }
}

pub fn summarize_basic_offset_table(
    file: &FileDicomObject<InMemDicomObject>,
) -> Result<BotSummary, DicomError> {
    let number_of_frames: u32 = FrameCount::try_from(file)?.into();
    let number_of_frames = number_of_frames as usize;

    let pixel_data = file
        .get(tags::PIXEL_DATA)
        .ok_or(DicomError::MissingPropertyError { name: "Pixel Data" })?;
    let value = pixel_data.value();

    let (offset_table, fragments) = match (value.offset_table(), value.fragments()) {
        (Some(offset_table), Some(fragments)) => (offset_table, fragments),
        _ => return Ok(BotSummary::non_encapsulated(number_of_frames)),
    };

    let fragment_count = fragments.len();
    let offset_count = offset_table.len();
    let offset_count_matches_frames = offset_count == number_of_frames;
    let starts_at_zero = offset_table.first().is_some_and(|&offset| offset == 0);
    let strictly_increasing = offset_table.windows(2).all(|window| window[0] < window[1]);
    let encoded_stream_len = fragments.iter().fold(0usize, |acc, fragment| {
        acc.saturating_add(fragment.len().saturating_add(OFFSET_ITEM_DELIMITER_BYTES))
    });
    let offsets_in_bounds = offset_table
        .iter()
        .all(|&offset| (offset as usize) < encoded_stream_len);
    let bot_required_for_decode_frame = fragment_count > 1 && fragment_count != number_of_frames;
    let bot_is_malformed = !(offset_count_matches_frames
        && starts_at_zero
        && strictly_increasing
        && offsets_in_bounds);
    let decode_frame_path_at_risk = bot_required_for_decode_frame && bot_is_malformed;
    let needs_correction = bot_is_malformed;

    Ok(BotSummary {
        file_is_encapsulated: true,
        number_of_frames,
        fragment_count,
        offset_count,
        offset_count_matches_frames,
        starts_at_zero,
        strictly_increasing,
        offsets_in_bounds,
        bot_required_for_decode_frame,
        decode_frame_path_at_risk,
        encoded_stream_len,
        offsets_preview: offset_table
            .iter()
            .take(OFFSETS_PREVIEW_LIMIT)
            .copied()
            .collect(),
        needs_correction,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use dicom::core::{DataElement, PrimitiveValue, VR};
    use dicom::dictionary_std::tags;
    use dicom::object::open_file;

    fn split_fragment_even(fragment: &[u8]) -> (Vec<u8>, Vec<u8>) {
        let midpoint = (fragment.len() / 2) & !1;
        let midpoint = midpoint.clamp(2, fragment.len().saturating_sub(2));
        (fragment[..midpoint].to_vec(), fragment[midpoint..].to_vec())
    }

    fn rewrite_to_two_frame_split_fragment_jpeg(
        mut dicom: FileDicomObject<InMemDicomObject>,
        offset_table: Vec<u32>,
    ) -> FileDicomObject<InMemDicomObject> {
        let pixel_data = dicom.get(tags::PIXEL_DATA).unwrap();
        let fragments = pixel_data.value().fragments().unwrap();

        let (f0a, f0b) = split_fragment_even(&fragments[0]);
        let (f1a, f1b) = split_fragment_even(&fragments[1]);
        let rewritten_fragments = vec![f0a, f0b, f1a, f1b];

        dicom.put_element(DataElement::new(
            tags::NUMBER_OF_FRAMES,
            VR::IS,
            PrimitiveValue::from("2"),
        ));
        let updated = dicom.update_value(tags::PIXEL_DATA, move |value| {
            let offset_table_mut = value.offset_table_mut().expect("pixel sequence BOT");
            *offset_table_mut = offset_table.clone().into();

            let fragments_mut = value.fragments_mut().expect("pixel sequence fragments");
            *fragments_mut = rewritten_fragments.clone().into();
        });
        assert!(updated, "expected pixel data to be present");
        dicom
    }

    #[test]
    fn summarize_bot_marks_valid_offsets_as_no_correction_needed() {
        let source = dicom_test_files::path("pydicom/color3d_jpeg_baseline.dcm").unwrap();
        let base_dicom = open_file(&source).unwrap();
        let pixel_data = base_dicom.get(tags::PIXEL_DATA).unwrap();
        let fragments = pixel_data.value().fragments().unwrap();
        let (f0a, f0b) = split_fragment_even(&fragments[0]);
        let valid_second_offset =
            (f0a.len() + OFFSET_ITEM_DELIMITER_BYTES + f0b.len() + OFFSET_ITEM_DELIMITER_BYTES)
                as u32;

        let valid_dicom = rewrite_to_two_frame_split_fragment_jpeg(
            open_file(&source).unwrap(),
            vec![0, valid_second_offset],
        );
        let summary = summarize_basic_offset_table(&valid_dicom).unwrap();
        assert!(summary.file_is_encapsulated);
        assert!(summary.bot_required_for_decode_frame);
        assert!(!summary.needs_correction);
    }

    #[test]
    fn summarize_bot_marks_malformed_offsets_as_needing_correction() {
        let source = dicom_test_files::path("pydicom/color3d_jpeg_baseline.dcm").unwrap();
        let malformed_dicom =
            rewrite_to_two_frame_split_fragment_jpeg(open_file(&source).unwrap(), vec![0, 0]);
        let summary = summarize_basic_offset_table(&malformed_dicom).unwrap();
        assert!(summary.file_is_encapsulated);
        assert!(summary.bot_required_for_decode_frame);
        assert!(summary.decode_frame_path_at_risk);
        assert!(summary.needs_correction);
    }

    #[test]
    fn summarize_bot_marks_malformed_even_when_decode_frame_path_not_at_risk() {
        let source = dicom_test_files::path("pydicom/color3d_jpeg_baseline.dcm").unwrap();
        let mut dicom = open_file(&source).unwrap();
        let updated = dicom.update_value(tags::PIXEL_DATA, |value| {
            let offset_table_mut = value.offset_table_mut().expect("pixel sequence BOT");
            *offset_table_mut = vec![0, 0].into();
        });
        assert!(updated);

        // Keep frame count consistent with source (64), where fragments == frames
        // and decode_frame path does not rely on BOT offsets.
        let summary = summarize_basic_offset_table(&dicom).unwrap();
        assert!(summary.file_is_encapsulated);
        assert!(!summary.bot_required_for_decode_frame);
        assert!(!summary.decode_frame_path_at_risk);
        assert!(summary.needs_correction);
    }

    #[test]
    fn summarize_bot_handles_native_pixel_data_without_correction() {
        let dicom = open_file(dicom_test_files::path("pydicom/CT_small.dcm").unwrap()).unwrap();
        let summary = summarize_basic_offset_table(&dicom).unwrap();
        assert!(!summary.file_is_encapsulated);
        assert!(!summary.needs_correction);
    }
}
