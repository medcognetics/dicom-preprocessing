use std::path::Path;

use dicom::object::{open_file, FileDicomObject, InMemDicomObject};
use dicom::pixeldata::ConvertOptions;
use image::DynamicImage;
use snafu::ResultExt;

use crate::errors::{dicom::ReadSnafu, DicomError};
use crate::preprocess::Preprocessor;
use crate::transform::volume::{
    DecodedStoredFrame, PreparedVolume, VolumeFramePlan, VolumeHandler,
};

#[derive(Debug, Clone)]
pub struct ViewerDicom {
    file: FileDicomObject<InMemDicomObject>,
    volume_handler: VolumeHandler,
    frame_plan: VolumeFramePlan,
}

impl ViewerDicom {
    pub fn open<P: AsRef<Path>>(
        path: P,
        volume_handler: VolumeHandler,
    ) -> Result<Self, DicomError> {
        let mut file = open_file(path.as_ref()).context(ReadSnafu)?;
        Preprocessor::sanitize_dicom(&mut file);
        Self::from_sanitized_object(file, volume_handler)
    }

    pub fn from_object(
        mut file: FileDicomObject<InMemDicomObject>,
        volume_handler: VolumeHandler,
    ) -> Result<Self, DicomError> {
        Preprocessor::sanitize_dicom(&mut file);
        Self::from_sanitized_object(file, volume_handler)
    }

    fn from_sanitized_object(
        file: FileDicomObject<InMemDicomObject>,
        volume_handler: VolumeHandler,
    ) -> Result<Self, DicomError> {
        let frame_plan = volume_handler.frame_plan(&file)?;
        Ok(Self {
            file,
            volume_handler,
            frame_plan,
        })
    }

    pub fn file(&self) -> &FileDicomObject<InMemDicomObject> {
        &self.file
    }

    pub fn volume_handler(&self) -> &VolumeHandler {
        &self.volume_handler
    }

    pub fn frame_plan(&self) -> &VolumeFramePlan {
        &self.frame_plan
    }

    pub fn prepare_volume_with_options(
        &self,
        options: &ConvertOptions,
        parallel: bool,
    ) -> Result<PreparedVolume, DicomError> {
        self.volume_handler
            .prepare_volume_with_options(&self.file, options, parallel)
    }

    pub fn decode_display_frame_with_options(
        &self,
        display_frame_index: usize,
        options: &ConvertOptions,
    ) -> Result<DynamicImage, DicomError> {
        let prepared = self.prepare_volume_with_options(options, false)?;
        prepared.images.into_iter().nth(display_frame_index).ok_or(
            DicomError::DisplayFrameIndexError {
                display_frame_index,
                number_of_frames: prepared.frame_plan.display_frames.len(),
            },
        )
    }

    pub fn decode_raw_display_frame(
        &self,
        display_frame_index: usize,
    ) -> Result<DecodedStoredFrame, DicomError> {
        let stored_frame_index = self
            .frame_plan
            .stored_frame_for_display(display_frame_index)?;
        VolumeHandler::decode_stored_frame_raw(&self.file, stored_frame_index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::volume::VolumeFrameSource;
    use dicom::core::header::HasLength;
    use dicom::core::{DataElement, PrimitiveValue, VR};
    use dicom::dictionary_std::tags;
    use dicom::object::open_file;

    #[test]
    fn from_object_sanitizes_empty_voi_lut_function() {
        let mut dicom = open_file(dicom_test_files::path("pydicom/CT_small.dcm").unwrap()).unwrap();
        dicom.put_element(DataElement::new(
            tags::VOILUT_FUNCTION,
            VR::LO,
            PrimitiveValue::Empty,
        ));
        assert!(dicom
            .get(tags::VOILUT_FUNCTION)
            .is_some_and(|element| element.is_empty()));

        let viewer = ViewerDicom::from_object(dicom, VolumeHandler::keep()).unwrap();

        assert!(viewer.file().get(tags::VOILUT_FUNCTION).is_none());
    }

    #[test]
    fn decode_raw_display_frame_uses_shared_frame_plan() {
        let viewer = ViewerDicom::open(
            dicom_test_files::path("pydicom/CT_small.dcm").unwrap(),
            VolumeHandler::keep(),
        )
        .unwrap();

        let raw = viewer.decode_raw_display_frame(0).unwrap();

        assert_eq!(
            viewer.frame_plan().display_frames,
            vec![VolumeFrameSource::StoredFrame {
                stored_frame_index: 0
            }]
        );
        assert!(!raw.data.is_empty());
    }
}
