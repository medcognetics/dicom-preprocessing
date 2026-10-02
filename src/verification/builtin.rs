use std::collections::BTreeMap;

use super::*;
use crate::{FlipOptions, PaddingDirection, Preprocessor};
use dicom::transfer_syntax::TransferSyntaxRegistry;
use dicom_pixeldata::{ConvertOptions, ModalityLutOption};

const MANIFEST: &str = include_str!("../../fixtures/verification/manifest.json");
const PREPROCESSING_FIXTURE: &str = "builtin/static-multiframe-native";
const DECODER_REGISTRATION: &str = "builtin/decoder-registration";

macro_rules! fixtures {
    ($($name:literal),* $(,)?) => {
        /// Every embedded fixture file and its bytes.
        pub(super) const FIXTURES: &[(&str, &[u8])] = &[
            $((concat!($name, ".dcm"), include_bytes!(concat!("../../fixtures/verification/", $name, ".dcm")))),*
        ];
    };
}

fixtures!(
    "implicit",
    "explicit",
    "big-endian",
    "deflated",
    "encapsulated-native",
    "deflated-frame",
    "rle",
    "jpeg-baseline",
    "jpeg-extended",
    "jpeg-extended-37x23",
    "jpeg-lossless",
    "jpeg-lossless-sv1",
    "j2k-lossless",
    "j2k-lossy",
    "j2k-part2-lossless",
    "j2k-part2-lossy",
    "htj2k-lossless",
    "htj2k-rpcl",
    "htj2k-lossy",
    "multiframe-native",
    "signed-16bit",
    "rgb",
    "planar-rgb",
    "rle-multiframe",
    "rle-16bit",
    "rle-rgb-16bit",
    "jpeg-baseline-multiframe",
);

/// Bytes embedded in the binary for the suite: every fixture plus the manifest.
#[cfg(test)]
pub(super) fn embedded_len() -> usize {
    MANIFEST.len() + FIXTURES.iter().map(|(_, bytes)| bytes.len()).sum::<usize>()
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Manifest {
    #[cfg_attr(not(test), allow(dead_code))] // Checked by tests.
    pub(super) schema_version: u32,
    tags: Vec<TagExpectation>,
    pub(super) pixels: BTreeMap<String, Vec<i32>>,
    pub(super) cases: Vec<ManifestCase>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ManifestCase {
    pub(super) id: String,
    pub(super) file: String,
    transfer_syntax_uid: String,
    rows: u32,
    columns: u32,
    pub(super) frames: Vec<ManifestFrame>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ManifestFrame {
    samples_per_pixel: u16,
    planar_configuration: u16,
    sample_type: SampleType,
    pub(super) pixels: String,
    absolute_tolerance: u32,
}

pub(super) fn manifest() -> Manifest {
    serde_json::from_str(MANIFEST).expect("checked fixture manifest")
}

/// Expands the manifest into one case per fixture. Rows and Columns join the shared tags.
pub(super) fn static_cases() -> Vec<VerificationCase> {
    let manifest = manifest();
    manifest
        .cases
        .iter()
        .map(|case| {
            let dimension = |element: u16, value: u32| TagExpectation {
                path: vec![TagPathComponent {
                    group: 0x0028,
                    element,
                    item: None,
                }],
                vr: "US".into(),
                values: vec![value.to_string()],
            };
            let mut tags = manifest.tags.clone();
            tags.push(dimension(0x0010, case.rows));
            tags.push(dimension(0x0011, case.columns));
            VerificationCase {
                id: case.id.clone(),
                dicom_bytes: FIXTURES
                    .iter()
                    .find(|(name, _)| *name == case.file)
                    .map(|(_, bytes)| *bytes)
                    .expect("embedded fixture"),
                transfer_syntax_uid: case.transfer_syntax_uid.clone(),
                tags,
                frames: case
                    .frames
                    .iter()
                    .map(|frame| FrameExpectation {
                        width: case.columns,
                        height: case.rows,
                        samples_per_pixel: frame.samples_per_pixel,
                        planar_configuration: frame.planar_configuration,
                        sample_type: frame.sample_type,
                        values: manifest.pixels[&frame.pixels].clone(),
                        absolute_tolerance: frame.absolute_tolerance,
                    })
                    .collect(),
            }
        })
        .collect()
}

pub(super) fn run() -> (Vec<VerificationCaseResult>, Vec<CodecCoverageResult>) {
    let cases = static_cases();
    let mut results: Vec<_> = cases.iter().map(run_case).collect();
    results.push(preprocessing(&cases));
    results.push(decoder_registration());
    let codecs = coverage(&results, decodable_transfer_syntaxes());
    (results, codecs)
}

/// Transfer syntaxes the registry can decode. Reference-only syntaxes carry no local pixels.
fn decodable_transfer_syntaxes() -> Vec<String> {
    TransferSyntaxRegistry
        .iter()
        .filter(|ts| ts.can_decode_all() && !ts.name().contains("Referenced"))
        .map(|ts| ts.uid().to_owned())
        .collect()
}

/// Requires a passing fixture for every decodable syntax, and the registration case for every
/// syntax whose reader this crate provides.
pub(super) fn coverage(
    results: &[VerificationCaseResult],
    decodable: Vec<String>,
) -> Vec<CodecCoverageResult> {
    let mut required: BTreeMap<String, Vec<String>> =
        decodable.into_iter().map(|uid| (uid, Vec::new())).collect();
    for result in results.iter().filter(|r| r.source == "embedded_fixture") {
        if let Some(uid) = &result.transfer_syntax_uid {
            required
                .entry(uid.clone())
                .or_default()
                .push(result.id.clone());
        }
    }
    for &(uid, _) in crate::codec::OVERRIDDEN_TRANSFER_SYNTAXES {
        required
            .entry(uid.into())
            .or_default()
            .push(DECODER_REGISTRATION.into());
    }
    required
        .into_iter()
        .map(|(uid, ids)| {
            let passed = ids.iter().any(|id| id != DECODER_REGISTRATION)
                && ids.iter().all(|id| {
                    results
                        .iter()
                        .any(|result| &result.id == id && result.passed)
                });
            CodecCoverageResult {
                transfer_syntax_uid: uid,
                required_cases: ids,
                passed,
            }
        })
        .collect()
}

/// Checks that this crate's JPEG and RLE readers, not dicom-rs's built-in ones, are registered.
///
/// A consumer that enables `dicom-pixeldata`'s `native`, `jpeg`, or `rle` features keeps the
/// built-in readers, which decode some inputs incorrectly (#128).
fn decoder_registration() -> VerificationCaseResult {
    let checks = crate::codec::decoder_registrations()
        .into_iter()
        .map(|registration| {
            let name = format!("reader_{}", registration.transfer_syntax_uid);
            if registration.active {
                VerificationCheck::new(name, true)
            } else {
                VerificationCheck::failure(
                    name,
                    "the dicom-rs built-in reader is registered; disable dicom-pixeldata's \
                     native, jpeg, and rle features",
                )
            }
        })
        .collect();
    result(DECODER_REGISTRATION, "builtin", None, checks)
}

/// Display conversion, caller-selected flips, nearest-neighbor resize, centered padding,
/// frame order, maximum-intensity projection, and serial/parallel repeatability, on the
/// three-frame 8-bit fixture.
fn preprocessing(cases: &[VerificationCase]) -> VerificationCaseResult {
    let run = || -> Result<Vec<VerificationCheck>, ()> {
        let case = cases
            .iter()
            .find(|case| case.id == PREPROCESSING_FIXTURE)
            .ok_or(())?;
        let file = from_reader(Cursor::new(case.dicom_bytes)).map_err(|_| ())?;
        let convert = ConvertOptions::default().with_modality_lut(ModalityLutOption::None);
        let viewer =
            ViewerDicom::from_object(file.clone(), VolumeHandler::keep()).map_err(|_| ())?;
        let mut checks = Vec::new();
        for (index, expected) in case.frames.iter().enumerate() {
            let image = viewer
                .decode_display_frame_with_options(index, &convert)
                .map_err(|_| ())?;
            checks.push(VerificationCheck::new(
                format!("display_{index}"),
                image.as_bytes() == expected.values.iter().map(|&v| v as u8).collect::<Vec<_>>(),
            ));
        }
        let preprocessor = Preprocessor {
            crop: false,
            size: Some((16, 20)),
            filter: crate::FilterType::Nearest,
            padding_direction: PaddingDirection::Center,
            use_padding: true,
            volume_handler: VolumeHandler::keep(),
            convert_options: convert.clone(),
            flip: FlipOptions {
                horizontal: true,
                vertical: true,
            },
            ..Default::default()
        };
        let expected_images: Vec<Vec<u8>> = case
            .frames
            .iter()
            .map(|frame| {
                let mut expected = vec![0u8; 16 * 20];
                for y in 0..16 {
                    for x in 0..16 {
                        expected[(y + 2) * 16 + x] =
                            frame.values[(7 - y / 2) * 8 + (7 - x / 2)] as u8;
                    }
                }
                expected
            })
            .collect();
        let mut first = None;
        for (run, parallel) in [false, true, false].into_iter().enumerate() {
            let (images, metadata, plan) = preprocessor
                .prepare_image_with_plan(&file, parallel)
                .map_err(|_| ())?;
            checks.push(VerificationCheck::new(
                format!("frame_order_{run}"),
                plan.stored_frame_order == vec![0, 1, 2],
            ));
            checks.push(VerificationCheck::new(
                format!("transforms_{run}"),
                images.len() == 3
                    && images
                        .iter()
                        .zip(&expected_images)
                        .all(|(image, expected)| {
                            image.width() == 16
                                && image.height() == 20
                                && image.as_bytes() == expected
                        }),
            ));
            checks.push(VerificationCheck::new(
                format!("flip_metadata_{run}"),
                metadata
                    .flip
                    .is_some_and(|flip| flip.horizontal && flip.vertical),
            ));
            let snapshot = images
                .iter()
                .map(|image| image.as_bytes().to_vec())
                .collect::<Vec<_>>();
            match &first {
                Some(first) => checks.push(VerificationCheck::new(
                    format!("transform_repeat_{run}"),
                    first == &snapshot,
                )),
                None => first = Some(snapshot),
            }
        }
        let projector = Preprocessor {
            crop: false,
            use_padding: false,
            volume_handler: VolumeHandler::max_intensity(0, 0),
            convert_options: convert,
            ..Default::default()
        };
        let (images, _) = projector.prepare_image(&file, true).map_err(|_| ())?;
        let expected = (0..64)
            .map(|i| {
                case.frames
                    .iter()
                    .map(|frame| frame.values[i] as u8)
                    .max()
                    .unwrap()
            })
            .collect::<Vec<_>>();
        checks.push(VerificationCheck::new(
            "max_projection",
            images.len() == 1 && images[0].as_bytes() == expected,
        ));
        Ok(checks)
    };
    let checks = run().unwrap_or_else(|()| {
        vec![VerificationCheck::failure(
            "preprocessing",
            "image conversion or preprocessing failed",
        )]
    });
    result("builtin/preprocessing", "builtin", None, checks)
}
