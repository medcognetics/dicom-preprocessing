use super::*;
use crate::{FlipOptions, PaddingDirection, Preprocessor};
use dicom::core::{value::PixelFragmentSequence, DataElement, PrimitiveValue};
use dicom::dictionary_std::tags;
use dicom::pixeldata::{ConvertOptions, ModalityLutOption, Transcode};
use dicom::transfer_syntax::{TransferSyntaxIndex, TransferSyntaxRegistry};

const MANIFEST: &str = include_str!("../../fixtures/verification/manifest.json");
const EXPLICIT: &str = "1.2.840.10008.1.2.1";
const GENERATED: &[(&str, &str)] = &[
    ("builtin/generated-native", EXPLICIT),
    ("builtin/generated-signed", EXPLICIT),
    ("builtin/generated-rgb", EXPLICIT),
    ("builtin/generated-planar-rgb", EXPLICIT),
    ("builtin/generated-rle", "1.2.840.10008.1.2.5"),
    ("builtin/generated-jpeg", "1.2.840.10008.1.2.4.50"),
];

fn embedded(name: &str) -> Option<&'static [u8]> {
    macro_rules! assets { ($($name:literal),* $(,)?) => { match name { $(concat!($name, ".dcm") => Some(include_bytes!(concat!("../../fixtures/verification/", $name, ".dcm"))),)* _ => None } }; }
    assets!(
        "implicit",
        "explicit",
        "big-endian",
        "deflated",
        "encapsulated-native",
        "deflated-frame",
        "rle",
        "jpeg-baseline",
        "jpeg-extended",
        "jpeg-lossless",
        "jpeg-lossless-sv1",
        "j2k-lossless",
        "j2k-lossy",
        "j2k-part2-lossless",
        "j2k-part2-lossy",
        "htj2k-lossless",
        "htj2k-rpcl",
        "htj2k-lossy"
    )
}

pub(super) fn static_cases() -> Vec<VerificationCase> {
    let entries: Vec<serde_json::Value> =
        serde_json::from_str(MANIFEST).expect("checked fixture manifest");
    entries
        .into_iter()
        .map(|mut entry| {
            let name = entry.as_object_mut().unwrap().remove("file").unwrap();
            let bytes = embedded(name.as_str().unwrap()).expect("embedded fixture");
            entry["dicom_bytes"] = serde_json::json!(bytes);
            serde_json::from_value(entry).expect("checked fixture expectation")
        })
        .collect()
}

pub(super) fn case_ids() -> Vec<(String, String)> {
    static_cases()
        .into_iter()
        .map(|case| (case.id, case.transfer_syntax_uid))
        .chain(
            GENERATED
                .iter()
                .map(|(id, uid)| ((*id).into(), (*uid).into())),
        )
        .collect()
}

pub(super) fn run() -> (Vec<VerificationCaseResult>, Vec<CodecDeclaration>) {
    let static_cases = static_cases();
    let mut results: Vec<_> = static_cases
        .iter()
        .map(|case| run_case(case, "embedded_fixture"))
        .collect();
    for &(id, uid) in GENERATED {
        results.push(match generated(id, uid) {
            Ok(case) => run_case(&case, "generated_fixture"),
            Err(()) => result(
                id.into(),
                "generated_fixture",
                VerificationPath::SharedLibrary,
                Some(uid.into()),
                vec![VerificationCheck::failure(
                    "generation",
                    "DICOM generation failed",
                )],
            ),
        });
    }
    results.push(preprocessing());
    let mut required: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (id, uid) in case_ids() {
        required.entry(uid).or_default().push(id);
    }
    // Reference-only dataset syntaxes do not contain local pixel data.
    for syntax in TransferSyntaxRegistry
        .iter()
        .filter(|ts| ts.can_decode_all() && !ts.name().contains("Referenced"))
    {
        required.entry(syntax.uid().into()).or_default();
    }
    // An enabled decoder without a fixture leaves an empty requirement: coverage fails.
    (
        results,
        vec![CodecDeclaration {
            id: "builtin/decoders".into(),
            required_cases: required,
        }],
    )
}

fn put_u16(file: &mut FileDicomObject<InMemDicomObject>, tag: Tag, value: u16) {
    file.put(DataElement::new(tag, VR::US, PrimitiveValue::from(value)));
}

pub(super) fn generated(id: &str, uid: &str) -> Result<VerificationCase, ()> {
    let mut case = static_cases()
        .into_iter()
        .find(|case| case.transfer_syntax_uid == EXPLICIT)
        .ok_or(())?;
    let mut file = from_reader(Cursor::new(&case.dicom_bytes)).map_err(|_| ())?;
    let signed = id.ends_with("signed");
    let rgb = id.ends_with("rgb");
    let planar = id.ends_with("planar-rgb");
    let sample_type = if signed {
        SampleType::I16
    } else {
        SampleType::U8
    };
    let mut state = 0x4d49434fu32;
    let mut bytes = Vec::new();
    case.frames.clear();
    for frame in 0..3 {
        let values = (0..64 * if rgb { 3 } else { 1 })
            .map(|i| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                if signed {
                    (state % 60001) as i32 - 30000
                } else {
                    (i * 3 + frame * 11) % 192
                }
            })
            .collect::<Vec<_>>();
        for &value in &values {
            if signed {
                bytes.extend_from_slice(&(value as i16).to_le_bytes());
            } else {
                bytes.push(value as u8);
            }
        }
        case.frames.push(FrameExpectation {
            width: 8,
            height: 8,
            samples_per_pixel: if rgb { 3 } else { 1 },
            planar_configuration: u16::from(planar),
            sample_type,
            values,
            absolute_tolerance: if id.ends_with("jpeg") { 12 } else { 0 },
        });
    }
    put_u16(&mut file, tags::SAMPLES_PER_PIXEL, if rgb { 3 } else { 1 });
    put_u16(&mut file, tags::PLANAR_CONFIGURATION, u16::from(planar));
    put_u16(&mut file, tags::BITS_ALLOCATED, if signed { 16 } else { 8 });
    put_u16(&mut file, tags::BITS_STORED, if signed { 16 } else { 8 });
    put_u16(&mut file, tags::HIGH_BIT, if signed { 15 } else { 7 });
    put_u16(&mut file, tags::PIXEL_REPRESENTATION, u16::from(signed));
    file.put(DataElement::new(
        tags::PHOTOMETRIC_INTERPRETATION,
        VR::CS,
        if rgb { "RGB" } else { "MONOCHROME2" },
    ));
    file.put(DataElement::new(tags::NUMBER_OF_FRAMES, VR::IS, "3"));
    let pixel_value = if signed {
        PrimitiveValue::U16(
            bytes
                .chunks_exact(2)
                .map(|v| u16::from_le_bytes([v[0], v[1]]))
                .collect(),
        )
    } else {
        PrimitiveValue::from(bytes.clone())
    };
    file.put(DataElement::new(
        tags::PIXEL_DATA,
        if signed { VR::OW } else { VR::OB },
        pixel_value,
    ));
    if id.ends_with("rle") {
        let fragments = bytes
            .chunks_exact(64)
            .map(|frame| {
                let mut fragment = Vec::new();
                fragment.extend_from_slice(&1u32.to_le_bytes());
                fragment.extend_from_slice(&64u32.to_le_bytes());
                fragment.resize(64, 0);
                // Each row is one PackBits literal run. Runs cannot cross rows.
                for row in frame.chunks_exact(8) {
                    fragment.push(7);
                    fragment.extend_from_slice(row);
                }
                fragment
            })
            .collect::<Vec<_>>();
        let mut offset = 0;
        let offsets = fragments
            .iter()
            .map(|fragment| {
                let current = offset;
                offset += fragment.len() as u32 + 8;
                current
            })
            .collect::<Vec<_>>();
        file.put(DataElement::new(
            tags::PIXEL_DATA,
            VR::OB,
            PixelFragmentSequence::new(offsets, fragments),
        ));
        file.meta_mut().transfer_syntax = uid.into();
    } else if id.ends_with("jpeg") {
        {
            let mut options = dicom::encoding::adapters::EncodeOptions::default();
            options.quality = Some(100);
            file.transcode_with_options(TransferSyntaxRegistry.get(uid).ok_or(())?, options)
        }
        .map_err(|_| ())?;
    }
    case.id = id.into();
    case.transfer_syntax_uid = uid.into();
    case.dicom_bytes.clear();
    file.write_all(&mut case.dicom_bytes).map_err(|_| ())?;
    Ok(case)
}

fn preprocessing() -> VerificationCaseResult {
    let run = || -> Result<Vec<VerificationCheck>, ()> {
        let case = generated("builtin/generated-native", EXPLICIT)?;
        let file = from_reader(Cursor::new(&case.dicom_bytes)).map_err(|_| ())?;
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
        let mut first = None;
        for parallel in [false, true, false] {
            let (images, metadata, plan) = preprocessor
                .prepare_image_with_plan(&file, parallel)
                .map_err(|_| ())?;
            checks.push(VerificationCheck::new(
                format!("frame_order_{}", checks.len()),
                plan.stored_frame_order == vec![0, 1, 2],
            ));
            let mut expected_images = Vec::new();
            for frame in &case.frames {
                let mut expected = vec![0u8; 16 * 20];
                for y in 0..16 {
                    for x in 0..16 {
                        expected[(y + 2) * 16 + x] =
                            frame.values[(7 - y / 2) * 8 + (7 - x / 2)] as u8;
                    }
                }
                expected_images.push(expected);
            }
            checks.push(VerificationCheck::new(
                format!("transforms_{}", checks.len()),
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
                format!("flip_metadata_{}", checks.len()),
                metadata
                    .flip
                    .is_some_and(|flip| flip.horizontal && flip.vertical),
            ));
            let snapshot = images
                .iter()
                .map(|image| image.as_bytes().to_vec())
                .collect::<Vec<_>>();
            if let Some(first) = &first {
                checks.push(VerificationCheck::new(
                    format!("transform_repeat_{}", checks.len()),
                    first == &snapshot,
                ));
            } else {
                first = Some(snapshot);
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
    result(
        "builtin/preprocessing".into(),
        "generated_fixture",
        VerificationPath::SharedLibrary,
        Some(EXPLICIT.into()),
        run().unwrap_or_else(|()| {
            vec![VerificationCheck::failure(
                "preprocessing",
                "image conversion or preprocessing failed",
            )]
        }),
    )
}
