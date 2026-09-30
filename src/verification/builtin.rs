use super::*;
use crate::{FlipOptions, PaddingDirection, Preprocessor};
use dicom::core::{value::PixelFragmentSequence, DataElement, PrimitiveValue};
use dicom::dictionary_std::tags;
use dicom::transfer_syntax::TransferSyntaxRegistry;
use dicom_pixeldata::{ConvertOptions, ModalityLutOption};

const MANIFEST: &str = include_str!("../../fixtures/verification/manifest.json");
const EXPLICIT: &str = "1.2.840.10008.1.2.1";
const GENERATED: &[(&str, &str)] = &[
    ("builtin/generated-native", EXPLICIT),
    ("builtin/generated-signed", EXPLICIT),
    ("builtin/generated-rgb", EXPLICIT),
    ("builtin/generated-planar-rgb", EXPLICIT),
    ("builtin/generated-rle", "1.2.840.10008.1.2.5"),
    ("builtin/generated-rle-u16", "1.2.840.10008.1.2.5"),
    ("builtin/generated-rgb-rle-u16", "1.2.840.10008.1.2.5"),
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
        "jpeg-extended-37x23",
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
    results.push(decoder_registration());
    let mut required: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (id, uid) in case_ids() {
        required.entry(uid).or_default().push(id);
    }
    for &(uid, _) in crate::codec::OVERRIDDEN_TRANSFER_SYNTAXES {
        required
            .entry(uid.into())
            .or_default()
            .push(DECODER_REGISTRATION.into());
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
    let unsigned_wide = id.ends_with("u16");
    let wide = signed || unsigned_wide;
    let rgb = id.contains("rgb");
    let planar = id.contains("planar-rgb");
    let rle = id.contains("-rle");
    let sample_type = if signed {
        SampleType::I16
    } else if unsigned_wide {
        SampleType::U16
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
                } else if unsigned_wide {
                    // Distinct high and low bytes expose swapped RLE segments.
                    ((i * 2053 + frame * 4099 + 17) * 31) % 65536
                } else {
                    (i * 3 + frame * 11) % 192
                }
            })
            .collect::<Vec<_>>();
        for &value in &values {
            if wide {
                bytes.extend_from_slice(&(value as u16).to_le_bytes());
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
    put_u16(&mut file, tags::BITS_ALLOCATED, if wide { 16 } else { 8 });
    put_u16(&mut file, tags::BITS_STORED, if wide { 16 } else { 8 });
    put_u16(&mut file, tags::HIGH_BIT, if wide { 15 } else { 7 });
    put_u16(&mut file, tags::PIXEL_REPRESENTATION, u16::from(signed));
    file.put(DataElement::new(
        tags::PHOTOMETRIC_INTERPRETATION,
        VR::CS,
        if rgb { "RGB" } else { "MONOCHROME2" },
    ));
    file.put(DataElement::new(tags::NUMBER_OF_FRAMES, VR::IS, "3"));
    let pixel_value = if wide {
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
        if wide { VR::OW } else { VR::OB },
        pixel_value,
    ));
    let fragments = if rle {
        let samples = if rgb { 3 } else { 1 };
        let bytes_per_sample = if wide { 2 } else { 1 };
        bytes
            .chunks_exact(64 * samples * bytes_per_sample)
            .map(|frame| rle_fragment(frame, samples, bytes_per_sample))
            .collect::<Vec<_>>()
    } else if id.ends_with("jpeg") {
        // Encoded independently of the libjpeg-turbo reader under test.
        bytes
            .chunks_exact(64)
            .map(|frame| {
                let mut fragment = Vec::new();
                jpeg_encoder::Encoder::new(&mut fragment, 100)
                    .encode(frame, 8, 8, jpeg_encoder::ColorType::Luma)
                    .map(|()| fragment)
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| ())?
    } else {
        Vec::new()
    };
    if !fragments.is_empty() {
        let mut offset = 0;
        let offsets = fragments
            .iter()
            .map(|fragment| {
                let current = offset;
                // Items are even-length; the encoder pads odd fragments.
                offset += fragment.len().next_multiple_of(2) as u32 + 8;
                current
            })
            .collect::<Vec<_>>();
        file.put(DataElement::new(
            tags::PIXEL_DATA,
            VR::OB,
            PixelFragmentSequence::new(offsets, fragments),
        ));
        file.meta_mut().transfer_syntax = uid.into();
    }
    case.id = id.into();
    case.transfer_syntax_uid = uid.into();
    case.dicom_bytes.clear();
    file.write_all(&mut case.dicom_bytes).map_err(|_| ())?;
    Ok(case)
}

/// Encodes one 8x8 frame of interleaved little endian samples as RLE Lossless.
///
/// Segments follow DICOM PS3.5 Annex G: each sample's most significant byte comes first.
fn rle_fragment(frame: &[u8], samples: usize, bytes_per_sample: usize) -> Vec<u8> {
    let segments = samples * bytes_per_sample;
    let mut fragment = vec![0; 64];
    fragment[..4].copy_from_slice(&(segments as u32).to_le_bytes());
    for segment in 0..segments {
        let offset = fragment.len() as u32;
        fragment[4 + 4 * segment..8 + 4 * segment].copy_from_slice(&offset.to_le_bytes());
        let byte = (segment / bytes_per_sample) * bytes_per_sample
            + (bytes_per_sample - 1 - segment % bytes_per_sample);
        let plane: Vec<u8> = frame
            .chunks_exact(segments)
            .map(|pixel| pixel[byte])
            .collect();
        // Each row is one PackBits literal run. Runs cannot cross rows.
        for row in plane.chunks_exact(8) {
            fragment.push(7);
            fragment.extend_from_slice(row);
        }
    }
    fragment
}

const DECODER_REGISTRATION: &str = "builtin/decoder-registration";

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
    result(
        DECODER_REGISTRATION.into(),
        "generated_fixture",
        VerificationPath::SharedLibrary,
        None,
        checks,
    )
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
