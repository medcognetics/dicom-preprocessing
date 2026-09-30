//! Pixel data readers registered with the dicom-rs transfer syntax registry.
//!
//! dicom-rs ships JPEG and RLE readers that decode some inputs incorrectly (#128), and a JPEG 2000
//! reader that decodes each frame on a single thread. This crate builds `dicom-pixeldata` without
//! its `jpeg`, `rle`, and `openjpeg-sys` features, so those transfer syntaxes are registry stubs,
//! and submits its own readers here. A submission only replaces a stub: if any crate in the final
//! dependency graph enables `dicom-pixeldata`'s `native`, `jpeg`, `rle`, `openjpeg-sys`, or
//! `openjp2` features, the built-in readers stay in place. [`decoder_registrations`] reports which reader
//! is active, so applications can reject a build in which one is not ours.
use std::borrow::Cow;

use dicom::encoding::adapters::{decode_error, DecodeResult, PixelDataObject};
use dicom::encoding::snafu::{ensure, OptionExt};
use dicom::encoding::{submit_ele_transfer_syntax, Codec, NeverAdapter, NeverPixelAdapter};
use dicom::transfer_syntax::{TransferSyntaxIndex, TransferSyntaxRegistry};

mod jpeg;
mod jpeg2000;
mod rle;

pub use jpeg::TurboJpegAdapter;
pub use jpeg2000::{jpeg2000_threads, set_jpeg2000_threads, OpenJpegAdapter, JPEG2000_THREADS_ENV};
pub use rle::RleAdapter;

/// Marker included in the name of every transfer syntax whose reader this crate provides.
pub const DECODER_NAME_MARKER: &str = "[dicom-preprocessing]";

/// Transfer syntaxes whose readers this crate provides, with the registered names.
pub const OVERRIDDEN_TRANSFER_SYNTAXES: &[(&str, &str)] = &[
    (
        "1.2.840.10008.1.2.4.50",
        "JPEG Baseline (Process 1) [dicom-preprocessing] libjpeg-turbo",
    ),
    (
        "1.2.840.10008.1.2.4.51",
        "JPEG Extended (Process 2 & 4) [dicom-preprocessing] libjpeg-turbo",
    ),
    (
        "1.2.840.10008.1.2.4.57",
        "JPEG Lossless, Non-Hierarchical (Process 14) [dicom-preprocessing] libjpeg-turbo",
    ),
    (
        "1.2.840.10008.1.2.4.70",
        "JPEG Lossless, Non-Hierarchical, First-Order Prediction [dicom-preprocessing] libjpeg-turbo",
    ),
    ("1.2.840.10008.1.2.5", "RLE Lossless [dicom-preprocessing]"),
    (
        "1.2.840.10008.1.2.4.90",
        "JPEG 2000 Image Compression (Lossless Only) [dicom-preprocessing] OpenJPEG",
    ),
    (
        "1.2.840.10008.1.2.4.91",
        "JPEG 2000 Image Compression [dicom-preprocessing] OpenJPEG",
    ),
    (
        "1.2.840.10008.1.2.4.92",
        "JPEG 2000 Part 2 Multi-component Image Compression (Lossless Only) [dicom-preprocessing] OpenJPEG",
    ),
    (
        "1.2.840.10008.1.2.4.93",
        "JPEG 2000 Part 2 Multi-component Image Compression [dicom-preprocessing] OpenJPEG",
    ),
    (
        "1.2.840.10008.1.2.4.201",
        "High-Throughput JPEG 2000 Image Compression (Lossless Only) [dicom-preprocessing] OpenJPEG",
    ),
    (
        "1.2.840.10008.1.2.4.202",
        "High-Throughput JPEG 2000 with RPCL Options Image Compression (Lossless Only) [dicom-preprocessing] OpenJPEG",
    ),
    (
        "1.2.840.10008.1.2.4.203",
        "High-Throughput JPEG 2000 Image Compression [dicom-preprocessing] OpenJPEG",
    ),
];

type JpegCodec = Codec<NeverAdapter, TurboJpegAdapter, NeverPixelAdapter>;
type RleCodec = Codec<NeverAdapter, RleAdapter, NeverPixelAdapter>;
type Jpeg2000Codec = Codec<NeverAdapter, OpenJpegAdapter, NeverPixelAdapter>;

/// Submits the reader for `OVERRIDDEN_TRANSFER_SYNTAXES[$index]`.
macro_rules! submit {
    ($index:literal, $codec:expr) => {
        submit_ele_transfer_syntax!(
            OVERRIDDEN_TRANSFER_SYNTAXES[$index].0,
            OVERRIDDEN_TRANSFER_SYNTAXES[$index].1,
            $codec
        );
    };
}

submit!(
    0,
    JpegCodec::EncapsulatedPixelData(Some(TurboJpegAdapter), None)
);
submit!(
    1,
    JpegCodec::EncapsulatedPixelData(Some(TurboJpegAdapter), None)
);
submit!(
    2,
    JpegCodec::EncapsulatedPixelData(Some(TurboJpegAdapter), None)
);
submit!(
    3,
    JpegCodec::EncapsulatedPixelData(Some(TurboJpegAdapter), None)
);
submit!(4, RleCodec::EncapsulatedPixelData(Some(RleAdapter), None));
submit!(
    5,
    Jpeg2000Codec::EncapsulatedPixelData(Some(OpenJpegAdapter), None)
);
submit!(
    6,
    Jpeg2000Codec::EncapsulatedPixelData(Some(OpenJpegAdapter), None)
);
submit!(
    7,
    Jpeg2000Codec::EncapsulatedPixelData(Some(OpenJpegAdapter), None)
);
submit!(
    8,
    Jpeg2000Codec::EncapsulatedPixelData(Some(OpenJpegAdapter), None)
);
submit!(
    9,
    Jpeg2000Codec::EncapsulatedPixelData(Some(OpenJpegAdapter), None)
);
submit!(
    10,
    Jpeg2000Codec::EncapsulatedPixelData(Some(OpenJpegAdapter), None)
);
submit!(
    11,
    Jpeg2000Codec::EncapsulatedPixelData(Some(OpenJpegAdapter), None)
);

/// The registered reader for one overridden transfer syntax.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecoderRegistration {
    /// Transfer syntax UID.
    pub transfer_syntax_uid: &'static str,
    /// Name of the transfer syntax entry currently in the registry, if any.
    pub registered_name: Option<String>,
    /// Whether this crate's reader is the one in use.
    pub active: bool,
}

/// Reports which reader the registry uses for each transfer syntax this crate overrides.
pub fn decoder_registrations() -> Vec<DecoderRegistration> {
    OVERRIDDEN_TRANSFER_SYNTAXES
        .iter()
        .map(|&(uid, name)| {
            let registered = TransferSyntaxRegistry.get(uid);
            let registered_name = registered.map(|ts| ts.name().to_owned());
            let active = registered.is_some_and(|ts| {
                ts.name() == name && matches!(ts.codec(), Codec::EncapsulatedPixelData(Some(_), _))
            });
            DecoderRegistration {
                transfer_syntax_uid: uid,
                registered_name,
                active,
            }
        })
        .collect()
}

/// Imaging attributes every reader needs.
#[derive(Debug, Clone, Copy)]
struct ImageInfo {
    rows: u16,
    cols: u16,
    samples_per_pixel: u16,
    bits_allocated: u16,
    frames: u32,
}

impl ImageInfo {
    fn read(src: &dyn PixelDataObject, frame: u32) -> DecodeResult<Self> {
        let info = ImageInfo {
            rows: src
                .rows()
                .context(decode_error::MissingAttributeSnafu { name: "Rows" })?,
            cols: src
                .cols()
                .context(decode_error::MissingAttributeSnafu { name: "Columns" })?,
            samples_per_pixel: src.samples_per_pixel().context(
                decode_error::MissingAttributeSnafu {
                    name: "SamplesPerPixel",
                },
            )?,
            bits_allocated: src
                .bits_allocated()
                .context(decode_error::MissingAttributeSnafu {
                    name: "BitsAllocated",
                })?,
            frames: src.number_of_frames().unwrap_or(1),
        };
        ensure!(
            frame < info.frames,
            decode_error::FrameRangeOutOfBoundsSnafu
        );
        if !matches!(info.samples_per_pixel, 1 | 3) {
            return custom(format!(
                "SamplesPerPixel {} is not supported",
                info.samples_per_pixel
            ));
        }
        if !matches!(info.bits_allocated, 8 | 16) {
            return custom(format!(
                "BitsAllocated {} is not supported",
                info.bits_allocated
            ));
        }
        Ok(info)
    }

    fn frame_len(&self) -> usize {
        usize::from(self.rows)
            * usize::from(self.cols)
            * usize::from(self.samples_per_pixel)
            * usize::from(self.bits_allocated / 8)
    }
}

fn custom<T>(message: impl Into<String>) -> DecodeResult<T> {
    Err(dicom::encoding::adapters::DecodeError::Custom {
        message: message.into(),
        source: None,
    })
}

fn fragment(src: &dyn PixelDataObject, index: usize) -> DecodeResult<Cow<'_, [u8]>> {
    match src.fragment(index) {
        Some(fragment) => Ok(fragment),
        None => custom(format!("Missing pixel data fragment #{index}")),
    }
}

/// Returns the compressed bytes of one frame, following dicom-rs's JPEG adapter: one fragment
/// per frame (or a single fragment for a single frame), otherwise the Basic Offset Table.
///
/// Adapted from dicom-rs `transfer-syntax-registry/src/adapters/jpeg.rs` (MIT OR Apache-2.0).
fn frame_fragments(
    src: &dyn PixelDataObject,
    frame: u32,
    frames: u32,
) -> DecodeResult<Cow<'_, [u8]>> {
    let count = src.number_of_fragments().unwrap_or(0) as usize;
    if count == 0 {
        return custom("Pixel data has no fragments");
    }
    if count == frames as usize || (count == 1 && frames == 1) {
        return fragment(src, frame as usize);
    }
    let offsets = src.offset_table().unwrap_or_default();
    let start = match (offsets.get(frame as usize), frame) {
        (Some(&offset), _) => offset as usize,
        (None, 0) => 0,
        (None, _) => {
            return custom(format!(
                "Frame #{frame} spans several fragments but has no Basic Offset Table entry"
            ))
        }
    };
    let end = offsets
        .get(frame as usize + 1)
        .map(|&offset| offset as usize);
    let mut bytes = Vec::new();
    let mut position = 0;
    for index in 0..count {
        if end.is_some_and(|end| position >= end) {
            break;
        }
        let fragment = fragment(src, index)?;
        if position >= start {
            bytes.extend_from_slice(&fragment);
        }
        // Offsets count each fragment's 8-byte item header.
        position += fragment.len() + 8;
    }
    if bytes.is_empty() {
        return custom(format!("No pixel data fragments found for frame #{frame}"));
    }
    Ok(Cow::Owned(bytes))
}

#[cfg(test)]
mod tests {
    use dicom_pixeldata::PixelDecoder;
    use rstest::rstest;

    /// Reference sums come from an independent decoder, Thomas Richter's libjpeg (through
    /// pylibjpeg-libjpeg). Lossy decoders may differ by rounding, so lossy cases compare the
    /// mean sample value within 0.02 instead of the exact sum.
    #[rstest]
    #[case::lossless_16_bit("pydicom/JPEG-LL.dcm", None, 3_596_452, true)]
    #[case::lossless_sv1_8_bit("pydicom/JPGLosslessP14SV1_1s_1f_8b.dcm", None, 13_572_107, true)]
    #[case::extended_12_bit("pydicom/JPEG-lossy.dcm", None, 3_770_427, false)]
    #[case::baseline_rgb("pydicom/SC_rgb_jpeg_dcmtk.dcm", None, 3_832_000, false)]
    #[case::baseline_ycbcr_labelled_rgb(
        "pydicom/SC_rgb_jpeg_lossy_gdcm.dcm",
        None,
        3_826_100,
        false
    )]
    #[case::baseline_multiframe("pydicom/color3d_jpeg_baseline.dcm", Some(0), 8_810_383, false)]
    fn public_jpeg_samples_match_an_independent_decoder(
        #[case] name: &str,
        #[case] frame: Option<u32>,
        #[case] reference_sum: u64,
        #[case] lossless: bool,
    ) {
        assert!(super::decoder_registrations().iter().all(|r| r.active));
        let object = dicom::object::open_file(dicom_test_files::path(name).unwrap()).unwrap();
        let decoded = match frame {
            Some(frame) => object.decode_pixel_data_frame(frame),
            None => object.decode_pixel_data(),
        }
        .unwrap();
        let samples: Vec<u64> = match decoded.bits_allocated() {
            16 => decoded
                .data()
                .chunks_exact(2)
                .map(|pair| u64::from(u16::from_le_bytes([pair[0], pair[1]])))
                .collect(),
            _ => decoded
                .data()
                .iter()
                .map(|&sample| u64::from(sample))
                .collect(),
        };
        let sum: u64 = samples.iter().sum();
        if lossless {
            assert_eq!(sum, reference_sum, "{name}");
        } else {
            let difference = (sum as f64 - reference_sum as f64).abs() / samples.len() as f64;
            assert!(difference <= 0.02, "{name}: mean differs by {difference}");
        }
    }

    /// Lossless files, and lossy files whose reference is their own uncompressed decode, must
    /// match the uncompressed reference exactly. `MR_small` is signed 16-bit, and `emri_small`
    /// has 10 frames.
    #[rstest]
    #[case::j2k_lossless_16_bit("pydicom/693_J2KR.dcm", "pydicom/693_UNCR.dcm")]
    #[case::j2k_lossless_rgb("pydicom/US1_J2KR.dcm", "pydicom/US1_UNCR.dcm")]
    #[case::j2k_lossless_signed("pydicom/MR_small_jp2klossless.dcm", "pydicom/MR_small.dcm")]
    #[case::j2k_lossless_multiframe(
        "pydicom/emri_small_jpeg_2k_lossless.dcm",
        "pydicom/emri_small.dcm"
    )]
    #[case::j2k_lossy("pydicom/693_J2KI.dcm", "pydicom/693_UNCI.dcm")]
    #[case::j2k_lossy_single_layer("pydicom/JPEG2000.dcm", "pydicom/JPEG2000_UNC.dcm")]
    #[case::rle_signed("pydicom/MR_small_RLE.dcm", "pydicom/MR_small.dcm")]
    #[case::rle_multiframe("pydicom/emri_small_RLE.dcm", "pydicom/emri_small.dcm")]
    fn public_samples_match_uncompressed_references(#[case] name: &str, #[case] reference: &str) {
        let open = |name| dicom::object::open_file(dicom_test_files::path(name).unwrap()).unwrap();
        let (object, reference) = (open(name), open(reference));
        let decoded = object.decode_pixel_data().unwrap();
        let expected = reference.decode_pixel_data().unwrap();
        assert_eq!(
            decoded.number_of_frames(),
            expected.number_of_frames(),
            "{name}"
        );
        assert!(
            decoded.data() == expected.data(),
            "{name} differs from its reference"
        );
    }

    #[test]
    fn jpeg2000_output_does_not_depend_on_the_thread_count() {
        let object =
            dicom::object::open_file(dicom_test_files::path("pydicom/693_J2KR.dcm").unwrap())
                .unwrap();
        super::set_jpeg2000_threads(1);
        let single = object.decode_pixel_data().unwrap().data().to_vec();
        super::set_jpeg2000_threads(4);
        let multi = object.decode_pixel_data().unwrap().data().to_vec();
        super::set_jpeg2000_threads(0);
        assert!(single == multi);
    }

    #[test]
    fn this_crates_readers_are_registered() {
        for registration in super::decoder_registrations() {
            assert!(registration.active, "{registration:?}");
        }
    }

    #[test]
    fn ycbcr_stream_labelled_rgb_is_color_converted() {
        // The file's header says RGB, but its JPEG components 1, 2, 3 without JFIF or Adobe
        // markers mean YCbCr. The top-left of this test pattern is red.
        let object = dicom::object::open_file(
            dicom_test_files::path("pydicom/SC_rgb_jpeg_lossy_gdcm.dcm").unwrap(),
        )
        .unwrap();
        let decoded = object.decode_pixel_data().unwrap();
        assert_eq!(decoded.data()[..3], [254, 0, 0]);
    }
}
