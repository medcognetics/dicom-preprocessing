//! JPEG Baseline, Extended, and Lossless reader backed by libjpeg-turbo 3.
use std::borrow::Cow;
use std::ffi::{c_int, CStr};

use dicom::encoding::adapters::{DecodeResult, PixelDataObject, PixelDataReader};
use turbojpeg_sys as sys;

use super::{custom, frame_fragments, ImageInfo};

/// Pixel data reader for JPEG Baseline (1.2.840.10008.1.2.4.50), Extended (.4.51), and
/// Lossless (.4.57, .4.70) transfer syntaxes.
///
/// Decodes 8-bit and 12-bit DCT images and 2-16-bit lossless images with libjpeg-turbo's
/// accurate integer IDCT. Three-component images are converted to interleaved RGB, which
/// `dicom-pixeldata` expects from every encapsulated reader.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TurboJpegAdapter;

impl PixelDataReader for TurboJpegAdapter {
    fn decode_frame(
        &self,
        src: &dyn PixelDataObject,
        frame: u32,
        dst: &mut Vec<u8>,
    ) -> DecodeResult<()> {
        let info = ImageInfo::read(src, frame)?;
        let data = frame_fragments(src, frame, info.frames)?;
        match decode_jpeg(&data, info) {
            Ok(decoded) => {
                dst.extend_from_slice(&decoded);
                Ok(())
            }
            Err(message) => custom(format!("JPEG decoding failed on frame {frame}: {message}")),
        }
    }
}

/// TurboJPEG functions that take `size_t`.
///
/// The pregenerated `turbojpeg-sys` bindings declare `size_t` as `c_ulong`, which is 32 bits on
/// Windows. These declarations use `usize`, which matches `size_t` on every supported target.
mod ffi {
    use std::ffi::{c_int, c_short, c_uchar, c_ushort};

    use turbojpeg_sys::tjhandle;

    #[allow(clashing_extern_declarations)]
    extern "C" {
        pub fn tj3DecompressHeader(handle: tjhandle, jpeg: *const c_uchar, size: usize) -> c_int;
        pub fn tj3Decompress8(
            handle: tjhandle,
            jpeg: *const c_uchar,
            size: usize,
            dst: *mut c_uchar,
            pitch: c_int,
            pixel_format: c_int,
        ) -> c_int;
        pub fn tj3Decompress12(
            handle: tjhandle,
            jpeg: *const c_uchar,
            size: usize,
            dst: *mut c_short,
            pitch: c_int,
            pixel_format: c_int,
        ) -> c_int;
        pub fn tj3Decompress16(
            handle: tjhandle,
            jpeg: *const c_uchar,
            size: usize,
            dst: *mut c_ushort,
            pitch: c_int,
            pixel_format: c_int,
        ) -> c_int;
    }
}

/// Owned TurboJPEG decompressor handle.
struct Decompressor(sys::tjhandle);

impl Decompressor {
    fn new() -> Result<Self, String> {
        // SAFETY: tj3Init has no preconditions; a null result is handled below.
        let handle = unsafe { sys::tj3Init(sys::TJINIT_TJINIT_DECOMPRESS as c_int) };
        if handle.is_null() {
            return Err(error_message(std::ptr::null_mut()));
        }
        let decompressor = Decompressor(handle);
        // Treat corrupt-data warnings as errors instead of returning partial images.
        decompressor.set(sys::TJPARAM_TJPARAM_STOPONWARNING, 1)?;
        // Bound the work a malicious progressive image can cause.
        decompressor.set(sys::TJPARAM_TJPARAM_SCANLIMIT, 500)?;
        Ok(decompressor)
    }

    fn set(&self, param: sys::TJPARAM, value: c_int) -> Result<(), String> {
        // SAFETY: the handle is valid for the lifetime of `self`.
        match unsafe { sys::tj3Set(self.0, param as c_int, value) } {
            0 => Ok(()),
            _ => Err(error_message(self.0)),
        }
    }

    fn get(&self, param: sys::TJPARAM) -> c_int {
        // SAFETY: the handle is valid for the lifetime of `self`.
        unsafe { sys::tj3Get(self.0, param as c_int) }
    }

    fn check(&self, status: c_int) -> Result<(), String> {
        match status {
            0 => Ok(()),
            _ => Err(error_message(self.0)),
        }
    }
}

impl Drop for Decompressor {
    fn drop(&mut self) {
        // SAFETY: the handle came from tj3Init and is destroyed exactly once.
        unsafe { sys::tj3Destroy(self.0) }
    }
}

fn error_message(handle: sys::tjhandle) -> String {
    // SAFETY: tj3GetErrorStr returns a NUL-terminated string owned by TurboJPEG (per handle, or
    // thread-local for a null handle). It is copied before any further TurboJPEG call.
    unsafe {
        let message = sys::tj3GetErrorStr(handle);
        if message.is_null() {
            "unknown libjpeg-turbo error".to_owned()
        } else {
            CStr::from_ptr(message).to_string_lossy().into_owned()
        }
    }
}

/// Decodes one JPEG stream into the sample layout `dicom-pixeldata` expects: interleaved
/// samples, with 16-bit allocations as little-endian `u16`.
fn decode_jpeg(data: &[u8], info: ImageInfo) -> Result<Vec<u8>, String> {
    let data = normalize_sequential_scans(data);
    let data = &*data;
    let decompressor = Decompressor::new()?;
    // SAFETY: `data` is valid for `data.len()` bytes during the call.
    decompressor
        .check(unsafe { ffi::tj3DecompressHeader(decompressor.0, data.as_ptr(), data.len()) })?;

    let width = decompressor.get(sys::TJPARAM_TJPARAM_JPEGWIDTH);
    let height = decompressor.get(sys::TJPARAM_TJPARAM_JPEGHEIGHT);
    if (width, height) != (c_int::from(info.cols), c_int::from(info.rows)) {
        return Err(format!(
            "image is {width}x{height}, but Columns x Rows is {}x{}",
            info.cols, info.rows
        ));
    }
    let components = match decompressor.get(sys::TJPARAM_TJPARAM_COLORSPACE) as sys::TJCS {
        sys::TJCS_TJCS_GRAY => 1,
        sys::TJCS_TJCS_RGB | sys::TJCS_TJCS_YCbCr => 3,
        colorspace => return Err(format!("unsupported JPEG colorspace {colorspace}")),
    };
    if components != info.samples_per_pixel {
        return Err(format!(
            "image has {components} components, but SamplesPerPixel is {}",
            info.samples_per_pixel
        ));
    }
    let pixel_format = if components == 1 {
        sys::TJPF_TJPF_GRAY
    } else {
        sys::TJPF_TJPF_RGB
    };
    let samples = usize::from(info.rows) * usize::from(info.cols) * usize::from(components);
    let precision = decompressor.get(sys::TJPARAM_TJPARAM_PRECISION);

    let decoded = match precision {
        2..=8 => {
            let mut buffer = vec![0u8; samples];
            // SAFETY: `buffer` holds width * height * components samples, the size TurboJPEG
            // writes with pitch 0 for this pixel format.
            decompressor.check(unsafe {
                ffi::tj3Decompress8(
                    decompressor.0,
                    data.as_ptr(),
                    data.len(),
                    buffer.as_mut_ptr(),
                    0,
                    pixel_format,
                )
            })?;
            match info.bits_allocated {
                8 => buffer,
                _ => buffer
                    .iter()
                    .flat_map(|&sample| u16::from(sample).to_le_bytes())
                    .collect(),
            }
        }
        9..=12 => {
            if info.bits_allocated != 16 {
                return Err(format!(
                    "{precision}-bit JPEG requires BitsAllocated 16, not {}",
                    info.bits_allocated
                ));
            }
            let mut buffer = vec![0i16; samples];
            // SAFETY: as above, with 16-bit sample storage.
            decompressor.check(unsafe {
                ffi::tj3Decompress12(
                    decompressor.0,
                    data.as_ptr(),
                    data.len(),
                    buffer.as_mut_ptr(),
                    0,
                    pixel_format,
                )
            })?;
            // 12-bit samples are non-negative, so reinterpreting them as u16 is lossless.
            buffer
                .iter()
                .flat_map(|&sample| (sample as u16).to_le_bytes())
                .collect()
        }
        13..=16 => {
            if info.bits_allocated != 16 {
                return Err(format!(
                    "{precision}-bit JPEG requires BitsAllocated 16, not {}",
                    info.bits_allocated
                ));
            }
            let mut buffer = vec![0u16; samples];
            // SAFETY: as above, with 16-bit sample storage.
            decompressor.check(unsafe {
                ffi::tj3Decompress16(
                    decompressor.0,
                    data.as_ptr(),
                    data.len(),
                    buffer.as_mut_ptr(),
                    0,
                    pixel_format,
                )
            })?;
            buffer
                .iter()
                .flat_map(|&sample| sample.to_le_bytes())
                .collect()
        }
        _ => return Err(format!("unsupported JPEG precision {precision}")),
    };
    debug_assert_eq!(decoded.len(), info.frame_len());
    Ok(decoded)
}

/// Sets the spectral selection and successive approximation fields of sequential DCT scans
/// (SOF0, SOF1) to the values T.81 requires (Ss 0, Se 63, Ah/Al 0).
///
/// Some encoders write zeroes there. libjpeg-turbo ignores these fields for sequential scans
/// but reports `JWRN_NOT_SEQUENTIAL`, which the decompressor treats as fatal because other
/// warnings signal corrupt data. Normalizing them keeps every other warning fatal without
/// changing the decoded samples.
fn normalize_sequential_scans(data: &[u8]) -> Cow<'_, [u8]> {
    let mut data = Cow::Borrowed(data);
    let mut sequential = false;
    // Walk marker segments by their declared lengths, so bytes inside APPn, COM, or table
    // payloads are never mistaken for markers. Malformed streams are left to libjpeg-turbo.
    if !data.starts_with(&[0xFF, 0xD8]) {
        return data;
    }
    let mut position = 2;
    loop {
        // Skip fill bytes before a marker (T.81 B.1.1.2).
        while data.get(position) == Some(&0xFF) && data.get(position + 1) == Some(&0xFF) {
            position += 1;
        }
        let (Some(&0xFF), Some(&marker)) = (data.get(position), data.get(position + 1)) else {
            return data;
        };
        position += 2;
        match marker {
            // Markers without a segment.
            0x01 | 0xD0..=0xD8 => continue,
            0xD9 => return data,
            _ => {}
        }
        let Some(length) = data
            .get(position..position + 2)
            .map(|bytes| usize::from(u16::from_be_bytes([bytes[0], bytes[1]])))
            .filter(|&length| length >= 2 && position + length <= data.len())
        else {
            return data;
        };
        let segment = position + 2..position + length;
        position += length;
        match marker {
            // SOFn; 0xC4 (DHT), 0xC8 (JPG), and 0xCC (DAC) share the range. Only sequential
            // DCT frames are normalized: progressive and lossless scans use these fields.
            0xC0..=0xCF if ![0xC4, 0xC8, 0xCC].contains(&marker) => {
                if sequential || !matches!(marker, 0xC0 | 0xC1) {
                    return data;
                }
                sequential = true;
            }
            0xDA => {
                if !sequential || segment.is_empty() {
                    return data;
                }
                // Ns, then Ns component selectors of 2 bytes, then Ss, Se, and Ah/Al.
                let components = usize::from(data[segment.start]);
                let fields = segment.start + 1 + 2 * components;
                if fields + 3 != segment.end {
                    return data;
                }
                if data[fields..fields + 3] != [0, 63, 0] {
                    data.to_mut()[fields..fields + 3].copy_from_slice(&[0, 63, 0]);
                }
                // Skip entropy-coded data up to the next marker. Inside it, 0xFF is followed by
                // 0x00 (stuffing) or a restart marker.
                loop {
                    match (data.get(position), data.get(position + 1)) {
                        (Some(&0xFF), Some(&next))
                            if next != 0x00 && !(0xD0..=0xD7).contains(&next) =>
                        {
                            break
                        }
                        (Some(_), Some(_)) => position += 1,
                        _ => return data,
                    }
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn gray16(rows: u16, cols: u16) -> ImageInfo {
        ImageInfo {
            rows,
            cols,
            samples_per_pixel: 1,
            bits_allocated: 16,
            frames: 1,
        }
    }

    fn as_u16(bytes: &[u8]) -> Vec<u16> {
        bytes
            .chunks_exact(2)
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
            .collect()
    }

    /// Minimal 16-bit lossless (SOF3, predictor 1) stream with an optional restart interval.
    ///
    /// Prediction follows ITU-T T.81 H.1.2.1: the first sample of each restart interval uses
    /// 2^(P-1), the rest of the interval's first row uses Ra, and each later row starts with Rb.
    fn lossless_stream(values: &[u16], width: usize, restart: usize) -> Vec<u8> {
        fn segment(marker: u8, payload: &[u8]) -> Vec<u8> {
            let mut bytes = vec![0xFF, marker];
            bytes.extend_from_slice(&(payload.len() as u16 + 2).to_be_bytes());
            bytes.extend_from_slice(payload);
            bytes
        }
        let height = values.len() / width;
        let mut data = vec![0xFF, 0xD8];
        // DC table: code lengths of 5 bits for categories 0-16.
        let mut dht = vec![0x00, 0, 0, 0, 0, 17];
        dht.extend([0; 11]);
        dht.extend(0..17u8);
        data.extend(segment(0xC4, &dht));
        let mut sof = vec![16];
        sof.extend((height as u16).to_be_bytes());
        sof.extend((width as u16).to_be_bytes());
        sof.extend([1, 1, 0x11, 0]);
        data.extend(segment(0xC3, &sof));
        if restart > 0 {
            data.extend(segment(0xDD, &(restart as u16).to_be_bytes()));
        }
        data.extend(segment(0xDA, &[1, 1, 0, 1, 0, 0]));
        let interval = if restart > 0 { restart } else { values.len() };
        for (number, start) in (0..values.len()).step_by(interval).enumerate() {
            let end = (start + interval).min(values.len());
            let mut bits = String::new();
            for i in start..end {
                let prediction = if i == start {
                    32768
                } else if i / width == start / width {
                    i32::from(values[i - 1])
                } else if i % width == 0 {
                    i32::from(values[i - width])
                } else {
                    i32::from(values[i - 1])
                };
                let difference = i32::from(values[i]) - prediction;
                let size = 32 - difference.unsigned_abs().leading_zeros();
                bits += &format!("{size:05b}");
                if size > 0 {
                    let magnitude = if difference >= 0 {
                        difference
                    } else {
                        difference + (1 << size) - 1
                    };
                    bits += &format!("{magnitude:0width$b}", width = size as usize);
                }
            }
            while !bits.len().is_multiple_of(8) {
                bits.push('1');
            }
            for byte in bits.as_bytes().chunks(8) {
                let byte = u8::from_str_radix(std::str::from_utf8(byte).unwrap(), 2).unwrap();
                data.push(byte);
                if byte == 0xFF {
                    data.push(0);
                }
            }
            if end < values.len() {
                data.extend([0xFF, 0xD0 + (number % 8) as u8]);
            }
        }
        data.extend([0xFF, 0xD9]);
        data
    }

    const LOSSLESS_VALUES: [u16; 12] = [
        32768, 32770, 32774, 32775, 32772, 32771, 32768, 32777, 32780, 32775, 32776, 32781,
    ];

    #[test]
    fn lossless_restart_at_each_row_is_exact() {
        let data = lossless_stream(&LOSSLESS_VALUES, 4, 4);
        let decoded = decode_jpeg(&data, gray16(3, 4)).unwrap();
        assert_eq!(as_u16(&decoded), LOSSLESS_VALUES);
    }

    #[test]
    fn lossless_without_restart_is_exact() {
        let data = lossless_stream(&LOSSLESS_VALUES, 4, 0);
        let decoded = decode_jpeg(&data, gray16(3, 4)).unwrap();
        assert_eq!(as_u16(&decoded), LOSSLESS_VALUES);
    }

    #[test]
    fn lossless_restart_inside_a_row_is_an_error() {
        // T.81 requires lossless restart intervals to be whole MCU rows; libjpeg-turbo rejects
        // others (JERR_BAD_RESTART) instead of guessing at prediction. The bundled 3.1.0 build
        // reports that error with the text of the preceding message table entry, so only the
        // rejection is asserted.
        let data = lossless_stream(&LOSSLESS_VALUES, 4, 3);
        assert!(decode_jpeg(&data, gray16(3, 4)).is_err());
    }

    #[test]
    fn sequential_scans_with_zero_spectral_selection_are_normalized() {
        // A minimal SOF1 header and SOS with Ss = Se = Ah/Al = 0, as some encoders write.
        let mut stream = vec![0xFF, 0xD8, 0xFF, 0xC1, 0, 11, 12, 0, 1, 0, 1, 1, 1, 0x11, 0];
        stream.extend([0xFF, 0xDA, 0, 8, 1, 1, 0, 0, 0, 0, 0x12, 0xFF, 0x00, 0x34]);
        let normalized = normalize_sequential_scans(&stream);
        let sos = stream.len() - 7;
        assert_eq!(normalized[sos..sos + 3], [0, 63, 0]);
        assert_eq!(normalized[..sos], stream[..sos]);
        assert_eq!(normalized[sos + 3..], stream[sos + 3..]);

        // Lossless scans use these fields for the predictor and point transform.
        let lossless = lossless_stream(&LOSSLESS_VALUES, 4, 4);
        assert!(matches!(
            normalize_sequential_scans(&lossless),
            Cow::Borrowed(_)
        ));
    }

    /// Inserts an APP15 segment right after SOI.
    fn with_app_segment(stream: &[u8], payload: &[u8]) -> Vec<u8> {
        let mut result = stream[..2].to_vec();
        result.extend([0xFF, 0xEF]);
        result.extend((payload.len() as u16 + 2).to_be_bytes());
        result.extend_from_slice(payload);
        result.extend_from_slice(&stream[2..]);
        result
    }

    /// SOS look-alike: Ns 1, selector 1/0, then Ss 5, Se 6, Ah/Al 7.
    const MARKER_LOOKALIKES: [u8; 12] = [
        0xFF, 0xC0, 0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x05, 0x06, 0x07,
    ];

    #[test]
    fn marker_bytes_inside_segment_payloads_are_ignored() {
        // SOF0 and SOS byte patterns inside an APP payload must not make a lossless stream look
        // sequential, and its predictor fields must stay untouched.
        let lossless =
            with_app_segment(&lossless_stream(&LOSSLESS_VALUES, 4, 4), &MARKER_LOOKALIKES);
        assert!(matches!(
            normalize_sequential_scans(&lossless),
            Cow::Borrowed(_)
        ));
        let decoded = decode_jpeg(&lossless, gray16(3, 4)).unwrap();
        assert_eq!(as_u16(&decoded), LOSSLESS_VALUES);

        // In a sequential stream, only the real SOS changes, not the look-alike in the payload.
        let mut sequential = vec![0xFF, 0xD8, 0xFF, 0xC1, 0, 11, 12, 0, 1, 0, 1, 1, 1, 0x11, 0];
        let sos = sequential.len();
        sequential.extend([0xFF, 0xDA, 0, 8, 1, 1, 0, 0, 0, 0]);
        sequential.extend([0x12, 0xFF, 0x00, 0x34, 0xFF, 0xD9]);
        let with_app = with_app_segment(&sequential, &MARKER_LOOKALIKES);
        let offset = 4 + MARKER_LOOKALIKES.len();
        let mut expected = with_app.clone();
        expected[offset + sos + 7..offset + sos + 10].copy_from_slice(&[0, 63, 0]);
        assert_eq!(*normalize_sequential_scans(&with_app), expected[..]);
    }

    #[test]
    fn every_scan_of_a_multi_scan_sequential_stream_is_normalized() {
        // Two scans separated by entropy data with a stuffed 0xFF and a restart marker.
        let mut stream = vec![0xFF, 0xD8];
        stream.extend([0xFF, 0xC0, 0, 14, 8, 0, 1, 0, 1, 2, 1, 0x11, 0, 2, 0x11, 0]);
        let first = stream.len();
        stream.extend([0xFF, 0xDA, 0, 8, 1, 1, 0, 0, 0, 0]);
        stream.extend([0x12, 0xFF, 0x00, 0x34, 0xFF, 0xD0, 0x56]);
        let second = stream.len();
        stream.extend([0xFF, 0xDA, 0, 8, 1, 2, 0, 1, 2, 3, 0x78, 0xFF, 0xD9]);
        let mut expected = stream.clone();
        expected[first + 7..first + 10].copy_from_slice(&[0, 63, 0]);
        expected[second + 7..second + 10].copy_from_slice(&[0, 63, 0]);
        assert_eq!(*normalize_sequential_scans(&stream), expected[..]);
    }

    #[test]
    fn truncated_segments_are_left_unchanged() {
        let mut stream = vec![0xFF, 0xD8, 0xFF, 0xC1, 0, 11, 12, 0, 1, 0, 1, 1, 1, 0x11, 0];
        stream.extend([0xFF, 0xDA, 0, 8, 1, 1, 0, 0]);
        assert!(matches!(
            normalize_sequential_scans(&stream),
            Cow::Borrowed(_)
        ));
        assert!(matches!(
            normalize_sequential_scans(&[0xFF, 0xD8, 0xFF, 0xE0, 0x10, 0x00]),
            Cow::Borrowed(_)
        ));
    }

    #[test]
    fn mismatched_attributes_are_errors() {
        let data = lossless_stream(&LOSSLESS_VALUES, 4, 4);
        assert!(decode_jpeg(&data, gray16(4, 3)).is_err());
        let rgb = ImageInfo {
            samples_per_pixel: 3,
            ..gray16(3, 4)
        };
        assert!(decode_jpeg(&data, rgb).is_err());
        let narrow = ImageInfo {
            bits_allocated: 8,
            ..gray16(3, 4)
        };
        assert!(decode_jpeg(&data, narrow).is_err());
    }

    #[test]
    fn corrupt_and_truncated_streams_are_errors() {
        let data = lossless_stream(&LOSSLESS_VALUES, 4, 4);
        assert!(decode_jpeg(&data[..data.len() / 2], gray16(3, 4)).is_err());
        assert!(decode_jpeg(b"not a jpeg", gray16(3, 4)).is_err());
        assert!(decode_jpeg(&[], gray16(3, 4)).is_err());
    }

    fn png_u16(path: &std::path::Path) -> (u32, u32, Vec<u16>) {
        let image = image::open(path).unwrap().into_luma16();
        (image.width(), image.height(), image.into_raw())
    }

    #[test]
    fn extended_12_bit_matches_libjpeg_turbo_references() {
        let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/jpeg12");
        let mut checked = 0;
        for entry in std::fs::read_dir(&directory).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_none_or(|extension| extension != "jpg") {
                continue;
            }
            let (width, height, expected) = png_u16(&path.with_extension("png"));
            let decoded = decode_jpeg(
                &std::fs::read(&path).unwrap(),
                gray16(height as u16, width as u16),
            )
            .unwrap_or_else(|error| panic!("{path:?}: {error}"));
            assert_eq!(as_u16(&decoded), expected, "{path:?}");
            checked += 1;
        }
        assert_eq!(checked, 6);
    }

    #[test]
    fn extended_12_bit_color_decodes_to_rgb() {
        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/jpeg12/color/rgb_37x23.jpg");
        let info = ImageInfo {
            rows: 23,
            cols: 37,
            samples_per_pixel: 3,
            bits_allocated: 16,
            frames: 1,
        };
        let decoded = decode_jpeg(&std::fs::read(path).unwrap(), info).unwrap();
        assert_eq!(decoded.len(), 23 * 37 * 3 * 2);
        assert!(as_u16(&decoded).iter().all(|&sample| sample <= 4095));
    }
}
