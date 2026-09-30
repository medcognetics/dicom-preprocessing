//! JPEG 2000 and High-Throughput JPEG 2000 reader backed by OpenJPEG, with multithreaded
//! code-block decoding.
use std::ffi::{c_char, c_void, CStr};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

use dicom::encoding::adapters::{DecodeResult, PixelDataObject, PixelDataReader};
use openjpeg_sys as sys;

use super::{custom, ImageInfo};

/// Environment variable that sets the default number of OpenJPEG threads per frame.
pub const JPEG2000_THREADS_ENV: &str = "DICOM_PREPROCESSING_JPEG2000_THREADS";

/// Upper bound of the automatic thread count. Frames and views are often decoded in parallel
/// as well, so more threads per frame mostly add contention.
const DEFAULT_MAX_THREADS: usize = 8;

/// Thread count set with [`set_jpeg2000_threads`]; 0 means not set.
static CONFIGURED_THREADS: AtomicUsize = AtomicUsize::new(0);

/// Sets the number of threads OpenJPEG uses to decode each JPEG 2000 frame.
///
/// `0` restores the default. `1` decodes each frame on the calling thread. The decoded samples
/// do not depend on the thread count.
pub fn set_jpeg2000_threads(threads: usize) {
    CONFIGURED_THREADS.store(threads, Ordering::Relaxed);
}

/// Returns the number of threads OpenJPEG uses to decode each JPEG 2000 frame.
///
/// In order of precedence: the value set with [`set_jpeg2000_threads`], a positive integer in
/// the [`JPEG2000_THREADS_ENV`] environment variable (read once), or the available parallelism
/// capped at 8.
pub fn jpeg2000_threads() -> usize {
    match CONFIGURED_THREADS.load(Ordering::Relaxed) {
        0 => *default_threads(),
        threads => threads,
    }
}

fn default_threads() -> &'static usize {
    static DEFAULT: OnceLock<usize> = OnceLock::new();
    DEFAULT.get_or_init(|| {
        std::env::var(JPEG2000_THREADS_ENV)
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|&threads| threads > 0)
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map_or(1, |threads| threads.get())
                    .min(DEFAULT_MAX_THREADS)
            })
    })
}

/// Pixel data reader for JPEG 2000 (1.2.840.10008.1.2.4.90 to .4.93) and High-Throughput
/// JPEG 2000 (.4.201 to .4.203).
///
/// Decodes with OpenJPEG using [`jpeg2000_threads`] threads per frame. Accepts raw codestreams
/// and JP2-wrapped streams. Samples are interleaved; 16-bit allocations are little-endian, and
/// signed samples keep their two's complement bit pattern.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OpenJpegAdapter;

impl PixelDataReader for OpenJpegAdapter {
    fn decode_frame(
        &self,
        src: &dyn PixelDataObject,
        frame: u32,
        dst: &mut Vec<u8>,
    ) -> DecodeResult<()> {
        let info = ImageInfo::read(src, frame)?;
        let Some(data) = src.frame_pixel_data(frame) else {
            return custom(format!("Missing pixel data for frame #{frame}"));
        };
        match decode_jpeg2000(&data, info, jpeg2000_threads()) {
            Ok(decoded) => {
                dst.extend_from_slice(&decoded);
                Ok(())
            }
            Err(message) => custom(format!(
                "JPEG 2000 decoding failed on frame {frame}: {message}"
            )),
        }
    }
}

/// Read position within a borrowed stream, owned by OpenJPEG through its user-data pointer.
struct Source {
    data: *const u8,
    len: usize,
    position: usize,
}

extern "C" fn read_source(buffer: *mut c_void, bytes: usize, source: *mut c_void) -> usize {
    // SAFETY: OpenJPEG passes back the `Source` registered with the stream, and a writable
    // buffer of `bytes` bytes.
    let source = unsafe { &mut *(source as *mut Source) };
    if source.position >= source.len {
        // OpenJPEG treats (size_t)-1 as end of stream.
        return usize::MAX;
    }
    let count = bytes.min(source.len - source.position);
    unsafe {
        std::ptr::copy_nonoverlapping(source.data.add(source.position), buffer as *mut u8, count)
    };
    source.position += count;
    count
}

extern "C" fn skip_source(bytes: i64, source: *mut c_void) -> i64 {
    // SAFETY: see `read_source`.
    let source = unsafe { &mut *(source as *mut Source) };
    let target = (source.position as i64)
        .saturating_add(bytes)
        .clamp(0, source.len as i64) as usize;
    let skipped = target as i64 - source.position as i64;
    source.position = target;
    skipped
}

extern "C" fn seek_source(offset: i64, source: *mut c_void) -> i32 {
    // SAFETY: see `read_source`.
    let source = unsafe { &mut *(source as *mut Source) };
    if offset < 0 || offset as usize > source.len {
        return 0;
    }
    source.position = offset as usize;
    1
}

extern "C" fn free_source(source: *mut c_void) {
    // SAFETY: the pointer came from `Box::into_raw` in `decode_jpeg2000`, and OpenJPEG calls
    // this exactly once when the stream is destroyed.
    drop(unsafe { Box::from_raw(source as *mut Source) })
}

extern "C" fn record_error(message: *const c_char, errors: *mut c_void) {
    if message.is_null() || errors.is_null() {
        return;
    }
    // SAFETY: `errors` is the `String` registered in `decode_jpeg2000`, which outlives the
    // codec; `message` is a NUL-terminated string from OpenJPEG.
    let errors = unsafe { &mut *(errors as *mut String) };
    let message = unsafe { CStr::from_ptr(message) }.to_string_lossy();
    if !errors.is_empty() {
        errors.push_str("; ");
    }
    errors.push_str(message.trim());
}

/// OpenJPEG objects destroyed in reverse order of creation.
struct Decoder {
    codec: *mut sys::opj_codec_t,
    stream: *mut sys::opj_stream_t,
    image: *mut sys::opj_image_t,
}

impl Drop for Decoder {
    fn drop(&mut self) {
        // SAFETY: each pointer is null or was created by OpenJPEG and is destroyed once.
        unsafe {
            if !self.image.is_null() {
                sys::opj_image_destroy(self.image);
            }
            if !self.stream.is_null() {
                sys::opj_stream_destroy(self.stream);
            }
            if !self.codec.is_null() {
                sys::opj_destroy_codec(self.codec);
            }
        }
    }
}

/// JP2 signature box (ISO/IEC 15444-1 I.5.1).
const JP2_SIGNATURE: &[u8] = &[
    0x00, 0x00, 0x00, 0x0C, 0x6A, 0x50, 0x20, 0x20, 0x0D, 0x0A, 0x87, 0x0A,
];

/// Decodes one JPEG 2000 stream into interleaved samples in the layout `dicom-pixeldata`
/// expects from encapsulated readers.
fn decode_jpeg2000(data: &[u8], info: ImageInfo, threads: usize) -> Result<Vec<u8>, String> {
    let format = if data.starts_with(JP2_SIGNATURE) {
        sys::CODEC_FORMAT::OPJ_CODEC_JP2
    } else {
        sys::CODEC_FORMAT::OPJ_CODEC_J2K
    };
    // Declared before `decoder` so it outlives the codec that writes to it.
    let mut errors = String::new();
    let failure = |errors: &str, step: &str| {
        if errors.is_empty() {
            format!("OpenJPEG {step} failed")
        } else {
            format!("OpenJPEG {step} failed: {errors}")
        }
    };
    let mut decoder = Decoder {
        // SAFETY: plain constructor; a null result is handled below.
        codec: unsafe { sys::opj_create_decompress(format) },
        stream: std::ptr::null_mut(),
        image: std::ptr::null_mut(),
    };
    if decoder.codec.is_null() {
        return Err("could not create an OpenJPEG decoder".to_owned());
    }
    // SAFETY: the codec is valid; the error string outlives it (see above); the stream reads
    // only from `data`, which outlives `decoder`; the `Source` box is freed by OpenJPEG.
    unsafe {
        sys::opj_set_error_handler(
            decoder.codec,
            Some(record_error),
            &mut errors as *mut String as *mut c_void,
        );
        let mut parameters: sys::opj_dparameters_t = std::mem::zeroed();
        sys::opj_set_default_decoder_parameters(&mut parameters);
        if sys::opj_setup_decoder(decoder.codec, &mut parameters) == 0 {
            return Err(failure(&errors, "setup"));
        }
        let threads = i32::try_from(threads).unwrap_or(i32::MAX);
        if threads > 1
            && sys::opj_has_thread_support() != 0
            && sys::opj_codec_set_threads(decoder.codec, threads) == 0
        {
            return Err(failure(&errors, "thread setup"));
        }

        decoder.stream = sys::opj_stream_default_create(1);
        if decoder.stream.is_null() {
            return Err("could not create an OpenJPEG stream".to_owned());
        }
        sys::opj_stream_set_read_function(decoder.stream, Some(read_source));
        sys::opj_stream_set_skip_function(decoder.stream, Some(skip_source));
        sys::opj_stream_set_seek_function(decoder.stream, Some(seek_source));
        sys::opj_stream_set_user_data_length(decoder.stream, data.len() as u64);
        let source = Box::into_raw(Box::new(Source {
            data: data.as_ptr(),
            len: data.len(),
            position: 0,
        }));
        sys::opj_stream_set_user_data(decoder.stream, source as *mut c_void, Some(free_source));

        if sys::opj_read_header(decoder.stream, decoder.codec, &mut decoder.image) == 0 {
            return Err(failure(&errors, "header decoding"));
        }
        if sys::opj_decode(decoder.codec, decoder.stream, decoder.image) == 0
            || sys::opj_end_decompress(decoder.codec, decoder.stream) == 0
        {
            return Err(failure(&errors, "decoding"));
        }
    }

    // SAFETY: after a successful decode, `image` and its `numcomps` components are valid, and
    // each component holds `w * h` samples.
    let image = unsafe { &*decoder.image };
    if image.comps.is_null() || image.numcomps == 0 {
        return Err("image has no components".to_owned());
    }
    let components = unsafe { std::slice::from_raw_parts(image.comps, image.numcomps as usize) };
    if components.len() != usize::from(info.samples_per_pixel) {
        return Err(format!(
            "image has {} components, but SamplesPerPixel is {}",
            components.len(),
            info.samples_per_pixel
        ));
    }
    let (cols, rows) = (u32::from(info.cols), u32::from(info.rows));
    let pixels = usize::from(info.cols) * usize::from(info.rows);
    let mut planes = Vec::with_capacity(components.len());
    for (index, component) in components.iter().enumerate() {
        if (component.w, component.h) != (cols, rows) || (component.dx, component.dy) != (1, 1) {
            return Err(format!(
                "component {index} is {}x{} with subsampling {}x{}, but Columns x Rows is \
                 {cols}x{rows}",
                component.w, component.h, component.dx, component.dy
            ));
        }
        if component.prec > u32::from(info.bits_allocated) {
            return Err(format!(
                "component {index} has {}-bit samples, but BitsAllocated is {}",
                component.prec, info.bits_allocated
            ));
        }
        if component.data.is_null() {
            return Err(format!("component {index} has no samples"));
        }
        planes.push(unsafe { std::slice::from_raw_parts(component.data, pixels) });
    }

    let bytes_per_sample = usize::from(info.bits_allocated / 8);
    let mut output = vec![0u8; pixels * planes.len() * bytes_per_sample];
    let stride = planes.len() * bytes_per_sample;
    for (index, plane) in planes.iter().enumerate() {
        let samples = output[index * bytes_per_sample..].chunks_mut(stride);
        // Truncation keeps the two's complement bit pattern of signed samples.
        match bytes_per_sample {
            1 => samples
                .zip(plane.iter())
                .for_each(|(out, &value)| out[0] = value as u8),
            _ => samples
                .zip(plane.iter())
                .for_each(|(out, &value)| out[..2].copy_from_slice(&(value as u16).to_le_bytes())),
        }
    }
    debug_assert_eq!(output.len(), info.frame_len());
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture(name: &str) -> Vec<u8> {
        std::fs::read(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("fixtures/jpeg2000")
                .join(name),
        )
        .unwrap()
    }

    /// The 37x23 source pattern of `fixtures/jpeg2000` (see its README).
    fn source_pattern() -> Vec<u16> {
        (0..23u32)
            .flat_map(|y| {
                (0..37u32).map(move |x| {
                    let v = (x * 97 + y * 53 + ((x * y) % 37) * 29) % 4096;
                    (if (x / 5 + y / 4) % 2 == 0 { v / 3 } else { v }) as u16
                })
            })
            .collect()
    }

    fn gray16() -> ImageInfo {
        ImageInfo {
            rows: 23,
            cols: 37,
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

    #[test]
    fn lossless_codestreams_are_exact_at_every_thread_count() {
        for name in [
            "j2k_lossless_37x23.j2c",
            "htj2k_lossless_37x23.j2c",
            "j2k_lossless_37x23.jp2",
        ] {
            for threads in [1, 2, 4, 8] {
                let decoded = decode_jpeg2000(&fixture(name), gray16(), threads)
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                assert_eq!(
                    as_u16(&decoded),
                    source_pattern(),
                    "{name}, {threads} threads"
                );
            }
        }
    }

    #[test]
    fn multi_component_images_are_interleaved() {
        // Components: the pattern, the pattern with rows reversed, and half the pattern.
        let pattern = source_pattern();
        let expected: Vec<u16> = (0..23 * 37)
            .flat_map(|i| {
                let (y, x) = (i / 37, i % 37);
                [pattern[i], pattern[(22 - y) * 37 + x], pattern[i] / 2]
            })
            .collect();
        let rgb = ImageInfo {
            samples_per_pixel: 3,
            ..gray16()
        };
        for threads in [1, 4] {
            let decoded =
                decode_jpeg2000(&fixture("j2k_lossless_rgb_37x23.j2c"), rgb, threads).unwrap();
            assert_eq!(as_u16(&decoded), expected);
        }
    }

    #[test]
    fn mismatched_attributes_are_errors() {
        let data = fixture("j2k_lossless_37x23.j2c");
        let wrong_size = ImageInfo {
            rows: 37,
            cols: 23,
            ..gray16()
        };
        assert!(decode_jpeg2000(&data, wrong_size, 1).is_err());
        let rgb = ImageInfo {
            samples_per_pixel: 3,
            ..gray16()
        };
        assert!(decode_jpeg2000(&data, rgb, 1).is_err());
        let narrow = ImageInfo {
            bits_allocated: 8,
            ..gray16()
        };
        assert!(decode_jpeg2000(&data, narrow, 1).is_err());
    }

    #[test]
    fn corrupt_and_truncated_streams_are_errors() {
        let data = fixture("j2k_lossless_37x23.j2c");
        assert!(decode_jpeg2000(&data[..data.len() / 2], gray16(), 1).is_err());
        assert!(decode_jpeg2000(b"not a codestream", gray16(), 1).is_err());
        assert!(decode_jpeg2000(&[], gray16(), 1).is_err());
    }

    #[test]
    fn thread_count_setting_overrides_the_default() {
        let default = jpeg2000_threads();
        assert!(
            (1..=DEFAULT_MAX_THREADS).contains(&default)
                || std::env::var_os(JPEG2000_THREADS_ENV).is_some()
        );
        set_jpeg2000_threads(3);
        assert_eq!(jpeg2000_threads(), 3);
        set_jpeg2000_threads(0);
        assert_eq!(jpeg2000_threads(), default);
    }
}
