//! RLE Lossless reader (DICOM PS3.5 Annex G).
use dicom::encoding::adapters::{DecodeResult, PixelDataObject, PixelDataReader};

use super::{custom, fragment, ImageInfo};

/// Pixel data reader for RLE Lossless (1.2.840.10008.1.2.5).
///
/// Each frame is one fragment: a 64-byte header of little-endian segment offsets, then one
/// PackBits segment per byte of each sample, most significant byte first. Output samples are
/// interleaved, with 16-bit allocations as little-endian `u16`.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RleAdapter;

impl PixelDataReader for RleAdapter {
    fn decode_frame(
        &self,
        src: &dyn PixelDataObject,
        frame: u32,
        dst: &mut Vec<u8>,
    ) -> DecodeResult<()> {
        let info = ImageInfo::read(src, frame)?;
        let fragments = src.number_of_fragments().unwrap_or(0);
        if fragments != info.frames {
            return custom(format!(
                "RLE Lossless requires one fragment per frame, found {fragments} for {} frames",
                info.frames
            ));
        }
        let data = fragment(src, frame as usize)?;
        match decode_rle_frame(
            &data,
            usize::from(info.rows) * usize::from(info.cols),
            usize::from(info.samples_per_pixel),
            usize::from(info.bits_allocated / 8),
        ) {
            Ok(decoded) => {
                dst.extend_from_slice(&decoded);
                Ok(())
            }
            Err(message) => custom(format!("RLE decoding failed on frame {frame}: {message}")),
        }
    }
}

const HEADER_LEN: usize = 64;
const MAX_SEGMENTS: usize = 15;

/// Decodes one RLE frame of `pixels` pixels into interleaved little-endian samples.
fn decode_rle_frame(
    data: &[u8],
    pixels: usize,
    samples_per_pixel: usize,
    bytes_per_sample: usize,
) -> Result<Vec<u8>, String> {
    if data.len() < HEADER_LEN {
        return Err(format!(
            "fragment of {} bytes has no RLE header",
            data.len()
        ));
    }
    let word = |index: usize| {
        u32::from_le_bytes(data[4 * index..4 * index + 4].try_into().unwrap()) as usize
    };
    let segments = word(0);
    let expected = samples_per_pixel * bytes_per_sample;
    if segments != expected || segments > MAX_SEGMENTS {
        return Err(format!(
            "header declares {segments} segments, but the image needs {expected}"
        ));
    }
    let mut bounds: Vec<usize> = (1..=segments).map(word).collect();
    bounds.push(data.len());
    if bounds[0] != HEADER_LEN || bounds.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(format!(
            "segment offsets {:?} are not increasing within the {}-byte fragment",
            &bounds[..segments],
            data.len()
        ));
    }

    let stride = expected;
    let mut output = vec![0u8; pixels * stride];
    // A single segment is the output itself; otherwise each segment is unpacked, then scattered.
    let mut plane = if stride == 1 {
        Vec::new()
    } else {
        vec![0u8; pixels]
    };
    for segment in 0..segments {
        let source = &data[bounds[segment]..bounds[segment + 1]];
        let unpacked = if stride == 1 {
            &mut output[..]
        } else {
            &mut plane[..]
        };
        unpack_segment(source, unpacked)
            .map_err(|message| format!("segment {segment}: {message}"))?;
        if stride > 1 {
            let sample = segment / bytes_per_sample;
            // Segments hold the most significant byte first; output samples are little endian.
            let byte =
                sample * bytes_per_sample + (bytes_per_sample - 1 - segment % bytes_per_sample);
            for (pixel, &value) in output[byte..].iter_mut().step_by(stride).zip(&plane) {
                *pixel = value;
            }
        }
    }
    Ok(output)
}

/// Expands a PackBits segment until `output` is full.
///
/// Bytes beyond the output length (such as even-length padding) are ignored. A segment that ends
/// before the output is full, or a run that crosses the end of the segment, is an error.
fn unpack_segment(segment: &[u8], output: &mut [u8]) -> Result<(), String> {
    let count = output.len();
    let mut written = 0;
    let mut position = 0;
    while written < count {
        let Some(&header) = segment.get(position) else {
            return Err(format!("ended after {written} of {count} bytes"));
        };
        position += 1;
        match header as i8 {
            // Literal run of header + 1 bytes.
            0..=127 => {
                let length = usize::from(header) + 1;
                let Some(literal) = segment.get(position..position + length) else {
                    return Err("literal run crosses the end of the segment".to_owned());
                };
                let take = length.min(count - written);
                output[written..written + take].copy_from_slice(&literal[..take]);
                written += take;
                position += length;
            }
            // No operation.
            -128 => {}
            // The next byte repeated 1 - header times.
            negative => {
                let Some(&value) = segment.get(position) else {
                    return Err("replicate run crosses the end of the segment".to_owned());
                };
                position += 1;
                let take = ((1 - i16::from(negative)) as usize).min(count - written);
                output[written..written + take].fill(value);
                written += take;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Encodes one frame with each segment as literal runs.
    fn rle_frame(segments: &[Vec<u8>]) -> Vec<u8> {
        let mut frame = vec![0; HEADER_LEN];
        frame[0..4].copy_from_slice(&(segments.len() as u32).to_le_bytes());
        for (i, segment) in segments.iter().enumerate() {
            let offset = frame.len() as u32;
            frame[4 + 4 * i..8 + 4 * i].copy_from_slice(&offset.to_le_bytes());
            for run in segment.chunks(128) {
                frame.push(run.len() as u8 - 1);
                frame.extend_from_slice(run);
            }
        }
        frame
    }

    #[test]
    fn eight_bit_monochrome_keeps_sample_positions() {
        let frame = rle_frame(&[vec![1, 2, 3, 4]]);
        assert_eq!(decode_rle_frame(&frame, 4, 1, 1).unwrap(), [1, 2, 3, 4]);
    }

    #[test]
    fn sixteen_bit_monochrome_is_little_endian() {
        // Samples 0x0102 and 0x0304: the MSB segment comes first.
        let frame = rle_frame(&[vec![0x01, 0x03], vec![0x02, 0x04]]);
        assert_eq!(
            decode_rle_frame(&frame, 2, 1, 2).unwrap(),
            [0x02, 0x01, 0x04, 0x03]
        );
    }

    #[test]
    fn eight_bit_rgb_is_interleaved() {
        let frame = rle_frame(&[vec![1, 4], vec![2, 5], vec![3, 6]]);
        assert_eq!(
            decode_rle_frame(&frame, 2, 3, 1).unwrap(),
            [1, 2, 3, 4, 5, 6]
        );
    }

    #[test]
    fn sixteen_bit_rgb_is_interleaved_little_endian() {
        // Pixels (0x0102, 0x0304, 0x0506) and (0x0708, 0x090A, 0x0B0C).
        let frame = rle_frame(&[
            vec![0x01, 0x07],
            vec![0x02, 0x08],
            vec![0x03, 0x09],
            vec![0x04, 0x0A],
            vec![0x05, 0x0B],
            vec![0x06, 0x0C],
        ]);
        assert_eq!(
            decode_rle_frame(&frame, 2, 3, 2).unwrap(),
            [0x02, 0x01, 0x04, 0x03, 0x06, 0x05, 0x08, 0x07, 0x0A, 0x09, 0x0C, 0x0B]
        );
    }

    #[test]
    fn replicate_runs_no_ops_and_padding() {
        let mut frame = rle_frame(&[vec![]]);
        frame.truncate(HEADER_LEN);
        frame[4..8].copy_from_slice(&(HEADER_LEN as u32).to_le_bytes());
        // Repeat 7 three times, no-op, literal [8, 9], then a padding byte.
        frame.extend([0xFE, 7, 0x80, 0x01, 8, 9, 0x00]);
        assert_eq!(decode_rle_frame(&frame, 5, 1, 1).unwrap(), [7, 7, 7, 8, 9]);
    }

    #[test]
    fn malformed_frames_are_errors() {
        let good = rle_frame(&[vec![1, 2, 3, 4]]);
        assert!(
            decode_rle_frame(&good[..32], 4, 1, 1).is_err(),
            "short header"
        );
        assert!(decode_rle_frame(&good, 5, 1, 1).is_err(), "too few bytes");
        assert!(decode_rle_frame(&good, 4, 1, 2).is_err(), "segment count");

        let mut offset_out_of_range = good.clone();
        offset_out_of_range[4..8].copy_from_slice(&1000u32.to_le_bytes());
        assert!(decode_rle_frame(&offset_out_of_range, 4, 1, 1).is_err());

        let mut offsets_not_increasing = rle_frame(&[vec![1, 2], vec![3, 4]]);
        offsets_not_increasing[8..12].copy_from_slice(&64u32.to_le_bytes());
        assert!(decode_rle_frame(&offsets_not_increasing, 2, 1, 2).is_err());

        let mut literal_overrun = good.clone();
        literal_overrun[HEADER_LEN] = 20;
        assert!(decode_rle_frame(&literal_overrun, 4, 1, 1).is_err());

        let mut replicate_overrun = good[..HEADER_LEN].to_vec();
        replicate_overrun.push(0xFD);
        assert!(decode_rle_frame(&replicate_overrun, 4, 1, 1).is_err());

        let mut too_many_segments = good;
        too_many_segments[0..4].copy_from_slice(&16u32.to_le_bytes());
        assert!(decode_rle_frame(&too_many_segments, 4, 1, 1).is_err());
    }
}
