//! Compares this crate's JPEG and RLE readers with the dicom-rs built-in readers on the same
//! DICOM objects, decoding every frame through `PixelDataReader::decode_frame`.
use dicom_encoding::adapters::PixelDataReader;
use dicom_encoding::Codec;
use dicom_preprocessing::codec::{RleAdapter, TurboJpegAdapter};
use dicom_transfer_syntax_registry::entries;
use std::time::Instant;

fn decode_all(
    reader: &dyn PixelDataReader,
    object: &dicom_object::DefaultDicomObject,
    frames: u32,
) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    for frame in 0..frames {
        reader
            .decode_frame(object, frame, &mut out)
            .map_err(|e| e.to_string())?;
    }
    Ok(out)
}

fn time(
    reader: &dyn PixelDataReader,
    object: &dicom_object::DefaultDicomObject,
    frames: u32,
    runs: usize,
) -> Result<(f64, f64, Vec<u8>), String> {
    let output = decode_all(reader, object, frames)?; // warm-up and output check
    let mut samples: Vec<f64> = (0..runs)
        .map(|_| {
            let start = Instant::now();
            std::hint::black_box(decode_all(reader, object, frames).unwrap());
            start.elapsed().as_secs_f64() * 1000.0
        })
        .collect();
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok((samples[runs / 2], samples[0], output))
}

fn max_difference(a: &[u8], b: &[u8], wide: bool) -> Option<i32> {
    if a.len() != b.len() {
        return None;
    }
    Some(if wide {
        a.chunks(2)
            .zip(b.chunks(2))
            .map(|(x, y)| {
                (u16::from_le_bytes([x[0], x[1]]) as i32 - u16::from_ne_bytes([y[0], y[1]]) as i32)
                    .abs()
            })
            .max()
            .unwrap_or(0)
    } else {
        a.iter()
            .zip(b)
            .map(|(x, y)| (*x as i32 - *y as i32).abs())
            .max()
            .unwrap_or(0)
    })
}

fn main() {
    let dir = std::env::args().nth(1).expect("data directory");
    let runs: usize = std::env::args()
        .nth(2)
        .map(|r| r.parse().unwrap())
        .unwrap_or(9);
    let Codec::EncapsulatedPixelData(Some(builtin_jpeg), _) = entries::JPEG_BASELINE.codec() else {
        panic!()
    };
    let Codec::EncapsulatedPixelData(Some(builtin_rle), _) = entries::RLE_LOSSLESS.codec() else {
        panic!()
    };
    if std::env::var_os("CHECKSUMS").is_some() {
        let mut names: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();
        names.sort();
        for path in names {
            let object = dicom_object::open_file(&path).unwrap();
            let ts = object
                .meta()
                .transfer_syntax()
                .trim_end_matches('\0')
                .to_string();
            let frames: u32 = object
                .element_by_name("NumberOfFrames")
                .ok()
                .and_then(|e| e.to_int().ok())
                .unwrap_or(1);
            let ours: &dyn PixelDataReader = if ts == "1.2.840.10008.1.2.5" {
                &RleAdapter
            } else {
                &TurboJpegAdapter
            };
            let out = decode_all(ours, &object, frames).unwrap();
            let hash = out.iter().fold(0xcbf29ce484222325u64, |h, &b| {
                (h ^ b as u64).wrapping_mul(0x100000001b3)
            });
            println!(
                "{} {:016x}",
                path.file_stem().unwrap().to_string_lossy(),
                hash
            );
        }
        return;
    }
    let mut names: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    names.sort();
    println!("| File | Frames | Built-in median (min) ms | Ours median (min) ms | Ours / built-in | Max sample difference |");
    println!("| --- | --- | --- | --- | --- | --- |");
    for path in names {
        let object = dicom_object::open_file(&path).unwrap();
        let ts = object
            .meta()
            .transfer_syntax()
            .trim_end_matches('\0')
            .to_string();
        let frames: u32 = object
            .element_by_name("NumberOfFrames")
            .ok()
            .and_then(|e| e.to_int().ok())
            .unwrap_or(1);
        let wide = object
            .element_by_name("BitsAllocated")
            .unwrap()
            .to_int::<u16>()
            .unwrap()
            == 16;
        let (builtin, ours): (&dyn PixelDataReader, &dyn PixelDataReader) =
            if ts == "1.2.840.10008.1.2.5" {
                (builtin_rle, &RleAdapter)
            } else {
                (builtin_jpeg, &TurboJpegAdapter)
            };
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        let b = time(builtin, &object, frames, runs);
        let o = time(ours, &object, frames, runs).expect("our reader failed");
        match b {
            Ok((bm, bmin, bout)) => {
                let diff = max_difference(&o.2, &bout, wide)
                    .map(|d| d.to_string())
                    .unwrap_or("length differs".into());
                println!(
                    "| {name} | {frames} | {bm:.1} ({bmin:.1}) | {:.1} ({:.1}) | {:.2}x | {diff} |",
                    o.0,
                    o.1,
                    o.0 / bm
                );
            }
            Err(e) => println!(
                "| {name} | {frames} | fails: {e} | {:.1} ({:.1}) | n/a | n/a |",
                o.0, o.1
            ),
        }
    }
}
