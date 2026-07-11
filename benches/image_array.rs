use criterion::{BatchSize, BenchmarkId, Criterion, Throughput};
use dicom_preprocessing::{
    images_to_array, DicomColorType, FrameCount, LoadFromTiff, PreprocessingMetadata, TiffSaver,
};
use image::{DynamicImage, ImageBuffer, Luma, Rgb};
use ndarray::Array4;
use num::Zero;
use std::hint::black_box;
use std::io::{Seek, SeekFrom};
use std::time::Duration;
use tempfile::spooled_tempfile;
use tiff::decoder::Decoder;
use tiff::encoder::compression::{Compressor, Uncompressed};
use tiff::encoder::TiffEncoder;

const FRAME_COUNTS: [usize; 2] = [1, 32];
const DEFAULT_WIDTH: u32 = 512;
const DEFAULT_HEIGHT: u32 = 512;
const HIGH_RES_WIDTH: u32 = 2048;
const HIGH_RES_HEIGHT: u32 = 1536;
const SPOOL_SIZE: usize = 64 * 1024 * 1024;
const HIGH_RES_BENCH_ENV: &str = "DICOM_BENCH_HIGH_RES";
const SAMPLE_SIZE: usize = 10;

fn metadata(frame_count: usize) -> PreprocessingMetadata {
    PreprocessingMetadata {
        flip: None,
        crop: None,
        resize: None,
        padding: None,
        resolution: None,
        num_frames: FrameCount::from(frame_count),
    }
}

fn tiff_round_trip<T>(images: &[DynamicImage], color_type: DicomColorType) -> (Array4<T>, u64, bool)
where
    T: Clone + Zero,
    Array4<T>: LoadFromTiff<T>,
{
    let saver = TiffSaver::new(Compressor::Uncompressed(Uncompressed), color_type);
    let mut buffer = spooled_tempfile(SPOOL_SIZE);
    {
        let mut encoder = TiffEncoder::new(&mut buffer).unwrap();
        let metadata = metadata(images.len());
        for image in images {
            saver.save(&mut encoder, image, &metadata).unwrap();
        }
    }
    let bytes_written = buffer.stream_position().unwrap();
    let spilled_to_disk = buffer.is_rolled();
    buffer.seek(SeekFrom::Start(0)).unwrap();
    let mut decoder = Decoder::new(buffer).unwrap();
    let array = Array4::<T>::decode(&mut decoder).unwrap();
    (array, bytes_written, spilled_to_disk)
}

fn luma8_frames(frame_count: usize, width: u32, height: u32) -> Vec<DynamicImage> {
    (0..frame_count)
        .map(|frame| {
            DynamicImage::ImageLuma8(ImageBuffer::from_fn(width, height, |x, y| {
                Luma([((x as usize + y as usize + frame) % (u8::MAX as usize + 1)) as u8])
            }))
        })
        .collect()
}

fn luma16_frames(frame_count: usize, width: u32, height: u32) -> Vec<DynamicImage> {
    (0..frame_count)
        .map(|frame| {
            DynamicImage::ImageLuma16(ImageBuffer::from_fn(width, height, |x, y| {
                Luma([((x as usize * 31 + y as usize * 17 + frame) % u16::MAX as usize) as u16])
            }))
        })
        .collect()
}

fn rgb8_frames(frame_count: usize, width: u32, height: u32) -> Vec<DynamicImage> {
    (0..frame_count)
        .map(|frame| {
            DynamicImage::ImageRgb8(ImageBuffer::from_fn(width, height, |x, y| {
                Rgb([
                    ((x as usize + frame) % (u8::MAX as usize + 1)) as u8,
                    ((y as usize + frame) % (u8::MAX as usize + 1)) as u8,
                    ((x as usize + y as usize + frame) % (u8::MAX as usize + 1)) as u8,
                ])
            }))
        })
        .collect()
}

fn benchmark_case<T>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    name: &str,
    images: &[DynamicImage],
    color_type: DicomColorType,
) where
    T: Clone + Zero,
    Array4<T>: LoadFromTiff<T>,
{
    let (_, bytes_written, spilled_to_disk) = tiff_round_trip::<T>(images, color_type.clone());
    let elements = images
        .iter()
        .map(|image| {
            u64::from(image.width())
                * u64::from(image.height())
                * u64::from(image.color().channel_count())
        })
        .sum::<u64>();
    let raw_bytes = images
        .iter()
        .map(DynamicImage::as_bytes)
        .map(<[u8]>::len)
        .sum::<usize>();
    eprintln!(
        "{name}: elements={elements}, raw_bytes={raw_bytes}, tiff_bytes={bytes_written}, spilled_to_disk={spilled_to_disk}",
    );
    group.throughput(Throughput::Elements(elements));
    group.bench_with_input(
        BenchmarkId::new("tiff_round_trip", name),
        images,
        |b, images| {
            b.iter(|| black_box(tiff_round_trip::<T>(black_box(images), color_type.clone()).0))
        },
    );
    group.bench_with_input(
        BenchmarkId::new("direct_array", name),
        images,
        |b, images| {
            b.iter_batched(
                || images.to_vec(),
                |images| black_box(images_to_array::<T>(black_box(images)).unwrap()),
                BatchSize::LargeInput,
            )
        },
    );
}

fn benchmark_dimensions(c: &mut Criterion, width: u32, height: u32) {
    let mut group = c.benchmark_group("image-array");
    group.sample_size(SAMPLE_SIZE);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));

    for frame_count in FRAME_COUNTS {
        let suffix = format!("{frame_count}x{width}x{height}");
        benchmark_case::<u8>(
            &mut group,
            &format!("luma8/{suffix}"),
            &luma8_frames(frame_count, width, height),
            DicomColorType::Gray8(tiff::encoder::colortype::Gray8),
        );
        benchmark_case::<u16>(
            &mut group,
            &format!("luma16/{suffix}"),
            &luma16_frames(frame_count, width, height),
            DicomColorType::Gray16(tiff::encoder::colortype::Gray16),
        );
        benchmark_case::<u8>(
            &mut group,
            &format!("rgb8/{suffix}"),
            &rgb8_frames(frame_count, width, height),
            DicomColorType::RGB8(tiff::encoder::colortype::RGB8),
        );
    }
    group.finish();
}

fn high_res_bench_enabled() -> bool {
    std::env::var(HIGH_RES_BENCH_ENV).is_ok_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes"
        )
    })
}

fn main() {
    let mut criterion = Criterion::default().configure_from_args();
    benchmark_dimensions(&mut criterion, DEFAULT_WIDTH, DEFAULT_HEIGHT);
    if high_res_bench_enabled() {
        benchmark_dimensions(&mut criterion, HIGH_RES_WIDTH, HIGH_RES_HEIGHT);
    }
    criterion.final_summary();
}
