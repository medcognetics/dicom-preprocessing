use criterion::{BenchmarkId, Criterion, Throughput};
use dicom_preprocessing::{InterpolateVolume, LaplacianMip};
use image::{DynamicImage, ImageBuffer, Luma};
use std::hint::black_box;
use std::time::Duration;

const INTERPOLATE_SOURCE_FRAMES: usize = 24;
const INTERPOLATE_TARGET_FRAMES: u32 = 48;
const LAP_MIP_FRAMES: usize = 32;
const HIGH_RES_LAP_MIP_FRAMES: usize = 50;
const WIDTH: u32 = 128;
const HEIGHT: u32 = 128;
const LARGE_LAP_MIP_WIDTH: u32 = 256;
const LARGE_LAP_MIP_HEIGHT: u32 = 256;
const HIGH_RES_LAP_MIP_WIDTH: u32 = 2048;
const HIGH_RES_LAP_MIP_HEIGHT: u32 = 1536;
const HIGH_RES_BENCH_ENV: &str = "DICOM_BENCH_HIGH_RES";
const LAPLACIAN_MIP_SAMPLE_SIZE: usize = 10;

fn high_res_bench_enabled() -> bool {
    match std::env::var(HIGH_RES_BENCH_ENV) {
        Ok(value) => {
            let normalized = value.trim().to_ascii_lowercase();
            normalized == "1" || normalized == "true" || normalized == "yes"
        }
        Err(_) => false,
    }
}

const fn pixels(width: u32, height: u32) -> u64 {
    (width as u64) * (height as u64)
}

fn synthetic_luma16_frames(num_frames: usize, width: u32, height: u32) -> Vec<DynamicImage> {
    (0..num_frames)
        .map(|frame_idx| {
            let image = ImageBuffer::from_fn(width, height, |x, y| {
                let value = ((x as usize + y as usize + frame_idx * 7) % u16::MAX as usize) as u16;
                Luma([value])
            });
            DynamicImage::ImageLuma16(image)
        })
        .collect()
}

fn bench_interpolation(c: &mut Criterion) {
    let mut group = c.benchmark_group("dbt-interpolate");
    let source_frames = synthetic_luma16_frames(INTERPOLATE_SOURCE_FRAMES, WIDTH, HEIGHT);
    let pixels_per_frame = pixels(WIDTH, HEIGHT);
    group.throughput(Throughput::Elements(
        pixels_per_frame * INTERPOLATE_TARGET_FRAMES as u64,
    ));
    group.sample_size(20);
    group.bench_with_input(
        BenchmarkId::new("interpolate_frames", INTERPOLATE_TARGET_FRAMES),
        &source_frames,
        |b, frames| {
            b.iter(|| {
                black_box(InterpolateVolume::interpolate_frames(
                    black_box(frames),
                    INTERPOLATE_TARGET_FRAMES,
                ))
            })
        },
    );
    group.finish();
}

fn bench_laplacian_case(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    laplacian_mip: &LaplacianMip,
    bench_name: &'static str,
    frames: &[DynamicImage],
    frame_count: usize,
    width: u32,
    height: u32,
) {
    group.throughput(Throughput::Elements(
        pixels(width, height) * frame_count as u64,
    ));
    group.bench_with_input(
        BenchmarkId::new(bench_name, frame_count),
        &frames,
        |b, input| {
            b.iter(|| {
                black_box(
                    laplacian_mip
                        .project_laplacian_mip(black_box(input))
                        .expect("Laplacian-MIP projection should succeed"),
                )
            })
        },
    );
}

fn bench_laplacian_mip(c: &mut Criterion) {
    let mut group = c.benchmark_group("dbt-laplacian-mip");
    let frames = synthetic_luma16_frames(LAP_MIP_FRAMES, WIDTH, HEIGHT);
    let large_frames =
        synthetic_luma16_frames(LAP_MIP_FRAMES, LARGE_LAP_MIP_WIDTH, LARGE_LAP_MIP_HEIGHT);
    group.sample_size(LAPLACIAN_MIP_SAMPLE_SIZE);

    let laplacian_mip = LaplacianMip::new(0, 0);
    bench_laplacian_case(
        &mut group,
        &laplacian_mip,
        "project_laplacian_mip_128x128",
        &frames,
        LAP_MIP_FRAMES,
        WIDTH,
        HEIGHT,
    );
    bench_laplacian_case(
        &mut group,
        &laplacian_mip,
        "project_laplacian_mip_256x256",
        &large_frames,
        LAP_MIP_FRAMES,
        LARGE_LAP_MIP_WIDTH,
        LARGE_LAP_MIP_HEIGHT,
    );
    group.finish();
}

fn bench_laplacian_mip_high_res(c: &mut Criterion) {
    if !high_res_bench_enabled() {
        return;
    }

    let mut group = c.benchmark_group("dbt-laplacian-mip-high-res");
    group.sample_size(LAPLACIAN_MIP_SAMPLE_SIZE);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));

    let frames = synthetic_luma16_frames(
        HIGH_RES_LAP_MIP_FRAMES,
        HIGH_RES_LAP_MIP_WIDTH,
        HIGH_RES_LAP_MIP_HEIGHT,
    );
    let laplacian_mip = LaplacianMip::new(0, 0);
    bench_laplacian_case(
        &mut group,
        &laplacian_mip,
        "project_laplacian_mip_2048x1536",
        &frames,
        HIGH_RES_LAP_MIP_FRAMES,
        HIGH_RES_LAP_MIP_WIDTH,
        HIGH_RES_LAP_MIP_HEIGHT,
    );
    group.finish();
}

fn main() {
    let mut criterion = Criterion::default().configure_from_args();
    bench_interpolation(&mut criterion);
    bench_laplacian_mip(&mut criterion);
    bench_laplacian_mip_high_res(&mut criterion);
}
