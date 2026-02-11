use criterion::{BenchmarkId, Criterion, Throughput};
use dicom_preprocessing::{InterpolateVolume, LaplacianMip};
use image::{DynamicImage, ImageBuffer, Luma};
use std::hint::black_box;

const INTERPOLATE_SOURCE_FRAMES: usize = 24;
const INTERPOLATE_TARGET_FRAMES: u32 = 48;
const LAP_MIP_FRAMES: usize = 32;
const WIDTH: u32 = 128;
const HEIGHT: u32 = 128;

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
    let pixels_per_frame = (WIDTH as u64) * (HEIGHT as u64);
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

fn bench_laplacian_mip(c: &mut Criterion) {
    let mut group = c.benchmark_group("dbt-laplacian-mip");
    let frames = synthetic_luma16_frames(LAP_MIP_FRAMES, WIDTH, HEIGHT);
    let pixels_per_frame = (WIDTH as u64) * (HEIGHT as u64);
    group.throughput(Throughput::Elements(
        pixels_per_frame * LAP_MIP_FRAMES as u64,
    ));
    group.sample_size(10);

    let laplacian_mip = LaplacianMip::new(0, 0);
    group.bench_with_input(
        BenchmarkId::new("project_laplacian_mip", LAP_MIP_FRAMES),
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
    group.finish();
}

fn main() {
    let mut criterion = Criterion::default().configure_from_args();
    bench_interpolation(&mut criterion);
    bench_laplacian_mip(&mut criterion);
}
