use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use dicom_preprocessing::{Padding, Transform};
use image::{DynamicImage, ImageBuffer, Luma};
use std::hint::black_box;
use std::time::Duration;

const WIDTH: u32 = 2048;
const HEIGHT: u32 = 1536;
const TARGET_WIDTH: u32 = 2304;
const TARGET_HEIGHT: u32 = 1792;
const PADDING_WIDTH: u32 = TARGET_WIDTH - WIDTH;
const PADDING_HEIGHT: u32 = TARGET_HEIGHT - HEIGHT;
const PADDING_LEFT: u32 = PADDING_WIDTH / 2;
const PADDING_TOP: u32 = PADDING_HEIGHT / 2;
const PADDING: Padding = Padding {
    left: PADDING_LEFT,
    top: PADDING_TOP,
    right: PADDING_WIDTH - PADDING_LEFT,
    bottom: PADDING_HEIGHT - PADDING_TOP,
};
const SAMPLE_SIZE: usize = 10;

fn synthetic_luma16_frame() -> DynamicImage {
    let image = ImageBuffer::from_fn(WIDTH, HEIGHT, |x, y| {
        let value = ((x.wrapping_mul(31) + y.wrapping_mul(17)) % u16::MAX as u32) as u16;
        Luma([value])
    });

    DynamicImage::ImageLuma16(image)
}

fn bench_padding(c: &mut Criterion) {
    let image = synthetic_luma16_frame();
    let bench_label = format!("{WIDTH}x{HEIGHT}_to_{TARGET_WIDTH}x{TARGET_HEIGHT}");

    let mut group = c.benchmark_group("padding");
    group.sample_size(SAMPLE_SIZE);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(
        (TARGET_WIDTH as u64) * (TARGET_HEIGHT as u64),
    ));
    group.bench_with_input(
        BenchmarkId::new("apply_luma16_center", bench_label),
        &image,
        |b, image| b.iter(|| black_box(PADDING.apply(black_box(image)))),
    );
    group.finish();
}

criterion_group!(benches, bench_padding);
criterion_main!(benches);
