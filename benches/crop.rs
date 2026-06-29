use criterion::{
    criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, BenchmarkId, Criterion,
    Throughput,
};
use dicom_preprocessing::Crop;
use image::{DynamicImage, ImageBuffer, Luma};
use std::hint::black_box;
use std::time::Duration;

const WIDTH: u32 = 2048;
const HEIGHT: u32 = 1536;
const FRAME_COUNT: usize = 8;
const FOREGROUND_LEFT: u32 = 280;
const FOREGROUND_TOP: u32 = 160;
const FOREGROUND_WIDTH: u32 = 1420;
const FOREGROUND_HEIGHT: u32 = 1180;
const ARTIFACT_SIZE: u32 = 24;
const FOREGROUND_VALUE: u16 = 32_000;
const ARTIFACT_VALUE: u16 = 48_000;
const SAMPLE_SIZE: usize = 10;
const SINGLE_FRAME_LABEL: &str = "2048x1536";
const MULTI_FRAME_LABEL: &str = "8x2048x1536";

fn in_rect(x: u32, y: u32, left: u32, top: u32, width: u32, height: u32) -> bool {
    x >= left && x < left + width && y >= top && y < top + height
}

fn synthetic_mammography_frame(frame_idx: usize) -> DynamicImage {
    let shift = frame_idx as u32 % 7;
    let image = ImageBuffer::from_fn(WIDTH, HEIGHT, |x, y| {
        let foreground = in_rect(
            x,
            y,
            FOREGROUND_LEFT + shift,
            FOREGROUND_TOP,
            FOREGROUND_WIDTH,
            FOREGROUND_HEIGHT,
        );
        let artifact = in_rect(x, y, 48 + shift, 64, ARTIFACT_SIZE, ARTIFACT_SIZE)
            || in_rect(
                x,
                y,
                WIDTH - 96,
                HEIGHT - 96 - shift,
                ARTIFACT_SIZE,
                ARTIFACT_SIZE,
            );

        let value = if foreground {
            FOREGROUND_VALUE + (x.wrapping_add(y) % 1024) as u16
        } else if artifact {
            ARTIFACT_VALUE
        } else {
            0
        };

        Luma([value])
    });

    DynamicImage::ImageLuma16(image)
}

fn synthetic_frames() -> Vec<DynamicImage> {
    (0..FRAME_COUNT).map(synthetic_mammography_frame).collect()
}

fn bench_crop_case(
    group: &mut BenchmarkGroup<'_, WallTime>,
    bench_name: &'static str,
    bench_label: &'static str,
    images: &[&DynamicImage],
) {
    group.throughput(Throughput::Elements(
        (WIDTH as u64) * (HEIGHT as u64) * images.len() as u64,
    ));
    group.bench_with_input(
        BenchmarkId::new(bench_name, bench_label),
        images,
        |b, images| {
            b.iter(|| black_box(Crop::new_from_images(black_box(images), true, true, None)))
        },
    );
}

fn bench_component_crop(c: &mut Criterion) {
    let frames = synthetic_frames();
    let single_frame = vec![&frames[0]];
    let multi_frame: Vec<&DynamicImage> = frames.iter().collect();

    let mut group = c.benchmark_group("component-crop");
    group.sample_size(SAMPLE_SIZE);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));

    bench_crop_case(
        &mut group,
        "single_frame_luma16",
        SINGLE_FRAME_LABEL,
        &single_frame,
    );
    bench_crop_case(
        &mut group,
        "multi_frame_luma16",
        MULTI_FRAME_LABEL,
        &multi_frame,
    );

    group.finish();
}

criterion_group!(benches, bench_component_crop);
criterion_main!(benches);
