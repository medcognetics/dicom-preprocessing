use criterion::{BenchmarkId, Criterion, Throughput};
use dicom::object::{open_file, FileDicomObject, InMemDicomObject};
use dicom_preprocessing::{FilterType, KeepVolume, PaddingDirection, Preprocessor, VolumeHandler};
use std::hint::black_box;
use std::time::Duration;

const BATCH_SIZES: [usize; 4] = [1, 8, 32, 64];
const SAMPLE_SIZE: usize = 10;
const RESIZE_DIMENSION: u32 = 96;

fn fixture_batch(batch_size: usize) -> Vec<FileDicomObject<InMemDicomObject>> {
    let fixture = open_file(dicom_test_files::path("pydicom/CT_small.dcm").unwrap()).unwrap();
    (0..batch_size).map(|_| fixture.clone()).collect()
}

fn preprocessor(with_transforms: bool) -> Preprocessor {
    Preprocessor {
        crop: with_transforms,
        size: with_transforms.then_some((RESIZE_DIMENSION, RESIZE_DIMENSION)),
        spacing: None,
        filter: FilterType::Triangle,
        padding_direction: PaddingDirection::Center,
        crop_max: false,
        volume_handler: VolumeHandler::Keep(KeepVolume),
        use_components: true,
        use_padding: with_transforms,
        border_frac: None,
        target_frames: 32,
        convert_options: Default::default(),
    }
}

fn benchmark_mode(c: &mut Criterion, with_transforms: bool) {
    let mode = if with_transforms {
        "crop-resize"
    } else {
        "decode-only"
    };
    let mut group = c.benchmark_group(format!("batch-preprocess/{mode}"));
    group.sample_size(SAMPLE_SIZE);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    let preprocessor = preprocessor(with_transforms);

    for batch_size in BATCH_SIZES {
        let files = fixture_batch(batch_size);
        group.throughput(Throughput::Elements(batch_size as u64));
        for (name, parallel) in [("serial", false), ("parallel", true)] {
            group.bench_with_input(BenchmarkId::new(name, batch_size), &files, |b, files| {
                b.iter(|| {
                    black_box(
                        preprocessor
                            .prepare_images_batch(black_box(files), parallel)
                            .unwrap(),
                    )
                })
            });
        }
    }
    group.finish();
}

fn main() {
    eprintln!("rayon_workers={}", rayon::current_num_threads());
    let mut criterion = Criterion::default().configure_from_args();
    benchmark_mode(&mut criterion, false);
    benchmark_mode(&mut criterion, true);
    criterion.final_summary();
}
