use criterion::{measurement::Measurement, BenchmarkGroup, BenchmarkId, Criterion, Throughput};
use dicom_preprocessing::file::InodeSort;
use std::path::PathBuf;
use tempfile::tempdir;

const NUM_FILES: usize = 1000;

fn setup(num_files: usize) -> Vec<PathBuf> {
    let temp_dir = tempdir().unwrap();
    // Create num_files random files in temp directory
    let mut paths = Vec::with_capacity(num_files);
    for i in 0..num_files {
        let file_path = temp_dir.path().join(format!("file{}.txt", i));
        std::fs::write(&file_path, format!("test content {}", i)).unwrap();
        paths.push(file_path);
    }
    paths
}

struct BenchDef {
    paths: Vec<PathBuf>,
    id: &'static str,
    sample_size: usize,
}

impl BenchDef {
    fn run<M: Measurement>(&self, group: &mut BenchmarkGroup<M>) {
        group
            .sample_size(self.sample_size)
            .throughput(Throughput::Elements(self.paths.len() as u64))
            .bench_with_input(
                BenchmarkId::new(self.id, self.paths.len()),
                &self.paths,
                |b, input| b.iter(|| input.iter().sorted_by_inode().collect::<Vec<_>>()),
            );
    }
}

fn main() {
    let mut c = Criterion::default().configure_from_args();
    let mut group = c.benchmark_group("inode-sort");
    let paths = setup(NUM_FILES);

    BenchDef {
        paths: paths.clone(),
        id: "inode_sort",
        sample_size: 500,
    }
    .run(&mut group);
}
