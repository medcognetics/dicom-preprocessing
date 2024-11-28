use criterion::{
    black_box, measurement::Measurement, BenchmarkGroup, BenchmarkId, Criterion, Throughput,
};
use dicom_preprocessing::file::{
    DicomFileOperations as DicomFileTrait, DICM_PREFIX, DICM_PREFIX_LOCATION,
};
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;
use std::path::PathBuf;
use tempfile::tempdir;

const DUMMY_CONTENT: &[u8] = b"test content";

pub trait Setup {
    fn setup<P: AsRef<Path>>(&self, path: P) -> Result<usize, std::io::Error>;

    fn subpath<P: AsRef<Path>>(&self, dir: P, index: usize) -> PathBuf;

    fn setup_many<P: AsRef<Path>>(
        &self,
        dir: P,
        num_files: usize,
    ) -> Result<Vec<PathBuf>, std::io::Error> {
        let mut paths = Vec::with_capacity(num_files);
        for i in 0..num_files {
            let file_path = self.subpath(dir.as_ref(), i);
            self.setup(&file_path)?;
            paths.push(file_path);
        }
        Ok(paths)
    }
}

struct DicomFile;
impl Setup for DicomFile {
    fn setup<P: AsRef<Path>>(&self, path: P) -> Result<usize, std::io::Error> {
        let mut file = std::fs::File::create(path).unwrap();
        file.seek(SeekFrom::Start(DICM_PREFIX_LOCATION))?;
        file.write(DICM_PREFIX)
    }

    fn subpath<P: AsRef<Path>>(&self, dir: P, index: usize) -> PathBuf {
        dir.as_ref().join(format!("file{}.dcm", index))
    }
}

struct OtherFile;
impl Setup for OtherFile {
    fn setup<P: AsRef<Path>>(&self, path: P) -> Result<usize, std::io::Error> {
        std::fs::write(path, DUMMY_CONTENT)?;
        Ok(DUMMY_CONTENT.len())
    }

    fn subpath<P: AsRef<Path>>(&self, dir: P, index: usize) -> PathBuf {
        dir.as_ref().join(format!("file{}.txt", index))
    }
}

struct ExtensionlessFile;
impl Setup for ExtensionlessFile {
    fn setup<P: AsRef<Path>>(&self, path: P) -> Result<usize, std::io::Error> {
        std::fs::write(path, DUMMY_CONTENT)?;
        Ok(DUMMY_CONTENT.len())
    }

    fn subpath<P: AsRef<Path>>(&self, dir: P, index: usize) -> PathBuf {
        dir.as_ref().join(format!("file{}", index))
    }
}

struct Dir;
impl Setup for Dir {
    fn setup<P: AsRef<Path>>(&self, path: P) -> Result<usize, std::io::Error> {
        std::fs::create_dir(path)?;
        Ok(0)
    }

    fn subpath<P: AsRef<Path>>(&self, dir: P, index: usize) -> PathBuf {
        dir.as_ref().join(format!("subdir{}", index))
    }
}

struct BenchDef {
    paths: Vec<PathBuf>,
    id: &'static str,
    sample_size: usize,
}

impl BenchDef {
    fn new(
        id: &'static str,
        sample_size: usize,
        num_dicom: usize,
        num_other: usize,
        num_extensionless: usize,
        num_dir: usize,
    ) -> Self {
        let temp_dir = tempdir().unwrap();
        // Create num_files random files in temp directory
        let total_files = num_dicom + num_other + num_extensionless + num_dir;
        let mut paths = Vec::with_capacity(total_files);
        for i in 0..num_dicom {
            paths.push(DicomFile.subpath(temp_dir.path(), i));
        }
        for i in 0..num_other {
            paths.push(OtherFile.subpath(temp_dir.path(), i));
        }
        for i in 0..num_extensionless {
            paths.push(ExtensionlessFile.subpath(temp_dir.path(), i));
        }
        for i in 0..num_dir {
            paths.push(Dir.subpath(temp_dir.path(), i));
        }
        Self {
            paths,
            id,
            sample_size,
        }
    }

    fn bench_is_dicom_file<M: Measurement>(&self, group: &mut BenchmarkGroup<M>) {
        let func = |path: &PathBuf| path.is_dicom_file();
        group
            .sample_size(self.sample_size)
            .throughput(Throughput::Elements(self.paths.len() as u64))
            .bench_with_input(
                BenchmarkId::new(self.id, self.paths.len()),
                &self.paths,
                |b, input| b.iter(|| black_box(input.iter()).map(func).collect::<Vec<_>>()),
            );
    }
}

fn main() {
    let mut c = Criterion::default().configure_from_args();
    let mut group = c.benchmark_group("dicom-file");
    // These have a fast path where only the extension is checked
    BenchDef::new("is_dicom_file-balanced", 500, 100, 100, 100, 100)
        .bench_is_dicom_file(&mut group);
    BenchDef::new("is_dicom_file-other", 500, 0, 10000, 0, 0).bench_is_dicom_file(&mut group);
    // These will be slower because they require file system access
    BenchDef::new("is_dicom_file-extensionless", 500, 0, 0, 10000, 0)
        .bench_is_dicom_file(&mut group);
    BenchDef::new("is_dicom_file-dir", 500, 0, 0, 0, 10000).bench_is_dicom_file(&mut group);
}
