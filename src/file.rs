use dicom::object::open_file;
use dicom::object::DefaultDicomObject;
use dicom::object::ReadError;
use indicatif::{ParallelProgressIterator, ProgressIterator};
use itertools::Itertools;
use rayon::iter::IntoParallelIterator;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::str::FromStr;
use tiff::decoder::Decoder;
use tiff::TiffError;

use indicatif::{ProgressBar, ProgressStyle};
use rust_search::SearchBuilder;
use std::os::unix::fs::MetadataExt;
use std::path::Path;

pub const DICM_PREFIX: &[u8; 4] = b"DICM";
pub const DICM_PREFIX_LOCATION: u64 = 128;

type IOResult<T> = Result<T, std::io::Error>;

pub fn default_bar(len: u64) -> ProgressBar {
    let pb = ProgressBar::new(len);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{msg} {spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta} @ {per_sec})",
            )
            .unwrap(),
    );
    pb
}

pub fn default_spinner() -> ProgressBar {
    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.blue} {msg}")
            .unwrap(),
    );
    spinner
}

/// Filter map function for boolean tuple results with the following logic:
/// - Boolean true results are propagated
/// - Boolean false results are filtered
/// - Errors are propagated
fn filter_fn_bool_tuple<T, E>(r: Result<(bool, T), E>) -> Option<Result<T, E>> {
    match r {
        Ok((true, p)) => Some(Ok(p)),
        Ok((false, _)) => None,
        Err(e) => Some(Err(e)),
    }
}

pub trait Inode
where
    Self: AsRef<Path>,
{
    /// Get the inode of a path.
    fn inode(&self) -> IOResult<u64> {
        let metadata = std::fs::metadata(self.as_ref())?;
        Ok(metadata.ino())
    }

    /// Get the inode of a path, or a default value if an error occurs.
    fn inode_or(&self, value: u64) -> u64 {
        let metadata = std::fs::metadata(self.as_ref());
        match metadata {
            Ok(metadata) => metadata.ino(),
            Err(_) => value,
        }
    }
}

impl<P: AsRef<Path>> Inode for P {}

pub trait InodeSort<P>
where
    P: AsRef<Path> + Inode,
    Self: Iterator<Item = P>,
{
    /// Sort paths by inode number.
    fn sorted_by_inode(&mut self) -> impl Iterator<Item = P> {
        self.map(|p| (p.inode_or(0), p))
            .sorted_unstable_by_key(|(i, _)| *i)
            .map(|(_, p)| p)
    }

    /// Like `sorted_by_inode`, but with a progress bar or spinner.
    /// The progress bar tracks the querying of the inode numbers, but not the sorting.
    fn sorted_by_inode_with_progress(&mut self) -> impl Iterator<Item = P> {
        let (_, total) = self.size_hint();
        let pb = match total {
            Some(total) => default_bar(total as u64),
            None => default_spinner(),
        };
        pb.set_message("Sorting paths by inode");
        self.map(|p| (p.inode_or(0), p))
            .progress_with(pb)
            .sorted_unstable_by_key(|(i, _)| *i)
            .map(|(_, p)| p)
    }
}

impl<P: AsRef<Path> + Inode, I: Iterator<Item = P>> InodeSort<P> for I {}

pub trait SourceFileOperations
where
    Self: AsRef<Path>,
{
    /// Read a file containing a list of paths and return an iterator of results.
    fn read_paths(&self) -> IOResult<impl Iterator<Item = IOResult<PathBuf>>> {
        let reader = BufReader::new(File::open(self.as_ref())?);
        let result = reader
            .lines()
            .map(|s| match s {
                Ok(s) => PathBuf::from_str(s.as_str())
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e)),
                Err(e) => return Err(e),
            })
            .into_iter();
        Ok(result)
    }
}

impl<P: AsRef<Path>> SourceFileOperations for P {}

pub trait DicomFileOperations
where
    Self: AsRef<Path>,
{
    /// Check if a file has a DICM prefix.
    /// This will only return an error if the file cannot be opened.
    /// Any other errors mapped to `false`.
    fn has_dicm_prefix(&self) -> IOResult<bool> {
        let mut reader = File::open(self.as_ref())?;
        let mut buffer = [0; DICM_PREFIX.len()];
        reader
            .seek(SeekFrom::Start(DICM_PREFIX_LOCATION))
            .and_then(|_| reader.read_exact(&mut buffer))
            .map_or(Ok(false), |_| Ok(&buffer == DICM_PREFIX))
    }

    /// Check if a file has a DICOM extension.
    fn has_dicom_extension(&self) -> bool {
        let path = self.as_ref();
        if let Some(ext) = path.extension() {
            return ext == "dcm" || ext == "dicom" || ext == "DCM" || ext == "DICOM";
        }
        return false;
    }

    /// Check if a path is a DICOM file as efficiently as possible.
    /// The function will use the file extension if available, otherwise it will check the DICM prefix.
    fn is_dicom_file(&self) -> IOResult<bool> {
        let path = self.as_ref();
        if self.has_dicom_extension() {
            Ok(path.is_file())
        } else if path.extension().is_some() || path.is_dir() {
            Ok(false)
        } else {
            self.has_dicm_prefix()
        }
    }

    /// Similar to `is_dicom_file`, but returns a default value if an error occurs.
    fn is_dicom_file_or(&self, default: bool) -> bool {
        self.is_dicom_file().unwrap_or(default)
    }

    /// Find all DICOM files in a directory.
    fn find_dicoms(&self) -> IOResult<impl Iterator<Item = PathBuf>> {
        let dir = self.as_ref();
        if !dir.is_dir() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Not a directory",
            ));
        }
        let result = SearchBuilder::default()
            .location(dir)
            .build()
            .filter(move |file| file.is_dicom_file_or(false))
            .map(|file| PathBuf::from_str(file.as_str()).unwrap());
        Ok(result)
    }

    /// Find all DICOM files in a directory, with a progress spinner.
    fn find_dicoms_with_spinner(&self) -> IOResult<impl Iterator<Item = PathBuf>> {
        let spinner = default_spinner();
        spinner.set_message("Searching for DICOM files");
        let result = self.find_dicoms()?.inspect(move |_| {
            spinner.tick();
        });
        Ok(result)
    }

    /// Read DICOM paths from a text file. Propagates any errors encountered in opening the text file
    /// or in parsing/validating the paths. Paths are filtered using `is_dicom_file`.
    fn read_dicom_paths(&self) -> IOResult<impl Iterator<Item = PathBuf>>
    where
        Self: SourceFileOperations,
    {
        let result = self
            .read_paths()?
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|r| match r {
                Ok(p) => p.is_dicom_file().map(|is_dicom| (is_dicom, p)),
                Err(e) => Err(e),
            })
            .filter_map(filter_fn_bool_tuple)
            .collect::<IOResult<Vec<_>>>()?
            .into_iter();
        Ok(result)
    }

    /// Like `read_dicom_paths`, but with a progress bar.
    fn read_dicom_paths_with_bar(&self) -> IOResult<impl Iterator<Item = PathBuf>>
    where
        Self: SourceFileOperations,
    {
        let paths = self.read_paths()?.collect::<Vec<_>>();

        let pb = default_bar(paths.len() as u64);
        pb.set_message("Reading DICOM paths from text file");

        let result = paths
            .into_par_iter()
            .progress_with(pb)
            .map(|r| match r {
                Ok(p) => p.is_dicom_file().map(|is_dicom| (is_dicom, p)),
                Err(e) => Err(e),
            })
            .filter_map(filter_fn_bool_tuple)
            .collect::<IOResult<Vec<_>>>()?
            .into_iter();
        Ok(result)
    }

    /// Read the DICOM file.
    fn dcmread(&self) -> Result<DefaultDicomObject, ReadError> {
        open_file(self.as_ref())
    }
}

impl<P: AsRef<Path>> DicomFileOperations for P {}

pub trait TiffFileOperations
where
    Self: AsRef<Path>,
{
    /// Check if a file has a TIFF extension.
    fn has_tiff_extension(&self) -> bool {
        let path = self.as_ref();
        if let Some(ext) = path.extension() {
            return ext == "tiff" || ext == "tif" || ext == "TIFF" || ext == "TIF";
        }
        return false;
    }

    /// Check if a path is a TIFF file.
    fn is_tiff_file(&self) -> Result<bool, std::io::Error> {
        // We don't handle extensionless TIFF files, extension and existence are the only checks.
        // Signature is matched against is_dicom_file for consistency.
        Ok(self.as_ref().is_file() && self.has_tiff_extension())
    }

    /// Similar to `is_tiff_file`, but returns a default value if an error occurs.
    fn is_tiff_file_or(&self, default: bool) -> bool {
        self.is_tiff_file().unwrap_or(default)
    }

    /// Find all TIFF files in a directory.
    fn find_tiffs(&self) -> Result<impl Iterator<Item = PathBuf>, std::io::Error> {
        let dir = self.as_ref();
        if !dir.is_dir() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Not a directory",
            ));
        }
        let result = SearchBuilder::default()
            .location(dir)
            .build()
            .filter(move |file| file.is_tiff_file_or(false))
            .map(|file| PathBuf::from_str(file.as_str()).unwrap());
        Ok(result)
    }

    /// Find all TIFF files in a directory, with a progress spinner.
    fn find_tiffs_with_spinner(&self) -> IOResult<impl Iterator<Item = PathBuf>> {
        let spinner = default_spinner();
        spinner.set_message("Searching for TIFF files");
        let result = self.find_tiffs()?.inspect(move |_| {
            spinner.tick();
        });
        Ok(result)
    }

    /// Read TIFF paths from a text file. Propagates any errors encountered in opening the text file
    /// or in parsing/validating the paths. Paths are filtered using `is_tiff_file`.
    fn read_tiff_paths(&self) -> IOResult<impl Iterator<Item = PathBuf>>
    where
        Self: SourceFileOperations,
    {
        let result = self
            .read_paths()?
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|r| match r {
                Ok(p) => p.is_tiff_file().map(|is_tiff| (is_tiff, p)),
                Err(e) => Err(e),
            })
            .filter_map(filter_fn_bool_tuple)
            .collect::<IOResult<Vec<_>>>()?
            .into_iter();
        Ok(result)
    }

    /// Like `read_tiff_paths`, but with a progress bar.
    fn read_tiff_paths_with_bar(&self) -> IOResult<impl Iterator<Item = PathBuf>>
    where
        Self: SourceFileOperations,
    {
        let paths = self.read_paths()?.collect::<Vec<_>>();

        let pb = default_bar(paths.len() as u64);
        pb.set_message("Reading TIFF paths from text file");

        let result = paths
            .into_par_iter()
            .progress_with(pb)
            .map(|r| match r {
                Ok(p) => p.is_tiff_file().map(|is_tiff| (is_tiff, p)),
                Err(e) => Err(e),
            })
            .filter_map(filter_fn_bool_tuple)
            .collect::<IOResult<Vec<_>>>()?
            .into_iter();
        Ok(result)
    }

    /// Read the TIFF file.
    fn tiffread(&self) -> Result<Decoder<BufReader<File>>, TiffError> {
        let file = File::open(self.as_ref()).map_err(|e| TiffError::IoError(e))?;
        let decoder = Decoder::new(BufReader::new(file))?;
        Ok(decoder)
    }
}

impl<P: AsRef<Path>> TiffFileOperations for P {}

#[cfg(test)]
mod tests {
    use super::*;

    use rstest::rstest;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[rstest]
    #[case::empty_file(vec![], false)]
    #[case::dicm_prefix(b"DICM".to_vec(), true)]
    #[case::wrong_prefix(b"NOT_DICM".to_vec(), false)]
    fn test_has_dicm_prefix(#[case] contents: Vec<u8>, #[case] expected: bool) {
        let mut temp = NamedTempFile::new().unwrap();
        temp.seek(SeekFrom::Start(DICM_PREFIX_LOCATION)).unwrap();
        temp.write_all(&contents).unwrap();

        let result = temp.path().has_dicm_prefix().unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_has_dicm_prefix_real_dicom() {
        let dicom_file_path = dicom_test_files::path("pydicom/CT_small.dcm").unwrap();
        let result = dicom_file_path.has_dicm_prefix().unwrap();
        assert!(result);
    }

    #[rstest]
    #[case::no_extension("test", false)]
    #[case::wrong_extension("test.txt", false)]
    #[case::dcm_extension("test.dcm", true)]
    #[case::dicom_extension("test.dicom", true)]
    #[case::dcm_extension_uppercase("test.DCM", true)]
    #[case::dicom_extension_uppercase("test.DICOM", true)]
    #[case::mixed_case("test.DiCoM", false)]
    fn test_has_dicom_extension(#[case] path: &str, #[case] expected: bool) {
        let path = PathBuf::from(path);
        let result = path.has_dicom_extension();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_is_dicom_file_real_dicom() {
        let path = dicom_test_files::path("pydicom/CT_small.dcm").unwrap();
        let result = path.is_dicom_file().unwrap();
        assert_eq!(result, true);
    }

    #[rstest]
    #[case::no_spinner(false)]
    #[case::spinner(true)]
    fn test_find_dicom_files(#[case] spinner: bool) {
        // Create a temp directory with some test files
        let temp_dir = tempfile::tempdir().unwrap();

        // Copy a real DICOM file
        let dicom_path = dicom_test_files::path("pydicom/CT_small.dcm").unwrap();
        let dicom_dest = temp_dir.path().join("test.dcm");
        std::fs::copy(&dicom_path, &dicom_dest).unwrap();

        // Create a non-DICOM file
        let text_path = temp_dir.path().join("test.txt");
        std::fs::write(&text_path, "not a DICOM file").unwrap();

        // Create a subdirectory with another DICOM file
        let sub_dir = temp_dir.path().join("subdir");
        std::fs::create_dir(&sub_dir).unwrap();
        let sub_dicom = sub_dir.join("sub.dcm");
        std::fs::copy(&dicom_path, &sub_dicom).unwrap();

        let files: Vec<_> = match spinner {
            true => temp_dir
                .path()
                .find_dicoms_with_spinner()
                .unwrap()
                .collect(),
            false => temp_dir.path().find_dicoms().unwrap().collect(),
        };
        assert_eq!(files.len(), 2);
        assert!(files.iter().any(|p| p == &dicom_dest));
        assert!(files.iter().any(|p| p == &sub_dicom));
    }

    #[rstest]
    #[case::no_spinner(false)]
    #[case::spinner(true)]
    fn test_inode_sort(#[case] spinner: bool) {
        // Create temp directory with test files
        let temp_dir = tempfile::tempdir().unwrap();

        // Create multiple files with known content
        let file1 = temp_dir.path().join("file1.txt");
        let file2 = temp_dir.path().join("file2.txt");
        let file3 = temp_dir.path().join("file3.txt");

        std::fs::write(&file1, "file1").unwrap();
        std::fs::write(&file2, "file2").unwrap();
        std::fs::write(&file3, "file3").unwrap();

        // Get paths in random order
        let paths = vec![&file2, &file3, &file1];

        // Sort by inode
        let sorted: Vec<_> = match spinner {
            true => paths.into_iter().sorted_by_inode_with_progress().collect(),
            false => paths.into_iter().sorted_by_inode().collect(),
        };

        // Verify files exist and are readable
        for path in &sorted {
            assert!(path.exists());
            let content = std::fs::read_to_string(path).unwrap();
            assert!(!content.is_empty());
        }

        // Verify we got back all files
        assert_eq!(sorted.len(), 3);

        // Verify inode ordering by checking each pair is ordered
        let inodes: Vec<u64> = sorted
            .iter()
            .map(|p| std::fs::metadata(p).unwrap().ino())
            .collect();

        for i in 1..inodes.len() {
            assert!(inodes[i - 1] <= inodes[i]);
        }
    }

    #[rstest]
    #[case::no_bar(false)]
    #[case::bar(true)]
    fn test_dicom_paths_from_file(#[case] bar: bool) {
        // Create temp file with list of paths
        let temp_file = tempfile::NamedTempFile::new().unwrap();
        let temp_dir = tempfile::tempdir().unwrap();

        // Create test files
        let file1 = temp_dir.path().join("file1.dcm");
        let file2 = temp_dir.path().join("file2.dcm");
        std::fs::write(&file1, "test1").unwrap();
        std::fs::write(&file2, "test2").unwrap();

        // Write paths to temp file
        writeln!(temp_file.as_file(), "{}", file1.display()).unwrap();
        writeln!(temp_file.as_file(), "{}", file2.display()).unwrap();

        // Read paths from file
        let paths = match bar {
            true => temp_file
                .path()
                .read_dicom_paths_with_bar()
                .unwrap()
                .collect::<Vec<_>>(),
            false => temp_file
                .path()
                .read_dicom_paths()
                .unwrap()
                .collect::<Vec<_>>(),
        };

        assert!(paths.len() == 2);
        assert!(paths[0].exists());
        assert!(paths[1].exists());
    }

    #[rstest]
    #[case("test.tiff", true)]
    #[case("test.tif", true)]
    #[case("test.TIFF", true)]
    #[case("test.TIF", true)]
    #[case("path/to/test.tiff", true)]
    #[case("test.txt", false)]
    #[case("test", false)]
    #[case("test.tif.txt", false)]
    #[case("path/to/test", false)]
    fn test_tiff_extension(#[case] path: &str, #[case] expected: bool) {
        let path = PathBuf::from(path);
        assert_eq!(path.has_tiff_extension(), expected);
    }

    #[rstest]
    #[case("test.tiff", true)]
    #[case("test.txt", false)]
    #[case("nonexistent.tiff", false)]
    fn test_is_tiff_file(#[case] filename: &str, #[case] expected: bool) {
        let temp_dir = tempfile::tempdir().unwrap();
        let file_path = temp_dir.path().join(filename);
        if !filename.starts_with("nonexistent") {
            std::fs::write(&file_path, "test").unwrap();
        }

        assert_eq!(file_path.is_tiff_file().unwrap(), expected);
        assert_eq!(file_path.is_tiff_file_or(!expected), expected);
    }

    #[rstest]
    #[case::no_spinner(false)]
    #[case::spinner(true)]
    fn test_find_tiffs(#[case] spinner: bool) {
        let temp_dir = tempfile::tempdir().unwrap();

        // Create test files
        let tiff_files = vec![
            temp_dir.path().join("test1.tiff"),
            temp_dir.path().join("test2.tif"),
            temp_dir.path().join("test3.TIFF"),
        ];
        let other_files = vec![
            temp_dir.path().join("test4.txt"),
            temp_dir.path().join("test5.doc"),
        ];

        // Create all test files
        for file in tiff_files.iter().chain(other_files.iter()) {
            std::fs::write(file, "test").unwrap();
        }

        // Find TIFF files
        let found: Vec<_> = match spinner {
            true => temp_dir.path().find_tiffs_with_spinner().unwrap().collect(),
            false => temp_dir.path().find_tiffs().unwrap().collect(),
        };

        // Verify we found all TIFF files and only TIFF files
        assert_eq!(found.len(), tiff_files.len());
        for file in &found {
            assert!(tiff_files.contains(file));
        }
    }

    #[rstest]
    #[case::no_bar(false)]
    #[case::bar(true)]
    fn test_tiff_paths_from_file(#[case] bar: bool) {
        // Create temp file with list of paths
        let mut temp_file = tempfile::NamedTempFile::new().unwrap();
        let temp_dir = tempfile::tempdir().unwrap();

        // Create test files
        let file1 = temp_dir.path().join("file1.tiff");
        let file2 = temp_dir.path().join("file2.tiff");
        std::fs::write(&file1, "test1").unwrap();
        std::fs::write(&file2, "test2").unwrap();

        // Write paths to temp file
        writeln!(temp_file.as_file(), "{}", file1.display()).unwrap();
        writeln!(temp_file.as_file(), "{}", file2.display()).unwrap();
        writeln!(temp_file.as_file(), "invalid/path").unwrap();
        temp_file.flush().unwrap();

        // Read paths from file
        let paths = match bar {
            true => temp_file
                .path()
                .read_tiff_paths_with_bar()
                .unwrap()
                .collect::<Vec<_>>(),
            false => temp_file
                .path()
                .read_tiff_paths()
                .unwrap()
                .collect::<Vec<_>>(),
        };

        // First two paths should be valid
        assert!(paths.len() == 2);
        assert!(paths[0].exists());
        assert!(paths[1].exists());
    }
}
