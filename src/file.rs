use dicom::object::open_file;
use dicom::object::DefaultDicomObject;
use dicom::object::ReadError;
use itertools::Itertools;
use rayon::iter::IntoParallelIterator;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::str::FromStr;

use indicatif::{ProgressBar, ProgressStyle};
use rust_search::SearchBuilder;
use std::os::unix::fs::MetadataExt;
use std::path::Path;

pub const DICM_PREFIX: &[u8; 4] = b"DICM";
pub const DICM_PREFIX_LOCATION: u64 = 128;

pub trait Inode
where
    Self: AsRef<Path>,
{
    /// Get the inode of a path.
    fn inode(&self) -> Result<u64, std::io::Error> {
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
        self.into_iter()
            .map(|p| (p.inode_or(0), p))
            .sorted_unstable_by_key(|(i, _)| *i)
            .map(|(_, p)| p)
    }
}

impl<P: AsRef<Path> + Inode, I: Iterator<Item = P>> InodeSort<P> for I {}

pub trait DicomFile
where
    Self: AsRef<Path>,
{
    /// Check if a file has a DICM prefix.
    /// This will only return an error if the file cannot be opened.
    /// Any other errors mapped to `false`.
    fn has_dicm_prefix(&self) -> Result<bool, std::io::Error> {
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
    /// If strict is false, the function will not check if the file exists.
    fn is_dicom_file(&self, strict: bool) -> Result<bool, std::io::Error> {
        let path = self.as_ref();
        if self.has_dicom_extension() {
            Ok(!strict || path.is_file())
        } else if path.extension().is_some() || path.is_dir() {
            Ok(false)
        } else {
            self.has_dicm_prefix()
        }
    }

    /// Similar to `is_dicom_file`, but returns a default value if an error occurs.
    fn is_dicom_file_or(&self, strict: bool, default: bool) -> bool {
        self.is_dicom_file(strict).unwrap_or(default)
    }

    /// Read the DICOM file.
    fn dcmread(&self) -> Result<DefaultDicomObject, ReadError> {
        open_file(self.as_ref())
    }
}

impl<P: AsRef<Path>> DicomFile for P {}

/// Find all DICOM files in a directory.
/// If bar is true, a progress bar will be shown.
pub fn find_dicom_files<P: AsRef<Path>>(dir: P, bar: bool) -> impl Iterator<Item = PathBuf> {
    let spinner = if bar {
        let spinner = ProgressBar::new_spinner();
        spinner.set_message("Searching for DICOM files");
        spinner.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.blue} {msg}")
                .unwrap(),
        );
        Some(spinner)
    } else {
        None
    };

    SearchBuilder::default()
        .location(dir)
        .build()
        .inspect(move |_| {
            if let Some(spinner) = &spinner {
                spinner.tick();
            }
        })
        .filter(move |file| file.is_dicom_file_or(false, false))
        .map(|file| PathBuf::from_str(file.as_str()).unwrap())
}

pub trait FileList
where
    Self: AsRef<Path>,
{
    /// Read a file containing a list of paths and return an iterator of results.
    fn read_paths(
        &self,
    ) -> Result<impl Iterator<Item = Result<PathBuf, std::io::Error>>, std::io::Error> {
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

    /// Read a file containing a list of DICOM and validate each path.
    fn read_valid_paths(&self, strict: bool, bar: bool) -> Result<Vec<PathBuf>, std::io::Error> {
        let paths = self.read_paths()?.collect::<Vec<_>>();

        // Set up spinner, checking files may take some time
        let spinner = if bar {
            let spinner = ProgressBar::new_spinner();
            spinner.set_message("Checking input paths from text file");
            spinner.set_style(
                ProgressStyle::default_spinner()
                    .template("{spinner:.blue} {msg}")
                    .unwrap(),
            );
            Some(spinner)
        } else {
            None
        };

        // Check each path
        let result = paths
            .into_par_iter()
            .inspect(|_| {
                if let Some(spinner) = &spinner {
                    spinner.tick()
                }
            })
            .map(|r| match r {
                Ok(path) => match path.is_dicom_file_or(false, false) {
                    true => Ok(path),
                    false => Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "Not a DICOM file",
                    )),
                },
                Err(e) => Err(e),
            });

        // For strict mode, return the errors
        if strict {
            result.collect::<Result<Vec<_>, _>>()
        // For non-strict mode, return the valid paths
        } else {
            Ok(result
                .collect::<Vec<_>>()
                .into_iter()
                .filter_map(|path| path.ok())
                .collect())
        }
    }
}

impl<P: AsRef<Path>> FileList for P {}

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
    fn test_has_dicm_prefix(
        #[case] contents: Vec<u8>,
        #[case] expected: bool,
    ) -> std::io::Result<()> {
        let mut temp = NamedTempFile::new()?;
        temp.seek(SeekFrom::Start(DICM_PREFIX_LOCATION))?;
        temp.write_all(&contents)?;

        let result = temp.path().has_dicm_prefix()?;
        assert_eq!(result, expected);
        Ok(())
    }

    #[test]
    fn test_has_dicm_prefix_real_dicom() -> std::io::Result<()> {
        let dicom_file_path = dicom_test_files::path("pydicom/CT_small.dcm").unwrap();
        let result = dicom_file_path.has_dicm_prefix()?;
        assert!(result);
        Ok(())
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

    #[rstest]
    #[case::strict_valid_dicom(true)]
    #[case::strict_invalid_dicom(false)]
    fn test_is_dicom_file_real_dicom(#[case] strict: bool) -> std::io::Result<()> {
        let path = dicom_test_files::path("pydicom/CT_small.dcm").unwrap();
        let result = path.is_dicom_file(strict)?;
        assert_eq!(result, true);
        Ok(())
    }

    #[rstest]
    #[case::no_bar(false)]
    #[case::bar(true)]
    fn test_find_dicom_files(#[case] bar: bool) -> std::io::Result<()> {
        // Create a temp directory with some test files
        let temp_dir = tempfile::tempdir()?;

        // Copy a real DICOM file
        let dicom_path = dicom_test_files::path("pydicom/CT_small.dcm").unwrap();
        let dicom_dest = temp_dir.path().join("test.dcm");
        std::fs::copy(&dicom_path, &dicom_dest)?;

        // Create a non-DICOM file
        let text_path = temp_dir.path().join("test.txt");
        std::fs::write(&text_path, "not a DICOM file")?;

        // Create a subdirectory with another DICOM file
        let sub_dir = temp_dir.path().join("subdir");
        std::fs::create_dir(&sub_dir)?;
        let sub_dicom = sub_dir.join("sub.dcm");
        std::fs::copy(&dicom_path, &sub_dicom)?;

        let files: Vec<_> = find_dicom_files(&temp_dir, bar).collect();
        assert_eq!(files.len(), 2);
        assert!(files.iter().any(|p| p == &dicom_dest));
        assert!(files.iter().any(|p| p == &sub_dicom));
        Ok(())
    }

    #[rstest]
    #[case::sequential(false)]
    #[case::parallel(true)]
    fn test_inode_sort(#[case] parallel: bool) -> std::io::Result<()> {
        // Create temp directory with test files
        let temp_dir = tempfile::tempdir()?;

        // Create multiple files with known content
        let file1 = temp_dir.path().join("file1.txt");
        let file2 = temp_dir.path().join("file2.txt");
        let file3 = temp_dir.path().join("file3.txt");

        std::fs::write(&file1, "file1")?;
        std::fs::write(&file2, "file2")?;
        std::fs::write(&file3, "file3")?;

        // Get paths in random order
        let paths = vec![&file2, &file3, &file1];

        // Sort by inode
        let sorted = if parallel {
            paths.into_iter().sorted_by_inode().collect::<Vec<_>>()
        } else {
            paths.into_iter().sorted_by_inode().collect::<Vec<_>>()
        };

        // Verify files exist and are readable
        for path in &sorted {
            assert!(path.exists());
            let content = std::fs::read_to_string(path)?;
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

        Ok(())
    }

    #[test]
    fn test_paths_from_file() -> std::io::Result<()> {
        // Create temp file with list of paths
        let temp_file = tempfile::NamedTempFile::new()?;
        let temp_dir = tempfile::tempdir()?;

        // Create test files
        let file1 = temp_dir.path().join("file1.txt");
        let file2 = temp_dir.path().join("file2.txt");
        std::fs::write(&file1, "test1")?;
        std::fs::write(&file2, "test2")?;

        // Write paths to temp file
        writeln!(temp_file.as_file(), "{}", file1.display())?;
        writeln!(temp_file.as_file(), "{}", file2.display())?;
        writeln!(temp_file.as_file(), "invalid/path")?;

        // Read paths from file
        let paths = temp_file.path().read_paths()?.collect::<Vec<_>>();

        // First two paths should be valid
        assert!(paths[0].as_ref().unwrap().exists());
        assert!(paths[1].as_ref().unwrap().exists());

        // Third path should be invalid but parseable
        assert!(paths[2].as_ref().unwrap().to_str().unwrap() == "invalid/path");
        assert!(!paths[2].as_ref().unwrap().exists());

        Ok(())
    }
}
