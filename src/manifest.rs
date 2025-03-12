use indicatif::ParallelProgressIterator;
use itertools::Itertools;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::path::Path;
use std::path::PathBuf;

use crate::file::{default_bar, TiffFileOperations};
use crate::metadata::Dimensions;

type IOResult<T> = Result<T, std::io::Error>;

/// Data structure for tracking metadata about a single preprocessed file
#[derive(Debug, Clone)]
pub struct ManifestEntry {
    path: PathBuf,
    sop_instance_uid: String,
    study_instance_uid: String,
    dimensions: Option<Dimensions>,
}

impl AsRef<Path> for ManifestEntry {
    fn as_ref(&self) -> &Path {
        &self.path
    }
}

impl ManifestEntry {
    pub fn new<P: AsRef<Path>>(
        path: P,
        sop_instance_uid: String,
        study_instance_uid: String,
    ) -> Self {
        Self {
            path: PathBuf::from(path.as_ref()),
            sop_instance_uid,
            study_instance_uid,
            dimensions: None,
        }
    }

    pub fn path(&self) -> &PathBuf {
        &self.path
    }

    pub fn sop_instance_uid(&self) -> &str {
        &self.sop_instance_uid
    }

    pub fn study_instance_uid(&self) -> &str {
        &self.study_instance_uid
    }

    pub fn dimensions(&self) -> Option<&Dimensions> {
        self.dimensions.as_ref()
    }

    /// Get the path of this entry relative to a root path
    pub fn relative_path<P: AsRef<Path>>(&self, root: P) -> PathBuf {
        let root = PathBuf::from(root.as_ref());
        self.path.strip_prefix(root).unwrap().to_path_buf()
    }

    /// Create a manifest entry from a preprocessed file. It is assumed that the file
    /// has a path structure of `{root}/{study_instance_uid}/{sop_instance_uid}.{tiff}`
    pub fn try_from_preprocessed_file<P: AsRef<Path>>(path: P) -> IOResult<Self> {
        // Get the file stem (without extension) which should be the SOP Instance UID
        let path = PathBuf::from(path.as_ref());
        let sop_instance_uid = path
            .file_stem()
            .ok_or(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "File should have a name",
            ))?
            .to_string_lossy()
            .to_string();

        // Parent directory should be the Study Instance UID
        let study_instance_uid = path
            .parent()
            .ok_or(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "File should be in a directory",
            ))?
            .file_name()
            .ok_or(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Parent directory should have a name",
            ))?
            .to_string_lossy()
            .to_string();

        // Try to read dimensions from the TIFF file
        let dimensions = if let Ok(file) = std::fs::File::open(&path) {
            if let Ok(mut decoder) = tiff::decoder::Decoder::new(file) {
                Dimensions::try_from(&mut decoder).ok()
            } else {
                None
            }
        } else {
            None
        };

        Ok(Self {
            path,
            sop_instance_uid,
            study_instance_uid,
            dimensions,
        })
    }
}

/// Builds a manifest from a root path. The root path is expected to contain
/// directories with the structure `{root}/{study_instance_uid}/{sop_instance_uid}.{tiff}`
pub fn get_manifest<P: AsRef<Path>>(root: P) -> IOResult<Vec<ManifestEntry>> {
    let manifest = root
        .find_tiffs()?
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|p| ManifestEntry::try_from_preprocessed_file(&p))
        .collect::<IOResult<Vec<_>>>()?;

    Ok(manifest)
}

/// Builds a manifest from a root path with progress indicators. The root path is expected to contain
/// directories with the structure `{root}/{study_instance_uid}/{sop_instance_uid}.{tiff}`
pub fn get_manifest_with_progress<P: AsRef<Path>>(root: P) -> IOResult<Vec<ManifestEntry>> {
    let tiff_files = root.find_tiffs_with_spinner()?.collect::<Vec<_>>();

    let bar = default_bar(tiff_files.len() as u64);
    bar.set_message("Building manifest entries");

    let manifest = tiff_files
        .into_par_iter()
        .progress_with(bar)
        .map(|p| ManifestEntry::try_from_preprocessed_file(&p))
        .collect::<IOResult<Vec<_>>>()?;

    // Sort by (study_instance_uid, sop_instance_uid)
    let manifest = manifest
        .into_iter()
        .sorted_by(|a, b| {
            a.study_instance_uid()
                .cmp(b.study_instance_uid())
                .then(a.sop_instance_uid().cmp(b.sop_instance_uid()))
        })
        .collect::<Vec<_>>();

    Ok(manifest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;
    use std::fs;
    use tempfile::TempDir;

    fn create_test_tiff(dir: &Path, filename: &str) -> IOResult<PathBuf> {
        let path = dir.join(filename);
        fs::write(&path, b"dummy tiff content")?;
        Ok(path)
    }

    fn setup_test_dir() -> IOResult<(TempDir, Vec<PathBuf>)> {
        let temp_dir = TempDir::new()?;

        // Create study directories
        let study1_dir = temp_dir.path().join("study1");
        let study2_dir = temp_dir.path().join("study2");
        fs::create_dir(&study1_dir)?;
        fs::create_dir(&study2_dir)?;

        // Create test files
        let mut paths = Vec::new();
        paths.push(create_test_tiff(&study1_dir, "image1.tiff")?);
        paths.push(create_test_tiff(&study1_dir, "image2.tiff")?);
        paths.push(create_test_tiff(&study2_dir, "image3.tiff")?);

        Ok((temp_dir, paths))
    }

    #[rstest]
    #[case("study1", "image1", "image1.tiff")]
    #[case("study2", "image", "image.tiff")]
    fn test_manifest_entry_from_file(
        #[case] study_uid: &str,
        #[case] sop_uid: &str,
        #[case] filename: &str,
    ) -> IOResult<()> {
        let temp_dir = TempDir::new()?;
        let study_dir = temp_dir.path().join(study_uid);
        fs::create_dir(&study_dir)?;
        let file_path = create_test_tiff(&study_dir, filename)?;

        let entry = ManifestEntry::try_from_preprocessed_file(&file_path)?;

        assert_eq!(entry.path, file_path);
        assert_eq!(entry.study_instance_uid, study_uid);
        assert_eq!(entry.sop_instance_uid, sop_uid);
        Ok(())
    }

    #[test]
    fn test_get_manifest() -> IOResult<()> {
        let (temp_dir, paths) = setup_test_dir()?;

        let manifest = get_manifest(temp_dir.path())?;

        assert_eq!(manifest.len(), 3);
        assert!(paths.iter().all(|p| manifest.iter().any(|e| e.path == *p)));
        Ok(())
    }

    #[test]
    fn test_get_manifest_with_progress() -> IOResult<()> {
        let (temp_dir, paths) = setup_test_dir()?;

        let manifest = get_manifest_with_progress(temp_dir.path())?;

        assert_eq!(manifest.len(), 3);
        assert!(paths.iter().all(|p| manifest.iter().any(|e| e.path == *p)));
        Ok(())
    }
}
