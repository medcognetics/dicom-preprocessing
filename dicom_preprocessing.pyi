from pathlib import Path
from typing import Dict, List

import numpy as np
import numpy.typing as npt

def load_tiff_u8(path: str) -> npt.NDArray[np.uint8]:
    """Load a TIFF file as an unsigned 8-bit numpy array.
    If the TIFF is of a different bit depth, it will be scaled to 8-bit.

    Args:
        path: Path to the TIFF file

    Returns:
        4D array with shape :math:`(N, H, W, C)`

    Raises:
        FileNotFoundError: If file cannot be found
        IOError: If file cannot be opened
        RuntimeError: If TIFF decoding fails
    """
    ...

def load_tiff_u16(path: str) -> npt.NDArray[np.uint16]:
    """Load a TIFF file as an unsigned 16-bit numpy array.
    If the TIFF is of a different bit depth, it will be scaled to 16-bit.
    Args:
        path: Path to the TIFF file

    Returns:
        4D array with shape :math:`(N, H, W, C)`

    Raises:
        FileNotFoundError: If file cannot be found
        IOError: If file cannot be opened
        RuntimeError: If TIFF decoding fails
    """
    ...

def load_tiff_f32(path: str) -> npt.NDArray[np.float32]:
    """Load a TIFF file as a 32-bit floating-point numpy array.
    Inputs are scaled to the range :math:`[0, 1]` according to the source bit depth.

    Args:
        path: Path to the TIFF file

    Returns:
        4D array with shape :math:`(N, H, W, C)`

    Raises:
        FileNotFoundError: If file cannot be found
        IOError: If file cannot be opened
        RuntimeError: If TIFF decoding fails
    """
    ...

def find_dicom_files(path: Path, spinner: bool = False) -> List[Path]:
    """Find all DICOM files in a directory recursively.

    Args:
        path: Directory path to search
        spinner: Whether to show a progress spinner

    Returns:
        List of paths to DICOM files
    """
    ...

def find_tiff_files(path: Path, spinner: bool = False) -> List[Path]:
    """Find all TIFF files in a directory recursively.

    Args:
        path: Directory path to search
        spinner: Whether to show a progress spinner

    Returns:
        List of paths to TIFF files
    """
    ...

def read_dicom_paths(path: Path, bar: bool = False) -> List[Path]:
    """Read paths to DICOM files from a text file containing one path per line.

    Args:
        path: Path to the text file
        bar: Whether to show a progress bar

    Returns:
        List of paths to DICOM files
    """
    ...

def read_tiff_paths(path: Path, bar: bool = False) -> List[Path]:
    """Read paths to TIFF files from a text file containing one path per line.

    Args:
        path: Path to the text file
        bar: Whether to show a progress bar

    Returns:
        List of paths to TIFF files
    """
    ...

def inode_sort(paths: List[Path], bar: bool = False) -> List[Path]:
    """Sort paths by inode number.

    Args:
        paths: List of paths to sort
        bar: Whether to show a progress bar

    Returns:
        Sorted list of paths
    """
    ...

class ManifestEntry:
    path: Path
    sop_instance_uid: str
    study_instance_uid: str
    inode: int
    dimensions: Dict[str, int]

    def relative_path(self, root: Path) -> Path:
        """Get the path of this entry relative to a root path"""
        ...

def get_manifest(path: Path, bar: bool = False) -> List[ManifestEntry]:
    """Gets manifest entries for preprocessed TIFF files in a directory.

    TIFF files are recursively searched for in the given directory.
    Preprocessed files are expected to be in the following directory structure:
    `{root}/{study_instance_uid}/{sop_instance_uid}.{tiff}`

    Args:
        path: Directory path to search
        bar: Whether to show a progress bar

    Returns:
        List of manifest entries
    """
    ...
