from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt

class Preprocessor:
    """Configuration for DICOM preprocessing.

    Args:
        crop: Whether to crop zero-valued borders
        size: Target size as (width, height) tuple
        filter: Interpolation filter for resizing. One of: nearest, triangle, catmull, gaussian, lanczos3
        padding_direction: Direction to pad when aspect ratio doesn't match target. One of: zero, center, edge
        crop_max: Whether to crop to maximum possible size
        volume_handler: How to handle multi-frame volumes. One of: keep, central, interpolate.
            - keep: Keep all frames
            - central: Keep the central frame
            - interpolate: Interpolate to `target_frames` using linear interpolation
        use_components: Whether to use color components for cropping
        use_padding: Whether to pad to target size
        border_frac: Optional fraction of border to keep when cropping
        target_frames: Target number of frames when using interpolation

    Raises:
        ValueError: If invalid filter type, padding direction or volume handler specified
    """

    def __init__(
        self,
        crop: bool = True,
        size: Optional[Tuple[int, int]] = None,
        filter: str = "triangle",
        padding_direction: str = "zero",
        crop_max: bool = True,
        volume_handler: str = "keep",
        use_components: bool = True,
        use_padding: bool = True,
        border_frac: Optional[float] = None,
        target_frames: int = 32,
    ) -> None: ...

def get_frame_count(path: Union[str, Path]) -> int:
    """Get the number of frames in a TIFF file.

    Args:
        path: Path to the TIFF file

    Returns:
        Number of frames in the TIFF file

    Raises:
        FileNotFoundError: If file cannot be found
        IOError: If file cannot be opened
        RuntimeError: If frame count cannot be determined
    """
    ...

def load_tiff_u8(path: Union[str, Path], frames: Optional[Sequence[int]] = None) -> npt.NDArray[np.uint8]:
    """Load a TIFF file as an unsigned 8-bit numpy array.
    If the TIFF is of a different bit depth, it will be scaled to 8-bit.

    Args:
        path: Path to the TIFF file
        frames: Optional sequence of frame indices to load. If None, loads all frames.

    Returns:
        4D array with shape :math:`(N, H, W, C)`

    Raises:
        FileNotFoundError: If file cannot be found
        IOError: If file cannot be opened
        RuntimeError: If TIFF decoding fails
    """
    ...

def load_tiff_u16(path: Union[str, Path], frames: Optional[Sequence[int]] = None) -> npt.NDArray[np.uint16]:
    """Load a TIFF file as an unsigned 16-bit numpy array.
    If the TIFF is of a different bit depth, it will be scaled to 16-bit.

    Args:
        path: Path to the TIFF file
        frames: Optional sequence of frame indices to load. If None, loads all frames.

    Returns:
        4D array with shape :math:`(N, H, W, C)`

    Raises:
        FileNotFoundError: If file cannot be found
        IOError: If file cannot be opened
        RuntimeError: If TIFF decoding fails
    """
    ...

def load_tiff_f32(path: Union[str, Path], frames: Optional[Sequence[int]] = None) -> npt.NDArray[np.float32]:
    """Load a TIFF file as a 32-bit floating-point numpy array.
    Inputs are scaled to the range :math:`[0, 1]` according to the source bit depth.

    Args:
        path: Path to the TIFF file
        frames: Optional sequence of frame indices to load. If None, loads all frames.

    Returns:
        4D array with shape :math:`(N, H, W, C)`

    Raises:
        FileNotFoundError: If file cannot be found
        IOError: If file cannot be opened
        RuntimeError: If TIFF decoding fails
    """
    ...

def preprocess_u8(
    path: Union[str, Path], preprocessor: Optional[Preprocessor] = None, parallel: bool = False
) -> npt.NDArray[np.uint8]:
    """Preprocess a DICOM file and return as 8-bit unsigned integer array.

    Args:
        path: Path to DICOM file
        preprocessor: Optional preprocessing configuration
        parallel: Whether to use parallel processing for multi-frame targets

    Returns:
        4D array with shape :math:`(N, H, W, C)`

    Raises:
        FileNotFoundError: If file cannot be found
        RuntimeError: If preprocessing fails
    """
    ...

def preprocess_u16(
    path: Union[str, Path], preprocessor: Optional[Preprocessor] = None, parallel: bool = False
) -> npt.NDArray[np.uint16]:
    """Preprocess a DICOM file and return as 16-bit unsigned integer array.

    Args:
        path: Path to DICOM file
        preprocessor: Optional preprocessing configuration
        parallel: Whether to use parallel processing for multi-frame targets

    Returns:
        4D array with shape :math:`(N, H, W, C)`

    Raises:
        FileNotFoundError: If file cannot be found
        RuntimeError: If preprocessing fails
    """
    ...

def preprocess_f32(
    path: Union[str, Path], preprocessor: Optional[Preprocessor] = None, parallel: bool = False
) -> npt.NDArray[np.float32]:
    """Preprocess a DICOM file and return as 32-bit floating point array.
    Values are scaled to the range :math:`[0, 1]`.

    Args:
        path: Path to DICOM file
        preprocessor: Optional preprocessing configuration
        parallel: Whether to use parallel processing for multi-frame targets

    Returns:
        4D array with shape :math:`(N, H, W, C)`

    Raises:
        FileNotFoundError: If file cannot be found
        RuntimeError: If preprocessing fails
    """
    ...

def preprocess_stream_u8(
    buffer: bytes, preprocessor: Optional[Preprocessor] = None, parallel: bool = False
) -> npt.NDArray[np.uint8]:
    """Preprocess a DICOM file from a bytes buffer and return as 8-bit unsigned integer array.

    Args:
        buffer: DICOM file contents as bytes
        preprocessor: Optional preprocessing configuration
        parallel: Whether to use parallel processing for multi-frame targets

    Returns:
        4D array with shape :math:`(N, H, W, C)`

    Raises:
        RuntimeError: If preprocessing fails
        ValueError: If buffer is not contiguous
    """
    ...

def preprocess_stream_u16(
    buffer: bytes, preprocessor: Optional[Preprocessor] = None, parallel: bool = False
) -> npt.NDArray[np.uint16]:
    """Preprocess a DICOM file from a bytes buffer and return as 16-bit unsigned integer array.

    Args:
        buffer: DICOM file contents as bytes
        preprocessor: Optional preprocessing configuration
        parallel: Whether to use parallel processing for multi-frame targets

    Returns:
        4D array with shape :math:`(N, H, W, C)`

    Raises:
        RuntimeError: If preprocessing fails
        ValueError: If buffer is not contiguous
    """
    ...

def preprocess_stream_f32(
    buffer: bytes, preprocessor: Optional[Preprocessor] = None, parallel: bool = False
) -> npt.NDArray[np.float32]:
    """Preprocess a DICOM file from a bytes buffer and return as 32-bit floating point array.
    Values are scaled to the range :math:`[0, 1]`.

    Args:
        buffer: DICOM file contents as bytes
        preprocessor: Optional preprocessing configuration
        parallel: Whether to use parallel processing for multi-frame targets

    Returns:
        4D array with shape :math:`(N, H, W, C)`

    Raises:
        RuntimeError: If preprocessing fails
        ValueError: If buffer is not contiguous
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

def load_tiff_f32_batched(
    paths: List[Path], batch_size: int, frames: Optional[Sequence[int]] = None
) -> Iterator[List[npt.NDArray[np.float32]]]:
    """Iterate over a list of TIFF file paths, loading and returning them in batches.

    Batches are loaded using parallel threads.

    Args:
        paths: List of paths to TIFF files
        batch_size: Number of files to load in each batch
        frames: Optional sequence of frame indices to load from each file. If None, loads all frames.

    Yields:
        A batch of TIFF files as 32-bit floating-point numpy arrays
    """
    ...
