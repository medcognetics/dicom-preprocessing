# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pydicom==3.0.2", "numpy==2.4.6", "pylibjpeg==2.1.0", "pylibjpeg-rle==2.2.0", "imagecodecs==2026.3.6"
# ]
# ///
"""Generate synthetic DICOM inputs for the decoder comparison. Requires libjpeg-turbo `cjpeg`.

Run with `uv run --script benches/decoder_comparison/generate.py <OUTPUT_DIR>`.
Images are synthetic: a breast-shaped region with smooth texture and noise on a zero background.
"""

import subprocess
import sys
from pathlib import Path

import imagecodecs
import numpy as np
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.encaps import encapsulate
from pydicom.uid import ExplicitVRLittleEndian, RLELossless, generate_uid

OUTPUT = Path(sys.argv[1])
OUTPUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(1234)


def mammo(rows=4096, cols=3328):
    y, x = np.mgrid[0:rows, 0:cols].astype(np.float32)
    # Breast-shaped region on a zero background, smooth tissue texture plus quantum noise.
    inside = ((x / (cols * 0.75)) ** 2 + ((y - rows / 2) / (rows * 0.48)) ** 2) < 1
    tissue = 1800 + 600 * np.sin(x / 97) * np.cos(y / 131) + 300 * np.sin((x + y) / 23)
    img = np.where(inside, tissue + rng.normal(0, 40, (rows, cols)), 0)
    return np.clip(img, 0, 4095).astype(np.uint16)


def dataset(rows, cols, spp, bits_alloc, bits_stored, frames, ts, pi):
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ts
    ds.file_meta.MediaStorageSOPClassUID = ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID = generate_uid()
    ds.Rows, ds.Columns, ds.SamplesPerPixel = rows, cols, spp
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = bits_alloc, bits_stored, bits_stored - 1, 0
    ds.PhotometricInterpretation = pi
    ds.NumberOfFrames = frames
    if spp == 3:
        ds.PlanarConfiguration = 0
    return ds


def cjpeg(arr, args, maxval):
    rows, cols = arr.shape[:2]
    if arr.ndim == 2:
        header = f"P5\n{cols} {rows}\n{maxval}\n".encode()
    else:
        header = f"P6\n{cols} {rows}\n{maxval}\n".encode()
    data = arr.astype(">u2" if maxval > 255 else "u1").tobytes()
    return subprocess.run(["cjpeg", *args], input=header + data, capture_output=True, check=True).stdout


def save_jpeg(name, frames_arrays, args, maxval, bits_alloc, bits_stored, ts, pi):
    rows, cols = frames_arrays[0].shape[:2]
    spp = 3 if frames_arrays[0].ndim == 3 else 1
    ds = dataset(rows, cols, spp, bits_alloc, bits_stored, len(frames_arrays), ts, pi)
    ds.PixelData = encapsulate([cjpeg(a, args, maxval) for a in frames_arrays])
    ds["PixelData"].VR = "OB"
    ds["PixelData"].is_undefined_length = True
    ds.save_as(OUTPUT / f"{name}.dcm", enforce_file_format=True)
    print(name)


m = mammo()
save_jpeg(
    "mammo-lossless-sv1-12bit",
    [m],
    ["-lossless", "1", "-precision", "12"],
    4095,
    16,
    12,
    "1.2.840.10008.1.2.4.70",
    "MONOCHROME2",
)
save_jpeg(
    "mammo-lossless-sv1-16bit",
    [m * 16],
    ["-lossless", "1", "-precision", "16"],
    65535,
    16,
    16,
    "1.2.840.10008.1.2.4.70",
    "MONOCHROME2",
)
save_jpeg(
    "mammo-extended-12bit",
    [m],
    ["-precision", "12", "-quality", "90"],
    4095,
    16,
    12,
    "1.2.840.10008.1.2.4.51",
    "MONOCHROME2",
)
m8 = (m >> 4).astype(np.uint8)
save_jpeg(
    "mammo-baseline-8bit", [m8], ["-quality", "90", "-grayscale"], 255, 8, 8, "1.2.840.10008.1.2.4.50", "MONOCHROME2"
)
# Ultrasound-like RGB cine: 30 frames of 1024x768.
cine = []
for f in range(30):
    y, x = np.mgrid[0:768, 0:1024]
    g = (128 + 60 * np.sin((x + 7 * f) / 29) * np.cos(y / 37) + rng.normal(0, 12, (768, 1024))).clip(0, 255)
    cine.append(np.stack([g, g * 0.9, g * 0.8], -1).astype(np.uint8))
save_jpeg("us-baseline-rgb-30f", cine, ["-quality", "90"], 255, 8, 8, "1.2.840.10008.1.2.4.50", "YBR_FULL_422")

for name, arr, bits in [("mammo-rle-16bit", m, 12), ("mammo-rle-8bit", m8, 8)]:
    ds = dataset(arr.shape[0], arr.shape[1], 1, arr.dtype.itemsize * 8, bits, 1, ExplicitVRLittleEndian, "MONOCHROME2")
    ds.PixelData = arr.tobytes()
    ds.compress(RLELossless, encoding_plugin="pylibjpeg")
    ds.save_as(OUTPUT / f"{name}.dcm", enforce_file_format=True)
    print(name)


def save_codestreams(name, streams, rows, cols, ts):
    ds = dataset(rows, cols, 1, 16, 12, len(streams), ts, "MONOCHROME2")
    ds.PixelData = encapsulate(streams)
    ds["PixelData"].VR = "OB"
    ds["PixelData"].is_undefined_length = True
    ds.save_as(OUTPUT / f"{name}.dcm", enforce_file_format=True)
    print(name)


# JPEG 2000 (OpenJPEG encoder, 6 resolutions) and HTJ2K (OpenJPH encoder).
j2k = dict(codecformat="J2K", bitspersample=12, numthreads=1)
save_codestreams(
    "mammo-j2k-lossless-12bit",
    [imagecodecs.jpeg2k_encode(m, level=0, reversible=True, **j2k)],
    *m.shape,
    "1.2.840.10008.1.2.4.90",
)
# Level is a PSNR target; 50 gives about 15:1 on this image.
save_codestreams(
    "mammo-j2k-lossy-12bit",
    [imagecodecs.jpeg2k_encode(m, level=50, reversible=False, **j2k)],
    *m.shape,
    "1.2.840.10008.1.2.4.91",
)
save_codestreams(
    "mammo-htj2k-lossless-12bit",
    [imagecodecs.htj2k_encode(m, reversible=True)],
    *m.shape,
    "1.2.840.10008.1.2.4.201",
)
# DBT-like: 30 lossless 1024 x 1024 frames.
dbt = [m[1000 + 20 * i : 2024 + 20 * i, 500:1524].copy() for i in range(30)]
save_codestreams(
    "dbt-j2k-lossless-30f",
    [imagecodecs.jpeg2k_encode(frame, level=0, reversible=True, **j2k) for frame in dbt],
    1024,
    1024,
    "1.2.840.10008.1.2.4.90",
)
