# /// script
# requires-python = ">=3.10"
# dependencies = ["pydicom==3.0.2", "numpy==2.4.6", "imagecodecs==2026.3.6"]
# ///
"""Regenerate synthetic verification fixtures. Requires OpenJPH ojph_compress 0.27.4.

Run with uv run --script scripts/ci/generate_verification_fixtures.py.
No patient data or downloaded DICOMs are used. Expected samples originate here,
not from the Rust decoder under test.
"""

import io
import json
import struct
import subprocess
import tempfile
import zlib
from pathlib import Path

import imagecodecs
import numpy as np
import pydicom
from pydicom._uid_dict import UID_dictionary
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.encaps import encapsulate
from pydicom.uid import UID

UID_dictionary["1.2.840.10008.1.2.8.1"] = (
    "Deflated Image Frame Compression",
    "Transfer Syntax",
    "",
    "",
    "DeflatedImageFrameCompression",
)

ROOT = Path(__file__).resolve().parents[2] / "fixtures" / "verification"
PREFIX = "1.2.840.10008.1.2"


def lossless_restart():
    """Minimal SOF3 predictor-1 stream: restart interval 4 resets prediction at each row."""
    values = [32768, 32770, 32774, 32775, 32772, 32771, 32768, 32777, 32780, 32775, 32776, 32781]

    def segment(marker, payload):
        return bytes([255, marker]) + struct.pack(">H", len(payload) + 2) + payload

    data = b"\xff\xd8"
    data += segment(0xC4, b"\x00" + bytes([0, 0, 0, 0, 17] + [0] * 11) + bytes(range(17)))
    data += segment(0xC3, struct.pack(">BHHB", 16, 3, 4, 1) + b"\x01\x11\x00")
    data += segment(0xDD, struct.pack(">H", 4))
    data += segment(0xDA, b"\x01\x01\x00\x01\x00\x00")
    for start in range(0, len(values), 4):
        bits = ""
        for i in range(start, start + 4):
            predictor = 32768 if i == start else values[i - 4] if i % 4 == 0 else values[i - 1]
            delta = values[i] - predictor
            size = abs(delta).bit_length()
            bits += f"{size:05b}"
            if size:
                bits += f"{delta if delta >= 0 else delta + (1 << size) - 1:0{size}b}"
        bits += "1" * (-len(bits) % 8)
        entropy = bytes(int(bits[i : i + 8], 2) for i in range(0, len(bits), 8))
        data += entropy.replace(b"\xff", b"\xff\x00")
        if start + 4 < len(values):
            data += bytes([255, 0xD0 + start // 4])
    return data + b"\xff\xd9", np.array(values, dtype=np.uint16).reshape(3, 4)


def rle(frame):
    """Encode one frame as RLE Lossless (PS3.5 Annex G).

    Each sample byte is one segment, most significant byte first. Every row is one PackBits
    literal run: DICOM runs must not cross a row boundary.
    """
    frame = frame.reshape(frame.shape[0], frame.shape[1], -1)
    width = frame.dtype.itemsize
    segments = []
    for sample in range(frame.shape[2]):
        for byte in reversed(range(width)):
            plane = ((frame[..., sample].astype(np.uint32) >> (8 * byte)) & 0xFF).astype("u1")
            segments.append(b"".join(bytes([len(row) - 1]) + row.tobytes() for row in plane))
    offsets, position = [], 64
    for segment in segments:
        offsets.append(position)
        position += len(segment)
    payload = b"".join(segments)
    payload += b"\x00" * (len(payload) % 2)
    return struct.pack("<16I", len(segments), *offsets, *([0] * (15 - len(offsets)))) + payload


def tag(group, element, vr, values, parent=None):
    path = [{"group": group, "element": element}]
    if parent is not None:
        path.insert(0, {"group": parent[0], "element": parent[1], "item": 0})
    return {"path": path, "vr": vr, "values": [str(value) for value in values]}


# Tag expectations shared by every fixture; Rows and Columns come from each case.
SHARED_TAGS = [
    tag(0x10, 0x10, "PN", ["SYNTHETIC^VERIFICATION"]),
    tag(0x28, 0x30, "DS", ["0.5", "0.75"]),
    tag(0x28, 0x1052, "DS", ["0"]),
    tag(0x28, 0x1053, "DS", ["1"]),
    tag(0x28, 0x1050, "DS", ["128"]),
    tag(0x28, 0x1051, "DS", ["256"]),
    tag(0x8, 0x100, "SH", ["SELFTEST"], (0x8, 0x1032)),
]


class Pixels:
    """Expected sample arrays, each stored once in the manifest and referenced by name."""

    def __init__(self):
        self.arrays = {}

    def name(self, preferred, values):
        values = [int(value) for value in values]
        for name, existing in self.arrays.items():
            if existing == values:
                return name
        name, suffix = preferred, 2
        while name in self.arrays:
            name, suffix = f"{preferred}-{suffix}", suffix + 1
        self.arrays[name] = values
        return name


def fixture(pixels_index, name, uid, pixels, compressed=None, tolerance=0, planar=False, bits_stored=None):
    """Write one fixture and return its manifest case.

    `pixels` is (rows, cols), (frames, rows, cols), or (frames, rows, cols, 3). `compressed`
    is one encapsulated fragment or a list with one fragment per frame. Expected values are
    each frame's raw sample order, which is planar when `planar` is set.
    """
    frames = pixels[None] if pixels.ndim == 2 else pixels
    samples = 3 if frames.ndim == 4 else 1
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = UID(uid)
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
    meta.MediaStorageSOPInstanceUID = "1.2.826.0.1.3680043.10.543.1"
    meta.ImplementationClassUID = "1.2.826.0.1.3680043.10.543"
    ds = FileDataset(None, {}, file_meta=meta, preamble=bytes(128))
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.PatientName = "SYNTHETIC^VERIFICATION"
    ds.PatientID = "NO-PATIENT-DATA"
    ds.Rows, ds.Columns = frames.shape[1:3]
    ds.SamplesPerPixel = samples
    ds.PhotometricInterpretation = "RGB" if samples == 3 else "MONOCHROME2"
    if samples == 3:
        ds.PlanarConfiguration = int(planar)
    ds.NumberOfFrames = str(len(frames))
    ds.BitsAllocated = frames.dtype.itemsize * 8
    ds.BitsStored = bits_stored or ds.BitsAllocated
    ds.HighBit = ds.BitsStored - 1
    ds.PixelRepresentation = int(frames.dtype.kind == "i")
    ds.PixelSpacing = ["0.5", "0.75"]
    ds.RescaleSlope, ds.RescaleIntercept = "1", "0"
    ds.WindowCenter, ds.WindowWidth = "128", "256"
    item = Dataset()
    item.CodeValue, item.CodingSchemeDesignator, item.CodeMeaning = "SELFTEST", "99TEST", "Synthetic test"
    ds.ProcedureCodeSequence = [item]
    raw_frames = frames.transpose(0, 3, 1, 2) if planar else frames
    raw = raw_frames.astype(frames.dtype.newbyteorder(">" if uid == PREFIX + ".2" else "<")).tobytes()
    if compressed is not None:
        ds.PixelData = encapsulate(compressed if isinstance(compressed, list) else [compressed])
        ds[0x7FE00010].is_undefined_length = True
    else:
        ds.PixelData = raw
    output = io.BytesIO()
    pydicom.dcmwrite(
        output, ds, enforce_file_format=True, little_endian=uid != PREFIX + ".2", implicit_vr=uid == PREFIX
    )
    (ROOT / f"{name}.dcm").write_bytes(output.getvalue())
    sample_type = f"{'i' if ds.PixelRepresentation else 'u'}{ds.BitsAllocated}"
    return {
        "id": f"builtin/static-{name}",
        "file": f"{name}.dcm",
        "transfer_syntax_uid": uid,
        "rows": ds.Rows,
        "columns": ds.Columns,
        "frames": [
            {
                "samples_per_pixel": samples,
                "planar_configuration": int(planar),
                "sample_type": sample_type,
                "pixels": pixels_index.name(f"{name}-{index}" if len(frames) > 1 else name, frame.ravel()),
                "absolute_tolerance": tolerance,
            }
            for index, frame in enumerate(raw_frames)
        ],
    }


def write_manifest(pixels_index, cases):
    """Write compact JSON: one line per pixel array and per case, for small embedded size."""

    def line(value):
        return json.dumps(value, separators=(",", ":"))

    lines = ["{", '"schema_version":1,', f'"tags":{line(SHARED_TAGS)},', '"pixels":{']
    arrays = list(pixels_index.arrays.items())
    lines += [
        f"{line(name)}:{line(values)}{',' if i < len(arrays) - 1 else ''}" for i, (name, values) in enumerate(arrays)
    ]
    lines += ["},", '"cases":[']
    lines += [f"{line(case)}{',' if i < len(cases) - 1 else ''}" for i, case in enumerate(cases)]
    lines += ["]", "}"]
    (ROOT / "manifest.json").write_text("\n".join(lines) + "\n")


def jpeg_tolerance(frames, streams):
    """Tolerance just above libjpeg-turbo's own decode error against the source samples."""
    error = max(
        int(np.abs(imagecodecs.jpeg8_decode(stream).astype(int) - frame.astype(int)).max())
        for frame, stream in zip(frames, streams)
    )
    return error + 1


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    pixels = np.arange(64, dtype=np.uint8).reshape(8, 8) * 3
    index = Pixels()
    cases = []

    def add(name, suffix, data=pixels, compressed=None, tolerance=0, **options):
        cases.append(fixture(index, name, PREFIX + suffix, data, compressed, tolerance, **options))

    add("implicit", "")
    add("explicit", ".1")
    add("big-endian", ".2", pixels.astype(np.uint16) * 101)
    add("deflated", ".1.99")
    add("encapsulated-native", ".1.98", compressed=pixels.tobytes())
    add("deflated-frame", ".8.1", compressed=zlib.compress(pixels.tobytes(), wbits=-15))
    add("rle", ".5", compressed=rle(pixels))
    add("jpeg-baseline", ".4.50", compressed=imagecodecs.jpeg8_encode(pixels, level=95), tolerance=3)
    extended = pixels.astype(np.uint16) * 16
    add(
        "jpeg-extended",
        ".4.51",
        extended,
        imagecodecs.jpeg8_encode(extended, level=100, bitspersample=12),
        3,
        bits_stored=12,
    )
    # 5 x 3 blocks, including partial edge blocks, exercise DC prediction and cropping.
    # libjpeg-turbo's own decode differs from these samples by at most 8.
    y, x = np.mgrid[0:23, 0:37]
    blocks = (x * 53 + y * 89 + (x * y) % 7 * 5).astype(np.uint16)
    add(
        "jpeg-extended-37x23",
        ".4.51",
        blocks,
        imagecodecs.jpeg8_encode(blocks, level=95, bitspersample=12),
        9,
        bits_stored=12,
    )
    restart, restart_pixels = lossless_restart()
    add("jpeg-lossless", ".4.57", restart_pixels, restart)
    add("jpeg-lossless-sv1", ".4.70", restart_pixels, restart)
    for name, suffix, reversible in [
        ("j2k-lossless", ".4.90", True),
        ("j2k-lossy", ".4.91", False),
        ("j2k-part2-lossless", ".4.92", True),
        ("j2k-part2-lossy", ".4.93", False),
    ]:
        stream = imagecodecs.jpeg2k_encode(pixels, codecformat="J2K", reversible=reversible, resolutions=2, level=0)
        add(name, suffix, compressed=stream, tolerance=0 if reversible else 2)
    with tempfile.TemporaryDirectory() as directory:
        source, output = Path(directory) / "input.pgm", Path(directory) / "output.j2c"
        source.write_bytes(b"P5\n8 8\n255\n" + pixels.tobytes())
        for name, suffix, reversible in [
            ("htj2k-lossless", ".4.201", True),
            ("htj2k-rpcl", ".4.202", True),
            ("htj2k-lossy", ".4.203", False),
        ]:
            subprocess.run(
                [
                    "ojph_compress",
                    "-i",
                    str(source),
                    "-o",
                    str(output),
                    "-num_decomps",
                    "1",
                    "-prog_order",
                    "RPCL",
                    "-reversible",
                    "true" if reversible else "false",
                ],
                check=True,
                capture_output=True,
            )
            add(name, suffix, compressed=output.read_bytes(), tolerance=0 if reversible else 3)

    # Multi-frame, signed, and RGB layouts. Values are explicit arithmetic patterns. The three
    # multi-frame frames differ so frame order is observable; other layouts use one frame to
    # keep the embedded corpus small.
    i = np.arange(64)
    frames_u8 = np.stack([((i * 3 + f * 11) % 192).reshape(8, 8) for f in range(3)]).astype(np.uint8)
    signed = ((i * 977) % 60001 - 30000).reshape(8, 8).astype(np.int16)
    rgb = ((np.arange(192) * 5) % 251).reshape(1, 8, 8, 3).astype(np.uint8)
    # Distinct high and low bytes expose swapped RLE segments.
    u16 = (((i * 2053 + 17) * 31) % 65536).reshape(8, 8).astype(np.uint16)
    rgb_u16 = (((np.arange(192) * 2053 + 17) * 31) % 65536).reshape(8, 8, 3).astype(np.uint16)
    add("multiframe-native", ".1", frames_u8)
    add("signed-16bit", ".1", signed)
    add("rgb", ".1", rgb)
    add("planar-rgb", ".1", rgb, planar=True)
    add("rle-multiframe", ".5", frames_u8, [rle(frame) for frame in frames_u8])
    add("rle-16bit", ".5", u16, rle(u16))
    add("rle-rgb-16bit", ".5", rgb_u16[None], rle(rgb_u16))
    streams = [imagecodecs.jpeg8_encode(frame, level=100) for frame in frames_u8]
    add("jpeg-baseline-multiframe", ".4.50", frames_u8, streams, jpeg_tolerance(frames_u8, streams))
    write_manifest(index, cases)


if __name__ == "__main__":
    main()
