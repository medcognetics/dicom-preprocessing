# Decoder comparison

Compares this crate's JPEG (libjpeg-turbo), JPEG 2000 (multithreaded OpenJPEG), and RLE
readers with the dicom-rs built-in readers (jpeg-decoder 0.3.2, single-threaded OpenJPEG, and
the RLE Lossless adapter) on the same DICOM objects. JPEG 2000 inputs run once on one thread
and once with the default thread count. Each
reader decodes every frame through `PixelDataReader::decode_frame`. The report gives median
and minimum wall time, and the largest sample difference between the two outputs.

This crate is not a workspace member. It enables the built-in readers, which the library must
never do.

```sh
uv run --script benches/decoder_comparison/generate.py /tmp/decoder-data
cargo run --release --manifest-path benches/decoder_comparison/Cargo.toml -- /tmp/decoder-data 15
```

`CHECKSUMS=1` prints a hash of this crate's output for each file instead. Use it to compare
builds, for example with and without NASM.

## Inputs

`generate.py` needs libjpeg-turbo `cjpeg`; JPEG 2000 and HTJ2K inputs come from imagecodecs
(OpenJPEG and OpenJPH encoders). It writes synthetic images: a 3328 x 4096 12-bit
breast-shaped region with smooth texture and noise, and a 30-frame 1024 x 768 RGB cine.

| File | Encoding |
| --- | --- |
| `mammo-baseline-8bit` | JPEG Baseline, 8-bit grayscale, quality 90 |
| `mammo-extended-12bit` | JPEG Extended, 12-bit, quality 90 |
| `mammo-lossless-sv1-12bit`, `-16bit` | JPEG Lossless SV1, 12-bit and 16-bit precision |
| `mammo-rle-16bit`, `-8bit` | RLE Lossless (pylibjpeg-rle encoder) |
| `us-baseline-rgb-30f` | JPEG Baseline, YCbCr 4:2:0, 30 frames |
| `mammo-j2k-lossless-12bit` | JPEG 2000 reversible, 6 resolutions |
| `mammo-j2k-lossy-12bit` | JPEG 2000 irreversible, about 15:1 |
| `mammo-htj2k-lossless-12bit` | HTJ2K reversible |
| `dbt-j2k-lossless-30f` | JPEG 2000 reversible, 30 frames of 1024 x 1024 |

## Results (2026-09-29)

AMD Ryzen Threadripper 3960X, Rust 1.97.1, release build, 15 runs. Times are medians in ms.

| File | Built-in | Ours, no NASM | Ours, NASM |
| --- | --- | --- | --- |
| `mammo-baseline-8bit` | 38.2 | 38.9 | 23.9 |
| `mammo-extended-12bit` | fails | 77.0 | 76.8 |
| `mammo-lossless-sv1-12bit` | 213.0 | 92.3 | 89.8 |
| `mammo-lossless-sv1-16bit` | 220.7 | 103.7 | 102.4 |
| `mammo-rle-16bit` | 50.3 | 35.6 | 35.4 |
| `mammo-rle-8bit` | 26.9 | 11.8 | 12.3 |
| `us-baseline-rgb-30f` | 233.1 | 247.8 | 134.5 |

- jpeg-decoder uses several threads for large images. With `RAYON_NUM_THREADS=1`, the
  built-in cine time rises to 337 ms and this crate's does not change.
- Output differences: lossless JPEG and 16-bit RLE are identical. Lossy JPEG differs by at
  most 1 (grayscale) and 3 (RGB) from IDCT and color conversion rounding. The built-in 8-bit
  RLE output is wrong by up to 173 because of its byte shift.
- This crate's output was byte-identical with and without NASM for every input.

### JPEG 2000 (2026-09-30)

Same machine, 7 runs, medians in ms. The default is 8 threads per frame.

| File | Built-in (1 thread) | Ours, 1 thread | Ours, 8 threads |
| --- | --- | --- | --- |
| `mammo-j2k-lossless-12bit` | 1087.7 | 1062.1 | 175.9 |
| `mammo-j2k-lossy-12bit` | 362.1 | 335.8 | 110.9 |
| `mammo-htj2k-lossless-12bit` | 178.5 | 145.1 | 54.3 |
| `dbt-j2k-lossless-30f`, frames decoded one after another | 4137.0 | 4110.6 | 690.6 |

- Output is identical to the built-in reader at every thread count.
- End to end, `dicom-preprocess -s 2048,1664 -c` on the lossless mammogram takes 354 ms with
  8 threads and 1229 ms with 1. On the 30-frame volume it is unchanged (1495 and 1530 ms),
  because frames already decode in parallel.
- Pure-Rust alternatives were slower than single-threaded OpenJPEG on the lossless mammogram:
  hayro-jpeg2000 0.4.0 took 1328 ms (and does not decode HTJ2K), and dicom-toolkit-jpeg2000
  0.5.0 took 1748 ms (and clamps signed samples).
