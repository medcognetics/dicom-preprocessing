# Synthetic verification corpus

All DICOMs in this directory are synthetic and released under this repository's
MIT license. No clinical files or patient data were used. They are embedded in the
Rust library, Python extension, and Node module with `include_bytes!`, so the suite
needs no files at run time.

Regenerate from the repository root:

```sh
uv run --script scripts/ci/generate_verification_fixtures.py
```

The script pins pydicom 3.0.2, NumPy 2.4.6, and imagecodecs 2026.3.6. HTJ2K generation
uses OpenJPH `ojph_compress` 0.27.4, installed separately only for regeneration.
No runtime verification step invokes this script or any encoder. Regenerating must
leave existing `.dcm` bytes unchanged; review every expected-value change
independently, and never change expected samples to agree with a decoder.

## Size budget

Embedded fixtures plus `manifest.json` must stay within 48 KiB; a unit test enforces
it. Fixtures are 8 x 8 except `jpeg-extended-37x23`. Multi-frame coverage uses three
frames; other layouts use one.

## Manifest

`manifest.json` is compact JSON with one entry per line:

- `tags`: tag expectations shared by every fixture. Each case adds Rows and Columns.
- `pixels`: each distinct expected sample array once, by name, in the decoder's raw
  layout (planar stays planar).
- `cases`: one per fixture, with the file, transfer syntax UID, dimensions, and per-frame
  sample type, pixel array name, and absolute tolerance.

Pixel expectations come from explicit arithmetic patterns, not a decoder:

- RLE uses literal PackBits runs, one segment per sample byte, most significant byte
  first. The 16-bit cases use values whose high and low bytes differ.
- JPEG Lossless uses a minimal SOF3 stream with a fixed Huffman table and predictor 1;
  restart interval 4 resets prediction at each 4-pixel row.
- JPEG Baseline and Extended use imagecodecs' JPEG encoder. `jpeg-extended-37x23` spans
  5 x 3 blocks with partial edge blocks; libjpeg-turbo's own decode differs from its
  samples by at most 8, so its tolerance is 9. The multi-frame Baseline tolerance is
  one above libjpeg-turbo's own decode error.
- JPEG 2000 uses imagecodecs' OpenJPEG encoder. The Part 2 UIDs use compatible
  single-component codestreams. HTJ2K uses OpenJPH with RPCL progression.

Lossy cases compare with the source samples within their tolerances, and every case must
decode identically twice in the same process.
