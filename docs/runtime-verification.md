# Runtime verification

`verify_runtime()` (Rust, Python) and `verifyRuntime()` (Node) run an offline self-test
of the installed build: DICOM parsing, pixel decoding for every enabled transfer syntax,
and preprocessing. Call it explicitly, for example at application startup. It needs no
network, files, temporary storage, or external tools; the synthetic fixtures are
embedded in the library.

```rust
let report = dicom_preprocessing::verify_runtime();
if !report.passed {
    // Inspect report.cases and report.codecs, and do not process studies.
}
```

```python
import dicom_preprocessing as dp

report = dp.verify_runtime()  # releases the GIL while it runs
if not report["passed"]:
    failures = [case["id"] for case in report["cases"] if not case["passed"]]
```

```typescript
import { verifyRuntime } from '@medcognetics/dicom-preprocessing'

const report = verifyRuntime() // synchronous; use a worker to keep an event loop free
if (!report.passed) {
  const failures = report.cases.filter(result => !result.passed).map(result => result.id)
}
```

## What is checked

- **Embedded fixtures** (`source` `"embedded_fixture"`): each fixture is parsed and
  decoded twice from its bytes. Checks cover the transfer syntax, primitive and sequence
  tags, frame count, frame layout, and every sample against independent expectations
  (exact for lossless, within a stated tolerance for lossy). The second load must match
  the first exactly. The corpus covers implicit and explicit little endian, big endian,
  dataset and frame deflate, encapsulated native pixels, RLE (8- and 16-bit, monochrome
  and RGB, multi-frame), JPEG Baseline, Extended (12-bit), and Lossless, JPEG 2000 (Part 1
  and compatible Part 2), and HTJ2K, plus signed 16-bit, interleaved and planar RGB, and
  multi-frame native data.
- **`builtin/preprocessing`**: display conversion, caller-selected flips,
  nearest-neighbor resize, centered padding, frame order, maximum-intensity projection,
  and serial/parallel repeatability, on the three-frame fixture.
- **`builtin/decoder-registration/<transfer syntax UID>`**: one case for each JPEG and RLE
  syntax that this crate's readers replace, checking that this crate's reader is the one
  registered. A Rust consumer that enables `dicom-pixeldata`'s `native`, `jpeg`, or `rle`
  features silently keeps the dicom-rs built-in readers, and the affected cases fail.
  Enabling only `rle`, for example, fails only the RLE case. See "JPEG and RLE Decoding"
  in the README.
- **Codec coverage** (`codecs`): every transfer syntax the registry can decode needs at
  least one passing fixture. A newly linked decoder without a fixture fails coverage. A
  syntax whose reader this crate replaces also needs its own registration case.

`passed` is true only if every case and every coverage entry passes. Failures are report
data, not exceptions. Applications decide what a failure means for them and keep their
own evidence records; the report omits timestamps so repeated runs compare equal.

`schema_version` changes when the report shape changes. `suite_version` changes when the
corpus or check semantics change.

## Maintenance

See [the fixture README](../fixtures/verification/README.md) for regeneration, the
manifest format, and the 48 KiB size budget. When a decoder is repaired or replaced,
the fixtures stay as they are: fix the decoder, not the expectations.
