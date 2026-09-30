# Runtime verification

Call verification explicitly after installation or at application startup. It uses
embedded synthetic DICOMs and fresh deterministic generated cases. It needs no
network connection, fixture directory, encoder executable, or Python package.
It does need writable temporary storage. Each fixture is reopened three times,
including one file load, and temporary files are removed after each case.

```rust
let report = dicom_preprocessing::verify_runtime();
if !report.passed {
    // Inspect report.cases and report.codecs before accepting this installation.
}
```

```python
import dicom_preprocessing as dp

report = dp.verify_runtime()
if not report["passed"]:
    failures = [case["id"] for case in report["cases"] if not case["passed"]]
```

```typescript
import { verifyRuntime } from '@medcognetics/dicom-preprocessing'

const report = verifyRuntime()
if (!report.passed) {
  const failures = report.cases.filter(test => !test.passed).map(test => test.id)
}
```

Calls are synchronous and uncached. Python releases the GIL for shared Rust checks
and reacquires it for callbacks. Node applications can run verification in a worker.
The report omits timestamps and durations so repeated runs can be compared directly.

## Decoders

With the pinned dependencies, the default report passes. This crate supplies its
own readers for two families of transfer syntaxes, because the dicom-rs built-in
readers decode some inputs incorrectly (#128):

- JPEG Baseline, Extended, and Lossless (`.4.50`, `.4.51`, `.4.57`, `.4.70`) use
  libjpeg-turbo 3.1 through `turbojpeg-sys`: 8-bit and 12-bit DCT (grayscale and
  color) and 2-16-bit lossless with all predictors. Three-component images are
  converted to RGB. Corrupt-data warnings are errors. A lossless restart interval
  that is not a whole number of rows is rejected, as ITU-T T.81 requires.
- RLE Lossless (`1.2.840.10008.1.2.5`) uses an in-crate reader with bounds-checked
  segments. The dicom-rs reader shifted 8-bit monochrome samples by one byte and
  swapped the bytes of 16-bit RGB samples.

These readers are registered with the dicom-rs transfer syntax registry, which only
lets a submission replace a stub. This crate therefore builds `dicom-pixeldata`
without its `native`, `jpeg`, and `rle` features. A consumer whose dependency graph
enables any of them, for example through `dicom-pixeldata` default features or the
`dicom` crate's `image`, `ndarray`, or `pixeldata` features, keeps the built-in
readers. The `builtin/decoder-registration` case then fails, and each overridden
transfer syntax's codec coverage fails with it. Rust consumers should depend on
`dicom-pixeldata` with `default-features = false`.

If a case fails, repair the decoder or the dependency features; do not relax the
fixture expectations. Applications must decide how to handle a failed report.

## Add a shared-library fixture

A fixture supplies serialized DICOM bytes, its transfer syntax UID, primitive tag
expectations, and every frame's raw pixel expectations. A tag path uses numeric
group/element pairs. For a sequence, intermediate components also specify a
zero-based `item` index. The final component selects a primitive element.

Tag `values` use the parser's canonical string representation, including multiple
values in order. Pixel `values` use the raw decoder layout: planar RGB remains
planar when the production decoder returns it that way. Values are integers;
`sample_type` supports `u8`, `i8`, `u16`, and `i16`. `absolute_tolerance` defaults to
zero. A nonzero tolerance applies only to expected-pixel comparisons; repeated
loads must still match exactly. Every case must include tags and frames.

For example, an application with a synthetic 2-by-1 unsigned 8-bit DICOM containing
samples `[10, 20]` can define:

```python
from pathlib import Path
import dicom_preprocessing as dp

case: dp.VerificationCase = {
    "id": "myapp/two-pixels",
    "dicom_bytes": Path("two-pixels.dcm").read_bytes(),
    "transfer_syntax_uid": "1.2.840.10008.1.2.1",
    "tags": [{"path": [{"group": 0x28, "element": 0x11}],
              "vr": "US", "values": ["2"]}],
    "frames": [{"width": 2, "height": 1, "samples_per_pixel": 1,
                "planar_configuration": 0, "sample_type": "u8", "values": [10, 20]}],
}
report = dp.verify_runtime(cases=[case])
```

```typescript
import { readFileSync } from 'node:fs'
import { verifyRuntime, type VerificationCase } from '@medcognetics/dicom-preprocessing'

const fixture: VerificationCase = {
  id: 'myapp/two-pixels', dicomBytes: readFileSync('two-pixels.dcm'),
  transferSyntaxUid: '1.2.840.10008.1.2.1',
  tags: [{ path: [{ group: 0x28, element: 0x11 }], vr: 'US', values: ['2'] }],
  frames: [{ width: 2, height: 1, samplesPerPixel: 1, planarConfiguration: 0,
             sampleType: 'u8', values: [10, 20] }],
}
const report = verifyRuntime({ cases: [fixture] })
```

```rust
use dicom_preprocessing::verification::*;

let case = VerificationCase {
    id: "myapp/two-pixels".into(),
    dicom_bytes: std::fs::read("two-pixels.dcm")?,
    transfer_syntax_uid: "1.2.840.10008.1.2.1".into(),
    tags: vec![TagExpectation {
        path: vec![TagPathComponent { group: 0x28, element: 0x11, item: None }],
        vr: "US".into(), values: vec!["2".into()],
    }],
    frames: vec![FrameExpectation {
        width: 2, height: 1, samples_per_pixel: 1, planar_configuration: 0,
        sample_type: SampleType::U8, values: vec![10, 20], absolute_tolerance: 0,
    }],
};
let report = verify_runtime_with(&VerificationOptions {
    cases: vec![case], ..Default::default()
})?;
```

## Add custom checks and declare codec coverage

Custom tests are trusted synchronous callbacks. Each returns a nonempty list of
uniquely named checks. They run once in declaration order, after shared checks.
They own any repetition, temporary resources, and application-specific assertions.
Codec declarations map each transfer syntax UID to its required case IDs. They do
not install or register codecs. IDs are unique across extension cases, tests, and
codec declarations, have a namespace such as `myapp/`, and cannot use `builtin/`.

```python
import dicom_preprocessing as dp

def check_application_decoder():
    # Replace this example assertion with your integration's independent oracle.
    actual = [10, 20]
    return [{"name": "expected_pixels", "passed": actual == [10, 20]}]

report = dp.verify_runtime(
    tests=[{"id": "myapp/decode", "run": check_application_decoder}],
    codecs=[{"id": "myapp/codec", "required_cases": {"1.2.3": ["myapp/decode"]}}],
)
```

```typescript
const report = verifyRuntime({
  tests: [{ id: 'myapp/decode', run: () => {
    const actual = [10, 20] // Replace with your application's decoder output.
    return [{ name: 'expected_pixels', passed: actual[0] === 10 && actual[1] === 20 }]
  } }],
  codecs: [{ id: 'myapp/codec', requiredCases: { '1.2.3': ['myapp/decode'] } }],
})
```

```rust
use std::collections::BTreeMap;
use dicom_preprocessing::verification::*;

struct ApplicationCheck;
impl VerificationTest for ApplicationCheck {
    fn id(&self) -> &str { "myapp/decode" }
    fn run(&self) -> Result<Vec<VerificationCheck>, String> {
        let actual = [10, 20]; // Replace with your application's decoder output.
        Ok(vec![VerificationCheck::new("expected_pixels", actual == [10, 20])])
    }
}
let test = ApplicationCheck;
let report = verify_runtime_with(&VerificationOptions {
    tests: vec![&test],
    codecs: vec![CodecDeclaration {
        id: "myapp/codec".into(),
        required_cases: BTreeMap::from([("1.2.3".into(), vec!["myapp/decode".into()])]),
    }],
    ..Default::default()
})?;
```

A custom test produces `path: caller_integration`. Its successful codec declaration
has `shared_library_verified: false` (`sharedLibraryVerified` in Node). Only a
successful shared fixture for that exact UID establishes shared-library coverage.
A declaration may reference built-in fixtures as well as extension fixtures and
tests. If linking another codec adds a production decoder to the registry, provide
shared fixtures for its UID: custom callbacks alone cannot satisfy the built-in
production-decoder inventory.

Invalid declarations fail before any checks run: Rust returns
`VerificationConfigurationError`; bindings raise an argument/type exception.
Test failures are report data. Ordinary callback exceptions become generic failure
entries and do not stop later callbacks. Node rejects unknown enumerable top-level
option names before reading fixtures or running callbacks. Python interruption exceptions propagate;
Node Promise results fail and rejected promises are consumed to avoid an unhandled
rejection. Rust callbacks must not panic. Explicit callback diagnostics are limited
to 256 characters; callers must keep them free of sensitive data.

## Coverage and maintenance

The embedded corpus covers implicit/explicit little endian, big endian, dataset
deflate, encapsulated native pixels, frame deflate, RLE, JPEG baseline/extended/
lossless, JPEG 2000, and HTJ2K. JPEG 2000 Part 2 cases exercise the compatible
single-component subset, not every Part 2 transform. JPEG Extended covers 12-bit
single-block and multi-block images with partial edge blocks. Generated cases add
signed 16-bit data, interleaved/planar RGB, unsigned 16-bit monochrome and RGB RLE,
and multiframe native/RLE/JPEG data.

Preprocessing checks use independent sample expectations for display conversion,
caller-selected flips, nearest-neighbor resize, centered zero padding, frame order,
and maximum-intensity projection, including serial/parallel repeatability.
This is a bounded installation test, not exhaustive DICOM conformance testing.
It cannot contain native crashes or sandbox callbacks.

See [fixture provenance](../fixtures/verification/README.md) for regeneration.
Review every expected-value change independently. Advance `suite_version` when
changing the corpus or check semantics, and `schema_version` for report-shape changes.
`make quality` and `make test` cover Rust and both bindings. Minimum-version CI
runs the same checks; cross-platform CI runs Rust verification and installed Node
verification. Wheel and npm package checks call the suite outside the source tree.
