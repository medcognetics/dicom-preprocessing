# DICOM Preprocessing

Implements a tool that preprocesses DICOM files into TIFF images. The primary motivation is to prepare DICOM images for use in computer vision tasks, with a focus on efficient storage and minimization of decode processing time.

### Building Distributable Artifacts

Run `make build` to create release artifacts for all supported Linux package surfaces:

- an executable-preserving archive of the Rust command-line binaries under `dist/rust/`
- a Python wheel under `dist/python/`
- a Node package tarball under `dist/node/`

Run `make test-build` afterward to smoke-test the Rust binaries, install the Python wheel and Node tarball in clean environments, and verify the full commit-pinned Node Git-install pathway. GitHub Actions performs this complete build nightly on the `beryl` self-hosted runner and retains the verified artifacts for 14 days.

### Transformation Sequence

Depending on the options used, the following transformations are applied to the DICOM image:

1. **Cropping** - if the `--crop` option is used, the image or volume is cropped such that all-zero rows and columns are removed
from the edges of the image.
2. **Resizing** - the image is resized to the target size, preserving the aspect ratio.
3. **Padding** - if the aspect ratio does not match the target size, the image is padded in the direction specified by the `--padding` option.

To enable mapping coordinates from the original image to the output image, the following TIFF tags will be set:
- `DefaultCropOrigin` - the origin of the initial cropping step as `(x, y)`
- `DefaultCropSize` - the size of the initial cropping step as `(width, height)`
- `DefaultScale` - the floating point scale of the resizing step as `(x, y)`
- `ActiveArea` - coordinates of the non-padded area of the image as `(left, top, right, bottom)`
- `PageNumber` - tuple of `(0, total)` indicating the total number of frames in the file

### Parallelization Flow

`dicom-preprocess` uses two levels of parallelism:
- file-level parallelism across input studies
- frame/pixel-level parallelism inside a single multi-frame volume

To avoid over-parallelizing across both levels, frame-level parallelism is only enabled when the available thread budget per input is greater than 1. Use `--threads` to cap worker threads (for example, memory-constrained hosts).

```mermaid
flowchart TD
    A["Collect input files"] --> B["Create rayon thread pool: --threads or system default"]
    B --> C{"More than one thread per input?"}
    C -->|Yes| D["Enable frame-level decode and transform parallelism"]
    C -->|No| E["Use file-level parallelism only"]
    D --> F["Process each file"]
    E --> F
    F --> G["Decode"]
    G --> H["Volume handler"]
    H --> I["Crop, resize, pad"]
    I --> J["Write TIFF"]
```


### Command Line Interface

```
Preprocess DICOM files into (multi-frame) TIFFs

Usage: dicom-preprocess [OPTIONS] <SOURCE> <OUTPUT>

Arguments:
  <SOURCE>  Source path. Can be a DICOM file, directory, or a text file with DICOM file paths
  <OUTPUT>  Output path. Can be a directory (for multiple files) or a file (for a single file)

Options:
  -c, --crop
          Crop the image. Pixels with value equal to zero are cropped away.
  -m, --crop-max
          Also include pixels with value equal to the data type's maximum value in the crop calculation
  -n, --no-components
          Do not use connected components for the crop calculation
  -b, --border-frac <BORDER_FRAC>
          Border fraction to exclude from crop calculation and grow final crop by
  -s, --size <SIZE>
          Target size (width,height)
      --spacing <SPACING>
          Target pixel/voxel spacing in mm (x,y or x,y,z)
  -f, --filter <FILTER>
          Filter type [default: triangle] [possible values: triangle, nearest, catmull-rom, gaussian, lanczos3, max-pool]
  -p, --padding <PADDING_DIRECTION>
          Padding direction [default: zero] [possible values: zero, top-left, bottom-right, center]
      --no-padding
          Disable padding
  -z, --compressor <COMPRESSOR>
          Compression type [default: packbits] [possible values: packbits, lzw, uncompressed]
  -v, --volume-handler <VOLUME_HANDLER>
          How to handle volumes [default: keep] [possible values: keep, central-slice, max-intensity, interpolate, laplacian-mip]
  -t, --target-frames <TARGET_FRAMES>
          Target number of frames when using interpolation [default: 32]
      --mip-weight <MIP_WEIGHT>
          LaplacianMip: weight for MIP Laplacian contribution (default 1.5, higher preserves calcifications better) [default: 1.5]
      --skip-frames <SKIP_FRAMES>
          LaplacianMip: frames to skip at start and end of volume (default 5, trims noisy edge frames) [default: 5]
      --projection-mode <PROJECTION_MODE>
          LaplacianMip: projection mode for computing the central frame (central-slice or parallel-beam) [default: parallel-beam]
      --strict
          Fail on input paths that are not DICOM files, or if any file processing fails
  -w, --window <WINDOW>
          Window center and width
  -j, --threads <THREADS>
          Maximum worker threads for preprocessing
  -h, --help
          Print help
  -V, --version
          Print version
```


### DICOM Validation

`dicom-validate` checks whether a single DICOM file has the metadata needed for the preprocessing path.
It strictly validates decode and TIFF-output requirements such as `PixelData`, dimensions, pixel format, `BitsAllocated`, `BitsStored`, and `HighBit`.
Optional display metadata such as windowing, VOI LUT, rescale, and spacing is reported diagnostically without making the file invalid.

Example usage:

```
dicom-validate /path/to/image.dcm
dicom-validate --format json /path/to/image.dcm
dicom-validate --decode none /path/to/image.dcm
```

Exit code `0` means the file passed validation, `1` means validation completed and found preprocessing blockers, and `2` means the tool hit a runtime error such as an unreadable path.


### Example Images

Below are example images demonstrating the effects of different cropping options (resized to 384x512):

| Original Image | Cropped (Zero Pixels) | Cropped (Zero + Maximum Pixels) |
|----------------|----------------|----------------------|
| ![Original Image](docs/nocrop.png) | ![Cropped (Zero)](docs/zerocrop.png) | ![Cropped (Zero + Max)](docs/zeromaxcrop.png) |

The maximum pixel cropping option (`-m`, `--crop-max`) prevents certain image watermarks from impacting the cropping calculation. Effective cropping can maximize the information extracted from the image at a given
resolution budget.

Below are example images demonstrating various volume handling options. Laplacian MIP uses pyramid-based fusion to combine central slice detail with calcification visibility from MIP [[1]](#references).

| Central Slice | Maximum Intensity | Laplacian MIP |
|---------------|-------------------|---------------|
| ![Central Slice](docs/central_slice.png) | ![Maximum Intensity](docs/max_intensity.png) | ![Laplacian MIP](docs/laplacian_mip.png) |
| ![Central Slice Crop](docs/central_slice_crop.png) | ![Maximum Intensity Crop](docs/max_intensity_crop.png) | ![Laplacian MIP Crop](docs/laplacian_mip_crop.png) |
| ![Central Slice Crop 2](docs/central_slice_crop2.png) | ![Maximum Intensity Crop 2](docs/max_intensity_crop2.png) | ![Laplacian MIP Crop 2](docs/laplacian_mip_crop2.png) |

The Laplacian MIP handler supports different projection modes for computing the central frame used in pyramid fusion. The default (`parallel-beam`) sums all slices along the z-axis, providing better depth integration. `central-slice` uses the middle slice directly, preserving single-slice sharpness.

For single-file multi-frame inputs, preprocessing resolves frame order from DICOM metadata before applying any volume handler. The precedence is dimension indices, stack ordinals, and then non-sampled patient geometry. Sampled/projection-style frames such as `TOMO_PROJ` are not slice-sorted from geometry alone. Multi-file `prepare_images_batch` and Python `*_slices` calls preserve caller order; Rust callers can opt into `Preprocessor::prepare_series` to validate one single-frame series, order it from patient geometry, derive center-to-center z spacing, and receive source-index provenance.

| Parallel Beam (default) | Central Slice |
|-------------------------|---------------|
| ![Parallel Beam](docs/laplacian_mip.png) | ![Central Slice](docs/laplacian_mip_central.png) |
| ![Parallel Beam Crop](docs/laplacian_mip_crop.png) | ![Central Slice Crop](docs/laplacian_mip_central_crop.png) |
| ![Parallel Beam Crop 2](docs/laplacian_mip_crop2.png) | ![Central Slice Crop 2](docs/laplacian_mip_central_crop2.png) |


### Optimization Notes

#### Compression and ZFS

Below is a comparison of file sizes for 26,474 digital breast tomosynthesis (DBT) volumes after preprocessing to TIFF when stored in ZFS.
Example decode times from a local NVMe SSD are also provided for each configuration. Note that the Rust PackBits decoder seems suboptimal,
as PackBits decoding is generally faster than LZW.


| TIFF Compression | Total Size | Total Size (LZ4 Compressed) | Peak Decode Time (ms) |
|------------------|------------|-----------------------------|-----------------------|
| Uncompressed     | 12TB       | 6.5TB                       | 3.204                 |
| Packbits         | 8.3TB      | 6.5TB                       | 67.288                |
| LZW              | 5.6TB      | 5.6TB                       | 44.080                |


PackBits compression does not yield a substantial reduction in stored file size on ZFS. The primary
determinant of compression algorithm then becomes the network bandwidth between the storage and compute nodes.
Uncompressed files will require higher bandwidth to transfer, but do not require decompression on the compute node. Furthermore, the elimination of decompression frees the CPU to do other train-time tasks like augmentation. Note that TrueNAS will store compressed blocks in adaptive replacement cache (ARC), thus uncompressed and PackBits compressed files will have similar memory footprint.

In summary, if you have sufficient storage capacity, network bandwidth, and are using an access pattern that saturates the network link, uncompressed TIFFs are a good choice. Local flash storage will be bottlenecked
by the decompression step, so uncompressed TIFFs are an ideal choice for maximum throughput.

#### Access Patterns

When loading preprocessed TIFFs from HDDs over a local network, access patterns become a significant determinant of throughput. Spinning rust HDDs suffer from high latencies, and thus random access patterns are suboptimal. Below is a comparison of sequential and random access patterns for a the dataset described above.
This is not an exact comparison, as the order of file reads differs between the two and thus the slice chosen from each DBT volume is different. However, the substantial difference in throughput demonstrates the impact of access patterns.

| Access Pattern | Throughput (files/s) |
|----------------|----------------------|
| Sequential     | 641.4                |
| Random         | 9.805                |

#### ARC

TrueNAS will store retrieved data in ARC. For sufficiently small datasets, it is possible that the entire dataset can be stored in ARC, thus eliminating bottlenecks associated with disk I/O and random access patterns. Below is a comparison of two benchmark runs, both using random access patterns with a consistent
seed between runs. The second run benefits from ARC, as the dataset is smaller than the available ARC capacity.

| Dataset Size | Throughput (files/s) |
|--------------|----------------------|
| Run 1        | 9.836                |
| Run 2        | 820.5                |

Given sufficient network bandwidth and ARC capacity, operations on datasets that have been cached will likely be bottlenecked by decode time.


### Manifest Creation

When dealing with large datasets stored on slow drives, it is useful to create a manifest of the dataset.
This manifest should track the preprocessed file paths that comprise the dataset, as well as the inode of the
preprocessed file (for optimizing sequential read performance). A binary, `dicom-manifest`, is provided to create a manifest from a directory of preprocessed TIFFs. It is assumed that the preprocessed TIFFs are named in the format of `{study_instance_uid}/{series_instance_uid}/{sop_instance_uid}.tiff`.

`dicom-manifest` can be pointed at any TIFF subtree (for example, a dataset's `images/` directory). The output
manifest can be written elsewhere (for example, the dataset root), and `path` values are written relative to the
manifest file location.

The manifest will contain the following columns, sorted by (`study_instance_uid`, `sop_instance_uid`):
- `study_instance_uid` - the study instance UID of the preprocessed file
- `sop_instance_uid` - the SOP instance UID of the preprocessed file
- `path` - the path of the preprocessed file relative to the manifest file location
- `inode` - the inode number of the preprocessed file
- `width` - the width of the preprocessed file
- `height` - the height of the preprocessed file
- `channels` - the number of channels in the preprocessed file
- `num_frames` - the number of frames in the preprocessed file

Example usage:

```
dicom-manifest /path/to/preprocessed/dataset/images /path/to/preprocessed/dataset/manifest.csv
```

### Rust Viewer API

Rust callers can use `ViewerDicom` and `VolumeHandler` when they need display-frame planning without committing to the TIFF preprocessing output path.
The viewer path uses the same frame ordering, volume handling, and DICOM sanitation logic as preprocessing, including empty `VOILUTFunction` cleanup.

```rust
use dicom_preprocessing::{ViewerDicom, VolumeFrameSource, VolumeHandler};

fn main() -> Result<(), dicom_preprocessing::DicomError> {
    let viewer = ViewerDicom::open("/path/to/image.dcm", VolumeHandler::keep())?;
    for (display_index, source) in viewer.frame_plan().display_frames.iter().enumerate() {
        match source {
            VolumeFrameSource::StoredFrame { stored_frame_index } => {
                println!("display {display_index} maps to stored frame {stored_frame_index}");
                let raw = viewer.decode_raw_display_frame(display_index)?;
                println!("raw bytes: {}", raw.data.len());
            }
            VolumeFrameSource::Derived => {
                println!("display {display_index} is synthesized");
            }
        }
    }
    Ok(())
}
```

`StoredFrame` entries are safe to use for frame-specific overlays such as GSPS, SR, or Parametric Map references.
`Derived` entries identify synthesized outputs such as interpolation, MIP, or Laplacian MIP where exact stored-frame overlays should not be drawn directly.

### Python Bindings

Python bindings are provided via the `pyo3` crate. The following features are supported:
 - Loading preprocessed TIFFs into Numpy arrays
 - Iterating, sorting, and discovering DICOM or TIFF files from various sources
 - Direct preprocessing of a DICOM file or buffer into a Numpy array
 - Validating whether a DICOM file is ready for preprocessing

Python preprocessing converts the resulting image stack directly into one NHWC NumPy array without creating an intermediate TIFF file.
The `u8`, `u16`, and `f32` entry points use the same numeric conversion semantics as the explicit TIFF-loading APIs, and DICOM pixel decoding and preprocessing release the Python GIL.
TIFF serialization remains available through the CLI and Rust output APIs when a persistent preprocessed file is desired.

Python exposes typed factories for the same volume handlers as Rust and Node. String names remain available, including `central` as an alias for `central-slice`.

```python
import dicom_preprocessing as dp

handler = dp.VolumeHandler.laplacian_mip(
    skip_start=2,
    skip_end=2,
    mip_weight=1.5,
    projection_mode="parallel-beam",
)
preprocessor = dp.Preprocessor(crop=False, volume_handler=handler)
projection = dp.preprocess_f32("/path/to/volume.dcm", preprocessor)
```

The validator API returns the same report schema as `dicom-validate --format json`.
Validation blockers are reported in the returned dictionary rather than raised as Python exceptions.
Runtime errors such as a missing source path still raise exceptions.

```python
import dicom_preprocessing as dp

report = dp.validate_dicom("/path/to/image.dcm", decode="frame")
if not report["summary"]["valid"]:
    print(report["errors"])
```

### Node/TypeScript Bindings

The repository root is the source-installable `@medcognetics/dicom-preprocessing` Node package. NAPI-RS source and generated entry points remain under `bindings/node`. The viewer path exposes the same frame-plan contract as the Rust `ViewerDicom` API without crop, resize, or pad.

Install an exact repository commit by using its full SHA:

```json
{
  "dependencies": {
    "@medcognetics/dicom-preprocessing": "git+https://github.com/medcognetics/dicom-preprocessing.git#<full-commit-sha>"
  }
}
```

Both initial and lockfile-based installs support omitting optional dependencies:

```shell
npm install --omit=optional
```

After removing `node_modules`, the generated lockfile can reproduce the installation with `npm ci --omit=optional`.

npm records the full commit SHA in `package-lock.json`. During a Git install, the package `prepare` script compiles the N-API module for the host and packs the generated JavaScript, declarations, and local `.node` binary. The source build requires Git, npm 10 or newer, Rust and Cargo 1.89.0 or newer, and a supported Node release: Node 22.13 or newer in the Node 22 line, Node 24, or Node 26. A native toolchain is also required: GNU build tools on Linux, Xcode command-line tools on macOS, or MSVC Build Tools on Windows.

Supported hosts are Linux x64 GNU, macOS arm64, and Windows x64 MSVC.

```ts
import { prepareDicom, renderDisplayFrame, renderFrame } from '@medcognetics/dicom-preprocessing'

const prepared = prepareDicom({ path: '/path/to/image.dcm' })
const raw = renderFrame(prepared, 0)
const display = renderDisplayFrame(prepared, 0)
console.log(display.width, display.height, display.dtype, display.source)
```

`renderFrame` returns raw stored-frame bytes and rejects derived display frames because they have no exact stored-frame source.
`renderDisplayFrame` returns the display-frame pixels, including derived volume-handler outputs such as `laplacian-mip`.
Frame data is returned as a Node `Buffer`. NAPI-RS can transfer Rust-owned buffers without copying in standard Node runtimes, but Electron may copy buffers because of V8 memory-cage constraints.


### CT Scan Stacking

CT scans are often represented using multiple DICOM files, one per axial slice. 
A CLI tool is provided to combine per-slice TIFF files (created through the preprocessing flow described above)
into a single multi-frame TIFF. 
This tool relies on accompanying CSV or Parquet metadata file which gives the `InstanceNumber`
for each `SOPInstanceUID` to properly order the frames in a series.


```
Combine single-frame TIFF files into multi-frame TIFFs

Usage: tiff-combine [OPTIONS] <SOURCE> <METADATA> <OUTPUT>

Arguments:
  <SOURCE>    Directory containing TIFF files with structure study_instance_uid/series_instance_uid/sop_instance_uid.tiff
  <METADATA>  CSV or Parquet file with series_instance_uid, sop_instance_uid, and instance_number columns
  <OUTPUT>    Output directory for combined multi-frame TIFFs

Options:
  -v, --verbose  Enable verbose logging
  -h, --help     Print help
  -V, --version  Print version
```

### References

1. Wei J, Chan HP, Helvie MA, Roubidoux MA, Neal CH, Lu Y, Hadjiiski LM, Zhou C. Synthesizing Mammogram from Digital Breast Tomosynthesis. *Phys Med Biol.* 2019;64(4):045011. doi:[10.1088/1361-6560/aafcda](https://doi.org/10.1088/1361-6560/aafcda)
