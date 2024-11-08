# Dicom Preprocessing

Implements a tool that preprocesses DICOM files into TIFF images.
DICOM files are opened and any applicable LUT transformations are applied.

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


### Command Line Interface

```
Preprocess DICOM files

Usage: dicom-preprocess [OPTIONS] <SOURCE> <OUTPUT>

Arguments:
  <SOURCE>  Source path. Can be a DICOM file, directory, or a text file with DICOM file paths
  <OUTPUT>  Output path. Can be a directory (for multiple files) or a file (for a single file)

Options:
  -c, --crop                         Crop the image
  -s, --size <SIZE>                  Target size (width,height)
  -f, --filter <FILTER>              Filter type 
  -p, --padding <PADDING_DIRECTION>  Padding direction 
  -h, --help                         Print help
  -V, --version                      Print version
```

### Limitations

This tool currently has the following limitations:
- Only 16-bit monochrome DICOM images are supported.
- Outputs will be 16-bit monochrome TIFF images compressed using packbits.