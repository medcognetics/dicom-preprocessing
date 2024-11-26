import timeit
from argparse import ArgumentParser, Namespace
from functools import partial
from pathlib import Path

import numpy as np
from dicom_preprocessing import load_tiff_f32, load_tiff_u16
from PIL import Image


def read_pil(filepath: Path, fp32: bool = False):
    """Reads a multi-frame TIFF file using PIL and returns a stacked array."""
    img = Image.open(filepath)
    frames = []

    try:
        while True:
            frames.append(img.copy())
            img.seek(img.tell() + 1)  # Move to the next frame
    except EOFError:
        pass  # End of sequence

    result = np.stack(frames)
    if fp32:
        delta = np.iinfo(result.dtype).max - np.iinfo(result.dtype).min
        result = result.astype(np.float32) / delta
    return result


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("path", type=Path, help="Path to TIFF file")
    parser.add_argument("--iterations", type=int, default=30, help="Number of iterations")
    parser.add_argument("--fp32", default=False, action="store_true", help="Load into float32. Otherwise load into u16")
    return parser.parse_args()


def main(args: Namespace):
    if not args.path.is_file():
        raise FileNotFoundError(f"File not found: {args.path}")

    if args.fp32:

        def rust_fn(p):
            return load_tiff_f32(str(p))

        pil_fn = partial(read_pil, fp32=True)
    else:

        def rust_fn(p):
            return load_tiff_u16(str(p))

        pil_fn = partial(read_pil, fp32=False)

    print(f"The test file size is {Path(args.path).stat().st_size / 1024**2:.2f}MB")
    arr1 = rust_fn(args.path)
    arr2 = pil_fn(args.path)
    print(arr1.dtype)
    print(arr2.dtype)
    assert np.allclose(arr1.squeeze(), arr2.squeeze())

    t1 = timeit.timeit(lambda: rust_fn(args.path), number=args.iterations) / args.iterations * 1000
    t2 = timeit.timeit(lambda: pil_fn(args.path), number=args.iterations) / args.iterations * 1000
    print(f"rust: {t1:.3f}ms")
    print(f"pil: {t2:.3f}ms")


def entrypoint():
    main(parse_args())


if __name__ == "__main__":
    entrypoint()
