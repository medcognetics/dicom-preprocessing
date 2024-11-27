from pathlib import Path
from dicom_preprocessing import find_dicom_files
import timeit
from argparse import ArgumentParser, Namespace
from typing import List

def is_dicom_file(path: Path, strict: bool) -> bool:
    ext = path.suffix.lower()
    if ext in (".dcm", ".dicom"):
        return not strict or path.is_file()
    elif ext or path.is_dir():
        return False
    raise NotImplementedError("DICM prefix check")

def find_dicom_files_python(path: Path) -> List[Path]:
    return list(p for p in path.rglob("*") if is_dicom_file(p, False))

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("path", type=Path, help="Path to directory")
    parser.add_argument("-i", "--iterations", type=int, default=1, help="Number of iterations")
    return parser.parse_args()


def main(args: Namespace):
    if not args.path.is_dir():
        raise NotADirectoryError(f"Directory not found: {args.path}")

    result_rust = find_dicom_files(str(args.path), True)
    result_python = find_dicom_files_python(args.path)
    # NOTE: Only check length - rust implementation uses unstable sort
    assert len(result_python) == len(result_rust)

    t1 = timeit.timeit(lambda: find_dicom_files(str(args.path), False), number=args.iterations) / args.iterations * 1000
    print(f"rust: {t1:.3f}ms")
    t2 = timeit.timeit(lambda: find_dicom_files_python(args.path), number=args.iterations) / args.iterations * 1000
    print(f"python: {t2:.3f}ms")


def entrypoint():
    main(parse_args())


if __name__ == "__main__":
    entrypoint()