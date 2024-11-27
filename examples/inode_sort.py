from pathlib import Path
from dicom_preprocessing import inode_sort
import timeit
from argparse import ArgumentParser, Namespace
from typing import List

def inode_sort_python(paths: List[Path]) -> List[Path]:
    return sorted(paths, key=lambda p: p.stat().st_ino)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("path", type=Path, help="Path to directory")
    parser.add_argument("-i", "--iterations", type=int, default=100, help="Number of iterations")
    return parser.parse_args()


def main(args: Namespace):
    if not args.path.is_dir():
        raise NotADirectoryError(f"Directory not found: {args.path}")

    print("Finding files...")
    files = list(args.path.rglob("*"))
    files = [p for p in files if p.is_file()]
    print(f"Found {len(files)} files")

    result_python = inode_sort_python(files)
    result_rust = inode_sort(files)
    # NOTE: Only check length - rust implementation uses unstable sort
    assert len(result_python) == len(result_rust)

    t1 = timeit.timeit(lambda: inode_sort(files), number=args.iterations) / args.iterations * 1000
    print(f"rust: {t1:.3f}ms")
    t2 = timeit.timeit(lambda: inode_sort_python(files), number=args.iterations) / args.iterations * 1000
    print(f"python: {t2:.3f}ms")


def entrypoint():
    main(parse_args())


if __name__ == "__main__":
    entrypoint()