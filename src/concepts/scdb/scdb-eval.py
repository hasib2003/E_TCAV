import argparse
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reorganize test images into per-class directories"
    )
    parser.add_argument(
        "--test-root",
        type=Path,
        default="/netscratch/aslam/TCAV/SCDB",
        help="Path to test image root (directory above relative paths)",
    )
    parser.add_argument(
        "--label-file",
        type=Path,
        default="/netscratch/aslam/TCAV/SCDB/test.csv",
        help="Path to test label file (path|class_idx format)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="/netscratch/aslam/TCAV/SCDB/test-dirs",
        help="Output directory (default: ./test-dirs)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.test_root.is_dir():
        raise NotADirectoryError(args.test_root)

    if not args.label_file.is_file():
        raise FileNotFoundError(args.label_file)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    num_processed = 0

    with open(args.label_file, "r") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                rel_path, label = line.split("|")
                label = str(int(label))  # normalize
            except ValueError:
                raise ValueError(
                    f"Malformed line {line_num} in {args.label_file}: {line}"
                )

            src_path = args.test_root / rel_path
            if not src_path.exists():
                raise FileNotFoundError(src_path)

            class_dir = args.output_dir / label
            class_dir.mkdir(parents=True, exist_ok=True)

            dst_path = class_dir / src_path.name

            shutil.copy2(src_path, dst_path)


            num_processed += 1

    print(f"✓ Organized {num_processed} files into {args.output_dir}")


if __name__ == "__main__":
    main()
