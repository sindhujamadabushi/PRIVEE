#!/usr/bin/env python3
"""
Create CSV versions of MNIST, CIFAR-10, CIFAR-100, UCI Adult Income,
and UCI Sensorless Drive Diagnosis.

Examples
--------
Create MNIST under the current directory:
    python create_dataset.py --dataset MNIST

Create CIFAR-100 under /home/msindhuja/PRIVEE:
    python create_dataset.py \
        --dataset CIFAR100 \
        --output-root /home/msindhuja/PRIVEE

Create all datasets:
    python create_dataset.py \
        --dataset ALL \
        --output-root /home/msindhuja/PRIVEE

The resulting layout is:
    <output-root>/datasets/
        MNIST/MNIST.csv
        CIFAR10/CIFAR10.csv
        CIFAR100/CIFAR100.csv
        ADULT/ADULT.csv
        DRIVE/DRIVE.csv

Image pixels are stored as uint8 values in [0, 255] to keep the already
large CSV files smaller. Divide image features by 255.0 when loading them.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


DATASET_NAMES = ("MNIST", "CIFAR10", "CIFAR100", "ADULT", "DRIVE")

ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "class",
]

UCI_ADULT_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/2/adult.zip"
)
UCI_DRIVE_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/325/"
    "dataset%2Bfor%2Bsensorless%2Bdrive%2Bdiagnosis.zip"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a dataset and create the CSV layout used by PRIVEE."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=str.upper,
        choices=(*DATASET_NAMES, "ALL"),
        help="Dataset to create: MNIST, CIFAR10, CIFAR100, ADULT, DRIVE, or ALL.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default="/Users/sindhujamadabushi/Desktop/PRIVEE/datasets",
        help=(
            "Parent directory in which the datasets folder is created. "
            "Default: the current working directory."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Rows written per chunk for image CSV files. Default: 1000.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing CSV instead of stopping.",
    )
    return parser.parse_args()


def download_file(url: str, destination: Path) -> Path:
    """Download a URL once and reuse the cached file on later runs."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and destination.stat().st_size > 0:
        print(f"Using cached download: {destination}")
        return destination

    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        temporary.unlink()

    print(f"Downloading: {url}")
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            with temporary.open("wb") as output_file:
                shutil.copyfileobj(response, output_file)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise

    temporary.replace(destination)
    return destination


def extract_zip(zip_path: Path, extraction_directory: Path) -> Path:
    extraction_directory.mkdir(parents=True, exist_ok=True)
    marker = extraction_directory / ".extracted"

    if marker.exists():
        return extraction_directory

    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(extraction_directory)

    marker.touch()
    return extraction_directory


def find_file(root: Path, filename: str) -> Path:
    matches = list(root.rglob(filename))
    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename!r} after extracting files under {root}."
        )
    return matches[0]


def prepare_output_path(
    datasets_root: Path,
    dataset_name: str,
    overwrite: bool,
) -> Path:
    dataset_directory = datasets_root / dataset_name
    dataset_directory.mkdir(parents=True, exist_ok=True)
    csv_path = dataset_directory / f"{dataset_name}.csv"

    if csv_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"{csv_path} already exists. Use --overwrite to replace it."
            )
        csv_path.unlink()

    return csv_path


def write_image_csv(
    images: np.ndarray,
    labels: np.ndarray,
    csv_path: Path,
    feature_names: list[str],
    chunk_size: int,
) -> None:
    """
    Write flattened image pixels and a final integer `class` column.

    Pixels remain uint8 values in [0, 255]. This avoids greatly increasing
    the size of the CIFAR CSV files.
    """
    if chunk_size <= 0:
        raise ValueError("--chunk-size must be a positive integer.")

    images = np.asarray(images)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)

    if images.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Image count {images.shape[0]} does not match label count "
            f"{labels.shape[0]}."
        )

    flat_dimension = int(np.prod(images.shape[1:]))
    if flat_dimension != len(feature_names):
        raise ValueError(
            f"Expected {flat_dimension} feature names, received "
            f"{len(feature_names)}."
        )

    total_rows = images.shape[0]
    temporary_path = csv_path.with_suffix(".csv.part")
    if temporary_path.exists():
        temporary_path.unlink()

    print(
        f"Writing {total_rows:,} rows and {flat_dimension:,} pixel columns "
        f"to {csv_path}"
    )

    try:
        for start in range(0, total_rows, chunk_size):
            end = min(start + chunk_size, total_rows)
            flattened = images[start:end].reshape(end - start, flat_dimension)

            frame = pd.DataFrame(flattened, columns=feature_names, copy=False)
            frame["class"] = labels[start:end]

            frame.to_csv(
                temporary_path,
                mode="w" if start == 0 else "a",
                header=(start == 0),
                index=False,
            )
            print(f"  wrote rows {start:,} through {end - 1:,}")
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    temporary_path.replace(csv_path)


def create_mnist(
    datasets_root: Path,
    cache_root: Path,
    overwrite: bool,
    chunk_size: int,
) -> Path:
    try:
        from torchvision.datasets import MNIST
    except ImportError as exc:
        raise RuntimeError(
            "torchvision is required. Install it with "
            "`python -m pip install torch torchvision`."
        ) from exc

    csv_path = prepare_output_path(datasets_root, "MNIST", overwrite)
    raw_root = cache_root / "torchvision"

    train = MNIST(root=raw_root, train=True, download=True)
    test = MNIST(root=raw_root, train=False, download=True)

    # Train first and official test set last. Your existing MNIST loader takes
    # the last 10,000 rows as the test set, so this ordering is intentional.
    images = np.concatenate(
        (train.data.numpy(), test.data.numpy()),
        axis=0,
    ).astype(np.uint8, copy=False)
    labels = np.concatenate(
        (
            np.asarray(train.targets, dtype=np.int64),
            np.asarray(test.targets, dtype=np.int64),
        )
    )

    feature_names = [
        f"pixel_r{row:02d}_c{column:02d}"
        for row in range(28)
        for column in range(28)
    ]

    write_image_csv(
        images=images,
        labels=labels,
        csv_path=csv_path,
        feature_names=feature_names,
        chunk_size=chunk_size,
    )
    return csv_path


def create_cifar(
    dataset_name: str,
    datasets_root: Path,
    cache_root: Path,
    overwrite: bool,
    chunk_size: int,
) -> Path:
    try:
        from torchvision.datasets import CIFAR10, CIFAR100
    except ImportError as exc:
        raise RuntimeError(
            "torchvision is required. Install it with "
            "`python -m pip install torch torchvision`."
        ) from exc

    dataset_class = CIFAR10 if dataset_name == "CIFAR10" else CIFAR100
    csv_path = prepare_output_path(datasets_root, dataset_name, overwrite)
    raw_root = cache_root / "torchvision"

    train = dataset_class(root=raw_root, train=True, download=True)
    test = dataset_class(root=raw_root, train=False, download=True)

    # torchvision stores CIFAR images as NHWC. Convert to NCHW before
    # flattening so adjacent CSV blocks correspond to image channels.
    train_images = np.transpose(train.data, (0, 3, 1, 2))
    test_images = np.transpose(test.data, (0, 3, 1, 2))
    images = np.concatenate((train_images, test_images), axis=0).astype(
        np.uint8,
        copy=False,
    )
    labels = np.concatenate(
        (
            np.asarray(train.targets, dtype=np.int64),
            np.asarray(test.targets, dtype=np.int64),
        )
    )

    feature_names = [
        f"pixel_ch{channel}_r{row:02d}_c{column:02d}"
        for channel in range(3)
        for row in range(32)
        for column in range(32)
    ]

    write_image_csv(
        images=images,
        labels=labels,
        csv_path=csv_path,
        feature_names=feature_names,
        chunk_size=chunk_size,
    )
    return csv_path


def create_adult(
    datasets_root: Path,
    cache_root: Path,
    overwrite: bool,
    chunk_size: int,
) -> Path:
    del chunk_size  # Not needed for this comparatively small dataset.

    csv_path = prepare_output_path(datasets_root, "ADULT", overwrite)
    zip_path = download_file(
        UCI_ADULT_ZIP_URL,
        cache_root / "uci" / "adult.zip",
    )
    extracted = extract_zip(
        zip_path,
        cache_root / "uci" / "adult",
    )

    train_path = find_file(extracted, "adult.data")
    test_path = find_file(extracted, "adult.test")

    common_arguments = {
        "names": ADULT_COLUMNS,
        "skipinitialspace": True,
        "na_values": ["?"],
    }

    train_frame = pd.read_csv(train_path, **common_arguments)
    test_frame = pd.read_csv(
        test_path,
        comment="|",
        **common_arguments,
    )

    frame = pd.concat(
        (train_frame, test_frame),
        axis=0,
        ignore_index=True,
    )

    # UCI's test labels end in a period. The split_data mapping expects
    # exactly '<=50K' and '>50K'.
    frame["class"] = (
        frame["class"]
        .astype("string")
        .str.strip()
        .str.removesuffix(".")
    )

    valid_labels = {"<=50K", ">50K"}
    observed_labels = set(frame["class"].dropna().unique())
    if observed_labels != valid_labels:
        raise ValueError(
            f"Unexpected ADULT labels: {sorted(observed_labels)}. "
            f"Expected {sorted(valid_labels)}."
        )

    frame.to_csv(csv_path, index=False)
    print(f"Writing {len(frame):,} rows and {frame.shape[1] - 1} features to {csv_path}")
    return csv_path


def create_drive(
    datasets_root: Path,
    cache_root: Path,
    overwrite: bool,
    chunk_size: int,
) -> Path:
    del chunk_size  # Not needed for this comparatively narrow dataset.

    csv_path = prepare_output_path(datasets_root, "DRIVE", overwrite)
    zip_path = download_file(
        UCI_DRIVE_ZIP_URL,
        cache_root / "uci" / "sensorless_drive_diagnosis.zip",
    )
    extracted = extract_zip(
        zip_path,
        cache_root / "uci" / "sensorless_drive_diagnosis",
    )

    text_path = find_file(extracted, "Sensorless_drive_diagnosis.txt")
    raw = pd.read_csv(text_path, sep=r"\s+", header=None)

    # The source file contains 48 real-valued features followed by a class
    # label taking values 1,...,11.
    if raw.shape[1] != 49:
        raise ValueError(
            f"Expected 49 DRIVE columns (48 features plus class), "
            f"but found {raw.shape[1]}."
        )

    feature_names = [f"feature_{index:02d}" for index in range(48)]
    frame = raw.iloc[:, :48].copy()
    frame.columns = feature_names

    original_labels = raw.iloc[:, 48].astype(np.int64)
    observed_labels = sorted(original_labels.unique().tolist())
    if observed_labels != list(range(1, 12)):
        raise ValueError(
            f"Unexpected DRIVE labels: {observed_labels}. "
            "Expected integer labels 1 through 11."
        )

    # PyTorch CrossEntropyLoss requires zero-based labels.
    frame["class"] = original_labels - 1

    frame.to_csv(csv_path, index=False)
    print(f"Writing {len(frame):,} rows and 48 features to {csv_path}")
    return csv_path


def main() -> int:
    arguments = parse_args()

    output_root = arguments.output_root.expanduser().resolve()
    datasets_root = output_root / "datasets"
    cache_root = output_root / ".dataset_cache"

    datasets_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    # Create the requested directory structure even when only one dataset
    # is generated in this invocation.
    for dataset_name in DATASET_NAMES:
        (datasets_root / dataset_name).mkdir(parents=True, exist_ok=True)

    creators: dict[str, Callable[[], Path]] = {
        "MNIST": lambda: create_mnist(
            datasets_root,
            cache_root,
            arguments.overwrite,
            arguments.chunk_size,
        ),
        "CIFAR10": lambda: create_cifar(
            "CIFAR10",
            datasets_root,
            cache_root,
            arguments.overwrite,
            arguments.chunk_size,
        ),
        "CIFAR100": lambda: create_cifar(
            "CIFAR100",
            datasets_root,
            cache_root,
            arguments.overwrite,
            arguments.chunk_size,
        ),
        "ADULT": lambda: create_adult(
            datasets_root,
            cache_root,
            arguments.overwrite,
            arguments.chunk_size,
        ),
        "DRIVE": lambda: create_drive(
            datasets_root,
            cache_root,
            arguments.overwrite,
            arguments.chunk_size,
        ),
    }

    requested = (
        list(DATASET_NAMES)
        if arguments.dataset == "ALL"
        else [arguments.dataset]
    )

    created_paths: list[Path] = []
    for dataset_name in requested:
        print(f"\n=== Creating {dataset_name} ===")
        created_paths.append(creators[dataset_name]())

    print("\nCreated:")
    for path in created_paths:
        print(f"  {path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
