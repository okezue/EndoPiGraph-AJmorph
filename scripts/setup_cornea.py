#!/usr/bin/env python3
"""
Setup script for Cornea Cells dataset.

Dataset: U-Net_Segmentation-Cornea_Cells
Source: https://github.com/svdeepak99/U-Net_Segmentation-Cornea_Cells
Format: Semantic segmentation masks (1=interior, 2=border, 3=background)

This script clones the repository if not already present.
"""

import subprocess
import sys
from pathlib import Path


def setup_cornea_dataset(data_dir: Path = None):
    """Clone the cornea cells dataset repository."""
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data" / "cornea_cells"

    if data_dir.exists():
        # Check if it has the expected structure
        labels_dir = data_dir / "labels"
        dataset_dir = data_dir / "dataset"

        if labels_dir.exists() and dataset_dir.exists():
            num_labels = len(list(labels_dir.glob("*.tif")))
            num_images = len(list(dataset_dir.glob("*.tif")))
            print(f"Cornea dataset already exists at {data_dir}")
            print(f"  Labels: {num_labels} files")
            print(f"  Images: {num_images} files")
            return data_dir
        else:
            print(f"Directory exists but incomplete, re-cloning...")
            import shutil
            shutil.rmtree(data_dir)

    print(f"Cloning cornea cells dataset to {data_dir}...")
    data_dir.parent.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        ["git", "clone",
         "https://github.com/svdeepak99/U-Net_Segmentation-Cornea_Cells.git",
         str(data_dir)],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"Error cloning repository: {result.stderr}")
        sys.exit(1)

    print("Dataset cloned successfully!")

    # Verify structure
    labels_dir = data_dir / "labels"
    dataset_dir = data_dir / "dataset"

    num_labels = len(list(labels_dir.glob("*.tif")))
    num_images = len(list(dataset_dir.glob("*.tif")))

    print(f"Dataset structure:")
    print(f"  Labels: {num_labels} files")
    print(f"  Images: {num_images} files")

    return data_dir


if __name__ == "__main__":
    setup_cornea_dataset()
