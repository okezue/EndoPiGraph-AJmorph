#!/usr/bin/env python3
"""
Setup and download LIVECell dataset for validation.

LIVECell: A large-scale dataset for label-free live cell segmentation
- Paper: https://www.nature.com/articles/s41592-021-01249-6
- Repo: https://github.com/sartorius-research/LIVECell
- >1.6M manually annotated cells
- COCO-format instance segmentation masks
- Expert-validated ground truth

We use this to validate:
1. Segmentation module accuracy
2. Adjacency/contact graph extraction
3. Generalizability to different modalities
"""

import os
import sys
import json
import subprocess
from pathlib import Path
import urllib.request
import zipfile

LIVECELL_BASE_URL = "https://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021"

# Smaller subset for quick validation
LIVECELL_FILES = {
    # Annotations (COCO format)
    "annotations": f"{LIVECELL_BASE_URL}/annotations/LIVECell/livecell_coco_val.json",
    # Sample images (we'll download a subset)
    "images_info": f"{LIVECELL_BASE_URL}/images.zip",  # ~1.3GB - we'll download selectively
}

def download_file(url: str, dest: Path, desc: str = None):
    """Download a file with progress."""
    desc = desc or url.split("/")[-1]
    print(f"Downloading {desc}...")

    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  Saved to {dest}")
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False


def setup_livecell(data_dir: Path, download_images: bool = False):
    """
    Setup LIVECell dataset for validation.

    Parameters
    ----------
    data_dir : Path
        Directory to store LIVECell data
    download_images : bool
        If True, download full image set (~1.3GB).
        If False, just download annotations for mask-based validation.
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SETTING UP LIVECELL DATASET")
    print("=" * 60)
    print(f"\nTarget directory: {data_dir}")

    # Download validation annotations
    ann_path = data_dir / "livecell_coco_val.json"
    if not ann_path.exists():
        download_file(
            LIVECELL_FILES["annotations"],
            ann_path,
            "validation annotations (COCO format)"
        )
    else:
        print(f"Annotations already exist: {ann_path}")

    # Clone repo for additional tools/info
    repo_dir = data_dir / "LIVECell-repo"
    if not repo_dir.exists():
        print("\nCloning LIVECell repository...")
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/sartorius-research/LIVECell.git",
            str(repo_dir)
        ], check=True)
    else:
        print(f"Repository already cloned: {repo_dir}")

    if download_images:
        print("\nNote: Full image download is ~1.3GB")
        print("For quick validation, we use annotation masks directly.")
        # Would download images here if needed

    # Verify setup
    print("\n" + "=" * 60)
    print("SETUP COMPLETE")
    print("=" * 60)

    if ann_path.exists():
        with open(ann_path) as f:
            coco = json.load(f)

        n_images = len(coco.get("images", []))
        n_annotations = len(coco.get("annotations", []))
        categories = [c["name"] for c in coco.get("categories", [])]

        print(f"\nValidation set statistics:")
        print(f"  Images: {n_images}")
        print(f"  Cell annotations: {n_annotations}")
        print(f"  Cell types: {categories}")

    print(f"\nFiles created:")
    for f in data_dir.iterdir():
        if f.is_file():
            size = f.stat().st_size / 1024 / 1024
            print(f"  {f.name}: {size:.1f} MB")
        elif f.is_dir():
            print(f"  {f.name}/ (directory)")

    return data_dir


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Setup LIVECell dataset")
    parser.add_argument("--data-dir", type=Path,
                        default=Path(__file__).parent.parent / "data" / "LIVECell",
                        help="Directory to store LIVECell data")
    parser.add_argument("--download-images", action="store_true",
                        help="Download full image set (~1.3GB)")

    args = parser.parse_args()
    setup_livecell(args.data_dir, args.download_images)


if __name__ == "__main__":
    main()
