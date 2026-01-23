#!/usr/bin/env python3
"""
Setup and download NuInsSeg dataset for validation.

NuInsSeg: A fully annotated dataset for nuclei instance segmentation
in H&E-stained histological images.

- Paper: https://www.nature.com/articles/s41597-024-03117-2
- GitHub: https://github.com/masih4/NuInsSeg
- Kaggle: https://www.kaggle.com/datasets/ipateam/nuinsseg
- 665 images (512x512) from 31 organs (human + mouse)
- Instance segmentation masks (labeled)

We use this to validate:
1. Adjacency/contact graph extraction on nuclei
2. Generalizability to histopathology images
3. Performance on dense, touching cell populations
"""

import os
import sys
from pathlib import Path
import zipfile
import shutil


def check_kaggle_api():
    """Check if Kaggle API is available."""
    try:
        import kaggle
        return True
    except ImportError:
        return False
    except OSError:
        # Kaggle credentials not configured
        return False


def setup_nuinsseg(data_dir: Path, use_kaggle: bool = True):
    """
    Setup NuInsSeg dataset for validation.

    Parameters
    ----------
    data_dir : Path
        Directory to store NuInsSeg data
    use_kaggle : bool
        If True, download from Kaggle API. Otherwise, provide manual instructions.
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SETTING UP NUINSSEG DATASET")
    print("=" * 60)
    print(f"\nTarget directory: {data_dir}")

    # Check if already downloaded
    tissue_dir = data_dir / "tissue images"
    label_dir = data_dir / "labeled masks"

    if tissue_dir.exists() and label_dir.exists():
        n_images = len(list(tissue_dir.glob("*/*.png")))
        n_masks = len(list(label_dir.glob("*/*.tif")))
        if n_images > 0 and n_masks > 0:
            print(f"\nDataset already exists:")
            print(f"  Images: {n_images}")
            print(f"  Masks: {n_masks}")
            return data_dir

    if use_kaggle and check_kaggle_api():
        print("\nDownloading from Kaggle...")
        try:
            import kaggle
            kaggle.api.dataset_download_files(
                'ipateam/nuinsseg',
                path=str(data_dir),
                unzip=True
            )
            print("Download complete!")
        except Exception as e:
            print(f"Kaggle download failed: {e}")
            use_kaggle = False

    if not use_kaggle or not check_kaggle_api():
        print("\n" + "-" * 60)
        print("MANUAL DOWNLOAD REQUIRED")
        print("-" * 60)
        print("""
To download NuInsSeg:

1. Go to: https://www.kaggle.com/datasets/ipateam/nuinsseg
2. Click "Download" (requires Kaggle account)
3. Extract the ZIP file to: {data_dir}

Or use Kaggle CLI:
  pip install kaggle
  kaggle datasets download -d ipateam/nuinsseg -p {data_dir} --unzip

The directory structure should be:
  {data_dir}/
    tissue images/
      <organ_name>/
        *.png
    labeled masks/
      <organ_name>/
        *.tif
""".format(data_dir=data_dir))
        return data_dir

    # Verify setup
    print("\n" + "=" * 60)
    print("SETUP COMPLETE")
    print("=" * 60)

    if tissue_dir.exists():
        organs = [d.name for d in tissue_dir.iterdir() if d.is_dir()]
        n_images = len(list(tissue_dir.glob("*/*.png")))
        print(f"\nDataset statistics:")
        print(f"  Organs: {len(organs)}")
        print(f"  Images: {n_images}")
        print(f"  Organ types: {', '.join(organs[:5])}...")

    if label_dir.exists():
        n_masks = len(list(label_dir.glob("*/*.tif")))
        print(f"  Instance masks: {n_masks}")

    return data_dir


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Setup NuInsSeg dataset")
    parser.add_argument("--data-dir", type=Path,
                        default=Path(__file__).parent.parent / "data" / "NuInsSeg",
                        help="Directory to store NuInsSeg data")
    parser.add_argument("--no-kaggle", action="store_true",
                        help="Skip Kaggle download, show manual instructions")

    args = parser.parse_args()
    setup_nuinsseg(args.data_dir, use_kaggle=not args.no_kaggle)


if __name__ == "__main__":
    main()
