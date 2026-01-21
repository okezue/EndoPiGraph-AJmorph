#!/usr/bin/env python3
"""
Process all EGM2 images and run B3 experiment on larger dataset.

This script:
1. Creates a manifest for available EGM2 images
2. Runs the EndoPiGraph pipeline on each image
3. Runs the typed vs untyped experiment on the larger dataset

Usage:
    python scripts/process_egm2_batch.py
"""

import subprocess
import sys
from pathlib import Path
import pandas as pd

def main():
    base_dir = Path(__file__).parent.parent
    images_dir = base_dir / "data/S-BIAD1540/images_egm2"
    egm2_manifest = base_dir / "data/S-BIAD1540/manifest_egm2_full.csv"
    output_dir = base_dir / "runs/egm2_full"

    # Check what images we have
    available_images = set(p.stem for p in images_dir.glob("*.tif"))
    print(f"Found {len(available_images)} downloaded EGM2 images")

    if len(available_images) < 30:
        print("WARNING: Less than 30 images available. Waiting for more downloads...")
        print("Run download_egm2.sh to download more images.")
        return

    # Load full manifest and filter to available images
    df = pd.read_csv(egm2_manifest)
    df['stem'] = df['path'].apply(lambda x: Path(x).stem)
    df_available = df[df['stem'].isin(available_images)].copy()

    print(f"Matched {len(df_available)} images in manifest")
    print(f"Conditions: {df_available['shear_stress'].value_counts().to_dict()}")

    # Create local manifest pointing to downloaded images
    local_manifest = output_dir / "manifest_egm2_local.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Adjust paths to point to local files
    df_available['path'] = df_available['stem'].apply(
        lambda x: f"data/S-BIAD1540/images_egm2/{x}.tif"
    )
    df_available.to_csv(local_manifest, index=False)
    print(f"Saved manifest: {local_manifest}")

    # Run pipeline
    pipeline_script = base_dir / "src/EndoPiGraph_AJmorph_v1_allinone.py"

    cmd = [
        sys.executable, str(pipeline_script), "run",
        "--manifest", str(local_manifest),
        "--out", str(output_dir),
        "--aj-channel", "VE-cadherin",
        "--cell-channel", "VE-cadherin",
        "--nuc-channel", "DAPI",
        "--golgi-channel", "GM130",
        "--use-cellpose", "1"
    ]

    print(f"\nRunning pipeline...")
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd=str(base_dir))

    if result.returncode != 0:
        print(f"Pipeline failed with return code {result.returncode}")
        return

    print("\nPipeline complete!")

    # Run B3 experiment on new data
    print("\n" + "="*70)
    print("Running typed vs untyped experiment on EGM2 dataset...")
    print("="*70)

    b3_script = base_dir / "scripts/typed_vs_untyped_experiment.py"
    cmd = [sys.executable, str(b3_script), str(output_dir)]

    result = subprocess.run(cmd, cwd=str(base_dir))

    print("\nDone!")


if __name__ == "__main__":
    main()
