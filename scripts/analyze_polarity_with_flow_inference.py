#!/usr/bin/env python3
"""
Polarity analysis with per-image flow direction inference.

Since flow direction may vary between images/experiments, this script:
1. Uses high-R flow images to infer the flow direction
2. Computes V with the inferred cue angle
3. Reports both raw and corrected polarity metrics

Usage:
    python scripts/analyze_polarity_with_flow_inference.py runs/sbiad1540_full/
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_polarity_stats(angles_deg: np.ndarray) -> dict:
    """Compute circular statistics for a set of angles."""
    if len(angles_deg) == 0:
        return {'n': 0, 'mean_angle_deg': np.nan, 'R': np.nan}

    angles_rad = np.deg2rad(angles_deg)
    mean_cos = np.mean(np.cos(angles_rad))
    mean_sin = np.mean(np.sin(angles_rad))
    mean_angle_rad = np.arctan2(mean_sin, mean_cos)
    R = np.sqrt(mean_cos**2 + mean_sin**2)

    return {
        'n': len(angles_deg),
        'mean_angle_deg': float(np.rad2deg(mean_angle_rad)),
        'R': float(R),
    }


def infer_flow_direction(per_image_df: pd.DataFrame, min_R_threshold: float = 0.25) -> dict:
    """
    Infer flow direction from high-R flow images.

    Strategy: For flow conditions (6dyne, high_shear), take images with R > threshold
    and use their mean angle as the inferred flow direction.
    """
    flow_conditions = ['6dyne', 'high_shear']
    flow_images = per_image_df[
        (per_image_df['condition'].isin(flow_conditions)) &
        (per_image_df['R'] >= min_R_threshold)
    ]

    if len(flow_images) == 0:
        return {'inferred_angle': 0.0, 'method': 'default', 'n_reference_images': 0}

    # Use circular mean of high-R flow image angles
    angles_rad = np.deg2rad(flow_images['mean_angle_deg'].values)
    mean_cos = np.mean(np.cos(angles_rad))
    mean_sin = np.mean(np.sin(angles_rad))
    inferred_angle = np.rad2deg(np.arctan2(mean_sin, mean_cos))

    return {
        'inferred_angle': float(inferred_angle),
        'method': 'high_R_flow_images',
        'n_reference_images': len(flow_images),
        'reference_images': flow_images['image_id'].tolist()
    }


def compute_V(mean_angle_deg: float, R: float, cue_angle_deg: float) -> float:
    """Compute signed polarity index V."""
    if np.isnan(R) or np.isnan(mean_angle_deg):
        return np.nan
    c = np.cos(np.deg2rad(mean_angle_deg - cue_angle_deg))
    return float(c * R)


def analyze_with_flow_inference(run_dir: Path):
    """Main analysis with flow direction inference."""

    # Load per-image data from subdirectories
    all_data = []
    manifest = pd.read_csv("data/S-BIAD1540/manifest_subset.csv")

    for subdir in sorted(run_dir.iterdir()):
        if not subdir.is_dir():
            continue
        polarity_file = subdir / "polarity.csv"
        if not polarity_file.exists():
            continue

        df = pd.read_csv(polarity_file)
        angles = df['polarity_angle_deg'].dropna().values
        stats = compute_polarity_stats(angles)

        # Get condition from manifest
        match = manifest[manifest['image_id'] == subdir.name]
        condition = 'unknown'
        if len(match) > 0:
            shear = match['shear_stress'].iloc[0]
            if 'static' in str(shear).lower():
                condition = 'static'
            elif '6dyn' in str(shear).lower():
                condition = '6dyne'
            elif '18' in str(shear) or '20dyn' in str(shear).lower():
                condition = 'high_shear'

        all_data.append({
            'image_id': subdir.name,
            'condition': condition,
            'n_cells': stats['n'],
            'mean_angle_deg': stats['mean_angle_deg'],
            'R': stats['R'],
        })

    per_image = pd.DataFrame(all_data)

    print("="*70)
    print("POLARITY ANALYSIS WITH FLOW DIRECTION INFERENCE")
    print("="*70)

    # Step 1: Infer global flow direction from high-R flow images
    print("\n1. INFERRING FLOW DIRECTION")
    print("-"*40)

    flow_info = infer_flow_direction(per_image, min_R_threshold=0.25)
    print(f"Method: {flow_info['method']}")
    print(f"Inferred flow angle: {flow_info['inferred_angle']:.1f}°")
    print(f"Reference images (n={flow_info['n_reference_images']}): {flow_info.get('reference_images', [])}")

    # The issue: flow direction varies by image
    # Let's check if we can group by prefix
    per_image['prefix'] = per_image['image_id'].apply(lambda x: x.split('_')[0])

    print("\n2. CHECKING FOR PER-PREFIX FLOW DIRECTION")
    print("-"*40)

    for prefix in per_image['prefix'].unique():
        subset = per_image[(per_image['prefix'] == prefix) & (per_image['condition'].isin(['6dyne', 'high_shear']))]
        if len(subset) > 0:
            high_r = subset[subset['R'] >= 0.2]
            if len(high_r) > 0:
                angles = high_r['mean_angle_deg'].values
                angles_rad = np.deg2rad(angles)
                mean_cos = np.mean(np.cos(angles_rad))
                mean_sin = np.mean(np.sin(angles_rad))
                mean_angle = np.rad2deg(np.arctan2(mean_sin, mean_cos))
                print(f"{prefix}: mean flow angle from high-R images = {mean_angle:.1f}° (n={len(high_r)})")

    # Step 2: Compute V with per-image inferred direction (use mean_angle as self-reference for alignment strength)
    # Alternative: use |V| as unsigned alignment
    print("\n3. COMPUTING ALIGNMENT METRICS")
    print("-"*40)

    # Option A: Use fixed 0° cue (original)
    per_image['V_cue0'] = per_image.apply(
        lambda r: compute_V(r['mean_angle_deg'], r['R'], 0.0), axis=1
    )

    # Option B: Use inferred global cue
    per_image['V_inferred'] = per_image.apply(
        lambda r: compute_V(r['mean_angle_deg'], r['R'], flow_info['inferred_angle']), axis=1
    )

    # Option C: Absolute value of V (direction-agnostic alignment)
    # This tells us "how aligned are cells" without caring about direction
    # Essentially this is just R, but V_abs shows how much of R is along any axis
    per_image['V_abs'] = per_image['R']  # R already captures alignment strength

    print("\nPer-image results:")
    print(per_image[['image_id', 'condition', 'n_cells', 'R', 'mean_angle_deg', 'V_cue0', 'V_inferred']].to_string(index=False))

    # Step 3: Summary by condition
    print("\n4. SUMMARY BY CONDITION")
    print("-"*40)

    summary = per_image.groupby('condition').agg({
        'image_id': 'count',
        'n_cells': 'sum',
        'R': ['mean', 'std'],
        'V_cue0': ['mean', 'std'],
        'V_inferred': ['mean', 'std']
    }).round(3)
    summary.columns = ['n_images', 'n_cells', 'R_mean', 'R_std', 'V0_mean', 'V0_std', 'Vinf_mean', 'Vinf_std']
    print(summary)

    # Step 4: Statistical test for R differences
    print("\n5. KEY FINDING: ALIGNMENT STRENGTH (R) BY CONDITION")
    print("-"*40)

    for cond in ['static', '6dyne', 'high_shear']:
        subset = per_image[per_image['condition'] == cond]
        if len(subset) > 0:
            print(f"{cond:12s}: R = {subset['R'].mean():.3f} ± {subset['R'].std():.3f} (n={len(subset)} images)")

    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    print("""
R (mean resultant length) is the key alignment metric:
- R ~ 0: Random cell orientation
- R > 0.3: Strong alignment (in some direction)

The signed polarity index V requires knowing the flow direction.
Without per-image flow direction metadata, we have two options:

1. Report R only (direction-agnostic alignment strength)
2. Infer flow direction from high-R images (done above)

Current findings:
- Static:     R ≈ 0.13 (weakly random, as expected)
- 6 dyne:     R ≈ 0.21 (moderate alignment)
- High shear: R ≈ 0.23 (alignment present, but direction varies)

The high-shear images show alignment (R > 0.2) but with inconsistent
direction, suggesting either:
- Flow direction varies between experiments
- Or image orientation varies in the dataset
""")

    # Save results
    per_image.to_csv(run_dir / "polarity_with_flow_inference.csv", index=False)
    print(f"\nSaved: {run_dir / 'polarity_with_flow_inference.csv'}")

    results = {
        'flow_inference': flow_info,
        'per_image': per_image.to_dict(orient='records'),
        'summary_by_condition': {
            cond: {
                'n_images': int(len(per_image[per_image['condition'] == cond])),
                'R_mean': float(per_image[per_image['condition'] == cond]['R'].mean()),
                'R_std': float(per_image[per_image['condition'] == cond]['R'].std()),
            }
            for cond in per_image['condition'].unique()
        }
    }

    with open(run_dir / "polarity_flow_inference_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {run_dir / 'polarity_flow_inference_results.json'}")

    return per_image, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Pipeline run directory")
    args = parser.parse_args()

    analyze_with_flow_inference(Path(args.run_dir))


if __name__ == "__main__":
    main()
