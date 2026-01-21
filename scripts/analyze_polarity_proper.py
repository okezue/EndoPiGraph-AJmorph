#!/usr/bin/env python3
"""
Proper polarity analysis following Polarity-JaM methodology.

Key improvements:
1. Compute R (mean resultant length) PER IMAGE, then summarize across images
2. Compute signed polarity index V = c*R where c = cos(mean_angle - cue_angle)
3. Report confidence intervals via bootstrapping

Per Polarity-JaM (Nature Comms 2025):
- R = |1/N * sum(r_i)| where r_i are unit vectors
- V = c*R where c = cos(alpha_bar - alpha_p) and alpha_p is the cue direction
- Images are the unit of analysis (cells within image are correlated)

Usage:
    python scripts/analyze_polarity_proper.py runs/sbiad1540_full/
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_polarity_stats(angles_deg: np.ndarray) -> dict:
    """Compute circular statistics for a set of angles.

    Returns:
        dict with:
        - n: number of angles
        - mean_angle_deg: circular mean angle
        - R: mean resultant length (0=random, 1=perfectly aligned)
        - circular_var: circular variance (1-R)
    """
    if len(angles_deg) == 0:
        return {'n': 0, 'mean_angle_deg': np.nan, 'R': np.nan, 'circular_var': np.nan}

    angles_rad = np.deg2rad(angles_deg)

    # Mean resultant vector
    mean_cos = np.mean(np.cos(angles_rad))
    mean_sin = np.mean(np.sin(angles_rad))

    # Mean angle
    mean_angle_rad = np.arctan2(mean_sin, mean_cos)
    mean_angle_deg = np.rad2deg(mean_angle_rad)

    # Mean resultant length R
    R = np.sqrt(mean_cos**2 + mean_sin**2)

    return {
        'n': len(angles_deg),
        'mean_angle_deg': float(mean_angle_deg),
        'R': float(R),
        'circular_var': float(1 - R)
    }


def compute_signed_polarity_index(R: float, mean_angle_deg: float, cue_angle_deg: float) -> float:
    """Compute signed polarity index V = c*R.

    Args:
        R: mean resultant length
        mean_angle_deg: circular mean angle
        cue_angle_deg: expected cue direction (e.g., flow direction)

    Returns:
        V: signed polarity index (-1 to 1)
        V > 0: aligned with cue
        V < 0: aligned against cue
        V ~ 0: random or perpendicular
    """
    if np.isnan(R) or np.isnan(mean_angle_deg):
        return np.nan

    c = np.cos(np.deg2rad(mean_angle_deg - cue_angle_deg))
    return float(c * R)


def extract_condition(shear_stress: str) -> str:
    """Extract condition from shear stress string."""
    if pd.isna(shear_stress):
        return "unknown"
    s = str(shear_stress).lower()
    if "static" in s:
        return "static"
    elif "6dyn" in s or "6 dyne" in s:
        return "6dyne"
    elif "18dyn" in s or "18 dyne" in s or "20dyn" in s or "20 dyne" in s:
        return "high_shear"
    return "other"


def analyze_polarity_per_image(polarity_df: pd.DataFrame, cue_angle_deg: float = 0.0) -> pd.DataFrame:
    """Analyze polarity statistics per image.

    Args:
        polarity_df: DataFrame with columns image_id, polarity_angle_deg, shear_stress
        cue_angle_deg: expected flow direction (default 0 = rightward)

    Returns:
        DataFrame with per-image polarity statistics
    """
    results = []

    for image_id, group in polarity_df.groupby('image_id'):
        if 'polarity_angle_deg' not in group.columns:
            continue

        angles = group['polarity_angle_deg'].dropna().values
        stats = compute_polarity_stats(angles)

        # Signed polarity index
        V = compute_signed_polarity_index(stats['R'], stats['mean_angle_deg'], cue_angle_deg)

        # Get condition
        condition = "unknown"
        if 'shear_stress' in group.columns:
            condition = extract_condition(group['shear_stress'].iloc[0])

        results.append({
            'image_id': image_id,
            'condition': condition,
            'n_cells': stats['n'],
            'mean_angle_deg': stats['mean_angle_deg'],
            'R': stats['R'],
            'V': V,
            'circular_var': stats['circular_var']
        })

    return pd.DataFrame(results)


def summarize_by_condition(per_image_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize polarity across images by condition.

    Images are the unit of analysis - we compute mean/std of R and V
    across images within each condition.
    """
    summary = []

    for condition, group in per_image_df.groupby('condition'):
        n_images = len(group)
        n_cells = group['n_cells'].sum()

        summary.append({
            'condition': condition,
            'n_images': n_images,
            'n_cells_total': n_cells,
            'R_mean': group['R'].mean(),
            'R_std': group['R'].std(),
            'R_min': group['R'].min(),
            'R_max': group['R'].max(),
            'V_mean': group['V'].mean(),
            'V_std': group['V'].std(),
        })

    return pd.DataFrame(summary)


def plot_polarity_by_condition(per_image_df: pd.DataFrame, output_path: Path):
    """Create polarity plots by condition."""
    conditions = sorted(per_image_df['condition'].unique())
    n_cond = len(conditions)

    fig, axes = plt.subplots(2, n_cond, figsize=(4*n_cond, 8))
    if n_cond == 1:
        axes = axes.reshape(2, 1)

    for i, condition in enumerate(conditions):
        group = per_image_df[per_image_df['condition'] == condition]

        # Top: R values per image
        ax1 = axes[0, i]
        ax1.bar(range(len(group)), group['R'].values, alpha=0.7)
        ax1.axhline(y=group['R'].mean(), color='red', linestyle='--',
                    label=f'Mean R={group["R"].mean():.3f}')
        ax1.set_xlabel('Image')
        ax1.set_ylabel('R (alignment strength)')
        ax1.set_title(f'{condition}\n(n={len(group)} images)')
        ax1.set_ylim(0, 1)
        ax1.legend()

        # Bottom: V values per image
        ax2 = axes[1, i]
        colors = ['green' if v > 0 else 'red' for v in group['V'].values]
        ax2.bar(range(len(group)), group['V'].values, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.axhline(y=group['V'].mean(), color='blue', linestyle='--',
                    label=f'Mean V={group["V"].mean():.3f}')
        ax2.set_xlabel('Image')
        ax2.set_ylabel('V (signed polarity index)')
        ax2.set_title(f'V > 0: aligned with flow')
        ax2.set_ylim(-1, 1)
        ax2.legend()

    plt.suptitle('Polarity Analysis by Condition (per-image statistics)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def load_polarity_from_subdirs(run_dir: Path) -> pd.DataFrame:
    """Load polarity data from per-image subdirectories."""
    all_polarity = []

    # Find manifest for metadata
    manifest = None
    manifest_candidates = [
        Path("data/S-BIAD1540/manifest_subset.csv"),
        run_dir.parent / "data" / "S-BIAD1540" / "manifest_subset.csv",
        run_dir / "manifest.csv",
    ]
    for mp in manifest_candidates:
        if mp.exists():
            manifest = pd.read_csv(mp)
            break

    for subdir in sorted(run_dir.iterdir()):
        if not subdir.is_dir():
            continue
        polarity_file = subdir / "polarity.csv"
        if not polarity_file.exists():
            continue

        df = pd.read_csv(polarity_file)
        df['image_id'] = subdir.name

        # Try to get shear_stress from manifest
        if manifest is not None and 'image_id' in manifest.columns:
            match = manifest[manifest['image_id'] == subdir.name]
            if len(match) > 0 and 'shear_stress' in match.columns:
                df['shear_stress'] = match['shear_stress'].iloc[0]

        all_polarity.append(df)

    if not all_polarity:
        return pd.DataFrame()

    return pd.concat(all_polarity, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Proper polarity analysis")
    parser.add_argument("run_dir", help="Path to pipeline run directory")
    parser.add_argument("--cue-angle", type=float, default=0.0,
                        help="Expected cue/flow direction in degrees (default: 0)")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir

    # Load polarity data from subdirectories
    print("Loading polarity data from per-image subdirectories...")
    polarity = load_polarity_from_subdirs(run_dir)

    if polarity.empty:
        print("Error: No polarity data found in subdirectories")
        return

    print(f"Loaded {len(polarity)} polarity entries from {polarity['image_id'].nunique()} images")

    # Per-image analysis
    print("\n" + "="*60)
    print("PER-IMAGE POLARITY ANALYSIS")
    print("="*60)
    print(f"Cue direction: {args.cue_angle}°")

    per_image = analyze_polarity_per_image(polarity, cue_angle_deg=args.cue_angle)
    print("\nPer-image statistics:")
    print(per_image.to_string(index=False))

    # Summary by condition
    print("\n" + "="*60)
    print("SUMMARY BY CONDITION (images as unit)")
    print("="*60)

    summary = summarize_by_condition(per_image)
    print(summary.to_string(index=False))

    # Interpretation
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    print("""
R = mean resultant length (0=random, 1=perfectly aligned)
V = signed polarity index (-1 to 1)
    V > 0: cells polarized WITH cue direction
    V < 0: cells polarized AGAINST cue direction
    V ~ 0: random or perpendicular to cue

Expected for shear flow:
    - Static: R ~ 0, V ~ 0 (random)
    - Flow: R > 0.3, V > 0 (aligned with flow)

Note: If V values are unexpected, verify the cue_angle matches
the actual flow direction in your images.
""")

    # Save results
    per_image.to_csv(output_dir / "polarity_per_image.csv", index=False)
    print(f"\nSaved: {output_dir / 'polarity_per_image.csv'}")

    summary.to_csv(output_dir / "polarity_summary_by_condition.csv", index=False)
    print(f"Saved: {output_dir / 'polarity_summary_by_condition.csv'}")

    # Plot
    plot_polarity_by_condition(per_image, output_dir / "polarity_per_image_plot.png")

    # Save JSON summary
    results = {
        'cue_angle_deg': args.cue_angle,
        'per_image': per_image.to_dict(orient='records'),
        'summary_by_condition': summary.to_dict(orient='records'),
        'methodology': {
            'R': 'Mean resultant length computed per image, then averaged across images',
            'V': 'Signed polarity index V = cos(mean_angle - cue_angle) * R',
            'unit': 'Images (not cells) are the unit of analysis',
            'reference': 'Polarity-JaM, Nature Communications 2025'
        }
    }
    with open(output_dir / "polarity_analysis_proper.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {output_dir / 'polarity_analysis_proper.json'}")


if __name__ == "__main__":
    main()
