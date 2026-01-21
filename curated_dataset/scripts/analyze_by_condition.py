#!/usr/bin/env python3
"""
Analyze EndoPiGraph results by experimental condition (shear stress).

This script compares AJ morphology, polarity, and graph features
across different shear stress conditions to validate the pipeline
reproduces expected biological effects.

Usage:
    python scripts/analyze_by_condition.py runs/sbiad1540_full/
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_run_data(run_dir: Path) -> tuple:
    """Load all data from a pipeline run."""
    cells = pd.read_csv(run_dir / "all_cells.csv")
    edges = pd.read_csv(run_dir / "all_edges.csv")
    polarity = pd.read_csv(run_dir / "all_polarity.csv")
    return cells, edges, polarity


def extract_shear_condition(shear_stress: str) -> str:
    """Normalize shear stress values to simple categories."""
    if pd.isna(shear_stress):
        return "unknown"
    s = str(shear_stress).lower()
    if "static" in s:
        return "static"
    elif "6dyne" in s or "6 dyne" in s:
        return "6dyne"
    elif "18dyne" in s or "18 dyne" in s or "20dyne" in s or "20 dyne" in s:
        return "high_shear"
    else:
        return "other"


def analyze_ajmorph_by_condition(edges: pd.DataFrame) -> pd.DataFrame:
    """Analyze AJ morphology distribution by shear condition."""
    edges = edges.copy()
    edges["condition"] = edges["shear_stress"].apply(extract_shear_condition)

    # Count AJ morphology classes per condition
    pivot = pd.crosstab(edges["condition"], edges["aj_morph"], normalize="index") * 100
    return pivot


def analyze_polarity_by_condition(polarity: pd.DataFrame) -> dict:
    """Analyze polarity vectors by shear condition."""
    polarity = polarity.copy()
    polarity["condition"] = polarity["shear_stress"].apply(extract_shear_condition)

    results = {}
    for cond, group in polarity.groupby("condition"):
        if "polarity_angle_deg" in group.columns:
            angles = np.deg2rad(group["polarity_angle_deg"].dropna())
            # Circular mean and std
            if len(angles) > 0:
                mean_cos = np.mean(np.cos(angles))
                mean_sin = np.mean(np.sin(angles))
                mean_angle = np.rad2deg(np.arctan2(mean_sin, mean_cos))
                # Circular variance (1 - R, where R is mean resultant length)
                R = np.sqrt(mean_cos**2 + mean_sin**2)
                circular_var = 1 - R
                results[cond] = {
                    "n": len(angles),
                    "mean_angle": mean_angle,
                    "R": R,  # alignment strength (0=random, 1=perfect)
                    "circular_var": circular_var,
                }
    return results


def analyze_features_by_condition(edges: pd.DataFrame) -> pd.DataFrame:
    """Analyze AJ features by shear condition."""
    edges = edges.copy()
    edges["condition"] = edges["shear_stress"].apply(extract_shear_condition)

    feature_cols = [
        "aj_mean_intensity", "aj_occupancy", "aj_cluster_count",
        "aj_skeleton_len", "aj_linearity_index", "aj_thickness_proxy"
    ]
    feature_cols = [c for c in feature_cols if c in edges.columns]

    summary = edges.groupby("condition")[feature_cols].agg(["mean", "std", "median"])
    return summary


def plot_ajmorph_distribution(pivot: pd.DataFrame, output_path: Path):
    """Plot AJ morphology distribution by condition."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Reorder columns for consistent display
    col_order = ["straight", "thick", "thick_to_reticular", "reticular", "fingers", "other"]
    col_order = [c for c in col_order if c in pivot.columns]
    pivot = pivot[col_order]

    pivot.plot(kind="bar", ax=ax, width=0.8)
    ax.set_xlabel("Shear Stress Condition")
    ax.set_ylabel("Percentage of Interfaces")
    ax.set_title("AJ Morphology Distribution by Shear Condition")
    ax.legend(title="AJ Morphology", bbox_to_anchor=(1.02, 1))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_polarity_roses(polarity: pd.DataFrame, output_path: Path):
    """Plot polarity rose diagrams by condition."""
    polarity = polarity.copy()
    polarity["condition"] = polarity["shear_stress"].apply(extract_shear_condition)

    conditions = sorted(polarity["condition"].unique())
    n_cond = len(conditions)

    fig, axes = plt.subplots(1, n_cond, figsize=(4*n_cond, 4),
                              subplot_kw={"projection": "polar"})
    if n_cond == 1:
        axes = [axes]

    for ax, cond in zip(axes, conditions):
        group = polarity[polarity["condition"] == cond]
        if "polarity_angle_deg" in group.columns:
            angles = np.deg2rad(group["polarity_angle_deg"].dropna())
            if len(angles) > 0:
                # Histogram
                n_bins = 36
                hist, bin_edges = np.histogram(angles, bins=n_bins, range=(-np.pi, np.pi))
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                ax.bar(bin_centers, hist, width=2*np.pi/n_bins, alpha=0.7)
                ax.set_title(f"{cond}\n(n={len(angles)})")

                # Add mean direction arrow
                mean_cos = np.mean(np.cos(angles))
                mean_sin = np.mean(np.sin(angles))
                R = np.sqrt(mean_cos**2 + mean_sin**2)
                mean_angle = np.arctan2(mean_sin, mean_cos)
                ax.annotate("", xy=(mean_angle, R * max(hist)), xytext=(0, 0),
                           arrowprops=dict(arrowstyle="->", color="red", lw=2))

    plt.suptitle("Cell Polarity by Shear Condition")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze results by condition")
    parser.add_argument("run_dir", help="Path to pipeline run directory")
    parser.add_argument("--output-dir", default=None, help="Output directory for plots")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir

    print(f"Loading data from {run_dir}...")
    cells, edges, polarity = load_run_data(run_dir)

    print(f"\nTotal cells: {len(cells)}")
    print(f"Total edges: {len(edges)}")
    print(f"Total polarity entries: {len(polarity)}")

    # AJ Morphology analysis
    print("\n" + "="*60)
    print("AJ MORPHOLOGY BY CONDITION")
    print("="*60)
    ajmorph_pivot = analyze_ajmorph_by_condition(edges)
    print(ajmorph_pivot.round(1).to_string())
    plot_ajmorph_distribution(ajmorph_pivot, output_dir / "ajmorph_by_condition.png")

    # Polarity analysis
    print("\n" + "="*60)
    print("POLARITY BY CONDITION")
    print("="*60)
    polarity_stats = analyze_polarity_by_condition(polarity)
    for cond, stats in polarity_stats.items():
        print(f"{cond}: n={stats['n']}, mean_angle={stats['mean_angle']:.1f}°, "
              f"R={stats['R']:.3f} (alignment strength)")

    # Expected: static should have low R (~0), flow should have high R (~0.5+)
    print("\nExpectation: R~0 for static (random), R>0.3 for flow (aligned)")

    if len(polarity) > 0:
        plot_polarity_roses(polarity, output_dir / "polarity_roses_by_condition.png")

    # Feature analysis
    print("\n" + "="*60)
    print("FEATURE SUMMARY BY CONDITION")
    print("="*60)
    feat_summary = analyze_features_by_condition(edges)
    print(feat_summary.round(3).to_string())

    # Save summary JSON
    summary = {
        "run_dir": str(run_dir),
        "n_cells": len(cells),
        "n_edges": len(edges),
        "ajmorph_by_condition": ajmorph_pivot.to_dict(),
        "polarity_by_condition": polarity_stats,
    }
    with open(output_dir / "condition_analysis.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {output_dir / 'condition_analysis.json'}")


if __name__ == "__main__":
    main()
