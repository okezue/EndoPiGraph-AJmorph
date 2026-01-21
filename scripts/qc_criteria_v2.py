#!/usr/bin/env python3
"""
Pipeline Correctness Criteria for EndoPiGraph-AJmorph (v2)
==========================================================

Validates pipeline outputs against objective QC thresholds
based on Polarity-JaM paper expectations for endothelial monolayers.

Improvements over v1:
- Computes ALL declared metrics
- Outputs qc_summary.csv with numeric values
- Generates auto-exclusion list
- More detailed per-image reporting

Usage:
    python scripts/qc_criteria_v2.py runs/sbiad1540_full/
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np


# =============================================================================
# QC THRESHOLDS (based on Polarity-JaM paper expectations)
# =============================================================================

SEGMENTATION_CRITERIA = {
    "min_cells_per_image": 20,          # confluent monolayers should have many cells
    "max_cells_per_image": 500,         # upper bound to catch over-segmentation
    "median_cell_area_min_px": 500,     # cells shouldn't be tiny
    "median_cell_area_max_px": 50000,   # cells shouldn't be giant
    "max_giant_cell_frac": 0.05,        # at most 5% "giant" cells (likely merges)
    "max_tiny_cell_frac": 0.1,          # at most 10% tiny cells (likely debris)
    "giant_cell_threshold_px": 20000,   # area threshold for "giant"
    "tiny_cell_threshold_px": 200,      # area threshold for "tiny"
}

GRAPH_CRITERIA = {
    "min_edges_per_image": 10,          # need some interfaces
    "min_mean_degree": 2.0,             # cells should have neighbors
    "max_mean_degree": 10.0,            # but not too many (over-segmentation)
    "max_isolated_node_frac": 0.1,      # at most 10% isolated cells
}

FEATURE_CRITERIA = {
    "occupancy_range": (0.0, 1.0),      # must be in [0, 1]
    "intensity_nonnegative": True,       # intensities >= 0
    "skeleton_len_nonnegative": True,    # skeleton length >= 0
    "cluster_count_nonnegative": True,   # cluster count >= 0
}


# =============================================================================
# QC Functions
# =============================================================================

def compute_qc_metrics(cells_df: pd.DataFrame, edges_df: pd.DataFrame, image_id: str) -> dict:
    """Compute all QC metrics for one image."""
    metrics = {"image_id": image_id}

    n_cells = len(cells_df)
    n_edges = len(edges_df)

    # Basic counts
    metrics["n_cells"] = n_cells
    metrics["n_edges"] = n_edges

    # Cell area metrics
    if "area" in cells_df.columns and n_cells > 0:
        areas = cells_df["area"]
        metrics["median_cell_area_px"] = float(areas.median())
        metrics["mean_cell_area_px"] = float(areas.mean())
        metrics["min_cell_area_px"] = float(areas.min())
        metrics["max_cell_area_px"] = float(areas.max())

        # Giant and tiny fractions
        giant_mask = areas > SEGMENTATION_CRITERIA["giant_cell_threshold_px"]
        tiny_mask = areas < SEGMENTATION_CRITERIA["tiny_cell_threshold_px"]
        metrics["giant_cell_frac"] = float(giant_mask.mean())
        metrics["tiny_cell_frac"] = float(tiny_mask.mean())
    else:
        metrics["median_cell_area_px"] = np.nan
        metrics["mean_cell_area_px"] = np.nan
        metrics["giant_cell_frac"] = np.nan
        metrics["tiny_cell_frac"] = np.nan

    # Graph metrics
    if n_cells > 0 and n_edges >= 0:
        metrics["mean_degree"] = 2 * n_edges / n_cells if n_cells > 0 else 0.0

        # Isolated nodes
        if "cell_i" in edges_df.columns and "cell_j" in edges_df.columns:
            cells_with_edges = set()
            cells_with_edges.update(edges_df["cell_i"].unique())
            cells_with_edges.update(edges_df["cell_j"].unique())

            if "cell_id" in cells_df.columns:
                all_cells = set(cells_df["cell_id"].unique())
                isolated = all_cells - cells_with_edges
                metrics["isolated_node_frac"] = len(isolated) / len(all_cells) if all_cells else 0.0
                metrics["n_isolated"] = len(isolated)
            else:
                metrics["isolated_node_frac"] = np.nan
                metrics["n_isolated"] = np.nan
        else:
            metrics["isolated_node_frac"] = np.nan
    else:
        metrics["mean_degree"] = np.nan
        metrics["isolated_node_frac"] = np.nan

    # Feature metrics
    if len(edges_df) > 0:
        if "aj_occupancy" in edges_df.columns:
            occ = edges_df["aj_occupancy"]
            metrics["occ_min"] = float(occ.min())
            metrics["occ_max"] = float(occ.max())
            metrics["occ_out_of_range"] = (occ < 0).any() or (occ > 1).any()

        # Check for negative values
        for col in ["aj_mean_intensity", "aj_max_intensity", "aj_skeleton_len", "aj_cluster_count"]:
            if col in edges_df.columns:
                metrics[f"{col}_has_negative"] = (edges_df[col] < 0).any()

        # NaN check
        nan_cols = edges_df.columns[edges_df.isna().any()].tolist()
        metrics["n_nan_columns"] = len(nan_cols)
        metrics["nan_columns"] = ",".join(nan_cols) if nan_cols else ""

    return metrics


def evaluate_metrics(metrics: dict) -> dict:
    """Evaluate metrics against thresholds and generate pass/fail status."""
    results = {
        "image_id": metrics["image_id"],
        "seg_passed": True,
        "graph_passed": True,
        "feature_passed": True,
        "issues": []
    }

    # Segmentation checks
    n_cells = metrics.get("n_cells", 0)
    if n_cells < SEGMENTATION_CRITERIA["min_cells_per_image"]:
        results["issues"].append(f"Too few cells: {n_cells}")
        results["seg_passed"] = False
    if n_cells > SEGMENTATION_CRITERIA["max_cells_per_image"]:
        results["issues"].append(f"Too many cells: {n_cells}")
        results["seg_passed"] = False

    median_area = metrics.get("median_cell_area_px", np.nan)
    if not np.isnan(median_area):
        if median_area < SEGMENTATION_CRITERIA["median_cell_area_min_px"]:
            results["issues"].append(f"Median area too small: {median_area:.0f}")
            results["seg_passed"] = False
        if median_area > SEGMENTATION_CRITERIA["median_cell_area_max_px"]:
            results["issues"].append(f"Median area too large: {median_area:.0f}")
            results["seg_passed"] = False

    giant_frac = metrics.get("giant_cell_frac", 0)
    if giant_frac > SEGMENTATION_CRITERIA["max_giant_cell_frac"]:
        results["issues"].append(f"Too many giant cells: {giant_frac:.1%}")
        results["seg_passed"] = False

    tiny_frac = metrics.get("tiny_cell_frac", 0)
    if tiny_frac > SEGMENTATION_CRITERIA["max_tiny_cell_frac"]:
        results["issues"].append(f"Too many tiny cells: {tiny_frac:.1%}")
        results["seg_passed"] = False

    # Graph checks
    n_edges = metrics.get("n_edges", 0)
    if n_edges < GRAPH_CRITERIA["min_edges_per_image"]:
        results["issues"].append(f"Too few edges: {n_edges}")
        results["graph_passed"] = False

    mean_deg = metrics.get("mean_degree", np.nan)
    if not np.isnan(mean_deg):
        if mean_deg < GRAPH_CRITERIA["min_mean_degree"]:
            results["issues"].append(f"Mean degree too low: {mean_deg:.2f}")
            results["graph_passed"] = False
        if mean_deg > GRAPH_CRITERIA["max_mean_degree"]:
            results["issues"].append(f"Mean degree too high: {mean_deg:.2f}")
            results["graph_passed"] = False

    iso_frac = metrics.get("isolated_node_frac", 0)
    if iso_frac > GRAPH_CRITERIA["max_isolated_node_frac"]:
        results["issues"].append(f"Too many isolated cells: {iso_frac:.1%}")
        results["graph_passed"] = False

    # Feature checks
    if metrics.get("occ_out_of_range", False):
        occ_min = metrics.get("occ_min", 0)
        occ_max = metrics.get("occ_max", 1)
        results["issues"].append(f"Occupancy out of [0,1]: [{occ_min:.2f}, {occ_max:.2f}]")
        results["feature_passed"] = False

    for col in ["aj_mean_intensity", "aj_max_intensity", "aj_skeleton_len", "aj_cluster_count"]:
        if metrics.get(f"{col}_has_negative", False):
            results["issues"].append(f"{col} has negative values")
            results["feature_passed"] = False

    if metrics.get("n_nan_columns", 0) > 0:
        results["issues"].append(f"NaN in columns: {metrics.get('nan_columns', '')}")
        results["feature_passed"] = False

    results["overall_passed"] = results["seg_passed"] and results["graph_passed"] and results["feature_passed"]

    return results


def run_qc(run_dir: Path) -> tuple:
    """Run all QC checks on a pipeline run.

    Returns:
        (metrics_df, results_df, summary)
    """
    run_dir = Path(run_dir)

    # Find image subdirectories
    image_dirs = [d for d in run_dir.iterdir() if d.is_dir() and (d / "cells.csv").exists()]

    all_metrics = []
    all_results = []

    for img_dir in sorted(image_dirs):
        image_id = img_dir.name

        cells_df = pd.read_csv(img_dir / "cells.csv")
        edges_df = pd.read_csv(img_dir / "edges.csv") if (img_dir / "edges.csv").exists() else pd.DataFrame()

        metrics = compute_qc_metrics(cells_df, edges_df, image_id)
        results = evaluate_metrics(metrics)

        all_metrics.append(metrics)
        all_results.append(results)

    metrics_df = pd.DataFrame(all_metrics)
    results_df = pd.DataFrame(all_results)

    # Generate summary
    n_images = len(results_df)
    n_seg_pass = results_df["seg_passed"].sum()
    n_graph_pass = results_df["graph_passed"].sum()
    n_feature_pass = results_df["feature_passed"].sum()
    n_overall_pass = results_df["overall_passed"].sum()

    summary = {
        "run_dir": str(run_dir),
        "n_images": n_images,
        "n_cells_total": int(metrics_df["n_cells"].sum()),
        "n_edges_total": int(metrics_df["n_edges"].sum()),
        "seg_passed": f"{n_seg_pass}/{n_images}",
        "graph_passed": f"{n_graph_pass}/{n_images}",
        "feature_passed": f"{n_feature_pass}/{n_images}",
        "overall_passed": f"{n_overall_pass}/{n_images}",
        "excluded_images": results_df[~results_df["overall_passed"]]["image_id"].tolist()
    }

    return metrics_df, results_df, summary


def print_qc_report(metrics_df: pd.DataFrame, results_df: pd.DataFrame, summary: dict):
    """Print a human-readable QC report."""
    print("="*70)
    print("PIPELINE QC REPORT (v2)")
    print("="*70)
    print(f"Run directory: {summary['run_dir']}")
    print(f"Images processed: {summary['n_images']}")
    print(f"Total cells: {summary['n_cells_total']}")
    print(f"Total edges: {summary['n_edges_total']}")
    print()

    # Summary
    print("SUMMARY")
    print("-"*40)
    print(f"Segmentation QC: {summary['seg_passed']} passed")
    print(f"Graph QC:        {summary['graph_passed']} passed")
    print(f"Feature QC:      {summary['feature_passed']} passed")
    print(f"Overall:         {summary['overall_passed']} passed")
    print()

    # Failed images
    failed = results_df[~results_df["overall_passed"]]
    if len(failed) > 0:
        print("FAILED IMAGES")
        print("-"*40)
        for _, row in failed.iterrows():
            print(f"  {row['image_id']}: {'; '.join(row['issues'])}")
        print()

    # Key metrics table
    print("KEY METRICS BY IMAGE")
    print("-"*40)
    cols = ["image_id", "n_cells", "n_edges", "mean_degree", "giant_cell_frac", "isolated_node_frac"]
    cols = [c for c in cols if c in metrics_df.columns]
    print(metrics_df[cols].to_string(index=False))
    print()

    # Thresholds reference
    print("QC THRESHOLDS")
    print("-"*40)
    print(f"Cells per image:     {SEGMENTATION_CRITERIA['min_cells_per_image']}-{SEGMENTATION_CRITERIA['max_cells_per_image']}")
    print(f"Mean degree:         {GRAPH_CRITERIA['min_mean_degree']}-{GRAPH_CRITERIA['max_mean_degree']}")
    print(f"Max giant cell frac: {SEGMENTATION_CRITERIA['max_giant_cell_frac']:.0%}")
    print(f"Max isolated frac:   {GRAPH_CRITERIA['max_isolated_node_frac']:.0%}")


def main():
    parser = argparse.ArgumentParser(description="Run QC checks on pipeline output (v2)")
    parser.add_argument("run_dir", help="Path to pipeline run directory")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of text")
    parser.add_argument("--output-dir", default=None, help="Output directory for CSV/JSON files")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir

    metrics_df, results_df, summary = run_qc(run_dir)

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print_qc_report(metrics_df, results_df, summary)

    # Save outputs
    metrics_df.to_csv(output_dir / "qc_metrics.csv", index=False)
    print(f"\nSaved: {output_dir / 'qc_metrics.csv'}")

    results_df.to_csv(output_dir / "qc_results.csv", index=False)
    print(f"Saved: {output_dir / 'qc_results.csv'}")

    # Exclusion list
    excluded = results_df[~results_df["overall_passed"]]["image_id"].tolist()
    with open(output_dir / "qc_excluded.txt", "w") as f:
        f.write("\n".join(excluded))
    print(f"Saved: {output_dir / 'qc_excluded.txt'} ({len(excluded)} images)")

    # JSON summary
    with open(output_dir / "qc_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {output_dir / 'qc_summary.json'}")

    sys.exit(0 if len(excluded) == 0 else 1)


if __name__ == "__main__":
    main()
