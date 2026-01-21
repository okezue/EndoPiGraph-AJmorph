#!/usr/bin/env python3
"""
Pipeline Correctness Criteria for EndoPiGraph-AJmorph
=====================================================

This script validates pipeline outputs against objective QC thresholds
based on Polarity-JaM paper expectations for endothelial monolayers.

Usage:
    python scripts/qc_criteria.py runs/sbiad1540_full/
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
    "max_border_touching_frac": 0.3,    # at most 30% touching border
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

def check_segmentation_qc(cells_df: pd.DataFrame, image_id: str) -> dict:
    """Check segmentation QC criteria for one image."""
    results = {"image_id": image_id, "passed": True, "issues": []}

    n_cells = len(cells_df)

    # Cell count
    if n_cells < SEGMENTATION_CRITERIA["min_cells_per_image"]:
        results["issues"].append(f"Too few cells: {n_cells}")
        results["passed"] = False
    if n_cells > SEGMENTATION_CRITERIA["max_cells_per_image"]:
        results["issues"].append(f"Too many cells (over-segmentation?): {n_cells}")
        results["passed"] = False

    # Cell area
    if "area" in cells_df.columns:
        median_area = cells_df["area"].median()
        if median_area < SEGMENTATION_CRITERIA["median_cell_area_min_px"]:
            results["issues"].append(f"Median cell area too small: {median_area:.0f}")
            results["passed"] = False
        if median_area > SEGMENTATION_CRITERIA["median_cell_area_max_px"]:
            results["issues"].append(f"Median cell area too large: {median_area:.0f}")
            results["passed"] = False

        # Giant cells
        giant_frac = (cells_df["area"] > SEGMENTATION_CRITERIA["giant_cell_threshold_px"]).mean()
        if giant_frac > SEGMENTATION_CRITERIA["max_giant_cell_frac"]:
            results["issues"].append(f"Too many giant cells: {giant_frac:.1%}")
            results["passed"] = False

        # Tiny cells
        tiny_frac = (cells_df["area"] < SEGMENTATION_CRITERIA["tiny_cell_threshold_px"]).mean()
        if tiny_frac > SEGMENTATION_CRITERIA["max_tiny_cell_frac"]:
            results["issues"].append(f"Too many tiny cells: {tiny_frac:.1%}")
            results["passed"] = False

    results["n_cells"] = n_cells
    results["median_area"] = cells_df["area"].median() if "area" in cells_df.columns else None

    return results


def check_graph_qc(cells_df: pd.DataFrame, edges_df: pd.DataFrame, image_id: str) -> dict:
    """Check graph QC criteria for one image."""
    results = {"image_id": image_id, "passed": True, "issues": []}

    n_cells = len(cells_df)
    n_edges = len(edges_df)

    # Edge count
    if n_edges < GRAPH_CRITERIA["min_edges_per_image"]:
        results["issues"].append(f"Too few edges: {n_edges}")
        results["passed"] = False

    # Mean degree
    if n_cells > 0:
        mean_degree = 2 * n_edges / n_cells
        if mean_degree < GRAPH_CRITERIA["min_mean_degree"]:
            results["issues"].append(f"Mean degree too low: {mean_degree:.2f}")
            results["passed"] = False
        if mean_degree > GRAPH_CRITERIA["max_mean_degree"]:
            results["issues"].append(f"Mean degree too high: {mean_degree:.2f}")
            results["passed"] = False
        results["mean_degree"] = mean_degree

    # Isolated nodes
    if n_cells > 0 and n_edges > 0:
        # Count cells that appear in edges
        cells_with_edges = set()
        if "cell_i" in edges_df.columns and "cell_j" in edges_df.columns:
            cells_with_edges.update(edges_df["cell_i"].unique())
            cells_with_edges.update(edges_df["cell_j"].unique())

        if "cell_id" in cells_df.columns:
            all_cells = set(cells_df["cell_id"].unique())
            isolated = all_cells - cells_with_edges
            isolated_frac = len(isolated) / len(all_cells)
            if isolated_frac > GRAPH_CRITERIA["max_isolated_node_frac"]:
                results["issues"].append(f"Too many isolated cells: {isolated_frac:.1%}")
                results["passed"] = False
            results["isolated_frac"] = isolated_frac

    results["n_edges"] = n_edges

    return results


def check_feature_qc(edges_df: pd.DataFrame, image_id: str) -> dict:
    """Check feature QC criteria."""
    results = {"image_id": image_id, "passed": True, "issues": []}

    # Occupancy range
    if "aj_occupancy" in edges_df.columns:
        occ = edges_df["aj_occupancy"]
        if occ.min() < 0 or occ.max() > 1:
            results["issues"].append(f"Occupancy out of [0,1]: [{occ.min():.2f}, {occ.max():.2f}]")
            results["passed"] = False

    # Non-negative features
    for col in ["aj_mean_intensity", "aj_max_intensity", "aj_skeleton_len", "aj_cluster_count"]:
        if col in edges_df.columns:
            if (edges_df[col] < 0).any():
                results["issues"].append(f"{col} has negative values")
                results["passed"] = False

    # NaN check
    nan_cols = edges_df.columns[edges_df.isna().any()].tolist()
    if nan_cols:
        results["issues"].append(f"NaN values in columns: {nan_cols}")
        results["passed"] = False

    return results


def run_qc(run_dir: Path) -> dict:
    """Run all QC checks on a pipeline run."""
    run_dir = Path(run_dir)

    # Load aggregate files
    cells_path = run_dir / "all_cells.csv"
    edges_path = run_dir / "all_edges.csv"
    report_path = run_dir / "run_report.json"

    if not cells_path.exists():
        return {"error": f"Missing {cells_path}"}

    cells_df = pd.read_csv(cells_path)
    edges_df = pd.read_csv(edges_path) if edges_path.exists() else pd.DataFrame()

    with open(report_path) as f:
        report = json.load(f)

    # Get image IDs from edges (which has image_id) or subdirectories
    if "image_id" in edges_df.columns:
        image_ids = edges_df["image_id"].unique()
    elif "image_id" in cells_df.columns:
        image_ids = cells_df["image_id"].unique()
    else:
        # Fall back to subdirectories
        image_ids = [d.name for d in run_dir.iterdir() if d.is_dir() and (d / "cells.csv").exists()]
        if not image_ids:
            image_ids = ["all"]

    all_results = {
        "run_dir": str(run_dir),
        "n_images": report.get("n_images", len(image_ids)),
        "n_cells_total": report.get("n_cells", len(cells_df)),
        "n_edges_total": report.get("n_edges", len(edges_df)),
        "segmentation_qc": [],
        "graph_qc": [],
        "feature_qc": [],
        "overall_pass": True,
    }

    # Check each image
    for img_id in image_ids:
        # Try to load per-image data from subdirectory first
        img_dir = run_dir / img_id
        if img_dir.exists() and (img_dir / "cells.csv").exists():
            img_cells = pd.read_csv(img_dir / "cells.csv")
            img_edges = pd.read_csv(img_dir / "edges.csv") if (img_dir / "edges.csv").exists() else pd.DataFrame()
        else:
            # Fall back to filtering aggregate data
            if "image_id" in edges_df.columns:
                img_edges = edges_df[edges_df["image_id"] == img_id]
            else:
                img_edges = edges_df
            if "image_id" in cells_df.columns:
                img_cells = cells_df[cells_df["image_id"] == img_id]
            else:
                img_cells = cells_df

        seg_qc = check_segmentation_qc(img_cells, img_id)
        graph_qc = check_graph_qc(img_cells, img_edges, img_id)
        feat_qc = check_feature_qc(img_edges, img_id)

        all_results["segmentation_qc"].append(seg_qc)
        all_results["graph_qc"].append(graph_qc)
        all_results["feature_qc"].append(feat_qc)

        if not seg_qc["passed"] or not graph_qc["passed"] or not feat_qc["passed"]:
            all_results["overall_pass"] = False

    return all_results


def print_qc_report(results: dict):
    """Print a human-readable QC report."""
    print("=" * 60)
    print("PIPELINE QC REPORT")
    print("=" * 60)
    print(f"Run directory: {results['run_dir']}")
    print(f"Images processed: {results['n_images']}")
    print(f"Total cells: {results['n_cells_total']}")
    print(f"Total edges: {results['n_edges_total']}")
    print()

    # Segmentation summary
    print("SEGMENTATION QC")
    print("-" * 40)
    seg_passed = sum(1 for r in results["segmentation_qc"] if r["passed"])
    print(f"Passed: {seg_passed}/{len(results['segmentation_qc'])} images")
    for r in results["segmentation_qc"]:
        if not r["passed"]:
            print(f"  FAIL {r['image_id']}: {', '.join(r['issues'])}")
    print()

    # Graph summary
    print("GRAPH QC")
    print("-" * 40)
    graph_passed = sum(1 for r in results["graph_qc"] if r["passed"])
    print(f"Passed: {graph_passed}/{len(results['graph_qc'])} images")
    for r in results["graph_qc"]:
        if not r["passed"]:
            print(f"  FAIL {r['image_id']}: {', '.join(r['issues'])}")
        else:
            deg = r.get("mean_degree", 0)
            print(f"  OK {r['image_id']}: {r['n_edges']} edges, degree={deg:.2f}")
    print()

    # Feature summary
    print("FEATURE QC")
    print("-" * 40)
    feat_passed = sum(1 for r in results["feature_qc"] if r["passed"])
    print(f"Passed: {feat_passed}/{len(results['feature_qc'])} images")
    for r in results["feature_qc"]:
        if not r["passed"]:
            print(f"  FAIL {r['image_id']}: {', '.join(r['issues'])}")
    print()

    # Overall
    print("=" * 60)
    if results["overall_pass"]:
        print("OVERALL: PASS")
    else:
        print("OVERALL: FAIL - see issues above")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run QC checks on pipeline output")
    parser.add_argument("run_dir", help="Path to pipeline run directory")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of text")
    args = parser.parse_args()

    results = run_qc(args.run_dir)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_qc_report(results)

    sys.exit(0 if results.get("overall_pass", False) else 1)


if __name__ == "__main__":
    main()
