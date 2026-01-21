#!/usr/bin/env python3
"""
Harden network discovery statistics with proper per-image replicate testing.

Fixes pseudo-replication issues by:
1. Computing metrics PER IMAGE first
2. Then testing across images (proper n = number of images, not cells/edges)
3. Reporting effect sizes with confidence intervals
"""

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
import networkx as nx


def load_all_data(runs_dir: Path):
    """Load all cells and edges data, tagged by image and condition."""
    all_cells = []
    all_edges = []

    # Try runs_dir/results first, then runs_dir directly
    results_dir = runs_dir / "results"
    if not results_dir.exists():
        results_dir = runs_dir

    for img_dir in sorted(results_dir.iterdir()):
        if not img_dir.is_dir():
            continue

        cells_file = img_dir / "cells.csv"
        edges_file = img_dir / "edges.csv"

        if not cells_file.exists() or not edges_file.exists():
            continue

        try:
            cells = pd.read_csv(cells_file)
            edges = pd.read_csv(edges_file)
        except Exception:
            continue

        if len(cells) == 0 or len(edges) == 0:
            continue

        image_id = img_dir.name

        # Determine condition from image name
        if "static" in image_id.lower():
            condition = "static"
        elif "6dyn" in image_id.lower() or "6dyne" in image_id.lower():
            condition = "6dyne"
        elif "20dyn" in image_id.lower():
            condition = "20dyne"
        else:
            condition = "unknown"

        cells["image_id"] = image_id
        cells["condition"] = condition
        edges["image_id"] = image_id
        edges["condition"] = condition

        all_cells.append(cells)
        all_edges.append(edges)

    return pd.concat(all_cells, ignore_index=True), pd.concat(all_edges, ignore_index=True)


def compute_per_image_clustering(cells_df, edges_df):
    """Compute mean clustering coefficient per image."""
    results = []

    for image_id in cells_df["image_id"].unique():
        img_cells = cells_df[cells_df["image_id"] == image_id]
        img_edges = edges_df[edges_df["image_id"] == image_id]
        condition = img_cells["condition"].iloc[0]

        # Build graph
        G = nx.Graph()
        G.add_nodes_from(img_cells["cell_id"].values)
        for _, row in img_edges.iterrows():
            G.add_edge(row["cell_i"], row["cell_j"])

        # Compute clustering per node, then mean
        clustering = nx.clustering(G)
        mean_clustering = np.mean(list(clustering.values())) if clustering else 0

        results.append({
            "image_id": image_id,
            "condition": condition,
            "mean_clustering": mean_clustering,
            "n_cells": len(img_cells),
            "n_edges": len(img_edges)
        })

    return pd.DataFrame(results)


def compute_per_image_reticular_pct(edges_df):
    """Compute reticular junction percentage per image."""
    results = []

    for image_id in edges_df["image_id"].unique():
        img_edges = edges_df[edges_df["image_id"] == image_id]
        condition = img_edges["condition"].iloc[0]

        if "aj_morph" not in img_edges.columns:
            continue

        total = len(img_edges)
        reticular = (img_edges["aj_morph"] == "reticular").sum()
        pct = reticular / total * 100 if total > 0 else 0

        results.append({
            "image_id": image_id,
            "condition": condition,
            "reticular_pct": pct,
            "n_reticular": reticular,
            "n_total": total
        })

    return pd.DataFrame(results)


def compute_per_image_degree_occupancy_corr(cells_df, edges_df):
    """Compute degree-occupancy Spearman correlation within each image."""
    results = []

    for image_id in cells_df["image_id"].unique():
        img_cells = cells_df[cells_df["image_id"] == image_id]
        img_edges = edges_df[edges_df["image_id"] == image_id]
        condition = img_cells["condition"].iloc[0]

        # Compute degree for each cell
        degree_counts = defaultdict(int)
        occupancy_sums = defaultdict(list)

        for _, row in img_edges.iterrows():
            degree_counts[row["cell_i"]] += 1
            degree_counts[row["cell_j"]] += 1
            if "aj_occupancy" in row:
                occupancy_sums[row["cell_i"]].append(row["aj_occupancy"])
                occupancy_sums[row["cell_j"]].append(row["aj_occupancy"])

        # Compute mean occupancy per cell
        degrees = []
        mean_occupancies = []
        for cell_id in degree_counts:
            if cell_id in occupancy_sums and len(occupancy_sums[cell_id]) > 0:
                degrees.append(degree_counts[cell_id])
                mean_occupancies.append(np.mean(occupancy_sums[cell_id]))

        if len(degrees) >= 5:  # Need minimum samples for correlation
            r, p = stats.spearmanr(degrees, mean_occupancies)
            results.append({
                "image_id": image_id,
                "condition": condition,
                "spearman_r": r,
                "p_value": p,
                "n_cells": len(degrees)
            })

    return pd.DataFrame(results)


def compute_per_image_triangle_stats(cells_df, edges_df):
    """Compute all-reticular triangle proportion per image."""
    results = []

    for image_id in cells_df["image_id"].unique():
        img_cells = cells_df[cells_df["image_id"] == image_id]
        img_edges = edges_df[edges_df["image_id"] == image_id]
        condition = img_cells["condition"].iloc[0]

        if "aj_morph" not in img_edges.columns:
            continue

        # Build graph
        G = nx.Graph()
        G.add_nodes_from(img_cells["cell_id"].values)

        edge_morph = {}
        for _, row in img_edges.iterrows():
            edge = tuple(sorted([row["cell_i"], row["cell_j"]]))
            G.add_edge(row["cell_i"], row["cell_j"])
            edge_morph[edge] = row["aj_morph"]

        # Find triangles
        triangles = [c for c in nx.enumerate_all_cliques(G) if len(c) == 3]

        if len(triangles) == 0:
            continue

        # Count all-reticular triangles
        all_reticular = 0
        for tri in triangles:
            edges_in_tri = [
                tuple(sorted([tri[0], tri[1]])),
                tuple(sorted([tri[1], tri[2]])),
                tuple(sorted([tri[0], tri[2]]))
            ]
            morphs = [edge_morph.get(e, "unknown") for e in edges_in_tri]
            if all(m == "reticular" for m in morphs):
                all_reticular += 1

        pct = all_reticular / len(triangles) * 100

        results.append({
            "image_id": image_id,
            "condition": condition,
            "all_reticular_triangle_pct": pct,
            "n_all_reticular": all_reticular,
            "n_triangles": len(triangles)
        })

    return pd.DataFrame(results)


def compute_per_image_area_degree_corr(cells_df, edges_df):
    """Compute area-degree Spearman correlation within each image."""
    results = []

    for image_id in cells_df["image_id"].unique():
        img_cells = cells_df[cells_df["image_id"] == image_id]
        img_edges = edges_df[edges_df["image_id"] == image_id]
        condition = img_cells["condition"].iloc[0]

        if "area" not in img_cells.columns:
            continue

        # Compute degree for each cell
        degree_counts = defaultdict(int)
        for _, row in img_edges.iterrows():
            degree_counts[row["cell_i"]] += 1
            degree_counts[row["cell_j"]] += 1

        # Match with area
        areas = []
        degrees = []
        for _, cell in img_cells.iterrows():
            cell_id = cell["cell_id"]
            if cell_id in degree_counts:
                areas.append(cell["area"])
                degrees.append(degree_counts[cell_id])

        if len(areas) >= 5:
            r, p = stats.spearmanr(areas, degrees)
            results.append({
                "image_id": image_id,
                "condition": condition,
                "spearman_r": r,
                "p_value": p,
                "n_cells": len(areas)
            })

    return pd.DataFrame(results)


def test_condition_difference(df, metric_col, condition_a="static", condition_b="6dyne"):
    """Mann-Whitney U test between conditions on per-image metric."""
    a = df[df["condition"] == condition_a][metric_col].dropna()
    b = df[df["condition"] == condition_b][metric_col].dropna()

    if len(a) < 2 or len(b) < 2:
        return None

    stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")

    # Effect size (rank-biserial correlation)
    n1, n2 = len(a), len(b)
    effect_size = 1 - (2 * stat) / (n1 * n2)

    # Bootstrap 95% CI for median difference
    np.random.seed(42)
    boot_diffs = []
    for _ in range(1000):
        a_boot = np.random.choice(a, size=len(a), replace=True)
        b_boot = np.random.choice(b, size=len(b), replace=True)
        boot_diffs.append(np.median(b_boot) - np.median(a_boot))
    ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])

    return {
        "condition_a": condition_a,
        "condition_b": condition_b,
        "n_a": len(a),
        "n_b": len(b),
        "median_a": float(np.median(a)),
        "median_b": float(np.median(b)),
        "median_diff": float(np.median(b) - np.median(a)),
        "ci_95_low": float(ci_low),
        "ci_95_high": float(ci_high),
        "mann_whitney_U": float(stat),
        "p_value": float(p),
        "effect_size_r": float(effect_size)
    }


def test_correlation_differs_from_zero(df, r_col="spearman_r"):
    """Wilcoxon signed-rank test that median correlation differs from 0."""
    rs = df[r_col].dropna()

    if len(rs) < 3:
        return None

    stat, p = stats.wilcoxon(rs)

    return {
        "n_images": len(rs),
        "median_r": float(np.median(rs)),
        "mean_r": float(np.mean(rs)),
        "std_r": float(np.std(rs)),
        "min_r": float(np.min(rs)),
        "max_r": float(np.max(rs)),
        "wilcoxon_stat": float(stat),
        "p_value": float(p)
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("runs_dir", help="Path to runs directory (e.g., runs/egm2_full)")
    parser.add_argument("--output", "-o", help="Output JSON file", default=None)
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)

    print("="*70)
    print("HARDENED NETWORK STATISTICS (Per-Image Replicate Testing)")
    print("="*70)

    print("\nLoading data...")
    cells, edges = load_all_data(runs_dir)

    n_images = cells["image_id"].nunique()
    print(f"Loaded {len(cells)} cells, {len(edges)} edges from {n_images} images")
    print(f"Conditions: {cells.groupby('condition')['image_id'].nunique().to_dict()}")

    results = {"meta": {"n_images": n_images, "n_cells": len(cells), "n_edges": len(edges)}}

    # Discovery 1: Clustering coefficient
    print("\n" + "-"*70)
    print("DISCOVERY 1: Clustering Coefficient (per-image)")
    print("-"*70)

    clustering_df = compute_per_image_clustering(cells, edges)
    print(f"\nPer-image clustering coefficients:")
    for cond in ["static", "6dyne"]:
        subset = clustering_df[clustering_df["condition"] == cond]["mean_clustering"]
        if len(subset) > 0:
            print(f"  {cond}: median={np.median(subset):.3f}, mean={np.mean(subset):.3f}, "
                  f"std={np.std(subset):.3f}, n={len(subset)}")

    test1 = test_condition_difference(clustering_df, "mean_clustering")
    if test1:
        print(f"\nMann-Whitney U test (static vs 6dyne):")
        print(f"  Median diff: {test1['median_diff']:.3f} [95% CI: {test1['ci_95_low']:.3f}, {test1['ci_95_high']:.3f}]")
        print(f"  U = {test1['mann_whitney_U']:.1f}, p = {test1['p_value']:.2e}")
        print(f"  Effect size r = {test1['effect_size_r']:.3f}")
        results["discovery_1_clustering"] = {
            "per_image_stats": clustering_df.to_dict(orient="records"),
            "test": test1
        }

    # Discovery 2: Reticular junction percentage
    print("\n" + "-"*70)
    print("DISCOVERY 2: Reticular Junction % (per-image)")
    print("-"*70)

    reticular_df = compute_per_image_reticular_pct(edges)
    print(f"\nPer-image reticular %:")
    for cond in ["static", "6dyne"]:
        subset = reticular_df[reticular_df["condition"] == cond]["reticular_pct"]
        if len(subset) > 0:
            print(f"  {cond}: median={np.median(subset):.1f}%, mean={np.mean(subset):.1f}%, "
                  f"std={np.std(subset):.1f}%, n={len(subset)}")

    test2 = test_condition_difference(reticular_df, "reticular_pct")
    if test2:
        print(f"\nMann-Whitney U test (static vs 6dyne):")
        print(f"  Median diff: {test2['median_diff']:.1f}% [95% CI: {test2['ci_95_low']:.1f}%, {test2['ci_95_high']:.1f}%]")
        print(f"  U = {test2['mann_whitney_U']:.1f}, p = {test2['p_value']:.2e}")
        print(f"  Effect size r = {test2['effect_size_r']:.3f}")
        results["discovery_2_reticular"] = {
            "per_image_stats": reticular_df.to_dict(orient="records"),
            "test": test2
        }

    # Discovery 3: Degree-occupancy correlation
    print("\n" + "-"*70)
    print("DISCOVERY 3: Degree-Occupancy Correlation (per-image)")
    print("-"*70)

    deg_occ_df = compute_per_image_degree_occupancy_corr(cells, edges)
    print(f"\nPer-image Spearman r (degree vs occupancy):")
    for cond in ["static", "6dyne"]:
        subset = deg_occ_df[deg_occ_df["condition"] == cond]["spearman_r"]
        if len(subset) > 0:
            print(f"  {cond}: median r={np.median(subset):.3f}, mean r={np.mean(subset):.3f}, "
                  f"range=[{np.min(subset):.3f}, {np.max(subset):.3f}], n={len(subset)}")

    test3 = test_correlation_differs_from_zero(deg_occ_df)
    if test3:
        print(f"\nWilcoxon test (median r differs from 0):")
        print(f"  Median r = {test3['median_r']:.3f}, p = {test3['p_value']:.2e}")
        results["discovery_3_degree_occupancy"] = {
            "per_image_stats": deg_occ_df.to_dict(orient="records"),
            "test": test3
        }

    # Discovery 4: All-reticular triangles
    print("\n" + "-"*70)
    print("DISCOVERY 4: All-Reticular Triangle % (per-image)")
    print("-"*70)

    triangle_df = compute_per_image_triangle_stats(cells, edges)
    print(f"\nPer-image all-reticular triangle %:")
    for cond in ["static", "6dyne"]:
        subset = triangle_df[triangle_df["condition"] == cond]["all_reticular_triangle_pct"]
        if len(subset) > 0:
            print(f"  {cond}: median={np.median(subset):.1f}%, mean={np.mean(subset):.1f}%, "
                  f"std={np.std(subset):.1f}%, n={len(subset)}")

    test4 = test_condition_difference(triangle_df, "all_reticular_triangle_pct")
    if test4:
        print(f"\nMann-Whitney U test (static vs 6dyne):")
        print(f"  Median diff: {test4['median_diff']:.1f}% [95% CI: {test4['ci_95_low']:.1f}%, {test4['ci_95_high']:.1f}%]")
        print(f"  U = {test4['mann_whitney_U']:.1f}, p = {test4['p_value']:.2e}")
        print(f"  Effect size r = {test4['effect_size_r']:.3f}")
        results["discovery_4_triangles"] = {
            "per_image_stats": triangle_df.to_dict(orient="records"),
            "test": test4
        }

    # Discovery 5: Area-degree correlation
    print("\n" + "-"*70)
    print("DISCOVERY 5: Area-Degree Correlation (per-image)")
    print("-"*70)

    area_deg_df = compute_per_image_area_degree_corr(cells, edges)
    print(f"\nPer-image Spearman r (area vs degree):")
    for cond in ["static", "6dyne"]:
        subset = area_deg_df[area_deg_df["condition"] == cond]["spearman_r"]
        if len(subset) > 0:
            print(f"  {cond}: median r={np.median(subset):.3f}, mean r={np.mean(subset):.3f}, "
                  f"range=[{np.min(subset):.3f}, {np.max(subset):.3f}], n={len(subset)}")

    test5_static = test_correlation_differs_from_zero(area_deg_df[area_deg_df["condition"] == "static"])
    test5_flow = test_correlation_differs_from_zero(area_deg_df[area_deg_df["condition"] == "6dyne"])
    test5_compare = test_condition_difference(area_deg_df, "spearman_r")

    if test5_static:
        print(f"\nStatic: median r = {test5_static['median_r']:.3f}, differs from 0: p = {test5_static['p_value']:.2e}")
    if test5_flow:
        print(f"6dyne: median r = {test5_flow['median_r']:.3f}, differs from 0: p = {test5_flow['p_value']:.2e}")
    if test5_compare:
        print(f"\nCondition comparison:")
        print(f"  Median diff: {test5_compare['median_diff']:.3f} [95% CI: {test5_compare['ci_95_low']:.3f}, {test5_compare['ci_95_high']:.3f}]")
        print(f"  U = {test5_compare['mann_whitney_U']:.1f}, p = {test5_compare['p_value']:.2e}")

    results["discovery_5_area_degree"] = {
        "per_image_stats": area_deg_df.to_dict(orient="records"),
        "test_static": test5_static,
        "test_flow": test5_flow,
        "test_compare": test5_compare
    }

    # Save results
    output_file = args.output or (runs_dir / "hardened_network_stats.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_file}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Per-Image Replicate Statistics")
    print("="*70)
    print("""
All statistics now use IMAGE as the sampling unit (not individual cells/edges).
This avoids pseudo-replication and provides robust p-values.

Effect sizes reported as rank-biserial correlation r:
  |r| < 0.1: negligible
  |r| 0.1-0.3: small
  |r| 0.3-0.5: medium
  |r| > 0.5: large

95% CIs computed via bootstrap (1000 resamples).
""")


if __name__ == "__main__":
    main()
