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
        # Note: "18-20dyn" means high shear (18-20 dyne/cm²)
        if "static" in image_id.lower():
            condition = "static"
        elif "6dyn" in image_id.lower() or "6dyne" in image_id.lower():
            condition = "6dyne"
        elif "18-20dyn" in image_id.lower() or "20dyn" in image_id.lower() or "high_shear" in image_id.lower():
            condition = "high_shear"
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
    """Compute mean clustering coefficient per image, with density metrics."""
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

        # Compute mean degree (for density control)
        degrees = [d for n, d in G.degree()]
        mean_degree = np.mean(degrees) if degrees else 0

        # Compute random baseline clustering (Erdos-Renyi expectation)
        n_nodes = G.number_of_nodes()
        n_edges_actual = G.number_of_edges()
        if n_nodes > 1:
            p = 2 * n_edges_actual / (n_nodes * (n_nodes - 1))  # edge probability
            c_random = p  # Expected clustering for random graph
        else:
            c_random = 0

        # Normalized clustering
        c_normalized = mean_clustering / c_random if c_random > 0 else 0

        results.append({
            "image_id": image_id,
            "condition": condition,
            "mean_clustering": mean_clustering,
            "mean_degree": mean_degree,
            "c_random": c_random,
            "c_normalized": c_normalized,
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


def test_condition_difference(df, metric_col, condition_a="static", condition_b="6dyne", condition_c=None):
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
    for cond in ["static", "6dyne", "high_shear"]:
        subset = clustering_df[clustering_df["condition"] == cond]["mean_clustering"]
        if len(subset) > 0:
            print(f"  {cond}: median={np.median(subset):.3f}, mean={np.mean(subset):.3f}, "
                  f"std={np.std(subset):.3f}, n={len(subset)}")

    # Test all pairwise comparisons
    test1_pairs = {}
    for cond_a, cond_b in [("static", "6dyne"), ("static", "high_shear"), ("6dyne", "high_shear")]:
        test = test_condition_difference(clustering_df, "mean_clustering", cond_a, cond_b)
        if test:
            print(f"\nMann-Whitney U test ({cond_a} vs {cond_b}):")
            print(f"  Median diff: {test['median_diff']:.3f} [95% CI: {test['ci_95_low']:.3f}, {test['ci_95_high']:.3f}]")
            print(f"  U = {test['mann_whitney_U']:.1f}, p = {test['p_value']:.2e}")
            print(f"  Effect size r = {test['effect_size_r']:.3f}")
            test1_pairs[f"{cond_a}_vs_{cond_b}"] = test

    # Density control: regression clustering ~ condition + mean_degree + n_cells
    print("\n** Density Control: Regression Analysis **")
    print("  Model: clustering ~ condition + mean_degree + n_cells")
    try:
        from scipy.stats import pearsonr
        import statsmodels.api as sm
        from statsmodels.formula.api import ols

        # Create condition dummies
        clustering_df["is_6dyne"] = (clustering_df["condition"] == "6dyne").astype(int)
        clustering_df["is_high_shear"] = (clustering_df["condition"] == "high_shear").astype(int)

        # Fit model
        model = ols("mean_clustering ~ is_6dyne + is_high_shear + mean_degree + n_cells",
                   data=clustering_df).fit()
        print(f"\n  Regression results (controlling for density):")
        print(f"    is_6dyne coef: {model.params['is_6dyne']:.4f}, p = {model.pvalues['is_6dyne']:.2e}")
        print(f"    is_high_shear coef: {model.params['is_high_shear']:.4f}, p = {model.pvalues['is_high_shear']:.2e}")
        print(f"    mean_degree coef: {model.params['mean_degree']:.4f}, p = {model.pvalues['mean_degree']:.2e}")
        print(f"    R-squared: {model.rsquared:.3f}")

        density_control = {
            "is_6dyne_coef": float(model.params['is_6dyne']),
            "is_6dyne_p": float(model.pvalues['is_6dyne']),
            "is_high_shear_coef": float(model.params['is_high_shear']),
            "is_high_shear_p": float(model.pvalues['is_high_shear']),
            "mean_degree_coef": float(model.params['mean_degree']),
            "mean_degree_p": float(model.pvalues['mean_degree']),
            "r_squared": float(model.rsquared)
        }

        # Also test normalized clustering (C / C_random)
        print(f"\n  Normalized clustering (C / C_random):")
        for cond in ["static", "6dyne", "high_shear"]:
            subset = clustering_df[clustering_df["condition"] == cond]["c_normalized"]
            if len(subset) > 0:
                print(f"    {cond}: median={np.median(subset):.2f}, mean={np.mean(subset):.2f}")

        test_norm = test_condition_difference(clustering_df, "c_normalized", "static", "6dyne")
        if test_norm:
            print(f"  Normalized clustering (static vs 6dyne): p = {test_norm['p_value']:.2e}")
            density_control["normalized_test_static_vs_6dyne"] = test_norm

    except ImportError:
        print("  (statsmodels not available - skipping regression)")
        density_control = None

    results["discovery_1_clustering"] = {
        "per_image_stats": clustering_df.to_dict(orient="records"),
        "tests": test1_pairs,
        "density_control": density_control
    }

    # Discovery 2: Reticular junction percentage
    print("\n" + "-"*70)
    print("DISCOVERY 2: Reticular Junction % (per-image)")
    print("-"*70)

    reticular_df = compute_per_image_reticular_pct(edges)
    print(f"\nPer-image reticular %:")
    for cond in ["static", "6dyne", "high_shear"]:
        subset = reticular_df[reticular_df["condition"] == cond]["reticular_pct"]
        if len(subset) > 0:
            print(f"  {cond}: median={np.median(subset):.1f}%, mean={np.mean(subset):.1f}%, "
                  f"std={np.std(subset):.1f}%, n={len(subset)}")

    test2_pairs = {}
    for cond_a, cond_b in [("static", "6dyne"), ("static", "high_shear"), ("6dyne", "high_shear")]:
        test = test_condition_difference(reticular_df, "reticular_pct", cond_a, cond_b)
        if test:
            print(f"\nMann-Whitney U test ({cond_a} vs {cond_b}):")
            print(f"  Median diff: {test['median_diff']:.1f}% [95% CI: {test['ci_95_low']:.1f}%, {test['ci_95_high']:.1f}%]")
            print(f"  U = {test['mann_whitney_U']:.1f}, p = {test['p_value']:.2e}")
            print(f"  Effect size r = {test['effect_size_r']:.3f}")
            test2_pairs[f"{cond_a}_vs_{cond_b}"] = test

    results["discovery_2_reticular"] = {
        "per_image_stats": reticular_df.to_dict(orient="records"),
        "tests": test2_pairs
    }

    # Discovery 3: Degree-occupancy correlation
    print("\n" + "-"*70)
    print("DISCOVERY 3: Degree-Occupancy Correlation (per-image)")
    print("-"*70)

    deg_occ_df = compute_per_image_degree_occupancy_corr(cells, edges)
    print(f"\nPer-image Spearman r (degree vs occupancy):")
    for cond in ["static", "6dyne", "high_shear"]:
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
    for cond in ["static", "6dyne", "high_shear"]:
        subset = triangle_df[triangle_df["condition"] == cond]["all_reticular_triangle_pct"]
        if len(subset) > 0:
            print(f"  {cond}: median={np.median(subset):.1f}%, mean={np.mean(subset):.1f}%, "
                  f"std={np.std(subset):.1f}%, n={len(subset)}")

    test4_pairs = {}
    for cond_a, cond_b in [("static", "6dyne"), ("static", "high_shear"), ("6dyne", "high_shear")]:
        test = test_condition_difference(triangle_df, "all_reticular_triangle_pct", cond_a, cond_b)
        if test:
            print(f"\nMann-Whitney U test ({cond_a} vs {cond_b}):")
            print(f"  Median diff: {test['median_diff']:.1f}% [95% CI: {test['ci_95_low']:.1f}%, {test['ci_95_high']:.1f}%]")
            print(f"  U = {test['mann_whitney_U']:.1f}, p = {test['p_value']:.2e}")
            print(f"  Effect size r = {test['effect_size_r']:.3f}")
            test4_pairs[f"{cond_a}_vs_{cond_b}"] = test

    results["discovery_4_triangles"] = {
        "per_image_stats": triangle_df.to_dict(orient="records"),
        "tests": test4_pairs
    }

    # Discovery 5: Area-degree correlation
    print("\n" + "-"*70)
    print("DISCOVERY 5: Area-Degree Correlation (per-image)")
    print("-"*70)

    area_deg_df = compute_per_image_area_degree_corr(cells, edges)
    print(f"\nPer-image Spearman r (area vs degree):")
    for cond in ["static", "6dyne", "high_shear"]:
        subset = area_deg_df[area_deg_df["condition"] == cond]["spearman_r"]
        if len(subset) > 0:
            print(f"  {cond}: median r={np.median(subset):.3f}, mean r={np.mean(subset):.3f}, "
                  f"range=[{np.min(subset):.3f}, {np.max(subset):.3f}], n={len(subset)}")

    # Test each condition differs from 0
    test5_by_cond = {}
    for cond in ["static", "6dyne", "high_shear"]:
        test = test_correlation_differs_from_zero(area_deg_df[area_deg_df["condition"] == cond])
        if test:
            print(f"\n{cond}: median r = {test['median_r']:.3f}, differs from 0: p = {test['p_value']:.2e}")
            test5_by_cond[cond] = test

    # Pairwise comparisons
    test5_pairs = {}
    print(f"\nPairwise condition comparisons:")
    for cond_a, cond_b in [("static", "6dyne"), ("static", "high_shear"), ("6dyne", "high_shear")]:
        test = test_condition_difference(area_deg_df, "spearman_r", cond_a, cond_b)
        if test:
            print(f"  {cond_a} vs {cond_b}: Median diff = {test['median_diff']:.3f}, p = {test['p_value']:.2e}")
            test5_pairs[f"{cond_a}_vs_{cond_b}"] = test

    results["discovery_5_area_degree"] = {
        "per_image_stats": area_deg_df.to_dict(orient="records"),
        "tests_differ_from_zero": test5_by_cond,
        "tests_pairwise": test5_pairs
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
