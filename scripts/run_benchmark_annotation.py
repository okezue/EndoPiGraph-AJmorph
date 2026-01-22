#!/usr/bin/env python3
"""
Run automatic junction annotation on the benchmark using three methods:

1. EndoPiGraph (novel) - Full feature set with heuristic classifier
2. Simple Baseline - Intensity/occupancy only classifier
3. Junction Mapper Equivalent - Emulates JM's metrics with rule-based classification

This script processes all benchmark images and generates predictions from each method,
then compares their agreement and characteristics.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import tifffile
from collections import Counter

from endopigraph.interfaces import extract_interfaces, interface_mask_from_coords
from endopigraph.ajmorph import (
    interface_marker_features,
    compute_threshold,
    heuristic_ajmorph_class,
)


# =============================================================================
# Classifier Definitions
# =============================================================================

def baseline_classifier(features: Dict) -> str:
    """
    Simple baseline classifier using ONLY intensity and occupancy.

    This represents a minimal approach that doesn't use cluster counting
    or skeleton analysis - just basic intensity metrics.

    Rules:
    - High occupancy (>0.6) + high intensity → "continuous"
    - Low occupancy (<0.2) → "discontinuous"
    - Medium occupancy + high intensity variation → "reticular"
    - Otherwise → "intermediate"
    """
    occ = features.get('occupancy', 0)
    mean_int = features.get('mean', 0)
    std_int = features.get('std', 0)
    max_int = features.get('max', 0)

    if not np.isfinite(occ):
        return 'unknown'

    # Coefficient of variation for intensity
    cv = std_int / mean_int if mean_int > 0 else 0

    # Simple rules based only on intensity/occupancy
    if occ > 0.6:
        if cv < 0.3:
            return 'straight'  # High occupancy, uniform intensity
        else:
            return 'thick'  # High occupancy, variable intensity
    elif occ < 0.2:
        return 'fingers'  # Very low occupancy
    elif occ < 0.4:
        return 'discontinuous'  # Low-medium occupancy
    else:
        if cv > 0.4:
            return 'reticular'  # Medium occupancy, high variation
        else:
            return 'thick_to_reticular'  # Medium occupancy, moderate variation


def junction_mapper_classifier(features: Dict) -> str:
    """
    Junction Mapper equivalent classifier.

    Junction Mapper (Tomlinson et al. eLife 2019) provides metrics but
    does NOT have automatic classification - users manually categorize.

    This emulates what a simple rule-based classifier using JM's metrics would do:
    - Fraction occupied (occupancy)
    - Number of clusters (cluster_count)
    - Mean cluster size
    - Mean intensity

    Note: JM doesn't provide cluster_density or skeleton metrics.
    """
    occ = features.get('occupancy', 0)
    n_clusters = features.get('cluster_count_cc', 0)  # JM uses connected components
    cluster_area = features.get('cluster_area_mean', 0)
    mean_int = features.get('mean', 0)

    if not np.isfinite(occ):
        return 'unknown'

    # Rules based on JM-available metrics
    if occ > 0.7 and n_clusters <= 2:
        return 'straight'  # High occupancy, few clusters = continuous

    if occ > 0.5 and n_clusters <= 4:
        return 'thick'  # High occupancy, some clustering

    if n_clusters >= 8:
        return 'reticular'  # Many clusters = fragmented/reticular

    if occ < 0.2:
        return 'fingers'  # Very sparse

    if occ < 0.4 and n_clusters >= 3:
        return 'discontinuous'  # Low occupancy with multiple clusters

    return 'thick_to_reticular'  # Default intermediate


def endopigraph_classifier(features: Dict, blur_robust: bool = False) -> str:
    """
    EndoPiGraph full classifier using all available features.

    This uses the heuristic_ajmorph_class function which considers:
    - occupancy
    - cluster_count (or skeleton metrics if blur_robust=True)
    - skeleton_len
    - thickness_proxy
    """
    return heuristic_ajmorph_class(features, blur_robust=blur_robust)


# =============================================================================
# Main Processing
# =============================================================================

def process_benchmark_image(
    benchmark_id: str,
    benchmark_dir: Path,
) -> List[Dict]:
    """Process a single benchmark image and classify all edges."""

    # Load image and mask
    img_path = benchmark_dir / 'images' / f"{benchmark_id}.tif"
    mask_path = benchmark_dir / 'masks' / f"{benchmark_id}_mask.tif"
    ann_path = benchmark_dir / 'annotations' / f"{benchmark_id}.json"

    image = tifffile.imread(img_path).astype(np.float32)
    masks = tifffile.imread(mask_path)

    with open(ann_path) as f:
        ann = json.load(f)

    # Extract interfaces
    iface = extract_interfaces(masks)

    # Compute global threshold
    boundary_vals = image[iface.all_boundary_mask]
    thresh = compute_threshold(boundary_vals, 'otsu')

    results = []

    for edge in ann['edges']:
        cell_i, cell_j = edge['cell_i'], edge['cell_j']

        # Get interface coordinates
        key = (min(cell_i, cell_j), max(cell_i, cell_j))
        coords = iface.boundary_coords.get(key)

        if coords is None or len(coords) == 0:
            # Edge not found in interface extraction
            results.append({
                'benchmark_id': benchmark_id,
                'cell_i': cell_i,
                'cell_j': cell_j,
                'contact_px': edge['contact_px'],
                'epg_class': 'unknown',
                'epg_robust_class': 'unknown',
                'baseline_class': 'unknown',
                'jm_class': 'unknown',
                'features': {},
            })
            continue

        # Create interface mask
        int_mask = interface_mask_from_coords(coords, masks.shape, dilate_px=2)

        # Compute features
        features = interface_marker_features(image, int_mask, thresh)

        # Run all classifiers
        epg_class = endopigraph_classifier(features, blur_robust=False)
        epg_robust_class = endopigraph_classifier(features, blur_robust=True)
        baseline_class = baseline_classifier(features)
        jm_class = junction_mapper_classifier(features)

        results.append({
            'benchmark_id': benchmark_id,
            'cell_i': cell_i,
            'cell_j': cell_j,
            'contact_px': edge['contact_px'],
            'epg_class': epg_class,
            'epg_robust_class': epg_robust_class,
            'baseline_class': baseline_class,
            'jm_class': jm_class,
            # Store key features for analysis
            'occupancy': features.get('occupancy', np.nan),
            'cluster_count': features.get('cluster_count_cc', 0),
            'skeleton_len': features.get('skeleton_len', 0),
            'mean_intensity': features.get('mean', np.nan),
            'complexity_score': features.get('complexity_score', 0),
        })

    return results


def run_full_benchmark_annotation(
    benchmark_dir: Path,
    max_images: int = None,
) -> pd.DataFrame:
    """Run all classifiers on the full benchmark."""

    manifest = pd.read_csv(benchmark_dir / 'manifest.csv')

    if max_images:
        manifest = manifest.head(max_images)

    all_results = []

    for i, row in manifest.iterrows():
        benchmark_id = row['benchmark_id']
        print(f"[{i+1}/{len(manifest)}] Processing {benchmark_id}...")

        try:
            results = process_benchmark_image(benchmark_id, benchmark_dir)
            all_results.extend(results)
            print(f"  Classified {len(results)} edges")
        except Exception as e:
            print(f"  Error: {e}")
            continue

    return pd.DataFrame(all_results)


def analyze_results(df: pd.DataFrame) -> Dict:
    """Analyze classification results across methods."""

    analysis = {}

    # Class distributions
    for method in ['epg_class', 'epg_robust_class', 'baseline_class', 'jm_class']:
        dist = df[method].value_counts(normalize=True).to_dict()
        analysis[f'{method}_distribution'] = dist

    # Agreement between methods
    methods = ['epg_class', 'baseline_class', 'jm_class']
    agreement_matrix = {}

    for m1 in methods:
        for m2 in methods:
            if m1 != m2:
                agreement = (df[m1] == df[m2]).mean()
                agreement_matrix[f'{m1}_vs_{m2}'] = agreement

    analysis['agreement'] = agreement_matrix

    # Full agreement (all 3 methods agree)
    full_agreement = ((df['epg_class'] == df['baseline_class']) &
                      (df['baseline_class'] == df['jm_class'])).mean()
    analysis['full_agreement'] = full_agreement

    return analysis


def print_results_summary(df: pd.DataFrame, analysis: Dict):
    """Print a formatted summary of results."""

    print("\n" + "=" * 70)
    print("BENCHMARK ANNOTATION RESULTS")
    print("=" * 70)

    print(f"\nTotal edges classified: {len(df)}")

    # Class distributions
    print("\n" + "-" * 70)
    print("CLASS DISTRIBUTIONS")
    print("-" * 70)

    methods = {
        'epg_class': 'EndoPiGraph (Full)',
        'epg_robust_class': 'EndoPiGraph (Blur-Robust)',
        'baseline_class': 'Baseline (Intensity/Occupancy)',
        'jm_class': 'Junction Mapper Equivalent',
    }

    for col, name in methods.items():
        print(f"\n{name}:")
        counts = df[col].value_counts()
        for cls, count in counts.items():
            pct = 100 * count / len(df)
            print(f"  {cls:<20} {count:>6} ({pct:>5.1f}%)")

    # Agreement analysis
    print("\n" + "-" * 70)
    print("METHOD AGREEMENT")
    print("-" * 70)

    print(f"\nPairwise agreement:")
    for key, val in analysis['agreement'].items():
        print(f"  {key}: {val:.1%}")

    print(f"\nFull agreement (all 3 methods): {analysis['full_agreement']:.1%}")

    # Feature statistics by class (EndoPiGraph classes)
    print("\n" + "-" * 70)
    print("FEATURE STATISTICS BY CLASS (EndoPiGraph)")
    print("-" * 70)

    feature_cols = ['occupancy', 'cluster_count', 'skeleton_len', 'complexity_score']

    print(f"\n{'Class':<20} {'Occupancy':>12} {'Clusters':>10} {'Skel Len':>10} {'Complexity':>12}")
    print("-" * 70)

    for cls in df['epg_class'].unique():
        if cls == 'unknown':
            continue
        subset = df[df['epg_class'] == cls]
        occ = subset['occupancy'].mean()
        clust = subset['cluster_count'].mean()
        skel = subset['skeleton_len'].mean()
        comp = subset['complexity_score'].mean()
        print(f"{cls:<20} {occ:>12.3f} {clust:>10.1f} {skel:>10.1f} {comp:>12.1f}")


def create_comparison_table(df: pd.DataFrame, output_dir: Path):
    """Create a detailed comparison table."""

    # Crosstab of EPG vs Baseline
    print("\n" + "-" * 70)
    print("CONFUSION: EndoPiGraph vs Baseline")
    print("-" * 70)
    ct1 = pd.crosstab(df['epg_class'], df['baseline_class'], margins=True)
    print(ct1)

    # Crosstab of EPG vs JM
    print("\n" + "-" * 70)
    print("CONFUSION: EndoPiGraph vs Junction Mapper")
    print("-" * 70)
    ct2 = pd.crosstab(df['epg_class'], df['jm_class'], margins=True)
    print(ct2)

    # Save crosstabs
    ct1.to_csv(output_dir / 'confusion_epg_vs_baseline.csv')
    ct2.to_csv(output_dir / 'confusion_epg_vs_jm.csv')


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Run benchmark annotation with multiple methods')
    parser.add_argument('--benchmark', type=str, default='benchmark', help='Benchmark directory')
    parser.add_argument('--max-images', type=int, default=None, help='Max images to process')
    parser.add_argument('--output', type=str, default=None, help='Output directory (default: benchmark)')

    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark)
    output_dir = Path(args.output) if args.output else benchmark_dir

    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("RUNNING MULTI-METHOD JUNCTION CLASSIFICATION")
    print("=" * 70)
    print(f"\nBenchmark: {benchmark_dir}")
    print(f"Output: {output_dir}")

    # Run classification
    df = run_full_benchmark_annotation(benchmark_dir, max_images=args.max_images)

    # Analyze results
    analysis = analyze_results(df)

    # Print summary
    print_results_summary(df, analysis)

    # Create comparison tables
    create_comparison_table(df, output_dir)

    # Save full results
    df.to_csv(output_dir / 'benchmark_annotations_all_methods.csv', index=False)

    # Save analysis JSON
    # Convert numpy types for JSON serialization
    analysis_json = {}
    for k, v in analysis.items():
        if isinstance(v, dict):
            analysis_json[k] = {str(kk): float(vv) if isinstance(vv, (np.floating, float)) else vv
                               for kk, vv in v.items()}
        else:
            analysis_json[k] = float(v) if isinstance(v, (np.floating, float)) else v

    with open(output_dir / 'benchmark_analysis.json', 'w') as f:
        json.dump(analysis_json, f, indent=2)

    print(f"\n{'=' * 70}")
    print("FILES SAVED")
    print("=" * 70)
    print(f"  {output_dir / 'benchmark_annotations_all_methods.csv'}")
    print(f"  {output_dir / 'benchmark_analysis.json'}")
    print(f"  {output_dir / 'confusion_epg_vs_baseline.csv'}")
    print(f"  {output_dir / 'confusion_epg_vs_jm.csv'}")

    return df, analysis


if __name__ == '__main__':
    main()
