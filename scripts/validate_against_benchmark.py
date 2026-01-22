#!/usr/bin/env python3
"""
Validate EndoPiGraph against the gold-standard benchmark.

This script:
1. Loads benchmark images and ground truth masks/edges
2. Runs EndoPiGraph segmentation and interface extraction
3. Compares predicted edges to ground truth
4. Reports precision, recall, F1 for edge detection
5. If manual junction annotations exist, evaluates classification accuracy
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import numpy as np
import pandas as pd
from pathlib import Path
import tifffile
from typing import Dict, List, Tuple

from endopigraph.segmentation import segment_cells
from endopigraph.interfaces import extract_interfaces
from endopigraph.ajmorph import interface_marker_features, compute_threshold, heuristic_ajmorph_class


def load_benchmark(benchmark_dir: Path) -> pd.DataFrame:
    """Load benchmark manifest and annotations."""
    manifest = pd.read_csv(benchmark_dir / 'manifest.csv')
    return manifest


def edge_set(edges: List[Dict]) -> set:
    """Convert edge list to set of tuples for comparison."""
    return set((e['cell_i'], e['cell_j']) for e in edges)


def edge_set_from_df(df: pd.DataFrame) -> set:
    """Convert edges DataFrame to set."""
    return set((row['cell_i'], row['cell_j']) for _, row in df.iterrows())


def compute_edge_metrics(gt_edges: set, pred_edges: set) -> Dict:
    """Compute precision, recall, F1 for edge detection."""
    if len(pred_edges) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': 0, 'fn': len(gt_edges)}

    tp = len(gt_edges & pred_edges)
    fp = len(pred_edges - gt_edges)
    fn = len(gt_edges - pred_edges)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
    }


def validate_segmentation(
    benchmark_dir: Path,
    use_gt_masks: bool = True,
    max_images: int = None,
) -> pd.DataFrame:
    """
    Validate EndoPiGraph edge detection against benchmark.

    Parameters
    ----------
    benchmark_dir : Path
        Path to benchmark directory
    use_gt_masks : bool
        If True, use benchmark masks for interface extraction (tests only interface detection).
        If False, run full segmentation (tests segmentation + interface detection).
    max_images : int
        Maximum number of images to process (for quick testing)
    """
    manifest = load_benchmark(benchmark_dir)

    if max_images:
        manifest = manifest.head(max_images)

    results = []

    for _, row in manifest.iterrows():
        benchmark_id = row['benchmark_id']
        print(f"Validating {benchmark_id}...")

        # Load ground truth
        ann_path = benchmark_dir / 'annotations' / f"{benchmark_id}.json"
        with open(ann_path) as f:
            gt_ann = json.load(f)

        gt_edges = edge_set(gt_ann['edges'])

        # Load image
        img_path = benchmark_dir / 'images' / f"{benchmark_id}.tif"
        image = tifffile.imread(img_path)

        if use_gt_masks:
            # Use ground truth masks
            mask_path = benchmark_dir / 'masks' / f"{benchmark_id}_mask.tif"
            masks = tifffile.imread(mask_path)
        else:
            # Run EndoPiGraph segmentation
            if image.ndim == 2:
                arr = image[np.newaxis, ...]
            else:
                arr = image
            channel_names = ['VE-cadherin']

            seg_config = {
                'method': 'watershed',
                'watershed': {'min_cell_size': 500}
            }
            masks = segment_cells(arr, channel_names, seg_config)

        # Extract interfaces
        iface = extract_interfaces(masks)
        pred_edges = edge_set_from_df(iface.edges)

        # Compute metrics
        metrics = compute_edge_metrics(gt_edges, pred_edges)

        results.append({
            'benchmark_id': benchmark_id,
            'condition': row.get('condition', 'unknown'),
            'gt_cells': gt_ann['n_cells'],
            'pred_cells': int(masks.max()),
            'gt_edges': len(gt_edges),
            'pred_edges': len(pred_edges),
            **metrics,
        })

        print(f"  GT edges: {len(gt_edges)}, Pred edges: {len(pred_edges)}")
        print(f"  P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")

    return pd.DataFrame(results)


def validate_junction_classification(
    benchmark_dir: Path,
    annotations_file: str = 'annotations_manual.csv',
    max_images: int = None,
) -> pd.DataFrame:
    """
    Validate junction type classification against manual annotations.

    Requires a CSV file with manual junction type annotations.
    """
    ann_path = benchmark_dir / annotations_file
    if not ann_path.exists():
        print(f"Manual annotations not found: {ann_path}")
        print("Use annotation_template.csv as starting point for manual annotation.")
        return pd.DataFrame()

    manual_ann = pd.read_csv(ann_path)
    manual_ann = manual_ann[manual_ann['junction_type'].notna()]

    if len(manual_ann) == 0:
        print("No manual annotations found in file.")
        return pd.DataFrame()

    # Group by image
    results = []

    for benchmark_id, group in manual_ann.groupby('benchmark_id'):
        print(f"Validating classification for {benchmark_id}...")

        # Load image and mask
        img_path = benchmark_dir / 'images' / f"{benchmark_id}.tif"
        mask_path = benchmark_dir / 'masks' / f"{benchmark_id}_mask.tif"

        image = tifffile.imread(img_path)
        masks = tifffile.imread(mask_path)

        # Extract interfaces
        iface = extract_interfaces(masks)

        # Compute features and predict
        marker = image.astype(np.float32)
        boundary_vals = marker[iface.all_boundary_mask]
        thresh = compute_threshold(boundary_vals, 'otsu')

        for _, ann_row in group.iterrows():
            cell_i, cell_j = ann_row['cell_i'], ann_row['cell_j']
            gt_type = ann_row['junction_type']

            # Find edge in interface data
            edge_match = iface.edges[
                ((iface.edges['cell_i'] == cell_i) & (iface.edges['cell_j'] == cell_j)) |
                ((iface.edges['cell_i'] == cell_j) & (iface.edges['cell_j'] == cell_i))
            ]

            if len(edge_match) == 0:
                continue

            # Get interface mask
            key = (min(cell_i, cell_j), max(cell_i, cell_j))
            coords = iface.boundary_coords.get(key)

            if coords is None or len(coords) == 0:
                continue

            # Create interface mask
            from endopigraph.interfaces import interface_mask_from_coords
            int_mask = interface_mask_from_coords(coords, masks.shape, dilate_px=2)

            # Compute features
            features = interface_marker_features(marker, int_mask, thresh)

            # Predict class
            pred_type = heuristic_ajmorph_class(features)

            results.append({
                'benchmark_id': benchmark_id,
                'cell_i': cell_i,
                'cell_j': cell_j,
                'gt_type': gt_type,
                'pred_type': pred_type,
                'correct': gt_type.lower() == pred_type.lower(),
            })

    results_df = pd.DataFrame(results)

    if len(results_df) > 0:
        accuracy = results_df['correct'].mean()
        print(f"\nClassification accuracy: {accuracy:.3f} ({results_df['correct'].sum()}/{len(results_df)})")

        # Confusion matrix
        print("\nConfusion matrix:")
        confusion = pd.crosstab(results_df['gt_type'], results_df['pred_type'])
        print(confusion)

    return results_df


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Validate EndoPiGraph against benchmark')
    parser.add_argument('--benchmark', type=str, default='benchmark', help='Benchmark directory')
    parser.add_argument('--max-images', type=int, default=None, help='Max images to process')
    parser.add_argument('--full-pipeline', action='store_true',
                        help='Run full segmentation (not just interface extraction)')
    parser.add_argument('--classification', action='store_true',
                        help='Also validate junction classification')

    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark)

    # Validate segmentation/edge detection
    print("=" * 60)
    print("EDGE DETECTION VALIDATION")
    print("=" * 60)

    results = validate_segmentation(
        benchmark_dir,
        use_gt_masks=not args.full_pipeline,
        max_images=args.max_images,
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\nImages validated: {len(results)}")
    print(f"Mean precision: {results['precision'].mean():.3f}")
    print(f"Mean recall: {results['recall'].mean():.3f}")
    print(f"Mean F1: {results['f1'].mean():.3f}")

    # By condition
    if 'condition' in results.columns:
        print("\nBy condition:")
        for cond, group in results.groupby('condition'):
            print(f"  {cond}: P={group['precision'].mean():.3f}, R={group['recall'].mean():.3f}, F1={group['f1'].mean():.3f}")

    # Save results
    results.to_csv(benchmark_dir / 'validation_results.csv', index=False)
    print(f"\nResults saved to: {benchmark_dir / 'validation_results.csv'}")

    # Optional: classification validation
    if args.classification:
        print("\n" + "=" * 60)
        print("JUNCTION CLASSIFICATION VALIDATION")
        print("=" * 60)
        class_results = validate_junction_classification(benchmark_dir, max_images=args.max_images)
        if len(class_results) > 0:
            class_results.to_csv(benchmark_dir / 'classification_results.csv', index=False)


if __name__ == '__main__':
    main()
