#!/usr/bin/env python3
"""
Benchmark multiple adjacency extraction methods on NuInsSeg gold-standard dataset.

Compares:
1. EndoPiGraph - Full interface extraction with contact_px filtering
2. Simple Dilation Baseline - Basic 1px dilation overlap detection
3. Centroid Distance Baseline - Adjacent if centroids within threshold
4. Voronoi Baseline - Adjacent if sharing Voronoi boundary + mask contact
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set
import warnings
warnings.filterwarnings('ignore')

try:
    from skimage import io as skio
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

from scipy import ndimage
from scipy.spatial.distance import cdist
from scipy.spatial import Voronoi
from endopigraph.interfaces import extract_interfaces


def load_label_mask(mask_path: Path) -> np.ndarray:
    """Load instance segmentation label mask from TIFF file."""
    mask = skio.imread(str(mask_path))
    if mask.ndim == 3:
        labels = mask[:, :, 0].astype(np.int32)
    else:
        labels = mask.astype(np.int32)
    return labels


def compute_ground_truth_adjacency(labels: np.ndarray, min_contact_px: int = 3) -> Set[Tuple[int, int]]:
    """Compute ground-truth adjacency from instance labels."""
    adjacency = set()
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]

    for label in unique_labels:
        cell_mask = labels == label
        dilated = ndimage.binary_dilation(cell_mask)
        neighbor_region = labels[dilated & ~cell_mask]
        neighbor_labels = np.unique(neighbor_region)
        neighbor_labels = neighbor_labels[neighbor_labels > 0]

        for neighbor in neighbor_labels:
            if neighbor != label:
                neighbor_mask = labels == neighbor
                contact = np.sum(dilated & neighbor_mask)
                if contact >= min_contact_px:
                    edge = tuple(sorted([int(label), int(neighbor)]))
                    adjacency.add(edge)

    return adjacency


# =============================================================================
# METHOD 1: EndoPiGraph
# =============================================================================

def endopigraph_adjacency(labels: np.ndarray, min_contact_px: int = 3) -> Set[Tuple[int, int]]:
    """Extract adjacency using EndoPiGraph interface extraction."""
    iface = extract_interfaces(labels)
    edges = iface.edges
    edges = edges[edges['contact_px'] >= min_contact_px]

    adjacency = set()
    for _, row in edges.iterrows():
        edge = tuple(sorted([int(row['cell_i']), int(row['cell_j'])]))
        adjacency.add(edge)

    return adjacency


# =============================================================================
# METHOD 2: Simple Dilation Baseline
# =============================================================================

def simple_dilation_adjacency(labels: np.ndarray, min_contact_px: int = 3) -> Set[Tuple[int, int]]:
    """Simple baseline: 1px dilation overlap detection."""
    adjacency = set()
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]

    dilated_masks = {}
    for label in unique_labels:
        cell_mask = labels == label
        dilated_masks[label] = ndimage.binary_dilation(cell_mask, iterations=1)

    for i, label_i in enumerate(unique_labels):
        for label_j in unique_labels[i+1:]:
            overlap = dilated_masks[label_i] & dilated_masks[label_j]
            contact = np.sum(overlap)
            if contact >= min_contact_px:
                edge = tuple(sorted([int(label_i), int(label_j)]))
                adjacency.add(edge)

    return adjacency


# =============================================================================
# METHOD 3: Centroid Distance Baseline
# =============================================================================

def centroid_distance_adjacency(labels: np.ndarray, max_distance: float = None) -> Set[Tuple[int, int]]:
    """Centroid-based baseline: Cells are adjacent if centroids are close."""
    adjacency = set()
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]

    if len(unique_labels) < 2:
        return adjacency

    centroids = []
    areas = []
    label_list = []

    for label in unique_labels:
        cell_mask = labels == label
        area = np.sum(cell_mask)
        if area > 0:
            coords = np.where(cell_mask)
            centroid = (np.mean(coords[0]), np.mean(coords[1]))
            centroids.append(centroid)
            areas.append(area)
            label_list.append(label)

    if len(centroids) < 2:
        return adjacency

    centroids = np.array(centroids)

    if max_distance is None:
        mean_area = np.mean(areas)
        mean_diameter = 2 * np.sqrt(mean_area / np.pi)
        max_distance = mean_diameter * 1.5

    distances = cdist(centroids, centroids)

    for i in range(len(label_list)):
        for j in range(i + 1, len(label_list)):
            if distances[i, j] <= max_distance:
                edge = tuple(sorted([int(label_list[i]), int(label_list[j])]))
                adjacency.add(edge)

    return adjacency


# =============================================================================
# METHOD 4: Voronoi Baseline
# =============================================================================

def voronoi_adjacency(labels: np.ndarray, min_contact_px: int = 3) -> Set[Tuple[int, int]]:
    """Voronoi-based baseline: Cells are adjacent if they share a Voronoi boundary."""
    adjacency = set()
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]

    if len(unique_labels) < 2:
        return adjacency

    centroids = []
    label_list = []

    for label in unique_labels:
        cell_mask = labels == label
        if np.sum(cell_mask) > 0:
            coords = np.where(cell_mask)
            centroid = (np.mean(coords[1]), np.mean(coords[0]))
            centroids.append(centroid)
            label_list.append(label)

    if len(centroids) < 4:
        return centroid_distance_adjacency(labels)

    centroids = np.array(centroids)

    try:
        vor = Voronoi(centroids)

        for ridge_points in vor.ridge_points:
            i, j = ridge_points
            if i >= 0 and j >= 0 and i < len(label_list) and j < len(label_list):
                mask_i = labels == label_list[i]
                mask_j = labels == label_list[j]
                dilated_i = ndimage.binary_dilation(mask_i)
                contact = np.sum(dilated_i & mask_j)

                if contact >= min_contact_px:
                    edge = tuple(sorted([int(label_list[i]), int(label_list[j])]))
                    adjacency.add(edge)
    except Exception:
        return centroid_distance_adjacency(labels)

    return adjacency


# =============================================================================
# Benchmarking Framework
# =============================================================================

def compute_metrics(gt_adjacency: Set, pred_adjacency: Set) -> Dict:
    """Compute precision, recall, F1 for predicted adjacency."""
    true_positives = len(gt_adjacency & pred_adjacency)
    false_positives = len(pred_adjacency - gt_adjacency)
    false_negatives = len(gt_adjacency - pred_adjacency)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'n_predicted': len(pred_adjacency),
        'n_ground_truth': len(gt_adjacency),
    }


def benchmark_methods(mask_paths: List[Path], min_contact_px: int = 3) -> Dict:
    """Benchmark all methods on a set of images."""

    methods = {
        'EndoPiGraph': lambda labels: endopigraph_adjacency(labels, min_contact_px),
        'Simple_Dilation': lambda labels: simple_dilation_adjacency(labels, min_contact_px),
        'Centroid_Distance': lambda labels: centroid_distance_adjacency(labels),
        'Voronoi': lambda labels: voronoi_adjacency(labels, min_contact_px),
    }

    results = {name: [] for name in methods}

    for i, mask_path in enumerate(mask_paths):
        print(f"  [{i+1}/{len(mask_paths)}] {mask_path.name}...", end=" ")

        try:
            labels = load_label_mask(mask_path)
            n_nuclei = len(np.unique(labels)) - 1

            if n_nuclei < 2:
                print("skipped (< 2 nuclei)")
                continue

            gt_adjacency = compute_ground_truth_adjacency(labels, min_contact_px)

            if len(gt_adjacency) == 0:
                print("skipped (no contacts)")
                continue

            method_results = []
            for name, method_fn in methods.items():
                pred_adjacency = method_fn(labels)
                metrics = compute_metrics(gt_adjacency, pred_adjacency)
                metrics['file'] = mask_path.name
                metrics['organ'] = mask_path.parent.parent.name
                metrics['n_nuclei'] = n_nuclei
                results[name].append(metrics)
                method_results.append(f"{name[:3]}={metrics['f1']:.2f}")

            print(" | ".join(method_results))

        except Exception as e:
            print(f"error: {e}")
            continue

    return results


def aggregate_results(results: Dict) -> Dict:
    """Aggregate per-image results into summary statistics."""
    import pandas as pd

    summary = {}
    for method_name, method_results in results.items():
        if not method_results:
            continue

        df = pd.DataFrame(method_results)

        summary[method_name] = {
            'n_images': len(method_results),
            'total_nuclei': int(df['n_nuclei'].sum()),
            'total_gt_edges': int(df['n_ground_truth'].sum()),
            'total_pred_edges': int(df['n_predicted'].sum()),
            'mean_precision': float(df['precision'].mean()),
            'mean_recall': float(df['recall'].mean()),
            'mean_f1': float(df['f1'].mean()),
            'std_f1': float(df['f1'].std()),
            'min_f1': float(df['f1'].min()),
            'max_f1': float(df['f1'].max()),
        }

    return summary


def main():
    print("=" * 70)
    print("NUINSSEG BENCHMARK: Multi-Method Adjacency Extraction Comparison")
    print("=" * 70)

    if not HAS_SKIMAGE:
        print("\nError: scikit-image required. Install with:")
        print("  pip install scikit-image")
        return

    # Paths
    data_dir = Path(__file__).parent.parent / "data" / "NuInsSeg"
    output_dir = Path(__file__).parent.parent / "runs" / "nuinsseg_validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        print(f"\nData not found: {data_dir}")
        print("Run setup_nuinsseg.py first")
        return

    # Find all mask files
    mask_paths = list(data_dir.glob("*/label masks/*.tif"))
    print(f"\nFound {len(mask_paths)} mask files")

    if len(mask_paths) == 0:
        print("No mask files found!")
        return

    # Use subset for validation
    max_images = 100
    if len(mask_paths) > max_images:
        print(f"Using random subset of {max_images} images for validation")
        np.random.seed(42)
        mask_paths = list(np.random.choice(mask_paths, max_images, replace=False))

    # Benchmark
    print("\n" + "-" * 70)
    print("BENCHMARKING METHODS")
    print("-" * 70)
    print("Methods:")
    print("  1. EndoPiGraph - Full pixel-level interface extraction")
    print("  2. Simple_Dilation - Basic 1px dilation overlap")
    print("  3. Centroid_Distance - Adjacent if centroids within 1.5x mean diameter")
    print("  4. Voronoi - Adjacent if sharing Voronoi boundary + mask contact")
    print()

    results = benchmark_methods(mask_paths, min_contact_px=3)
    summary = aggregate_results(results)

    # Print results table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Method':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'F1 Std':>10}")
    print("-" * 70)

    for method_name, stats in summary.items():
        print(f"{method_name:<25} {stats['mean_precision']:>9.1%} {stats['mean_recall']:>9.1%} "
              f"{stats['mean_f1']:>9.1%} {stats['std_f1']:>9.1%}")

    # Save results
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    output = {
        'summary': convert_numpy(summary),
        'per_image': {name: convert_numpy(res) for name, res in results.items()},
    }

    with open(output_dir / "method_comparison_results.json", 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_dir / 'method_comparison_results.json'}")


if __name__ == "__main__":
    main()
