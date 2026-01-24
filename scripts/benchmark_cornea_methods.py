#!/usr/bin/env python3
"""
Multi-method benchmark on Cornea Cells dataset.

Compares adjacency extraction methods:
1. EndoPiGraph (ours)
2. Simple Dilation
3. Centroid Distance
4. Voronoi-based

Dataset: U-Net_Segmentation-Cornea_Cells
Source: https://github.com/svdeepak99/U-Net_Segmentation-Cornea_Cells
Modality: Specular microscopy of corneal endothelium
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime

# Add src directory to path for endopigraph imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import tifffile
from scipy import ndimage
from scipy.spatial import Voronoi, Delaunay
from scipy.spatial.distance import cdist
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from tqdm import tqdm

from endopigraph.interfaces import extract_interfaces


def semantic_to_instance(semantic_mask: np.ndarray, min_distance: int = 15) -> np.ndarray:
    """Convert semantic segmentation mask to instance segmentation."""
    cell_border = (semantic_mask == 2)
    background = (semantic_mask == 3)

    dist_from_border = ndimage.distance_transform_edt(~cell_border)
    local_maxi = peak_local_max(dist_from_border, min_distance=min_distance, exclude_border=False)

    markers = np.zeros_like(semantic_mask, dtype=np.int32)
    for i, (y, x) in enumerate(local_maxi):
        markers[y, x] = i + 1

    instance_mask = watershed(-dist_from_border, markers, mask=~background)
    return instance_mask


def get_ground_truth_adjacency(mask: np.ndarray, min_contact_pixels: int = 2):
    """Extract ground truth adjacency from instance mask."""
    adjacency = set()
    labels = np.unique(mask)
    labels = labels[labels > 0]

    for label in labels:
        cell_mask = (mask == label)
        dilated = ndimage.binary_dilation(cell_mask, iterations=1)
        neighbor_region = dilated & ~cell_mask
        neighbor_labels = np.unique(mask[neighbor_region])
        neighbor_labels = neighbor_labels[neighbor_labels > 0]
        neighbor_labels = neighbor_labels[neighbor_labels != label]

        for neighbor in neighbor_labels:
            neighbor_mask = (mask == neighbor)
            dilated_neighbor = ndimage.binary_dilation(neighbor_mask, iterations=1)
            contact = dilated & dilated_neighbor & ~cell_mask & ~neighbor_mask
            contact_pixels = contact.sum()

            if contact_pixels >= min_contact_pixels:
                edge = tuple(sorted([int(label), int(neighbor)]))
                adjacency.add(edge)

    return adjacency


def endopigraph_adjacency(mask: np.ndarray, min_contact_px: int = 2):
    """Get adjacency using EndoPiGraph interface extraction."""
    try:
        iface = extract_interfaces(mask)
        edges = iface.edges
        edges = edges[edges['contact_px'] >= min_contact_px]

        adjacency = set()
        for _, row in edges.iterrows():
            edge = tuple(sorted([int(row['cell_i']), int(row['cell_j'])]))
            adjacency.add(edge)
        return adjacency
    except:
        return set()


def simple_dilation_adjacency(mask: np.ndarray, dilation_px: int = 3):
    """Get adjacency using simple dilation overlap."""
    adjacency = set()
    labels = np.unique(mask)
    labels = labels[labels > 0]

    dilated_masks = {}
    for label in labels:
        cell_mask = (mask == label)
        dilated_masks[label] = ndimage.binary_dilation(cell_mask, iterations=dilation_px)

    for i, label1 in enumerate(labels):
        for label2 in labels[i+1:]:
            if np.any(dilated_masks[label1] & dilated_masks[label2]):
                adjacency.add((int(label1), int(label2)))

    return adjacency


def centroid_distance_adjacency(mask: np.ndarray, distance_threshold: float = 50):
    """Get adjacency using centroid distance threshold."""
    adjacency = set()
    labels = np.unique(mask)
    labels = labels[labels > 0]

    centroids = {}
    for label in labels:
        coords = np.where(mask == label)
        centroids[label] = (np.mean(coords[0]), np.mean(coords[1]))

    centroid_array = np.array([centroids[l] for l in labels])

    if len(centroid_array) > 1:
        distances = cdist(centroid_array, centroid_array)

        for i in range(len(labels)):
            for j in range(i+1, len(labels)):
                if distances[i, j] < distance_threshold:
                    adjacency.add((int(labels[i]), int(labels[j])))

    return adjacency


def voronoi_adjacency(mask: np.ndarray):
    """Get adjacency using Voronoi tessellation of centroids, verified by mask overlap."""
    adjacency = set()
    labels = np.unique(mask)
    labels = labels[labels > 0]

    if len(labels) < 4:
        return adjacency

    centroids = {}
    for label in labels:
        coords = np.where(mask == label)
        centroids[label] = (np.mean(coords[1]), np.mean(coords[0]))  # x, y

    points = np.array([centroids[l] for l in labels])

    try:
        delaunay = Delaunay(points)

        for simplex in delaunay.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    l1, l2 = labels[simplex[i]], labels[simplex[j]]
                    edge = tuple(sorted([int(l1), int(l2)]))

                    # Verify with mask dilation
                    mask1 = ndimage.binary_dilation(mask == l1, iterations=1)
                    mask2 = ndimage.binary_dilation(mask == l2, iterations=1)
                    if np.any(mask1 & mask2):
                        adjacency.add(edge)
    except:
        pass

    return adjacency


def benchmark_all_methods(data_dir: Path, output_dir: Path):
    """Run benchmark comparing all methods on Cornea dataset."""
    labels_dir = data_dir / "labels"
    label_paths = sorted(labels_dir.glob("*.tif"))

    print(f"Found {len(label_paths)} images")

    methods = {
        "EndoPiGraph": endopigraph_adjacency,
        "Simple Dilation": lambda m: simple_dilation_adjacency(m, dilation_px=3),
        "Centroid Distance": lambda m: centroid_distance_adjacency(m, distance_threshold=40),
        "Voronoi": voronoi_adjacency
    }

    results = {method: {"tp": 0, "fp": 0, "fn": 0, "total_cells": 0} for method in methods}

    for label_path in tqdm(label_paths, desc="Benchmarking"):
        semantic_mask = tifffile.imread(label_path)
        instance_mask = semantic_to_instance(semantic_mask, min_distance=15)

        num_cells = instance_mask.max()
        if num_cells < 2:
            continue

        gt_adjacency = get_ground_truth_adjacency(instance_mask, min_contact_pixels=2)

        for method_name, method_func in methods.items():
            pred_adjacency = method_func(instance_mask)

            tp = len(gt_adjacency & pred_adjacency)
            fp = len(pred_adjacency - gt_adjacency)
            fn = len(gt_adjacency - pred_adjacency)

            results[method_name]["tp"] += tp
            results[method_name]["fp"] += fp
            results[method_name]["fn"] += fn
            results[method_name]["total_cells"] += num_cells

    # Calculate final metrics
    final_results = {
        "dataset": "Cornea Cells",
        "source": "https://github.com/svdeepak99/U-Net_Segmentation-Cornea_Cells",
        "modality": "Specular microscopy",
        "timestamp": datetime.now().isoformat(),
        "methods": {}
    }

    print("\n" + "="*70)
    print("CORNEA CELLS BENCHMARK RESULTS")
    print("="*70)
    print(f"{'Method':<20} {'Precision':>12} {'Recall':>12} {'F1 Score':>12}")
    print("-"*70)

    for method_name, data in results.items():
        tp, fp, fn = data["tp"], data["fp"], data["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        final_results["methods"][method_name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "total_cells": int(data["total_cells"])
        }

        print(f"{method_name:<20} {precision*100:>11.1f}% {recall*100:>11.1f}% {f1*100:>11.1f}%")

    print("="*70)

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "method_comparison_results.json"
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    return final_results


if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / "data" / "cornea_cells"
    output_dir = Path(__file__).parent.parent / "runs" / "cornea_validation"

    if not data_dir.exists():
        print(f"Data not found at {data_dir}")
        print("Please run setup_cornea.py first")
        sys.exit(1)

    benchmark_all_methods(data_dir, output_dir)
