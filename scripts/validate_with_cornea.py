#!/usr/bin/env python3
"""
Validate EndoPiGraph adjacency extraction using Cornea Cells dataset.

Dataset: U-Net_Segmentation-Cornea_Cells
Source: https://github.com/svdeepak99/U-Net_Segmentation-Cornea_Cells
Modality: Specular microscopy of corneal endothelium
Format: Semantic segmentation (1=interior, 2=border, 3=background)

This script converts semantic masks to instance masks using watershed,
then validates EndoPiGraph's adjacency extraction against ground truth.
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
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from tqdm import tqdm

from endopigraph.interfaces import extract_interfaces


def semantic_to_instance(semantic_mask: np.ndarray, min_distance: int = 15) -> np.ndarray:
    """
    Convert semantic segmentation mask to instance segmentation.

    Args:
        semantic_mask: Semantic mask with values 1=interior, 2=border, 3=background
        min_distance: Minimum distance between cell centers for peak detection

    Returns:
        Instance mask where each cell has a unique integer ID
    """
    cell_border = (semantic_mask == 2)
    background = (semantic_mask == 3)

    # Distance transform from borders
    dist_from_border = ndimage.distance_transform_edt(~cell_border)

    # Find local maxima as cell centers
    local_maxi = peak_local_max(
        dist_from_border,
        min_distance=min_distance,
        exclude_border=False
    )

    # Create markers at local maxima
    markers = np.zeros_like(semantic_mask, dtype=np.int32)
    for i, (y, x) in enumerate(local_maxi):
        markers[y, x] = i + 1

    # Watershed segmentation (mask out background)
    instance_mask = watershed(
        -dist_from_border,
        markers,
        mask=~background
    )

    return instance_mask


def get_ground_truth_adjacency(mask: np.ndarray, min_contact_pixels: int = 2):
    """
    Extract ground truth adjacency from instance segmentation mask.

    Two cells are adjacent if they share at least min_contact_pixels
    boundary pixels (8-connectivity check).
    """
    adjacency = set()
    labels = np.unique(mask)
    labels = labels[labels > 0]  # Remove background

    # For each cell, find neighbors by checking boundary pixels
    for label in labels:
        cell_mask = (mask == label)
        # Dilate by 1 pixel to find neighbors
        dilated = ndimage.binary_dilation(cell_mask, iterations=1)
        # Find labels that overlap with dilation (excluding self and background)
        neighbor_region = dilated & ~cell_mask
        neighbor_labels = np.unique(mask[neighbor_region])
        neighbor_labels = neighbor_labels[neighbor_labels > 0]
        neighbor_labels = neighbor_labels[neighbor_labels != label]

        for neighbor in neighbor_labels:
            # Count contact pixels
            neighbor_mask = (mask == neighbor)
            dilated_neighbor = ndimage.binary_dilation(neighbor_mask, iterations=1)
            contact = dilated & dilated_neighbor & ~cell_mask & ~neighbor_mask
            contact_pixels = contact.sum()

            if contact_pixels >= min_contact_pixels:
                edge = tuple(sorted([int(label), int(neighbor)]))
                adjacency.add(edge)

    return adjacency


def validate_cornea_dataset(data_dir: Path = None, output_dir: Path = None):
    """Run validation on Cornea Cells dataset."""
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data" / "cornea_cells"

    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "runs" / "cornea_validation"

    output_dir.mkdir(parents=True, exist_ok=True)

    labels_dir = data_dir / "labels"

    if not labels_dir.exists():
        print(f"Labels directory not found: {labels_dir}")
        print("Please run setup_cornea.py first")
        sys.exit(1)

    # Find all label files
    label_paths = sorted(labels_dir.glob("*.tif"))
    print(f"Found {len(label_paths)} label files")

    # Validation results
    results = {
        "dataset": "Cornea Cells",
        "source": "https://github.com/svdeepak99/U-Net_Segmentation-Cornea_Cells",
        "modality": "Specular microscopy",
        "timestamp": datetime.now().isoformat(),
        "images": [],
        "summary": {}
    }

    total_gt_edges = 0
    total_pred_edges = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_cells = 0

    for label_path in tqdm(label_paths, desc="Validating"):
        # Load semantic mask
        semantic_mask = tifffile.imread(label_path)

        # Convert to instance mask
        instance_mask = semantic_to_instance(semantic_mask, min_distance=15)

        num_cells = instance_mask.max()
        if num_cells < 2:
            continue

        total_cells += num_cells

        # Get ground truth adjacency
        gt_adjacency = get_ground_truth_adjacency(instance_mask, min_contact_pixels=2)

        # Get EndoPiGraph adjacency
        try:
            iface = extract_interfaces(instance_mask)
            edges = iface.edges
            edges = edges[edges['contact_px'] >= 2]
            pred_adjacency = set()
            for _, row in edges.iterrows():
                edge = tuple(sorted([int(row['cell_i']), int(row['cell_j'])]))
                pred_adjacency.add(edge)
        except Exception as e:
            print(f"Error processing {label_path.name}: {e}")
            continue

        # Calculate metrics
        tp = len(gt_adjacency & pred_adjacency)
        fp = len(pred_adjacency - gt_adjacency)
        fn = len(gt_adjacency - pred_adjacency)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        total_gt_edges += len(gt_adjacency)
        total_pred_edges += len(pred_adjacency)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        results["images"].append({
            "filename": label_path.name,
            "num_cells": int(num_cells),
            "gt_edges": len(gt_adjacency),
            "pred_edges": len(pred_adjacency),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4)
        })

    # Calculate overall metrics
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

    results["summary"] = {
        "total_images": len(results["images"]),
        "total_cells": total_cells,
        "total_gt_edges": total_gt_edges,
        "total_pred_edges": total_pred_edges,
        "total_true_positives": total_tp,
        "total_false_positives": total_fp,
        "total_false_negatives": total_fn,
        "overall_precision": round(overall_precision, 4),
        "overall_recall": round(overall_recall, 4),
        "overall_f1_score": round(overall_f1, 4)
    }

    # Save results
    results_path = output_dir / "endopigraph_validation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("CORNEA CELLS VALIDATION RESULTS")
    print("="*60)
    print(f"Dataset: Cornea Cells (Specular Microscopy)")
    print(f"Images validated: {results['summary']['total_images']}")
    print(f"Total cells: {results['summary']['total_cells']}")
    print(f"Total ground truth edges: {results['summary']['total_gt_edges']}")
    print(f"Total predicted edges: {results['summary']['total_pred_edges']}")
    print("-"*60)
    print(f"True Positives:  {results['summary']['total_true_positives']}")
    print(f"False Positives: {results['summary']['total_false_positives']}")
    print(f"False Negatives: {results['summary']['total_false_negatives']}")
    print("-"*60)
    print(f"PRECISION: {results['summary']['overall_precision']*100:.1f}%")
    print(f"RECALL:    {results['summary']['overall_recall']*100:.1f}%")
    print(f"F1 SCORE:  {results['summary']['overall_f1_score']*100:.1f}%")
    print("="*60)
    print(f"\nResults saved to: {results_path}")

    return results


if __name__ == "__main__":
    validate_cornea_dataset()
