#!/usr/bin/env python3
"""
Create a gold-standard benchmark dataset using S-BIAD1540 HUVEC images.

This version uses:
1. S-BIAD1540 HUVEC images (dense monolayers, better for segmentation)
2. Cellpose for cell segmentation (deep learning, more accurate)
3. Automatic adjacency extraction from segmentation
4. Framework for manual junction type annotation

The benchmark includes:
- Cell masks (Cellpose-generated, can be manually refined)
- Adjacency truth (extracted from masks)
- Edge-level placeholder annotations (to be manually filled)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import tifffile
from scipy import ndimage
from skimage.measure import label, regionprops

# Try to import Cellpose
try:
    from cellpose.models import CellposeModel
    CELLPOSE_AVAILABLE = True
except ImportError:
    CELLPOSE_AVAILABLE = False
    print("Warning: Cellpose not available, using watershed fallback")


def segment_cells_cellpose(image: np.ndarray, diameter: int = 30) -> np.ndarray:
    """Segment cells using Cellpose."""
    if not CELLPOSE_AVAILABLE:
        return segment_cells_watershed(image)

    model = CellposeModel(gpu=False, model_type='cyto2')
    masks, flows, styles = model.eval(image, diameter=diameter)
    return masks


def segment_cells_watershed(image: np.ndarray, min_cell_size: int = 200) -> np.ndarray:
    """Fallback watershed segmentation."""
    from skimage.filters import threshold_otsu, sobel
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max

    img = image.astype(float)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    edges = sobel(img)
    thresh = threshold_otsu(img)
    binary = img < thresh

    distance = ndimage.distance_transform_edt(binary)
    coords = peak_local_max(distance, min_distance=15, threshold_abs=5)

    markers = np.zeros(distance.shape, dtype=int)
    for i, (y, x) in enumerate(coords):
        markers[y, x] = i + 1
    markers = ndimage.grey_dilation(markers, size=3)

    labels = watershed(edges, markers, mask=binary)

    for region in regionprops(labels):
        if region.area < min_cell_size:
            labels[labels == region.label] = 0

    return label(labels > 0)


def extract_adjacency(masks: np.ndarray) -> List[Dict]:
    """Extract cell adjacency with contact information."""
    adjacency = []

    # Find boundaries between cells
    for cell_id in range(1, masks.max() + 1):
        cell_mask = masks == cell_id
        dilated = ndimage.binary_dilation(cell_mask, iterations=2)

        # Find neighbors
        neighbor_region = dilated & (masks != cell_id) & (masks > 0)
        neighbor_ids = np.unique(masks[neighbor_region])

        for neighbor_id in neighbor_ids:
            if neighbor_id > cell_id:  # Avoid duplicates
                # Calculate contact length
                neighbor_mask = masks == neighbor_id
                contact = dilated & ndimage.binary_dilation(neighbor_mask, iterations=2)
                contact_pixels = contact.sum()

                # Find contact centroid
                contact_coords = np.argwhere(contact)
                if len(contact_coords) > 0:
                    centroid = contact_coords.mean(axis=0)
                else:
                    centroid = [0, 0]

                adjacency.append({
                    'cell_i': int(cell_id),
                    'cell_j': int(neighbor_id),
                    'contact_px': int(contact_pixels),
                    'contact_centroid': [float(centroid[1]), float(centroid[0])],  # x, y
                    'junction_type': None,  # To be manually annotated
                })

    return adjacency


def process_sbiad1540_images(
    data_dir: Path,
    output_dir: Path,
    max_images: int = 50,
    use_cellpose: bool = True,
) -> List[Dict]:
    """Process S-BIAD1540 images for the benchmark."""

    images_dir = data_dir / 'S-BIAD1540' / 'images_egm2'
    if not images_dir.exists():
        print(f"Warning: {images_dir} not found")
        return []

    tiff_files = list(images_dir.glob('*.tif'))[:max_images]
    print(f"Found {len(tiff_files)} images in S-BIAD1540")

    benchmark_entries = []

    for i, tiff_path in enumerate(tiff_files):
        print(f"[{i+1}/{len(tiff_files)}] Processing {tiff_path.name}...")

        try:
            # Load image
            raw = tifffile.imread(tiff_path)
            if raw.ndim == 3:
                # Multi-channel: use first channel (VE-cadherin)
                ve_cadherin = raw[0]
            else:
                ve_cadherin = raw

            # Segment cells
            print("  Segmenting cells...")
            if use_cellpose and CELLPOSE_AVAILABLE:
                masks = segment_cells_cellpose(ve_cadherin, diameter=30)
            else:
                masks = segment_cells_watershed(ve_cadherin)

            n_cells = int(masks.max())
            print(f"  Found {n_cells} cells")

            if n_cells < 5:
                print("  Skipping: too few cells")
                continue

            # Extract adjacency
            adjacency = extract_adjacency(masks)
            print(f"  Found {len(adjacency)} cell-cell edges")

            if len(adjacency) < 3:
                print("  Skipping: too few edges")
                continue

            # Create benchmark ID
            benchmark_id = f"sbiad1540_{tiff_path.stem}".replace(' ', '_').replace('-', '_')

            # Determine condition from filename
            if 'static' in tiff_path.name.lower():
                condition = 'static'
            elif '18' in tiff_path.name or '20' in tiff_path.name:
                condition = 'high_shear'
            elif '6dyn' in tiff_path.name:
                condition = 'low_shear'
            else:
                condition = 'unknown'

            # Save image
            img_out = output_dir / 'images' / f"{benchmark_id}.tif"
            tifffile.imwrite(img_out, ve_cadherin.astype(np.uint16))

            # Save mask
            mask_out = output_dir / 'masks' / f"{benchmark_id}_mask.tif"
            tifffile.imwrite(mask_out, masks.astype(np.uint16))

            # Create annotation
            entry = {
                'benchmark_id': benchmark_id,
                'source': 'S-BIAD1540',
                'condition': condition,
                'original_file': tiff_path.name,
                'image_shape': [int(x) for x in ve_cadherin.shape],
                'n_cells': n_cells,
                'n_edges': len(adjacency),
                'edges': adjacency,
                'annotation_status': 'auto_generated',
                'notes': 'Masks generated with Cellpose. Adjacency extracted automatically. Junction types need manual annotation.',
            }

            # Save annotation JSON
            ann_out = output_dir / 'annotations' / f"{benchmark_id}.json"
            with open(ann_out, 'w') as f:
                json.dump(entry, f, indent=2)

            benchmark_entries.append(entry)
            print(f"  Saved: {benchmark_id}")

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    return benchmark_entries


def create_annotation_template(output_dir: Path, entries: List[Dict]):
    """Create a CSV template for manual junction annotation."""

    rows = []
    for entry in entries:
        for edge in entry['edges']:
            rows.append({
                'benchmark_id': entry['benchmark_id'],
                'cell_i': edge['cell_i'],
                'cell_j': edge['cell_j'],
                'contact_px': edge['contact_px'],
                'centroid_x': edge['contact_centroid'][0],
                'centroid_y': edge['contact_centroid'][1],
                'junction_type': '',  # To be filled manually
                'confidence': '',     # Optional: high/medium/low
                'notes': '',          # Optional: annotator notes
            })

    df = pd.DataFrame(rows)
    template_path = output_dir / 'annotation_template.csv'
    df.to_csv(template_path, index=False)
    print(f"\nAnnotation template saved: {template_path}")
    print(f"Total edges to annotate: {len(rows)}")

    return df


def create_benchmark_documentation(output_dir: Path, metadata: Dict):
    """Create benchmark documentation."""

    doc = f"""# EndoPiGraph-AJmorph Gold Standard Benchmark v2

## Overview

This benchmark provides cell segmentation masks and adjacency ground truth for validating EndoPiGraph-AJmorph.

## Statistics

- **Images**: {metadata['n_images']}
- **Total cells**: {metadata['n_total_cells']}
- **Total edges**: {metadata['n_total_edges']}
- **Source**: S-BIAD1540 (HUVEC, various shear conditions)

## Directory Structure

```
benchmark/
├── images/                  # VE-cadherin images (16-bit TIFF)
├── masks/                   # Cell instance masks (16-bit TIFF, Cellpose-generated)
├── annotations/             # Per-image JSON with edge lists
├── metadata/                # Benchmark metadata
├── annotation_template.csv  # Template for manual junction annotation
├── manifest.csv             # Image manifest
└── BENCHMARK.md             # This documentation
```

## Annotation Format

Each annotation JSON contains:

```json
{{
  "benchmark_id": "sbiad1540_EGM2_static_01",
  "n_cells": 45,
  "n_edges": 120,
  "edges": [
    {{
      "cell_i": 1,
      "cell_j": 2,
      "contact_px": 85,
      "contact_centroid": [512.5, 384.2],
      "junction_type": null  // To be manually annotated
    }}
  ]
}}
```

## Junction Type Categories

For manual annotation, use these categories:

| Category | Description |
|----------|-------------|
| `straight` | Linear, continuous junction |
| `thick` | Thick, dense junction |
| `reticular` | Complex, fragmented/reticular pattern |
| `fingers` | Finger-like protrusions |
| `discontinuous` | Interrupted, gapped junction |

## Manual Annotation Instructions

1. Open `annotation_template.csv`
2. For each edge, examine the junction at the centroid location
3. Fill in `junction_type` with one of the categories above
4. Optionally add `confidence` (high/medium/low) and `notes`
5. Save as `annotations_manual.csv`

### Using ImageJ/FIJI for annotation

```
1. Open image: benchmark/images/[benchmark_id].tif
2. Open mask: benchmark/masks/[benchmark_id]_mask.tif
3. Overlay mask on image (Image > Overlay > Add Image)
4. Navigate to edge centroids listed in annotation_template.csv
5. Record junction type observations
```

## Validating EndoPiGraph

```python
import json
import pandas as pd
from endopigraph import extract_interfaces, compute_interface_features

# Load ground truth
with open('benchmark/annotations/sbiad1540_example.json') as f:
    gt = json.load(f)

gt_edges = set((e['cell_i'], e['cell_j']) for e in gt['edges'])

# Run EndoPiGraph
import tifffile
mask = tifffile.imread('benchmark/masks/sbiad1540_example_mask.tif')
interfaces = extract_interfaces(mask)
pred_edges = set((row['cell_i'], row['cell_j']) for _, row in interfaces.edges.iterrows())

# Calculate edge detection metrics
tp = len(gt_edges & pred_edges)
precision = tp / len(pred_edges)
recall = tp / len(gt_edges)
f1 = 2 * precision * recall / (precision + recall)

print(f"Edge detection: P={{precision:.3f}}, R={{recall:.3f}}, F1={{f1:.3f}}")
```

## Citation

If you use this benchmark, please cite:

1. EndoPiGraph-AJmorph
2. S-BIAD1540 dataset (BioImage Archive)

---

*Generated: January 2026*
*Segmentation: {'Cellpose' if CELLPOSE_AVAILABLE else 'Watershed'}*
"""

    with open(output_dir / 'BENCHMARK.md', 'w') as f:
        f.write(doc)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Create gold-standard benchmark v2')
    parser.add_argument('--max-images', type=int, default=30, help='Maximum number of images')
    parser.add_argument('--data-root', type=str, default='data', help='Data root directory')
    parser.add_argument('--output', type=str, default='benchmark', help='Output directory')
    parser.add_argument('--no-cellpose', action='store_true', help='Use watershed instead of Cellpose')

    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output)

    # Create directories
    (output_dir / 'images').mkdir(parents=True, exist_ok=True)
    (output_dir / 'masks').mkdir(parents=True, exist_ok=True)
    (output_dir / 'annotations').mkdir(parents=True, exist_ok=True)
    (output_dir / 'metadata').mkdir(parents=True, exist_ok=True)

    # Process images
    entries = process_sbiad1540_images(
        data_root,
        output_dir,
        max_images=args.max_images,
        use_cellpose=not args.no_cellpose,
    )

    if not entries:
        print("No images processed!")
        return

    # Create metadata
    metadata = {
        'benchmark_name': 'EndoPiGraph-AJmorph Gold Standard Benchmark v2',
        'version': '2.0',
        'date_created': pd.Timestamp.now().isoformat(),
        'source_dataset': 'S-BIAD1540 (BioImage Archive)',
        'segmentation_method': 'Cellpose' if (CELLPOSE_AVAILABLE and not args.no_cellpose) else 'Watershed',
        'n_images': len(entries),
        'n_total_cells': sum(e['n_cells'] for e in entries),
        'n_total_edges': sum(e['n_edges'] for e in entries),
        'conditions': list(set(e['condition'] for e in entries)),
        'images': [e['benchmark_id'] for e in entries],
    }

    # Save metadata
    with open(output_dir / 'metadata' / 'benchmark_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # Create manifest
    manifest = pd.DataFrame([
        {
            'benchmark_id': e['benchmark_id'],
            'image_file': f"images/{e['benchmark_id']}.tif",
            'mask_file': f"masks/{e['benchmark_id']}_mask.tif",
            'annotation_file': f"annotations/{e['benchmark_id']}.json",
            'n_cells': e['n_cells'],
            'n_edges': e['n_edges'],
            'condition': e['condition'],
        }
        for e in entries
    ])
    manifest.to_csv(output_dir / 'manifest.csv', index=False)

    # Create annotation template
    create_annotation_template(output_dir, entries)

    # Create documentation
    create_benchmark_documentation(output_dir, metadata)

    print(f"\n{'='*60}")
    print("BENCHMARK CREATION COMPLETE")
    print(f"{'='*60}")
    print(f"Images: {len(entries)}")
    print(f"Total cells: {metadata['n_total_cells']}")
    print(f"Total edges: {metadata['n_total_edges']}")
    print(f"Output: {output_dir}")
    print(f"\nNext steps:")
    print(f"1. Review masks in benchmark/masks/")
    print(f"2. Fill in junction types in annotation_template.csv")
    print(f"3. Run validation script to test EndoPiGraph")


if __name__ == '__main__':
    main()
