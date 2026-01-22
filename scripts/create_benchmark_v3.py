#!/usr/bin/env python3
"""
Create a gold-standard benchmark using EndoPiGraph's own segmentation.

This version uses EndoPiGraph's internal pipeline for consistency.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import numpy as np
import pandas as pd
from pathlib import Path
import tifffile

from endopigraph.segmentation import segment_cells
from endopigraph.interfaces import extract_interfaces


def process_images(
    data_dir: Path,
    output_dir: Path,
    max_images: int = 30,
) -> list:
    """Process images using EndoPiGraph pipeline."""

    images_dir = data_dir / 'S-BIAD1540' / 'images_egm2'
    if not images_dir.exists():
        print(f"Error: {images_dir} not found")
        return []

    tiff_files = list(images_dir.glob('*.tif'))[:max_images]
    print(f"Found {len(tiff_files)} images")

    # Create directories
    (output_dir / 'images').mkdir(parents=True, exist_ok=True)
    (output_dir / 'masks').mkdir(parents=True, exist_ok=True)
    (output_dir / 'annotations').mkdir(parents=True, exist_ok=True)
    (output_dir / 'metadata').mkdir(parents=True, exist_ok=True)

    benchmark_entries = []

    for i, tiff_path in enumerate(tiff_files):
        print(f"[{i+1}/{len(tiff_files)}] Processing {tiff_path.name}...")

        try:
            # Load image
            raw = tifffile.imread(tiff_path)
            if raw.ndim == 3:
                ve_cadherin = raw[0]
                channel_names = ['VE-cadherin', 'DAPI', 'GM130'][:raw.shape[0]]
            else:
                ve_cadherin = raw
                raw = raw[np.newaxis, ...]
                channel_names = ['VE-cadherin']

            # Segment using watershed (EndoPiGraph default)
            print("  Segmenting cells...")
            seg_config = {
                'method': 'watershed',
                'watershed': {
                    'min_cell_size': 500,
                    'marker_threshold_method': 'otsu',
                }
            }
            masks = segment_cells(raw, channel_names, seg_config)

            n_cells = int(masks.max())
            print(f"  Found {n_cells} cells")

            if n_cells < 5:
                print("  Skipping: too few cells")
                continue

            # Extract interfaces
            print("  Extracting interfaces...")
            iface = extract_interfaces(masks)
            n_edges = len(iface.edges)
            print(f"  Found {n_edges} edges")

            if n_edges < 3:
                print("  Skipping: too few edges")
                continue

            # Create benchmark ID
            benchmark_id = f"sbiad1540_{tiff_path.stem}".replace(' ', '_').replace('-', '_')

            # Determine condition
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

            # Create edge list with annotations
            edges = []
            for _, row in iface.edges.iterrows():
                edges.append({
                    'cell_i': int(row['cell_i']),
                    'cell_j': int(row['cell_j']),
                    'contact_px': int(row['contact_px']),
                    'junction_type': None,  # To be annotated
                })

            # Create entry
            entry = {
                'benchmark_id': benchmark_id,
                'source': 'S-BIAD1540',
                'condition': condition,
                'original_file': tiff_path.name,
                'image_shape': [int(x) for x in ve_cadherin.shape],
                'n_cells': n_cells,
                'n_edges': n_edges,
                'edges': edges,
            }

            # Save annotation
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


def create_annotation_template(output_dir: Path, entries: list):
    """Create CSV template for manual annotation."""
    rows = []
    for entry in entries:
        for edge in entry['edges']:
            rows.append({
                'benchmark_id': entry['benchmark_id'],
                'cell_i': edge['cell_i'],
                'cell_j': edge['cell_j'],
                'contact_px': edge['contact_px'],
                'junction_type': '',
                'confidence': '',
                'notes': '',
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / 'annotation_template.csv', index=False)
    print(f"\nAnnotation template: {output_dir / 'annotation_template.csv'}")
    print(f"Total edges to annotate: {len(rows)}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--max-images', type=int, default=30)
    parser.add_argument('--data-root', type=str, default='data')
    parser.add_argument('--output', type=str, default='benchmark')
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output)

    entries = process_images(data_root, output_dir, args.max_images)

    if not entries:
        print("No images processed!")
        return

    # Create metadata
    metadata = {
        'benchmark_name': 'EndoPiGraph-AJmorph Gold Standard Benchmark',
        'version': '1.0',
        'date_created': pd.Timestamp.now().isoformat(),
        'source': 'S-BIAD1540',
        'segmentation': 'EndoPiGraph watershed',
        'n_images': len(entries),
        'n_total_cells': sum(e['n_cells'] for e in entries),
        'n_total_edges': sum(e['n_edges'] for e in entries),
    }

    with open(output_dir / 'metadata' / 'benchmark_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # Create manifest
    manifest = pd.DataFrame([{
        'benchmark_id': e['benchmark_id'],
        'n_cells': e['n_cells'],
        'n_edges': e['n_edges'],
        'condition': e['condition'],
    } for e in entries])
    manifest.to_csv(output_dir / 'manifest.csv', index=False)

    # Create annotation template
    create_annotation_template(output_dir, entries)

    # Create docs
    doc = f"""# EndoPiGraph Gold Standard Benchmark

## Summary
- Images: {len(entries)}
- Total cells: {sum(e['n_cells'] for e in entries)}
- Total edges: {sum(e['n_edges'] for e in entries)}

## Files
- `images/` - VE-cadherin images
- `masks/` - Cell segmentation masks
- `annotations/` - Edge lists (JSON)
- `annotation_template.csv` - Template for manual junction type annotation

## Junction Types (for manual annotation)
| Type | Description |
|------|-------------|
| straight | Linear, continuous |
| thick | Dense, wide junction |
| reticular | Fragmented pattern |
| fingers | Finger-like protrusions |
| discontinuous | Interrupted |

## Usage
```python
import json
import tifffile

# Load benchmark image
with open('benchmark/annotations/sbiad1540_example.json') as f:
    ann = json.load(f)

img = tifffile.imread(f"benchmark/images/{{ann['benchmark_id']}}.tif")
mask = tifffile.imread(f"benchmark/masks/{{ann['benchmark_id']}}_mask.tif")

print(f"Cells: {{ann['n_cells']}}, Edges: {{ann['n_edges']}}")
```
"""
    with open(output_dir / 'BENCHMARK.md', 'w') as f:
        f.write(doc)

    print(f"\n{'='*60}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*60}")
    print(f"Images: {len(entries)}")
    print(f"Total cells: {metadata['n_total_cells']}")
    print(f"Total edges: {metadata['n_total_edges']}")
    print(f"Output: {output_dir}")


if __name__ == '__main__':
    main()
