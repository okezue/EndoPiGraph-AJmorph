# EndoPiGraph Gold Standard Benchmark

## Summary
- Images: 30
- Total cells: 10380
- Total edges: 25104

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

img = tifffile.imread(f"benchmark/images/{ann['benchmark_id']}.tif")
mask = tifffile.imread(f"benchmark/masks/{ann['benchmark_id']}_mask.tif")

print(f"Cells: {ann['n_cells']}, Edges: {ann['n_edges']}")
```
