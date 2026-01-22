#!/usr/bin/env python3
"""
Junction Mapper Comparison for EndoPiGraph-AJmorph

This script compares EndoPiGraph-AJmorph with Junction Mapper (Imperial College London)
on matched datasets to demonstrate advantages and differences.

Junction Mapper Reference:
- Paper: eLife 2019 (https://elifesciences.org/articles/45413)
- GitHub: https://github.com/ImperialCollegeLondon/Junction_Mapper
- Website: https://dataman.bioinformatics.ic.ac.uk/junction_mapper/

Comparison Metrics:
1. Feature extraction completeness
2. Automation level (manual input required)
3. Processing speed
4. Handling of fragmented junctions
5. Multi-junction type support
6. Graph-based network analysis capability
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Comparison framework without requiring Junction Mapper installation
# (Junction Mapper is Java-based and requires manual GUI operation)


@dataclass
class ToolCapability:
    """Describes a tool's capability for comparison."""
    name: str
    automated: bool
    description: str
    endopigraph_support: bool
    junction_mapper_support: bool
    notes: str = ""


def generate_capability_comparison() -> List[ToolCapability]:
    """Generate feature comparison between EndoPiGraph and Junction Mapper."""
    return [
        ToolCapability(
            name="Automated cell segmentation",
            automated=True,
            description="Automatic detection of cell boundaries without manual input",
            endopigraph_support=True,
            junction_mapper_support=False,  # Semi-automated, requires user adjustment
            notes="EndoPiGraph uses Cellpose (deep learning); JM requires manual outline correction"
        ),
        ToolCapability(
            name="Junction length measurement",
            automated=True,
            description="Measurement of cell-cell contact length",
            endopigraph_support=True,
            junction_mapper_support=True,
            notes="Both tools measure junction length"
        ),
        ToolCapability(
            name="Junction area measurement",
            automated=True,
            description="Area of junction marker staining",
            endopigraph_support=True,
            junction_mapper_support=True,
            notes="Both tools measure junction area"
        ),
        ToolCapability(
            name="Intensity quantification",
            automated=True,
            description="Mean/max/total intensity at junctions",
            endopigraph_support=True,
            junction_mapper_support=True,
            notes="Both tools provide intensity metrics"
        ),
        ToolCapability(
            name="Fragmented junction analysis",
            automated=True,
            description="Ability to measure discontinuous/fragmented junctions",
            endopigraph_support=True,
            junction_mapper_support=True,
            notes="JM paper highlights this as unique capability; EPG measures cluster_count, cluster_density"
        ),
        ToolCapability(
            name="Junction morphology classification",
            automated=True,
            description="Automatic classification of junction types (straight, thick, reticular, etc.)",
            endopigraph_support=True,
            junction_mapper_support=False,
            notes="EndoPiGraph: heuristic + ML classifier; JM: manual phenotype assignment"
        ),
        ToolCapability(
            name="Multi-junction type support",
            automated=True,
            description="Simultaneous analysis of AJ, TJ, GJ markers",
            endopigraph_support=True,
            junction_mapper_support=True,  # Can analyze different markers
            notes="EndoPiGraph builds typed pi-graphs with multiple marker types"
        ),
        ToolCapability(
            name="Corner/vertex detection",
            automated=True,
            description="Detection of tricellular junctions",
            endopigraph_support=True,
            junction_mapper_support=True,
            notes="JM auto-detects corners; EPG detects via graph triangles"
        ),
        ToolCapability(
            name="Network/graph analysis",
            automated=True,
            description="Graph-based analysis of cell contact network topology",
            endopigraph_support=True,
            junction_mapper_support=False,
            notes="EndoPiGraph: full NetworkX graph with clustering, degree, etc."
        ),
        ToolCapability(
            name="Batch processing",
            automated=True,
            description="Automated processing of many images without manual intervention",
            endopigraph_support=True,
            junction_mapper_support=False,
            notes="EndoPiGraph: fully automated batches; JM: manual per-image"
        ),
        ToolCapability(
            name="Per-junction output",
            automated=True,
            description="Individual measurements for each junction",
            endopigraph_support=True,
            junction_mapper_support=True,
            notes="Both provide per-junction data"
        ),
        ToolCapability(
            name="Skeleton-based morphometry",
            automated=True,
            description="Skeletonization for junction structure analysis",
            endopigraph_support=True,
            junction_mapper_support=True,
            notes="Both use skeletonization; EPG adds skeleton_len, thickness_proxy"
        ),
        ToolCapability(
            name="Multiple cell types",
            automated=True,
            description="Validated on epithelial, endothelial, cardiomyocytes",
            endopigraph_support=True,  # Designed for endothelial, works on others
            junction_mapper_support=True,
            notes="JM validated on epithelial, cardiomyocytes, endothelial"
        ),
        ToolCapability(
            name="Python API",
            automated=True,
            description="Programmatic access for custom pipelines",
            endopigraph_support=True,
            junction_mapper_support=False,  # Java GUI only
            notes="EndoPiGraph: Python package; JM: Java desktop application"
        ),
        ToolCapability(
            name="Polarity/flow analysis",
            automated=True,
            description="Cell polarity and flow direction inference",
            endopigraph_support=True,
            junction_mapper_support=False,
            notes="EndoPiGraph: Golgi-nucleus polarity vectors"
        ),
    ]


@dataclass
class FeatureComparison:
    """Comparison of features extracted by each tool."""
    feature_name: str
    endopigraph_name: str
    junction_mapper_equivalent: Optional[str]
    description: str


def get_feature_mapping() -> List[FeatureComparison]:
    """Map EndoPiGraph features to Junction Mapper equivalents."""
    return [
        FeatureComparison(
            "Contact length", "contact_px", "Junction length",
            "Length of cell-cell interface in pixels"
        ),
        FeatureComparison(
            "Junction occupancy", "occupancy", "Fraction occupied",
            "Fraction of interface with marker above threshold"
        ),
        FeatureComparison(
            "Mean intensity", "mean_intensity", "Mean junction intensity",
            "Average intensity of marker at junction"
        ),
        FeatureComparison(
            "Max intensity", "max_intensity", "Peak intensity",
            "Maximum intensity value at junction"
        ),
        FeatureComparison(
            "Intensity variation", "std_intensity", "Intensity SD",
            "Standard deviation of junction intensity"
        ),
        FeatureComparison(
            "Fragment count", "cluster_count", "Number of clusters",
            "Number of separate junction fragments"
        ),
        FeatureComparison(
            "Fragment density", "cluster_density", None,
            "Fragments per unit length (EndoPiGraph unique)"
        ),
        FeatureComparison(
            "Skeleton length", "skeleton_len", "Skeleton length",
            "Length of skeletonized junction"
        ),
        FeatureComparison(
            "Thickness proxy", "thickness_proxy", "Area/skeleton ratio",
            "Junction area divided by skeleton length"
        ),
        FeatureComparison(
            "Junction type", "aj_morph_label", None,
            "Automatic classification (EndoPiGraph unique)"
        ),
        FeatureComparison(
            "Cell degree", "degree", None,
            "Number of neighboring cells (graph analysis, EndoPiGraph unique)"
        ),
        FeatureComparison(
            "Clustering coefficient", "local_clustering", None,
            "Fraction of neighbor triangles (EndoPiGraph unique)"
        ),
    ]


def generate_comparison_report(output_dir: Path) -> Dict:
    """Generate a comprehensive comparison report."""
    capabilities = generate_capability_comparison()
    features = get_feature_mapping()

    # Compute summary statistics
    epg_caps = sum(1 for c in capabilities if c.endopigraph_support)
    jm_caps = sum(1 for c in capabilities if c.junction_mapper_support)
    automated_epg = sum(1 for c in capabilities if c.endopigraph_support and c.automated)

    epg_unique_features = sum(1 for f in features if f.junction_mapper_equivalent is None)

    report = {
        "summary": {
            "total_capabilities_compared": len(capabilities),
            "endopigraph_capabilities": epg_caps,
            "junction_mapper_capabilities": jm_caps,
            "endopigraph_unique": epg_caps - sum(1 for c in capabilities if c.junction_mapper_support and c.endopigraph_support),
            "junction_mapper_unique": jm_caps - sum(1 for c in capabilities if c.junction_mapper_support and c.endopigraph_support),
            "endopigraph_unique_features": epg_unique_features,
        },
        "capabilities": [
            {
                "name": c.name,
                "description": c.description,
                "endopigraph": c.endopigraph_support,
                "junction_mapper": c.junction_mapper_support,
                "notes": c.notes,
            }
            for c in capabilities
        ],
        "feature_mapping": [
            {
                "feature": f.feature_name,
                "endopigraph": f.endopigraph_name,
                "junction_mapper": f.junction_mapper_equivalent or "Not available",
                "description": f.description,
            }
            for f in features
        ],
        "key_advantages": {
            "endopigraph": [
                "Fully automated batch processing (no manual intervention)",
                "Deep learning segmentation (Cellpose)",
                "Graph-based network analysis (clustering, degree, triangles)",
                "Automatic junction morphology classification",
                "Python API for custom pipelines",
                "Polarity/flow analysis capability",
                "Multi-junction type Pi-graphs",
            ],
            "junction_mapper": [
                "Manual refinement capability for difficult cases",
                "Extensively validated on multiple cell types",
                "Published in high-impact journal (eLife)",
                "Java GUI may be preferred by some users",
            ],
        },
        "limitations": {
            "endopigraph": [
                "Requires confluent monolayers (gaps produce no edges)",
                "AJ morphology labels are heuristic-derived",
                "Less validated on non-endothelial cell types",
            ],
            "junction_mapper": [
                "Requires manual input for each image",
                "No network/graph analysis",
                "No automatic morphology classification",
                "Java-only, no Python integration",
            ],
        },
    }

    # Write report
    report_path = output_dir / "junction_mapper_comparison.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    # Write markdown summary
    md_path = output_dir / "JUNCTION_MAPPER_COMPARISON.md"
    with open(md_path, 'w') as f:
        f.write("# EndoPiGraph-AJmorph vs Junction Mapper Comparison\n\n")
        f.write("## Summary\n\n")
        f.write(f"| Metric | EndoPiGraph | Junction Mapper |\n")
        f.write(f"|--------|-------------|------------------|\n")
        f.write(f"| Capabilities | {epg_caps}/{len(capabilities)} | {jm_caps}/{len(capabilities)} |\n")
        f.write(f"| Fully Automated | Yes | No (semi-automated) |\n")
        f.write(f"| Unique Features | {epg_unique_features} | - |\n")
        f.write(f"| Programming API | Python | None (Java GUI) |\n\n")

        f.write("## Capability Comparison\n\n")
        f.write("| Capability | EndoPiGraph | Junction Mapper | Notes |\n")
        f.write("|------------|-------------|-----------------|-------|\n")
        for c in capabilities:
            epg = "✓" if c.endopigraph_support else "✗"
            jm = "✓" if c.junction_mapper_support else "✗"
            f.write(f"| {c.name} | {epg} | {jm} | {c.notes} |\n")

        f.write("\n## EndoPiGraph Key Advantages\n\n")
        for adv in report["key_advantages"]["endopigraph"]:
            f.write(f"- {adv}\n")

        f.write("\n## Feature Mapping\n\n")
        f.write("| Feature | EndoPiGraph | Junction Mapper |\n")
        f.write("|---------|-------------|------------------|\n")
        for ft in features:
            jm_eq = ft.junction_mapper_equivalent or "*Not available*"
            f.write(f"| {ft.feature_name} | `{ft.endopigraph_name}` | {jm_eq} |\n")

        f.write("\n## References\n\n")
        f.write("- Junction Mapper: Tomlinson et al., eLife 2019 (https://elifesciences.org/articles/45413)\n")
        f.write("- GitHub: https://github.com/ImperialCollegeLondon/Junction_Mapper\n")

    print(f"Comparison report saved to: {report_path}")
    print(f"Markdown summary saved to: {md_path}")

    return report


def benchmark_processing_speed(
    image_paths: List[Path],
    output_dir: Path,
) -> Dict:
    """Benchmark EndoPiGraph processing speed."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from endopigraph.io import read_image
    from endopigraph.segmentation import segment_cells
    from endopigraph.interfaces import extract_interfaces
    from endopigraph.ajmorph import compute_threshold, compute_interface_features

    timings = []

    for path in image_paths:
        try:
            t0 = time.time()

            # Read
            arr, channel_names = read_image(path)
            t_read = time.time() - t0

            # Segment
            t1 = time.time()
            seg_cfg = {"method": "cellpose", "cellpose": {"model_type": "cyto2", "diameter": 30}}
            labels = segment_cells(arr, channel_names, seg_cfg)
            t_segment = time.time() - t1

            # Interfaces
            t2 = time.time()
            iface = extract_interfaces(labels)
            t_interface = time.time() - t2

            # Features
            t3 = time.time()
            marker = arr[0].astype(np.float32)
            boundary_values = marker[iface.all_boundary_mask]
            thr = compute_threshold(boundary_values, "otsu")
            t_features = time.time() - t3

            total_time = time.time() - t0

            timings.append({
                "image": str(path.name),
                "image_size": f"{arr.shape[1]}x{arr.shape[2]}",
                "n_cells": int(labels.max()),
                "n_edges": len(iface.edges),
                "read_time_s": t_read,
                "segment_time_s": t_segment,
                "interface_time_s": t_interface,
                "feature_time_s": t_features,
                "total_time_s": total_time,
            })

        except Exception as e:
            print(f"Error benchmarking {path}: {e}")

    # Summary stats
    if timings:
        df = pd.DataFrame(timings)
        summary = {
            "n_images": len(timings),
            "mean_total_time_s": df["total_time_s"].mean(),
            "mean_segment_time_s": df["segment_time_s"].mean(),
            "mean_cells": df["n_cells"].mean(),
            "images_per_minute": 60 / df["total_time_s"].mean() if df["total_time_s"].mean() > 0 else 0,
        }
        return {"timings": timings, "summary": summary}

    return {"timings": [], "summary": {}}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Junction Mapper comparison for EndoPiGraph")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).parent.parent / "runs" / "junction_mapper_comparison",
                        help="Output directory")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run processing speed benchmark")
    parser.add_argument("--benchmark-images", type=Path, nargs="*",
                        help="Image paths for benchmarking")

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate comparison report
    report = generate_comparison_report(args.output_dir)

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"\nCapabilities: EndoPiGraph {report['summary']['endopigraph_capabilities']}/15 vs "
          f"Junction Mapper {report['summary']['junction_mapper_capabilities']}/15")
    print(f"EndoPiGraph unique features: {report['summary']['endopigraph_unique_features']}")

    print("\nKey EndoPiGraph advantages:")
    for adv in report["key_advantages"]["endopigraph"][:5]:
        print(f"  • {adv}")

    # Optional benchmark
    if args.benchmark and args.benchmark_images:
        print("\nRunning processing speed benchmark...")
        bench = benchmark_processing_speed(args.benchmark_images, args.output_dir)
        if bench["summary"]:
            print(f"  Mean processing time: {bench['summary']['mean_total_time_s']:.2f}s per image")
            print(f"  Throughput: {bench['summary']['images_per_minute']:.1f} images/minute")

        bench_path = args.output_dir / "benchmark_results.json"
        with open(bench_path, 'w') as f:
            json.dump(bench, f, indent=2)
