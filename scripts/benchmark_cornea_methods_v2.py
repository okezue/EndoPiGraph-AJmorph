#!/usr/bin/env python3
import json
import sys
import os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import tifffile
from scipy import ndimage
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
from skimage.segmentation import watershed
from skimage.measure import label, regionprops
from tqdm import tqdm

from endopigraph.interfaces import extract_interfaces


def semantic_to_instance_watershed(sem: np.ndarray) -> np.ndarray:
    interior = (sem == 1)
    border = (sem == 2)
    background = (sem == 3)
    seeds = label(interior)
    dist = ndimage.distance_transform_edt(~border & ~background)
    inst = watershed(-dist, markers=seeds, mask=~background)
    return inst.astype(np.int32)


def gt_adjacency_from_semantic(sem: np.ndarray, inst: np.ndarray) -> set:
    border = (sem == 2)
    H, W = sem.shape
    adj = set()
    br, bc = np.nonzero(border)
    if len(br) == 0:
        return adj
    offsets = [(-1,0),(1,0),(0,-1),(0,1)]
    neighbor_labels_per_border = []
    for r, c in zip(br.tolist(), bc.tolist()):
        labels_here = set()
        for dr, dc in offsets:
            nr, nc = r+dr, c+dc
            if 0 <= nr < H and 0 <= nc < W:
                lbl = int(inst[nr, nc])
                if lbl > 0:
                    labels_here.add(lbl)
        if len(labels_here) >= 2:
            ls = sorted(labels_here)
            for i in range(len(ls)):
                for j in range(i+1, len(ls)):
                    adj.add((ls[i], ls[j]))
    return adj


def endopigraph_adj(inst: np.ndarray, min_px: int = 1) -> set:
    iface = extract_interfaces(inst, min_contact_px=min_px)
    edges = iface.edges
    return {(int(min(r["cell_i"],r["cell_j"])),int(max(r["cell_i"],r["cell_j"])))
            for _, r in edges.iterrows()}


def dilation_adj(inst: np.ndarray, px: int = 2) -> set:
    adj = set()
    labels = np.unique(inst)
    labels = labels[labels > 0]
    dilated = {}
    for lbl in labels:
        dilated[lbl] = ndimage.binary_dilation(inst == lbl, iterations=px)
    for i, l1 in enumerate(labels):
        for l2 in labels[i+1:]:
            if np.any(dilated[l1] & dilated[l2]):
                a, b = int(l1), int(l2)
                adj.add((min(a,b), max(a,b)))
    return adj


def centroid_adj(inst: np.ndarray, thresh: float = 40) -> set:
    adj = set()
    props = regionprops(inst)
    if len(props) < 2:
        return adj
    labels = [p.label for p in props]
    centroids = np.array([p.centroid for p in props])
    dists = cdist(centroids, centroids)
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            if dists[i,j] < thresh:
                a, b = int(labels[i]), int(labels[j])
                adj.add((min(a,b), max(a,b)))
    return adj


def delaunay_adj(inst: np.ndarray) -> set:
    adj = set()
    props = regionprops(inst)
    if len(props) < 4:
        return adj
    labels = [p.label for p in props]
    pts = np.array([p.centroid for p in props])
    try:
        tri = Delaunay(pts)
    except:
        return adj
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i+1, 3):
                l1, l2 = labels[simplex[i]], labels[simplex[j]]
                m1 = ndimage.binary_dilation(inst == l1, iterations=1)
                m2 = ndimage.binary_dilation(inst == l2, iterations=1)
                if np.any(m1 & m2):
                    adj.add((int(min(l1,l2)), int(max(l1,l2))))
    return adj


def benchmark(data_dir: Path, output_dir: Path):
    labels_dir = data_dir / "labels"
    paths = sorted(labels_dir.glob("*.tif"))
    print(f"Found {len(paths)} images")

    methods = {
        "EndoPiGraph": lambda inst: endopigraph_adj(inst, min_px=1),
        "Dilation (2px)": lambda inst: dilation_adj(inst, px=2),
        "Dilation (3px)": lambda inst: dilation_adj(inst, px=3),
        "Centroid (40px)": lambda inst: centroid_adj(inst, thresh=40),
        "Centroid (60px)": lambda inst: centroid_adj(inst, thresh=60),
        "Delaunay+verify": lambda inst: delaunay_adj(inst),
    }

    totals = {m: {"tp":0,"fp":0,"fn":0,"cells":0} for m in methods}

    for lp in tqdm(paths, desc="Benchmarking"):
        sem = tifffile.imread(lp)
        inst = semantic_to_instance_watershed(sem)
        n = int(inst.max())
        if n < 2:
            continue

        gt = gt_adjacency_from_semantic(sem, inst)
        if len(gt) == 0:
            continue

        for mname, mfn in methods.items():
            try:
                pred = mfn(inst)
            except:
                continue
            tp = len(gt & pred)
            fp = len(pred - gt)
            fn = len(gt - pred)
            totals[mname]["tp"] += tp
            totals[mname]["fp"] += fp
            totals[mname]["fn"] += fn
            totals[mname]["cells"] += n

    results = {"dataset": "Cornea Cells", "timestamp": datetime.now().isoformat(),
               "gt_method": "semantic border adjacency", "methods": {}}

    print("\n" + "="*75)
    print("CORNEA CELLS BENCHMARK (Fixed GT)")
    print("="*75)
    print(f"{'Method':<22} {'Precision':>11} {'Recall':>11} {'F1':>11}")
    print("-"*75)

    for m, t in totals.items():
        p = t["tp"]/(t["tp"]+t["fp"]) if (t["tp"]+t["fp"])>0 else 0
        r = t["tp"]/(t["tp"]+t["fn"]) if (t["tp"]+t["fn"])>0 else 0
        f = 2*p*r/(p+r) if (p+r)>0 else 0
        results["methods"][m] = {
            "precision": round(p,4), "recall": round(r,4), "f1": round(f,4),
            "tp": t["tp"], "fp": t["fp"], "fn": t["fn"], "cells": t["cells"],
        }
        print(f"{m:<22} {p*100:>10.1f}% {r*100:>10.1f}% {f*100:>10.1f}%")

    print("="*75)

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "benchmark_v2_results.json"
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {out}")
    return results


if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / "data" / "cornea_cells"
    output_dir = Path(__file__).parent.parent / "runs" / "cornea_validation_v2"
    benchmark(data_dir, output_dir)
