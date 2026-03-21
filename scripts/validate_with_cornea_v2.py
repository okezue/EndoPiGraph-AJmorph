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
from skimage.segmentation import watershed
from skimage.measure import label
from tqdm import tqdm

from endopigraph.interfaces import extract_interfaces


def semantic_to_instance_cc(semantic_mask: np.ndarray) -> np.ndarray:
    interior = (semantic_mask == 1)
    instances = label(interior)
    return instances.astype(np.int32)


def semantic_to_instance_watershed(semantic_mask: np.ndarray) -> np.ndarray:
    interior = (semantic_mask == 1)
    border = (semantic_mask == 2)
    background = (semantic_mask == 3)
    instances_cc = label(interior)
    dist = ndimage.distance_transform_edt(~border & ~background)
    instances = watershed(-dist, markers=instances_cc, mask=~background)
    return instances.astype(np.int32)


def gt_adjacency_from_semantic(semantic_mask: np.ndarray, instance_mask: np.ndarray) -> set:
    border = (semantic_mask == 2)
    H, W = semantic_mask.shape
    adjacency = set()

    border_r, border_c = np.nonzero(border)
    if len(border_r) == 0:
        return adjacency

    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
        nr = border_r + dr
        nc = border_c + dc
        valid = (nr>=0) & (nr<H) & (nc>=0) & (nc<W)
        nr = nr[valid]
        nc = nc[valid]
        br = border_r[valid]
        bc = border_c[valid]

        neighbor_labels = instance_mask[nr, nc]
        cell_mask = neighbor_labels > 0

        for dr2, dc2 in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            nr2 = br + dr2
            nc2 = bc + dc2
            v2 = (nr2>=0) & (nr2<H) & (nc2>=0) & (nc2<W) & cell_mask
            nr2 = nr2[v2]
            nc2 = nc2[v2]
            nl1 = neighbor_labels[v2]
            nl2 = instance_mask[nr2, nc2]

            diff = (nl2 > 0) & (nl1 != nl2)
            for a, b in zip(nl1[diff].tolist(), nl2[diff].tolist()):
                adjacency.add((min(a,b), max(a,b)))

    return adjacency


def gt_adjacency_from_instances(instance_mask: np.ndarray, min_contact: int = 1) -> set:
    H, W = instance_mask.shape
    adjacency = set()

    a_h = instance_mask[:, :-1]
    b_h = instance_mask[:, 1:]
    diff_h = (a_h != b_h) & (a_h > 0) & (b_h > 0)
    i_h = np.minimum(a_h[diff_h], b_h[diff_h])
    j_h = np.maximum(a_h[diff_h], b_h[diff_h])

    a_v = instance_mask[:-1, :]
    b_v = instance_mask[1:, :]
    diff_v = (a_v != b_v) & (a_v > 0) & (b_v > 0)
    i_v = np.minimum(a_v[diff_v], b_v[diff_v])
    j_v = np.maximum(a_v[diff_v], b_v[diff_v])

    all_i = np.concatenate([i_h, i_v])
    all_j = np.concatenate([j_h, j_v])

    if len(all_i) == 0:
        return adjacency

    codes = all_i.astype(np.int64) * 100000 + all_j.astype(np.int64)
    uniq, counts = np.unique(codes, return_counts=True)

    for code, cnt in zip(uniq, counts):
        if cnt >= min_contact:
            a = int(code // 100000)
            b = int(code % 100000)
            adjacency.add((a, b))

    return adjacency


def validate_cornea_v2(data_dir: Path = None, output_dir: Path = None):
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data" / "cornea_cells"
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "runs" / "cornea_validation_v2"

    output_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = data_dir / "labels"

    if not labels_dir.exists():
        print(f"Labels directory not found: {labels_dir}")
        sys.exit(1)

    label_paths = sorted(labels_dir.glob("*.tif"))
    print(f"Found {len(label_paths)} label files")

    methods = {
        "cc_semantic_gt": {
            "instance_fn": semantic_to_instance_cc,
            "gt_fn": "semantic",
        },
        "watershed_semantic_gt": {
            "instance_fn": semantic_to_instance_watershed,
            "gt_fn": "semantic",
        },
        "watershed_instance_gt": {
            "instance_fn": semantic_to_instance_watershed,
            "gt_fn": "instance",
        },
    }

    results_all = {}
    for method_name, mcfg in methods.items():
        total = {"tp":0,"fp":0,"fn":0,"cells":0,"images":0}
        per_image = []

        for lp in tqdm(label_paths, desc=method_name):
            sem = tifffile.imread(lp)
            inst = mcfg["instance_fn"](sem)

            n_cells = int(inst.max())
            if n_cells < 2:
                continue

            if mcfg["gt_fn"] == "semantic":
                gt = gt_adjacency_from_semantic(sem, inst)
            else:
                gt = gt_adjacency_from_instances(inst, min_contact=3)

            try:
                iface = extract_interfaces(inst, min_contact_px=1)
                edges = iface.edges
                pred = set()
                for _, row in edges.iterrows():
                    pred.add((int(min(row["cell_i"], row["cell_j"])),
                              int(max(row["cell_i"], row["cell_j"]))))
            except:
                continue

            tp = len(gt & pred)
            fp = len(pred - gt)
            fn = len(gt - pred)

            prec = tp/(tp+fp) if (tp+fp)>0 else 0
            rec = tp/(tp+fn) if (tp+fn)>0 else 0
            f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0

            total["tp"] += tp
            total["fp"] += fp
            total["fn"] += fn
            total["cells"] += n_cells
            total["images"] += 1

            per_image.append({
                "file": lp.name,
                "cells": n_cells,
                "gt_edges": len(gt),
                "pred_edges": len(pred),
                "tp": tp, "fp": fp, "fn": fn,
                "precision": round(prec,4),
                "recall": round(rec,4),
                "f1": round(f1,4),
            })

        p = total["tp"]/(total["tp"]+total["fp"]) if (total["tp"]+total["fp"])>0 else 0
        r = total["tp"]/(total["tp"]+total["fn"]) if (total["tp"]+total["fn"])>0 else 0
        f = 2*p*r/(p+r) if (p+r)>0 else 0

        results_all[method_name] = {
            "summary": {
                "images": total["images"],
                "cells": total["cells"],
                "precision": round(p,4),
                "recall": round(r,4),
                "f1": round(f,4),
                "tp": total["tp"],
                "fp": total["fp"],
                "fn": total["fn"],
            },
            "per_image": per_image,
        }

        print(f"\n{method_name}: P={p*100:.1f}% R={r*100:.1f}% F1={f*100:.1f}%")

    out_path = output_dir / "cornea_v2_results.json"
    with open(out_path, 'w') as f:
        json.dump({
            "dataset": "Cornea Cells",
            "timestamp": datetime.now().isoformat(),
            "methods": results_all,
        }, f, indent=2)

    print(f"\nResults saved to: {out_path}")
    return results_all


if __name__ == "__main__":
    validate_cornea_v2()
