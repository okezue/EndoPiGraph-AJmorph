#!/usr/bin/env python3
import json
import sys
import os
from pathlib import Path
from datetime import datetime
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import tifffile
from scipy import ndimage
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
from tqdm import tqdm

from endopigraph.interfaces import extract_interfaces


def semantic_to_instance(sem: np.ndarray, min_distance: int = 10) -> np.ndarray:
    border = (sem == 2)
    background = (sem == 3)
    dist = ndimage.distance_transform_edt(~border & ~background)
    peaks = peak_local_max(dist, min_distance=min_distance, exclude_border=False)
    markers = np.zeros_like(sem, dtype=np.int32)
    for i, (y, x) in enumerate(peaks, start=1):
        markers[y, x] = i
    inst = watershed(-dist, markers, mask=~background)
    return inst.astype(np.int32)


def gt_adjacency_semantic_border(sem: np.ndarray, inst: np.ndarray) -> set:
    border_mask = (sem == 2)
    H, W = sem.shape
    adj = set()
    br, bc = np.nonzero(border_mask)
    if len(br) == 0:
        return adj

    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr = np.clip(br+dr, 0, H-1)
        nc = np.clip(bc+dc, 0, W-1)
        lbl_n = inst[nr, nc]
        lbl_b = inst[br, bc]
        diff = (lbl_n > 0) & (lbl_b > 0) & (lbl_n != lbl_b)
        for a, b in zip(lbl_n[diff].tolist(), lbl_b[diff].tolist()):
            adj.add((min(a,b), max(a,b)))
    return adj


def gt_adjacency_direct(inst: np.ndarray, min_contact: int = 3) -> set:
    adj = set()
    a_h = inst[:, :-1]; b_h = inst[:, 1:]
    d_h = (a_h!=b_h) & (a_h>0) & (b_h>0)
    i_h = np.minimum(a_h[d_h], b_h[d_h])
    j_h = np.maximum(a_h[d_h], b_h[d_h])

    a_v = inst[:-1, :]; b_v = inst[1:, :]
    d_v = (a_v!=b_v) & (a_v>0) & (b_v>0)
    i_v = np.minimum(a_v[d_v], b_v[d_v])
    j_v = np.maximum(a_v[d_v], b_v[d_v])

    all_i = np.concatenate([i_h, i_v])
    all_j = np.concatenate([j_h, j_v])
    if len(all_i) == 0:
        return adj

    codes = all_i.astype(np.int64) * 1000000 + all_j.astype(np.int64)
    uniq, counts = np.unique(codes, return_counts=True)
    for code, cnt in zip(uniq, counts):
        if cnt >= min_contact:
            adj.add((int(code//1000000), int(code%1000000)))
    return adj


def endopigraph_adj(inst, min_px=3):
    iface = extract_interfaces(inst, min_contact_px=min_px)
    return {(int(min(r["cell_i"],r["cell_j"])),int(max(r["cell_i"],r["cell_j"])))
            for _, r in iface.edges.iterrows()}


def dilation_adj(inst, px=2):
    adj = set()
    ulabels = np.unique(inst)
    ulabels = ulabels[ulabels > 0]
    if len(ulabels) > 200:
        ulabels = ulabels[:200]
    dilated = {int(l): ndimage.binary_dilation(inst==l, iterations=px) for l in ulabels}
    for i, l1 in enumerate(ulabels):
        for l2 in ulabels[i+1:]:
            if np.any(dilated[int(l1)] & dilated[int(l2)]):
                adj.add((int(min(l1,l2)), int(max(l1,l2))))
    return adj


def centroid_adj(inst, thresh=50):
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
                adj.add((min(labels[i],labels[j]), max(labels[i],labels[j])))
    return adj


def delaunay_adj(inst):
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
                adj.add((min(l1,l2), max(l1,l2)))
    return adj


def benchmark(data_dir: Path, output_dir: Path):
    labels_dir = data_dir / "labels"
    paths = sorted(labels_dir.glob("*.tif"))
    print(f"Found {len(paths)} images")

    methods = {
        "EndoPiGraph": lambda inst: endopigraph_adj(inst, min_px=3),
        "Dilation (2px)": lambda inst: dilation_adj(inst, px=2),
        "Centroid (50px)": lambda inst: centroid_adj(inst, thresh=50),
        "Delaunay": lambda inst: delaunay_adj(inst),
    }

    totals_sem = {m: {"tp":0,"fp":0,"fn":0} for m in methods}
    totals_dir = {m: {"tp":0,"fp":0,"fn":0} for m in methods}
    timings = {m: 0.0 for m in methods}
    n_imgs = 0

    for lp in tqdm(paths, desc="Benchmark"):
        sem = tifffile.imread(lp)
        inst = semantic_to_instance(sem, min_distance=10)
        n = int(inst.max())
        if n < 3:
            continue
        n_imgs += 1

        gt_sem = gt_adjacency_semantic_border(sem, inst)
        gt_dir = gt_adjacency_direct(inst, min_contact=3)

        for mname, mfn in methods.items():
            t0 = time.time()
            try:
                pred = mfn(inst)
            except:
                pred = set()
            timings[mname] += time.time() - t0

            for gt, tots in [(gt_sem, totals_sem), (gt_dir, totals_dir)]:
                tp = len(gt & pred)
                fp = len(pred - gt)
                fn = len(gt - pred)
                tots[mname]["tp"] += tp
                tots[mname]["fp"] += fp
                tots[mname]["fn"] += fn

    output_dir.mkdir(parents=True, exist_ok=True)

    def _metrics(t):
        p = t["tp"]/(t["tp"]+t["fp"]) if (t["tp"]+t["fp"])>0 else 0
        r = t["tp"]/(t["tp"]+t["fn"]) if (t["tp"]+t["fn"])>0 else 0
        f = 2*p*r/(p+r) if (p+r)>0 else 0
        return round(p,4), round(r,4), round(f,4)

    print(f"\n{'='*80}")
    print(f"CORNEA CELLS BENCHMARK — {n_imgs} images")
    print(f"{'='*80}")

    print(f"\nGT: Semantic border adjacency")
    print(f"{'Method':<22} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Time':>8}")
    print("-"*55)
    results_sem = {}
    for m in methods:
        p,r,f = _metrics(totals_sem[m])
        t = timings[m]
        results_sem[m] = {"precision":p, "recall":r, "f1":f, **totals_sem[m], "time_s": round(t,2)}
        print(f"{m:<22} {p*100:>7.1f}% {r*100:>7.1f}% {f*100:>7.1f}% {t:>7.1f}s")

    print(f"\nGT: Direct pixel adjacency (min_contact=3)")
    print(f"{'Method':<22} {'Prec':>8} {'Rec':>8} {'F1':>8}")
    print("-"*47)
    results_dir = {}
    for m in methods:
        p,r,f = _metrics(totals_dir[m])
        results_dir[m] = {"precision":p, "recall":r, "f1":f, **totals_dir[m]}
        print(f"{m:<22} {p*100:>7.1f}% {r*100:>7.1f}% {f*100:>7.1f}%")

    print(f"{'='*80}")

    out = output_dir / "cornea_final_benchmark.json"
    with open(out, 'w') as f:
        json.dump({
            "dataset": "Cornea Cells (Specular Microscopy)",
            "n_images": n_imgs,
            "timestamp": datetime.now().isoformat(),
            "instance_method": "peak_local_max + watershed (min_distance=10)",
            "semantic_gt": results_sem,
            "direct_gt": results_dir,
        }, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / "data" / "cornea_cells"
    output_dir = Path(__file__).parent.parent / "runs" / "cornea_validation_v2"
    benchmark(data_dir, output_dir)
