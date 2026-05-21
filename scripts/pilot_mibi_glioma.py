from __future__ import annotations
import argparse, json, time, sys, re
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
from endopigraph.codex_pi import (type_cells_from_markers,interface_marker_features_multichannel,
    load_codex_image,CODEX_LINEAGES)
from endopigraph.cell_typing import add_edge_types,edge_type_summary

def _find_one(root:Path,*names):
    for n in names:
        ms=list(root.rglob(n))
        if ms: return ms[0]
    return None

EXCLUDE_CHANNELS={"seg","seg_wholecell","seg_nuclear","Nuclear","Au","Noodle","Fe","chan_39","chan_48","chan_70","FoxP3_nuc_include","TMEM119_nuc_exclude"}

def _load_per_marker_stack(fov_dir:Path):
    tifs=sorted([p for p in fov_dir.glob("*.tiff") if "seg" not in p.stem.lower() and "mask" not in p.stem.lower()])
    if not tifs: return None,None,None
    arrs=[]; names=[]
    for p in tifs:
        if p.stem in EXCLUDE_CHANNELS: continue
        try:
            a=tifffile.imread(p)
        except Exception: continue
        if a.ndim==3 and a.shape[0]==1: a=a[0]
        if a.ndim!=2: continue
        arrs.append(a.astype(np.float32))
        names.append(p.stem)
    if not arrs: return None,None,None
    arr=np.stack(arrs,axis=0)
    return arr,names,tifs[0]

def _per_cell_means(arr:np.ndarray,labels:np.ndarray,channel_names:list)->pd.DataFrame:
    cids=np.unique(labels)
    cids=cids[cids>0]
    rows=[]
    for ci,name in enumerate(channel_names):
        ch=arr[ci]
        s=np.bincount(labels.ravel(),weights=ch.ravel().astype(np.float64),minlength=int(cids.max())+1)
        cnt=np.bincount(labels.ravel(),minlength=int(cids.max())+1).astype(np.float64)
        m=np.where(cnt>0,s/np.clip(cnt,1,None),0.0)
        rows.append((name,m[cids]))
    out=pd.DataFrame({k:v for k,v in rows})
    out.insert(0,"cell_id",cids.astype(np.int64))
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--fov-dir",required=True,help="Dir containing the FOV image stack and segmentation mask")
    ap.add_argument("--out",required=True)
    ap.add_argument("--dilate-px",type=int,default=2)
    ap.add_argument("--min-contact-px",type=int,default=10)
    ap.add_argument("--max-cells",type=int,default=None)
    args=ap.parse_args()
    fov=Path(args.fov_dir); outd=Path(args.out); outd.mkdir(parents=True,exist_ok=True)
    log={}; t0=time.time()

    print("[1/6] locating files...",flush=True)
    mask_p=_find_one(fov,"*whole_cell*.tiff","*wholecell*.tiff","*_mask.tiff","*labels*.tif*","seg_wholecell.tiff")
    if mask_p is None:
        cand=[p.name for p in fov.rglob("*.tif*")][:20]
        sys.exit(f"missing mask\ncandidates: {cand}")
    print(f"  mask: {mask_p}",flush=True)
    arr,ch_names,_first=_load_per_marker_stack(fov)
    if arr is None:
        img_p=_find_one(fov,"*expressions*.ome.tif","*expressions*.ome.tiff","*.ome.tif","*.ome.tiff","*stack*.tif*")
        if img_p is None: sys.exit("no markers and no stack found")
        arr,ch_names=load_codex_image(img_p)
    print(f"  channels: {len(ch_names)} ({ch_names[:8]}...)",flush=True)

    print("[2/6] loading mask...",flush=True)
    labels=tifffile.imread(mask_p).astype(np.int32)
    if labels.ndim==3: labels=labels[0]
    if arr.shape[-2:]!=labels.shape:
        if arr.shape[-2:][::-1]==labels.shape:
            arr=np.swapaxes(arr,-1,-2)
        else: sys.exit(f"shape mismatch: img {arr.shape} vs mask {labels.shape}")
    n_cells=int(labels.max())
    if args.max_cells is not None and n_cells>args.max_cells:
        keep=np.zeros(n_cells+1,dtype=bool); keep[:args.max_cells+1]=True
        labels=np.where(keep[labels],labels,0)
    log["img_shape"]=list(arr.shape)
    log["n_channels"]=int(arr.shape[0])
    log["channel_names"]=ch_names[:20]
    log["n_cells_total"]=int(labels.max())

    print("[3/6] per-cell mean intensities...",flush=True)
    cxm=_per_cell_means(arr,labels,ch_names)
    cxm.to_parquet(outd/"cells_x_marker.parquet",index=False)

    print("[4/6] typing cells from markers...",flush=True)
    types=type_cells_from_markers(cxm,min_z=0.5,min_fold=1.2)
    types.to_parquet(outd/"cell_types.parquet",index=False)
    log["lineage_counts"]=types["lineage"].value_counts().to_dict()
    log["n_cells_used"]=int(len(types))

    print("[5/6] interface marker features (multi-channel boundary integral)...",flush=True)
    t1=time.time()
    edges,_=interface_marker_features_multichannel(arr,labels,channel_names=ch_names,
        dilate_px=args.dilate_px,min_contact_px=args.min_contact_px)
    log["t_iface_features_s"]=round(time.time()-t1,2)
    log["n_edges"]=int(len(edges))
    edges.to_parquet(outd/"edges_x_marker.parquet",index=False)

    print("[6/6] typed edges + summary...",flush=True)
    typed=add_edge_types(edges,types,type_col="lineage")
    typed.to_parquet(outd/"edges_typed.parquet",index=False)
    et=edge_type_summary(typed)
    et=et.sort_values("n_edges",ascending=False).reset_index(drop=True)
    et.to_parquet(outd/"edge_type_summary.parquet",index=False)
    log["n_edge_types"]=int(len(et))
    log["top_edge_types"]=et.head(15)[["edge_type","n_edges","total_contact_px"]].to_dict(orient="records") if "total_contact_px" in et.columns else et.head(15)[["edge_type","n_edges"]].to_dict(orient="records")

    log["total_s"]=round(time.time()-t0,2)
    (outd/"pilot_summary.json").write_text(json.dumps(log,indent=2,default=str))
    print(json.dumps(log,indent=2,default=str))

if __name__=="__main__":
    main()
