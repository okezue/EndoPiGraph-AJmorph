from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

def positions_allen(subset=True)->pd.DataFrame:
    p="runs/allen_merfish_subset_ec2/cell_types.parquet" if subset else "runs/allen_merfish_ec2/cell_types.parquet"
    d=pd.read_parquet(p)
    d["lineage"]=d["lineage"].fillna("unassigned")
    return d[["cell_id","x","y","lineage","brain_section_label","class"]]

def positions_xenium_breast()->Optional[pd.DataFrame]:
    bd=Path("data/xenium_breast_rep1/outs/cell_boundaries.parquet")
    if not bd.exists(): return None
    polys=pd.read_parquet(bd)
    cat=polys["cell_id"].astype("category")
    cid_map={c:i+1 for i,c in enumerate(cat.cat.categories)}
    polys["_cid"]=polys["cell_id"].map(cid_map).astype(np.int32)
    c=polys.groupby("_cid").agg(x=("vertex_x","mean"),y=("vertex_y","mean")).reset_index().rename(columns={"_cid":"cell_id"})
    ct=pd.read_parquet("runs/xenium_full_ec2/cell_types.parquet")
    return c.merge(ct[["cell_id","lineage"]],on="cell_id",how="inner")

def positions_visium()->pd.DataFrame:
    sp=pd.read_csv("data/visium_breast/spatial/tissue_positions.csv")
    sp=sp[sp["in_tissue"]==1].copy()
    sp=sp.reset_index(drop=True)
    sp["cell_id"]=np.arange(1,len(sp)+1)
    sp["x"]=sp["pxl_col_in_fullres"]*0.430
    sp["y"]=sp["pxl_row_in_fullres"]*0.430
    st=pd.read_parquet("runs/visium_breast_local/spot_types.parquet")
    return sp[["cell_id","x","y"]].merge(st[["cell_id","lineage"]],on="cell_id",how="inner")

def positions_stereoseq(which="brain")->Optional[pd.DataFrame]:
    p={"brain":"data/stereoseq_brain/brain_cell_bin.h5ad","embryo":"data/stereoseq_embryo_e95.h5ad"}[which]
    if not Path(p).exists(): return None
    try:
        import anndata as ad
    except ImportError:
        return None
    a=ad.read_h5ad(p,backed="r")
    xy=np.asarray(a.obsm["spatial"])
    n=xy.shape[0]
    ct=pd.read_parquet(f"runs/stereoseq_{which}_local/cell_types.parquet" if which=="brain" else "runs/stereoseq_embryo_e95_local/cell_types.parquet")
    df=pd.DataFrame({"cell_id":np.arange(1,n+1),"x":xy[:,0],"y":xy[:,1]})
    return df.merge(ct[["cell_id","lineage"]],on="cell_id",how="inner")

def positions_cosmx(which="colon")->Optional[pd.DataFrame]:
    p=f"data/cosmx_{which}/meta.csv.gz"
    if not Path(p).exists(): return None
    m=pd.read_csv(p,compression="gzip",low_memory=False,
        usecols=["fov","cell_ID","CenterX_global_px","CenterY_global_px"])
    m["_k"]=m["fov"].astype(str)+"_"+m["cell_ID"].astype(str)
    m=m.drop_duplicates("_k").reset_index(drop=True)
    m["cell_id"]=np.arange(1,len(m)+1)
    m["x"]=m["CenterX_global_px"]*0.18
    m["y"]=m["CenterY_global_px"]*0.18
    ct=pd.read_parquet(f"runs/cosmx_{which}_ec2/cell_types.parquet")
    return m[["cell_id","x","y","fov"]].merge(ct[["cell_id","lineage"]],on="cell_id",how="inner")

def positions_mibi()->Optional[pd.DataFrame]:
    import tifffile
    p="data/mibi/fov/seg_wholecell.tiff"
    if not Path(p).exists(): return None
    L=tifffile.imread(p)
    if L.ndim==3: L=L.squeeze() if L.shape[0]==1 else L[0]
    from skimage.measure import regionprops_table
    props=regionprops_table(L,properties=("label","centroid"))
    df=pd.DataFrame({"cell_id":props["label"].astype(np.int64),
                     "y":props["centroid-0"],"x":props["centroid-1"]})
    ct=pd.read_parquet("runs/mibi_glioma_local/cell_types.parquet")
    return df.merge(ct[["cell_id","lineage"]],on="cell_id",how="inner")

def all_positions(verbose=True)->dict:
    out={}
    for name,fn in [("allen_merfish",lambda:positions_allen()),
                    ("xenium_breast",positions_xenium_breast),
                    ("visium_breast",positions_visium),
                    ("stereoseq_brain",lambda:positions_stereoseq("brain")),
                    ("stereoseq_embryo_e95",lambda:positions_stereoseq("embryo")),
                    ("cosmx_colon",lambda:positions_cosmx("colon")),
                    ("cosmx_pancreas",lambda:positions_cosmx("pancreas")),
                    ("mibi_glioma",positions_mibi)]:
        try:
            d=fn()
            if d is not None and len(d):
                out[name]=d
                if verbose:
                    print(f"  {name}: n={len(d)} cells, lineages={d['lineage'].nunique()}, x∈[{d['x'].min():.0f},{d['x'].max():.0f}]")
        except Exception as e:
            if verbose: print(f"  {name}: FAILED {e}")
    return out

if __name__=="__main__":
    all_positions()
