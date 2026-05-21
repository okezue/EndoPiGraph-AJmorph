from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import pandas as pd

def load_filtered_h5(p)->pd.DataFrame:
    import h5py
    with h5py.File(p,"r") as f:
        m=f["matrix"]
        data=m["data"][:]
        indices=m["indices"][:]
        indptr=m["indptr"][:]
        shape=tuple(m["shape"][:])
        barcodes=[b.decode() for b in m["barcodes"][:]]
        feats=m["features"]
        gene_names=[g.decode() for g in feats["name"][:]]
    from scipy.sparse import csc_matrix
    X=csc_matrix((data,indices,indptr),shape=shape).T.tocsr()
    seen={}; gene_unique=[]
    for g in gene_names:
        if g not in seen:
            seen[g]=len(seen); gene_unique.append(g)
    if len(gene_unique)<len(gene_names):
        n=len(gene_unique)
        col_remap=np.array([seen[g] for g in gene_names],dtype=np.int64)
        from scipy.sparse import coo_matrix
        Xc=X.tocoo()
        Xn=coo_matrix((Xc.data,(Xc.row,col_remap[Xc.col])),shape=(X.shape[0],n)).tocsr()
        X=Xn
        gene_names=gene_unique
    df=pd.DataFrame(X.toarray().astype(np.float32),columns=gene_names)
    df.insert(0,"barcode",barcodes)
    df.insert(0,"cell_id",np.arange(1,len(df)+1))
    return df

def load_tissue_positions(spatial_dir:Path)->pd.DataFrame:
    sp=Path(spatial_dir)
    for n in ("tissue_positions.csv","tissue_positions_list.csv"):
        p=sp/n
        if p.exists():
            df=pd.read_csv(p,header=0 if n=="tissue_positions.csv" else None)
            if df.shape[1]==6 and df.columns[0]!="barcode":
                df.columns=["barcode","in_tissue","array_row","array_col","pxl_row_in_fullres","pxl_col_in_fullres"]
            return df
    raise FileNotFoundError(f"no tissue_positions in {sp}")

def visium_cells_x_gene(h5_path,spatial_dir,*,um_per_pixel:Optional[float]=None,
                       scalefactors_path:Optional[Path]=None,in_tissue_only:bool=True
                       )->Tuple[pd.DataFrame,float]:
    cxg=load_filtered_h5(h5_path)
    pos=load_tissue_positions(Path(spatial_dir))
    pos=pos[["barcode","in_tissue","array_row","array_col","pxl_row_in_fullres","pxl_col_in_fullres"]]
    df=cxg.merge(pos,on="barcode",how="inner")
    if in_tissue_only and "in_tissue" in df.columns:
        df=df[df["in_tissue"]==1].reset_index(drop=True)
    if um_per_pixel is None and scalefactors_path is not None:
        import json
        sf=json.load(open(scalefactors_path))
        spot_d_px=float(sf.get("spot_diameter_fullres",1.0))
        um_per_pixel=55.0/spot_d_px if spot_d_px>0 else 1.0
    if um_per_pixel is None: um_per_pixel=1.0
    df["x"]=df["pxl_col_in_fullres"].astype(np.float32)*um_per_pixel
    df["y"]=df["pxl_row_in_fullres"].astype(np.float32)*um_per_pixel
    return df,float(um_per_pixel)
