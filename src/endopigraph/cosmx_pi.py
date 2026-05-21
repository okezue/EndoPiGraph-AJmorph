from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

def load_cosmx_metadata(p)->pd.DataFrame:
    df=pd.read_csv(p,compression="infer",low_memory=False)
    df.columns=[c.strip() for c in df.columns]
    rn={}
    for cand,std in [("CenterX_global_px","x_px"),("CenterY_global_px","y_px"),
                     ("CenterX_local_px","x_local_px"),("CenterY_local_px","y_local_px"),
                     ("cell_ID","cell_id"),("cell","cell_id"),("fov","fov")]:
        if cand in df.columns: rn[cand]=std
    df=df.rename(columns=rn)
    return df

def load_cosmx_expr(p,*,index_cols=("cell_ID","fov","cell"))->pd.DataFrame:
    df=pd.read_csv(p,compression="infer",low_memory=False)
    df.columns=[c.strip() for c in df.columns]
    return df

def make_cells_x_gene(expr:pd.DataFrame,meta:pd.DataFrame,*,um_per_px:float=0.18)->pd.DataFrame:
    expr=expr.copy(); meta=meta.copy()
    id_col=None
    for c in ("cell_ID","cell","cell_id"):
        if c in expr.columns and c in meta.columns: id_col=c; break
    if id_col is None:
        if "cell" in expr.columns and "fov" in expr.columns and "cell" in meta.columns and "fov" in meta.columns:
            expr["_cid"]=expr["fov"].astype(str)+"_"+expr["cell"].astype(str)
            meta["_cid"]=meta["fov"].astype(str)+"_"+meta["cell"].astype(str)
            id_col="_cid"
    if id_col is None and "fov" in expr.columns and "cell" in expr.columns:
        expr["cell_ID"]=expr["fov"].astype(str)+"_"+expr["cell"].astype(str)
        if "fov" in meta.columns and ("cell" in meta.columns or "CellId" in meta.columns):
            cc="cell" if "cell" in meta.columns else "CellId"
            meta["cell_ID"]=meta["fov"].astype(str)+"_"+meta[cc].astype(str)
            id_col="cell_ID"
    if id_col is None: raise ValueError("could not align expr/meta on a common cell id")

    meta_keep=[c for c in (id_col,"x_px","y_px","x_local_px","y_local_px","fov") if c in meta.columns]
    m=meta[meta_keep].drop_duplicates(id_col)
    df=expr.merge(m,on=id_col,how="inner")
    drop=[c for c in expr.columns if c.lower() in("cell_id","cell","fov","cell_ID")]
    gene_cols=[c for c in expr.columns if c not in drop]
    x_col="x_px" if "x_px" in df.columns else ("CenterX_global_px" if "CenterX_global_px" in df.columns else "x_local_px")
    y_col="y_px" if "y_px" in df.columns else ("CenterY_global_px" if "CenterY_global_px" in df.columns else "y_local_px")
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"missing coord columns in metadata; have {list(df.columns)[:20]}")
    df["x"]=df[x_col].astype(np.float64)*um_per_px
    df["y"]=df[y_col].astype(np.float64)*um_per_px
    df["cell_id"]=np.arange(1,len(df)+1)
    keep=["cell_id","x","y"]+(["fov"] if "fov" in df.columns else [])+gene_cols
    return df[keep].reset_index(drop=True)
