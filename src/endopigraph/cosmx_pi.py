from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

def load_cosmx_metadata(p)->pd.DataFrame:
    df=pd.read_csv(p,compression="infer",low_memory=False)
    df.columns=[c.strip() for c in df.columns]
    if "CenterX_global_px" in df.columns: df["x_px"]=df["CenterX_global_px"].astype(float)
    elif "CenterX_local_px" in df.columns: df["x_px"]=df["CenterX_local_px"].astype(float)
    if "CenterY_global_px" in df.columns: df["y_px"]=df["CenterY_global_px"].astype(float)
    elif "CenterY_local_px" in df.columns: df["y_px"]=df["CenterY_local_px"].astype(float)
    return df

def load_cosmx_expr(p,*,index_cols=("cell_ID","fov","cell"))->pd.DataFrame:
    import pyarrow as pa
    import pyarrow.csv as pcsv
    import gzip
    op=gzip.open(str(p),"rt") if str(p).endswith(".gz") else open(str(p),"rt")
    with op as f: header=[h.strip() for h in f.readline().rstrip("\n").split(",")]
    id_cols=set(index_cols)|{"CellId","CellID"}
    types={}
    for h in header:
        types[h]=pa.int32() if h not in id_cols else pa.int64()
    tbl=pcsv.read_csv(str(p),
        read_options=pcsv.ReadOptions(use_threads=True,block_size=1<<27),
        convert_options=pcsv.ConvertOptions(column_types=types))
    df=tbl.to_pandas()
    df.columns=[c.strip() for c in df.columns]
    return df

def make_cells_x_gene(expr:pd.DataFrame,meta:pd.DataFrame,*,um_per_px:float=0.18)->pd.DataFrame:
    e=expr; m=meta
    if not("fov" in e.columns and "cell_ID" in e.columns):
        raise ValueError(f"expr needs fov + cell_ID, got {list(e.columns)[:6]}")
    if not("fov" in m.columns and "cell_ID" in m.columns):
        raise ValueError(f"meta needs fov + cell_ID, got {list(m.columns)[:6]}")
    if not("x_px" in m.columns and "y_px" in m.columns):
        raise ValueError(f"meta missing x_px/y_px; got {list(m.columns)[:10]}")
    gene_cols=[c for c in e.columns if c not in("fov","cell_ID")]
    e2=e[["fov","cell_ID"]+gene_cols].copy()
    e2["_k"]=e2["fov"].astype(str)+"_"+e2["cell_ID"].astype(str)
    m2=m[["fov","cell_ID","x_px","y_px"]].drop_duplicates(["fov","cell_ID"]).copy()
    m2["_k"]=m2["fov"].astype(str)+"_"+m2["cell_ID"].astype(str)
    df=e2.merge(m2[["_k","x_px","y_px"]],on="_k",how="inner")
    df["x"]=df["x_px"].astype(np.float64)*um_per_px
    df["y"]=df["y_px"].astype(np.float64)*um_per_px
    df["cell_id"]=np.arange(1,len(df)+1)
    return df[["cell_id","x","y","fov"]+gene_cols].reset_index(drop=True)
