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

def load_cosmx_expr(p,*,index_cols=("cell_ID","fov","cell"),max_rows=None,max_fovs=None
                    )->pd.DataFrame:
    import pyarrow as pa
    import pyarrow.csv as pcsv
    import gzip
    op=gzip.open(str(p),"rt") if str(p).endswith(".gz") else open(str(p),"rt")
    with op as f: header=[h.strip() for h in f.readline().rstrip("\n").split(",")]
    id_cols=set(index_cols)|{"CellId","CellID"}
    types={h:(pa.int32() if h not in id_cols else pa.int64()) for h in header}
    reader=pcsv.open_csv(str(p),
        read_options=pcsv.ReadOptions(use_threads=True,block_size=1<<26),
        convert_options=pcsv.ConvertOptions(column_types=types))
    batches=[]; total=0; fovs=set()
    for batch in reader:
        if max_fovs is not None and "fov" in batch.schema.names:
            fov_arr=batch.column("fov").to_numpy()
            mask=np.array([f in fovs or len(fovs)<max_fovs for f in fov_arr])
            for f in fov_arr[:1000]:
                if len(fovs)<max_fovs: fovs.add(int(f))
            if not mask.any(): continue
            batch=batch.filter(pa.array(mask))
        batches.append(batch); total+=batch.num_rows
        if max_rows is not None and total>=max_rows: break
    tbl=pa.Table.from_batches(batches)
    if max_rows is not None and tbl.num_rows>max_rows: tbl=tbl.slice(0,max_rows)
    df=tbl.to_pandas(self_destruct=True,split_blocks=True)
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
    m2=m[["fov","cell_ID","x_px","y_px"]].drop_duplicates(["fov","cell_ID"]).copy()
    m2["_k"]=m2["fov"].astype(np.int64)*100000+m2["cell_ID"].astype(np.int64)
    lookup=dict(zip(m2["_k"].values,zip(m2["x_px"].values,m2["y_px"].values)))
    keys=e["fov"].astype(np.int64).values*100000+e["cell_ID"].astype(np.int64).values
    xs=np.empty(len(keys),dtype=np.float64); ys=np.empty(len(keys),dtype=np.float64)
    mask=np.ones(len(keys),dtype=bool)
    for i,k in enumerate(keys):
        v=lookup.get(int(k))
        if v is None: mask[i]=False; continue
        xs[i],ys[i]=v
    if not mask.all():
        e=e[mask].reset_index(drop=True); xs=xs[mask]; ys=ys[mask]
    e["x"]=xs*um_per_px
    e["y"]=ys*um_per_px
    e["cell_id"]=np.arange(1,len(e)+1)
    gene_cols=[c for c in e.columns if c not in("fov","cell_ID","x","y","cell_id")]
    return e[["cell_id","x","y","fov"]+gene_cols].reset_index(drop=True)
