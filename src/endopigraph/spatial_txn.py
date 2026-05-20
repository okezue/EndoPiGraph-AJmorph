from __future__ import annotations
from pathlib import Path
from typing import Tuple, Dict
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from skimage.draw import polygon as _dp
from skimage.morphology import binary_dilation, disk
from .interfaces import compute_interfaces, local_interface_mask

XENIUM_TX_RENAME={"x_location":"x","y_location":"y","feature_name":"gene","target":"gene"}
MERSCOPE_TX_RENAME={"global_x":"x","global_y":"y","gene":"gene"}

def _read_any(p):
    s=str(p)
    if s.endswith(".parquet"): return pd.read_parquet(p)
    return pd.read_csv(p)

def load_transcripts(p,qv_min=20.0,platform="auto"):
    df=_read_any(p)
    cols=df.columns
    if platform=="auto":
        platform="xenium" if ("x_location" in cols or "feature_name" in cols) else "merscope"
    rn=XENIUM_TX_RENAME if platform=="xenium" else MERSCOPE_TX_RENAME
    df=df.rename(columns={k:v for k,v in rn.items() if k in cols})
    if "qv" in df.columns and qv_min is not None: df=df[df["qv"]>=qv_min]
    if "gene" in df.columns:
        df=df.assign(gene=df["gene"].astype(str))
        df=df[df["gene"].str.match(r"^[A-Za-z][\w\-\.]*$")]
    return df[["x","y","gene"]].copy()

def load_polygons(p)->Tuple[Dict[int,np.ndarray],Dict[int,str]]:
    df=_read_any(p)
    cid_col="cell_id" if "cell_id" in df.columns else df.columns[0]
    cat=df[cid_col].astype("category")
    df=df.assign(_cid=cat.cat.codes.astype(np.int32)+1)
    polys={}
    for c,g in df.groupby("_cid",sort=False):
        polys[int(c)]=g[["vertex_x","vertex_y"]].values.astype(np.float32)
    name_map=dict(zip(range(1,len(cat.cat.categories)+1),[str(x) for x in cat.cat.categories]))
    return polys,name_map

def rasterize_polygons(polys,shape,pxsz=1.0):
    L=np.zeros(shape,dtype=np.int32)
    H,W=shape
    for cid,v in polys.items():
        r=np.clip((v[:,1]/pxsz).astype(np.int32),0,H-1)
        c=np.clip((v[:,0]/pxsz).astype(np.int32),0,W-1)
        rr,cc=_dp(r,c,shape=shape)
        L[rr,cc]=cid
    return L

def shape_from_polygons(polys,pxsz=1.0,pad=8):
    mx=my=0.0
    for v in polys.values():
        if v.size==0: continue
        mx=max(mx,v[:,0].max()); my=max(my,v[:,1].max())
    return (int(my/pxsz)+pad,int(mx/pxsz)+pad)

def _sparse_tx_grid(tx,shape,pxsz):
    H,W=shape
    px=np.floor(tx["x"].values/pxsz).astype(np.int64)
    py=np.floor(tx["y"].values/pxsz).astype(np.int64)
    ok=(px>=0)&(px<W)&(py>=0)&(py<H)
    px,py=px[ok],py[ok]
    g=tx["gene"].values[ok]
    gidx=pd.Index(np.unique(g))
    gmap={x:i for i,x in enumerate(gidx)}
    gc=np.fromiter((gmap[x] for x in g),dtype=np.int32,count=len(g))
    flat=py*W+px
    S=coo_matrix((np.ones(len(gc),dtype=np.int32),(flat,gc)),shape=(H*W,len(gidx))).tocsr()
    return S,gidx

def junctional_counts(labels,tx,*,pxsz=0.2125,dilate_px=3,min_contact_px=10,return_iface=False):
    H,W=labels.shape
    S,gidx=_sparse_tx_grid(tx,(H,W),pxsz)
    iface=compute_interfaces(labels,min_contact_px=min_contact_px)
    eds=iface.edges.reset_index(drop=True)
    M=np.zeros((len(eds),len(gidx)),dtype=np.int32)
    for ei,r in eds.iterrows():
        i,j=int(r.cell_i),int(r.cell_j)
        co=iface.boundary_coords[(min(i,j),max(i,j))]
        if co.size==0: continue
        bb,ml=local_interface_mask(co,dilate_px,(H,W))
        r0,r1,c0,c1=bb
        if ml.size==0: continue
        ml=binary_dilation(ml,footprint=disk(dilate_px))
        yy,xx=np.where(ml)
        if yy.size==0: continue
        pidx=(yy+r0)*W+(xx+c0)
        v=np.asarray(S[pidx,:].sum(axis=0)).ravel()
        M[ei]=v
    out=pd.DataFrame(M,columns=gidx.tolist())
    out.insert(0,"cell_i",eds["cell_i"].values)
    out.insert(1,"cell_j",eds["cell_j"].values)
    out.insert(2,"contact_px",eds["contact_px"].values)
    if return_iface: return out,iface
    return out

def cell_x_gene_from_transcripts(labels,tx,*,pxsz=0.2125):
    H,W=labels.shape
    px=np.floor(tx["x"].values/pxsz).astype(np.int64)
    py=np.floor(tx["y"].values/pxsz).astype(np.int64)
    ok=(px>=0)&(px<W)&(py>=0)&(py<H)
    px,py=px[ok],py[ok]
    g=tx["gene"].values[ok]
    cid=labels[py,px]
    keep=cid>0
    cid,g=cid[keep],g[keep]
    gidx=pd.Index(np.unique(g))
    gmap={x:i for i,x in enumerate(gidx)}
    gc=np.fromiter((gmap[x] for x in g),dtype=np.int32,count=len(g))
    cids=np.unique(cid)
    cmap={c:i for i,c in enumerate(cids)}
    cc=np.fromiter((cmap[c] for c in cid),dtype=np.int32,count=len(cid))
    M=coo_matrix((np.ones(len(cc),dtype=np.int32),(cc,gc)),shape=(len(cids),len(gidx))).toarray()
    df=pd.DataFrame(M,columns=gidx.tolist())
    df.insert(0,"cell_id",cids)
    return df
