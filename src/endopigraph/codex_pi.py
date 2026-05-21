from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import tifffile
from scipy.sparse import coo_matrix
from skimage.morphology import binary_dilation, disk
from .interfaces import compute_interfaces, local_interface_mask

CODEX_LINEAGES={
    "tcell":["CD3","CD3D","CD3E","CD3G","CD4","CD8","CD8A","CD8B","CD45RO","CD45RA","TRAC","FOXP3","ICOS"],
    "bcell":["CD20","CD19","CD79A","MS4A1","CD22","CD138"],
    "myeloid":["CD68","CD163","CD14","CD11b","CD11c","CD16","HLA-DR","HLADR","CSF1R","CD15","CD66b","CD33","CD206","Iba1","AIF1"],
    "nk":["CD56","NKG7","KLRD1","NCAM1"],
    "endothelial":["CD31","PECAM1","PECAM","CDH5","VE-Cadherin","VECadherin","CD34","VWF","ERG","ENG","CD105"],
    "epithelial":["EPCAM","CDH1","ECadherin","Cytokeratin","PanCK","PanCytokeratin","CK","CK7","CK8","CK14","CK18","CK19","CK20","KRT5","KRT7","KRT8","KRT14","KRT17","KRT18","KRT19","KRT20"],
    "fibroblast":["Vimentin","Podoplanin","PDPN","FAP","S100A4","ColIa1","COL1A1","CD90","Thy1"],
    "mural_smc":["aSMA","SMA","ACTA2","MyHC","Desmin","DES","RGS5","PDGFRB"],
    "neural":["MAP2","NeuN","Synaptophysin","SYP","GFAP","NSE","Tuj1","NEFL","NEUN"],
    "tumor_prolif":["Ki67","MKI67","Cyclin","CyclinD1","PCNA","TOP2A","MCM2"],
}

def _norm(s):
    return str(s).lower().replace("-","").replace("_","").replace(" ","").replace(".","").replace("/","")

def _match_marker(col,patterns):
    cl=_norm(col)
    for p in patterns:
        if cl==_norm(p): return True
    return False

def type_cells_from_markers(cell_x_marker:pd.DataFrame,*,id_col="cell_id",
                            min_fold:float=1.5,min_z:float=1.0)->pd.DataFrame:
    df=cell_x_marker
    cols=[c for c in df.columns if c!=id_col]
    scores={}
    for lin,pats in CODEX_LINEAGES.items():
        matched=[c for c in cols if _match_marker(c,pats)]
        if not matched: continue
        vals=df[matched].astype(np.float32).values
        s=vals.mean(axis=1)
        z=(s-s.mean())/max(s.std(),1e-6)
        scores[lin]={"matched":matched,"z":z,"raw":s}
    if not scores:
        return pd.DataFrame({id_col:df[id_col] if id_col in df else pd.RangeIndex(len(df)),
                             "lineage":"unassigned"})
    S=pd.DataFrame({k:v["z"] for k,v in scores.items()})
    top=S.idxmax(axis=1).values
    top_score=S.max(axis=1).values
    S2=S.values.copy()
    ci={k:i for i,k in enumerate(S.columns)}
    for i,k in enumerate(top): S2[i,ci[k]]=-np.inf
    fold=top_score-S2.max(axis=1)
    assigned=(top_score>=min_z)&(fold>=np.log(min_fold))
    lab=np.where(assigned,top,"unassigned")
    out=pd.DataFrame({id_col:df[id_col] if id_col in df else pd.RangeIndex(len(df)),
                      "lineage":lab,"lineage_z":top_score.astype(np.float32)})
    return out

def interface_marker_features_multichannel(arr:np.ndarray,labels:np.ndarray,*,
    channel_names:List[str],dilate_px:int=2,min_contact_px:int=10,
    thresholds:Optional[Dict[str,float]]=None)->Tuple[pd.DataFrame,object]:
    if arr.ndim==2: arr=arr[None,...]
    C,H,W=arr.shape
    assert labels.shape==(H,W),f"labels {labels.shape} != image {(H,W)}"
    iface=compute_interfaces(labels,min_contact_px=min_contact_px)
    eds=iface.edges.reset_index(drop=True)
    if len(eds)==0:
        return pd.DataFrame(),iface
    thresholds=thresholds or {}
    mean_M=np.zeros((len(eds),C),dtype=np.float32)
    occ_M=np.zeros((len(eds),C),dtype=np.float32)
    contact_lens=np.zeros(len(eds),dtype=np.int32)
    for ei,r in eds.iterrows():
        i,j=int(r.cell_i),int(r.cell_j)
        co=iface.boundary_coords[(min(i,j),max(i,j))]
        if co.size==0: continue
        bb,ml=local_interface_mask(co,dilate_px,(H,W))
        if ml.size==0: continue
        ml=binary_dilation(ml,footprint=disk(dilate_px))
        r0,r1,c0,c1=bb
        yy,xx=np.where(ml)
        if yy.size==0: continue
        gy=yy+r0; gx=xx+c0
        contact_lens[ei]=int(len(co))
        for ci,name in enumerate(channel_names):
            patch=arr[ci,gy,gx].astype(np.float32)
            mean_M[ei,ci]=float(patch.mean()) if patch.size else 0.0
            thr=thresholds.get(name)
            if thr is not None:
                occ_M[ei,ci]=float((patch>=thr).mean())
    out=pd.DataFrame({"cell_i":eds["cell_i"].values,"cell_j":eds["cell_j"].values,
                      "contact_px":eds["contact_px"].values})
    for ci,name in enumerate(channel_names):
        out[f"{name}_mean"]=mean_M[:,ci]
        if any(name in thresholds for name in channel_names if thresholds.get(name) is not None):
            out[f"{name}_occ"]=occ_M[:,ci]
    return out,iface

def load_codex_image(p:str|Path)->Tuple[np.ndarray,List[str]]:
    p=Path(p)
    arr=tifffile.imread(p)
    if arr.ndim==4 and arr.shape[0]==1: arr=arr[0]
    if arr.ndim==4: arr=arr.max(axis=0)
    if arr.ndim==2: arr=arr[None,...]
    try:
        with tifffile.TiffFile(p) as tf:
            ome=tf.ome_metadata or ""
        import re
        names=re.findall(r'Name="([^"]+)"',ome)
        if len(names)==arr.shape[0]: return arr,names
    except Exception: pass
    return arr,[f"ch{i}" for i in range(arr.shape[0])]
