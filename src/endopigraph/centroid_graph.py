from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, cKDTree

def delaunay_edges(xy:np.ndarray,max_edge_um:Optional[float]=None)->pd.DataFrame:
    if len(xy)<3:
        return pd.DataFrame(columns=["cell_i","cell_j","dist_um"])
    tri=Delaunay(xy)
    pairs=set()
    for s in tri.simplices:
        for a,b in ((s[0],s[1]),(s[1],s[2]),(s[0],s[2])):
            pairs.add((min(int(a),int(b)),max(int(a),int(b))))
    ij=np.array(sorted(pairs),dtype=np.int64)
    if ij.size==0: return pd.DataFrame(columns=["cell_i","cell_j","dist_um"])
    d=np.linalg.norm(xy[ij[:,0]]-xy[ij[:,1]],axis=1)
    out=pd.DataFrame({"cell_i":ij[:,0],"cell_j":ij[:,1],"dist_um":d})
    if max_edge_um is not None: out=out[out["dist_um"]<=max_edge_um].reset_index(drop=True)
    return out

def knn_edges(xy:np.ndarray,k:int=8,max_edge_um:Optional[float]=None)->pd.DataFrame:
    if len(xy)<2: return pd.DataFrame(columns=["cell_i","cell_j","dist_um"])
    tree=cKDTree(xy)
    d,idx=tree.query(xy,k=k+1)
    rows=[]
    for i in range(len(xy)):
        for j,dij in zip(idx[i,1:],d[i,1:]):
            a,b=int(i),int(j)
            if a==b: continue
            if max_edge_um is not None and dij>max_edge_um: continue
            rows.append((min(a,b),max(a,b),float(dij)))
    out=pd.DataFrame(rows,columns=["cell_i","cell_j","dist_um"]).drop_duplicates(["cell_i","cell_j"]).reset_index(drop=True)
    return out

def build_centroid_graph(cells:pd.DataFrame,*,x_col="x",y_col="y",
                         section_col:Optional[str]=None,
                         method:str="delaunay",max_edge_um:Optional[float]=30.0,
                         knn_k:int=8)->pd.DataFrame:
    if section_col is not None and section_col in cells.columns:
        outs=[]
        for sec,g in cells.groupby(section_col,sort=False):
            xy=g[[x_col,y_col]].values.astype(np.float32)
            local_ids=g.index.values
            if method=="delaunay":
                e=delaunay_edges(xy,max_edge_um=max_edge_um)
            else:
                e=knn_edges(xy,k=knn_k,max_edge_um=max_edge_um)
            if len(e)==0: continue
            e["cell_i"]=local_ids[e["cell_i"].values]
            e["cell_j"]=local_ids[e["cell_j"].values]
            e[section_col]=sec
            outs.append(e)
        return pd.concat(outs,ignore_index=True) if outs else pd.DataFrame(columns=["cell_i","cell_j","dist_um",section_col])
    xy=cells[[x_col,y_col]].values.astype(np.float32)
    if method=="delaunay":
        e=delaunay_edges(xy,max_edge_um=max_edge_um)
    else:
        e=knn_edges(xy,k=knn_k,max_edge_um=max_edge_um)
    return e

def edges_with_types(edges:pd.DataFrame,cell_types:pd.Series)->pd.DataFrame:
    ti=cell_types.iloc[edges["cell_i"].values].astype(str).values
    tj=cell_types.iloc[edges["cell_j"].values].astype(str).values
    a=np.minimum(ti,tj); b=np.maximum(ti,tj)
    out=edges.copy()
    out["type_i"]=a; out["type_j"]=b; out["edge_type"]=a+"__"+b
    return out

def edge_type_counts(typed:pd.DataFrame)->pd.DataFrame:
    g=typed.groupby("edge_type")
    n=g.size().rename("n_edges")
    if "dist_um" in typed.columns:
        md=g["dist_um"].median().rename("median_dist_um")
        return pd.concat([n,md],axis=1).reset_index()
    return n.reset_index()
