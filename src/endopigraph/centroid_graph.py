from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, Voronoi, cKDTree

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
    if method=="voronoi":
        return build_voronoi_graph(cells,x_col=x_col,y_col=y_col,section_col=section_col,max_edge_um=max_edge_um)
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
    out=[n]
    if "dist_um" in typed.columns: out.append(g["dist_um"].median().rename("median_dist_um"))
    if "contact_len_um" in typed.columns: out.append(g["contact_len_um"].sum().rename("total_contact_len_um"))
    return pd.concat(out,axis=1).reset_index()

def voronoi_edges(xy:np.ndarray,max_edge_um:Optional[float]=None)->pd.DataFrame:
    if len(xy)<4: return pd.DataFrame(columns=["cell_i","cell_j","dist_um","contact_len_um"])
    vor=Voronoi(xy)
    V=vor.vertices
    rows=[]
    for (pi,pj),(v1,v2) in zip(vor.ridge_points,vor.ridge_vertices):
        a,b=min(int(pi),int(pj)),max(int(pi),int(pj))
        d=float(np.linalg.norm(xy[a]-xy[b]))
        if max_edge_um is not None and d>max_edge_um: continue
        if v1>=0 and v2>=0:
            L=float(np.linalg.norm(V[v1]-V[v2]))
        else:
            L=np.nan
        rows.append((a,b,d,L))
    out=pd.DataFrame(rows,columns=["cell_i","cell_j","dist_um","contact_len_um"])
    return out.drop_duplicates(["cell_i","cell_j"]).reset_index(drop=True)

def build_voronoi_graph(cells:pd.DataFrame,*,x_col="x",y_col="y",
                        section_col:Optional[str]=None,
                        max_edge_um:Optional[float]=30.0,
                        impute_unbounded:bool=True)->pd.DataFrame:
    def _one(g):
        xy=g[[x_col,y_col]].values.astype(np.float64)
        e=voronoi_edges(xy,max_edge_um=max_edge_um)
        if len(e)==0: return e
        if impute_unbounded and e["contact_len_um"].isna().any():
            med=float(np.nanmedian(e["contact_len_um"])) if e["contact_len_um"].notna().any() else 1.0
            e["contact_len_um"]=e["contact_len_um"].fillna(med)
        return e
    if section_col is not None and section_col in cells.columns:
        outs=[]
        for sec,g in cells.groupby(section_col,sort=False):
            e=_one(g)
            if len(e)==0: continue
            local_ids=g.index.values
            e["cell_i"]=local_ids[e["cell_i"].values]
            e["cell_j"]=local_ids[e["cell_j"].values]
            e[section_col]=sec
            outs.append(e)
        return pd.concat(outs,ignore_index=True) if outs else pd.DataFrame(columns=["cell_i","cell_j","dist_um","contact_len_um",section_col])
    return _one(cells.reset_index(drop=True))

def approx_junctional_expression(edges:pd.DataFrame,cell_x_gene:pd.DataFrame,*,
                                 mode:str="min",weight_by_contact:bool=True,
                                 gene_cols:Optional[list]=None)->pd.DataFrame:
    cxg=cell_x_gene.set_index("cell_id") if "cell_id" in cell_x_gene.columns else cell_x_gene
    if gene_cols is None:
        gene_cols=[c for c in cxg.columns if c not in("x","y","class","lineage","cell_id","brain_section_label")]
    G=cxg[gene_cols].values.astype(np.float32)
    ei=G[edges["cell_i"].values]
    ej=G[edges["cell_j"].values]
    if mode=="min": E=np.minimum(ei,ej)
    elif mode=="geometric": E=np.sqrt(np.clip(ei,0,None)*np.clip(ej,0,None))
    elif mode=="product": E=ei*ej
    elif mode=="sum": E=(ei+ej)*0.5
    else: raise ValueError(f"unknown mode {mode}")
    if weight_by_contact and "contact_len_um" in edges.columns:
        L=edges["contact_len_um"].values.astype(np.float32).reshape(-1,1)
        L=np.where(np.isfinite(L),L,1.0)
        E=E*L
    out=pd.DataFrame(E,columns=gene_cols)
    out.insert(0,"cell_i",edges["cell_i"].values)
    out.insert(1,"cell_j",edges["cell_j"].values)
    if "contact_len_um" in edges.columns:
        out.insert(2,"contact_len_um",edges["contact_len_um"].values)
    return out
