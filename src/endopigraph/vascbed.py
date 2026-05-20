from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

BED_MARKERS={
    "arterial":["GJA5","HEY1","DLL4","BMX","EFNB2","SOX17","GJA4"],
    "venous":["NR2F2","EPHB4","NRP2","SELP","ACKR1"],
    "capillary":["CA4","RGCC","KDR","CD36","SGK1","SPARC"],
    "lymphatic":["PROX1","LYVE1","PDPN","FLT4","CCL21","TBX1"],
    "tip":["ESM1","APLN","CXCR4","ANGPT2","DLL4","UNC5B"],
}

def _to_mouse(g): return g[0]+g[1:].lower()

def _resolve(cols,species="human"):
    cs=set(cols)
    out={}
    for k,gs in BED_MARKERS.items():
        gs2=gs if species=="human" else [_to_mouse(x) for x in gs]
        out[k]=[g for g in gs2 if g in cs]
    return out

def _pick_k(X,k_min=2,k_max=8,random_state=0):
    best=(np.inf,None,None)
    for k in range(k_min,k_max+1):
        try:
            g=GaussianMixture(n_components=k,covariance_type="diag",random_state=random_state,reg_covar=1e-4).fit(X)
            b=g.bic(X)
            if b<best[0]: best=(b,k,g)
        except Exception:
            continue
    return best

def cluster_edges(edge_gene_df:pd.DataFrame,*,species="human",k_min=2,k_max=8,
                  normalize=True,random_state=0,min_total_counts=1):
    meta_cols=[c for c in("cell_i","cell_j","contact_px") if c in edge_gene_df.columns]
    gene_cols=[c for c in edge_gene_df.columns if c not in meta_cols]
    M=edge_gene_df[gene_cols].values.astype(np.float32)
    tot=M.sum(axis=1)
    keep=tot>=min_total_counts
    Xall=np.log1p(M)
    if normalize:
        Xall=Xall/np.clip(np.linalg.norm(Xall,axis=1,keepdims=True),1e-6,None)
    X=Xall[keep]
    if len(X)<k_min:
        out=edge_gene_df[meta_cols].copy()
        out["bed_cluster"]=-1; out["bed_label"]="unassigned"; out["bed_conf"]=0.0
        return out,{"k":0,"bic":None,"sizes":{}}
    bic,k,gm=_pick_k(X,k_min,k_max,random_state)
    lab=np.full(len(edge_gene_df),-1,dtype=np.int32)
    conf=np.zeros(len(edge_gene_df),dtype=np.float32)
    p=gm.predict_proba(X); l=gm.predict(X)
    lab[np.where(keep)[0]]=l
    conf[np.where(keep)[0]]=p.max(axis=1).astype(np.float32)
    mk=_resolve(gene_cols,species=species)
    cluster_label=_label_clusters(edge_gene_df,gene_cols,lab,mk)
    out=edge_gene_df[meta_cols].copy()
    out["bed_cluster"]=lab
    out["bed_label"]=[cluster_label.get(int(c),"unassigned") for c in lab]
    out["bed_conf"]=conf
    sizes={int(c):int((lab==c).sum()) for c in np.unique(lab)}
    return out,{"k":int(k),"bic":float(bic),"sizes":sizes,"cluster_label":cluster_label}

def _label_clusters(edge_gene_df,gene_cols,labels,bed_to_genes):
    M=edge_gene_df[gene_cols].values.astype(np.float32)
    cls=sorted([int(c) for c in np.unique(labels) if c>=0])
    beds=list(bed_to_genes.keys())
    score=np.full((len(cls),len(beds)),-np.inf,dtype=np.float32)
    for i,c in enumerate(cls):
        rows=M[labels==c]
        if rows.size==0: continue
        means=rows.mean(axis=0)
        for j,bed in enumerate(beds):
            gs=bed_to_genes[bed]
            if not gs: continue
            idx=[gene_cols.index(g) for g in gs if g in gene_cols]
            if not idx: continue
            score[i,j]=float(np.log1p(means[idx]).mean())
    out={}
    used_beds=set()
    order=np.argsort(-score.max(axis=1))
    for i in order:
        c=cls[i]; row=score[i].copy()
        for j,b in enumerate(beds):
            if b in used_beds: row[j]=-np.inf
        if np.isneginf(row).all():
            j=int(np.argmax(score[i])); b=beds[j]; out[c]=f"{b}_subtype"
        else:
            j=int(np.argmax(row)); b=beds[j]
            out[c]=b; used_beds.add(b)
    return out
