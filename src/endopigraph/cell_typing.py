from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

LINEAGES_HUMAN={
    "endothelial":["PECAM1","CDH5","VWF","KDR","CLDN5","FLT1","TIE1","ESAM","CD34","ENG"],
    "epithelial":["EPCAM","KRT8","KRT18","KRT19","CDH1","KRT7"],
    "basal_myoep":["KRT5","KRT14","KRT17","MYH11","TP63","ACTA2"],
    "fibroblast":["COL1A1","COL1A2","COL3A1","DCN","LUM","PDGFRA","FAP","S100A4"],
    "mural_smc":["ACTA2","MYH11","TAGLN","CNN1","DES","RGS5","PDGFRB"],
    "tcell":["CD3D","CD3E","CD3G","CD8A","CD8B","CD4","TRAC"],
    "bcell":["CD19","CD79A","CD79B","MS4A1","JCHAIN","IGHM","IGHG1"],
    "myeloid":["CD68","CD163","AIF1","LYZ","CSF1R","ITGAX","C1QA","C1QB","CD14"],
    "mast":["TPSAB1","TPSB2","KIT","CPA3"],
    "nk":["NKG7","KLRD1","GNLY","NCAM1","PRF1"],
    "neural":["MAP2","RBFOX3","SYN1","NEFL","SLC17A7","GAD1","GAD2","TH","CHAT"],
    "glia":["GFAP","AQP4","S100B","MBP","PLP1","MOG","OLIG1","OLIG2"],
    "hepatocyte":["ALB","TTR","APOA1","APOA2","HNF4A","CYP3A4"],
    "tumor_breast":["ESR1","ERBB2","MKI67","FOXA1","GATA3","TFF1","TFF3"],
    "proliferating":["MKI67","TOP2A","PCNA","CCNB1","CDK1","BIRC5","UBE2C"],
}

def _mouse(g): return g[0]+g[1:].lower()
LINEAGES_MOUSE={k:[_mouse(g) for g in v] for k,v in LINEAGES_HUMAN.items()}

def _detect_species(cols):
    cs=set(map(str,cols))
    h=sum(1 for v in LINEAGES_HUMAN.values() for g in v if g in cs)
    m=sum(1 for v in LINEAGES_MOUSE.values() for g in v if g in cs)
    return "mouse" if m>h else "human"

def score_lineages(cell_x_gene:pd.DataFrame,*,species="auto",
                   min_total=3,min_fold=1.5,
                   panels:Optional[Dict[str,List[str]]]=None)->pd.DataFrame:
    df=cell_x_gene
    has_cid="cell_id" in df.columns
    cid=df["cell_id"].values if has_cid else np.arange(len(df))
    gcols=[c for c in df.columns if c!="cell_id"]
    cs=set(gcols)
    if panels is None:
        sp=species if species!="auto" else _detect_species(gcols)
        panels=LINEAGES_HUMAN if sp=="human" else LINEAGES_MOUSE
    else:
        sp=species if species!="auto" else "custom"
    avail={k:[g for g in v if g in cs] for k,v in panels.items()}
    avail={k:v for k,v in avail.items() if len(v)>=1}
    if not avail:
        return pd.DataFrame({"cell_id":cid,"lineage":"unassigned",
                             "lineage_score":0.0,"lineage_runnerup":"unassigned",
                             "species":sp})
    Mtot=df[gcols].sum(axis=1).values.astype(np.float32)
    Mtot=np.clip(Mtot,1.0,None)
    scores={}
    for k,gs in avail.items():
        s=df[gs].sum(axis=1).values.astype(np.float32)/len(gs)
        scores[k]=np.log1p(s)-np.log1p(Mtot/(2*max(len(gs),1)))
    S=pd.DataFrame(scores)
    counts={k:df[avail[k]].sum(axis=1).values for k in avail}
    Cdf=pd.DataFrame(counts)
    top=S.idxmax(axis=1).values
    top_score=S.max(axis=1).values
    S2=S.values.copy()
    col_idx={k:i for i,k in enumerate(S.columns)}
    for i,k in enumerate(top): S2[i,col_idx[k]]=-np.inf
    runner=np.array([S.columns[j] for j in S2.argmax(axis=1)])
    runner_score=S2.max(axis=1)
    top_count=np.array([Cdf.iloc[i][k] for i,k in enumerate(top)])
    fold=top_score-runner_score
    assigned=(top_count>=min_total)&(fold>=np.log(min_fold))
    lab=np.where(assigned,top,"unassigned")
    return pd.DataFrame({
        "cell_id":cid,"lineage":lab,"lineage_score":top_score.astype(np.float32),
        "lineage_runnerup":runner,"runnerup_score":runner_score.astype(np.float32),
        "n_top_markers":top_count.astype(np.int32),"species":sp,
    })

def add_edge_types(edge_df:pd.DataFrame,cell_types:pd.DataFrame,
                   *,type_col="lineage")->pd.DataFrame:
    m=dict(zip(cell_types["cell_id"].astype(int),cell_types[type_col].astype(str)))
    ti=edge_df["cell_i"].map(lambda x:m.get(int(x),"unassigned"))
    tj=edge_df["cell_j"].map(lambda x:m.get(int(x),"unassigned"))
    a=np.minimum(ti,tj); b=np.maximum(ti,tj)
    out=edge_df.copy()
    out["type_i"]=a.values; out["type_j"]=b.values
    out["edge_type"]=a.values+"__"+b.values
    return out

def edge_type_summary(typed_edges:pd.DataFrame)->pd.DataFrame:
    gene_cols=[c for c in typed_edges.columns if c not in
               ("cell_i","cell_j","contact_px","type_i","type_j","edge_type")]
    g=typed_edges.groupby("edge_type")
    n=g.size().rename("n_edges")
    contact=g["contact_px"].sum().rename("total_contact_px") if "contact_px" in typed_edges.columns else None
    mean_expr=g[gene_cols].mean()
    out=mean_expr.copy()
    out.insert(0,"n_edges",n)
    if contact is not None: out.insert(1,"total_contact_px",contact)
    return out.reset_index()
