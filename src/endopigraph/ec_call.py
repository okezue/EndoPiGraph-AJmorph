from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

EC_HUMAN=["PECAM1","CDH5","VWF","KDR","CLDN5","FLT1","TIE1","ESAM","CD34","ENG"]
EC_MOUSE=["Pecam1","Cdh5","Vwf","Kdr","Cldn5","Flt1","Tie1","Esam","Cd34","Eng"]
LYMPH_HUMAN=["PROX1","LYVE1","PDPN","FLT4"]
LYMPH_MOUSE=["Prox1","Lyve1","Pdpn","Flt4"]

def _resolve_markers(cols,species="auto"):
    cs=set(map(str,cols))
    h=sum(1 for m in EC_HUMAN if m in cs)
    m=sum(1 for m in EC_MOUSE if m in cs)
    sp="mouse" if (species=="mouse" or (species=="auto" and m>h)) else "human"
    ec=EC_HUMAN if sp=="human" else EC_MOUSE
    ly=LYMPH_HUMAN if sp=="human" else LYMPH_MOUSE
    return [g for g in ec if g in cs],[g for g in ly if g in cs],sp

def call_ec(cell_x_gene:pd.DataFrame,*,species="auto",min_markers=3,
            post_threshold=0.85,min_total_counts=5,random_state=0):
    df=cell_x_gene
    gcols=[c for c in df.columns if c!="cell_id"]
    ec,ly,sp=_resolve_markers(gcols,species=species)
    if len(ec)<min_markers:
        return pd.DataFrame({"cell_id":df.get("cell_id",pd.RangeIndex(len(df))),
                             "is_ec":False,"is_lymph":False,"ec_score":0.0,
                             "n_ec_markers":len(ec),"species":sp,"sep":0.0})
    sub=df[ec].astype(np.float32)
    raw=sub.sum(axis=1).values
    s=np.log1p(raw).reshape(-1,1)
    sep=0.0; is_ec=np.zeros(len(df),dtype=bool); score=np.zeros(len(df),dtype=np.float32)
    if np.unique(s).size>=2:
        g=GaussianMixture(n_components=2,random_state=random_state,covariance_type="full").fit(s)
        mu=g.means_.ravel()
        hi=int(np.argmax(mu))
        sep=float(mu[hi]-mu[1-hi])
        p=g.predict_proba(s)
        is_ec=(p[:,hi]>=post_threshold)&(raw>=min_total_counts)
        score=p[:,hi].astype(np.float32)
    is_lymph=np.zeros(len(df),dtype=bool)
    if ly:
        ls=np.log1p(df[ly].astype(np.float32).sum(axis=1).values)
        is_lymph=is_ec&(ls>=np.quantile(ls[is_ec],0.5) if is_ec.any() else False)
    return pd.DataFrame({
        "cell_id":df.get("cell_id",pd.RangeIndex(len(df))).values,
        "is_ec":is_ec,"is_lymph":is_lymph,"ec_score":score,
        "n_ec_markers":len(ec),"species":sp,"sep":sep,
    })

def filter_edges_to_ec(edge_df:pd.DataFrame,ec_calls:pd.DataFrame)->pd.DataFrame:
    ec_set=set(ec_calls.loc[ec_calls["is_ec"],"cell_id"].astype(int).tolist())
    m=edge_df["cell_i"].isin(ec_set)&edge_df["cell_j"].isin(ec_set)
    return edge_df[m].reset_index(drop=True)
