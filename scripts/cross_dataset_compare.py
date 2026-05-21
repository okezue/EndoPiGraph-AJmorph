from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

def _load(run_dir):
    p=Path(run_dir)
    s=json.load(open(p/"pilot_summary.json"))
    et=pd.read_parquet(p/"edge_type_summary.parquet")
    ct=pd.read_parquet(p/"cell_types.parquet")
    return s,et,ct

def _normalize_etype(s):
    parts=sorted(s.split("__"))
    return "__".join(parts)

def _comp_vector(et,top=None):
    v=et.set_index("edge_type")["n_edges"].copy()
    v.index=[_normalize_etype(x) for x in v.index]
    v=v.groupby(level=0).sum()
    if top is not None:
        v=v.nlargest(top)
    return v

def _align(vecs):
    keys=sorted(set().union(*[v.index for v in vecs]))
    out=pd.DataFrame(0.0,index=keys,columns=list(range(len(vecs))))
    for i,v in enumerate(vecs):
        out.loc[v.index,i]=v.values
    return out

def js_matrix(vecs,labels):
    A=_align(vecs).values.astype(np.float64)
    A=A/np.clip(A.sum(axis=0,keepdims=True),1e-9,None)
    n=A.shape[1]
    M=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            M[i,j]=float(jensenshannon(A[:,i],A[:,j],base=2))
    return pd.DataFrame(M,index=labels,columns=labels)

def lineage_table(cts,labels):
    rows=[]
    for ct,lab in zip(cts,labels):
        v=ct["lineage"].value_counts(normalize=True)
        rows.append(v.rename(lab))
    return pd.concat(rows,axis=1).fillna(0.0)

def top_unique_edges(vecs,labels,k=10):
    A=_align(vecs).copy()
    A.columns=labels
    Anorm=A.div(A.sum(axis=0)+1e-9)
    out={}
    for c in labels:
        others=[x for x in labels if x!=c]
        delta=Anorm[c]-Anorm[others].mean(axis=1)
        top=delta.nlargest(k)
        out[c]=top.to_dict()
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--runs",nargs="+",required=True,help="dataset_label=path pairs")
    ap.add_argument("--out",default="runs/cross_dataset")
    ap.add_argument("--top",type=int,default=50)
    args=ap.parse_args()
    outd=Path(args.out); outd.mkdir(parents=True,exist_ok=True)
    labels=[]; vecs=[]; cts=[]; summs={}
    for spec in args.runs:
        if "=" not in spec: raise SystemExit(f"need label=path, got {spec}")
        lab,pth=spec.split("=",1)
        s,et,ct=_load(pth)
        labels.append(lab); vecs.append(_comp_vector(et,top=args.top)); cts.append(ct)
        summs[lab]={"cells":s.get("n_cells_used"),"transcripts":s.get("n_transcripts","-"),
                    "edges":s.get("n_edges"),"edge_types":s.get("n_edge_types"),
                    "ec_cells":s.get("n_ec_cells","-"),"ec_edges":s.get("n_ec_edges","-"),
                    "species":(ct["species"].iloc[0] if "species" in ct.columns and len(ct) else "?")}
    js=js_matrix(vecs,labels)
    lin=lineage_table(cts,labels)
    uniq=top_unique_edges(vecs,labels,k=15)
    js.to_csv(outd/"edge_type_js_distance.csv")
    lin.to_csv(outd/"lineage_fractions.csv")
    Path(outd/"cross_summary.json").write_text(json.dumps({
        "datasets":summs,"unique_edge_types":uniq,
        "js_pairs":{f"{a}__{b}":float(js.loc[a,b]) for a in labels for b in labels if a<b},
    },indent=2,default=str))
    print("=== datasets ===")
    print(pd.DataFrame(summs).T)
    print("\n=== Jensen-Shannon distance between edge-type compositions ===")
    print(js.round(3))
    print("\n=== Lineage fractions (rows=lineage, cols=dataset) ===")
    print(lin.round(3))
    print("\n=== Top edge types unique to each dataset (Δ vs mean of others) ===")
    for k,v in uniq.items():
        print(f"\n  {k}:")
        for et,d in list(v.items())[:8]:
            print(f"    {et:48s}  Δ={d:+.4f}")

if __name__=="__main__":
    main()
