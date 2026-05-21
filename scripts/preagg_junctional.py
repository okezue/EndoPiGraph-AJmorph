from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

META_COLS={"cell_i","cell_j","contact_len_um","contact_px","type_i","type_j","edge_type","brain_section_label","fov"}

def aggregate(parq_path:Path,out_path:Path,batch_size:int=100_000):
    pf=pq.ParquetFile(parq_path)
    schema=pf.schema_arrow
    cols=pf.schema.names
    import pyarrow as pa
    gene_cols=[]
    for c in cols:
        if c in META_COLS: continue
        t=schema.field(c).type
        if pa.types.is_integer(t) or pa.types.is_floating(t): gene_cols.append(c)
    n_genes=len(gene_cols)
    s=np.zeros(n_genes,dtype=np.float64)
    s2=np.zeros(n_genes,dtype=np.float64)
    nz=np.zeros(n_genes,dtype=np.int64)
    n_total=0
    for batch in pf.iter_batches(batch_size=batch_size,columns=gene_cols):
        A=batch.to_pandas().values.astype(np.float64)
        s+=A.sum(axis=0); s2+=(A*A).sum(axis=0); nz+=(A>0).sum(axis=0); n_total+=len(A)
    mean=s/max(n_total,1)
    var=s2/max(n_total,1)-mean**2
    var=np.clip(var,0,None)
    sd=np.sqrt(var)
    import pandas as pd
    df=pd.DataFrame({"gene":gene_cols,"mean":mean,"sd":sd,"frac_nonzero":nz/max(n_total,1),"n_edges":n_total})
    df.to_csv(out_path,index=False)
    print(f"{parq_path} -> {out_path}: n_edges={n_total}, n_genes={n_genes}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--runs",nargs="+",required=True,help="label=path entries")
    ap.add_argument("--out-dir",required=True)
    args=ap.parse_args()
    od=Path(args.out_dir); od.mkdir(parents=True,exist_ok=True)
    for spec in args.runs:
        lab,p=spec.split("=",1)
        for fname in("edge_junctional_expression.parquet","edges_x_gene.parquet","edges_typed.parquet"):
            pq_path=Path(p)/fname
            if pq_path.exists(): break
        else:
            print(f"  skip {lab}: no gene-level edge parquet"); continue
        aggregate(pq_path,od/f"{lab}_gene_signal.csv")

if __name__=="__main__":
    main()
