from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pyarrow as pa, pyarrow.parquet as pq

META={"cell_i","cell_j","contact_len_um","contact_px","type_i","type_j","edge_type","fov","brain_section_label"}
MARKERS_PROTECT={"PECAM1","CDH5","VWF","KDR","CLDN5","CD31","CDH1","CTNNB1","EPCAM","KRT8","KRT19",
    "ESR1","ERBB2","FOXA1","GATA3","COL1A1","COL3A1","ACTA2","MYH11","CD3D","CD3E","CD8A","CD4",
    "CD19","CD20","MS4A1","CD68","CD163","CD14","CD45","FOXP3","MKI67","KRT5","KRT14",
    "PROX1","LYVE1","PDPN","GJA5","HEY1","DLL4","NR2F2","EPHB4","CA4","RGCC",
    "Slc17a7","Gfap","Aqp4","Nrn1","Epha4","Igf2","Gad1","Gad2","Calb1","Pecam1","Cdh5","Vwf","Pvalb"}

def subset(in_path:Path,out_path:Path,max_edges=50_000,max_genes=2000,seed=0):
    pf=pq.ParquetFile(in_path)
    cols=pf.schema_arrow
    n_rows=pf.metadata.num_rows
    schema_cols=cols.names
    meta_cols=[c for c in schema_cols if c in META]
    all_genes=[c for c in schema_cols if c not in META and (pa.types.is_integer(cols.field(c).type) or pa.types.is_floating(cols.field(c).type))]
    print(f"  {in_path.name}: rows={n_rows:,} meta={meta_cols} genes={len(all_genes)}",flush=True)

    s=np.zeros(len(all_genes),dtype=np.float64)
    s2=np.zeros(len(all_genes),dtype=np.float64)
    n=0
    for batch in pf.iter_batches(batch_size=200_000,columns=all_genes):
        A=batch.to_pandas().values.astype(np.float64)
        s+=A.sum(axis=0); s2+=(A*A).sum(axis=0); n+=len(A)
    mean=s/max(n,1); var=np.clip(s2/max(n,1)-mean**2,0,None)
    order=np.argsort(-var)
    keep_idx=set(order[:max_genes].tolist())
    keep_idx|={i for i,g in enumerate(all_genes) if g in MARKERS_PROTECT}
    gene_keep=[all_genes[i] for i in sorted(keep_idx)]
    print(f"    keeping {len(gene_keep)} of {len(all_genes)} genes (variance + markers)",flush=True)

    rng=np.random.default_rng(seed)
    if n_rows>max_edges:
        idx=np.sort(rng.choice(n_rows,size=max_edges,replace=False))
    else:
        idx=np.arange(n_rows)
    keep_cols=meta_cols+gene_keep
    out_batches=[]
    pos=0
    for batch in pf.iter_batches(batch_size=200_000,columns=keep_cols):
        bsz=batch.num_rows
        lo,hi=pos,pos+bsz
        in_range=idx[(idx>=lo)&(idx<hi)]-lo
        if len(in_range)>0:
            out_batches.append(batch.take(pa.array(in_range)))
        pos+=bsz
    tbl=pa.Table.from_batches(out_batches)
    pq.write_table(tbl,out_path,compression="zstd",compression_level=9)
    print(f"    wrote {out_path} ({out_path.stat().st_size/1e6:.1f} MB)",flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--in",dest="in_path",required=True)
    ap.add_argument("--out",dest="out_path",required=True)
    ap.add_argument("--max-edges",type=int,default=50000)
    ap.add_argument("--max-genes",type=int,default=2000)
    args=ap.parse_args()
    subset(Path(args.in_path),Path(args.out_path),args.max_edges,args.max_genes)

if __name__=="__main__":
    main()
