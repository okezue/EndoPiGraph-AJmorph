from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

META={"cell_i","cell_j","contact_px","contact_len_um","type_i","type_j","edge_type"}

def per_cell_boundary_sum(edges_path:Path,gene_cols:list)->pd.DataFrame:
    pf=pq.ParquetFile(edges_path)
    avail=[c for c in gene_cols if c in pf.schema.names]
    out={}
    needed=avail+["cell_i","cell_j"]
    for batch in pf.iter_batches(batch_size=200_000,columns=needed):
        d=batch.to_pandas()
        for col in avail:
            v=d[col].values.astype(np.float64)
            si=pd.Series(v).groupby(d["cell_i"]).sum()
            sj=pd.Series(v).groupby(d["cell_j"]).sum()
            full=si.add(sj,fill_value=0)
            if col not in out: out[col]=full
            else: out[col]=out[col].add(full,fill_value=0)
    df=pd.DataFrame(out).fillna(0)
    df.index.name="cell_id"
    return df.reset_index()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cxg",required=True,help="cells_x_gene.parquet")
    ap.add_argument("--edges",required=True,help="edges_typed.parquet")
    ap.add_argument("--cell-types",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--max-genes",type=int,default=200)
    args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)

    cxg=pd.read_parquet(args.cxg)
    ct=pd.read_parquet(args.cell_types)[["cell_id","lineage"]]
    pf_ed=pq.ParquetFile(args.edges)
    ed_gene_cols=[c for c in pf_ed.schema.names if c not in META and pf_ed.schema_arrow.field(c).type in (pf_ed.schema_arrow.field(c).type,) and (str(pf_ed.schema_arrow.field(c).type).startswith("int") or str(pf_ed.schema_arrow.field(c).type).startswith("float"))]
    cxg_gene_cols=[c for c in cxg.columns if c not in("cell_id","barcode","fov","x","y","x_px","y_px","in_tissue","array_row","array_col","pxl_row_in_fullres","pxl_col_in_fullres") and pd.api.types.is_numeric_dtype(cxg[c])]
    shared=[g for g in cxg_gene_cols if g in ed_gene_cols]
    print(f"shared genes (cells_x_gene ∩ edges_typed): {len(shared)}")
    if len(shared)>args.max_genes:
        var=cxg[shared].var(numeric_only=True).sort_values(ascending=False)
        markers={"Aqp4","Gfap","Cdh5","Vwf","Pecam1","Slc17a7","Gad1","Gad2","Epha4","Nrn1","Calb1",
                 "Pvalb","Cldn5","Nr2f2","Cd31","Olig2","Mbp","Plp1","Aif1","P2ry12","Rbfox3","Map2",
                 "Igf2","Igfbp4","Penk","Lamp5","Rorb","Bcl11b","Necab1","Dcn","Tagln","Actb","Gapdh"}
        shared_m=[g for g in shared if g in markers]
        rest=[g for g in shared if g not in markers]
        keep_n=max(0,args.max_genes-len(shared_m))
        topvar=var.reindex(rest).dropna().head(keep_n).index.tolist()
        shared=sorted(set(shared_m+topvar),key=lambda g:-cxg[g].sum())
    print(f"analyzing {len(shared)} genes")

    cb=per_cell_boundary_sum(Path(args.edges),shared)
    body=cxg[["cell_id"]+shared].copy()
    bnd=cb.copy()
    body=body.set_index("cell_id")
    bnd=bnd.set_index("cell_id").reindex(body.index).fillna(0)
    body_sub=body[shared].astype(np.float64).values
    bnd_sub=bnd[shared].astype(np.float64).values
    total=np.maximum(body_sub,1e-9)
    frac=bnd_sub/np.maximum(body_sub,bnd_sub+1e-9)
    rows=[]
    for k,g in enumerate(shared):
        v=frac[:,k]
        mask=body_sub[:,k]>=2
        if mask.sum()<10:
            rows.append((g,np.nan,np.nan,np.nan,np.nan,np.nan,int(mask.sum())))
            continue
        m=np.median(v[mask])
        mu=np.mean(v[mask])
        rows.append((g,m,mu,float(np.percentile(v[mask],25)),float(np.percentile(v[mask],75)),
                     float(body_sub[mask,k].mean()),int(mask.sum())))
    out_df=pd.DataFrame(rows,columns=["gene","median_boundary_frac","mean_boundary_frac",
                                      "p25","p75","mean_cell_total","n_cells_used"])
    out_df=out_df.sort_values("median_boundary_frac",ascending=False)
    out_df.to_csv(out/"boundary_vs_body.csv",index=False)
    print(out_df.head(30).to_string(index=False))
    print()
    print("=== bottom 15 (cytoplasmic-like) ===")
    print(out_df.tail(15).to_string(index=False))

if __name__=="__main__":
    main()
