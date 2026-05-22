from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

META={"cell_i","cell_j","contact_px","contact_len_um","type_i","type_j","edge_type"}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cxg",required=True)
    ap.add_argument("--edges",required=True)
    ap.add_argument("--cell-types",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--target-edge-types",nargs="+",
                    default=["epithelial__epithelial","fibroblast__fibroblast","tcell__tcell","myeloid__myeloid"])
    ap.add_argument("--genes",nargs="+",default=None)
    args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)

    cxg=pd.read_parquet(args.cxg)
    ed=pd.read_parquet(args.edges)
    ed_gene_cols=[c for c in ed.columns if c not in META and pd.api.types.is_numeric_dtype(ed[c])]
    cxg_gene_cols=[c for c in cxg.columns if c not in("cell_id","barcode","fov","x","y","x_px","y_px") and pd.api.types.is_numeric_dtype(cxg[c])]
    shared=[g for g in cxg_gene_cols if g in ed_gene_cols]
    if args.genes:
        shared=[g for g in shared if g in args.genes]
    body=cxg.set_index("cell_id")

    rows=[]
    for et in args.target_edge_types:
        sub=ed[ed["edge_type"]==et]
        n=len(sub)
        if n<20: continue
        for g in shared:
            v=sub[g].astype(np.float64).values
            ci=sub["cell_i"].values; cj=sub["cell_j"].values
            body_i=body[g].reindex(ci).fillna(0).values
            body_j=body[g].reindex(cj).fillna(0).values
            denom=np.maximum(np.maximum(body_i,body_j),1e-9)
            ratio=v/(denom+v+1e-9)
            mask=(body_i+body_j+v)>=2
            if mask.sum()<10: continue
            rows.append((et,g,float(np.mean(ratio[mask])),float(np.median(ratio[mask])),int(mask.sum()),float(np.mean(v[mask]))))
    df=pd.DataFrame(rows,columns=["edge_type","gene","mean_bnd_frac","median_bnd_frac","n_edges","mean_edge_signal"])
    df.to_csv(out/"boundary_by_edge_type.csv",index=False)

    print("=== top 5 genes per edge_type by mean_bnd_frac ===")
    for et,grp in df.groupby("edge_type"):
        top=grp.sort_values("mean_bnd_frac",ascending=False).head(8)
        print(f"\n  {et}  (n_edges={top['n_edges'].iloc[0]}):")
        for _,r in top.iterrows():
            print(f"    {r['gene']:12s} mean_bnd_frac={r['mean_bnd_frac']:.3f}  median={r['median_bnd_frac']:.3f}")

if __name__=="__main__":
    main()
