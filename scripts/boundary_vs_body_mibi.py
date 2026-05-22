from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

META={"cell_i","cell_j","contact_px","contact_len_um","type_i","type_j","edge_type"}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cxm",required=True,help="cells_x_marker.parquet")
    ap.add_argument("--edges",required=True,help="edges_typed.parquet (with *_mean cols)")
    ap.add_argument("--out",required=True)
    args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)

    cxm=pd.read_parquet(args.cxm)
    ed=pd.read_parquet(args.edges)
    body_cols=[c for c in cxm.columns if c!="cell_id"]
    edge_mean_cols=[c for c in ed.columns if c.endswith("_mean")]
    shared=[]
    for c in body_cols:
        if f"{c}_mean" in edge_mean_cols: shared.append(c)
    print(f"shared markers: {len(shared)}")
    body=cxm.set_index("cell_id")
    rows=[]
    for marker in shared:
        ec=f"{marker}_mean"
        v_edge=ed[ec].astype(np.float64).values
        ci=ed["cell_i"].values; cj=ed["cell_j"].values
        ed_i_marker=pd.Series(v_edge,index=ci).groupby(level=0).mean()
        ed_j_marker=pd.Series(v_edge,index=cj).groupby(level=0).mean()
        ed_per_cell=ed_i_marker.combine_first(ed_j_marker).reindex(body.index).fillna(np.nan)
        body_v=body[marker].astype(np.float64).reindex(body.index).fillna(np.nan)
        m=(~ed_per_cell.isna())&(~body_v.isna())&(body_v>0)
        if m.sum()<10:
            rows.append((marker,np.nan,np.nan,np.nan,int(m.sum()))); continue
        ratio=ed_per_cell[m].values/body_v[m].values
        rows.append((marker,float(np.nanmedian(ratio)),float(np.nanmean(ratio)),
                     float(body_v[m].mean()),int(m.sum())))
    df=pd.DataFrame(rows,columns=["marker","median_ratio_bnd_over_body","mean_ratio","mean_body","n_cells_used"])
    df=df.sort_values("median_ratio_bnd_over_body",ascending=False)
    df.to_csv(out/"boundary_vs_body_mibi.csv",index=False)
    print(df.to_string(index=False))

if __name__=="__main__":
    main()
