from __future__ import annotations
import argparse, json, time, sys
from pathlib import Path
import numpy as np
import pandas as pd
from endopigraph.visium_pi import visium_cells_x_gene
from endopigraph.centroid_graph import build_voronoi_graph,approx_junctional_expression,edge_type_counts
from endopigraph.cell_typing import score_lineages,add_edge_types,edge_type_summary
from endopigraph.ec_call import call_ec
from endopigraph.vascbed import cluster_edges

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--h5",required=True)
    ap.add_argument("--spatial-dir",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--max-edge-um",type=float,default=200.0,
        help="Visium spots are ~100um center-to-center; use 200um to cover ring neighbors")
    ap.add_argument("--junct-mode",choices=["min","geometric","product","sum"],default="min")
    args=ap.parse_args()
    outd=Path(args.out); outd.mkdir(parents=True,exist_ok=True)
    log={}; t0=time.time()

    print("[1/6] loading filtered h5 + tissue positions...",flush=True)
    spatial_dir=Path(args.spatial_dir)
    sf_path=spatial_dir/"scalefactors_json.json"
    df,upp=visium_cells_x_gene(args.h5,spatial_dir,scalefactors_path=sf_path if sf_path.exists() else None)
    log["n_spots"]=int(len(df))
    log["um_per_pixel"]=float(upp)
    print(f"  {len(df):,} in-tissue spots, um/px={upp:.4f}",flush=True)

    gene_cols=[c for c in df.columns if c not in("cell_id","barcode","in_tissue","array_row","array_col","pxl_row_in_fullres","pxl_col_in_fullres","x","y")]
    log["n_genes"]=len(gene_cols)

    print("[2/6] lineage typing per spot (multi-cell mixture)...",flush=True)
    cxg_for_typing=df[["cell_id"]+gene_cols].copy()
    types=score_lineages(cxg_for_typing,species="human",min_total=5,min_fold=1.3)
    types.to_parquet(outd/"spot_types.parquet",index=False)
    log["lineage_counts"]=types["lineage"].value_counts().to_dict()
    log["n_spots_typed"]=int((types["lineage"]!="unassigned").sum())

    print("[3/6] Voronoi adjacency on spots...",flush=True)
    t1=time.time()
    edges=build_voronoi_graph(df,x_col="x",y_col="y",max_edge_um=args.max_edge_um)
    log["t_voronoi_s"]=round(time.time()-t1,2)
    log["n_edges"]=int(len(edges))
    if len(edges)==0: sys.exit("no edges")

    print("[4/6] approx junctional expression...",flush=True)
    t2=time.time()
    cxg_for_junct=df[["cell_id"]+gene_cols].copy()
    je=approx_junctional_expression(edges,cxg_for_junct,mode=args.junct_mode,
                                    weight_by_contact=True,gene_cols=gene_cols)
    log["t_junct_s"]=round(time.time()-t2,2)
    je.to_parquet(outd/"edge_junctional_expression.parquet",index=False)
    log["junct_mode"]=args.junct_mode

    print("[5/6] typed edges + summary...",flush=True)
    typed=add_edge_types(edges,types,type_col="lineage")
    typed.to_parquet(outd/"edges_typed.parquet",index=False)
    et=typed.groupby("edge_type").size().rename("n_edges").reset_index().sort_values("n_edges",ascending=False).reset_index(drop=True)
    if "contact_len_um" in typed.columns:
        tl=typed.groupby("edge_type")["contact_len_um"].sum().rename("total_contact_len_um").reset_index()
        et=et.merge(tl,on="edge_type",how="left")
    et.to_parquet(outd/"edge_type_summary.parquet",index=False)
    log["n_edge_types"]=int(len(et))
    log["top_edge_types"]=et.head(15).to_dict(orient="records")

    print("[6/6] EC subanalysis on junctional expression...",flush=True)
    cxg_ec=df[["cell_id"]+gene_cols].copy()
    ec=call_ec(cxg_ec)
    ec.to_parquet(outd/"ec_calls.parquet",index=False)
    log["n_ec_spots"]=int(ec["is_ec"].sum())
    log["ec_sep"]=float(ec["sep"].iloc[0]) if len(ec) else 0.0
    ec_set=set(ec.loc[ec["is_ec"],"cell_id"].astype(int))
    ec_pairs=edges[edges["cell_i"].isin(ec_set)&edges["cell_j"].isin(ec_set)][["cell_i","cell_j"]]
    ec_je=je.merge(ec_pairs,on=["cell_i","cell_j"])
    log["n_ec_edges"]=int(len(ec_je))
    if len(ec_je)>=10:
        ec_for=ec_je.copy(); ec_for["contact_px"]=ec_je["contact_len_um"]
        beds,bm=cluster_edges(ec_for,species="human",k_min=2,k_max=6,min_total_counts=0.1)
        beds.to_parquet(outd/"ec_edges_bed.parquet",index=False)
        log["bed_meta"]={"k":bm["k"],"cluster_label":bm["cluster_label"],"sizes":bm["sizes"]}

    log["total_s"]=round(time.time()-t0,2)
    (outd/"pilot_summary.json").write_text(json.dumps(log,indent=2,default=str))
    print(json.dumps(log,indent=2,default=str))

if __name__=="__main__":
    main()
