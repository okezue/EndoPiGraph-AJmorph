from __future__ import annotations
import argparse, json, time, sys
from pathlib import Path
import numpy as np
import pandas as pd
from endopigraph.cosmx_pi import load_cosmx_metadata,load_cosmx_expr,make_cells_x_gene
from endopigraph.centroid_graph import build_voronoi_graph,approx_junctional_expression
from endopigraph.cell_typing import score_lineages,add_edge_types,edge_type_summary
from endopigraph.ec_call import call_ec
from endopigraph.vascbed import cluster_edges

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--expr",required=True)
    ap.add_argument("--metadata",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--um-per-px",type=float,default=0.18)
    ap.add_argument("--max-edge-um",type=float,default=20.0)
    ap.add_argument("--junct-mode",choices=["min","geometric","product","sum"],default="min")
    ap.add_argument("--max-cells",type=int,default=None)
    ap.add_argument("--limit-fovs",type=int,default=None)
    ap.add_argument("--max-genes",type=int,default=1000,
        help="cap genes used in junctional matrix to top-N by variance (memory)")
    args=ap.parse_args()
    outd=Path(args.out); outd.mkdir(parents=True,exist_ok=True)
    log={}; t0=time.time()

    print("[1/6] loading expression matrix + metadata...",flush=True)
    expr=load_cosmx_expr(args.expr,max_rows=args.max_cells,max_fovs=args.limit_fovs)
    meta=load_cosmx_metadata(args.metadata)
    log["n_expr_rows"]=int(len(expr))
    log["n_meta_rows"]=int(len(meta))
    log["meta_cols"]=list(meta.columns)[:20]
    log["expr_cols_head"]=list(expr.columns)[:10]
    print(f"  expr rows={len(expr):,}  meta rows={len(meta):,}",flush=True)

    print("[2/6] joining + scaling to microns...",flush=True)
    cxg=make_cells_x_gene(expr,meta,um_per_px=args.um_per_px)
    if args.limit_fovs is not None and "fov" in cxg.columns:
        keep_fovs=cxg["fov"].drop_duplicates().head(args.limit_fovs).tolist()
        cxg=cxg[cxg["fov"].isin(keep_fovs)].reset_index(drop=True)
        cxg["cell_id"]=np.arange(1,len(cxg)+1)
        log["limited_to_fovs"]=keep_fovs
    if args.max_cells is not None and len(cxg)>args.max_cells:
        cxg=cxg.sample(n=args.max_cells,random_state=0).reset_index(drop=True)
        cxg["cell_id"]=np.arange(1,len(cxg)+1)
    log["n_cells"]=int(len(cxg))
    log["n_genes"]=int(sum(1 for c in cxg.columns if c not in("cell_id","x","y","fov")))
    print(f"  joined: {len(cxg):,} cells, {log['n_genes']:,} genes",flush=True)

    print("[3/6] lineage typing...",flush=True)
    gene_cols=[c for c in cxg.columns if c not in("cell_id","x","y","fov")]
    types=score_lineages(cxg[["cell_id"]+gene_cols].copy(),species="human",min_total=3,min_fold=1.3)
    types.to_parquet(outd/"cell_types.parquet",index=False)
    log["lineage_counts"]=types["lineage"].value_counts().to_dict()
    log["n_cells_typed"]=int((types["lineage"]!="unassigned").sum())

    print(f"[4/6] Voronoi adjacency (max_edge_um={args.max_edge_um})...",flush=True)
    t1=time.time()
    edges=build_voronoi_graph(cxg,x_col="x",y_col="y",
                              section_col="fov" if "fov" in cxg.columns else None,
                              max_edge_um=args.max_edge_um)
    log["t_voronoi_s"]=round(time.time()-t1,2)
    log["n_edges"]=int(len(edges))
    if len(edges)==0: sys.exit("no edges")

    print("[5/6] approx junctional expression...",flush=True)
    from endopigraph.cell_typing import LINEAGES_HUMAN
    from endopigraph.ec_call import EC_HUMAN
    from endopigraph.vascbed import BED_MARKERS
    keep_marker=set()
    for v in LINEAGES_HUMAN.values(): keep_marker.update(v)
    keep_marker.update(EC_HUMAN)
    for v in BED_MARKERS.values(): keep_marker.update(v)
    marker_present=[g for g in gene_cols if g in keep_marker]
    others=[g for g in gene_cols if g not in keep_marker]
    if len(others)>args.max_genes-len(marker_present):
        var=cxg[others].var(numeric_only=True).sort_values(ascending=False)
        top_var=var.index[:max(0,args.max_genes-len(marker_present))].tolist()
    else: top_var=others
    junct_genes=list(dict.fromkeys(marker_present+top_var))
    log["n_junct_genes"]=len(junct_genes)
    print(f"  junctional matrix on {len(junct_genes)} genes ({len(marker_present)} markers + {len(top_var)} top-variable)",flush=True)
    t2=time.time()
    je=approx_junctional_expression(edges,cxg[["cell_id"]+junct_genes].copy(),
                                    mode=args.junct_mode,weight_by_contact=True,gene_cols=junct_genes)
    log["t_junct_s"]=round(time.time()-t2,2)
    je.to_parquet(outd/"edge_junctional_expression.parquet",index=False)

    print("[6/6] typed edges + EC subanalysis...",flush=True)
    typed=add_edge_types(edges,types,type_col="lineage")
    typed.to_parquet(outd/"edges_typed.parquet",index=False)
    et=typed.groupby("edge_type").size().rename("n_edges").reset_index().sort_values("n_edges",ascending=False).reset_index(drop=True)
    if "contact_len_um" in typed.columns:
        tl=typed.groupby("edge_type")["contact_len_um"].sum().rename("total_contact_len_um").reset_index()
        et=et.merge(tl,on="edge_type",how="left")
    et.to_parquet(outd/"edge_type_summary.parquet",index=False)
    log["n_edge_types"]=int(len(et))
    log["top_edge_types"]=et.head(15).to_dict(orient="records")

    ec=call_ec(cxg[["cell_id"]+gene_cols].copy(),species="human")
    ec.to_parquet(outd/"ec_calls.parquet",index=False)
    log["n_ec_cells"]=int(ec["is_ec"].sum())
    log["ec_sep"]=float(ec["sep"].iloc[0]) if len(ec) else 0.0
    ec_set=set(ec.loc[ec["is_ec"],"cell_id"].astype(int))
    ec_pairs=edges[edges["cell_i"].isin(ec_set)&edges["cell_j"].isin(ec_set)][["cell_i","cell_j"]]
    ec_je=je.merge(ec_pairs,on=["cell_i","cell_j"])
    log["n_ec_edges"]=int(len(ec_je))
    if len(ec_je)>=20:
        ec_for=ec_je.copy(); ec_for["contact_px"]=ec_je["contact_len_um"]
        beds,bm=cluster_edges(ec_for,species="human",k_min=2,k_max=6,min_total_counts=0.1)
        beds.to_parquet(outd/"ec_edges_bed.parquet",index=False)
        log["bed_meta"]={"k":bm["k"],"cluster_label":bm["cluster_label"],"sizes":bm["sizes"]}

    log["total_s"]=round(time.time()-t0,2)
    (outd/"pilot_summary.json").write_text(json.dumps(log,indent=2,default=str))
    print(json.dumps(log,indent=2,default=str))

if __name__=="__main__":
    main()
