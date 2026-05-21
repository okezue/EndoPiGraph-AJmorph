from __future__ import annotations
import argparse, json, time, sys
from pathlib import Path
import numpy as np
import pandas as pd
from endopigraph.centroid_graph import (build_voronoi_graph,approx_junctional_expression,
    edges_with_types,edge_type_counts)
from endopigraph.cell_typing import score_lineages,add_edge_types,edge_type_summary,LINEAGES_MOUSE
from endopigraph.ec_call import call_ec
from endopigraph.vascbed import cluster_edges

def _resolve_xy(adata):
    if "spatial" in adata.obsm: return np.asarray(adata.obsm["spatial"])
    for k in ("X_spatial","spatial_aligned","spatial_warped"):
        if k in adata.obsm: return np.asarray(adata.obsm[k])
    for cols in (("x","y"),("X","Y"),("x_centroid","y_centroid"),("center_x","center_y")):
        if cols[0] in adata.obs.columns and cols[1] in adata.obs.columns:
            return adata.obs[[cols[0],cols[1]]].values.astype(np.float64)
    raise ValueError("no spatial coords found")

def _resolve_classes(adata):
    for c in ("annotation","Annotation","cell_type","celltype","class","subclass","leiden","clusters","layer","region"):
        if c in adata.obs.columns: return c
    return None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--h5ad",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--max-edge-um",type=float,default=25.0)
    ap.add_argument("--junct-mode",choices=["min","geometric","product","sum"],default="min")
    ap.add_argument("--max-cells",type=int,default=None)
    args=ap.parse_args()
    outd=Path(args.out); outd.mkdir(parents=True,exist_ok=True)
    log={}; t0=time.time()

    print("[1/6] loading h5ad...",flush=True)
    import anndata as ad
    adata=ad.read_h5ad(args.h5ad)
    log["n_cells_total"]=int(adata.n_obs)
    log["n_genes"]=int(adata.n_vars)
    print(f"  {adata.n_obs:,} cells x {adata.n_vars:,} genes, obs cols: {list(adata.obs.columns)[:10]}",flush=True)
    if args.max_cells is not None and adata.n_obs>args.max_cells:
        idx=np.random.default_rng(0).choice(adata.n_obs,args.max_cells,replace=False)
        adata=adata[idx,:].copy()
        log["subsampled_to"]=int(adata.n_obs)

    xy=_resolve_xy(adata)
    class_col=_resolve_classes(adata)
    log["spatial_range"]={"x":[float(xy[:,0].min()),float(xy[:,0].max())],
                          "y":[float(xy[:,1].min()),float(xy[:,1].max())]}
    log["class_col"]=class_col

    print(f"[2/6] building cell_x_gene + lineage typing (class_col={class_col})...",flush=True)
    X=adata.X
    try: X=X.toarray()
    except Exception: X=np.asarray(X)
    gene_names=[str(g) for g in adata.var.index]
    if len(set(gene_names))<len(gene_names):
        seen={}; keep=[]; uniq=[]
        for i,g in enumerate(gene_names):
            if g not in seen: seen[g]=len(seen); uniq.append(g); keep.append(i)
        X=X[:,keep]; gene_names=uniq
    cxg=pd.DataFrame(X.astype(np.float32),columns=gene_names)
    cxg.insert(0,"cell_id",np.arange(1,len(cxg)+1))
    if class_col is not None:
        from scripts.pilot_allen_merfish import _map_class
        anno=adata.obs[class_col].astype(str).values
        cell_types=pd.DataFrame({"cell_id":cxg["cell_id"].values,"lineage":[_map_class(a) for a in anno],
                                 "raw_class":anno,"species":"mouse"})
        if (cell_types["lineage"]=="unassigned").mean()>=0.8:
            print("  >80% unassigned from annotation; falling back to marker scoring",flush=True)
            cell_types=score_lineages(cxg,species="mouse",min_total=3,min_fold=1.3)
    else:
        cell_types=score_lineages(cxg,species="mouse",min_total=3,min_fold=1.3)
    cell_types.to_parquet(outd/"cell_types.parquet",index=False)
    log["lineage_counts"]=cell_types["lineage"].value_counts().to_dict()
    log["n_cells_used"]=int(len(cell_types))

    print(f"[3/6] Voronoi adjacency (max_edge_um={args.max_edge_um})...",flush=True)
    df_xy=pd.DataFrame({"x":xy[:,0],"y":xy[:,1]})
    t1=time.time()
    edges=build_voronoi_graph(df_xy,x_col="x",y_col="y",max_edge_um=args.max_edge_um)
    log["t_voronoi_s"]=round(time.time()-t1,2)
    log["n_edges"]=int(len(edges))
    if len(edges)==0: sys.exit("no edges - check max_edge_um vs spatial units")

    print("[4/6] approx junctional expression...",flush=True)
    t2=time.time()
    je=approx_junctional_expression(edges,cxg,mode=args.junct_mode,
                                    weight_by_contact=True,gene_cols=gene_names)
    log["t_junct_s"]=round(time.time()-t2,2)
    je.to_parquet(outd/"edge_junctional_expression.parquet",index=False)

    print("[5/6] typed edges + summary...",flush=True)
    typed=add_edge_types(edges,cell_types,type_col="lineage")
    typed.to_parquet(outd/"edges_typed.parquet",index=False)
    et=typed.groupby("edge_type").size().rename("n_edges").reset_index().sort_values("n_edges",ascending=False).reset_index(drop=True)
    if "contact_len_um" in typed.columns:
        tl=typed.groupby("edge_type")["contact_len_um"].sum().rename("total_contact_len_um").reset_index()
        et=et.merge(tl,on="edge_type",how="left")
    et.to_parquet(outd/"edge_type_summary.parquet",index=False)
    log["n_edge_types"]=int(len(et))
    log["top_edge_types"]=et.head(15).to_dict(orient="records")

    print("[6/6] EC subanalysis...",flush=True)
    ec=call_ec(cxg,species="mouse")
    ec.to_parquet(outd/"ec_calls.parquet",index=False)
    log["n_ec_cells"]=int(ec["is_ec"].sum())
    log["ec_sep"]=float(ec["sep"].iloc[0]) if len(ec) else 0.0
    ec_set=set(ec.loc[ec["is_ec"],"cell_id"].astype(int))
    ec_pairs=edges[edges["cell_i"].isin(ec_set)&edges["cell_j"].isin(ec_set)][["cell_i","cell_j"]]
    ec_je=je.merge(ec_pairs,on=["cell_i","cell_j"])
    log["n_ec_edges"]=int(len(ec_je))
    if len(ec_je)>=20:
        ec_for=ec_je.copy(); ec_for["contact_px"]=ec_je["contact_len_um"]
        beds,bm=cluster_edges(ec_for,species="mouse",k_min=2,k_max=6,min_total_counts=0.1)
        beds.to_parquet(outd/"ec_edges_bed.parquet",index=False)
        log["bed_meta"]={"k":bm["k"],"cluster_label":bm["cluster_label"],"sizes":bm["sizes"]}

    log["total_s"]=round(time.time()-t0,2)
    (outd/"pilot_summary.json").write_text(json.dumps(log,indent=2,default=str))
    print(json.dumps(log,indent=2,default=str))

if __name__=="__main__":
    main()
