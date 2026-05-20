from __future__ import annotations
import argparse, json, time, sys
from pathlib import Path
import numpy as np
import pandas as pd
from endopigraph.spatial_txn import (
    load_transcripts,load_polygons,rasterize_polygons,shape_from_polygons,
    junctional_counts,cell_x_gene_from_transcripts,
)
from endopigraph.cell_typing import score_lineages,add_edge_types,edge_type_summary
from endopigraph.ec_call import call_ec,filter_edges_to_ec
from endopigraph.vascbed import cluster_edges

XENIUM_PXSZ=0.2125

def _find(root:Path,*names):
    for n in names:
        ms=list(root.rglob(n))
        if ms: return ms[0]
    return None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--outs",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--qv-min",type=float,default=20.0)
    ap.add_argument("--dilate-px",type=int,default=3)
    ap.add_argument("--min-contact-px",type=int,default=10)
    ap.add_argument("--bbox",nargs=4,type=float,default=None,
                    metavar=("XMIN","YMIN","XMAX","YMAX"))
    ap.add_argument("--max-cells",type=int,default=None)
    ap.add_argument("--ec-subanalysis",action="store_true",
                    help="Also run EC-restricted vascular-bed clustering")
    args=ap.parse_args()
    outs=Path(args.outs); outd=Path(args.out); outd.mkdir(parents=True,exist_ok=True)
    log={}
    t0=time.time()

    tx_p=_find(outs,"transcripts.parquet","transcripts.csv.gz","transcripts.csv")
    bd_p=_find(outs,"cell_boundaries.parquet","cell_boundaries.csv.gz","cell_boundaries.csv")
    if tx_p is None or bd_p is None: sys.exit(f"missing: tx={tx_p} bd={bd_p}")
    print(f"transcripts: {tx_p}\nboundaries: {bd_p}",flush=True)

    print("[1/7] loading polygons...",flush=True)
    polys,_=load_polygons(bd_p)
    log["n_cells_total"]=len(polys)

    ox=oy=0.0
    if args.bbox is not None:
        xmin,ymin,xmax,ymax=args.bbox
        polys={c:v for c,v in polys.items()
               if v[:,0].min()>=xmin and v[:,0].max()<=xmax
               and v[:,1].min()>=ymin and v[:,1].max()<=ymax}
        ox,oy=xmin,ymin
        polys={c:np.stack([v[:,0]-ox,v[:,1]-oy],axis=1) for c,v in polys.items()}
        log["bbox_um"]=list(args.bbox)
    if args.max_cells is not None and len(polys)>args.max_cells:
        keep=list(polys.keys())[:args.max_cells]
        polys={c:polys[c] for c in keep}
    log["n_cells_used"]=len(polys)
    if len(polys)==0: sys.exit("no cells after filter")

    print("[2/7] rasterizing labels (origin-shifted)...",flush=True)
    H,W=shape_from_polygons(polys,pxsz=XENIUM_PXSZ,pad=8)
    log["labels_shape"]=[H,W]
    L=rasterize_polygons(polys,(H,W),pxsz=XENIUM_PXSZ)
    log["n_label_pixels"]=int((L>0).sum())

    print("[3/7] loading transcripts...",flush=True)
    tx=load_transcripts(tx_p,qv_min=args.qv_min,platform="xenium")
    if args.bbox is not None:
        tx=tx[(tx.x>=args.bbox[0])&(tx.x<=args.bbox[2])&(tx.y>=args.bbox[1])&(tx.y<=args.bbox[3])].copy()
        tx["x"]=tx["x"]-ox; tx["y"]=tx["y"]-oy
    log["n_transcripts"]=int(len(tx))
    log["n_genes_total"]=int(tx["gene"].nunique())

    print("[4/7] junctional gene counts (all edges)...",flush=True)
    t1=time.time()
    edges_x_gene=junctional_counts(L,tx,pxsz=XENIUM_PXSZ,
                                   dilate_px=args.dilate_px,
                                   min_contact_px=args.min_contact_px)
    log["t_junctional_counts_s"]=round(time.time()-t1,2)
    log["n_edges"]=int(len(edges_x_gene))

    print("[5/7] cell x gene + lineage typing...",flush=True)
    cxg=cell_x_gene_from_transcripts(L,tx,pxsz=XENIUM_PXSZ)
    cxg.to_parquet(outd/"cells_x_gene.parquet",index=False)
    types=score_lineages(cxg,species="auto",min_total=3,min_fold=1.5)
    types.to_parquet(outd/"cell_types.parquet",index=False)
    tcounts=types["lineage"].value_counts().to_dict()
    log["lineage_counts"]=tcounts

    print("[6/7] adding typed edges + edge-type summary...",flush=True)
    typed=add_edge_types(edges_x_gene,types,type_col="lineage")
    typed.to_parquet(outd/"edges_typed.parquet",index=False)
    et_sum=edge_type_summary(typed)
    et_sum.to_parquet(outd/"edge_type_summary.parquet",index=False)
    et_top=et_sum.nlargest(15,"n_edges")[["edge_type","n_edges","total_contact_px"]]
    log["top_edge_types"]=et_top.to_dict(orient="records")
    log["n_edge_types"]=int(et_sum["edge_type"].nunique())

    if args.ec_subanalysis:
        print("[7/7] EC subanalysis (vascular bed clustering)...",flush=True)
        ec=call_ec(cxg)
        ec.to_parquet(outd/"ec_calls.parquet",index=False)
        ec_edges=filter_edges_to_ec(edges_x_gene,ec)
        log["n_ec_cells"]=int(ec["is_ec"].sum())
        log["ec_sep"]=float(ec["sep"].iloc[0]) if len(ec) else 0.0
        log["n_ec_edges"]=int(len(ec_edges))
        if len(ec_edges)>=10:
            beds,bm=cluster_edges(ec_edges,k_min=2,k_max=6)
            beds.to_parquet(outd/"ec_edges_bed.parquet",index=False)
            log["bed_meta"]=bm

    log["total_s"]=round(time.time()-t0,2)
    (outd/"pilot_summary.json").write_text(json.dumps(log,indent=2,default=str))
    print(json.dumps(log,indent=2,default=str))

if __name__=="__main__":
    main()
