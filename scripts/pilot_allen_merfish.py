from __future__ import annotations
import argparse, json, time, sys, re
from pathlib import Path
import numpy as np
import pandas as pd
from endopigraph.centroid_graph import build_centroid_graph,edges_with_types,edge_type_counts

ALLEN_CLASS_TO_LINEAGE={
    "IT-ET Glut":"neural","NP-CT-L6b Glut":"neural","OB-CR Glut":"neural",
    "DG-IMN Glut":"neural","HY-EA Glut":"neural","TH Glut":"neural",
    "MH-LH Glut":"neural","HY Glut":"neural","P Glut":"neural","MY Glut":"neural",
    "MB Glut":"neural","CB Glut":"neural","CNU-MGE GABA":"neural","CGE GABA":"neural",
    "CNU-LGE GABA":"neural","HY MM Glut":"neural","HY Gnrh1 Glut":"neural",
    "MB GABA":"neural","HY GABA":"neural","TH GABA":"neural","Sero":"neural",
    "Astro-Epen":"glia","OPC-Oligo":"glia","Oligo NN":"glia","Astro-TE NN":"glia",
    "Astro-NT NN":"glia","Bergmann NN":"glia","Astroependymal":"glia",
    "OEC NN":"glia","Olfactory ensheathing NN":"glia","Tanycyte NN":"glia",
    "Ependymal NN":"glia","Choroid NN":"glia","Hypendymal NN":"glia",
    "Endo NN":"endothelial","Peri NN":"mural_smc","SMC NN":"mural_smc",
    "VLMC NN":"fibroblast","ABC NN":"fibroblast",
    "Micro-PVM NN":"myeloid","Microglia NN":"myeloid","BAM NN":"myeloid",
    "Lymphoid NN":"tcell","Mast NN":"mast","DC NN":"myeloid",
    "RBC NN":"unassigned","HEM-MB-HB":"unassigned",
}

def _map_class(c:str)->str:
    if c is None or (isinstance(c,float) and np.isnan(c)): return "unassigned"
    s=str(c)
    if s in ALLEN_CLASS_TO_LINEAGE: return ALLEN_CLASS_TO_LINEAGE[s]
    sl=s.lower()
    if "glut" in sl or "gaba" in sl or "neur" in sl: return "neural"
    if "astro" in sl or "oligo" in sl or "opc" in sl or "glia" in sl or "ependym" in sl: return "glia"
    if "endo" in sl: return "endothelial"
    if "peri" in sl or "smc" in sl or "mural" in sl: return "mural_smc"
    if "vlmc" in sl or "abc" in sl or "fibro" in sl: return "fibroblast"
    if "micro" in sl or "pvm" in sl or "macro" in sl: return "myeloid"
    if "lymph" in sl or "tcell" in sl: return "tcell"
    return "unassigned"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--metadata",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--class-col",default="class")
    ap.add_argument("--x-col",default="x")
    ap.add_argument("--y-col",default="y")
    ap.add_argument("--section-col",default="brain_section_label")
    ap.add_argument("--max-edge-um",type=float,default=30.0)
    ap.add_argument("--method",choices=["delaunay","knn"],default="delaunay")
    ap.add_argument("--limit-sections",type=int,default=None)
    args=ap.parse_args()
    outd=Path(args.out); outd.mkdir(parents=True,exist_ok=True)
    log={}; t0=time.time()

    print("[1/4] loading metadata...",flush=True)
    df=pd.read_csv(args.metadata,low_memory=False)
    log["n_cells_total"]=len(df)
    print(f"  loaded {len(df):,} cells, cols: {list(df.columns)[:12]}...",flush=True)

    needed_cols=[args.x_col,args.y_col,args.class_col]
    missing=[c for c in needed_cols if c not in df.columns]
    if missing:
        cand_x=[c for c in df.columns if c.lower() in ("x","x_section","center_x","x_centroid","x_ccf")][:3]
        cand_y=[c for c in df.columns if c.lower() in ("y","y_section","center_y","y_centroid","y_ccf")][:3]
        cand_cls=[c for c in df.columns if "class" in c.lower() or "subclass" in c.lower()][:5]
        sys.exit(f"missing {missing}. candidates: x={cand_x}, y={cand_y}, class={cand_cls}")

    df=df[[args.x_col,args.y_col,args.class_col]+([args.section_col] if args.section_col in df.columns else [])].copy()
    df=df.dropna(subset=[args.x_col,args.y_col]).reset_index(drop=True)
    df["lineage"]=df[args.class_col].map(_map_class)
    log["lineage_counts"]=df["lineage"].value_counts().to_dict()

    if args.limit_sections is not None and args.section_col in df.columns:
        secs=df[args.section_col].dropna().unique()[:args.limit_sections]
        df=df[df[args.section_col].isin(secs)].reset_index(drop=True)
        log["n_sections_used"]=int(len(secs))
    log["n_cells_used"]=int(len(df))

    print(f"[2/4] building centroid graph ({args.method}, max_edge_um={args.max_edge_um})...",flush=True)
    t1=time.time()
    edges=build_centroid_graph(df,x_col=args.x_col,y_col=args.y_col,
                                section_col=args.section_col if args.section_col in df.columns else None,
                                method=args.method,max_edge_um=args.max_edge_um)
    log["t_graph_build_s"]=round(time.time()-t1,2)
    log["n_edges"]=int(len(edges))
    print(f"  built {len(edges):,} edges in {log['t_graph_build_s']}s",flush=True)

    print("[3/4] adding edge types...",flush=True)
    typed=edges_with_types(edges,df["lineage"])
    typed.to_parquet(outd/"edges_typed.parquet",index=False)

    print("[4/4] edge type summary...",flush=True)
    et=edge_type_counts(typed)
    et=et.sort_values("n_edges",ascending=False).reset_index(drop=True)
    et.to_parquet(outd/"edge_type_summary.parquet",index=False)
    log["n_edge_types"]=int(len(et))
    log["top_edge_types"]=et.head(15).to_dict(orient="records")
    cells_out=df[[args.x_col,args.y_col,args.class_col,"lineage"]+([args.section_col] if args.section_col in df.columns else [])].copy()
    cells_out.insert(0,"cell_id",np.arange(1,len(cells_out)+1))
    cells_out.to_parquet(outd/"cell_types.parquet",index=False)

    log["total_s"]=round(time.time()-t0,2)
    (outd/"pilot_summary.json").write_text(json.dumps(log,indent=2,default=str))
    print(json.dumps(log,indent=2,default=str))

if __name__=="__main__":
    main()
