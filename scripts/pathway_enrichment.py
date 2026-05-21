from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import urllib.request, urllib.error, ssl
import pandas as pd
try:
    import certifi
    _SSL_CTX=ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX=ssl.create_default_context()

GPROFILER_URL="https://biit.cs.ut.ee/gprofiler/api/gost/profile/"

PANELS=[
    ("brain_3plat","brain_3plat_gene_table.csv","mmusculus","Mouse brain (3 platforms shared)"),
    ("brain_xenium_allen","brain_xenium_allen_gene_table.csv","mmusculus","Xenium MB vs Allen MERFISH"),
    ("brain_allen_stereo","brain_allen_stereo_gene_table.csv","mmusculus","Allen MERFISH vs Stereo-seq MB"),
    ("brain_xenium_stereo","brain_xenium_stereo_gene_table.csv","mmusculus","Xenium MB vs Stereo-seq MB"),
    ("colon_xenium_cosmx","colon_xenium_cosmx_gene_table.csv","hsapiens","Human colon (Xenium vs CosMx)"),
    ("breast_xenium_visium","breast_xenium_visium_gene_table.csv","hsapiens","Human breast (Xenium vs Visium)"),
    ("dev_vs_adult","dev_vs_adult_gene_table.csv","mmusculus","Embryo E9.5 vs adult brain"),
    ("cosmx_colon_pancreas","cosmx_colon_pancreas_gene_table.csv","hsapiens","CosMx colon vs pancreas"),
]
SOURCES=["GO:BP","GO:CC","GO:MF","KEGG","REAC","HP","HPA"]

def gprofiler(genes,organism):
    body={"organism":organism,"query":list(genes),"sources":SOURCES,
          "user_threshold":0.05,"significance_threshold_method":"g_SCS",
          "no_iea":False,"all_results":False,"ordered":True,"output":"json"}
    req=urllib.request.Request(GPROFILER_URL,data=json.dumps(body).encode(),
                                headers={"Content-Type":"application/json","Accept":"application/json"})
    try:
        with urllib.request.urlopen(req,timeout=120,context=_SSL_CTX) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error":f"HTTP {e.code}: {e.read().decode()[:300]}"}
    except Exception as e:
        return {"error":str(e)}

def run_one(tbl_path:Path,panel:str,organism:str,top_k:int,out_dir:Path):
    df=pd.read_csv(tbl_path)
    df=df.sort_values("rank_mean").head(top_k)
    genes=df["gene"].tolist()
    print(f"[{panel}] organism={organism} top {len(genes)} genes",flush=True)
    res=gprofiler(genes,organism)
    if "error" in res:
        print(f"  ERROR: {res['error']}",flush=True); return None
    rows=res.get("result",[])
    if not rows:
        print(f"  no enriched terms",flush=True); return None
    out=pd.DataFrame(rows)
    keep=[c for c in("source","native","name","p_value","intersection_size","term_size","query_size","precision","recall","intersections") if c in out.columns]
    out=out[keep].copy()
    out["panel"]=panel; out["organism"]=organism; out["top_k"]=top_k
    out_dir.mkdir(parents=True,exist_ok=True)
    out.to_csv(out_dir/f"enrich_{panel}.csv",index=False)
    print(f"  {len(out)} terms (top: {out.iloc[0]['name'][:60]}, p={out.iloc[0]['p_value']:.2e})",flush=True)
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--tables-dir",required=True)
    ap.add_argument("--out-dir",required=True)
    ap.add_argument("--top-k",type=int,default=50)
    args=ap.parse_args()
    tbl_dir=Path(args.tables_dir); out_dir=Path(args.out_dir)
    all_results=[]
    for panel,fname,org,_ in PANELS:
        p=tbl_dir/fname
        if not p.exists():
            print(f"  skip {panel}: {p} missing",flush=True); continue
        r=run_one(p,panel,org,args.top_k,out_dir)
        if r is not None: all_results.append(r)
        time.sleep(0.5)
    if all_results:
        combo=pd.concat(all_results,ignore_index=True)
        combo.to_csv(out_dir/"enrich_all_panels.csv",index=False)
        print(f"=== wrote {len(combo)} total enriched terms across {len(all_results)} panels ===",flush=True)
    else:
        print("no enrichment results",flush=True)

if __name__=="__main__":
    main()
