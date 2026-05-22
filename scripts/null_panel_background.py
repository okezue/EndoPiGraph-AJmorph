from __future__ import annotations
import argparse, json, ssl, time
from pathlib import Path
import urllib.request
import pandas as pd
try:
    import certifi
    _CTX=ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _CTX=ssl.create_default_context()

GP="https://biit.cs.ut.ee/gprofiler/api/gost/profile/"
SOURCES=["GO:BP","GO:CC","GO:MF","KEGG","REAC"]

def gprofiler(genes,organism,background=None):
    body={"organism":organism,"query":list(genes),"sources":SOURCES,
          "user_threshold":0.05,"significance_threshold_method":"g_SCS",
          "no_iea":False,"all_results":False,"ordered":True,"output":"json"}
    if background is not None:
        body["background"]=list(background)
        body["domain_scope"]="custom"
    req=urllib.request.Request(GP,data=json.dumps(body).encode(),
        headers={"Content-Type":"application/json","Accept":"application/json"})
    with urllib.request.urlopen(req,timeout=120,context=_CTX) as r:
        return json.loads(r.read())

def run_one(label,gene_table_csv,organism,top_k,out_dir):
    df=pd.read_csv(gene_table_csv).sort_values("rank_mean")
    universe=df["gene"].tolist()
    top=universe[:top_k]
    print(f"\n=== {label} ({organism}) ===")
    print(f"  universe (shared panel): {len(universe)} genes")
    print(f"  query (top by rank_mean): {len(top)} genes")
    print(f"  examples query: {top[:8]}")
    out_dir.mkdir(parents=True,exist_ok=True)

    res_full=gprofiler(top,organism,background=None)
    f1=[r for r in res_full.get("result",[]) if r["source"] in("GO:CC","GO:BP")]
    print(f"\n  --- whole-organism background ---")
    print(f"    {len(res_full.get('result',[]))} total enriched terms")
    if f1:
        for r in sorted(f1,key=lambda x:x["p_value"])[:5]:
            print(f"    {r['source']:6s} {r['name'][:55]:55s} p={r['p_value']:.2e} k={r['intersection_size']}/{r['query_size']}")

    res_panel=gprofiler(top,organism,background=universe)
    f2=[r for r in res_panel.get("result",[]) if r["source"] in("GO:CC","GO:BP")]
    print(f"\n  --- shared-panel background ({len(universe)} genes) ---")
    print(f"    {len(res_panel.get('result',[]))} total enriched terms")
    if f2:
        for r in sorted(f2,key=lambda x:x["p_value"])[:5]:
            print(f"    {r['source']:6s} {r['name'][:55]:55s} p={r['p_value']:.2e} k={r['intersection_size']}/{r['query_size']}")
    else:
        print(f"    (NONE — the cell-junction enrichment evaporates relative to the shared panel)")

    for tag,res in [("whole_genome_bg",res_full),("panel_bg",res_panel)]:
        rows=res.get("result",[])
        if not rows: continue
        keep=["source","native","name","p_value","intersection_size","term_size","query_size","precision","recall"]
        out=pd.DataFrame([{k:r.get(k) for k in keep} for r in rows])
        out["panel"]=label; out["background"]=tag
        out.to_csv(out_dir/f"null_{label}_{tag}.csv",index=False)
    return res_full,res_panel

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--tables-dir",required=True)
    ap.add_argument("--out-dir",required=True)
    ap.add_argument("--top-k",type=int,default=50)
    args=ap.parse_args()
    tdir=Path(args.tables_dir); od=Path(args.out_dir)
    panels=[
      ("brain_3plat","brain_3plat_gene_table.csv","mmusculus"),
      ("brain_xenium_allen","brain_xenium_allen_gene_table.csv","mmusculus"),
      ("brain_xenium_stereo","brain_xenium_stereo_gene_table.csv","mmusculus"),
      ("brain_allen_stereo","brain_allen_stereo_gene_table.csv","mmusculus"),
      ("colon_xenium_cosmx","colon_xenium_cosmx_gene_table.csv","hsapiens"),
      ("breast_xenium_visium","breast_xenium_visium_gene_table.csv","hsapiens"),
    ]
    for lab,fn,org in panels:
        p=tdir/fn
        if not p.exists(): print(f"skip {lab}: {p} missing"); continue
        run_one(lab,p,org,args.top_k,od)
        time.sleep(0.5)

if __name__=="__main__":
    main()
