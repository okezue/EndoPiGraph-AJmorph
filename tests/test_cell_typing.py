import numpy as np
import pandas as pd
from endopigraph.cell_typing import score_lineages,add_edge_types,edge_type_summary,LINEAGES_HUMAN

rng=np.random.default_rng(0)

def _make_cxg():
    pools={
        "endothelial":["PECAM1","CDH5","VWF","KDR","CLDN5"],
        "epithelial":["EPCAM","KRT8","KRT18","KRT19"],
        "tcell":["CD3D","CD3E","CD8A","TRAC"],
        "fibroblast":["COL1A1","COL1A2","DCN","LUM"],
    }
    all_g=sorted(set(g for v in pools.values() for g in v))+["ACTB","GAPDH"]
    rows=[]; truth=[]
    for lin,gs in pools.items():
        for _ in range(20):
            v={g:int(rng.poisson(1)) for g in all_g}
            for g in gs: v[g]=int(rng.poisson(20))
            rows.append(v); truth.append(lin)
    df=pd.DataFrame(rows); df.insert(0,"cell_id",np.arange(1,len(df)+1))
    return df,np.array(truth)

class TestScoreLineages:
    def test_assigns_known_lineages(self):
        cxg,truth=_make_cxg()
        res=score_lineages(cxg)
        acc=(res["lineage"].values==truth).mean()
        assert acc>=0.85,f"accuracy {acc} too low"

    def test_unassigned_when_no_signal(self):
        all_g=["ACTB","GAPDH","TUBB"]
        df=pd.DataFrame({g:rng.poisson(5,size=20) for g in all_g})
        df.insert(0,"cell_id",np.arange(1,21))
        res=score_lineages(df)
        assert (res["lineage"]=="unassigned").mean()>=0.8

    def test_species_autodetect_mouse(self):
        g=["Pecam1","Cdh5","Vwf","Kdr","Actb"]
        df=pd.DataFrame(rng.poisson(10,size=(10,len(g))),columns=g)
        df.insert(0,"cell_id",np.arange(1,11))
        res=score_lineages(df)
        assert (res["species"]=="mouse").all()

class TestEdgeTypes:
    def test_add_edge_types_symmetric(self):
        cxg,_=_make_cxg()
        types=score_lineages(cxg)
        edges=pd.DataFrame({"cell_i":[1,1,21,41],"cell_j":[2,21,41,61],
                            "contact_px":[10,10,10,10],"PECAM1":[5,0,0,0]})
        out=add_edge_types(edges,types)
        assert "edge_type" in out.columns
        assert (out["type_i"]<=out["type_j"]).all()
        assert out["edge_type"].str.contains("__").all()

    def test_summary_aggregates(self):
        cxg,_=_make_cxg()
        types=score_lineages(cxg)
        edges=pd.DataFrame({"cell_i":[1,2,3],"cell_j":[4,5,6],
                            "contact_px":[10,20,30],"GENE":[1,2,3]})
        out=add_edge_types(edges,types)
        s=edge_type_summary(out)
        assert "n_edges" in s.columns
        assert "GENE" in s.columns
        assert s["n_edges"].sum()==3
