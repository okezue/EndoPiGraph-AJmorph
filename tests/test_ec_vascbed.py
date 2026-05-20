import numpy as np
import pandas as pd
from endopigraph.ec_call import call_ec,filter_edges_to_ec
from endopigraph.vascbed import cluster_edges,BED_MARKERS

rng=np.random.default_rng(0)

def _cxg_with_ec(n_ec=20,n_other=40):
    genes=["PECAM1","CDH5","VWF","KDR","CLDN5","FLT1","TIE1","ESAM","CD34","ENG","ACTB","GAPDH"]
    ec=rng.poisson(20,size=(n_ec,len(genes))).astype(np.int32)
    ec[:,-2:]=rng.poisson(5,size=(n_ec,2))
    ot=rng.poisson(1,size=(n_other,len(genes))).astype(np.int32)
    ot[:,-2:]=rng.poisson(20,size=(n_other,2))
    M=np.vstack([ec,ot])
    df=pd.DataFrame(M,columns=genes)
    df.insert(0,"cell_id",np.arange(1,n_ec+n_other+1))
    return df,n_ec

class TestCallEC:
    def test_identifies_ec_cluster(self):
        cxg,n_ec=_cxg_with_ec()
        res=call_ec(cxg)
        assert res["is_ec"].iloc[:n_ec].sum()>=int(0.8*n_ec)
        assert res["is_ec"].iloc[n_ec:].sum()<=int(0.2*(len(cxg)-n_ec))
        assert res["sep"].iloc[0]>0

    def test_filter_edges_to_ec(self):
        cxg,n_ec=_cxg_with_ec()
        res=call_ec(cxg)
        edges=pd.DataFrame({"cell_i":[1,1,n_ec+1],"cell_j":[2,n_ec+1,n_ec+2],"contact_px":[10,10,10]})
        f=filter_edges_to_ec(edges,res)
        assert (f["cell_i"]==1).all()
        assert all(int(c) in set(res.loc[res.is_ec,"cell_id"]) for c in f["cell_i"])

    def test_few_markers(self):
        df=pd.DataFrame({"cell_id":[1,2,3],"PECAM1":[10,0,0],"FOO":[1,1,1]})
        res=call_ec(df,min_markers=3)
        assert not res["is_ec"].any()

    def test_mouse_species_detected(self):
        genes=["Pecam1","Cdh5","Vwf","Kdr","Cldn5","Actb"]
        M=rng.poisson(10,size=(10,len(genes)))
        df=pd.DataFrame(M,columns=genes); df.insert(0,"cell_id",np.arange(1,11))
        res=call_ec(df)
        assert (res["species"]=="mouse").all()

class TestClusterEdges:
    def test_assigns_known_beds(self):
        all_genes=sorted(set(g for gs in BED_MARKERS.values() for g in gs))+["ACTB","GAPDH"]
        rows=[]
        for bed,gs in [("arterial",BED_MARKERS["arterial"]),
                       ("venous",BED_MARKERS["venous"]),
                       ("capillary",BED_MARKERS["capillary"])]:
            for _ in range(15):
                v={g:0 for g in all_genes}
                for g in gs: v[g]=int(rng.poisson(25))
                rows.append(v)
        df=pd.DataFrame(rows)
        df.insert(0,"cell_i",np.arange(1,len(df)+1))
        df.insert(1,"cell_j",np.arange(2,len(df)+2))
        df.insert(2,"contact_px",10)
        out,meta=cluster_edges(df,k_min=2,k_max=5)
        assert meta["k"]>=2
        labs=set(out["bed_label"].unique())
        assert len(labs.intersection({"arterial","venous","capillary"}))>=2

    def test_empty_returns_unassigned(self):
        df=pd.DataFrame({"cell_i":[1],"cell_j":[2],"contact_px":[10],"FOO":[0]})
        out,meta=cluster_edges(df,k_min=2,k_max=3,min_total_counts=1)
        assert "bed_label" in out.columns
