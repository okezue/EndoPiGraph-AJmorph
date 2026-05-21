import numpy as np
import pandas as pd
from endopigraph.centroid_graph import (delaunay_edges,knn_edges,build_centroid_graph,
    edges_with_types,edge_type_counts,voronoi_edges,build_voronoi_graph,
    approx_junctional_expression)

class TestDelaunay:
    def test_4_corners(self):
        xy=np.array([[0,0],[10,0],[0,10],[10,10]],dtype=np.float32)
        e=delaunay_edges(xy)
        assert len(e)>=5
        assert (e["cell_i"]<e["cell_j"]).all()

    def test_max_edge_filter(self):
        xy=np.array([[0,0],[1,0],[0,1],[100,100]],dtype=np.float32)
        e_lo=delaunay_edges(xy,max_edge_um=5)
        e_hi=delaunay_edges(xy,max_edge_um=200)
        assert len(e_lo)<len(e_hi)

    def test_too_few_points(self):
        e=delaunay_edges(np.array([[0,0],[1,1]],dtype=np.float32))
        assert len(e)==0

class TestKnn:
    def test_basic(self):
        rng=np.random.default_rng(0)
        xy=rng.uniform(0,100,size=(50,2)).astype(np.float32)
        e=knn_edges(xy,k=3)
        assert len(e)>0
        assert (e["cell_i"]<e["cell_j"]).all()

class TestEdgeTypes:
    def test_assigns_and_counts(self):
        xy=np.array([[0,0],[5,0],[10,0],[15,0]],dtype=np.float32)
        df=pd.DataFrame(xy,columns=["x","y"])
        df["lineage"]=["A","A","B","B"]
        edges=build_centroid_graph(df,method="knn",knn_k=2)
        typed=edges_with_types(edges,df["lineage"])
        assert "edge_type" in typed.columns
        assert (typed["type_i"]<=typed["type_j"]).all()
        s=edge_type_counts(typed)
        assert s["n_edges"].sum()==len(typed)

class TestSection:
    def test_section_isolation(self):
        df=pd.DataFrame({"x":[0,10,100,110],"y":[0,0,0,0],
                         "lineage":["A","B","A","B"],"sec":["s1","s1","s2","s2"]})
        e=build_centroid_graph(df,section_col="sec",method="knn",knn_k=1,max_edge_um=20)
        if "sec" in e.columns:
            for _,r in e.iterrows():
                a,b=int(r["cell_i"]),int(r["cell_j"])
                assert df.iloc[a]["sec"]==df.iloc[b]["sec"]

class TestVoronoi:
    def test_4_corners_finite_ridges(self):
        xy=np.array([[0,0],[10,0],[0,10],[10,10],[5,5]],dtype=np.float64)
        e=voronoi_edges(xy)
        assert len(e)>0
        assert "contact_len_um" in e.columns
        finite_ridges=e["contact_len_um"].notna().sum()
        assert finite_ridges>=1

    def test_max_edge_filter(self):
        rng=np.random.default_rng(0)
        xy=rng.uniform(0,100,size=(40,2))
        e_all=voronoi_edges(xy)
        e_filt=voronoi_edges(xy,max_edge_um=20)
        assert len(e_filt)<=len(e_all)

    def test_build_voronoi_imputes_unbounded(self):
        rng=np.random.default_rng(1)
        xy=rng.uniform(0,50,size=(20,2))
        df=pd.DataFrame(xy,columns=["x","y"])
        e=build_voronoi_graph(df,impute_unbounded=True,max_edge_um=None)
        assert e["contact_len_um"].notna().all()

class TestApproxJunctional:
    def test_min_mode(self):
        cxg=pd.DataFrame({"cell_id":[0,1,2],"G1":[10.0,5.0,0.0],"G2":[1.0,1.0,8.0]})
        edges=pd.DataFrame({"cell_i":[0,1],"cell_j":[1,2],"contact_len_um":[2.0,4.0]})
        out=approx_junctional_expression(edges,cxg,mode="min",weight_by_contact=True)
        assert out.loc[0,"G1"]==5.0*2.0
        assert out.loc[0,"G2"]==1.0*2.0
        assert out.loc[1,"G1"]==0.0
        assert out.loc[1,"G2"]==1.0*4.0

    def test_geometric_mode(self):
        cxg=pd.DataFrame({"cell_id":[0,1],"G":[16.0,4.0]})
        edges=pd.DataFrame({"cell_i":[0],"cell_j":[1],"contact_len_um":[1.0]})
        out=approx_junctional_expression(edges,cxg,mode="geometric")
        assert abs(out.loc[0,"G"]-8.0)<1e-5

    def test_unweighted(self):
        cxg=pd.DataFrame({"cell_id":[0,1],"G":[5.0,3.0]})
        edges=pd.DataFrame({"cell_i":[0],"cell_j":[1],"contact_len_um":[100.0]})
        out=approx_junctional_expression(edges,cxg,mode="min",weight_by_contact=False)
        assert out.loc[0,"G"]==3.0
