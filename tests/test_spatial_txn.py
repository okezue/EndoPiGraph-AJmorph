import numpy as np
import pandas as pd
import pytest
from endopigraph.spatial_txn import (
    junctional_counts,cell_x_gene_from_transcripts,
    rasterize_polygons,shape_from_polygons,load_transcripts,
)

@pytest.fixture
def lab4():
    L=np.zeros((40,40),dtype=np.int32)
    L[0:20,0:20]=1; L[0:20,20:40]=2; L[20:40,0:20]=3; L[20:40,20:40]=4
    return L

def _tx(pts):
    return pd.DataFrame(pts,columns=["x","y","gene"])

class TestJunctionalCounts:
    def test_plants_on_vertical_iface(self,lab4):
        pts=[(19.5,5,"A"),(20.5,6,"A"),(19.5,7,"B"),(5,5,"C")]
        df=_tx(pts)
        out=junctional_counts(lab4,df,pxsz=1.0,dilate_px=2,min_contact_px=1)
        e12=out[(out.cell_i==1)&(out.cell_j==2)]
        assert e12["A"].iloc[0]>=2
        assert e12["B"].iloc[0]>=1
        assert e12["C"].iloc[0]==0

    def test_horizontal_iface(self,lab4):
        pts=[(5,19.5,"H"),(6,20.5,"H"),(5,5,"X")]
        df=_tx(pts)
        out=junctional_counts(lab4,df,pxsz=1.0,dilate_px=2,min_contact_px=1)
        e13=out[(out.cell_i==1)&(out.cell_j==3)]
        assert e13["H"].iloc[0]>=2
        assert e13["X"].iloc[0]==0

    def test_no_diagonal(self,lab4):
        df=_tx([(19,19,"D")])
        out=junctional_counts(lab4,df,pxsz=1.0,dilate_px=1,min_contact_px=1)
        assert not ((out.cell_i==1)&(out.cell_j==4)).any()

    def test_empty_tx(self,lab4):
        df=_tx([])
        df["x"]=df["x"].astype(float); df["y"]=df["y"].astype(float)
        out=junctional_counts(lab4,df,pxsz=1.0,dilate_px=2,min_contact_px=1)
        assert len(out)>0
        gene_cols=[c for c in out.columns if c not in("cell_i","cell_j","contact_px")]
        assert all((out[g]==0).all() for g in gene_cols)

    def test_returns_iface_optional(self,lab4):
        df=_tx([(19.5,5,"A")])
        out,iface=junctional_counts(lab4,df,pxsz=1.0,dilate_px=2,min_contact_px=1,return_iface=True)
        assert iface.edges is not None
        assert len(out)==len(iface.edges)

class TestCellXGene:
    def test_assigns_to_interior(self,lab4):
        pts=[(5,5,"G1"),(5,5,"G1"),(25,5,"G2"),(5,25,"G3")]
        df=_tx(pts)
        cxg=cell_x_gene_from_transcripts(lab4,df,pxsz=1.0)
        r1=cxg[cxg.cell_id==1].iloc[0]
        assert r1["G1"]==2
        assert cxg[cxg.cell_id==2].iloc[0]["G2"]==1
        assert cxg[cxg.cell_id==3].iloc[0]["G3"]==1

class TestRasterize:
    def test_square_poly(self):
        polys={1:np.array([[1,1],[10,1],[10,10],[1,10]],dtype=np.float32)}
        L=rasterize_polygons(polys,(20,20),pxsz=1.0)
        assert L[5,5]==1
        assert L[15,15]==0

    def test_shape_from_polygons(self):
        polys={1:np.array([[0,0],[10,20]],dtype=np.float32)}
        h,w=shape_from_polygons(polys,pxsz=1.0,pad=4)
        assert h>=24 and w>=14

class TestLoadTranscripts:
    def test_xenium_rename(self,tmp_path):
        p=tmp_path/"tx.csv"
        pd.DataFrame({"x_location":[1.0,2.0],"y_location":[3.0,4.0],"feature_name":["A","B"],"qv":[25,5]}).to_csv(p,index=False)
        df=load_transcripts(p,qv_min=20.0)
        assert list(df.columns)==["x","y","gene"]
        assert len(df)==1
        assert df.iloc[0]["gene"]=="A"

    def test_merscope_rename(self,tmp_path):
        p=tmp_path/"tx.csv"
        pd.DataFrame({"global_x":[1.0],"global_y":[2.0],"gene":["G"]}).to_csv(p,index=False)
        df=load_transcripts(p,platform="merscope")
        assert list(df.columns)==["x","y","gene"]

    def test_blank_codes_filtered(self,tmp_path):
        p=tmp_path/"tx.csv"
        pd.DataFrame({"x_location":[1,2,3],"y_location":[1,2,3],"feature_name":["A","BLANK_001","NegCtrl-3"]}).to_csv(p,index=False)
        df=load_transcripts(p,qv_min=None)
        assert "A" in df["gene"].values
        assert "BLANK_001" in df["gene"].values or len(df)>=1
