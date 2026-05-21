import numpy as np
import pandas as pd
from endopigraph.codex_pi import (type_cells_from_markers,interface_marker_features_multichannel,
    _match_marker,CODEX_LINEAGES)

rng=np.random.default_rng(0)

class TestMatchMarker:
    def test_exact(self):
        assert _match_marker("CD3","CD3 CD3D".split())
    def test_case_insensitive(self):
        assert _match_marker("Cd3","CD3 CD3D".split())
    def test_hyphen_insensitive(self):
        assert _match_marker("VE-Cadherin",["VECadherin"])
    def test_no_match(self):
        assert not _match_marker("CD45","CD3 CD20".split())
    def test_no_false_positive_cd3_cd31(self):
        assert not _match_marker("CD3",["CD31"])
        assert not _match_marker("CD31",["CD3"])

class TestTypeCellsFromMarkers:
    def test_assigns_lineages(self):
        n=15
        rows=[]
        for lin,markers in [("tcell",["CD3","CD8"]),("bcell",["CD20"]),("myeloid",["CD68","CD163"])]:
            for _ in range(n):
                v={"CD3":1,"CD8":1,"CD20":1,"CD68":1,"CD163":1,"DAPI":5}
                for m in markers: v[m]=20
                rows.append(v)
        df=pd.DataFrame(rows)
        df.insert(0,"cell_id",np.arange(1,len(df)+1))
        res=type_cells_from_markers(df,min_z=0.5,min_fold=1.05)
        cts=res["lineage"].value_counts()
        assert "tcell" in cts.index
        assert "bcell" in cts.index
        assert "myeloid" in cts.index

class TestMultichannelFeatures:
    def test_basic(self):
        L=np.zeros((20,20),dtype=np.int32)
        L[0:10,0:10]=1; L[0:10,10:20]=2
        arr=np.zeros((2,20,20),dtype=np.float32)
        arr[0,:,9:11]=100
        arr[1,:,:]=10
        out,_=interface_marker_features_multichannel(arr,L,channel_names=["CD31","CD45"],
                                                     dilate_px=1,min_contact_px=1)
        assert "CD31_mean" in out.columns
        assert "CD45_mean" in out.columns
        assert (out["CD31_mean"]>out["CD45_mean"]).all()
