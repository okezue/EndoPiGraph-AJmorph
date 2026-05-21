import numpy as np
import pandas as pd
import tempfile, json
from pathlib import Path
import pytest
from endopigraph.visium_pi import load_tissue_positions,visium_cells_x_gene

class TestLoadTissuePositions:
    def test_with_header(self,tmp_path):
        p=tmp_path/"tissue_positions.csv"
        pd.DataFrame({"barcode":["A","B"],"in_tissue":[1,1],"array_row":[0,0],"array_col":[0,2],
                      "pxl_row_in_fullres":[100.0,100.0],"pxl_col_in_fullres":[50.0,250.0]}).to_csv(p,index=False)
        df=load_tissue_positions(tmp_path)
        assert len(df)==2 and "barcode" in df.columns

    def test_no_header(self,tmp_path):
        p=tmp_path/"tissue_positions_list.csv"
        pd.DataFrame([["A",1,0,0,100,50],["B",1,0,2,100,250]]).to_csv(p,index=False,header=False)
        df=load_tissue_positions(tmp_path)
        assert df.iloc[0]["barcode"]=="A"
        assert df.iloc[1]["pxl_col_in_fullres"]==250
