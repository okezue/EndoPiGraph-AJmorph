import json
import tempfile
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest
from endopigraph.graph_build import build_graph, write_graph_outputs, graph_to_dict


class TestBuildGraph:
    def test_basic(self, cells_df, edges_df):
        G = build_graph(cells_df, edges_df, junction_types=["AJ"])
        assert G.number_of_nodes() == 4
        assert G.number_of_edges() == 4

    def test_node_attrs(self, cells_df, edges_df):
        G = build_graph(cells_df, edges_df, junction_types=["AJ"])
        assert "area_px" in G.nodes[1]
        assert "cy" in G.nodes[1]

    def test_edge_junction_types(self, cells_df, edges_df):
        G = build_graph(cells_df, edges_df, junction_types=["AJ"])
        assert "AJ" in G.edges[1, 2]["junction_types"]
        assert "AJ" not in G.edges[1, 3]["junction_types"]

    def test_pi_incidence(self, cells_df, edges_df):
        G = build_graph(cells_df, edges_df, junction_types=["AJ"])
        assert "pi" in G.nodes[1]
        assert "AJ" in G.nodes[1]["pi"]
        pi_3 = G.nodes[3]["pi"]
        assert "AJ" in pi_3

    def test_missing_node_col(self, cells_df, edges_df):
        bad = cells_df.drop(columns=["cell_id"])
        with pytest.raises(KeyError):
            build_graph(bad, edges_df, junction_types=["AJ"])

    def test_missing_edge_col(self, cells_df, edges_df):
        bad = edges_df.drop(columns=["cell_i"])
        with pytest.raises(KeyError):
            build_graph(cells_df, bad, junction_types=["AJ"])

    def test_empty(self):
        c = pd.DataFrame({"cell_id": [], "area_px": []})
        e = pd.DataFrame({"cell_i": [], "cell_j": [], "contact_px": []})
        G = build_graph(c, e, junction_types=["AJ"])
        assert G.number_of_nodes() == 0


class TestWriteGraphOutputs:
    def test_writes_files(self, cells_df, edges_df):
        G = build_graph(cells_df, edges_df, junction_types=["AJ"])
        with tempfile.TemporaryDirectory() as td:
            paths = write_graph_outputs(G, Path(td) / "test_graph")
            assert Path(paths["graphml"]).exists()
            assert Path(paths["json"]).exists()

    def test_json_readable(self, cells_df, edges_df):
        G = build_graph(cells_df, edges_df, junction_types=["AJ"])
        with tempfile.TemporaryDirectory() as td:
            paths = write_graph_outputs(G, Path(td) / "test_graph")
            with open(paths["json"]) as f:
                d = json.load(f)
            assert "nodes" in d
            assert "edges" in d
            assert len(d["nodes"]) == 4


class TestGraphToDict:
    def test_structure(self, cells_df, edges_df):
        G = build_graph(cells_df, edges_df, junction_types=["AJ"])
        d = graph_to_dict(G)
        assert d["directed"] is False
        assert len(d["nodes"]) == 4
        assert len(d["edges"]) == 4

    def test_nan_to_none(self):
        G = nx.Graph()
        G.add_node(1, val=float("nan"))
        d = graph_to_dict(G)
        assert d["nodes"][0]["val"] is None
