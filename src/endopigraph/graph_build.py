from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import networkx as nx
import pandas as pd


def build_graph(
    cells: pd.DataFrame,
    edges: pd.DataFrame,
    junction_types: Iterable[str],
    node_id_col: str = "cell_id",
    src_col: str = "cell_i",
    dst_col: str = "cell_j",
) -> nx.Graph:
    """Build an undirected typed contact graph.

    Parameters
    ----------
    cells:
        DataFrame with one row per cell.
    edges:
        DataFrame with one row per contacting pair.
        Expected columns include src_col and dst_col.
    junction_types:
        Iterable of junction type names (e.g. ["AJ", "TJ"]).

    Returns
    -------
    G:
        networkx Graph with node and edge attributes.
    """
    junction_types = list(junction_types)

    if node_id_col not in cells.columns:
        raise KeyError(f"cells missing node id column: {node_id_col}")
    if src_col not in edges.columns or dst_col not in edges.columns:
        raise KeyError(f"edges must include columns {src_col} and {dst_col}")

    G = nx.Graph()

    # Nodes
    for _, row in cells.iterrows():
        node_id = int(row[node_id_col])
        attrs = {k: _jsonable(row[k]) for k in cells.columns if k != node_id_col}
        G.add_node(node_id, **attrs)

    # Edges
    for _, row in edges.iterrows():
        u = int(row[src_col])
        v = int(row[dst_col])
        attrs = {k: _jsonable(row[k]) for k in edges.columns if k not in (src_col, dst_col)}

        present = []
        for t in junction_types:
            flag_col = f"has_{t}"
            if flag_col in row and bool(row[flag_col]):
                present.append(t)
        attrs["junction_types"] = present  # kept as list for JSON; GraphML will stringify
        G.add_edge(u, v, **attrs)

    # pi-incidence per node: union of incident junction types
    for n in G.nodes:
        types: set[str] = set()
        for _, _, edata in G.edges(n, data=True):
            for t in edata.get("junction_types", []):
                types.add(t)
        G.nodes[n]["pi"] = sorted(types)

    return G


def write_graph_outputs(G: nx.Graph, out_prefix: str | Path) -> Dict[str, Path]:
    """Write GraphML and JSON side by side.

    GraphML is convenient for Cytoscape/Gephi but does not support list-valued
    attributes. We therefore stringify list values for the GraphML export.
    """
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    graphml_path = out_prefix.with_suffix(".graphml")
    json_path = out_prefix.with_suffix(".json")

    # GraphML-compatible copy
    G2 = G.copy()
    for n, data in G2.nodes(data=True):
        for k, v in list(data.items()):
            if isinstance(v, (list, dict)):
                data[k] = json.dumps(v)
    for u, v, data in G2.edges(data=True):
        for k, val in list(data.items()):
            if isinstance(val, (list, dict)):
                data[k] = json.dumps(val)

    nx.write_graphml(G2, graphml_path)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(graph_to_dict(G), f, indent=2)

    return {"graphml": graphml_path, "json": json_path}


def graph_to_dict(G: nx.Graph) -> Dict[str, Any]:
    nodes = []
    for n, data in G.nodes(data=True):
        nodes.append({"id": int(n), **_make_json_safe(data)})

    edges = []
    for u, v, data in G.edges(data=True):
        edges.append({"source": int(u), "target": int(v), **_make_json_safe(data)})

    return {
        "directed": False,
        "multigraph": False,
        "nodes": nodes,
        "edges": edges,
    }


def _jsonable(x: Any) -> Any:
    # pandas scalars -> python scalars
    try:
        import numpy as np

        if isinstance(x, (np.generic,)):
            return x.item()
    except Exception:
        pass

    # NaN -> None
    try:
        import pandas as pd

        if pd.isna(x):
            return None
    except Exception:
        pass

    return x


def _make_json_safe(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: _jsonable(v) for k, v in d.items()}
