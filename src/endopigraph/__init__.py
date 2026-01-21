"""EndoPiGraph-AJmorph v1.

Core idea:
- segment endothelial cells (instance labels)
- infer cell-cell contacts (interfaces)
- quantify junction-marker signal on each interface (e.g. VE-cad for AJ)
- build a typed cell-contact graph ("pi-graph" in the manuscript sense)

This package aims to be:
- explicit about assumptions
- easy to run on public microscopy data (e.g. BioImage Archive accessions)
- easy to extend with better segmentation / classification models
"""

from importlib.metadata import version as _version

__all__ = ["__version__"]

try:
    __version__ = _version("endopigraph-ajmorph")
except Exception:  # pragma: no cover
    __version__ = "0.0.0"
