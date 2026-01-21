from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple


def ensure_dir(path: os.PathLike | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha1_of_file(path: os.PathLike | str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def read_json(path: os.PathLike | str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: os.PathLike | str, indent: int = 2) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent)


def parse_percentile(spec: str) -> Optional[float]:
    """Parse strings like 'percentile:95' into 95.0."""
    s = spec.strip().lower()
    if not s.startswith("percentile"):
        return None
    parts = s.split(":", 1)
    if len(parts) != 2:
        return None
    try:
        return float(parts[1])
    except ValueError:
        return None


def pair_code(a: int, b: int) -> int:
    """Pack two uint32 into one uint64-ish Python int."""
    if a > b:
        a, b = b, a
    return (int(a) << 32) | int(b)


def unpack_pair_code(code: int) -> Tuple[int, int]:
    a = (code >> 32) & 0xFFFFFFFF
    b = code & 0xFFFFFFFF
    return int(a), int(b)

