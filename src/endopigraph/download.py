from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Literal, Optional

import requests


def fetch_bioimagearchive_info(accession: str, timeout_s: int = 60) -> dict:
    """Fetch BioStudies/BioImage Archive study info as JSON.

    This uses the endpoint:
        https://www.ebi.ac.uk/biostudies/api/v1/studies/<accession>/info

    The response typically contains an `ftpLink` field pointing to the root folder.
    """
    url = f"https://www.ebi.ac.uk/biostudies/api/v1/studies/{accession}/info"
    r = requests.get(url, timeout=timeout_s)
    r.raise_for_status()
    return r.json()


def download_study(accession: str, out_dir: str | Path, method: Literal["print", "wget"] = "print") -> dict:
    """Download (or print download commands) for a BioImage Archive study.

    Parameters
    ----------
    accession:
        Study accession, e.g. 'S-BIAD1540'.
    out_dir:
        Output directory where files should land.
    method:
        - 'print': do not download; just print ftpLink and recommended commands.
        - 'wget': use wget to mirror the `Files/` directory.

    Notes
    -----
    For large studies, `wget` can be slow or unreliable. BioImage Archive recommends Aspera for large transfers.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    info = fetch_bioimagearchive_info(accession)
    ftp_link = info.get("ftpLink") or info.get("ftp") or None
    if not ftp_link:
        raise RuntimeError(f"No ftpLink found in info for {accession}. Keys: {list(info.keys())}")

    print(f"Study: {accession}")
    print(f"ftpLink: {ftp_link}")

    files_link = ftp_link.rstrip("/") + "/Files/"
    target = out_dir / accession
    target.mkdir(parents=True, exist_ok=True)

    if method == "print":
        print("\nRecommended download commands (choose one):\n")
        print("# 1) wget (simple but may be slow for many files)")
        print(f"wget -r -np -nH --cut-dirs=100 -P {target} {files_link}")
        print("\n# 2) Aspera (recommended for big studies; see BioImage Archive docs)")
        print("# Example (requires IBM Aspera CLI + EBI key file):")
        print(f"ascp -QT -l 300m -P 33001 -i <path-to-aspera-key> era-fasp@fasp.ebi.ac.uk:{files_link.replace('ftp://ftp.ebi.ac.uk','')} {target}")
        return info

    if method == "wget":
        cmd = [
            "wget",
            "-r",
            "-np",
            "-nH",
            "--cut-dirs=100",
            "-P",
            str(target),
            files_link,
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        return info

    raise ValueError(f"Unknown method: {method}")
