"""Sample interfaces stratified by AJMORPH state for blinded expert annotation.

Outputs:
  runs/ajmorph_annotation/sample_<seed>.csv  - patch list + blinded ID + empty annotator columns
  runs/ajmorph_annotation/patches/<blinded_id>.png  - copied patches under blinded filenames
  runs/ajmorph_annotation/key_<seed>.csv  - blinded->true mapping (DO NOT share with annotators)
  runs/ajmorph_annotation/INSTRUCTIONS.md  - annotation protocol

Two annotators independently label each patch using the controlled vocabulary; we then
compute Cohen's kappa, consensus labels, and consensus-vs-PiMorph macro-F1. The 400-patch
default gives ~80 patches per state with five states, sufficient for kappa CIs of width <0.1.

Usage:
  python scripts/sample_ajmorph_annotation.py --n 400 --seed 42
"""
from __future__ import annotations
import argparse, hashlib, shutil
from pathlib import Path
import pandas as pd

VOCAB=["straight","thick","reticular","fingers","thick_to_reticular","other","unclassifiable"]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--patch-index",default="runs/sbiad1540_full/patches/patch_index.csv")
    p.add_argument("--out",default="runs/ajmorph_annotation")
    p.add_argument("--n",type=int,default=400)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--per-state-min",type=int,default=40)
    a=p.parse_args()

    df=pd.read_csv(a.patch_index)
    df=df[df["aj_morph"].notna()].reset_index(drop=True)
    out=Path(a.out);(out/"patches").mkdir(parents=True,exist_ok=True)

    rng=__import__("numpy").random.RandomState(a.seed)
    # stratified by aj_morph
    parts=[]
    for st,g in df.groupby("aj_morph"):
        k=max(a.per_state_min,int(round(a.n*len(g)/len(df))))
        k=min(k,len(g))
        parts.append(g.sample(n=k,random_state=a.seed))
    sample=pd.concat(parts).sample(frac=1,random_state=a.seed).reset_index(drop=True)
    if len(sample)>a.n: sample=sample.head(a.n)

    sample["blinded_id"]=[hashlib.sha256(f"{a.seed}:{i}".encode()).hexdigest()[:12] for i in range(len(sample))]
    for _,r in sample.iterrows():
        src=Path(r["patch_path"])
        if src.exists():
            shutil.copy(src,out/"patches"/f"{r['blinded_id']}.png")

    blind=sample[["blinded_id"]].copy()
    blind["annotator_A_label"]=""
    blind["annotator_A_confidence_1to5"]=""
    blind["annotator_A_notes"]=""
    blind["annotator_B_label"]=""
    blind["annotator_B_confidence_1to5"]=""
    blind["annotator_B_notes"]=""
    blind.to_csv(out/f"sample_{a.seed}.csv",index=False)

    sample.to_csv(out/f"key_{a.seed}.csv",index=False)

    (out/"INSTRUCTIONS.md").write_text(f"""# AJMORPH annotation protocol

## Task
Classify each VE-cadherin junction patch into one of these classes:

{chr(10).join("- **"+v+"**" for v in VOCAB)}

## Class definitions
- **straight**: continuous, linear junction with high VE-cadherin coverage along a single axis
- **thick**: continuous and visibly broader than straight; high occupancy spread perpendicular to the contact axis
- **reticular**: fragmented, web/network-like pattern with multiple bright pieces and gaps
- **fingers**: sparse, finger-like protrusions; elongated thin elements with low overall occupancy
- **thick_to_reticular**: transitional appearance with both continuous-thick and fragmented regions
- **other**: legitimate junction not fitting the above categories
- **unclassifiable**: image quality, crop artifact, or no recognizable junction visible

## Procedure
1. Open `sample_{a.seed}.csv`. Each row corresponds to `patches/<blinded_id>.png`.
2. View patches in any image viewer. Do **not** look up the original source image.
3. Fill in `annotator_<X>_label` (one of the values above), `annotator_<X>_confidence_1to5` (1=guess, 5=very confident), and optional `annotator_<X>_notes`.
4. Save your completed CSV as `sample_{a.seed}_annotator_<NAME>.csv`.
5. Do NOT compare with the other annotator until both submissions are received.

## Blinding
- `blinded_id` is a SHA-256 hash of (seed, position). True image/cell IDs are in `key_{a.seed}.csv` and must not be shared with annotators until annotations are complete.
- The PiMorph-assigned label is also in the key file; do not consult it during annotation.

## Analysis
After both annotators submit, run:
  python scripts/analyze_ajmorph_annotations.py --sample sample_{a.seed}.csv --A annotator_X.csv --B annotator_Y.csv

Outputs Cohen's kappa, per-class F1, consensus labels, and consensus-vs-PiMorph macro-F1.
""")

    counts=sample["aj_morph"].value_counts().to_dict()
    print(f"Sampled {len(sample)} patches into {out}")
    print("Per-state counts:",counts)
    print(f"Annotator CSV: {out}/sample_{a.seed}.csv")
    print(f"Key file (DO NOT SHARE): {out}/key_{a.seed}.csv")

if __name__=="__main__": main()
