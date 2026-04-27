"""Analyze blinded annotator submissions for AJMORPH.

Computes:
  - Cohen's kappa between annotators (with bootstrap CI)
  - Confusion matrices: A vs B, consensus vs PiMorph
  - Per-class precision/recall/F1 for PiMorph against consensus
  - Macro-F1 with bootstrap CI

Usage:
  python scripts/analyze_ajmorph_annotations.py \\
      --key runs/ajmorph_annotation/key_42.csv \\
      --A   sample_42_annotator_A.csv \\
      --B   sample_42_annotator_B.csv \\
      --out runs/ajmorph_annotation/report_42.md
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report, f1_score

def boot_kappa(a,b,n=1000,seed=0):
    rng=np.random.RandomState(seed)
    ks=[]
    for _ in range(n):
        idx=rng.choice(len(a),len(a),replace=True)
        ks.append(cohen_kappa_score(a[idx],b[idx]))
    return np.percentile(ks,[2.5,50,97.5])

def boot_f1(y,p,n=1000,seed=0):
    rng=np.random.RandomState(seed)
    fs=[]
    for _ in range(n):
        idx=rng.choice(len(y),len(y),replace=True)
        try: fs.append(f1_score(y[idx],p[idx],average="macro",zero_division=0))
        except: pass
    return np.percentile(fs,[2.5,50,97.5])

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--key",required=True)
    p.add_argument("--A",required=True)
    p.add_argument("--B",required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args()

    key=pd.read_csv(a.key)
    A=pd.read_csv(a.A);B=pd.read_csv(a.B)
    A=A.rename(columns=lambda c:c.replace("annotator_A_",""))
    B=B.rename(columns=lambda c:c.replace("annotator_B_",""))
    m=key[["blinded_id","aj_morph"]].rename(columns={"aj_morph":"pimorph"}).merge(
        A[["blinded_id","label"]].rename(columns={"label":"A"}),on="blinded_id").merge(
        B[["blinded_id","label"]].rename(columns={"label":"B"}),on="blinded_id")
    m=m[(m["A"].notna())&(m["B"].notna())].copy()
    n=len(m)
    if n==0:
        Path(a.out).write_text("# No annotated rows found.\n");return

    kappa=cohen_kappa_score(m["A"],m["B"])
    klo,kmed,khi=boot_kappa(m["A"].values,m["B"].values)

    m["consensus"]=np.where(m["A"]==m["B"],m["A"],"DISAGREE")
    agree=(m["A"]==m["B"]).sum()

    consen=m[m["consensus"]!="DISAGREE"]
    if len(consen)>0:
        macro=f1_score(consen["consensus"],consen["pimorph"],average="macro",zero_division=0)
        flo,fmed,fhi=boot_f1(consen["consensus"].values,consen["pimorph"].values)
        rep=classification_report(consen["consensus"],consen["pimorph"],zero_division=0)
    else:
        macro=flo=fmed=fhi=float("nan");rep="(no consensus rows)"

    cm=confusion_matrix(m["A"],m["B"],labels=sorted(set(m["A"])|set(m["B"])))
    cm_df=pd.DataFrame(cm,index=sorted(set(m["A"])|set(m["B"])),columns=sorted(set(m["A"])|set(m["B"])))

    md=f"""# AJMORPH manual annotation analysis

- N annotated: {n}
- Annotators agree on: {agree}/{n} ({agree/n:.1%})

## Inter-rater reliability
- Cohen's kappa = **{kappa:.3f}**  [95% CI {klo:.3f} - {khi:.3f}]
- Interpretation: <0.2 poor, 0.2-0.4 fair, 0.4-0.6 moderate, 0.6-0.8 substantial, >0.8 almost perfect.

## A vs B confusion matrix
{cm_df.to_markdown()}

## PiMorph vs consensus (only patches where annotators agreed; n={len(consen)})
- Macro-F1 = **{macro:.3f}**  [95% CI {flo:.3f} - {fhi:.3f}]

```
{rep}
```
"""
    Path(a.out).write_text(md)
    print(md)

if __name__=="__main__": main()
