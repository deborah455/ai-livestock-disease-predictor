# collect_vision_probs.py — merge vision inference CSVs into one file for fusion
from pathlib import Path
import pandas as pd

HCSV = Path("results/vision_infer_healthy/predictions.csv")
UCSV = Path("results/vision_infer_unhealthy/predictions.csv")
OUT  = Path("data/exp2b_fusion"); OUT.mkdir(parents=True, exist_ok=True)

def load(csv, label):
    df = pd.read_csv(csv)
    df = df.rename(columns={"p_unhealthy":"p_vis"})
    df["label"] = label
    # keep only what we need
    keep = ["path","p_vis","label"]
    return df[keep].copy()

vh = load(HCSV, 0)
vu = load(UCSV, 1)
allv = pd.concat([vh,vu], ignore_index=True)
allv.to_csv(OUT/"vision_probs.csv", index=False)
print("Wrote", len(allv), "rows →", OUT/"vision_probs.csv")
