import json, os
from pathlib import Path
import pandas as pd

R = Path("results")
OUT = R / "summary"; OUT.mkdir(parents=True, exist_ok=True)

def read_vision():
    p = R/"vision"/"classification_report.json"
    if not p.exists(): return None
    d = json.load(open(p))
    rep = d["report"]
    return dict(
        exp="Exp 1 — Vision",
        acc=rep.get("accuracy", None),
        f1=rep.get("weighted avg",{}).get("f1-score", None),
        prec=rep.get("weighted avg",{}).get("precision", None),
        rec=rep.get("weighted avg",{}).get("recall", None),
        auc=d.get("auc", None),
        tp=None, fp=None, fn=None, tn=None, notes="Cattle skin (healthy vs LSD)"
    )

def read_sensors():
    p = R/"exp2_sensors"/"classification_report.json"
    if not p.exists(): return None
    d = json.load(open(p))
    rep = d["report"]
    return dict(
        exp="Exp 2 — Sensors",
        acc=rep.get("accuracy", None),
        f1=rep.get("weighted avg",{}).get("f1-score", None),
        prec=rep.get("weighted avg",{}).get("precision", None),
        rec=rep.get("weighted avg",{}).get("recall", None),
        auc=d.get("auc", None),
        tp=None, fp=None, fn=None, tn=None, notes="Synthetic; split by animal_id"
    )

def read_fusion():
    p = R/"exp2b_fusion"/"fusion_summary.json"
    if not p.exists(): return None
    d = json.load(open(p))
    # stacking has a confusion matrix + report we can use
    rep = d["stacking"]["report"]
    cm  = d["stacking"]["confusion_matrix"]  # [[tn, fp],[fn, tp]]
    tn, fp = cm[0]
    fn, tp = cm[1]
    return dict(
        exp=f"Exp 2b — Fusion (best α={d['weighted_fusion']['best_alpha']:.2f})",
        acc=rep.get("accuracy", None),
        f1=rep.get("weighted avg",{}).get("f1-score", None),
        prec=rep.get("weighted avg",{}).get("precision", None),
        rec=rep.get("weighted avg",{}).get("recall", None),
        auc=d["stacking"]["auc"],
        tp=tp, fp=fp, fn=fn, tn=tn,
        notes="Proxy paired slice (upper bound)"
    )

rows = []
for fn in (read_vision, read_sensors, read_fusion):
    x = fn()
    if x: rows.append(x)

df = pd.DataFrame(rows, columns=["exp","acc","f1","prec","rec","auc","tp","fp","fn","tn","notes"])

# Save CSV
csv_path = OUT/"paper_metrics_table.csv"
df.to_csv(csv_path, index=False)

# Save Markdown
def fmt(x):
    return "—" if x is None else (f"{x:.3f}" if isinstance(x,(float,int)) else str(x))
md = ["| Experiment | Acc | F1 | Prec | Rec | AUROC | TP | FP | FN | TN | Notes |",
      "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
for _,r in df.iterrows():
    md.append("| " + " | ".join([
        r["exp"],
        fmt(r["acc"]), fmt(r["f1"]), fmt(r["prec"]), fmt(r["rec"]), fmt(r["auc"]),
        fmt(r["tp"]), fmt(r["fp"]), fmt(r["fn"]), fmt(r["tn"]),
        r["notes"]
    ]) + " |")
(Path(OUT/"paper_metrics_table.md")).write_text("\n".join(md))

print(f"Wrote:\n- {csv_path}\n- {OUT/'paper_metrics_table.md'}")
print("Tip: run recompute_confusions.py to fill TP/FP/FN/TN for Exp1 & Exp2 if you need them.")
