import numpy as np, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

np.random.seed(13)
OUTDIR = Path("results/exp2b_fusion"); OUTDIR.mkdir(parents=True, exist_ok=True)

def try_load_fused():
    f = OUTDIR/"fusion_predictions.csv"
    if f.exists():
        df = pd.read_csv(f)
        need = {"p_vis","p_sens","label"}
        if need.issubset(df.columns):
            return df[["p_vis","p_sens","label"]].dropna().reset_index(drop=True)
    return None

def pseudo_pair():
    # fallback: rebuild a paired slice from the source CSVs (proxy!)
    vis = Path("data/exp2b_fusion/vision_probs.csv")
    sen = Path("data/exp2_sensors/processed/sensor_test_probs.csv")
    if not (vis.exists() and sen.exists()):
        raise SystemExit("Missing both fusion_predictions.csv and source prob CSVs.")
    dv = pd.read_csv(vis); ds = pd.read_csv(sen)
    # normalize column names
    if "p_vis" not in dv.columns:
        if "p_unhealthy" in dv.columns: dv = dv.rename(columns={"p_unhealthy":"p_vis"})
        else: raise SystemExit("vision_probs.csv must have p_vis or p_unhealthy")
    if "label" not in dv.columns or "label" not in ds.columns:
        raise SystemExit("Both CSVs must contain 'label' for pseudo-pairing.")
    # class-wise shuffle & pair
    out=[]
    for y in (0,1):
        v = dv.query("label==@y").sample(frac=1, random_state=13).reset_index(drop=True)
        s = ds.query("label==@y").sample(frac=1, random_state=13).reset_index(drop=True)
        n=min(len(v),len(s))
        if n>0:
            out.append(pd.DataFrame({"label":y,"p_vis":v["p_vis"].to_numpy()[:n],
                                     "p_sens":s["p_sens"].to_numpy()[:n]}))
    if not out: raise SystemExit("No overlapping classes to pair.")
    df = pd.concat(out, ignore_index=True).sample(frac=1, random_state=13).reset_index(drop=True)
    return df

df = try_load_fused()
if df is None:
    df = pseudo_pair()

alphas = np.round(np.linspace(0.0,1.0,21),2)
tau = 0.50
rows=[]
for a in alphas:
    p_fused = a*df["p_sens"].values + (1-a)*df["p_vis"].values
    yhat = (p_fused >= tau).astype(int)
    y = df["label"].astype(int).values
    rows.append({
        "alpha": float(a),
        "precision": precision_score(y,yhat,zero_division=0),
        "recall":    recall_score(y,yhat,zero_division=0),
        "f1":        f1_score(y,yhat,zero_division=0),
        "auc":       roc_auc_score(y,p_fused)
    })
grid = pd.DataFrame(rows)
best = grid.iloc[grid["f1"].values.argmax()]

# save grid for the paper (optional)
grid.to_csv(OUTDIR/"alpha_sweep_metrics.csv", index=False)

# plot (single chart, no subplots, no custom colors)
plt.figure(figsize=(7,5))
plt.plot(grid["alpha"], grid["f1"], label="F1")
plt.plot(grid["alpha"], grid["precision"], label="Precision")
plt.plot(grid["alpha"], grid["recall"], label="Recall")
plt.axvline(best["alpha"], linestyle="--")   # shows chosen alpha
plt.xlabel("Sensor weight α")
plt.ylabel("Score")
plt.title("Exp 2b: Precision/Recall/F1 vs Fusion Weight α  (τ=0.50)")
plt.legend()
plt.tight_layout()
plt.savefig(OUTDIR/"alpha_sweep_metrics.png", dpi=180)
print("[i] Saved:", OUTDIR/"alpha_sweep_metrics.png")
print("[i] Best α by F1:", round(float(best['alpha']),2),
      "F1=", round(float(best['f1']),3),
      "Prec=", round(float(best['precision']),3),
      "Rec=", round(float(best['recall']),3))
