# run_fusion_exp2b.py — late fusion of vision + sensors for Exp 2b
from pathlib import Path
import numpy as np, pandas as pd, json, matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

SEED=13
SENS = Path("data/exp2_sensors/processed/sensor_test_probs.csv")
VIS  = Path("data/exp2b_fusion/vision_probs.csv")
OUT  = Path("results/exp2b_fusion"); OUT.mkdir(parents=True, exist_ok=True)

# load
sens = pd.read_csv(SENS)   # columns: species, animal_id, t, label, p_sens
vis  = pd.read_csv(VIS)    # columns: path, p_vis, label

# (optional) focus on Cattle in sensors to better match vision domain
sens_cattle = sens[sens["species"]=="Cattle"].copy()
if len(sens_cattle) >= 100: sens = sens_cattle

# balance by class via pairing (pseudo-paired — for demonstration)
def pair_by_class(df_vis, df_sens, label):
    a = df_vis[df_vis["label"]==label].sample(frac=1, random_state=SEED).reset_index(drop=True)
    b = df_sens[df_sens["label"]==label].sample(frac=1, random_state=SEED).reset_index(drop=True)
    n = min(len(a), len(b))
    a, b = a.iloc[:n], b.iloc[:n]
    return pd.DataFrame({"label": label,
                         "p_vis": a["p_vis"].to_numpy()[:n],
                         "p_sens": b["p_sens"].to_numpy()[:n]})

pairs0 = pair_by_class(vis, sens, 0)
pairs1 = pair_by_class(vis, sens, 1)
pairs = pd.concat([pairs0, pairs1], ignore_index=True)
pairs = pairs.sample(frac=1, random_state=SEED).reset_index(drop=True)
print(f"Paired dataset size: {len(pairs)}  (class0={len(pairs0)}, class1={len(pairs1)})")

# split pairs for evaluation
X = pairs[["p_vis","p_sens"]].values
y = pairs["label"].values
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.30, random_state=SEED, stratify=y)

# 1) Weighted fusion sweep
grid = np.linspace(0.0, 1.0, 21)  # alpha grid
records=[]
for a in grid:
    p_fused = a*X_te[:,1] + (1-a)*X_te[:,0]
    try: auc = roc_auc_score(y_te, p_fused)
    except: auc = float("nan")
    # derive a 0.5 decision just to report a confusion matrix
    pred = (p_fused>=0.5).astype(int)
    rep  = classification_report(y_te, pred, output_dict=True, digits=4)
    records.append({"alpha":float(a), "auc":float(auc),
                    "f1":float(rep["weighted avg"]["f1-score"]),
                    "precision":float(rep["weighted avg"]["precision"]),
                    "recall":float(rep["weighted avg"]["recall"])})
rec_df = pd.DataFrame(records)
best_row = rec_df.sort_values(["auc","f1"], ascending=False).iloc[0]
best_alpha = float(best_row["alpha"])
print(f"Best alpha (AUC): {best_alpha:.2f} → AUC={best_row['auc']:.3f}, F1={best_row['f1']:.3f}")

# plot ROC curves for modalities and best fused
def plot_roc_curve(y_true, scores, title, path):
    fpr,tpr,_=roc_curve(y_true, scores)
    plt.figure(figsize=(4,4)); plt.plot(fpr,tpr); plt.plot([0,1],[0,1],'--')
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(title); plt.tight_layout(); plt.savefig(path); plt.close()

plot_roc_curve(y_te, X_te[:,0], "ROC (Vision only)", OUT/"roc_vis.png")
plot_roc_curve(y_te, X_te[:,1], "ROC (Sensors only)", OUT/"roc_sens.png")
p_fused_best = best_alpha*X_te[:,1] + (1-best_alpha)*X_te[:,0]
plot_roc_curve(y_te, p_fused_best, f"ROC (Fusion α={best_alpha:.2f})", OUT/"roc_fused.png")

# 2) Stacking (logistic regression on [p_vis, p_sens])
clf = LogisticRegression(random_state=SEED, max_iter=1000)
clf.fit(X_tr, y_tr)
p_stack = clf.predict_proba(X_te)[:,1]
auc_stack = roc_auc_score(y_te, p_stack)
pred_stack = (p_stack>=0.5).astype(int)
rep_stack  = classification_report(y_te, pred_stack, output_dict=True, digits=4)
cm_stack   = confusion_matrix(y_te, pred_stack)

# save summary
summary = {
  "pair_sizes": {"total": int(len(pairs)), "class0": int(len(pairs0)), "class1": int(len(pairs1))},
  "weighted_fusion": {
     "grid": grid.tolist(),
     "best_alpha": best_alpha,
     "best_auc": float(best_row["auc"]),
     "best_f1": float(best_row["f1"])
  },
  "stacking": {
     "auc": float(auc_stack),
     "report": rep_stack,
     "confusion_matrix": cm_stack.tolist()
  }
}
json.dump(summary, open(OUT/"fusion_summary.json","w"), indent=2)
rec_df.to_csv(OUT/"fusion_grid.csv", index=False)
print("Saved fusion results →", OUT)
