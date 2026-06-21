# run_sensors.py — Train/Eval sensor-only baseline (Exp-2)
# Input: data/exp2_sensors/raw/exp2_sensor_samples.csv
# Output: results/exp2_sensors/{classification_report.json, confusion_matrix.png, roc.png, feature_importance.png, model_rf.joblib}

import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import joblib

SEED=13
RAW = Path("data/exp2_sensors/raw/exp2_sensor_samples.csv")
OUT = Path("results/exp2_sensors"); OUT.mkdir(parents=True, exist_ok=True)

# Load
df = pd.read_csv(RAW)
# Features
FEATS = ["body_temp_c","heart_rate_bpm","resp_rate_bpm","activity_idx","cough_count",
         "rumination_min","ambient_temp_c","ambient_humidity_pct"]
X = df[FEATS].copy()
y = df["label"].values
animal = df["animal_id"].values

# Split by animal_id to avoid leakage
animals = np.unique(animal)
train_an, temp_an = train_test_split(animals, test_size=0.30, random_state=SEED, shuffle=True)
val_an, test_an   = train_test_split(temp_an, test_size=0.50, random_state=SEED, shuffle=True)

def mask_from(an_ids): return np.isin(animal, an_ids)
tr_mask, va_mask, te_mask = mask_from(train_an), mask_from(val_an), mask_from(test_an)
Xtr, Xva, Xte = X[tr_mask], X[va_mask], X[te_mask]
ytr, yva, yte = y[tr_mask], y[va_mask], y[te_mask]

# Pipeline: scale (for smoothness) + RF
pipe = Pipeline([
    ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ("rf", RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        class_weight="balanced", random_state=SEED, n_jobs=-1))
])

pipe.fit(Xtr, ytr)
proba_va = pipe.predict_proba(Xva)[:,1]
proba_te = pipe.predict_proba(Xte)[:,1]
pred_te  = (proba_te >= 0.5).astype(int)

rep = classification_report(yte, pred_te, output_dict=True, digits=4)
cm  = confusion_matrix(yte, pred_te)
try: auc = float(roc_auc_score(yte, proba_te))
except: auc = float("nan")

# Save report
json.dump({"auc":auc, "report":rep,
           "sizes":{"train":int(tr_mask.sum()),"val":int(va_mask.sum()),"test":int(te_mask.sum())}},
          open(OUT/"classification_report.json","w"), indent=2)

# Confusion matrix
fig, ax = plt.subplots(figsize=(4,4))
im = ax.imshow(cm, interpolation='nearest')
ax.set_title("Confusion Matrix (Sensors)"); ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(["healthy","unhealthy"]); ax.set_yticklabels(["healthy","unhealthy"])
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j,i,cm[i,j],ha="center",va="center")
ax.set_xlabel("Predicted"); ax.set_ylabel("True"); fig.tight_layout(); fig.savefig(OUT/"confusion_matrix.png"); plt.close(fig)

# ROC
fpr,tpr,_=roc_curve(yte, proba_te)
plt.figure(figsize=(4,4)); plt.plot(fpr,tpr); plt.plot([0,1],[0,1],'--')
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC (AUC={auc:.3f})")
plt.tight_layout(); plt.savefig(OUT/"roc.png"); plt.close()

# Feature importance (mean decrease in impurity)
rf = pipe.named_steps["rf"]
importances = rf.feature_importances_
order = np.argsort(importances)[::-1]
plt.figure(figsize=(6,4))
plt.bar(range(len(FEATS)), importances[order])
plt.xticks(range(len(FEATS)), [FEATS[i] for i in order], rotation=30, ha="right")
plt.ylabel("Importance"); plt.title("Feature Importance (RF)")
plt.tight_layout(); plt.savefig(OUT/"feature_importance.png"); plt.close()

# Save model
joblib.dump(pipe, OUT/"model_rf.joblib")
print(f"Saved results to {OUT}")
