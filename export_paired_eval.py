from pathlib import Path
import numpy as np, pandas as pd, json
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

SEED=13
sens = pd.read_csv("data/exp2_sensors/processed/sensor_test_probs.csv")  # label, p_sens (+species/animal_id/t)
vis  = pd.read_csv("data/exp2b_fusion/vision_probs.csv")                 # label, p_vis (+path)

# optional: restrict sensors to Cattle to match vision domain better
sens_cattle = sens[sens["species"]=="Cattle"]
if len(sens_cattle)>=100: sens = sens_cattle

def pair_by_class(dv, ds, y):
    a = dv[dv.label==y].sample(frac=1, random_state=SEED).reset_index(drop=True)
    b = ds[ds.label==y].sample(frac=1, random_state=SEED).reset_index(drop=True)
    n=min(len(a),len(b)); a=a.iloc[:n]; b=b.iloc[:n]
    return pd.DataFrame({"label":y, "p_vis":a.p_vis.to_numpy(), "p_sens":b.p_sens.to_numpy()})

pairs = pd.concat([pair_by_class(vis,sens,0), pair_by_class(vis,sens,1)], ignore_index=True)
pairs = pairs.sample(frac=1, random_state=SEED).reset_index(drop=True)
pairs.to_csv("results/exp2b_fusion/paired_eval.csv", index=False)

# split, compute AUCs
X = pairs[["p_vis","p_sens"]].values; y = pairs["label"].values
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.30, random_state=SEED, stratify=y)
auc_vis  = roc_auc_score(y_te, X_te[:,0])
auc_sens = roc_auc_score(y_te, X_te[:,1])

# load fusion summary to get best alpha
d = json.load(open("results/exp2b_fusion/fusion_summary.json"))
a = float(d["weighted_fusion"]["best_alpha"])
import numpy as np
p_fused = a*X_te[:,1] + (1-a)*X_te[:,0]
auc_fused = roc_auc_score(y_te, p_fused)

print(f"AUC (vision) : {auc_vis:.3f}")
print(f"AUC (sensors): {auc_sens:.3f}")
print(f"AUC (fused)  : {auc_fused:.3f}  @ alpha={a:.2f}")
