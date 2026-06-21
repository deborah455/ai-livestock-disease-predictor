# export_sensor_probs.py — save sensor test-set probabilities for Exp 2b fusion
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split

SEED=13
RAW = Path("data/exp2_sensors/raw/exp2_sensor_samples.csv")
MODEL = Path("results/exp2_sensors/model_rf.joblib")
OUT = Path("data/exp2_sensors/processed"); OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW)
FEATS = ["body_temp_c","heart_rate_bpm","resp_rate_bpm","activity_idx","cough_count",
         "rumination_min","ambient_temp_c","ambient_humidity_pct"]
X = df[FEATS].copy()
y = df["label"].values
animal = df["animal_id"].values

animals = np.unique(animal)
train_an, temp_an = train_test_split(animals, test_size=0.30, random_state=SEED, shuffle=True)
val_an, test_an   = train_test_split(temp_an,  test_size=0.50, random_state=SEED, shuffle=True)

def mask(an_ids): return np.isin(animal, an_ids)
te_mask = mask(test_an)
Xte = X[te_mask]; yte = y[te_mask]
df_te = df.loc[te_mask, ["species","animal_id","t","label"]].reset_index(drop=True)

pipe = joblib.load(MODEL)
p_sens = pipe.predict_proba(Xte)[:,1]
out = df_te.copy()
out["p_sens"] = p_sens
out.to_csv(OUT/"sensor_test_probs.csv", index=False)
print(f"Wrote {len(out)} rows → {OUT/'sensor_test_probs.csv'}")
