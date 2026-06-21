# simulate_sensors.py — Experiment 2 dataset generator
# Output: data/exp2_sensors/raw/exp2_sensor_samples.csv

import numpy as np, pandas as pd, random
from pathlib import Path
rng = np.random.default_rng(13)
random.seed(13)

OUT = Path("data/exp2_sensors/raw"); OUT.mkdir(parents=True, exist_ok=True)
CSV = OUT / "exp2_sensor_samples.csv"

species_list = ["Cattle","Goat","Sheep","Pig","Chicken"]
# Per-species baselines (rough, for simulation only — not clinical references)
BASE = {
    "Cattle":  dict(temp=38.5,  hr=60,  rr=24, rum=40, act=0.55),
    "Goat":    dict(temp=39.1,  hr=85,  rr=25, rum=35, act=0.60),
    "Sheep":   dict(temp=39.0,  hr=75,  rr=24, rum=35, act=0.58),
    "Pig":     dict(temp=38.6,  hr=75,  rr=25, rum=0,  act=0.50),
    "Chicken": dict(temp=41.2,  hr=280, rr=30, rum=0,  act=0.65),
}

N_ANIMALS_PER_SPECIES = 40        # total animals per species
HOURS_PER_ANIMAL = 72             # samples per animal (e.g., 3 days hourly)
P_UNHEALTHY = 0.30                # prevalence per sample (episodic)

rows = []
for sp in species_list:
    base = BASE[sp]
    for a in range(N_ANIMALS_PER_SPECIES):
        animal_id = f"{sp[:2].upper()}_{a:03d}"
        # small per-animal offsets
        off_temp = rng.normal(0, 0.15)
        off_hr   = rng.normal(0, 3.0)
        off_rr   = rng.normal(0, 1.5)
        off_act  = rng.normal(0, 0.05)
        off_rum  = rng.normal(0, 3.0)

        # simulate ambient profile (daily cycle)
        ambient_base = 22 + (5 if sp in ["Pig","Chicken"] else 0)
        for t in range(HOURS_PER_ANIMAL):
            hour = t % 24
            amb_temp = ambient_base + 6*np.sin(2*np.pi*hour/24.0) + rng.normal(0, 1.0)
            amb_hum  = np.clip(65 + 10*np.sin(2*np.pi*(hour-4)/24.0) + rng.normal(0, 5.0), 30, 95)

            # health state — with mild temporal persistence
            unhealthy = 1 if rng.uniform() < P_UNHEALTHY else 0
            if t>0 and rows and rows[-1]["animal_id"]==animal_id and rows[-1]["label"]==1:
                if rng.uniform()<0.5: unhealthy = 1  # persistence

            # base signals
            temp = base["temp"] + off_temp + 0.02*(amb_temp-22) + rng.normal(0, 0.1)
            hr   = base["hr"]   + off_hr   + 0.2*(amb_temp-22)    + rng.normal(0, 2.0)
            rr   = base["rr"]   + off_rr   + 0.1*(amb_temp-22)    + rng.normal(0, 1.0)
            act  = np.clip(base["act"] + off_act + rng.normal(0, 0.05), 0, 1)
            rum  = max(0, base["rum"] + off_rum + rng.normal(0, 2.0))
            cough= max(0, int(rng.poisson(0.2)))

            if unhealthy:
                # fever & respiratory pattern + behavior changes
                temp += 1.0 + rng.normal(0, 0.2)
                hr   += 0.20*base["hr"] + rng.normal(0, 3.0)
                rr   += 0.30*base["rr"] + rng.normal(0, 1.5)
                act   = np.clip(act - 0.25 + rng.normal(0, 0.05), 0, 1)
                cough = max(0, int(rng.poisson(3.0)))
                if sp in ["Cattle","Goat","Sheep"]:
                    rum = max(0, rum - 15 + rng.normal(0, 3.0))

            rows.append(dict(
                species=sp, animal_id=animal_id, t=t,
                body_temp_c=round(float(temp),3),
                heart_rate_bpm=round(float(hr),2),
                resp_rate_bpm=round(float(rr),2),
                activity_idx=round(float(act),3),
                cough_count=int(cough),
                rumination_min=round(float(rum),2),
                ambient_temp_c=round(float(amb_temp),2),
                ambient_humidity_pct=round(float(amb_hum),2),
                label=int(unhealthy)
            ))

df = pd.DataFrame(rows)
df.to_csv(CSV, index=False)
print(f"Wrote {len(df)} rows to {CSV}")
print(df.head(5))
