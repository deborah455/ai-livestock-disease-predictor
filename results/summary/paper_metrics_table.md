| Experiment | Acc | F1 | Prec | Rec | AUROC | TP | FP | FN | TN | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Exp 1 — Vision | 0.896 | 0.894 | 0.895 | 0.896 | 0.950 | nan | nan | nan | nan | Cattle skin (healthy vs LSD) |
| Exp 2 — Sensors | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | nan | nan | nan | nan | Synthetic; split by animal_id |
| Exp 2b — Fusion (best α=0.45) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 46.000 | 0.000 | 0.000 | 62.000 | Proxy paired slice (upper bound) |