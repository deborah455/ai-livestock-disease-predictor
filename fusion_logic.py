# fusion_logic.py — Exp 2b (Vision + Sensors) late-fusion inference and metrics
# - Inputs: CSV with vision probabilities, CSV with sensor probabilities
# - Modes:
#    * pseudo_pair  (default): pair within each class (healthy/unhealthy) — proxy eval
#    * join         : join rows by a key column (e.g., animal_id+t) when you have real alignment
#    * zip          : fuse rows by order (only if both CSVs are one-to-one in the same order)
# - Output: fused CSV with p_vis, p_sens, p_fused, prediction; optional ROC & confusion matrix

import os
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")  # quiet backend warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

SEED = 13
rng  = np.random.default_rng(SEED)

def load_vision(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Normalize column names
    if "p_vis" not in df.columns:
        if "p_unhealthy" in df.columns:
            df = df.rename(columns={"p_unhealthy": "p_vis"})
        else:
            raise SystemExit("Vision CSV must have 'p_vis' or 'p_unhealthy' column.")
    # Optional label for evaluation (0/1). If missing, we can still fuse.
    if "label" in df.columns:
        df["label"] = df["label"].astype(int)
    return df

def load_sensor(csv_path: Path, restrict_species=None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "p_sens" not in df.columns:
        # allow alternate names
        if "proba" in df.columns:
            df = df.rename(columns={"proba": "p_sens"})
        else:
            raise SystemExit("Sensor CSV must have 'p_sens' column (or 'proba').")
    if "label" in df.columns:
        df["label"] = df["label"].astype(int)
    if restrict_species and "species" in df.columns:
        df = df[df["species"] == restrict_species].reset_index(drop=True)
    return df

def pseudo_pair(vision: pd.DataFrame, sensors: pd.DataFrame) -> pd.DataFrame:
    """Pair samples by class (healthy/unhealthy) without time alignment. Proxy only."""
    if "label" not in vision.columns or "label" not in sensors.columns:
        raise SystemExit("Pseudo-pairing needs 'label' column in both CSVs.")
    out = []
    for y in (0,1):
        va = vision[vision["label"]==y].sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        sb = sensors[sensors["label"]==y].sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        n = min(len(va), len(sb))
        if n == 0:
            continue
        dfy = pd.DataFrame({
            "label": y,
            "p_vis": va["p_vis"].to_numpy()[:n],
            "p_sens": sb["p_sens"].to_numpy()[:n]
        })
        out.append(dfy)
    if not out:
        raise SystemExit("No overlapping classes to pair.")
    pairs = pd.concat(out, ignore_index=True)
    pairs = pairs.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    return pairs

def join_on(vision: pd.DataFrame, sensors: pd.DataFrame, v_key: str, s_key: str) -> pd.DataFrame:
    if v_key not in vision.columns or s_key not in sensors.columns:
        raise SystemExit(f"Join keys not found. Vision has {vision.columns.tolist()}, Sensors has {sensors.columns.tolist()}")
    merged = vision.merge(sensors, left_on=v_key, right_on=s_key, how="inner", suffixes=("_vis","_sens"))
    # prefer 'label_vis' if available, else 'label_sens'
    if "label_vis" in merged.columns:
        merged = merged.rename(columns={"label_vis":"label"})
    elif "label_sens" in merged.columns:
        merged = merged.rename(columns={"label_sens":"label"})
    return merged

def zip_fuse(vision: pd.DataFrame, sensors: pd.DataFrame) -> pd.DataFrame:
    n = min(len(vision), len(sensors))
    if n == 0:
        raise SystemExit("Nothing to zip. One of the CSVs is empty.")
    v = vision.iloc[:n].reset_index(drop=True)
    s = sensors.iloc[:n].reset_index(drop=True)
    df = pd.DataFrame({"p_vis": v["p_vis"], "p_sens": s["p_sens"]})
    if "label" in v.columns:
        df["label"] = v["label"].astype(int)
    elif "label" in s.columns:
        df["label"] = s["label"].astype(int)
    return df

def compute_and_save_metrics(y_true, scores, thr, outdir: Path, tag: str):
    outdir.mkdir(parents=True, exist_ok=True)
    pred = (scores >= thr).astype(int)
    rep  = classification_report(y_true, pred, output_dict=True, digits=4)
    cm   = confusion_matrix(y_true, pred)
    try:
        auc = float(roc_auc_score(y_true, scores))
    except Exception:
        auc = float("nan")
    # save json
    json.dump({"auc": auc, "report": rep, "threshold": thr},
              open(outdir/f"classification_report_{tag}.json","w"), indent=2)
    # confusion matrix plot
    fig, ax = plt.subplots(figsize=(4,4))
    ax.imshow(cm); ax.set_title(f"Confusion Matrix ({tag})")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["healthy","unhealthy"]); ax.set_yticklabels(["healthy","unhealthy"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j,i,cm[i,j],ha="center",va="center")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.tight_layout(); fig.savefig(outdir/f"confusion_matrix_{tag}.png"); plt.close(fig)
    # ROC plot
    fpr, tpr, _ = roc_curve(y_true, scores)
    plt.figure(figsize=(4,4)); plt.plot(fpr,tpr); plt.plot([0,1],[0,1],'--')
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC ({tag}) AUC={auc:.3f}")
    plt.tight_layout(); plt.savefig(outdir/f"roc_{tag}.png"); plt.close()
    return auc, rep

def main():
    ap = argparse.ArgumentParser(description="Exp 2b Fusion logic (vision + sensors)")
    ap.add_argument("--vision_csv", default="data/exp2b_fusion/vision_probs.csv",
                    help="CSV with p_vis (and optional label)")
    ap.add_argument("--sensor_csv", default="data/exp2_sensors/processed/sensor_test_probs.csv",
                    help="CSV with p_sens (and optional label)")
    ap.add_argument("--mode", choices=["pseudo_pair","join","zip"], default="pseudo_pair",
                    help="How to align rows for fusion")
    ap.add_argument("--join_on_vision_col", default=None, help="Column in vision CSV to join on (mode=join)")
    ap.add_argument("--join_on_sensor_col", default=None, help="Column in sensor CSV to join on (mode=join)")
    ap.add_argument("--species_filter", default=None, help="If set, filter sensor CSV to this species (e.g., Cattle)")
    ap.add_argument("--alpha", type=float, default=0.45, help="Weight on sensors (0..1)")
    ap.add_argument("--threshold", type=float, default=0.50, help="Decision threshold on p_fused")
    ap.add_argument("--out", default="results/exp2b_fusion_infer", help="Output folder")
    args = ap.parse_args()

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    vis = load_vision(Path(args.vision_csv))
    sens = load_sensor(Path(args.sensor_csv), restrict_species=args.species_filter)

    if args.mode == "pseudo_pair":
        df = pseudo_pair(vis, sens)
    elif args.mode == "join":
        if not args.join_on_vision_col or not args.join_on_sensor_col:
            raise SystemExit("mode=join requires --join_on_vision_col and --join_on_sensor_col.")
        df = join_on(vis, sens, args.join_on_vision_col, args.join_on_sensor_col)
        # ensure standardized column names
        if "p_vis" not in df.columns and "p_vis_vis" in df.columns:
            df = df.rename(columns={"p_vis_vis":"p_vis"})
        if "p_sens" not in df.columns and "p_sens_sens" in df.columns:
            df = df.rename(columns={"p_sens_sens":"p_sens"})
        if "label" in df.columns:
            df["label"] = df["label"].astype(int)
    else:  # zip
        df = zip_fuse(vis, sens)

    # Compute fused score
    a = float(args.alpha)
    df["p_fused"] = a*df["p_sens"] + (1.0 - a)*df["p_vis"]
    df["prediction"] = (df["p_fused"] >= args.threshold).astype(int)

    # Save fused CSV
    fused_csv = outdir/"fusion_predictions.csv"
    keep_cols = [c for c in ["label","p_vis","p_sens","p_fused","prediction"] if c in df.columns]
    df[keep_cols].to_csv(fused_csv, index=False)
    print(f"[i] Saved fused CSV → {fused_csv}  (rows={len(df)})")

    # Metrics & plots if labels exist
    if "label" in df.columns:
        auc, rep = compute_and_save_metrics(df["label"].values, df["p_fused"].values,
                                            args.threshold, outdir, tag="fusion")
        # Also record per-modality AUCs for the *same paired/joined* rows
        try:
            auc_vis  = float(roc_auc_score(df["label"].values, df["p_vis"].values))
            auc_sens = float(roc_auc_score(df["label"].values, df["p_sens"].values))
        except Exception:
            auc_vis, auc_sens = float("nan"), float("nan")

        summary = {
            "alpha": a,
            "threshold": args.threshold,
            "rows": int(len(df)),
            "auc_fused": auc,
            "auc_vision_on_slice": auc_vis,
            "auc_sensors_on_slice": auc_sens,
            "report_f1_weighted": rep["weighted avg"]["f1-score"]
        }
        json.dump(summary, open(outdir/"fusion_logic_summary.json","w"), indent=2)
        print(f"[i] AUCs on this slice → vision={auc_vis:.3f}  sensors={auc_sens:.3f}  fused={auc:.3f}")
        print(f"[i] Weighted F1 (τ={args.threshold:.2f}) = {rep['weighted avg']['f1-score']:.3f}")
    else:
        print("[i] No 'label' column found; skipped metrics/plots (fused probabilities only).")

if __name__ == "__main__":
    main()
