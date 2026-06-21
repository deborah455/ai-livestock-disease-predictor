from pathlib import Path
import json, numpy as np, pandas as pd

# -------- Exp 2 (sensors): trivial to recompute from saved model --------
def recompute_exp2():
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix
    import joblib
    RAW = Path("data/exp2_sensors/raw/exp2_sensor_samples.csv")
    MODEL = Path("results/exp2_sensors/model_rf.joblib")
    if not RAW.exists() or not MODEL.exists():
        return None
    df = pd.read_csv(RAW)
    FEATS = ["body_temp_c","heart_rate_bpm","resp_rate_bpm","activity_idx","cough_count",
             "rumination_min","ambient_temp_c","ambient_humidity_pct"]
    X = df[FEATS].copy()
    y = df["label"].astype(int).values
    animal = df["animal_id"].values
    SEED=13
    animals = np.unique(animal)
    train_an, temp_an = train_test_split(animals, test_size=0.30, random_state=SEED, shuffle=True)
    val_an, test_an   = train_test_split(temp_an,  test_size=0.50, random_state=SEED, shuffle=True)
    te_mask = np.isin(animal, test_an)
    Xte, yte = X[te_mask], y[te_mask]
    pipe = joblib.load(MODEL)
    pred = (pipe.predict_proba(Xte)[:,1] >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
    return dict(TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn))

# -------- Exp 1 (vision): re-eval test using saved checkpoint --------
def recompute_exp1():
    import torch, torchvision as tv
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix
    from torch.utils.data import Subset

    data_dir = Path("data/images")
    ckpt = Path("results/vision/best.pth")
    if not data_dir.exists() or not ckpt.exists():
        return None

    # dataset
    tfm = tv.transforms.Compose([
        tv.transforms.Resize((224,224)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    ds = tv.datasets.ImageFolder(str(data_dir), transform=tfm)  # class_to_idx: {'healthy':0,'unhealthy':1}

    # stratified split by label (reproducible)
    labels = np.array([y for _,y in ds.samples])
    idx = np.arange(len(labels))
    SEED=13
    idx_train, idx_temp = train_test_split(idx, test_size=0.30, random_state=SEED, stratify=labels)
    idx_val, idx_test   = train_test_split(idx_temp, test_size=0.50, random_state=SEED,
                                           stratify=labels[idx_temp])

    test_loader = torch.utils.data.DataLoader(Subset(ds, idx_test), batch_size=32, shuffle=False)

    # model (MobileNetV2 head with 2 classes)
    model = tv.models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    ys=[]; ps=[]
    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            ys.append(y.numpy()); ps.append(pred)
    y_true = np.concatenate(ys); y_pred = np.concatenate(ps)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return dict(TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn))

def main():
    OUT = Path("results/summary"); OUT.mkdir(parents=True, exist_ok=True)
    conf = {"exp1": recompute_exp1(), "exp2": recompute_exp2()}
    json.dump(conf, open(OUT/"confusions.json","w"), indent=2)
    print("Wrote", OUT/"confusions.json")
    # Optional: update the Markdown table if it exists
    md = OUT/"paper_metrics_table.md"
    if md.exists():
        txt = md.read_text()
        j = conf
        def repl(tag, val):
            return txt.replace(tag, str(val)) if val is not None else txt
        if j["exp1"]:
            txt = repl("[TP1]", j["exp1"]["TP"])
            txt = repl("[FP1]", j["exp1"]["FP"])
            txt = repl("[FN1]", j["exp1"]["FN"])
            txt = repl("[TN1]", j["exp1"]["TN"])
        if j["exp2"]:
            txt = repl("[TP2]", j["exp2"]["TP"])
            txt = repl("[FP2]", j["exp2"]["FP"])
            txt = repl("[FN2]", j["exp2"]["FN"])
            txt = repl("[TN2]", j["exp2"]["TN"])
        md.write_text(txt)
        print("Patched confusion counts into", md)
    else:
        print("Note: run autofill_tables.py first to create the Markdown table.")

if __name__=="__main__":
    main()
