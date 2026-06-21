# demo_cli.py
# Interactive CLI for Exp 1 (vision), Exp 2 (sensors), and Exp 2b (fusion)

import os, sys, json, math
from pathlib import Path

# --------- Imports for Vision ----------
import torch
import torchvision as tv
from PIL import Image

# --------- Imports for Sensors ----------
import joblib
import numpy as np

# ---------- Paths (adjust if needed) ----------
VISION_CKPT = Path("results/vision/best.pth")
SENSORS_MODEL = Path("results/exp2_sensors/model_rf.joblib")

# ---------- Vision: model + transform ----------
def load_vision_model():
    if not VISION_CKPT.exists():
        raise FileNotFoundError(f"Missing vision checkpoint: {VISION_CKPT}")
    model = tv.models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    state = torch.load(VISION_CKPT, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    tfm = tv.transforms.Compose([
        tv.transforms.Resize((224,224)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return model, tfm

@torch.no_grad()
def vision_predict_one(model, tfm, img_path: Path):
    img = Image.open(img_path).convert("RGB")
    x = tfm(img).unsqueeze(0)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    p_healthy, p_unhealthy = float(probs[0]), float(probs[1])
    return p_unhealthy, p_healthy

def iter_images(path: Path):
    exts = {".jpg",".jpeg",".png",".bmp",".webp"}
    if path.is_file() and path.suffix.lower() in exts:
        yield path
    elif path.is_dir():
        for p in sorted(path.rglob("*")):
            if p.suffix.lower() in exts:
                yield p
    else:
        raise FileNotFoundError(f"No images at: {path}")

# ---------- Sensors: model + features ----------
SENSOR_FEATURES = [
    "body_temp_c","heart_rate_bpm","resp_rate_bpm","activity_idx",
    "cough_count","rumination_min","ambient_temp_c","ambient_humidity_pct"
]

def load_sensor_model():
    if not SENSORS_MODEL.exists():
        raise FileNotFoundError(f"Missing sensor model: {SENSORS_MODEL}")
    pipe = joblib.load(SENSORS_MODEL)
    return pipe

def prompt_float(msg, default=None):
    while True:
        raw = input(f"{msg} " + (f"[default {default}]: " if default is not None else ": "))
        if raw.strip()=="" and default is not None:
            return float(default)
        try:
            return float(raw)
        except:
            print("Enter a number.")

def prompt_int(msg, default=None):
    while True:
        raw = input(f"{msg} " + (f"[default {default}]: " if default is not None else ": "))
        if raw.strip()=="" and default is not None:
            return int(default)
        try:
            return int(raw)
        except:
            print("Enter an integer.")

def get_sensor_vector_interactive():
    print("\nEnter sensor values (typical healthy ranges in brackets):")
    bt   = prompt_float("Body temperature °C [37.5–39.5]", 38.3)
    hr   = prompt_int  ("Heart rate bpm (ruminants ~48–84)", 70)
    rr   = prompt_int  ("Resp rate bpm (ruminants ~10–30)", 22)
    act  = prompt_float("Activity index 0–1", 0.6)
    cough= prompt_int  ("Cough count (per hour)", 0)
    rum  = prompt_float("Rumination minutes (ruminants) per hour", 40.0)
    ambt = prompt_float("Ambient temperature °C", 27.0)
    ambh = prompt_float("Ambient humidity %", 65.0)
    x = np.array([[bt, hr, rr, act, cough, rum, ambt, ambh]], dtype=float)
    return x

# ---------- Decision helpers ----------
def decide(p, tau):
    return "UNHEALTHY" if p >= tau else "HEALTHY"

def clamp01(x): 
    return max(0.0, min(1.0, float(x)))

# ---------- Modes ----------
def mode_vision():
    print("\n[Vision Mode — Exp 1]")
    path_str = input("Enter image FILE path or FOLDER path: ").strip().strip('"').strip("'")
    tau = prompt_float("Decision threshold τ for UNHEALTHY [0–1]", 0.50)
    tau = clamp01(tau)

    model, tfm = load_vision_model()
    p = Path(path_str)
    total=0; unhealthy=0
    for imgp in iter_images(p):
        pu, ph = vision_predict_one(model, tfm, imgp)
        pred = decide(pu, tau)
        print(f"[{imgp.name}] p_unhealthy={pu:.3f} → {pred}")
        total += 1
        unhealthy += (pred=="UNHEALTHY")
    if total>1:
        print(f"\nSummary: {unhealthy}/{total} flagged as UNHEALTHY at τ={tau:.2f}")

def mode_sensors():
    print("\n[Sensors Mode — Exp 2]")
    tau = prompt_float("Decision threshold τ for UNHEALTHY [0–1]", 0.50)
    tau = clamp01(tau)
    pipe = load_sensor_model()
    x = get_sensor_vector_interactive()
    p_unhealthy = float(pipe.predict_proba(x)[:,1][0])
    pred = decide(p_unhealthy, tau)
    print(f"\nSensor probability p_unhealthy={p_unhealthy:.3f} → {pred}")

def mode_fusion():
    print("\n[Fusion Mode — Exp 2b]")
    alpha = prompt_float("Fusion weight α (sensor weight) [0–1]", 0.45)
    tau   = prompt_float("Decision threshold τ for UNHEALTHY [0–1]", 0.50)
    alpha = clamp01(alpha); tau = clamp01(tau)

    # vision part
    path_str = input("Enter image FILE path (for vision): ").strip().strip('"').strip("'")
    model, tfm = load_vision_model()
    imgp = Path(path_str)
    if not imgp.exists():
        print("Image path not found."); return
    p_vis, _p_h = vision_predict_one(model, tfm, imgp)

    # sensor part
    pipe = load_sensor_model()
    x = get_sensor_vector_interactive()
    p_sens = float(pipe.predict_proba(x)[:,1][0])

    # fuse
    p_fused = alpha*p_sens + (1.0-alpha)*p_vis
    pv = p_vis; ps = p_sens; pf = p_fused
    pred = decide(p_fused, tau)

    print("\n--- Fusion Summary ---")
    print(f"p_vis  (image)   = {pv:.3f}")
    print(f"p_sens (sensors) = {ps:.3f}")
    print(f"α (sensor weight)= {alpha:.2f}")
    print(f"p_fused          = α*ps + (1-α)*pv = {pf:.3f}")
    print(f"τ (threshold)    = {tau:.2f}")
    print(f"Decision         → {pred}")

def main():
    print("Interactive Health Scoring (Exp 1, Exp 2, Exp 2b)")
    while True:
        print("\nChoose mode:")
        print("  1) Vision (image or folder)  [Exp 1]")
        print("  2) Sensors (type values)     [Exp 2]")
        print("  3) Fusion (image + values)   [Exp 2b]")
        print("  4) Quit")
        choice = input("Enter 1/2/3/4: ").strip()
        try:
            if choice == "1": mode_vision()
            elif choice == "2": mode_sensors()
            elif choice == "3": mode_fusion()
            elif choice == "4": break
            else: print("Pick 1, 2, 3, or 4.")
        except FileNotFoundError as e:
            print(f"[Error] {e}")
        except Exception as e:
            print(f"[Unexpected] {e}")

if __name__ == "__main__":
    main()
