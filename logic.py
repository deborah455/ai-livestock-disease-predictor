# logic.py — Inference for Experiment 1 (Vision-only, PyTorch)
# Input: image file OR folder
# Output: annotated images + predictions.csv

# Silence optional backend warnings (safe; no accuracy change)
import os
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"  # hide NNPACK/Fbgemm warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
from pathlib import Path
from typing import List
import torch, torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v2
from PIL import Image, ImageDraw, ImageFont
import pandas as pd

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".PNG", ".JPEG"}

def list_images(p: Path) -> List[Path]:
    if p.is_file() and p.suffix in ALLOWED_EXT:
        return [p]
    if p.is_dir():
        return sorted([q for q in p.rglob('*') if q.suffix in ALLOWED_EXT], key=lambda x: x.as_posix())
    raise SystemExit(f"[!] Input path not found: {p}")

def build_preprocess():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def build_model_from_checkpoint(ckpt: Path) -> torch.nn.Module:
    if not ckpt.exists():
        raise SystemExit(f"[!] Checkpoint not found: {ckpt}")
    m = mobilenet_v2(weights=None)
    for p in m.features.parameters():
        p.requires_grad = False
    m.classifier[1] = nn.Linear(m.last_channel, 2)
    state = torch.load(ckpt, map_location="cpu")
    m.load_state_dict(state)
    m.eval()
    return m

def load_scripted(scripted_path: Path) -> torch.jit.ScriptModule:
    if not scripted_path.exists():
        raise SystemExit(f"[!] TorchScript file not found: {scripted_path}")
    m = torch.jit.load(str(scripted_path), map_location="cpu")
    m.eval()
    return m

def prob_unhealthy(model, img_tensor: torch.Tensor) -> float:
    with torch.no_grad():
        logits = model(img_tensor)         # [1,2]
        probs = torch.softmax(logits, 1)   # softmax over classes
        return float(probs.cpu().numpy()[0,1])  # class 1 = UNHEALTHY

def annotate(image_path: Path, out_path: Path, label: str, p: float, thresh: float):
    im = Image.open(image_path).convert("RGB")
    W, H = im.size
    d = ImageDraw.Draw(im)
    color = (220,20,60) if label=="UNHEALTHY" else (34,139,34)  # red/green
    t = max(4, min(W,H)//200)
    for i in range(t):
        d.rectangle([i,i,W-1-i,H-1-i], outline=color)
    bar = max(24, H//18)
    d.rectangle([0,0,W,bar], fill=color)
    txt = f"{label}  p_unhealthy={p:.3f}  (τ={thresh:.2f})"
    try: font = ImageFont.load_default()
    except: font = None
    d.text((8, max(2, bar//8)), txt, fill=(255,255,255), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)

def main():
    ap = argparse.ArgumentParser(description="Exp1 Vision Inference (PyTorch)")
    ap.add_argument("--input", required=True, help="Image file OR folder")
    ap.add_argument("--ckpt", default="results/vision/best.pth", help="Path to .pth checkpoint")
    ap.add_argument("--scripted", default="results/vision/model_scripted.pt", help="TorchScript .pt (optional)")
    ap.add_argument("--use_scripted", action="store_true", help="Use TorchScript instead of checkpoint")
    ap.add_argument("--out", default="results/vision_infer", help="Output folder")
    ap.add_argument("--threshold", type=float, default=0.50, help="Decision threshold τ")
    args = ap.parse_args()

    inp = Path(args.input)
    files = list_images(inp)
    if not files:
        raise SystemExit("[!] No images found at input.")

    # Load model
    if args.use_scripted:
        print(f"[i] Loading TorchScript: {args.scripted}")
        model = load_scripted(Path(args.scripted))
    else:
        print(f"[i] Loading checkpoint: {args.ckpt}")
        model = build_model_from_checkpoint(Path(args.ckpt))

    preprocess = build_preprocess()
    rows = []

    for i, p in enumerate(files, 1):
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            print(f"[!] Skipping unreadable file: {p} ({e})")
            continue
        x = preprocess(img).unsqueeze(0)  # [1,3,H,W]
        punh = prob_unhealthy(model, x)
        label = "UNHEALTHY" if punh >= args.threshold else "HEALTHY"

        rel = p.name if inp.is_file() else p.relative_to(inp)
        out_img = Path(args.out) / "annotated" / rel
        annotate(p, out_img, label, punh, args.threshold)

        rows.append({"index": i, "path": str(p), "p_unhealthy": round(punh,6), "prediction": label})
        print(f"[{i:04d}] {p.name}  p_unhealthy={punh:.3f}  → {label}")

    # Save CSV
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "predictions.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    unhealthy_cnt = sum(1 for r in rows if r["prediction"] == "UNHEALTHY")
    print(f"\n[i] Saved CSV → {csv_path}")
    print(f"[i] Summary: {unhealthy_cnt}/{len(rows)} flagged as UNHEALTHY (τ={args.threshold})")

if __name__ == "__main__":
    main()
