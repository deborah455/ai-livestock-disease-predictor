# Experiment 1 (Vision-only) — PyTorch CPU
# Folders expected:
#   data/images/healthy/     data/images/unhealthy/
import json, math, random, os
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# ----------------- config -----------------
SEED = 13
IMG_SIZE = 224
BATCH = 32
EPOCHS = 12
OUT = Path("results/vision"); OUT.mkdir(parents=True, exist_ok=True)
H_DIR = Path("data/images/healthy")
U_DIR = Path("data/images/unhealthy")
DEVICE = "cpu"  # force CPU
# -----------------------------------------

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# Collect files
def list_images(root: Path):
    exts = {".jpg",".jpeg",".png",".bmp",".JPG",".PNG",".JPEG"}
    return [str(p) for p in root.rglob("*") if p.suffix in exts]

healthy = list_images(H_DIR)
unhealthy = list_images(U_DIR)
if len(healthy)==0 or len(unhealthy)==0:
    raise SystemExit("Put images into data/images/healthy and data/images/unhealthy, then rerun.")

X = np.array(healthy + unhealthy)
y = np.array([0]*len(healthy) + [1]*len(unhealthy), dtype=np.int64)

# 70/15/15 stratified split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
X_val,   X_test, y_val, y_test   = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED)

# Transforms (use ImageNet normalization; fall back if pretrained weights unavailable)
IMAGENET_MEAN = [0.485,0.456,0.406]
IMAGENET_STD  = [0.229,0.224,0.225]

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.9,1.0)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

class ImageList(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = list(paths)
        self.labels = list(labels)
        self.transform = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), int(self.labels[i])

ds_train = ImageList(X_train, y_train, train_tf)
ds_val   = ImageList(X_val,   y_val,   eval_tf)
ds_test  = ImageList(X_test,  y_test,  eval_tf)

dl_train = DataLoader(ds_train, batch_size=BATCH, shuffle=True,  num_workers=2, pin_memory=False)
dl_val   = DataLoader(ds_val,   batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=False)
dl_test  = DataLoader(ds_test,  batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=False)

print(f"Total images: {len(X)} (healthy={len(healthy)}, unhealthy={len(unhealthy)})")
print(f"Split sizes → train={len(ds_train)}, val={len(ds_val)}, test={len(ds_test)}")

# Load model
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
try:
    weights = MobileNet_V2_Weights.IMAGENET1K_V1
    model = mobilenet_v2(weights=weights)
    print("Loaded MobileNetV2 with ImageNet weights.")
except Exception as e:
    weights = None
    model = mobilenet_v2(weights=None)
    print("Could not load pretrained weights; training from scratch:", e)

# Freeze backbone, replace classifier
for p in model.features.parameters():
    p.requires_grad = False
model.classifier[1] = nn.Linear(model.last_channel, 2)

model = model.to(DEVICE)

# Class weights (handle imbalance)
counts = Counter(y_train)
total = sum(counts.values())
w0 = total/(2.0*counts.get(0,1)); w1 = total/(2.0*counts.get(1,1))
ce_weight = torch.tensor([w0, w1], dtype=torch.float32, device=DEVICE)
criterion = nn.CrossEntropyLoss(weight=ce_weight)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4)

# Training with early stopping on val accuracy
best_acc = 0.0
patience = 3
stale = 0

def epoch_run(dataloader, train=True):
    if train: model.train()
    else:     model.eval()
    total, correct, running_loss = 0, 0, 0.0
    all_logits, all_labels = [], []
    with torch.set_grad_enabled(train):
        for xb, yb in dataloader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = criterion(logits, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            running_loss += loss.item()*xb.size(0)
            preds = logits.argmax(1)
            correct += (preds==yb).sum().item()
            total += xb.size(0)
            all_logits.append(logits.detach().cpu())
            all_labels.append(yb.detach().cpu())
    avg_loss = running_loss/max(1,total)
    acc = correct/max(1,total)
    logits = torch.cat(all_logits) if all_logits else torch.empty(0,2)
    labels = torch.cat(all_labels) if all_labels else torch.empty(0, dtype=torch.long)
    return avg_loss, acc, logits, labels

for epoch in range(1, EPOCHS+1):
    tr_loss, tr_acc, _, _ = epoch_run(dl_train, train=True)
    va_loss, va_acc, _, _  = epoch_run(dl_val,   train=False)
    print(f"Epoch {epoch:02d}: train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f}")

    if va_acc > best_acc:
        best_acc = va_acc
        stale = 0
        torch.save(model.state_dict(), OUT/"best.pth")
    else:
        stale += 1
        if stale >= patience:
            print("Early stopping.")
            break

# Load best & evaluate on test
model.load_state_dict(torch.load(OUT/"best.pth", map_location=DEVICE))
model.eval()
_, _, test_logits, test_labels = epoch_run(dl_test, train=False)
y_true = test_labels.numpy()
probs = torch.softmax(test_logits, dim=1).numpy()
y_pred = probs.argmax(1)
y_score = probs[:,1] if probs.shape[1]==2 else probs[:,0]  # binary score

# Metrics & plots
rep = classification_report(y_true, y_pred, output_dict=True, digits=4)
cm  = confusion_matrix(y_true, y_pred)
try: auc = float(roc_auc_score(y_true, y_score))
except Exception: auc = float("nan")

with open(OUT/"classification_report.json","w") as f:
    json.dump({"auc":auc, "report":rep, "class_weight":[float(w0), float(w1)],
               "sizes":{"train":len(ds_train),"val":len(ds_val),"test":len(ds_test)}}, f, indent=2)

# Confusion matrix
fig, ax = plt.subplots(figsize=(4,4))
im = ax.imshow(cm, interpolation='nearest')
ax.set_title("Confusion Matrix"); ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(["healthy","unhealthy"]); ax.set_yticklabels(["healthy","unhealthy"])
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j,i,cm[i,j],ha="center",va="center")
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
fig.tight_layout(); fig.savefig(OUT/"confusion_matrix.png"); plt.close(fig)

# ROC
fpr, tpr, _ = roc_curve(y_true, y_score)
plt.figure(figsize=(4,4)); plt.plot(fpr,tpr); plt.plot([0,1],[0,1],'--')
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC (AUC={auc:.3f})")
plt.tight_layout(); plt.savefig(OUT/"roc.png"); plt.close()

# Save model (state dict + TorchScript)
torch.save(model.state_dict(), OUT/"model.pth")
try:
    example = torch.randn(1,3,IMG_SIZE,IMG_SIZE)
    scripted = torch.jit.trace(model, example)
    scripted.save(str(OUT/"model_scripted.pt"))
except Exception as e:
    print("TorchScript export failed (ok to ignore):", e)

print("\n[Vision/PyTorch] Done. See results in", OUT)
