"""
04_model_custom_cnn.py
======================
Custom CNN training script for the PlantVillage leaf-disease dataset.

Features vs. the original notebook:
  - Checkpoint / resume  : saves full training state every epoch; resumes
                           automatically if a checkpoint is found.
  - Plain-text progress  : no tqdm.notebook; uses tqdm (terminal) + explicit
                           print() lines so every line is visible in SLURM logs.
  - Flush-on-print       : all prints go through a helper that flushes stdout
                           immediately (important when PYTHONUNBUFFERED is set).

Usage
-----
  python 04_model_custom_cnn.py            # start / resume automatically
  python 04_model_custom_cnn.py --reset    # ignore any checkpoint, start fresh
"""

import argparse
import json
import multiprocessing
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe on headless nodes
import matplotlib.pyplot as plt
import seaborn as sns


# ── Helpers ───────────────────────────────────────────────────────────────────

def log(msg: str = "") -> None:
    """Print with immediate flush so lines appear in SLURM logs right away."""
    print(msg, flush=True)


def separator(char: str = "─", width: int = 60) -> None:
    log(char * width)


# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument(
    "--reset",
    action="store_true",
    help="Ignore existing checkpoint and train from scratch.",
)
args = parser.parse_args()


# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR    = os.environ.get("LEAF_PRED_ROOT", os.path.expanduser("~/leaf-pred"))
PROC_DIR    = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LOGS_DIR    = os.path.join(BASE_DIR, "logs")
MODEL_NAME  = "custom_cnn"

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR,    exist_ok=True)

BEST_MODEL_PATH  = os.path.join(MODELS_DIR, f"{MODEL_NAME}_best.pth")
CHECKPOINT_PATH  = os.path.join(MODELS_DIR, f"{MODEL_NAME}_checkpoint.pth")


# ── Startup banner ────────────────────────────────────────────────────────────

separator("=")
log("  NOTEBOOK 04 — Custom CNN  (script version with checkpointing)")
separator("=")
log(f"  BASE_DIR    : {BASE_DIR}")
log(f"  MODELS_DIR  : {MODELS_DIR}")
log(f"  RESULTS_DIR : {RESULTS_DIR}")
log(f"  Start time  : {time.strftime('%Y-%m-%d %H:%M:%S')}")
separator("=")
log()


# ── Device ────────────────────────────────────────────────────────────────────

log(f"PyTorch : {torch.__version__}")
log(f"CUDA    : {torch.cuda.is_available()}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"Device  : {DEVICE}")
if DEVICE.type == "cuda":
    log(f"GPU     : {torch.cuda.get_device_name(0)}")
    log(f"VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
torch.backends.cudnn.benchmark = True

if DEVICE.type != "cuda":
    log()
    log("ERROR: CUDA is not available — training on CPU would take ~20h per run.")
    log("       Check that --nv is passed to singularity exec in the SLURM script.")
    log("       Exiting now to avoid wasting the job slot.")
    sys.exit(1)
log()


# ── Config ────────────────────────────────────────────────────────────────────

with open(os.path.join(PROC_DIR, "config.json"),         "r") as f: config   = json.load(f)
with open(os.path.join(PROC_DIR, "class_manifest.json"), "r") as f: manifest = json.load(f)
with open(os.path.join(PROC_DIR, "dataset_stats.json"),  "r") as f: stats    = json.load(f)

manifest_df = pd.DataFrame(manifest)
N_CLASSES   = config["n_classes"]
IMG_SIZE    = config["img_size"]
MEAN_RGB    = stats["mean_rgb"]
STD_RGB     = stats["std_rgb"]

log(f"N_CLASSES : {N_CLASSES}")
log(f"IMG_SIZE  : {IMG_SIZE}")
log()


# ── Dataset & DataLoaders ─────────────────────────────────────────────────────

class LeafDiseaseDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths     = paths
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN_RGB, std=STD_RGB),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
])

eval_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN_RGB, std=STD_RGB),
])

all_paths  = np.load(os.path.join(PROC_DIR, "all_paths.npy"),  allow_pickle=True)
all_labels = np.load(os.path.join(PROC_DIR, "all_labels.npy"), allow_pickle=True).astype(int)
train_idx  = np.load(os.path.join(PROC_DIR, "train_idx.npy"))
val_idx    = np.load(os.path.join(PROC_DIR, "val_idx.npy"))
test_idx   = np.load(os.path.join(PROC_DIR, "test_idx.npy"))

_NUM_WORKERS   = min(16, max(1, multiprocessing.cpu_count() - 1))  # 16 is plenty; more just wastes RAM & I/O
BATCH_SIZE     = 512   # A100 80GB can handle this easily; drop to 256 if you see OOM
_PREFETCH      = 4     # batches to prepare ahead per worker

_DL_KWARGS = dict(
    batch_size=BATCH_SIZE,
    num_workers=_NUM_WORKERS,
    pin_memory=True,
    persistent_workers=(_NUM_WORKERS > 0),
    prefetch_factor=_PREFETCH if _NUM_WORKERS > 0 else None,
)

train_loader = DataLoader(
    LeafDiseaseDataset(all_paths[train_idx], all_labels[train_idx], train_transforms),
    shuffle=True,  **_DL_KWARGS)

val_loader = DataLoader(
    LeafDiseaseDataset(all_paths[val_idx], all_labels[val_idx], eval_transforms),
    shuffle=False, **_DL_KWARGS)

test_loader = DataLoader(
    LeafDiseaseDataset(all_paths[test_idx], all_labels[test_idx], eval_transforms),
    shuffle=False, **_DL_KWARGS)

log(f"Train batches : {len(train_loader)}  ({len(train_idx)} images)")
log(f"Val   batches : {len(val_loader)}  ({len(val_idx)} images)")
log(f"Test  batches : {len(test_loader)}  ({len(test_idx)} images)")
log(f"Batch size    : {BATCH_SIZE}   num_workers: {_NUM_WORKERS}   prefetch: {_PREFETCH}")
log()


# ── Model ─────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Two conv layers with BN, ReLU, MaxPool, and Dropout."""
    def __init__(self, in_ch, out_ch, dropout=0.25):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.block(x)


class CustomCNN(nn.Module):
    def __init__(self, n_classes=38):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3,   32,  dropout=0.10),
            ConvBlock(32,  64,  dropout=0.15),
            ConvBlock(64,  128, dropout=0.20),
            ConvBlock(128, 256, dropout=0.25),
            ConvBlock(256, 512, dropout=0.30),
        )
        self.gap        = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x


model = CustomCNN(n_classes=N_CLASSES).to(DEVICE)

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
log(f"Total parameters     : {total_params:,}")
log(f"Trainable parameters : {trainable_params:,}")
log()


# ── Loss / Optimiser / Scheduler ──────────────────────────────────────────────

class_weights  = np.load(os.path.join(PROC_DIR, "class_weights.npy"))
weights_tensor = torch.FloatTensor(class_weights).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=weights_tensor)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3)

log("Loss      : CrossEntropyLoss (class-weighted)")
log("Optimizer : AdamW  (lr=1e-3, weight_decay=1e-4)")
log("Scheduler : ReduceLROnPlateau (factor=0.5, patience=3)")
log()


# ── AMP ───────────────────────────────────────────────────────────────────────

_amp_enabled = (DEVICE.type == "cuda")
scaler       = torch.cuda.amp.GradScaler(enabled=_amp_enabled)
_autocast    = lambda: torch.cuda.amp.autocast(enabled=_amp_enabled)
log(f"Mixed precision (AMP) : {'enabled' if _amp_enabled else 'disabled'}")
log()


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(epoch: int, best_val_loss: float, patience_count: int, history: dict) -> None:
    """Save full training state so the job can resume after being killed."""
    torch.save({
        "epoch":          epoch,
        "model_state":    model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state":   scaler.state_dict(),
        "best_val_loss":  best_val_loss,
        "patience_count": patience_count,
        "history":        history,
    }, CHECKPOINT_PATH)


def load_checkpoint():
    """
    Returns (start_epoch, best_val_loss, patience_count, history) loaded from
    disk, or the initial values if no checkpoint exists / --reset was passed.
    """
    if args.reset:
        log("--reset flag set: ignoring any existing checkpoint.")
        return 1, float("inf"), 0, {"train_loss": [], "train_acc": [],
                                     "val_loss":   [], "val_acc":   []}

    if not os.path.isfile(CHECKPOINT_PATH):
        log("No checkpoint found — starting from epoch 1.")
        return 1, float("inf"), 0, {"train_loss": [], "train_acc": [],
                                     "val_loss":   [], "val_acc":   []}

    log(f"Checkpoint found at: {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    scaler.load_state_dict(ckpt["scaler_state"])

    start_epoch    = ckpt["epoch"] + 1           # resume AFTER the saved epoch
    best_val_loss  = ckpt["best_val_loss"]
    patience_count = ckpt["patience_count"]
    history        = ckpt["history"]

    log(f"  Resuming from epoch {start_epoch}  "
        f"(best_val_loss={best_val_loss:.4f}, patience={patience_count})")
    return start_epoch, best_val_loss, patience_count, history


# ── Train / Eval loops ────────────────────────────────────────────────────────

def train_epoch(loader):
    model.train()
    total = 0

    # Accumulate loss/correct as GPU tensors — no per-batch CPU sync
    running_loss_t = torch.tensor(0.0, device=DEVICE)
    correct_t      = torch.tensor(0,   device=DEVICE)

    bar = tqdm(loader, desc="  Train", leave=False, unit="batch",
               dynamic_ncols=True, file=sys.stdout)
    for imgs, labels in bar:
        imgs, labels = (imgs.to(DEVICE, non_blocking=True),
                        labels.to(DEVICE, non_blocking=True))
        optimizer.zero_grad(set_to_none=True)
        with _autocast():
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss_t += loss.detach() * imgs.size(0)
        correct_t      += (outputs.detach().argmax(1) == labels).sum()
        total          += imgs.size(0)
        # tqdm postfix update: sync once per batch is acceptable for display
        bar.set_postfix(loss=f"{running_loss_t.item()/total:.4f}",
                        acc=f"{correct_t.item()/total:.4f}")

    bar.close()
    # Single CPU sync at end of epoch
    return running_loss_t.item() / total, correct_t.item() / total


def eval_epoch(loader, desc="  Val  "):
    model.eval()
    total = 0

    running_loss_t = torch.tensor(0.0, device=DEVICE)
    correct_t      = torch.tensor(0,   device=DEVICE)

    bar = tqdm(loader, desc=desc, leave=False, unit="batch",
               dynamic_ncols=True, file=sys.stdout)
    with torch.no_grad():
        for imgs, labels in bar:
            imgs, labels = (imgs.to(DEVICE, non_blocking=True),
                            labels.to(DEVICE, non_blocking=True))
            with _autocast():
                outputs = model(imgs)
                loss    = criterion(outputs, labels)
            running_loss_t += loss * imgs.size(0)
            correct_t      += (outputs.argmax(1) == labels).sum()
            total          += imgs.size(0)
            bar.set_postfix(loss=f"{running_loss_t.item()/total:.4f}",
                            acc=f"{correct_t.item()/total:.4f}")

    bar.close()
    return running_loss_t.item() / total, correct_t.item() / total


# ── Training loop ─────────────────────────────────────────────────────────────

N_EPOCHS       = 50
EARLY_STOP_PAT = 7

start_epoch, best_val_loss, patience_count, history = load_checkpoint()

separator()
log(f"  Training  |  epochs {start_epoch} → {N_EPOCHS}  |  early-stop patience={EARLY_STOP_PAT}")
separator()
log()

for epoch in range(start_epoch, N_EPOCHS + 1):
    t0 = time.time()

    log(f"[Epoch {epoch:02d}/{N_EPOCHS}]  {time.strftime('%H:%M:%S')}  — training...")
    train_loss, train_acc = train_epoch(train_loader)

    log(f"[Epoch {epoch:02d}/{N_EPOCHS}]  {time.strftime('%H:%M:%S')}  — validating...")
    val_loss, val_acc = eval_epoch(val_loader)

    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]["lr"]

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    elapsed = time.time() - t0

    # ── Per-epoch summary line ──
    log(
        f"[Epoch {epoch:02d}/{N_EPOCHS}]  "
        f"tr_loss={train_loss:.4f}  tr_acc={train_acc:.4f}  "
        f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
        f"lr={current_lr:.2e}  time={elapsed:.1f}s"
    )

    # ── Best model ──
    if val_loss < best_val_loss:
        best_val_loss  = val_loss
        patience_count = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        log(f"  ✓ New best model saved  (val_loss={best_val_loss:.4f})")
    else:
        patience_count += 1
        log(f"  ✗ No improvement  (patience {patience_count}/{EARLY_STOP_PAT})")
        if patience_count >= EARLY_STOP_PAT:
            log(f"\nEarly stopping triggered at epoch {epoch} "
                f"(no improvement for {EARLY_STOP_PAT} epochs).")
            # Save checkpoint before exiting so results are safe
            save_checkpoint(epoch, best_val_loss, patience_count, history)
            break

    # ── Checkpoint every epoch ──
    save_checkpoint(epoch, best_val_loss, patience_count, history)
    log(f"  Checkpoint saved  → {CHECKPOINT_PATH}")
    log()

log("\nTraining complete.")
separator()
log()


# ── Plot training curves ──────────────────────────────────────────────────────

log("Plotting training curves...")
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
epochs_ran = range(1, len(history["train_loss"]) + 1)

axes[0].plot(epochs_ran, history["train_loss"], label="Train", color="steelblue")
axes[0].plot(epochs_ran, history["val_loss"],   label="Val",   color="tomato")
axes[0].set_title(f"{MODEL_NAME} — Loss")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].legend()

axes[1].plot(epochs_ran, history["train_acc"], label="Train", color="steelblue")
axes[1].plot(epochs_ran, history["val_acc"],   label="Val",   color="tomato")
axes[1].set_title(f"{MODEL_NAME} — Accuracy")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy"); axes[1].legend()

plt.tight_layout()
curves_path = os.path.join(RESULTS_DIR, f"04_{MODEL_NAME}_curves.png")
plt.savefig(curves_path, dpi=150)
plt.close()
log(f"  Saved → {curves_path}")
log()


# ── Evaluate on test set ──────────────────────────────────────────────────────

log("Loading best model weights for test evaluation...")
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
model.eval()

all_preds, all_targets = [], []
bar = tqdm(test_loader, desc="Testing", unit="batch",
           dynamic_ncols=True, file=sys.stdout)
with torch.no_grad():
    for imgs, labels in bar:
        imgs  = imgs.to(DEVICE)
        preds = model(imgs).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(labels.numpy())
bar.close()

all_preds   = np.array(all_preds)
all_targets = np.array(all_targets)
test_acc    = (all_preds == all_targets).mean()
log(f"Test Accuracy : {test_acc:.4f}  ({test_acc * 100:.2f}%)")
log()


# ── Confusion matrix ──────────────────────────────────────────────────────────

log("Generating confusion matrix...")
class_names = [
    row["disease"][:15]
    for _, row in manifest_df.sort_values("class_idx").iterrows()
]

cm = confusion_matrix(all_targets, all_preds)
fig, ax = plt.subplots(figsize=(16, 14))
sns.heatmap(cm, annot=False, cmap="Blues", ax=ax,
            xticklabels=class_names, yticklabels=class_names)
ax.set_title(f"{MODEL_NAME} — Confusion Matrix (Test Set)", fontsize=12)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
plt.xticks(rotation=90, fontsize=7); plt.yticks(rotation=0, fontsize=7)
plt.tight_layout()
cm_path = os.path.join(RESULTS_DIR, f"04_{MODEL_NAME}_confusion_matrix.png")
plt.savefig(cm_path, dpi=150)
plt.close()
log(f"  Saved → {cm_path}")
log()


# ── Per-class metrics ─────────────────────────────────────────────────────────

log("Per-class classification report:")
report = classification_report(
    all_targets, all_preds,
    target_names=class_names,
    output_dict=True,
)
report_df = pd.DataFrame(report).T
log(report_df.round(3).to_string())
log()


# ── Save results JSON ─────────────────────────────────────────────────────────

results = {
    "model_name":    MODEL_NAME,
    "test_accuracy": float(test_acc),
    "best_val_loss": float(best_val_loss),
    "history":       history,
    "n_params":      trainable_params,
    "per_class":     report,
}
results_path = os.path.join(PROC_DIR, f"results_{MODEL_NAME}.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
log(f"Results JSON saved → {results_path}")
log()


# ── Final summary ─────────────────────────────────────────────────────────────

separator("=")
log("  NOTEBOOK 04 — COMPLETE")
separator("=")
log(f"  Model            : Custom CNN (from scratch)")
log(f"  Parameters       : {trainable_params:,}")
log(f"  Test Accuracy    : {test_acc * 100:.2f}%")
log(f"  Best Val Loss    : {best_val_loss:.4f}")
log(f"  Epochs trained   : {len(history['train_loss'])}")
log()
log("  Saved to models/:")
log(f"    {MODEL_NAME}_best.pth")
log(f"    {MODEL_NAME}_checkpoint.pth")
log()
log("  Saved to results/:")
log(f"    04_{MODEL_NAME}_curves.png")
log(f"    04_{MODEL_NAME}_confusion_matrix.png")
log()
log("  Saved to data/processed/:")
log(f"    results_{MODEL_NAME}.json")
log()
log("  Next: Notebook 05 — Model B: ResNet50 Transfer Learning")
separator("=")
