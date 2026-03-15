"""
07_model_vit.py
===============
ViT-B/16 transfer learning — two-phase fine-tuning on PlantVillage.

Phase 1 : freeze backbone, train classification head only    (10 epochs, patience 4)
Phase 2 : unfreeze encoder layers 8-11 + head, diff LRs     (20 epochs, patience 6)

Note on batch size: ViT-B/16 has higher memory usage than CNNs due to the
self-attention O(n²) scaling. On A100 80GB, BATCH_SIZE=256 is safe. The
original notebook used 16 (designed for a T4 with 16GB).

Also saves attention maps from the last encoder layer after test evaluation.

Usage
-----
  python 07_model_vit.py            # start / resume automatically
  python 07_model_vit.py --reset    # ignore checkpoints, start fresh
"""

import argparse
import json
import multiprocessing
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


# ── Helpers ───────────────────────────────────────────────────────────────────

def log(msg=""):
    print(msg, flush=True)

def separator(char="─", width=60):
    log(char * width)


# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--reset", action="store_true",
                    help="Ignore existing checkpoints and train from scratch.")
parser.add_argument("--eval-only", action="store_true",
                    help="Skip training. Load best weights and run test "
                         "evaluation, attention maps, and results saving.")
args = parser.parse_args()


# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR    = os.environ.get("LEAF_PRED_ROOT", os.path.expanduser("~/leaf-pred"))
PROC_DIR    = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODEL_NAME  = "vit_b16"

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(MODELS_DIR, f"{MODEL_NAME}_best.pth")
CKPT_P1         = os.path.join(MODELS_DIR, f"{MODEL_NAME}_ckpt_phase1.pth")
CKPT_P2         = os.path.join(MODELS_DIR, f"{MODEL_NAME}_ckpt_phase2.pth")


# ── Banner ────────────────────────────────────────────────────────────────────

separator("=")
log("  07 — ViT-B/16 Transfer Learning  (2-phase fine-tuning)")
separator("=")
log(f"  BASE_DIR    : {BASE_DIR}")
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
    log("ERROR: CUDA not available — refusing to train on CPU.")
    log("       Add --nv to singularity exec in the SLURM script.")
    sys.exit(1)

log()


# ── Config ────────────────────────────────────────────────────────────────────

with open(os.path.join(PROC_DIR, "config.json"),         "r") as f: config   = json.load(f)
with open(os.path.join(PROC_DIR, "class_manifest.json"), "r") as f: manifest = json.load(f)
with open(os.path.join(PROC_DIR, "dataset_stats.json"),  "r") as f: stats    = json.load(f)

manifest_df = pd.DataFrame(manifest)
N_CLASSES   = config["n_classes"]
IMG_SIZE    = config["img_size"]   # 224 — matches ViT-B/16 pretraining resolution
MEAN_RGB    = stats["mean_rgb"]
STD_RGB     = stats["std_rgb"]

log(f"N_CLASSES : {N_CLASSES}   IMG_SIZE : {IMG_SIZE}")
log()


# ── Data ──────────────────────────────────────────────────────────────────────

class LeafDiseaseDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths, self.labels, self.transform = paths, labels, transform
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

_NUM_WORKERS = min(16, max(1, multiprocessing.cpu_count() - 1))
# ViT self-attention is O(n²) in sequence length so it uses more memory than CNNs,
# but on 80GB VRAM batch 256 is still very comfortable.
BATCH_SIZE   = 256
_PREFETCH    = 4

_DL_KWARGS = dict(
    batch_size=BATCH_SIZE, num_workers=_NUM_WORKERS,
    pin_memory=True, persistent_workers=(_NUM_WORKERS > 0),
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

log(f"Train : {len(train_loader)} batches ({len(train_idx)} images)")
log(f"Val   : {len(val_loader)} batches ({len(val_idx)} images)")
log(f"Test  : {len(test_loader)} batches ({len(test_idx)} images)")
log(f"Batch size: {BATCH_SIZE}   num_workers: {_NUM_WORKERS}   prefetch: {_PREFETCH}")
log()


# ── Model ─────────────────────────────────────────────────────────────────────

def build_vit_b16(n_classes, dropout=0.4):
    m = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    in_features = m.heads.head.in_features  # 768
    m.heads.head = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, 512),
        nn.GELU(),
        nn.Dropout(dropout * 0.5),
        nn.Linear(512, n_classes),
    )
    return m

def freeze_backbone(model):
    for name, p in model.named_parameters():
        if "heads" not in name:
            p.requires_grad = False

def unfreeze_encoder_layers(model, layer_indices):
    for idx in layer_indices:
        for p in model.encoder.layers[idx].parameters():
            p.requires_grad = True

def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


model = build_vit_b16(N_CLASSES).to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
log(f"Total parameters : {total_params:,}")
log(f"  Patch size     : 16×16")
log(f"  Num patches    : {(IMG_SIZE // 16)**2}  (+1 CLS = {(IMG_SIZE//16)**2 + 1})")
log(f"  Hidden dim     : 768   Encoder layers : 12")
log()


# ── Loss / AMP ────────────────────────────────────────────────────────────────

class_weights  = np.load(os.path.join(PROC_DIR, "class_weights.npy"))
weights_tensor = torch.FloatTensor(class_weights).to(DEVICE)
criterion      = nn.CrossEntropyLoss(weight=weights_tensor)

scaler    = torch.cuda.amp.GradScaler()
_autocast = lambda: torch.cuda.amp.autocast()

log("Loss : CrossEntropyLoss (class-weighted)")
log("AMP  : enabled")
log()


# ── Train / Eval loops ────────────────────────────────────────────────────────

def train_epoch(loader, optimizer):
    model.train()
    total = 0
    loss_t    = torch.tensor(0.0, device=DEVICE)
    correct_t = torch.tensor(0,   device=DEVICE)
    bar = tqdm(loader, desc="  Train", leave=False, unit="batch",
               dynamic_ncols=True, file=sys.stdout)
    for imgs, labels in bar:
        imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast():
            out  = model(imgs)
            loss = criterion(out, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_t    += loss.detach() * imgs.size(0)
        correct_t += (out.detach().argmax(1) == labels).sum()
        total     += imgs.size(0)
        bar.set_postfix(loss=f"{loss_t.item()/total:.4f}",
                        acc=f"{correct_t.item()/total:.4f}")
    bar.close()
    return loss_t.item() / total, correct_t.item() / total


def eval_epoch(loader, desc="  Val  "):
    model.eval()
    total = 0
    loss_t    = torch.tensor(0.0, device=DEVICE)
    correct_t = torch.tensor(0,   device=DEVICE)
    bar = tqdm(loader, desc=desc, leave=False, unit="batch",
               dynamic_ncols=True, file=sys.stdout)
    with torch.no_grad():
        for imgs, labels in bar:
            imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            with _autocast():
                out  = model(imgs)
                loss = criterion(out, labels)
            loss_t    += loss * imgs.size(0)
            correct_t += (out.argmax(1) == labels).sum()
            total     += imgs.size(0)
            bar.set_postfix(loss=f"{loss_t.item()/total:.4f}",
                            acc=f"{correct_t.item()/total:.4f}")
    bar.close()
    return loss_t.item() / total, correct_t.item() / total


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_ckpt(path, epoch, best_val_loss, patience_count, history, optimizer, scheduler):
    torch.save({
        "epoch": epoch, "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state":    scaler.state_dict(),
        "best_val_loss":   best_val_loss,
        "patience_count":  patience_count,
        "history":         history,
    }, path)

def load_ckpt(path, optimizer, scheduler):
    if args.reset or not os.path.isfile(path):
        if args.reset:
            log("--reset: ignoring checkpoint.")
        else:
            log(f"No checkpoint at {path} — starting fresh.")
        return 1, float("inf"), 0, {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}
    log(f"Resuming from checkpoint: {path}")
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    scaler.load_state_dict(ckpt["scaler_state"])
    log(f"  → epoch {ckpt['epoch']+1}, best_val_loss={ckpt['best_val_loss']:.4f}, "
        f"patience={ckpt['patience_count']}")
    return ckpt["epoch"]+1, ckpt["best_val_loss"], ckpt["patience_count"], ckpt["history"]


# ── Generic training loop ─────────────────────────────────────────────────────

def run_phase(phase_name, ckpt_path, optimizer, scheduler,
              n_epochs, early_stop_pat, history=None):
    start_epoch, best_val_loss, patience_count, hist = load_ckpt(
        ckpt_path, optimizer, scheduler)
    if history is not None and not hist["train_loss"]:
        hist = history

    separator()
    log(f"  {phase_name}  |  epochs {start_epoch} → {n_epochs}  |  patience={early_stop_pat}")
    separator()

    for epoch in range(start_epoch, n_epochs + 1):
        t0 = time.time()
        log(f"[Epoch {epoch:02d}/{n_epochs}]  {time.strftime('%H:%M:%S')}  — training...")
        tr_loss, tr_acc = train_epoch(train_loader, optimizer)
        log(f"[Epoch {epoch:02d}/{n_epochs}]  {time.strftime('%H:%M:%S')}  — validating...")
        vl_loss, vl_acc = eval_epoch(val_loader)
        scheduler.step(vl_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        hist["train_loss"].append(tr_loss); hist["train_acc"].append(tr_acc)
        hist["val_loss"].append(vl_loss);   hist["val_acc"].append(vl_acc)

        log(f"[Epoch {epoch:02d}/{n_epochs}]  "
            f"tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.4f}  "
            f"vl_loss={vl_loss:.4f}  vl_acc={vl_acc:.4f}  "
            f"lr={current_lr:.2e}  time={time.time()-t0:.1f}s")

        if vl_loss < best_val_loss:
            best_val_loss  = vl_loss
            patience_count = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            log(f"  ✓ New best saved  (val_loss={best_val_loss:.4f})")
        else:
            patience_count += 1
            log(f"  ✗ No improvement  (patience {patience_count}/{early_stop_pat})")
            if patience_count >= early_stop_pat:
                log(f"\nEarly stopping at epoch {epoch}.")
                save_ckpt(ckpt_path, epoch, best_val_loss, patience_count,
                          hist, optimizer, scheduler)
                break

        save_ckpt(ckpt_path, epoch, best_val_loss, patience_count,
                  hist, optimizer, scheduler)
        log(f"  Checkpoint saved → {ckpt_path}")
        log()

    return hist, best_val_loss


# ── Phase 1 ───────────────────────────────────────────────────────────────────

if args.eval_only:
    log("--eval-only: skipping training phases.")
    if not os.path.isfile(BEST_MODEL_PATH):
        log(f"ERROR: best model not found at {BEST_MODEL_PATH}")
        log("       Run without --eval-only first to train the model.")
        sys.exit(1)
    history      = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
else:
    freeze_backbone(model)
    log(f"Phase 1 — Trainable parameters : {count_trainable(model):,}  (head only)")

    optimizer_p1 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3, weight_decay=1e-4)
    scheduler_p1 = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_p1, mode="min", factor=0.5, patience=2)

    history, best_val_loss = run_phase(
        "Phase 1 — Feature Extraction (head only)",
        CKPT_P1, optimizer_p1, scheduler_p1,
        n_epochs=10, early_stop_pat=4,
    )

    # ── Phase 2 ───────────────────────────────────────────────────────────────

    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    unfreeze_encoder_layers(model, layer_indices=[8, 9, 10, 11])
    log(f"Phase 2 — Trainable parameters : {count_trainable(model):,}  (layers 8-11 + head)")

    optimizer_p2 = optim.AdamW([
        {"params": model.encoder.layers[8].parameters(),  "lr": 1e-6},
        {"params": model.encoder.layers[9].parameters(),  "lr": 5e-6},
        {"params": model.encoder.layers[10].parameters(), "lr": 1e-5},
        {"params": model.encoder.layers[11].parameters(), "lr": 5e-5},
        {"params": model.heads.parameters(),              "lr": 1e-3},
    ], weight_decay=1e-4)
    scheduler_p2 = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_p2, mode="min", factor=0.5, patience=3)

    history, best_val_loss = run_phase(
        "Phase 2 — Fine-tuning (encoder layers 8-11 + head)",
        CKPT_P2, optimizer_p2, scheduler_p2,
        n_epochs=20, early_stop_pat=6,
        history=history,
    )

    log("\nTraining complete.")
    separator()
    log()

    # ── Training curves ───────────────────────────────────────────────────────

    log("Plotting training curves...")
    phase1_epochs = min(10, len(history["train_loss"]))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    epochs_ran = range(1, len(history["train_loss"]) + 1)

    for ax, (m0, m1), title in zip(axes,
        [("train_loss","val_loss"), ("train_acc","val_acc")], ["Loss","Accuracy"]):
        ax.plot(epochs_ran, history[m0], label="Train", color="steelblue")
        ax.plot(epochs_ran, history[m1], label="Val",   color="tomato")
        ax.axvline(phase1_epochs, color="gray", linestyle="--", alpha=0.7, label="Phase 2 starts")
        ax.set_title(f"{MODEL_NAME} — {title}")
        ax.set_xlabel("Epoch"); ax.set_ylabel(title); ax.legend()

    plt.tight_layout()
    curves_path = os.path.join(RESULTS_DIR, f"07_{MODEL_NAME}_curves.png")
    plt.savefig(curves_path, dpi=150); plt.close()
    log(f"  Saved → {curves_path}")
    log()


# ── Test evaluation ───────────────────────────────────────────────────────────

log("Loading best weights for test evaluation...")
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
model.eval()

all_preds, all_targets = [], []
bar = tqdm(test_loader, desc="Testing", unit="batch", dynamic_ncols=True, file=sys.stdout)
with torch.no_grad():
    for imgs, labels in bar:
        preds = model(imgs.to(DEVICE)).argmax(1).cpu().numpy()
        all_preds.extend(preds); all_targets.extend(labels.numpy())
bar.close()

all_preds   = np.array(all_preds)
all_targets = np.array(all_targets)
test_acc    = (all_preds == all_targets).mean()
log(f"Test Accuracy : {test_acc:.4f}  ({test_acc*100:.2f}%)")
log()


# ── Attention maps ────────────────────────────────────────────────────────────

log("Generating attention maps (4 test samples)...")

mean_t = torch.tensor(MEAN_RGB).view(3, 1, 1)
std_t  = torch.tensor(STD_RGB).view(3, 1, 1)

def denormalize(t):
    return torch.clamp(t * std_t + mean_t, 0, 1).permute(1, 2, 0).numpy()

def get_attention_map(img_tensor):
    """
    Extract CLS-token attention from the last encoder layer.

    PyTorch 2.x MultiheadAttention returns (attn_output, None) by default
    because average_attn_weights is applied internally and the raw weights
    are discarded before the hook fires. We temporarily patch the module to
    force it to return per-head weights so the hook can capture them.
    """
    attn_weights = []
    last_mha     = model.encoder.layers[-1].self_attention
    original_fwd = last_mha.forward

    def patched_forward(query, key, value, **kwargs):
        kwargs["need_weights"]         = True
        kwargs["average_attn_weights"] = False
        return original_fwd(query, key, value, **kwargs)

    last_mha.forward = patched_forward

    def hook_fn(module, input, output):
        if output[1] is not None:
            attn_weights.append(output[1].detach().cpu())

    hook = last_mha.register_forward_hook(hook_fn)

    try:
        with torch.no_grad():
            _ = model(img_tensor.unsqueeze(0).to(DEVICE))
    finally:
        hook.remove()
        last_mha.forward = original_fwd   # always restore, even if forward crashes

    if not attn_weights:
        log("  Warning: attention weights unavailable — skipping map.")
        return None

    attn     = attn_weights[0].squeeze(0)  # (heads, seq_len, seq_len)
    cls_attn = attn[:, 0, 1:]              # (heads, num_patches) — CLS row, drop CLS col
    return cls_attn.numpy()

idx_to_disease = {row["class_idx"]: row["disease"] for _, row in manifest_df.iterrows()}
np.random.seed(7)
sample_indices = np.random.choice(len(test_idx), size=4, replace=False)

fig, axes = plt.subplots(4, 2, figsize=(8, 16))
for row_i, sample_i in enumerate(sample_indices):
    img_path   = all_paths[test_idx[sample_i]]
    true_label = all_labels[test_idx[sample_i]]
    disease    = idx_to_disease.get(int(true_label), "?")
    pil_img    = Image.open(img_path).convert("RGB")
    img_tensor = eval_transforms(pil_img)
    orig_img   = denormalize(img_tensor)

    axes[row_i, 0].imshow(orig_img)
    axes[row_i, 0].set_title(f"{disease}\nOriginal", fontsize=8)
    axes[row_i, 0].axis("off")

    attn = get_attention_map(img_tensor)
    if attn is not None:
        grid_size = int(attn.shape[1] ** 0.5)
        avg_attn  = attn.mean(axis=0).reshape(grid_size, grid_size)
        attn_img  = Image.fromarray(
            (avg_attn / avg_attn.max() * 255).astype("uint8")
        ).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        axes[row_i, 1].imshow(orig_img)
        axes[row_i, 1].imshow(np.array(attn_img) / 255.0, cmap="hot", alpha=0.5)
        axes[row_i, 1].set_title(f"{disease}\nAttention Map", fontsize=8)
    else:
        axes[row_i, 1].text(0.5, 0.5, "Attention\nunavailable",
                            ha="center", va="center", transform=axes[row_i,1].transAxes)
    axes[row_i, 1].axis("off")

plt.suptitle("ViT Attention Maps — CLS Token (last layer, avg across heads)", fontsize=10)
plt.tight_layout()
attn_path = os.path.join(RESULTS_DIR, f"07_{MODEL_NAME}_attention_maps.png")
plt.savefig(attn_path, dpi=150); plt.close()
log(f"  Saved → {attn_path}")
log()


# ── Confusion matrix ──────────────────────────────────────────────────────────

log("Generating confusion matrix...")
class_names = [row["disease"][:15] for _, row in manifest_df.sort_values("class_idx").iterrows()]
cm = confusion_matrix(all_targets, all_preds)
fig, ax = plt.subplots(figsize=(16, 14))
sns.heatmap(cm, annot=False, cmap="Purples", ax=ax,
            xticklabels=class_names, yticklabels=class_names)
ax.set_title(f"{MODEL_NAME} — Confusion Matrix (Test Set)", fontsize=12)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
plt.xticks(rotation=90, fontsize=7); plt.yticks(rotation=0, fontsize=7)
plt.tight_layout()
cm_path = os.path.join(RESULTS_DIR, f"07_{MODEL_NAME}_confusion_matrix.png")
plt.savefig(cm_path, dpi=150); plt.close()
log(f"  Saved → {cm_path}")
log()


# ── Per-class report ──────────────────────────────────────────────────────────

log("Per-class classification report:")
report    = classification_report(all_targets, all_preds,
                                   target_names=class_names, output_dict=True)
report_df = pd.DataFrame(report).T
log(report_df.round(3).to_string())
log()


# ── Save results JSON ─────────────────────────────────────────────────────────

results = {
    "model_name":    MODEL_NAME,
    "test_accuracy": float(test_acc),
    "best_val_loss": float(best_val_loss),
    "history":       history,
    "n_params":      int(total_params),
    "per_class":     report,
}
results_path = os.path.join(PROC_DIR, f"results_{MODEL_NAME}.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
log(f"Results JSON → {results_path}")
log()


# ── Summary ───────────────────────────────────────────────────────────────────

separator("=")
log("  07 — ViT-B/16 COMPLETE")
separator("=")
log(f"  Strategy       : 2-phase fine-tuning")
log(f"  Total params   : {total_params:,}")
log(f"  Test Accuracy  : {test_acc*100:.2f}%")
log(f"  Best Val Loss  : {best_val_loss:.4f}")
log(f"  Epochs trained : {len(history['train_loss'])}")
log()
log("  Next: notebook 08_results_comparison.ipynb  (run in Jupyter/Colab)")
separator("=")