"""
Wattson — 1D-CNN (PyTorch)
Attack detection from raw current signal using deep learning

Requirements:
    pip install torch scikit-learn numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import time, os, sys
import pickle

# Usage: python cnn_model.py [TAG]
TAG = f"_{sys.argv[1]}" if len(sys.argv) > 1 else ""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)

print(f"PyTorch {torch.__version__} | GPU: {torch.cuda.is_available()}")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device in use: {DEVICE}")

# ─── FIXED SEED (for reproducibility) ────────────────────
SEED = 42
import random
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ─── LOAD DATA ─────────────────────────────────────────────
print(f"\nLoading windowed_data{TAG}.npz...")
data  = np.load(f"windowed_data{TAG}.npz", allow_pickle=True)
X_raw = data['X']          # (N, 4882) — per-window z-score normalized
y_raw = data['y'].astype(str)
sessions = data['sessions'].astype(str) if 'sessions' in data else None

# Merge PortScan subclasses
y_raw = np.where(
    np.char.startswith(y_raw, 'Port_Scan'), 'PortScan', y_raw
)

le      = LabelEncoder()
y       = le.fit_transform(y_raw)
classes = le.classes_
n_cls   = len(classes)
WIN     = X_raw.shape[1]

print(f"Shape: {X_raw.shape}  |  Classes: {classes}")

# Class distribution
unique, counts = np.unique(y, return_counts=True)
for i, c in zip(unique, counts):
    print(f"  {classes[i]:<18}: {c:>5,}")

# ─── TRAIN / VAL / TEST — SESSION-BASED SPLIT ─────────────────
# Windows from the same recording session (overlapping sliding window)
# are very similar to each other — if we split at the window level at
# random, the same session leaks into both train and test, and the
# model memorizes session-specific hardware/noise signatures instead
# of the actual attack pattern (spuriously high accuracy). For this
# reason we split at the SESSION level: for each class, some independent
# sessions go only into train, others only into val/test.
if sessions is None:
    raise SystemExit(
        "[ERROR] 'sessions' not found in windowed_data.npz — "
        "re-run prepare_cnn_data.py with its current version."
    )

rng = np.random.RandomState(42)
train_idx, val_idx, test_idx = [], [], []

for cls_i in np.unique(y):
    cls_sessions = np.unique(sessions[y == cls_i])
    rng.shuffle(cls_sessions)
    n = len(cls_sessions)
    if n < 3:
        raise SystemExit(
            f"[ERROR] Class {classes[cls_i]} has only {n} session(s) — "
            f"at least 3 are required for a session-based train/val/test split."
        )
    n_test  = max(1, round(n * 0.2))
    n_val   = max(1, round(n * 0.2))
    n_train = n - n_val - n_test
    if n_train < 1:
        n_train, n_val, n_test = n - 2, 1, 1

    tr_sess  = set(cls_sessions[:n_train])
    val_sess = set(cls_sessions[n_train:n_train + n_val])
    te_sess  = set(cls_sessions[n_train + n_val:])

    cls_mask = (y == cls_i)
    train_idx.extend(np.where(cls_mask & np.isin(sessions, list(tr_sess)))[0])
    val_idx.extend(np.where(cls_mask & np.isin(sessions, list(val_sess)))[0])
    test_idx.extend(np.where(cls_mask & np.isin(sessions, list(te_sess)))[0])

    print(f"  {classes[cls_i]:<18}: train sessions={sorted(tr_sess)} "
          f"val sessions={sorted(val_sess)} test sessions={sorted(te_sess)}")

train_idx = np.array(train_idx); val_idx = np.array(val_idx); test_idx = np.array(test_idx)
X_tr,  y_tr  = X_raw[train_idx], y[train_idx]
X_val, y_val = X_raw[val_idx],   y[val_idx]
X_te,  y_te  = X_raw[test_idx],  y[test_idx]

tr_s, val_s, te_s = set(sessions[train_idx]), set(sessions[val_idx]), set(sessions[test_idx])
assert not (tr_s & val_s) and not (tr_s & te_s) and not (val_s & te_s), \
    "Session leakage: a session was found in more than one split"
print(f"\nTrain: {len(X_tr):,} | Val: {len(X_val):,} | Test: {len(X_te):,}  (session-based split)")

# Convert to tensors — (N, 1, T) shape (batch, channels, time)
def to_tensor(X, y_arr):
    Xt = torch.tensor(X[:, np.newaxis, :], dtype=torch.float32)
    yt = torch.tensor(y_arr, dtype=torch.long)
    return TensorDataset(Xt, yt)

BATCH = 64
tr_loader  = DataLoader(to_tensor(X_tr,  y_tr),  batch_size=BATCH, shuffle=True,  num_workers=0)
val_loader = DataLoader(to_tensor(X_val, y_val), batch_size=BATCH, shuffle=False, num_workers=0)
te_loader  = DataLoader(to_tensor(X_te,  y_te),  batch_size=BATCH, shuffle=False, num_workers=0)

# Class weights
cw_vals = compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
class_weights = torch.tensor(cw_vals, dtype=torch.float32).to(DEVICE)
print(f"Class weights: { {classes[i]: round(v,2) for i,v in enumerate(cw_vals)} }")

# ─── 1D-CNN ARCHITECTURE ────────────────────────────────────
class ResBlock1D(nn.Module):
    """Residual-connection Conv block — improves gradient flow"""
    def __init__(self, channels, kernel_size):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.conv(x))


class CNN1D(nn.Module):
    def __init__(self, win_size, n_classes):
        super().__init__()
        # Stage 1: Wide receptive field (low-frequency trends)
        self.stage1 = nn.Sequential(
            nn.Conv1d(1,  32, kernel_size=64, stride=4, padding=32, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )
        self.res1 = ResBlock1D(32, 7)

        # Stage 2: Mid-scale patterns
        self.stage2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=16, stride=2, padding=8, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )
        self.res2 = ResBlock1D(64, 5)

        # Stage 3: Fine details
        self.stage3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=8, padding=4, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Global average pooling → classifier
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.res1(x)
        x = self.stage2(x)
        x = self.res2(x)
        x = self.stage3(x)
        x = self.gap(x)
        return self.classifier(x)


model = CNN1D(WIN, n_cls).to(DEVICE)

# Parameter count
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel parameters: {n_params:,}")
print(model)

# ─── OPTIMIZATION ───────────────────────────────────────────
criterion  = nn.CrossEntropyLoss(weight=class_weights)
optimizer  = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler  = ReduceLROnPlateau(optimizer, factor=0.5, patience=4, min_lr=1e-5)

# ─── TRAINING LOOP ─────────────────────────────────────────
EPOCHS     = 60
PATIENCE   = 10
best_val   = float('inf')
patience_c = 0
history    = {'train_loss': [], 'val_loss': [], 'val_acc': []}
best_state = None

print(f"\n── Training starting ({EPOCHS} epoch max, patience={PATIENCE}) ──")
t_start = time.time()

for epoch in range(1, EPOCHS + 1):
    # Training
    model.train()
    tr_loss = 0.0
    for Xb, yb in tr_loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        optimizer.step()
        tr_loss += loss.item() * len(yb)
    tr_loss /= len(y_tr)

    # Validation
    model.eval()
    val_loss, val_correct = 0.0, 0
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            out = model(Xb)
            val_loss    += criterion(out, yb).item() * len(yb)
            val_correct += (out.argmax(1) == yb).sum().item()
    val_loss /= len(y_val)
    val_acc   = val_correct / len(y_val)

    history['train_loss'].append(tr_loss)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    scheduler.step(val_loss)

    # Early stopping
    if val_loss < best_val:
        best_val   = val_loss
        patience_c = 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        torch.save(best_state, f"cnn_best{TAG}.pt")
    else:
        patience_c += 1

    if epoch % 5 == 0 or patience_c == 0:
        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:3d}/{EPOCHS}  "
              f"train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_acc={val_acc:.3f}  lr={lr_now:.2e}"
              f"{'  ★' if patience_c == 0 else ''}")

    if patience_c >= PATIENCE:
        print(f"\nEarly stop — epoch {epoch} (patience={PATIENCE})")
        break

train_time = time.time() - t_start
print(f"\nTraining time: {train_time:.1f}s")

# Load best weights
model.load_state_dict(best_state)
model.eval()

# ─── TEST EVALUATION ───────────────────────────────
all_preds, all_probs = [], []
with torch.no_grad():
    for Xb, _ in te_loader:
        out   = model(Xb.to(DEVICE))
        probs = torch.softmax(out, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        all_probs.append(probs)
        all_preds.append(preds)

y_proba = np.vstack(all_probs)
y_pred  = np.concatenate(all_preds)

auc_cnn = roc_auc_score(y_te, y_proba, multi_class='ovr', average='weighted')

print("\n── Test Results (1D-CNN) ──")
print(classification_report(y_te, y_pred, target_names=classes, digits=3))
print(f"Weighted AUC: {auc_cnn:.4f}")

# Inference speed — single window (CPU, without ONNX)
model.to('cpu').eval()
single = torch.tensor(X_te[:1, np.newaxis, :], dtype=torch.float32)
N_inf  = 300
with torch.no_grad():
    t0 = time.perf_counter()
    for _ in range(N_inf):
        model(single)
inf_ms_cpu = (time.perf_counter() - t0) / N_inf * 1000
print(f"Inference (CPU, 1 window): {inf_ms_cpu:.2f} ms")
print(f"Real-time (100ms threshold): {'✓' if inf_ms_cpu < 100 else '✗'}")

# ─── PLOTS ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Training curve
ax = axes[0]
ep_range = range(1, len(history['train_loss']) + 1)
ax.plot(ep_range, history['train_loss'], label='Train Loss', color='#3b82f6', lw=2)
ax.plot(ep_range, history['val_loss'],   label='Val Loss',   color='#ef4444', lw=2)
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.set_title('Training Curve', fontweight='bold')
ax.legend(); ax.grid(alpha=0.3)
best_ep = np.argmin(history['val_loss']) + 1
ax.axvline(best_ep, color='gray', linestyle='--', lw=1, alpha=0.6)
ax.text(best_ep + 0.5, max(history['train_loss']) * 0.9,
        f'Best: epoch {best_ep}', fontsize=8, color='gray')

# 2. Confusion Matrix
ax = axes[1]
cm = confusion_matrix(y_te, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=classes)
disp.plot(ax=ax, colorbar=False, cmap='Blues', values_format='d')
ax.set_title(f'1D-CNN Confusion Matrix\nAUC: {auc_cnn:.4f}', fontweight='bold')
ax.tick_params(axis='x', rotation=30, labelsize=8)

# 3. Model comparison bar chart
ax = axes[2]
model_names = ['Dragon_Slice\n(baseline)', 'SVM\n(RBF)', 'XGBoost', 'Random\nForest', '1D-CNN\n(Proposed)']
aucs_comp   = [0.883, 0.9504, 0.9717, 0.9687, auc_cnn]
colors_bar  = ['#94a3b8', '#f87171', '#fb923c', '#60a5fa', '#4ade80']
bars = ax.bar(model_names, aucs_comp, color=colors_bar, edgecolor='white', width=0.55)
ax.set_ylim([0.75, 1.0])
ax.set_ylabel('Weighted AUC')
ax.set_title('Model Comparison', fontweight='bold')
ax.axhline(0.876, color='gray', linestyle='--', lw=1, alpha=0.5)
ax.axhline(0.890, color='gray', linestyle=':', lw=1, alpha=0.5)
ax.grid(axis='y', alpha=0.3)
for bar, v in zip(bars, aucs_comp):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.003,
            f'{v:.4f}', ha='center', fontsize=9, fontweight='bold')

plt.suptitle('Wattson — 1D-CNN Results (PyTorch)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"cnn_results{TAG}.png", dpi=150, bbox_inches='tight')
plt.show()
print(f"\nSaved: cnn_results{TAG}.png")

# ─── TORCH → ONNX EXPORT (for RPi) ────────────────────────
try:
    onnx_path = f"cnn_model{TAG}.onnx"
    dummy = torch.zeros(1, 1, WIN, dtype=torch.float32)
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        opset_version=14
    )
    sz_kb = os.path.getsize(onnx_path) / 1024
    print(f"Saved: {onnx_path} ({sz_kb:.0f} KB) — to be copied to the RPi")

    # ONNX inference speed
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path)
    inp_name = sess.get_inputs()[0].name
    inp_np   = dummy.numpy()
    t0 = time.perf_counter()
    for _ in range(N_inf):
        sess.run(None, {inp_name: inp_np})
    inf_ms_onnx = (time.perf_counter() - t0) / N_inf * 1000
    print(f"ONNX inference (CPU): {inf_ms_onnx:.2f} ms/window")
except Exception as e:
    print(f"ONNX export: {e}")

# ─── SUMMARY ───────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"SUMMARY — PUBLICATION RESULTS")
print(f"{'='*55}")
print(f"  Model:               1D-CNN (Conv1D×3 + ResBlock + GAP)")
print(f"  Weighted AUC:        {auc_cnn:.4f}")
print(f"  Dragon_Slice AUC:    0.876–0.890")
delta = (auc_cnn - 0.883) / 0.883 * 100
print(f"  Improvement:         {delta:+.1f}% (relative to Dragon_Slice average)")
print(f"  RF vs CNN:           RF={0.9097:.4f}  CNN={auc_cnn:.4f}")
print(f"  Inference (CPU):     {inf_ms_cpu:.2f} ms/window")
print(f"  Training time:       {train_time:.1f}s")
print(f"  Parameter count:     {n_params:,}")
print(f"  Window size:         {WIN} sample (~100ms @ 48.8kHz)")
