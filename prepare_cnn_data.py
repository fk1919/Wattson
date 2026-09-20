"""
Wattson — Raw Window Data for the 1D-CNN
Extracts raw current-signal windows from the CSV files.
Output: windowed_data.npz  (X: float32, shape [N, WIN_SIZE], y: str labels,
                           sessions: str — the recording session the window came from)

NOTE: different from feature_matrix.npz — here the raw signal is kept (no
feature extraction). The CNN learns directly from the raw signal.

IMPORTANT — SESSION LEAKAGE:
There are MULTIPLE independent recording sessions per class (see
new_data/multisession, 5 sessions/class, 2026-09-15). The `sessions` array
lets cnn_model.py perform its train/val/test split at the SESSION level
rather than the WINDOW level — otherwise overlapping windows from the same
session would leak into both train and test, and the model would memorize
session-specific hardware/noise signatures instead of the actual attack
pattern.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import re
import sys
import time

# ─── SETTINGS ────────────────────────────────────────────────
# NOTE: Since the hardware (ACS712 VCC 3.3V→5V, loose contact, IP+/IP- power
# line series wiring) was completely replaced and recalibrated on
# 2026-09-15, the old "pi_*_large" data (collected under a
# different/faulty hardware condition) is NO LONGER USED — mixing it with
# the new data would create a scale/noise mismatch.
# Usage: python prepare_cnn_data.py [BASE_DIR] [OUTPUT_FILE]
BASE     = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("new_data/full_dataset")
OUT_FILE = sys.argv[2] if len(sys.argv) > 2 else "windowed_data.npz"
WIN_SIZE = 4882    # ~100ms assuming 48828 Hz (actual Fs ~7300-7800 Hz)
STEP     = 2441    # 50% overlap (for more data)
CHUNK    = 500_000

# Extract the class label from the filename: pi_<label>_<timestamp>.csv
LABEL_MAP = {
    'normal':         'Normal',
    'portscan':       'PortScan',
    'ssh_bruteforce': 'SSH_Bruteforce',
    'dos':            'DoS',
}

def label_from_filename(path: Path) -> str:
    m = re.match(r'pi_(.+?)_\d{8}_\d{6}\.csv$', path.name)
    key = m.group(1) if m else ''
    if key not in LABEL_MAP:
        raise ValueError(f"Could not derive class from filename: {path.name}")
    return LABEL_MAP[key]

FILES = sorted(BASE.glob("pi_*.csv"))
if not FILES:
    raise SystemExit(f"[ERROR] No pi_*.csv found in {BASE}")

def process_file(filepath: Path, class_label: str):
    wins, labs = [], []
    buffer = np.array([], dtype=np.float32)

    for chunk_df in pd.read_csv(filepath, chunksize=CHUNK, usecols=['Current', 'anno_type']):
        curr   = chunk_df['Current'].values.astype(np.float32)
        labels = chunk_df['anno_type'].values

        if len(buffer) > 0:
            curr   = np.concatenate([buffer, curr])
            labels = np.concatenate([np.full(len(buffer), labels[0]), labels])

        n_wins = (len(curr) - WIN_SIZE) // STEP + 1 if len(curr) >= WIN_SIZE else 0
        for i in range(n_wins):
            s, e = i * STEP, i * STEP + WIN_SIZE
            win_labels = labels[s:e]
            if len(np.unique(win_labels)) > 1:
                continue
            wins.append(curr[s:e])
            labs.append(win_labels[0])

        used   = n_wins * STEP
        buffer = curr[used:] if used < len(curr) else np.array([], dtype=np.float32)

    return wins, labs

# ─── MAIN LOOP ──────────────────────────────────────────────
all_X, all_y, all_sessions = [], [], []
label_counts = {}
session_counts = {}

t0 = time.time()
for filepath in FILES:
    class_label = label_from_filename(filepath)
    session_id  = filepath.stem   # e.g. pi_normal_20260915_223505
    print(f"Processing: {filepath.name} [{class_label}] session={session_id}")
    wins, labs = process_file(filepath, class_label)
    all_X.extend(wins)
    all_y.extend(labs)
    all_sessions.extend([session_id] * len(labs))
    label_counts[class_label] = label_counts.get(class_label, 0) + len(labs)
    session_counts[session_id] = len(labs)
    print(f"  → {len(labs):,} windows")

X = np.array(all_X, dtype=np.float32)          # shape: (N, 4882)
y = np.array(all_y)                            # string labels
sessions = np.array(all_sessions)              # string session ids

# ─── BALANCE ALL CLASSES ────────────────────────────────────
# same rationale as in feature_extraction.py: whichever class has the most
# sessions/windows drowns out the others. For each class, an equal share is
# drawn per session via session-based subsampling up to a shared cap
# (MAX_WINDOWS_PER_CLASS).
MAX_WINDOWS_PER_CLASS = 8000
rng = np.random.RandomState(42)
keep_idx = []
for cls in np.unique(y):
    cls_mask = (y == cls)
    cls_sessions = np.unique(sessions[cls_mask])
    if cls_mask.sum() > MAX_WINDOWS_PER_CLASS:
        per_session_quota = max(1, MAX_WINDOWS_PER_CLASS // len(cls_sessions))
        for s in cls_sessions:
            s_idx = np.where(cls_mask & (sessions == s))[0]
            n_take = min(len(s_idx), per_session_quota)
            keep_idx.extend(rng.choice(s_idx, size=n_take, replace=False))
        print(f"{cls:15s} balanced: {cls_mask.sum():,} -> ~{MAX_WINDOWS_PER_CLASS:,} windows "
              f"(equal share from {len(cls_sessions)} sessions, all kept)")
    else:
        keep_idx.extend(np.where(cls_mask)[0])
keep_idx = np.array(sorted(keep_idx))
X, y, sessions = X[keep_idx], y[keep_idx], sessions[keep_idx]

print(f"\n{'─'*50}")
print(f"Total: {len(X):,} windows | Shape: {X.shape}")
print(f"Number of sessions: {len(set(sessions))}")
print(f"\nClass distribution:")
unique, counts = np.unique(y, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  {u:<18}: {c:>5,} windows")

# ─── GLOBAL NORMALIZATION (z-score per sample) ──────────────────
# Normalize each window internally for the CNN
# (mean=0, std=1 per window)
mu  = X.mean(axis=1, keepdims=True)
std = X.std(axis=1, keepdims=True) + 1e-9
X_norm = (X - mu) / std

print(f"\nNormalized (per-window z-score)")
print(f"X_norm: min={X_norm.min():.3f}, max={X_norm.max():.3f}, mean={X_norm.mean():.4f}")

np.savez_compressed(OUT_FILE, X=X_norm, y=y, sessions=sessions)
print(f"\nSaved: {OUT_FILE} (X, y, sessions)")
print(f"Estimated file size: ~{X_norm.nbytes/1e6:.1f} MB")
print(f"Duration: {time.time()-t0:.1f}s")
print(f"\n✓ Next step: python cnn_model.py")
