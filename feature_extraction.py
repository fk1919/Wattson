"""
Wattson — Feature Extraction Pipeline
Extracts 40 features from each ~100ms window, saves as feature_matrix.npz.
Reads large CSVs chunk by chunk, RAM-friendly.

NOTE: Since the hardware was fixed and recalibrated on 2026-09-15, the old
"pi_*_large" data is NO LONGER USED — the 40 independent sessions in
new_data/multisession (5x DoS/PortScan + 15x Normal/SSH_Bruteforce) are used
instead. The `sessions` array is kept for the same reason as in
cnn_model.py (to prevent session leakage).
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import find_peaks
from pathlib import Path
import re
import sys
import time

# ─── SETTINGS ────────────────────────────────────────────────
# Usage: python feature_extraction.py [BASE_DIR] [OUTPUT_FILE]
BASE      = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("new_data/full_dataset")
OUT_FILE  = sys.argv[2] if len(sys.argv) > 2 else "feature_matrix.npz"
WIN_SIZE  = 4882           # Window size (~100ms assuming 48828Hz)
STEP      = 4882           # Stride (no overlap)
N_FFT     = 32             # Number of frequency components to take from the FFT
CHUNK     = 500_000        # Rows read per iteration

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

# ─── FEATURE EXTRACTION FUNCTION ─────────────────────────────
# 43 features: 8 statistical + 32 FFT magnitudes + 3 "packet-arrival-rate"
# proxies (zero-crossing rate, peak rate, spectral entropy) — added to
# strengthen the DoS/PortScan distinction (see confusion matrix finding:
# DoS<->PortScan is the biggest confusion pair, both being "high packet
# volume, low CPU-processing load" classes; these 3 features aim to capture
# the TEMPORAL irregularity/periodicity character of the signal to help
# separate them).
def extract_features(window: np.ndarray) -> np.ndarray:
    mean_   = np.mean(window)
    std_    = np.std(window)
    rms_    = np.sqrt(np.mean(window**2))
    peak2pk = np.ptp(window)
    skew_   = stats.skew(window)
    kurt_   = stats.kurtosis(window)
    crest   = np.max(np.abs(window)) / (rms_ + 1e-9)
    energy  = np.sum(window**2)

    stat_feats = np.array([mean_, std_, rms_, peak2pk, skew_, kurt_, crest, energy])

    fft_mag = np.abs(np.fft.rfft(window - mean_))[:N_FFT]
    fft_mag = fft_mag / (np.sum(fft_mag) + 1e-9)

    # Zero-crossing rate: how often the signal changes direction around its
    # mean -- a proxy for the density of high-frequency/jittery events
    # (such as packet-processing interrupts).
    centered = window - mean_
    zcr = np.mean(np.diff(np.sign(centered)) != 0)

    # Peak rate: number of distinguishable "events" per window
    # (peaks with prominence exceeding a fraction of std).
    peaks, _ = find_peaks(np.abs(centered), prominence=0.5 * (std_ + 1e-9))
    peak_rate = len(peaks) / len(window)

    # Spectral entropy: how "sharp/periodic" (low entropy) or
    # "flat/irregular" (high entropy) the FFT magnitude distribution is.
    # Normalized to the [0,1] range (by dividing by log(N_FFT)).
    p = fft_mag + 1e-12
    spectral_entropy = -np.sum(p * np.log(p)) / np.log(len(p))

    extra_feats = np.array([zcr, peak_rate, spectral_entropy])

    return np.concatenate([stat_feats, fft_mag, extra_feats])  # 43-dimensional vector

def process_file(filepath: Path):
    feats, labs = [], []
    buffer = np.array([], dtype=np.float32)

    for chunk_df in pd.read_csv(filepath, chunksize=CHUNK, usecols=['Current', 'anno_type']):
        current    = chunk_df['Current'].values.astype(np.float32)
        labels_raw = chunk_df['anno_type'].values

        if len(buffer) > 0:
            current = np.concatenate([buffer, current])
            labels_raw = np.concatenate([np.full(len(buffer), labels_raw[0]), labels_raw])

        n_windows = (len(current) - WIN_SIZE) // STEP + 1 if len(current) >= WIN_SIZE else 0
        for i in range(n_windows):
            start, end = i * STEP, i * STEP + WIN_SIZE
            win = current[start:end]
            label_win = labels_raw[start:end]
            if len(np.unique(label_win)) > 1:
                continue
            feats.append(extract_features(win))
            labs.append(label_win[0])

        used = n_windows * STEP
        buffer = current[used:] if used < len(current) else np.array([], dtype=np.float32)

    return feats, labs

# ─── MAIN LOOP ──────────────────────────────────────────────
all_features, all_labels, all_sessions = [], [], []
label_counts = {}

t0 = time.time()
for filepath in FILES:
    class_label = label_from_filename(filepath)
    session_id  = filepath.stem
    print(f"Processing: {filepath.name} [{class_label}] session={session_id}")
    feats, labs = process_file(filepath)
    all_features.extend(feats)
    all_labels.extend(labs)
    all_sessions.extend([session_id] * len(labs))
    label_counts[class_label] = label_counts.get(class_label, 0) + len(labs)
    print(f"  → {len(labs)} windows added")

X = np.array(all_features, dtype=np.float32)
y = np.array(all_labels)
sessions = np.array(all_sessions)

# ─── BALANCE ALL CLASSES ────────────────────────────────────
# The number of sessions can vary a lot between classes (sometimes there are
# way too many Normal sessions, sometimes way too many attack-class
# sessions). Training on the raw distribution lets the class with the most
# windows "drown out" the others (see earlier attempts: DoS F1 0.36, then
# Normal F1 0.20). For each class, an equal share of windows is sampled from
# EVERY session up to a shared cap (MAX_WINDOWS_PER_CLASS) -- sessions are
# never dropped, only the volume is capped, so diversity is preserved.
MAX_WINDOWS_PER_CLASS = 3000
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
print(f"Total windows: {len(X):,}")
print(f"Feature dimension: {X.shape[1]}")
print(f"Number of sessions: {len(set(sessions))}")
print(f"\nClass distribution:")
unique, counts = np.unique(y, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  {u:15s}: {c:6,} windows")

np.savez_compressed(OUT_FILE, X=X, y=y, sessions=sessions)
print(f"\nSaved: {OUT_FILE} (X, y, sessions)")
print(f"Estimated file size: ~{X.nbytes/1e6:.1f} MB")
print(f"Duration: {time.time()-t0:.1f} seconds")
