"""
Noise robustness test (parallel to Zeng et al. 2026's AWGN sweep):
measures how the AUC/accuracy of an already-trained XGBoost model degrades
under progressively added synthetic noise (AWGN) on the raw (non-normalized)
signal. The previous version was INCORRECT: windowed_data_ethernet_fixed.npz
is already per-window z-score normalized (for the CNN), which destroys
amplitude information and makes the statistical features meaningless. This
version re-extracts RAW (non-normalized) windows directly from the CSVs,
from the SAME test sessions used by feature_extraction.py.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import re
import pickle
from scipy import stats
from scipy.signal import find_peaks
from sklearn.metrics import roc_auc_score, accuracy_score

WIN_SIZE = 4882
N_FFT = 32
BASE = Path("new_data/ethernet_dataset")

LABEL_MAP = {'normal': 'Normal', 'portscan': 'PortScan',
             'ssh_bruteforce': 'SSH_Bruteforce', 'dos': 'DoS'}

def label_from_filename(path):
    m = re.match(r'pi_(.+?)_\d{8}_\d{6}\.csv$', path.name)
    key = m.group(1) if m else ''
    return LABEL_MAP.get(key)

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
    centered = window - mean_
    zcr = np.mean(np.diff(np.sign(centered)) != 0)
    peaks, _ = find_peaks(np.abs(centered), prominence=0.5 * (std_ + 1e-9))
    peak_rate = len(peaks) / len(window)
    p = fft_mag + 1e-12
    spectral_entropy = -np.sum(p * np.log(p)) / np.log(len(p))
    extra_feats = np.array([zcr, peak_rate, spectral_entropy])
    return np.concatenate([stat_feats, fft_mag, extra_feats])

with open("xgboost_model_ethernet_fixed.pkl", "rb") as f:
    bundle = pickle.load(f)
model, scaler, le = bundle['model'], bundle['scaler'], bundle['le']
classes = le.classes_

# Re-derive the SAME session-based test split used in
# feature_matrix_ethernet_fixed.npz (using the sessions list from
# feature_matrix to reproduce the same rng logic as feature_extraction.py +
# multi_model_comparison.py)
data = np.load("feature_matrix_ethernet_fixed.npz", allow_pickle=True)
y_raw_all = data['y'].astype(str)
sessions_all = data['sessions'].astype(str)
y_all = le.transform(y_raw_all)

rng = np.random.RandomState(42)
test_sessions_set = set()
for cls_i in np.unique(y_all):
    cls_sessions = np.unique(sessions_all[y_all == cls_i])
    rng.shuffle(cls_sessions)
    n = len(cls_sessions)
    n_test = max(1, round(n * 0.2))
    n_train = n - n_test
    test_sessions_set.update(cls_sessions[n_train:])

print(f"Number of test sessions: {len(test_sessions_set)}")

# Draw RAW windows from a few test sessions per class (capped for speed)
FILES = sorted(BASE.glob("pi_*.csv"))
session_to_file = {f.stem: f for f in FILES}

rng2 = np.random.RandomState(0)
per_class_files = {}
for s in test_sessions_set:
    if s not in session_to_file:
        continue
    f = session_to_file[s]
    lbl = label_from_filename(f)
    per_class_files.setdefault(lbl, []).append(f)

MAX_WIN_PER_CLASS = 300
raw_windows, raw_labels = [], []
for cls_name, files in per_class_files.items():
    rng2.shuffle(files)
    collected = 0
    for f in files:
        if collected >= MAX_WIN_PER_CLASS:
            break
        df = pd.read_csv(f, usecols=['Current'])
        curr = df['Current'].values.astype(np.float32)
        n_wins = len(curr) // WIN_SIZE
        for i in range(n_wins):
            if collected >= MAX_WIN_PER_CLASS:
                break
            raw_windows.append(curr[i*WIN_SIZE:(i+1)*WIN_SIZE])
            raw_labels.append(cls_name)
            collected += 1
    print(f"  {cls_name}: {collected} raw windows collected")

X_te_raw = np.array(raw_windows, dtype=np.float32)
y_te = le.transform(np.array(raw_labels))
print(f"Total test windows: {len(X_te_raw)}")

def add_awgn(x, snr_db, seed=123):
    if snr_db is None:
        return x
    sig_power = np.mean(x**2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = sig_power / snr_linear
    noise = np.random.RandomState(seed).normal(0, np.sqrt(noise_power), size=x.shape)
    return x + noise

print(f"\n{'SNR(dB)':>10}{'Accuracy':>12}{'WeightedAUC':>14}")
for snr_db in [None, 30, 20, 10, 5, 0, -5]:
    feats = []
    for w in X_te_raw:
        wn = add_awgn(w, snr_db)
        feats.append(extract_features(wn))
    X_te_feat = np.array(feats, dtype=np.float32)
    X_te_s = scaler.transform(X_te_feat)
    y_pred = model.predict(X_te_s)
    y_proba = model.predict_proba(X_te_s)
    acc = accuracy_score(y_te, y_pred)
    try:
        auc = roc_auc_score(y_te, y_proba, multi_class='ovr', average='weighted')
    except ValueError:
        auc = float('nan')
    label = "clean" if snr_db is None else f"{snr_db}"
    print(f"{label:>10}{acc:>12.4f}{auc:>14.4f}")

print(f"\nReference (Zeng et al. 2026): clean->0.96 F1 (attack ID), 0dB->0.417 F1 (major collapse)")
