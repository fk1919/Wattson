"""
SHAP-based explainability analysis (paralleling Zeng et al. 2026's claim of
being the first study to integrate SHAP into a defensive power side-channel
IDS). Shows which of the 43 features the trained XGBoost model relies on most.
"""
import numpy as np
import pickle

with open("xgboost_model_ethernet_fixed.pkl", "rb") as f:
    bundle = pickle.load(f)
model, scaler, le = bundle['model'], bundle['scaler'], bundle['le']
classes = le.classes_

FEATURE_NAMES = (
    ['mean', 'std', 'rms', 'ptp', 'skew', 'kurtosis', 'crest', 'energy'] +
    [f'fft_bin_{i}' for i in range(32)] +
    ['zero_crossing_rate', 'peak_rate', 'spectral_entropy']
)

data = np.load("feature_matrix_ethernet_fixed.npz", allow_pickle=True)
X = data['X'].astype(np.float32)
y_raw = data['y'].astype(str)
sessions = data['sessions'].astype(str)
y = le.transform(y_raw)

rng = np.random.RandomState(42)
test_idx = []
for cls_i in np.unique(y):
    cls_sessions = np.unique(sessions[y == cls_i])
    rng.shuffle(cls_sessions)
    n = len(cls_sessions)
    n_test = max(1, round(n * 0.2))
    n_train = n - n_test
    te_sess = set(cls_sessions[n_train:])
    cls_mask = (y == cls_i)
    test_idx.extend(np.where(cls_mask & np.isin(sessions, list(te_sess)))[0])
test_idx = np.array(test_idx)
X_te = X[test_idx]
X_te_s = scaler.transform(X_te)

try:
    import shap
except ImportError:
    print("[ERROR] shap library is not installed, installing...")
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "shap", "-q"])
    import shap

explainer = shap.TreeExplainer(model)
# Subsample for speed
rng2 = np.random.RandomState(0)
sub = rng2.choice(len(X_te_s), size=min(500, len(X_te_s)), replace=False)
X_sub = X_te_s[sub]

shap_values = explainer.shap_values(X_sub)
# XGBoost multiclass: shap_values shape is (n_samples, n_features, n_classes) in the new API
if isinstance(shap_values, list):
    mean_abs_per_class = [np.mean(np.abs(sv), axis=0) for sv in shap_values]
    global_importance = np.mean(mean_abs_per_class, axis=0)
else:
    sv = np.array(shap_values)
    if sv.ndim == 3:
        global_importance = np.mean(np.abs(sv), axis=(0, 2))
        mean_abs_per_class = [np.mean(np.abs(sv[:, :, c]), axis=0) for c in range(sv.shape[2])]
    else:
        global_importance = np.mean(np.abs(sv), axis=0)
        mean_abs_per_class = None

order = np.argsort(global_importance)[::-1]
print("=" * 60)
print("GLOBAL SHAP IMPORTANCE RANKING (averaged across all classes, top 15)")
print("=" * 60)
for i in order[:15]:
    print(f"  {FEATURE_NAMES[i]:<22s}: {global_importance[i]:.5f}")

if mean_abs_per_class is not None:
    print("\n" + "=" * 60)
    print("TOP 5 MOST IMPORTANT FEATURES PER CLASS")
    print("=" * 60)
    for c_i, c_name in enumerate(classes):
        c_order = np.argsort(mean_abs_per_class[c_i])[::-1][:5]
        feats_str = ", ".join(f"{FEATURE_NAMES[i]}({mean_abs_per_class[c_i][i]:.4f})" for i in c_order)
        print(f"  {c_name:<16s}: {feats_str}")

# Total contribution by feature category (statistical vs FFT vs packet-arrival-rate)
stat_idx = list(range(0, 8))
fft_idx = list(range(8, 40))
extra_idx = list(range(40, 43))
total = global_importance.sum()
print("\n" + "=" * 60)
print("TOTAL SHAP CONTRIBUTION BY FEATURE CATEGORY")
print("=" * 60)
print(f"  Statistical (8 features)              : {global_importance[stat_idx].sum()/total*100:.1f}%")
print(f"  FFT magnitude (32 features)            : {global_importance[fft_idx].sum()/total*100:.1f}%")
print(f"  Packet-arrival-rate proxy (3 features) : {global_importance[extra_idx].sum()/total*100:.1f}%")
