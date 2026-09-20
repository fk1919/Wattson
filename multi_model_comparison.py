"""
Wattson — Multi-Model Comparison
RF vs SVM vs XGBoost — AUC, F1, Inference time

Requirements:
    pip install xgboost scikit-learn numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import pickle
import sys

# Usage: python multi_model_comparison.py [TAG]
# If TAG is given: feature_matrix_{TAG}.npz is read, all outputs are saved with the _{TAG} suffix.
TAG = f"_{sys.argv[1]}" if len(sys.argv) > 1 else ""

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report, roc_auc_score,
    roc_curve, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.utils.class_weight import compute_class_weight

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    print("[WARNING] XGBoost not found. pip install xgboost")
    HAS_XGB = False

# ─── LOAD DATA ─────────────────────────────────────────────
print(f"Loading feature_matrix{TAG}.npz...")
data = np.load(f"feature_matrix{TAG}.npz", allow_pickle=True)
X = data['X'].astype(np.float32)
y_raw = data['y']
sessions = data['sessions'].astype(str) if 'sessions' in data else None

# Merge PortScan subclasses
y_raw = np.where(np.char.startswith(y_raw.astype(str), 'Port_Scan'), 'PortScan', y_raw)

le = LabelEncoder()
y = le.fit_transform(y_raw)
class_names = le.classes_
n_classes   = len(class_names)
print(f"Classes: {class_names}  |  Total: {len(X):,} windows")

# ─── TRAIN/TEST SPLIT — SESSION-BASED ─────────────────────────
# Same rationale as in cnn_model.py: splitting windows randomly leaks
# windows from the same session into both train and test, producing a
# falsely high accuracy. We perform the split at the session level instead.
if sessions is None:
    raise SystemExit(
        "[ERROR] 'sessions' not found in feature_matrix.npz — "
        "re-run the current version of feature_extraction.py."
    )

rng = np.random.RandomState(42)
train_idx, test_idx = [], []
for cls_i in np.unique(y):
    cls_sessions = np.unique(sessions[y == cls_i])
    rng.shuffle(cls_sessions)
    n = len(cls_sessions)
    n_test  = max(1, round(n * 0.2))
    n_train = n - n_test
    tr_sess = set(cls_sessions[:n_train])
    te_sess = set(cls_sessions[n_train:])
    cls_mask = (y == cls_i)
    train_idx.extend(np.where(cls_mask & np.isin(sessions, list(tr_sess)))[0])
    test_idx.extend(np.where(cls_mask & np.isin(sessions, list(te_sess)))[0])
    print(f"  {class_names[cls_i]:<18}: train sessions={sorted(tr_sess)} test sessions={sorted(te_sess)}")

train_idx, test_idx = np.array(train_idx), np.array(test_idx)
X_tr, y_tr = X[train_idx], y[train_idx]
X_te, y_te = X[test_idx],  y[test_idx]
assert not (set(sessions[train_idx]) & set(sessions[test_idx])), "Session leakage!"

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s  = scaler.transform(X_te)
print(f"Training: {len(X_tr):,}  |  Test: {len(X_te):,}  (session-based split)")

# ─── DEFINE MODELS ──────────────────────────────────────────
models = {}

# 1. Random Forest (load if already trained, otherwise retrain)
try:
    with open(f"rf_model{TAG}.pkl", "rb") as f:
        bundle = pickle.load(f)
    rf = bundle['model']
    print(f"✓ RF loaded from rf_model{TAG}.pkl")
except FileNotFoundError:
    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                random_state=42, n_jobs=-1)
models['Random Forest'] = rf

# 2. SVM (RBF kernel, requires normalized data)
svm = SVC(kernel='rbf', C=10, gamma='scale',
          class_weight='balanced', probability=True, random_state=42)
models['SVM (RBF)'] = svm

# 3. XGBoost
if HAS_XGB:
    weights = compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
    sample_w = np.array([weights[yi] for yi in y_tr], dtype=np.float32)
    xgb_clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='mlogloss',
        tree_method='hist',
        random_state=42,
        n_jobs=-1
    )
    models['XGBoost'] = xgb_clf

# ─── TRAINING & EVALUATION ───────────────────────────────
results = {}

for name, model in models.items():
    print(f"\n{'─'*50}")
    print(f"► Training {name}...")

    # Training time
    t0 = time.time()
    if name == 'XGBoost' and HAS_XGB:
        model.fit(X_tr_s, y_tr, sample_weight=sample_w)
    elif name == 'Random Forest' and hasattr(model, 'estimators_'):
        pass  # Already trained
    else:
        model.fit(X_tr_s, y_tr)
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s")

    # Prediction
    y_pred  = model.predict(X_te_s)
    y_proba = model.predict_proba(X_te_s)

    # Metrics
    auc = roc_auc_score(y_te, y_proba, multi_class='ovr', average='weighted')
    print(classification_report(y_te, y_pred, target_names=class_names, digits=3))
    print(f"  Weighted AUC: {auc:.4f}")

    # Single-window inference time
    single = X_te_s[:1]
    N = 500
    t0 = time.perf_counter()
    for _ in range(N):
        model.predict(single)
    inf_ms = (time.perf_counter() - t0) / N * 1000
    print(f"  Inference time (1 window): {inf_ms:.3f} ms")

    results[name] = {
        'model':      model,
        'y_pred':     y_pred,
        'y_proba':    y_proba,
        'auc':        auc,
        'inf_ms':     inf_ms,
        'train_time': train_time,
    }

# ─── SUMMARY TABLE ─────────────────────────────────────────
print(f"\n{'='*65}")
print(f"{'Model':<22} {'AUC':>8} {'Acc':>7} {'Inf(ms)':>10} {'Train(s)':>10}")
print(f"{'─'*65}")
for name, r in results.items():
    acc = np.mean(r['y_pred'] == y_te)
    rt  = '✓' if r['inf_ms'] < 100 else '✗'
    print(f"{name:<22} {r['auc']:>8.4f} {acc:>7.3f} {r['inf_ms']:>9.3f}ms {r['train_time']:>9.1f}s  {rt}")

print(f"\n  Dragon_Slice baseline AUC: 0.876–0.89")
print(f"  Real-time threshold: < 100ms/window")

# ─── FIGURES ──────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
n_cols = max(len(results), len(class_names))
gs  = gridspec.GridSpec(2, n_cols, figure=fig)

colors = ['#3b82f6', '#ef4444', '#22c55e', '#f97316']

# Top row: Confusion Matrix
for col, (name, r) in enumerate(results.items()):
    ax = fig.add_subplot(gs[0, col])
    cm = confusion_matrix(y_te, r['y_pred'])
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=False, cmap='Blues', values_format='d')
    ax.set_title(f"{name}\nAUC: {r['auc']:.4f}", fontsize=11, fontweight='bold')
    ax.tick_params(axis='x', rotation=35, labelsize=8)
    ax.tick_params(axis='y', labelsize=8)

# Bottom row: ROC curves (all models together, one subplot per class)
for cls_idx, cls_name in enumerate(class_names):
    ax = fig.add_subplot(gs[1, cls_idx])
    ax.set_title(f"ROC — {cls_name}", fontsize=10)
    ax.set_xlabel("False Positive Rate", fontsize=9)
    ax.set_ylabel("True Positive Rate", fontsize=9)
    ax.plot([0,1],[0,1],'k--', alpha=0.4, lw=1)

    for (name, r), col in zip(results.items(), colors):
        y_bin  = (y_te == cls_idx).astype(int)
        y_sc   = r['y_proba'][:, cls_idx]
        fpr, tpr, _ = roc_curve(y_bin, y_sc)
        auc_cls = roc_auc_score(y_bin, y_sc)
        ax.plot(fpr, tpr, color=col, lw=1.8, label=f"{name} ({auc_cls:.3f})")

    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0,1]); ax.set_ylim([0,1])

plt.suptitle("Wattson — Multi-Model Comparison", fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f"multi_model_comparison{TAG}.png", dpi=150, bbox_inches='tight')
plt.show()
print(f"\nSaved: multi_model_comparison{TAG}.png")

# ─── BAR CHART — AUC Comparison ───────────────────────
fig2, axes = plt.subplots(1, 2, figsize=(12, 5))

names_list = list(results.keys())
aucs  = [results[n]['auc']    for n in names_list]
infs  = [results[n]['inf_ms'] for n in names_list]
c_list = colors[:len(names_list)]

ax1, ax2 = axes

bars1 = ax1.barh(names_list, aucs, color=c_list, edgecolor='white', height=0.5)
ax1.axvline(0.89, color='red', linestyle='--', lw=1.5, label='Dragon_Slice upper (0.89)')
ax1.axvline(0.876, color='orange', linestyle='--', lw=1.2, alpha=0.7, label='Dragon_Slice lower (0.876)')
ax1.set_xlabel("Weighted AUC")
ax1.set_title("Model AUC Comparison", fontweight='bold')
ax1.set_xlim([0.7, 1.0])
ax1.legend(fontsize=8)
for bar, v in zip(bars1, aucs):
    ax1.text(v + 0.002, bar.get_y() + bar.get_height()/2,
             f'{v:.4f}', va='center', fontsize=10, fontweight='bold')

bars2 = ax2.barh(names_list, infs, color=c_list, edgecolor='white', height=0.5)
ax2.axvline(100, color='red', linestyle='--', lw=1.5, label='100ms (real-time limit)')
ax2.set_xlabel("Inference Time (ms/window)")
ax2.set_title("Inference Speed Comparison", fontweight='bold')
ax2.legend(fontsize=8)
for bar, v in zip(bars2, infs):
    ax2.text(v + 0.2, bar.get_y() + bar.get_height()/2,
             f'{v:.2f}ms', va='center', fontsize=10, fontweight='bold')

plt.suptitle("Wattson — Model Performance Summary", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"model_comparison_bars{TAG}.png", dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved: model_comparison_bars{TAG}.png")

# ─── SAVE MODELS ───────────────────────────────────────────
for name, r in results.items():
    fname = name.lower().replace(' ', '_').replace('(', '').replace(')', '') + f"_model{TAG}.pkl"
    with open(fname, "wb") as f:
        pickle.dump({'model': r['model'], 'scaler': scaler, 'le': le}, f)
    print(f"Saved: {fname}")
