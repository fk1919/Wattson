"""
Wattson — ML Training Pipeline
Random Forest baseline + evaluation
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, ConfusionMatrixDisplay)
from sklearn.utils.class_weight import compute_class_weight
import time

# ─── LOAD DATA ────────────────────────────────────────────
print("Loading feature_matrix.npz...")
data = np.load("feature_matrix.npz", allow_pickle=True)
X = data['X']
y_raw = data['y']

# Merge PortScan subclasses
y_raw = np.where(np.char.startswith(y_raw.astype(str), 'Port_Scan'), 'PortScan', y_raw)

print(f"X: {X.shape}, unique y: {np.unique(y_raw, return_counts=True)}")

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y_raw)
class_names = le.classes_
print(f"\nClasses: {class_names}")

# ─── CLASS IMBALANCE ────────────────────────────────────────
# Compute class weights manually (instead of sklearn's class_weight='balanced')
weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = dict(zip(np.unique(y), weights))
print(f"\nClass weights: { {class_names[k]:round(v,2) for k,v in class_weight_dict.items()} }")

# ─── TRAIN/TEST SPLIT ──────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"\nTraining: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

# ─── MODEL 1: RANDOM FOREST ────────────────────────────────
print("\n─── Training Random Forest... ───")
t0 = time.time()
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
print(f"Training time: {time.time()-t0:.1f}s")

y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)

print("\n── Classification Report (Random Forest) ──")
print(classification_report(y_test, y_pred_rf, target_names=class_names))

# AUC (multiclass)
auc_rf = roc_auc_score(y_test, y_prob_rf, multi_class='ovr', average='weighted')
print(f"Weighted AUC: {auc_rf:.4f}")

# ─── CONFUSION MATRIX ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm = confusion_matrix(y_test, y_pred_rf)
disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title(f"Random Forest\nWeighted AUC: {auc_rf:.4f}")
axes[0].tick_params(axis='x', rotation=30)

# ─── MODEL 2: GRADIENT BOOSTING ────────────────────────────
print("\n─── Training Gradient Boosting... ───")
t0 = time.time()
gb = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
gb.fit(X_train, y_train)
print(f"Training time: {time.time()-t0:.1f}s")

y_pred_gb = gb.predict(X_test)
y_prob_gb = gb.predict_proba(X_test)

print("\n── Classification Report (Gradient Boosting) ──")
print(classification_report(y_test, y_pred_gb, target_names=class_names))

auc_gb = roc_auc_score(y_test, y_prob_gb, multi_class='ovr', average='weighted')
print(f"Weighted AUC: {auc_gb:.4f}")

cm2 = confusion_matrix(y_test, y_pred_gb)
disp2 = ConfusionMatrixDisplay(cm2, display_labels=class_names)
disp2.plot(ax=axes[1], colorbar=False, cmap='Oranges')
axes[1].set_title(f"Gradient Boosting\nWeighted AUC: {auc_gb:.4f}")
axes[1].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig("model_results.png", dpi=150)
plt.show()
print("\nPlot saved: model_results.png")

# ─── FEATURE IMPORTANCES ─────────────────────────────────
feat_names = (['mean','std','rms','peak2pk','skew','kurt','crest','energy']
              + [f'fft_{i}' for i in range(32)])
importances = rf.feature_importances_
top10_idx = np.argsort(importances)[::-1][:10]
print("\n─── Top 10 Most Important Features (RF) ───")
for i, idx in enumerate(top10_idx):
    print(f"  {i+1:2d}. {feat_names[idx]:12s}: {importances[idx]:.4f}")

# ─── SAVE MODEL ───────────────────────────────────────────
import pickle
with open("rf_model.pkl", "wb") as f:
    pickle.dump({'model': rf, 'scaler': scaler, 'le': le}, f)
print("\nModel saved: rf_model.pkl")
print("\nCompare against the Dragon_Slice baseline (AUC 0.876-0.89)!")
print(f"Our RF AUC: {auc_rf:.4f}  |  GB AUC: {auc_gb:.4f}")
