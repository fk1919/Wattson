import pickle
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score

# ==================== XGBoost ====================
with open("xgboost_model_ethernet_fixed.pkl", "rb") as f:
    b = pickle.load(f)
model, scaler, le = b['model'], b['scaler'], b['le']
classes = le.classes_

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
X_te, y_te = X[test_idx], y[test_idx]
X_te_s = scaler.transform(X_te)
y_pred = model.predict(X_te_s)
y_proba = model.predict_proba(X_te_s)

print("=" * 60)
print("XGBOOST (ETHERNET CORRECTED) - CONFUSION MATRIX (row=actual, column=predicted)")
print("=" * 60)
cm = confusion_matrix(y_te, y_pred)
print("Classes:", list(classes))
print(cm)
for i, c in enumerate(classes):
    row = cm[i]; total = row.sum()
    print(f"{c:16s}: " + " ".join(f"{classes[j]}={row[j]}({row[j]/total*100:.1f}%)" for j in range(len(classes))))

normal_idx = list(classes).index('Normal')
normal_mask = (y_te == normal_idx)
fp = np.sum((y_te == normal_idx) & (y_pred != normal_idx))
fp_rate = fp / normal_mask.sum()
print(f"\nXGBoost FALSE POSITIVE RATE (Normal misclassified as attack): {fp}/{normal_mask.sum()} = {fp_rate*100:.2f}%")
auc = roc_auc_score(y_te, y_proba, multi_class='ovr', average='weighted')
print(f"XGBoost Weighted AUC: {auc:.4f}  |  Test session count: {len(set(sessions[test_idx]))}")

# ==================== Random Forest ====================
print()
print("=" * 60)
print("RANDOM FOREST (ETHERNET CORRECTED) - CONFUSION MATRIX")
print("=" * 60)
with open("random_forest_model_ethernet_fixed.pkl", "rb") as f:
    b2 = pickle.load(f)
rf_model, rf_scaler, rf_le = b2['model'], b2['scaler'], b2['le']
# Uses the SAME session-based test split (test_idx) -- we re-encode y_raw with
# rf_le only because the label encoding may differ, but WHICH SAMPLES are in
# the test set is exactly identical to XGBoost (fair comparison).
y2 = rf_le.transform(y_raw)
test_idx2 = test_idx
X_te2, y_te2 = X[test_idx2], y2[test_idx2]
X_te2_s = rf_scaler.transform(X_te2)
y_pred2 = rf_model.predict(X_te2_s)
y_proba2 = rf_model.predict_proba(X_te2_s)

cm2 = confusion_matrix(y_te2, y_pred2)
print("Classes:", list(rf_le.classes_))
print(cm2)
for i, c in enumerate(rf_le.classes_):
    row = cm2[i]; total = row.sum()
    print(f"{c:16s}: " + " ".join(f"{rf_le.classes_[j]}={row[j]}({row[j]/total*100:.1f}%)" for j in range(len(rf_le.classes_))))

normal_idx2 = list(rf_le.classes_).index('Normal')
normal_mask2 = (y_te2 == normal_idx2)
fp2 = np.sum((y_te2 == normal_idx2) & (y_pred2 != normal_idx2))
fp_rate2 = fp2 / normal_mask2.sum()
print(f"\nRF FALSE POSITIVE RATE: {fp2}/{normal_mask2.sum()} = {fp_rate2*100:.2f}%")
auc2 = roc_auc_score(y_te2, y_proba2, multi_class='ovr', average='weighted')
print(f"RF Weighted AUC: {auc2:.4f}")

# ==================== CNN ====================
print()
print("=" * 60)
print("CNN (ETHERNET CORRECTED) - CONFUSION MATRIX")
print("=" * 60)

import torch
import torch.nn as nn

class ResBlock1D(nn.Module):
    def __init__(self, channels, kernel_size):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(channels), nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(x + self.conv(x))

class CNN1D(nn.Module):
    def __init__(self, win_size, n_classes):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=64, stride=4, padding=32, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(4))
        self.res1 = ResBlock1D(32, 7)
        self.stage2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=16, stride=2, padding=8, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4))
        self.res2 = ResBlock1D(64, 5)
        self.stage3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=8, padding=4, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3))
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(0.4), nn.Linear(64, n_classes))
    def forward(self, x):
        x = self.stage1(x); x = self.res1(x)
        x = self.stage2(x); x = self.res2(x)
        x = self.stage3(x); x = self.gap(x)
        return self.classifier(x)

cnn_data = np.load("windowed_data_ethernet_fixed.npz", allow_pickle=True)
X_raw = cnn_data['X']
y_raw2 = cnn_data['y'].astype(str)
sessions2 = cnn_data['sessions'].astype(str)
le3 = le
y3 = le3.transform(y_raw2)

rng3 = np.random.RandomState(42)
test_idx3 = []
for cls_i in np.unique(y3):
    cls_sessions = np.unique(sessions2[y3 == cls_i])
    rng3.shuffle(cls_sessions)
    n = len(cls_sessions)
    n_test = max(1, round(n * 0.2))
    n_val = max(1, round(n * 0.2))
    n_train = n - n_val - n_test
    if n_train < 1:
        n_train, n_val, n_test = n - 2, 1, 1
    te_sess = set(cls_sessions[n_train + n_val:])
    cls_mask = (y3 == cls_i)
    test_idx3.extend(np.where(cls_mask & np.isin(sessions2, list(te_sess)))[0])
test_idx3 = np.array(test_idx3)
X_te3, y_te3 = X_raw[test_idx3], y3[test_idx3]

model_cnn = CNN1D(X_raw.shape[1], len(classes))
model_cnn.load_state_dict(torch.load("cnn_best_ethernet_fixed.pt", map_location='cpu'))
model_cnn.eval()

all_preds, all_probs = [], []
with torch.no_grad():
    BATCH = 256
    for i in range(0, len(X_te3), BATCH):
        Xb = torch.tensor(X_te3[i:i+BATCH][:, np.newaxis, :], dtype=torch.float32)
        out = model_cnn(Xb)
        probs = torch.softmax(out, dim=1).numpy()
        all_probs.append(probs)
        all_preds.append(np.argmax(probs, axis=1))
y_pred3 = np.concatenate(all_preds)
y_proba3 = np.vstack(all_probs)

cm3 = confusion_matrix(y_te3, y_pred3)
print("Classes:", list(classes))
print(cm3)
for i, c in enumerate(classes):
    row = cm3[i]; total = row.sum()
    print(f"{c:16s}: " + " ".join(f"{classes[j]}={row[j]}({row[j]/total*100:.1f}%)" for j in range(len(classes))))

normal_mask3 = (y_te3 == normal_idx)
fp3 = np.sum((y_te3 == normal_idx) & (y_pred3 != normal_idx))
fp_rate3 = fp3 / normal_mask3.sum()
print(f"\nCNN FALSE POSITIVE RATE: {fp3}/{normal_mask3.sum()} = {fp_rate3*100:.2f}%")
auc3 = roc_auc_score(y_te3, y_proba3, multi_class='ovr', average='weighted')
print(f"CNN Weighted AUC: {auc3:.4f}  |  Test session count: {len(set(sessions2[test_idx3]))}")
