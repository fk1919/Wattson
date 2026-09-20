"""
Validation using a chronological (time-ordered) train/test split (in response
to Lemus-Prieto et al. 2026 criticizing their own random split and suggesting
"a chronological split would be more appropriate, there is a leakage risk"):
instead of a random session-based split, this checks for temporal leakage by
training on EARLY sessions and testing on LATE sessions for EACH CLASS.
"""
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from xgboost import XGBClassifier

data = np.load("feature_matrix_ethernet_fixed.npz", allow_pickle=True)
X = data['X'].astype(np.float32)
y_raw = data['y'].astype(str)
sessions = data['sessions'].astype(str)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y_raw)
classes = le.classes_

def session_timestamp(s):
    m = re.search(r'(\d{8}_\d{6})$', s)
    return m.group(1) if m else s

train_idx, test_idx = [], []
for cls_i in np.unique(y):
    cls_mask = (y == cls_i)
    cls_sessions = np.unique(sessions[cls_mask])
    cls_sessions_sorted = sorted(cls_sessions, key=session_timestamp)
    n = len(cls_sessions_sorted)
    n_train = int(n * 0.8)
    tr_sess = set(cls_sessions_sorted[:n_train])
    te_sess = set(cls_sessions_sorted[n_train:])
    train_idx.extend(np.where(cls_mask & np.isin(sessions, list(tr_sess)))[0])
    test_idx.extend(np.where(cls_mask & np.isin(sessions, list(te_sess)))[0])
train_idx, test_idx = np.array(train_idx), np.array(test_idx)

print(f"Chronological split: train={len(train_idx)}  test={len(test_idx)}")
print(f"(comparison: the random session-based split had train~9506 test~2380)")

X_tr, X_te = X[train_idx], X[test_idx]
y_tr, y_te = y[train_idx], y[test_idx]

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)

model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                       random_state=42, eval_metric='mlogloss', n_jobs=-1)
model.fit(X_tr_s, y_tr)
y_pred = model.predict(X_te_s)
y_proba = model.predict_proba(X_te_s)

print("\n" + "=" * 55)
print("CHRONOLOGICAL SPLIT RESULT (XGBoost)")
print("=" * 55)
print(classification_report(y_te, y_pred, target_names=classes, digits=3))
auc = roc_auc_score(y_te, y_proba, multi_class='ovr', average='weighted')
f1 = f1_score(y_te, y_pred, average='macro')
print(f"Weighted AUC: {auc:.4f}  |  Macro F1: {f1:.4f}")
print(f"\nComparison -- random session-based split (previous result): AUC=0.9717, macroF1=0.8205")
print(f"Difference: AUC {auc-0.9717:+.4f}, macroF1 {f1-0.8205:+.4f}")
print("(If NO major drop is observed, this confirms there is no temporal leakage/drift risk)")
