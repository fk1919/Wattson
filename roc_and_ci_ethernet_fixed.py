import pickle
import numpy as np
from sklearn.metrics import roc_auc_score

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
y_proba = model.predict_proba(X_te_s)
y_pred = model.predict(X_te_s)

print("PER-CLASS ROC AUC (one-vs-rest) -- CORRECTED ETHERNET, XGBoost:")
for i, c in enumerate(classes):
    y_bin = (y_te == i).astype(int)
    auc_c = roc_auc_score(y_bin, y_proba[:, i])
    print(f"  {c:16s}: AUC = {auc_c:.4f}")

print(f"\nBOOTSTRAP 95% CONFIDENCE INTERVAL (session-based resample, N=1000):")
sess_te = sessions[test_idx]
unique_sess = np.unique(sess_te)
accs = []
aucs_boot = []
rng2 = np.random.RandomState(1)
correct_all = (y_pred == y_te)
for _ in range(1000):
    sample_sess = rng2.choice(unique_sess, size=len(unique_sess), replace=True)
    mask = np.isin(sess_te, sample_sess)
    if mask.sum() == 0:
        continue
    accs.append(correct_all[mask].mean())
    try:
        aucs_boot.append(roc_auc_score(y_te[mask], y_proba[mask], multi_class='ovr', average='weighted'))
    except ValueError:
        pass
accs = np.array(accs)
aucs_boot = np.array(aucs_boot)
lo, hi = np.percentile(accs, [2.5, 97.5])
alo, ahi = np.percentile(aucs_boot, [2.5, 97.5])
print(f"  Accuracy       : {accs.mean()*100:.2f}%  (95% CI: [{lo*100:.2f}%, {hi*100:.2f}%])  |  {len(unique_sess)} test sessions")
print(f"  Weighted AUC   : {aucs_boot.mean():.4f}  (95% CI: [{alo:.4f}, {ahi:.4f}])")
