import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, f1_score
import xgboost as xgb

data = np.load("feature_matrix_ethernet_fixed.npz", allow_pickle=True)
X_full = data['X'].astype(np.float32)
y_raw = data['y'].astype(str)
sessions = data['sessions'].astype(str)

le = LabelEncoder()
y = le.fit_transform(y_raw)
classes = le.classes_

rng = np.random.RandomState(42)
train_idx, test_idx = [], []
for cls_i in np.unique(y):
    cls_sessions = np.unique(sessions[y == cls_i])
    rng.shuffle(cls_sessions)
    n = len(cls_sessions)
    n_test = max(1, round(n * 0.2))
    n_train = n - n_test
    tr_sess = set(cls_sessions[:n_train])
    te_sess = set(cls_sessions[n_train:])
    cls_mask = (y == cls_i)
    train_idx.extend(np.where(cls_mask & np.isin(sessions, list(tr_sess)))[0])
    test_idx.extend(np.where(cls_mask & np.isin(sessions, list(te_sess)))[0])
train_idx, test_idx = np.array(train_idx), np.array(test_idx)

def run(X, label):
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                               eval_metric='mlogloss', tree_method='hist',
                               random_state=42, n_jobs=-1)
    from sklearn.utils.class_weight import compute_class_weight
    w = compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
    sw = np.array([w[yi] for yi in y_tr])
    model.fit(X_tr_s, y_tr, sample_weight=sw)
    y_pred = model.predict(X_te_s)
    y_proba = model.predict_proba(X_te_s)
    auc = roc_auc_score(y_te, y_proba, multi_class='ovr', average='weighted')
    macro_f1 = f1_score(y_te, y_pred, average='macro')
    print(f"\n{'='*55}\n{label}\n{'='*55}")
    print(classification_report(y_te, y_pred, target_names=classes, digits=3))
    print(f"Weighted AUC: {auc:.4f}  |  Macro F1: {macro_f1:.4f}")
    return auc, macro_f1

auc40, f1_40 = run(X_full[:, :40], "40 FEATURES (statistical + FFT, packet-arrival-rate EXCLUDED)")
auc43, f1_43 = run(X_full, "43 FEATURES (+ zero-crossing, peak-rate, spectral-entropy)")

print(f"\n{'='*55}\nSUMMARY (CORRECTED ETHERNET DATA)\n{'='*55}")
print(f"AUC:      {auc40:.4f} -> {auc43:.4f}  ({(auc43-auc40)*100:+.2f} points)")
print(f"Macro F1: {f1_40:.4f} -> {f1_43:.4f}  ({(f1_43-f1_40)*100:+.2f} points)")
