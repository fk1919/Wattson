"""
Repeated training for XGBoost and RF with 5 different random seeds
(session-based train/test split + model init) -- reports mean+/-std AUC
and macro-F1. The reference paper did not do this, but Q1 journal
reviewers may expect a statistical confidence interval.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score
from xgboost import XGBClassifier

data = np.load("feature_matrix_ethernet_fixed.npz", allow_pickle=True)
X_all = data['X'].astype(np.float32)
y_raw = data['y'].astype(str)
sessions = data['sessions'].astype(str)

le = LabelEncoder()
y_all = le.fit_transform(y_raw)
classes = le.classes_

SEEDS = [42, 7, 123, 2024, 31415]

def session_split(seed):
    rng = np.random.RandomState(seed)
    train_idx, test_idx = [], []
    for cls_i in np.unique(y_all):
        cls_sessions = np.unique(sessions[y_all == cls_i])
        rng.shuffle(cls_sessions)
        n = len(cls_sessions)
        n_test = max(1, round(n * 0.2))
        n_train = n - n_test
        tr_sess = set(cls_sessions[:n_train])
        te_sess = set(cls_sessions[n_train:])
        cls_mask = (y_all == cls_i)
        train_idx.extend(np.where(cls_mask & np.isin(sessions, list(tr_sess)))[0])
        test_idx.extend(np.where(cls_mask & np.isin(sessions, list(te_sess)))[0])
    return np.array(train_idx), np.array(test_idx)

results = {'XGBoost': {'auc': [], 'f1': []}, 'RandomForest': {'auc': [], 'f1': []}}

for seed in SEEDS:
    tr_idx, te_idx = session_split(seed)
    X_tr, X_te = X_all[tr_idx], X_all[te_idx]
    y_tr, y_te = y_all[tr_idx], y_all[te_idx]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # XGBoost
    xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                         random_state=seed, eval_metric='mlogloss', n_jobs=-1)
    xgb.fit(X_tr_s, y_tr)
    proba = xgb.predict_proba(X_te_s)
    pred = xgb.predict(X_te_s)
    auc = roc_auc_score(y_te, proba, multi_class='ovr', average='weighted')
    f1 = f1_score(y_te, pred, average='macro')
    results['XGBoost']['auc'].append(auc)
    results['XGBoost']['f1'].append(f1)
    print(f"seed={seed:6d}  XGBoost   AUC={auc:.4f}  macroF1={f1:.4f}")

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=None,
                                 class_weight='balanced', random_state=seed, n_jobs=-1)
    rf.fit(X_tr_s, y_tr)
    proba2 = rf.predict_proba(X_te_s)
    pred2 = rf.predict(X_te_s)
    auc2 = roc_auc_score(y_te, proba2, multi_class='ovr', average='weighted')
    f12 = f1_score(y_te, pred2, average='macro')
    results['RandomForest']['auc'].append(auc2)
    results['RandomForest']['f1'].append(f12)
    print(f"seed={seed:6d}  RF        AUC={auc2:.4f}  macroF1={f12:.4f}")

print(f"\n{'='*60}\nSUMMARY -- 5-SEED MEAN +/- STD\n{'='*60}")
for model_name, r in results.items():
    aucs = np.array(r['auc'])
    f1s = np.array(r['f1'])
    print(f"{model_name:14s}  AUC={aucs.mean():.4f} +/- {aucs.std():.4f}  (min={aucs.min():.4f}, max={aucs.max():.4f})")
    print(f"{'':14s}  macroF1={f1s.mean():.4f} +/- {f1s.std():.4f}  (min={f1s.min():.4f}, max={f1s.max():.4f})")
