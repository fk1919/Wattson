"""
Reconstructs the 5-seed XGBoost vs Random Forest comparison from
multi_seed_stability.py and adds a real paired t-test between the two
models' per-seed AUC scores (scipy.stats.ttest_rel). This was missing --
a figure title elsewhere hardcoded "paired t-test: p=0.0107" with no
traceable computation. This script produces a verifiable, reproducible
number to replace it.
"""
import numpy as np
from scipy.stats import ttest_rel
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

xgb_aucs, rf_aucs = [], []
xgb_f1s, rf_f1s = [], []

for seed in SEEDS:
    tr_idx, te_idx = session_split(seed)
    X_tr, X_te = X_all[tr_idx], X_all[te_idx]
    y_tr, y_te = y_all[tr_idx], y_all[te_idx]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                         random_state=seed, eval_metric='mlogloss', n_jobs=-1)
    xgb.fit(X_tr_s, y_tr)
    proba = xgb.predict_proba(X_te_s)
    pred = xgb.predict(X_te_s)
    auc = roc_auc_score(y_te, proba, multi_class='ovr', average='weighted')
    f1 = f1_score(y_te, pred, average='macro')
    xgb_aucs.append(auc)
    xgb_f1s.append(f1)

    rf = RandomForestClassifier(n_estimators=200, max_depth=None,
                                 class_weight='balanced', random_state=seed, n_jobs=-1)
    rf.fit(X_tr_s, y_tr)
    proba2 = rf.predict_proba(X_te_s)
    pred2 = rf.predict(X_te_s)
    auc2 = roc_auc_score(y_te, proba2, multi_class='ovr', average='weighted')
    f12 = f1_score(y_te, pred2, average='macro')
    rf_aucs.append(auc2)
    rf_f1s.append(f12)

    print(f"seed={seed:6d}  XGBoost AUC={auc:.4f} F1={f1:.4f}  |  RF AUC={auc2:.4f} F1={f12:.4f}")

xgb_aucs = np.array(xgb_aucs)
rf_aucs = np.array(rf_aucs)

print(f"\n{'='*60}")
print(f"XGBoost AUC: {xgb_aucs.mean():.4f} +/- {xgb_aucs.std():.4f}")
print(f"RF      AUC: {rf_aucs.mean():.4f} +/- {rf_aucs.std():.4f}")

t_stat, p_val = ttest_rel(xgb_aucs, rf_aucs)
print(f"\nPaired t-test (XGBoost vs RF AUC, n=5 seeds): t={t_stat:.4f}, p={p_val:.4f}")

f1_stat, f1_p = ttest_rel(np.array(xgb_f1s), np.array(rf_f1s))
print(f"Paired t-test (XGBoost vs RF macro-F1, n=5 seeds): t={f1_stat:.4f}, p={f1_p:.4f}")
