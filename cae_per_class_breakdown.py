"""
Using the saved CAE (256, dropout) MSE scores, breaks down the binary
Anomaly label by original attack class (DoS/PortScan/SSH_Bruteforce)
and computes a SEPARATE AUC FOR EACH CLASS (against Normal). The reference
paper only reports a pooled "Anomaly" AUC -- this breakdown shows which
attack type is detected more easily/with more difficulty by the CAE.
No retraining is needed; it simply reuses the already-saved scores.
"""
import numpy as np
import re
from sklearn.metrics import roc_auc_score

d = np.load("cae_256_dropout_mse_scores.npz", allow_pickle=True)
mse_scores = d['mse_scores']
y_test = d['y_test']
sess_test = d['sess_test'].astype(str)

LABEL_MAP = {'normal': 'Normal', 'portscan': 'PortScan',
             'ssh_bruteforce': 'SSH_Bruteforce', 'dos': 'DoS'}

def label_from_session(s):
    m = re.match(r'pi_(.+?)_\d{8}_\d{6}$', s)
    key = m.group(1) if m else ''
    return LABEL_MAP.get(key, 'Unknown')

session_labels = np.array([label_from_session(s) for s in sess_test])

# Apply MAF (MA=30) -- same post-processing as the main script
best_ma = 30
smoothed = np.zeros_like(mse_scores)
for s in np.unique(sess_test):
    mask = np.where(sess_test == s)[0]
    vals = mse_scores[mask]
    kernel = np.ones(best_ma) / best_ma
    smoothed[mask] = np.convolve(vals, kernel, mode='same')

normal_mask = (y_test == 0)
print(f"Number of normal windows: {normal_mask.sum()}")
print(f"\n{'Attack Class':<18}{'Windows':>10}{'Raw AUC':>10}{'MAF AUC':>10}")
for cls in ['DoS', 'PortScan', 'SSH_Bruteforce']:
    cls_mask = (session_labels == cls)
    n_cls = cls_mask.sum()
    if n_cls == 0:
        continue
    combined_mask = normal_mask | cls_mask
    y_bin = np.where(cls_mask[combined_mask], 1, 0)
    raw_auc = roc_auc_score(y_bin, mse_scores[combined_mask])
    maf_auc = roc_auc_score(y_bin, smoothed[combined_mask])
    print(f"{cls:<18}{n_cls:>10}{raw_auc:>10.4f}{maf_auc:>10.4f}")

print(f"\nThe reference paper only reports a pooled (all attack types combined) AUC,")
print(f"without a per-class breakdown -- this analysis is extra rigor beyond the reference.")
