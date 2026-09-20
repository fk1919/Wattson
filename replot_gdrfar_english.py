"""
Replots the GDR-vs-FAR curve from the saved CAE MSE scores (WITHOUT
needing retraining) with English labels.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

d = np.load("cae_256_dropout_mse_scores.npz", allow_pickle=True)
mse_scores = d['mse_scores']
y_test = d['y_test']
sess_test = d['sess_test'].astype(str)

WIN_SIZE = 4882
WINDOWS_PER_HOUR = 3600 / (WIN_SIZE / 48828.0)
best_ma = 30

smoothed = np.zeros_like(mse_scores)
for s in np.unique(sess_test):
    mask = np.where(sess_test == s)[0]
    vals = mse_scores[mask]
    kernel = np.ones(best_ma) / best_ma
    smoothed[mask] = np.convolve(vals, kernel, mode='same')

normal_mask = (y_test == 0)
anomaly_mask = (y_test == 1)
n_normal = normal_mask.sum()

normal_scores = smoothed[normal_mask]
pctls = np.linspace(0.0, 100.0, 300)
thresholds = np.percentile(normal_scores, pctls)

gdr_far = []
for pctl, th in zip(pctls, thresholds):
    tp = np.sum(smoothed[anomaly_mask] >= th)
    fp = np.sum(smoothed[normal_mask] >= th)
    gdr = tp / anomaly_mask.sum() * 100
    far_per_hour = (fp / n_normal) * WINDOWS_PER_HOUR
    gdr_far.append((pctl, th, gdr, far_per_hour))

fars = [g[3] for g in gdr_far]
gdrs = [g[2] for g in gdr_far]

# ---- Verified palette ----
BLUE = '#2a78d6'
plt.figure(figsize=(7, 5))
plt.plot(fars, gdrs, marker='.', color=BLUE, label="Wattson (256, dropout)")
plt.axhline(65, color='gray', linestyle='--', alpha=0.5, label="Reference dropout @0 FA/h (~65%)")
plt.axhline(50, color='lightgray', linestyle=':', alpha=0.5, label="Reference non-dropout @0 FA/h (~50%)")
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.xlabel("False Alarm Rate (per hour)")
plt.ylabel("Good Detection Rate (%)")
plt.title("GDR vs FAR — Dragon_Slice Methodology Replication (Corrected)")
plt.grid(alpha=0.3)
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig("figures_final/gdr_far_curve_ethernet_fixed_v2.png", dpi=150)
print("Saved: figures_final/gdr_far_curve_ethernet_fixed_v2.png (English)")
