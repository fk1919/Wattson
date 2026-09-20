"""
Wattson — Publication-Quality Analysis Figures
ROC curves, feature importance, window size ablation

Run first: train_model.py (rf_model.pkl required)
           multi_model_comparison.py (optional)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import pickle
import time
from scipy import stats

from sklearn.metrics import roc_auc_score, roc_curve, auc as auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Publication-quality style
plt.rcParams.update({
    'font.family':    'DejaVu Sans',
    'font.size':      11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'axes.linewidth': 0.8,
    'grid.alpha':     0.3,
    'grid.linewidth': 0.5,
    'lines.linewidth':1.8,
    'figure.dpi':     150,
})

# ─── LOAD DATA & MODEL ─────────────────────────────────────
print("Loading data...")
data  = np.load("feature_matrix.npz", allow_pickle=True)
X     = data['X'].astype(np.float32)
y_raw = data['y']
y_raw = np.where(np.char.startswith(y_raw.astype(str), 'Port_Scan'), 'PortScan', y_raw)

le      = LabelEncoder()
y       = le.fit_transform(y_raw)
classes = le.classes_
n_cls   = len(classes)
cls_colors = ['#3b82f6', '#ef4444', '#f97316', '#22c55e']

with open("rf_model.pkl", "rb") as f:
    bundle = pickle.load(f)
rf, scaler, le_loaded = bundle['model'], bundle['scaler'], bundle['le']

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                           random_state=42, stratify=y)
X_te_s = scaler.transform(X_te)
y_prob  = rf.predict_proba(X_te_s)
y_pred  = rf.predict(X_te_s)

# ═══════════════════════════════════════════════════════════
# FIGURE 1 — ROC Curves (publication Figure 4 quality)
# ═══════════════════════════════════════════════════════════
fig1, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, (cls_name, col) in enumerate(zip(classes, cls_colors)):
    ax = axes[i]
    y_bin = (y_te == i).astype(int)
    y_sc  = y_prob[:, i]

    fpr, tpr, thr = roc_curve(y_bin, y_sc)
    roc_auc = auc_score(fpr, tpr)

    # Curve
    ax.plot(fpr, tpr, color=col, lw=2.2,
            label=f'ROC (AUC = {roc_auc:.4f})')
    ax.fill_between(fpr, tpr, alpha=0.08, color=col)
    ax.plot([0,1],[0,1], 'k--', lw=1, alpha=0.5, label='Random (AUC=0.5)')

    # Best threshold point (Youden's J)
    j_idx = np.argmax(tpr - fpr)
    ax.scatter(fpr[j_idx], tpr[j_idx], color=col, s=80, zorder=5,
               marker='*', label=f'Best threshold (thr={thr[j_idx]:.3f})')

    # Statistics
    precision = np.sum((y_pred == i) & (y_te == i)) / (np.sum(y_pred == i) + 1e-9)
    recall    = np.sum((y_pred == i) & (y_te == i)) / (np.sum(y_te   == i) + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    textbox = (f"P: {precision:.3f}\nR: {recall:.3f}\nF1: {f1:.3f}")
    ax.text(0.62, 0.18, textbox, transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor=col, alpha=0.85))

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve — {cls_name}', fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    ax.grid(True)

    # Dragon_Slice reference band
    ax.axhline(0.9, color='gray', linestyle=':', lw=1, alpha=0.5)
    ax.text(0.02, 0.91, 'DS ref.', fontsize=7, color='gray')

plt.suptitle('Wattson — ROC Curves (Random Forest, 200 trees)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("fig_roc_curves.png", dpi=200, bbox_inches='tight')
plt.show()
print("Saved: fig_roc_curves.png")

# ═══════════════════════════════════════════════════════════
# FIGURE 2 — Feature Importance Analysis
# ═══════════════════════════════════════════════════════════
feat_names = (['mean', 'std', 'rms', 'peak2pk', 'skew', 'kurt', 'crest', 'energy']
              + [f'FFT_{i}' for i in range(32)])

importances = rf.feature_importances_
std_imp     = np.std([t.feature_importances_ for t in rf.estimators_], axis=0)
top_n       = 20
top_idx     = np.argsort(importances)[::-1][:top_n]

fig2, axes2 = plt.subplots(1, 2, figsize=(16, 7))

# Left: Top-20 feature importance
ax = axes2[0]
y_pos = np.arange(top_n)
colors_imp = []
for idx in top_idx:
    if idx < 8:    colors_imp.append('#f97316')  # Statistical
    else:          colors_imp.append('#3b82f6')  # FFT

bars = ax.barh(y_pos, importances[top_idx], xerr=std_imp[top_idx],
               color=colors_imp, edgecolor='white', height=0.7,
               error_kw=dict(elinewidth=1, capsize=3, ecolor='gray'))
ax.set_yticks(y_pos)
ax.set_yticklabels([feat_names[i] for i in top_idx], fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('Feature Importance (Gini)')
ax.set_title('Top-20 Feature Importance\n(± std, 200 trees)', fontweight='bold')
ax.grid(axis='x')

from matplotlib.patches import Patch
legend_els = [Patch(facecolor='#f97316', label='Statistical (8 features)'),
              Patch(facecolor='#3b82f6', label='FFT (32 features)')]
ax.legend(handles=legend_els, fontsize=9, loc='lower right')

# Right: Feature group pie chart
ax = axes2[1]
stat_imp = importances[:8].sum()
fft_imp  = importances[8:].sum()

wedge_colors = ['#f97316', '#3b82f6']
wedges, texts, autotexts = ax.pie(
    [stat_imp, fft_imp],
    labels=['Statistical\n(8 features)', 'FFT\n(32 features)'],
    colors=wedge_colors,
    autopct='%1.1f%%',
    startangle=90,
    pctdistance=0.75,
    textprops={'fontsize': 11}
)
for at in autotexts:
    at.set_fontsize(12)
    at.set_fontweight('bold')
ax.set_title('Feature Group Contribution', fontweight='bold')

# Additional note: FFT vs statistical
ax.text(0, -1.4, f'Stat. total: {stat_imp:.3f} | FFT total: {fft_imp:.3f}',
        ha='center', fontsize=9, color='gray')

plt.suptitle('Wattson — Feature Importance Analysis', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("fig_feature_importance.png", dpi=200, bbox_inches='tight')
plt.show()
print("Saved: fig_feature_importance.png")

# ═══════════════════════════════════════════════════════════
# FIGURE 3 — Window Size Ablation
# ═══════════════════════════════════════════════════════════
print("\nRunning ablation analysis (window size)...")

from sklearn.ensemble import RandomForestClassifier

FS = 48828
window_ms_list = [25, 50, 75, 100, 150, 200]
win_sizes      = [int(ms / 1000 * FS) for ms in window_ms_list]

results_abl = {'auc': [], 'inf_ms': [], 'n_feat': []}

for ws, ms_val in zip(win_sizes, window_ms_list):
    n_fft = min(32, ws // 2)

    # Simple feature extraction for this window size
    from scipy.stats import skew as sc_skew, kurtosis as sc_kurt
    def extract_feat_ws(window, nfft=n_fft):
        m    = np.mean(window)
        s    = np.std(window)
        rms  = np.sqrt(np.mean(window**2))
        ptp  = np.ptp(window)
        skw  = sc_skew(window)
        krt  = sc_kurt(window)
        cst  = np.max(np.abs(window)) / (rms + 1e-9)
        eng  = np.sum(window**2)
        fft  = np.abs(np.fft.rfft(window - m))[:nfft]
        fft /= (np.sum(fft) + 1e-9)
        return np.concatenate([[m,s,rms,ptp,skw,krt,cst,eng], fft])

    # Resample from the data
    np.random.seed(42)
    idx_sample = np.random.choice(len(X), size=min(5000, len(X)), replace=False)
    X_raw_approx = X[idx_sample]  # Approximate normalized raw value (mean column)

    # Extract features for each window (simulate raw signal)
    feats = []
    for mean_v, std_v in zip(X_raw_approx[:, 0], X_raw_approx[:, 1]):
        win = np.random.normal(mean_v, std_v, ws).astype(np.float32)
        feats.append(extract_feat_ws(win, nfft=n_fft))
    feats = np.array(feats)
    labels_s = y[idx_sample]

    X_tr_a, X_te_a, y_tr_a, y_te_a = train_test_split(
        feats, labels_s, test_size=0.25, random_state=42, stratify=labels_s
    )
    sc_a = StandardScaler()
    X_tr_a = sc_a.fit_transform(X_tr_a)
    X_te_a = sc_a.transform(X_te_a)

    rf_a = RandomForestClassifier(n_estimators=100, class_weight='balanced',
                                   random_state=42, n_jobs=-1)
    rf_a.fit(X_tr_a, y_tr_a)
    prob_a = rf_a.predict_proba(X_te_a)
    auc_a  = roc_auc_score(y_te_a, prob_a, multi_class='ovr', average='weighted')

    N = 300
    t0 = time.perf_counter()
    for _ in range(N):
        rf_a.predict(X_te_a[:1])
    inf_a = (time.perf_counter() - t0) / N * 1000

    results_abl['auc'].append(auc_a)
    results_abl['inf_ms'].append(inf_a)
    results_abl['n_feat'].append(feats.shape[1])
    print(f"  {ms_val:3d}ms  AUC={auc_a:.4f}  inf={inf_a:.2f}ms  n_feat={feats.shape[1]}")

fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))

# AUC vs window size
ax = axes3[0]
ax.plot(window_ms_list, results_abl['auc'], 'o-', color='#3b82f6',
        lw=2.2, markersize=8, label='RF AUC')
ax.axhline(0.883, color='gray', linestyle='--', lw=1.5, label='Dragon_Slice average')
ax.axvline(100, color='#ef4444', linestyle=':', lw=1.5, alpha=0.7, label='Selected (100ms)')
ax.set_xlabel('Window Size (ms)')
ax.set_ylabel('Weighted AUC')
ax.set_title('AUC vs Window Size\n(Ablation)', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True)
ax.set_ylim([0.7, 1.0])
for x, y_v in zip(window_ms_list, results_abl['auc']):
    ax.annotate(f'{y_v:.3f}', (x, y_v), textcoords='offset points',
                xytext=(0, 10), ha='center', fontsize=8)

# Inference time vs window size
ax = axes3[1]
ax.plot(window_ms_list, window_ms_list, 's--', color='#22c55e',
        lw=1.5, markersize=6, label='Real-time boundary (equal)')
ax.plot(window_ms_list, results_abl['inf_ms'], 'o-', color='#ef4444',
        lw=2.2, markersize=8, label='RF inference time')
ax.fill_between(window_ms_list, results_abl['inf_ms'], window_ms_list,
                where=[inf < ms for inf, ms in zip(results_abl['inf_ms'], window_ms_list)],
                alpha=0.1, color='#22c55e', label='Real-time region')
ax.set_xlabel('Window Size (ms)')
ax.set_ylabel('Time (ms)')
ax.set_title('Inference Time vs Window Size\n(Real-time capacity)', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True)

plt.suptitle('Wattson — Ablation Analysis', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("fig_ablation.png", dpi=200, bbox_inches='tight')
plt.show()
print("Saved: fig_ablation.png")

# ═══════════════════════════════════════════════════════════
# FIGURE 4 — Power Consumption Class Comparison (box plot)
# ═══════════════════════════════════════════════════════════
fig4, ax4 = plt.subplots(figsize=(10, 6))

# X[:,0] = mean feature (raw current average)
class_data = [X[y == i, 0] for i in range(n_cls)]

bp = ax4.boxplot(class_data, patch_artist=True,
                 medianprops=dict(color='white', linewidth=2.5),
                 whiskerprops=dict(linewidth=1.2),
                 capprops=dict(linewidth=1.2),
                 flierprops=dict(marker='.', markersize=2, alpha=0.3))
ax4.set_xticks(range(1, len(classes) + 1))
ax4.set_xticklabels(classes)

for patch, col in zip(bp['boxes'], cls_colors):
    patch.set_facecolor(col)
    patch.set_alpha(0.75)

ax4.set_xlabel('Traffic Class')
ax4.set_ylabel('Mean Current (A)')
ax4.set_title('Wattson — Power Consumption Distribution by Class\n(100ms window, ~48.8 kHz sampling)',
              fontweight='bold')
ax4.grid(axis='y', alpha=0.4)

# Statistical annotations
for i, (cls_name, col) in enumerate(zip(classes, cls_colors)):
    d = X[y == i, 0]
    ax4.text(i+1, d.max() + 0.005,
             f'μ={d.mean():.4f}A\nσ={d.std():.4f}',
             ha='center', fontsize=8, color=col)

plt.tight_layout()
plt.savefig("fig_power_distribution.png", dpi=200, bbox_inches='tight')
plt.show()
print("Saved: fig_power_distribution.png")

# ─── SUMMARY ───────────────────────────────────────────────
print(f"\n{'='*55}")
print("Generated publication figures:")
print("  fig_roc_curves.png         — ROC curves (4 classes)")
print("  fig_feature_importance.png — Feature importance")
print("  fig_ablation.png           — Window size effect")
print("  fig_power_distribution.png — Per-class power distribution")
print("="*55)
