"""
Generates publication-ready figures for all the additional analyses missing
from the paper (CAE ablation, confusion matrix, SHAP, noise robustness,
detection latency, tool generalization, WiFi-Ethernet comparison, multi-seed
stability, cost comparison, chronological split). Colors are taken from the
dataviz skill's verified (colorblind-safe) default palette.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.ticker as mticker

# ---- Verified categorical palette (dataviz skill references/palette.md) ----
BLUE    = '#2a78d6'
ORANGE  = '#eb6834'
AQUA    = '#1baf7a'
YELLOW  = '#eda100'
MAGENTA = '#e87ba4'
GREEN   = '#008300'
VIOLET  = '#4a3aa7'
RED     = '#e34948'

INK_PRIMARY   = '#0b0b0b'
INK_SECONDARY = '#52514e'
INK_MUTED     = '#898781'
GRIDLINE      = '#e1e0d9'
BASELINE      = '#c3c2b7'
SURFACE       = '#fcfcfb'

SEQ_BLUE = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Segoe UI', 'DejaVu Sans', 'Arial'],
    'axes.edgecolor': BASELINE,
    'axes.labelcolor': INK_PRIMARY,
    'text.color': INK_PRIMARY,
    'xtick.color': INK_SECONDARY,
    'ytick.color': INK_SECONDARY,
    'axes.grid': True,
    'grid.color': GRIDLINE,
    'grid.linewidth': 0.8,
    'figure.facecolor': SURFACE,
    'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE,
})

OUT = "figures_final"


def style_ax(ax, ygrid_only=True):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(BASELINE)
    ax.spines['bottom'].set_color(BASELINE)
    if ygrid_only:
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)


# ============================================================
# 1) CAE FULL ABLATION — Raw/MAF AUC, 6 configurations vs reference
# ============================================================
def fig_cae_ablation():
    configs = ['1024\n(no dropout)', '512\n(no dropout)', '256\n(no dropout)',
               '1024\n(dropout)', '512\n(dropout)', '256\n(dropout)']
    raw_auc = [0.9100, 0.9154, 0.9201, 0.9209, 0.9199, 0.9291]
    maf_auc = [0.9158, 0.9213, 0.9309, 0.9369, 0.9363, 0.9388]
    ref_raw = [0.778, 0.768, 0.772, 0.766, 0.759, 0.752]
    ref_maf = [0.876, 0.874, 0.868, 0.890, 0.886, 0.884]

    x = np.arange(len(configs))
    w = 0.2
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - 1.5*w, raw_auc, w, label='Proposed Raw AUC', color=BLUE)
    ax.bar(x - 0.5*w, maf_auc, w, label='Proposed MAF AUC', color=AQUA)
    ax.bar(x + 0.5*w, ref_raw, w, label='Reference Raw AUC', color=BLUE, alpha=0.35, hatch='//')
    ax.bar(x + 1.5*w, ref_maf, w, label='Reference MAF AUC', color=AQUA, alpha=0.35, hatch='//')

    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=9)
    ax.set_ylabel('AUC')
    ax.set_ylim(0.7, 1.0)
    ax.set_title('CAE Full Ablation: All 6 Configurations Beat the Reference', fontsize=12, fontweight='bold', pad=14)
    ax.legend(loc='lower right', frameon=False, fontsize=9, ncol=2)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_cae_ablation.png", dpi=200)
    plt.close(fig)
    print("Saved: fig_cae_ablation.png")


# ============================================================
# 2) CONFUSION MATRIX HEATMAPS — XGBoost, RF, CNN (Ethernet-fixed)
# ============================================================
def fig_confusion_matrices():
    classes = ['DoS', 'Normal', 'PortScan', 'SSH_BF']
    cms = {
        'XGBoost': np.array([[477, 100, 8, 3], [62, 521, 0, 5], [7, 1, 453, 141], [12, 8, 80, 502]]),
        'Random Forest': np.array([[474, 104, 5, 5], [50, 531, 1, 6], [8, 2, 424, 168], [6, 9, 47, 540]]),
        'CNN': np.array([[1006, 392, 46, 110], [1, 1176, 96, 281], [1, 21, 1088, 481], [1, 41, 63, 1486]]),
    }
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('seqblue', SEQ_BLUE)
    for ax, (name, cm) in zip(axes, cms.items()):
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                val = cm_norm[i, j]
                txt_color = 'white' if val > 0.55 else INK_PRIMARY
                ax.text(j, i, f"{cm[i,j]}\n({val*100:.0f}%)", ha='center', va='center',
                        fontsize=8.5, color=txt_color)
        ax.set_xticks(range(4)); ax.set_xticklabels(classes, fontsize=8.5, rotation=20)
        ax.set_yticks(range(4)); ax.set_yticklabels(classes, fontsize=8.5)
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted');
        if name == 'XGBoost':
            ax.set_ylabel('Actual')
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle('Confusion Matrix Comparison (Corrected Ethernet, session-based test)', fontsize=12, fontweight='bold', y=1.03)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_confusion_matrices.png", dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("Saved: fig_confusion_matrices.png")


# ============================================================
# 3) SHAP FEATURE IMPORTANCE
# ============================================================
def fig_shap_importance():
    feats = ['mean', 'rms', 'ptp', 'peak_rate', 'std', 'skew', 'zero_crossing_rate',
             'kurtosis', 'crest', 'fft_bin_23', 'fft_bin_26', 'fft_bin_7', 'fft_bin_4',
             'fft_bin_17', 'fft_bin_22']
    vals = [1.30671, 1.01976, 0.40280, 0.33358, 0.31170, 0.20277, 0.17154,
            0.12517, 0.09806, 0.05977, 0.05907, 0.05148, 0.04974, 0.04907, 0.04706]
    colors = [BLUE if 'fft' not in f else AQUA for f in feats]
    colors = [ORANGE if f in ('zero_crossing_rate', 'peak_rate') else c for f, c in zip(feats, colors)]

    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(feats))[::-1]
    ax.barh(y, vals, color=colors, height=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels(feats, fontsize=9.5)
    ax.set_xlabel('Mean |SHAP value|')
    ax.set_title('XGBoost Global SHAP Feature Importance (top 15)', fontsize=12, fontweight='bold', pad=12)
    legend_elems = [Patch(facecolor=BLUE, label='Statistical'),
                    Patch(facecolor=AQUA, label='FFT magnitude'),
                    Patch(facecolor=ORANGE, label='Packet-arrival-rate proxy')]
    ax.legend(handles=legend_elems, loc='lower right', frameon=False, fontsize=9)
    style_ax(ax, ygrid_only=False)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_shap_importance.png", dpi=200)
    plt.close(fig)
    print("Saved: fig_shap_importance.png")


# ============================================================
# 4) NOISE ROBUSTNESS
# ============================================================
def fig_noise_robustness():
    snr = [30, 20, 10, 5, 0, -5]
    snr_labels = ['Clean'] + [str(s) for s in snr]
    acc = [0.8608, 0.8667, 0.5892, 0.2892, 0.2650, 0.2542, 0.2550]
    auc = [0.9784, 0.9764, 0.9056, 0.7791, 0.7054, 0.6513, 0.6986]
    x = np.arange(len(snr_labels))

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(x, acc, marker='o', color=BLUE, linewidth=2.2, markersize=6, label='Accuracy')
    ax.plot(x, auc, marker='s', color=AQUA, linewidth=2.2, markersize=6, label='Weighted AUC')
    ax.axhline(0.25, color=BASELINE, linestyle='--', linewidth=1, label='Random guess (4 classes)')
    ax.set_xticks(x)
    ax.set_xticklabels(snr_labels)
    ax.set_xlabel('SNR (dB) — added synthetic AWGN noise')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.05)
    ax.set_title('Noise Robustness: Robust Down to 20 dB SNR, Collapses Below', fontsize=12, fontweight='bold', pad=12)
    ax.legend(loc='center left', frameon=False, fontsize=9)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_noise_robustness.png", dpi=200)
    plt.close(fig)
    print("Saved: fig_noise_robustness.png")


# ============================================================
# 5) DETECTION LATENCY
# ============================================================
def fig_detection_latency():
    labels = ['XGBoost\nPortScan', 'XGBoost\nSSH_BF', 'XGBoost\nPortScan(t2)', 'XGBoost\nSSH_BF(t2)',
              'RF\nDoS', 'WiFi\nDoS', 'WiFi\nPortScan']
    first_det = [0.74, 0.05, 1.46, 0.05, 12.27, None, 27.41]
    sustained = [2.19, 2.88, 5.33, 3.46, None, None, None]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(labels))
    w = 0.35
    first_vals = [v if v is not None else 0 for v in first_det]
    sustained_vals = [v if v is not None else 0 for v in sustained]
    bars1 = ax.bar(x - w/2, first_vals, w, label='First detection', color=BLUE)
    bars2 = ax.bar(x + w/2, sustained_vals, w, label='Sustained detection (≥3 windows)', color=AQUA)

    for i, v in enumerate(first_det):
        if v is None:
            ax.text(x[i] - w/2, 0.5, 'no\ndetection', ha='center', fontsize=7.5, color=RED)
    for i, v in enumerate(sustained):
        if v is None:
            ax.text(x[i] + w/2, 0.5, 'n/a', ha='center', fontsize=7.5, color=INK_MUTED)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel('Seconds')
    ax.set_title('Detection Latency: Sub-Second for Ethernet+XGBoost, WiFi Regime Impractical', fontsize=11.5, fontweight='bold', pad=12)
    ax.legend(loc='upper left', frameon=False, fontsize=9)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_detection_latency.png", dpi=200)
    plt.close(fig)
    print("Saved: fig_detection_latency.png")


# ============================================================
# 6) TOOL-SETTING / TOOL-GENERALIZATION ROBUSTNESS
# ============================================================
def fig_tool_robustness():
    tests = ['nmap T1\n(paranoid)', 'nmap T5\n(insane)', 'hydra -t1', 'hydra -t16',
             'hping3\nslow', 'hping3\nACK', 'netcat\n(different tool)', 'medusa\n(different tool)']
    baseline = [83, 83, 95, 95, 98, 98, 83, 95]  # approx. baseline performance at training setting
    result = [1.7, 43.2, 88.9, 94.6, 9.1, 2.2, 97.7, 85.7]
    colors = [RED, RED, GREEN, GREEN, RED, RED, GREEN, GREEN]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(tests))
    ax.bar(x, baseline, width=0.55, color=BASELINE, alpha=0.5, label='Baseline performance at training setting (~)')
    ax.bar(x, result, width=0.4, color=colors, label='Result with different setting/tool')
    for i, v in enumerate(result):
        ax.text(x[i], v + 2, f"{v}%", ha='center', fontsize=9, fontweight='bold', color=INK_PRIMARY)

    ax.set_xticks(x)
    ax.set_xticklabels(tests, fontsize=8.5)
    ax.set_ylabel('Detection rate (%)')
    ax.set_ylim(0, 110)
    ax.set_title('Tool-Setting Robustness: Fragile to Speed/Timing, Robust to Tool Identity', fontsize=12, fontweight='bold', pad=12)
    legend_elems = [Patch(facecolor=RED, label='Collapsed (parameter change)'),
                    Patch(facecolor=GREEN, label='Stayed robust (parameter or tool change)')]
    ax.legend(handles=legend_elems, loc='upper center', frameon=False, fontsize=9, ncol=2)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_tool_robustness.png", dpi=200)
    plt.close(fig)
    print("Saved: fig_tool_robustness.png")


# ============================================================
# 7) WiFi vs ETHERNET COMPARISON
# ============================================================
def fig_wifi_vs_ethernet():
    classes = ['DoS', 'PortScan', 'SSH_Bruteforce']
    wifi_live = [0.0, 2.0, 87.0]
    eth_live = [97.9, 84.2, 97.9]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(classes))
    w = 0.35
    ax = axes[0]
    ax.bar(x - w/2, wifi_live, w, label='WiFi (old)', color=RED)
    ax.bar(x + w/2, eth_live, w, label='Ethernet (corrected)', color=GREEN)
    ax.set_xticks(x); ax.set_xticklabels(classes)
    ax.set_ylabel('Live detection rate (%)')
    ax.set_ylim(0, 105)
    ax.set_title('Live Validation: WiFi vs Ethernet', fontsize=11, fontweight='bold')
    ax.legend(frameon=False, fontsize=9)
    style_ax(ax)

    ax2 = axes[1]
    models = ['XGBoost']
    wifi_auc = [0.8772]
    eth_auc = [0.9717]
    x2 = np.arange(len(models))
    ax2.bar(x2 - w/2, wifi_auc, w, label='WiFi (old)', color=RED)
    ax2.bar(x2 + w/2, eth_auc, w, label='Ethernet (corrected)', color=GREEN)
    ax2.set_xticks(x2); ax2.set_xticklabels(models)
    ax2.set_ylabel('Weighted AUC (offline)')
    ax2.set_ylim(0.7, 1.0)
    ax2.set_title('Offline AUC: WiFi vs Ethernet', fontsize=11, fontweight='bold')
    ax2.legend(frameon=False, fontsize=9)
    style_ax(ax2)

    fig.suptitle('Effect of the Network Interface Regime Change', fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_wifi_vs_ethernet.png", dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("Saved: fig_wifi_vs_ethernet.png")


# ============================================================
# 8) MULTI-SEED STABILITY
# ============================================================
def fig_multiseed_stability():
    xgb = np.array([0.9723, 0.9733, 0.9729, 0.9699, 0.9727])
    rf = np.array([0.9687, 0.9722, 0.9709, 0.9673, 0.9715])

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    bp = ax.boxplot([xgb, rf], tick_labels=['XGBoost', 'Random Forest'], patch_artist=True,
                     widths=0.45, medianprops=dict(color=INK_PRIMARY, linewidth=1.5))
    for patch, color in zip(bp['boxes'], [BLUE, AQUA]):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    for i, data in enumerate([xgb, rf]):
        xs = np.random.RandomState(0).normal(i + 1, 0.03, size=len(data))
        ax.scatter(xs, data, color=INK_PRIMARY, s=22, zorder=5)

    ax.set_ylabel('Weighted AUC (5 different random seeds)')
    ax.set_title('Multi-Seed Statistical Stability\n(paired t-test: p=0.0107, XGBoost significantly better)',
                  fontsize=11.5, fontweight='bold', pad=12)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_multiseed_stability.png", dpi=200)
    plt.close(fig)
    print("Saved: fig_multiseed_stability.png")


# ============================================================
# 9) COST COMPARISON (across the literature, log scale)
# ============================================================
def fig_cost_comparison():
    names = ['Myridakis et al.\n(resistor+Arduino)', 'Wattson\n(Proposed, ACS712+MCP3008)',
              'DeepAuditor\n(custom INA219 board)', 'Wijethilaka et al.\n(RPi4 network-based NIDS)',
              'Monsoon\nPower Monitor', 'Lightbody et al.\n(Keysight N6705A)']
    costs = [2, 4, 25, 75, 929, 6500]
    colors = [AQUA, GREEN, BLUE, BLUE, ORANGE, RED]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    y = np.arange(len(names))[::-1]
    bars = ax.barh(y, costs, color=colors, height=0.6)
    ax.set_xscale('log')
    ax.set_xlabel('Hardware cost (USD, log scale)')
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9.5)
    for i, c in enumerate(costs):
        ax.text(c * 1.15, y[i], f"${c:,}", va='center', fontsize=9, fontweight='bold')
    ax.set_title('Hardware Cost Spectrum Across the Power Side-Channel IDS Literature', fontsize=12, fontweight='bold', pad=12)
    style_ax(ax, ygrid_only=False)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, which='both', alpha=0.4)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_cost_comparison.png", dpi=200)
    plt.close(fig)
    print("Saved: fig_cost_comparison.png")


# ============================================================
# 10) CHRONOLOGICAL vs RANDOM SPLIT
# ============================================================
def fig_chronological_split():
    metrics = ['Weighted AUC', 'Macro F1']
    random_split = [0.9717, 0.8205]
    chrono_split = [0.9734, 0.8346]

    x = np.arange(len(metrics))
    w = 0.32
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.bar(x - w/2, random_split, w, label='Random session-based split', color=BLUE)
    ax.bar(x + w/2, chrono_split, w, label='Chronological (time-ordered) split', color=AQUA)
    for i in range(len(metrics)):
        ax.text(x[i]-w/2, random_split[i]+0.01, f"{random_split[i]:.3f}", ha='center', fontsize=8.5)
        ax.text(x[i]+w/2, chrono_split[i]+0.01, f"{chrono_split[i]:.3f}", ha='center', fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylim(0.7, 1.0)
    ax.set_title('Temporal Leakage Check: No Difference', fontsize=12, fontweight='bold', pad=12)
    ax.legend(loc='lower center', frameon=False, fontsize=9)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_chronological_split.png", dpi=200)
    plt.close(fig)
    print("Saved: fig_chronological_split.png")


# ============================================================
# 11) FALSE POSITIVE TEST UNDER BENIGN LOAD
# ============================================================
def fig_benign_false_positive():
    tests = ['Idle\n(reference FP)', 'SCP file\ntransfer', 'CPU load\n(repeat-1)',
              'CPU load\n(repeat-2)', 'Disk I/O\n(isolated)']
    fp_rate = [11.4, 100, 100, 100, 100]
    colors = [GREEN, RED, RED, RED, RED]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    x = np.arange(len(tests))
    ax.bar(x, fp_rate, color=colors, width=0.55)
    for i, v in enumerate(fp_rate):
        ax.text(x[i], v + 2, f"%{v:.0f}", ha='center', fontsize=10, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(tests, fontsize=9)
    ax.set_ylabel('False "attack" alarm rate (%)')
    ax.set_ylim(0, 115)
    ax.set_title('Benign-Load Tests: Legitimate Activity Consistently Triggers False Alarms', fontsize=11.5, fontweight='bold', pad=12)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_benign_false_positive.png", dpi=200)
    plt.close(fig)
    print("Saved: fig_benign_false_positive.png")


# ============================================================
# 12) CAE PER-CLASS BREAKDOWN
# ============================================================
def fig_cae_per_class():
    classes = ['PortScan', 'SSH_Bruteforce', 'DoS']
    raw_auc = [0.9850, 0.9764, 0.8101]
    maf_auc = [0.9893, 0.9897, 0.8338]

    x = np.arange(len(classes))
    w = 0.32
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.bar(x - w/2, raw_auc, w, label='Raw AUC', color=BLUE)
    ax.bar(x + w/2, maf_auc, w, label='MAF AUC', color=AQUA)
    for i in range(len(classes)):
        ax.text(x[i]-w/2, raw_auc[i]+0.01, f"{raw_auc[i]:.3f}", ha='center', fontsize=8.5)
        ax.text(x[i]+w/2, maf_auc[i]+0.01, f"{maf_auc[i]:.3f}", ha='center', fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(classes)
    ax.set_ylabel('AUC (Normal vs. this attack type)')
    ax.set_ylim(0.7, 1.02)
    ax.set_title('CAE Per-Attack-Type Breakdown: DoS Notably Weaker', fontsize=12, fontweight='bold', pad=12)
    ax.legend(loc='lower left', frameon=False, fontsize=9)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_cae_per_class.png", dpi=200)
    plt.close(fig)
    print("Saved: fig_cae_per_class.png")


if __name__ == '__main__':
    import os
    os.makedirs(OUT, exist_ok=True)
    fig_cae_ablation()
    fig_confusion_matrices()
    fig_shap_importance()
    fig_noise_robustness()
    fig_detection_latency()
    fig_tool_robustness()
    fig_wifi_vs_ethernet()
    fig_multiseed_stability()
    fig_cost_comparison()
    fig_chronological_split()
    fig_benign_false_positive()
    fig_cae_per_class()
    print("\nALL FIGURES GENERATED ->", OUT)
