"""
Wattson - Visualize the Normal → Attack Transition
Known from the legend: the first attack starts around row ~1,455,074
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path("pi")
bf_file = BASE / "pi_bruteforce_large" / "pi_bruteforce_large" / "pi_hydra_t1_cke.csv"

# Read the transition region: between rows 1,400,000 and 1,600,000
SKIP    = 1_400_000
N_READ  = 200_000

print(f"Reading transition region (row {SKIP:,} → {SKIP+N_READ:,})...")
df = pd.read_csv(bf_file, skiprows=range(1, SKIP+1), nrows=N_READ)
# Column names were lost, re-add them
df.columns = ['idx','Time','Current','anno_string','anno_type','anno_specific']

print(f"Label distribution:\n{df['anno_type'].value_counts()}")
print(f"\nCurrent Normal:  mean={df[df['anno_type']=='Normal']['Current'].mean():.4f} A")
attack_mask = df['anno_type'] != 'Normal'
if attack_mask.sum() > 0:
    print(f"Current Attack:  mean={df[attack_mask]['Current'].mean():.4f} A")

# ── Plot ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 9))

# 1. Raw signal — transition region
colors_map = {'Normal': 'steelblue', 'SSH_Bruteforce': 'red',
              'Reconnaissance': 'orange', 'DoS': 'purple', 'CTF': 'green'}
c = [colors_map.get(t, 'gray') for t in df['anno_type']]
axes[0].scatter(df['Time'], df['Current'], c=c, s=0.2, linewidths=0)
axes[0].set_title("Raw Current Signal — Transition Region (blue=Normal, red=Attack)")
axes[0].set_ylabel("Current (A)")
axes[0].set_xlabel("Time (s)")

# 2. FFT comparison
FS = 48828
WINDOW = 4096  # ~84ms

# Normal window
normal_chunk = df[df['anno_type'] == 'Normal']['Current'].values
if len(normal_chunk) >= WINDOW:
    fft_norm = np.abs(np.fft.rfft(normal_chunk[:WINDOW] - normal_chunk[:WINDOW].mean()))
    freqs = np.fft.rfftfreq(WINDOW, 1/FS)
    axes[1].semilogy(freqs[:500], fft_norm[:500], color='steelblue', label='Normal', alpha=0.8)

# Attack window
attack_chunk = df[attack_mask]['Current'].values
if len(attack_chunk) >= WINDOW:
    fft_atk = np.abs(np.fft.rfft(attack_chunk[:WINDOW] - attack_chunk[:WINDOW].mean()))
    axes[1].semilogy(freqs[:500], fft_atk[:500], color='red', label='SSH Brute-force', alpha=0.8)

axes[1].set_title("FFT Spectrum Comparison (first 500 Hz)")
axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_ylabel("Amplitude (log)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 3. Mean current per window (100ms windows)
STEP = 4882  # ~100ms
times, means, labels = [], [], []
for i in range(0, len(df) - STEP, STEP):
    chunk = df.iloc[i:i+STEP]
    majority = chunk['anno_type'].mode()[0]
    times.append(chunk['Time'].mean())
    means.append(chunk['Current'].mean())
    labels.append(majority)

colors_w = ['red' if l != 'Normal' else 'steelblue' for l in labels]
axes[2].bar(times, means, width=0.09, color=colors_w, alpha=0.8)
axes[2].set_title("Mean Current per 100ms Window (blue=Normal, red=Attack)")
axes[2].set_ylabel("Mean Current (A)")
axes[2].set_xlabel("Time (s)")

plt.tight_layout()
plt.savefig("transition_analysis.png", dpi=150)
plt.show()
print("\nPlot saved: transition_analysis.png")
