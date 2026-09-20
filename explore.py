"""
Wattson Dataset - Exploration Script
Reads only the first N rows (RAM-friendly)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ─── SETTINGS ────────────────────────────────────────────────
BASE = Path("pi")          # root folder
N_ROWS = 500_000           # number of rows to read first (~10 seconds of data)
# ─────────────────────────────────────────────────────────────

# 1) Read one brute-force file
bf_file = BASE / "pi_bruteforce_large" / "pi_bruteforce_large" / "pi_hydra_t1_cke.csv"
print(f"Reading: {bf_file}")
df_bf = pd.read_csv(bf_file, nrows=N_ROWS)
print(f"Shape: {df_bf.shape}")
print(f"Columns: {df_bf.columns.tolist()}")
print(f"\nFirst 3 rows:\n{df_bf.head(3)}")
print(f"\nLabel distribution:\n{df_bf['anno_type'].value_counts()}")
print(f"\nCurrent statistics:\n{df_bf['Current'].describe()}")

# 2) Read a normal file
norm_file = BASE / "pi_norm_large" / "pi_norm_large" / "pi_norm_1.csv"
print(f"\nReading normal file: {norm_file}")
df_norm = pd.read_csv(norm_file, nrows=N_ROWS)
print(f"Label distribution:\n{df_norm['anno_type'].value_counts()}")

# 3) Compute sampling rate
dt = df_bf["Time"].iloc[1] - df_bf["Time"].iloc[0]
fs = 1 / dt
print(f"\nSampling rate: {fs:.0f} Hz (~{fs/1000:.1f} kHz)")

# 4) Plot: Normal vs Attack comparison
fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=False)

# Normal: first 10,000 samples
axes[0].plot(df_norm["Time"].iloc[:10000], df_norm["Current"].iloc[:10000],
             color="steelblue", linewidth=0.5)
axes[0].set_title("Normal Traffic — Power Consumption")
axes[0].set_ylabel("Current (A)")
axes[0].set_xlabel("Time (s)")

# Attack section: find the Normal → Attack transition
attack_start_idx = df_bf[df_bf["anno_type"] != "Normal"].index
if len(attack_start_idx) > 0:
    start = max(0, attack_start_idx[0] - 5000)
    end = min(len(df_bf), attack_start_idx[0] + 5000)
    chunk = df_bf.iloc[start:end]
    colors = ["red" if t != "Normal" else "steelblue" for t in chunk["anno_type"]]
    axes[1].scatter(chunk["Time"], chunk["Current"], c=colors, s=0.3, linewidths=0)
    axes[1].set_title("Brute-Force Attack Transition Region (blue=Normal, red=Attack)")
    axes[1].set_ylabel("Current (A)")
    axes[1].set_xlabel("Time (s)")
else:
    axes[1].text(0.5, 0.5, "No attack in these 500K samples\n(increase N_ROWS)",
                 ha='center', va='center', transform=axes[1].transAxes)

plt.tight_layout()
plt.savefig("wattson_exploration.png", dpi=150)
plt.show()
print("\nPlot saved: wattson_exploration.png")

# 5) Suggested window size
window_sec = 0.1   # 100ms window
window_samples = int(fs * window_sec)
print(f"\nSuggested window: {window_sec*1000:.0f} ms = {window_samples} samples")
print("FFT + statistical features will be extracted using these windows")
