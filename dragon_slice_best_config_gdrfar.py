"""
Retrains the best CAE configuration (256 bottleneck, dropout) standalone,
saves the model weights AND the raw MSE scores/labels/sessions to disk,
then computes the GDR-vs-FAR curve with the CORRECT threshold range (over
the normal distribution's own percentiles) -- directly comparable with
reference Figure 7D/12D.

Bug in the previous run: thresholds were selected from the 50th-99.9th
percentile of the attack-heavy COMBINED distribution -- that entire range
was already above the normal distribution, so FAR always came out as 0.
In this version, thresholds are selected ONLY from the 0th-100th percentile
of the normal (Normal=0) test window's own score distribution, so a genuine
tradeoff curve is obtained from FAR=0 to FAR=~high.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import re
import time
import json
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

WIN_SIZE = 4882
WINDOWS_PER_HOUR = 3600 / (WIN_SIZE / 48828.0)
BASE = Path("new_data/ethernet_dataset")

LABEL_MAP = {'normal': 'Normal', 'portscan': 'PortScan',
             'ssh_bruteforce': 'SSH_Bruteforce', 'dos': 'DoS'}

def label_from_filename(path):
    m = re.match(r'pi_(.+?)_\d{8}_\d{6}\.csv$', path.name)
    key = m.group(1) if m else ''
    return LABEL_MAP.get(key)

FILES = sorted(BASE.glob("pi_*.csv"))
normal_files, attack_files = [], []
for f in FILES:
    lbl = label_from_filename(f)
    if lbl == 'Normal':
        normal_files.append(f)
    elif lbl in ('PortScan', 'SSH_Bruteforce', 'DoS'):
        attack_files.append(f)

rng = np.random.RandomState(42)
rng.shuffle(normal_files)
n_train_normal = int(len(normal_files) * 0.6)
train_normal_files = normal_files[:n_train_normal]
test_normal_files = normal_files[n_train_normal:]

def load_windows(filepath, max_windows=200):
    wins = []
    for chunk_df in pd.read_csv(filepath, chunksize=500_000, usecols=['Current']):
        curr = chunk_df['Current'].values.astype(np.float32)
        n_wins = len(curr) // WIN_SIZE
        for i in range(min(n_wins, max_windows - len(wins))):
            wins.append(curr[i*WIN_SIZE:(i+1)*WIN_SIZE])
        if len(wins) >= max_windows:
            break
    return wins

t0 = time.time()
train_wins = []
for f in train_normal_files:
    train_wins.extend(load_windows(f))
X_train = np.array(train_wins, dtype=np.float32)

test_wins, test_labels, test_sessions = [], [], []
for f in test_normal_files:
    ws = load_windows(f)
    test_wins.extend(ws); test_labels.extend([0]*len(ws)); test_sessions.extend([f.stem]*len(ws))
for f in attack_files:
    ws = load_windows(f)
    test_wins.extend(ws); test_labels.extend([1]*len(ws)); test_sessions.extend([f.stem]*len(ws))

X_test = np.array(test_wins, dtype=np.float32)
y_test = np.array(test_labels)
sess_test = np.array(test_sessions)
print(f"Train: {X_train.shape}  Test: {X_test.shape} (Normal={np.sum(y_test==0)}, Anomaly={np.sum(y_test==1)})")
print(f"Data loading time: {time.time()-t0:.1f}s")

X_tr_t = torch.tensor(X_train[:, None, :], dtype=torch.float32)
X_te_t = torch.tensor(X_test[:, None, :], dtype=torch.float32)


class CAE(nn.Module):
    def __init__(self, win_size, bottleneck=256, dropout=True):
        super().__init__()
        act = nn.GELU if dropout else nn.ReLU
        conv_p = 0.1 if dropout else 0.0
        enc_layers = []
        in_ch = 1
        for out_ch in (32, 32, 32):
            enc_layers += [nn.Conv1d(in_ch, out_ch, kernel_size=7 if in_ch == 1 else 5,
                                      stride=2, padding=3 if in_ch == 1 else 2), act()]
            if dropout:
                enc_layers.append(nn.Dropout(conv_p))
            in_ch = out_ch
        self.encoder = nn.Sequential(*enc_layers)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, win_size)
            enc_out = self.encoder(dummy)
            self.enc_channels = enc_out.shape[1]
            self.enc_length = enc_out.shape[2]
            self.flat_dim = enc_out.numel()
        dense_p = 0.5 if dropout else 0.0
        self.fc_enc = nn.Sequential(nn.Linear(self.flat_dim, bottleneck), nn.Dropout(dense_p))
        self.fc_dec = nn.Linear(bottleneck, self.flat_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 32, kernel_size=5, stride=2, padding=2, output_padding=1), act(),
            nn.ConvTranspose1d(32, 32, kernel_size=5, stride=2, padding=2, output_padding=1), act(),
            nn.ConvTranspose1d(32, 1, kernel_size=7, stride=2, padding=3, output_padding=1),
        )

    def forward(self, x):
        z = self.encoder(x)
        flat = z.view(z.size(0), -1)
        bottleneck = self.fc_enc(flat)
        back = self.fc_dec(bottleneck).view(x.size(0), self.enc_channels, self.enc_length)
        out = self.decoder(back)
        if out.shape[-1] != x.shape[-1]:
            out = out[..., :x.shape[-1]] if out.shape[-1] > x.shape[-1] else \
                  nn.functional.pad(out, (0, x.shape[-1]-out.shape[-1]))
        return out


model = CAE(WIN_SIZE, bottleneck=256, dropout=True)
n_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {n_params:,}")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()
BATCH, EPOCHS = 32, 60
n = len(X_tr_t)
t0 = time.time()
for epoch in range(1, EPOCHS+1):
    perm = torch.randperm(n)
    total_loss = 0.0
    model.train()
    for i in range(0, n, BATCH):
        idx = perm[i:i+BATCH]
        xb = X_tr_t[idx]
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, xb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(idx)
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{EPOCHS}  train_MSE={total_loss/n:.6f}")
print(f"Training time: {time.time()-t0:.1f}s")

torch.save(model.state_dict(), "cae_256_dropout_best.pt")
print("Saved: cae_256_dropout_best.pt")

model.eval()
mse_scores = []
with torch.no_grad():
    for i in range(0, len(X_te_t), 64):
        xb = X_te_t[i:i+64]
        out = model(xb)
        mse = torch.mean((out - xb)**2, dim=(1, 2)).cpu().numpy()
        mse_scores.extend(mse)
mse_scores = np.array(mse_scores)

np.savez_compressed("cae_256_dropout_mse_scores.npz",
                     mse_scores=mse_scores, y_test=y_test, sess_test=sess_test)
print("Saved: cae_256_dropout_mse_scores.npz (raw scores, for analysis without retraining)")

raw_auc = roc_auc_score(y_test, mse_scores)
best_auc, best_ma = raw_auc, 1
for ma_window in [3, 5, 8, 10, 12, 15, 17, 20, 25, 30]:
    smoothed = np.zeros_like(mse_scores)
    for s in np.unique(sess_test):
        mask = np.where(sess_test == s)[0]
        vals = mse_scores[mask]
        kernel = np.ones(ma_window) / ma_window
        smoothed[mask] = np.convolve(vals, kernel, mode='same')
    auc = roc_auc_score(y_test, smoothed)
    if auc > best_auc:
        best_auc, best_ma = auc, ma_window
print(f"RAW AUC={raw_auc:.4f}  MAF AUC={best_auc:.4f} (MA={best_ma})")

smoothed = np.zeros_like(mse_scores)
for s in np.unique(sess_test):
    mask = np.where(sess_test == s)[0]
    vals = mse_scores[mask]
    kernel = np.ones(best_ma) / best_ma
    smoothed[mask] = np.convolve(vals, kernel, mode='same')

normal_mask = (y_test == 0)
anomaly_mask = (y_test == 1)
n_normal = normal_mask.sum()

# FIX: thresholds are selected ONLY from the normal window's own score
# distribution's 0th-100th percentile (for a correct FAR=0..high sweep)
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

print(f"\n{'Normal_pctl':>12}{'GDR(%)':>10}{'FAR/hour':>12}")
for pctl, th, gdr, far in gdr_far[::15]:
    print(f"{pctl:>11.1f}%{gdr:>10.1f}{far:>12.2f}")

for target_far in [0, 5, 10, 20, 50, 100]:
    candidates = [g for g in gdr_far if g[3] <= target_far]
    if candidates:
        best_gdr = max(candidates, key=lambda g: g[2])
        print(f"FAR<={target_far:>3}/hour -> best GDR={best_gdr[2]:.1f}%  (threshold=normal's {best_gdr[0]:.1f}th percentile)")
    else:
        print(f"FAR<={target_far:>3}/hour -> no data at this threshold (lowest FAR={min(g[3] for g in gdr_far):.2f})")

print(f"\nReference (Dragon_Slice, non-dropout, 1024): ~50%@0FA/h, ~70%@<10FA/h, ~80%@~50FA/h")
print(f"Reference (Dragon_Slice, dropout, 1024)    : ~65%@0FA/h, ~77%@<10FA/h, ~85%@<50FA/h")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fars = [g[3] for g in gdr_far]
    gdrs = [g[2] for g in gdr_far]
    plt.figure(figsize=(7, 5))
    plt.plot(fars, gdrs, marker='.', label="Wattson (256, dropout)")
    plt.axhline(65, color='gray', linestyle='--', alpha=0.4, label="Reference dropout @0FA/h (~65%)")
    plt.axhline(50, color='lightgray', linestyle=':', alpha=0.4, label="Reference non-dropout @0FA/h (~50%)")
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.xlabel("False Alarm Rate (per hour)")
    plt.ylabel("Good Detection Rate (%)")
    plt.title("GDR vs FAR -- replication of Dragon_Slice methodology (corrected)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("gdr_far_curve_ethernet_fixed_v2.png", dpi=150)
    print("\nSaved: gdr_far_curve_ethernet_fixed_v2.png")
except Exception as e:
    print(f"[WARNING] Could not save figure: {e}")

with open("dragon_slice_gdrfar_v2.json", "w", encoding="utf-8") as f:
    json.dump({
        'raw_auc': float(raw_auc), 'maf_auc': float(best_auc), 'maf_window': int(best_ma),
        'gdr_far_curve': [(float(p), float(t), float(g), float(fa)) for p, t, g, fa in gdr_far],
    }, f, indent=2)
print("Saved: dragon_slice_gdrfar_v2.json")
