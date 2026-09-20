"""
FULL ablation replication of the Dragon_Slice methodology (corrected Ethernet data):
- Same exact structure as the reference (Lightbody et al. 2024) Table 4 (no dropout)
  and Table 5 (dropout): 3 compression ratios (1024/512/256) x {no dropout, with dropout}
  = 6 CAE configurations.
- Dropout variant: p=0.1 on conv layers, p=0.5 on the dense (bottleneck) layer,
  GELU activation (same specification as the reference).
- For each configuration: raw MSE AUC + best AUC after MAF (moving average).
- Extra, for the best configuration: GDR-vs-FAR curve (% detection rate against
  X false alarms per hour) -- the headline metric DIRECTLY comparable with the
  reference's Figure 7D/12D.
- Example window duration ~100ms (WIN_SIZE=4882 @ Fs~48828Hz) -> 36,000 windows
  per hour, FAR/hour = (false_positive_count / normal_window_count) * 36000.
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
WINDOWS_PER_HOUR = 3600 / (WIN_SIZE / 48828.0)  # ~36,000
BASE = Path("new_data/ethernet_dataset")

LABEL_MAP = {'normal': 'Normal', 'portscan': 'PortScan',
             'ssh_bruteforce': 'SSH_Bruteforce', 'dos': 'DoS'}

def label_from_filename(path):
    m = re.match(r'pi_(.+?)_\d{8}_\d{6}\.csv$', path.name)
    key = m.group(1) if m else ''
    return LABEL_MAP.get(key)

FILES = sorted(BASE.glob("pi_*.csv"))
print(f"Total files: {len(FILES)}")

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

print(f"Normal sessions: {len(normal_files)} -> train(clean-room)={len(train_normal_files)}, test={len(test_normal_files)}")
print(f"Attack sessions (test, Anomaly): {len(attack_files)}")

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
print(f"Training windows: {X_train.shape}")

test_wins, test_labels, test_sessions = [], [], []
for f in test_normal_files:
    ws = load_windows(f)
    test_wins.extend(ws)
    test_labels.extend([0]*len(ws))
    test_sessions.extend([f.stem]*len(ws))
for f in attack_files:
    ws = load_windows(f)
    test_wins.extend(ws)
    test_labels.extend([1]*len(ws))
    test_sessions.extend([f.stem]*len(ws))

X_test = np.array(test_wins, dtype=np.float32)
y_test = np.array(test_labels)
sess_test = np.array(test_sessions)
print(f"Test windows: {X_test.shape}  (Normal={np.sum(y_test==0)}, Anomaly={np.sum(y_test==1)})")
print(f"Data loading time: {time.time()-t0:.1f}s\n")

X_tr_t = torch.tensor(X_train[:, None, :], dtype=torch.float32)
X_te_t = torch.tensor(X_test[:, None, :], dtype=torch.float32)


class CAE(nn.Module):
    def __init__(self, win_size, bottleneck=512, dropout=False):
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


def train_and_eval(bottleneck, dropout, epochs=60, batch=32, lr=1e-4):
    tag = f"{bottleneck}{'_dropout' if dropout else ''}"
    print(f"\n{'='*60}\nCONFIGURATION: bottleneck={bottleneck}  dropout={dropout}\n{'='*60}")
    model = CAE(WIN_SIZE, bottleneck=bottleneck, dropout=dropout)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    n = len(X_tr_t)
    t0 = time.time()
    for epoch in range(1, epochs+1):
        perm = torch.randperm(n)
        total_loss = 0.0
        model.train()
        for i in range(0, n, batch):
            idx = perm[i:i+batch]
            xb = X_tr_t[idx]
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, xb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
        if epoch % 20 == 0 or epoch == 1 or epoch == epochs:
            print(f"  Epoch {epoch:3d}/{epochs}  train_MSE={total_loss/n:.6f}")
    train_time = time.time() - t0

    model.eval()
    mse_scores = []
    with torch.no_grad():
        for i in range(0, len(X_te_t), 64):
            xb = X_te_t[i:i+64]
            out = model(xb)
            mse = torch.mean((out - xb)**2, dim=(1, 2)).cpu().numpy()
            mse_scores.extend(mse)
    mse_scores = np.array(mse_scores)

    raw_auc = roc_auc_score(y_test, mse_scores)

    best_auc, best_ma = raw_auc, 1
    for ma_window in [3, 5, 8, 10, 12, 15, 17, 20, 25, 30]:
        smoothed = np.zeros_like(mse_scores)
        for s in np.unique(sess_test):
            mask = np.where(sess_test == s)[0]
            vals = mse_scores[mask]
            kernel = np.ones(ma_window) / ma_window
            sm = np.convolve(vals, kernel, mode='same')
            smoothed[mask] = sm
        auc = roc_auc_score(y_test, smoothed)
        if auc > best_auc:
            best_auc, best_ma = auc, ma_window

    print(f"  Training time: {train_time:.1f}s")
    print(f"  RAW MSE AUC: {raw_auc:.4f}  |  BEST MAF AUC: {best_auc:.4f} (MA={best_ma})")

    return {
        'tag': tag, 'bottleneck': bottleneck, 'dropout': dropout,
        'n_params': n_params, 'train_time_s': train_time,
        'raw_auc': raw_auc, 'maf_auc': best_auc, 'maf_window': best_ma,
        'mse_scores': mse_scores,
    }


# ==================== RUN ALL 6 CONFIGURATIONS ====================
CONFIGS = [(1024, False), (512, False), (256, False),
           (1024, True), (512, True), (256, True)]

results = []
for bottleneck, dropout in CONFIGS:
    r = train_and_eval(bottleneck, dropout)
    results.append(r)

# ==================== SUMMARY TABLE (in reference Table 4/5 format) ====================
print(f"\n\n{'='*70}")
print("SUMMARY -- CORRECTED ETHERNET DATA, FULL ABLATION")
print(f"{'='*70}")
print(f"{'Config':<16}{'Params':>12}{'Train(s)':>10}{'RawAUC':>9}{'MAF_AUC':>9}{'MA_win':>7}")
for r in results:
    print(f"{r['tag']:<16}{r['n_params']:>12,}{r['train_time_s']:>10.1f}{r['raw_auc']:>9.4f}{r['maf_auc']:>9.4f}{r['maf_window']:>7d}")

print(f"\nReference (Dragon_Slice, non-dropout): 1024->raw0.778/MAF0.876 | 512->raw0.768/MAF0.874 | 256->raw0.772/MAF0.868")
print(f"Reference (Dragon_Slice, dropout)    : 1024->raw0.766/MAF0.890 | 512->raw0.759/MAF0.886 | 256->raw0.752/MAF0.884")

# ==================== GDR-vs-FAR CURVE FOR THE BEST CONFIGURATION ====================
best = max(results, key=lambda r: r['maf_auc'])
print(f"\n{'='*70}")
print(f"BEST CONFIGURATION: {best['tag']}  (MAF AUC={best['maf_auc']:.4f})")
print("Computing GDR-vs-FAR curve (for comparison with reference Figure 7D/12D)...")
print(f"{'='*70}")

mse_scores = best['mse_scores']
ma_window = best['maf_window']
smoothed = np.zeros_like(mse_scores)
for s in np.unique(sess_test):
    mask = np.where(sess_test == s)[0]
    vals = mse_scores[mask]
    kernel = np.ones(ma_window) / ma_window
    smoothed[mask] = np.convolve(vals, kernel, mode='same')

normal_mask = (y_test == 0)
anomaly_mask = (y_test == 1)
n_normal = normal_mask.sum()

thresholds = np.quantile(smoothed, np.linspace(0.50, 0.999, 200))
gdr_far = []
for th in thresholds:
    tp = np.sum(smoothed[anomaly_mask] >= th)
    fp = np.sum(smoothed[normal_mask] >= th)
    gdr = tp / anomaly_mask.sum() * 100
    far_per_hour = (fp / n_normal) * WINDOWS_PER_HOUR
    gdr_far.append((th, gdr, far_per_hour))

print(f"{'Threshold(pctl)':>12}{'GDR(%)':>10}{'FAR/hour':>12}")
for i in range(0, len(gdr_far), 10):
    th, gdr, far = gdr_far[i]
    print(f"{i/2:>11.1f}%{gdr:>10.1f}{far:>12.2f}")

for target_far in [0, 10, 50]:
    candidates = [g for g in gdr_far if g[2] <= target_far]
    if candidates:
        best_gdr = max(candidates, key=lambda g: g[1])
        print(f"FAR<={target_far:>3}/hour  -> best GDR={best_gdr[1]:.1f}%  (reference non-dropout: ~50/70/80%, dropout: ~65/77/85%)")
    else:
        print(f"FAR<={target_far:>3}/hour  -> no data at this threshold")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fars = [g[2] for g in gdr_far]
    gdrs = [g[1] for g in gdr_far]
    plt.figure(figsize=(7, 5))
    plt.plot(fars, gdrs, marker='.', label=f"Wattson ({best['tag']})")
    plt.xlim(0, 100)
    plt.xlabel("False Alarm Rate (per hour)")
    plt.ylabel("Good Detection Rate (%)")
    plt.title("GDR vs FAR -- replication of Dragon_Slice methodology")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("gdr_far_curve_ethernet_fixed.png", dpi=150)
    print("\nSaved: gdr_far_curve_ethernet_fixed.png")
except Exception as e:
    print(f"[WARNING] Could not save figure: {e}")

# ==================== SAVE RESULTS TO JSON ====================
out_json = []
for r in results:
    r2 = {k: v for k, v in r.items() if k != 'mse_scores'}
    out_json.append(r2)
with open("dragon_slice_ablation_results.json", "w", encoding="utf-8") as f:
    json.dump({'configs': out_json, 'gdr_far_best': [(float(t), float(g), float(fa)) for t, g, fa in gdr_far],
                'best_config': best['tag']}, f, indent=2)
print("\nSaved: dragon_slice_ablation_results.json")
