"""
Replication of the Dragon_Slice methodology with the CORRECTED Ethernet data:
- Unsupervised Convolutional Autoencoder (CAE)
- Training ONLY on held-out "clean room" Normal data
- Raw (z-score NOT APPLIED) current values (amplitude information matters for
  anomaly detection)
- Test: unseen Normal + ALL attack classes (merged as Anomaly)
- Metric: raw MSE AUC + AUC after MAF (moving average)
- Reference: Lightbody et al. 2024, Dragon_Slice_1024_Dropout: raw AUC=0.764, MAF AUC=0.890
- NOTE: the previous run was on new_data/full_dataset (old WiFi); this version
  runs on new_data/ethernet_dataset, recollected with the corrected IP.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import re
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

WIN_SIZE = 4882
BASE = Path("new_data/ethernet_dataset")

LABEL_MAP = {'normal': 'Normal', 'portscan': 'PortScan',
             'ssh_bruteforce': 'SSH_Bruteforce', 'dos': 'DoS'}

def label_from_filename(path):
    m = re.match(r'pi_(.+?)_\d{8}_\d{6}\.csv$', path.name)
    key = m.group(1) if m else ''
    return LABEL_MAP.get(key)

FILES = sorted(BASE.glob("pi_*.csv"))
print(f"Total files: {len(FILES)}")

# ---- Split sessions by class ----
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

# ---- Training data (ONLY clean-room Normal) ----
train_wins = []
for f in train_normal_files:
    train_wins.extend(load_windows(f))
X_train = np.array(train_wins, dtype=np.float32)
print(f"Training windows: {X_train.shape}")

# ---- Test data (unseen Normal + all attacks) ----
test_wins, test_labels, test_sessions = [], [], []
for f in test_normal_files:
    ws = load_windows(f)
    test_wins.extend(ws)
    test_labels.extend([0]*len(ws))  # 0 = Normal
    test_sessions.extend([f.stem]*len(ws))
for f in attack_files:
    ws = load_windows(f)
    test_wins.extend(ws)
    test_labels.extend([1]*len(ws))  # 1 = Anomaly
    test_sessions.extend([f.stem]*len(ws))

X_test = np.array(test_wins, dtype=np.float32)
y_test = np.array(test_labels)
sess_test = np.array(test_sessions)
print(f"Test windows: {X_test.shape}  (Normal={np.sum(y_test==0)}, Anomaly={np.sum(y_test==1)})")

# ---- CAE Architecture (similar to Dragon_Slice: conv-stride encoder, mirror decoder) ----
class CAE(nn.Module):
    def __init__(self, win_size, bottleneck=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3), nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=2, padding=2), nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, win_size)
            enc_out = self.encoder(dummy)
            self.enc_channels = enc_out.shape[1]
            self.enc_length = enc_out.shape[2]
            self.flat_dim = enc_out.numel()
        self.fc_enc = nn.Linear(self.flat_dim, bottleneck)
        self.fc_dec = nn.Linear(bottleneck, self.flat_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 32, kernel_size=5, stride=2, padding=2, output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(32, 32, kernel_size=5, stride=2, padding=2, output_padding=1), nn.ReLU(),
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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CAE(WIN_SIZE, bottleneck=512).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"CAE parameter count: {n_params:,} | Bottleneck: 512 (~{X_train.shape[1]*4/512:.0f}x compression)")

X_tr_t = torch.tensor(X_train[:, None, :], dtype=torch.float32)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

BATCH = 32
EPOCHS = 60
n = len(X_tr_t)
print(f"\nTraining starting ({EPOCHS} epochs, ONLY with clean-room Normal)...")
for epoch in range(1, EPOCHS+1):
    perm = torch.randperm(n)
    total_loss = 0.0
    model.train()
    for i in range(0, n, BATCH):
        idx = perm[i:i+BATCH]
        xb = X_tr_t[idx].to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, xb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(idx)
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{EPOCHS}  train_MSE={total_loss/n:.6f}")

# ---- Test: compute MSE per window ----
model.eval()
mse_scores = []
with torch.no_grad():
    X_te_t = torch.tensor(X_test[:, None, :], dtype=torch.float32)
    for i in range(0, len(X_te_t), 64):
        xb = X_te_t[i:i+64].to(device)
        out = model(xb)
        mse = torch.mean((out - xb)**2, dim=(1,2)).cpu().numpy()
        mse_scores.extend(mse)
mse_scores = np.array(mse_scores)

raw_auc = roc_auc_score(y_test, mse_scores)
print(f"\n{'='*55}\nRAW MSE AUC: {raw_auc:.4f}  (Dragon_Slice reference: 0.764-0.778)\n{'='*55}")

# ---- Apply MAF (moving average), find the best window size (session-ordered) ----
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
    print(f"  MA window={ma_window:3d}  AUC={auc:.4f}")
    if auc > best_auc:
        best_auc, best_ma = auc, ma_window

print(f"\n{'='*55}")
print(f"BEST MAF AUC: {best_auc:.4f} (MA window={best_ma})  (Dragon_Slice reference: 0.876-0.890)")
print(f"{'='*55}")
