"""
Wattson — ACS712 Data Collection Script
MCP3008 (SPI) → saves annotated CSV in Wattson format

Usage:
    python data_collector.py --label Normal     --duration 60  # 60s normal traffic
    python data_collector.py --label DoS        --duration 30
    python data_collector.py --label SSH_Bruteforce --duration 30
    python data_collector.py --label PortScan   --duration 30
    python data_collector.py --sim              # Simulation if no hardware

Output:
    pi_<label>_<timestamp>.csv  — Current, anno_type columns
    (directly compatible with Wattson / prepare_cnn_data.py)

Setup:
    pip install spidev numpy
"""

import argparse
import time
import csv
import json
import sys
import os
from datetime import datetime

import numpy as np

# ─── ARGUMENTS ──────────────────────────────────────────────
VALID_LABELS = ['Normal', 'DoS', 'SSH_Bruteforce', 'PortScan']

parser = argparse.ArgumentParser(description='Wattson Data Collector')
parser.add_argument('--label',    required=True if '--sim' not in sys.argv else False,
                    default='Normal',
                    choices=VALID_LABELS,
                    help='Traffic type during recording')
parser.add_argument('--duration', type=float, default=60.0,
                    help='Recording duration (seconds, default: 60)')
parser.add_argument('--channel',  type=int, default=0,
                    help='MCP3008 channel (default: 0)')
parser.add_argument('--outdir',   default='.',
                    help='Output directory (default: .)')
parser.add_argument('--sim',      action='store_true',
                    help='Simulate if SPI hardware is not available')
parser.add_argument('--calibrate', action='store_true',
                    help='Re-measure and save the 0A reference point (ACS_ZERO)')
parser.add_argument('--match-load', action='store_true',
                    help='Simulate the SAME computational load (FFT+ONNX) as '
                         'rpi_inference.py on every window — to avoid a '
                         'training/deployment CPU-load mismatch '
                         '(see the live_validation finding)')
parser.add_argument('--infer-model', default='xgb_pipeline.onnx',
                    help='ONNX file to use for --match-load (default: xgb_pipeline.onnx)')
args = parser.parse_args()

# ─── CONSTANTS ────────────────────────────────────────────────
FS        = 48828.0       # Target sampling rate (Hz) — matches Wattson
DT        = 1.0 / FS      # ~20.48 µs
SPI_SPEED = 1_200_000     # 1.2 MHz
VREF      = 3.3           # MCP3008 reference voltage
ACS_SENS  = 0.185         # ACS712-5A sensitivity (V/A)
CHUNK     = 10_000        # Write to disk every chunk (memory control)
WIN_SIZE  = 4882          # Same window size as rpi_inference.py (~100ms @ 48828Hz assumed)
os.makedirs(args.outdir, exist_ok=True)
CALIB_FILE = os.path.join(args.outdir, 'acs712_calibration.json')

# ─── (OPTIONAL) MATCHING DEPLOYMENT LOAD ───────────────────
# Finding: while running live, rpi_inference.py performs FFT+ONNX
# computation on every window, whereas data_collector.py did not — this
# CPU/power load difference created a baseline shift never seen during
# training, driving live detection performance to zero (see the
# live_validation/ tests). With --match-load, the SAME computation is
# performed during collection as well (the result is discarded), so the
# collected data matches the real deployment conditions.
infer_sess = infer_input_name = None
if args.match_load:
    from scipy import stats as _stats
    from scipy.signal import find_peaks as _find_peaks
    import onnxruntime as _ort
    infer_sess = _ort.InferenceSession(args.infer_model, providers=['CPUExecutionProvider'])
    infer_input_name = infer_sess.get_inputs()[0].name
    print(f"✓ --match-load active: the same computation as {args.infer_model} will run on every window")

def _extract_features_like_inference(window: np.ndarray) -> np.ndarray:
    m     = np.mean(window)
    s     = np.std(window)
    rms   = np.sqrt(np.mean(window ** 2))
    ptp   = np.ptp(window)
    skw   = float(_stats.skew(window))
    krt   = float(_stats.kurtosis(window))
    crest = float(np.max(np.abs(window)) / (rms + 1e-9))
    eng   = float(np.sum(window ** 2))
    fft   = np.abs(np.fft.rfft(window - m))[:32]
    fft   = fft / (np.sum(fft) + 1e-9)

    centered = window - m
    zcr = float(np.mean(np.diff(np.sign(centered)) != 0))
    peaks, _ = _find_peaks(np.abs(centered), prominence=0.5 * (s + 1e-9))
    peak_rate = len(peaks) / len(window)
    p = fft + 1e-12
    spectral_entropy = float(-np.sum(p * np.log(p)) / np.log(len(p)))

    return np.concatenate([[m, s, rms, ptp, skw, krt, crest, eng], fft,
                            [zcr, peak_rate, spectral_entropy]]).astype(np.float32)

# ─── START SPI ──────────────────────────────────────────────
spi = None
if not args.sim:
    try:
        import spidev
        spi = spidev.SpiDev()
        spi.open(0, 0)
        spi.max_speed_hz = SPI_SPEED
        spi.mode = 0
        print(f"✓ SPI started (CE0, {SPI_SPEED/1e6:.1f} MHz)")
    except Exception as e:
        print(f"[WARNING] SPI could not be started: {e} → Switching to simulation mode.")
        args.sim = True

# ─── READ ADC ─────────────────────────────────────────────────
def read_adc(ch: int) -> float:
    """Reads voltage from the MCP3008 (0.0 – VREF)"""
    r = spi.xfer2([0x01, (0x08 | ch) << 4, 0x00])
    return ((r[1] & 0x03) << 8 | r[2]) * VREF / 1023.0

def volt_to_ampere(v: float) -> float:
    """ACS712-5A: converts voltage to current (A)"""
    return (v - ACS_ZERO) / ACS_SENS

# ─── 0A REFERENCE CALIBRATION ────────────────────────────────
# ACS_ZERO depends on the ACS712's VCC (VCC/2) — if VCC is changed
# from 3.3V to 5V, a fixed assumption (VREF/2) produces an incorrect
# offset. For this reason we measure the actual 0A point from the
# hardware and store it in a file, so that all classes
# (Normal/DoS/SSH_Bruteforce/PortScan) use the same reference and can
# be recalibrated with a single command even if VCC changes.
def calibrate_zero(n_samples: int = 5000) -> float:
    vals = [read_adc(args.channel) for _ in range(n_samples)]
    zero = float(np.mean(vals))
    with open(CALIB_FILE, 'w') as f:
        json.dump({'acs_zero': zero, 'timestamp': datetime.now().isoformat()}, f)
    print(f"✓ Calibration complete: ACS_ZERO = {zero:.4f} V → {CALIB_FILE}")
    return zero

if args.sim:
    ACS_ZERO = VREF / 2
elif args.calibrate or not os.path.exists(CALIB_FILE):
    print("Measuring calibration (keep the RPi as idle as possible)...")
    ACS_ZERO = calibrate_zero()
else:
    with open(CALIB_FILE) as f:
        ACS_ZERO = json.load(f)['acs_zero']
    print(f"✓ Using saved calibration: ACS_ZERO = {ACS_ZERO:.4f} V")

# ─── SIMULATION ──────────────────────────────────────────────
_t = 0.0
def read_sim_voltage() -> float:
    """Simulates a different current profile depending on the class"""
    global _t
    _t += DT
    label = args.label
    if label == 'Normal':
        base, noise_amp = 1.65, 0.03
        sig = base + noise_amp * np.sin(2 * np.pi * 50 * _t)
    elif label == 'DoS':
        # SYN flood: sudden high, periodic spike
        base, noise_amp = 1.75, 0.12
        sig = base + noise_amp * abs(np.sin(2 * np.pi * 8 * _t))
    elif label == 'SSH_Bruteforce':
        # Regular connection attempts: mid-frequency packets
        base, noise_amp = 1.70, 0.07
        sig = base + noise_amp * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * _t))
    else:  # PortScan
        # Fast port scanning: high-frequency small packets
        base, noise_amp = 1.68, 0.06
        sig = base + noise_amp * np.sin(2 * np.pi * 20 * _t) ** 2
    return float(np.clip(sig + np.random.normal(0, 0.005), 0, VREF))

read_voltage = read_sim_voltage if args.sim else lambda: read_adc(args.channel)

# ─── OUTPUT FILE ───────────────────────────────────────────
os.makedirs(args.outdir, exist_ok=True)
ts_str   = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = os.path.join(args.outdir,
                        f"pi_{args.label.lower()}_{ts_str}.csv")

total_samples = int(args.duration * FS)
print(f"\n{'='*55}")
print(f"  Wattson Data Collector")
print(f"{'='*55}")
print(f"  Label     : {args.label}")
print(f"  Duration  : {args.duration:.0f}s  ({total_samples:,} sample)")
print(f"  Fs        : {FS:.0f} Hz")
print(f"  Channel   : MCP3008 CH{args.channel}")
print(f"  Output    : {filename}")
if args.sim:
    print(f"  ⚠ SIMULATION MODE")
print(f"{'='*55}")
print("\nRecording about to start — stop with Ctrl+C\n3...")
time.sleep(1); print("2..."); time.sleep(1); print("1..."); time.sleep(1)
print("RECORDING STARTED!\n")

# ─── RECORDING LOOP ───────────────────────────────────────────
t_next    = time.perf_counter()
t_start   = t_next
collected = 0
chunk_buf = []          # (current_A, label) rows
win_buf   = []          # window accumulator for --match-load

try:
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Current', 'anno_type'])   # Wattson format

        while collected < total_samples:
            # Timing
            now = time.perf_counter()
            if now < t_next:
                pass  # busy-wait
            t_next += DT

            v = read_voltage()
            i = volt_to_ampere(v)
            chunk_buf.append((f"{i:.6f}", args.label))
            collected += 1

            if infer_sess is not None:
                win_buf.append(i)
                if len(win_buf) >= WIN_SIZE:
                    feat = _extract_features_like_inference(np.array(win_buf, dtype=np.float32))
                    infer_sess.run(None, {infer_input_name: feat.reshape(1, -1)})
                    win_buf = []

            # Write once chunk is full
            if len(chunk_buf) >= CHUNK:
                writer.writerows(chunk_buf)
                chunk_buf.clear()

            # Progress report (every 48828 samples = 1s)
            if collected % int(FS) == 0:
                elapsed  = time.perf_counter() - t_start
                pct      = collected / total_samples * 100
                real_fs  = collected / elapsed
                remaining = (total_samples - collected) / FS
                print(f"  {pct:5.1f}%  {collected:>8,}/{total_samples:,} sample"
                      f"  Fs≈{real_fs:.0f}Hz  remaining:{remaining:.0f}s")

        # Write remaining chunk
        if chunk_buf:
            writer.writerows(chunk_buf)

except KeyboardInterrupt:
    # CSV is still valid if interrupted midway
    if chunk_buf:
        with open(filename, 'a', newline='') as f:
            csv.writer(f).writerows(chunk_buf)
    print(f"\n\nStopped early — {collected:,} samples saved.")

finally:
    if spi:
        spi.close()

elapsed   = time.perf_counter() - t_start
real_fs   = collected / elapsed if elapsed > 0 else 0
file_size = os.path.getsize(filename) / 1e6 if os.path.exists(filename) else 0

print(f"\n{'='*55}")
print(f"RECORDING COMPLETE")
print(f"{'='*55}")
print(f"  File      : {filename}")
print(f"  Size      : {file_size:.1f} MB")
print(f"  Samples   : {collected:,}")
print(f"  Duration  : {elapsed:.1f}s")
print(f"  Actual Fs : {real_fs:.0f} Hz  (target: {FS:.0f})")
print(f"  Label     : {args.label}")
print(f"\nNext step:")
print(f"  python prepare_cnn_data.py   ← produces windowed_data.npz")
print(f"  python cnn_model.py          ← retrains the CNN")
print(f"{'='*55}")
