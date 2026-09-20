"""
Wattson — RPi Unified Inference Script
Supports CNN (raw signal) and RF (feature extraction)

Usage:
    python rpi_inference.py                    # CNN ONNX, live mode
    python rpi_inference.py --model rf         # RF pipeline ONNX
    python rpi_inference.py --demo             # Simulation without hardware
    python rpi_inference.py --model rf --demo  # RF simulation
    python rpi_inference.py --log alerts.csv   # Log attacks

Requirements (on the RPi):
    pip install onnxruntime numpy scipy spidev
"""

import argparse
import time
import sys
import csv
import os
from datetime import datetime

import numpy as np
from scipy import stats
from scipy.signal import find_peaks

# ─── ARGUMENTS ──────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Wattson — RPi IDS')
parser.add_argument('--model',   default='cnn', choices=['cnn', 'rf', 'xgb'],
                    help='Model: cnn (cnn_model.onnx), rf (rf_pipeline.onnx) or xgb (xgb_pipeline.onnx)')
parser.add_argument('--channel', type=int, default=0,
                    help='MCP3008 ADC channel (default: 0)')
parser.add_argument('--demo',    action='store_true',
                    help='Simulation mode — no hardware required')
parser.add_argument('--verbose', action='store_true',
                    help='Print all class probabilities for every window')
parser.add_argument('--log',     default='',
                    help='Log attacks to this CSV file')
args = parser.parse_args()

# ─── CONSTANTS ────────────────────────────────────────────────
FS        = 48828.0
DT        = 1.0 / FS
WIN_SIZE  = 4882          # ~100ms @ 48828 Hz
SPI_SPEED = 1_200_000
VREF      = 3.3
ACS_SENS  = 0.185         # V/A (ACS712-5A)

# data_collector.py uses EXACTLY the same calibration (see acs712_calibration.json).
# Important: without this value, the features extracted for RF/XGB (mean/rms/energy
# etc.) remain in the raw Volt scale instead of the Amp scale used during training,
# and the StandardScaler receives completely out-of-distribution input (this was the
# root cause of live detection performance dropping to zero).
_CALIB_FILE = 'acs712_calibration.json'
if os.path.exists(_CALIB_FILE):
    import json as _json
    with open(_CALIB_FILE) as _f:
        ACS_ZERO = _json.load(_f)['acs_zero']
    print(f"✓ Calibration loaded: ACS_ZERO = {ACS_ZERO:.4f} V")
else:
    ACS_ZERO = VREF / 2
    print(f"[WARNING] {_CALIB_FILE} not found, using default ACS_ZERO = {ACS_ZERO:.4f} V")

CLASSES = ['DOS', 'Normal', 'PortScan', 'SSH_Bruteforce']

ONNX_FILES = {
    'cnn': 'cnn_model.onnx',
    'rf':  'rf_pipeline.onnx',
    'xgb': 'xgb_pipeline.onnx',
}

COLORS = {
    'Normal':         '\033[92m',
    'DOS':            '\033[91m',
    'DoS':            '\033[91m',
    'SSH_Bruteforce': '\033[93m',
    'PortScan':       '\033[95m',
}
RESET = '\033[0m'

# ─── LOAD ONNX ──────────────────────────────────────────────
onnx_file = ONNX_FILES[args.model]
if not os.path.exists(onnx_file):
    print(f"[ERROR] {onnx_file} not found!")
    if args.model == 'cnn':
        print("       python cnn_model.py → produces cnn_model.onnx")
    else:
        print("       python rf_onnx_export.py → produces rf_pipeline.onnx")
    sys.exit(1)

try:
    import onnxruntime as ort
    sess     = ort.InferenceSession(onnx_file,
                   providers=['CPUExecutionProvider'])
    inp_name = sess.get_inputs()[0].name
    n_outputs = len(sess.get_outputs())
    print(f"✓ Model loaded: {onnx_file} ({os.path.getsize(onnx_file)/1e6:.1f} MB)")
    print(f"  Input shape  : {sess.get_inputs()[0].shape}")
    for o in sess.get_outputs():
        print(f"  Output       : {o.name}  {o.shape}")
except ImportError:
    print("[ERROR] onnxruntime is not installed: pip install onnxruntime")
    sys.exit(1)

# ─── FEATURE EXTRACTION (for RF/XGB) ────────────────────────
def extract_features(window: np.ndarray) -> np.ndarray:
    """
    Extracts 43 features for the RF/XGB model:
      8 statistical + 32 normalized FFT magnitudes + 3 packet-arrival-rate proxies
      (zero-crossing rate, peak rate, spectral entropy)
    Must be IDENTICAL to feature_extraction.py — otherwise the
    StandardScaler receives out-of-distribution input (see the
    calibration lesson learned).
    NOT USED for the CNN — the CNN consumes the raw signal directly.
    """
    m     = np.mean(window)
    s     = np.std(window)
    rms   = np.sqrt(np.mean(window ** 2))
    ptp   = np.ptp(window)
    skw   = float(stats.skew(window))
    krt   = float(stats.kurtosis(window))
    crest = float(np.max(np.abs(window)) / (rms + 1e-9))
    eng   = float(np.sum(window ** 2))
    fft   = np.abs(np.fft.rfft(window - m))[:32]
    fft   = fft / (np.sum(fft) + 1e-9)

    centered = window - m
    zcr = float(np.mean(np.diff(np.sign(centered)) != 0))
    peaks, _ = find_peaks(np.abs(centered), prominence=0.5 * (s + 1e-9))
    peak_rate = len(peaks) / len(window)
    p = fft + 1e-12
    spectral_entropy = float(-np.sum(p * np.log(p)) / np.log(len(p)))

    return np.concatenate([[m, s, rms, ptp, skw, krt, crest, eng], fft,
                            [zcr, peak_rate, spectral_entropy]]).astype(np.float32)

# ─── INFERENCE ───────────────────────────────────────────────
def infer(window: np.ndarray):
    """
    window: (WIN_SIZE,) float32 — raw voltage/current values

    CNN: z-score normalized raw signal → (1, 1, WIN_SIZE)
    RF : feature extraction → (1, 40), rf_pipeline.onnx applies the scaler internally

    Returns: (class_name: str, confidence: float, all_probabilities: ndarray)
    """
    if args.model == 'cnn':
        # Per-window z-score (same preprocessing as cnn_model.py)
        mu  = window.mean()
        std = window.std() + 1e-9
        win_norm = ((window - mu) / std)[np.newaxis, np.newaxis, :].astype(np.float32)

        logits = sess.run(None, {inp_name: win_norm})[0][0]
        probs  = np.exp(logits - logits.max())
        probs /= probs.sum()

    else:  # rf
        # Feature extraction → rf_pipeline.onnx (scaler is inside the pipeline)
        feats = extract_features(window).reshape(1, -1)   # (1, 40), raw — scaler is inside the pipeline
        out   = sess.run(None, {inp_name: feats})
        # out[0] = label indices, out[1] = probabilities (ndarray, shape [1, 4])
        if n_outputs >= 2 and isinstance(out[1], np.ndarray):
            probs = out[1][0].astype(np.float32)
        elif n_outputs >= 2 and isinstance(out[1], list):
            probs = np.array([out[1][0].get(c, 0.0) for c in CLASSES], dtype=np.float32)
        else:
            # Only label output is available, no probability
            idx = int(out[0][0])
            one_hot = np.zeros(len(CLASSES), dtype=np.float32)
            one_hot[idx] = 1.0
            probs = one_hot

    idx   = int(np.argmax(probs))
    label = CLASSES[idx] if idx < len(CLASSES) else str(idx)
    return label, float(probs[idx]), probs

# ─── START SPI ──────────────────────────────────────────────
spi = None
if not args.demo:
    try:
        import spidev
        spi = spidev.SpiDev()
        spi.open(0, 0)
        spi.max_speed_hz = SPI_SPEED
        spi.mode = 0
        print(f"✓ SPI opened (CE0, {SPI_SPEED/1e6:.1f} MHz)")
    except Exception as e:
        print(f"[WARNING] SPI could not be opened: {e} → Switching to demo mode.")
        args.demo = True

# ─── SAMPLING ───────────────────────────────────────────────
# NOTE: converted to exactly the same unit (Amps) as data_collector.py —
# the training data was in Amps, not raw voltage (see the ACS_ZERO note above).
def read_adc_voltage() -> float:
    r = spi.xfer2([0x01, (0x08 | args.channel) << 4, 0x00])
    v = ((r[1] & 0x03) << 8 | r[2]) * VREF / 1023.0
    return (v - ACS_ZERO) / ACS_SENS

_sim_t = 0.0
def read_sim_voltage() -> float:
    global _sim_t
    _sim_t += DT
    v = float(np.clip(1.65 + 0.05 * np.sin(2 * np.pi * 50 * _sim_t)
                      + np.random.normal(0, 0.01), 0, VREF))
    return (v - ACS_ZERO) / ACS_SENS

read_voltage = read_sim_voltage if args.demo else read_adc_voltage

# ─── LOG FILE ─────────────────────────────────────────────
log_writer = log_file = None
if args.log:
    log_file   = open(args.log, 'w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['timestamp', 'class', 'confidence'] + CLASSES)
    print(f"✓ Log: {args.log}")

# ─── DEMO MODE — speed test + scenarios ─────────────────────
if args.demo:
    print(f"\n{'='*55}")
    print(f"  DEMO MODE — Model: {args.model.upper()}")
    print(f"{'='*55}")

    # Speed test
    dummy = np.random.normal(1.65, 0.03, WIN_SIZE).astype(np.float32)
    N = 300
    t0 = time.perf_counter()
    for _ in range(N):
        infer(dummy)
    ms = (time.perf_counter() - t0) / N * 1000
    print(f"\n── Speed Test ({N} repetitions) ──")
    print(f"  Inference     : {ms:.2f} ms/window")
    print(f"  Real-time     : {'✓ YES' if ms < 100 else '✗ NO'} (threshold: 100ms)")
    print(f"  RPi estimate  : ~{ms*7:.1f} ms/window")

    # Scenario tests
    scenarios = [
        ("Normal traffic",            1.65, 0.03, "Normal"),
        ("SSH Brute-Force",           1.75, 0.07, "SSH_Bruteforce"),
        ("DoS SYN Flood",             1.80, 0.10, "DOS"),
        ("Port Scan (Nmap)",          1.68, 0.05, "PortScan"),
    ]
    print(f"\n── Scenario Tests ──")
    print(f"{'Scenario':<26} {'Prediction':<18} {'Confidence':>6}")
    print("─" * 55)
    for name, mu, sigma, _ in scenarios:
        win = np.random.normal(mu, sigma, WIN_SIZE).astype(np.float32)
        lbl, conf, _ = infer(win)
        color = COLORS.get(lbl, '')
        print(f"{name:<26} {color}{lbl:<18}{RESET} {conf:5.1%}")

    print(f"\n{'='*55}")
    print(f"SUMMARY — For the Paper")
    print(f"{'='*55}")
    print(f"  Model          : {args.model.upper()} ONNX")
    print(f"  Model size     : {os.path.getsize(onnx_file)/1e6:.1f} MB")
    print(f"  Inference (PC) : {ms:.2f} ms/window")
    print(f"  RPi estimate   : ~{ms*7:.1f} ms/window")
    print(f"  Real-time      : {'Yes' if ms*7 < 100 else 'No'}")

    if log_file:
        log_file.close()
    if spi:
        spi.close()
    sys.exit(0)

# ─── LIVE MODE ───────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  Wattson Live IDS  |  Model: {args.model.upper()}")
print(f"  Channel: CH{args.channel}  |  Window: {WIN_SIZE} sample (~100ms)")
print(f"{'='*55}\n")

buffer    = np.zeros(WIN_SIZE, dtype=np.float32)
buf_idx   = 0
win_count = 0
alert_cnt = 0
t_start = t_next = time.perf_counter()

try:
    while True:
        now = time.perf_counter()
        if now < t_next:
            pass
        t_next += DT

        buffer[buf_idx] = read_voltage()
        buf_idx += 1

        if buf_idx >= WIN_SIZE:
            buf_idx   = 0
            win_count += 1

            label, conf, probs = infer(buffer.copy())
            ts    = datetime.now().strftime('%H:%M:%S.%f')[:-3]
            color = COLORS.get(label, '')
            is_attack = label != 'Normal'

            if is_attack or args.verbose:
                bar = '█' * int(conf * 20) + '░' * (20 - int(conf * 20))
                icon = '⚠ ATTACK' if is_attack else '✓ Normal '
                print(f"[{ts}] {color}{icon}{RESET}  {label:<18} {conf:5.1%}  [{bar}]")
                if args.verbose:
                    for c, p in zip(CLASSES, probs):
                        print(f"         {c:<18} {p:5.1%}{'  ←' if c == label else ''}")

            if is_attack and log_writer:
                log_writer.writerow([ts, label, f'{conf:.4f}'] +
                                    [f'{p:.4f}' for p in probs])
                log_file.flush()
                alert_cnt += 1

            if win_count % 60 == 0:
                elapsed = time.perf_counter() - t_start
                print(f"\n── {win_count} windows | {elapsed:.0f}s | "
                      f"alerts: {alert_cnt} ──\n")

except KeyboardInterrupt:
    elapsed = time.perf_counter() - t_start
    print(f"\nStopped — {win_count} windows, {elapsed:.1f}s, {alert_cnt} alerts")

finally:
    if spi:
        spi.close()
    if log_file:
        log_file.close()
