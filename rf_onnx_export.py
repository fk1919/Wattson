"""
Wattson — Converting the RF Model to ONNX
rf_model.pkl (RandomForest + StandardScaler + LabelEncoder)
→ rf_pipeline.onnx  (via skl2onnx — runs with onnxruntime on the RPi)

Setup:
    pip install skl2onnx onnxruntime numpy scikit-learn

Usage:
    python rf_onnx_export.py            # rf_model.pkl → rf_pipeline.onnx
    python rf_onnx_export.py --test     # also run a validation test

Why ONNX?
    - rf_model.pkl: ~50 MB (requires the scikit-learn dependency)
    - rf_pipeline.onnx: ~45 MB (only onnxruntime needed, no scikit-learn)
    - onnxruntime inference on the RPi: ~15ms/window (faster than pickle)
"""

import argparse
import pickle
import time
import sys
import os

import numpy as np

parser = argparse.ArgumentParser(description='RF → ONNX export')
parser.add_argument('--input',  default='rf_model.pkl',
                    help='Input pkl file (default: rf_model.pkl)')
parser.add_argument('--output', default='rf_pipeline.onnx',
                    help='Output ONNX file (default: rf_pipeline.onnx)')
parser.add_argument('--test',   action='store_true',
                    help='Run a validation test after export')
parser.add_argument('--data',   default='feature_matrix.npz',
                    help='Test data (default: feature_matrix.npz)')
args = parser.parse_args()

# ─── skl2onnx INSTALLATION CHECK ───────────────────────────────
try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    from sklearn.pipeline import Pipeline
except ImportError:
    print("[ERROR] skl2onnx not found.")
    print("       pip install skl2onnx")
    sys.exit(1)

# ─── LOAD PKL ───────────────────────────────────────────────────
if not os.path.exists(args.input):
    print(f"[ERROR] {args.input} not found!")
    print("       Run multi_model_comparison.py first.")
    sys.exit(1)

print(f"Loading: {args.input} ({os.path.getsize(args.input)/1e6:.1f} MB)")
with open(args.input, 'rb') as f:
    bundle = pickle.load(f)

rf      = bundle['model']    # RandomForestClassifier
scaler  = bundle['scaler']   # StandardScaler
le      = bundle['le']       # LabelEncoder
classes = le.classes_
n_feat  = scaler.n_features_in_

print(f"✓ Model loaded")
print(f"  Classes     : {classes}")
print(f"  Features    : {n_feat}")
print(f"  Tree count  : {rf.n_estimators}")

# ─── BUILD SKLEARN PIPELINE AND CONVERT TO ONNX ─────────────
# skl2onnx also supports converting a pipeline in a single step
pipe = Pipeline([
    ('scaler', scaler),
    ('rf',     rf),
])

print(f"\nConverting to ONNX (this may take a few minutes)...")
t0 = time.time()

initial_type = [('float_input', FloatTensorType([None, n_feat]))]

try:
    onnx_model = convert_sklearn(
        pipe,
        initial_types=initial_type,
        target_opset=15,
        options={id(rf): {'zipmap': False}},   # make the probability output an array
    )
except Exception as e:
    print(f"[WARNING] zipmap=False attempt failed: {e}")
    print("        Retrying with zipmap=True...")
    onnx_model = convert_sklearn(
        pipe,
        initial_types=initial_type,
        target_opset=15,
    )

convert_time = time.time() - t0
print(f"Conversion time: {convert_time:.1f}s")

# Save the ONNX file
with open(args.output, 'wb') as f:
    f.write(onnx_model.SerializeToString())

size_mb = os.path.getsize(args.output) / 1e6
print(f"\n✓ Saved: {args.output}  ({size_mb:.1f} MB)")

# ─── ONNX OUTPUT STRUCTURE ───────────────────────────────────
try:
    import onnxruntime as ort

    sess     = ort.InferenceSession(args.output,
                   providers=['CPUExecutionProvider'])
    inp_name = sess.get_inputs()[0].name
    outputs  = [(o.name, o.shape) for o in sess.get_outputs()]
    print(f"\nONNX input  : {inp_name}  shape={sess.get_inputs()[0].shape}")
    for oname, oshape in outputs:
        print(f"ONNX output : {oname}  shape={oshape}")

    # Detect: is the probability output an array or a dict?
    dummy = np.zeros((1, n_feat), dtype=np.float32)
    out   = sess.run(None, {inp_name: dummy})
    if len(out) >= 2 and isinstance(out[1], np.ndarray):
        PROB_IDX    = 1
        PROB_FORMAT = 'array'
        print(f"Probability format: ndarray  shape={out[1].shape}")
    elif len(out) >= 2 and isinstance(out[1], list):
        PROB_IDX    = 1
        PROB_FORMAT = 'zipmap'
        print(f"Probability format: zipmap (list of dicts)")
    else:
        PROB_IDX    = None
        PROB_FORMAT = 'unknown'
        print(f"[WARNING] Unrecognized output format — update rpi_inference.py")

    print(f"\nUsage in rpi_inference.py:")
    if PROB_FORMAT == 'array':
        print(f"    out = sess.run(None, {{inp_name: features}})")
        print(f"    probs = out[1][0]  # shape: ({len(classes)},)")
    elif PROB_FORMAT == 'zipmap':
        print(f"    out = sess.run(None, {{inp_name: features}})")
        print(f"    probs = np.array([out[1][0].get(c, 0.0) for c in CLASSES])")

except ImportError:
    print("\n[WARNING] onnxruntime is not installed, validation skipped.")
    print("        pip install onnxruntime")

# ─── VALIDATION TEST ─────────────────────────────────────────
if args.test:
    print(f"\n{'─'*50}")
    print("Starting validation test...")

    if not os.path.exists(args.data):
        print(f"[WARNING] {args.data} not found, test skipped.")
    else:
        try:
            from sklearn.metrics import roc_auc_score
            import onnxruntime as ort

            data  = np.load(args.data, allow_pickle=True)
            X_raw = data['X'].astype(np.float32)
            y_raw = data['y'].astype(str)
            y_raw = np.where(np.char.startswith(y_raw, 'Port_Scan'), 'PortScan', y_raw)
            y     = le.transform(y_raw)

            # Test data (20%)
            from sklearn.model_selection import train_test_split
            _, X_te, _, y_te = train_test_split(
                X_raw, y, test_size=0.2, random_state=42, stratify=y
            )

            sess_test = ort.InferenceSession(args.output,
                            providers=['CPUExecutionProvider'])
            inp_name  = sess_test.get_inputs()[0].name

            print(f"Test set: {len(X_te):,} samples")

            # Batch inference
            BATCH  = 256
            preds, probs_all = [], []
            t0 = time.time()

            for i in range(0, len(X_te), BATCH):
                Xb  = X_te[i:i+BATCH]
                out = sess_test.run(None, {inp_name: Xb})

                # Label predictions
                preds.extend(out[0].tolist())

                # Probabilities
                if PROB_FORMAT == 'array':
                    probs_all.append(out[1])
                elif PROB_FORMAT == 'zipmap':
                    pb = np.array([[d.get(c, 0.0) for c in classes]
                                   for d in out[1]], dtype=np.float32)
                    probs_all.append(pb)

            inf_time = (time.time() - t0) / len(X_te) * 1000

            y_pred  = np.array(preds)
            y_proba = np.vstack(probs_all) if probs_all else None

            acc = np.mean(y_pred == y_te)
            print(f"Accuracy  : {acc:.3f}")

            if y_proba is not None:
                auc = roc_auc_score(y_te, y_proba,
                                    multi_class='ovr', average='weighted')
                print(f"Weighted AUC: {auc:.4f}  (original RF: ~0.9097)")

            print(f"Inference : {inf_time:.3f} ms/sample (batch={BATCH})")

            # RPi estimate (~7x slower)
            print(f"\nEstimate on RPi (ARMv7, ~7× CPU difference):")
            print(f"  ~{inf_time * 7:.0f} ms/window  "
                  f"({'✓' if inf_time * 7 < 100 else '✗'} real-time)")

        except Exception as e:
            print(f"[ERROR] Error during test: {e}")
            import traceback
            traceback.print_exc()

# ─── SUMMARY ────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"SUMMARY")
print(f"{'='*55}")
print(f"  Input      : {args.input}  ({os.path.getsize(args.input)/1e6:.1f} MB)")
print(f"  Output     : {args.output}  ({os.path.getsize(args.output)/1e6:.1f} MB)")
print(f"  Classes    : {', '.join(classes)}")
print(f"\nFiles to copy to the RPi:")
print(f"  {args.output}")
print(f"  rpi_inference.py  (use with the --model rf flag)")
print(f"{'='*55}")
print(f"\nUsage:")
print(f"  scp {args.output} pi@<rpi-ip>:~/wattson/")
print(f"  python rpi_inference.py --model rf")
