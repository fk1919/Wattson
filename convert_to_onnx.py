"""
Random Forest → ONNX conversion
The resulting .onnx file will be copied to the Raspberry Pi
"""

import pickle
import numpy as np
import onnxruntime as rt
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import time

# ─── LOAD MODEL ──────────────────────────────────────────
print("Loading rf_model.pkl...")
with open("rf_model.pkl", "rb") as f:
    bundle = pickle.load(f)

rf     = bundle['model']
scaler = bundle['scaler']
le     = bundle['le']
class_names = le.classes_
print(f"Classes: {class_names}")
print(f"Number of features: {rf.n_features_in_}")

# ─── ONNX CONVERSION ─────────────────────────────────────────
print("\nConverting to ONNX...")
n_features = rf.n_features_in_
initial_type = [('float_input', FloatTensorType([None, n_features]))]

onnx_model = convert_sklearn(
    rf,
    initial_types=initial_type,
    target_opset=15,
    options={rf.__class__: {'zipmap': False}}  # return probabilities as an array
)

with open("rf_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

import os
size_kb = os.path.getsize("rf_model.onnx") / 1024
print(f"Saved: rf_model.onnx ({size_kb:.0f} KB)")

# Also save the scaler parameters (for normalization on the RPi)
np.save("scaler_mean.npy", scaler.mean_.astype(np.float32))
np.save("scaler_scale.npy", scaler.scale_.astype(np.float32))
np.save("class_names.npy", class_names)
print("Saved: scaler_mean.npy, scaler_scale.npy, class_names.npy")

# ─── VALIDATION (test on Windows) ───────────────────────────
print("\n── Validation on Windows ──")
data = np.load("feature_matrix.npz", allow_pickle=True)
X_test = data['X'][:100].astype(np.float32)
y_test = data['y'][:100]

# Normalize
X_norm = (X_test - scaler.mean_.astype(np.float32)) / scaler.scale_.astype(np.float32)

# sklearn prediction
y_pred_sk = rf.predict(scaler.transform(X_test))

# ONNX prediction
sess = rt.InferenceSession("rf_model.onnx")
input_name = sess.get_inputs()[0].name
y_pred_onnx_idx = sess.run(None, {input_name: X_norm})[0]
y_pred_onnx = le.inverse_transform(y_pred_onnx_idx)

# Consistency check
match = np.sum(y_pred_sk == y_pred_onnx)
print(f"sklearn vs ONNX agreement: {match}/100 ({'✓ Consistent' if match==100 else '✗ Mismatch'})")

# ─── SPEED TEST ──────────────────────────────────────────────
print("\n── Speed test (Windows, 1 window) ──")
single = X_norm[:1]
N = 1000
t0 = time.perf_counter()
for _ in range(N):
    sess.run(None, {input_name: single})
elapsed = (time.perf_counter() - t0) / N * 1000
print(f"Average inference time: {elapsed:.3f} ms/window")
print(f"Real-time capacity: {1000/elapsed:.0f} windows/second")
print(f"(Each window = 100ms of data → {'✓ Real-time' if elapsed < 100 else '✗ Slow'})")

print("\n─────────────────────────────────────────────────")
print("Files to copy to the RPi:")
print("  1. rf_model.onnx")
print("  2. scaler_mean.npy")
print("  3. scaler_scale.npy")
print("  4. class_names.npy")
print("  5. rpi_inference.py  (to be created in the next step)")
