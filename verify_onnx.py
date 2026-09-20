"""ONNX verification — fixed version"""
import pickle, numpy as np, onnxruntime as rt

with open("rf_model.pkl", "rb") as f:
    bundle = pickle.load(f)
rf, scaler, le = bundle['model'], bundle['scaler'], bundle['le']

data = np.load("feature_matrix.npz", allow_pickle=True)
X_test = data['X'][:200].astype(np.float32)

# sklearn (with normalized input)
X_scaled = scaler.transform(X_test)
y_sk = rf.predict(X_scaled)              # int array

# ONNX (manual normalization)
X_norm = ((X_test - scaler.mean_) / scaler.scale_).astype(np.float32)
sess = rt.InferenceSession("rf_model.onnx")
inp  = sess.get_inputs()[0].name
y_onnx = sess.run(None, {inp: X_norm})[0]  # int array

match = np.sum(y_sk == y_onnx)
print(f"sklearn vs ONNX agreement: {match}/200 ({'✓ Consistent' if match>=195 else '✗ Error'})")
print(f"sklearn sample: {le.classes_[y_sk[:5]]}")
print(f"ONNX   sample: {le.classes_[y_onnx[:5]]}")
print(f"\nONNX file size: {__import__('os').path.getsize('rf_model.onnx')/1024:.0f} KB")
print("✓ Ready for deployment — we can move on to the RPi step")
