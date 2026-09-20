import pickle, os, time
import numpy as np

with open("svm_rbf_model_ethernet_fixed.pkl", "rb") as f:
    bundle = pickle.load(f)

svm_clf = bundle['model']
scaler  = bundle['scaler']
le      = bundle['le']
classes = le.classes_
n_feat  = scaler.n_features_in_

print(f"Classes: {classes} | Features: {n_feat}")

from sklearn.pipeline import Pipeline
pipe = Pipeline([('scaler', scaler), ('svm', svm_clf)])

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

initial_type = [('float_input', FloatTensorType([None, n_feat]))]
t0 = time.time()
onnx_model = convert_sklearn(
    pipe, initial_types=initial_type, target_opset=15,
    options={id(svm_clf): {'zipmap': False}},
)
print(f"Conversion time: {time.time()-t0:.1f}s")

with open("svm_pipeline_ethernet_fixed.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

size_mb = os.path.getsize("svm_pipeline_ethernet_fixed.onnx") / 1e6
print(f"Saved: svm_pipeline_ethernet_fixed.onnx ({size_mb:.1f} MB)")

import onnxruntime as ort
sess = ort.InferenceSession("svm_pipeline_ethernet_fixed.onnx", providers=['CPUExecutionProvider'])
inp_name = sess.get_inputs()[0].name
for o in sess.get_outputs():
    print(f"Output: {o.name} {o.shape}")

dummy = np.zeros((1, n_feat), dtype=np.float32)
out = sess.run(None, {inp_name: dummy})
print("label out:", out[0], "prob type:", type(out[1]))

data = np.load("feature_matrix_ethernet_fixed.npz", allow_pickle=True)
X_raw = data['X'].astype(np.float32)
y_raw = data['y'].astype(str)
sessions = data['sessions'].astype(str)
y = le.transform(y_raw)

rng = np.random.RandomState(42)
test_idx = []
for cls_i in np.unique(y):
    cls_sessions = np.unique(sessions[y == cls_i])
    rng.shuffle(cls_sessions)
    n = len(cls_sessions)
    n_test = max(1, round(n * 0.2))
    n_train = n - n_test
    te_sess = set(cls_sessions[n_train:])
    cls_mask = (y == cls_i)
    test_idx.extend(np.where(cls_mask & np.isin(sessions, list(te_sess)))[0])
test_idx = np.array(test_idx)
X_te, y_te = X_raw[test_idx], y[test_idx]

from sklearn.metrics import roc_auc_score, classification_report
out = sess.run(None, {inp_name: X_te})
y_pred = out[0]
y_proba = out[1]
print(classification_report(y_te, y_pred, target_names=classes, digits=3))
auc = roc_auc_score(y_te, y_proba, multi_class='ovr', average='weighted')
print(f"Weighted AUC (session-based test): {auc:.4f}")
