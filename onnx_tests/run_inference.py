# run_inference.py
import onnxruntime as ort, numpy as np
sess = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
def predict(X: np.ndarray):
    return sess.run(None, {"X": X.astype(np.float32)})[0]

