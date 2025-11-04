# dump_onnx_probs.py  (run in ~/dev/etoxpred)
import numpy as np, pandas as pd, onnxruntime as ort
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem
from rdkit import DataStructs
rdBase.DisableLog('rdApp.warning')

SMI = "tcm600_nr.smi"
ONNX = "model.onnx"
OUT  = "onnx_probs.csv"

def load_smi_as_ecfp4(path):
    df = pd.read_csv(path, sep="\t", names=["smiles","name"])
    xs, smiles, names = [], df.smiles.tolist(), df.name.tolist()
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        m = Chem.AddHs(m)
        fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=1024)
        arr = np.zeros((1024,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        xs.append(arr.astype(np.float32))
    return np.asarray(xs, np.float32), smiles, names

def onnx_predict_proba(sess, X):
    out_names = [o.name for o in sess.get_outputs()]
    try:
        prob_idx = out_names.index("output_probability")
    except ValueError:
        prob_idx = next((i for i,n in enumerate(out_names) if "prob" in n.lower()), len(out_names)-1)
    inp = sess.get_inputs()[0].name
    prob_out = sess.run(out_names, {inp: X})[prob_idx]

    # ZipMap list-of-dicts → arrays
    if isinstance(prob_out, list) and prob_out and isinstance(prob_out[0], dict):
        p0, p1 = [], []
        for d in prob_out:
            dd = {str(k): float(v) for k, v in d.items()}
            if "0" in dd and "1" in dd:
                p0.append(dd["0"]); p1.append(dd["1"])
            else:
                keys = sorted([k for k in dd if k.isdigit()], key=lambda k: int(k))
                if len(keys) >= 2:
                    p0.append(dd[keys[0]]); p1.append(dd[keys[1]])
                else:
                    v1 = max(dd.values()); v0 = 1.0 - v1
                    p0.append(v0); p1.append(v1)
        return np.asarray(p0, np.float32), np.asarray(p1, np.float32)

    arr = np.asarray(prob_out, np.float32)
    if arr.ndim == 1: arr = arr.reshape(-1, 1)
    if arr.shape[1] == 2:
        return arr[:, 0], arr[:, 1]
    if arr.shape[1] == 1:
        p1 = arr[:, 0]; p0 = 1.0 - p1
        return p0, p1
    return arr[:, 0], arr[:, -1]

X, smiles, names = load_smi_as_ecfp4(SMI)
sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
p0, p1 = onnx_predict_proba(sess, X)

pd.DataFrame({
    "name": names,
    "smiles": smiles,
    "onnx_prob_0": p0,
    "onnx_prob_1": p1,
}).to_csv(OUT, index=False)

print(f"Wrote {OUT} ({len(names)} rows)")

