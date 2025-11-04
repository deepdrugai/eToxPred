import argparse
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit import rdBase
rdBase.DisableLog('rdApp.warning')

from sascore import SAscore
import onnxruntime as ort
import log

log.setLevel('INFO')


def myargs():
    p = argparse.ArgumentParser()
    p.add_argument('--datafile', required=True, help='input .smi file (SMILES<TAB>Name)')
    p.add_argument('--modelfile', required=True, help='path to the ONNX model file')
    p.add_argument('--outputfile', default='./results.csv', help='output CSV (default: results.csv)')
    p.add_argument('--prob_class', type=int, default=0,
                   help='Which class prob to use for Tox-score (0 or 1). Default: 1')
    return p.parse_args()


def load_data(filename):
    """
    Returns:
      X: float32 (n, 1024) with 0/1 ECFP4 (radius=2) bits
      smiles_list, names, mols
    """
    df = pd.read_csv(filename, sep='\t', names=['smiles', 'name'])
    smiles_list = df['smiles'].tolist()
    names = df['name'].tolist()

    X, mols = [], []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            log.warning(f'Error parsing SMILES at line {i+1}: {smi}')
            continue
        # Match training pipeline: AddHs, ECFP4, 1024 bits
        mol = Chem.AddHs(mol)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        arr = np.zeros((1024,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)  # -> 0/1
        X.append(arr.astype(np.float32))
        mols.append(mol)

    X = np.asarray(X, dtype=np.float32)
    return X, smiles_list, names, mols


def _pick_label_and_prob_outputs(sess):
    """
    Find indices of label and probability outputs, if present.
    Returns (label_idx, prob_idx, out_names).
    """
    out_names = [o.name for o in sess.get_outputs()]
    label_idx = next((i for i, n in enumerate(out_names) if 'label' in n.lower()), None)
    try:
        prob_idx = out_names.index('output_probability')
    except ValueError:
        prob_idx = next((i for i, n in enumerate(out_names) if 'prob' in n.lower()), len(out_names) - 1)
    return label_idx, prob_idx, out_names


def _extract_two_class_probs(prob_out):
    """
    Return a dict of probabilities:
      - If ZipMap dict: return keys as prob_0/prob_1 etc.
      - If tensor with 2 cols: prob_0, prob_1.
      - If tensor with 1 col: prob_single.
    """
    if isinstance(prob_out, list) and prob_out and isinstance(prob_out[0], dict):
        d = prob_out[0]
        ds = {str(k): float(v) for k, v in d.items()}
        out = {}
        if '0' in ds: out['prob_0'] = ds['0']
        if '1' in ds: out['prob_1'] = ds['1']
        for k, v in ds.items():
            out[f'prob_{k}'] = v
        return out

    arr = np.asarray(prob_out, dtype=np.float32).reshape(1, -1)
    out = {}
    if arr.shape[1] >= 2:
        out['prob_0'] = float(arr[0, 0])
        out['prob_1'] = float(arr[0, 1])
        for j in range(arr.shape[1]):
            out[f'prob_col{j}'] = float(arr[0, j])
        return out
    if arr.shape[1] == 1:
        out['prob_single'] = float(arr[0, 0])
        return out
    return out


def predict(opt):
    X, smiles_list, names, _ = load_data(opt.datafile)

    # sanity check: fingerprint sums
    fp_sums = [int(X[j].sum()) for j in range(min(5, len(X)))]
    log.warning(f"fp row sums (first 5): {fp_sums}")

    log.info('...loading ONNX model')
    sess = ort.InferenceSession(opt.modelfile, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    label_idx, prob_idx, out_names = _pick_label_and_prob_outputs(sess)
    log.info(f'ONNX outputs: {out_names}')

    reg = SAscore()
    rows = []

    log.info('...starting prediction')
    for i in range(X.shape[0]):
        feeds = {input_name: X[i:i+1]}
        outs = sess.run(out_names, feeds)
        label_out = outs[label_idx] if label_idx is not None else None
        prob_out = outs[prob_idx]

        probs = _extract_two_class_probs(prob_out)
        tox_score = probs.get(f'prob_{opt.prob_class}',
                              probs.get('prob_1',
                                        probs.get('prob_single', np.nan)))
        sa_score = float(reg(smiles_list[i]))

        row = {'name': names[i], 'smiles': smiles_list[i],
               'Tox-score': tox_score, 'SAscore': sa_score}
        row.update(probs)
        rows.append(row)

    df_out = pd.DataFrame(rows)
    # Log first 20 rows
    log.info("Sample predictions:\n" + df_out.head(20).to_string(index=False))

    df_out.to_csv(opt.outputfile, index=False)
    log.info("Prediction completed successfully")


if __name__ == "__main__":
    opt = myargs()
    predict(opt)