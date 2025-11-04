# compare_probs.py  (run in ~/dev/etoxpred)
import numpy as np, pandas as pd

GT  = "tcm_results.csv"     # your ground-truth CSV
SK  = "sk_probs.csv"        # written by dump_sk_probs.py
ON  = "onnx_probs.csv"      # written by dump_onnx_probs.py
OUT = "probs_merged.csv"

gt = pd.read_csv(GT)
sk = pd.read_csv(SK)
on = pd.read_csv(ON)

df = sk.merge(on[["name","onnx_prob_0","onnx_prob_1"]], on="name", how="inner") \
       .merge(gt[["name","Tox-score"]], on="name", how="left")

def mae(a,b): return float(np.mean(np.abs(a - b)))
print(f"MAE vs GT  onnx_prob_0: {mae(df['Tox-score'], df['onnx_prob_0']):.6f}")
print(f"MAE vs GT  onnx_prob_1: {mae(df['Tox-score'], df['onnx_prob_1']):.6f}")
print(f"MAE vs GT   sk_prob_0 : {mae(df['Tox-score'], df['sk_prob_0']):.6f}")
print(f"MAE vs GT   sk_prob_1 : {mae(df['Tox-score'], df['sk_prob_1']):.6f}")

df.to_csv(OUT, index=False)
print(f"Wrote {OUT} ({len(df)} rows)")

