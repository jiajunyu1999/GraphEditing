"""
iclr19-graph2graph

Copyright (c) 2019 Wengong Jin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
#!/usr/bin/env python
import numpy as np
from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import AllChem
from rdkit import DataStructs
import pickle
import os
import os.path as op
rdBase.DisableLog('rdApp.error')

"""Scores based on an ECFP classifier for activity."""

clf_model = None
onnx_session = None
onnx_input_name = 'input'

def load_model():
    global clf_model, onnx_session, onnx_input_name
    base_dir = op.dirname(__file__)

    # Prefer ONNX if present (works on modern Python)
    onnx_path = op.join(base_dir, 'clf.onnx')
    onnx_err = None
    if op.exists(onnx_path):
        try:
            import onnxruntime as ort
            onnx_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
            # Try to detect input name if available
            if onnx_session.get_inputs():
                onnx_input_name = onnx_session.get_inputs()[0].name
            return
        except Exception as e:
            onnx_err = e
            onnx_session = None

    # Fallback: legacy pickle (requires compatible Python/sklearn)
    pkl_path = op.join(base_dir, 'clf_py36.pkl')
    if not op.exists(pkl_path):
        if onnx_err is not None:
            raise ImportError(
                f"Found '{onnx_path}' but failed to load it (install `onnxruntime`?). "
                f"Fallback pickle '{pkl_path}' is also missing. Original error: {onnx_err}"
            )
        raise FileNotFoundError(
            f"DRD2 scorer model not found. Expected either '{onnx_path}' or '{pkl_path}'."
        )
    # sklearn import is needed for unpickling certain model classes
    from sklearn import svm  # noqa: F401
    with open(pkl_path, "rb") as f:
        clf_model = pickle.load(f)

def get_score(mol):
    if clf_model is None and onnx_session is None:
        load_model()

    if mol:
        fp = fingerprints_from_mol(mol)
        if onnx_session is not None:
            # onnxruntime expects float32
            x = fp.astype(np.float32)
            outputs = onnx_session.run(None, {onnx_input_name: x})

            # Handle outputs like [array([0], dtype=int64), [{0: 0.97, 1: 0.03}]]
            # Find probs if present
            for out in outputs:
                # If output is a list of dict (prob map), use that
                if isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict):
                    # Find first dict with class 1 prob
                    prob_dict = out[0]
                    if 1 in prob_dict:
                        return float(prob_dict[1])
                # If output is a dict itself
                elif isinstance(out, dict) and 1 in out:
                    return float(out[1])
                # If output is a 2d array of shape [n, 2]: classic probability output
                elif isinstance(out, np.ndarray) and out.ndim == 2 and out.shape[1] == 2:
                    return float(out[0, 1])
            # As a last resort, if a 1d/2d array, try to return a scalar
            for out in outputs:
                if isinstance(out, np.ndarray):
                    return float(np.ravel(out)[0])
        else:
            score = clf_model.predict_proba(fp)[:, 1]
            return float(score)
    return 0.0

def fingerprints_from_mol(mol):
    fp = AllChem.GetMorganFingerprint(mol, 3, useCounts=True, useFeatures=True)
    size = 2048
    nfp = np.zeros((1, size), np.int32)
    for idx,v in fp.GetNonzeroElements().items():
        nidx = idx%size
        nfp[0, nidx] += int(v)
    return nfp
if __name__ == "__main__":
    # Quick smoke check
    smiles = "C(N(C)C(c1cc(OC)c(cc1F)F)Cn1c(nc2c3c(n(c(n2C)=O)C)nc(N)nc13)=O)C"
    mol = Chem.MolFromSmiles(smiles)
    
    print(
        get_score(
            mol
        )
    )
