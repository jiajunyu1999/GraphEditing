from __future__ import annotations

import argparse
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd


def _require_rdkit() -> None:
    try:
        import rdkit  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency: RDKit. Install it (e.g. `conda install -c conda-forge rdkit`)."
        ) from exc


def _compute_one(
    smi: str,
    fp_radius: int,
    fp_nbits: int,
) -> tuple[bool, Any]:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False, None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, int(fp_radius), nBits=int(fp_nbits))
    return True, fp


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast fingerprint-based diversity sampler.")
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--smiles_col", type=str, default="source_smiles")
    parser.add_argument("--sample_n", type=int, default=None)
    parser.add_argument("--keep_ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--fp_radius", type=int, default=2)
    parser.add_argument("--fp_nbits", type=int, default=2048)
    args = parser.parse_args()

    _require_rdkit()

    df = pd.read_csv(args.input_csv, engine="c")
    if args.smiles_col not in df.columns:
        raise KeyError(f"Missing smiles_col={args.smiles_col} in {args.input_csv}. Columns: {list(df.columns)}")

    smiles_list = df[args.smiles_col].astype(str).tolist()
    keep_mask: list[bool] = [False] * len(smiles_list)
    fps: list[Any] = [None] * len(smiles_list)

    def _assign(i: int, smi: str) -> None:
        ok, fp = _compute_one(smi, args.fp_radius, args.fp_nbits)
        keep_mask[i] = ok
        fps[i] = fp

    num_workers = max(1, int(args.num_workers))
    if num_workers <= 1:
        for i, smi in enumerate(smiles_list):
            _assign(i, smi)
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            futures = [ex.submit(_assign, i, smi) for i, smi in enumerate(smiles_list)]
            for f in futures:
                f.result()

    valid_idx = [i for i, ok in enumerate(keep_mask) if ok]
    if not valid_idx:
        raise ValueError("No valid SMILES found for diversity sampling")

    df = df.loc[valid_idx].copy().reset_index(drop=True)
    fps_valid = [fps[i] for i in valid_idx]

    sample_n = args.sample_n
    if sample_n is None and args.keep_ratio is not None:
        sample_n = int(round(len(df) * float(args.keep_ratio)))
    if sample_n is None or sample_n <= 0 or sample_n >= len(df):
        sample_n = len(df)

    from rdkit import DataStructs

    rng = random.Random(int(args.seed))
    n = len(fps_valid)
    if sample_n >= n:
        picks = list(range(n))
    else:
        first = rng.randrange(n)
        picks = [first]
        min_dist = [1.0] * n
        min_dist[first] = -1.0
        while len(picks) < sample_n:
            sims = DataStructs.BulkTanimotoSimilarity(fps_valid[picks[-1]], fps_valid)
            for i, sim in enumerate(sims):
                if min_dist[i] < 0:
                    continue
                dist = 1.0 - float(sim)
                if dist < min_dist[i]:
                    min_dist[i] = dist
            next_idx = max(range(n), key=lambda i: min_dist[i])
            if min_dist[next_idx] < 0:
                break
            picks.append(next_idx)
            min_dist[next_idx] = -1.0

    out_df = df.iloc[list(picks)].reset_index(drop=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(out_df)} rows -> {args.output_csv}")


if __name__ == "__main__":
    main()
