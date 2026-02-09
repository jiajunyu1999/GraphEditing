from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from .config import Config

Sign = Literal["pos", "neg"]
OpName = Literal["add", "remove", "replace"]


@dataclass(frozen=True)
class FilterResult:
    df: pd.DataFrame
    keep_indices: list[int]
    dropped_nan_rows: int
    notes: list[str]


def _delta_cols(df: pd.DataFrame, property_names: list[str]) -> list[str]:
    cols = [f"{p}_delta" for p in property_names if f"{p}_delta" in df.columns]
    if not cols:
        raise KeyError(
            "No `*_delta` columns found. Run `compute_properties.py` first to create delta columns."
        )
    return cols


def _score(df: pd.DataFrame, delta_cols: list[str]) -> pd.Series:
    # Normalize by per-column std so different property scales contribute comparably.
    std = df[delta_cols].std(axis=0, skipna=True).replace(0.0, 1.0)
    z = df[delta_cols].abs().div(std, axis=1)
    return z.sum(axis=1)


def _prefilter_top_delta(
    df: pd.DataFrame,
    *,
    delta_cols: list[str],
    quantile: float,
) -> tuple[pd.DataFrame, int]:
    """
    Keep rows where at least one |delta| is in the top `quantile` (per-column).
    Example: quantile=0.9 keeps rows with any |delta_col| >= q90(|delta_col|).
    """
    q = float(quantile)
    if not (0.0 < q < 1.0):
        raise ValueError("quantile must be in (0, 1)")
    if len(df) == 0:
        return df, 0

    absd = df[delta_cols].abs()
    thr = absd.quantile(q, axis=0).fillna(np.inf)
    mask = absd.ge(thr, axis=1).any(axis=1)
    before = len(df)
    df2 = df.loc[mask].reset_index(drop=True)
    return df2, before - len(df2)


def _sign_mask(df: pd.DataFrame, col: str, *, eps: float) -> tuple[pd.Series, pd.Series]:
    d = df[col]
    return d > eps, d < -eps


def _safe_json_loads(s: Any) -> Any:
    try:
        return json.loads(s) if isinstance(s, str) and s else None
    except Exception:
        return None


def _require_rdkit() -> None:
    try:
        import rdkit  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency: RDKit is required for atom-type balancing. "
            "Install it (e.g. `conda install -c conda-forge rdkit`) or use `--no-balance_atoms`."
        ) from exc


@lru_cache(maxsize=4096)
def _atom_map_to_bin(start_smiles_tagged: str) -> dict[int, str]:
    """
    Build a map from atom-map number -> atom bin (C/N/O/other) for a tagged SMILES.
    Cached because many rows share the same `start_smiles_tagged`.
    """
    _require_rdkit()
    from rdkit import Chem

    mol = Chem.MolFromSmiles(start_smiles_tagged)
    if mol is None:
        return {}
    out: dict[int, str] = {}
    for a in mol.GetAtoms():
        amap = int(a.GetAtomMapNum() or 0)
        if amap <= 0:
            continue
        z = int(a.GetAtomicNum())
        if z == 6:
            b = "C"
        elif z == 7:
            b = "N"
        elif z == 8:
            b = "O"
        else:
            b = "other"
        out[amap] = b
    return out


def _score_np(delta: np.ndarray, *, col_std: np.ndarray) -> np.ndarray:
    col_std = np.where(col_std == 0.0, 1.0, col_std)
    z = np.abs(delta) / col_std[None, :]
    return z.sum(axis=1)


def _minibatch_kmeans(
    x: np.ndarray,
    *,
    k: int,
    steps: int,
    batch_size: int,
    seed: int,
    init_size: int = 10000,
) -> np.ndarray:
    """
    Very small dependency-free MiniBatchKMeans for diversity sampling.
    - x: [N, D] float32/float64
    Returns: centroids [k, D]
    """
    if x.ndim != 2:
        raise ValueError("x must be 2D")
    n, d = x.shape
    if n == 0:
        raise ValueError("empty x")
    k = int(max(1, min(k, n)))
    steps = int(max(1, steps))
    batch_size = int(max(1, min(batch_size, n)))

    rng = np.random.default_rng(int(seed))
    init_n = int(min(n, max(k, init_size)))
    init_idx = rng.choice(n, size=init_n, replace=False)
    xi = x[init_idx]

    # kmeans++ init on xi
    centers = np.empty((k, d), dtype=x.dtype)
    first = int(rng.integers(0, init_n))
    centers[0] = xi[first]
    dist2 = np.sum((xi - centers[0]) ** 2, axis=1)
    for c in range(1, k):
        probs = dist2 / max(float(dist2.sum()), 1e-12)
        j = int(rng.choice(init_n, p=probs))
        centers[c] = xi[j]
        dist2 = np.minimum(dist2, np.sum((xi - centers[c]) ** 2, axis=1))

    counts = np.zeros((k,), dtype=np.int64)
    for _ in range(steps):
        bidx = rng.integers(0, n, size=batch_size)
        xb = x[bidx]  # [B, D]
        # distances [B, K]
        d2 = np.sum((xb[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        assign = np.argmin(d2, axis=1)
        for cid in range(k):
            mask = assign == cid
            if not np.any(mask):
                continue
            pts = xb[mask]
            m = pts.mean(axis=0)
            counts[cid] += int(mask.sum())
            eta = 1.0 / float(counts[cid])
            centers[cid] = (1.0 - eta) * centers[cid] + eta * m
    return centers


def _assign_clusters(x: np.ndarray, centers: np.ndarray, *, chunk_size: int = 20000) -> np.ndarray:
    n = x.shape[0]
    out = np.empty((n,), dtype=np.int32)
    k = centers.shape[0]
    for s in range(0, n, chunk_size):
        e = min(n, s + chunk_size)
        xb = x[s:e]
        d2 = np.sum((xb[:, None, :] - centers[None, :, :]) ** 2, axis=2)  # [B, K]
        out[s:e] = np.argmin(d2, axis=1).astype(np.int32)
    return out


def _quota_by_size(cluster_sizes: np.ndarray, *, keep_n: int, min_each: int = 1) -> np.ndarray:
    k = int(cluster_sizes.size)
    keep_n = int(keep_n)
    if keep_n <= 0:
        raise ValueError("keep_n must be positive")
    nonempty = np.where(cluster_sizes > 0)[0]
    if nonempty.size == 0:
        return np.zeros((k,), dtype=np.int64)
    # If we have more clusters than budget, keep the biggest clusters.
    if keep_n < nonempty.size * min_each:
        order = nonempty[np.argsort(-cluster_sizes[nonempty])]
        take = order[:keep_n]
        q = np.zeros((k,), dtype=np.int64)
        q[take] = 1
        return q

    frac = cluster_sizes / float(cluster_sizes.sum())
    q = np.maximum(min_each, np.round(frac * keep_n).astype(np.int64))
    # Cap by cluster size.
    q = np.minimum(q, cluster_sizes.astype(np.int64))

    # Adjust sum to match keep_n.
    cur = int(q.sum())
    if cur > keep_n:
        # decrement from largest quotas first
        order = np.argsort(-q)
        for i in order:
            while cur > keep_n and q[i] > min_each:
                q[i] -= 1
                cur -= 1
            if cur == keep_n:
                break
    elif cur < keep_n:
        # increment where capacity remains (prefer larger clusters)
        capacity = cluster_sizes.astype(np.int64) - q
        order = np.argsort(-capacity)
        for i in order:
            while cur < keep_n and capacity[i] > 0:
                q[i] += 1
                capacity[i] -= 1
                cur += 1
            if cur == keep_n:
                break
    return q


def _parse_edits_counts(
    df: pd.DataFrame,
    *,
    indices: list[int],
    edits_col: str,
    start_smiles_tagged_col: str,
    use_atoms: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (op_counts [M,3], atom_counts [M,4], num_edits [M])
    atom bins: C/N/O/other. If use_atoms is False, atom_counts is zeros.
    """
    op_names = ("add", "remove", "replace")
    op_to_i = {n: i for i, n in enumerate(op_names)}
    atom_bins = ("C", "N", "O", "other")
    atom_to_i = {n: i for i, n in enumerate(atom_bins)}

    m = len(indices)
    op_counts = np.zeros((m, 3), dtype=np.float32)
    atom_counts = np.zeros((m, 4), dtype=np.float32)
    num_edits = np.zeros((m,), dtype=np.float32)

    if use_atoms:
        _require_rdkit()

    for j, idx in enumerate(indices):
        edits = _safe_json_loads(df.at[idx, edits_col])
        amap_to_bin: dict[int, str] = {}
        if use_atoms:
            amap_to_bin = _atom_map_to_bin(str(df.at[idx, start_smiles_tagged_col]))

        if not isinstance(edits, list):
            continue

        for e in edits:
            if not isinstance(e, dict):
                continue
            op = e.get("op")
            if op in op_to_i:
                op_counts[j, op_to_i[str(op)]] += 1.0
            if use_atoms:
                try:
                    amap = int(e.get("anchor_atom_map"))
                except Exception:
                    amap = -1
                b = amap_to_bin.get(amap)
                if b in atom_to_i:
                    atom_counts[j, atom_to_i[str(b)]] += 1.0
        num_edits[j] = float(sum(op_counts[j]))
    return op_counts, atom_counts, num_edits


def _repair_min_signs(
    delta: np.ndarray,
    *,
    selected_pos: list[int],
    remaining_pos: list[int],
    eps: float,
    min_per_sign: int,
    notes: list[str],
) -> tuple[list[int], list[int]]:
    """
    Enforce at least `min_per_sign` positive and negative examples per property (if feasible),
    via swaps between selected and remaining pools.
    - delta: [N, P] for the full (post-dropna) df
    - selected_pos/remaining_pos: positions (0..N-1) in the *same* delta array
    """
    if delta.size == 0:
        return selected_pos, remaining_pos
    p = int(delta.shape[1])
    min_req = max(1, int(min_per_sign))

    sel = set(selected_pos)
    rem = list(remaining_pos)

    def _counts(indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
        d = delta[np.array(indices, dtype=np.int64)]
        pos = (d > eps).sum(axis=0)
        neg = (d < -eps).sum(axis=0)
        return pos.astype(np.int64), neg.astype(np.int64)

    def _can_remove(i: int, pos: np.ndarray, neg: np.ndarray) -> bool:
        row = delta[i]
        for j in range(p):
            if row[j] > eps and pos[j] <= min_req:
                return False
            if row[j] < -eps and neg[j] <= min_req:
                return False
        return True

    # Cap min_req per property/sign by what's feasible in the union of pools.
    all_pos = (delta > eps).sum(axis=0)
    all_neg = (delta < -eps).sum(axis=0)
    cap_pos = np.minimum(all_pos, min_req)
    cap_neg = np.minimum(all_neg, min_req)

    swaps = 0
    for _ in range(10_000):  # hard bound
        sel_list = sorted(sel)
        pos, neg = _counts(sel_list)
        miss_j = None
        miss_sign: Sign | None = None
        for j in range(p):
            if pos[j] < cap_pos[j]:
                miss_j, miss_sign = j, "pos"
                break
            if neg[j] < cap_neg[j]:
                miss_j, miss_sign = j, "neg"
                break
        if miss_j is None:
            break

        if miss_sign == "pos":
            cand_add = next((i for i in rem if delta[i, miss_j] > eps), None)
        else:
            cand_add = next((i for i in rem if delta[i, miss_j] < -eps), None)
        if cand_add is None:
            notes.append(f"cannot_satisfy_sign:p{miss_j}:{miss_sign}")
            # mark this constraint as infeasible by setting cap to current
            if miss_sign == "pos":
                cap_pos[miss_j] = pos[miss_j]
            else:
                cap_neg[miss_j] = neg[miss_j]
            continue

        # remove the lowest-impact row (prefer those not contributing to missing sign)
        remove_candidates = [i for i in sel_list if _can_remove(i, pos, neg)]
        if not remove_candidates:
            notes.append("cannot_find_removal_candidate_for_signs")
            break
        cand_remove = None
        if miss_sign == "pos":
            cand_remove = next((i for i in remove_candidates if not (delta[i, miss_j] > eps)), None)
        else:
            cand_remove = next((i for i in remove_candidates if not (delta[i, miss_j] < -eps)), None)
        if cand_remove is None:
            cand_remove = remove_candidates[0]

        sel.remove(cand_remove)
        sel.add(cand_add)
        rem.remove(cand_add)
        rem.append(cand_remove)
        swaps += 1

    if swaps:
        notes.append(f"sign_swaps={swaps}")
    return sorted(sel), rem


def filter_topk_swap(
    df: pd.DataFrame,
    *,
    property_names: list[str],
    keep_ratio: float,
    eps: float,
    drop_nan: bool,
    min_per_sign: int,
    top_delta_quantile: float = 0.9,
    top_delta_filter: bool = True,
    edits_col: str = "edits_json",
    start_smiles_tagged_col: str = "start_smiles_tagged",
    balance_ops: bool = True,
    min_op_fraction: float = 0.10,
    max_op_fraction: float = 0.85,
    balance_atoms: bool = True,
    min_atom_fraction: float = 0.05,
    max_atom_fraction: float = 0.95,
    max_swaps: int = 5000,
) -> FilterResult:
    notes: list[str] = []
    delta_cols = _delta_cols(df, property_names)

    dropped_nan_rows = 0
    if drop_nan:
        before = len(df)
        df = df.dropna(subset=delta_cols).reset_index(drop=True)
        dropped_nan_rows = before - len(df)
        if dropped_nan_rows:
            notes.append(f"dropped_nan_rows={dropped_nan_rows}")

    if len(df) == 0:
        raise ValueError("No rows left after dropping NaNs")

    if bool(top_delta_filter):
        before = len(df)
        df, dropped = _prefilter_top_delta(df, delta_cols=delta_cols, quantile=float(top_delta_quantile))
        if dropped:
            notes.append(f"top_delta_filter:q={float(top_delta_quantile):.3f} dropped={dropped} kept={len(df)}/{before}")
        if len(df) == 0:
            raise ValueError("No rows left after top-delta filter")

    keep_n = int(math.ceil(len(df) * keep_ratio))
    keep_n = max(1, min(keep_n, len(df)))

    score = _score(df, delta_cols)
    order = score.sort_values(ascending=False).index.tolist()

    selected = set(order[:keep_n])
    remaining = [i for i in order[keep_n:]]

    # Determine minimum required counts per sign per property.
    min_req = max(1, int(min_per_sign))

    def counts(sel: set[int]) -> dict[str, dict[Sign, int]]:
        out: dict[str, dict[Sign, int]] = {}
        for c in delta_cols:
            pos_mask, neg_mask = _sign_mask(df.loc[list(sel)], c, eps=eps)
            out[c] = {"pos": int(pos_mask.sum()), "neg": int(neg_mask.sum())}
        return out

    def missing(cnt: dict[str, dict[Sign, int]]) -> list[tuple[str, Sign]]:
        miss: list[tuple[str, Sign]] = []
        for c in delta_cols:
            if cnt[c]["pos"] < min_req:
                miss.append((c, "pos"))
            if cnt[c]["neg"] < min_req:
                miss.append((c, "neg"))
        return miss

    def can_remove(idx: int, sel: set[int], cnt: dict[str, dict[Sign, int]]) -> bool:
        # idx must not be critical for satisfying min_req on any property/sign.
        row = df.loc[idx, delta_cols]
        for c in delta_cols:
            v = float(row[c])
            if v > eps and cnt[c]["pos"] <= min_req:
                return False
            if v < -eps and cnt[c]["neg"] <= min_req:
                return False
        return True

    cnt = counts(selected)
    miss = missing(cnt)

    # Try to repair missing sign coverage by swapping in rows from the remaining pool.
    while miss and remaining:
        c, s = miss[0]
        if s == "pos":
            candidates = [i for i in remaining if float(df.loc[i, c]) > eps]
        else:
            candidates = [i for i in remaining if float(df.loc[i, c]) < -eps]
        if not candidates:
            notes.append(f"cannot_satisfy:{c}:{s}")
            miss = miss[1:]
            continue

        add_idx = candidates[0]  # highest score due to remaining order
        # choose a removable low-score element from selected
        selected_sorted_low = sorted(selected, key=lambda i: float(score.loc[i]))
        remove_idx = next((i for i in selected_sorted_low if can_remove(i, selected, cnt)), None)
        if remove_idx is None:
            notes.append("cannot_find_removal_candidate")
            break

        # apply swap
        selected.remove(remove_idx)
        selected.add(add_idx)
        remaining.remove(add_idx)

        cnt = counts(selected)
        miss = missing(cnt)

    if miss:
        notes.append(f"unresolved_missing={miss}")

    # Optional: balance op distribution and anchor-atom bins for the kept set.
    if balance_ops or balance_atoms:
        if edits_col not in df.columns:
            notes.append(f"missing_column:{edits_col} (skip op/atom balancing)")
        elif balance_atoms and start_smiles_tagged_col not in df.columns:
            notes.append(f"missing_column:{start_smiles_tagged_col} (skip atom balancing)")
        else:
            # Precompute row-level contributions.
            op_names: list[OpName] = ["add", "remove", "replace"]
            row_op_counts: list[Counter[str]] = []
            row_atom_counts: list[Counter[str]] = []

            if balance_atoms:
                try:
                    _require_rdkit()
                except Exception as e:
                    notes.append(f"skip_atom_balance:{type(e).__name__}")
                    balance_atoms = False

            for i in range(len(df)):
                edits = _safe_json_loads(df.at[i, edits_col])
                c_ops: Counter[str] = Counter()
                c_atoms: Counter[str] = Counter()
                if isinstance(edits, list):
                    if balance_atoms:
                        amap_to_bin = _atom_map_to_bin(str(df.at[i, start_smiles_tagged_col]))
                    else:
                        amap_to_bin = {}
                    for e in edits:
                        if not isinstance(e, dict):
                            continue
                        op = e.get("op")
                        if op in op_names:
                            c_ops[str(op)] += 1
                        anchor_map = e.get("anchor_atom_map")
                        try:
                            anchor_map_i = int(anchor_map)
                        except Exception:
                            anchor_map_i = -1
                        if balance_atoms and anchor_map_i > 0:
                            b = amap_to_bin.get(anchor_map_i)
                            if b:
                                c_atoms[b] += 1
                row_op_counts.append(c_ops)
                row_atom_counts.append(c_atoms)

            def _sum_counts(sel: set[int], rows: list[Counter[str]]) -> Counter[str]:
                out: Counter[str] = Counter()
                for idx in sel:
                    out.update(rows[idx])
                return out

            def _total_edits(ops: Counter[str]) -> int:
                return int(sum(ops.values()))

            full_ops = _sum_counts(set(range(len(df))), row_op_counts)
            full_atoms = _sum_counts(set(range(len(df))), row_atom_counts) if balance_atoms else Counter()

            swaps = 0
            while swaps < max_swaps:
                ops_cnt = _sum_counts(selected, row_op_counts) if balance_ops else Counter()
                atoms_cnt = _sum_counts(selected, row_atom_counts) if balance_atoms else Counter()
                total_edits = _total_edits(ops_cnt) if balance_ops else _total_edits(atoms_cnt)
                if total_edits <= 0:
                    notes.append("no_edits_in_selected (skip op/atom balancing)")
                    break

                # Build min/max targets only for keys that exist in the full dataset.
                op_min: dict[str, int] = {}
                op_max: dict[str, int] = {}
                if balance_ops:
                    for op in op_names:
                        if full_ops.get(op, 0) <= 0:
                            continue
                        desired_min = max(1, int(math.ceil(min_op_fraction * total_edits)))
                        feasible_min = min(int(full_ops.get(op, 0)), desired_min)
                        if feasible_min < desired_min:
                            notes.append(f"cap_min_op:{op}={feasible_min}<desired={desired_min}")
                        op_min[op] = feasible_min

                        desired_max = int(math.floor(max_op_fraction * total_edits))
                        op_max[op] = desired_max
                        op_max[op] = max(op_max[op], op_min[op])

                atom_bins = ["C", "N", "O", "other"]
                atom_min: dict[str, int] = {}
                atom_max: dict[str, int] = {}
                if balance_atoms:
                    for b in atom_bins:
                        if full_atoms.get(b, 0) <= 0:
                            continue
                        desired_min = max(1, int(math.ceil(min_atom_fraction * total_edits)))
                        feasible_min = min(int(full_atoms.get(b, 0)), desired_min)
                        if feasible_min < desired_min:
                            notes.append(f"cap_min_atom:{b}={feasible_min}<desired={desired_min}")
                        atom_min[b] = feasible_min

                        desired_max = int(math.floor(max_atom_fraction * total_edits))
                        atom_max[b] = desired_max
                        atom_max[b] = max(atom_max[b], atom_min[b])

                def _can_remove_idx(idx: int, sel: set[int], sign_cnt: dict[str, dict[Sign, int]]) -> bool:
                    # Preserve property sign mins.
                    row = df.loc[idx, delta_cols]
                    for c in delta_cols:
                        v = float(row[c])
                        if v > eps and sign_cnt[c]["pos"] <= min_req:
                            return False
                        if v < -eps and sign_cnt[c]["neg"] <= min_req:
                            return False

                    # Preserve op/atom mins.
                    if balance_ops:
                        for op, mn in op_min.items():
                            if row_op_counts[idx].get(op, 0) > 0 and ops_cnt.get(op, 0) - row_op_counts[idx].get(op, 0) < mn:
                                return False
                    if balance_atoms:
                        for b, mn in atom_min.items():
                            if row_atom_counts[idx].get(b, 0) > 0 and atoms_cnt.get(b, 0) - row_atom_counts[idx].get(b, 0) < mn:
                                return False
                    return True

                sign_cnt = counts(selected)

                deficits: list[tuple[str, int, str]] = []
                # (kind, deficit, key)
                if balance_ops:
                    for op, mn in op_min.items():
                        d = mn - int(ops_cnt.get(op, 0))
                        if d > 0:
                            deficits.append(("op", d, op))
                if balance_atoms:
                    for b, mn in atom_min.items():
                        d = mn - int(atoms_cnt.get(b, 0))
                        if d > 0:
                            deficits.append(("atom", d, b))
                deficits.sort(key=lambda t: t[1], reverse=True)

                excesses: list[tuple[str, int, str]] = []
                if balance_ops:
                    for op, mx in op_max.items():
                        e = int(ops_cnt.get(op, 0)) - mx
                        if e > 0:
                            excesses.append(("op", e, op))
                if balance_atoms:
                    for b, mx in atom_max.items():
                        e = int(atoms_cnt.get(b, 0)) - mx
                        if e > 0:
                            excesses.append(("atom", e, b))
                excesses.sort(key=lambda t: t[1], reverse=True)

                if not deficits and not excesses:
                    break

                # Helper lists sorted by score.
                selected_low = sorted(selected, key=lambda i: float(score.loc[i]))
                # remaining is already in descending score order.

                did_swap = False

                # Fix deficits first.
                if deficits:
                    kind, _, key = deficits[0]
                    if kind == "op":
                        add_candidates = [i for i in remaining if row_op_counts[i].get(key, 0) > 0]
                    else:
                        add_candidates = [i for i in remaining if row_atom_counts[i].get(key, 0) > 0]
                    if add_candidates:
                        add_idx = add_candidates[0]

                        # Remove a low-score row that does NOT contribute to the deficient key if possible.
                        def _contrib(idx: int) -> int:
                            if kind == "op":
                                return int(row_op_counts[idx].get(key, 0))
                            return int(row_atom_counts[idx].get(key, 0))

                        remove_idx = None
                        for i in selected_low:
                            if _contrib(i) != 0:
                                continue
                            if _can_remove_idx(i, selected, sign_cnt):
                                remove_idx = i
                                break
                        if remove_idx is None:
                            for i in selected_low:
                                if _can_remove_idx(i, selected, sign_cnt):
                                    remove_idx = i
                                    break

                        if remove_idx is not None:
                            selected.remove(remove_idx)
                            selected.add(add_idx)
                            remaining.remove(add_idx)
                            did_swap = True

                # If still needed, try to fix the most overrepresented bin.
                if not did_swap and excesses:
                    kind, _, key = excesses[0]
                    if kind == "op":
                        remove_candidates = [i for i in selected_low if row_op_counts[i].get(key, 0) > 0]
                    else:
                        remove_candidates = [i for i in selected_low if row_atom_counts[i].get(key, 0) > 0]

                    remove_idx = next((i for i in remove_candidates if _can_remove_idx(i, selected, sign_cnt)), None)
                    if remove_idx is not None:
                        # Add a high-score row that contributes less to the overrepresented key.
                        def _contrib(idx: int) -> int:
                            if kind == "op":
                                return int(row_op_counts[idx].get(key, 0))
                            return int(row_atom_counts[idx].get(key, 0))

                        rm_contrib = _contrib(remove_idx)
                        add_candidates = [i for i in remaining if _contrib(i) < rm_contrib]
                        if not add_candidates:
                            add_candidates = list(remaining)
                        if add_candidates:
                            add_idx = add_candidates[0]
                            selected.remove(remove_idx)
                            selected.add(add_idx)
                            remaining.remove(add_idx)
                            did_swap = True

                if not did_swap:
                    notes.append("op_atom_balance_stuck")
                    break

                swaps += 1

            if swaps:
                notes.append(f"op_atom_swaps={swaps}")

    keep_indices = sorted(selected)
    out_df = df.loc[keep_indices].reset_index(drop=True)
    return FilterResult(df=out_df, keep_indices=keep_indices, dropped_nan_rows=dropped_nan_rows, notes=notes)


def filter_topk_cluster(
    df: pd.DataFrame,
    *,
    property_names: list[str],
    keep_ratio: float,
    candidate_ratio: float,
    eps: float,
    drop_nan: bool,
    min_per_sign: int,
    top_delta_quantile: float = 0.9,
    top_delta_filter: bool = True,
    edits_col: str = "edits_json",
    start_smiles_tagged_col: str = "start_smiles_tagged",
    use_ops: bool = True,
    use_atoms: bool = True,
    cluster_k: int = 256,
    cluster_steps: int = 2000,
    cluster_batch_size: int = 512,
    seed: int = 42,
) -> FilterResult:
    notes: list[str] = []
    delta_cols = _delta_cols(df, property_names)

    dropped_nan_rows = 0
    if drop_nan:
        before = len(df)
        df = df.dropna(subset=delta_cols).reset_index(drop=True)
        dropped_nan_rows = before - len(df)
        if dropped_nan_rows:
            notes.append(f"dropped_nan_rows={dropped_nan_rows}")

    if len(df) == 0:
        raise ValueError("No rows left after dropping NaNs")

    if bool(top_delta_filter):
        before = len(df)
        df, dropped = _prefilter_top_delta(df, delta_cols=delta_cols, quantile=float(top_delta_quantile))
        if dropped:
            notes.append(f"top_delta_filter:q={float(top_delta_quantile):.3f} dropped={dropped} kept={len(df)}/{before}")
        if len(df) == 0:
            raise ValueError("No rows left after top-delta filter")

    keep_n = int(math.ceil(len(df) * keep_ratio))
    keep_n = max(1, min(keep_n, len(df)))

    candidate_ratio = float(candidate_ratio)
    if not (0.0 < candidate_ratio <= 1.0):
        raise ValueError("candidate_ratio must be in (0, 1]")
    candidate_n = int(math.ceil(len(df) * candidate_ratio))
    candidate_n = max(keep_n, min(candidate_n, len(df)))

    # Delta matrix + score for candidate prefilter.
    delta = df[delta_cols].to_numpy(dtype=np.float32, copy=False)
    col_std = np.nanstd(delta, axis=0)
    score = _score_np(delta, col_std=col_std)
    order = np.argsort(-score)
    candidates = order[:candidate_n].tolist()

    # Parse edits once for candidates.
    op_counts, atom_counts, num_edits = _parse_edits_counts(
        df,
        indices=candidates,
        edits_col=edits_col,
        start_smiles_tagged_col=start_smiles_tagged_col,
        use_atoms=bool(use_atoms),
    )

    # Build feature matrix for clustering.
    feats: list[np.ndarray] = []
    # (1) normalized deltas
    delta_cand = delta[np.array(candidates, dtype=np.int64)]
    col_std_safe = np.where(col_std == 0.0, 1.0, col_std).astype(np.float32)
    feats.append(delta_cand / col_std_safe[None, :])
    # (2) delta signs (encourage +/- diversity)
    feats.append((delta_cand > eps).astype(np.float32))
    feats.append((delta_cand < -eps).astype(np.float32))
    # (3) op hist
    denom = np.maximum(1.0, num_edits)[:, None]
    if use_ops:
        feats.append(op_counts / denom)
    # (4) atom hist
    if use_atoms:
        feats.append(atom_counts / denom)
    # (5) num_edits (weak signal)
    feats.append(np.log1p(num_edits).reshape(-1, 1).astype(np.float32))

    x = np.concatenate(feats, axis=1).astype(np.float32, copy=False)
    # Normalize feature scales for clustering.
    x_mean = x.mean(axis=0, keepdims=True)
    x_std = x.std(axis=0, keepdims=True)
    x_std = np.where(x_std == 0.0, 1.0, x_std)
    xz = (x - x_mean) / x_std

    k = int(max(1, min(cluster_k, xz.shape[0], keep_n)))
    centers = _minibatch_kmeans(
        xz,
        k=k,
        steps=int(cluster_steps),
        batch_size=int(cluster_batch_size),
        seed=int(seed),
    )
    labels = _assign_clusters(xz, centers)

    # Select quotas per cluster; within each cluster, take highest-score rows.
    cluster_sizes = np.bincount(labels, minlength=k).astype(np.int64)
    quotas = _quota_by_size(cluster_sizes, keep_n=keep_n, min_each=1)

    # Build per-cluster candidate lists sorted by score desc.
    score_cand = score[np.array(candidates, dtype=np.int64)]
    by_cluster: list[list[int]] = [[] for _ in range(k)]
    for local_pos, cid in enumerate(labels.tolist()):
        by_cluster[int(cid)].append(local_pos)
    for cid in range(k):
        by_cluster[cid].sort(key=lambda lp: float(score_cand[lp]), reverse=True)

    selected_local: list[int] = []
    for cid in range(k):
        q = int(quotas[cid])
        if q <= 0:
            continue
        selected_local.extend(by_cluster[cid][:q])

    # Adjust (rare rounding issues).
    if len(selected_local) > keep_n:
        selected_local.sort(key=lambda lp: float(score_cand[lp]), reverse=True)
        selected_local = selected_local[:keep_n]
    elif len(selected_local) < keep_n:
        # fill from remaining best
        pool = [lp for cid in range(k) for lp in by_cluster[cid] if lp not in set(selected_local)]
        pool.sort(key=lambda lp: float(score_cand[lp]), reverse=True)
        need = keep_n - len(selected_local)
        selected_local.extend(pool[:need])

    selected_pos = sorted({candidates[lp] for lp in selected_local})
    remaining_pos = [i for i in candidates if i not in set(selected_pos)]
    # Also allow swaps from outside the candidate pool if needed.
    remaining_pos.extend([i for i in order[candidate_n:].tolist()])

    # Repair minimal +/- sign coverage (fast, array-based).
    selected_pos, remaining_pos = _repair_min_signs(
        delta,
        selected_pos=selected_pos,
        remaining_pos=remaining_pos,
        eps=float(eps),
        min_per_sign=int(min_per_sign),
        notes=notes,
    )

    keep_indices = selected_pos
    out_df = df.loc[keep_indices].reset_index(drop=True)
    return FilterResult(df=out_df, keep_indices=keep_indices, dropped_nan_rows=dropped_nan_rows, notes=notes)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Filter a computed-properties dataset to keep large-change rows while preserving diversity "
            "(delta props / op types / anchor atom types)."
        )
    )
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--input", type=str, default="data/edit_dataset1000_large_props.csv")
    parser.add_argument("--output", type=str, default="data/gnn_edit_dataset_props_filtered.csv")
    parser.add_argument("--method", type=str, default="cluster", choices=["cluster", "swap"])
    parser.add_argument("--keep_ratio", type=float, default=0.5)
    parser.add_argument(
        "--candidate_ratio",
        type=float,
        default=0.8,
        help="For clustering: take top `candidate_ratio` by |delta| as the candidate pool, then select `keep_ratio` diversely.",
    )
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument(
        "--drop_nan",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop rows with NaN deltas (default: true). Use `--no-drop_nan` to keep NaNs.",
    )
    parser.add_argument("--min_per_sign", type=int, default=1)
    parser.add_argument("--top_delta_quantile", type=float, default=0.9)
    parser.add_argument(
        "--top_delta_filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep rows where any |*_delta| is in the top quantile per property (default: true).",
    )
    parser.add_argument("--edits_col", type=str, default="edits_json")
    parser.add_argument("--start_smiles_tagged_col", type=str, default="start_smiles_tagged")

    parser.add_argument(
        "--use_ops",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include op histogram in clustering features (default: true).",
    )

    parser.add_argument(
        "--use_atoms",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include anchor atom-type histogram in clustering features (default: true; requires RDKit).",
    )

    parser.add_argument("--cluster_k", type=int, default=256)
    parser.add_argument("--cluster_steps", type=int, default=2000)
    parser.add_argument("--cluster_batch_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)

    # Legacy swap-based balancing knobs.
    parser.add_argument("--max_swaps", type=int, default=5000, help="For --method swap only.")
    parser.add_argument("--min_op_fraction", type=float, default=0.10, help="For --method swap only.")
    parser.add_argument("--max_op_fraction", type=float, default=0.85, help="For --method swap only.")
    parser.add_argument("--min_atom_fraction", type=float, default=0.05, help="For --method swap only.")
    parser.add_argument("--max_atom_fraction", type=float, default=0.95, help="For --method swap only.")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    property_names = list(cfg.raw["chemistry"]["property_names"])

    input_path = cfg.resolve_path(args.input)
    output_path = cfg.resolve_path(args.output)

    df = pd.read_csv(input_path, engine="c")
    if str(args.method) == "cluster":
        res = filter_topk_cluster(
            df,
            property_names=property_names,
            keep_ratio=float(args.keep_ratio),
            candidate_ratio=float(args.candidate_ratio),
            eps=float(args.eps),
            drop_nan=bool(args.drop_nan),
            min_per_sign=int(args.min_per_sign),
            top_delta_quantile=float(args.top_delta_quantile),
            top_delta_filter=bool(args.top_delta_filter),
            edits_col=str(args.edits_col),
            start_smiles_tagged_col=str(args.start_smiles_tagged_col),
            use_ops=bool(args.use_ops),
            use_atoms=bool(args.use_atoms),
            cluster_k=int(args.cluster_k),
            cluster_steps=int(args.cluster_steps),
            cluster_batch_size=int(args.cluster_batch_size),
            seed=int(args.seed),
        )
    else:
        res = filter_topk_swap(
            df,
            property_names=property_names,
            keep_ratio=float(args.keep_ratio),
            eps=float(args.eps),
            drop_nan=bool(args.drop_nan),
            min_per_sign=int(args.min_per_sign),
            top_delta_quantile=float(args.top_delta_quantile),
            top_delta_filter=bool(args.top_delta_filter),
            edits_col=str(args.edits_col),
            start_smiles_tagged_col=str(args.start_smiles_tagged_col),
            balance_ops=True,
            min_op_fraction=float(args.min_op_fraction),
            max_op_fraction=float(args.max_op_fraction),
            balance_atoms=True,
            min_atom_fraction=float(args.min_atom_fraction),
            max_atom_fraction=float(args.max_atom_fraction),
            max_swaps=int(args.max_swaps),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    res.df.to_csv(output_path, index=False)
    print(f"Wrote {len(res.df)} rows -> {output_path}")
    if res.notes:
        print("Notes:")
        for n in res.notes:
            print(f"- {n}")


if __name__ == "__main__":
    main()
