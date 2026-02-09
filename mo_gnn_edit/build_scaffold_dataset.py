from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

import pandas as pd

from .chem_utils import (
    atom_idx_by_map,
    ensure_atom_maps,
    mol_from_smiles,
    mol_to_tagged_smiles,
    normalize_bracket_atom_hydrogens,
    smiles_without_atom_maps,
)
from .edits import Edit, attach_functional_group, remove_non_scaffold_branch, replace_non_scaffold_branch

if TYPE_CHECKING:  # pragma: no cover
    from rdkit import Chem


_WORKER_CTX: dict[str, Any] | None = None


@dataclass(frozen=True)
class FunctionalGroups:
    smiles_to_id: dict[str, int]

    @staticmethod
    def load(path: Path) -> "FunctionalGroups":
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict):
            raise ValueError("fg_list_json must be a dict: {fg_smiles: id}")
        smiles_to_id: dict[str, int] = {}
        for k, v in raw.items():
            if not isinstance(k, str) or not isinstance(v, int):
                raise ValueError("fg_list_json keys must be str and values must be int")
            smiles_to_id[k] = v
        if not smiles_to_id:
            raise ValueError("fg_list_json is empty")
        return FunctionalGroups(smiles_to_id=smiles_to_id)

    @property
    def smiles_list(self) -> list[str]:
        return list(self.smiles_to_id.keys())

    def fg_id(self, smiles: str) -> int:
        return self.smiles_to_id[smiles]


def _init_worker(ctx: dict[str, Any]) -> None:
    global _WORKER_CTX
    _WORKER_CTX = ctx
    try:  # pragma: no cover
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
    except Exception:
        pass


def _weighted_choice(rng: random.Random, items: list[tuple[str, float]]) -> str:
    total = sum(w for _, w in items)
    r = rng.random() * total
    upto = 0.0
    for name, weight in items:
        upto += weight
        if upto >= r:
            return name
    return items[-1][0]


def _murcko_scaffold_smiles(mol: "Chem.Mol") -> str:
    from rdkit.Chem.Scaffolds import MurckoScaffold

    scaf = MurckoScaffold.GetScaffoldForMol(mol)
    if scaf is None or scaf.GetNumAtoms() == 0:
        return _acyclic_scaffold_smiles(mol)
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)


def _acyclic_scaffold_smiles(mol: "Chem.Mol") -> str:
    from rdkit import Chem

    rw = Chem.RWMol(Chem.Mol(mol))
    changed = True
    while changed and rw.GetNumAtoms() > 2:
        changed = False
        remove_idx = [
            a.GetIdx()
            for a in rw.GetAtoms()
            if a.GetAtomicNum() > 1 and a.GetDegree() <= 1
        ]
        if not remove_idx:
            break
        for idx in sorted(remove_idx, reverse=True):
            rw.RemoveAtom(int(idx))
        changed = True
    out = rw.GetMol()
    if out.GetNumAtoms() == 0:
        return ""
    return Chem.MolToSmiles(out, isomericSmiles=True, canonical=True)


def _scaffold_atom_indices(mol: "Chem.Mol") -> set[int]:
    from rdkit.Chem.Scaffolds import MurckoScaffold

    scaf = MurckoScaffold.GetScaffoldForMol(mol)
    if scaf is None or scaf.GetNumAtoms() == 0:
        return set()
    match = mol.GetSubstructMatch(scaf)
    if not match:
        return set()
    return set(int(i) for i in match)


def _add_anchor_maps(mol: "Chem.Mol") -> list[int]:
    anchors: list[int] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        amap = atom.GetAtomMapNum()
        if amap <= 0:
            continue
        if atom.GetTotalNumHs() > 0:
            anchors.append(amap)
    return anchors


def _remove_candidates(
    mol: "Chem.Mol", scaffold_idx: set[int]
) -> list[tuple[int, int]]:
    """
    Return (anchor_map, removed_map) pairs where removed is outside scaffold,
    and anchor is its scaffold neighbor.
    """
    pairs: list[tuple[int, int]] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        idx = atom.GetIdx()
        amap = atom.GetAtomMapNum()
        if amap <= 0:
            continue
        if idx in scaffold_idx:
            continue
        scaffold_nbrs = [n for n in atom.GetNeighbors() if n.GetIdx() in scaffold_idx]
        if not scaffold_nbrs:
            continue
        anchor = scaffold_nbrs[0]
        anchor_map = anchor.GetAtomMapNum()
        if anchor_map <= 0:
            continue
        pairs.append((anchor_map, amap))
    return pairs


def generate_one_scaffold_oneshot(
    smiles: str,
    rng: random.Random,
    functional_groups: FunctionalGroups,
    max_attempts_per_edit: int,
    num_edits: int,
    op_weights: dict[str, float],
) -> dict[str, Any]:
    mol0 = normalize_bracket_atom_hydrogens(ensure_atom_maps(mol_from_smiles(smiles)))
    scaffold_idx = _scaffold_atom_indices(mol0)
    scaffold_smiles = _murcko_scaffold_smiles(mol0)

    add_anchors = _add_anchor_maps(mol0)
    remove_pairs = _remove_candidates(mol0, scaffold_idx)

    eligible_any = set(add_anchors) | {a for a, _ in remove_pairs}
    if not eligible_any:
        raise RuntimeError("No eligible anchors for edits in molecule")
    num_edits = min(num_edits, len(eligible_any))

    op_items = [(k, float(v)) for k, v in op_weights.items()]
    used_anchors: set[int] = set()
    used_removed: set[int] = set()
    edits: list[Edit] = []

    start_smiles_tagged = mol_to_tagged_smiles(mol0)
    start_smiles = smiles_without_atom_maps(mol0)

    for step in range(1, num_edits + 1):
        success = False
        for _ in range(max_attempts_per_edit):
            op = _weighted_choice(rng, op_items)
            if op == "add":
                candidates = [a for a in add_anchors if a not in used_anchors and a not in used_removed]
                if not candidates:
                    continue
                anchor_map = rng.choice(candidates)
                fg = rng.choice(functional_groups.smiles_list)
                edits.append(
                    Edit(
                        step=step,
                        anchor_atom_map=anchor_map,
                        op="add",
                        fg_smiles=fg,
                        fg_id=functional_groups.fg_id(fg),
                    )
                )
                used_anchors.add(anchor_map)
                success = True
            elif op in ("remove", "replace"):
                candidates = [
                    (a, r)
                    for a, r in remove_pairs
                    if a not in used_anchors and r not in used_removed and r not in used_anchors
                ]
                if not candidates:
                    continue
                anchor_map, removed_map = rng.choice(candidates)
                if op == "remove":
                    edits.append(
                        Edit(
                            step=step,
                            anchor_atom_map=anchor_map,
                            op="remove",
                            removed_atom_map=removed_map,
                        )
                    )
                else:
                    fg = rng.choice(functional_groups.smiles_list)
                    edits.append(
                        Edit(
                            step=step,
                            anchor_atom_map=anchor_map,
                            op="replace",
                            fg_smiles=fg,
                            fg_id=functional_groups.fg_id(fg),
                            removed_atom_map=removed_map,
                        )
                    )
                used_anchors.add(anchor_map)
                used_removed.add(removed_map)
                success = True
            else:
                raise ValueError(f"Unknown op: {op}")
            if success:
                break
        if not success:
            break

    def _op_rank(op: str) -> int:
        return {"remove": 0, "replace": 1, "add": 2}[op]

    mol = mol0
    edits_applied = sorted(edits, key=lambda x: (_op_rank(x.op), x.anchor_atom_map))
    edits_applied = [
        Edit(
            step=i,
            anchor_atom_map=e.anchor_atom_map,
            op=e.op,
            fg_smiles=e.fg_smiles,
            fg_id=e.fg_id,
            removed_atom_map=e.removed_atom_map,
        )
        for i, e in enumerate(edits_applied, start=1)
    ]

    for e in edits_applied:
        if e.op == "add":
            anchor_idx = atom_idx_by_map(mol, e.anchor_atom_map)
            mol = attach_functional_group(mol, anchor_idx, str(e.fg_smiles))
        elif e.op == "remove":
            mol, _ = remove_non_scaffold_branch(mol, e.anchor_atom_map, int(e.removed_atom_map or -1))
        elif e.op == "replace":
            mol, _ = replace_non_scaffold_branch(
                mol, e.anchor_atom_map, int(e.removed_atom_map or -1), str(e.fg_smiles)
            )
        else:
            raise ValueError(f"Unknown op: {e.op}")

    target_smiles_tagged = mol_to_tagged_smiles(mol)
    target_smiles = smiles_without_atom_maps(mol)

    return {
        "source_smiles": smiles,
        "source_smiles_tagged": start_smiles_tagged,
        "start_smiles_tagged": start_smiles_tagged,
        "start_smiles": start_smiles,
        "target_smiles_tagged": target_smiles_tagged,
        "target_smiles": target_smiles,
        "generated_smiles_tagged": target_smiles_tagged,
        "generated_smiles": target_smiles,
        "num_edits": len(edits_applied),
        "scaffold_smiles": scaffold_smiles,
        "edits_json": json.dumps([e.to_dict() for e in edits_applied], ensure_ascii=False),
    }


def _build_one_task(task: tuple[int, int, str]) -> tuple[dict[str, Any] | None, str | None]:
    assert _WORKER_CTX is not None
    ctx = _WORKER_CTX
    i, sample_idx, smiles = task
    last_err: str | None = None
    for attempt in range(int(ctx["max_attempts_per_row"])):
        row_rng = random.Random(int(ctx["seed"]) + i * 1000003 + sample_idx * 9176 + attempt)
        num_edits = row_rng.randint(int(ctx["num_edits_min"]), int(ctx["num_edits_max"]))
        try:
            row = generate_one_scaffold_oneshot(
                smiles=smiles,
                rng=row_rng,
                functional_groups=ctx["functional_groups"],
                max_attempts_per_edit=int(ctx["max_attempts_per_edit"]),
                num_edits=num_edits,
                op_weights=ctx["op_weights"],
            )
            if int(row.get("num_edits", 0)) < int(ctx["num_edits_min"]):
                continue
            row["source_idx"] = int(i)
            row["sample_idx"] = int(sample_idx)
            return row, None
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            continue
    return None, last_err or "unknown_error"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate scaffold-based edit dataset from SMILES-only CSV.")
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--smiles_col", type=str, default="source_smiles")
    parser.add_argument("--fg_list_json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples_per_mol", type=int, default=5)
    parser.add_argument("--num_edits_min", type=int, default=1)
    parser.add_argument("--num_edits_max", type=int, default=5)
    parser.add_argument("--max_attempts_per_edit", type=int, default=10)
    parser.add_argument("--max_attempts_per_row", type=int, default=10)
    parser.add_argument("--executor", type=str, default="thread", choices=["thread", "process"])
    parser.add_argument("--chunksize", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--skip_failed_rows", action="store_true")
    parser.add_argument("--op_weights", type=str, default='{"add": 0.33, "remove": 0.34, "replace": 0.33}')
    args = parser.parse_args()

    try:  # pragma: no cover
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
    except Exception:
        pass

    op_weights = json.loads(args.op_weights)
    if not isinstance(op_weights, dict):
        raise ValueError("--op_weights must be a JSON dict")

    df = pd.read_csv(args.input_csv, engine="c")
    if args.smiles_col not in df.columns:
        raise KeyError(f"Missing smiles_col={args.smiles_col} in {args.input_csv}. Columns: {list(df.columns)}")

    functional_groups = FunctionalGroups.load(args.fg_list_json)
    smiles_list = df[args.smiles_col].astype(str).tolist()
    tasks: list[tuple[int, int, str]] = []
    for i, smiles in enumerate(smiles_list):
        for sample_idx in range(max(1, int(args.num_samples_per_mol))):
            tasks.append((i, sample_idx, smiles))

    total = len(tasks)

    def _render_progress(done: int, *, start_time: float) -> None:
        elapsed = max(1e-9, time.time() - start_time)
        rate = done / elapsed
        msg = f"build_scaffold_dataset: {done}/{total} ({done/total*100:5.1f}%)  {rate:6.1f} it/s"
        if sys.stdout.isatty():
            sys.stdout.write("\r" + msg)
            sys.stdout.flush()
        else:
            print(msg, flush=True)

    ctx = {
        "seed": int(args.seed),
        "num_edits_min": int(args.num_edits_min),
        "num_edits_max": int(args.num_edits_max),
        "max_attempts_per_edit": int(args.max_attempts_per_edit),
        "max_attempts_per_row": int(args.max_attempts_per_row),
        "op_weights": op_weights,
        "functional_groups": functional_groups,
    }
    _init_worker(ctx)

    if int(args.num_workers) <= 1:
        start = time.time()
        last = 0.0
        out_rows = []
        print(f"build_scaffold_dataset: total={total} workers=1 executor=thread", flush=True)
        for idx, task in enumerate(tasks, start=1):
            row, err = _build_one_task(task)
            if row is not None:
                out_rows.append(row)
            elif not args.skip_failed_rows:
                raise RuntimeError(f"Failed to build dataset for task={task}: {err}")
            now = time.time()
            if idx == total or (now - last) >= 0.2:
                _render_progress(idx, start_time=start)
                last = now
        sys.stdout.write("\n")
    else:
        start = time.time()
        last = 0.0
        done = 0
        out_rows = []
        print(
            f"build_scaffold_dataset: total={total} workers={args.num_workers} executor={args.executor}",
            flush=True,
        )
        if args.executor == "process":
            import multiprocessing as mp

            ctx_mp = mp.get_context("spawn")
            with ctx_mp.Pool(processes=int(args.num_workers), initializer=_init_worker, initargs=(ctx,)) as pool:
                for row, err in pool.imap_unordered(_build_one_task, tasks, chunksize=max(1, int(args.chunksize))):
                    done += 1
                    if row is not None:
                        out_rows.append(row)
                    elif not args.skip_failed_rows:
                        raise RuntimeError(f"Failed to build dataset for a task: {err}")
                    now = time.time()
                    if done == total or (now - last) >= 0.2:
                        _render_progress(done, start_time=start)
                        last = now
        else:
            with ThreadPoolExecutor(max_workers=int(args.num_workers), initializer=_init_worker, initargs=(ctx,)) as ex:
                for row, err in ex.map(_build_one_task, tasks):
                    done += 1
                    if row is not None:
                        out_rows.append(row)
                    elif not args.skip_failed_rows:
                        raise RuntimeError(f"Failed to build dataset for a task: {err}")
                    now = time.time()
                    if done == total or (now - last) >= 0.2:
                        _render_progress(done, start_time=start)
                        last = now
        sys.stdout.write("\n")

    out_rows.sort(key=lambda r: (int(r.get("source_idx", 0)), int(r.get("sample_idx", 0))))
    out_df = pd.DataFrame(out_rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(out_df)} rows -> {args.output_csv}")


if __name__ == "__main__":
    main()
