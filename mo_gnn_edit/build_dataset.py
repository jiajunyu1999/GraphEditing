from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

import pandas as pd

from .chem_utils import (
    atom_idx_by_map,
    ensure_atom_maps,
    mol_from_smiles,
    mol_to_tagged_smiles,
    normalize_bracket_atom_hydrogens,
    terminal_neighbors,
    smiles_without_atom_maps,
)
from .config import Config
from .edits import Edit, attach_functional_group, remove_terminal_substituent
from .properties import build_property_fns, calc_properties

if TYPE_CHECKING:  # pragma: no cover
    from rdkit import Chem


_WORKER_CTX: dict[str, Any] | None = None


def _init_worker(ctx: dict[str, Any]) -> None:
    global _WORKER_CTX
    # Build non-picklable objects here so this works for both threads and processes.
    if "functional_groups" not in ctx:
        ctx["functional_groups"] = FunctionalGroups(smiles_to_id=dict(ctx["fg_smiles_to_id"]))
    if "property_fns" not in ctx:
        if bool(ctx["include_properties_json"]):
            ctx["property_fns"] = build_property_fns(list(ctx["property_names"]))
        else:
            ctx["property_fns"] = {}
    _WORKER_CTX = ctx
    try:  # pragma: no cover
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
    except Exception:
        pass


def _build_one_task(task: tuple[int, int, str]) -> tuple[dict[str, Any] | None, str | None]:
    assert _WORKER_CTX is not None
    ctx = _WORKER_CTX
    i, sample_idx, smiles = task
    last_err: str | None = None

    for attempt in range(int(ctx["max_attempts_per_row"])):
        row_rng = random.Random(int(ctx["seed"]) + i * 1000003 + sample_idx * 9176 + attempt)
        num_edits = row_rng.randint(int(ctx["num_edits_min"]), int(ctx["num_edits_max"]))
        try:
            row = generate_one_oneshot(
                smiles=smiles,
                rng=row_rng,
                functional_groups=ctx["functional_groups"],
                property_fns=ctx["property_fns"],
                max_attempts_per_edit=int(ctx["max_attempts_per_edit"]),
                num_edits=num_edits,
                op_weights=ctx["op_weights"],
                include_trajectory=bool(ctx["include_trajectory"]),
                include_properties_json=bool(ctx["include_properties_json"]),
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


@dataclass(frozen=True)
class FunctionalGroups:
    smiles_to_id: dict[str, int]

    @staticmethod
    def load(path: Path) -> "FunctionalGroups":
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict):
            raise ValueError("functional_groups_json must be a dict: {fg_smiles: id}")
        smiles_to_id: dict[str, int] = {}
        for k, v in raw.items():
            if not isinstance(k, str) or not isinstance(v, int):
                raise ValueError("functional_groups_json keys must be str and values must be int")
            smiles_to_id[k] = v
        if not smiles_to_id:
            raise ValueError("functional_groups_json is empty")
        return FunctionalGroups(smiles_to_id=smiles_to_id)

    @property
    def smiles_list(self) -> list[str]:
        return list(self.smiles_to_id.keys())

    def fg_id(self, smiles: str) -> int:
        return self.smiles_to_id[smiles]


def _weighted_choice(rng: random.Random, items: list[tuple[str, float]]) -> str:
    total = sum(w for _, w in items)
    r = rng.random() * total
    upto = 0.0
    for name, weight in items:
        upto += weight
        if upto >= r:
            return name
    return items[-1][0]


def generate_one_oneshot(
    smiles: str,
    rng: random.Random,
    functional_groups: FunctionalGroups,
    property_fns: dict[str, Callable[["Chem.Mol"], float]],
    max_attempts_per_edit: int,
    num_edits: int,
    op_weights: dict[str, float],
    include_trajectory: bool,
    include_properties_json: bool,
) -> dict[str, Any]:
    """
    Generate an unordered set of edits that are all defined on the *start* molecule.
    This avoids edit-order dependence and makes training a single-pass multi-label classification problem.
    """
    # Build normalized start mol.
    # Fast path: start from raw SMILES -> atom-mapped mol (avoid SMILES roundtrip).
    mol0 = normalize_bracket_atom_hydrogens(ensure_atom_maps(mol_from_smiles(smiles)))

    start_props = calc_properties(mol0, property_fns) if include_properties_json else {}

    # Pre-compute eligible anchors on the start molecule.
    add_anchors: list[int] = []
    rm_anchors: list[int] = []
    for atom in mol0.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        amap = atom.GetAtomMapNum()
        if amap <= 0:
            continue
        # Conservative add-mask: substitute an implicit H on C/N/O.
        if atom.GetNumImplicitHs() > 0 and atom.GetAtomicNum() in (6, 7, 8):
            if not (atom.GetIsAromatic() and atom.GetAtomicNum() != 6):
                add_anchors.append(amap)
        # Remove/replace mask: anchor has terminal neighbor(s).
        if terminal_neighbors(mol0, atom.GetIdx()):
            rm_anchors.append(amap)

    eligible_any = sorted(set(add_anchors) | set(rm_anchors))
    if not eligible_any:
        raise RuntimeError("No eligible anchors for oneshot edits in molecule")
    num_edits = min(num_edits, len(eligible_any))

    op_items = [(k, float(v)) for k, v in op_weights.items()]
    used_anchors: set[int] = set()
    used_removed: set[int] = set()
    edits: list[Edit] = []
    start_smiles_tagged = mol_to_tagged_smiles(mol0)
    start_smiles = smiles_without_atom_maps(mol0)
    trajectory_tagged: list[str] = [start_smiles_tagged]

    # Select edits (unordered) on the start molecule.
    for step in range(1, num_edits + 1):
        success = False
        last_err: Exception | None = None
        for _ in range(max_attempts_per_edit):
            try:
                op = _weighted_choice(rng, op_items)
                if op == "add":
                    candidates = [a for a in add_anchors if a not in used_anchors and a not in used_removed]
                elif op in ("remove", "replace"):
                    candidates = [a for a in rm_anchors if a not in used_anchors and a not in used_removed]
                else:
                    raise ValueError(f"Unknown op: {op}")
                if not candidates:
                    raise ValueError("No candidates for sampled op")

                anchor_map = rng.choice(candidates)
                anchor_idx0 = atom_idx_by_map(mol0, anchor_map)

                if op == "add":
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
                elif op == "remove":
                    nbrs = terminal_neighbors(mol0, anchor_idx0)
                    if not nbrs:
                        raise ValueError("No terminal neighbor candidates")
                    # pick a terminal neighbor not already removed by another op
                    cand_maps = [
                        mol0.GetAtomWithIdx(i).GetAtomMapNum() for i in nbrs if mol0.GetAtomWithIdx(i).GetAtomMapNum() > 0
                    ]
                    cand_maps = [m for m in cand_maps if m not in used_removed and m not in used_anchors]
                    if not cand_maps:
                        raise ValueError("No unused removable terminal neighbors")
                    removed_map = rng.choice(cand_maps)
                    edits.append(
                        Edit(
                            step=step,
                            anchor_atom_map=anchor_map,
                            op="remove",
                            removed_atom_map=removed_map,
                        )
                    )
                    used_removed.add(removed_map)
                elif op == "replace":
                    nbrs = terminal_neighbors(mol0, anchor_idx0)
                    if not nbrs:
                        raise ValueError("No terminal neighbor candidates")
                    cand_maps = [
                        mol0.GetAtomWithIdx(i).GetAtomMapNum() for i in nbrs if mol0.GetAtomWithIdx(i).GetAtomMapNum() > 0
                    ]
                    cand_maps = [m for m in cand_maps if m not in used_removed and m not in used_anchors]
                    if not cand_maps:
                        raise ValueError("No unused removable terminal neighbors")
                    removed_map = rng.choice(cand_maps)
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
                    used_removed.add(removed_map)

                used_anchors.add(anchor_map)
                success = True
                break
            except Exception as e:
                last_err = e
                continue
        if not success:
            # Too many requested steps can exhaust valid anchors/candidates; stop early.
            break

    # Apply edits in a deterministic order to produce a single generated molecule.
    # Also write `edits_json` in this same order so `validate_dataset.py` can replay.
    def _op_rank(op: str) -> int:
        return {"remove": 0, "replace": 1, "add": 2}[op]

    mol = mol0
    edits_applied = sorted(edits, key=lambda x: (_op_rank(x.op), x.anchor_atom_map))
    # Renumber steps to match the applied order.
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
            mol, _ = remove_terminal_substituent(mol, e.anchor_atom_map, removed_atom_map=int(e.removed_atom_map or -1))
        elif e.op == "replace":
            mol2, _ = remove_terminal_substituent(mol, e.anchor_atom_map, removed_atom_map=int(e.removed_atom_map or -1))
            anchor_idx2 = atom_idx_by_map(mol2, e.anchor_atom_map)
            mol = attach_functional_group(mol2, anchor_idx2, str(e.fg_smiles))
        else:
            raise ValueError(f"Unknown op: {e.op}")
        if include_trajectory:
            trajectory_tagged.append(mol_to_tagged_smiles(mol))

    target_props = calc_properties(mol, property_fns) if include_properties_json else {}

    target_smiles_tagged = mol_to_tagged_smiles(mol)
    target_smiles = smiles_without_atom_maps(mol)

    row: dict[str, Any] = {
        "source_smiles": smiles,
        "source_smiles_tagged": start_smiles_tagged,
        "start_smiles_tagged": start_smiles_tagged,
        "start_smiles": start_smiles,
        "target_smiles_tagged": target_smiles_tagged,
        "target_smiles": target_smiles,
        "generated_smiles_tagged": target_smiles_tagged,
        "generated_smiles": target_smiles,
        "num_edits": len(edits_applied),
        # order is not used for training; keep as-is for reproducibility/debugging
        "edits_json": json.dumps([e.to_dict() for e in edits_applied], ensure_ascii=False),
    }
    if include_properties_json:
        row["start_props_json"] = json.dumps(start_props, ensure_ascii=False)
        row["target_props_json"] = json.dumps(target_props, ensure_ascii=False)
    if include_trajectory:
        row["trajectory_tagged_json"] = json.dumps(trajectory_tagged, ensure_ascii=False)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate random molecule-edit dataset from a SMILES-only CSV.")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    # Suppress RDKit stderr warnings like "Can't kekulize mol ..."
    try:  # pragma: no cover
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
    except Exception:
        pass

    cfg = Config.load(args.config)
    input_csv = cfg.resolve_path(cfg.raw["input"]["csv_path"])
    output_csv = cfg.resolve_path(cfg.raw["output"]["csv_path"])
    include_trajectory = bool(cfg.raw["output"].get("include_trajectory", False))
    include_properties_json = bool(cfg.raw["output"].get("include_properties_json", True))

    seed = int(cfg.raw["random"]["seed"])
    num_samples_per_mol = int(cfg.raw["random"].get("num_samples_per_mol", 1))
    num_edits_min = int(cfg.raw["random"]["num_edits_min"])
    num_edits_max = int(cfg.raw["random"]["num_edits_max"])
    max_attempts_per_edit = int(cfg.raw["random"]["max_attempts_per_edit"])
    max_attempts_per_row = int(cfg.raw["random"].get("max_attempts_per_row", 1))
    executor = str(cfg.raw["random"].get("executor", "thread"))
    chunksize = int(cfg.raw["random"].get("chunksize", 20))
    num_workers = int(cfg.raw["random"].get("num_workers", 1))
    skip_failed_rows = bool(cfg.raw["random"].get("skip_failed_rows", False))
    op_weights = dict(cfg.raw["random"]["op_weights"])

    smiles_col = str(cfg.raw["input"]["smiles_col"])
    functional_groups_path = cfg.resolve_path(cfg.raw["chemistry"]["functional_groups_json"])
    property_names = list(cfg.raw["chemistry"]["property_names"])

    df = pd.read_csv(input_csv, engine="c")
    if smiles_col not in df.columns:
        raise KeyError(f"Missing smiles_col={smiles_col} in {input_csv}. Columns: {list(df.columns)}")

    functional_groups = FunctionalGroups.load(functional_groups_path)
    property_fns = build_property_fns(property_names) if include_properties_json else {}

    smiles_list = df[smiles_col].astype(str).tolist()
    tasks: list[tuple[int, int, str]] = []
    for i, smiles in enumerate(smiles_list):
        for sample_idx in range(max(1, num_samples_per_mol)):
            tasks.append((i, sample_idx, smiles))

    total = len(tasks)

    def _render_progress(done: int, *, start_time: float) -> None:
        elapsed = max(1e-9, time.time() - start_time)
        rate = done / elapsed
        msg = f"build_dataset: {done}/{total} ({done/total*100:5.1f}%)  {rate:6.1f} it/s"
        if sys.stdout.isatty():
            sys.stdout.write("\r" + msg)
            sys.stdout.flush()
        else:
            print(msg, flush=True)

    ctx = {
        "seed": seed,
        "num_edits_min": num_edits_min,
        "num_edits_max": num_edits_max,
        "max_attempts_per_edit": max_attempts_per_edit,
        "max_attempts_per_row": max_attempts_per_row,
        "include_trajectory": include_trajectory,
        "include_properties_json": include_properties_json,
        "op_weights": op_weights,
        "property_names": property_names,
        "fg_smiles_to_id": functional_groups.smiles_to_id,
    }
    # Fast path for threads: reuse already-built objects to avoid rebuilding in each thread.
    ctx_thread = dict(ctx)
    ctx_thread["functional_groups"] = functional_groups
    ctx_thread["property_fns"] = property_fns
    _init_worker(ctx_thread)

    if num_workers <= 1:
        start = time.time()
        last = 0.0
        out_rows = []
        print(f"build_dataset: total={total} workers=1 executor=thread", flush=True)
        for idx, task in enumerate(tasks, start=1):
            row, err = _build_one_task(task)
            if row is not None:
                out_rows.append(row)
            elif not skip_failed_rows:
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
        print(f"build_dataset: total={total} workers={num_workers} executor={executor}", flush=True)
        if executor == "process":
            import multiprocessing as mp

            # Avoid ProcessPoolExecutor sysconf checks in sandbox; use multiprocessing Pool instead.
            ctx_mp = mp.get_context("spawn")
            with ctx_mp.Pool(processes=num_workers, initializer=_init_worker, initargs=(ctx,)) as pool:
                for row, err in pool.imap_unordered(_build_one_task, tasks, chunksize=max(1, chunksize)):
                    done += 1
                    if row is not None:
                        out_rows.append(row)
                    elif not skip_failed_rows:
                        raise RuntimeError(f"Failed to build dataset for a task: {err}")
                    now = time.time()
                    if done == total or (now - last) >= 0.2:
                        _render_progress(done, start_time=start)
                        last = now
        else:
            with ThreadPoolExecutor(max_workers=num_workers, initializer=_init_worker, initargs=(ctx_thread,)) as ex:
                for row, err in ex.map(_build_one_task, tasks):
                    done += 1
                    if row is not None:
                        out_rows.append(row)
                    elif not skip_failed_rows:
                        raise RuntimeError(f"Failed to build dataset for a task: {err}")
                    now = time.time()
                    if done == total or (now - last) >= 0.2:
                        _render_progress(done, start_time=start)
                        last = now
        sys.stdout.write("\n")

    # Stable output order
    out_rows.sort(key=lambda r: (int(r.get("source_idx", 0)), int(r.get("sample_idx", 0))))

    out_df = pd.DataFrame(out_rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"Wrote {len(out_df)} rows -> {output_csv}")


if __name__ == "__main__":
    main()
