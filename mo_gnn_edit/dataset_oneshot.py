from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
import os
import time
from pathlib import Path
from typing import Any

import torch

from .featurize import featurize_tagged_smiles
from .pyg_utils import require_torch_geometric
from .chem_utils import ensure_atom_maps, mol_from_smiles, normalize_bracket_atom_hydrogens
from .scaffold_utils import scaffold_mask

# Keep replace (4-way op classification).
OP_TO_ID: dict[str, int] = {"none": 0, "add": 1, "remove": 2, "replace": 3}
ID_TO_OP: dict[int, str] = {v: k for k, v in OP_TO_ID.items()}


@dataclass(frozen=True)
class OneShotRow:
    start_smiles_tagged: str
    edits_json: str
    props: torch.Tensor  # [1, num_props] delta-only
    props_src: torch.Tensor  # [1, num_props] source values aligned with props
    task_id: str | None


class OneShotEditDataset(torch.utils.data.Dataset):
    """
    One sample = one start graph + a *set* of edits (order ignored).

    Labels are per-node:
      - y_op: 0 none, 1 add, 2 remove, 3 replace
      - y_fg: fg_id for add/replace else -1
      - y_removed_atom_map: removed atom-map for remove/replace else -1
    """

    def __init__(
        self,
        csv_path: str,
        property_names: list[str],
        *,
        seed: int = 42,
        prop_mask_by_task: dict[str, list[int]] | None = None,
        mask_non_task_props: bool = False,
    ):
        self.csv_path = csv_path
        self.property_names = property_names
        self.seed = int(seed)
        self.prop_mask_by_task = prop_mask_by_task or {}
        self.mask_non_task_props = bool(mask_non_task_props)

        with Path(csv_path).open("r", newline="") as f:
            reader = csv.DictReader(f)
            header = list(reader.fieldnames or [])

            required = {"start_smiles_tagged", "edits_json"}
            missing = required - set(header)
            if missing:
                raise KeyError(f"Missing columns in {csv_path}: {sorted(missing)}")

            # Property conditioning vector.
            # Simplified: only use deltas per property, ordered by property_names.
            self.prop_cols = [f"{p}_delta" for p in property_names if f"{p}_delta" in header]
            # Optional: matching source-property columns aligned with prop_cols.
            self.prop_src_cols: list[str | None] = []
            for c in self.prop_cols:
                base = c[: -len("_delta")] if c.endswith("_delta") else c
                src_c = f"{base}_src"
                self.prop_src_cols.append(src_c if src_c in header else None)

            def _safe_float(x: Any) -> float:
                try:
                    v = float(x)
                    if v != v or v in (float("inf"), float("-inf")):
                        return 0.0
                    return v
                except Exception:
                    return 0.0

            self.rows = []
            max_task_id = -1
            for row in reader:
                if not row:
                    continue
                task_id = str(row.get("task_id", "")).strip() or None
                if task_id and task_id.isdigit():
                    max_task_id = max(max_task_id, int(task_id))
                if self.prop_cols:
                    props = torch.tensor([_safe_float(row.get(c, 0.0)) for c in self.prop_cols], dtype=torch.float32)
                    props = props.unsqueeze(0)
                    props = torch.nan_to_num(props, nan=0.0, posinf=0.0, neginf=0.0)
                    props_src = torch.tensor(
                        [_safe_float(row.get(c, 0.0)) if c else 0.0 for c in self.prop_src_cols],
                        dtype=torch.float32,
                    )
                    props_src = props_src.unsqueeze(0)
                    props_src = torch.nan_to_num(props_src, nan=0.0, posinf=0.0, neginf=0.0)
                else:
                    props = torch.empty((1, 0), dtype=torch.float32)
                    props_src = torch.empty((1, 0), dtype=torch.float32)
                self.rows.append(
                    OneShotRow(
                        start_smiles_tagged=str(row.get("start_smiles_tagged", "")),
                        edits_json=str(row.get("edits_json", "")),
                        props=props,
                        props_src=props_src,
                        task_id=task_id,
                    )
                )
            self.max_task_id = int(max_task_id)
        # Shuffle row order deterministically to avoid any CSV ordering artifacts.
        rng = random.Random(self.seed)
        rng.shuffle(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        require_torch_geometric()
        from torch_geometric.data import Data

        t0 = time.perf_counter()
        r = self.rows[idx]
        t1 = time.perf_counter()
        f = featurize_tagged_smiles(r.start_smiles_tagged)
        t2 = time.perf_counter()

        mol = normalize_bracket_atom_hydrogens(ensure_atom_maps(mol_from_smiles(r.start_smiles_tagged)))
        t3 = time.perf_counter()
        scaffold = scaffold_mask(mol)
        t4 = time.perf_counter()
        map_to_idx = {int(m.item()): i for i, m in enumerate(f.atom_map)}
        num_nodes = int(f.z.size(0))

        add_allowed = torch.tensor(
            [a.GetAtomicNum() > 1 and int(a.GetTotalNumHs()) > 0 for a in mol.GetAtoms()],
            dtype=torch.bool,
        )
        remove_allowed = torch.zeros((num_nodes,), dtype=torch.bool)
        if int(f.edge_index.numel()) > 0 and int(scaffold.numel()) == num_nodes:
            src = f.edge_index[0]
            dst = f.edge_index[1]
            mask_edges = scaffold[src] & (~scaffold[dst])
            if bool(mask_edges.any().item()):
                remove_allowed.scatter_(0, src[mask_edges], True)
        t5 = time.perf_counter()

        op_allowed = torch.zeros((num_nodes, 4), dtype=torch.bool)
        op_allowed[:, OP_TO_ID["none"]] = True
        op_allowed[:, OP_TO_ID["add"]] = add_allowed
        op_allowed[:, OP_TO_ID["remove"]] = remove_allowed
        op_allowed[:, OP_TO_ID["replace"]] = add_allowed & remove_allowed
        anchor_mask = op_allowed[:, 1:].any(dim=-1)

        y_op = torch.zeros((num_nodes,), dtype=torch.long)  # none
        y_fg = torch.full((num_nodes,), -1, dtype=torch.long)
        y_removed_atom_map = torch.full((num_nodes,), -1, dtype=torch.long)

        edits = json.loads(r.edits_json)
        if not isinstance(edits, list):
            raise ValueError("edits_json must be a list")
        for e in edits:
            op = str(e["op"])
            anchor_map = int(e["anchor_atom_map"])
            if anchor_map not in map_to_idx:
                raise KeyError(
                    f"anchor_atom_map={anchor_map} not found in start_smiles_tagged. "
                    "Use random.mode=oneshot to avoid anchors on newly-added atoms."
                )
            anchor_idx = map_to_idx[anchor_map]
            if op not in OP_TO_ID:
                raise ValueError(f"Unsupported op={op!r} in edits_json. Supported: {sorted(OP_TO_ID)}.")
            y_op[anchor_idx] = OP_TO_ID[op]
            if op in ("add", "replace"):
                y_fg[anchor_idx] = int(e["fg_id"])
            if op in ("remove", "replace"):
                removed_map = e.get("removed_atom_map")
                if removed_map is not None:
                    y_removed_atom_map[anchor_idx] = int(removed_map)
        t6 = time.perf_counter()

        t7 = t6
        if self.prop_cols:
            props = r.props
            props_src = r.props_src
            props = torch.nan_to_num(props, nan=0.0, posinf=0.0, neginf=0.0)
            props_src = torch.nan_to_num(props_src, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            props = r.props
            props_src = r.props_src
        t7 = time.perf_counter()

        # Optionally zero out deltas for properties not defined for this task_id.
        # This makes training conditioning match task-style inference where unspecified deltas are 0.
        if self.mask_non_task_props and self.prop_cols and r.task_id:
            keep_idxs = self.prop_mask_by_task.get(str(r.task_id))
            if isinstance(keep_idxs, list):
                num_props = int(props.size(-1))
                keep = torch.zeros((num_props,), dtype=torch.bool)
                for j in keep_idxs:
                    try:
                        jj = int(j)
                    except Exception:
                        continue
                    if 0 <= jj < num_props:
                        keep[jj] = True
                if bool((~keep).any().item()):
                    props = props.clone()
                    props[:, ~keep] = 0.0
        t8 = time.perf_counter()

        data = Data(
            z=f.z,
            x=f.x,
            edge_index=f.edge_index,
            edge_type=f.edge_type,
            atom_map=f.atom_map,
            scaffold_mask=scaffold,
            add_allowed=add_allowed,
            remove_allowed=remove_allowed,
            props=props,
            props_src=props_src,
            anchor_mask=anchor_mask,
            op_allowed=op_allowed,
            y_op=y_op,
            y_fg=y_fg,
            y_removed_atom_map=y_removed_atom_map,
            task_id=torch.tensor(int(r.task_id) if (r.task_id and str(r.task_id).isdigit()) else -1, dtype=torch.long),
        )
        t9 = time.perf_counter()

        if os.environ.get("PROFILE_DATASET_TIMING") == "1":
            stats = getattr(self, "_timing_stats", None)
            if stats is None:
                stats = {
                    "n": 0,
                    "row": 0.0,
                    "featurize": 0.0,
                    "mol": 0.0,
                    "scaffold": 0.0,
                    "op_mask": 0.0,
                    "edits": 0.0,
                    "props": 0.0,
                    "mask_props": 0.0,
                    "data": 0.0,
                    "total": 0.0,
                }
                setattr(self, "_timing_stats", stats)
            stats["n"] += 1
            stats["row"] += t1 - t0
            stats["featurize"] += t2 - t1
            stats["mol"] += t3 - t2
            stats["scaffold"] += t4 - t3
            stats["op_mask"] += t5 - t4
            stats["edits"] += t6 - t5
            stats["props"] += t7 - t6
            stats["mask_props"] += t8 - t7
            stats["data"] += t9 - t8
            stats["total"] += t9 - t0
            if stats["n"] % 1000 == 0:
                n = float(stats["n"])
                print(
                    "[dataset_timing] "
                    f"n={int(n)} "
                    f"row={stats['row']/n:.4f}s "
                    f"featurize={stats['featurize']/n:.4f}s "
                    f"mol={stats['mol']/n:.4f}s "
                    f"scaffold={stats['scaffold']/n:.4f}s "
                    f"op_mask={stats['op_mask']/n:.4f}s "
                    f"edits={stats['edits']/n:.4f}s "
                    f"props={stats['props']/n:.4f}s "
                    f"mask_props={stats['mask_props']/n:.4f}s "
                    f"data={stats['data']/n:.4f}s "
                    f"total={stats['total']/n:.4f}s"
                )
        return data
