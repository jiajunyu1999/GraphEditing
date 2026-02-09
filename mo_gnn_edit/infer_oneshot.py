from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import json
import math
import random
import sys
import re
import time
from concurrent.futures import ThreadPoolExecutor
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

from .config import Config
from .chem_utils import (
    ensure_atom_maps,
    mol_from_smiles,
    mol_to_tagged_smiles,
    normalize_bracket_atom_hydrogens,
    smiles_without_atom_maps,
)
from .vocab import FunctionalGroupVocab
from .dataset_oneshot import ID_TO_OP, OP_TO_ID

if TYPE_CHECKING:  # pragma: no cover
    import torch


def _finite_or_zero(x: Any) -> float:
    try:
        v = float(x)
        if v != v or v in (float("inf"), float("-inf")):
            return 0.0
        return v
    except Exception:
        return 0.0


def _infer_moe_from_checkpoint_name(path: str | Path) -> tuple[int | None, int | None]:
    """
    Infer (num_experts, moe_topk) from checkpoint filename.

    Examples:
      base_moe16_4_*.pt -> (16, 4)
      base_moe8_3_*.pt  -> (8, 3)
    """
    name = Path(path).name
    m = re.search(r"moe(\d+)_(\d+)", name)
    if not m:
        return None, None
    try:
        return int(m.group(1)), int(m.group(2))
    except Exception:
        return None, None


def _as_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(int(x))
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("1", "true", "yes", "y", "on"):
            return True
        if s in ("0", "false", "no", "n", "off", ""):
            return False
    raise ValueError(f"Expected boolean-like value, got {x!r}")


def _require_yaml() -> None:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Missing dependency: PyYAML. Install it (e.g. `pip install pyyaml`).") from exc


def _load_yaml(path: str | Path) -> dict[str, Any]:
    _require_yaml()
    import yaml  # type: ignore

    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"YAML root must be a dict, got {type(raw)}")
    return raw


def _disable_rdkit_warnings() -> None:
    try:  # pragma: no cover
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
    except Exception:
        pass


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    props: list[str]
    trend: str  # string of 0/1 aligned with props


@dataclass(frozen=True)
class _Candidate:
    smiles: str
    score: float
    edits: list[dict[str, Any]]
    moe_snapshot: dict[str, Any] | None = None


@dataclass
class _Profiler:
    enabled: bool = False
    sync_cuda: bool = False
    error_topk: int = 20
    max_error_samples: int = 20
    error_maxlen: int = 200
    time_s: dict[str, float] = None  # type: ignore[assignment]
    time_n: dict[str, int] = None  # type: ignore[assignment]
    counts: dict[str, int] = None  # type: ignore[assignment]
    meta: dict[str, Any] = None  # type: ignore[assignment]
    error_msg_counts: dict[str, Counter[str]] = None  # type: ignore[assignment]
    error_samples: list[dict[str, Any]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.time_s = defaultdict(float)
        self.time_n = defaultdict(int)
        self.counts = defaultdict(int)
        self.meta = {}
        self.error_msg_counts = defaultdict(Counter)
        self.error_samples = []

    def inc(self, key: str, n: int = 1) -> None:
        if not self.enabled:
            return
        self.counts[str(key)] += int(n)

    def record_error(self, *, phase: str, exc: Exception, context: dict[str, Any] | None = None) -> None:
        if not self.enabled:
            return
        phase = str(phase)
        exc_type = type(exc).__name__
        msg = str(exc).strip()
        if not msg:
            msg = "<empty>"
        if int(self.error_maxlen) > 0 and len(msg) > int(self.error_maxlen):
            msg = msg[: max(1, int(self.error_maxlen) - 3)] + "..."

        self.inc("reject.exception")
        self.inc(f"reject.exception.{exc_type}")
        self.inc(f"reject.phase.{phase}")
        self.inc(f"reject.phase.{phase}.{exc_type}")
        self.error_msg_counts[exc_type][msg] += 1
        self.error_msg_counts[f"{phase}.{exc_type}"][msg] += 1

        if len(self.error_samples) < int(self.max_error_samples):
            payload: dict[str, Any] = {
                "phase": phase,
                "exc_type": exc_type,
                "exc_msg": msg,
            }
            if isinstance(context, dict):
                for k, v in context.items():
                    payload[str(k)] = v
            self.error_samples.append(payload)

    @contextmanager
    def timer(self, key: str, *, n: int = 1, sync_fn=None):
        if not self.enabled:
            yield
            return
        if sync_fn is not None:
            try:
                sync_fn()
            except Exception:
                pass
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if sync_fn is not None:
                try:
                    sync_fn()
                except Exception:
                    pass
            dt = time.perf_counter() - t0
            self.time_s[str(key)] += float(dt)
            self.time_n[str(key)] += int(n)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "meta": dict(self.meta),
            "timings": {},
            "counts": dict(self.counts),
            "errors": {
                "top_messages": {},
                "samples": list(self.error_samples),
            },
        }
        for k, v in sorted(self.time_s.items(), key=lambda kv: (-float(kv[1]), str(kv[0]))):
            n = int(self.time_n.get(k, 0))
            out["timings"][k] = {"total_s": float(v), "n": int(n), "avg_s": float(v) / float(n) if n > 0 else 0.0}
        topk = max(1, int(self.error_topk))
        for k, counter in self.error_msg_counts.items():
            try:
                out["errors"]["top_messages"][str(k)] = [(m, int(c)) for m, c in counter.most_common(topk)]
            except Exception:
                continue
        return out

    def print_top(self, *, top_k: int = 25) -> None:
        if not self.enabled:
            return
        items = sorted(self.time_s.items(), key=lambda kv: float(kv[1]), reverse=True)[: max(1, int(top_k))]
        print("[profile] top timings:")
        for k, v in items:
            n = int(self.time_n.get(k, 0))
            avg = float(v) / float(n) if n > 0 else 0.0
            print(f"[profile] {k}: total_s={v:.3f} n={n} avg_s={avg:.6f}")


def _load_infer_config(path: str | Path) -> dict[str, Any]:
    raw = _load_yaml(path)
    if not isinstance(raw, dict):
        raise ValueError("infer_config must be a dict")
    return raw


def _load_control_config(path: str | Path) -> dict[str, Any]:
    raw = _load_yaml(path)
    if not isinstance(raw, dict):
        raise ValueError("control_config must be a dict")
    return raw


def _normalize_control_inputs(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    items: list[Any]
    if isinstance(raw, dict):
        if "inputs" in raw:
            items = raw.get("inputs")
        elif "input" in raw:
            items = [raw.get("input")]
        elif "smiles" in raw or "smiles_tagged" in raw:
            items = [raw]
        else:
            raise ValueError("control_config must contain inputs or smiles_tagged/smiles")
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError("control_config must be a dict or list of inputs")
    if not isinstance(items, list):
        raise ValueError("control_config inputs must be a list")
    out: list[dict[str, Any]] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"control input #{i} must be a dict")
        out.append(item)
    return out


def _fg_ids_from_candidates(raw: Any, fg_smiles_to_id: dict[str, int]) -> list[int]:
    if raw is None:
        return []
    values = raw if isinstance(raw, (list, tuple)) else [raw]
    ids: list[int] = []
    for v in values:
        if v is None:
            continue
        if isinstance(v, bool):
            continue
        if isinstance(v, int):
            ids.append(int(v))
            continue
        if isinstance(v, float):
            ids.append(int(v))
            continue
        if isinstance(v, str):
            s = v.strip()
            if not s:
                continue
            if s.isdigit():
                ids.append(int(s))
                continue
            fid = fg_smiles_to_id.get(s)
            if fid is not None:
                ids.append(int(fid))
            continue
        try:
            ids.append(int(v))
        except Exception:
            continue
    return sorted(set(ids))


def _parse_control_spec(item: dict[str, Any], fg_smiles_to_id: dict[str, int]) -> dict[str, Any]:
    spec = item.get("control") if isinstance(item.get("control"), dict) else item
    if not isinstance(spec, dict):
        return {}

    out: dict[str, Any] = {}
    if spec.get("task_id") is not None or spec.get("task") is not None:
        out["task_id"] = str(spec.get("task_id", spec.get("task")))
    delta_raw = spec.get("delta_props", spec.get("delta_prop", spec.get("delta")))
    if isinstance(delta_raw, dict):
        delta_clean: dict[str, float] = {}
        for k, v in delta_raw.items():
            if k is None:
                continue
            key = str(k)
            if key.endswith("_delta"):
                key = key[: -len("_delta")]
            try:
                delta_clean[key] = float(v)
            except Exception:
                continue
        if delta_clean:
            out["delta_props"] = delta_clean
    anchor_raw = (
        spec.get("anchor_atom_maps")
        if spec.get("anchor_atom_maps") is not None
        else spec.get("anchor_maps")
        if spec.get("anchor_maps") is not None
        else spec.get("atom_maps")
        if spec.get("atom_maps") is not None
        else spec.get("region_atom_maps")
    )
    if anchor_raw is not None:
        anchors = anchor_raw if isinstance(anchor_raw, (list, tuple)) else [anchor_raw]
        anchor_ids: list[int] = []
        for v in anchors:
            try:
                anchor_ids.append(int(v))
            except Exception:
                continue
        if anchor_ids:
            out["anchor_atom_maps"] = sorted(set(anchor_ids))
        mode = str(spec.get("anchor_mode", "include") or "include").strip().lower()
        if mode in ("include", "only", "whitelist", "allow"):
            out["anchor_mode"] = "include"
        elif mode in ("exclude", "blacklist", "deny", "except"):
            out["anchor_mode"] = "exclude"
        else:
            raise ValueError(f"Unknown anchor_mode={mode!r}")

    ops_raw = spec.get("op_allow", spec.get("ops"))
    if ops_raw is not None:
        ops = ops_raw if isinstance(ops_raw, (list, tuple)) else [ops_raw]
        allow_ops: list[str] = []
        for op in ops:
            s = str(op).strip().lower()
            if s in ("add", "remove", "replace", "none"):
                if s not in allow_ops:
                    allow_ops.append(s)
        if allow_ops:
            out["op_allow"] = allow_ops

    fg_candidates = spec.get("fg_candidates")
    if fg_candidates is None:
        fg_ids_raw = spec.get("fg_ids")
        fg_smiles_raw = spec.get("fg_smiles")
        if fg_ids_raw is not None or fg_smiles_raw is not None:
            fg_candidates = []
            if fg_ids_raw is not None:
                fg_candidates.extend(fg_ids_raw if isinstance(fg_ids_raw, (list, tuple)) else [fg_ids_raw])
            if fg_smiles_raw is not None:
                fg_candidates.extend(fg_smiles_raw if isinstance(fg_smiles_raw, (list, tuple)) else [fg_smiles_raw])

    if isinstance(fg_candidates, dict):
        fg_allowed: dict[str, list[int]] = {}
        add_ids = _fg_ids_from_candidates(fg_candidates.get("add"), fg_smiles_to_id)
        rep_ids = _fg_ids_from_candidates(fg_candidates.get("replace"), fg_smiles_to_id)
        if add_ids:
            fg_allowed["add"] = add_ids
        if rep_ids:
            fg_allowed["replace"] = rep_ids
        if fg_allowed:
            out["fg_allowed_ids"] = fg_allowed
    elif fg_candidates is not None:
        ids = _fg_ids_from_candidates(fg_candidates, fg_smiles_to_id)
        if ids:
            out["fg_allowed_ids"] = ids

    return out


def _apply_control_to_op_allowed(op_allowed, atom_map, control: dict[str, Any] | None):
    if not control:
        return op_allowed
    import torch

    out = op_allowed.clone()
    anchor_maps = control.get("anchor_atom_maps")
    if anchor_maps:
        allow = {int(a) for a in anchor_maps}
        mask = torch.tensor([int(v) in allow for v in atom_map.tolist()], device=op_allowed.device, dtype=torch.bool)
        if str(control.get("anchor_mode", "include")) == "exclude":
            mask = ~mask
        out[~mask, :] = False
        out[~mask, OP_TO_ID["none"]] = True
    ops = control.get("op_allow")
    if ops:
        allow_ops = {str(o).strip().lower() for o in ops}
        for op_name in ("add", "remove", "replace"):
            if op_name not in allow_ops:
                out[:, OP_TO_ID[op_name]] = False
    return out


def _parse_delta_raw_ranges(raw: Any) -> dict[str, tuple[float, float]]:
    """
    Parse per-property raw-delta sampling ranges.

    Expected format in infer_config.yaml:
      delta_raw_ranges:
        logp: [0, 10]
        qed: [0.0, 0.2]
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("delta_raw_ranges must be a dict like {prop: [min, max], ...}")

    out: dict[str, tuple[float, float]] = {}
    for k, v in raw.items():
        prop = str(k)
        if isinstance(v, (list, tuple)) and len(v) == 2:
            vmin, vmax = float(v[0]), float(v[1])
        else:
            raise ValueError(f"delta_raw_ranges[{prop}] must be a 2-list [min, max], got {v!r}")
        if not (vmin <= vmax):
            raise ValueError(f"delta_raw_ranges[{prop}] must satisfy min <= max, got [{vmin}, {vmax}]")
        out[prop] = (vmin, vmax)
    return out


def _load_tasks(task_yaml: str | Path) -> dict[str, TaskSpec]:
    raw = _load_yaml(task_yaml)
    taskid_prop = raw.get("taskid_prop")
    prop_trend = raw.get("prop_trend")
    if not isinstance(taskid_prop, dict) or not isinstance(prop_trend, dict):
        raise ValueError("task config must contain dict keys: taskid_prop and prop_trend")

    out: dict[str, TaskSpec] = {}
    for k, props in taskid_prop.items():
        tid = str(k)
        if not isinstance(props, list) or not all(isinstance(p, str) for p in props):
            raise ValueError(f"taskid_prop[{tid}] must be a list[str]")
        trend = prop_trend.get(k, prop_trend.get(tid))
        if not isinstance(trend, str):
            raise ValueError(f"Missing prop_trend for task_id={tid}")
        if len(trend) != len(props):
            raise ValueError(f"prop_trend[{tid}] length must match taskid_prop[{tid}]")
        if any(c not in "01" for c in trend):
            raise ValueError(f"prop_trend[{tid}] must be a 0/1 string")
        out[tid] = TaskSpec(task_id=tid, props=list(props), trend=trend)
    if not out:
        raise ValueError("No tasks found in task config")
    return out


def _load_tasks_from_dict(raw: dict[str, Any]) -> dict[str, TaskSpec]:
    taskid_prop = raw.get("taskid_prop")
    prop_trend = raw.get("prop_trend")
    if not isinstance(taskid_prop, dict) or not isinstance(prop_trend, dict):
        raise ValueError("task_defs must contain dict keys: taskid_prop and prop_trend")

    # Type coercion happens in _load_tasks; reimplement minimal here to avoid IO.
    out: dict[str, TaskSpec] = {}
    for k, props in taskid_prop.items():
        tid = str(k)
        if not isinstance(props, list) or not all(isinstance(p, str) for p in props):
            raise ValueError(f"taskid_prop[{tid}] must be a list[str]")
        trend = prop_trend.get(k, prop_trend.get(tid))
        if not isinstance(trend, str):
            raise ValueError(f"Missing prop_trend for task_id={tid}")
        if len(trend) != len(props):
            raise ValueError(f"prop_trend[{tid}] length must match taskid_prop[{tid}]")
        if any(c not in "01" for c in trend):
            raise ValueError(f"prop_trend[{tid}] must be a 0/1 string")
        out[tid] = TaskSpec(task_id=tid, props=list(props), trend=trend)
    if not out:
        raise ValueError("No tasks found in task_defs")
    return out


def _prepare_start_smiles_tagged(smiles: str) -> str:
    mol = mol_from_smiles(smiles)
    mol = ensure_atom_maps(mol)
    mol = normalize_bracket_atom_hydrogens(mol)
    return mol_to_tagged_smiles(mol)


def _compute_op_allowed(data) -> torch.Tensor:
    """
    Return [N, 4] boolean op-allowed mask aligned with OP_TO_ID order.
    Scaffold-aware constraints when available.
    """
    import torch

    n = int(data.z.size(0))
    add_allowed = getattr(data, "add_allowed", None)
    remove_allowed = getattr(data, "remove_allowed", None)
    if add_allowed is None or remove_allowed is None:
        return torch.ones((n, 4), device=data.z.device, dtype=torch.bool)
    add_allowed = add_allowed.to(device=data.z.device)
    remove_allowed = remove_allowed.to(device=data.z.device)
    allowed = torch.zeros((n, 4), device=data.z.device, dtype=torch.bool)
    allowed[:, OP_TO_ID["none"]] = True
    allowed[:, OP_TO_ID["add"]] = add_allowed
    allowed[:, OP_TO_ID["remove"]] = remove_allowed
    allowed[:, OP_TO_ID["replace"]] = add_allowed & remove_allowed
    return allowed


def _desired_props_from_task(
    *,
    task: TaskSpec,
    property_cols: list[str],
    trend_alpha: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Create a [1, P] raw delta target aligned with property_cols (delta-only).
    Unspecified properties get 0; trend bit 1-> +trend_alpha, 0 -> -trend_alpha.
    """
    import torch

    p = len(property_cols)
    if p == 0:
        return torch.zeros((1, 0), device=device)

    # property_cols: expected to be delta cols (only).
    col_to_prop = []
    for c in property_cols:
        if c.endswith("_delta"):
            col_to_prop.append(c[: -len("_delta")])
        else:
            col_to_prop.append(c)
    prop_to_i = {n: i for i, n in enumerate(col_to_prop)}

    z = torch.zeros((1, p), device=device, dtype=torch.float32)
    for prop_name, bit in zip(task.props, task.trend):
        if prop_name not in prop_to_i:
            continue
        i = prop_to_i[prop_name]
        z[0, i] = float(trend_alpha) if bit == "1" else -float(trend_alpha)

    return z


def _sample_actions_k(
    model: Any,
    data0,
    *,
    fg_vocab: FunctionalGroupVocab,
    k: int,
    max_edits: int,
    temperature_op: float,
    temperature_fg: float,
    temperature_remove: float,
    seed: int,
    forced_k_target: int | None = None,
    fg_allowed_ids: list[int] | dict[str, list[int]] | None = None,
) -> list[list[dict[str, Any]]]:
    # Backward-compatible wrapper: sampling logic lives on the model to avoid
    # inference-time coupling to internal head wiring / tensor shapes.
    return model.sample_edit_sets(
        data0,
        fg_vocab=fg_vocab,
        k_samples=int(k),
        max_edits=int(max_edits),
        temperature_op=float(temperature_op),
        temperature_fg=float(temperature_fg),
        temperature_remove=float(temperature_remove),
        seed=int(seed),
        forced_k_target=forced_k_target,
        fg_allowed_ids=fg_allowed_ids,
    )


def _top_action_pool_once(
    model: Any,
    data0,
    *,
    fg_vocab: FunctionalGroupVocab,
    pool_n: int,
    op_topk: int,
    fg_topk: int,
    remove_topk: int,
    temperature_op: float,
    temperature_fg: float,
    temperature_remove: float,
    fg_allowed_ids: list[int] | dict[str, list[int]] | None = None,
) -> list[dict[str, Any]]:
    """
    Run one forward pass and return a pool of high-confidence single-edit actions.

    Returns:
      - actions: list[edit_dict] sorted by descending confidence (no step fields).
    """
    from .pyg_utils import require_torch_geometric

    require_torch_geometric()
    import torch
    import torch.nn.functional as F
    from torch_geometric.data import Batch

    device = next(model.parameters()).device
    model.eval()
    batch = Batch.from_data_list([data0]).to(device)
    out = model.forward(batch, teacher_forcing=False)
    # Keep MoE routing diagnostics consistent with iterative mode (best-effort).
    try:
        model._last_moe_topi = getattr(out, "moe_topi", None).detach().cpu().tolist() if getattr(out, "moe_topi", None) is not None else None  # type: ignore[attr-defined]
        model._last_moe_topv = getattr(out, "moe_topv", None).detach().cpu().tolist() if getattr(out, "moe_topv", None) is not None else None  # type: ignore[attr-defined]
        model._last_moe_entropy = getattr(out, "moe_entropy", None).detach().cpu().tolist() if getattr(out, "moe_entropy", None) is not None else None  # type: ignore[attr-defined]
        model._last_moe_topk_mass = getattr(out, "moe_topk_mass", None).detach().cpu().tolist() if getattr(out, "moe_topk_mass", None) is not None else None  # type: ignore[attr-defined]
        model._last_moe_gate_probs = getattr(out, "moe_gate_probs", None).detach().cpu().tolist() if getattr(out, "moe_gate_probs", None) is not None else None  # type: ignore[attr-defined]
    except Exception:
        pass

    op_allowed = getattr(batch, "op_allowed", None)
    if op_allowed is None:
        op_allowed = torch.ones((int(batch.z.size(0)), 4), device=device, dtype=torch.bool)
    op_logits = out.op_logits.masked_fill(~op_allowed, -1e9)
    op_probs = F.softmax(op_logits / max(1e-6, float(temperature_op)), dim=-1)  # [N,4]

    fg_logits = out.fg_logits
    fg_probs_all = F.softmax(fg_logits / max(1e-6, float(temperature_fg)), dim=-1)  # [N,V]

    fg_id_to_smiles = getattr(fg_vocab, "id_to_smiles", {}) or {}
    node0 = (batch.batch == 0).nonzero(as_tuple=True)[0]
    n_nodes = int(batch.z.size(0))
    deg = torch.bincount(batch.edge_index[0].to(torch.long), minlength=n_nodes) if hasattr(batch, "edge_index") else None

    # Optional FG restriction from retrieval.
    def _fg_ids_for_op(op_name: str) -> list[int] | None:
        if fg_allowed_ids is None:
            return None
        if isinstance(fg_allowed_ids, dict):
            vals = fg_allowed_ids.get(op_name)
            return list(vals) if isinstance(vals, (list, tuple)) else None
        if isinstance(fg_allowed_ids, (list, tuple)):
            return list(fg_allowed_ids)
        return None

    def _build_allowed_mask(ids: list[int] | None) -> torch.Tensor | None:
        if not ids:
            return None
        mask = torch.zeros((int(fg_probs_all.size(-1)),), device=device, dtype=torch.bool)
        for fid in ids:
            if 0 <= int(fid) < int(fg_probs_all.size(-1)):
                mask[int(fid)] = True
        return mask

    allowed_mask_add = _build_allowed_mask(_fg_ids_for_op("add"))
    allowed_mask_replace = _build_allowed_mask(_fg_ids_for_op("replace"))

    def _terminal_neighbor_candidates(anchor_idx: int) -> torch.Tensor:
        if not hasattr(batch, "edge_index") or deg is None:
            return torch.empty((0,), device=device, dtype=torch.long)
        src = batch.edge_index[0]
        dst = batch.edge_index[1]
        nbrs = dst[src == int(anchor_idx)]
        if int(nbrs.numel()) == 0:
            return nbrs
        scaffold = getattr(batch, "scaffold_mask", None)
        if scaffold is not None:
            if not bool(scaffold[int(anchor_idx)].item()):
                return nbrs.new_empty((0,))
            return nbrs[~scaffold[nbrs]]
        term = nbrs[deg[nbrs] <= 1]
        return term if int(term.numel()) else nbrs

    def _add_action(e: dict[str, Any], score: float, out_list: list[tuple[float, dict[str, Any]]]) -> None:
        # Score uses log-prob to avoid underflow; higher is better.
        e2 = dict(e)
        e2["_score"] = float(score)
        out_list.append((float(score), e2))

    # Build all candidates, then keep top pool_n.
    cand_scored: list[tuple[float, dict[str, Any]]] = []
    op_topk = max(1, int(op_topk))
    fg_topk = max(1, int(fg_topk))
    remove_topk = max(1, int(remove_topk))

    for gi in node0.tolist():
        # Pick top ops among non-none (add/remove/replace). Indices are 1..3 in OP_TO_ID order.
        p_ops = op_probs[gi, 1:]  # [3]
        # If nothing is allowed, skip.
        if float(p_ops.sum().item()) <= 0.0:
            continue
        k_op = min(int(op_topk), 3)
        topv, topi = torch.topk(p_ops, k=k_op, largest=True)
        for vv, ii in zip(topv.tolist(), topi.tolist()):
            op_id = int(ii) + 1
            op_name = str(ID_TO_OP.get(op_id, "none"))
            if op_name == "none":
                continue
            p_op = max(1e-12, float(vv))
            base = {
                "anchor_atom_map": int(batch.atom_map[gi].item()),
                "op": op_name,
            }
            score_op = float(torch.log(torch.tensor(p_op)).item())

            if op_name in ("add", "replace"):
                probs = fg_probs_all[gi]
                allowed_mask = allowed_mask_add if op_name == "add" else allowed_mask_replace
                if allowed_mask is not None and bool(allowed_mask.any().item()):
                    masked = probs * allowed_mask.to(dtype=probs.dtype)
                    if float(masked.sum().item()) > 0.0:
                        probs = masked
                probs = probs / probs.sum().clamp_min(1e-12)
                k_fg = min(int(fg_topk), int(probs.numel()))
                fg_v, fg_i = torch.topk(probs, k=k_fg, largest=True)

                if op_name == "add":
                    for p_fg, fg_id in zip(fg_v.tolist(), fg_i.tolist()):
                        fg_smiles = fg_id_to_smiles.get(int(fg_id))
                        if fg_smiles is None:
                            continue
                        e = dict(base)
                        e["fg_id"] = int(fg_id)
                        e["fg_smiles"] = str(fg_smiles)
                        score = score_op + float(torch.log(torch.tensor(max(1e-12, float(p_fg)))).item())
                        _add_action(e, score, cand_scored)
                else:  # replace: cross fg and removed
                    cand = _terminal_neighbor_candidates(int(gi))
                    if int(cand.numel()) == 0:
                        continue
                    q = model.remove_query(torch.cat([out.node_h[gi], out.graph_h[0]], dim=-1))
                    scores = (out.node_h[cand] * q.unsqueeze(0)).sum(dim=-1)
                    rm_probs = F.softmax(scores / max(1e-6, float(temperature_remove)), dim=-1)
                    k_rm = min(int(remove_topk), int(rm_probs.numel()))
                    rm_v, rm_i = torch.topk(rm_probs, k=k_rm, largest=True)
                    for p_fg, fg_id in zip(fg_v.tolist(), fg_i.tolist()):
                        fg_smiles = fg_id_to_smiles.get(int(fg_id))
                        if fg_smiles is None:
                            continue
                        for p_rm, ridx in zip(rm_v.tolist(), rm_i.tolist()):
                            removed_map = int(batch.atom_map[int(cand[int(ridx)].item())].item())
                            e = dict(base)
                            e["fg_id"] = int(fg_id)
                            e["fg_smiles"] = str(fg_smiles)
                            e["removed_atom_map"] = int(removed_map)
                            score = score_op
                            score += float(torch.log(torch.tensor(max(1e-12, float(p_fg)))).item())
                            score += float(torch.log(torch.tensor(max(1e-12, float(p_rm)))).item())
                            _add_action(e, score, cand_scored)

            elif op_name == "remove":
                cand = _terminal_neighbor_candidates(int(gi))
                if int(cand.numel()) == 0:
                    continue
                q = model.remove_query(torch.cat([out.node_h[gi], out.graph_h[0]], dim=-1))
                scores = (out.node_h[cand] * q.unsqueeze(0)).sum(dim=-1)
                rm_probs = F.softmax(scores / max(1e-6, float(temperature_remove)), dim=-1)
                k_rm = min(int(remove_topk), int(rm_probs.numel()))
                rm_v, rm_i = torch.topk(rm_probs, k=k_rm, largest=True)
                for p_rm, ridx in zip(rm_v.tolist(), rm_i.tolist()):
                    removed_map = int(batch.atom_map[int(cand[int(ridx)].item())].item())
                    e = dict(base)
                    e["removed_atom_map"] = int(removed_map)
                    score = score_op + float(torch.log(torch.tensor(max(1e-12, float(p_rm)))).item())
                    _add_action(e, score, cand_scored)

    cand_scored.sort(key=lambda x: x[0], reverse=True)
    actions = [e for _, e in cand_scored[: max(1, int(pool_n))]]
    return actions


def _pick_edit_set_from_pool(
    pool: list[dict[str, Any]],
    *,
    k_target: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """
    Sample up to k_target edits from a pool while enforcing unique anchors.
    Pool items may contain an optional float key "_score" (log-prob); higher is better.
    """
    k_target = max(0, int(k_target))
    if k_target <= 0 or not pool:
        return []

    # Softmax over scores for weighted sampling without replacement.
    idxs = list(range(len(pool)))
    picked: list[dict[str, Any]] = []
    used_anchors: set[int] = set()
    # Sample without replacement by repeated weighted draws (pool is tiny: <= 10).
    while idxs and len(picked) < k_target:
        # Renormalize on the remaining indices.
        scores = [float(pool[j].get("_score", 0.0)) for j in idxs]
        mx = max(scores) if scores else 0.0
        import math

        ws = [math.exp(s - mx) for s in scores]
        z = sum(ws) if ws else 1.0
        r = rng.random() * float(z)
        acc = 0.0
        chosen_pos = 0
        for pos, w in enumerate(ws):
            acc += float(w)
            if r <= acc:
                chosen_pos = pos
                break
        j = idxs.pop(chosen_pos)
        e = pool[j]
        anchor = int(e.get("anchor_atom_map", -1))
        if anchor in used_anchors:
            continue
        used_anchors.add(anchor)
        # Drop bookkeeping fields.
        e2 = {k: v for k, v in e.items() if not str(k).startswith("_")}
        picked.append(e2)

    def _op_rank(op_name: str) -> int:
        return {"remove": 0, "replace": 1, "add": 2}.get(op_name, 99)

    picked = sorted(picked, key=lambda e: (_op_rank(str(e.get("op", ""))), int(e.get("anchor_atom_map", 0))))
    for si, e in enumerate(picked, start=1):
        e["step"] = si
    return picked


def _load_state_dict_forgiving(model, state_dict: dict[str, Any]) -> None:
    """
    `strict=False` does not ignore tensor size mismatches; filter those keys out so
    older checkpoints can still be loaded (with partially re-initialized modules).
    """
    try:
        from torch.nn.parameter import UninitializedParameter  # type: ignore
    except Exception:  # pragma: no cover
        UninitializedParameter = ()  # type: ignore

    model_sd = model.state_dict()
    filtered: dict[str, Any] = {}
    dropped: list[str] = []
    for k, v in (state_dict or {}).items():
        if k not in model_sd:
            continue
        # Allow lazy modules (e.g., LazyLinear) to load weights even if the local
        # state_dict has uninitialized placeholder tensors.
        if isinstance(model_sd[k], UninitializedParameter):
            filtered[k] = v
            continue
        try:
            if hasattr(v, "shape") and hasattr(model_sd[k], "shape") and tuple(v.shape) != tuple(model_sd[k].shape):
                dropped.append(k)
                continue
        except Exception:
            dropped.append(k)
            continue
        filtered[k] = v
    model.load_state_dict(filtered, strict=False)
    if dropped:
        print(f"[warn] dropped {len(dropped)} mismatched keys (e.g. {dropped[:5]})")

def _canonicalize_smiles(smiles: str) -> str | None:
    s = str(smiles).strip()
    if not s:
        return None
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return None
        return str(Chem.MolToSmiles(mol, canonical=True))
    except Exception:
        return s


def _infer_removed_fg_smiles(
    *,
    start_smiles_tagged: str,
    anchor_atom_map: int,
    removed_atom_map: int,
) -> str | None:
    """
    Infer a functional-group-like fragment SMILES (with a '*' dummy) for a remove/replace action.

    We cut the bond between anchor and removed atom and return the substituent fragment containing
    the removed atom, augmented with a dummy atom connected at the attachment site.
    """
    try:
        from rdkit import Chem
    except Exception:
        return None

    try:
        mol = ensure_atom_maps(mol_from_smiles(str(start_smiles_tagged)))
        if mol is None:
            return None

        anchor_idx = None
        removed_idx = None
        for a in mol.GetAtoms():
            amap = int(a.GetAtomMapNum())
            if amap == int(anchor_atom_map):
                anchor_idx = int(a.GetIdx())
            elif amap == int(removed_atom_map):
                removed_idx = int(a.GetIdx())
        if anchor_idx is None or removed_idx is None:
            return None

        bond = mol.GetBondBetweenAtoms(int(anchor_idx), int(removed_idx))
        if bond is None:
            return None
        bond_idx = int(bond.GetIdx())

        frag_mol = Chem.FragmentOnBonds(mol, [bond_idx], addDummies=False)
        frags = Chem.GetMolFrags(frag_mol, asMols=True, sanitizeFrags=False)
        if not frags:
            return None

        target = None
        for f in frags:
            for a in f.GetAtoms():
                if int(a.GetAtomMapNum()) == int(removed_atom_map):
                    target = f
                    break
            if target is not None:
                break
        if target is None:
            return None

        rw = Chem.RWMol(target)
        frag_removed_idx = None
        for a in rw.GetAtoms():
            if int(a.GetAtomMapNum()) == int(removed_atom_map):
                frag_removed_idx = int(a.GetIdx())
                break
        if frag_removed_idx is None:
            return None

        dummy = Chem.Atom(0)
        dummy.SetAtomMapNum(0)
        dummy_idx = int(rw.AddAtom(dummy))
        rw.AddBond(int(frag_removed_idx), int(dummy_idx), Chem.BondType.SINGLE)
        out = rw.GetMol()
        Chem.SanitizeMol(out, catchErrors=True)
        return str(Chem.MolToSmiles(out, canonical=True))
    except Exception:
        return None


def main() -> None:
    _disable_rdkit_warnings()
    parser = argparse.ArgumentParser(description="Inference for one-shot edit-set model (predict K edited SMILES per input).")
    parser.add_argument("--infer_config", type=str, default=None, help="YAML with tasks + IO + model settings (optional).")
    parser.add_argument("--config", type=str, default="config.yaml", help="Base config.yaml (functional groups + property list).")
    parser.add_argument("--train_config", type=str, default="train_oneshot_config.yaml", help="Model hyperparams YAML (hidden_dim/num_layers).")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_oneshot.pt")
    parser.add_argument("--tasks", type=str, default=None, help="YAML containing taskid_prop and prop_trend.")
    parser.add_argument(
        "--task_id",
        type=str,
        default=None,
        help='Which task_id to run; use "all" to run all tasks (default: only task if tasks file has one).',
    )
    parser.add_argument(
        "--task_ids",
        type=str,
        default=None,
        help='Comma-separated task ids to run (overrides --task_id). Example: "101,102,201".',
    )

    parser.add_argument("--input_csv", type=str, default=None)
    parser.add_argument("--control_config", type=str, default=None, help="YAML with tagged smiles + control constraints.")
    parser.add_argument("--smiles_col", type=str, default="mol")
    parser.add_argument("--output_csv", type=str, default=None, help="(Legacy) Single output CSV path.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help='Output directory. Writes one file per task: "<task_id>_task.csv".',
    )

    parser.add_argument("--k", type=int, default=1, help="Number of sampled predictions per molecule.")
    parser.add_argument("--max_edits", type=int, default=10)
    parser.add_argument(
        "--output_actions",
        action="store_true",
        help="Save predicted edit actions per sample (pred_actions_list_json).",
    )
    parser.add_argument("--max_attempts_per_sample", type=int, default=10, help="Retries to get a valid edited molecule for each sample.")
    parser.add_argument(
        "--max_total_attempts",
        type=int,
        default=None,
        help="Maximum total sampling attempts per (input_mol, task_id). Defaults to k*max_attempts_per_sample*50.",
    )
    parser.add_argument(
        "--allow_duplicates",
        action="store_true",
        help="Do not enforce uniqueness across the k returned SMILES for each input/task.",
    )
    parser.add_argument(
        "--allow_invalid_smiles",
        action="store_true",
        help='Allow invalid/empty SMILES in output (otherwise they are retried and may raise if budget is exhausted).',
    )
    parser.add_argument(
        "--allow_fewer_than_k",
        action="store_true",
        help="If uniqueness/validity cannot be satisfied within attempt budget, output fewer than k instead of raising.",
    )

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trend_alpha", type=float, default=1.0, help="Standardized delta magnitude for specified props (±alpha).")
    parser.add_argument("--temperature_op", type=float, default=1.0)
    parser.add_argument("--temperature_fg", type=float, default=1.0)
    parser.add_argument("--temperature_remove", type=float, default=1.0)
    # Retrieval-conditioned inference: use a similar (mol + src props) training example to set delta props.
    parser.add_argument("--retrieval", action="store_true", help="Enable retrieval-conditioned delta props.")
    parser.add_argument("--retrieval_csv", type=str, default="", help="CSV used as retrieval bank (must include *_src/*_delta columns).")
    parser.add_argument("--retrieval_k", type=int, default=8, help="Top-K retrieval neighbors to average.")
    parser.add_argument("--retrieval_top_m", type=int, default=2000, help="Preselect top-M by src-prop distance before Tanimoto.")
    parser.add_argument("--retrieval_alpha", type=float, default=10.0, help="Softmax temperature over retrieval scores (higher = sharper).")
    parser.add_argument("--retrieval_w_sim", type=float, default=1.0, help="Weight on (1 - Tanimoto).")
    parser.add_argument("--retrieval_w_prop", type=float, default=1.0, help="Weight on normalized src-prop distance.")
    parser.add_argument(
        "--retrieval_delta_hypotheses",
        type=int,
        default=1,
        help="Try multiple retrieval delta hypotheses (best-of over generated candidates). 1 uses weighted-mean delta (default).",
    )
    parser.add_argument(
        "--retrieval_delta_strategy",
        type=str,
        default="top",
        help='How to form hypotheses from retrieval topK: "top", "sample", "mean_top", "mean_sample".',
    )
    # Optional weak constraints derived from retrieval topK.
    parser.add_argument("--retrieval_use_num_edits", action="store_true", help="Force edit count to rounded mean num_edits from retrieval topK.")
    parser.add_argument("--retrieval_use_fg_pool", action="store_true", help="Restrict add/replace FG sampling to FG IDs seen in retrieval topK.")
    parser.add_argument("--retrieval_fg_max_ids", type=int, default=0, help="Optional cap on FG pool size (0 = no cap).")

    # Sampling strategy:
    # - iterative: current behavior, run multiple sampling attempts (may recompute model each time).
    # - top_actions: one forward pass, build a top-N action pool, then sample/edit-set combos from it.
    # - conf_beam: confidence-guided beam search over single-edit steps.
    # - best_first: global best-first search with light re-ranking.
    parser.add_argument(
        "--sampling_mode",
        type=str,
        default="",
        help='Sampling mode: "iterative", "top_actions", "conf_beam", or "best_first" (default from infer_config or iterative).',
    )
    parser.add_argument("--top_actions_n", type=int, default=10, help="Pool size N for --sampling_mode top_actions.")
    parser.add_argument("--top_actions_op_topk", type=int, default=2, help="Per-anchor op top-k for action pool.")
    parser.add_argument("--top_actions_fg_topk", type=int, default=3, help="Per-anchor fg top-k for action pool.")
    parser.add_argument("--top_actions_remove_topk", type=int, default=3, help="Per-anchor remove pointer top-k for action pool.")
    parser.add_argument(
        "--top_actions_fallback_iterative",
        action="store_true",
        help="If top_actions cannot produce enough valid unique samples, fall back to iterative sampling.",
    )
    # Step-by-step rollout sampling: one edit per forward/apply, repeated for multiple steps.
    parser.add_argument(
        "--rollout_steps",
        type=int,
        default=0,
        help="Number of rollout steps for --sampling_mode rollout (0 = use --max_edits).",
    )
    parser.add_argument("--rollout_size", type=int, default=8, help="Beam size for --sampling_mode rollout.")
    parser.add_argument("--rollout_batch_size", type=int, default=8, help="Batch size for rollout forward passes.")
    parser.add_argument(
        "--rollout_scaffold_mode",
        type=str,
        default="fixed",
        help='Scaffold definition during rollout: "fixed" (source scaffold atom-maps) or "dynamic" (recompute Murcko each step).',
    )
    parser.add_argument(
        "--conf_beam_size",
        type=int,
        default=2,
        help="Beam size for --sampling_mode conf_beam.",
    )
    parser.add_argument(
        "--conf_beam_topk",
        type=int,
        default=10,
        help="Top-K actions per beam state for --sampling_mode conf_beam.",
    )
    parser.add_argument(
        "--conf_beam_alpha",
        type=float,
        default=1.0,
        help="Length penalty alpha for conf_beam scoring: sum log conf / (T^alpha).",
    )
    parser.add_argument(
        "--conf_beam_tau",
        type=float,
        default=0.0,
        help="Absolute confidence threshold for conf_beam early stop.",
    )
    parser.add_argument(
        "--conf_beam_r",
        type=float,
        default=0.0,
        help="Relative confidence threshold for conf_beam early stop: max_conf_next < r * conf_prev.",
    )
    parser.add_argument(
        "--conf_beam_max_steps",
        type=int,
        default=0,
        help="Max steps for conf_beam (0 = use --max_edits).",
    )
    parser.add_argument(
        "--best_first_open_size",
        type=int,
        default=128,
        help="Open list size for --sampling_mode best_first.",
    )
    parser.add_argument(
        "--best_first_expand",
        type=int,
        default=2,
        help="Number of paths to expand per iteration for best_first.",
    )
    parser.add_argument(
        "--best_first_topk",
        type=int,
        default=10,
        help="Top-K actions per path for best_first.",
    )
    parser.add_argument(
        "--best_first_alpha",
        type=float,
        default=1.0,
        help="Length penalty alpha for best_first scoring.",
    )
    parser.add_argument(
        "--best_first_lambda",
        type=float,
        default=0.1,
        help="Deprecated (unused): trend margin removed from best_first score.",
    )
    parser.add_argument(
        "--best_first_beta",
        type=float,
        default=0.0,
        help="Step penalty for best_first scoring.",
    )
    parser.add_argument(
        "--best_first_max_steps",
        type=int,
        default=0,
        help="Max steps for best_first (0 = use --max_edits).",
    )
    parser.add_argument(
        "--best_first_max_expansions",
        type=int,
        default=200,
        help="Max number of path expansions for best_first.",
    )
    parser.add_argument(
        "--rdkit_threads",
        type=int,
        default=0,
        help="Number of threads for RDKit-heavy rollout ops (0 = disable).",
    )
    parser.add_argument("--profile_timing", action="store_true", help="Collect timing breakdown + retry stats.")
    parser.add_argument(
        "--profile_timing_json",
        type=str,
        default="",
        help='Optional JSON output path for profiling (default: "<output_dir>/timing_profile.json").',
    )
    parser.add_argument(
        "--profile_timing_sync_cuda",
        action="store_true",
        help="Synchronize CUDA around timed blocks for more accurate GPU timings (slower).",
    )
    parser.add_argument("--profile_error_topk", type=int, default=20, help="Top-K exception messages to keep per error bucket.")
    parser.add_argument("--profile_error_samples", type=int, default=20, help="Max number of error samples to save into timing_profile.json.")
    parser.add_argument("--profile_error_maxlen", type=int, default=200, help="Max length of saved exception messages (truncate).")
    parser.add_argument("--max_rows", type=int, default=None, help="Optional cap on number of input rows processed (debug).")
    args = parser.parse_args()
    # CLI should take precedence over infer_config. Track which argparse dests were explicitly provided.
    provided_dests: set[str] = set()
    for tok in sys.argv[1:]:
        if not tok.startswith("--"):
            continue
        opt = tok.split("=", 1)[0]
        action = parser._option_string_actions.get(opt)  # type: ignore[attr-defined]
        if action is not None and getattr(action, "dest", None):
            provided_dests.add(str(action.dest))

    import torch

    from .apply import apply_edits
    from .model_oneshot import OneShotTwoStageEditModel
    from .pyg_utils import require_torch_geometric
    from .properties import build_property_fns, calc_properties

    tasks_inline: dict[str, Any] | None = None
    delta_raw_ranges: dict[str, tuple[float, float]] = {}
    retrieval_cfg: dict[str, Any] | None = None
    control_inline: dict[str, Any] | None = None
    control_inputs_raw: list[dict[str, Any]] = []
    t_main0 = time.perf_counter()
    icfg: dict[str, Any] | None = None
    if args.infer_config:
        icfg = _load_infer_config(args.infer_config)

    prof = _Profiler(
        enabled=bool(getattr(args, "profile_timing", False)),
        sync_cuda=bool(getattr(args, "profile_timing_sync_cuda", False)),
        error_topk=int(getattr(args, "profile_error_topk", 20)),
        max_error_samples=int(getattr(args, "profile_error_samples", 20)),
        error_maxlen=int(getattr(args, "profile_error_maxlen", 200)),
    )

    if icfg is not None:
        profiling_cfg = icfg.get("profiling") if isinstance(icfg.get("profiling"), dict) else None
        if isinstance(profiling_cfg, dict):
            if "profile_timing" not in provided_dests and profiling_cfg.get("enabled", None) is not None:
                prof.enabled = bool(_as_bool(profiling_cfg.get("enabled")))
            if "profile_timing_sync_cuda" not in provided_dests and profiling_cfg.get("sync_cuda", None) is not None:
                prof.sync_cuda = bool(_as_bool(profiling_cfg.get("sync_cuda")))
            if "profile_timing_json" not in provided_dests and profiling_cfg.get("json_path", None) is not None:
                args.profile_timing_json = str(profiling_cfg.get("json_path") or "")
            if "profile_error_topk" not in provided_dests and profiling_cfg.get("error_topk", None) is not None:
                prof.error_topk = int(profiling_cfg.get("error_topk"))
            if "profile_error_samples" not in provided_dests and profiling_cfg.get("error_samples", None) is not None:
                prof.max_error_samples = int(profiling_cfg.get("error_samples"))
            if "profile_error_maxlen" not in provided_dests and profiling_cfg.get("error_maxlen", None) is not None:
                prof.error_maxlen = int(profiling_cfg.get("error_maxlen"))
            if "max_rows" not in provided_dests and profiling_cfg.get("max_rows", None) is not None:
                args.max_rows = int(profiling_cfg.get("max_rows"))

    if icfg is not None:
        rollout_cfg = icfg.get("rollout") if isinstance(icfg.get("rollout"), dict) else {}
        if not isinstance(rollout_cfg, dict):
            rollout_cfg = {}
        # Respect explicit CLI overrides: only take infer_config values for args not provided on CLI.
        if "config" not in provided_dests:
            args.config = str(icfg.get("base_config", args.config))
        if "train_config" not in provided_dests:
            args.train_config = str(icfg.get("train_config", args.train_config))
        if "checkpoint" not in provided_dests:
            args.checkpoint = str(icfg.get("checkpoint", args.checkpoint))
        task_defs = icfg.get("task_defs")
        if isinstance(task_defs, dict) and ("tasks" not in provided_dests):
            tasks_inline = task_defs
        else:
            if "tasks" not in provided_dests and icfg.get("tasks", None) is not None:
                args.tasks = str(icfg.get("tasks", args.tasks))
        if "task_id" not in provided_dests and icfg.get("task_id", None) is not None:
            args.task_id = str(icfg.get("task_id"))
        if "task_ids" not in provided_dests and icfg.get("task_ids", None) is not None:
            args.task_ids = str(icfg.get("task_ids"))
        if "input_csv" not in provided_dests and icfg.get("input_csv", None) is not None:
            args.input_csv = str(icfg.get("input_csv"))
        if "smiles_col" not in provided_dests and icfg.get("smiles_col", None) is not None:
            args.smiles_col = str(icfg.get("smiles_col"))
        if "output_csv" not in provided_dests and icfg.get("output_csv", None) is not None:
            args.output_csv = str(icfg.get("output_csv"))
        if "output_dir" not in provided_dests and icfg.get("output_dir", None) is not None:
            args.output_dir = str(icfg.get("output_dir"))
        if "k" not in provided_dests and icfg.get("k", None) is not None:
            args.k = int(icfg.get("k"))
        if "max_edits" not in provided_dests and icfg.get("max_edits", None) is not None:
            args.max_edits = int(icfg.get("max_edits"))
        if "max_attempts_per_sample" not in provided_dests and icfg.get("max_attempts_per_sample", None) is not None:
            args.max_attempts_per_sample = int(icfg.get("max_attempts_per_sample"))
        if "max_total_attempts" not in provided_dests and icfg.get("max_total_attempts", None) is not None:
            args.max_total_attempts = int(icfg.get("max_total_attempts"))
        if "allow_duplicates" not in provided_dests and icfg.get("allow_duplicates", None) is not None:
            args.allow_duplicates = _as_bool(icfg.get("allow_duplicates"))
        if "allow_invalid_smiles" not in provided_dests and icfg.get("allow_invalid_smiles", None) is not None:
            args.allow_invalid_smiles = _as_bool(icfg.get("allow_invalid_smiles"))
        if "allow_fewer_than_k" not in provided_dests and icfg.get("allow_fewer_than_k", None) is not None:
            args.allow_fewer_than_k = _as_bool(icfg.get("allow_fewer_than_k"))
        if "device" not in provided_dests and icfg.get("device", None) is not None:
            args.device = str(icfg.get("device"))
        if "seed" not in provided_dests and icfg.get("seed", None) is not None:
            args.seed = int(icfg.get("seed"))
        if "sampling_mode" not in provided_dests and icfg.get("sampling_mode", None) is not None:
            args.sampling_mode = str(icfg.get("sampling_mode") or "")
        if "rollout_steps" not in provided_dests and (icfg.get("rollout_steps", None) is not None or rollout_cfg.get("steps", None) is not None):
            args.rollout_steps = int(icfg.get("rollout_steps", rollout_cfg.get("steps", args.rollout_steps)))
        if "rollout_size" not in provided_dests and (icfg.get("rollout_size", None) is not None or rollout_cfg.get("size", None) is not None):
            args.rollout_size = int(icfg.get("rollout_size", rollout_cfg.get("size", args.rollout_size)))
        if "rollout_batch_size" not in provided_dests and (
            icfg.get("rollout_batch_size", None) is not None or rollout_cfg.get("batch_size", None) is not None
        ):
            args.rollout_batch_size = int(icfg.get("rollout_batch_size", rollout_cfg.get("batch_size", args.rollout_batch_size)))
        if "rollout_scaffold_mode" not in provided_dests and (
            icfg.get("rollout_scaffold_mode", None) is not None or rollout_cfg.get("scaffold_mode", None) is not None
        ):
            args.rollout_scaffold_mode = str(
                icfg.get("rollout_scaffold_mode", rollout_cfg.get("scaffold_mode", args.rollout_scaffold_mode)) or ""
            )
        if "rdkit_threads" not in provided_dests and icfg.get("rdkit_threads", None) is not None:
            args.rdkit_threads = int(icfg.get("rdkit_threads"))
        if "trend_alpha" not in provided_dests and icfg.get("trend_alpha", None) is not None:
            args.trend_alpha = float(icfg.get("trend_alpha"))
        if icfg.get("delta_raw_ranges", None) is not None:
            # delta_raw_ranges is not a CLI arg; treat infer_config as authoritative.
            delta_raw_ranges = _parse_delta_raw_ranges(icfg.get("delta_raw_ranges"))
        temps = icfg.get("temperatures", {})
        if isinstance(temps, dict):
            if "temperature_op" not in provided_dests and temps.get("op", None) is not None:
                args.temperature_op = float(temps.get("op", args.temperature_op))
            if "temperature_fg" not in provided_dests and temps.get("fg", None) is not None:
                args.temperature_fg = float(temps.get("fg", args.temperature_fg))
            if "temperature_remove" not in provided_dests and temps.get("remove", None) is not None:
                args.temperature_remove = float(temps.get("remove", args.temperature_remove))
        retrieval_cfg = icfg.get("retrieval") if isinstance(icfg.get("retrieval"), dict) else None
        if retrieval_cfg:
            if retrieval_cfg.get("enabled", None) is not None:
                if "retrieval" not in provided_dests:
                    args.retrieval = _as_bool(retrieval_cfg.get("enabled"))
            if "retrieval_csv" not in provided_dests:
                args.retrieval_csv = str(retrieval_cfg.get("data_csv", args.retrieval_csv) or "")
            if "retrieval_k" not in provided_dests and retrieval_cfg.get("k", None) is not None:
                args.retrieval_k = int(retrieval_cfg.get("k", args.retrieval_k))
            if "retrieval_top_m" not in provided_dests and retrieval_cfg.get("top_m", None) is not None:
                args.retrieval_top_m = int(retrieval_cfg.get("top_m", args.retrieval_top_m))
            if "retrieval_alpha" not in provided_dests and retrieval_cfg.get("alpha", None) is not None:
                args.retrieval_alpha = float(retrieval_cfg.get("alpha", args.retrieval_alpha))
            if "retrieval_w_sim" not in provided_dests and retrieval_cfg.get("w_sim", None) is not None:
                args.retrieval_w_sim = float(retrieval_cfg.get("w_sim", args.retrieval_w_sim))
            if "retrieval_w_prop" not in provided_dests and retrieval_cfg.get("w_prop", None) is not None:
                args.retrieval_w_prop = float(retrieval_cfg.get("w_prop", args.retrieval_w_prop))
            if "retrieval_delta_hypotheses" not in provided_dests and retrieval_cfg.get("delta_hypotheses", None) is not None:
                args.retrieval_delta_hypotheses = int(retrieval_cfg.get("delta_hypotheses", args.retrieval_delta_hypotheses))
            if "retrieval_delta_strategy" not in provided_dests and retrieval_cfg.get("delta_strategy", None) is not None:
                args.retrieval_delta_strategy = str(retrieval_cfg.get("delta_strategy", args.retrieval_delta_strategy) or "")
            rcon = retrieval_cfg.get("constraints", {})
            if isinstance(rcon, dict):
                if rcon.get("use_num_edits", None) is not None:
                    if "retrieval_use_num_edits" not in provided_dests:
                        args.retrieval_use_num_edits = _as_bool(rcon.get("use_num_edits"))
                if rcon.get("use_fg_pool", None) is not None:
                    if "retrieval_use_fg_pool" not in provided_dests:
                        args.retrieval_use_fg_pool = _as_bool(rcon.get("use_fg_pool"))
                if rcon.get("fg_max_ids", None) is not None:
                    if "retrieval_fg_max_ids" not in provided_dests:
                        args.retrieval_fg_max_ids = int(rcon.get("fg_max_ids"))

        sampling_cfg = icfg.get("sampling") if isinstance(icfg.get("sampling"), dict) else None
        if sampling_cfg:
            if "sampling_mode" not in provided_dests and sampling_cfg.get("mode", None) is not None:
                args.sampling_mode = str(sampling_cfg.get("mode") or "")
            top_cfg = sampling_cfg.get("top_actions", {})
            if isinstance(top_cfg, dict):
                if "top_actions_n" not in provided_dests and top_cfg.get("n", None) is not None:
                    args.top_actions_n = int(top_cfg.get("n"))
                if "top_actions_op_topk" not in provided_dests and top_cfg.get("op_topk", None) is not None:
                    args.top_actions_op_topk = int(top_cfg.get("op_topk"))
                if "top_actions_fg_topk" not in provided_dests and top_cfg.get("fg_topk", None) is not None:
                    args.top_actions_fg_topk = int(top_cfg.get("fg_topk"))
                if "top_actions_remove_topk" not in provided_dests and top_cfg.get("remove_topk", None) is not None:
                    args.top_actions_remove_topk = int(top_cfg.get("remove_topk"))
                if "top_actions_fallback_iterative" not in provided_dests and top_cfg.get("fallback_to_iterative", None) is not None:
                    if _as_bool(top_cfg.get("fallback_to_iterative")):
                        args.top_actions_fallback_iterative = True
            conf_cfg = sampling_cfg.get("conf_beam", {})
            if isinstance(conf_cfg, dict):
                if "conf_beam_size" not in provided_dests and conf_cfg.get("beam", None) is not None:
                    args.conf_beam_size = int(conf_cfg.get("beam"))
                if "conf_beam_topk" not in provided_dests and conf_cfg.get("topk", None) is not None:
                    args.conf_beam_topk = int(conf_cfg.get("topk"))
                if "conf_beam_alpha" not in provided_dests and conf_cfg.get("alpha", None) is not None:
                    args.conf_beam_alpha = float(conf_cfg.get("alpha"))
                if "conf_beam_tau" not in provided_dests and conf_cfg.get("tau", None) is not None:
                    args.conf_beam_tau = float(conf_cfg.get("tau"))
                if "conf_beam_r" not in provided_dests and conf_cfg.get("r", None) is not None:
                    args.conf_beam_r = float(conf_cfg.get("r"))
                if "conf_beam_max_steps" not in provided_dests and conf_cfg.get("max_steps", None) is not None:
                    args.conf_beam_max_steps = int(conf_cfg.get("max_steps"))
            best_cfg = sampling_cfg.get("best_first", {})
            if isinstance(best_cfg, dict):
                if "best_first_open_size" not in provided_dests and best_cfg.get("open_size", None) is not None:
                    args.best_first_open_size = int(best_cfg.get("open_size"))
                if "best_first_expand" not in provided_dests and best_cfg.get("expand", None) is not None:
                    args.best_first_expand = int(best_cfg.get("expand"))
                if "best_first_topk" not in provided_dests and best_cfg.get("topk", None) is not None:
                    args.best_first_topk = int(best_cfg.get("topk"))
                if "best_first_alpha" not in provided_dests and best_cfg.get("alpha", None) is not None:
                    args.best_first_alpha = float(best_cfg.get("alpha"))
                if "best_first_lambda" not in provided_dests and best_cfg.get("lambda", None) is not None:
                    args.best_first_lambda = float(best_cfg.get("lambda"))
                if "best_first_beta" not in provided_dests and best_cfg.get("beta", None) is not None:
                    args.best_first_beta = float(best_cfg.get("beta"))
                if "best_first_max_steps" not in provided_dests and best_cfg.get("max_steps", None) is not None:
                    args.best_first_max_steps = int(best_cfg.get("max_steps"))
                if "best_first_max_expansions" not in provided_dests and best_cfg.get("max_expansions", None) is not None:
                    args.best_first_max_expansions = int(best_cfg.get("max_expansions"))

        ctrl_cfg = icfg.get("control")
        if isinstance(ctrl_cfg, dict):
            if any(k in ctrl_cfg for k in ("inputs", "input", "smiles", "smiles_tagged")):
                control_inline = ctrl_cfg
            else:
                ctrl_path = ctrl_cfg.get("yaml", ctrl_cfg.get("path"))
                if ctrl_path and "control_config" not in provided_dests:
                    args.control_config = str(ctrl_path)
        if "control_config" not in provided_dests and icfg.get("control_config", None) is not None:
            args.control_config = str(icfg.get("control_config"))
        if "control_config" not in provided_dests and icfg.get("control_yaml", None) is not None:
            args.control_config = str(icfg.get("control_yaml"))

    if not getattr(args, "sampling_mode", ""):
        args.sampling_mode = "iterative"

    if control_inline is not None:
        control_inputs_raw = _normalize_control_inputs(control_inline)
    elif args.control_config:
        ctrl_path = Path(str(args.control_config))
        if not ctrl_path.is_absolute():
            base_dir = Path(args.infer_config).parent if args.infer_config else Path.cwd()
            ctrl_path = base_dir / ctrl_path
        control_cfg = _load_control_config(ctrl_path)
        control_inputs_raw = _normalize_control_inputs(control_cfg)
        prof.meta["control_config"] = str(ctrl_path)

    if not args.tasks and tasks_inline is None:
        raise ValueError("--tasks is required (or provide task_defs via --infer_config)")
    if not args.input_csv and not control_inputs_raw:
        raise ValueError("--input_csv is required (or provide control_config with inputs)")
    if not args.output_dir and not args.output_csv:
        raise ValueError("--output_dir or --output_csv is required (or provide it via --infer_config)")

    rdkit_threads = max(0, int(getattr(args, "rdkit_threads", 0) or 0))
    rdkit_pool = ThreadPoolExecutor(max_workers=rdkit_threads) if rdkit_threads > 0 else None
    if rdkit_pool is not None:
        prof.meta["rdkit_threads"] = int(rdkit_threads)

    with prof.timer("setup.load_base_config"):
        cfg = Config.load(args.config)
        fg_vocab_path = cfg.resolve_path(cfg.raw["chemistry"]["functional_groups_json"])
        fg_vocab = FunctionalGroupVocab.load(fg_vocab_path)
    try:
        fg_smiles_to_id = {str(smi): int(fid) for fid, smi in (getattr(fg_vocab, "id_to_smiles", {}) or {}).items() if smi}
    except Exception:
        fg_smiles_to_id = {}

    with prof.timer("setup.load_tasks"):
        tasks = _load_tasks_from_dict(tasks_inline) if isinstance(tasks_inline, dict) else _load_tasks(str(args.tasks))

    # ---------------- Retrieval bank (optional) ----------------
    retrieval_bank = None
    retrieval_meta = None
    if bool(args.retrieval):
        if not args.retrieval_csv:
            raise ValueError("--retrieval_csv is required when --retrieval is enabled.")
        try:
            from rdkit import Chem, DataStructs
            from rdkit.Chem import AllChem
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Retrieval requires RDKit (Chem, DataStructs, AllChem).") from exc

        import csv as _csv
        import math as _math

        # property_cols comes from checkpoint later; defer bank building until after checkpoint load.
        retrieval_bank = {"csv_path": str(args.retrieval_csv)}
        retrieval_meta = {
            "fp_radius": 2,
            "fp_bits": 2048,
            "eps": 1e-6,
        }

    # Choose which tasks to run.
    if args.task_ids:
        want = [t.strip() for t in str(args.task_ids).split(",") if t.strip()]
        unknown = [t for t in want if t not in tasks]
        if unknown:
            raise KeyError(f"Unknown task_ids={unknown}. Available: {sorted(tasks.keys())}")
        run_task_ids = want
    else:
        if args.task_id is None:
            if len(tasks) != 1:
                raise ValueError(f'--task_id is required when tasks file contains multiple tasks (or use --task_id "all")')
            run_task_ids = [next(iter(tasks.keys()))]
        else:
            tid = str(args.task_id)
            if tid.lower() == "all":
                run_task_ids = sorted(tasks.keys())
            else:
                if tid not in tasks:
                    raise KeyError(f"Unknown task_id={tid}. Available: {sorted(tasks.keys())}")
                run_task_ids = [tid]

    with prof.timer("setup.load_train_config"):
        train_cfg = _load_yaml(args.train_config)
    model_cfg = train_cfg.get("model", {})
    data_cfg = train_cfg.get("data", {}) if isinstance(train_cfg.get("data", {}), dict) else {}
    with prof.timer("setup.load_checkpoint"):
        ckpt = torch.load(cfg.resolve_path(args.checkpoint), map_location="cpu")
    property_cols = list(ckpt.get("property_cols", []))

    # Keep inference conditioning consistent with training-time masking.
    # Priority:
    #   1) checkpoint payload (best source of truth)
    #   2) checkpoint filename heuristic (legacy)
    #   3) train_config.yaml (fallback)
    cfg_mask = bool(data_cfg.get("mask_non_task_props", data_cfg.get("mask_prop_by_task", False)))
    ckpt_mask_raw = ckpt.get("mask_non_task_props", ckpt.get("mask_prop_by_task", None))
    if ckpt_mask_raw is not None:
        try:
            mask_non_task_props = bool(_as_bool(ckpt_mask_raw))
        except Exception:
            mask_non_task_props = bool(ckpt_mask_raw)
    else:
        # Legacy checkpoints didn't store the flag; infer from filename.
        mask_non_task_props = bool(cfg_mask)
        name = Path(str(args.checkpoint)).name.lower()
        if "mask_prop" in name or "maskprop" in name:
            mask_non_task_props = True
        elif "no_mask" in name or "nomask" in name:
            mask_non_task_props = False

    if bool(mask_non_task_props) != bool(cfg_mask):
        print(
            f"[info] mask_non_task_props resolved to {mask_non_task_props} from checkpoint/name; "
            f"train_config had {cfg_mask}."
        )

    # Optional: compute source properties for conditioning (aligned with delta cols).
    props_for_src = sorted({c[: -len("_delta")] for c in property_cols if isinstance(c, str) and c.endswith("_delta")})
    with prof.timer("setup.build_property_fns"):
        property_fns = build_property_fns(list(props_for_src)) if props_for_src else {}
    props_cache: dict[str, dict[str, float]] = {}
    props_cache_lock = threading.Lock()

    def _props_for_smiles(smiles_plain: str) -> dict[str, float]:
        if not property_fns:
            return {}
        key = str(smiles_plain)
        with props_cache_lock:
            cached = props_cache.get(key)
        if cached is not None:
            prof.inc("props.cache_hit")
            return cached
        prof.inc("props.cache_miss")
        try:
            from rdkit import Chem

            with prof.timer("props.calc"):
                mol = Chem.MolFromSmiles(key)
                if mol is None:
                    out: dict[str, float] = {}
                else:
                    out = calc_properties(mol, property_fns)
        except Exception:
            out = {}
        with props_cache_lock:
            props_cache[key] = out
        return out


    # Align MoE config with checkpoint to avoid size mismatches.
    state_dict = ckpt.get("model_state_dict", {})
    ckpt_moe_weight = state_dict.get("moe_gate.weight")
    ckpt_num_experts = int(ckpt_moe_weight.shape[0]) if ckpt_moe_weight is not None else int(model_cfg.get("num_experts", 0))
    name_num_experts, name_topk = _infer_moe_from_checkpoint_name(args.checkpoint)

    # MoE is enabled if requested by config or present in the checkpoint.
    use_moe = bool(model_cfg.get("use_moe", ckpt_moe_weight is not None))

    # num_experts must match checkpoint tensor shapes; trust checkpoint over filename/config.
    num_experts = int(model_cfg.get("num_experts", ckpt_num_experts if ckpt_num_experts > 0 else 0))
    if ckpt_num_experts > 0:
        num_experts = int(ckpt_num_experts)
    if name_num_experts is not None and ckpt_num_experts > 0 and int(name_num_experts) != int(ckpt_num_experts):
        print(
            f"[warn] checkpoint name suggests num_experts={name_num_experts} but state_dict has {ckpt_num_experts}; "
            f"using {ckpt_num_experts}."
        )

    # moe_topk (activated experts) is an inference-time setting; prefer filename hint, else ckpt metadata, else config.
    moe_topk = int(model_cfg.get("moe_topk", 1))
    if ckpt.get("moe_topk", None) is not None:
        try:
            moe_topk = int(ckpt.get("moe_topk"))
        except Exception:
            pass
    if name_topk is not None:
        moe_topk = int(name_topk)
    moe_topk = max(1, min(int(moe_topk), max(1, int(num_experts))))

    num_tasks_cfg = int(model_cfg.get("num_tasks", -1))
    if num_tasks_cfg <= 0:
        num_tasks_cfg = int(ckpt.get("num_tasks", 0))
    task_emb_dim_cfg = int(model_cfg.get("task_emb_dim", -1))
    if task_emb_dim_cfg <= 0:
        task_emb_dim_cfg = int(ckpt.get("task_emb_dim", 32))
    model = OneShotTwoStageEditModel(
        props_dim=len(property_cols),
        fg_vocab_size=fg_vocab.size,
        fg_num_classes=2,
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
        num_layers=int(model_cfg.get("num_layers", 5)),
        gine_layers=model_cfg.get("gine_layers"),
        mlp_layers=model_cfg.get("mlp_layers"),
        num_tasks=num_tasks_cfg,
        task_emb_dim=task_emb_dim_cfg,
        dropout=float(model_cfg.get("dropout", 0.1)),
        backbone=str(model_cfg.get("backbone", "gine")),
        use_moe=use_moe,
        num_experts=num_experts,
        moe_topk=moe_topk,
        moe_gate_temperature=float(model_cfg.get("moe_gate_temperature", 1.0)),
        moe_gate_noise=0.0,
    )
    with prof.timer("setup.load_model_state"):
        _load_state_dict_forgiving(model, ckpt.get("model_state_dict", {}))
    device = torch.device(str(args.device))
    with prof.timer("setup.model_to_device"):
        model.to(device)
    require_torch_geometric()
    from torch_geometric.data import Data

    import pandas as pd

    with prof.timer("setup.load_input_csv"):
        control_by_row: list[dict[str, Any]] | None = None
        if control_inputs_raw:
            rows: list[dict[str, Any]] = []
            control_by_row = []
            for i, item in enumerate(control_inputs_raw):
                smi_tagged = item.get("smiles_tagged")
                smi_plain = item.get("smiles")
                smi = smi_tagged if smi_tagged not in (None, "") else smi_plain
                if smi is None or str(smi).strip() == "":
                    raise ValueError(f"control input #{i} missing smiles_tagged/smiles")
                rows.append({str(args.smiles_col): str(smi)})
                control_by_row.append(_parse_control_spec(item, fg_smiles_to_id))
            df_in = pd.DataFrame(rows)
            prof.meta["control_inputs"] = int(len(rows))
        else:
            df_in = pd.read_csv(cfg.resolve_path(args.input_csv), engine="c")
    if args.smiles_col not in df_in.columns:
        raise KeyError(f"Missing smiles_col={args.smiles_col}. Columns: {list(df_in.columns)}")
    if args.max_rows is not None and int(args.max_rows) >= 0:
        df_in = df_in.head(int(args.max_rows)).reset_index(drop=True)
        prof.meta["max_rows"] = int(args.max_rows)
        if control_by_row is not None:
            control_by_row = control_by_row[: int(len(df_in))]
    prof.meta["input_rows"] = int(len(df_in))

    # Collect per-task outputs: list length must match input rows.
    pred_json_by_task: dict[str, list[str]] = {tid: [] for tid in run_task_ids}
    pred_actions_by_task: dict[str, list[str]] = {tid: [] for tid in run_task_ids}
    op_counts_by_task: dict[str, dict[str, int]] = {tid: {"add": 0, "remove": 0, "replace": 0, "none": 0} for tid in run_task_ids}
    # FG distributions: add/replace have explicit fg_id; remove/replace use inferred removed fragment (if matchable).
    fg_counts_by_task: dict[str, dict[int, int]] = {tid: {} for tid in run_task_ids}  # add+replace combined (legacy)
    add_fg_counts_by_task: dict[str, dict[int, int]] = {tid: {} for tid in run_task_ids}
    replace_fg_counts_by_task: dict[str, dict[int, int]] = {tid: {} for tid in run_task_ids}
    remove_fg_counts_by_task: dict[str, dict[int, int]] = {tid: {} for tid in run_task_ids}
    remove_fg_unknown_by_task: dict[str, int] = {tid: 0 for tid in run_task_ids}
    edit_len_hist_by_task: dict[str, dict[int, int]] = {tid: {} for tid in run_task_ids}
    # Mean number of predicted edit actions per sampled set.
    edit_count_stats_by_task: dict[str, dict[str, float]] = {tid: {"sum": 0.0, "n": 0.0} for tid in run_task_ids}
    # MoE routing distribution (best-effort; present only for MoE checkpoints).
    moe_expert_count_by_task: dict[str, dict[int, int]] = {tid: {} for tid in run_task_ids}
    moe_expert_weight_sum_by_task: dict[str, dict[int, float]] = {tid: {} for tid in run_task_ids}
    moe_entropy_sum_by_task: dict[str, float] = {tid: 0.0 for tid in run_task_ids}
    moe_topk_mass_sum_by_task: dict[str, float] = {tid: 0.0 for tid in run_task_ids}
    moe_n_by_task: dict[str, int] = {tid: 0 for tid in run_task_ids}

    def _accum_moe_stats(task_id: str) -> None:
        # model.sample_edit_sets and _top_action_pool_once set these attributes when available.
        topi = getattr(model, "_last_moe_topi", None)
        topv = getattr(model, "_last_moe_topv", None)
        ent = getattr(model, "_last_moe_entropy", None)
        mass = getattr(model, "_last_moe_topk_mass", None)
        if topi is None or topv is None:
            return
        # We run 1-graph batches; take graph-0 routing.
        try:
            topi0 = topi[0] if isinstance(topi, list) and topi and isinstance(topi[0], list) else topi
            topv0 = topv[0] if isinstance(topv, list) and topv and isinstance(topv[0], list) else topv
            if not isinstance(topi0, list) or not isinstance(topv0, list) or len(topi0) != len(topv0):
                return
            for i, w in zip(topi0, topv0):
                ei = int(i)
                moe_expert_count_by_task[task_id][ei] = int(moe_expert_count_by_task[task_id].get(ei, 0)) + 1
                moe_expert_weight_sum_by_task[task_id][ei] = float(moe_expert_weight_sum_by_task[task_id].get(ei, 0.0)) + float(w)
            if isinstance(ent, list) and ent:
                moe_entropy_sum_by_task[task_id] += float(ent[0])
            if isinstance(mass, list) and mass:
                moe_topk_mass_sum_by_task[task_id] += float(mass[0])
            moe_n_by_task[task_id] = int(moe_n_by_task[task_id]) + 1
        except Exception:
            return

    def _maybe_sync_cuda() -> None:
        if not prof.enabled or not prof.sync_cuda:
            return
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)

    # Build retrieval index once we know which property_cols this checkpoint expects.
    if retrieval_bank is not None and retrieval_meta is not None:
        csv_path = Path(str(retrieval_bank["csv_path"]))
        if not csv_path.exists():
            raise FileNotFoundError(f"retrieval_csv not found: {csv_path}")

        # Only delta columns are conditioned; src props are used for distance.
        delta_cols_only = [c for c in property_cols if isinstance(c, str) and c.endswith("_delta")]
        prop_names = [c[: -len("_delta")] for c in delta_cols_only]
        src_cols = [f"{p}_src" for p in prop_names]

        # Per-task lists: (fp, src_vec[prop_names], delta_vec[prop_names], num_edits, fg_id_list)
        bank: dict[str, list[tuple[object, list[float], list[float], int, list[int]]]] = {}
        # Per-task normalization for src prop distance.
        src_means: dict[str, list[float]] = {}
        src_stds: dict[str, list[float]] = {}

        def _fp_from_smiles(smiles_plain: str):
            m = Chem.MolFromSmiles(str(smiles_plain))
            if m is None:
                return None
            return AllChem.GetMorganFingerprintAsBitVect(m, int(retrieval_meta["fp_radius"]), nBits=int(retrieval_meta["fp_bits"]))

        def _safe_float(x: object) -> float:
            try:
                v = float(x)  # type: ignore[arg-type]
                if v != v or v in (float("inf"), float("-inf")):
                    return 0.0
                return v
            except Exception:
                return 0.0

        def _parse_edits_meta(edits_json: str) -> tuple[int, list[int]]:
            """
            Return (num_edits, fg_ids) for add/replace ops.
            Keep duplicates in fg_ids (acts as a frequency signal), caller may dedupe/cap.
            """
            try:
                edits = json.loads(str(edits_json))
            except Exception:
                return 0, []
            if not isinstance(edits, list):
                return 0, []
            fg_ids: list[int] = []
            for e in edits:
                if not isinstance(e, dict):
                    continue
                op = str(e.get("op", "")).strip()
                if op in ("add", "replace"):
                    fg = e.get("fg_id", None)
                    if fg is None:
                        continue
                    try:
                        fg_ids.append(int(fg))
                    except Exception:
                        pass
            return int(len(edits)), fg_ids

        # First pass: load raw vectors.
        with prof.timer("retrieval.bank_read"):
            with csv_path.open("r", newline="") as f:
                r = _csv.DictReader(f)
                header = set(r.fieldnames or [])
                # Choose a molecule column for retrieval.
                mol_col = "start_smiles"
                if mol_col not in header:
                    mol_col = "source_smiles" if "source_smiles" in header else mol_col
                if mol_col not in header and "start_smiles_tagged" in header:
                    mol_col = "start_smiles_tagged"
                if mol_col not in header:
                    raise KeyError(
                        f"retrieval_csv missing molecule column (need start_smiles/source_smiles/start_smiles_tagged): {csv_path}"
                    )
                if "task_id" not in header:
                    raise KeyError(f"retrieval_csv missing task_id: {csv_path}")
                missing = [c for c in delta_cols_only if c not in header]
                if missing:
                    raise KeyError(f"retrieval_csv missing delta columns: {missing[:10]}{'...' if len(missing)>10 else ''}")
                missing_src = [c for c in src_cols if c not in header]
                if missing_src:
                    raise KeyError(f"retrieval_csv missing src columns: {missing_src[:10]}{'...' if len(missing_src)>10 else ''}")
                if (bool(args.retrieval_use_num_edits) or bool(args.retrieval_use_fg_pool)) and "edits_json" not in header:
                    raise KeyError("retrieval_csv missing edits_json (required for retrieval constraints).")

                for row in r:
                    prof.inc("retrieval.bank_rows_total")
                    tid = str(row.get("task_id", "")).strip()
                    if not tid:
                        continue
                    smi = str(row.get(mol_col, "")).strip()
                    if not smi:
                        continue
                    if mol_col.endswith("_tagged"):
                        try:
                            plain = smiles_without_atom_maps(mol_from_smiles(smi))
                            smi_plain = str(plain)
                        except Exception:
                            smi_plain = smi
                    else:
                        smi_plain = smi
                    fp = _fp_from_smiles(smi_plain)
                    if fp is None:
                        continue
                    src_vec = [_safe_float(row.get(c, 0.0)) for c in src_cols]
                    delta_vec = [_safe_float(row.get(c, 0.0)) for c in delta_cols_only]
                    num_edits, fg_ids = _parse_edits_meta(str(row.get("edits_json", "")))
                    bank.setdefault(tid, []).append((fp, src_vec, delta_vec, int(num_edits), fg_ids))

        with prof.timer("retrieval.bank_compute_stats"):
            # Compute per-task mean/std for src distance normalization.
            for tid, items in bank.items():
                if not items:
                    continue
                d = len(items[0][1])
                means = [0.0] * d
                for _, src_vec, _, _, _ in items:
                    for i, v in enumerate(src_vec):
                        means[i] += float(v)
                means = [m / float(len(items)) for m in means]
                vars_ = [0.0] * d
                for _, src_vec, _, _, _ in items:
                    for i, v in enumerate(src_vec):
                        dv = float(v) - means[i]
                        vars_[i] += dv * dv
                stds = [(_math.sqrt(v / float(max(1, len(items) - 1))) if len(items) > 1 else 1.0) for v in vars_]
                stds = [s if s > float(retrieval_meta["eps"]) else 1.0 for s in stds]
                src_means[tid] = means
                src_stds[tid] = stds

        retrieval_bank = {
            "delta_cols_only": delta_cols_only,
            "prop_names": prop_names,
            "prop_to_idx": {p: i for i, p in enumerate(prop_names)},
            "src_cols": src_cols,
            "bank": bank,
            "src_means": src_means,
            "src_stds": src_stds,
            "fp_radius": int(retrieval_meta["fp_radius"]),
            "fp_bits": int(retrieval_meta["fp_bits"]),
        }

        loaded_rows = int(sum(len(v) for v in bank.values()))
        prof.meta["retrieval.bank_rows_loaded"] = loaded_rows
        prof.meta["retrieval.bank_tasks"] = int(len(bank))
        prof.meta["retrieval.csv"] = str(csv_path)
        print(f"retrieval: loaded {loaded_rows} rows across {len(bank)} tasks from {csv_path}")
    for row_i, row in df_in.iterrows():
        prof.inc("rows.total")
        smiles = str(row[args.smiles_col])
        try:
            with prof.timer("row.prepare_start_smiles_tagged"):
                start_smiles_tagged = _prepare_start_smiles_tagged(smiles)
        except Exception:
            prof.inc("rows.bad_start_smiles")
            empty = json.dumps([], ensure_ascii=False)
            for tid in run_task_ids:
                pred_json_by_task[tid].append(empty)
            continue

        # Compute source property values once per input molecule (used for conditioning).
        try:
            with prof.timer("row.plain_for_props"):
                plain_for_props = smiles_without_atom_maps(mol_from_smiles(start_smiles_tagged))
        except Exception:
            plain_for_props = smiles
        with prof.timer("row.src_props"):
            src_props_by_prop = _props_for_smiles(str(plain_for_props))

        # Build graph features (shared across tasks)
        from .featurize import featurize_tagged_smiles

        with prof.timer("row.featurize"):
            f = featurize_tagged_smiles(start_smiles_tagged)
        with prof.timer("row.mask_mol"):
            mol_for_masks = normalize_bracket_atom_hydrogens(ensure_atom_maps(mol_from_smiles(start_smiles_tagged)))
        from .scaffold_utils import scaffold_mask as _scaffold_mask

        with prof.timer("row.scaffold_masks"):
            scaffold = _scaffold_mask(mol_for_masks)
            add_allowed = torch.tensor(
                [a.GetAtomicNum() > 1 and int(a.GetTotalNumHs()) > 0 for a in mol_for_masks.GetAtoms()],
                dtype=torch.bool,
            )
            remove_allowed = torch.zeros((int(f.z.size(0)),), dtype=torch.bool)
            if int(f.edge_index.numel()) > 0 and int(scaffold.numel()) == int(f.z.size(0)):
                src = f.edge_index[0]
                dst = f.edge_index[1]
                mask_edges = scaffold[src] & (~scaffold[dst])
                if bool(mask_edges.any().item()):
                    remove_allowed.scatter_(0, src[mask_edges], True)

        pred_by_task: dict[str, list[str]] = {}
        pred_actions_by_task_row: dict[str, list[list[dict[str, Any]]]] = {}
        control_spec = control_by_row[int(row_i)] if control_by_row is not None and int(row_i) < int(len(control_by_row)) else None
        control_task_id = str(control_spec.get("task_id")) if isinstance(control_spec, dict) and control_spec.get("task_id") is not None else None
        if control_task_id is not None and control_task_id not in tasks:
            raise ValueError(f"control task_id={control_task_id} not found in tasks: {sorted(tasks.keys())}")

        for tid in run_task_ids:
            if control_task_id is not None and str(tid) != str(control_task_id):
                pred_by_task[tid] = []
                pred_actions_by_task_row[tid] = []
                continue
            task_time_start = time.perf_counter()
            task = tasks[tid]
            task_prop_set = set(str(p) for p in task.props)
            trend_sign_by_prop = {p: (1.0 if bit == "1" else -1.0) for p, bit in zip(task.props, task.trend)}

            def _score_candidate_smiles(out_smiles_plain: str) -> float:
                """
                Heuristic best-of score: maximize number of task property trends satisfied.
                Tie-breaker prefers larger signed improvements.
                """
                if not property_fns or not task.props:
                    return 0.0
                props_out = _props_for_smiles(out_smiles_plain)
                if not props_out:
                    return float("-inf")
                ok = 0
                margin = 0.0
                for p in task.props:
                    if p not in props_out:
                        continue
                    sign = float(trend_sign_by_prop.get(p, 0.0))
                    if sign == 0.0:
                        continue
                    d = float(props_out.get(p, 0.0)) - float(src_props_by_prop.get(p, 0.0))
                    if d * sign > 0:
                        ok += 1
                    margin += d * sign
                return float(ok) + 1e-3 * float(margin)

            def _trend_ok_and_margin(out_smiles_plain: str) -> tuple[bool, float]:
                if not property_fns or not task.props:
                    return True, 0.0
                props_out = _props_for_smiles(out_smiles_plain)
                if not props_out:
                    return False, float("-inf")
                ok = 0
                margin = 0.0
                for p in task.props:
                    if p not in props_out:
                        continue
                    sign = float(trend_sign_by_prop.get(p, 0.0))
                    if sign == 0.0:
                        continue
                    d = float(props_out.get(p, 0.0)) - float(src_props_by_prop.get(p, 0.0))
                    if d * sign > 0:
                        ok += 1
                    margin += d * sign
                trend_ok = bool(ok == len(task.props)) if task.props else True
                return trend_ok, float(margin)

            delta_cols_only = [c for c in property_cols if c.endswith("_delta")]
            delta_z = _desired_props_from_task(
                task=task,
                property_cols=delta_cols_only,
                trend_alpha=float(args.trend_alpha),
                device=torch.device("cpu"),
            ).squeeze(0)
            delta_z_by_prop = {c[: -len("_delta")]: float(delta_z[j].item()) for j, c in enumerate(delta_cols_only)}

            # Retrieval-conditioned delta (optional): use a similar (mol + src props) example from retrieval_csv.
            retrieved_delta_by_prop: dict[str, float] | None = None
            retrieved_delta_hypotheses: list[dict[str, float]] | None = None
            retrieved_forced_k: int | None = None
            retrieved_fg_pool: list[int] | None = None
            control_fg_allowed = control_spec.get("fg_allowed_ids") if isinstance(control_spec, dict) else None
            control_delta_by_prop = control_spec.get("delta_props") if isinstance(control_spec, dict) else None
            if retrieval_bank is not None:
                with prof.timer("retrieval.query"):
                    try:
                        from rdkit import Chem, DataStructs
                        from rdkit.Chem import AllChem
                    except Exception:
                        retrieved_delta_by_prop = None
                    else:
                        bank = retrieval_bank.get("bank", {})
                        items = bank.get(str(tid), [])
                        prof.inc("retrieval.query_n")
                        prof.inc("retrieval.items_scanned", int(len(items)))
                        if items:
                            qmol = Chem.MolFromSmiles(str(plain_for_props))
                            qfp = (
                                AllChem.GetMorganFingerprintAsBitVect(
                                    qmol, int(retrieval_bank["fp_radius"]), nBits=int(retrieval_bank["fp_bits"])
                                )
                                if qmol is not None
                                else None
                            )
                            if qfp is not None:
                                # src-prop vector for this task (aligned with retrieval_bank.prop_names)
                                src_vec_q = [float(_finite_or_zero(src_props_by_prop.get(p, 0.0))) for p in retrieval_bank["prop_names"]]
                                stds = retrieval_bank["src_stds"].get(str(tid))
                                if stds is None:
                                    stds = [1.0] * len(src_vec_q)
                                prop_to_idx = retrieval_bank.get("prop_to_idx", {})

                                # Preselect top-M by normalized src-prop distance.
                                import heapq

                                top_m = max(1, int(args.retrieval_top_m))
                                cand = []
                                for fp, src_vec, delta_vec, num_edits, fg_ids in items:
                                    dist = 0.0
                                    for p in task.props:
                                        j = prop_to_idx.get(p)
                                        if j is None:
                                            continue
                                        dist += abs((float(src_vec[j]) - float(src_vec_q[j])) / float(stds[j]))
                                    cand.append((dist, fp, delta_vec, int(num_edits), fg_ids))
                                if len(cand) > top_m:
                                    cand = heapq.nsmallest(top_m, cand, key=lambda x: x[0])
                                else:
                                    cand.sort(key=lambda x: x[0])

                                scored = []
                                for dist, fp, delta_vec, num_edits, fg_ids in cand:
                                    sim = float(DataStructs.TanimotoSimilarity(qfp, fp))
                                    score = float(args.retrieval_w_sim) * (1.0 - sim) + float(args.retrieval_w_prop) * float(dist)
                                    scored.append((score, delta_vec, int(num_edits), fg_ids))
                                scored.sort(key=lambda x: x[0])
                                top = scored[: max(1, int(args.retrieval_k))]
                                if top:
                                    # Softmax(-alpha*score) weights.
                                    import math

                                    alpha = float(args.retrieval_alpha)
                                    exps = [math.exp(-alpha * float(s)) for s, *_ in top]
                                    z = sum(exps) if exps else 1.0
                                    wts = [e / z for e in exps]
                                    delta = [0.0] * len(retrieval_bank["prop_names"])
                                    for w, (_, dv, _, _) in zip(wts, top):
                                        for i, v in enumerate(dv):
                                            delta[i] += float(w) * float(v)
                                    retrieved_delta_by_prop = {p: float(delta[i]) for i, p in enumerate(retrieval_bank["prop_names"])}
                                    # Optional: multi-hypothesis deltas (best-of at generation time).
                                    try:
                                        n_h = int(getattr(args, "retrieval_delta_hypotheses", 1))
                                    except Exception:
                                        n_h = 1
                                    n_h = max(1, int(n_h))
                                    strat = str(getattr(args, "retrieval_delta_strategy", "top") or "top").strip().lower()
                                    if n_h > 1:
                                        # Base deltas from topK neighbors (no averaging).
                                        neighbor_deltas = [
                                            {p: float(dv[i]) for i, p in enumerate(retrieval_bank["prop_names"])}
                                            for _, dv, _, _ in top
                                        ]
                                        # Deterministic sampling based on input/task.
                                        det_seed = int(args.seed) + int(row_i) * 10007 + (abs(hash(str(tid))) % 1000003) + 99991
                                        rng = random.Random(int(det_seed))

                                        def _sample_indices_without_replacement(weights: list[float], n: int) -> list[int]:
                                            idxs = list(range(len(weights)))
                                            w = [max(0.0, float(x)) for x in weights]
                                            out: list[int] = []
                                            for _ in range(min(int(n), len(idxs))):
                                                z2 = float(sum(w))
                                                if z2 <= 0:
                                                    out.append(idxs.pop(0))
                                                    w.pop(0)
                                                    continue
                                                r = rng.random() * z2
                                                acc = 0.0
                                                pick_pos = 0
                                                for j, ww in enumerate(w):
                                                    acc += float(ww)
                                                    if r <= acc:
                                                        pick_pos = int(j)
                                                        break
                                                out.append(idxs.pop(pick_pos))
                                                w.pop(pick_pos)
                                            return out

                                        hyps: list[dict[str, float]] = []
                                        if strat in ("top", "topk"):
                                            hyps = neighbor_deltas[:n_h]
                                        elif strat in ("sample", "samplek"):
                                            pick = _sample_indices_without_replacement(wts, n_h)
                                            hyps = [neighbor_deltas[i] for i in pick]
                                        elif strat in ("mean_top", "mean+top"):
                                            hyps = [dict(retrieved_delta_by_prop)]
                                            hyps.extend(neighbor_deltas[: max(0, n_h - 1)])
                                        elif strat in ("mean_sample", "mean+sample"):
                                            hyps = [dict(retrieved_delta_by_prop)]
                                            pick = _sample_indices_without_replacement(wts, max(0, n_h - 1))
                                            hyps.extend(neighbor_deltas[i] for i in pick)
                                        else:
                                            hyps = neighbor_deltas[:n_h]
                                        # Always keep at least one hypothesis.
                                        retrieved_delta_hypotheses = hyps or [dict(retrieved_delta_by_prop)]
                                    if bool(args.retrieval_use_num_edits):
                                        mean_edits = sum(int(ne) for _, _, ne, _ in top) / float(len(top))
                                        retrieved_forced_k = int(round(float(mean_edits)))
                                    if bool(args.retrieval_use_fg_pool):
                                        fg_all: list[int] = []
                                        for _, _, _, fg_ids in top:
                                            for fg in fg_ids:
                                                try:
                                                    fg_all.append(int(fg))
                                                except Exception:
                                                    pass
                                        if fg_all:
                                            if int(args.retrieval_fg_max_ids) > 0:
                                                from collections import Counter

                                                c = Counter(fg_all)
                                                fg_all = [fg for fg, _ in c.most_common(int(args.retrieval_fg_max_ids))]
                                            else:
                                                fg_all = sorted(set(fg_all))
                                            retrieved_fg_pool = fg_all

            fg_allowed_ids = control_fg_allowed if control_fg_allowed is not None else retrieved_fg_pool
            if isinstance(control_delta_by_prop, dict) and control_delta_by_prop:
                retrieved_delta_by_prop = dict(control_delta_by_prop)
                retrieved_delta_hypotheses = None

            def _build_data_with_sampled_delta(
                *, sample_seed: int, retrieved_delta_override: dict[str, float] | None = None, control: dict[str, Any] | None = None
            ):
                rng = random.Random(int(sample_seed))
                use_retrieved = retrieved_delta_override if retrieved_delta_override is not None else retrieved_delta_by_prop
                control_delta = control.get("delta_props") if isinstance(control, dict) else None

                delta_raw_by_prop: dict[str, float] = {}
                for c in property_cols:
                    prop_name = c[: -len("_delta")] if c.endswith("_delta") else c
                    if control_delta is not None and prop_name in control_delta:
                        delta_raw_by_prop[prop_name] = float(control_delta.get(prop_name, 0.0))
                    elif use_retrieved is not None and c.endswith("_delta"):
                        delta_raw_by_prop[prop_name] = float(use_retrieved.get(prop_name, 0.0))
                    elif prop_name in delta_raw_ranges and prop_name in trend_sign_by_prop:
                        vmin, vmax = delta_raw_ranges[prop_name]
                        mag = float(rng.uniform(float(vmin), float(vmax)))
                        delta_raw_by_prop[prop_name] = float(trend_sign_by_prop[prop_name]) * mag
                    else:
                        delta_raw_by_prop[prop_name] = float(delta_z_by_prop.get(prop_name, 0.0))
                if mask_non_task_props:
                    # Match training-time masking: non-task property deltas are always 0.
                    for prop_name in list(delta_raw_by_prop.keys()):
                        if prop_name not in task_prop_set:
                            delta_raw_by_prop[prop_name] = 0.0

                props_raw = torch.zeros((1, len(property_cols)), dtype=torch.float32)
                props_src_raw = torch.zeros((1, len(property_cols)), dtype=torch.float32)
                for i, c in enumerate(property_cols):
                    prop_name = c[: -len("_delta")] if c.endswith("_delta") else c
                    props_raw[0, i] = float(delta_raw_by_prop.get(prop_name, 0.0))
                    if c.endswith("_delta"):
                        props_src_raw[0, i] = float(_finite_or_zero(src_props_by_prop.get(prop_name, 0.0)))

                props = props_raw.to(device)
                props_src = props_src_raw.to(device)

                with prof.timer("data.build_data"):
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
                        task_id=torch.tensor(int(tid) if str(tid).isdigit() else -1, dtype=torch.long),
                    )
                    data.op_allowed = _compute_op_allowed(data)
                    data.op_allowed = _apply_control_to_op_allowed(data.op_allowed, data.atom_map, control)
                    data.anchor_mask = data.op_allowed[:, 1:].any(dim=-1)
                    if bool((~data.anchor_mask).any().item()):
                        op_allowed = data.op_allowed.clone()
                        op_allowed[~data.anchor_mask, :] = False
                        op_allowed[~data.anchor_mask, OP_TO_ID["none"]] = True
                        data.op_allowed = op_allowed
                    return data

            def _accum_moe_snapshot(task_id: str, snap: dict[str, Any] | None) -> None:
                if not snap:
                    return
                topi = snap.get("topi")
                topv = snap.get("topv")
                ent = snap.get("ent")
                mass = snap.get("mass")
                if topi is None or topv is None:
                    return
                try:
                    topi0 = topi[0] if isinstance(topi, list) and topi and isinstance(topi[0], list) else topi
                    topv0 = topv[0] if isinstance(topv, list) and topv and isinstance(topv[0], list) else topv
                    if not isinstance(topi0, list) or not isinstance(topv0, list) or len(topi0) != len(topv0):
                        return
                    for i, w in zip(topi0, topv0):
                        ei = int(i)
                        moe_expert_count_by_task[task_id][ei] = int(moe_expert_count_by_task[task_id].get(ei, 0)) + 1
                        moe_expert_weight_sum_by_task[task_id][ei] = float(moe_expert_weight_sum_by_task[task_id].get(ei, 0.0)) + float(w)
                    if isinstance(ent, list) and ent:
                        moe_entropy_sum_by_task[task_id] += float(ent[0])
                    if isinstance(mass, list) and mass:
                        moe_topk_mass_sum_by_task[task_id] += float(mass[0])
                    moe_n_by_task[task_id] = int(moe_n_by_task[task_id]) + 1
                except Exception:
                    return

            def _record_sample(task_id: str, *, out_smi: str, edits: list[dict[str, Any]], moe_snapshot: dict[str, Any] | None) -> None:
                edit_count_stats_by_task[task_id]["sum"] += float(len(edits))
                edit_count_stats_by_task[task_id]["n"] += 1.0
                edit_len = int(len(edits))
                edit_len_hist_by_task[task_id][edit_len] = int(edit_len_hist_by_task[task_id].get(edit_len, 0)) + 1
                for e in edits:
                    op = str(e.get("op", "none"))
                    if op not in op_counts_by_task[task_id]:
                        op = "none"
                    op_counts_by_task[task_id][op] += 1
                    if op in ("add", "replace") and "fg_id" in e:
                        fg_id = int(e["fg_id"])
                        fg_counts_by_task[task_id][fg_id] = int(fg_counts_by_task[task_id].get(fg_id, 0)) + 1
                        if op == "add":
                            add_fg_counts_by_task[task_id][fg_id] = int(add_fg_counts_by_task[task_id].get(fg_id, 0)) + 1
                        else:
                            replace_fg_counts_by_task[task_id][fg_id] = int(replace_fg_counts_by_task[task_id].get(fg_id, 0)) + 1
                    if op in ("remove", "replace"):
                        rm_map = e.get("removed_atom_map")
                        if rm_map is not None:
                            fg_smi = _infer_removed_fg_smiles(
                                start_smiles_tagged=start_smiles_tagged,
                                anchor_atom_map=int(e.get("anchor_atom_map", -1)),
                                removed_atom_map=int(rm_map),
                            )
                            if fg_smi and fg_smi in fg_smiles_to_id:
                                rid = int(fg_smiles_to_id[fg_smi])
                                remove_fg_counts_by_task[task_id][rid] = int(remove_fg_counts_by_task[task_id].get(rid, 0)) + 1
                            else:
                                remove_fg_unknown_by_task[task_id] = int(remove_fg_unknown_by_task[task_id]) + 1
                _accum_moe_snapshot(task_id, moe_snapshot)

            enforce_unique = not bool(args.allow_duplicates)
            enforce_valid = not bool(args.allow_invalid_smiles)
            allow_fewer = bool(args.allow_fewer_than_k)
            k_want = int(args.k)
            if k_want < 0:
                raise ValueError("--k must be >= 0")
                if k_want == 0:
                    pred_by_task[tid] = []
                    continue

            max_total_attempts = (
                int(args.max_total_attempts)
                if args.max_total_attempts is not None
                else int(args.max_attempts_per_sample) * int(k_want) * 50
            )
            max_total_attempts = max(1, int(max_total_attempts))

            pred_smiles: list[str] = []
            pred_actions: list[list[dict[str, Any]]] = []
            seen: set[str] = set()
            attempts = 0
            base_seed = int(args.seed) + int(row_i) * 10007 + (abs(hash(tid)) % 1000003)
            sampling_mode = str(getattr(args, "sampling_mode", "iterative") or "iterative").strip().lower()
            if sampling_mode in ("step", "step_by_step", "stepbystep"):
                sampling_mode = "rollout"
            if sampling_mode in ("confidence_beam", "confidence_beam_search", "conf_beam_search"):
                sampling_mode = "conf_beam"
            if sampling_mode in ("bestfirst", "best-first"):
                sampling_mode = "best_first"

            # Multi-hypothesis retrieval: generate candidates per delta hypothesis, then best-of rank.
            delta_hyps = retrieved_delta_hypotheses if retrieved_delta_hypotheses else ([retrieved_delta_by_prop] if retrieved_delta_by_prop is not None else [None])
            if retrieved_delta_by_prop is None:
                # No retrieval delta available -> single hypothesis (uses delta_raw_ranges / trend / delta_z).
                delta_hyps = [None]
            if not delta_hyps:
                delta_hyps = [retrieved_delta_by_prop] if retrieved_delta_by_prop is not None else [None]
            try:
                n_hyps = int(getattr(args, "retrieval_delta_hypotheses", 1))
            except Exception:
                n_hyps = 1
            if n_hyps <= 1:
                # Preserve legacy behavior: use weighted-mean delta only.
                delta_hyps = [retrieved_delta_by_prop] if retrieved_delta_by_prop is not None else [None]

            candidates: list[_Candidate] = []
            max_total_attempts_per_hyp = max(1, int(max_total_attempts) // max(1, len(delta_hyps)))

            def _snapshot_moe(graph_i: int | None = None) -> dict[str, Any]:
                def _pick(v: Any):
                    if graph_i is None:
                        return v
                    if isinstance(v, list) and len(v) > int(graph_i):
                        return v[int(graph_i)]
                    return v

                return {
                    "topi": _pick(getattr(model, "_last_moe_topi", None)),
                    "topv": _pick(getattr(model, "_last_moe_topv", None)),
                    "ent": _pick(getattr(model, "_last_moe_entropy", None)),
                    "mass": _pick(getattr(model, "_last_moe_topk_mass", None)),
                }

            def _try_add_candidate(out_smi: str, edits: list[dict[str, Any]]) -> None:
                with prof.timer("score.candidate"):
                    s = _score_candidate_smiles(str(out_smi))
                candidates.append(_Candidate(smiles=str(out_smi), score=float(s), edits=list(edits), moe_snapshot=_snapshot_moe()))

            # Strategy A (default): iterative sampling, re-sample until we collect k valid/unique molecules.
            # Strategy B: one forward pass, build a top-N action pool, then sample edit-set combinations from it.
            if sampling_mode in ("top_actions", "topactions"):
                last_err = None
                # For each delta hypothesis, build its own pool once and try to sample candidates.
                for h_i, delta_h in enumerate(delta_hyps):
                    attempts_h = 0
                    base_seed_h = int(base_seed) + int(h_i) * 10000019
                    data = _build_data_with_sampled_delta(
                        sample_seed=base_seed_h,
                        retrieved_delta_override=delta_h,
                        control=control_spec,
                    )
                    with prof.timer("sampling.top_actions.pool_once", sync_fn=_maybe_sync_cuda):
                        pool = _top_action_pool_once(
                            model,
                            data,
                            fg_vocab=fg_vocab,
                            pool_n=int(getattr(args, "top_actions_n", 10)),
                            op_topk=int(getattr(args, "top_actions_op_topk", 2)),
                            fg_topk=int(getattr(args, "top_actions_fg_topk", 3)),
                            remove_topk=int(getattr(args, "top_actions_remove_topk", 3)),
                            temperature_op=float(args.temperature_op),
                            temperature_fg=float(args.temperature_fg),
                            temperature_remove=float(args.temperature_remove),
                            fg_allowed_ids=fg_allowed_ids,
                        )
                    k_target = retrieved_forced_k
                    if k_target is None:
                        k_target = 1
                    k_target = max(1, min(int(k_target), int(args.max_edits)))

                    while attempts_h < max_total_attempts_per_hyp and len(candidates) < int(k_want) * int(len(delta_hyps)):
                        attempts += 1
                        attempts_h += 1
                        if not pool:
                            break
                        rng = random.Random(base_seed_h + attempts_h * 104729)
                        with prof.timer("sampling.top_actions.pick_edit_set"):
                            edits = _pick_edit_set_from_pool(pool, k_target=k_target, rng=rng)
                        if not edits:
                            break
                        try:
                            with prof.timer("chem.apply_edits"):
                                tagged = apply_edits(start_smiles_tagged, edits)
                                plain = smiles_without_atom_maps(mol_from_smiles(tagged))
                        except Exception as exc:
                            prof.record_error(
                                phase="apply_edits",
                                exc=exc,
                                context={
                                    "mode": "top_actions",
                                    "row_i": int(row_i),
                                    "task_id": str(tid),
                                    "attempt": int(attempts),
                                    "smiles": str(smiles),
                                    "edits": edits,
                                },
                            )
                            last_err = exc
                            continue

                        with prof.timer("chem.canonicalize"):
                            canon = _canonicalize_smiles(plain)
                        if enforce_valid and canon is None:
                            prof.inc("reject.invalid_smiles")
                            continue
                        out_smi = canon if canon is not None else str(plain).strip()
                        if enforce_unique:
                            key = str(out_smi)
                            if key in seen:
                                prof.inc("reject.duplicate")
                                continue
                            seen.add(key)
                        _try_add_candidate(str(out_smi), edits)

                # Optional fallback to iterative sampling if the pool couldn't yield enough valid/unique outputs.
                if len(candidates) < k_want and bool(getattr(args, "top_actions_fallback_iterative", False)):
                    sampling_mode = "iterative"
                    prof.inc("sampling.top_actions.fallback_to_iterative")

                if sampling_mode == "top_actions" and len(candidates) < k_want and (not allow_fewer):
                    msg = (
                        f"Failed to sample {k_want} unique/valid SMILES for task_id={tid} "
                        f"using top_actions pool (got {len(candidates)}) within max_total_attempts={max_total_attempts}."
                    )
                    if last_err is not None:
                        msg += f" Last error: {type(last_err).__name__}: {last_err}"
                    raise RuntimeError(msg)

            if sampling_mode == "iterative":
                # Always sample one-at-a-time so we can enforce uniqueness and validity.
                last_err: Exception | None = None
                for h_i, delta_h in enumerate(delta_hyps):
                    attempts_h = 0
                    # Each hypothesis gets an independent seed stream, but total compute is capped.
                    base_seed_h = int(base_seed) + int(h_i) * 10000019
                    while attempts_h < max_total_attempts_per_hyp and len(candidates) < int(k_want) * int(len(delta_hyps)):
                        si = len(candidates)
                        attempt_seed = base_seed_h + int(si) * 100003 + int(attempts_h) * 31
                        data = (
                            _build_data_with_sampled_delta(
                                sample_seed=attempt_seed,
                                retrieved_delta_override=delta_h,
                                control=control_spec,
                            )
                            if delta_raw_ranges
                            else _build_data_with_sampled_delta(
                                sample_seed=base_seed_h,
                                retrieved_delta_override=delta_h,
                                control=control_spec,
                            )
                        )
                        ok = False
                        for retry in range(int(args.max_attempts_per_sample)):
                            attempts += 1
                            attempts_h += 1
                            if attempts_h > max_total_attempts_per_hyp:
                                break
                            try:
                                with prof.timer("sampling.sample_edit_sets", sync_fn=_maybe_sync_cuda):
                                    edits = _sample_actions_k(
                                        model,
                                        data,
                                        fg_vocab=fg_vocab,
                                        k=1,
                                        max_edits=int(args.max_edits),
                                        temperature_op=float(args.temperature_op),
                                        temperature_fg=float(args.temperature_fg),
                                        temperature_remove=float(args.temperature_remove),
                                        seed=attempt_seed + int(retry),
                                        forced_k_target=retrieved_forced_k,
                                        fg_allowed_ids=fg_allowed_ids,
                                    )[0]
                            except Exception as exc:
                                prof.record_error(
                                    phase="sample_edit_sets",
                                    exc=exc,
                                    context={
                                        "mode": "iterative",
                                        "row_i": int(row_i),
                                        "task_id": str(tid),
                                        "attempt_seed": int(attempt_seed),
                                        "retry": int(retry),
                                        "smiles": str(smiles),
                                    },
                                )
                                last_err = exc
                                continue

                            try:
                                with prof.timer("chem.apply_edits"):
                                    tagged = apply_edits(start_smiles_tagged, edits)
                                    plain = smiles_without_atom_maps(mol_from_smiles(tagged))
                            except Exception as exc:
                                prof.record_error(
                                    phase="apply_edits",
                                    exc=exc,
                                    context={
                                        "mode": "iterative",
                                        "row_i": int(row_i),
                                        "task_id": str(tid),
                                        "attempt_seed": int(attempt_seed),
                                        "retry": int(retry),
                                        "smiles": str(smiles),
                                        "edits": edits,
                                    },
                                )
                                last_err = exc
                                continue

                            with prof.timer("chem.canonicalize"):
                                canon = _canonicalize_smiles(plain)

                            if enforce_valid and canon is None:
                                prof.inc("reject.invalid_smiles")
                                continue
                            out_smi = canon if canon is not None else str(plain).strip()
                            if enforce_unique:
                                key = str(out_smi)
                                if key in seen:
                                    prof.inc("reject.duplicate")
                                    continue
                                seen.add(key)

                            _try_add_candidate(str(out_smi), edits)
                            ok = True
                            break
                        if not ok:
                            continue

                if not candidates and (not allow_fewer):
                    msg = (
                        f"Failed to sample {k_want} unique/valid SMILES for task_id={tid} "
                        f"after {attempts} attempts (row={row_i}, smiles={smiles!r})."
                    )
                    if last_err is not None:
                        msg += f" Last error: {type(last_err).__name__}: {last_err}"
                    raise RuntimeError(msg)
            elif sampling_mode in ("rollout", "step-by-step", "step_by_step", "stepbystep"):
                last_err = None
                rollout_steps = int(getattr(args, "rollout_steps", 0) or 0)
                if rollout_steps <= 0:
                    rollout_steps = int(args.max_edits)
                rollout_steps = max(1, int(rollout_steps))
                beam_size = max(1, int(getattr(args, "rollout_size", 8) or 8))
                rollout_batch_size = max(1, int(getattr(args, "rollout_batch_size", 8) or 8))
                # Disable per-state expansion; sample a single edit per beam state per step.
                expand_per_state = 1
                scaffold_mode = str(getattr(args, "rollout_scaffold_mode", "fixed") or "fixed").strip().lower()
                if scaffold_mode not in ("fixed", "dynamic"):
                    scaffold_mode = "fixed"
                prof.meta["rollout.scaffold_mode"] = str(scaffold_mode)

                fixed_scaffold_maps: set[int] | None = None
                if scaffold_mode == "fixed":
                    fixed_scaffold_maps = set()
                    try:
                        for ai, atom in enumerate(mol_for_masks.GetAtoms()):
                            if ai < int(scaffold.numel()) and bool(scaffold[ai].item()):
                                amap = int(atom.GetAtomMapNum())
                                if amap > 0:
                                    fixed_scaffold_maps.add(amap)
                    except Exception:
                        fixed_scaffold_maps = None

                def _sample_props_tensors(*, sample_seed: int, retrieved_delta_override: dict[str, float] | None = None):
                    rng = random.Random(int(sample_seed))
                    use_retrieved = retrieved_delta_override if retrieved_delta_override is not None else retrieved_delta_by_prop
                    control_delta = control_spec.get("delta_props") if isinstance(control_spec, dict) else None

                    delta_raw_by_prop: dict[str, float] = {}
                    for c in property_cols:
                        prop_name = c[: -len("_delta")] if c.endswith("_delta") else c
                        if control_delta is not None and prop_name in control_delta:
                            delta_raw_by_prop[prop_name] = float(control_delta.get(prop_name, 0.0))
                        elif use_retrieved is not None and c.endswith("_delta"):
                            delta_raw_by_prop[prop_name] = float(use_retrieved.get(prop_name, 0.0))
                        elif prop_name in delta_raw_ranges and prop_name in trend_sign_by_prop:
                            vmin, vmax = delta_raw_ranges[prop_name]
                            mag = float(rng.uniform(float(vmin), float(vmax)))
                            delta_raw_by_prop[prop_name] = float(trend_sign_by_prop[prop_name]) * mag
                        else:
                            delta_raw_by_prop[prop_name] = float(delta_z_by_prop.get(prop_name, 0.0))
                    if mask_non_task_props:
                        for prop_name in list(delta_raw_by_prop.keys()):
                            if prop_name not in task_prop_set:
                                delta_raw_by_prop[prop_name] = 0.0

                    props_raw = torch.zeros((1, len(property_cols)), dtype=torch.float32)
                    props_src_raw = torch.zeros((1, len(property_cols)), dtype=torch.float32)
                    for i, c in enumerate(property_cols):
                        prop_name = c[: -len("_delta")] if c.endswith("_delta") else c
                        props_raw[0, i] = float(delta_raw_by_prop.get(prop_name, 0.0))
                        if c.endswith("_delta"):
                            props_src_raw[0, i] = float(_finite_or_zero(src_props_by_prop.get(prop_name, 0.0)))
                    return props_raw.to(device), props_src_raw.to(device)

                def _build_data_from_smiles(
                    *,
                    smiles_tagged: str,
                    props: "torch.Tensor",
                    props_src: "torch.Tensor",
                ):
                    from .featurize import featurize_tagged_smiles
                    from .scaffold_utils import scaffold_mask as _scaffold_mask

                    with prof.timer("rollout.featurize"):
                        f2 = featurize_tagged_smiles(smiles_tagged)
                    with prof.timer("rollout.mask_mol"):
                        mol_for_masks2 = normalize_bracket_atom_hydrogens(ensure_atom_maps(mol_from_smiles(smiles_tagged)))
                    with prof.timer("rollout.scaffold_masks"):
                        if scaffold_mode == "fixed" and fixed_scaffold_maps:
                            try:
                                amaps2 = f2.atom_map.detach().cpu().tolist()
                                scaffold2 = torch.tensor(
                                    [int(m) in fixed_scaffold_maps for m in amaps2],
                                    dtype=torch.bool,
                                )
                            except Exception:
                                scaffold2 = _scaffold_mask(mol_for_masks2)
                        else:
                            scaffold2 = _scaffold_mask(mol_for_masks2)
                        add_allowed2 = torch.tensor(
                            [a.GetAtomicNum() > 1 and int(a.GetTotalNumHs()) > 0 for a in mol_for_masks2.GetAtoms()],
                            dtype=torch.bool,
                        )
                        remove_allowed2 = torch.zeros((int(f2.z.size(0)),), dtype=torch.bool)
                        if int(f2.edge_index.numel()) > 0 and int(scaffold2.numel()) == int(f2.z.size(0)):
                            src2 = f2.edge_index[0]
                            dst2 = f2.edge_index[1]
                            mask_edges2 = scaffold2[src2] & (~scaffold2[dst2])
                            if bool(mask_edges2.any().item()):
                                remove_allowed2.scatter_(0, src2[mask_edges2], True)

                    with prof.timer("rollout.build_data"):
                        data2 = Data(
                            z=f2.z.to(device),
                            x=f2.x.to(device),
                            edge_index=f2.edge_index.to(device),
                            edge_type=f2.edge_type.to(device),
                            atom_map=f2.atom_map.to(device),
                            scaffold_mask=scaffold2.to(device),
                            add_allowed=add_allowed2.to(device),
                            remove_allowed=remove_allowed2.to(device),
                            props=props,
                            props_src=props_src,
                            task_id=torch.tensor(int(tid) if str(tid).isdigit() else -1, dtype=torch.long, device=device),
                        )
                        data2.op_allowed = _compute_op_allowed(data2)
                        data2.op_allowed = _apply_control_to_op_allowed(data2.op_allowed, data2.atom_map, control_spec)
                        data2.anchor_mask = data2.op_allowed[:, 1:].any(dim=-1)
                        if bool((~data2.anchor_mask).any().item()):
                            op_allowed2 = data2.op_allowed.clone()
                            op_allowed2[~data2.anchor_mask, :] = False
                            op_allowed2[~data2.anchor_mask, OP_TO_ID["none"]] = True
                            data2.op_allowed = op_allowed2
                        return data2

                @dataclass(frozen=True)
                class _RolloutState:
                    smiles_tagged: str
                    smiles_plain: str
                    edits: tuple[dict[str, Any], ...]
                    score: float
                    moe_snapshot: dict[str, Any] | None = None

                def _apply_edits_candidate(
                    *,
                    base_smiles_tagged: str,
                    base_smiles_plain: str,
                    edits1_norm: list[dict[str, Any]],
                    row_i: int,
                    tid: str,
                    step_i: int,
                ):
                    try:
                        with prof.timer("rollout.apply_edits"):
                            tagged2 = apply_edits(base_smiles_tagged, edits1_norm)
                            plain2 = smiles_without_atom_maps(mol_from_smiles(tagged2))
                    except Exception as exc:
                        prof.record_error(
                            phase="rollout.apply_edits",
                            exc=exc,
                            context={
                                "row_i": int(row_i),
                                "task_id": str(tid),
                                "step": int(step_i),
                                "smiles": str(base_smiles_plain),
                                "edits": edits1_norm,
                            },
                        )
                        return None

                    with prof.timer("rollout.canonicalize"):
                        canon2 = _canonicalize_smiles(plain2)
                    if enforce_valid and canon2 is None:
                        prof.inc("reject.invalid_smiles")
                        return None
                    out_smi2 = str(canon2 if canon2 is not None else str(plain2).strip())
                    s2 = float(_score_candidate_smiles(out_smi2))
                    return (out_smi2, tagged2, edits1_norm, s2)

                for h_i, delta_h in enumerate(delta_hyps):
                    base_seed_h = int(base_seed) + int(h_i) * 10000019
                    props_h, props_src_h = _sample_props_tensors(sample_seed=base_seed_h, retrieved_delta_override=delta_h)

                    init_score = float(_score_candidate_smiles(str(smiles)))
                    beam: list[_RolloutState] = [
                        _RolloutState(
                            smiles_tagged=str(start_smiles_tagged),
                            smiles_plain=str(smiles),
                            edits=tuple(),
                            score=float(init_score),
                        )
                    ]

                    for step_i in range(int(rollout_steps)):
                        next_best: dict[str, _RolloutState] = {}
                        for chunk_start in range(0, len(beam), int(rollout_batch_size)):
                            chunk = beam[chunk_start : chunk_start + int(rollout_batch_size)]
                            data_list = []
                            st_list: list[_RolloutState] = []
                            seed_list: list[int] = []
                            for b_i, st in enumerate(chunk, start=chunk_start):
                                attempt_seed = int(base_seed_h) + int(step_i) * 10007 + int(b_i) * 1000003
                                try:
                                    data2 = _build_data_from_smiles(
                                        smiles_tagged=st.smiles_tagged,
                                        props=props_h,
                                        props_src=props_src_h,
                                    )
                                except Exception as exc:
                                    prof.record_error(
                                        phase="rollout.build_data",
                                        exc=exc,
                                        context={
                                            "row_i": int(row_i),
                                            "task_id": str(tid),
                                            "step": int(step_i),
                                            "smiles": str(st.smiles_plain),
                                        },
                                    )
                                    last_err = exc
                                    continue
                                data_list.append(data2)
                                st_list.append(st)
                                seed_list.append(int(attempt_seed))

                            if not data_list:
                                continue

                            try:
                                with prof.timer("rollout.sample_edit_sets", sync_fn=_maybe_sync_cuda):
                                    edit_sets_list = model.sample_edit_sets_batch(
                                        data_list,
                                        fg_vocab=fg_vocab,
                                        k_samples=int(expand_per_state),
                                        max_edits=1,
                                        temperature_op=float(args.temperature_op),
                                        temperature_fg=float(args.temperature_fg),
                                        temperature_remove=float(args.temperature_remove),
                                        seeds=seed_list,
                                        forced_k_target=1,
                                        fg_allowed_ids=fg_allowed_ids,
                                    )
                            except Exception as exc:
                                prof.record_error(
                                    phase="rollout.sample_edit_sets",
                                    exc=exc,
                                    context={"row_i": int(row_i), "task_id": str(tid), "step": int(step_i)},
                                )
                                last_err = exc
                                continue

                            for st_i, edit_sets in enumerate(edit_sets_list):
                                st = st_list[st_i]
                                edit_payloads: list[list[dict[str, Any]]] = []
                                for edits1 in edit_sets:
                                    attempts += 1
                                    if not edits1:
                                        continue
                                    edits1_norm: list[dict[str, Any]] = []
                                    for e in edits1:
                                        e2 = dict(e)
                                        e2.pop("step", None)
                                        edits1_norm.append(e2)
                                    edit_payloads.append(edits1_norm)

                                results = []
                                if rdkit_pool is None:
                                    for edits1_norm in edit_payloads:
                                        res = _apply_edits_candidate(
                                            base_smiles_tagged=st.smiles_tagged,
                                            base_smiles_plain=st.smiles_plain,
                                            edits1_norm=edits1_norm,
                                            row_i=int(row_i),
                                            tid=str(tid),
                                            step_i=int(step_i),
                                        )
                                        if res is not None:
                                            results.append(res)
                                else:
                                    futures = [
                                        rdkit_pool.submit(
                                            _apply_edits_candidate,
                                            base_smiles_tagged=st.smiles_tagged,
                                            base_smiles_plain=st.smiles_plain,
                                            edits1_norm=edits1_norm,
                                            row_i=int(row_i),
                                            tid=str(tid),
                                            step_i=int(step_i),
                                        )
                                        for edits1_norm in edit_payloads
                                    ]
                                    for fut in futures:
                                        res = fut.result()
                                        if res is not None:
                                            results.append(res)

                                for out_smi2, tagged2, edits1_norm, s2 in results:
                                    if enforce_unique and out_smi2 in seen:
                                        prof.inc("reject.duplicate")
                                        continue
                                    merged = [dict(e) for e in st.edits] + edits1_norm
                                    for si, e in enumerate(merged, start=1):
                                        e["step"] = int(si)
                                    key2 = out_smi2
                                    cand_state = _RolloutState(
                                        smiles_tagged=str(tagged2),
                                        smiles_plain=str(out_smi2),
                                        edits=tuple(merged),
                                        score=float(s2),
                                        moe_snapshot=_snapshot_moe(int(st_i)),
                                    )
                                    prev = next_best.get(key2)
                                    if prev is None or float(cand_state.score) > float(prev.score):
                                        next_best[key2] = cand_state

                        if not next_best:
                            break
                        beam = sorted(next_best.values(), key=lambda s: float(s.score), reverse=True)[: int(beam_size)]

                    for st in beam:
                        candidates.append(
                            _Candidate(
                                smiles=str(st.smiles_plain),
                                score=float(st.score),
                                edits=list(st.edits),
                                moe_snapshot=st.moe_snapshot,
                            )
                        )

                if not candidates and (not allow_fewer):
                    msg = (
                        f"Failed to sample {k_want} unique/valid SMILES for task_id={tid} "
                        f"after rollout_steps={rollout_steps} (row={row_i}, smiles={smiles!r})."
                    )
                    if last_err is not None:
                        msg += f" Last error: {type(last_err).__name__}: {last_err}"
                    raise RuntimeError(msg)
            elif sampling_mode in ("conf_beam", "confidence_beam"):
                last_err = None
                conf_beam_size = max(1, int(getattr(args, "conf_beam_size", 2) or 2))
                conf_beam_topk = max(1, int(getattr(args, "conf_beam_topk", 10) or 10))
                conf_beam_alpha = float(getattr(args, "conf_beam_alpha", 1.0) or 1.0)
                conf_beam_tau = float(getattr(args, "conf_beam_tau", 0.0) or 0.0)
                conf_beam_r = float(getattr(args, "conf_beam_r", 0.0) or 0.0)
                conf_beam_steps = int(getattr(args, "conf_beam_max_steps", 0) or 0)
                if conf_beam_steps <= 0:
                    conf_beam_steps = int(args.max_edits)
                conf_beam_steps = max(1, int(conf_beam_steps))

                scaffold_mode = str(getattr(args, "rollout_scaffold_mode", "fixed") or "fixed").strip().lower()
                if scaffold_mode not in ("fixed", "dynamic"):
                    scaffold_mode = "fixed"
                prof.meta["conf_beam.scaffold_mode"] = str(scaffold_mode)

                fixed_scaffold_maps: set[int] | None = None
                if scaffold_mode == "fixed":
                    fixed_scaffold_maps = set()
                    try:
                        for ai, atom in enumerate(mol_for_masks.GetAtoms()):
                            if ai < int(scaffold.numel()) and bool(scaffold[ai].item()):
                                amap = int(atom.GetAtomMapNum())
                                if amap > 0:
                                    fixed_scaffold_maps.add(amap)
                    except Exception:
                        fixed_scaffold_maps = None

                def _sample_props_tensors(*, sample_seed: int, retrieved_delta_override: dict[str, float] | None = None):
                    rng = random.Random(int(sample_seed))
                    use_retrieved = retrieved_delta_override if retrieved_delta_override is not None else retrieved_delta_by_prop
                    control_delta = control_spec.get("delta_props") if isinstance(control_spec, dict) else None

                    delta_raw_by_prop: dict[str, float] = {}
                    for c in property_cols:
                        prop_name = c[: -len("_delta")] if c.endswith("_delta") else c
                        if control_delta is not None and prop_name in control_delta:
                            delta_raw_by_prop[prop_name] = float(control_delta.get(prop_name, 0.0))
                        elif use_retrieved is not None and c.endswith("_delta"):
                            delta_raw_by_prop[prop_name] = float(use_retrieved.get(prop_name, 0.0))
                        elif prop_name in delta_raw_ranges and prop_name in trend_sign_by_prop:
                            vmin, vmax = delta_raw_ranges[prop_name]
                            mag = float(rng.uniform(float(vmin), float(vmax)))
                            delta_raw_by_prop[prop_name] = float(trend_sign_by_prop[prop_name]) * mag
                        else:
                            delta_raw_by_prop[prop_name] = float(delta_z_by_prop.get(prop_name, 0.0))
                    if mask_non_task_props:
                        for prop_name in list(delta_raw_by_prop.keys()):
                            if prop_name not in task_prop_set:
                                delta_raw_by_prop[prop_name] = 0.0

                    props_raw = torch.zeros((1, len(property_cols)), dtype=torch.float32)
                    props_src_raw = torch.zeros((1, len(property_cols)), dtype=torch.float32)
                    for i, c in enumerate(property_cols):
                        prop_name = c[: -len("_delta")] if c.endswith("_delta") else c
                        props_raw[0, i] = float(delta_raw_by_prop.get(prop_name, 0.0))
                        if c.endswith("_delta"):
                            props_src_raw[0, i] = float(_finite_or_zero(src_props_by_prop.get(prop_name, 0.0)))
                    return props_raw.to(device), props_src_raw.to(device)

                def _build_data_from_smiles(
                    *,
                    smiles_tagged: str,
                    props: "torch.Tensor",
                    props_src: "torch.Tensor",
                ):
                    from .featurize import featurize_tagged_smiles
                    from .scaffold_utils import scaffold_mask as _scaffold_mask

                    f2 = featurize_tagged_smiles(smiles_tagged)
                    mol_for_masks2 = normalize_bracket_atom_hydrogens(ensure_atom_maps(mol_from_smiles(smiles_tagged)))
                    if scaffold_mode == "fixed" and fixed_scaffold_maps:
                        try:
                            amaps2 = f2.atom_map.detach().cpu().tolist()
                            scaffold2 = torch.tensor(
                                [int(m) in fixed_scaffold_maps for m in amaps2],
                                dtype=torch.bool,
                            )
                        except Exception:
                            scaffold2 = _scaffold_mask(mol_for_masks2)
                    else:
                        scaffold2 = _scaffold_mask(mol_for_masks2)
                    add_allowed2 = torch.tensor(
                        [a.GetAtomicNum() > 1 and int(a.GetTotalNumHs()) > 0 for a in mol_for_masks2.GetAtoms()],
                        dtype=torch.bool,
                    )
                    remove_allowed2 = torch.zeros((int(f2.z.size(0)),), dtype=torch.bool)
                    if int(f2.edge_index.numel()) > 0 and int(scaffold2.numel()) == int(f2.z.size(0)):
                        src2 = f2.edge_index[0]
                        dst2 = f2.edge_index[1]
                        mask_edges2 = scaffold2[src2] & (~scaffold2[dst2])
                        if bool(mask_edges2.any().item()):
                            remove_allowed2.scatter_(0, src2[mask_edges2], True)

                    data2 = Data(
                        z=f2.z.to(device),
                        x=f2.x.to(device),
                        edge_index=f2.edge_index.to(device),
                        edge_type=f2.edge_type.to(device),
                        atom_map=f2.atom_map.to(device),
                        scaffold_mask=scaffold2.to(device),
                        add_allowed=add_allowed2.to(device),
                        remove_allowed=remove_allowed2.to(device),
                        props=props,
                        props_src=props_src,
                        task_id=torch.tensor(int(tid) if str(tid).isdigit() else -1, dtype=torch.long, device=device),
                    )
                    data2.op_allowed = _compute_op_allowed(data2)
                    data2.op_allowed = _apply_control_to_op_allowed(data2.op_allowed, data2.atom_map, control_spec)
                    data2.anchor_mask = data2.op_allowed[:, 1:].any(dim=-1)
                    if bool((~data2.anchor_mask).any().item()):
                        op_allowed2 = data2.op_allowed.clone()
                        op_allowed2[~data2.anchor_mask, :] = False
                        op_allowed2[~data2.anchor_mask, OP_TO_ID["none"]] = True
                        data2.op_allowed = op_allowed2
                    return data2

                @dataclass(frozen=True)
                class _ConfBeamState:
                    smiles_tagged: str
                    smiles_plain: str
                    edits: tuple[dict[str, Any], ...]
                    sum_log_conf: float
                    last_conf: float | None
                    score: float
                    moe_snapshot: dict[str, Any] | None = None

                for h_i, delta_h in enumerate(delta_hyps):
                    base_seed_h = int(base_seed) + int(h_i) * 10000019
                    props_h, props_src_h = _sample_props_tensors(sample_seed=base_seed_h, retrieved_delta_override=delta_h)

                    init_state = _ConfBeamState(
                        smiles_tagged=str(start_smiles_tagged),
                        smiles_plain=str(smiles),
                        edits=tuple(),
                        sum_log_conf=0.0,
                        last_conf=None,
                        score=0.0,
                    )
                    beam = [init_state]
                    final_states: list[_ConfBeamState] = []

                    for step_i in range(int(conf_beam_steps)):
                        next_candidates: list[_ConfBeamState] = []
                        step_seen: set[str] = set()

                        for b_i, st in enumerate(beam):
                            try:
                                data2 = _build_data_from_smiles(
                                    smiles_tagged=st.smiles_tagged,
                                    props=props_h,
                                    props_src=props_src_h,
                                )
                            except Exception as exc:
                                prof.record_error(
                                    phase="conf_beam.build_data",
                                    exc=exc,
                                    context={
                                        "row_i": int(row_i),
                                        "task_id": str(tid),
                                        "step": int(step_i),
                                        "smiles": str(st.smiles_plain),
                                    },
                                )
                                last_err = exc
                                final_states.append(st)
                                continue

                            with prof.timer("conf_beam.top_actions", sync_fn=_maybe_sync_cuda):
                                pool = _top_action_pool_once(
                                    model,
                                    data2,
                                    fg_vocab=fg_vocab,
                                    pool_n=int(conf_beam_topk),
                                    op_topk=int(getattr(args, "top_actions_op_topk", 2)),
                                    fg_topk=int(getattr(args, "top_actions_fg_topk", 3)),
                                    remove_topk=int(getattr(args, "top_actions_remove_topk", 3)),
                                    temperature_op=float(args.temperature_op),
                                    temperature_fg=float(args.temperature_fg),
                                    temperature_remove=float(args.temperature_remove),
                                    fg_allowed_ids=fg_allowed_ids,
                                )

                            if not pool:
                                final_states.append(st)
                                continue

                            max_conf_next = max(float(math.exp(float(a.get("_score", -1e9)))) for a in pool)
                            if max_conf_next < float(conf_beam_tau):
                                final_states.append(st)
                                continue
                            if st.last_conf is not None and max_conf_next < float(conf_beam_r) * float(st.last_conf):
                                final_states.append(st)
                                continue

                            for action in pool:
                                edits1_norm = [dict(action)]
                                for e in edits1_norm:
                                    e.pop("_score", None)
                                try:
                                    with prof.timer("conf_beam.apply_edits"):
                                        tagged2 = apply_edits(st.smiles_tagged, edits1_norm)
                                        plain2 = smiles_without_atom_maps(mol_from_smiles(tagged2))
                                except Exception as exc:
                                    prof.record_error(
                                        phase="conf_beam.apply_edits",
                                        exc=exc,
                                        context={
                                            "row_i": int(row_i),
                                            "task_id": str(tid),
                                            "step": int(step_i),
                                            "smiles": str(st.smiles_plain),
                                            "edits": edits1_norm,
                                        },
                                    )
                                    last_err = exc
                                    continue

                                with prof.timer("conf_beam.canonicalize"):
                                    canon = _canonicalize_smiles(plain2)
                                if enforce_valid and canon is None:
                                    prof.inc("reject.invalid_smiles")
                                    continue
                                out_smi = str(canon if canon is not None else str(plain2).strip())
                                if out_smi in step_seen:
                                    prof.inc("reject.duplicate")
                                    continue
                                step_seen.add(out_smi)

                                log_conf = float(action.get("_score", 0.0))
                                conf = float(math.exp(log_conf))
                                new_edits = list(st.edits) + edits1_norm
                                for si, e in enumerate(new_edits, start=1):
                                    e["step"] = int(si)
                                t = max(1, len(new_edits))
                                sum_log_conf = float(st.sum_log_conf) + float(log_conf)
                                score = float(sum_log_conf) / float(pow(float(t), float(conf_beam_alpha)))
                                next_candidates.append(
                                    _ConfBeamState(
                                        smiles_tagged=str(tagged2),
                                        smiles_plain=str(out_smi),
                                        edits=tuple(new_edits),
                                        sum_log_conf=float(sum_log_conf),
                                        last_conf=float(conf),
                                        score=float(score),
                                        moe_snapshot=_snapshot_moe(int(b_i)),
                                    )
                                )

                        if not next_candidates:
                            break
                        next_candidates.sort(key=lambda s: float(s.score), reverse=True)
                        beam = next_candidates[: int(conf_beam_size)]

                    final_states.extend(beam)
                    if not final_states and (not allow_fewer):
                        msg = (
                            f"Failed to sample {k_want} unique/valid SMILES for task_id={tid} "
                            f"after conf_beam steps={conf_beam_steps} (row={row_i}, smiles={smiles!r})."
                        )
                        if last_err is not None:
                            msg += f" Last error: {type(last_err).__name__}: {last_err}"
                        raise RuntimeError(msg)

                    for st in final_states:
                        candidates.append(
                            _Candidate(
                                smiles=str(st.smiles_plain),
                                score=float(st.score),
                                edits=list(st.edits),
                                moe_snapshot=st.moe_snapshot,
                            )
                        )
            elif sampling_mode in ("best_first", "bestfirst"):
                last_err = None
                best_open_size = max(1, int(getattr(args, "best_first_open_size", 128) or 128))
                best_expand = max(1, int(getattr(args, "best_first_expand", 2) or 2))
                best_topk = max(1, int(getattr(args, "best_first_topk", 10) or 10))
                best_alpha = float(getattr(args, "best_first_alpha", 1.0) or 1.0)
                best_beta = float(getattr(args, "best_first_beta", 0.0) or 0.0)
                best_steps = int(getattr(args, "best_first_max_steps", 0) or 0)
                if best_steps <= 0:
                    best_steps = int(args.max_edits)
                best_steps = max(1, int(best_steps))
                best_max_expansions = max(1, int(getattr(args, "best_first_max_expansions", 200) or 200))

                scaffold_mode = str(getattr(args, "rollout_scaffold_mode", "fixed") or "fixed").strip().lower()
                if scaffold_mode not in ("fixed", "dynamic"):
                    scaffold_mode = "fixed"
                prof.meta["best_first.scaffold_mode"] = str(scaffold_mode)

                fixed_scaffold_maps: set[int] | None = None
                if scaffold_mode == "fixed":
                    fixed_scaffold_maps = set()
                    try:
                        for ai, atom in enumerate(mol_for_masks.GetAtoms()):
                            if ai < int(scaffold.numel()) and bool(scaffold[ai].item()):
                                amap = int(atom.GetAtomMapNum())
                                if amap > 0:
                                    fixed_scaffold_maps.add(amap)
                    except Exception:
                        fixed_scaffold_maps = None

                def _sample_props_tensors(*, sample_seed: int, retrieved_delta_override: dict[str, float] | None = None):
                    rng = random.Random(int(sample_seed))
                    use_retrieved = retrieved_delta_override if retrieved_delta_override is not None else retrieved_delta_by_prop
                    control_delta = control_spec.get("delta_props") if isinstance(control_spec, dict) else None

                    delta_raw_by_prop: dict[str, float] = {}
                    for c in property_cols:
                        prop_name = c[: -len("_delta")] if c.endswith("_delta") else c
                        if control_delta is not None and prop_name in control_delta:
                            delta_raw_by_prop[prop_name] = float(control_delta.get(prop_name, 0.0))
                        elif use_retrieved is not None and c.endswith("_delta"):
                            delta_raw_by_prop[prop_name] = float(use_retrieved.get(prop_name, 0.0))
                        elif prop_name in delta_raw_ranges and prop_name in trend_sign_by_prop:
                            vmin, vmax = delta_raw_ranges[prop_name]
                            mag = float(rng.uniform(float(vmin), float(vmax)))
                            delta_raw_by_prop[prop_name] = float(trend_sign_by_prop[prop_name]) * mag
                        else:
                            delta_raw_by_prop[prop_name] = float(delta_z_by_prop.get(prop_name, 0.0))
                    if mask_non_task_props:
                        for prop_name in list(delta_raw_by_prop.keys()):
                            if prop_name not in task_prop_set:
                                delta_raw_by_prop[prop_name] = 0.0

                    props_raw = torch.zeros((1, len(property_cols)), dtype=torch.float32)
                    props_src_raw = torch.zeros((1, len(property_cols)), dtype=torch.float32)
                    for i, c in enumerate(property_cols):
                        prop_name = c[: -len("_delta")] if c.endswith("_delta") else c
                        props_raw[0, i] = float(delta_raw_by_prop.get(prop_name, 0.0))
                        if c.endswith("_delta"):
                            props_src_raw[0, i] = float(_finite_or_zero(src_props_by_prop.get(prop_name, 0.0)))
                    return props_raw.to(device), props_src_raw.to(device)

                def _build_data_from_smiles(
                    *,
                    smiles_tagged: str,
                    props: "torch.Tensor",
                    props_src: "torch.Tensor",
                ):
                    from .featurize import featurize_tagged_smiles
                    from .scaffold_utils import scaffold_mask as _scaffold_mask

                    f2 = featurize_tagged_smiles(smiles_tagged)
                    mol_for_masks2 = normalize_bracket_atom_hydrogens(ensure_atom_maps(mol_from_smiles(smiles_tagged)))
                    if scaffold_mode == "fixed" and fixed_scaffold_maps:
                        try:
                            amaps2 = f2.atom_map.detach().cpu().tolist()
                            scaffold2 = torch.tensor(
                                [int(m) in fixed_scaffold_maps for m in amaps2],
                                dtype=torch.bool,
                            )
                        except Exception:
                            scaffold2 = _scaffold_mask(mol_for_masks2)
                    else:
                        scaffold2 = _scaffold_mask(mol_for_masks2)
                    add_allowed2 = torch.tensor(
                        [a.GetAtomicNum() > 1 and int(a.GetTotalNumHs()) > 0 for a in mol_for_masks2.GetAtoms()],
                        dtype=torch.bool,
                    )
                    remove_allowed2 = torch.zeros((int(f2.z.size(0)),), dtype=torch.bool)
                    if int(f2.edge_index.numel()) > 0 and int(scaffold2.numel()) == int(f2.z.size(0)):
                        src2 = f2.edge_index[0]
                        dst2 = f2.edge_index[1]
                        mask_edges2 = scaffold2[src2] & (~scaffold2[dst2])
                        if bool(mask_edges2.any().item()):
                            remove_allowed2.scatter_(0, src2[mask_edges2], True)

                    data2 = Data(
                        z=f2.z.to(device),
                        x=f2.x.to(device),
                        edge_index=f2.edge_index.to(device),
                        edge_type=f2.edge_type.to(device),
                        atom_map=f2.atom_map.to(device),
                        scaffold_mask=scaffold2.to(device),
                        add_allowed=add_allowed2.to(device),
                        remove_allowed=remove_allowed2.to(device),
                        props=props,
                        props_src=props_src,
                        task_id=torch.tensor(int(tid) if str(tid).isdigit() else -1, dtype=torch.long, device=device),
                    )
                    data2.op_allowed = _compute_op_allowed(data2)
                    data2.op_allowed = _apply_control_to_op_allowed(data2.op_allowed, data2.atom_map, control_spec)
                    data2.anchor_mask = data2.op_allowed[:, 1:].any(dim=-1)
                    if bool((~data2.anchor_mask).any().item()):
                        op_allowed2 = data2.op_allowed.clone()
                        op_allowed2[~data2.anchor_mask, :] = False
                        op_allowed2[~data2.anchor_mask, OP_TO_ID["none"]] = True
                        data2.op_allowed = op_allowed2
                    return data2

                @dataclass(frozen=True)
                class _BestFirstState:
                    smiles_tagged: str
                    smiles_plain: str
                    edits: tuple[dict[str, Any], ...]
                    sum_log_conf: float
                    step: int
                    score: float
                    moe_snapshot: dict[str, Any] | None = None

                def _bf_score(sum_log_conf: float, step: int) -> float:
                    t = max(1, int(step))
                    return (float(sum_log_conf) / float(pow(float(t), float(best_alpha)))) - float(best_beta) * float(step)

                for h_i, delta_h in enumerate(delta_hyps):
                    base_seed_h = int(base_seed) + int(h_i) * 10000019
                    props_h, props_src_h = _sample_props_tensors(sample_seed=base_seed_h, retrieved_delta_override=delta_h)

                    open_states: list[_BestFirstState] = [
                        _BestFirstState(
                            smiles_tagged=str(start_smiles_tagged),
                            smiles_plain=str(smiles),
                            edits=tuple(),
                            sum_log_conf=0.0,
                            step=0,
                            score=0.0,
                        )
                    ]
                    final_states: list[_BestFirstState] = []
                    best_score_by_smiles: dict[str, float] | None = {str(smiles): 0.0} if enforce_unique else None
                    expansions = 0

                    while open_states and expansions < best_max_expansions:
                        open_states.sort(key=lambda s: float(s.score), reverse=True)
                        to_expand = open_states[: int(best_expand)]
                        open_states = open_states[int(best_expand) :]

                        for st in to_expand:
                            if st.step >= int(best_steps):
                                final_states.append(st)
                                continue
                            try:
                                data2 = _build_data_from_smiles(
                                    smiles_tagged=st.smiles_tagged,
                                    props=props_h,
                                    props_src=props_src_h,
                                )
                            except Exception as exc:
                                prof.record_error(
                                    phase="best_first.build_data",
                                    exc=exc,
                                    context={
                                        "row_i": int(row_i),
                                        "task_id": str(tid),
                                        "step": int(st.step),
                                        "smiles": str(st.smiles_plain),
                                    },
                                )
                                last_err = exc
                                final_states.append(st)
                                continue

                            with prof.timer("best_first.top_actions", sync_fn=_maybe_sync_cuda):
                                pool = _top_action_pool_once(
                                    model,
                                    data2,
                                    fg_vocab=fg_vocab,
                                    pool_n=int(best_topk),
                                    op_topk=int(getattr(args, "top_actions_op_topk", 2)),
                                    fg_topk=int(getattr(args, "top_actions_fg_topk", 3)),
                                    remove_topk=int(getattr(args, "top_actions_remove_topk", 3)),
                                    temperature_op=float(args.temperature_op),
                                    temperature_fg=float(args.temperature_fg),
                                    temperature_remove=float(args.temperature_remove),
                                    fg_allowed_ids=fg_allowed_ids,
                                )
                            expansions += 1

                            if not pool:
                                final_states.append(st)
                                continue

                            any_child = False
                            for action in pool:
                                edits1_norm = [dict(action)]
                                for e in edits1_norm:
                                    e.pop("_score", None)
                                try:
                                    with prof.timer("best_first.apply_edits"):
                                        tagged2 = apply_edits(st.smiles_tagged, edits1_norm)
                                        plain2 = smiles_without_atom_maps(mol_from_smiles(tagged2))
                                except Exception as exc:
                                    prof.record_error(
                                        phase="best_first.apply_edits",
                                        exc=exc,
                                        context={
                                            "row_i": int(row_i),
                                            "task_id": str(tid),
                                            "step": int(st.step),
                                            "smiles": str(st.smiles_plain),
                                            "edits": edits1_norm,
                                        },
                                    )
                                    last_err = exc
                                    continue

                                with prof.timer("best_first.canonicalize"):
                                    canon = _canonicalize_smiles(plain2)
                                if enforce_valid and canon is None:
                                    prof.inc("reject.invalid_smiles")
                                    continue
                                out_smi = str(canon if canon is not None else str(plain2).strip())
                                log_conf = float(action.get("_score", 0.0))
                                new_edits = list(st.edits) + edits1_norm
                                for si, e in enumerate(new_edits, start=1):
                                    e["step"] = int(si)
                                step_n = int(st.step) + 1
                                sum_log_conf = float(st.sum_log_conf) + float(log_conf)
                                score = _bf_score(sum_log_conf, step_n)
                                if best_score_by_smiles is not None:
                                    prev = best_score_by_smiles.get(out_smi)
                                    if prev is not None and float(score) <= float(prev):
                                        continue
                                    best_score_by_smiles[out_smi] = float(score)
                                open_states.append(
                                    _BestFirstState(
                                        smiles_tagged=str(tagged2),
                                        smiles_plain=str(out_smi),
                                        edits=tuple(new_edits),
                                        sum_log_conf=float(sum_log_conf),
                                        step=int(step_n),
                                        score=float(score),
                                        moe_snapshot=_snapshot_moe(None),
                                    )
                                )
                                any_child = True

                            if not any_child:
                                final_states.append(st)

                        if len(open_states) > int(best_open_size):
                            open_states.sort(key=lambda s: float(s.score), reverse=True)
                            open_states = open_states[: int(best_open_size)]

                    if open_states:
                        final_states.extend(open_states)

                    if not final_states and (not allow_fewer):
                        msg = (
                            f"Failed to sample {k_want} unique/valid SMILES for task_id={tid} "
                            f"after best_first expansions={best_max_expansions} (row={row_i}, smiles={smiles!r})."
                        )
                        if last_err is not None:
                            msg += f" Last error: {type(last_err).__name__}: {last_err}"
                        raise RuntimeError(msg)

                    for st in final_states:
                        candidates.append(
                            _Candidate(
                                smiles=str(st.smiles_plain),
                                score=float(st.score),
                                edits=list(st.edits),
                                moe_snapshot=st.moe_snapshot,
                            )
                        )

            # Best-of selection across all candidates (and across retrieval delta hypotheses).
            if candidates:
                use_conf_beam = bool(sampling_mode in ("conf_beam", "confidence_beam"))
                use_best_first = bool(sampling_mode in ("best_first", "bestfirst"))
                pred_smiles = []
                pred_actions = []
                chosen_seen: set[str] = set()
                if use_conf_beam or use_best_first:
                    ranked = []
                    for cand in candidates:
                        trend_ok, margin = _trend_ok_and_margin(str(cand.smiles))
                        ranked.append((int(trend_ok), float(margin), float(cand.score), cand))
                    # if use_best_first:
                    #     trend_keep = sum(1 for ok, _, __, ___ in ranked if ok)
                    #     print(
                    #         f"[best_first] row={int(row_i)} task_id={tid} "
                    #         f"trend_keep={trend_keep} candidates={len(ranked)}"
                    #     )
                    ranked.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
                    for _, __, ___, cand in ranked:
                        key = str(cand.smiles)
                        if enforce_unique and key in chosen_seen:
                            continue
                        chosen_seen.add(key)
                        pred_smiles.append(str(cand.smiles))
                        pred_actions.append(list(cand.edits))
                        _record_sample(tid, out_smi=str(cand.smiles), edits=cand.edits, moe_snapshot=cand.moe_snapshot)
                        if len(pred_smiles) >= k_want:
                            break
                else:
                    candidates_sorted = sorted(candidates, key=lambda c: float(c.score), reverse=True)
                    # Re-enforce uniqueness at selection time, in case duplicates slipped through.
                    for cand in candidates_sorted:
                        key = str(cand.smiles)
                        if enforce_unique and key in chosen_seen:
                            continue
                        chosen_seen.add(key)
                        pred_smiles.append(str(cand.smiles))
                        pred_actions.append(list(cand.edits))
                        _record_sample(tid, out_smi=str(cand.smiles), edits=cand.edits, moe_snapshot=cand.moe_snapshot)
                        if len(pred_smiles) >= k_want:
                            break

            if len(pred_smiles) < k_want:
                if not allow_fewer:
                    raise RuntimeError(
                        f"Failed to sample {k_want} unique/valid SMILES for task_id={tid} "
                        f"(got {len(pred_smiles)}) within max_total_attempts={max_total_attempts}."
                    )
                if len(pred_smiles) == 0 and not bool(args.allow_invalid_smiles):
                    raise RuntimeError(
                        f"Failed to sample any valid SMILES for task_id={tid} within max_total_attempts={max_total_attempts}."
                    )

            pred_by_task[tid] = pred_smiles
            pred_actions_by_task_row[tid] = pred_actions
            prof.inc("sampling.attempts_total", int(attempts))
            prof.inc("sampling.candidates_total", int(len(candidates)))
            prof.inc("sampling.outputs_total", int(len(pred_smiles)))
            elapsed_ms = (time.perf_counter() - task_time_start) * 1000.0
            print(
                f"[sample_time] row={int(row_i)} task_id={tid} time_ms={elapsed_ms:.2f}"
            )

        for tid in run_task_ids:
            pred_json_by_task[tid].append(json.dumps(pred_by_task[tid], ensure_ascii=False))
            if bool(args.output_actions):
                pred_actions_by_task[tid].append(json.dumps(pred_actions_by_task_row[tid], ensure_ascii=False))

    # Output:
    # - If output_dir is set, write one CSV per task with a single column `pred_smiles_list_json`.
    # - Else write a single CSV (legacy).
    if args.output_dir:
        out_dir = cfg.resolve_path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for tid in run_task_ids:
            out_path = out_dir / f"{tid}_task.csv"
            payload = {"pred_smiles_list_json": pred_json_by_task[tid]}
            if bool(args.output_actions):
                payload["pred_actions_list_json"] = pred_actions_by_task[tid]
            pd.DataFrame(payload).to_csv(out_path, index=False, header=None)
        print(f"Wrote {len(run_task_ids)} task files -> {out_dir}")
        stats_dir = out_dir
    else:
        # Legacy combined output for backward compatibility.
        if len(run_task_ids) == 1:
            payload = pred_json_by_task[run_task_ids[0]]
            df_payload: dict[str, Any] = {"pred_smiles_list_json": payload}
            if bool(args.output_actions):
                df_payload["pred_actions_list_json"] = pred_actions_by_task[run_task_ids[0]]
            df_out = pd.DataFrame(df_payload)
        else:
            combined = []
            for i in range(len(df_in)):
                combined.append(
                    json.dumps({tid: json.loads(pred_json_by_task[tid][i]) for tid in run_task_ids}, ensure_ascii=False)
                )
            df_payload = {"pred_smiles_list_json": combined}
            if bool(args.output_actions):
                combined_actions = []
                for i in range(len(df_in)):
                    combined_actions.append(
                        json.dumps(
                            {tid: json.loads(pred_actions_by_task[tid][i]) for tid in run_task_ids},
                            ensure_ascii=False,
                        )
                    )
                df_payload["pred_actions_list_json"] = combined_actions
            df_out = pd.DataFrame(df_payload)

        out_path = cfg.resolve_path(args.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(out_path, index=False,header=None)
        print(f"Wrote -> {out_path}")
        stats_dir = out_path.parent

    if rdkit_pool is not None:
        rdkit_pool.shutdown(wait=True)

    # Profiling output (optional).
    if bool(prof.enabled):
        prof.meta["device"] = str(device)
        prof.meta["sampling_mode"] = str(getattr(args, "sampling_mode", ""))
        prof.meta["k"] = int(getattr(args, "k", 0))
        prof.meta["retrieval.enabled"] = bool(getattr(args, "retrieval", False))
        prof.meta["total_runtime_s"] = float(time.perf_counter() - t_main0)
        try:
            out_prof_path = str(getattr(args, "profile_timing_json", "") or "").strip()
            prof_path = cfg.resolve_path(out_prof_path) if out_prof_path else (stats_dir / "timing_profile.json")
            prof_path.parent.mkdir(parents=True, exist_ok=True)
            prof_path.write_text(json.dumps(prof.as_dict(), indent=2, ensure_ascii=False))
            print(f"[profile] wrote -> {prof_path}")
            prof.print_top(top_k=30)
        except Exception as exc:
            print(f"[warn] failed to write timing_profile.json: {type(exc).__name__}: {exc}")

    # Write per-task distributions for debugging/analysis.
    try:
        import json as _json
        from collections import Counter

        def _topk(counter: dict[int, int], k: int = 30) -> list[tuple[int, int]]:
            return Counter({int(a): int(b) for a, b in (counter or {}).items()}).most_common(int(k))

        stats_out: dict[str, Any] = {
            "checkpoint": str(args.checkpoint),
            "input_csv": str(args.input_csv),
            "k": int(args.k),
            "sampling_mode": str(getattr(args, "sampling_mode", "iterative")),
            "mask_non_task_props": bool(mask_non_task_props),
            "per_task": {},
        }
        for tid in run_task_ids:
            s = float(edit_count_stats_by_task.get(tid, {}).get("sum", 0.0))
            n = float(edit_count_stats_by_task.get(tid, {}).get("n", 0.0))
            mn = int(moe_n_by_task.get(tid, 0))
            ent_avg = float(moe_entropy_sum_by_task.get(tid, 0.0)) / float(mn) if mn > 0 else 0.0
            mass_avg = float(moe_topk_mass_sum_by_task.get(tid, 0.0)) / float(mn) if mn > 0 else 0.0
            moe_counts = moe_expert_count_by_task.get(tid, {}) or {}
            moe_weights = moe_expert_weight_sum_by_task.get(tid, {}) or {}
            moe_top_counts = Counter({int(k): int(v) for k, v in moe_counts.items()}).most_common(20) if moe_counts else []
            moe_top_weight_sum = sorted(((int(k), float(v)) for k, v in moe_weights.items()), key=lambda kv: (-float(kv[1]), int(kv[0])))[:20] if moe_weights else []
            moe_top_weight_avg = (
                sorted(((int(k), float(v) / float(mn)) for k, v in moe_weights.items()), key=lambda kv: (-float(kv[1]), int(kv[0])))[:20]
                if (moe_weights and mn > 0)
                else []
            )
            stats_out["per_task"][str(tid)] = {
                "op_counts": dict(op_counts_by_task.get(tid, {})),
                "add_fg_top": _topk(add_fg_counts_by_task.get(tid, {})),
                "replace_fg_top": _topk(replace_fg_counts_by_task.get(tid, {})),
                "remove_fg_top": _topk(remove_fg_counts_by_task.get(tid, {})),
                "remove_fg_unknown": int(remove_fg_unknown_by_task.get(tid, 0)),
                "edit_len_hist": {str(k): int(v) for k, v in sorted((edit_len_hist_by_task.get(tid, {}) or {}).items())},
                "avg_actions_per_edit_set": (s / n) if n > 0 else 0.0,
                "moe": {
                    "n": int(mn),
                    "entropy_avg": float(ent_avg),
                    "topk_mass_avg": float(mass_avg),
                    "expert_top_counts": moe_top_counts,
                    "expert_top_weight_sum": moe_top_weight_sum,
                    "expert_top_weight_avg": moe_top_weight_avg,
                },
            }
        (stats_dir / "stats_by_task.json").write_text(_json.dumps(stats_out, indent=2, ensure_ascii=False))
    except Exception as exc:
        print(f"[warn] failed to write stats_by_task.json: {type(exc).__name__}: {exc}")

    # Op distribution summary (counts over sampled edit-sets).
    for tid in run_task_ids:
        counts = op_counts_by_task.get(tid, {})
        total = int(sum(int(v) for v in counts.values()))
        if total <= 0:
            print(f"task_id={tid} op_counts={{}}")
            continue
        props = {k: float(v) / float(total) for k, v in counts.items()}
        print(f"task_id={tid} op_counts={counts} op_props={props} total_ops={total}")

        fg_counts = fg_counts_by_task.get(tid, {})
        fg_total = int(sum(int(v) for v in fg_counts.values()))
        if fg_total <= 0:
            print(f"task_id={tid} fg_counts={{}}")
        else:
            top = sorted(fg_counts.items(), key=lambda kv: (-int(kv[1]), int(kv[0])))[:30]
            fg_props_top = [(int(fid), _finite_or_zero(cnt) / float(fg_total)) for fid, cnt in top]
            print(f"task_id={tid} fg_total={fg_total} fg_top_counts={top} fg_top_props={fg_props_top}")

        add_counts = add_fg_counts_by_task.get(tid, {})
        add_total = int(sum(int(v) for v in add_counts.values()))
        if add_total > 0:
            top_add = sorted(add_counts.items(), key=lambda kv: (-int(kv[1]), int(kv[0])))[:30]
            print(f"task_id={tid} add_fg_total={add_total} add_fg_top={top_add}")

        rep_counts = replace_fg_counts_by_task.get(tid, {})
        rep_total = int(sum(int(v) for v in rep_counts.values()))
        if rep_total > 0:
            top_rep = sorted(rep_counts.items(), key=lambda kv: (-int(kv[1]), int(kv[0])))[:30]
            print(f"task_id={tid} replace_fg_total={rep_total} replace_fg_top={top_rep}")

        rm_counts = remove_fg_counts_by_task.get(tid, {})
        rm_total = int(sum(int(v) for v in rm_counts.values()))
        if rm_total > 0 or int(remove_fg_unknown_by_task.get(tid, 0)) > 0:
            top_rm = sorted(rm_counts.items(), key=lambda kv: (-int(kv[1]), int(kv[0])))[:30]
            print(
                f"task_id={tid} remove_fg_total={rm_total} remove_fg_top={top_rm} remove_fg_unknown={int(remove_fg_unknown_by_task.get(tid, 0))}"
            )

        h = edit_len_hist_by_task.get(tid, {})
        if h:
            items = sorted(((int(k), int(v)) for k, v in h.items()), key=lambda kv: kv[0])
            print(f"task_id={tid} edit_len_hist={items}")

        mn = int(moe_n_by_task.get(tid, 0))
        if mn > 0:
            counts_moe = moe_expert_count_by_task.get(tid, {}) or {}
            weights_moe = moe_expert_weight_sum_by_task.get(tid, {}) or {}
            top_cnt = sorted(counts_moe.items(), key=lambda kv: (-int(kv[1]), int(kv[0])))[:10]
            top_w = sorted(((int(k), float(v) / float(mn)) for k, v in weights_moe.items()), key=lambda kv: (-float(kv[1]), int(kv[0])))[:10]
            ent_avg = float(moe_entropy_sum_by_task.get(tid, 0.0)) / float(mn)
            mass_avg = float(moe_topk_mass_sum_by_task.get(tid, 0.0)) / float(mn)
            print(f"task_id={tid} moe_n={mn} moe_entropy_avg={ent_avg:.3f} moe_topk_mass_avg={mass_avg:.3f} moe_top_counts={top_cnt} moe_top_weight_avg={top_w}")

        s = float(edit_count_stats_by_task.get(tid, {}).get("sum", 0.0))
        n = float(edit_count_stats_by_task.get(tid, {}).get("n", 0.0))
        avg = s / max(1.0, n)
        print(f"task_id={tid} avg_actions_per_edit_set={avg:.3f} (sum={s:.0f}, n={n:.0f})")


if __name__ == "__main__":
    main()
