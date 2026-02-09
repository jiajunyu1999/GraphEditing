from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch.nn.parameter import UninitializedParameter

from .config import Config
from .dataset_oneshot import OP_TO_ID, OneShotEditDataset
from .model_oneshot import OneShotTwoStageEditModel
from .pyg_utils import require_torch_geometric
from .vocab import FunctionalGroupVocab


def _disable_rdkit_warnings() -> None:
    try:  # pragma: no cover
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
    except Exception:
        pass


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


def _write_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    _require_yaml()
    import yaml  # type: ignore

    Path(path).write_text(yaml.safe_dump(payload, sort_keys=False))


def _resolve_path(base: Path, maybe_path: str) -> Path:
    p = Path(maybe_path)
    return p if p.is_absolute() else (base / p)


def _task_ids_from_infer_config(infer_cfg: dict[str, Any], *, infer_cfg_path: Path) -> list[str]:
    task_defs = infer_cfg.get("task_defs")
    if isinstance(task_defs, dict):
        taskid_prop = task_defs.get("taskid_prop")
    else:
        tasks_path = infer_cfg.get("tasks")
        if not tasks_path:
            raise ValueError("infer_config must include task_defs or tasks path for eval.")
        tasks_cfg = _load_yaml(_resolve_path(infer_cfg_path.parent, str(tasks_path)))
        taskid_prop = tasks_cfg.get("taskid_prop")

    if not isinstance(taskid_prop, dict) or not taskid_prop:
        raise ValueError("infer_config task definitions are empty or invalid.")

    all_task_ids = sorted(str(t) for t in taskid_prop.keys())

    task_ids_raw = infer_cfg.get("task_ids")
    if task_ids_raw:
        if isinstance(task_ids_raw, list):
            want = [str(t).strip() for t in task_ids_raw if str(t).strip()]
        else:
            want = [t.strip() for t in str(task_ids_raw).split(",") if t.strip()]
        return want

    tid = infer_cfg.get("task_id")
    if tid is None:
        if len(all_task_ids) != 1:
            raise ValueError('infer_config requires task_id when multiple tasks exist (or use "all").')
        return [all_task_ids[0]]
    tid = str(tid)
    if tid.lower() == "all":
        return all_task_ids
    if tid not in all_task_ids:
        raise ValueError(f"infer_config task_id={tid} not in tasks: {all_task_ids}")
    return [tid]


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if float(den) > 0 else 0.0


def _op_class_weights(*, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # Prevent collapse to all-none on imbalanced data.
    return torch.tensor([0.05, 1.0, 1.0, 1.0], device=device, dtype=dtype)


def _build_fg_id_to_class(*, fg_vocab: FunctionalGroupVocab) -> torch.Tensor:
    # Hierarchy removed; keep a flat zero class for backward compatibility.
    return torch.zeros((int(fg_vocab.size),), dtype=torch.long)


def _extract_fg_ids(edits_json: str) -> list[int]:
    try:
        edits = json.loads(str(edits_json))
    except Exception:
        return []
    if not isinstance(edits, list):
        return []
    fg_ids: list[int] = []
    for e in edits:
        if not isinstance(e, dict):
            continue
        if "fg_id" not in e:
            continue
        try:
            fg_ids.append(int(e["fg_id"]))
        except Exception:
            continue
    return fg_ids


def _build_fg_balance_weights(rows, indices: list[int], *, power: float, min_w: float | None, max_w: float | None) -> list[float]:
    counts: dict[int, int] = {}
    for idx in indices:
        for fg_id in _extract_fg_ids(rows[idx].edits_json):
            counts[fg_id] = counts.get(fg_id, 0) + 1
    if not counts:
        return []
    weights: list[float] = []
    for idx in indices:
        fg_ids = _extract_fg_ids(rows[idx].edits_json)
        if not fg_ids:
            w = 1.0
        else:
            inv = [(1.0 / max(1, counts.get(fg, 1))) ** power for fg in fg_ids]
            w = float(sum(inv) / max(1, len(inv)))
        if min_w is not None:
            w = max(float(min_w), w)
        if max_w is not None:
            w = min(float(max_w), w)
        weights.append(w)
    return weights


def _graph_task_ids(batch, *, num_graphs: int) -> torch.Tensor | None:
    if not hasattr(batch, "task_id"):
        return None
    task_id = batch.task_id
    if task_id.dim() == 0:
        task_id = task_id.view(1)
    if int(task_id.numel()) == int(num_graphs):
        return task_id.view(-1)
    if hasattr(batch, "batch") and int(task_id.numel()) == int(batch.batch.numel()):
        task_ids = []
        for g in range(int(num_graphs)):
            idx = (batch.batch == g).nonzero(as_tuple=True)[0]
            if int(idx.numel()) == 0:
                task_ids.append(torch.tensor(-1, device=task_id.device, dtype=task_id.dtype))
            else:
                task_ids.append(task_id[idx[0]])
        return torch.stack(task_ids, dim=0)
    return task_id.view(-1)[: int(num_graphs)]


def _task_weight_tensor(
    task_ids: torch.Tensor | None,
    *,
    task_weights: dict[str, float] | None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if task_ids is None or not task_weights:
        return torch.ones((int(task_ids.numel()) if task_ids is not None else 0), device=device, dtype=dtype)
    weights = []
    for tid in task_ids.tolist():
        tid_int = int(tid)
        if tid_int < 0:
            w = 1.0
        else:
            w = float(task_weights.get(str(tid_int), 1.0))
        weights.append(w)
    return torch.tensor(weights, device=device, dtype=dtype)


def _count_params(model: torch.nn.Module) -> float:
    total = 0
    for p in model.parameters():
        if isinstance(p, UninitializedParameter):
            continue
        total += int(p.numel())
    return float(total) / 1e6


def _f1(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return float(2 * tp) / float(denom) if denom > 0 else 0.0


def _terminal_neighbor_candidates(
    edge_index: torch.Tensor,
    *,
    anchor_idx: int,
    deg: torch.Tensor,
    scaffold_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Candidate set for removed_atom_map prediction (graph-only heuristic):
    - Prefer terminal neighbors (deg<=1) of the anchor.
    - Fallback to all neighbors if no terminal neighbors exist.
    """
    src = edge_index[0]
    dst = edge_index[1]
    nbrs = dst[src == int(anchor_idx)]
    if int(nbrs.numel()) == 0:
        return nbrs
    if scaffold_mask is not None:
        if not bool(scaffold_mask[int(anchor_idx)].item()):
            return nbrs.new_empty((0,))
        non_scaf = nbrs[~scaffold_mask[nbrs]]
        return non_scaf
    term = nbrs[deg[nbrs] <= 1]
    return term if int(term.numel()) else nbrs


def _edit_targets(batch, *, eps: float, num_props: int, device: torch.device) -> torch.Tensor:
    """
    Graph-level binary labels indicating whether an edit is expected.
    y_edit=1 if any |delta| > eps, else 0.
    """
    if num_props <= 0 or not hasattr(batch, "props"):
        return torch.ones((int(batch.num_graphs),), device=device, dtype=torch.float32)
    props = batch.props
    if props.dim() == 1:
        props = props.unsqueeze(0)
    mask = (props.abs().max(dim=1).values > float(eps)).float()
    return mask.to(device=device, dtype=torch.float32)


@torch.no_grad()
def _acc_by_class(pred: torch.Tensor, target: torch.Tensor, *, num_classes: int) -> tuple[list[int], list[int]]:
    total_by = [0] * int(num_classes)
    correct_by = [0] * int(num_classes)
    for i in range(int(num_classes)):
        m = target == i
        total = int(m.sum().item())
        total_by[i] += total
        if total:
            correct_by[i] += int((pred[m] == i).sum().item())
    return correct_by, total_by


def train_one_epoch(
    model: OneShotTwoStageEditModel,
    loader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    max_epochs: int,
    log_every_steps: int,
    eval_every_steps: int = 0,
    eval_num_batches: int = 0,
    val_loader=None,
    on_val: Callable[[float, int], None] | None = None,
    global_step_start: int,
    moe_balance_weight: float,
    task_weights: dict[str, float] | None = None,
    max_steps: int = 0,
    infer_every_steps: int = 0,
    infer_eval_cb: Callable[[int], None] | None = None,
    fg_curriculum: bool = False,
    fg_curriculum_start: float = 0.2,
    fg_curriculum_end: float = 0.8,
    extra_boost_cb: Callable[[dict[str, float | int | dict], int], int] | None = None,
    remove_loss_weight: float = 0.0,
    dynamic_task_weighting: bool = False,
    dynamic_task_every_steps: int = 0,
    dynamic_task_topk: int = 0,
    dynamic_task_boost: float = 2.0,
    profile_one_step: bool = False,
) -> int:
    model.train()
    global_step = int(global_step_start)

    def _fg_curriculum_weights() -> tuple[float, float]:
        if not fg_curriculum:
            return 1.0, 0.0
        if max_epochs <= 1:
            return 1.0, 0.0
        t = float(epoch - 1) / float(max_epochs - 1)
        start = float(fg_curriculum_start)
        end = float(fg_curriculum_end)
        if t <= start:
            return 1.0, 0.0
        if t >= end:
            return 0.0, 1.0
        # Linear transition from label->pred.
        w_pred = (t - start) / max(end - start, 1e-8)
        w_label = 1.0 - w_pred
        return w_label, w_pred

    sum_total = 0.0
    sum_op = 0.0
    sum_fg = 0.0
    sum_remove = 0.0
    sum_moe_bal = 0.0
    sum_moe_ent = 0.0
    sum_moe_topk_mass = 0.0
    sum_moe_maxp = 0.0
    moe_topk_counts = None
    window_steps = 0

    op_correct = 0
    op_total = 0
    op_correct_by = [0, 0, 0, 0]
    op_total_by = [0, 0, 0, 0]

    fg_correct = 0
    fg_total = 0
    remove_correct = 0
    remove_total = 0

    per_task_stats: dict[str, dict[str, object]] = {}
    # Mutable mapping that can be updated mid-epoch (e.g. hard-task mining).
    task_weights_live: dict[str, float] | None = task_weights
    profile_one_step = bool(profile_one_step)
    profile_done = False
    prev_end = time.perf_counter()

    def _task_bucket(tid: str) -> dict[str, object]:
        bucket = per_task_stats.get(tid)
        if bucket is None:
            bucket = {
                "graphs": 0,
                "graphs_valid": 0,
                "op_loss_sum": 0.0,
                "fg_loss_sum": 0.0,
                "op_counts": [0, 0, 0, 0],
                "fg_counts": {},
            }
            per_task_stats[tid] = bucket
        return bucket
    for batch in loader:
        if profile_one_step and not profile_done and device.type == "cuda":
            torch.cuda.synchronize()
        t_data_ready = time.perf_counter()
        data_wait = t_data_ready - prev_end

        if profile_one_step and not profile_done:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_to_start = time.perf_counter()
            batch = batch.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_to_end = time.perf_counter()
        else:
            batch = batch.to(device)

        if profile_one_step and not profile_done:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_fwd_start = time.perf_counter()
            out = model(batch, teacher_forcing=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_fwd_end = time.perf_counter()
        else:
            out = model(batch, teacher_forcing=True)

        op_allowed = getattr(batch, "op_allowed", None)
        if op_allowed is not None:
            op_logits = out.op_logits.masked_fill(~op_allowed, -1e9)
            invalid_target = ~op_allowed.gather(1, batch.y_op.view(-1, 1)).squeeze(1)
            y_op_for_loss = batch.y_op
            ignore_index = -100
            if bool(invalid_target.any().item()):
                y_op_for_loss = batch.y_op.clone()
                y_op_for_loss[invalid_target] = ignore_index
        else:
            op_logits = out.op_logits
            invalid_target = None
            y_op_for_loss = batch.y_op
            ignore_index = -100

        num_graphs = int(batch.num_graphs)
        graph_task_ids = _graph_task_ids(batch, num_graphs=num_graphs)
        if graph_task_ids is None or not task_weights_live:
            graph_weight = torch.ones((num_graphs,), device=device, dtype=out.op_logits.dtype)
        else:
            graph_weight = _task_weight_tensor(
                graph_task_ids,
                task_weights=task_weights_live,
                device=device,
                dtype=out.op_logits.dtype,
            )
        node_weight = graph_weight[batch.batch] if hasattr(batch, "batch") else graph_weight

        op_weight = _op_class_weights(device=device, dtype=out.op_logits.dtype)
        op_loss_raw = F.cross_entropy(
            op_logits,
            y_op_for_loss,
            weight=op_weight,
            ignore_index=ignore_index,
            reduction="none",
        )
        valid_mask = y_op_for_loss != ignore_index
        if bool(valid_mask.any().item()):
            denom = node_weight[valid_mask].sum().clamp_min(1.0)
            loss_op = (op_loss_raw[valid_mask] * node_weight[valid_mask]).sum() / denom
        else:
            loss_op = torch.tensor(0.0, device=device)

        # fg loss with optional curriculum: label-mask -> pred-mask
        if bool((batch.y_fg >= 0).any().item()):
            fg_loss_raw = F.cross_entropy(out.fg_logits, batch.y_fg, ignore_index=-1, reduction="none")
            valid_fg = batch.y_fg >= 0
            pred_op = op_logits.detach().argmax(dim=-1)
            pred_fg_mask = (pred_op == OP_TO_ID["add"]) | (pred_op == OP_TO_ID["replace"])
            w_label, w_pred = _fg_curriculum_weights()
            fg_mask = (w_label * valid_fg.to(dtype=fg_loss_raw.dtype)) + (w_pred * pred_fg_mask.to(dtype=fg_loss_raw.dtype))
            denom = fg_mask.sum().clamp_min(1.0)
            loss_fg = (fg_loss_raw * fg_mask).sum() / denom
        else:
            loss_fg = torch.tensor(0.0, device=device)
            fg_loss_raw = None
            fg_mask = None

        total = loss_op + loss_fg

        # Anchor-level removed_atom_map pointer loss for remove/replace ops.
        remove_loss = torch.tensor(0.0, device=device)
        if float(remove_loss_weight) > 0 and hasattr(batch, "y_removed_atom_map") and hasattr(batch, "edge_index"):
            n_nodes = int(batch.y_op.numel())
            deg = torch.bincount(batch.edge_index[0].to(torch.long), minlength=n_nodes).to(device=device)
            anchor_mask = ((batch.y_op == OP_TO_ID["remove"]) | (batch.y_op == OP_TO_ID["replace"])) & (batch.y_removed_atom_map >= 0)
            anchor_idx = anchor_mask.nonzero(as_tuple=True)[0]
            if int(anchor_idx.numel()) > 0:
                cand_lists: list[torch.Tensor] = []
                tgt_pos: list[int] = []
                keep_anchor: list[int] = []
                for ai in anchor_idx.tolist():
                    cand = _terminal_neighbor_candidates(
                        batch.edge_index,
                        anchor_idx=ai,
                        deg=deg,
                        scaffold_mask=getattr(batch, "scaffold_mask", None),
                    )
                    if int(cand.numel()) == 0:
                        continue
                    removed_map = int(batch.y_removed_atom_map[ai].item())
                    pos = (batch.atom_map[cand] == removed_map).nonzero(as_tuple=True)[0]
                    if int(pos.numel()) != 1:
                        continue
                    cand_lists.append(cand)
                    tgt_pos.append(int(pos[0].item()))
                    keep_anchor.append(int(ai))
                if cand_lists:
                    max_c = max(int(c.numel()) for c in cand_lists)
                    a = len(cand_lists)
                    cand_mat = torch.full((a, max_c), -1, device=device, dtype=torch.long)
                    cand_mask = torch.zeros((a, max_c), device=device, dtype=torch.bool)
                    for i, c in enumerate(cand_lists):
                        k = int(c.numel())
                        cand_mat[i, :k] = c
                        cand_mask[i, :k] = True
                    anchor_tensor = torch.tensor(keep_anchor, device=device, dtype=torch.long)
                    g = batch.batch[anchor_tensor]
                    q = model.remove_query(torch.cat([out.node_h[anchor_tensor], out.graph_h[g]], dim=-1))  # [A, H]
                    keys = out.node_h[cand_mat.clamp_min(0)]  # [A, C, H]
                    logits = (keys * q.unsqueeze(1)).sum(dim=-1)  # [A, C]
                    logits = logits.masked_fill(~cand_mask, -1e9)
                    y = torch.tensor(tgt_pos, device=device, dtype=torch.long)
                    raw = F.cross_entropy(logits, y, reduction="none")
                    w = node_weight[anchor_tensor] if int(node_weight.numel()) else torch.ones((a,), device=device, dtype=raw.dtype)
                    denom = w.sum().clamp_min(1.0)
                    remove_loss = (raw * w).sum() / denom
                    total = total + float(remove_loss_weight) * remove_loss

        moe_bal_loss = torch.tensor(0.0, device=device)
        if float(moe_balance_weight) > 0 and out.moe_gate_probs is not None and out.moe_topi is not None:
            gate_probs = out.moe_gate_probs  # [G, E]
            topi = out.moe_topi  # [G, K]
            gsz, esz = int(gate_probs.size(0)), int(gate_probs.size(1))
            ksz = int(topi.size(1)) if int(topi.numel()) else 0
            if gsz > 0 and esz > 0 and ksz > 0:
                importance = gate_probs.mean(dim=0)  # [E]
                load = gate_probs.new_zeros((esz,))
                ones = gate_probs.new_ones((gsz * ksz,))
                load.scatter_add_(0, topi.reshape(-1), ones)
                load = load / float(max(1, gsz * ksz))
                moe_bal_loss = float(esz) * torch.sum(importance * load)
                total = total + float(moe_balance_weight) * moe_bal_loss

        if profile_one_step and not profile_done:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_loss_end = time.perf_counter()
            t_grad_start = t_loss_end

        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()

        if profile_one_step and not profile_done:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_grad_end = time.perf_counter()

        with torch.no_grad():
            pred_op = op_logits.argmax(dim=-1)
            if invalid_target is not None:
                valid = ~invalid_target
                op_correct += int((pred_op[valid] == batch.y_op[valid]).sum().item())
                op_total += int(valid.sum().item())
                c_by, t_by = _acc_by_class(pred_op[valid], batch.y_op[valid], num_classes=4)
            else:
                op_correct += int((pred_op == batch.y_op).sum().item())
                op_total += int(batch.y_op.numel())
                c_by, t_by = _acc_by_class(pred_op, batch.y_op, num_classes=4)
            for i in range(4):
                op_correct_by[i] += c_by[i]
                op_total_by[i] += t_by[i]

            if bool((batch.y_fg >= 0).any().item()):
                valid_fg = batch.y_fg >= 0
                fg_target = batch.y_fg[valid_fg]
                pred_fg = out.fg_logits[valid_fg].argmax(dim=-1)
                fg_correct += int((pred_fg == fg_target).sum().item())
                fg_total += int(valid_fg.sum().item())

            if float(remove_loss_weight) > 0 and hasattr(batch, "y_removed_atom_map") and hasattr(batch, "edge_index"):
                n_nodes = int(batch.y_op.numel())
                deg = torch.bincount(batch.edge_index[0].to(torch.long), minlength=n_nodes).to(device=device)
                anchor_mask = ((batch.y_op == OP_TO_ID["remove"]) | (batch.y_op == OP_TO_ID["replace"])) & (batch.y_removed_atom_map >= 0)
                anchor_idx = anchor_mask.nonzero(as_tuple=True)[0]
                if int(anchor_idx.numel()) > 0:
                    ok = 0
                    tot = 0
                    for ai in anchor_idx.tolist():
                        cand = _terminal_neighbor_candidates(
                            batch.edge_index,
                            anchor_idx=ai,
                            deg=deg,
                            scaffold_mask=getattr(batch, "scaffold_mask", None),
                        )
                        if int(cand.numel()) == 0:
                            continue
                        removed_map = int(batch.y_removed_atom_map[ai].item())
                        pos = (batch.atom_map[cand] == removed_map).nonzero(as_tuple=True)[0]
                        if int(pos.numel()) != 1:
                            continue
                        g = int(batch.batch[ai].item())
                        q = model.remove_query(torch.cat([out.node_h[ai], out.graph_h[g]], dim=-1))
                        scores = (out.node_h[cand] * q.unsqueeze(0)).sum(dim=-1)
                        pred = int(scores.argmax().item())
                        ok += int(pred == int(pos[0].item()))
                        tot += 1
                    remove_correct += ok
                    remove_total += tot

            sum_total += float(total.detach().item())
            sum_op += float(loss_op.detach().item())
            sum_fg += float(loss_fg.detach().item())
            sum_remove += float(remove_loss.detach().item())
            sum_moe_bal += float(moe_bal_loss.detach().item())

            if out.moe_gate_probs is not None:
                gp = out.moe_gate_probs
                sum_moe_ent += float((out.moe_entropy.mean().item()) if out.moe_entropy is not None else 0.0)
                sum_moe_topk_mass += float((out.moe_topk_mass.mean().item()) if out.moe_topk_mass is not None else 0.0)
                sum_moe_maxp += float(gp.max(dim=-1).values.mean().item())
                if out.moe_topi is not None:
                    _, e = gp.shape
                    if moe_topk_counts is None or int(moe_topk_counts.numel()) != int(e):
                        moe_topk_counts = torch.zeros((int(e),), device="cpu", dtype=torch.int64)
                    flat = out.moe_topi.reshape(-1).detach().to("cpu")
                    moe_topk_counts.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.int64))
            window_steps += 1

            if graph_task_ids is not None:
                node_task_ids = graph_task_ids[batch.batch].detach().cpu()
                task_ids = graph_task_ids.detach().cpu().tolist()
                num_graphs = int(batch.num_graphs)

                per_graph_op_loss = torch.zeros((num_graphs,), device=device)
                per_graph_op_cnt = torch.zeros((num_graphs,), device=device)
                if bool(valid_mask.any().item()):
                    per_graph_op_loss.scatter_add_(0, batch.batch[valid_mask], op_loss_raw[valid_mask])
                    per_graph_op_cnt.scatter_add_(0, batch.batch[valid_mask], torch.ones_like(op_loss_raw[valid_mask]))
                per_graph_op_loss = per_graph_op_loss / per_graph_op_cnt.clamp_min(1.0)

                per_graph_fg_loss = torch.zeros((num_graphs,), device=device)
                per_graph_fg_cnt = torch.zeros((num_graphs,), device=device)
                if bool((batch.y_fg >= 0).any().item()):
                    fg_mask_cpu = fg_mask.detach()
                    per_graph_fg_loss.scatter_add_(0, batch.batch, fg_loss_raw * fg_mask_cpu)
                    per_graph_fg_cnt.scatter_add_(0, batch.batch, fg_mask_cpu)
                per_graph_fg_loss = per_graph_fg_loss / per_graph_fg_cnt.clamp_min(1.0)

                if invalid_target is not None:
                    valid_nodes = ~invalid_target
                    y_non_none = ((batch.y_op != OP_TO_ID["none"]) & valid_nodes).to(dtype=torch.long)
                else:
                    y_non_none = (batch.y_op != OP_TO_ID["none"]).to(dtype=torch.long)
                y_count_local = torch.zeros((num_graphs,), device=device, dtype=torch.long)
                y_count_local.index_add_(0, batch.batch, y_non_none)
                graph_has_edit = (y_count_local > 0).detach().cpu().tolist()

                y_op_cpu = batch.y_op.detach().cpu()
                if invalid_target is not None:
                    valid_nodes_cpu = (~invalid_target).detach().cpu()
                else:
                    valid_nodes_cpu = torch.ones_like(y_op_cpu, dtype=torch.bool)

                for g in range(num_graphs):
                    tid_raw = int(task_ids[g])
                    tid = str(tid_raw) if tid_raw >= 0 else "NA"
                    bucket = _task_bucket(tid)
                    bucket["graphs"] = int(bucket["graphs"]) + 1
                    if graph_has_edit[g]:
                        bucket["graphs_valid"] = int(bucket["graphs_valid"]) + 1
                    bucket["op_loss_sum"] = float(bucket["op_loss_sum"]) + float(per_graph_op_loss[g].item())
                    bucket["fg_loss_sum"] = float(bucket["fg_loss_sum"]) + float(per_graph_fg_loss[g].item())

                for i in range(int(y_op_cpu.numel())):
                    if not bool(valid_nodes_cpu[i].item()):
                        continue
                    tid_raw = int(node_task_ids[i].item())
                    tid = str(tid_raw) if tid_raw >= 0 else "NA"
                    bucket = _task_bucket(tid)
                    oc = bucket["op_counts"]
                    oc[int(y_op_cpu[i].item())] += 1

                if bool((batch.y_fg >= 0).any().item()):
                    fg_ids = batch.y_fg.detach().cpu()
                    valid_fg_cpu = (fg_ids >= 0)
                    for i in range(int(fg_ids.numel())):
                        if not bool(valid_fg_cpu[i].item()):
                            continue
                        tid_raw = int(node_task_ids[i].item())
                        tid = str(tid_raw) if tid_raw >= 0 else "NA"
                        bucket = _task_bucket(tid)
                        fg_counts = bucket["fg_counts"]
                        fg_id = int(fg_ids[i].item())
                        fg_counts[fg_id] = fg_counts.get(fg_id, 0) + 1

        if profile_one_step and not profile_done:
            total_step = t_grad_end - t_data_ready
            to_device = t_to_end - t_to_start
            fwd = t_fwd_end - t_fwd_start
            loss = t_loss_end - t_fwd_end
            grad = t_grad_end - t_grad_start
            print(
                "profile_one_step "
                f"step={global_step + 1} "
                f"data_wait_ms={data_wait * 1000.0:.3f} "
                f"to_device_ms={to_device * 1000.0:.3f} "
                f"forward_ms={fwd * 1000.0:.3f} "
                f"loss_ms={loss * 1000.0:.3f} "
                f"grad_step_ms={grad * 1000.0:.3f} "
                f"total_ms={total_step * 1000.0:.3f}"
            )
            profile_done = True

        prev_end = time.perf_counter()
        global_step += 1

        if (
            dynamic_task_weighting
            and graph_task_ids is not None
            and int(dynamic_task_every_steps) > 0
            and int(dynamic_task_topk) > 0
            and float(dynamic_task_boost) > 1.0
            and (global_step % int(dynamic_task_every_steps) == 0)
            and per_task_stats
        ):
            # Pick the top-K hardest tasks (by avg loss over the recent window) and boost their weights.
            scores = []
            for tid, s in per_task_stats.items():
                if tid == "NA":
                    continue
                graphs = int(s.get("graphs", 0) or 0)
                if graphs <= 0:
                    continue
                op_loss_avg = float(s.get("op_loss_sum", 0.0)) / float(graphs)
                fg_loss_avg = float(s.get("fg_loss_sum", 0.0)) / float(graphs)
                scores.append((str(tid), op_loss_avg + fg_loss_avg))
            scores.sort(key=lambda x: x[1], reverse=True)
            pick = scores[: int(dynamic_task_topk)]
            if pick:
                if task_weights_live is None:
                    task_weights_live = {}
                # Reset all previously-seen tasks to 1.0, then boost top-k.
                for tid, _ in scores:
                    task_weights_live[tid] = 1.0
                for tid, _ in pick:
                    task_weights_live[tid] = float(dynamic_task_boost)
                pick_str = ", ".join(f"{tid}:{loss:.4f}" for tid, loss in pick)
                print(
                    f"step={global_step} dynamic_task_weights boost={float(dynamic_task_boost):.2f} "
                    f"topk={int(dynamic_task_topk)} every={int(dynamic_task_every_steps)} picked={pick_str}"
                )

        if log_every_steps and global_step % log_every_steps == 0:
            avg_total = _safe_div(sum_total, window_steps)
            avg_op = _safe_div(sum_op, window_steps)
            avg_fg = _safe_div(sum_fg, window_steps)
            avg_remove = _safe_div(sum_remove, window_steps)
            avg_moe_bal = _safe_div(sum_moe_bal, window_steps)
            avg_moe_ent = _safe_div(sum_moe_ent, window_steps)
            avg_moe_topk_mass = _safe_div(sum_moe_topk_mass, window_steps)
            avg_moe_maxp = _safe_div(sum_moe_maxp, window_steps)

            op_acc = _safe_div(float(op_correct), float(op_total))
            acc_none = _safe_div(float(op_correct_by[OP_TO_ID["none"]]), float(op_total_by[OP_TO_ID["none"]]))
            acc_add = _safe_div(float(op_correct_by[OP_TO_ID["add"]]), float(op_total_by[OP_TO_ID["add"]]))
            acc_remove = _safe_div(float(op_correct_by[OP_TO_ID["remove"]]), float(op_total_by[OP_TO_ID["remove"]]))
            acc_replace = _safe_div(float(op_correct_by[OP_TO_ID["replace"]]), float(op_total_by[OP_TO_ID["replace"]]))
            fg_acc = _safe_div(float(fg_correct), float(fg_total))
            remove_acc = _safe_div(float(remove_correct), float(remove_total))
            print(
                f"step={global_step} train_loss={avg_total:.4f} op_loss={avg_op:.4f} fg_loss={avg_fg:.4f} "
                f"op_acc={op_acc:.3f} acc_none={acc_none:.3f} acc_add={acc_add:.3f} "
                f"acc_remove={acc_remove:.3f} acc_replace={acc_replace:.3f} fg_acc={fg_acc:.3f} "
                f"remove_loss={avg_remove:.4f} remove_acc={remove_acc:.3f} "
                f"moe_bal={avg_moe_bal:.4f} moe_ent={avg_moe_ent:.3f} moe_topk_mass={avg_moe_topk_mass:.3f} moe_maxp={avg_moe_maxp:.3f} "
                f"op_n={op_total} fg_n={fg_total}"
            )
            if moe_topk_counts is not None:
                top = torch.topk(moe_topk_counts.to(dtype=torch.float32), k=min(8, int(moe_topk_counts.numel()))).indices.tolist()
                pairs = ", ".join([f"{i}:{int(moe_topk_counts[i])}" for i in top])
                print(f"step={global_step} moe_topk_counts_top={pairs}")

            if per_task_stats:
                for tid in sorted(per_task_stats.keys(), key=lambda x: (x != "NA", x)):
                    s = per_task_stats[tid]
                    graphs = int(s["graphs"])
                    if graphs <= 0:
                        continue
                    valid_ratio = float(s["graphs_valid"]) / float(graphs)
                    op_loss_avg = float(s["op_loss_sum"]) / float(graphs)
                    fg_loss_avg = float(s["fg_loss_sum"]) / float(graphs)
                    op_counts = s["op_counts"]
                    op_total_t = max(1, sum(op_counts))
                    op_dist = [float(c) / float(op_total_t) for c in op_counts]
                    fg_counts = s["fg_counts"]
                    if fg_counts:
                        top_fg = sorted(fg_counts.items(), key=lambda x: (-x[1], x[0]))[:5]
                        fg_top_str = ",".join(f"{k}:{v}" for k, v in top_fg)
                    else:
                        fg_top_str = ""
                    print(
                        f"step={global_step} task={tid} graphs={graphs} valid_ratio={valid_ratio:.3f} "
                        f"op_loss={op_loss_avg:.4f} fg_loss={fg_loss_avg:.4f} "
                        f"op_dist={op_dist[0]:.2f}/{op_dist[1]:.2f}/{op_dist[2]:.2f}/{op_dist[3]:.2f} "
                        f"fg_top={fg_top_str}"
                    )

            sum_total = 0.0
            sum_op = 0.0
            sum_fg = 0.0
            sum_remove = 0.0
            sum_moe_bal = 0.0
            sum_moe_ent = 0.0
            sum_moe_topk_mass = 0.0
            sum_moe_maxp = 0.0
            moe_topk_counts = None
            window_steps = 0
            op_correct = 0
            op_total = 0
            op_correct_by = [0, 0, 0, 0]
            op_total_by = [0, 0, 0, 0]
            fg_correct = 0
            fg_total = 0
            remove_correct = 0
            remove_total = 0
            per_task_stats = {}

        if eval_every_steps and val_loader is not None and global_step % eval_every_steps == 0:
            was_training = model.training
            metrics = eval_metrics(
                model,
                val_loader,
                device=device,
                max_batches=0,
            )
            if was_training:
                model.train()
            print(
                f"step={global_step} val_loss={metrics['loss']:.4f} op_loss={metrics['op_loss']:.4f} fg_loss={metrics['fg_loss']:.4f} "
                f"op_acc={metrics['op_acc']:.3f} acc_none={metrics['acc_none']:.3f} acc_add={metrics['acc_add']:.3f} "
                f"acc_remove={metrics['acc_remove']:.3f} acc_replace={metrics['acc_replace']:.3f} fg_acc={metrics['fg_acc']:.3f} "
                f"remove_loss={metrics.get('remove_loss', 0.0):.4f} remove_acc={metrics.get('remove_acc', 0.0):.3f} "
                f"op_f1={metrics['op_f1_macro']:.3f} fg_f1={metrics['fg_f1']:.3f} op_n={metrics['op_n']} fg_n={metrics['fg_n']}"
            )
            if on_val is not None:
                on_val(float(metrics["op_f1_macro"]), int(global_step))
            if extra_boost_cb is not None:
                global_step = int(extra_boost_cb(metrics, int(global_step)))

        if infer_every_steps and infer_eval_cb is not None and global_step % int(infer_every_steps) == 0:
            infer_eval_cb(int(global_step))

        if max_steps and (global_step - global_step_start) >= int(max_steps):
            break

    return global_step


@torch.no_grad()
def eval_metrics(
    model: OneShotTwoStageEditModel,
    loader,
    *,
    device: torch.device,
    max_batches: int = 0,
) -> dict[str, float | int | dict[str, float] | dict[str, int]]:
    model.eval()

    sum_total = 0.0
    sum_op = 0.0
    sum_fg = 0.0
    sum_remove = 0.0
    batches = 0

    op_correct = 0
    op_total = 0
    op_correct_by = [0, 0, 0, 0]
    op_total_by = [0, 0, 0, 0]
    op_tp = [0, 0, 0, 0]
    op_fp = [0, 0, 0, 0]
    op_fn = [0, 0, 0, 0]

    fg_correct = 0
    fg_total = 0
    fg_tp = 0
    fg_fp = 0
    fg_fn = 0
    remove_total = 0
    remove_correct = 0
    per_task_correct: dict[str, int] = {}
    per_task_total: dict[str, int] = {}
    per_task_fg_loss_sum: dict[str, float] = {}
    per_task_graphs: dict[str, int] = {}

    for batch in loader:
        batch = batch.to(device)

        out = model(batch, teacher_forcing=True)

        op_allowed = getattr(batch, "op_allowed", None)
        if op_allowed is not None:
            op_logits = out.op_logits.masked_fill(~op_allowed, -1e9)
            invalid_target = ~op_allowed.gather(1, batch.y_op.view(-1, 1)).squeeze(1)
            y_op_for_loss = batch.y_op
            ignore_index = -100
            if bool(invalid_target.any().item()):
                y_op_for_loss = batch.y_op.clone()
                y_op_for_loss[invalid_target] = ignore_index
        else:
            op_logits = out.op_logits
            invalid_target = None
            y_op_for_loss = batch.y_op
            ignore_index = -100

        op_weight = _op_class_weights(device=device, dtype=out.op_logits.dtype)
        loss_op = F.cross_entropy(op_logits, y_op_for_loss, weight=op_weight, ignore_index=ignore_index)

        fg_loss_raw = None
        if bool((batch.y_fg >= 0).any().item()):
            fg_loss_raw = F.cross_entropy(out.fg_logits, batch.y_fg, ignore_index=-1, reduction="none")
            loss_fg = F.cross_entropy(out.fg_logits, batch.y_fg, ignore_index=-1)
        else:
            loss_fg = torch.tensor(0.0, device=device)

        # Removed-atom pointer metrics (graph-only candidate heuristic).
        remove_loss = torch.tensor(0.0, device=device)
        if hasattr(batch, "y_removed_atom_map") and hasattr(batch, "edge_index"):
            n_nodes = int(batch.y_op.numel())
            deg = torch.bincount(batch.edge_index[0].to(torch.long), minlength=n_nodes).to(device=device)
            anchor_mask = ((batch.y_op == OP_TO_ID["remove"]) | (batch.y_op == OP_TO_ID["replace"])) & (batch.y_removed_atom_map >= 0)
            anchor_idx = anchor_mask.nonzero(as_tuple=True)[0]
            if int(anchor_idx.numel()) > 0:
                losses = []
                ok = 0
                tot = 0
                for ai in anchor_idx.tolist():
                    cand = _terminal_neighbor_candidates(
                        batch.edge_index,
                        anchor_idx=ai,
                        deg=deg,
                        scaffold_mask=getattr(batch, "scaffold_mask", None),
                    )
                    if int(cand.numel()) == 0:
                        continue
                    removed_map = int(batch.y_removed_atom_map[ai].item())
                    pos = (batch.atom_map[cand] == removed_map).nonzero(as_tuple=True)[0]
                    if int(pos.numel()) != 1:
                        continue
                    g = int(batch.batch[ai].item())
                    q = model.remove_query(torch.cat([out.node_h[ai], out.graph_h[g]], dim=-1))
                    scores = (out.node_h[cand] * q.unsqueeze(0)).sum(dim=-1)
                    y = int(pos[0].item())
                    losses.append(F.cross_entropy(scores.view(1, -1), torch.tensor([y], device=device)))
                    ok += int(int(scores.argmax().item()) == y)
                    tot += 1
                if losses:
                    remove_loss = torch.stack(losses).mean()
                remove_correct += ok
                remove_total += tot

        total = (loss_op + loss_fg + remove_loss).detach()

        pred_op = op_logits.argmax(dim=-1)
        valid_mask = ~invalid_target if invalid_target is not None else torch.ones_like(batch.y_op, dtype=torch.bool)
        task_ids_graph = _graph_task_ids(batch, num_graphs=int(batch.num_graphs))
        if task_ids_graph is not None:
            node_task_ids = task_ids_graph[batch.batch]
            unique_ids = torch.unique(node_task_ids[valid_mask])
            for tid in unique_ids.tolist():
                tid_int = int(tid)
                if tid_int < 0:
                    continue
                m = valid_mask & (node_task_ids == tid)
                if not bool(m.any().item()):
                    continue
                correct = int((pred_op[m] == batch.y_op[m]).sum().item())
                total = int(m.sum().item())
                key = str(tid_int)
                per_task_correct[key] = per_task_correct.get(key, 0) + correct
                per_task_total[key] = per_task_total.get(key, 0) + total
            num_graphs = int(batch.num_graphs)
            per_graph_fg_loss = torch.zeros((num_graphs,), device=device)
            per_graph_fg_cnt = torch.zeros((num_graphs,), device=device)
            if fg_loss_raw is not None:
                valid_fg = batch.y_fg >= 0
                per_graph_fg_loss.scatter_add_(0, batch.batch[valid_fg], fg_loss_raw[valid_fg])
                per_graph_fg_cnt.scatter_add_(0, batch.batch[valid_fg], torch.ones_like(fg_loss_raw[valid_fg]))
            per_graph_fg_loss = per_graph_fg_loss / per_graph_fg_cnt.clamp_min(1.0)
            for g in range(num_graphs):
                tid_raw = int(task_ids_graph[g].item())
                tid = str(tid_raw) if tid_raw >= 0 else "NA"
                per_task_graphs[tid] = per_task_graphs.get(tid, 0) + 1
                per_task_fg_loss_sum[tid] = per_task_fg_loss_sum.get(tid, 0.0) + float(per_graph_fg_loss[g].item())
        if invalid_target is not None:
            valid = ~invalid_target
            op_correct += int((pred_op[valid] == batch.y_op[valid]).sum().item())
            op_total += int(valid.sum().item())
            c_by, t_by = _acc_by_class(pred_op[valid], batch.y_op[valid], num_classes=4)
        else:
            op_correct += int((pred_op == batch.y_op).sum().item())
            op_total += int(batch.y_op.numel())
            c_by, t_by = _acc_by_class(pred_op, batch.y_op, num_classes=4)
        for i in range(4):
            op_correct_by[i] += c_by[i]
            op_total_by[i] += t_by[i]
            if invalid_target is not None:
                m = ~invalid_target
                op_tp[i] += int(((pred_op == i) & (batch.y_op == i) & m).sum().item())
                op_fp[i] += int(((pred_op == i) & (batch.y_op != i) & m).sum().item())
                op_fn[i] += int(((pred_op != i) & (batch.y_op == i) & m).sum().item())
            else:
                op_tp[i] += int(((pred_op == i) & (batch.y_op == i)).sum().item())
                op_fp[i] += int(((pred_op == i) & (batch.y_op != i)).sum().item())
                op_fn[i] += int(((pred_op != i) & (batch.y_op == i)).sum().item())

        if bool((batch.y_fg >= 0).any().item()):
            valid_fg = batch.y_fg >= 0
            fg_target = batch.y_fg[valid_fg]
            pred_fg = out.fg_logits[valid_fg].argmax(dim=-1)
            fg_correct += int((pred_fg == fg_target).sum().item())
            fg_total += int(valid_fg.sum().item())
            tp = int((pred_fg == fg_target).sum().item())
            fg_tp += tp
            fg_fp += int((pred_fg != fg_target).sum().item())
            fg_fn += int((pred_fg != fg_target).sum().item())

        sum_total += float(total)
        sum_op += float(loss_op.detach().item())
        sum_fg += float(loss_fg.detach().item())
        sum_remove += float(remove_loss.detach().item())
        batches += 1

        if max_batches and batches >= max_batches:
            break

    if batches == 0:
        return {
            "loss": float("inf"),
            "op_loss": float("inf"),
            "fg_loss": float("inf"),
            "op_acc": 0.0,
            "acc_none": 0.0,
            "acc_add": 0.0,
            "acc_remove": 0.0,
            "acc_replace": 0.0,
            "fg_acc": 0.0,
            "op_n": 0,
            "fg_n": 0,
            "per_task_op_acc": {},
            "per_task_op_n": {},
        }

    per_task_op_acc = {k: _safe_div(float(per_task_correct[k]), float(per_task_total[k])) for k in per_task_total}

    return {
        "loss": _safe_div(sum_total, float(batches)),
        "op_loss": _safe_div(sum_op, float(batches)),
        "fg_loss": _safe_div(sum_fg, float(batches)),
        "remove_loss": _safe_div(sum_remove, float(batches)),
        "op_acc": _safe_div(float(op_correct), float(op_total)),
        "acc_none": _safe_div(float(op_correct_by[OP_TO_ID["none"]]), float(op_total_by[OP_TO_ID["none"]])),
        "acc_add": _safe_div(float(op_correct_by[OP_TO_ID["add"]]), float(op_total_by[OP_TO_ID["add"]])),
        "acc_remove": _safe_div(float(op_correct_by[OP_TO_ID["remove"]]), float(op_total_by[OP_TO_ID["remove"]])),
        "acc_replace": _safe_div(float(op_correct_by[OP_TO_ID["replace"]]), float(op_total_by[OP_TO_ID["replace"]])),
        "fg_acc": _safe_div(float(fg_correct), float(fg_total)),
        "op_f1_macro": sum(_f1(op_tp[i], op_fp[i], op_fn[i]) for i in range(4)) / 4.0,
        "fg_f1": _f1(fg_tp, fg_fp, fg_fn),
        "remove_acc": _safe_div(float(remove_correct), float(remove_total)),
        "op_n": int(op_total),
        "fg_n": int(fg_total),
        "per_task_op_acc": per_task_op_acc,
        "per_task_op_n": per_task_total,
        "per_task_fg_loss": {k: _safe_div(float(per_task_fg_loss_sum[k]), float(per_task_graphs[k])) for k in per_task_graphs},
    }


@torch.no_grad()
def eval_loss(
    model: OneShotTwoStageEditModel,
    loader,
    *,
    device: torch.device,
    max_batches: int = 0,
) -> float:
    return float(
        eval_metrics(
            model,
            loader,
            device=device,
            max_batches=max_batches,
        )["loss"]
    )


def _build_prop_mask_by_task(*, task_defs_path: str, property_names: list[str]) -> dict[str, list[int]]:
    if not task_defs_path:
        return {}
    try:
        raw = _load_yaml(task_defs_path)
    except Exception:
        return {}
    task_defs = raw.get("task_defs", raw)
    taskid_prop = task_defs.get("taskid_prop") if isinstance(task_defs, dict) else None
    if not isinstance(taskid_prop, dict):
        return {}
    name_to_idx = {p: i for i, p in enumerate(property_names)}
    mask: dict[str, list[int]] = {}
    for tid, props in taskid_prop.items():
        if not isinstance(props, list):
            continue
        idxs = [name_to_idx[p] for p in props if p in name_to_idx]
        mask[str(tid)] = idxs
    return mask


def main() -> None:
    _disable_rdkit_warnings()
    parser = argparse.ArgumentParser(
        description="Minimal one-shot training: no chemistry constraints; predict per-node op + per-node fg on add/replace."
    )
    parser.add_argument("--train_config", type=str, default="train_oneshot_config.yaml")
    parser.add_argument(
        "--model_scale",
        type=str,
        default="",
        help="Optional preset model size: small | medium | large (overrides hidden_dim/num_layers).",
    )
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default="",
        help="Optional checkpoint path to load model weights before training.",
    )
    args = parser.parse_args()

    train_cfg_path = Path(args.train_config).resolve()
    cfg = _load_yaml(args.train_config)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    eval_cfg = cfg.get("eval", {}) or {}
    if not isinstance(eval_cfg, dict):
        raise ValueError("eval config must be a dict if provided.")
    # Task definitions used to build per-task property masks for conditioning.
    # Support both legacy (top-level) and nested (train.*) config placement.
    task_defs_path = str(cfg.get("task_defs_yaml") or train_cfg.get("task_defs_yaml") or "infer_config.yaml")

    base_cfg = Config.load(str(data_cfg["config_yaml"]))
    property_names = list(base_cfg.raw["chemistry"]["property_names"])
    fg_vocab_path = base_cfg.resolve_path(base_cfg.raw["chemistry"]["functional_groups_json"])
    fg_vocab = FunctionalGroupVocab.load(fg_vocab_path)

    prop_mask_by_task = _build_prop_mask_by_task(task_defs_path=task_defs_path, property_names=property_names)

    seed = int(data_cfg.get("seed", 42))
    # Control whether to zero out deltas for non-task properties during training.
    # Keep this under data.* only so the dataset behavior is fully determined by the dataset section.
    # Support legacy key name `data.mask_prop_by_task` as an alias.
    mask_non_task_props = bool(data_cfg.get("mask_non_task_props", data_cfg.get("mask_prop_by_task", False)))
    dataset = OneShotEditDataset(
        str(data_cfg["dataset_csv"]),
        property_names=property_names,
        seed=seed,
        prop_mask_by_task=prop_mask_by_task,
        mask_non_task_props=mask_non_task_props,
    )
    num_tasks_cfg = int(model_cfg.get("num_tasks", -1))
    if num_tasks_cfg < 0:
        max_task_id = int(getattr(dataset, "max_task_id", -1))
        num_tasks_cfg = max_task_id + 1 if max_task_id >= 0 else 0
    model_cfg = {**model_cfg, "num_tasks": int(num_tasks_cfg), "task_emb_dim": int(model_cfg.get("task_emb_dim", 32))}
    n = len(dataset)
    if n < 10:
        print(f"Warning: dataset is very small (n={n}).")

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    val_ratio = float(data_cfg.get("val_ratio", 0.1))
    val_n = max(1, int(math.ceil(n * val_ratio)))
    val_idx = perm[:val_n]
    train_idx = perm[val_n:]

    require_torch_geometric()
    from torch_geometric.loader import DataLoader

    fg_balance_cfg = data_cfg.get("fg_balance", {})
    fg_balance_enabled = False
    fg_balance_power = 1.0
    fg_balance_min = None
    fg_balance_max = None
    if isinstance(fg_balance_cfg, dict):
        fg_balance_enabled = bool(fg_balance_cfg.get("enabled", False))
        fg_balance_power = float(fg_balance_cfg.get("power", 1.0))
        fg_balance_min = fg_balance_cfg.get("min_weight", None)
        fg_balance_max = fg_balance_cfg.get("max_weight", None)
    elif isinstance(fg_balance_cfg, bool):
        fg_balance_enabled = fg_balance_cfg

    train_subset = torch.utils.data.Subset(dataset, train_idx)
    train_sampler = None
    if fg_balance_enabled:
        weights = _build_fg_balance_weights(
            dataset.rows,
            train_idx,
            power=fg_balance_power,
            min_w=fg_balance_min,
            max_w=fg_balance_max,
        )
        if weights:
            train_sampler = torch.utils.data.WeightedRandomSampler(
                weights=weights,
                num_samples=len(weights),
                replacement=True,
                generator=g,
            )

    train_loader = DataLoader(
        train_subset,
        batch_size=int(data_cfg.get("batch_size", 8)),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=int(data_cfg.get("num_workers", 0)),
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(dataset, val_idx),
        batch_size=int(data_cfg.get("batch_size", 8)),
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 0)),
    )

    scale = str(getattr(args, "model_scale", "")).lower().strip()
    if scale:
        presets = {
            "small": {"hidden_dim": 192, "num_layers": 3},
            "medium": {"hidden_dim": 256, "num_layers": 5},
            "large": {"hidden_dim": 384, "num_layers": 7},
        }
        if scale not in presets:
            raise ValueError(f"Unknown model_scale={scale!r}; choose from {sorted(presets)}")
        model_cfg = {**model_cfg, **presets[scale]}

    num_props = len(dataset.prop_cols)
    model = OneShotTwoStageEditModel(
        props_dim=num_props,
        fg_vocab_size=fg_vocab.size,
        fg_num_classes=2,
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
        num_layers=int(model_cfg.get("num_layers", 5)),
        gine_layers=model_cfg.get("gine_layers"),
        mlp_layers=model_cfg.get("mlp_layers"),
        num_tasks=int(model_cfg.get("num_tasks", 0)),
        task_emb_dim=int(model_cfg.get("task_emb_dim", 32)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        backbone=str(model_cfg.get("backbone", "gine")),
        use_moe=bool(model_cfg.get("use_moe", True)),
        num_experts=int(model_cfg.get("num_experts", 8)),
        moe_topk=int(model_cfg.get("moe_topk", 1)),
        moe_gate_temperature=float(model_cfg.get("moe_gate_temperature", 1.0)),
        moe_gate_noise=float(model_cfg.get("moe_gate_noise", 0.0)),
    )

    resume_path = str(train_cfg.get("resume_from", "")) or str(args.resume_checkpoint or "")
    if resume_path:
        ckpt_path = Path(resume_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"resume_checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[resume] missing keys: {missing[:6]}{'...' if len(missing) > 6 else ''}")
        if unexpected:
            print(f"[resume] unexpected keys: {unexpected[:6]}{'...' if len(unexpected) > 6 else ''}")
        print(f"[resume] loaded model weights from {ckpt_path}")

    device = torch.device(str(train_cfg.get("device", "cpu")))
    model.to(device)
    total_params_m = _count_params(model)
    scale_tag = scale if scale else "custom"
    print(f"Model scale={scale_tag}, params={total_params_m:.2f}M")
    model.set_fg_hierarchy(fg_id_to_class=_build_fg_id_to_class(fg_vocab=fg_vocab).to(device))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
    )

    best_val = 0.0
    best_metric = 0.0
    patience = int(train_cfg.get("early_stop_patience", 8))
    bad_epochs = 0
    save_path = Path(str(train_cfg.get("save_path", "checkpoints/best_oneshot.pt")))
    save_path.parent.mkdir(parents=True, exist_ok=True)

    eval_infer_cfg_path = eval_cfg.get("infer_config")
    use_infer_eval = bool(eval_infer_cfg_path)
    metric_name = "avg_loose" if use_infer_eval else "op_f1"

    def _checkpoint_payload() -> dict[str, Any]:
        return {
            "model_state_dict": model.state_dict(),
            "property_cols": dataset.prop_cols,
            "fg_vocab_path": str(fg_vocab_path),
            "config_yaml": str(data_cfg["config_yaml"]),
            "num_tasks": int(model_cfg.get("num_tasks", 0)),
            "task_emb_dim": int(model_cfg.get("task_emb_dim", 32)),
            # Persist training-time masking so inference can stay consistent even if a different train_config.yaml is used.
            "mask_non_task_props": bool(mask_non_task_props),
        }

    def _save_checkpoint(path: Path) -> None:
        torch.save(_checkpoint_payload(), path)

    def _maybe_save(val: float, *, tag: str) -> None:
        nonlocal best_val, best_metric
        if val > best_metric:
            best_metric = float(val)
            _save_checkpoint(save_path)
            print(f"saved: {save_path} ({tag} best_{metric_name}={best_metric:.4f})")

    infer_cfg = None
    infer_cfg_path = None
    eval_every_epochs = 0
    eval_every_steps = 0
    eval_output_dir = None
    eval_test_data = None
    eval_mol_col = None
    eval_tasks_cfg = None
    eval_seed = None
    fg_curriculum = bool(train_cfg.get("fg_curriculum", False))
    fg_curriculum_start = float(train_cfg.get("fg_curriculum_start", 0.2))
    fg_curriculum_end = float(train_cfg.get("fg_curriculum_end", 0.8))
    extra_boost_cfg = train_cfg.get("extra_task_boost", {})
    if isinstance(extra_boost_cfg, dict):
        extra_boost_enabled = bool(extra_boost_cfg.get("enabled", False))
        extra_boost_topk = int(extra_boost_cfg.get("topk", 2))
        extra_boost_steps = int(extra_boost_cfg.get("steps", 1))
        extra_boost_dir = Path(str(extra_boost_cfg.get("data_dir", "data/task_small")))
        extra_boost_every_steps = int(extra_boost_cfg.get("every_steps", 0))
    else:
        extra_boost_enabled = bool(extra_boost_cfg)
        extra_boost_topk = 2
        extra_boost_steps = 1
        extra_boost_dir = Path("data/task_small")
        extra_boost_every_steps = 0
    extra_task_cache: dict[str, OneShotEditDataset] = {}

    def _load_task_small_dataset(task_id: str) -> OneShotEditDataset | None:
        path = extra_boost_dir / f"task_{task_id}.csv"
        if not path.exists():
            return None
        ds = extra_task_cache.get(task_id)
        if ds is None:
            ds = OneShotEditDataset(
                str(path),
                property_names=property_names,
                seed=int(data_cfg.get("seed", 42)),
                prop_mask_by_task=prop_mask_by_task,
                mask_non_task_props=mask_non_task_props,
            )
            extra_task_cache[task_id] = ds
        return ds

    def _extra_boost_cb(metrics: dict[str, float | int | dict], step: int) -> int:
        nonlocal global_step
        if not extra_boost_enabled or int(extra_boost_topk) <= 0 or int(extra_boost_steps) <= 0:
            return int(step)
        print(f"extra_task_boost: check step={int(step)}")
        if extra_boost_every_steps > 0 and (int(step) % int(extra_boost_every_steps) != 0):
            print(f"extra_task_boost: skip (every_steps={int(extra_boost_every_steps)})")
            return int(step)
        per_fg = metrics.get("per_task_fg_loss", {})
        if not isinstance(per_fg, dict):
            print("extra_task_boost: missing per_task_fg_loss (check task_id in val data)")
            return int(step)
        if not per_fg:
            print("extra_task_boost: empty per-task losses (check task_id in val data)")
            return int(step)
        scores = []
        for tid, v in per_fg.items():
            try:
                score = float(v)
            except Exception:
                continue
            scores.append((str(tid), score))
        scores.sort(key=lambda x: x[1], reverse=True)
        pick = [tid for tid, _ in scores[: int(extra_boost_topk)]]
        ds_list = []
        for tid in pick:
            ds = _load_task_small_dataset(tid)
            if ds is None:
                print(f"extra_task_boost: missing dataset for task={tid} (path={extra_boost_dir}/task_{tid}.csv)")
            elif len(ds) == 0:
                print(f"extra_task_boost: empty dataset for task={tid}")
            else:
                ds_list.append(ds)
        if not ds_list:
            print("extra_task_boost: no datasets loaded for selected tasks")
            return int(step)
        extra_ds = torch.utils.data.ConcatDataset(ds_list)
        extra_loader = DataLoader(
            extra_ds,
            batch_size=int(data_cfg.get("batch_size", 8)),
            shuffle=True,
            num_workers=int(data_cfg.get("num_workers", 0)),
        )
        print(f"extra_task_boost tasks={pick} steps={int(extra_boost_steps)}")
        global_step = train_one_epoch(
            model,
            extra_loader,
            device=device,
            optimizer=optimizer,
            epoch=epoch,
            max_epochs=int(train_cfg.get("max_epochs", 50)),
            log_every_steps=0,
            val_loader=None,
            eval_every_steps=0,
            eval_num_batches=0,
            on_val=None,
            global_step_start=step,
            moe_balance_weight=float(train_cfg.get("moe_balance_weight", 0.0)),
            task_weights=None,
            max_steps=int(extra_boost_steps),
            infer_every_steps=0,
            infer_eval_cb=None,
            fg_curriculum=fg_curriculum,
            fg_curriculum_start=fg_curriculum_start,
            fg_curriculum_end=fg_curriculum_end,
            extra_boost_cb=None,
            remove_loss_weight=float(train_cfg.get("remove_loss_weight", 0.0)),
            dynamic_task_weighting=False,
            dynamic_task_every_steps=0,
            dynamic_task_topk=0,
            dynamic_task_boost=1.0,
            profile_one_step=False,
        )
        return int(global_step)

    def _infer_eval_step(step: int) -> None:
        infer_res = _infer_and_eval(tag=f"step_{step}")
        if infer_res is None:
            return
        avg_loose, per_task_acc = infer_res
        best_before = float(best_metric)
        _maybe_save(float(avg_loose), tag=f"step={step}")
        if per_task_acc:
            per_task_str = ", ".join(f"{k}:{float(v):.2f}" for k, v in sorted(per_task_acc.items()))
        else:
            per_task_str = ""
        print(f"infer_eval step={step} avg_loose={float(avg_loose):.2f} best_before={best_before:.2f}")
        if per_task_str:
            print(f"infer_eval step={step} per_task_loose={per_task_str}")

    if use_infer_eval:
        infer_cfg_path = _resolve_path(train_cfg_path.parent, str(eval_infer_cfg_path))
        infer_cfg = _load_yaml(infer_cfg_path)
        eval_every_epochs = int(eval_cfg.get("every_epochs", 1))
        eval_every_steps = int(eval_cfg.get("every_steps", 0))
        if eval_every_steps > 0:
            eval_every_epochs = 0
        eval_output_dir = _resolve_path(
            train_cfg_path.parent,
            str(eval_cfg.get("output_dir", infer_cfg.get("output_dir", "out_tasks_eval"))),
        )
        eval_test_data_raw = eval_cfg.get("test_data", infer_cfg.get("input_csv"))
        if not eval_test_data_raw:
            raise ValueError("eval.test_data (or infer_config.input_csv) is required for infer eval.")
        eval_test_data = _resolve_path(train_cfg_path.parent, str(eval_test_data_raw))
        eval_mol_col = eval_cfg.get("mol_col", infer_cfg.get("smiles_col", "mol"))
        eval_tasks_cfg = eval_cfg.get("tasks_config")
        eval_seed = eval_cfg.get("seed")

    def _infer_and_eval(*, tag: str) -> tuple[float, dict[str, float]] | None:
        if not use_infer_eval or infer_cfg is None or infer_cfg_path is None:
            return None
        import evaluate_infer

        tmp_ckpt = save_path.parent / f".eval_ckpt_{tag}.pt"
        tmp_infer_cfg = save_path.parent / f".infer_eval_{tag}.yaml"
        tmp_tasks_cfg = None

        infer_cfg_run = dict(infer_cfg)
        infer_cfg_run["checkpoint"] = str(tmp_ckpt)
        infer_cfg_run["train_config"] = str(Path(args.train_config).resolve())
        infer_cfg_run["device"] = str(device)

        eval_out_dir = Path(str(eval_output_dir)).resolve()
        eval_in_csv = Path(str(eval_test_data)).resolve()
        infer_cfg_run["output_dir"] = str(eval_out_dir)
        infer_cfg_run["input_csv"] = str(eval_in_csv)
        infer_cfg_run["smiles_col"] = str(eval_mol_col)
        if eval_seed is not None:
            infer_cfg_run["seed"] = int(eval_seed)

        run_task_ids = _task_ids_from_infer_config(infer_cfg_run, infer_cfg_path=infer_cfg_path)

        if eval_tasks_cfg:
            tasks_cfg_path = _resolve_path(infer_cfg_path.parent, str(eval_tasks_cfg))
        else:
            task_defs_inline = infer_cfg_run.get("task_defs")
            if isinstance(task_defs_inline, dict):
                tmp_tasks_cfg = save_path.parent / f".tasks_eval_{tag}.yaml"
                _write_yaml(tmp_tasks_cfg, task_defs_inline)
                tasks_cfg_path = tmp_tasks_cfg
            else:
                tasks_path = infer_cfg_run.get("tasks")
                if not tasks_path:
                    raise ValueError("eval tasks config missing (set eval.tasks_config or infer_config.tasks/task_defs).")
                tasks_cfg_path = _resolve_path(infer_cfg_path.parent, str(tasks_path))

        try:
            _save_checkpoint(tmp_ckpt)
            _write_yaml(tmp_infer_cfg, infer_cfg_run)

            cmd = [
                sys.executable,
                str(Path(__file__).resolve().parents[1] / "infer_oneshot.py"),
                "--infer_config",
                str(tmp_infer_cfg),
            ]
            subprocess.run(cmd, check=True)

            infer_paths = [str(eval_out_dir / f"{tid}_task.csv") for tid in run_task_ids]
            per_task_metrics, avg_metrics = evaluate_infer.evaluate_infer_paths(
                infer_paths=infer_paths,
                task_ids=run_task_ids,
                config_path=str(tasks_cfg_path),
                test_data=str(eval_in_csv),
                mol_col=str(eval_mol_col),
                seed=None,
                verbose=True,
            )
            per_task_loose = {k: float(v[0]) for k, v in per_task_metrics.items()}
            return float(avg_metrics[0]), per_task_loose
        finally:
            try:
                tmp_ckpt.unlink()
            except Exception:
                pass
            try:
                tmp_infer_cfg.unlink()
            except Exception:
                pass
            if tmp_tasks_cfg is not None:
                try:
                    tmp_tasks_cfg.unlink()
                except Exception:
                    pass

    global_step = 0
    for epoch in range(1, int(train_cfg.get("max_epochs", 50)) + 1):
        best_before_epoch = float(best_metric)
        dyn_cfg = train_cfg.get("dynamic_task_weights", {})
        dyn_enabled = bool(dyn_cfg.get("enabled", False)) if isinstance(dyn_cfg, dict) else bool(dyn_cfg)
        dyn_every = int(dyn_cfg.get("every_steps", 0)) if isinstance(dyn_cfg, dict) else 0
        dyn_topk = int(dyn_cfg.get("topk", 5)) if isinstance(dyn_cfg, dict) else 5
        dyn_boost = float(dyn_cfg.get("boost", 2.0)) if isinstance(dyn_cfg, dict) else 2.0
        task_weights_live: dict[str, float] | None = {} if dyn_enabled else None
        global_step = train_one_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            epoch=epoch,
            max_epochs=int(train_cfg.get("max_epochs", 50)),
            log_every_steps=int(train_cfg.get("log_every_steps", 50)),
            val_loader=val_loader,
            eval_every_steps=int(train_cfg.get("eval_every_steps", 0)),
            eval_num_batches=int(train_cfg.get("eval_num_batches", 0)),
            on_val=None if use_infer_eval else (lambda v, s: _maybe_save(v, tag=f"step={s}")),
            global_step_start=global_step,
            moe_balance_weight=float(train_cfg.get("moe_balance_weight", 0.0)),
            task_weights=task_weights_live,
            infer_every_steps=int(eval_every_steps),
            infer_eval_cb=_infer_eval_step if (use_infer_eval and eval_every_steps > 0) else None,
            fg_curriculum=fg_curriculum,
            fg_curriculum_start=fg_curriculum_start,
            fg_curriculum_end=fg_curriculum_end,
            extra_boost_cb=_extra_boost_cb if extra_boost_enabled and extra_boost_every_steps > 0 else None,
            remove_loss_weight=float(train_cfg.get("remove_loss_weight", 0.0)),
            dynamic_task_weighting=bool(dyn_enabled),
            dynamic_task_every_steps=int(dyn_every),
            dynamic_task_topk=int(dyn_topk),
            dynamic_task_boost=float(dyn_boost),
            profile_one_step=bool(train_cfg.get("profile_one_step", False)),
        )
        metrics = eval_metrics(
            model,
            val_loader,
            device=device,
        )
        if extra_boost_enabled and extra_boost_every_steps <= 0:
            _extra_boost_cb(metrics, int(global_step))
        avg_loose = None
        per_task_acc = None
        if use_infer_eval and eval_every_epochs > 0 and epoch % eval_every_epochs == 0:
            try:
                infer_res = _infer_and_eval(tag=f"epoch_{epoch}")
                if infer_res is not None:
                    avg_loose, per_task_acc = infer_res
                    best_before = float(best_metric)
                    if per_task_acc:
                        per_task_str = ", ".join(f"{k}:{float(v):.2f}" for k, v in sorted(per_task_acc.items()))
                    else:
                        per_task_str = ""
                    print(f"infer_eval epoch={epoch} avg_loose={float(avg_loose):.2f} best_before={best_before:.2f}")
                    if per_task_str:
                        print(f"infer_eval epoch={epoch} per_task_loose={per_task_str}")
            except Exception as exc:
                print(f"infer-eval failed: {type(exc).__name__}: {exc}")
        if avg_loose is not None:
            _maybe_save(float(avg_loose), tag=f"epoch={epoch}")
        elif not use_infer_eval:
            _maybe_save(float(metrics["op_f1_macro"]), tag=f"epoch={epoch}")
        print(
            f"epoch={epoch} val_loss={metrics['loss']:.4f} op_loss={metrics['op_loss']:.4f} fg_loss={metrics['fg_loss']:.4f} "
            f"op_acc={metrics['op_acc']:.3f} acc_none={metrics['acc_none']:.3f} acc_add={metrics['acc_add']:.3f} "
            f"acc_remove={metrics['acc_remove']:.3f} acc_replace={metrics['acc_replace']:.3f} fg_acc={metrics['fg_acc']:.3f} "
            f"remove_loss={metrics.get('remove_loss', 0.0):.4f} remove_acc={metrics.get('remove_acc', 0.0):.3f} "
            f"op_f1={metrics['op_f1_macro']:.3f} fg_f1={metrics['fg_f1']:.3f} op_n={metrics['op_n']} fg_n={metrics['fg_n']} "
            f"best_{metric_name}={best_metric:.4f}"
        )

        if use_infer_eval:
            if avg_loose is None:
                continue
            if float(avg_loose) > best_before_epoch:
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"early stop: no improvement for {patience} epochs")
                    break
        else:
            if float(metrics["op_f1_macro"]) > best_before_epoch:
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"early stop: no improvement for {patience} epochs")
                    break


if __name__ == "__main__":
    main()
