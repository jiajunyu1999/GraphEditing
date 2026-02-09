from __future__ import annotations

"""
Single-step PPO fine-tuning for `OneShotTwoStageEditModel`.

Design:
  - State: (start_smiles_tagged, desired property deltas)
  - Action: one sampled edit-set (per-node op + fg for add/replace)
  - Transition: apply edits -> edited SMILES
  - Reward: user-defined; default is "trend alignment" on properties
    - Optional: worst-task reward over all tasks in tasks.yaml (min over task rewards).

This script loads a supervised checkpoint (best_oneshot.pt) and continues training with PPO.
"""

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .chem_utils import mol_from_smiles, smiles_without_atom_maps
from .config import Config
from .dataset_oneshot import OP_TO_ID
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


def _load_tasks(path: str | Path) -> tuple[dict[str, tuple[list[str], str]], dict[str, float]]:
    raw = _load_yaml(str(path))
    taskid_prop = raw.get("taskid_prop")
    prop_trend = raw.get("prop_trend")
    prop_threshold = raw.get("prop_threshold", {})
    if not isinstance(taskid_prop, dict) or not isinstance(prop_trend, dict):
        raise ValueError("tasks_yaml must contain dict keys: taskid_prop and prop_trend")
    if not isinstance(prop_threshold, dict):
        prop_threshold = {}
    out: dict[str, tuple[list[str], str]] = {}
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
        out[tid] = (list(props), trend)
    if not out:
        raise ValueError("No tasks found in tasks_yaml")
    thresholds = {str(k): float(v) for k, v in prop_threshold.items()}
    return out, thresholds


def _finite_or_zero(x: Any) -> float:
    try:
        v = float(x)
        if not math.isfinite(v):
            return 0.0
        return v
    except Exception:
        return 0.0


def _parse_prop_cols(property_cols: list[str]) -> tuple[list[str], list[str], list[str]]:
    delta_cols = [c for c in property_cols if c.endswith("_delta")]
    src_cols = [c for c in property_cols if c.endswith("_src")]
    gen_cols = [c for c in property_cols if c.endswith("_gen")]
    return delta_cols, src_cols, gen_cols


def _build_props_tensor(
    *,
    property_cols: list[str],
    src_raw: dict[str, float],
    delta_raw_by_prop: dict[str, float],
    device: torch.device,
) -> torch.Tensor:
    # property_cols is the training-time conditioning vector layout.
    props_raw = torch.zeros((1, len(property_cols)), dtype=torch.float32)
    for i, c in enumerate(property_cols):
        if c.endswith("_delta"):
            p = c[: -len("_delta")]
            props_raw[0, i] = float(delta_raw_by_prop.get(p, 0.0))
        elif c.endswith("_src"):
            p = c[: -len("_src")]
            props_raw[0, i] = float(src_raw.get(p, 0.0))
        elif c.endswith("_gen"):
            p = c[: -len("_gen")]
            props_raw[0, i] = float(src_raw.get(p, 0.0) + delta_raw_by_prop.get(p, 0.0))
        else:
            props_raw[0, i] = 0.0
    return props_raw.to(device)


def _prepare_start_smiles_tagged(smiles: str) -> str:
    from .chem_utils import ensure_atom_maps, mol_to_tagged_smiles, normalize_bracket_atom_hydrogens

    mol = mol_from_smiles(smiles)
    mol = ensure_atom_maps(mol)
    mol = normalize_bracket_atom_hydrogens(mol)
    return mol_to_tagged_smiles(mol)


@dataclass(frozen=True)
class StepBatch:
    start_smiles_tagged: str
    start_smiles_plain: str
    props: torch.Tensor  # [1, D]
    delta_cols_only: list[str]
    delta_raw_by_prop: dict[str, float]
    src_raw: dict[str, float]


def _reward_trend_alignment(
    *,
    start_smiles_plain: str,
    final_smiles_plain: str | None,
    props_seen: set[str],
    delta_raw_by_prop: dict[str, float],
    src_raw: dict[str, float],
    property_fns: dict[str, Any],
    wrong_dir_coef: float,
    invalid_penalty: float,
) -> float:
    if final_smiles_plain is None:
        return -float(invalid_penalty)

    try:
        from .properties import calc_properties
    except Exception:
        raise RuntimeError("Missing mo_gnn_edit/properties.py with calc_properties() required for reward.")

    try:
        final_raw = calc_properties(mol_from_smiles(final_smiles_plain), property_fns)
    except Exception:
        return -float(invalid_penalty)

    reward = 0.0
    for p in props_seen:
        desired = float(delta_raw_by_prop.get(p, 0.0))
        if desired == 0.0:
            continue
        sr = float(src_raw.get(p, 0.0))
        fr = float(final_raw.get(p, sr))
        actual = fr - sr
        mag = abs(actual)
        if desired * actual >= 0:
            reward += mag
        else:
            reward -= float(wrong_dir_coef) * mag
    return float(reward)


def _reward_task_trend_from_raw(
    *,
    task_props: list[str],
    task_trend: str,
    src_raw: dict[str, float],
    final_raw: dict[str, float],
    wrong_dir_coef: float,
) -> float:
    reward = 0.0
    for prop, bit in zip(task_props, task_trend):
        sign = 1.0 if bit == "1" else -1.0
        sr = _finite_or_zero(src_raw.get(prop, 0.0))
        fr = _finite_or_zero(final_raw.get(prop, sr))
        actual = fr - sr
        mag = abs(actual)
        if mag <= 0.0:
            continue
        if sign * actual >= 0.0:
            reward += mag
        else:
            reward -= float(wrong_dir_coef) * mag
    return float(reward)


class ValueHead(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, *, graph_h: torch.Tensor) -> torch.Tensor:
        return self.net(graph_h).squeeze(-1)  # [B]


def _sample_edit_set_and_logprob(
    model,
    data0,
    *,
    fg_vocab: FunctionalGroupVocab,
    max_edits: int,
    temperature_op: float,
    temperature_fg: float,
    seed: int,
) -> tuple[list[dict[str, Any]], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample one edit-set and return (edits, logprob, entropy).
    logprob/entropy are scalar tensors on model device.
    """
    require_torch_geometric()
    from torch_geometric.data import Batch

    device = next(model.parameters()).device
    try:
        g = torch.Generator(device=device)
    except Exception:
        g = torch.Generator()
    g.manual_seed(int(seed))

    batch = Batch.from_data_list([data0]).to(device)
    node_h, graph_h, props_tokens = model.encode(batch)
    fused, _ = model._fuse_node_inputs(node_h, props_tokens, batch)

    op_logits = model.op_head(fused)  # [N,4]
    if hasattr(batch, "op_allowed"):
        op_allowed = batch.op_allowed
    else:
        op_allowed = torch.ones((int(batch.z.size(0)), 4), device=batch.z.device, dtype=torch.bool)
    op_logits = op_logits.masked_fill(~op_allowed, -1e9)

    op_log_probs = F.log_softmax(op_logits / max(1e-6, float(temperature_op)), dim=-1)
    op_probs = op_log_probs.exp()
    op_id = torch.multinomial(op_probs, num_samples=1, generator=g).squeeze(-1)  # [N]

    if int(max_edits) > 0:
        non_none = (op_id != OP_TO_ID["none"]).nonzero(as_tuple=True)[0]
        if int(non_none.numel()) > int(max_edits):
            score = (1.0 - op_probs[:, OP_TO_ID["none"]]).detach()
            top = torch.topk(score[non_none], k=int(max_edits), largest=True).indices
            keep_nodes = non_none[top]
            keep_mask = torch.zeros((int(op_id.numel()),), device=device, dtype=torch.bool)
            keep_mask[keep_nodes] = True
            op_id[(op_id != OP_TO_ID["none"]) & (~keep_mask)] = OP_TO_ID["none"]

    fg_mask = (op_id == OP_TO_ID["add"]) | (op_id == OP_TO_ID["replace"])
    fg_logits = torch.zeros((int(node_h.size(0)), int(model.fg_vocab_size)), device=device, dtype=op_logits.dtype)
    if bool(fg_mask.any().item()):
        fg_logits[fg_mask] = model.fg_head(fused[fg_mask])
    fg_log_probs = F.log_softmax(fg_logits / max(1e-6, float(temperature_fg)), dim=-1)

    # logprob = sum_i log p(op_i) + sum_{i in add/rep} log p(fg_i)
    logprob_op = op_log_probs.gather(1, op_id.unsqueeze(-1)).squeeze(-1).sum()
    entropy_op = -(op_log_probs * op_probs).sum(dim=-1).sum()

    fg_id = torch.full((int(op_id.numel()),), -1, device=device, dtype=torch.long)
    logprob_fg = torch.zeros((), device=device)
    entropy_fg = torch.zeros((), device=device)
    if bool(fg_mask.any().item()):
        fg_probs = fg_log_probs.exp()
        fg_id_sel = torch.multinomial(fg_probs[fg_mask], num_samples=1, generator=g).squeeze(-1)
        fg_id[fg_mask] = fg_id_sel
        logprob_fg = fg_log_probs[fg_mask].gather(1, fg_id_sel.unsqueeze(-1)).squeeze(-1).sum()
        entropy_fg = -(fg_log_probs[fg_mask] * fg_probs[fg_mask]).sum(dim=-1).sum()

    logprob = logprob_op + logprob_fg
    entropy = entropy_op + entropy_fg

    fg_id_to_smiles = getattr(fg_vocab, "id_to_smiles", {}) or {}
    node0 = (batch.batch == 0).nonzero(as_tuple=True)[0]
    edits: list[dict[str, Any]] = []
    for gi in node0.tolist():
        oid = int(op_id[gi].item())
        if oid == OP_TO_ID["none"]:
            continue
        op = {0: "none", 1: "add", 2: "remove", 3: "replace"}[oid]
        anchor_map = int(batch.atom_map[gi].item())
        e: dict[str, Any] = {"anchor_atom_map": anchor_map, "op": op}
        if op in ("add", "replace"):
            mid = int(fg_id[gi].item())
            smi = fg_id_to_smiles.get(mid)
            if smi is None:
                continue
            e["fg_id"] = int(mid)
            e["fg_smiles"] = str(smi)
        edits.append(e)

    # Stable ordering
    def _op_rank(op_name: str) -> int:
        return {"remove": 0, "replace": 1, "add": 2}.get(op_name, 99)

    edits = sorted(edits, key=lambda e: (_op_rank(str(e["op"])), int(e["anchor_atom_map"])))
    for si, e in enumerate(edits, start=1):
        e["step"] = si
    return edits, logprob, entropy, op_id.detach(), fg_id.detach()


def _logprob_entropy_for_action(
    model,
    data0,
    *,
    op_id: torch.Tensor,
    fg_id: torch.Tensor,
    temperature_op: float,
    temperature_fg: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Recompute (logprob, entropy, value_features) for a stored (op_id, fg_id) action.
    Returns (logprob, entropy, fused) on model device.
    """
    require_torch_geometric()
    from torch_geometric.data import Batch

    device = next(model.parameters()).device
    batch = Batch.from_data_list([data0]).to(device)
    node_h, graph_h, props_tokens = model.encode(batch)
    fused, _ = model._fuse_node_inputs(node_h, props_tokens, batch)

    op_logits = model.op_head(fused)
    if hasattr(batch, "op_allowed"):
        op_allowed = batch.op_allowed
    else:
        op_allowed = torch.ones((int(batch.z.size(0)), 4), device=batch.z.device, dtype=torch.bool)
    op_logits = op_logits.masked_fill(~op_allowed, -1e9)

    op_id = op_id.to(device=device, dtype=torch.long)
    fg_id = fg_id.to(device=device, dtype=torch.long)

    op_log_probs = F.log_softmax(op_logits / max(1e-6, float(temperature_op)), dim=-1)
    op_probs = op_log_probs.exp()
    logprob_op = op_log_probs.gather(1, op_id.unsqueeze(-1)).squeeze(-1).sum()
    entropy_op = -(op_log_probs * op_probs).sum(dim=-1).sum()

    fg_mask = (op_id == OP_TO_ID["add"]) | (op_id == OP_TO_ID["replace"])
    fg_logits = torch.zeros((int(node_h.size(0)), int(model.fg_vocab_size)), device=device, dtype=op_logits.dtype)
    if bool(fg_mask.any().item()):
        fg_logits[fg_mask] = model.fg_head(fused[fg_mask])
    fg_log_probs = F.log_softmax(fg_logits / max(1e-6, float(temperature_fg)), dim=-1)
    fg_probs = fg_log_probs.exp()
    logprob_fg = torch.zeros((), device=device)
    entropy_fg = torch.zeros((), device=device)
    if bool(fg_mask.any().item()):
        fg_id_sel = fg_id[fg_mask].clamp_min(0).clamp_max(int(model.fg_vocab_size) - 1)
        logprob_fg = fg_log_probs[fg_mask].gather(1, fg_id_sel.unsqueeze(-1)).squeeze(-1).sum()
        entropy_fg = -(fg_log_probs[fg_mask] * fg_probs[fg_mask]).sum(dim=-1).sum()

    return logprob_op + logprob_fg, entropy_op + entropy_fg, graph_h


def main() -> None:
    _disable_rdkit_warnings()
    parser = argparse.ArgumentParser(description="PPO fine-tuning for OneShotTwoStageEditModel (single-step episodes).")
    parser.add_argument("--checkpoint", type=str, required=True, help="Supervised checkpoint .pt from train_oneshot.")
    parser.add_argument("--train_config", type=str, default="train_oneshot_config.yaml", help="YAML with model hyperparams.")
    parser.add_argument("--input_csv", type=str, required=True, help="CSV with input molecules.")
    parser.add_argument("--smiles_col", type=str, default="mol")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--steps_per_epoch", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--max_edits", type=int, default=10)
    parser.add_argument("--temperature_op", type=float, default=1.0)
    parser.add_argument("--temperature_fg", type=float, default=1.0)
    parser.add_argument("--invalid_resample", type=int, default=3, help="Resample action if edit yields invalid molecule.")

    parser.add_argument(
        "--reward_mode",
        type=str,
        default="trend",
        choices=["trend", "worst_task"],
        help="Reward function: trend uses delta conditioning; worst_task uses min reward across tasks.yaml.",
    )
    parser.add_argument("--tasks_yaml", type=str, default="tasks.yaml")
    parser.add_argument("--delta_sparsity", type=float, default=0.3, help="Fraction of props to activate per step.")
    parser.add_argument("--trend_alpha", type=float, default=1.0, help="Magnitude in z-space; mapped by model delta_scale.")
    parser.add_argument("--reward_interval", type=int, default=10, help="Compute reward every N steps (others get 0).")

    parser.add_argument("--wrong_dir_coef", type=float, default=10.0)
    parser.add_argument("--invalid_penalty", type=float, default=1.0)

    parser.add_argument("--save_path", type=str, default="checkpoints/ppo_finetuned.pt")
    args = parser.parse_args()

    require_torch_geometric()
    from torch_geometric.data import Data

    import pandas as pd

    from .apply import apply_edits
    from .featurize import featurize_tagged_smiles
    from .properties import build_property_fns, calc_properties
    from .model_oneshot import OneShotTwoStageEditModel

    device = torch.device(str(args.device))
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    train_cfg = _load_yaml(args.train_config)
    model_cfg = train_cfg.get("model", {})

    ckpt = torch.load(Path(args.checkpoint), map_location="cpu")
    property_cols = list(ckpt.get("property_cols", []))
    if not property_cols:
        raise ValueError("Checkpoint missing property_cols; cannot build conditioning vector.")

    # Model init from config + checkpoint.
    fg_vocab_path = ckpt.get("fg_vocab_path")
    config_yaml = ckpt.get("config_yaml")
    if not fg_vocab_path or not config_yaml:
        raise ValueError("Checkpoint missing fg_vocab_path/config_yaml; required for PPO.")

    cfg = Config.load(str(config_yaml))
    fg_vocab = FunctionalGroupVocab.load(str(fg_vocab_path))

    num_tasks_cfg = int(model_cfg.get("num_tasks", -1))
    if num_tasks_cfg <= 0:
        num_tasks_cfg = int(ckpt.get("num_tasks", 0))
    task_emb_dim_cfg = int(model_cfg.get("task_emb_dim", -1))
    if task_emb_dim_cfg <= 0:
        task_emb_dim_cfg = int(ckpt.get("task_emb_dim", 32))
    model = OneShotTwoStageEditModel(
        props_dim=len(property_cols),
        fg_vocab_size=fg_vocab.size,
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
        num_layers=int(model_cfg.get("num_layers", 5)),
        gine_layers=model_cfg.get("gine_layers"),
        mlp_layers=model_cfg.get("mlp_layers"),
        num_tasks=num_tasks_cfg,
        task_emb_dim=task_emb_dim_cfg,
        dropout=float(model_cfg.get("dropout", 0.1)),
        backbone=str(model_cfg.get("backbone", "gine")),
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device)
    model.train()

    value_head = ValueHead(model.hidden_dim).to(device)
    value_head.train()

    # Property reward functions.
    property_names = list(cfg.raw.get("chemistry", {}).get("property_names", []))
    property_fns = build_property_fns(property_names)
    tasks, prop_thresholds = _load_tasks(args.tasks_yaml)

    # PPO optimizer (policy + value).
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(value_head.parameters()),
        lr=float(args.lr),
        weight_decay=float(train_cfg.get("train", {}).get("weight_decay", 1e-5)),
    )

    df = pd.read_csv(cfg.resolve_path(args.input_csv), engine="c")
    if args.smiles_col not in df.columns:
        raise KeyError(f"Missing smiles_col={args.smiles_col}. Columns: {list(df.columns)}")

    delta_cols, src_cols, gen_cols = _parse_prop_cols(property_cols)
    props_seen = set([c[: -len("_delta")] for c in delta_cols])
    if args.reward_mode == "trend" and not props_seen:
        raise ValueError("property_cols has no *_delta columns; PPO expects delta conditioning.")
    if args.reward_mode == "worst_task" and not tasks:
        raise ValueError("reward_mode=worst_task requires tasks_yaml with task definitions.")

    # Use model buffers for delta scaling.
    delta_scale = getattr(model, "prop_delta_scale", None)
    prop_list = list(cfg.raw.get("chemistry", {}).get("property_names", []))
    scale_by_prop: dict[str, float] = {}
    if isinstance(delta_scale, torch.Tensor) and int(delta_scale.numel()) > 0:
        for i, p in enumerate(prop_list):
            if i >= int(delta_scale.numel()):
                break
            scale_by_prop[str(p)] = float(delta_scale.view(-1)[i].detach().cpu().item())

    def _sample_delta_raw_by_prop(*, rng: random.Random) -> dict[str, float]:
        out: dict[str, float] = {}
        props = sorted(props_seen)
        k = max(1, int(round(float(args.delta_sparsity) * len(props))))
        chosen = set(rng.sample(props, k=min(k, len(props))))
        for p in props:
            if p not in chosen:
                out[p] = 0.0
                continue
            sign = 1.0 if rng.random() < 0.5 else -1.0
            z = sign * float(args.trend_alpha)
            scale = float(scale_by_prop.get(p, 1.0))
            out[p] = scale * z
        return out

    # Rollout storage.
    roll_data: list[Data] = []
    roll_edits: list[list[dict[str, Any]]] = []
    roll_op_id: list[torch.Tensor] = []
    roll_fg_id: list[torch.Tensor] = []
    roll_logp_old: list[torch.Tensor] = []
    roll_reward: list[torch.Tensor] = []

    for epoch in range(1, int(args.epochs) + 1):
        roll_data.clear()
        roll_edits.clear()
        roll_op_id.clear()
        roll_fg_id.clear()
        roll_logp_old.clear()
        roll_reward.clear()

        rng = random.Random(int(args.seed) + epoch * 10007)
        rows = df.sample(n=int(args.steps_per_epoch), replace=True, random_state=int(args.seed) + epoch).to_dict("records")

        task_hits_loose = {tid: 0 for tid in tasks}
        task_hits_strict = {tid: 0 for tid in tasks}
        task_counts = {tid: 0 for tid in tasks}
        valid_steps = 0
        total_steps = 0

        for step_i, row in enumerate(rows):
            smiles_plain = str(row[args.smiles_col])
            try:
                start_mol = mol_from_smiles(smiles_plain)
            except Exception:
                continue
            start_smiles_plain = smiles_without_atom_maps(start_mol)
            try:
                start_smiles_tagged = _prepare_start_smiles_tagged(start_smiles_plain)
            except Exception:
                start_smiles_tagged = start_smiles_plain

            try:
                src_raw = calc_properties(mol_from_smiles(start_smiles_plain), property_fns) if property_fns else {}
            except Exception:
                src_raw = {}

            delta_raw_by_prop = _sample_delta_raw_by_prop(rng=rng)
            props = _build_props_tensor(
                property_cols=property_cols,
                src_raw=src_raw,
                delta_raw_by_prop=delta_raw_by_prop,
                device=device,
            )

            f = featurize_tagged_smiles(start_smiles_tagged)
            data0 = Data(
                z=f.z,
                x=f.x,
                edge_index=f.edge_index,
                edge_type=f.edge_type,
                atom_map=f.atom_map,
                props=props,
            )
            data0.op_allowed = torch.ones((int(f.z.size(0)), 4), device=device, dtype=torch.bool)

            edits = []
            logp_old = None
            op_id = None
            fg_id = None
            final_smiles_plain = None
            max_tries = max(0, int(args.invalid_resample)) + 1
            for attempt in range(max_tries):
                edits, logp_old, _entropy, op_id, fg_id = _sample_edit_set_and_logprob(
                    model,
                    data0,
                    fg_vocab=fg_vocab,
                    max_edits=int(args.max_edits),
                    temperature_op=float(args.temperature_op),
                    temperature_fg=float(args.temperature_fg),
                    seed=int(args.seed) + epoch * 100000 + step_i * 997 + attempt,
                )
                try:
                    tagged = apply_edits(start_smiles_tagged, edits)
                    final_smiles_plain = smiles_without_atom_maps(mol_from_smiles(tagged))
                except Exception:
                    final_smiles_plain = None
                if final_smiles_plain is not None:
                    break

            compute_reward = int(args.reward_interval) > 0 and ((step_i + 1) % int(args.reward_interval) == 0)
            if not compute_reward:
                r = 0.0
            elif args.reward_mode == "trend":
                r = _reward_trend_alignment(
                    start_smiles_plain=start_smiles_plain,
                    final_smiles_plain=final_smiles_plain,
                    props_seen=props_seen,
                    delta_raw_by_prop=delta_raw_by_prop,
                    src_raw=src_raw,
                    property_fns=property_fns,
                    wrong_dir_coef=float(args.wrong_dir_coef),
                    invalid_penalty=float(args.invalid_penalty),
                )
            else:
                if final_smiles_plain is None:
                    r = -float(args.invalid_penalty)
                else:
                    try:
                        final_raw = calc_properties(mol_from_smiles(final_smiles_plain), property_fns)
                    except Exception:
                        final_raw = {}
                    if not final_raw:
                        r = -float(args.invalid_penalty)
                    else:
                        task_rewards = [
                            _reward_task_trend_from_raw(
                                task_props=props,
                                task_trend=trend,
                                src_raw=src_raw,
                                final_raw=final_raw,
                                wrong_dir_coef=float(args.wrong_dir_coef),
                            )
                            for props, trend in tasks.values()
                        ]
                        r = float(min(task_rewards)) if task_rewards else -float(args.invalid_penalty)

            if compute_reward and tasks:
                total_steps += 1
                if final_smiles_plain is None:
                    final_raw = None
                else:
                    try:
                        final_raw = calc_properties(mol_from_smiles(final_smiles_plain), property_fns)
                    except Exception:
                        final_raw = None
                if final_raw is not None:
                    valid_steps += 1
                for tid, (props, trend) in tasks.items():
                    task_counts[tid] += 1
                    if final_raw is None:
                        continue
                    loose = True
                    strict = True
                    for prop, bit in zip(props, trend):
                        sign = 1.0 if bit == "1" else -1.0
                        sr = _finite_or_zero(src_raw.get(prop, 0.0))
                        fr = _finite_or_zero(final_raw.get(prop, sr))
                        delta = fr - sr
                        thr = float(prop_thresholds.get(prop, 0.0))
                        if sign > 0:
                            if delta <= 0.0:
                                loose = False
                            if delta <= thr:
                                strict = False
                        else:
                            if delta >= 0.0:
                                loose = False
                            if delta >= -thr:
                                strict = False
                    if loose:
                        task_hits_loose[tid] += 1
                    if strict:
                        task_hits_strict[tid] += 1

            roll_data.append(data0.cpu())
            roll_edits.append(edits)
            roll_op_id.append(op_id.cpu())
            roll_fg_id.append(fg_id.cpu())
            roll_logp_old.append(logp_old.detach())
            roll_reward.append(torch.tensor(r, device=device, dtype=torch.float32))

        if not roll_reward:
            raise RuntimeError("No rollouts collected; check inputs and dependencies.")

        rewards = torch.stack(roll_reward)  # [T]
        if int(args.reward_interval) > 0 and tasks:
            loose_str = ", ".join(
                f"{tid}:{(task_hits_loose[tid] / task_counts[tid]):.3f}" if task_counts[tid] else f"{tid}:0.000"
                for tid in sorted(tasks.keys())
            )
            strict_str = ", ".join(
                f"{tid}:{(task_hits_strict[tid] / task_counts[tid]):.3f}" if task_counts[tid] else f"{tid}:0.000"
                for tid in sorted(tasks.keys())
            )
            print(f"epoch={epoch} task_acc_loose[{int(args.reward_interval)}]={loose_str}")
            print(f"epoch={epoch} task_acc_strict[{int(args.reward_interval)}]={strict_str}")
            valid_ratio = (valid_steps / total_steps) if total_steps else 0.0
            print(f"epoch={epoch} valid_ratio[{int(args.reward_interval)}]={valid_ratio:.3f}")

        # PPO updates (single-step, so no GAE needed).
        for _ in range(int(args.ppo_epochs)):
            idxs = torch.randperm(rewards.size(0), device=device)
            for start in range(0, int(idxs.numel()), int(args.batch_size)):
                mb = idxs[start : start + int(args.batch_size)]
                if int(mb.numel()) == 0:
                    continue

                logp_old = torch.stack([roll_logp_old[int(i)].to(device) for i in mb]).detach()

                logp_new_list: list[torch.Tensor] = []
                ent_list: list[torch.Tensor] = []
                v_list: list[torch.Tensor] = []
                for i in mb.tolist():
                    data0 = roll_data[int(i)]
                    op_id = roll_op_id[int(i)]
                    fg_id = roll_fg_id[int(i)]
                    logp_new, entropy, graph_h = _logprob_entropy_for_action(
                        model,
                        data0,
                        op_id=op_id,
                        fg_id=fg_id,
                        temperature_op=float(args.temperature_op),
                        temperature_fg=float(args.temperature_fg),
                    )
                    v = value_head(graph_h=graph_h).squeeze(0)
                    logp_new_list.append(logp_new)
                    ent_list.append(entropy)
                    v_list.append(v)

                logp_new = torch.stack(logp_new_list)
                entropy = torch.stack(ent_list)
                v = torch.stack(v_list)

                returns = rewards[mb]
                adv = (returns - v.detach())

                ratio = torch.exp(logp_new - logp_old)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - float(args.clip_eps), 1.0 + float(args.clip_eps)) * adv
                policy_loss = -(torch.min(surr1, surr2)).mean()
                value_loss = F.mse_loss(v, returns)
                loss = policy_loss + float(args.value_coef) * value_loss - float(args.entropy_coef) * entropy.mean()

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(list(model.parameters()) + list(value_head.parameters()), max_norm=float(args.max_grad_norm))
                optimizer.step()

        save_path = Path(str(args.save_path))
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "value_state_dict": value_head.state_dict(),
                "property_cols": property_cols,
                "fg_vocab_path": str(fg_vocab_path),
                "config_yaml": str(config_yaml),
            },
            save_path,
        )
        print(f"epoch={epoch} reward_mean={float(rewards.mean().item()):.4f} saved={save_path}")


if __name__ == "__main__":
    main()
