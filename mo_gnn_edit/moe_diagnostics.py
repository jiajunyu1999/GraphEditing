from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from .config import Config
from .dataset_oneshot import OneShotEditDataset
from .model_oneshot import OneShotTwoStageEditModel
from .vocab import FunctionalGroupVocab


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


def _fmt_top_pairs(pairs: list[tuple[int, float]], *, k: int = 10) -> str:
    pairs = sorted(pairs, key=lambda kv: (-float(kv[1]), int(kv[0])))[: int(k)]
    return ", ".join([f"{i}:{v:.3f}" for i, v in pairs])


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="MoE gating diagnostics for OneShotTwoStageEditModel.")
    parser.add_argument("--train_config", type=str, default="train_oneshot_config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Defaults to train.save_path from train_config.")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--graphs", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default=None, help="Override train.device")
    args = parser.parse_args()

    cfg_all = _load_yaml(args.train_config)
    data_cfg = cfg_all.get("data", {}) or {}
    model_cfg = cfg_all.get("model", {}) or {}
    train_cfg = cfg_all.get("train", {}) or {}

    ckpt_path = args.checkpoint or str(train_cfg.get("save_path", "checkpoints/best_oneshot.pt"))
    ckpt = torch.load(Path(ckpt_path), map_location="cpu")

    base_cfg = Config.load(str(data_cfg.get("config_yaml", "config.yaml")))
    fg_vocab_path = base_cfg.resolve_path(base_cfg.raw["chemistry"]["functional_groups_json"])
    fg_vocab = FunctionalGroupVocab.load(fg_vocab_path)

    property_names = list(base_cfg.raw.get("chemistry", {}).get("property_names", []))
    dataset = OneShotEditDataset(
        str(data_cfg.get("dataset_csv", "")),
        property_names,
        seed=int(data_cfg.get("seed", 42)),
    )

    # Same split strategy as training.
    val_ratio = float(data_cfg.get("val_ratio", 0.05))
    n = len(dataset)
    gen = torch.Generator().manual_seed(int(data_cfg.get("seed", 42)))
    perm = torch.randperm(n, generator=gen).tolist()
    n_val = int(math.floor(float(n) * float(val_ratio)))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    idx = train_idx if str(args.split) == "train" else val_idx

    # Limit graphs for speed.
    want = int(args.graphs)
    if want > 0:
        idx = idx[:want]

    subset = torch.utils.data.Subset(dataset, idx)
    loader = torch.utils.data.DataLoader(subset, batch_size=int(args.batch_size), shuffle=False, num_workers=0)

    # Model init must match checkpoint shapes.
    state_dict = ckpt.get("model_state_dict", {})
    ckpt_moe_w = state_dict.get("moe_gate.weight")
    ckpt_num_experts = int(ckpt_moe_w.shape[0]) if ckpt_moe_w is not None else int(model_cfg.get("num_experts", 0))
    use_moe = bool(model_cfg.get("use_moe", ckpt_moe_w is not None))
    num_experts = int(model_cfg.get("num_experts", ckpt_num_experts if ckpt_num_experts > 0 else 0))
    moe_topk = int(model_cfg.get("moe_topk", 1))

    prop_cols = list(ckpt.get("property_cols", []))
    num_tasks_cfg = int(model_cfg.get("num_tasks", -1))
    if num_tasks_cfg <= 0:
        num_tasks_cfg = int(ckpt.get("num_tasks", 0))
    task_emb_dim_cfg = int(model_cfg.get("task_emb_dim", -1))
    if task_emb_dim_cfg <= 0:
        task_emb_dim_cfg = int(ckpt.get("task_emb_dim", 32))
    model = OneShotTwoStageEditModel(
        props_dim=len(prop_cols),
        fg_vocab_size=fg_vocab.size,
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

    # Forgiving load (ignore missing/unexpected keys and size mismatches).
    try:
        from torch.nn.parameter import UninitializedParameter  # type: ignore
    except Exception:  # pragma: no cover
        UninitializedParameter = ()  # type: ignore

    model_sd = model.state_dict()
    filtered: dict[str, Any] = {}
    for k, v in (state_dict or {}).items():
        if k not in model_sd:
            continue
        if isinstance(model_sd[k], UninitializedParameter):
            filtered[k] = v
            continue
        try:
            if hasattr(v, "shape") and hasattr(model_sd[k], "shape") and tuple(v.shape) != tuple(model_sd[k].shape):
                continue
        except Exception:
            continue
        filtered[k] = v
    model.load_state_dict(filtered, strict=False)

    device = torch.device(str(args.device) if args.device else str(train_cfg.get("device", "cpu")))
    model.to(device)
    model.eval()

    # Aggregation (overall + by task_id).
    sum_ent = 0.0
    sum_maxp = 0.0
    sum_topk_mass = 0.0
    sum_g = 0
    sum_gate = None
    topk_present = None  # [E] count of graphs where expert appears in topk

    by_task = defaultdict(lambda: {"n": 0, "sum_ent": 0.0, "sum_maxp": 0.0, "sum_topk_mass": 0.0, "present": None})

    for batch in loader:
        batch = batch.to(device)
        out = model(batch, teacher_forcing=False)
        if out.moe_gate_probs is None or out.moe_topi is None:
            raise RuntimeError("Model did not return MoE routing outputs; ensure use_moe=True and num_experts>0.")
        gate_probs = out.moe_gate_probs  # [G, E]
        topi = out.moe_topi  # [G, K]
        entropy = out.moe_entropy
        topk_mass = out.moe_topk_mass
        if entropy is None or topk_mass is None:
            raise RuntimeError("Missing moe_entropy/moe_topk_mass in model output.")

        gsz, esz = int(gate_probs.size(0)), int(gate_probs.size(1))
        if sum_gate is None:
            sum_gate = gate_probs.detach().sum(dim=0).to(device="cpu")
        else:
            sum_gate = sum_gate + gate_probs.detach().sum(dim=0).to(device="cpu")

        maxp = gate_probs.max(dim=-1).values
        sum_ent += float(entropy.sum().item())
        sum_maxp += float(maxp.sum().item())
        sum_topk_mass += float(topk_mass.sum().item())
        sum_g += int(gsz)

        present = torch.zeros((gsz, esz), device=gate_probs.device, dtype=torch.float32)
        present.scatter_(1, topi, 1.0)
        present_cnt = present.sum(dim=0).to(device="cpu")  # [E]
        if topk_present is None:
            topk_present = present_cnt
        else:
            topk_present = topk_present + present_cnt

        if hasattr(batch, "task_id"):
            tids = batch.task_id.view(-1).detach().to("cpu")
            for tid in torch.unique(tids).tolist():
                tid = int(tid)
                if tid < 0:
                    continue
                m = tids == tid
                nn = int(m.sum().item())
                if nn <= 0:
                    continue
                bt = by_task[tid]
                bt["n"] += nn
                bt["sum_ent"] += float(entropy.detach().to("cpu")[m].sum().item())
                bt["sum_maxp"] += float(maxp.detach().to("cpu")[m].sum().item())
                bt["sum_topk_mass"] += float(topk_mass.detach().to("cpu")[m].sum().item())
                pc = present.detach().to("cpu")[m].sum(dim=0)  # [E]
                if bt["present"] is None:
                    bt["present"] = pc
                else:
                    bt["present"] = bt["present"] + pc

    if sum_g <= 0 or sum_gate is None or topk_present is None:
        raise RuntimeError("No graphs processed.")

    mean_ent = sum_ent / float(sum_g)
    mean_ppl = math.exp(mean_ent) if mean_ent < 50 else float("inf")
    mean_maxp = sum_maxp / float(sum_g)
    mean_topk_mass = sum_topk_mass / float(sum_g)

    mean_gate = (sum_gate / float(sum_g)).tolist()
    topk_freq = (topk_present / float(sum_g)).tolist()

    print(f"MoE diagnostics split={args.split} graphs={sum_g} experts={len(mean_gate)} topk={int(model.moe_topk)}")
    print(f"overall: mean_entropy={mean_ent:.3f} mean_perplexity={mean_ppl:.2f} mean_max_prob={mean_maxp:.3f} mean_topk_mass={mean_topk_mass:.3f}")

    top = sorted([(i, float(v)) for i, v in enumerate(topk_freq)], key=lambda kv: (-kv[1], kv[0]))[:10]
    print("overall: topk_freq_per_graph (top): " + _fmt_top_pairs(top, k=10))
    top = sorted([(i, float(v)) for i, v in enumerate(mean_gate)], key=lambda kv: (-kv[1], kv[0]))[:10]
    print("overall: mean_gate_prob (top): " + _fmt_top_pairs(top, k=10))

    for tid in sorted(by_task.keys()):
        bt = by_task[tid]
        n = int(bt["n"])
        if n <= 0:
            continue
        ent = float(bt["sum_ent"]) / float(n)
        maxp = float(bt["sum_maxp"]) / float(n)
        tkm = float(bt["sum_topk_mass"]) / float(n)
        present = bt["present"]
        if present is None:
            continue
        freq = (present / float(n)).tolist()
        top = sorted([(i, float(v)) for i, v in enumerate(freq)], key=lambda kv: (-kv[1], kv[0]))[:5]
        print(
            f"task_id={tid} n={n} mean_entropy={ent:.3f} mean_max_prob={maxp:.3f} topk_mass={tkm:.3f} topk_freq_top="
            + _fmt_top_pairs(top, k=5)
        )


if __name__ == "__main__":
    main()
