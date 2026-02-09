from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .dataset_oneshot import OP_TO_ID
from .featurize import featurize_tagged_smiles
from .pyg_utils import require_torch_geometric


@dataclass(frozen=True)
class Sample:
    z: torch.Tensor  # [N]
    x: torch.Tensor  # [N, 6]
    atom_map: torch.Tensor  # [N]
    edge_index: torch.Tensor  # [2, E]
    edge_type: torch.Tensor  # [E]
    props: torch.Tensor  # [P]
    y_op: torch.Tensor  # [N]
    op_allowed: torch.Tensor  # [N, 4]
    anchor_mask: torch.Tensor  # [N]


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return 0.0
        return v
    except Exception:
        return 0.0


def _load_rows(csv_path: str | Path, *, max_rows: int, seed: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with Path(csv_path).open("r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: (v if v is not None else "") for k, v in r.items()})
            if max_rows and len(rows) >= int(max_rows):
                break
    rng = random.Random(int(seed))
    rng.shuffle(rows)
    return rows


def _resolve_property_cols(header: list[str], property_names: list[str]) -> list[str]:
    delta = [f"{p}_delta" for p in property_names if f"{p}_delta" in header]
    src = [f"{p}_src" for p in property_names if f"{p}_src" in header]
    gen = [f"{p}_gen" for p in property_names if f"{p}_gen" in header]
    if delta and src and gen and len(delta) == len(src) == len(gen):
        return delta + src + gen
    if delta:
        return delta
    raise KeyError("No property columns found: expected at least `*_delta` columns in the CSV header.")


def _compute_op_allowed(z: torch.Tensor, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
    """
    Mirror the current conservative mask used in training/inference.
    - add_ok: implicitHs>0 and not aromatic heteroatom (kept within C/N/O)
    - remove_ok: anchor has a terminal heavy neighbor via SINGLE non-ring bond (approximated by neighbor !in_ring)
    - replace_ok: remove_ok & (implicitHs==0)
    """
    total_h = x[:, 3]
    is_arom = x[:, 4] > 0.5
    add_ok = (total_h > 0.5) & ((z == 6) | (z == 7) | (z == 8)) & (~(is_arom & (z != 6)))

    deg = x[:, 0].round().long()
    in_ring = x[:, 5].round().long()
    is_terminal_heavy = (deg == 1) & (in_ring == 0) & (z > 1)

    src = edge_index[0]
    dst = edge_index[1]
    is_single = edge_type == 0
    ok = is_terminal_heavy[dst] & is_single
    rm_ok = torch.zeros((int(z.numel()),), dtype=torch.bool)
    if bool(ok.any().item()):
        rm_ok[src[ok]] = True
    rep_ok = rm_ok & (total_h < 0.5)

    n = int(z.numel())
    return torch.stack(
        [
            torch.ones((n,), dtype=torch.bool),
            add_ok,
            rm_ok,
            rep_ok,
        ],
        dim=-1,
    )


def _build_sample(row: dict[str, str], *, prop_cols: list[str]) -> Sample | None:
    start_smiles_tagged = str(row.get("start_smiles_tagged", "")).strip()
    edits_json = str(row.get("edits_json", "")).strip()
    if not start_smiles_tagged or not edits_json:
        return None

    f = featurize_tagged_smiles(start_smiles_tagged)
    map_to_idx = {int(m.item()): i for i, m in enumerate(f.atom_map)}

    y_op = torch.zeros((int(f.z.size(0)),), dtype=torch.long)
    try:
        edits = json.loads(edits_json)
    except Exception:
        return None
    if not isinstance(edits, list):
        return None
    for e in edits:
        try:
            op = str(e["op"])
            anchor_map = int(e["anchor_atom_map"])
        except Exception:
            continue
        if anchor_map not in map_to_idx:
            # Dataset row is inconsistent with start_smiles_tagged; skip for this debug script.
            return None
        y_op[map_to_idx[anchor_map]] = int(OP_TO_ID.get(op, 0))

    props = torch.tensor([_safe_float(row.get(c, 0.0)) for c in prop_cols], dtype=torch.float32)

    op_allowed = _compute_op_allowed(f.z, f.x, f.edge_index, f.edge_type)
    anchor_mask = op_allowed[:, 1:].any(dim=-1)

    return Sample(
        z=f.z,
        x=f.x,
        atom_map=f.atom_map,
        edge_index=f.edge_index,
        edge_type=f.edge_type,
        props=props,
        y_op=y_op,
        op_allowed=op_allowed,
        anchor_mask=anchor_mask,
    )


class SimpleOpModel(nn.Module):
    def __init__(self, *, prop_dim: int, hidden: int = 128, max_atomic_num: int = 100) -> None:
        super().__init__()
        self.atom_emb = nn.Embedding(max_atomic_num + 1, hidden)
        self.x_proj = nn.Linear(6, hidden)
        self.prop_proj = nn.Sequential(nn.Linear(prop_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, 4))

    def forward(self, *, z: torch.Tensor, x: torch.Tensor, batch: torch.Tensor, props: torch.Tensor) -> torch.Tensor:
        z = z.clamp_min(0).clamp_max(self.atom_emb.num_embeddings - 1)
        node_h = self.atom_emb(z) + self.x_proj(x)
        ctx = self.prop_proj(props)  # [B, H]
        logits = self.head(torch.cat([node_h, ctx[batch]], dim=-1))
        return logits


class SimpleGINOpModel(nn.Module):
    def __init__(
        self,
        *,
        prop_dim: int,
        hidden: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0,
        max_atomic_num: int = 100,
        num_bond_types: int = 5,
    ) -> None:
        super().__init__()
        require_torch_geometric()
        from torch_geometric.nn import GINEConv

        self.atom_emb = nn.Embedding(max_atomic_num + 1, hidden)
        self.bond_emb = nn.Embedding(num_bond_types, hidden)
        self.x_proj = nn.Linear(6, hidden)
        self.prop_proj = nn.Sequential(nn.Linear(prop_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.dropout = float(dropout)

        def gin_mlp() -> nn.Module:
            return nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))

        self.convs = nn.ModuleList([GINEConv(gin_mlp(), edge_dim=hidden) for _ in range(int(num_layers))])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(int(num_layers))])

        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, 4))

    def forward(
        self,
        *,
        z: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        batch: torch.Tensor,
        props: torch.Tensor,
    ) -> torch.Tensor:
        z = z.clamp_min(0).clamp_max(self.atom_emb.num_embeddings - 1)
        node_h = self.atom_emb(z) + self.x_proj(x)
        node_h = F.relu(node_h)
        edge_h = self.bond_emb(edge_type.clamp_min(0).clamp_max(self.bond_emb.num_embeddings - 1))
        for conv, norm in zip(self.convs, self.norms):
            node_h = conv(node_h, edge_index, edge_h)
            node_h = norm(node_h)
            node_h = F.relu(node_h)
            if self.dropout:
                node_h = F.dropout(node_h, p=self.dropout, training=self.training)

        ctx = self.prop_proj(props)
        return self.head(torch.cat([node_h, ctx[batch]], dim=-1))


def _stack(samples: list[Sample], device: torch.device) -> tuple[torch.Tensor, ...]:
    z_all = []
    x_all = []
    y_all = []
    allowed_all = []
    anchor_all = []
    batch_all = []
    props_all = []
    edge_index_all = []
    edge_type_all = []

    node_off = 0
    for i, s in enumerate(samples):
        n = int(s.z.size(0))
        z_all.append(s.z)
        x_all.append(s.x)
        y_all.append(s.y_op)
        allowed_all.append(s.op_allowed)
        anchor_all.append(s.anchor_mask)
        batch_all.append(torch.full((n,), i, dtype=torch.long))
        props_all.append(s.props.unsqueeze(0))
        if int(s.edge_index.numel()):
            edge_index_all.append(s.edge_index + node_off)
            edge_type_all.append(s.edge_type)
        node_off += n

    z = torch.cat(z_all, dim=0).to(device)
    x = torch.cat(x_all, dim=0).to(device)
    y = torch.cat(y_all, dim=0).to(device)
    op_allowed = torch.cat(allowed_all, dim=0).to(device)
    anchor_mask = torch.cat(anchor_all, dim=0).to(device)
    batch = torch.cat(batch_all, dim=0).to(device)
    props = torch.cat(props_all, dim=0).to(device)  # [B, P]
    if edge_index_all:
        edge_index = torch.cat(edge_index_all, dim=1).to(device)
        edge_type = torch.cat(edge_type_all, dim=0).to(device)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_type = torch.empty((0,), dtype=torch.long, device=device)
    return z, x, edge_index, edge_type, y, op_allowed, anchor_mask, batch, props


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Debug script: can a tiny non-GNN baseline overfit op prediction with property conditioning?"
    )
    parser.add_argument("--csv", type=str, required=True, help="Training CSV with start_smiles_tagged + edits_json + props.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Base config.yaml (to read property_names).")
    parser.add_argument("--max_rows_read", type=int, default=2000)
    parser.add_argument("--subset_size", type=int, default=32, help="Number of molecules to overfit on.")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--model", type=str, default="gin", choices=["mlp", "gin"])
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--loss_mode",
        type=str,
        default="anchor",
        choices=["anchor", "heavy"],
        help='Which nodes contribute to op loss: "anchor" (op_allowed-based) or "heavy" (z>1).',
    )
    parser.add_argument(
        "--only_pos",
        action="store_true",
        help='If set, only compute op loss on nodes with y_op!=none (good for checking "can it memorize positives").',
    )
    parser.add_argument(
        "--drop_invalid_gt",
        action="store_true",
        help="If set, drop nodes whose GT op is masked out by op_allowed (diagnoses label/mask mismatch).",
    )
    args = parser.parse_args()

    cfg = Config.load(args.config)
    property_names = list(cfg.raw["chemistry"]["property_names"])

    # Read header first to resolve prop cols deterministically.
    with Path(args.csv).open("r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
    prop_cols = _resolve_property_cols(header, property_names)

    rows = _load_rows(args.csv, max_rows=int(args.max_rows_read), seed=int(args.seed))
    samples: list[Sample] = []
    for r in rows:
        s = _build_sample(r, prop_cols=prop_cols)
        if s is None:
            continue
        samples.append(s)
        if len(samples) >= int(args.subset_size):
            break
    if not samples:
        raise RuntimeError("No usable samples. Check CSV columns: start_smiles_tagged, edits_json, and property columns.")

    device = torch.device(str(args.device))
    model = SimpleOpModel(prop_dim=len(prop_cols), hidden=int(args.hidden))
    if args.model == "gin":
        model = SimpleGINOpModel(
            prop_dim=len(prop_cols),
            hidden=int(args.hidden),
            num_layers=int(args.num_layers),
            dropout=float(args.dropout),
        )
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    rng = random.Random(int(args.seed))

    def iter_minibatch() -> list[Sample]:
        bs = min(int(args.batch_size), len(samples))
        idx = [rng.randrange(len(samples)) for _ in range(bs)]
        return [samples[i] for i in idx]

    # Quick dataset sanity stats on the chosen subset.
    with torch.no_grad():
        z0, x0, edge_index0, edge_type0, y0, op_allowed0, anchor0, batch0, props0 = _stack(samples, device)
        heavy0 = z0 > 1
        gt_allowed0 = op_allowed0.gather(1, y0.clamp_min(0).clamp_max(3).unsqueeze(-1)).squeeze(-1)
        invalid0 = int((heavy0 & (~gt_allowed0) & (y0 != OP_TO_ID["none"])).sum().item())
        print(
            f"loaded_samples={len(samples)} prop_dim={len(prop_cols)} "
            f"avg_nodes={z0.numel()/len(samples):.1f} "
            f"anchor_frac={anchor0.float().mean().item():.3f} "
            f"pos_frac@heavy={(y0[heavy0]!=OP_TO_ID['none']).float().mean().item():.3f} "
            f"invalid_pos_ops@heavy={invalid0}"
        )

    for step in range(1, int(args.steps) + 1):
        batch_samples = iter_minibatch()
        z, x, edge_index, edge_type, y, op_allowed, anchor_mask, batch, props = _stack(batch_samples, device)

        if args.loss_mode == "heavy":
            loss_mask = z > 1
        else:
            loss_mask = anchor_mask
        if bool(args.only_pos):
            loss_mask = loss_mask & (y != OP_TO_ID["none"])

        gt_allowed = op_allowed.gather(1, y.clamp_min(0).clamp_max(3).unsqueeze(-1)).squeeze(-1)
        invalid_gt = loss_mask & (~gt_allowed)
        if bool(args.drop_invalid_gt):
            loss_mask = loss_mask & gt_allowed

        if args.model == "gin":
            logits = model(z=z, x=x, edge_index=edge_index, edge_type=edge_type, batch=batch, props=props)
        else:
            logits = model(z=z, x=x, batch=batch, props=props)
        loss = F.cross_entropy(logits[loss_mask], y[loss_mask]) if bool(loss_mask.any().item()) else logits.sum() * 0.0

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 50 == 0 or step == 1:
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                acc = (pred[loss_mask] == y[loss_mask]).float().mean().item() if bool(loss_mask.any().item()) else 0.0
                pos = (y[loss_mask] != OP_TO_ID["none"]).float().mean().item() if bool(loss_mask.any().item()) else 0.0
                invalid_n = int(invalid_gt.sum().item())
                print(
                    f"step={step} loss={loss.item():.4f} acc={acc:.3f} pos_rate={pos:.3f} "
                    f"nodes={int(loss_mask.sum().item())} invalid_gt={invalid_n}"
                )


if __name__ == "__main__":
    main()
