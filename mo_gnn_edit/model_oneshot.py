from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset_oneshot import ID_TO_OP, OP_TO_ID
from .pyg_utils import require_torch_geometric


@dataclass(frozen=True)
class OneShotModelOutput:
    op_logits: torch.Tensor  # [num_nodes, 4]
    fg_logits: torch.Tensor  # [num_nodes, fg_vocab]
    node_h: torch.Tensor  # [num_nodes, hidden]
    graph_h: torch.Tensor  # [batch, hidden]
    moe_gate_probs: torch.Tensor | None = None  # [batch, num_experts]
    moe_topi: torch.Tensor | None = None  # [batch, topk]
    moe_topv: torch.Tensor | None = None  # [batch, topk] normalized to sum=1
    moe_entropy: torch.Tensor | None = None  # [batch]
    moe_topk_mass: torch.Tensor | None = None  # [batch] sum of unnormalized topk probs


class SinusoidalProjection(nn.Module):
    """Project scalar |delta| to a sinusoidal embedding."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)
        half = max(1, self.dim // 2)
        inv_freq = torch.exp(-math.log(10000.0) * torch.arange(0, half, dtype=torch.float32) / float(max(half - 1, 1)))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [..., 1] positive scalar magnitude.
        returns: [..., dim]
        """
        if x.dim() < 1:
            raise ValueError("SinusoidalProjection expects input with at least 1 dimension")
        # [..., 1] -> [..., half]
        arg = x * self.inv_freq.to(device=x.device, dtype=x.dtype)
        sin = torch.sin(arg)
        cos = torch.cos(arg)
        out = torch.cat([sin, cos], dim=-1)
        if out.size(-1) < self.dim:
            pad = torch.zeros(*out.shape[:-1], self.dim - out.size(-1), device=out.device, dtype=out.dtype)
            out = torch.cat([out, pad], dim=-1)
        elif out.size(-1) > self.dim:
            out = out[..., : self.dim]
        return out


class ConditionEncoder(nn.Module):
    """
    Minimal conditioning block:
      - type embedding distinguishes which property to edit
      - sign embedding encodes increase / keep / decrease
      - magnitude MLP projects |delta| to hidden space
      - value projection projects |src| (optional) to hidden space
      - value projection projects |target| to hidden space
      - fusion MLP combines signals per-property
    """

    def __init__(self, num_props: int, d_model: int, *, sign_eps: float = 1e-4) -> None:
        super().__init__()
        self.num_props = int(num_props)
        self.d_model = int(d_model)
        self.sign_eps = float(sign_eps)
        self.d_type = 16
        self.d_sign = 8
        self.d_cont = 32
        self.type_emb = nn.Embedding(max(1, self.num_props), self.d_type)
        self.sign_emb = nn.Embedding(3, self.d_sign)  # 0=negative, 1=zero, 2=positive
        self.mag_proj = SinusoidalProjection(self.d_cont)
        self.val_proj = SinusoidalProjection(self.d_cont)
        self.tgt_proj = SinusoidalProjection(self.d_cont)
        self.fusion = nn.Sequential(
            nn.Linear(self.d_type + 3 * self.d_sign + 3 * self.d_cont, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        delta_vals: torch.Tensor,
        src_vals: torch.Tensor | None = None,
        prop_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        delta_vals: [batch, num_props] raw property deltas.
        src_vals: optional [batch, num_props] raw property values on the source molecule.
        prop_ids: optional [batch, num_props] or [num_props]; defaults to 0..num_props-1.
        Returns:
          - prop_tokens: [batch, num_props, d_model]
        """
        if delta_vals.dim() != 2:
            raise ValueError("delta_vals must have shape [batch, num_props]")
        b, p = int(delta_vals.size(0)), int(delta_vals.size(1))
        if p != int(self.num_props):
            raise ValueError(f"Expected num_props={self.num_props}, got {p}")

        if prop_ids is None:
            prop_ids = torch.arange(self.num_props, device=delta_vals.device, dtype=torch.long).unsqueeze(0).expand(b, -1)
        elif prop_ids.dim() == 1:
            prop_ids = prop_ids.unsqueeze(0).expand(b, -1)
        elif prop_ids.dim() != 2:
            raise ValueError("prop_ids must have shape [num_props] or [batch, num_props]")

        if int(prop_ids.size(1)) != p:
            raise ValueError("prop_ids second dimension must match num_props")

        e_type = self.type_emb(prop_ids)  # [B, P, d_type]

        sign_raw = torch.sign(delta_vals)
        sign_raw = torch.where(delta_vals.abs() <= self.sign_eps, torch.zeros_like(sign_raw), sign_raw)
        signs = sign_raw.long() + 1  # -1->0, 0->1, +1->2
        signs = signs.clamp(min=0, max=2)
        e_sign = self.sign_emb(signs)  # [B, P, d_sign]

        mags = torch.abs(delta_vals).unsqueeze(-1)  # [B, P, 1]
        e_mag = self.mag_proj(mags)  # [B, P, d_cont]

        if src_vals is None:
            src_vals = torch.zeros_like(delta_vals)
        if src_vals.dim() != 2 or int(src_vals.size(0)) != b or int(src_vals.size(1)) != p:
            raise ValueError("src_vals must have shape [batch, num_props] when provided")
        src_mags = torch.abs(src_vals).unsqueeze(-1)  # [B, P, 1]
        e_src = self.val_proj(src_mags)  # [B, P, d_cont]

        tgt_vals = src_vals + delta_vals
        tgt_mags = torch.abs(tgt_vals).unsqueeze(-1)  # [B, P, 1]
        e_tgt = self.tgt_proj(tgt_mags)  # [B, P, d_cont]

        src_sign_raw = torch.sign(src_vals)
        src_sign_raw = torch.where(src_vals.abs() <= self.sign_eps, torch.zeros_like(src_sign_raw), src_sign_raw)
        src_signs = src_sign_raw.long() + 1
        src_signs = src_signs.clamp(min=0, max=2)
        e_src_sign = self.sign_emb(src_signs)  # [B, P, d_sign]

        tgt_sign_raw = torch.sign(tgt_vals)
        tgt_sign_raw = torch.where(tgt_vals.abs() <= self.sign_eps, torch.zeros_like(tgt_sign_raw), tgt_sign_raw)
        tgt_signs = tgt_sign_raw.long() + 1
        tgt_signs = tgt_signs.clamp(min=0, max=2)
        e_tgt_sign = self.sign_emb(tgt_signs)  # [B, P, d_sign]

        combined = torch.cat([e_type, e_sign, e_mag, e_src, e_src_sign, e_tgt, e_tgt_sign], dim=-1)
        fused = self.fusion(combined)  # [B, P, H]
        return fused


class OneShotTwoStageEditModel(nn.Module):
    """
    One forward pass predicts a *set* of edits:
      - per-node `op` in {none, add, remove, replace}
      - per-node `fg_id` for add/replace (2nd-stage head)
      - per-anchor removed-atom choice for remove/replace (2nd-stage pointer head)
    """

    def __init__(
        self,
        *,
        props_dim: int,
        fg_vocab_size: int,
        fg_num_classes: int = 2,
        hidden_dim: int = 256,
        num_layers: int = 5,
        gine_layers: int | None = None,
        mlp_layers: int | None = None,
        num_tasks: int = 0,
        task_emb_dim: int = 32,
        dropout: float = 0.1,
        backbone: str = "gine",
        max_atomic_num: int = 100,
        num_bond_types: int = 5,
        use_moe: bool = True,
        num_experts: int = 8,
        moe_topk: int = 1,
        moe_gate_temperature: float = 1.0,
        moe_gate_noise: float = 0.0,
        prop_attn_heads: int = 4,
        node_self_attn_layers: int = 1,
        prop_self_attn_layers: int = 1,
        cross_attn_ffn_layers: int = 1,
        cross_attn_ffn_mult: int = 4,
        cross_attn_ffn_dropout: float | None = None,
    ) -> None:
        super().__init__()
        require_torch_geometric()
        from torch_geometric.nn import GINEConv

        self.props_dim = int(props_dim)
        self.num_props = int(props_dim)
        self.fg_vocab_size = int(fg_vocab_size)
        self.fg_num_classes = int(fg_num_classes)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.backbone = str(backbone)
        self.use_moe = bool(use_moe)
        self.num_experts = max(0, int(num_experts))
        self.moe_topk = max(1, int(moe_topk))
        self.moe_gate_temperature = float(moe_gate_temperature)
        self.moe_gate_noise = float(moe_gate_noise)
        self.num_tasks = max(0, int(num_tasks))
        self.task_emb_dim = max(1, int(task_emb_dim))
        self.condition_encoder = ConditionEncoder(self.num_props, self.hidden_dim) if self.num_props > 0 else None
        self.prop_q = None
        self.prop_k = None
        self.prop_v = None
        if self.num_tasks > 0:
            self.task_emb = nn.Embedding(self.num_tasks + 1, self.task_emb_dim)
        else:
            self.task_emb = None

        self.atom_emb = nn.Embedding(max_atomic_num + 1, hidden_dim)
        self.bond_emb = nn.Embedding(num_bond_types, hidden_dim)
        self.atom_scalar = nn.LazyLinear(hidden_dim)
        self.atom_in = nn.Linear(hidden_dim * 2, hidden_dim)
        cond_in_dim = hidden_dim if self.num_props <= 0 else hidden_dim * self.num_props
        attn_heads = int(prop_attn_heads)
        if attn_heads < 1:
            raise ValueError("prop_attn_heads must be >= 1")
        if hidden_dim % attn_heads != 0:
            raise ValueError(f"prop_attn_heads={attn_heads} must divide hidden_dim={hidden_dim}.")
        # FiLM conditioning: (gamma, beta) from flattened prop tokens.
        self.cond_to_film = nn.Sequential(
            nn.Linear(cond_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )
        self.node_self_attn_layers = max(0, int(node_self_attn_layers))
        self.prop_self_attn_layers = max(0, int(prop_self_attn_layers))
        self.node_self_attn = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=attn_heads,
                    dropout=self.dropout,
                    batch_first=True,
                )
                for _ in range(self.node_self_attn_layers)
            ]
        )
        self.node_self_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(self.node_self_attn_layers)])
        self.prop_self_attn = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=attn_heads,
                    dropout=self.dropout,
                    batch_first=True,
                )
                for _ in range(self.prop_self_attn_layers)
            ]
        )
        self.prop_self_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(self.prop_self_attn_layers)])
        self.prop_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=attn_heads,
            dropout=self.dropout,
            batch_first=True,
        )
        self.prop_norm = nn.LayerNorm(hidden_dim)
        self.cross_ffn_layers = max(0, int(cross_attn_ffn_layers))
        self.cross_ffn_mult = max(1, int(cross_attn_ffn_mult))
        self.cross_ffn_dropout = (
            float(self.dropout) if cross_attn_ffn_dropout is None else float(cross_attn_ffn_dropout)
        )
        if self.cross_ffn_layers > 0:
            self.cross_ffn = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim * self.cross_ffn_mult),
                        nn.ReLU(),
                        nn.Dropout(self.cross_ffn_dropout),
                        nn.Linear(hidden_dim * self.cross_ffn_mult, hidden_dim),
                    )
                    for _ in range(self.cross_ffn_layers)
                ]
            )
            self.cross_ffn_norms = nn.ModuleList(
                [nn.LayerNorm(hidden_dim) for _ in range(self.cross_ffn_layers)]
            )
        else:
            self.cross_ffn = None
            self.cross_ffn_norms = None
        fused_dim = hidden_dim * 2
        self.node_prop_fuse = nn.Linear(fused_dim, hidden_dim)

        def gin_mlp() -> nn.Module:
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        if self.backbone == "gine":
            depth = int(gine_layers) if gine_layers is not None else int(num_layers)
            self.convs = nn.ModuleList([GINEConv(gin_mlp(), edge_dim=hidden_dim) for _ in range(depth)])
            self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(depth)])
            self.mlp_layers = None
            self.node_blocks = None
            self.edge_bias = None
            self.degree_emb = None
            self.cond_to_attn_bias = None
        elif self.backbone == "mlp":
            self.convs = None
            self.norms = None
            self.mlp_layers = None
            depth = int(mlp_layers) if mlp_layers is not None else int(num_layers)
            self.mlp_linears = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(depth)])
            self.mlp_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(depth)])
            self.node_blocks = None
            self.edge_bias = None
            self.degree_emb = None
            self.cond_to_attn_bias = None
        else:
            raise ValueError(f"Unknown backbone={self.backbone!r}; expected 'gine' or 'mlp'")

        self.op_head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )

        self.fg_head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.fg_vocab_size),
        )
        if self.use_moe and self.num_experts > 0:
            # Gate on conditioning + graph embedding:
            #   - if P==0: graph_h [B, H]
            #   - else: concat([props_flat, graph_h]) [B, P*H + H]
            task_dim = self.task_emb_dim if self.task_emb is not None else 0
            moe_gate_in_dim = hidden_dim if self.num_props <= 0 else hidden_dim * self.num_props
            self.moe_gate = nn.Linear(moe_gate_in_dim, self.num_experts)
            self.op_experts = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(fused_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, 4),
                    )
                    for _ in range(self.num_experts)
                ]
            )
            self.fg_experts = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(fused_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, self.fg_vocab_size),
                    )
                    for _ in range(self.num_experts)
                ]
            )
        else:
            self.moe_gate = None
            self.op_experts = None
            self.fg_experts = None

        self.remove_query = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def set_fg_hierarchy(self, *, fg_id_to_class: torch.Tensor) -> None:
        # Backward compatibility no-op (hierarchical FG classes removed).
        return

    def _props_flat(self, props_tokens: torch.Tensor, *, batch_size: int) -> torch.Tensor:
        # Flatten per-property tokens.
        if self.num_props <= 0:
            return props_tokens.new_zeros((int(batch_size), 0))
        if int(props_tokens.numel()) != 0:
            _, p, h = props_tokens.shape
            return props_tokens.reshape(int(batch_size), int(p) * int(h))
        # If props are expected but missing, gate on a zero vector (keeps shapes consistent).
        return props_tokens.new_zeros((int(batch_size), int(self.hidden_dim) * int(self.num_props)))

    def _task_embedding(self, data, *, batch_size: int) -> torch.Tensor | None:
        if self.task_emb is None or self.num_tasks <= 0:
            return None
        if not hasattr(data, "task_id"):
            unk = torch.full(
                (int(batch_size),),
                self.num_tasks,
                device=next(self.task_emb.parameters()).device,
                dtype=torch.long,
            )
            return self.task_emb(unk)
        task_id = data.task_id
        if task_id.dim() == 0:
            task_id = task_id.view(1)
        if int(task_id.numel()) != int(batch_size):
            if hasattr(data, "batch") and int(task_id.numel()) == int(data.batch.numel()):
                task_ids = []
                num_graphs = int(batch_size)
                for g in range(num_graphs):
                    idx = (data.batch == g).nonzero(as_tuple=True)[0]
                    if int(idx.numel()) == 0:
                        task_ids.append(torch.tensor(self.num_tasks, device=task_id.device, dtype=task_id.dtype))
                    else:
                        task_ids.append(task_id[idx[0]])
                task_id = torch.stack(task_ids, dim=0)
            else:
                task_id = task_id.view(-1)[:batch_size]
        task_id = task_id.to(dtype=torch.long)
        unk = torch.full_like(task_id, self.num_tasks)
        task_id = torch.where((task_id < 0) | (task_id >= self.num_tasks), unk, task_id)
        return self.task_emb(task_id)

    def _moe_gate_input(self, props_tokens: torch.Tensor, graph_h: torch.Tensor, data) -> torch.Tensor:
        """
        Returns MoE gate input:
        - if P==0: zeros [B, H]
        - else: props_flat [B, P*H]
        """
        batch_size = int(graph_h.size(0))
        if self.num_props <= 0:
            return graph_h.new_zeros((int(batch_size), int(self.hidden_dim)))
        return self._props_flat(props_tokens, batch_size=batch_size)

    def _film_params(self, gate_input: torch.Tensor, *, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        gate_input: [B, H*P] (or [B, H] if P==0)
        Returns: (gamma, beta) both [B, H]
        """
        if batch_size == 0:
            return gate_input.new_zeros((0, self.hidden_dim)), gate_input.new_zeros((0, self.hidden_dim))
        if int(gate_input.numel()) == 0:
            return gate_input.new_zeros((batch_size, self.hidden_dim)), gate_input.new_zeros((batch_size, self.hidden_dim))
        gb = self.cond_to_film(gate_input)  # [B, 2H]
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = torch.tanh(gamma)  # keep scale stable; applied as (1+gamma)
        beta = 0.1 * beta
        return gamma, beta

    def _moe_head(
        self,
        fused: torch.Tensor,
        batch: torch.Tensor,
        gate_input: torch.Tensor,
        experts: nn.ModuleList,
        out_dim: int,
    ) -> torch.Tensor:
        logits = fused.new_zeros((int(fused.size(0)), int(out_dim)))
        if experts is None or self.moe_gate is None or self.num_experts == 0 or int(fused.numel()) == 0:
            return logits
        num_graphs = int(batch.max().item()) + 1 if int(batch.numel()) else 0
        _, topi, topv, _, _ = self._moe_route(gate_input)
        return self._moe_apply(fused, batch, experts, out_dim=out_dim, num_graphs=num_graphs, topi=topi, topv=topv)

    def _moe_route(
        self, gate_input: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          gate_probs: [G, E]
          topi: [G, K]
          topv: [G, K] normalized to sum=1 (used as mixture weights)
          entropy: [G]
          topk_mass: [G] sum of unnormalized topk probs (before renorm)
        """
        if self.moe_gate is None or self.num_experts <= 0:
            z = gate_input.new_zeros((int(gate_input.size(0)), 0))
            return z, z.to(dtype=torch.long), z, z.sum(dim=-1), z.sum(dim=-1)

        gate_logits = self.moe_gate(gate_input)  # [G, E]
        if self.training and float(self.moe_gate_noise) > 0:
            gate_logits = gate_logits + float(self.moe_gate_noise) * torch.randn_like(gate_logits)
        temp = max(1e-6, float(self.moe_gate_temperature))
        gate_probs = torch.softmax(gate_logits / temp, dim=-1)
        topk = min(int(self.moe_topk), int(self.num_experts))
        topv_raw, topi = torch.topk(gate_probs, k=topk, dim=-1)
        topk_mass = topv_raw.sum(dim=-1)
        topv = topv_raw / topk_mass.clamp_min(1e-12).unsqueeze(-1)
        eps = 1e-12
        entropy = -(gate_probs.clamp_min(eps) * gate_probs.clamp_min(eps).log()).sum(dim=-1)
        return gate_probs, topi, topv, entropy, topk_mass

    def _moe_apply(
        self,
        fused: torch.Tensor,
        batch: torch.Tensor,
        experts: nn.ModuleList,
        *,
        out_dim: int,
        num_graphs: int,
        topi: torch.Tensor,
        topv: torch.Tensor,
    ) -> torch.Tensor:
        logits = fused.new_zeros((int(fused.size(0)), int(out_dim)))
        topk = int(topi.size(1)) if int(topi.numel()) else 0
        if topk <= 0 or experts is None:
            return logits
        for g in range(int(num_graphs)):
            idx = (batch == g).nonzero(as_tuple=True)[0]
            if int(idx.numel()) == 0:
                continue
            if topk == 1:
                expert = experts[int(topi[g, 0])]
                logits[idx] = expert(fused[idx])
            else:
                acc = 0.0
                for k in range(topk):
                    expert = experts[int(topi[g, k])]
                    acc = acc + topv[g, k] * expert(fused[idx])
                logits[idx] = acc
        return logits

    def _encode_nodes(
        self,
        x: torch.Tensor,
        data,
        *,
        film_gamma: torch.Tensor,
        film_beta: torch.Tensor,
    ) -> torch.Tensor:
        if self.backbone == "gine":
            if self.convs is None or self.norms is None:
                raise RuntimeError("GINE backbone modules are not initialized.")
            edge_h = self.bond_emb(data.edge_type)
            for conv, norm in zip(self.convs, self.norms):
                x = x + conv(x, data.edge_index, edge_h)
                x = norm(x)
                if int(film_gamma.numel()) != 0:
                    g = film_gamma[data.batch]
                    b = film_beta[data.batch]
                    x = x * (1.0 + g) + b
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            return x

        if self.backbone == "mlp":
            if self.mlp_linears is None or self.mlp_norms is None:
                raise RuntimeError("MLP backbone modules are not initialized.")
            for lin, norm in zip(self.mlp_linears, self.mlp_norms):
                x = lin(x)
                x = norm(x)
                if int(film_gamma.numel()) != 0:
                    g = film_gamma[data.batch]
                    b = film_beta[data.batch]
                    x = x * (1.0 + g) + b
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            return x

        raise RuntimeError(f"Unexpected backbone={self.backbone!r}")

    def _node_self_attend(self, node_h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if not self.node_self_attn:
            return node_h
        out = node_h
        num_graphs = int(batch.max().item()) + 1 if int(batch.numel()) else 0
        for attn, norm in zip(self.node_self_attn, self.node_self_norms):
            layer_out = out.new_zeros(out.size())
            for g in range(num_graphs):
                idx = (batch == g).nonzero(as_tuple=True)[0]
                if int(idx.numel()) == 0:
                    continue
                q = out[idx].unsqueeze(0)
                ctx, _ = attn(q, q, q, need_weights=False)
                layer_out[idx] = norm(q + ctx).squeeze(0)
            out = layer_out
        return out

    def _prop_self_attend(self, props_tokens: torch.Tensor) -> torch.Tensor:
        if int(props_tokens.size(1)) == 0 or not self.prop_self_attn:
            return props_tokens
        out = props_tokens
        for attn, norm in zip(self.prop_self_attn, self.prop_self_norms):
            ctx, _ = attn(out, out, out, need_weights=False)
            out = norm(out + ctx)
        return out

    def _prop_context(self, node_h: torch.Tensor, props_tokens: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if int(props_tokens.size(1)) == 0:
            return node_h.new_zeros(node_h.size())
        out = node_h.new_zeros(node_h.size())
        num_graphs = int(batch.max().item()) + 1 if int(batch.numel()) else 0
        for g in range(num_graphs):
            idx = (batch == g).nonzero(as_tuple=True)[0]
            if int(idx.numel()) == 0:
                continue
            tokens = props_tokens[g : g + 1]
            if int(tokens.size(1)) == 0:
                continue
            q = node_h[idx].unsqueeze(0)
            attn_out, _ = self.prop_attn(q, tokens, tokens, need_weights=False)
            y = self.prop_norm(q + attn_out).squeeze(0)
            if self.cross_ffn is not None and self.cross_ffn_norms is not None:
                for ffn, norm in zip(self.cross_ffn, self.cross_ffn_norms):
                    y = norm(y + ffn(y))
            out[idx] = y
        return out

    def _fuse_node_inputs(
        self, node_h: torch.Tensor, props_tokens: torch.Tensor, data
    ) -> tuple[torch.Tensor, torch.Tensor]:
        c_i = self._prop_context(node_h, props_tokens, data.batch)
        fused = torch.cat([node_h, c_i], dim=-1)
        node_ctx = self.node_prop_fuse(fused)
        return fused, node_ctx

    def encode(self, data):
        require_torch_geometric()
        from torch_geometric.nn import global_mean_pool

        z = data.z.clamp_min(0).clamp_max(self.atom_emb.num_embeddings - 1)
        atom_h = self.atom_emb(z)
        scalar_h = self.atom_scalar(data.x)
        x = self.atom_in(torch.cat([atom_h, scalar_h], dim=-1))
        x = F.relu(x)

        # Graph-level condition embedding.
        batch_size = int(data.batch.max().item()) + 1 if int(data.batch.numel()) else 0
        if self.condition_encoder is None or not hasattr(data, "props") or self.num_props <= 0:
            props_tokens = torch.zeros((batch_size, 0, self.hidden_dim), device=x.device, dtype=x.dtype)
        else:
            props = data.props
            if props.dim() != 2:
                raise ValueError("data.props must have shape [batch, num_props]")
            if int(props.size(1)) != self.num_props:
                raise ValueError(f"Unexpected props dim={int(props.size(1))}; expected {self.num_props}")
            if hasattr(data, "props_src"):
                props_src = data.props_src
                if props_src.dim() != 2:
                    raise ValueError("data.props_src must have shape [batch, num_props]")
                if int(props_src.size(1)) != self.num_props:
                    raise ValueError(f"Unexpected props_src dim={int(props_src.size(1))}; expected {self.num_props}")
            else:
                props_src = torch.zeros_like(props)
            props_tokens = self.condition_encoder(props, props_src)

        if self.num_props <= 0:
            gate_input = x.new_zeros((batch_size, self.hidden_dim))
        else:
            gate_input = self._props_flat(props_tokens, batch_size=batch_size)
        film_gamma, film_beta = self._film_params(gate_input, batch_size=batch_size)

        # Node backbone encoding.
        x = self._encode_nodes(x, data, film_gamma=film_gamma, film_beta=film_beta)
        x = self._node_self_attend(x, data.batch)
        props_tokens = self._prop_self_attend(props_tokens)
        graph_h = global_mean_pool(x, data.batch)
        return x, graph_h, props_tokens

    def forward(self, data, *, teacher_forcing: bool = True) -> OneShotModelOutput:
        node_h, graph_h, props_tokens = self.encode(data)
        fused, node_ctx = self._fuse_node_inputs(node_h, props_tokens, data)
        moe_gate_input = self._moe_gate_input(props_tokens, graph_h, data)

        moe_gate_probs = None
        moe_topi = None
        moe_topv = None
        moe_entropy = None
        moe_topk_mass = None
        if self.use_moe and self.moe_gate is not None and self.num_experts > 0 and int(graph_h.numel()) != 0:
            moe_gate_probs, moe_topi, moe_topv, moe_entropy, moe_topk_mass = self._moe_route(moe_gate_input)
            num_graphs = int(graph_h.size(0))
        else:
            num_graphs = int(graph_h.size(0)) if int(graph_h.numel()) else 0

        # ---------- op prediction ----------
        if self.use_moe and self.op_experts is not None and moe_topi is not None and moe_topv is not None:
            op_logits = self._moe_apply(fused, data.batch, self.op_experts, out_dim=4, num_graphs=num_graphs, topi=moe_topi, topv=moe_topv)
        elif self.use_moe and self.op_experts is not None:
            op_logits = self._moe_head(fused, data.batch, moe_gate_input, self.op_experts, out_dim=4)
        else:
            op_logits = self.op_head(fused)
        pred_op_id = op_logits.argmax(dim=-1)  # [num_nodes]

        # ---------- fg prediction ----------
        # Compute FG logits for all nodes so training can backprop regardless of op correctness.
        if self.use_moe and self.fg_experts is not None and moe_topi is not None and moe_topv is not None:
            fg_logits = self._moe_apply(
                fused,
                data.batch,
                self.fg_experts,
                out_dim=self.fg_vocab_size,
                num_graphs=num_graphs,
                topi=moe_topi,
                topv=moe_topv,
            )
        elif self.use_moe and self.fg_experts is not None:
            fg_logits = self._moe_head(fused, data.batch, moe_gate_input, self.fg_experts, out_dim=self.fg_vocab_size)
        else:
            fg_logits = self.fg_head(fused)

        return OneShotModelOutput(
            op_logits=op_logits,
            fg_logits=fg_logits,
            node_h=node_ctx,
            graph_h=graph_h,
            moe_gate_probs=moe_gate_probs,
            moe_topi=moe_topi,
            moe_topv=moe_topv,
            moe_entropy=moe_entropy,
            moe_topk_mass=moe_topk_mass,
        )

    @torch.no_grad()
    def sample_edit_sets(
        self,
        data0,
        *,
        fg_vocab: Any,
        k_samples: int,
        max_edits: int,
        temperature_op: float,
        temperature_fg: float,
        temperature_remove: float,
        seed: int,
        forced_k_target: int | None = None,
        fg_allowed_ids: list[int] | dict[str, list[int]] | None = None,
    ) -> list[list[dict[str, Any]]]:
        """
        Sample K unordered edit sets for a single graph Data object.
        Each set is a list of Edit-like dicts compatible with `apply_edits`.

        This method intentionally lives with the model so inference code does not depend
        on internal head wiring (encode/op_head/fg_head shapes).
        """
        require_torch_geometric()
        from torch_geometric.data import Batch

        device = next(self.parameters()).device
        try:
            g = torch.Generator(device=device)
        except Exception:
            g = torch.Generator()
        g.manual_seed(int(seed))

        self.eval()
        batch = Batch.from_data_list([data0]).to(device)

        node_h, graph_h, props_tokens = self.encode(batch)
        moe_gate_input = self._moe_gate_input(props_tokens, graph_h, batch)
        fused, node_ctx = self._fuse_node_inputs(node_h, props_tokens, batch)
        if self.use_moe and self.moe_gate is not None and self.num_experts > 0 and int(graph_h.numel()) != 0:
            gate_probs, topi, topv, entropy, topk_mass = self._moe_route(moe_gate_input)
            # Expose MoE routing diagnostics to inference without changing return types.
            # These fields are best-effort and may be absent for non-MoE models.
            try:
                self._last_moe_topi = topi.detach().cpu().tolist()  # type: ignore[attr-defined]
                self._last_moe_topv = topv.detach().cpu().tolist()  # type: ignore[attr-defined]
                self._last_moe_entropy = entropy.detach().cpu().tolist()  # type: ignore[attr-defined]
                self._last_moe_topk_mass = topk_mass.detach().cpu().tolist()  # type: ignore[attr-defined]
                self._last_moe_gate_probs = gate_probs.detach().cpu().tolist()  # type: ignore[attr-defined]
            except Exception:
                pass
            op_logits = self._moe_apply(
                fused,
                batch.batch,
                self.op_experts,
                out_dim=4,
                num_graphs=int(graph_h.size(0)),
                topi=topi,
                topv=topv,
            )
            full_fg_logits = self._moe_apply(
                fused,
                batch.batch,
                self.fg_experts,
                out_dim=self.fg_vocab_size,
                num_graphs=int(graph_h.size(0)),
                topi=topi,
                topv=topv,
            )
        else:
            op_logits = self.op_head(fused)
            full_fg_logits = self.fg_head(fused)
            # Clear MoE diagnostics for non-MoE models.
            try:
                self._last_moe_topi = None  # type: ignore[attr-defined]
                self._last_moe_topv = None  # type: ignore[attr-defined]
                self._last_moe_entropy = None  # type: ignore[attr-defined]
                self._last_moe_topk_mass = None  # type: ignore[attr-defined]
                self._last_moe_gate_probs = None  # type: ignore[attr-defined]
            except Exception:
                pass
        if hasattr(batch, "op_allowed"):
            op_allowed = batch.op_allowed
        else:
            op_allowed = torch.ones((int(batch.z.size(0)), 4), device=batch.z.device, dtype=torch.bool)
        op_logits = op_logits.masked_fill(~op_allowed, -1e9)

        fg_id_to_smiles = getattr(fg_vocab, "id_to_smiles", {}) or {}

        results: list[list[dict[str, Any]]] = []
        node0 = (batch.batch == 0).nonzero(as_tuple=True)[0]
        n_nodes = int(batch.z.size(0))
        deg = torch.bincount(batch.edge_index[0].to(torch.long), minlength=n_nodes) if hasattr(batch, "edge_index") else None

        def _terminal_neighbor_candidates(anchor_idx: int) -> torch.Tensor:
            if not hasattr(batch, "edge_index") or deg is None:
                return torch.empty((0,), device=device, dtype=torch.long)
            src = batch.edge_index[0]
            dst = batch.edge_index[1]
            nbrs = dst[src == int(anchor_idx)]
            if int(nbrs.numel()) == 0:
                return nbrs
            # Prefer non-scaffold neighbors when scaffold_mask is available (matches training/inference
            # remove_allowed definition and apply_edits(remove_non_scaffold_branch) expectations).
            scaffold = getattr(batch, "scaffold_mask", None)
            if scaffold is not None:
                try:
                    if not bool(scaffold[int(anchor_idx)].item()):
                        return nbrs.new_empty((0,))
                    non_scaf = nbrs[~scaffold[nbrs]]
                    if int(non_scaf.numel()) > 0:
                        return non_scaf
                except Exception:
                    pass
            term = nbrs[deg[nbrs] <= 1]
            return term if int(term.numel()) else nbrs

        for _ in range(int(k_samples)):
            op_probs = F.softmax(op_logits / max(1e-6, float(temperature_op)), dim=-1)  # [N, 4]
            op_id = torch.full((int(op_probs.size(0)),), OP_TO_ID["none"], device=device, dtype=torch.long)

            # Optional: if caller forces edit-count, select K anchors then force exactly K non-none ops.
            if forced_k_target is not None:
                k_target = max(0, min(int(forced_k_target), int(max_edits)))
                if k_target > 0 and int(node0.numel()) > 0:
                    edit_score = (1.0 - op_probs[:, OP_TO_ID["none"]]).detach()
                    can_edit = op_allowed[:, 1:].any(dim=-1)
                    cand = node0[can_edit[node0]]
                    if int(cand.numel()) > 0:
                        k_pick = min(int(k_target), int(cand.numel()))
                        top = torch.topk(edit_score[cand], k=k_pick, largest=True).indices
                        anchors = cand[top]
                        probs_non_none = op_probs[anchors, 1:]
                        probs_non_none = probs_non_none / probs_non_none.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                        sampled = torch.multinomial(probs_non_none, num_samples=1, generator=g).squeeze(-1) + 1
                        op_id[anchors] = sampled
            else:
                # Fallback: sample ops independently per node, then truncate to max_edits.
                sampled = torch.multinomial(op_probs, num_samples=1, generator=g).squeeze(-1)  # [N]
                op_id = sampled
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
            fg_logits = torch.zeros(
                (int(node_h.size(0)), int(self.fg_vocab_size)),
                device=node_h.device,
                dtype=op_logits.dtype,
            )
            if bool(fg_mask.any().item()):
                fg_logits[fg_mask] = full_fg_logits[fg_mask]
            fg_probs = F.softmax(fg_logits / max(1e-6, float(temperature_fg)), dim=-1)
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
                mask = torch.zeros((int(self.fg_vocab_size),), device=device, dtype=torch.bool)
                for fid in ids:
                    if 0 <= int(fid) < int(self.fg_vocab_size):
                        mask[int(fid)] = True
                return mask

            allowed_mask_add = _build_allowed_mask(_fg_ids_for_op("add"))
            allowed_mask_replace = _build_allowed_mask(_fg_ids_for_op("replace"))

            edits: list[dict[str, Any]] = []
            for gi in node0.tolist():
                oid = int(op_id[gi].item())
                if oid == OP_TO_ID["none"]:
                    continue
                op = ID_TO_OP[oid]
                anchor_map = int(batch.atom_map[gi].item())

                e: dict[str, Any] = {"anchor_atom_map": anchor_map, "op": op}
                if op in ("add", "replace"):
                    probs = fg_probs[gi]
                    allowed_mask = allowed_mask_add if op == "add" else allowed_mask_replace
                    if allowed_mask is not None and bool(allowed_mask.any().item()):
                        masked = probs * allowed_mask.to(dtype=probs.dtype)
                        if float(masked.sum().item()) > 0.0:
                            probs = masked
                    probs = probs / probs.sum().clamp_min(1e-12)
                    fg_id = int(torch.multinomial(probs, num_samples=1, generator=g).item())
                    fg_smiles = fg_id_to_smiles.get(fg_id)
                    if fg_smiles is None:
                        fg_id = int(fg_probs[gi].argmax().item())
                        fg_smiles = fg_id_to_smiles.get(fg_id)
                    e["fg_smiles"] = str(fg_smiles) if fg_smiles is not None else ""
                    e["fg_id"] = int(fg_id)
                if op in ("remove", "replace"):
                    cand = _terminal_neighbor_candidates(int(gi))
                    if int(cand.numel()) > 0:
                        q = self.remove_query(torch.cat([node_ctx[gi], graph_h[0]], dim=-1))
                        scores = (node_ctx[cand] * q.unsqueeze(0)).sum(dim=-1)  # [C]
                        probs = F.softmax(scores / max(1e-6, float(temperature_remove)), dim=-1)
                        pick = int(torch.multinomial(probs, num_samples=1, generator=g).item())
                        removed_map = int(batch.atom_map[int(cand[pick].item())].item())
                        e["removed_atom_map"] = int(removed_map)
                edits.append(e)

            def _op_rank(op_name: str) -> int:
                return {"remove": 0, "replace": 1, "add": 2}.get(op_name, 99)

            edits = sorted(edits, key=lambda e: (_op_rank(str(e["op"])), int(e["anchor_atom_map"])))
            # Conflict resolution: if a remove/replace explicitly removes an atom-map, drop any other
            # edits anchored on that atom (it won't exist when we apply edits sequentially).
            removed_maps: set[int] = set()
            for e in edits:
                if str(e.get("op")) not in ("remove", "replace"):
                    continue
                rm = e.get("removed_atom_map")
                if rm is None:
                    continue
                try:
                    rmi = int(rm)
                except Exception:
                    continue
                if rmi > 0:
                    removed_maps.add(rmi)
            if removed_maps:
                edits = [e for e in edits if int(e.get("anchor_atom_map", -1)) not in removed_maps]
            for si, e in enumerate(edits, start=1):
                e["step"] = si
            results.append(edits)

        return results


    @torch.no_grad()
    def sample_edit_sets_batch(
        self,
        data_list,
        *,
        fg_vocab: Any,
        k_samples: int,
        max_edits: int,
        temperature_op: float,
        temperature_fg: float,
        temperature_remove: float,
        seeds: list[int],
        forced_k_target: int | None = None,
        fg_allowed_ids: list[int] | dict[str, list[int]] | None = None,
    ) -> list[list[list[dict[str, Any]]]]:
        """
        Batch version of sample_edit_sets: returns per-graph edit sets.
        Output shape: [num_graphs][k_samples][edit_dict].
        """
        require_torch_geometric()
        from torch_geometric.data import Batch

        if not data_list:
            return []
        if len(seeds) != len(data_list):
            raise ValueError("seeds length must match data_list length")

        device = next(self.parameters()).device
        self.eval()
        batch = Batch.from_data_list(list(data_list)).to(device)

        node_h, graph_h, props_tokens = self.encode(batch)
        moe_gate_input = self._moe_gate_input(props_tokens, graph_h, batch)
        fused, node_ctx = self._fuse_node_inputs(node_h, props_tokens, batch)
        if self.use_moe and self.moe_gate is not None and self.num_experts > 0 and int(graph_h.numel()) != 0:
            gate_probs, topi, topv, entropy, topk_mass = self._moe_route(moe_gate_input)
            try:
                self._last_moe_topi = topi.detach().cpu().tolist()  # type: ignore[attr-defined]
                self._last_moe_topv = topv.detach().cpu().tolist()  # type: ignore[attr-defined]
                self._last_moe_entropy = entropy.detach().cpu().tolist()  # type: ignore[attr-defined]
                self._last_moe_topk_mass = topk_mass.detach().cpu().tolist()  # type: ignore[attr-defined]
                self._last_moe_gate_probs = gate_probs.detach().cpu().tolist()  # type: ignore[attr-defined]
            except Exception:
                pass
            op_logits = self._moe_apply(
                fused,
                batch.batch,
                self.op_experts,
                out_dim=4,
                num_graphs=int(graph_h.size(0)),
                topi=topi,
                topv=topv,
            )
            full_fg_logits = self._moe_apply(
                fused,
                batch.batch,
                self.fg_experts,
                out_dim=self.fg_vocab_size,
                num_graphs=int(graph_h.size(0)),
                topi=topi,
                topv=topv,
            )
        else:
            op_logits = self.op_head(fused)
            full_fg_logits = self.fg_head(fused)
            try:
                self._last_moe_topi = None  # type: ignore[attr-defined]
                self._last_moe_topv = None  # type: ignore[attr-defined]
                self._last_moe_entropy = None  # type: ignore[attr-defined]
                self._last_moe_topk_mass = None  # type: ignore[attr-defined]
                self._last_moe_gate_probs = None  # type: ignore[attr-defined]
            except Exception:
                pass
        if hasattr(batch, "op_allowed"):
            op_allowed = batch.op_allowed
        else:
            op_allowed = torch.ones((int(batch.z.size(0)), 4), device=batch.z.device, dtype=torch.bool)
        op_logits = op_logits.masked_fill(~op_allowed, -1e9)

        fg_id_to_smiles = getattr(fg_vocab, "id_to_smiles", {}) or {}

        n_nodes = int(batch.z.size(0))
        deg = torch.bincount(batch.edge_index[0].to(torch.long), minlength=n_nodes) if hasattr(batch, "edge_index") else None

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
                try:
                    if not bool(scaffold[int(anchor_idx)].item()):
                        return nbrs.new_empty((0,))
                    non_scaf = nbrs[~scaffold[nbrs]]
                    if int(non_scaf.numel()) > 0:
                        return non_scaf
                except Exception:
                    pass
            term = nbrs[deg[nbrs] <= 1]
            return term if int(term.numel()) else nbrs

        results_all: list[list[list[dict[str, Any]]]] = []
        num_graphs = int(batch.num_graphs)
        for g_i in range(num_graphs):
            node_idx = (batch.batch == int(g_i)).nonzero(as_tuple=True)[0]
            if int(node_idx.numel()) == 0:
                results_all.append([[] for _ in range(int(k_samples))])
                continue
            op_logits_g = op_logits[node_idx]
            op_allowed_g = op_allowed[node_idx]
            results: list[list[dict[str, Any]]] = []

            try:
                ggen = torch.Generator(device=device)
            except Exception:
                ggen = torch.Generator()
            ggen.manual_seed(int(seeds[g_i]))

            for _ in range(int(k_samples)):
                op_probs = F.softmax(op_logits_g / max(1e-6, float(temperature_op)), dim=-1)
                op_id = torch.full((int(op_probs.size(0)),), OP_TO_ID["none"], device=device, dtype=torch.long)

                if forced_k_target is not None:
                    k_target = max(0, min(int(forced_k_target), int(max_edits)))
                    if k_target > 0 and int(node_idx.numel()) > 0:
                        edit_score = (1.0 - op_probs[:, OP_TO_ID["none"]]).detach()
                        can_edit = op_allowed_g[:, 1:].any(dim=-1)
                        cand = can_edit.nonzero(as_tuple=True)[0]
                        if int(cand.numel()) > 0:
                            k_pick = min(int(k_target), int(cand.numel()))
                            top = torch.topk(edit_score[cand], k=k_pick, largest=True).indices
                            anchors = cand[top]
                            probs_non_none = op_probs[anchors, 1:]
                            probs_non_none = probs_non_none / probs_non_none.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                            sampled = torch.multinomial(probs_non_none, num_samples=1, generator=ggen).squeeze(-1) + 1
                            op_id[anchors] = sampled
                else:
                    sampled = torch.multinomial(op_probs, num_samples=1, generator=ggen).squeeze(-1)
                    op_id = sampled
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
                fg_logits = torch.zeros(
                    (int(op_logits_g.size(0)), int(self.fg_vocab_size)),
                    device=node_h.device,
                    dtype=op_logits.dtype,
                )
                if bool(fg_mask.any().item()):
                    fg_logits[fg_mask] = full_fg_logits[node_idx][fg_mask]
                fg_probs = F.softmax(fg_logits / max(1e-6, float(temperature_fg)), dim=-1)
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
                    mask = torch.zeros((int(self.fg_vocab_size),), device=device, dtype=torch.bool)
                    for fid in ids:
                        if 0 <= int(fid) < int(self.fg_vocab_size):
                            mask[int(fid)] = True
                    return mask

                allowed_mask_add = _build_allowed_mask(_fg_ids_for_op("add"))
                allowed_mask_replace = _build_allowed_mask(_fg_ids_for_op("replace"))

                edits: list[dict[str, Any]] = []
                for li in range(int(node_idx.numel())):
                    oid = int(op_id[li].item())
                    if oid == OP_TO_ID["none"]:
                        continue
                    op = ID_TO_OP[oid]
                    gi = int(node_idx[li].item())
                    anchor_map = int(batch.atom_map[gi].item())

                    e: dict[str, Any] = {"anchor_atom_map": anchor_map, "op": op}
                    if op in ("add", "replace"):
                        probs = fg_probs[li]
                        allowed_mask = allowed_mask_add if op == "add" else allowed_mask_replace
                        if allowed_mask is not None and bool(allowed_mask.any().item()):
                            masked = probs * allowed_mask.to(dtype=probs.dtype)
                            if float(masked.sum().item()) > 0.0:
                                probs = masked
                        probs = probs / probs.sum().clamp_min(1e-12)
                        fg_id = int(torch.multinomial(probs, num_samples=1, generator=ggen).item())
                        fg_smiles = fg_id_to_smiles.get(fg_id)
                        if fg_smiles is None:
                            fg_id = int(fg_probs[li].argmax().item())
                            fg_smiles = fg_id_to_smiles.get(fg_id)
                        e["fg_smiles"] = str(fg_smiles) if fg_smiles is not None else ""
                        e["fg_id"] = int(fg_id)
                    if op in ("remove", "replace"):
                        cand = _terminal_neighbor_candidates(int(gi))
                        if int(cand.numel()) > 0:
                            q = self.remove_query(torch.cat([node_ctx[gi], graph_h[int(g_i)]], dim=-1))
                            scores = (node_ctx[cand] * q.unsqueeze(0)).sum(dim=-1)
                            probs = F.softmax(scores / max(1e-6, float(temperature_remove)), dim=-1)
                            pick = int(torch.multinomial(probs, num_samples=1, generator=ggen).item())
                            removed_map = int(batch.atom_map[int(cand[pick].item())].item())
                            e["removed_atom_map"] = int(removed_map)
                    edits.append(e)

                def _op_rank(op_name: str) -> int:
                    return {"remove": 0, "replace": 1, "add": 2}.get(op_name, 99)

                edits = sorted(edits, key=lambda e: (_op_rank(str(e["op"])), int(e["anchor_atom_map"])))
                removed_maps: set[int] = set()
                for e in edits:
                    if str(e.get("op")) not in ("remove", "replace"):
                        continue
                    rm = e.get("removed_atom_map")
                    if rm is None:
                        continue
                    try:
                        rmi = int(rm)
                    except Exception:
                        continue
                    if rmi > 0:
                        removed_maps.add(rmi)
                if removed_maps:
                    edits = [e for e in edits if int(e.get("anchor_atom_map", -1)) not in removed_maps]
                for si, e in enumerate(edits, start=1):
                    e["step"] = si
                results.append(edits)

            results_all.append(results)

        return results_all


    def removed_scores_for_anchor(
        self, data, out: OneShotModelOutput, anchor_global_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return `(graph_node_idx, scores)` where `scores` is aligned with `graph_node_idx` (both 1D tensors).
        """
        g = int(data.batch[anchor_global_idx].item())
        q = self.remove_query(
            torch.cat([out.node_h[anchor_global_idx], out.graph_h[g]], dim=-1)
        )  # [H]
        idx = (data.batch == g).nonzero(as_tuple=True)[0]
        scores = (out.node_h[idx] * q.unsqueeze(0)).sum(dim=-1)
        return idx, scores

    @torch.no_grad()
    def decode_one(self, data) -> list[dict[str, int]]:
        """
        Greedy decode for a single-graph Data object.
        Returns a list of predicted edits (orderless).
        """
        require_torch_geometric()
        from torch_geometric.data import Batch

        self.eval()
        batch = Batch.from_data_list([data])
        out = self.forward(batch, teacher_forcing=False)

        if hasattr(batch, "op_allowed"):
            op_allowed = batch.op_allowed
        else:
            op_allowed = torch.ones((int(batch.z.size(0)), 4), device=batch.z.device, dtype=torch.bool)

        op_logits = out.op_logits.masked_fill(~op_allowed, -1e9)
        op_id = op_logits.argmax(dim=-1)  # [num_nodes]
        actions: list[dict[str, int]] = []

        node0 = (batch.batch == 0).nonzero(as_tuple=True)[0]
        n_nodes = int(batch.z.size(0))
        deg = torch.bincount(batch.edge_index[0].to(torch.long), minlength=n_nodes) if hasattr(batch, "edge_index") else None

        def _terminal_neighbor_candidates(anchor_idx: int) -> torch.Tensor:
            if not hasattr(batch, "edge_index") or deg is None:
                return torch.empty((0,), device=batch.z.device, dtype=torch.long)
            src = batch.edge_index[0]
            dst = batch.edge_index[1]
            nbrs = dst[src == int(anchor_idx)]
            if int(nbrs.numel()) == 0:
                return nbrs
            scaffold = getattr(batch, "scaffold_mask", None)
            if scaffold is not None:
                try:
                    if not bool(scaffold[int(anchor_idx)].item()):
                        return nbrs.new_empty((0,))
                    non_scaf = nbrs[~scaffold[nbrs]]
                    if int(non_scaf.numel()) > 0:
                        return non_scaf
                except Exception:
                    pass
            term = nbrs[deg[nbrs] <= 1]
            return term if int(term.numel()) else nbrs

        for gi in node0.tolist():
            oid = int(op_id[gi].item())
            if oid == OP_TO_ID["none"]:
                continue
            anchor_map = int(batch.atom_map[gi].item())
            op = ID_TO_OP[oid]
            if op in ("add", "replace"):
                fg_id = int(out.fg_logits[gi].argmax().item())
            else:
                fg_id = -1

            removed_map = -1
            if op in ("remove", "replace"):
                cand = _terminal_neighbor_candidates(int(gi))
                if int(cand.numel()) > 0:
                    q = self.remove_query(torch.cat([out.node_h[gi], out.graph_h[0]], dim=-1))
                    scores = (out.node_h[cand] * q.unsqueeze(0)).sum(dim=-1)
                    pick = int(scores.argmax().item())
                    removed_map = int(batch.atom_map[int(cand[pick].item())].item())
            actions.append(
                {
                    "anchor_atom_map": anchor_map,
                    "op_id": oid,
                    "fg_id": fg_id,
                    "removed_atom_map": removed_map,
                }
            )
        return actions
