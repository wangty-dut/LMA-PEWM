from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        y = float(x)
        if math.isnan(y) or math.isinf(y):
            return default
        return y
    except Exception:
        return default


class StructuredSemanticVectorizer:
    def __init__(self, dim: int = 64, max_numeric: int = 2048, max_tokens: int = 2048) -> None:
        self.dim = int(dim)
        self.max_numeric = int(max_numeric)
        self.max_tokens = int(max_tokens)
        if self.dim < 16:
            raise ValueError("semantic vector dimension must be at least 16")

    def _collect(self, obj: Any) -> Tuple[List[float], List[str]]:
        numeric: List[float] = []
        tokens: List[str] = []

        def rec(x: Any) -> None:
            if len(numeric) >= self.max_numeric and len(tokens) >= self.max_tokens:
                return
            if isinstance(x, bool):
                if len(numeric) < self.max_numeric:
                    numeric.append(1.0 if x else 0.0)
                return
            if isinstance(x, (int, float)):
                if len(numeric) < self.max_numeric:
                    numeric.append(_safe_float(x, 0.0))
                return
            if isinstance(x, dict):
                for k in sorted(x.keys(), key=str):
                    if len(tokens) < self.max_tokens:
                        tokens.append(str(k))
                    rec(x[k])
                return
            if isinstance(x, (list, tuple)):
                for item in x:
                    rec(item)
                return
            if x is not None and len(tokens) < self.max_tokens:
                s = str(x).strip()
                if s:
                    tokens.append(s)

        rec(obj)
        return numeric, tokens

    @staticmethod
    def _numeric_summary(values: List[float]) -> List[float]:
        if not values:
            return [0.0] * 12
        vals = [v for v in values if math.isfinite(v)]
        if not vals:
            return [0.0] * 12
        n = float(len(vals))
        s = float(sum(vals))
        mean = s / n
        var = sum((v - mean) ** 2 for v in vals) / n
        std = math.sqrt(max(var, 0.0))
        mn = min(vals)
        mx = max(vals)
        abs_mean = sum(abs(v) for v in vals) / n
        pos = sum(1 for v in vals if v > 0.0) / n
        neg = sum(1 for v in vals if v < 0.0) / n
        zero = sum(1 for v in vals if v == 0.0) / n
        l1 = sum(abs(v) for v in vals)
        l2 = math.sqrt(sum(v * v for v in vals))
        return [n, s, mean, std, mn, mx, abs_mean, pos, neg, zero, l1, l2]

    def vectorize(self, obj: Any) -> List[float]:
        numeric, tokens = self._collect(obj)
        base = self._numeric_summary(numeric)
        hash_dim = self.dim - len(base)
        hashed = [0.0] * hash_dim
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8", errors="ignore")).hexdigest()
            idx = int(digest, 16) % hash_dim
            hashed[idx] += 1.0
        hashed = [math.log1p(v) for v in hashed]
        out = base + hashed
        if len(out) < self.dim:
            out.extend([0.0] * (self.dim - len(out)))
        return out[: self.dim]

    def tensor(self, obj: Any, device: Optional[torch.device] = None) -> torch.Tensor:
        return torch.tensor(self.vectorize(obj), dtype=torch.float32, device=device)


@dataclass
class LMAPEWorldModelConfig:
    production_input_dim: int = 10
    energy_input_dim: int = 11
    coupling_input_dim: int = 8
    semantic_dim: int = 64
    action_dim: int = 12
    production_target_dim: int = 4
    energy_target_dim: int = 10
    coupling_target_dim: int = 3
    production_stage_classes: int = 5
    hidden_dim: int = 128
    latent_dim: int = 192
    time_embed_dim: int = 32
    action_embed_dim: int = 64
    num_attention_heads: int = 4
    dropout: float = 0.1


class BranchStateEncoder(nn.Module):
    def __init__(self, raw_dim: int, semantic_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.raw = nn.Sequential(
            nn.Linear(raw_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.semantic = nn.Sequential(
            nn.Linear(semantic_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, raw: torch.Tensor, semantic: torch.Tensor) -> torch.Tensor:
        if semantic.dim() == 2:
            semantic = semantic.unsqueeze(1).expand(-1, raw.size(1), -1)
        if semantic.size(1) != raw.size(1):
            if semantic.size(1) == 1:
                semantic = semantic.expand(-1, raw.size(1), -1)
            else:
                semantic = semantic[:, -1:, :].expand(-1, raw.size(1), -1)
        x = torch.cat([self.raw(raw), self.semantic(semantic)], dim=-1)
        return self.fuse(x)


class VariableStepGRU(nn.Module):
    def __init__(self, hidden_dim: int, time_embed_dim: int, dropout: float) -> None:
        super().__init__()
        self.time_encoder = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.Tanh(),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.GELU(),
        )
        self.gru = nn.GRU(
            input_size=hidden_dim + time_embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        if delta_t.dim() == 2:
            delta_t = delta_t.unsqueeze(-1)
        dt = torch.log1p(torch.clamp(delta_t, min=0.0))
        dt_emb = self.time_encoder(dt)
        y, _ = self.gru(torch.cat([x, dt_emb], dim=-1))
        return self.norm(self.dropout(y) + x)


class MBCA(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.p_from_e = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.e_from_p = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.c_from_pe = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.coupling_gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.Sigmoid())
        self.p_norm = nn.LayerNorm(hidden_dim)
        self.e_norm = nn.LayerNorm(hidden_dim)
        self.c_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        production_history: torch.Tensor,
        energy_history: torch.Tensor,
        coupling_history: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p_cross, _ = self.p_from_e(production_history, energy_history, energy_history, need_weights=False)
        e_cross, _ = self.e_from_p(energy_history, production_history, production_history, need_weights=False)
        joint_kv = torch.cat([production_history, energy_history], dim=1)
        c_cross, _ = self.c_from_pe(coupling_history, joint_kv, joint_kv, need_weights=False)
        gate = self.coupling_gate(coupling_history[:, -1, :])
        p_gate, e_gate = gate.chunk(2, dim=-1)
        p = self.p_norm(production_history + p_cross * p_gate.unsqueeze(1))
        e = self.e_norm(energy_history + e_cross * e_gate.unsqueeze(1))
        c = self.c_norm(coupling_history + c_cross)
        return p, e, c


class JointStateFusion(nn.Module):
    def __init__(self, hidden_dim: int, latent_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 3, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, p: torch.Tensor, e: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([p[:, -1, :], e[:, -1, :], c[:, -1, :]], dim=-1))


class WorldModelDynamics(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, action_embed_dim: int, time_embed_dim: int) -> None:
        super().__init__()
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, action_embed_dim),
            nn.GELU(),
            nn.Linear(action_embed_dim, action_embed_dim),
            nn.LayerNorm(action_embed_dim),
        )
        self.time_encoder = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.Tanh(),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.GELU(),
        )
        self.cell = nn.GRUCell(action_embed_dim + time_embed_dim, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, latent: torch.Tensor, action: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        if delta_t.dim() == 1:
            delta_t = delta_t.unsqueeze(-1)
        action_emb = self.action_encoder(action)
        time_emb = self.time_encoder(torch.log1p(torch.clamp(delta_t, min=0.0)))
        next_latent = self.cell(torch.cat([action_emb, time_emb], dim=-1), latent)
        return self.norm(next_latent)


class LMAPEWorldModel(nn.Module):
    def __init__(self, config: Optional[LMAPEWorldModelConfig] = None) -> None:
        super().__init__()
        self.config = config or LMAPEWorldModelConfig()
        cfg = self.config

        self.production_encoder = BranchStateEncoder(cfg.production_input_dim, cfg.semantic_dim, cfg.hidden_dim, cfg.dropout)
        self.energy_encoder = BranchStateEncoder(cfg.energy_input_dim, cfg.semantic_dim, cfg.hidden_dim, cfg.dropout)
        self.coupling_encoder = BranchStateEncoder(cfg.coupling_input_dim, cfg.semantic_dim, cfg.hidden_dim, cfg.dropout)

        self.production_vs_gru = VariableStepGRU(cfg.hidden_dim, cfg.time_embed_dim, cfg.dropout)
        self.energy_vs_gru = VariableStepGRU(cfg.hidden_dim, cfg.time_embed_dim, cfg.dropout)
        self.coupling_vs_gru = VariableStepGRU(cfg.hidden_dim, cfg.time_embed_dim, cfg.dropout)

        self.mbca = MBCA(cfg.hidden_dim, cfg.num_attention_heads, cfg.dropout)
        self.fusion = JointStateFusion(cfg.hidden_dim, cfg.latent_dim, cfg.dropout)
        self.dynamics = WorldModelDynamics(cfg.latent_dim, cfg.action_dim, cfg.action_embed_dim, cfg.time_embed_dim)

        self.production_numeric_decoder = nn.Linear(cfg.latent_dim, cfg.production_target_dim)
        self.production_stage_decoder = nn.Linear(cfg.latent_dim, cfg.production_stage_classes)
        self.energy_numeric_decoder = nn.Linear(cfg.latent_dim, cfg.energy_target_dim)
        self.energy_risk_decoder = nn.Linear(cfg.latent_dim, 1)
        self.coupling_numeric_decoder = nn.Linear(cfg.latent_dim, cfg.coupling_target_dim)
        self.feasibility_decoder = nn.Linear(cfg.latent_dim, 1)
        self.reward_decoder = nn.Linear(cfg.latent_dim, 1)

    def encode_history(
        self,
        production_history: torch.Tensor,
        energy_history: torch.Tensor,
        coupling_history: torch.Tensor,
        history_delta_t: torch.Tensor,
        production_semantic: torch.Tensor,
        energy_semantic: torch.Tensor,
        coupling_semantic: torch.Tensor,
    ) -> torch.Tensor:
        p = self.production_encoder(production_history, production_semantic)
        e = self.energy_encoder(energy_history, energy_semantic)
        c = self.coupling_encoder(coupling_history, coupling_semantic)

        p = self.production_vs_gru(p, history_delta_t)
        e = self.energy_vs_gru(e, history_delta_t)
        c = self.coupling_vs_gru(c, history_delta_t)

        p, e, c = self.mbca(p, e, c)
        return self.fusion(p, e, c)

    def _decode_latent(self, latent: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "production_numeric": self.production_numeric_decoder(latent),
            "production_stage_logits": self.production_stage_decoder(latent),
            "energy_numeric": self.energy_numeric_decoder(latent),
            "energy_risk_logit": self.energy_risk_decoder(latent).squeeze(-1),
            "coupling_numeric": self.coupling_numeric_decoder(latent),
            "feasibility_logit": self.feasibility_decoder(latent).squeeze(-1),
            "reward": self.reward_decoder(latent).squeeze(-1),
        }

    def forward(
        self,
        production_history: torch.Tensor,
        energy_history: torch.Tensor,
        coupling_history: torch.Tensor,
        history_delta_t: torch.Tensor,
        production_semantic: torch.Tensor,
        energy_semantic: torch.Tensor,
        coupling_semantic: torch.Tensor,
        future_actions: torch.Tensor,
        future_delta_t: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        initial_latent = self.encode_history(
            production_history=production_history,
            energy_history=energy_history,
            coupling_history=coupling_history,
            history_delta_t=history_delta_t,
            production_semantic=production_semantic,
            energy_semantic=energy_semantic,
            coupling_semantic=coupling_semantic,
        )

        latent = initial_latent
        horizon = future_actions.size(1)
        latent_steps: List[torch.Tensor] = []
        decoded: Dict[str, List[torch.Tensor]] = {
            "production_numeric": [],
            "production_stage_logits": [],
            "energy_numeric": [],
            "energy_risk_logit": [],
            "coupling_numeric": [],
            "feasibility_logit": [],
            "reward": [],
        }

        for k in range(horizon):
            latent = self.dynamics(latent, future_actions[:, k, :], future_delta_t[:, k])
            latent_steps.append(latent)
            step = self._decode_latent(latent)
            for key, value in step.items():
                decoded[key].append(value)

        out = {key: torch.stack(values, dim=1) for key, values in decoded.items()}
        out["latent_trajectory"] = torch.stack(latent_steps, dim=1)
        out["initial_latent"] = initial_latent
        return out

    @torch.no_grad()
    def imagine(self, **batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        training = self.training
        self.eval()
        out = self.forward(**batch)
        if training:
            self.train()
        return out

    def checkpoint_payload(self) -> Dict[str, Any]:
        return {
            "model_config": asdict(self.config),
            "model_state_dict": self.state_dict(),
        }

    @classmethod
    def from_checkpoint(cls, payload: Dict[str, Any], map_location: Optional[str] = None) -> "LMAPEWorldModel":
        cfg = LMAPEWorldModelConfig(**dict(payload.get("model_config") or {}))
        model = cls(cfg)
        state = payload.get("model_state_dict") or payload.get("state_dict")
        if not isinstance(state, dict):
            raise ValueError("checkpoint does not contain model_state_dict")
        model.load_state_dict(state)
        if map_location:
            model.to(torch.device(map_location))
        return model


def save_world_model(model: LMAPEWorldModel, path: str, extra: Optional[Dict[str, Any]] = None) -> None:
    payload = model.checkpoint_payload()
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_world_model(path: str, device: str = "cpu") -> LMAPEWorldModel:
    payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict):
        raise ValueError("invalid world-model checkpoint")
    model = LMAPEWorldModel.from_checkpoint(payload, map_location=device)
    model.eval()
    return model
