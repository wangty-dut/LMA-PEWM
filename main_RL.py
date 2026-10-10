from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Dataset

from g_lma_pewm import LMAPEWorldModel, StructuredSemanticVectorizer, load_world_model
from mechanism_energy_projection import PhysicsConsistentEnergyProjector
from mechanism_schedule_projection import PhysicsConsistentVariationalProjector


PRODUCTION_STAGE_NAMES = ["BOF", "LF", "Casting", "Rolling", "Other"]


PRODUCTION_INPUT_DIM = 10
ENERGY_INPUT_DIM = 11
COUPLING_INPUT_DIM = 8
ACTION_DIM = 12


def safe_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def safe_list(x: Any) -> List[Any]:
    if isinstance(x, list):
        return x
    if x is None:
        return []
    return [x]


def nested_get(obj: Dict[str, Any], paths: Sequence[Sequence[str]], default: float = 0.0) -> float:
    for path in paths:
        cur: Any = obj
        ok = True
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                ok = False
                break
            cur = cur[key]
        if ok:
            return safe_float(cur, default)
    return float(default)


def signed_log1p(x: float) -> float:
    value = safe_float(x, 0.0)
    if value == 0.0:
        return 0.0
    return math.copysign(math.log1p(abs(value)), value)


def stage_from_patch(patch: Dict[str, Any]) -> str:
    patch = safe_dict(patch)
    raw = str(
        patch.get("stage")
        or patch.get("representative_stage")
        or patch.get("process")
        or patch.get("operation")
        or ""
    ).strip()
    if not raw:
        stages = safe_list(patch.get("stages"))
        raw = str(stages[0]) if stages else ""
    value = raw.upper().replace(" ", "").replace("-", "_")
    if "BOF" in value or "CONVERTER" in value or "LDG" in value:
        return "BOF"
    if value == "LF" or "LADLE" in value or "REFIN" in value:
        return "LF"
    if "CAST" in value or value == "CC" or value.startswith("CC"):
        return "Casting"
    if "ROLL" in value or re.fullmatch(r"11\d\d", value):
        return "Rolling"
    return "Other"


def stage_index(stage: str) -> int:
    try:
        return PRODUCTION_STAGE_NAMES.index(stage)
    except ValueError:
        return len(PRODUCTION_STAGE_NAMES) - 1


def duration_min(patch: Dict[str, Any], default: float = 1.0) -> float:
    patch = safe_dict(patch)
    for key in ["duration_min", "duration", "dt", "delta_t"]:
        if patch.get(key) is not None:
            value = safe_float(patch.get(key), default)
            if value > 0:
                return value
    t = safe_dict(patch.get("t"))
    starts = [
        patch.get("start_min"),
        patch.get("t0"),
        t.get("t0"),
        t.get("s0"),
    ]
    ends = [
        patch.get("end_min"),
        patch.get("t1"),
        t.get("t1"),
        t.get("s1"),
    ]
    start = next((safe_float(v, float("nan")) for v in starts if v is not None), float("nan"))
    end = next((safe_float(v, float("nan")) for v in ends if v is not None), float("nan"))
    if not math.isnan(start) and not math.isnan(end) and end > start:
        return end - start
    return float(default)


def production_target_features(patch: Dict[str, Any]) -> Tuple[List[float], int]:
    patch = safe_dict(patch)
    active_heats = safe_list(patch.get("active_heats") or patch.get("heats") or patch.get("heat_ids"))
    rolling = safe_list(patch.get("active_rolling_units") or patch.get("rolling_units"))
    stages = safe_list(patch.get("stages"))
    if not stages:
        stage = stage_from_patch(patch)
        stages = [] if stage == "Other" else [stage]
    numeric = [
        signed_log1p(len(active_heats)),
        signed_log1p(len(rolling)),
        signed_log1p(duration_min(patch, 1.0)),
        signed_log1p(len(stages)),
    ]
    return numeric, stage_index(stage_from_patch(patch))


def production_input_features(patch: Dict[str, Any]) -> List[float]:
    numeric, st_idx = production_target_features(patch)
    one_hot = [1.0 if i == st_idx else 0.0 for i in range(len(PRODUCTION_STAGE_NAMES))]
    converter = 1.0 if safe_dict(patch).get("converter") or safe_dict(patch).get("unit") else 0.0
    return (numeric + one_hot + [converter])[:PRODUCTION_INPUT_DIM]


def energy_numeric_features(patch: Dict[str, Any]) -> List[float]:
    patch = safe_dict(patch)
    raw = [
        nested_get(patch, [["g", "b", "tot"], ["g", "b_tot"], ["bfg_generation"], ["BFG_generation"]]),
        nested_get(patch, [["g", "c", "tot"], ["g", "c_tot"], ["cog_generation"], ["COG_generation"]]),
        nested_get(patch, [["g", "l", "tot"], ["g", "l_tot"], ["ldg_generation"], ["LDG_generation"]]),
        nested_get(patch, [["st", "b", "e"], ["st", "b_e"], ["bfg_holder"], ["BFG_holder"]]),
        nested_get(patch, [["st", "c", "e"], ["st", "c_e"], ["cog_holder"], ["COG_holder"]]),
        nested_get(patch, [["st", "l", "e"], ["st", "l_e"], ["ldg_holder"], ["LDG_holder"]]),
        nested_get(patch, [["dm", "s", "tot"], ["dm", "s_tot"], ["steam_load"], ["steam_demand"]]),
        nested_get(patch, [["dm", "e", "tot"], ["dm", "e_tot"], ["power_load"], ["power_demand"]]),
        nested_get(patch, [["ex", "s", "tot"], ["ex", "s_tot"], ["steam_purchase"], ["purchased_steam"]]),
        nested_get(patch, [["ex", "e", "tot"], ["ex", "e_tot"], ["power_purchase"], ["purchased_power"]]),
    ]
    return [signed_log1p(v) for v in raw]


def energy_risk_count(patch: Dict[str, Any]) -> int:
    patch = safe_dict(patch)
    risks = patch.get("rk") or patch.get("risk_flags") or patch.get("risk_labels") or patch.get("key_risks")
    return len(safe_list(risks))


def energy_input_features(patch: Dict[str, Any]) -> List[float]:
    return energy_numeric_features(patch) + [signed_log1p(energy_risk_count(patch))]


def coupling_input_features(prod_patch: Dict[str, Any], energy_patch: Dict[str, Any]) -> List[float]:
    p_num, _ = production_target_features(prod_patch)
    e = energy_numeric_features(energy_patch)
    generation_total = sum(math.expm1(max(0.0, v)) for v in e[:3])
    holder_mean = sum(e[3:6]) / 3.0 if len(e) >= 6 else 0.0
    load_total = e[6] + e[7] if len(e) >= 8 else 0.0
    purchase_total = e[8] + e[9] if len(e) >= 10 else 0.0
    return (p_num + [signed_log1p(generation_total), holder_mean, load_total, purchase_total])[:COUPLING_INPUT_DIM]


def action_features(action: Dict[str, Any]) -> List[float]:
    action = safe_dict(action)
    pa = safe_dict(action.get("production_action"))
    ea = safe_dict(action.get("energy_action"))
    ptype = str(pa.get("type") or "keep").strip().lower()
    ptypes = ["keep", "shift_heat", "shift_rolling_unit", "other"]
    pkey = ptype if ptype in ptypes[:-1] else "other"
    p_onehot = [1.0 if pkey == name else 0.0 for name in ptypes]
    p_delta = math.tanh(safe_float(pa.get("delta_min"), 0.0) / 60.0)
    etype = str(ea.get("type") or "keep").strip().lower()
    etypes = ["keep", "increase_gas_use", "increase_boiler_load", "increase_power_purchase", "reduce_power_purchase", "other"]
    ekey = etype if etype in etypes[:-1] else "other"
    e_onehot = [1.0 if ekey == name else 0.0 for name in etypes]
    level = str(ea.get("delta_level") or "none").strip().lower()
    level_value = {"none": 0.0, "low": 0.33, "medium": 0.67, "high": 1.0}.get(level, 0.0)
    return (p_onehot + [p_delta] + e_onehot + [level_value])[:ACTION_DIM]


def action_magnitude(action: Dict[str, Any]) -> float:
    pa = safe_dict(safe_dict(action).get("production_action"))
    ea = safe_dict(safe_dict(action).get("energy_action"))
    value = abs(safe_float(pa.get("delta_min"), 0.0)) / 15.0
    level = str(ea.get("delta_level") or "none").lower()
    value += {"none": 0.0, "low": 0.5, "medium": 1.0, "high": 1.5}.get(level, 0.5)
    if str(pa.get("type") or "keep") != "keep":
        value += 0.5
    if str(ea.get("type") or "keep") != "keep":
        value += 0.5
    return float(value)


def _pad_sequence(seq: Sequence[Any], length: int, pad: Any) -> List[Any]:
    values = list(seq[-length:])
    if len(values) < length:
        values = [pad for _ in range(length - len(values))] + values
    return values


def load_semantic_records(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            value = line.strip()
            if not value:
                continue
            obj = json.loads(value)
            if isinstance(obj, dict):
                records.append(obj)
    if not records:
        raise ValueError(f"No records found in semantic cache: {path}")
    return records


class BehaviorStateDataset(Dataset):
    def __init__(
        self,
        records: Sequence[Dict[str, Any]],
        semantic_dim: int,
    ) -> None:
        self.records = list(records)
        self.vectorizer = StructuredSemanticVectorizer(dim=semantic_dim)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        record = safe_dict(self.records[index])
        tr = safe_dict(record.get("training_record"))
        sem = safe_dict(record.get("llm_semantics"))
        h = int(tr.get("history_window") or 5)
        k = int(tr.get("horizon") or 5)
        ph = _pad_sequence(safe_list(tr.get("production_history")), h, {})
        eh = _pad_sequence(safe_list(tr.get("energy_history")), h, {})
        ch = _pad_sequence(safe_list(tr.get("coupling_history")), h, {})
        hist_dt = _pad_sequence(safe_list(tr.get("history_delta_t")), h, 1.0)
        fut_dt = _pad_sequence(safe_list(tr.get("future_delta_t")), k, 1.0)[-k:]
        coupling_rows = []
        for i in range(h):
            row = safe_dict(ch[i])
            coupling_rows.append(
                coupling_input_features(
                    safe_dict(row.get("production")) if row else ph[i],
                    safe_dict(row.get("energy")) if row else eh[i],
                )
            )
        return {
            "production_history": torch.tensor([production_input_features(x) for x in ph], dtype=torch.float32),
            "energy_history": torch.tensor([energy_input_features(x) for x in eh], dtype=torch.float32),
            "coupling_history": torch.tensor(coupling_rows, dtype=torch.float32),
            "history_delta_t": torch.tensor(hist_dt, dtype=torch.float32).unsqueeze(-1),
            "production_semantic": torch.tensor(self.vectorizer.vectorize(sem.get("production")), dtype=torch.float32),
            "energy_semantic": torch.tensor(self.vectorizer.vectorize(sem.get("energy")), dtype=torch.float32),
            "coupling_semantic": torch.tensor(self.vectorizer.vectorize(sem.get("coupling")), dtype=torch.float32),
            "future_delta_t": torch.tensor(fut_dt, dtype=torch.float32).unsqueeze(-1),
        }


WorldModelTensorDataset = BehaviorStateDataset


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        y = float(x)
        if math.isnan(y) or math.isinf(y):
            return float(default)
        return y
    except Exception:
        return float(default)


def json_safe(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return x.detach().cpu().item()
        return x.detach().cpu().tolist()
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    return x


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def build_joint_action_space() -> List[Dict[str, Any]]:
    production_actions = [
        {"type": "keep", "target": "production_schedule", "delta_min": 0},
        {"type": "shift_heat", "target": "focus_heat", "delta_min": -15},
        {"type": "shift_heat", "target": "focus_heat", "delta_min": 15},
        {"type": "shift_rolling_unit", "target": "focus_rolling_unit", "delta_min": -15},
        {"type": "shift_rolling_unit", "target": "focus_rolling_unit", "delta_min": 15},
    ]
    energy_actions = [
        {"type": "keep", "target": "energy_system", "delta_level": "none"},
        {"type": "increase_gas_use", "target": "byproduct_gas", "delta_level": "medium"},
        {"type": "increase_boiler_load", "target": "boiler", "delta_level": "medium"},
        {"type": "increase_power_purchase", "target": "grid", "delta_level": "medium"},
        {"type": "reduce_power_purchase", "target": "grid", "delta_level": "medium"},
    ]
    actions: List[Dict[str, Any]] = []
    for i, pa in enumerate(production_actions):
        for j, ea in enumerate(energy_actions):
            actions.append(
                {
                    "action_id": f"JA_{i}_{j}",
                    "production_action": dict(pa),
                    "energy_action": dict(ea),
                }
            )
    return actions


@dataclass
class RestorationConfig:
    max_iterations: int = 3
    damping: float = 0.75
    tolerance: float = 1e-4
    latent_correction_clip: float = 5.0


class KKTNewtonLocalMechanismRestorer:
    def __init__(
        self,
        world_model: LMAPEWorldModel,
        config: Optional[RestorationConfig] = None,
    ) -> None:
        self.world_model = world_model
        self.config = config or RestorationConfig()
        self.energy_projector = PhysicsConsistentEnergyProjector()
        self.schedule_projector = PhysicsConsistentVariationalProjector()
        weights = [
            self.world_model.production_numeric_decoder.weight.detach(),
            self.world_model.energy_numeric_decoder.weight.detach(),
            self.world_model.coupling_numeric_decoder.weight.detach(),
        ]
        self.decoder_jacobian = torch.cat(weights, dim=0)
        self.decoder_pinv = torch.linalg.pinv(self.decoder_jacobian)

    def _inverse_signed_log(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * torch.expm1(torch.abs(x))

    def _signed_log(self, x: float) -> float:
        if x == 0.0:
            return 0.0
        return math.copysign(math.log1p(abs(float(x))), float(x))

    def _energy_vector_to_prediction(self, vector: torch.Tensor) -> Dict[str, Any]:
        raw = self._inverse_signed_log(vector.detach().cpu()).tolist()
        values = [safe_float(v, 0.0) for v in raw]
        return {
            "future_patch_predictions": [
                {
                    "step": 1,
                    "patch_id": "imagined_step",
                    "id": "imagined_step",
                    "p": "imagined_step",
                    "g": {
                        "b": {"tot": values[0]},
                        "c": {"tot": values[1]},
                        "l": {"tot": values[2]},
                    },
                    "st": {
                        "b": {"e": values[3], "m": values[3]},
                        "c": {"e": values[4], "m": values[4]},
                        "l": {"e": values[5], "m": values[5]},
                    },
                    "dm": {
                        "s": {"tot": values[6], "pk": values[6]},
                        "e": {"tot": values[7], "pk": values[7]},
                    },
                    "ex": {
                        "s": {"tot": values[8]},
                        "e": {"tot": values[9]},
                    },
                    "rk": [],
                    "confidence": 1.0,
                }
            ]
        }

    def _energy_prediction_to_vector(self, prediction: Dict[str, Any], device: torch.device) -> torch.Tensor:
        steps = prediction.get("future_patch_predictions")
        row = steps[0] if isinstance(steps, list) and steps and isinstance(steps[0], dict) else {}
        def nested(path: Sequence[str]) -> float:
            cur: Any = row
            for key in path:
                if not isinstance(cur, dict):
                    return 0.0
                cur = cur.get(key)
            return safe_float(cur, 0.0)
        raw = [
            nested(["g", "b", "tot"]),
            nested(["g", "c", "tot"]),
            nested(["g", "l", "tot"]),
            nested(["st", "b", "e"]),
            nested(["st", "c", "e"]),
            nested(["st", "l", "e"]),
            nested(["dm", "s", "tot"]),
            nested(["dm", "e", "tot"]),
            nested(["ex", "s", "tot"]),
            nested(["ex", "e", "tot"]),
        ]
        return torch.tensor([self._signed_log(v) for v in raw], dtype=torch.float32, device=device)

    def _repair_energy(self, energy: torch.Tensor) -> torch.Tensor:
        device = energy.device
        raw_prediction = self._energy_vector_to_prediction(energy)
        context = {
            "future_patch_context": [{"id": "imagined_step", "patch_id": "imagined_step"}],
            "energy_history": [],
            "production_operation_segment_summary": {},
            "constraints": {},
        }
        result = self.energy_projector.project(raw_prediction, repair_context=context)
        prediction = result.get("prediction") if isinstance(result, dict) else None
        if not isinstance(prediction, dict):
            prediction = raw_prediction
        return self._energy_prediction_to_vector(prediction, device)

    def _repair_production(
        self,
        production: torch.Tensor,
        stage_logits: torch.Tensor,
        delta_t: float,
    ) -> torch.Tensor:
        repaired = torch.clamp(production, min=0.0)
        stage_index = int(torch.argmax(stage_logits).detach().cpu().item())
        stage = PRODUCTION_STAGE_NAMES[min(max(stage_index, 0), len(PRODUCTION_STAGE_NAMES) - 1)]
        if stage == "Other":
            return repaired
        duration_raw = max(0.5, math.expm1(max(0.0, safe_float(repaired[2].detach().cpu().item(), 0.0))))
        horizon_end = max(duration_raw, safe_float(delta_t, 1.0), 1.0)
        segment = {
            "segment_id": "imagined_segment",
            "heat_id": "IMAGINED_HEAT",
            "stage": stage,
            "unit": stage,
            "start_min": 0.0,
            "end_min": duration_raw,
            "duration_min": duration_raw,
            "source": "model_output",
            "confidence": 1.0,
        }
        result = self.schedule_projector.project(
            [segment],
            repair_context={
                "future_time_window": {"start_min": 0.0, "end_min": horizon_end},
                "stage_order": ["BOF", "LF", "Casting", "Rolling"],
            },
        )
        segments = result.get("repaired_segments") if isinstance(result, dict) else None
        if isinstance(segments, list) and segments and isinstance(segments[0], dict):
            duration = safe_float(segments[0].get("duration_min"), duration_raw)
            repaired = repaired.clone()
            repaired[2] = float(self._signed_log(max(duration, 0.0)))
        return repaired

    def _repair_coupling(self, coupling: torch.Tensor) -> torch.Tensor:
        out = coupling.clone()
        if out.numel() >= 1:
            out[0] = torch.clamp(out[0], min=0.0)
        if out.numel() >= 2:
            out[1] = torch.clamp(out[1], min=0.0)
        if out.numel() >= 3:
            out[2] = torch.clamp(out[2], min=0.0, max=1.0)
        return out

    def _state_vector(self, decoded: Dict[str, torch.Tensor], index: int) -> torch.Tensor:
        return torch.cat(
            [
                decoded["production_numeric"][index],
                decoded["energy_numeric"][index],
                decoded["coupling_numeric"][index],
            ],
            dim=-1,
        )

    def _repair_state_vector(
        self,
        decoded: Dict[str, torch.Tensor],
        index: int,
        delta_t: float,
    ) -> torch.Tensor:
        p = self._repair_production(
            decoded["production_numeric"][index],
            decoded["production_stage_logits"][index],
            delta_t,
        )
        e = self._repair_energy(decoded["energy_numeric"][index])
        c = self._repair_coupling(decoded["coupling_numeric"][index])
        return torch.cat([p, e, c], dim=-1)

    def _mechanism_residual(self, decoded: Dict[str, torch.Tensor]) -> torch.Tensor:
        p = decoded["production_numeric"]
        e = decoded["energy_numeric"]
        c = decoded["coupling_numeric"]
        residual = torch.relu(-p).pow(2).sum(dim=-1)
        residual = residual + torch.relu(-e).pow(2).sum(dim=-1)
        residual = residual + torch.relu(-c[..., :2]).pow(2).sum(dim=-1)
        if c.size(-1) >= 3:
            residual = residual + torch.relu(-c[..., 2]).pow(2)
            residual = residual + torch.relu(c[..., 2] - 1.0).pow(2)
        if e.size(-1) >= 10:
            residual = residual + 0.25 * torch.relu(e[..., 8] - e[..., 6]).pow(2)
            residual = residual + 0.25 * torch.relu(e[..., 9] - e[..., 7]).pow(2)
        return residual

    @torch.no_grad()
    def restore(
        self,
        latent: torch.Tensor,
        delta_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, Any]]:
        z = latent.detach().clone()
        initial_decoded = self.world_model._decode_latent(z)
        residual_before = self._mechanism_residual(initial_decoded)
        iteration_logs: List[Dict[str, Any]] = []
        batch_size = z.size(0)

        for iteration in range(self.config.max_iterations):
            decoded = self.world_model._decode_latent(z)
            predicted = torch.stack([self._state_vector(decoded, i) for i in range(batch_size)], dim=0)
            target = torch.stack(
                [
                    self._repair_state_vector(
                        decoded,
                        i,
                        safe_float(delta_t[i].reshape(-1)[0].detach().cpu().item(), 1.0),
                    )
                    for i in range(batch_size)
                ],
                dim=0,
            )
            delta = target - predicted
            max_change = float(delta.abs().max().detach().cpu().item()) if delta.numel() else 0.0
            correction = delta @ self.decoder_pinv.transpose(0, 1)
            correction = torch.clamp(
                correction,
                min=-self.config.latent_correction_clip,
                max=self.config.latent_correction_clip,
            )
            z = z + self.config.damping * correction
            iteration_logs.append(
                {
                    "iteration": iteration + 1,
                    "max_state_correction": max_change,
                    "mean_state_correction": float(delta.abs().mean().detach().cpu().item()) if delta.numel() else 0.0,
                }
            )
            if max_change <= self.config.tolerance:
                break

        final_decoded = self.world_model._decode_latent(z)
        residual_after = self._mechanism_residual(final_decoded)
        diagnostics = {
            "method": "KKT-NLMR",
            "world_model_frozen": True,
            "newton_iterations": len(iteration_logs),
            "residual_before_mean": float(residual_before.mean().detach().cpu().item()),
            "residual_after_mean": float(residual_after.mean().detach().cpu().item()),
            "residual_before": residual_before.detach().cpu().tolist(),
            "residual_after": residual_after.detach().cpu().tolist(),
            "iteration_logs": iteration_logs,
        }
        return z.detach(), final_decoded, diagnostics


class ActorNetwork(nn.Module):
    def __init__(self, latent_dim: int, num_actions: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class CriticNetwork(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


@dataclass
class BehaviorTrainingConfig:
    gamma: float = 0.95
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    violation_penalty_coef: float = 2.0
    hidden_dim: int = 256
    imagination_horizon: int = 15
    epochs: int = 50
    batch_size: int = 32
    max_grad_norm: float = 5.0
    seed: int = 42
    device: str = "auto"
    validation_fraction: float = 0.1
    restore_max_iterations: int = 3
    restore_damping: float = 0.75


class BehaviorModelTrainer:
    def __init__(
        self,
        world_model: LMAPEWorldModel,
        config: BehaviorTrainingConfig,
        action_space: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.config = config
        self.device = self._resolve_device(config.device)
        self.world_model = world_model.to(self.device)
        self.world_model.eval()
        for parameter in self.world_model.parameters():
            parameter.requires_grad_(False)

        self.action_space = action_space or build_joint_action_space()
        latent_dim = int(self.world_model.config.latent_dim)
        self.actor = ActorNetwork(latent_dim, len(self.action_space), config.hidden_dim).to(self.device)
        self.critic = CriticNetwork(latent_dim, config.hidden_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.restorer = KKTNewtonLocalMechanismRestorer(
            self.world_model,
            RestorationConfig(
                max_iterations=config.restore_max_iterations,
                damping=config.restore_damping,
            ),
        )
        self.action_tensor = torch.tensor(
            [action_features(a) for a in self.action_space],
            dtype=torch.float32,
            device=self.device,
        )
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

    def _resolve_device(self, requested: str) -> torch.device:
        value = str(requested or "auto").lower()
        if value == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(value)

    def _move(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.to(self.device) for k, v in batch.items()}

    @torch.no_grad()
    def _initial_latent(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.world_model.encode_history(
            production_history=batch["production_history"],
            energy_history=batch["energy_history"],
            coupling_history=batch["coupling_history"],
            history_delta_t=batch["history_delta_t"],
            production_semantic=batch["production_semantic"],
            energy_semantic=batch["energy_semantic"],
            coupling_semantic=batch["coupling_semantic"],
        ).detach()

    def _step_reward(
        self,
        decoded: Dict[str, torch.Tensor],
        action_indices: torch.Tensor,
        residual_before: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        consistency = torch.clamp(decoded["coupling_numeric"][..., 2], 0.0, 1.0)
        feasibility = torch.sigmoid(decoded["feasibility_logit"])
        risk = torch.sigmoid(decoded["energy_risk_logit"])
        energy = decoded["energy_numeric"]
        purchase = torch.zeros_like(consistency)
        if energy.size(-1) >= 10:
            steam_purchase = torch.expm1(torch.clamp(energy[..., 8], min=0.0))
            power_purchase = torch.expm1(torch.clamp(energy[..., 9], min=0.0))
            purchase = steam_purchase + power_purchase

        magnitudes = torch.tensor(
            [action_magnitude(self.action_space[int(i)]) for i in action_indices.detach().cpu().tolist()],
            dtype=torch.float32,
            device=self.device,
        )
        learned_reward = decoded["reward"]
        structured_reward = (
            2.0 * consistency
            + 1.0 * feasibility
            - 1.5 * risk
            - 0.05 * purchase
            - 0.02 * magnitudes
            - self.config.violation_penalty_coef * residual_before
        )
        reward = 0.5 * learned_reward + 0.5 * structured_reward
        terms = {
            "learned_reward": learned_reward.detach(),
            "structured_reward": structured_reward.detach(),
            "consistency": consistency.detach(),
            "feasibility": feasibility.detach(),
            "risk": risk.detach(),
            "purchase": purchase.detach(),
            "action_magnitude": magnitudes.detach(),
            "mechanism_residual_penalty": (
                self.config.violation_penalty_coef * residual_before
            ).detach(),
        }
        return reward.detach(), terms

    def _discounted_returns(self, rewards: torch.Tensor) -> torch.Tensor:
        horizon, batch_size = rewards.shape
        returns = torch.zeros_like(rewards)
        running = torch.zeros(batch_size, dtype=torch.float32, device=rewards.device)
        for t in range(horizon - 1, -1, -1):
            running = rewards[t] + self.config.gamma * running
            returns[t] = running
        return returns

    def imagined_rollout(
        self,
        batch: Dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> Dict[str, Any]:
        batch = self._move(batch)
        latent = self._initial_latent(batch)
        batch_size = latent.size(0)
        future_dt = batch["future_delta_t"]
        dt_len = max(1, int(future_dt.size(1)))

        log_probs: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []
        values: List[torch.Tensor] = []
        rewards: List[torch.Tensor] = []
        actions: List[torch.Tensor] = []
        residual_before_rows: List[torch.Tensor] = []
        residual_after_rows: List[torch.Tensor] = []
        reward_terms: List[Dict[str, torch.Tensor]] = []

        for step in range(self.config.imagination_horizon):
            policy_state = latent.detach()
            logits = self.actor(policy_state)
            dist = Categorical(logits=logits)
            action_idx = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
            log_prob = dist.log_prob(action_idx)
            entropy = dist.entropy()
            value = self.critic(policy_state)
            action_vector = self.action_tensor[action_idx]
            dt = future_dt[:, step % dt_len, :]
            with torch.no_grad():
                predicted_latent = self.world_model.dynamics(latent, action_vector, dt)
                initial_decoded = self.world_model._decode_latent(predicted_latent)
                residual_before = self.restorer._mechanism_residual(initial_decoded)
                repaired_latent, repaired_decoded, restoration_diag = self.restorer.restore(
                    predicted_latent,
                    dt,
                )
                residual_after = torch.tensor(
                    restoration_diag["residual_after"],
                    dtype=torch.float32,
                    device=self.device,
                )
                reward, terms = self._step_reward(
                    repaired_decoded,
                    action_idx,
                    residual_before,
                )
            log_probs.append(log_prob)
            entropies.append(entropy)
            values.append(value)
            rewards.append(reward)
            actions.append(action_idx)
            residual_before_rows.append(residual_before.detach())
            residual_after_rows.append(residual_after.detach())
            reward_terms.append(terms)
            latent = repaired_latent.detach()

        return {
            "log_probs": torch.stack(log_probs, dim=0),
            "entropies": torch.stack(entropies, dim=0),
            "values": torch.stack(values, dim=0),
            "rewards": torch.stack(rewards, dim=0),
            "actions": torch.stack(actions, dim=0),
            "residual_before": torch.stack(residual_before_rows, dim=0),
            "residual_after": torch.stack(residual_after_rows, dim=0),
            "reward_terms": reward_terms,
            "final_latent": latent.detach(),
        }

    def train_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.actor.train()
        self.critic.train()
        rollout = self.imagined_rollout(batch, deterministic=False)
        rewards = rollout["rewards"]
        returns = self._discounted_returns(rewards)
        values = rollout["values"]
        log_probs = rollout["log_probs"]
        entropies = rollout["entropies"]

        advantage = returns - values.detach()
        actor_loss = -(log_probs * advantage.detach()).mean() - self.config.entropy_coef * entropies.mean()
        critic_loss = F.mse_loss(values, returns.detach())

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad(set_to_none=True)
        (self.config.value_coef * critic_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
        self.critic_optimizer.step()

        return {
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "mean_reward": float(rewards.mean().detach().cpu().item()),
            "mean_return": float(returns[0].mean().detach().cpu().item()),
            "entropy": float(entropies.mean().detach().cpu().item()),
            "mechanism_residual_before": float(rollout["residual_before"].mean().detach().cpu().item()),
            "mechanism_residual_after": float(rollout["residual_after"].mean().detach().cpu().item()),
        }

    @torch.no_grad()
    def evaluate_loader(self, loader: DataLoader) -> Dict[str, float]:
        self.actor.eval()
        self.critic.eval()
        rows: List[Dict[str, float]] = []
        for batch in loader:
            rollout = self.imagined_rollout(batch, deterministic=True)
            rewards = rollout["rewards"]
            returns = self._discounted_returns(rewards)
            rows.append(
                {
                    "mean_reward": float(rewards.mean().cpu().item()),
                    "mean_return": float(returns[0].mean().cpu().item()),
                    "mechanism_residual_before": float(rollout["residual_before"].mean().cpu().item()),
                    "mechanism_residual_after": float(rollout["residual_after"].mean().cpu().item()),
                }
            )
        if not rows:
            return {
                "mean_reward": 0.0,
                "mean_return": 0.0,
                "mechanism_residual_before": 0.0,
                "mechanism_residual_after": 0.0,
            }
        return {
            key: sum(row[key] for row in rows) / len(rows)
            for key in rows[0]
        }

    def checkpoint_payload(self) -> Dict[str, Any]:
        return {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "behavior_config": asdict(self.config),
            "world_model_config": asdict(self.world_model.config),
            "action_space": self.action_space,
            "world_model_frozen": True,
        }

    def save(self, path: Path, extra: Optional[Dict[str, Any]] = None) -> None:
        payload = self.checkpoint_payload()
        if extra:
            payload.update(extra)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)


def split_records(
    records: Sequence[Dict[str, Any]],
    validation_fraction: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows = list(records)
    rnd = random.Random(seed)
    rnd.shuffle(rows)
    if len(rows) <= 1:
        return rows, rows
    n_val = max(1, int(round(len(rows) * validation_fraction)))
    n_val = min(n_val, len(rows) - 1)
    return rows[n_val:], rows[:n_val]


def average_metrics(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {
        key: sum(float(row.get(key, 0.0)) for row in rows) / len(rows)
        for key in keys
    }


def write_history_csv(history: Sequence[Dict[str, Any]], path: Path) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    seen = set()
    for row in history:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def verify_world_model_frozen(model: LMAPEWorldModel) -> None:
    bad = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if bad:
        raise RuntimeError(f"World-model parameters are not frozen: {bad[:10]}")


def train_behavior_model(
    world_model_checkpoint: str,
    semantic_cache: str,
    output_dir: str,
    config: BehaviorTrainingConfig,
) -> Dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    world_model = load_world_model(world_model_checkpoint, device="cpu")
    records = load_semantic_records(semantic_cache)
    train_records, val_records = split_records(
        records,
        validation_fraction=config.validation_fraction,
        seed=config.seed,
    )

    train_dataset = WorldModelTensorDataset(
        train_records,
        semantic_dim=world_model.config.semantic_dim,
    )
    val_dataset = WorldModelTensorDataset(
        val_records,
        semantic_dim=world_model.config.semantic_dim,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
    )

    trainer = BehaviorModelTrainer(world_model, config)
    verify_world_model_frozen(trainer.world_model)

    history: List[Dict[str, Any]] = []
    best_return = -float("inf")
    best_path = output / "actor_critic_best.pt"
    last_path = output / "actor_critic_last.pt"

    for epoch in range(1, config.epochs + 1):
        batch_rows = [trainer.train_batch(batch) for batch in train_loader]
        train_metrics = average_metrics(batch_rows)
        val_metrics = trainer.evaluate_loader(val_loader)
        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        print(
            f"[Behavior] epoch={epoch}/{config.epochs} "
            f"train_return={train_metrics.get('mean_return', 0.0):.6f} "
            f"val_return={val_metrics.get('mean_return', 0.0):.6f} "
            f"residual={val_metrics.get('mechanism_residual_before', 0.0):.6f}"
            f"->{val_metrics.get('mechanism_residual_after', 0.0):.6f}"
        )
        current_return = float(val_metrics.get("mean_return", -float("inf")))
        if current_return > best_return:
            best_return = current_return
            trainer.save(
                best_path,
                extra={
                    "epoch": epoch,
                    "best_validation_return": best_return,
                    "world_model_checkpoint": str(world_model_checkpoint),
                    "semantic_cache": str(semantic_cache),
                },
            )

    trainer.save(
        last_path,
        extra={
            "epoch": config.epochs,
            "best_validation_return": best_return,
            "world_model_checkpoint": str(world_model_checkpoint),
            "semantic_cache": str(semantic_cache),
        },
    )
    save_json(history, output / "behavior_training_history.json")
    write_history_csv(history, output / "behavior_training_history.csv")
    summary = {
        "world_model_checkpoint": str(world_model_checkpoint),
        "semantic_cache": str(semantic_cache),
        "world_model_frozen": True,
        "num_records": len(records),
        "num_train_records": len(train_records),
        "num_validation_records": len(val_records),
        "num_actions": len(trainer.action_space),
        "best_validation_return": best_return,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "behavior_config": asdict(config),
    }
    save_json(summary, output / "behavior_training_summary.json")
    return summary


def evaluate_behavior_model(
    world_model_checkpoint: str,
    semantic_cache: str,
    behavior_checkpoint: str,
    output_dir: str,
    device: str = "auto",
    batch_size: int = 32,
) -> Dict[str, Any]:
    payload = torch.load(behavior_checkpoint, map_location="cpu")
    cfg = BehaviorTrainingConfig(**dict(payload.get("behavior_config") or {}))
    cfg.device = device
    cfg.batch_size = batch_size

    world_model = load_world_model(world_model_checkpoint, device="cpu")
    records = load_semantic_records(semantic_cache)
    dataset = WorldModelTensorDataset(
        records,
        semantic_dim=world_model.config.semantic_dim,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    trainer = BehaviorModelTrainer(
        world_model,
        cfg,
        action_space=list(payload.get("action_space") or build_joint_action_space()),
    )
    trainer.actor.load_state_dict(payload["actor_state_dict"])
    trainer.critic.load_state_dict(payload["critic_state_dict"])
    verify_world_model_frozen(trainer.world_model)
    metrics = trainer.evaluate_loader(loader)
    result = {
        "world_model_checkpoint": str(world_model_checkpoint),
        "behavior_checkpoint": str(behavior_checkpoint),
        "world_model_frozen": True,
        "num_records": len(records),
        "metrics": metrics,
    }
    save_json(result, Path(output_dir) / "behavior_evaluation.json")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "evaluate"], default="train")
    parser.add_argument("--world_model_checkpoint", default="world_model_artifacts/lma_pewm_best.pt")
    parser.add_argument("--semantic_cache", default="world_model_artifacts/semantic_world_model_records.jsonl")
    parser.add_argument("--behavior_checkpoint", default="behavior_model_artifacts/actor_critic_best.pt")
    parser.add_argument("--output_dir", default="behavior_model_artifacts")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--imagination_horizon", type=int, default=15)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--violation_penalty_coef", type=float, default=2.0)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--validation_fraction", type=float, default=0.1)
    parser.add_argument("--restore_max_iterations", type=int, default=3)
    parser.add_argument("--restore_damping", type=float, default=0.75)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "train":
        config = BehaviorTrainingConfig(
            gamma=args.gamma,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            entropy_coef=args.entropy_coef,
            value_coef=args.value_coef,
            violation_penalty_coef=args.violation_penalty_coef,
            hidden_dim=args.hidden_dim,
            imagination_horizon=args.imagination_horizon,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_grad_norm=args.max_grad_norm,
            seed=args.seed,
            device=args.device,
            validation_fraction=args.validation_fraction,
            restore_max_iterations=args.restore_max_iterations,
            restore_damping=args.restore_damping,
        )
        summary = train_behavior_model(
            world_model_checkpoint=args.world_model_checkpoint,
            semantic_cache=args.semantic_cache,
            output_dir=args.output_dir,
            config=config,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    result = evaluate_behavior_model(
        world_model_checkpoint=args.world_model_checkpoint,
        semantic_cache=args.semantic_cache,
        behavior_checkpoint=args.behavior_checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
