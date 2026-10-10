from __future__ import annotations

import copy
import json
import math
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from c2_prod_agent_adapter import ProductionAgentAdapter
from d2_energy_agent_adapter import EnergyAgentAdapter
from e2_couple_agent_adapter import CouplingAgentAdapter
from f_transition_engine import TransitionEngine
from g_lma_pewm import LMAPEWorldModel, LMAPEWorldModelConfig, StructuredSemanticVectorizer


PRODUCTION_STAGE_NAMES = ["BOF", "LF", "Casting", "Rolling", "Other"]
PRODUCTION_INPUT_DIM = 10
PRODUCTION_TARGET_DIM = 4
ENERGY_INPUT_DIM = 11
ENERGY_TARGET_DIM = 10
COUPLING_INPUT_DIM = 8
COUPLING_TARGET_DIM = 3
ACTION_DIM = 12


def safe_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def safe_list(x: Any) -> List[Any]:
    if isinstance(x, list):
        return x
    if x is None:
        return []
    return [x]


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, str) and not x.strip():
            return default
        y = float(x)
        if math.isnan(y) or math.isinf(y):
            return default
        return y
    except Exception:
        return default


def nested_get(obj: Any, paths: Sequence[Sequence[str]], default: float = 0.0) -> float:
    for path in paths:
        cur = obj
        ok = True
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                ok = False
                break
            cur = cur[key]
        if ok and cur is not None:
            return safe_float(cur, default)
    return default


def signed_log1p(x: Any) -> float:
    v = safe_float(x, 0.0)
    return math.copysign(math.log1p(abs(v)), v)


def canonical_stage(x: Any) -> str:
    s = str(x or "").strip().upper().replace(" ", "").replace("_", "")
    if "BOF" in s or "CONVERTER" in s or "LDG" in s:
        return "BOF"
    if s == "LF" or "LADLE" in s or "REFIN" in s:
        return "LF"
    if "CAST" in s or s == "CC" or s.startswith("CC"):
        return "Casting"
    if "ROLL" in s:
        return "Rolling"
    return "Other"


def stage_index(x: Any) -> int:
    stage = canonical_stage(x)
    return PRODUCTION_STAGE_NAMES.index(stage) if stage in PRODUCTION_STAGE_NAMES else 4


def stage_from_patch(patch: Dict[str, Any]) -> str:
    stage = patch.get("stage") or patch.get("process") or patch.get("operation")
    if stage:
        return canonical_stage(stage)
    stages = safe_list(patch.get("stages"))
    if stages:
        return canonical_stage(stages[0])
    active = safe_list(patch.get("active_heats"))
    for item in active:
        if isinstance(item, dict):
            s = item.get("stage") or item.get("process")
            if s:
                return canonical_stage(s)
    return "Other"


def duration_min(item: Dict[str, Any], default: float = 1.0) -> float:
    direct = item.get("duration_min") or item.get("dt") or item.get("delta_t")
    if direct is not None:
        return max(1e-6, safe_float(direct, default))
    t0 = item.get("t0_min") or item.get("start_min")
    t1 = item.get("t1_min") or item.get("end_min")
    if t0 is not None and t1 is not None:
        return max(1e-6, safe_float(t1, default) - safe_float(t0, 0.0))
    t = safe_dict(item.get("t"))
    t0 = t.get("t0") or t.get("s0") or t.get("start_min")
    t1 = t.get("t1") or t.get("s1") or t.get("end_min")
    if t0 is not None and t1 is not None:
        return max(1e-6, safe_float(t1, default) - safe_float(t0, 0.0))
    return max(1e-6, default)


def production_target_features(patch: Dict[str, Any]) -> Tuple[List[float], int]:
    patch = safe_dict(patch)
    active_heats = safe_list(patch.get("active_heats") or patch.get("heats") or patch.get("heat_ids"))
    rolling = safe_list(patch.get("active_rolling_units") or patch.get("rolling_units"))
    stages = safe_list(patch.get("stages"))
    if not stages:
        st = stage_from_patch(patch)
        stages = [] if st == "Other" else [st]
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
    converter = 1.0 if patch.get("converter") or patch.get("unit") else 0.0
    out = numeric + one_hot + [converter]
    return out[:PRODUCTION_INPUT_DIM]


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
    return [signed_log1p(x) for x in raw]


def energy_risk_count(patch: Dict[str, Any]) -> int:
    patch = safe_dict(patch)
    risks = patch.get("rk") or patch.get("risk_flags") or patch.get("risk_labels") or patch.get("key_risks")
    return len(safe_list(risks))


def energy_input_features(patch: Dict[str, Any]) -> List[float]:
    return energy_numeric_features(patch) + [signed_log1p(energy_risk_count(patch))]


def coupling_target_features(pair: Dict[str, Any]) -> Tuple[List[float], float]:
    pair = safe_dict(pair)
    activity = safe_dict(pair.get("production_operation_activity"))
    num_segments = safe_float(activity.get("num_segments"), 0.0)
    risk_count = len(safe_list(pair.get("risk_flags")))
    feasible = 1.0 if bool(pair.get("feasible", True)) else 0.0
    if pair.get("is_aligned") is not None:
        consistency = 1.0 if bool(pair.get("is_aligned")) else 0.0
    else:
        consistency = 1.0 if feasible and risk_count == 0 else (0.5 if feasible else 0.0)
    return [signed_log1p(num_segments), signed_log1p(risk_count), consistency], feasible


def coupling_input_features(prod_patch: Dict[str, Any], energy_patch: Dict[str, Any]) -> List[float]:
    p_num, _ = production_target_features(prod_patch)
    e = energy_numeric_features(energy_patch)
    generation_total = sum(math.expm1(max(0.0, v)) for v in e[:3])
    holder_mean = sum(e[3:6]) / 3.0 if len(e) >= 6 else 0.0
    load_total = e[6] + e[7] if len(e) >= 8 else 0.0
    purchase_total = e[8] + e[9] if len(e) >= 10 else 0.0
    return p_num + [signed_log1p(generation_total), holder_mean, load_total, purchase_total]


def action_features(action: Dict[str, Any]) -> List[float]:
    action = safe_dict(action)
    pa = safe_dict(action.get("production_action"))
    ea = safe_dict(action.get("energy_action"))
    ptype = str(pa.get("type") or "keep").strip().lower()
    ptypes = ["keep", "shift_heat", "shift_rolling_unit", "other"]
    pkey = ptype if ptype in ptypes[:-1] else "other"
    p_onehot = [1.0 if pkey == name else 0.0 for name in ptypes]
    delta_min = safe_float(pa.get("delta_min"), 0.0)
    p_delta = math.tanh(delta_min / 60.0)

    etype = str(ea.get("type") or "keep").strip().lower()
    etypes = ["keep", "increase_gas_use", "increase_boiler_load", "increase_power_purchase", "reduce_power_purchase", "other"]
    ekey = etype if etype in etypes[:-1] else "other"
    e_onehot = [1.0 if ekey == name else 0.0 for name in etypes]
    level = str(ea.get("delta_level") or "none").strip().lower()
    level_value = {"none": 0.0, "low": 0.33, "medium": 0.67, "high": 1.0}.get(level, 0.0)
    out = p_onehot + [p_delta] + e_onehot + [level_value]
    return out[:ACTION_DIM]


def default_joint_action() -> Dict[str, Any]:
    return {
        "action_id": "observed_noop",
        "production_action": {"type": "keep", "target": "production_schedule", "delta_min": 0},
        "energy_action": {"type": "keep", "target": "energy_system", "delta_level": "none"},
        "source": "world_model_training_default",
    }


def action_magnitude(action: Dict[str, Any]) -> float:
    pa = safe_dict(safe_dict(action).get("production_action"))
    ea = safe_dict(safe_dict(action).get("energy_action"))
    mag = abs(safe_float(pa.get("delta_min"), 0.0)) / 15.0
    level = str(ea.get("delta_level") or "none").lower()
    mag += {"none": 0.0, "low": 0.5, "medium": 1.0, "high": 1.5}.get(level, 0.5)
    if str(pa.get("type") or "keep") != "keep":
        mag += 0.5
    if str(ea.get("type") or "keep") != "keep":
        mag += 0.5
    return float(mag)


def derive_reward_target(energy_patch: Dict[str, Any], pair: Dict[str, Any], action: Dict[str, Any]) -> float:
    e = energy_numeric_features(energy_patch)
    risk_count = energy_risk_count(energy_patch)
    _, feasible = coupling_target_features(pair)
    consistency = coupling_target_features(pair)[0][2]
    purchase_log = 0.0
    if len(e) >= 10:
        purchase_log = max(0.0, e[8]) + max(0.0, e[9])
    return float(
        2.0 * consistency
        + 1.0 * feasible
        - 1.5 * min(float(risk_count), 10.0)
        - 0.05 * purchase_log
        - 0.02 * action_magnitude(action)
    )


def _pad_sequence(seq: List[Any], length: int, pad: Any) -> List[Any]:
    seq = list(seq[-length:])
    if len(seq) < length:
        seq = [copy.deepcopy(pad) for _ in range(length - len(seq))] + seq
    return seq


def build_training_record_from_joint_sample(
    joint_sample: Dict[str, Any],
    candidate_action: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    joint_sample = safe_dict(joint_sample)
    comp = safe_dict(joint_sample.get("component_samples"))
    prod_sample = safe_dict(comp.get("production"))
    energy_sample = safe_dict(comp.get("energy"))
    prod_input = safe_dict(prod_sample.get("input_state"))
    energy_input = safe_dict(energy_sample.get("input_state"))
    target = safe_dict(joint_sample.get("target_future"))
    prod_target = safe_dict(prod_sample.get("target_future"))
    energy_target = safe_dict(energy_sample.get("target_future"))

    prod_history = safe_list(prod_input.get("history_patches"))
    energy_history = safe_list(energy_input.get("energy_history"))
    prod_future = safe_list(target.get("production_future") or prod_target.get("future_patches"))
    energy_future = safe_list(target.get("energy_future") or energy_target.get("energy_future"))

    functional = safe_dict(target.get("functional_coupling_target") or target.get("coupling_target") or joint_sample.get("coupling_target"))
    coupling_future = safe_list(functional.get("future_pairs"))
    if not coupling_future:
        coupling_future = [{} for _ in range(max(len(prod_future), len(energy_future)))]

    h = int(joint_sample.get("history_window") or max(len(prod_history), len(energy_history), 1))
    k = int(joint_sample.get("horizon") or max(len(prod_future), len(energy_future), 1))
    prod_history = _pad_sequence(prod_history, h, {})
    energy_history = _pad_sequence(energy_history, h, {})
    prod_future = _pad_sequence(prod_future, k, {})[-k:]
    energy_future = _pad_sequence(energy_future, k, {})[-k:]
    coupling_future = _pad_sequence(coupling_future, k, {})[-k:]

    coupling_history = [
        {"production": prod_history[i], "energy": energy_history[i]}
        for i in range(h)
    ]

    history_dt = []
    for i in range(h):
        dt = duration_min(prod_history[i], duration_min(energy_history[i], 1.0))
        history_dt.append(dt)
    future_dt = []
    for i in range(k):
        dt = duration_min(prod_future[i], duration_min(energy_future[i], 1.0))
        future_dt.append(dt)

    action = safe_dict(candidate_action or joint_sample.get("candidate_action") or safe_dict(joint_sample.get("input_state")).get("candidate_action"))
    if not action:
        action = default_joint_action()

    return {
        "sample_id": joint_sample.get("sample_id"),
        "day_id": joint_sample.get("day_id"),
        "anchor_patch_idx": joint_sample.get("anchor_patch_idx"),
        "history_window": h,
        "horizon": k,
        "production_history": prod_history,
        "energy_history": energy_history,
        "coupling_history": coupling_history,
        "production_future": prod_future,
        "energy_future": energy_future,
        "coupling_future": coupling_future,
        "history_delta_t": history_dt,
        "future_delta_t": future_dt,
        "candidate_action": action,
    }


class FixedLLMSemanticExtractor:
    def __init__(
        self,
        prod_model_path: str,
        energy_model_path: str,
        couple_model_path: str,
        device: str = "auto",
        prod_tokens: int = 8192,
        energy_tokens: int = 8192,
        couple_tokens: int = 8192,
        auto_release: bool = True,
    ) -> None:
        self.engine = TransitionEngine(
            prod_agent_factory=lambda: ProductionAgentAdapter(
                model_path=prod_model_path,
                device=device,
                max_new_tokens=prod_tokens,
            ),
            energy_agent_factory=lambda: EnergyAgentAdapter(
                model_path=energy_model_path,
                device=device,
                max_new_tokens=energy_tokens,
            ),
            couple_agent_factory=lambda: CouplingAgentAdapter(
                model_path=couple_model_path,
                device=device,
                max_new_tokens=couple_tokens,
            ),
            auto_release=auto_release,
            verbose=False,
        )

    def extract(self, joint_sample: Dict[str, Any]) -> Dict[str, Any]:
        result = self.engine.predict_joint_horizon(joint_sample)
        return {
            "production": safe_dict(getattr(result, "production_prediction", {})),
            "energy": safe_dict(getattr(result, "energy_prediction", {})),
            "coupling": safe_dict(getattr(result, "coupling_prediction", {})),
            "production_metrics": safe_dict(getattr(result, "production_metrics", {})),
            "energy_metrics": safe_dict(getattr(result, "energy_metrics", {})),
            "coupling_metrics": safe_dict(getattr(result, "coupling_metrics", {})),
            "joint_metrics": safe_dict(getattr(result, "joint_metrics", {})),
        }


@dataclass
class SemanticCacheConfig:
    history_window: int = 5
    horizon: int = 5
    stride: int = 1
    max_samples_per_day: Optional[int] = None
    use_llm_semantics: bool = True
    stop_on_error: bool = False


def prepare_semantic_records(
    assembler: Any,
    day_ids: Sequence[str],
    cache_path: str,
    semantic_extractor: Optional[FixedLLMSemanticExtractor],
    config: SemanticCacheConfig,
) -> List[Dict[str, Any]]:
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    with path.open("w", encoding="utf-8") as f:
        for day_id in day_ids:
            samples = assembler.build_joint_horizon_dataset(
                day_id=day_id,
                horizon=config.horizon,
                history_window=config.history_window,
                stride=config.stride,
                max_samples=config.max_samples_per_day,
            )
            for sample in samples:
                try:
                    action = safe_dict(sample.get("candidate_action")) or default_joint_action()
                    sample = copy.deepcopy(sample)
                    sample["candidate_action"] = action
                    input_state = sample.setdefault("input_state", {})
                    if isinstance(input_state, dict):
                        input_state["candidate_action"] = action
                    comp = safe_dict(sample.get("component_samples"))
                    for key in ["production", "energy"]:
                        component = safe_dict(comp.get(key))
                        cin = component.setdefault("input_state", {})
                        if isinstance(cin, dict):
                            cin["candidate_action"] = action
                        if component:
                            comp[key] = component
                    sample["component_samples"] = comp

                    training_record = (
                        assembler.build_world_model_training_record(sample, candidate_action=action)
                        if hasattr(assembler, "build_world_model_training_record")
                        else build_training_record_from_joint_sample(sample, candidate_action=action)
                    )

                    if config.use_llm_semantics:
                        if semantic_extractor is None:
                            raise ValueError("semantic_extractor is required when use_llm_semantics=True")
                        semantics = semantic_extractor.extract(sample)
                    else:
                        semantics = {
                            "production": safe_dict(safe_dict(comp.get("production")).get("input_state")),
                            "energy": safe_dict(safe_dict(comp.get("energy")).get("input_state")),
                            "coupling": safe_dict(sample.get("input_state")),
                        }

                    record = {
                        "sample_id": sample.get("sample_id"),
                        "day_id": day_id,
                        "training_record": training_record,
                        "llm_semantics": semantics,
                    }
                    records.append(record)
                    f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                    f.flush()
                except Exception as exc:
                    error = {
                        "day_id": day_id,
                        "sample_id": safe_dict(sample).get("sample_id"),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                    errors.append(error)
                    if config.stop_on_error:
                        raise

    if errors:
        error_path = path.with_suffix(path.suffix + ".errors.json")
        error_path.write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
    if not records:
        raise RuntimeError("No semantic world-model records were generated")
    return records


def load_semantic_records(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                records.append(obj)
    if not records:
        raise ValueError(f"No records found in semantic cache: {path}")
    return records


class WorldModelTensorDataset(Dataset):
    def __init__(
        self,
        records: Sequence[Dict[str, Any]],
        semantic_dim: int = 64,
        history_window: Optional[int] = None,
        horizon: Optional[int] = None,
    ) -> None:
        self.records = list(records)
        self.vectorizer = StructuredSemanticVectorizer(dim=semantic_dim)
        self.history_window = history_window
        self.horizon = horizon

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        rec = safe_dict(self.records[index])
        tr = safe_dict(rec.get("training_record"))
        sem = safe_dict(rec.get("llm_semantics"))
        h = int(self.history_window or tr.get("history_window") or 5)
        k = int(self.horizon or tr.get("horizon") or 5)

        ph = _pad_sequence(safe_list(tr.get("production_history")), h, {})
        eh = _pad_sequence(safe_list(tr.get("energy_history")), h, {})
        ch = _pad_sequence(safe_list(tr.get("coupling_history")), h, {})
        pf = _pad_sequence(safe_list(tr.get("production_future")), k, {})[-k:]
        ef = _pad_sequence(safe_list(tr.get("energy_future")), k, {})[-k:]
        cf = _pad_sequence(safe_list(tr.get("coupling_future")), k, {})[-k:]

        hist_dt = list(tr.get("history_delta_t") or [])
        fut_dt = list(tr.get("future_delta_t") or [])
        hist_dt = _pad_sequence(hist_dt, h, 1.0)
        fut_dt = _pad_sequence(fut_dt, k, 1.0)[-k:]

        action = safe_dict(tr.get("candidate_action")) or default_joint_action()
        action_vec = action_features(action)
        future_actions = [action_vec for _ in range(k)]

        prod_target = []
        prod_stage = []
        energy_target = []
        energy_risk = []
        coupling_target = []
        feasibility = []
        reward_target = []
        for i in range(k):
            pnum, pst = production_target_features(pf[i])
            enum = energy_numeric_features(ef[i])
            cnum, feas = coupling_target_features(cf[i])
            prod_target.append(pnum)
            prod_stage.append(pst)
            energy_target.append(enum)
            energy_risk.append(1.0 if energy_risk_count(ef[i]) > 0 else 0.0)
            coupling_target.append(cnum)
            feasibility.append(feas)
            reward_target.append(derive_reward_target(ef[i], cf[i], action))

        return {
            "production_history": torch.tensor([production_input_features(x) for x in ph], dtype=torch.float32),
            "energy_history": torch.tensor([energy_input_features(x) for x in eh], dtype=torch.float32),
            "coupling_history": torch.tensor([
                coupling_input_features(
                    safe_dict(x.get("production")) if isinstance(x, dict) else {},
                    safe_dict(x.get("energy")) if isinstance(x, dict) else {},
                ) for x in ch
            ], dtype=torch.float32),
            "history_delta_t": torch.tensor(hist_dt, dtype=torch.float32).unsqueeze(-1),
            "production_semantic": torch.tensor(self.vectorizer.vectorize(sem.get("production")), dtype=torch.float32),
            "energy_semantic": torch.tensor(self.vectorizer.vectorize(sem.get("energy")), dtype=torch.float32),
            "coupling_semantic": torch.tensor(self.vectorizer.vectorize(sem.get("coupling")), dtype=torch.float32),
            "future_actions": torch.tensor(future_actions, dtype=torch.float32),
            "future_delta_t": torch.tensor(fut_dt, dtype=torch.float32).unsqueeze(-1),
            "target_production_numeric": torch.tensor(prod_target, dtype=torch.float32),
            "target_production_stage": torch.tensor(prod_stage, dtype=torch.long),
            "target_energy_numeric": torch.tensor(energy_target, dtype=torch.float32),
            "target_energy_risk": torch.tensor(energy_risk, dtype=torch.float32),
            "target_coupling_numeric": torch.tensor(coupling_target, dtype=torch.float32),
            "target_feasibility": torch.tensor(feasibility, dtype=torch.float32),
            "target_reward": torch.tensor(reward_target, dtype=torch.float32),
        }


@dataclass
class WorldModelLossWeights:
    production_numeric: float = 1.0
    production_stage: float = 1.0
    energy_numeric: float = 1.0
    energy_risk: float = 0.5
    coupling_numeric: float = 1.0
    feasibility: float = 0.5
    reward: float = 0.25
    mechanism: float = 0.1


class WorldModelLoss(nn.Module):
    def __init__(self, weights: Optional[WorldModelLossWeights] = None) -> None:
        super().__init__()
        self.weights = weights or WorldModelLossWeights()

    def mechanism_loss(self, pred: Dict[str, torch.Tensor]) -> torch.Tensor:
        p = pred["production_numeric"]
        e = pred["energy_numeric"]
        c = pred["coupling_numeric"]
        penalty = torch.relu(-p).pow(2).mean()
        penalty = penalty + torch.relu(-e).pow(2).mean()
        penalty = penalty + torch.relu(-c[..., :2]).pow(2).mean()
        if e.size(-1) >= 10:
            steam_load = e[..., 6]
            power_load = e[..., 7]
            steam_purchase = e[..., 8]
            power_purchase = e[..., 9]
            penalty = penalty + 0.25 * torch.relu(steam_purchase - steam_load).pow(2).mean()
            penalty = penalty + 0.25 * torch.relu(power_purchase - power_load).pow(2).mean()
        if e.size(1) > 1 and e.size(-1) >= 6:
            holders = e[..., 3:6]
            holder_jump = holders[:, 1:, :] - holders[:, :-1, :]
            penalty = penalty + 0.01 * holder_jump.pow(2).mean()
        return penalty

    def forward(self, pred: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        losses: Dict[str, torch.Tensor] = {}
        losses["production_numeric"] = F.mse_loss(pred["production_numeric"], batch["target_production_numeric"])
        losses["production_stage"] = F.cross_entropy(
            pred["production_stage_logits"].reshape(-1, pred["production_stage_logits"].size(-1)),
            batch["target_production_stage"].reshape(-1),
        )
        losses["energy_numeric"] = F.mse_loss(pred["energy_numeric"], batch["target_energy_numeric"])
        losses["energy_risk"] = F.binary_cross_entropy_with_logits(pred["energy_risk_logit"], batch["target_energy_risk"])
        losses["coupling_numeric"] = F.mse_loss(pred["coupling_numeric"], batch["target_coupling_numeric"])
        losses["feasibility"] = F.binary_cross_entropy_with_logits(pred["feasibility_logit"], batch["target_feasibility"])
        losses["reward"] = F.mse_loss(pred["reward"], batch["target_reward"])
        losses["mechanism"] = self.mechanism_loss(pred)

        w = self.weights
        total = (
            w.production_numeric * losses["production_numeric"]
            + w.production_stage * losses["production_stage"]
            + w.energy_numeric * losses["energy_numeric"]
            + w.energy_risk * losses["energy_risk"]
            + w.coupling_numeric * losses["coupling_numeric"]
            + w.feasibility * losses["feasibility"]
            + w.reward * losses["reward"]
            + w.mechanism * losses["mechanism"]
        )
        losses["total"] = total
        return total, losses


@dataclass
class WorldModelTrainingConfig:
    history_window: int = 5
    horizon: int = 5
    semantic_dim: int = 64
    hidden_dim: int = 128
    latent_dim: int = 192
    num_attention_heads: int = 4
    dropout: float = 0.1
    batch_size: int = 16
    epochs: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 5.0
    validation_ratio: float = 0.15
    seed: int = 42
    device: str = "auto"
    num_workers: int = 0
    use_amp: bool = True
    patience: int = 10
    checkpoint_dir: str = "world_model_artifacts"
    loss_weights: WorldModelLossWeights = field(default_factory=WorldModelLossWeights)


def resolve_device(device: str) -> torch.device:
    d = str(device or "auto").lower()
    if d == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(d)


def build_model_config(cfg: WorldModelTrainingConfig) -> LMAPEWorldModelConfig:
    return LMAPEWorldModelConfig(
        production_input_dim=PRODUCTION_INPUT_DIM,
        energy_input_dim=ENERGY_INPUT_DIM,
        coupling_input_dim=COUPLING_INPUT_DIM,
        semantic_dim=cfg.semantic_dim,
        action_dim=ACTION_DIM,
        production_target_dim=PRODUCTION_TARGET_DIM,
        energy_target_dim=ENERGY_TARGET_DIM,
        coupling_target_dim=COUPLING_TARGET_DIM,
        production_stage_classes=len(PRODUCTION_STAGE_NAMES),
        hidden_dim=cfg.hidden_dim,
        latent_dim=cfg.latent_dim,
        num_attention_heads=cfg.num_attention_heads,
        dropout=cfg.dropout,
    )


def split_records(records: Sequence[Dict[str, Any]], validation_ratio: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    items = list(records)
    rng = random.Random(seed)
    rng.shuffle(items)
    if len(items) <= 1 or validation_ratio <= 0.0:
        return items, []
    n_val = max(1, int(round(len(items) * validation_ratio)))
    n_val = min(n_val, len(items) - 1)
    return items[n_val:], items[:n_val]


class WorldModelTrainer:
    def __init__(
        self,
        model: LMAPEWorldModel,
        config: Optional[WorldModelTrainingConfig] = None,
    ) -> None:
        self.config = config or WorldModelTrainingConfig()
        self.device = resolve_device(self.config.device)
        self.model = model.to(self.device)
        self.loss_fn = WorldModelLoss(self.config.loss_weights).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.use_amp = bool(self.config.use_amp and self.device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.history: List[Dict[str, Any]] = []
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def _move_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    @staticmethod
    def _model_inputs(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        keys = [
            "production_history",
            "energy_history",
            "coupling_history",
            "history_delta_t",
            "production_semantic",
            "energy_semantic",
            "coupling_semantic",
            "future_actions",
            "future_delta_t",
        ]
        return {k: batch[k] for k in keys}

    def _run_loader(self, loader: DataLoader, training: bool) -> Dict[str, float]:
        if training:
            self.model.train()
        else:
            self.model.eval()
        totals: Dict[str, float] = {}
        count = 0

        for batch in loader:
            batch = self._move_batch(batch)
            if training:
                self.optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    pred = self.model(**self._model_inputs(batch))
                    loss, parts = self.loss_fn(pred, batch)
                if training:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            bs = int(batch["production_history"].size(0))
            count += bs
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu().item()) * bs

        if count == 0:
            return {"total": float("nan")}
        return {key: value / count for key, value in totals.items()}

    def _save_checkpoint(self, path: Path, epoch: int, best_val: float) -> None:
        payload = {
            "model_config": asdict(self.model.config),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_config": {
                **asdict(self.config),
                "loss_weights": asdict(self.config.loss_weights),
            },
            "feature_spec": {
                "production_stage_names": PRODUCTION_STAGE_NAMES,
                "production_input_dim": PRODUCTION_INPUT_DIM,
                "energy_input_dim": ENERGY_INPUT_DIM,
                "coupling_input_dim": COUPLING_INPUT_DIM,
                "action_dim": ACTION_DIM,
            },
            "epoch": int(epoch),
            "best_validation_loss": float(best_val),
            "training_history": self.history,
        }
        torch.save(payload, path)

    def fit(self, records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

        train_records, val_records = split_records(records, self.config.validation_ratio, self.config.seed)
        train_ds = WorldModelTensorDataset(
            train_records,
            semantic_dim=self.config.semantic_dim,
            history_window=self.config.history_window,
            horizon=self.config.horizon,
        )
        val_ds = WorldModelTensorDataset(
            val_records,
            semantic_dim=self.config.semantic_dim,
            history_window=self.config.history_window,
            horizon=self.config.horizon,
        ) if val_records else None

        train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
        ) if val_ds is not None else None

        best_val = float("inf")
        best_epoch = 0
        patience_left = int(self.config.patience)
        checkpoint_dir = Path(self.config.checkpoint_dir)
        best_path = checkpoint_dir / "lma_pewm_best.pt"
        last_path = checkpoint_dir / "lma_pewm_last.pt"

        for epoch in range(1, int(self.config.epochs) + 1):
            train_metrics = self._run_loader(train_loader, training=True)
            val_metrics = self._run_loader(val_loader, training=False) if val_loader is not None else dict(train_metrics)
            val_total = float(val_metrics.get("total", train_metrics.get("total", float("inf"))))
            row = {
                "epoch": epoch,
                "train": train_metrics,
                "validation": val_metrics,
            }
            self.history.append(row)
            self._save_checkpoint(last_path, epoch, min(best_val, val_total))

            if val_total < best_val:
                best_val = val_total
                best_epoch = epoch
                patience_left = int(self.config.patience)
                self._save_checkpoint(best_path, epoch, best_val)
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        history_path = checkpoint_dir / "world_model_training_history.json"
        history_path.write_text(json.dumps(self.history, ensure_ascii=False, indent=2), encoding="utf-8")
        summary = {
            "num_records": len(records),
            "num_train_records": len(train_records),
            "num_validation_records": len(val_records),
            "best_epoch": best_epoch,
            "best_validation_loss": best_val,
            "best_checkpoint": str(best_path),
            "last_checkpoint": str(last_path),
            "history_path": str(history_path),
        }
        (checkpoint_dir / "world_model_training_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return summary


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: str,
    records: Sequence[Dict[str, Any]],
    device: str = "auto",
    batch_size: int = 16,
) -> Dict[str, float]:
    dev = resolve_device(device)
    payload = torch.load(checkpoint_path, map_location=dev)
    model_cfg = LMAPEWorldModelConfig(**safe_dict(payload.get("model_config")))
    model = LMAPEWorldModel(model_cfg).to(dev)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    training_cfg = safe_dict(payload.get("training_config"))
    loss_weights = WorldModelLossWeights(**safe_dict(training_cfg.get("loss_weights"))) if training_cfg.get("loss_weights") else WorldModelLossWeights()
    loss_fn = WorldModelLoss(loss_weights).to(dev)
    ds = WorldModelTensorDataset(
        records,
        semantic_dim=model_cfg.semantic_dim,
        history_window=int(training_cfg.get("history_window", 5)),
        horizon=int(training_cfg.get("horizon", 5)),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    totals: Dict[str, float] = {}
    count = 0
    for batch in loader:
        batch = {k: v.to(dev) for k, v in batch.items()}
        pred = model(**WorldModelTrainer._model_inputs(batch))
        _, parts = loss_fn(pred, batch)
        bs = int(batch["production_history"].size(0))
        count += bs
        for key, value in parts.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu().item()) * bs
    return {key: value / max(count, 1) for key, value in totals.items()}
