from __future__ import annotations

import csv
import gc
import json
import math
import traceback
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, Iterable, List, Optional

try:
    import torch
except Exception:
    torch = None

@dataclass
class CandidatePlan:
    plan_id: str
    day_id: str
    heat_id: Optional[str] = None
    task: str = "production_next_state_prediction"
    mode: str = "production_state_predictor"
    candidate_actions: List[Dict[str, Any]] = field(default_factory=list)
    candidate_summary: Dict[str, Any] = field(default_factory=dict)
    production_statistics: Dict[str, Any] = field(default_factory=dict)
    plan_snapshot: Dict[str, Any] = field(default_factory=dict)
    raw_output: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class EnergyTrajectory:
    trajectory_id: str
    plan_id: str
    day_id: str
    heat_id: Optional[str] = None
    patch_id: Optional[str] = None
    task: str = "energy_next_state_prediction_day"
    mode: str = "energy_state_predictor_batch_day"
    optimized_patch_summary: Dict[str, Any] = field(default_factory=dict)
    optimized_patch_minute_trajectory: List[Dict[str, Any]] = field(default_factory=list)
    patch_summaries: List[Dict[str, Any]] = field(default_factory=list)
    num_patches: Optional[int] = None
    num_feasible_patches: Optional[int] = None
    feasible_patch_ratio: Optional[float] = None
    energy_state_snapshot: Dict[str, Any] = field(default_factory=dict)
    raw_output: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class EvaluationResult:
    plan_id: str
    day_id: str
    heat_id: Optional[str] = None
    task: str = "joint_state_consistency_prediction"
    mode: str = "joint_state_consistency_evaluator"
    joint_feasibility: bool = True
    economic_score: float = 0.0
    risk_score: float = 0.0
    consistency_score: float = 0.0
    coupling_risks: List[str] = field(default_factory=list)
    major_conflicts: List[str] = field(default_factory=list)
    violation_list: List[str] = field(default_factory=list)
    failure_reasons: List[str] = field(default_factory=list)
    recommended_joint_decisions: List[str] = field(default_factory=list)
    revision_suggestions: List[str] = field(default_factory=list)
    next_state_summary: Dict[str, Any] = field(default_factory=dict)
    reward: Optional[float] = None
    violation_cost: Optional[float] = None
    risk_cost: Optional[float] = None
    raw_output: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class JointHorizonResult:
    sample_id: str
    day_id: str
    anchor_patch_idx: int
    history_window: int
    horizon: int
    production_prediction: Dict[str, Any]
    production_metrics: Dict[str, Any]
    energy_prediction: Dict[str, Any]
    energy_metrics: Dict[str, Any]
    coupling_prediction: Dict[str, Any]
    coupling_metrics: Dict[str, Any]
    joint_metrics: Dict[str, Any]
    sample_meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

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

def safe_optional_float(x: Any) -> Optional[float]:

    try:
        if x is None:
            return None
        if isinstance(x, str) and not x.strip():
            return None
        y = float(x)
        if math.isnan(y) or math.isinf(y):
            return None
        return y
    except Exception:
        return None

def is_valid_number(x: Any) -> bool:
    if isinstance(x, bool):
        return True
    if not isinstance(x, (int, float)):
        return False
    try:
        return not (math.isnan(float(x)) or math.isinf(float(x)))
    except Exception:
        return False

def json_safe(obj: Any) -> Any:
    try:
        import numpy as np
    except Exception:
        np = None
    try:
        import pandas as pd
    except Exception:
        pd = None

    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]
    if isinstance(obj, set):
        return [json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if np is not None and isinstance(obj, (np.integer,)):
        return int(obj)
    if np is not None and isinstance(obj, (np.floating,)):
        return float(obj)
    if np is not None and isinstance(obj, np.ndarray):
        return obj.tolist()
    if pd is not None and hasattr(pd, "Timestamp") and isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj

def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(obj), f, ensure_ascii=False, indent=2)

def _flatten_numeric_metrics(prefix: str, metrics: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in safe_dict(metrics).items():
        if isinstance(v, bool):
            out[f"{prefix}{k}"] = 1.0 if v else 0.0
        elif isinstance(v, (int, float)):
            out[f"{prefix}{k}"] = safe_float(v)
    return out

_RUNTIME_BANNED_FIELDS = {
    "_raw_text", "raw_text", "raw_output", "_raw_list_output",
    "model_input", "prompt", "full_prompt", "messages",
    "comment", "comments", "traceback", "debug", "debug_info",
    "exception", "stack", "stacktrace", "reasoning", "analysis",
}

_OPERATION_SEGMENT_FIELDS = {
    "predicted_operation_segments",
    "model_predicted_operation_segments",
    "derived_operation_segments_from_patches",
    "operation_segments",
    "patch_operation_segments",
    "production_operation_segments",
    "production_patch_operation_segments",
    "history_operation_segments",
    "target_operation_segments",
}

def strip_runtime_fields_keep_operation_segments(obj: Any) -> Any:

    if isinstance(obj, dict):
        return {
            str(k): strip_runtime_fields_keep_operation_segments(v)
            for k, v in obj.items()
            if k not in _RUNTIME_BANNED_FIELDS
        }
    if isinstance(obj, list):
        return [strip_runtime_fields_keep_operation_segments(x) for x in obj]
    return obj

def strip_for_coupling(obj: Any) -> Any:

    banned = set(_RUNTIME_BANNED_FIELDS) | set(_OPERATION_SEGMENT_FIELDS)
    if isinstance(obj, dict):
        return {
            str(k): strip_for_coupling(v)
            for k, v in obj.items()
            if k not in banned
        }
    if isinstance(obj, list):
        return [strip_for_coupling(x) for x in obj]
    return obj

def _operation_segment_source(prediction: Dict[str, Any]) -> str:
    pred = safe_dict(prediction)
    source = pred.get("operation_segment_source")
    if not source:
        source = safe_dict(pred.get("repair_meta")).get("operation_segment_source")
    if not source:
        source = safe_dict(pred.get("operation_segment_diagnostics")).get("operation_segment_source")
    if not source:
        if isinstance(pred.get("predicted_operation_segments"), list) and pred.get("predicted_operation_segments"):
            source = "model_output_unspecified"
        elif isinstance(pred.get("derived_operation_segments_from_patches"), list) and pred.get("derived_operation_segments_from_patches"):
            source = "derived_from_future_production_states"
        else:
            source = "missing_model_operation_segments"
    return str(source)

def _is_model_operation_segment_source(source: str) -> bool:

    s = str(source or "").strip().lower()
    if not s:
        return False
    invalid_tokens = ["missing", "derived", "fallback", "calendar", "target", "patch"]
    if any(tok in s for tok in invalid_tokens):
        return False
    return ("model" in s) or ("llm" in s) or ("predicted" in s) or (s == "model_output_unspecified")

def _segment_source(seg: Dict[str, Any]) -> str:
    s = (
        seg.get("source")
        or seg.get("source_kind")
        or seg.get("origin")
        or seg.get("created_by")
        or ""
    )
    return str(s).strip()

def _is_valid_model_operation_segment(seg: Dict[str, Any], prediction_source: str) -> bool:

    if not isinstance(seg, dict):
        return False

    valid_flag = seg.get("is_valid_gantt_prediction")
    if valid_flag is False:
        return False

    s = _segment_source(seg)
    if s:
        sl = s.lower()
        invalid_tokens = ["missing", "derived", "fallback", "calendar", "target", "patch"]
        if any(tok in sl for tok in invalid_tokens):
            return False
        if ("model" in sl) or ("llm" in sl) or ("predicted" in sl):
            return True
        return _is_model_operation_segment_source(prediction_source)

    return _is_model_operation_segment_source(prediction_source)

def _model_predicted_operation_segments(prediction: Dict[str, Any]) -> List[Dict[str, Any]]:
    pred = safe_dict(prediction)
    segs = pred.get("predicted_operation_segments")
    if not isinstance(segs, list):
        return []
    source = _operation_segment_source(pred)
    return [
        x for x in segs
        if isinstance(x, dict) and _is_valid_model_operation_segment(x, source)
    ]

def _raw_predicted_operation_segments(prediction: Dict[str, Any]) -> List[Dict[str, Any]]:
    pred = safe_dict(prediction)
    segs = pred.get("predicted_operation_segments")
    return [x for x in segs if isinstance(x, dict)] if isinstance(segs, list) else []

def _derived_operation_segments(prediction: Dict[str, Any]) -> List[Dict[str, Any]]:
    pred = safe_dict(prediction)
    out: List[Dict[str, Any]] = []
    for key in [
        "derived_operation_segments_from_patches",
        "fallback_operation_segments",
        "patch_operation_segments",
    ]:
        segs = pred.get(key)
        if isinstance(segs, list):
            out.extend([x for x in segs if isinstance(x, dict)])
    return out

def _build_operation_segment_compact_context(
    prediction: Dict[str, Any],
    max_examples: int = 24,
) -> Dict[str, Any]:

    pred = safe_dict(prediction)
    diag = _operation_segment_diagnostics(pred)
    model_segs = _model_predicted_operation_segments(pred)

    heat_ids: List[str] = []
    stage_counts: Dict[str, int] = {}
    unit_counts: Dict[str, int] = {}
    starts: List[float] = []
    ends: List[float] = []
    examples: List[Dict[str, Any]] = []

    for seg in model_segs:
        hid = str(seg.get("heat_id") or seg.get("id") or seg.get("hn") or "").strip()
        if hid and hid not in heat_ids:
            heat_ids.append(hid)

        stage = str(seg.get("stage") or seg.get("process") or seg.get("operation") or "unknown")
        stage_counts[stage] = stage_counts.get(stage, 0) + 1

        unit = str(seg.get("unit") or seg.get("resource") or seg.get("lane") or "unknown")
        unit_counts[unit] = unit_counts.get(unit, 0) + 1

        s = safe_optional_float(seg.get("start_min"))
        e = safe_optional_float(seg.get("end_min"))
        if s is not None:
            starts.append(s)
        if e is not None:
            ends.append(e)

        if len(examples) < max_examples:
            examples.append({
                "heat_id": hid,
                "stage": stage,
                "unit": unit,
                "start_min": s,
                "end_min": e,
                "duration_min": safe_optional_float(seg.get("duration_min")),
                "source_steps": safe_list(seg.get("source_steps")),
            })

    num_segments = len(model_segs)
    available = bool(diag["operation_segment_prediction_available"] and num_segments > 0)
    repair_meta = safe_dict(pred.get("repair_meta"))
    projection_diag = (
        safe_dict(pred.get("mechanism_projection_diagnostics"))
        or safe_dict(pred.get("mechanism_repair_diagnostics"))
        or safe_dict(repair_meta.get("mechanism_projection_diagnostics"))
    )
    return {
        "task": "production_operation_gantt_prediction_context",
        "mode": "compact_operation_segment_summary",
        "available": available,
        "source": diag["operation_segment_source"],
        "segment_source": diag["operation_segment_source"],
        "trust_level": "model_output" if available else "missing_or_diagnostic_only",
        "operation_segment_source": diag["operation_segment_source"],
        "operation_segment_prediction_available": diag["operation_segment_prediction_available"],
        "operation_segment_missing": diag["operation_segment_missing"],
        "num_segments": num_segments,
        "valid_model_segment_count": num_segments,
        "num_model_predicted_operation_segments": diag["num_model_predicted_operation_segments"],
        "num_raw_predicted_operation_segments": diag["num_raw_predicted_operation_segments"],
        "num_rejected_raw_operation_segments": diag["num_rejected_raw_operation_segments"],
        "num_derived_operation_segments": diag["num_derived_operation_segments"],
        "diagnostic_segment_count": diag["num_derived_operation_segments"] + diag["num_rejected_raw_operation_segments"],
        "heat_ids": heat_ids[:max_examples],
        "stage_counts": stage_counts,
        "unit_counts": unit_counts,
        "time_span": {
            "start_min": min(starts) if starts else None,
            "end_min": max(ends) if ends else None,
        },
        "examples": examples,
        "conditioning_policy": "schedule_conditioned" if available else "history_only_no_valid_model_schedule",
        "pcvp": {
            "available": bool(projection_diag),
            "method": projection_diag.get("method") if projection_diag else None,
            "strong_repair_policy": bool(repair_meta.get("projection_strong_repair_policy")),
            "repair_rate": repair_meta.get("projection_repair_rate"),
            "uncertain_rate": repair_meta.get("projection_uncertain_rate"),
            "reject_rate": repair_meta.get("projection_reject_rate"),
            "violations_before": projection_diag.get("violations_before") if projection_diag else None,
            "violations_after": projection_diag.get("violations_after") if projection_diag else None,
        },
        "policy": "valid production context comes only from model_output predicted_operation_segments; derived/fallback segments are diagnostics only",
    }

def _build_schedule_conditioning_payload(
    production_prediction: Dict[str, Any],
    energy_sample: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    pred = safe_dict(production_prediction)
    summary = _build_operation_segment_compact_context(pred)
    input_state = safe_dict(safe_dict(energy_sample).get("input_state"))
    energy_repair_context = safe_dict(input_state.get("energy_repair_context"))
    return {
        "production_conditioning_available": bool(summary.get("available")),
        "production_activity_basis": "schedule_conditioned" if summary.get("available") else "history_only_no_valid_model_schedule",
        "production_operation_segment_summary": summary,
        "operation_segment_summary": summary,
        "pcvp_projection_summary": summary.get("pcvp", {}),
        "no_target_future_labels": True,
        "energy_repair_context_available": bool(energy_repair_context),
        "policy": (
            "Energy horizon prediction must condition on production_operation_segment_summary when available; "
            "full Gantt segments and target_future labels are not sent into the energy prompt."
        ),
    }

def _prediction_nested_meta(prediction: Dict[str, Any]) -> Dict[str, Any]:
    pred = safe_dict(prediction)
    meta: Dict[str, Any] = {}
    for key in ["retry_meta", "repair_meta", "projection_meta", "mechanism_meta"]:
        obj = safe_dict(pred.get(key))
        if obj:
            meta.update(obj)
    return meta

def _bool_meta(prediction: Dict[str, Any], *keys: str, default: bool = False) -> bool:
    pred = safe_dict(prediction)
    meta = _prediction_nested_meta(pred)
    for key in keys:
        if key in pred:
            return bool(pred.get(key))
        if key in meta:
            return bool(meta.get(key))
    return default

def _num_meta(prediction: Dict[str, Any], *keys: str, default: float = 0.0) -> float:
    pred = safe_dict(prediction)
    meta = _prediction_nested_meta(pred)
    for key in keys:
        if key in pred:
            return safe_float(pred.get(key), default)
        if key in meta:
            return safe_float(meta.get(key), default)
    return default

def _operation_segment_diagnostics(prediction: Dict[str, Any]) -> Dict[str, Any]:
    source = _operation_segment_source(prediction)
    model_segs = _model_predicted_operation_segments(prediction)
    derived_segs = _derived_operation_segments(prediction)
    raw_pred_segs = _raw_predicted_operation_segments(prediction)
    eligible = _is_model_operation_segment_source(source) and len(model_segs) > 0
    rejected = max(0, len(raw_pred_segs) - len(model_segs))
    return {
        "operation_segment_source": source,
        "operation_segment_evaluation_eligible": bool(eligible),
        "operation_segment_prediction_available": bool(eligible),
        "operation_segment_missing": not bool(eligible),
        "num_model_predicted_operation_segments": len(model_segs),
        "num_raw_predicted_operation_segments": len(raw_pred_segs),
        "num_rejected_raw_operation_segments": rejected,
        "num_derived_operation_segments": len(derived_segs),

        "retry_enabled": _bool_meta(prediction, "retry_enabled"),
        "num_horizon_attempts": _num_meta(prediction, "num_horizon_attempts", "num_attempts"),
        "num_horizon_retries_used": _num_meta(prediction, "num_horizon_retries_used", "num_retries_used"),
        "first_pass_missing": _bool_meta(prediction, "first_pass_missing"),
        "retry_success": _bool_meta(prediction, "retry_success"),
        "retry_uses_future_target_labels": _bool_meta(prediction, "retry_uses_future_target_labels", "uses_future_target_labels"),
        "rule_fallback_enabled": _bool_meta(prediction, "rule_fallback_enabled"),
        "rule_fallback_used": _bool_meta(prediction, "rule_fallback_used"),
        "rule_fallback_success": _bool_meta(prediction, "rule_fallback_success"),
        "rule_fallback_evaluate_as_prediction": _bool_meta(prediction, "rule_fallback_evaluate_as_prediction"),
        "rule_fallback_uses_future_target_labels": _bool_meta(prediction, "rule_fallback_uses_future_target_labels"),
        "rule_fallback_num_raw_segments": _num_meta(prediction, "rule_fallback_num_raw_segments"),
        "rule_fallback_num_projected_segments": _num_meta(prediction, "rule_fallback_num_projected_segments"),
        "warning": "" if eligible else (
            "predicted_operation_segments is missing or diagnostic-only; "
            "valid model-output or explicitly evaluated rule-assisted segments are required for Gantt evidence"
        ),
    }

def _attach_prediction_audit_metrics(metrics: Dict[str, Any], prediction: Dict[str, Any], prefix: str = "") -> None:

    if not isinstance(metrics, dict) or not isinstance(prediction, dict):
        return
    meta = _prediction_nested_meta(prediction)
    source_dicts = [prediction, meta]
    keys = [
        "retry_enabled", "num_horizon_attempts", "num_horizon_retries_used",
        "first_pass_missing", "retry_success", "retry_uses_future_target_labels",
        "rule_fallback_enabled", "rule_fallback_used", "rule_fallback_success",
        "rule_fallback_evaluate_as_prediction", "rule_fallback_uses_future_target_labels",
        "rule_fallback_num_raw_segments", "rule_fallback_num_projected_segments",
        "physics_consistent_projection_applied", "projection_uses_future_target_labels",
        "projection_repair_rate", "projection_uncertain_rate", "projection_reject_rate",
        "projection_hard_violations_before", "projection_hard_violations_after",
        "energy_pcvp_applied", "physics_consistent_energy_projection_applied",
        "energy_projection_repair_rate", "energy_projection_uncertain_rate", "energy_projection_reject_rate",
        "coupling_retry_enabled", "coupling_num_horizon_attempts", "coupling_num_horizon_retries_used",
        "coupling_first_pass_missing_or_invalid", "coupling_retry_success",
        "coupling_rule_fallback_used", "coupling_rule_fallback_uses_future_target_labels",
    ]
    for key in keys:
        for d in source_dicts:
            if key in d:
                value = d.get(key)
                if isinstance(value, bool):
                    metrics[prefix + key] = 1.0 if value else 0.0
                elif isinstance(value, (int, float)):
                    metrics[prefix + key] = value
                else:
                    metrics[prefix + key] = value
                break

def _target_operation_segments_from_sample(joint_sample: Dict[str, Any]) -> List[Dict[str, Any]]:

    sample = safe_dict(joint_sample)

    target = safe_dict(sample.get("target_future"))
    for key in [
        "production_operation_segments",
        "target_operation_segments",
        "operation_segments",
        "future_operation_segments",
    ]:
        val = target.get(key)
        if isinstance(val, list):
            return [x for x in val if isinstance(x, dict)]

    comp = safe_dict(sample.get("component_samples"))
    prod_sample = safe_dict(comp.get("production"))
    prod_target = safe_dict(prod_sample.get("target_future"))
    for key in [
        "production_operation_segments",
        "target_operation_segments",
        "operation_segments",
        "future_operation_segments",
    ]:
        val = prod_target.get(key)
        if isinstance(val, list):
            return [x for x in val if isinstance(x, dict)]

    return []

def _build_gantt_artifact(joint_sample: Dict[str, Any], result: "JointHorizonResult") -> Dict[str, Any]:

    pred = safe_dict(result.production_prediction)
    diag = _operation_segment_diagnostics(pred)
    target_segments = _target_operation_segments_from_sample(joint_sample)

    return {
        "sample_id": result.sample_id,
        "day_id": result.day_id,
        "anchor_patch_idx": result.anchor_patch_idx,
        "history_window": result.history_window,
        "horizon": result.horizon,
        "operation_segment_source": diag["operation_segment_source"],
        "operation_segment_evaluation_eligible": diag["operation_segment_evaluation_eligible"],
        "operation_segment_prediction_available": diag["operation_segment_prediction_available"],
        "predicted_operation_segments": _model_predicted_operation_segments(pred),
        "raw_predicted_operation_segments": _raw_predicted_operation_segments(pred),
        "diagnostic_derived_operation_segments": _derived_operation_segments(pred),
        "raw_predicted_operation_segments_before_projection": safe_list(pred.get("raw_predicted_operation_segments_before_projection")),
        "rolling_repaired_operation_segments_before_projection": safe_list(pred.get("rolling_repaired_operation_segments_before_projection")),
        "uncertain_operation_segments": safe_list(pred.get("uncertain_operation_segments")),
        "rejected_operation_segments": safe_list(pred.get("rejected_operation_segments")),
        "mechanism_projection_diagnostics": safe_dict(pred.get("mechanism_projection_diagnostics") or safe_dict(pred.get("repair_meta")).get("mechanism_projection_diagnostics")),
        "target_operation_segments": target_segments,
        "counts": {
            "num_model_predicted_operation_segments": diag["num_model_predicted_operation_segments"],
            "num_raw_predicted_operation_segments": diag["num_raw_predicted_operation_segments"],
            "num_rejected_raw_operation_segments": diag["num_rejected_raw_operation_segments"],
            "num_derived_operation_segments": diag["num_derived_operation_segments"],
            "num_target_operation_segments": len(target_segments),
            "num_uncertain_operation_segments": len(safe_list(pred.get("uncertain_operation_segments"))),
            "num_rejected_operation_segments_by_projection": len(safe_list(pred.get("rejected_operation_segments"))),
        },
        "plotting_policy": {
            "draw_predicted_from": "predicted_operation_segments only",
            "draw_target_from": "target_operation_segments",
            "do_not_use_as_prediction": [
                "raw_predicted_operation_segments rejected by source gate",
                "diagnostic_derived_operation_segments",
                "fallback_operation_segments",
                "patch_operation_segments",
            ],
        },
        "warning": diag["warning"],
    }

def safe_target_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}

class TransitionEngine:

    def __init__(
        self,
        prod_agent: Optional[Any] = None,
        energy_agent: Optional[Any] = None,
        couple_agent: Optional[Any] = None,
        prod_agent_factory: Optional[Callable[[], Any]] = None,
        energy_agent_factory: Optional[Callable[[], Any]] = None,
        couple_agent_factory: Optional[Callable[[], Any]] = None,
        auto_release: bool = False,
        verbose: bool = True,
    ) -> None:
        self.prod_agent = prod_agent
        self.energy_agent = energy_agent
        self.couple_agent = couple_agent
        self.prod_agent_factory = prod_agent_factory
        self.energy_agent_factory = energy_agent_factory
        self.couple_agent_factory = couple_agent_factory
        self.auto_release = auto_release
        self.verbose = verbose
        self._validate_init()

    def _validate_init(self) -> None:
        if self.prod_agent is None and self.prod_agent_factory is None:
            raise ValueError("Either prod_agent or prod_agent_factory must be provided.")
        if self.energy_agent is None and self.energy_agent_factory is None:
            raise ValueError("Either energy_agent or energy_agent_factory must be provided.")
        if self.couple_agent is None and self.couple_agent_factory is None:
            raise ValueError("Either couple_agent or couple_agent_factory must be provided.")

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _clear_memory(self) -> None:
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass

    def _release_agent(self, agent: Any) -> None:
        try:
            if hasattr(agent, "chat_model"):
                cm = getattr(agent, "chat_model")
                for attr in ["model", "tokenizer"]:
                    if hasattr(cm, attr):
                        try:
                            delattr(cm, attr)
                        except Exception:
                            pass
        except Exception:
            pass
        self._clear_memory()

    def _get_agent(self, stage: str) -> Any:
        if stage == "prod":
            return self.prod_agent if self.prod_agent is not None else self.prod_agent_factory()
        if stage == "energy":
            return self.energy_agent if self.energy_agent is not None else self.energy_agent_factory()
        if stage == "couple":
            return self.couple_agent if self.couple_agent is not None else self.couple_agent_factory()
        raise ValueError(f"Unknown stage: {stage}")

    def _should_release(self, stage: str) -> bool:
        if not self.auto_release:
            return False
        if stage == "prod":
            return self.prod_agent is None and self.prod_agent_factory is not None
        if stage == "energy":
            return self.energy_agent is None and self.energy_agent_factory is not None
        if stage == "couple":
            return self.couple_agent is None and self.couple_agent_factory is not None
        return False

    def predict_joint_horizon(self, joint_sample: Dict[str, Any]) -> JointHorizonResult:
        sample_id = str(joint_sample.get("sample_id", "unknown_sample"))
        day_id = str(joint_sample.get("day_id", "unknown_day"))
        anchor_patch_idx = int(joint_sample.get("anchor_patch_idx", -1))
        history_window = int(joint_sample.get("history_window", 0))
        horizon = int(joint_sample.get("horizon", 0))
        component_samples = safe_dict(joint_sample.get("component_samples"))
        production_sample = safe_dict(component_samples.get("production"))
        energy_sample = safe_dict(component_samples.get("energy"))

        if not production_sample:
            raise ValueError("joint_sample is missing component_samples.production.")
        if not energy_sample:
            raise ValueError("joint_sample is missing component_samples.energy.")

        self._log(f"[JointHorizon] sample={sample_id}, K={horizon}, anchor={anchor_patch_idx}")

        prod_agent = self._get_agent("prod")
        production_prediction: Dict[str, Any] = {}
        production_metrics: Dict[str, Any] = {}
        try:
            self._log("[JointHorizon] Stage 1/3: production horizon prediction")
            raw_prod = prod_agent.predict_horizon(production_sample)
            production_prediction = raw_prod if isinstance(raw_prod, dict) else {
                "task": "production_horizon_prediction",
                "mode": "invalid_type_fallback",
                "predicted_operation_segments": [],
                "error": f"production prediction returned {type(raw_prod).__name__}",
            }

            target = safe_target_dict(production_sample.get("target_future"))
            if target and hasattr(prod_agent, "evaluate_horizon_prediction"):
                production_metrics = prod_agent.evaluate_horizon_prediction(production_prediction, target)

            op_diag = _operation_segment_diagnostics(production_prediction)
            production_prediction["operation_segment_diagnostics"] = op_diag

            production_prediction["operation_segment_source"] = op_diag["operation_segment_source"]
            production_prediction["operation_segment_evaluation_eligible"] = op_diag["operation_segment_evaluation_eligible"]
            production_prediction["operation_segment_prediction_available"] = op_diag["operation_segment_prediction_available"]
            production_prediction["production_task_reduction"] = {
                "production_task": "operation_level_gantt_segment_prediction_only",
                "removed_tasks": [
                    "patch_level_future_production_state_prediction",
                    "production_heat_patch_alignment_prediction",
                ],
            }
            operation_summary = _build_operation_segment_compact_context(production_prediction)
            production_prediction["operation_segment_summary"] = operation_summary
            production_prediction["production_operation_segment_summary"] = operation_summary
            production_prediction["production_conditioning_available_for_energy"] = bool(operation_summary.get("available"))
            production_prediction["production_activity_basis_for_energy"] = (
                "schedule_conditioned" if operation_summary.get("available") else "history_only_no_valid_model_schedule"
            )

            production_metrics["operation_segment_source"] = op_diag["operation_segment_source"]
            production_metrics["operation_segment_evaluation_eligible"] = 1.0 if op_diag["operation_segment_evaluation_eligible"] else 0.0
            production_metrics["operation_segment_prediction_available"] = 1.0 if op_diag["operation_segment_prediction_available"] else 0.0
            production_metrics["operation_segment_missing"] = 1.0 if op_diag["operation_segment_missing"] else 0.0
            production_metrics["num_model_predicted_operation_segments"] = op_diag["num_model_predicted_operation_segments"]
            production_metrics["num_raw_predicted_operation_segments"] = op_diag["num_raw_predicted_operation_segments"]
            production_metrics["num_rejected_raw_operation_segments"] = op_diag["num_rejected_raw_operation_segments"]
            production_metrics["num_derived_operation_segments"] = op_diag["num_derived_operation_segments"]
            production_metrics["num_predicted_operation_segments"] = op_diag["num_model_predicted_operation_segments"]
            for audit_key in [
                "retry_enabled", "num_horizon_attempts", "num_horizon_retries_used",
                "first_pass_missing", "retry_success", "retry_uses_future_target_labels",
                "rule_fallback_enabled", "rule_fallback_used", "rule_fallback_success",
                "rule_fallback_evaluate_as_prediction", "rule_fallback_uses_future_target_labels",
                "rule_fallback_num_raw_segments", "rule_fallback_num_projected_segments",
            ]:
                production_metrics[audit_key] = op_diag.get(audit_key)
            _attach_prediction_audit_metrics(production_metrics, production_prediction)

            if not op_diag["operation_segment_evaluation_eligible"]:

                production_metrics["operation_start_mae"] = None
                production_metrics["operation_end_mae"] = None
                production_metrics["operation_duration_mae"] = None
        except Exception as e:
            production_prediction = {
                "task": "production_horizon_prediction",
                "mode": "engine_exception_fallback",
                "predicted_operation_segments": [],
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc(),
            }
            production_metrics = {
                "coverage": 0.0,
                "heat_jaccard": 0.0,
                "heat_exact_match_rate": 0.0,
                "stage_accuracy": 0.0,
                "rolling_unit_jaccard": 0.0,
                "operation_segment_evaluation_eligible": 0.0,
                "num_model_predicted_operation_segments": 0,
                "num_raw_predicted_operation_segments": 0,
                "num_derived_operation_segments": 0,
                "num_rejected_raw_operation_segments": 0,
                "operation_segment_prediction_available": 0.0,
                "operation_segment_missing": 1.0,
                "operation_start_mae": None,
                "operation_end_mae": None,
                "operation_duration_mae": None,
                "operation_segment_source": "production_exception",
                "operation_segment_precision": 0.0,
                "operation_segment_recall": 0.0,
                "operation_segment_f1": 0.0,
                "operation_segment_mean_iou": 0.0,
                "operation_segment_false_positive_count": 0,
                "operation_segment_false_negative_count": 0,
                "unmatched_predicted_segment_rate": 0.0,
                "unmatched_target_segment_rate": 1.0,
                "error_type": type(e).__name__,
            }
            self._log(f"[JointHorizon][WARN] production stage failed: {type(e).__name__}: {e}")
        finally:
            if self._should_release("prod"):
                self._release_agent(prod_agent)

        energy_agent = self._get_agent("energy")
        energy_prediction: Dict[str, Any] = {}
        energy_metrics: Dict[str, Any] = {}
        try:
            self._log("[JointHorizon] Stage 2/3: energy horizon prediction")
            production_energy_context = _build_schedule_conditioning_payload(
                production_prediction=production_prediction,
                energy_sample=energy_sample,
            )
            energy_sample_for_prediction = deepcopy(energy_sample)
            energy_input_state = safe_dict(energy_sample_for_prediction.get("input_state"))
            energy_input_state["production_operation_segment_summary"] = production_energy_context["production_operation_segment_summary"]
            energy_input_state["operation_segment_summary"] = production_energy_context["production_operation_segment_summary"]
            energy_input_state["production_conditioning_available"] = production_energy_context["production_conditioning_available"]
            energy_input_state["production_activity_basis"] = production_energy_context["production_activity_basis"]
            energy_input_state["schedule_conditioning_payload"] = production_energy_context
            energy_sample_for_prediction["input_state"] = energy_input_state

            raw_energy = energy_agent.predict_horizon(
                energy_sample_for_prediction,
                production_prediction=production_energy_context,
            )
            energy_prediction = raw_energy if isinstance(raw_energy, dict) else {
                "task": "energy_horizon_prediction",
                "mode": "invalid_type_fallback",
                "predicted_energy_future": [],
                "error": f"energy prediction returned {type(raw_energy).__name__}",
            }
            energy_prediction.setdefault("production_operation_segment_summary", production_energy_context["production_operation_segment_summary"])
            energy_prediction["production_conditioning_available"] = bool(production_energy_context["production_conditioning_available"])
            energy_prediction["production_activity_basis"] = production_energy_context["production_activity_basis"]
            energy_prediction["schedule_conditioning_payload"] = production_energy_context
            energy_prediction["energy_stage_input_policy"] = {
                "uses_production_operation_segment_summary": bool(production_energy_context["production_conditioning_available"]),
                "does_not_use_full_gantt_segments_in_prompt": True,
                "does_not_use_target_future_labels": True,
            }

            summary_obj = safe_dict(energy_prediction.get("summary") or energy_prediction.get("horizon_summary"))
            if summary_obj:
                summary_obj.setdefault("production_conditioning_available", bool(production_energy_context["production_conditioning_available"]))
                summary_obj.setdefault("production_activity_basis", production_energy_context["production_activity_basis"])
                energy_prediction["summary"] = summary_obj

            target = safe_target_dict(energy_sample.get("target_future"))
            if target and hasattr(energy_agent, "evaluate_horizon_prediction"):
                energy_metrics = energy_agent.evaluate_horizon_prediction(energy_prediction, target)
            energy_metrics["production_conditioning_available"] = 1.0 if production_energy_context["production_conditioning_available"] else 0.0
            energy_metrics["production_schedule_conditioned"] = 1.0 if production_energy_context["production_activity_basis"] == "schedule_conditioned" else 0.0
            energy_metrics["production_operation_segment_summary_num_segments"] = safe_float(
                production_energy_context["production_operation_segment_summary"].get("num_segments")
            )

            energy_diag = safe_dict(energy_prediction.get("mechanism_energy_projection_diagnostics") or energy_prediction.get("energy_projection_diagnostics"))
            energy_retry = safe_dict(energy_prediction.get("retry_meta") or energy_prediction.get("energy_retry_meta"))
            energy_metrics["physics_consistent_energy_projection_applied"] = 1.0 if energy_diag else safe_float(energy_prediction.get("energy_pcvp_applied"))
            energy_metrics["energy_pcvp_applied"] = energy_metrics["physics_consistent_energy_projection_applied"]
            energy_metrics["energy_projection_repair_rate"] = safe_float(energy_diag.get("repair_rate") or energy_diag.get("energy_projection_repair_rate"))
            energy_metrics["energy_projection_uncertain_rate"] = safe_float(energy_diag.get("uncertain_rate") or energy_diag.get("energy_projection_uncertain_rate"))
            energy_metrics["energy_projection_reject_rate"] = safe_float(energy_diag.get("reject_rate") or energy_diag.get("energy_projection_reject_rate"))
            for k in [
                "energy_retry_enabled", "energy_num_horizon_attempts", "energy_num_horizon_retries_used",
                "energy_first_pass_missing", "energy_retry_success", "energy_retry_uses_future_target_labels",
                "energy_rule_fallback_enabled", "energy_rule_fallback_used", "energy_rule_fallback_success",
                "energy_rule_fallback_evaluate_as_prediction", "energy_rule_fallback_uses_future_target_labels",
            ]:
                if k in energy_prediction:
                    energy_metrics[k] = energy_prediction.get(k)
                elif k in energy_retry:
                    energy_metrics[k] = energy_retry.get(k)
            _attach_prediction_audit_metrics(energy_metrics, energy_prediction, prefix="")
        except Exception as e:
            energy_prediction = {
                "task": "energy_horizon_prediction",
                "mode": "engine_exception_fallback",
                "predicted_energy_future": [],
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc(),
            }
            energy_metrics = {
                "coverage": 0.0,
                "energy_mae": 0.0,
                "energy_rmse": 0.0,
                "risk_precision": 0.0,
                "risk_recall": 0.0,
                "risk_f1": 0.0,
                "error_type": type(e).__name__,
            }
            self._log(f"[JointHorizon][WARN] energy stage failed: {type(e).__name__}: {e}")
        finally:
            if self._should_release("energy"):
                self._release_agent(energy_agent)

        couple_agent = self._get_agent("couple")
        coupling_prediction: Dict[str, Any] = {}
        coupling_metrics: Dict[str, Any] = {}

        prod_for_couple = strip_runtime_fields_keep_operation_segments(production_prediction)
        energy_for_couple = strip_for_coupling(energy_prediction)
        if isinstance(energy_for_couple, dict):
            energy_for_couple["production_conditioning_available"] = energy_prediction.get("production_conditioning_available")
            energy_for_couple["production_activity_basis"] = energy_prediction.get("production_activity_basis")
            energy_for_couple["production_operation_segment_summary"] = energy_prediction.get("production_operation_segment_summary")

        try:
            self._log("[JointHorizon] Stage 3/3: operation-schedule-to-energy functional coupling")
            raw_coupling = couple_agent.predict_horizon(
                joint_sample=joint_sample,
                production_prediction=prod_for_couple,
                energy_prediction=energy_for_couple,
            )
            coupling_prediction = raw_coupling if isinstance(raw_coupling, dict) else {
                "task": "joint_horizon_coupling_prediction",
                "mode": "invalid_type_fallback",
                "predicted_coupling_future": [],
                "horizon_summary": {
                    "mean_consistency": 0.0,
                    "feasibility_rate": 0.0,
                    "major_conflicts": [f"coupling prediction returned {type(raw_coupling).__name__}"],
                },
            }
        except Exception as e:
            coupling_prediction = {
                "task": "operation_schedule_energy_functional_coupling_prediction",
                "mode": "engine_exception_fallback",
                "predicted_coupling_future": [],
                "horizon_summary": {
                    "mean_consistency": 0.0,
                    "feasibility_rate": 0.0,
                    "major_conflicts": [f"coupling_exception:{type(e).__name__}"],
                    "comment": str(e),
                },
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc(),
            }
            self._log(f"[JointHorizon][WARN] coupling stage failed: {type(e).__name__}: {e}")

        try:
            if hasattr(couple_agent, "evaluate_horizon_prediction"):
                target = safe_target_dict(joint_sample.get("target_future"))
                try:
                    coupling_metrics = couple_agent.evaluate_horizon_prediction(
                        prediction=coupling_prediction,
                        target_future=target,
                        joint_sample=joint_sample,
                        production_prediction=prod_for_couple,
                        energy_prediction=energy_for_couple,
                    )
                except TypeError:

                    coupling_metrics = couple_agent.evaluate_horizon_prediction(coupling_prediction, target)
                _attach_prediction_audit_metrics(coupling_metrics, coupling_prediction)
                retry_meta = safe_dict(coupling_prediction.get("retry_meta") or coupling_prediction.get("coupling_retry_meta"))
                for k in [
                    "coupling_retry_enabled", "coupling_num_horizon_attempts", "coupling_num_horizon_retries_used",
                    "coupling_first_pass_missing_or_invalid", "coupling_retry_success",
                    "coupling_rule_fallback_used", "coupling_rule_fallback_uses_future_target_labels",
                ]:
                    if k in coupling_prediction:
                        coupling_metrics[k] = coupling_prediction.get(k)
                    elif k in retry_meta:
                        coupling_metrics[k] = retry_meta.get(k)
        except Exception as e:
            coupling_metrics = {
                "num_target_steps": 0,
                "num_predicted_steps": 0,
                "coverage": 0.0,
                "functional_coupling_score": 0.0,
                "schedule_energy_response_consistency": 0.0,
                "production_activity_detection_accuracy": 0.0,
                "risk_precision": 0.0,
                "risk_recall": 0.0,
                "risk_f1": 0.0,
                "mean_consistency": 0.0,
                "feasibility_rate": 0.0,
                "error_type": type(e).__name__,
                "error_message": str(e),
            }
            self._log(f"[JointHorizon][WARN] coupling metric evaluation failed: {type(e).__name__}: {e}")
        finally:
            if self._should_release("couple"):
                self._release_agent(couple_agent)

        joint_metrics = self._build_joint_metrics(
            production_metrics=production_metrics,
            energy_metrics=energy_metrics,
            coupling_metrics=coupling_metrics,
        )

        return JointHorizonResult(
            sample_id=sample_id,
            day_id=day_id,
            anchor_patch_idx=anchor_patch_idx,
            history_window=history_window,
            horizon=horizon,
            production_prediction=production_prediction,
            production_metrics=production_metrics,
            energy_prediction=energy_prediction,
            energy_metrics=energy_metrics,
            coupling_prediction=coupling_prediction,
            coupling_metrics=coupling_metrics,
            joint_metrics=joint_metrics,
            sample_meta=safe_dict(joint_sample.get("meta")),
        )

    def predict_dataset(
        self,
        assembler: Any,
        day_id: str,
        horizon: int,
        history_window: int = 5,
        stride: int = 1,
        max_samples: Optional[int] = None,
        anchor_indices: Optional[Iterable[int]] = None,
        output_dir: Optional[Path] = None,
    ) -> List[JointHorizonResult]:

        horizon = int(horizon)
        build_errors: List[Dict[str, Any]] = []

        if anchor_indices is not None:
            samples: List[Dict[str, Any]] = []
            for anchor in anchor_indices:
                try:
                    sample = assembler.build_joint_horizon_sample(
                        day_id=day_id,
                        anchor_patch_idx=int(anchor),
                        history_window=history_window,
                        horizon=horizon,
                        include_target=True,
                    )
                    samples.append(sample)
                except Exception as exc:
                    build_errors.append({
                        "sample_id": f"{day_id}_joint_anchor_{anchor}_W{history_window}_K{horizon}",
                        "day_id": day_id,
                        "anchor_patch_idx": int(anchor),
                        "history_window": int(history_window),
                        "horizon": horizon,
                        "error": f"build_joint_horizon_sample_failed:{type(exc).__name__}",
                        "error_message": str(exc),
                    })
            if max_samples is not None:
                samples = samples[:max_samples]
        else:
            samples = assembler.build_joint_horizon_dataset(
                day_id=day_id,
                history_window=history_window,
                horizon=horizon,
                stride=stride,
                max_samples=max_samples,
            )

            last_errors = getattr(assembler, "_last_horizon_sample_build_errors", None)
            if isinstance(last_errors, list):
                build_errors.extend([x for x in last_errors if isinstance(x, dict)])

        valid_samples: List[Dict[str, Any]] = []
        invalid_samples: List[Dict[str, Any]] = []

        for idx, sample in enumerate(samples, start=1):
            if not isinstance(sample, dict):
                item = {
                    "index": idx,
                    "error": "invalid_sample_object",
                    "error_message": f"sample object type is {type(sample).__name__}",
                }
                invalid_samples.append(item)
                self._log(f"[JointHorizon][WARN] skip invalid sample object: {type(sample).__name__}")
                continue

            if sample.get("error"):
                item = {
                    "index": idx,
                    "sample_id": sample.get("sample_id"),
                    "anchor_patch_idx": sample.get("anchor_patch_idx"),
                    "horizon": sample.get("horizon"),
                    "error": sample.get("error"),
                    "error_message": sample.get("error_message"),
                }
                invalid_samples.append(item)
                self._log(
                    "[JointHorizon][WARN] skip sample build error: "
                    f"sample={sample.get('sample_id')} error={sample.get('error_message')}"
                )
                continue

            comp = safe_dict(sample.get("component_samples"))
            prod_sample = safe_dict(comp.get("production"))
            energy_sample = safe_dict(comp.get("energy"))
            if not prod_sample or not energy_sample:
                item = {
                    "index": idx,
                    "sample_id": sample.get("sample_id"),
                    "anchor_patch_idx": sample.get("anchor_patch_idx"),
                    "horizon": sample.get("horizon"),
                    "error": "missing_component_samples",
                    "sample_keys": list(sample.keys()),
                    "component_keys": list(comp.keys()),
                    "has_component_samples_production": bool(prod_sample),
                    "has_component_samples_energy": bool(energy_sample),
                }
                invalid_samples.append(item)
                self._log(
                    "[JointHorizon][WARN] skip invalid joint sample without component samples: "
                    f"sample={sample.get('sample_id')} keys={list(sample.keys())} "
                    f"component_keys={list(comp.keys())}"
                )
                continue

            valid_samples.append(sample)

        if output_dir is not None and (build_errors or invalid_samples):
            diag_dir = Path(output_dir) / f"K_{horizon:02d}"
            save_json({
                "day_id": day_id,
                "horizon": horizon,
                "history_window": history_window,
                "stride": stride,
                "num_raw_samples": len(samples),
                "num_valid_samples": len(valid_samples),
                "num_build_errors": len(build_errors),
                "num_invalid_samples": len(invalid_samples),
                "build_errors": build_errors,
                "invalid_samples": invalid_samples,
            }, diag_dir / "horizon_sample_build_errors.json")

        if not valid_samples:
            raise ValueError(
                "No valid joint horizon samples were built. "
                "Each sample must contain component_samples.production and component_samples.energy. "
                f"day_id={day_id}, horizon={horizon}, history_window={history_window}, "
                f"num_raw_samples={len(samples)}, num_build_errors={len(build_errors)}, "
                f"num_invalid_samples={len(invalid_samples)}. "
                "See horizon_sample_build_errors.json for details."
            )

        results: List[JointHorizonResult] = []
        for idx, sample in enumerate(valid_samples, start=1):
            result = self.predict_joint_horizon(sample)
            results.append(result)
            if output_dir is not None:
                sample_dir = Path(output_dir) / f"K_{horizon:02d}" / f"sample_{idx:03d}_anchor_{result.anchor_patch_idx}"
                save_json(sample, sample_dir / "joint_horizon_sample.json")
                save_json(result.to_dict(), sample_dir / "joint_horizon_result.json")
                save_json(result.production_prediction, sample_dir / "production_prediction.json")
                save_json(result.production_metrics, sample_dir / "production_metrics.json")
                save_json(_build_gantt_artifact(sample, result), sample_dir / "gantt_operation_segments.json")
                save_json(result.energy_prediction, sample_dir / "energy_prediction.json")
                save_json(result.energy_metrics, sample_dir / "energy_metrics.json")
                save_json(result.coupling_prediction, sample_dir / "coupling_prediction.json")
                save_json(result.coupling_metrics, sample_dir / "coupling_metrics.json")
                save_json(result.joint_metrics, sample_dir / "joint_metrics.json")
        return results

    def _build_joint_metrics(
        self,
        production_metrics: Dict[str, Any],
        energy_metrics: Dict[str, Any],
        coupling_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        pm = safe_dict(production_metrics)
        em = safe_dict(energy_metrics)
        cm = safe_dict(coupling_metrics)

        return {
            "task_production_operation_gantt_prediction": 1.0,
            "task_energy_horizon_prediction": 1.0,
            "task_functional_schedule_energy_coupling": 1.0,

            "prod_num_target_operation_segments": safe_float(pm.get("num_target_operation_segments")),
            "prod_num_predicted_operation_segments": safe_float(pm.get("num_predicted_operation_segments")),
            "prod_num_model_predicted_operation_segments": safe_float(pm.get("num_model_predicted_operation_segments")),
            "prod_num_raw_predicted_operation_segments": safe_float(pm.get("num_raw_predicted_operation_segments")),
            "prod_num_derived_operation_segments": safe_float(pm.get("num_derived_operation_segments")),
            "prod_num_rejected_raw_operation_segments": safe_float(pm.get("num_rejected_raw_operation_segments")),
            "prod_operation_segment_evaluation_eligible": safe_float(pm.get("operation_segment_evaluation_eligible")),
            "prod_operation_segment_prediction_available": safe_float(pm.get("operation_segment_prediction_available")),
            "prod_operation_segment_missing": safe_float(pm.get("operation_segment_missing")),
            "prod_operation_segment_precision": safe_float(pm.get("operation_segment_precision")),
            "prod_operation_segment_recall": safe_float(pm.get("operation_segment_recall")),
            "prod_operation_segment_f1": safe_float(pm.get("operation_segment_f1")),
            "prod_operation_segment_mean_iou": safe_float(pm.get("operation_segment_mean_iou")),
            "prod_operation_segment_false_positive_count": safe_float(pm.get("operation_segment_false_positive_count")),
            "prod_operation_segment_false_negative_count": safe_float(pm.get("operation_segment_false_negative_count")),
            "prod_unmatched_predicted_segment_rate": safe_float(pm.get("unmatched_predicted_segment_rate")),
            "prod_unmatched_target_segment_rate": safe_float(pm.get("unmatched_target_segment_rate")),
            "prod_operation_start_mae": safe_optional_float(pm.get("operation_start_mae")),
            "prod_operation_end_mae": safe_optional_float(pm.get("operation_end_mae")),
            "prod_operation_duration_mae": safe_optional_float(pm.get("operation_duration_mae")),
            "prod_physics_consistent_projection_applied": safe_float(pm.get("physics_consistent_projection_applied")),
            "prod_projection_uses_future_target_labels": safe_float(pm.get("projection_uses_future_target_labels")),
            "prod_projection_repair_rate": safe_float(pm.get("projection_repair_rate")),
            "prod_projection_uncertain_rate": safe_float(pm.get("projection_uncertain_rate")),
            "prod_projection_reject_rate": safe_float(pm.get("projection_reject_rate")),
            "prod_projection_hard_violations_before": safe_float(pm.get("projection_hard_violations_before")),
            "prod_projection_hard_violations_after": safe_float(pm.get("projection_hard_violations_after")),
            "prod_num_uncertain_operation_segments": safe_float(pm.get("num_uncertain_operation_segments")),
            "prod_num_rejected_operation_segments_by_projection": safe_float(pm.get("num_rejected_operation_segments_by_projection")),
            "prod_retry_enabled": safe_float(pm.get("retry_enabled")),
            "prod_num_horizon_attempts": safe_float(pm.get("num_horizon_attempts")),
            "prod_num_horizon_retries_used": safe_float(pm.get("num_horizon_retries_used")),
            "prod_first_pass_missing": safe_float(pm.get("first_pass_missing")),
            "prod_retry_success": safe_float(pm.get("retry_success")),
            "prod_retry_uses_future_target_labels": safe_float(pm.get("retry_uses_future_target_labels")),
            "prod_rule_fallback_enabled": safe_float(pm.get("rule_fallback_enabled")),
            "prod_rule_fallback_used": safe_float(pm.get("rule_fallback_used")),
            "prod_rule_fallback_success": safe_float(pm.get("rule_fallback_success")),
            "prod_rule_fallback_evaluate_as_prediction": safe_float(pm.get("rule_fallback_evaluate_as_prediction")),
            "prod_rule_fallback_uses_future_target_labels": safe_float(pm.get("rule_fallback_uses_future_target_labels")),
            "prod_rule_fallback_num_projected_segments": safe_float(pm.get("rule_fallback_num_projected_segments")),

            "energy_coverage": safe_float(em.get("coverage")),
            "energy_num_target_steps": safe_float(em.get("num_targets") or em.get("num_target_steps")),
            "energy_num_predicted_steps": safe_float(em.get("num_predictions") or em.get("num_predicted_steps")),
            "energy_mae": safe_float(em.get("energy_mae")),
            "energy_rmse": safe_float(em.get("energy_rmse")),
            "generation_mae": safe_float(em.get("generation_mae")),
            "holder_mae": safe_float(em.get("holder_mae")),
            "demand_mae": safe_float(em.get("demand_mae")),
            "purchase_mae": safe_float(em.get("purchase_mae")),
            "purchased_power_mae": safe_float(em.get("purchased_power_mae")),
            "purchased_steam_mae": safe_float(em.get("purchased_steam_mae")),
            "energy_risk_precision": safe_float(em.get("risk_precision")),
            "energy_risk_recall": safe_float(em.get("risk_recall")),
            "energy_risk_f1": safe_float(em.get("risk_f1")),
            "energy_risk_precision_excluding_ov": safe_float(em.get("risk_precision_excluding_ov")),
            "energy_risk_recall_excluding_ov": safe_float(em.get("risk_recall_excluding_ov")),
            "energy_risk_f1_excluding_ov": safe_float(em.get("risk_f1_excluding_ov")),
            "energy_purchase_risk_f1": safe_float(em.get("purchase_risk_f1")),
            "energy_holder_limit_risk_f1": safe_float(em.get("holder_limit_risk_f1")),
            "energy_production_conditioning_available": safe_float(em.get("production_conditioning_available")),
            "energy_production_schedule_conditioned": safe_float(em.get("production_schedule_conditioned")),
            "energy_operation_segment_summary_num_segments": safe_float(em.get("production_operation_segment_summary_num_segments")),
            "energy_physics_consistent_projection_applied": safe_float(em.get("physics_consistent_energy_projection_applied") or em.get("energy_projection_applied")),
            "energy_projection_repair_rate": safe_float(em.get("energy_projection_repair_rate") or em.get("projection_repair_rate")),
            "energy_projection_uncertain_rate": safe_float(em.get("energy_projection_uncertain_rate") or em.get("projection_uncertain_rate")),
            "energy_projection_reject_rate": safe_float(em.get("energy_projection_reject_rate") or em.get("projection_reject_rate")),
            "energy_retry_enabled": safe_float(em.get("energy_retry_enabled") or em.get("retry_enabled")),
            "energy_num_horizon_attempts": safe_float(em.get("energy_num_horizon_attempts") or em.get("num_horizon_attempts")),
            "energy_num_horizon_retries_used": safe_float(em.get("energy_num_horizon_retries_used") or em.get("num_horizon_retries_used")),
            "energy_first_pass_missing": safe_float(em.get("energy_first_pass_missing") or em.get("first_pass_missing")),
            "energy_retry_success": safe_float(em.get("energy_retry_success") or em.get("retry_success")),
            "energy_retry_uses_future_target_labels": safe_float(em.get("energy_retry_uses_future_target_labels") or em.get("retry_uses_future_target_labels")),
            "energy_rule_fallback_used": safe_float(em.get("energy_rule_fallback_used") or em.get("rule_fallback_used")),
            "energy_rule_fallback_success": safe_float(em.get("energy_rule_fallback_success") or em.get("rule_fallback_success")),
            "energy_rule_fallback_uses_future_target_labels": safe_float(em.get("energy_rule_fallback_uses_future_target_labels") or em.get("rule_fallback_uses_future_target_labels")),

            "coupling_functional_coupling_score": safe_float(
                cm.get("functional_coupling_score")
                or cm.get("operation_energy_functional_coupling_score")
                or cm.get("mean_consistency")
            ),
            "coupling_schedule_energy_response_consistency": safe_float(
                cm.get("schedule_energy_response_consistency")
                or cm.get("production_energy_response_consistency")
                or cm.get("mean_consistency")
            ),
            "coupling_production_activity_detection_accuracy": safe_float(
                cm.get("production_activity_detection_accuracy")
                or cm.get("production_activity_accuracy")
            ),
            "coupling_energy_response_detection_accuracy": safe_float(
                cm.get("energy_response_detection_accuracy")
                or cm.get("energy_response_accuracy")
            ),
            "coupling_risk_precision": safe_float(cm.get("risk_precision")),
            "coupling_risk_recall": safe_float(cm.get("risk_recall")),
            "coupling_risk_f1": safe_float(cm.get("risk_f1")),
            "coupling_risk_precision_excluding_ov": safe_float(cm.get("risk_precision_excluding_ov")),
            "coupling_risk_recall_excluding_ov": safe_float(cm.get("risk_recall_excluding_ov")),
            "coupling_risk_f1_excluding_ov": safe_float(cm.get("risk_f1_excluding_ov")),
            "coupling_purchase_risk_f1": safe_float(cm.get("purchase_risk_f1")),
            "coupling_holder_limit_risk_f1": safe_float(cm.get("holder_limit_risk_f1")),
            "coupling_mean_consistency": safe_float(cm.get("mean_consistency")),
            "coupling_feasibility_rate": safe_float(cm.get("feasibility_rate")),
            "coupling_coverage": safe_float(cm.get("coverage")),
            "coupling_num_target_steps": safe_float(cm.get("num_target_steps")),
            "coupling_num_predicted_steps": safe_float(cm.get("num_predicted_steps")),
            "coupling_retry_enabled": safe_float(cm.get("coupling_retry_enabled") or cm.get("retry_enabled")),
            "coupling_num_horizon_attempts": safe_float(cm.get("coupling_num_horizon_attempts") or cm.get("num_horizon_attempts")),
            "coupling_num_horizon_retries_used": safe_float(cm.get("coupling_num_horizon_retries_used") or cm.get("num_horizon_retries_used")),
            "coupling_first_pass_missing_or_invalid": safe_float(cm.get("coupling_first_pass_missing_or_invalid") or cm.get("first_pass_missing_or_invalid")),
            "coupling_retry_success": safe_float(cm.get("coupling_retry_success") or cm.get("retry_success")),
            "coupling_rule_fallback_used": safe_float(cm.get("coupling_rule_fallback_used") or cm.get("rule_fallback_used")),
            "coupling_rule_fallback_uses_future_target_labels": safe_float(cm.get("coupling_rule_fallback_uses_future_target_labels") or cm.get("rule_fallback_uses_future_target_labels")),
        }

    @staticmethod
    def summarize_results(results: List[JointHorizonResult]) -> Dict[str, Any]:
        if not results:
            return {"num_samples": 0}

        numeric_keys = sorted({
            key
            for r in results
            for key, value in safe_dict(r.joint_metrics).items()
            if is_valid_number(value)
        })

        summary: Dict[str, Any] = {
            "horizon": results[0].horizon,
            "history_window": results[0].history_window,
            "num_samples": len(results),
            "day_id": results[0].day_id,
        }
        for key in numeric_keys:
            vals = [
                float(r.joint_metrics.get(key))
                for r in results
                if is_valid_number(safe_dict(r.joint_metrics).get(key))
            ]
            summary[f"mean_{key}"] = sum(vals) / max(len(vals), 1) if vals else None
            summary[f"min_{key}"] = min(vals) if vals else None
            summary[f"max_{key}"] = max(vals) if vals else None
            summary[f"num_valid_{key}"] = len(vals)
        return summary

    @staticmethod
    def write_sample_metrics_csv(results: List[JointHorizonResult], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows: List[Dict[str, Any]] = []
        for r in results:
            row: Dict[str, Any] = {
                "sample_id": r.sample_id,
                "day_id": r.day_id,
                "anchor_patch_idx": r.anchor_patch_idx,
                "history_window": r.history_window,
                "horizon": r.horizon,
            }
            row.update(_flatten_numeric_metrics("prod_", r.production_metrics))
            row.update(_flatten_numeric_metrics("energy_", r.energy_metrics))
            row.update(_flatten_numeric_metrics("coupling_", r.coupling_metrics))
            row.update(_flatten_numeric_metrics("joint_", r.joint_metrics))
            rows.append(row)

        fieldnames = sorted({k for row in rows for k in row.keys()}) if rows else []
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    @staticmethod
    def write_horizon_summary_csv(summaries: List[Dict[str, Any]], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({k for s in summaries for k in s.keys()}) if summaries else []
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for s in summaries:
                writer.writerow(s)


@dataclass
class LearnedWorldModelRolloutResult:
    production_numeric: Any
    production_stage_logits: Any
    energy_numeric: Any
    energy_risk_logit: Any
    coupling_numeric: Any
    feasibility_logit: Any
    reward: Any
    latent_trajectory: Any


class LearnedWorldModelTransitionEngine:
    def __init__(self, checkpoint_path: Optional[str] = None, model: Any = None, device: str = "cpu") -> None:
        import torch
        from g_lma_pewm import LMAPEWorldModel, LMAPEWorldModelConfig

        self.torch = torch
        self.device = torch.device(device)
        if model is not None:
            self.model = model.to(self.device)
        elif checkpoint_path:
            payload = torch.load(checkpoint_path, map_location=self.device)
            config = LMAPEWorldModelConfig(**dict(payload.get("model_config") or {}))
            self.model = LMAPEWorldModel(config).to(self.device)
            self.model.load_state_dict(payload["model_state_dict"])
        else:
            raise ValueError("Either checkpoint_path or model must be provided.")
        self.model.eval()

    @staticmethod
    def model_input_keys() -> List[str]:
        return [
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

    def rollout(self, batch: Dict[str, Any]) -> LearnedWorldModelRolloutResult:
        inputs = {}
        for key in self.model_input_keys():
            value = batch[key]
            if not self.torch.is_tensor(value):
                value = self.torch.tensor(value, dtype=self.torch.float32)
            inputs[key] = value.to(self.device)
        with self.torch.no_grad():
            out = self.model(**inputs)
        return LearnedWorldModelRolloutResult(
            production_numeric=out["production_numeric"].detach().cpu(),
            production_stage_logits=out["production_stage_logits"].detach().cpu(),
            energy_numeric=out["energy_numeric"].detach().cpu(),
            energy_risk_logit=out["energy_risk_logit"].detach().cpu(),
            coupling_numeric=out["coupling_numeric"].detach().cpu(),
            feasibility_logit=out["feasibility_logit"].detach().cpu(),
            reward=out["reward"].detach().cpu(),
            latent_trajectory=out["latent_trajectory"].detach().cpu(),
        )
