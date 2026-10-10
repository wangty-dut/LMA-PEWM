from __future__ import annotations

import copy
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

@dataclass
class ProjectionConfig:
    min_duration_min: float = 0.5
    default_duration_min: float = 5.0
    high_confidence_threshold: float = 0.85
    downstream_confidence_threshold: float = 0.55
    reject_low_confidence_threshold: float = 0.20
    max_iterations: int = 3
    clip_to_horizon: bool = True
    allow_duration_projection: bool = True
    allow_unit_inference: bool = True
    uncertain_on_waiting_violation: bool = True
    default_duration_priors: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "BOF": {"min": 8.0, "median": 16.0, "max": 60.0},
        "LF": {"min": 5.0, "median": 18.0, "max": 90.0},
        "Casting": {"min": 2.0, "median": 6.0, "max": 60.0},
        "Rolling": {"min": 3.0, "median": 15.0, "max": 120.0},
    })
    default_waiting_priors: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "BOF->LF": {"min": 0.0, "median": 10.0, "max": 90.0},
        "BOF->Casting": {"min": 0.0, "median": 15.0, "max": 120.0},
        "LF->Casting": {"min": 0.0, "median": 10.0, "max": 90.0},
        "Casting->Rolling": {"min": 0.0, "median": 60.0, "max": 360.0},
    })
    objective_weights: Dict[str, float] = field(default_factory=lambda: {
        "llm_deviation": 1.0,
        "duration_prior": 0.3,
        "stage_order": 2.0,
        "waiting_window": 1.0,
        "unit_overlap": 2.0,
        "casting_continuity": 1.5,
        "rolling_queue": 1.5,
        "energy_proxy": 0.2,
    })

def _safe_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}

def _safe_list(x: Any) -> List[Any]:
    if isinstance(x, list):
        return x
    if x is None:
        return []
    return [x]

def _safe_str(x: Any, default: str = "") -> str:
    if x is None:
        return default
    return str(x).strip()

def _safe_float(x: Any, default: Optional[float] = 0.0) -> Optional[float]:
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

def _normalize_heat_id(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip().replace("\\", "/").split("/")[-1]
    s = s.replace("-", "_").replace(".", "_").replace(" ", "")
    if not s:
        return ""
    for prefix in [
        "Converter1_", "Converter2_", "Converter3_", "Converter4_",
        "BOF1_", "BOF2_", "BOF3_", "BOF4_",
        "Furnace1_", "Furnace2_", "Furnace3_", "Furnace4_",
    ]:
        if s.startswith(prefix):
            s = s[len(prefix):]
    m = re.findall(r"[A-Za-z0-9]*LDG_\d+", s, flags=re.I)
    if m:
        return m[-1].upper()
    parts = [p for p in s.split("_") if p]
    if len(parts) >= 2 and parts[-2].upper().endswith("LDG") and parts[-1].isdigit():
        return f"{parts[-2].upper()}_{parts[-1]}"
    return s.upper() if "LDG" in s.upper() else s

def _canonical_stage(x: Any) -> str:
    s = str(x or "").strip()
    u = s.upper().replace(" ", "").replace("-", "_")
    if not u:
        return "Unknown"
    if "BOF" in u or "CONVERTER" in u or "LDGGENERATION" in u or re.search(r"\d#LDG", u):
        return "BOF"
    if u == "LF" or "LADLE" in u or "REFIN" in u:
        return "LF"
    if "CAST" in u or u == "CC" or u.startswith("CC"):
        return "Casting"
    if "ROLL" in u or re.fullmatch(r"11\d\d", u):
        return "Rolling"
    return s

def _stage_rank(stage: Any) -> int:
    return {"BOF": 0, "LF": 1, "Casting": 2, "Rolling": 3}.get(_canonical_stage(stage), 99)

def _pair(prev_stage: Any, next_stage: Any) -> str:
    return f"{_canonical_stage(prev_stage)}->{_canonical_stage(next_stage)}"

def _looks_calendar_based(reason: Any) -> bool:
    s = str(reason or "").lower()
    return "calendar" in s or "patch step" in s or "known future" in s

def _dedup(items: Iterable[Any]) -> List[Any]:
    out, seen = [], set()
    for x in items:
        key = json.dumps(x, ensure_ascii=False, sort_keys=True, default=str) if isinstance(x, (dict, list)) else str(x)
        if key in seen:
            continue
        seen.add(key)
        out.append(x)
    return out

class PhysicsConsistentVariationalProjector:

    def __init__(self, config: Optional[ProjectionConfig] = None) -> None:
        self.config = config or ProjectionConfig()

    def project(self, raw_segments: List[Dict[str, Any]], repair_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ctx = self._normalize_context(_safe_dict(repair_context))
        raw_segments = [x for x in _safe_list(raw_segments) if isinstance(x, dict)]

        normalized, rejected = self._normalize_and_filter(raw_segments, ctx)
        violations_before = self._violations(normalized, ctx)
        objective_before = self._objective(normalized, ctx)

        repaired = normalized
        uncertain: List[Dict[str, Any]] = []
        logs: List[Dict[str, Any]] = []

        for iteration in range(1, self.config.max_iterations + 1):
            prev_hard = self._hard_violation_count(self._violations(repaired, ctx))
            iter_log: Dict[str, Any] = {"iteration": iteration, "num_segments_in": len(repaired)}

            repaired, u1, r1, l1 = self._deduplicate_heat_stage(repaired, ctx)
            repaired, u2, r2, l2 = self._project_route_and_waiting(repaired, ctx)
            repaired, u3, r3, l3 = self._project_duration(repaired, ctx)
            repaired, u4, r4, l4 = self._project_unit_no_overlap(repaired, ctx)
            repaired, u5, r5, l5 = self._project_casting_continuity(repaired, ctx)
            repaired, u6, r6, l6 = self._project_rolling_queue(repaired, ctx)
            repaired, u7, r7, l7 = self._final_check(repaired, ctx)

            uncertain.extend(u1 + u2 + u3 + u4 + u5 + u6 + u7)
            rejected.extend(r1 + r2 + r3 + r4 + r5 + r6 + r7)
            iter_log.update({
                "num_segments_out": len(repaired),
                "steps": [l1, l2, l3, l4, l5, l6, l7],
                "violations_after_iteration": self._violations(repaired, ctx),
            })
            logs.append(iter_log)
            if self._hard_violation_count(iter_log["violations_after_iteration"]) == 0:
                break
            if self._hard_violation_count(iter_log["violations_after_iteration"]) >= prev_hard and iteration > 1:
                break

        repaired = self._finalize_repaired(repaired)
        uncertain = self._unique(uncertain)
        rejected = self._unique(rejected)
        violations_after = self._violations(repaired, ctx)
        objective_after = self._objective(repaired, ctx)
        energy_proxy = self._production_energy_proxy(repaired, ctx)

        diagnostics = {
            "method": "LLM-guided Physics-consistent Variational Projection",
            "method_short": "PCVP",
            "uses_future_target_labels": False,
            "num_raw_segments": len(raw_segments),
            "num_normalized_segments": len(normalized),
            "num_repaired_segments": len(repaired),
            "num_uncertain_segments": len(uncertain),
            "num_rejected_segments": len(rejected),
            "horizon_window": ctx["horizon_window"],
            "violations_before": violations_before,
            "violations_after": violations_after,
            "objective_terms_before": objective_before,
            "objective_terms_after": objective_after,
            "objective_total_before": self._weighted_objective(objective_before),
            "objective_total_after": self._weighted_objective(objective_after),
            "repair_iterations": len(logs),
            "iteration_logs": logs,
            "production_energy_proxy": energy_proxy,
            "projection_policy": {
                "draw_predicted_from": "repaired_segments",
                "do_not_use_as_main_prediction": ["uncertain_segments", "rejected_segments", "raw_segments_before_projection"],
            },
        }

        return {
            "repaired_segments": repaired,
            "uncertain_segments": uncertain,
            "rejected_segments": rejected,
            "repair_diagnostics": diagnostics,
            "predicted_operation_segments": repaired,
            "mechanism_projection_diagnostics": diagnostics,
        }

    def _normalize_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        horizon = _safe_dict(context.get("horizon_window") or context.get("future_time_window"))
        h0 = _safe_float(horizon.get("start_min") or horizon.get("t0") or horizon.get("start"), None)
        h1 = _safe_float(horizon.get("end_min") or horizon.get("t1") or horizon.get("end"), None)
        future_calendar = _safe_list(context.get("known_future_patch_calendar") or context.get("future_patch_calendar") or context.get("future_context"))
        if h0 is None or h1 is None:
            starts, ends = [], []
            for row in future_calendar:
                if not isinstance(row, dict):
                    continue
                s = _safe_float(row.get("start_min") or row.get("t0") or row.get("patch_start_minute_index"), None)
                e = _safe_float(row.get("end_min") or row.get("t1") or row.get("patch_end_minute_index"), None)
                if s is not None:
                    starts.append(s)
                if e is not None:
                    ends.append(e)
            h0 = min(starts) if h0 is None and starts else (h0 if h0 is not None else 0.0)
            h1 = max(ends) if h1 is None and ends else (h1 if h1 is not None else h0 + 1.0)
        if h1 <= h0:
            h1 = h0 + 1.0

        return {
            "horizon_window": {"start_min": float(h0), "end_min": float(h1)},
            "stage_order": [_canonical_stage(x) for x in _safe_list(context.get("stage_order") or ["BOF", "LF", "Casting", "Rolling"])],
            "duration_priors": self._normalize_duration_priors(_safe_dict(context.get("duration_priors") or context.get("operation_duration_priors"))),
            "waiting_priors": self._normalize_waiting_priors(_safe_dict(context.get("waiting_time_priors") or context.get("transfer_time_priors"))),
            "unit_compatibility": self._normalize_unit_compatibility(_safe_dict(context.get("unit_compatibility"))),
            "unit_capacity": self._normalize_unit_capacity(_safe_dict(context.get("unit_capacity"))),
            "history_tail_operation_segments": _safe_list(context.get("history_tail_operation_segments") or context.get("history_operation_segments")),
            "known_future_patch_calendar": future_calendar,
            "rolling_queue_prior": self._normalize_rolling_queue(context.get("rolling_queue_prior") or context.get("rolling_queue")),
            "casting_sequence_prior": self._normalize_casting_sequences(context.get("casting_sequence_prior") or context.get("cast_sequence_prior")),
            "energy_coefficients": _safe_dict(context.get("energy_coefficients") or context.get("production_energy_coefficients")),
        }

    def _normalize_duration_priors(self, priors: Dict[str, Any]) -> Dict[str, Any]:
        out = copy.deepcopy(self.config.default_duration_priors)
        for key, val in priors.items():
            if not isinstance(val, dict):
                continue
            stage = _canonical_stage(key)
            if any(isinstance(v, dict) for v in val.values()) and not all(k in val for k in ["min", "median", "max"]):
                out.setdefault(stage, {})
                for unit, p in val.items():
                    if isinstance(p, dict):
                        out[stage][_safe_str(unit)] = self._duration_prior(stage, p)
            else:
                out[stage] = self._duration_prior(stage, val)
        return out

    def _duration_prior(self, stage: str, p: Dict[str, Any]) -> Dict[str, float]:
        d = self.config.default_duration_priors.get(stage, {"min": 0.5, "median": 5.0, "max": 60.0})
        mn = float(max(self.config.min_duration_min, _safe_float(p.get("min") or p.get("p05") or p.get("q05"), d["min"]) or d["min"]))
        med = float(max(mn, _safe_float(p.get("median") or p.get("med") or p.get("mean"), d["median"]) or d["median"]))
        mx = float(max(med, _safe_float(p.get("max") or p.get("p95") or p.get("q95"), d["max"]) or d["max"]))
        return {"min": mn, "median": med, "max": mx}

    def _normalize_waiting_priors(self, priors: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        out = copy.deepcopy(self.config.default_waiting_priors)
        for key, val in priors.items():
            if not isinstance(val, dict):
                continue
            pair = str(key).replace(" ", "") if "->" in str(key) else _pair(val.get("from"), val.get("to"))
            d = out.get(pair, {"min": 0.0, "median": 0.0, "max": 120.0})
            mn = float(max(0.0, _safe_float(val.get("min") or val.get("min_gap"), d["min"]) or 0.0))
            med = float(max(mn, _safe_float(val.get("median") or val.get("mean"), d["median"]) or mn))
            mx = float(max(med, _safe_float(val.get("max") or val.get("max_gap"), d["max"]) or med))
            out[pair] = {"min": mn, "median": med, "max": mx}
        return out

    def _normalize_unit_compatibility(self, comp: Dict[str, Any]) -> Dict[str, List[str]]:
        out = {
            "BOF": ["BOF", "Converter1", "Converter2", "Converter3", "Converter4", "1#LDG Generation", "2#LDG Generation", "3#LDG Generation", "4#LDG Generation"],
            "LF": ["LF", "LF1", "LF2", "LadleFurnace"],
            "Casting": ["Casting", "CC", "CC1", "CC2", "Caster", "Caster1", "Caster2"],
            "Rolling": ["Rolling", "1130", "1131", "1132"],
        }
        for stage, units in comp.items():
            st = _canonical_stage(stage)
            vals = list(units.keys()) if isinstance(units, dict) else [_safe_str(x) for x in _safe_list(units)]
            if vals:
                out[st] = _dedup(out.get(st, []) + vals)
        return out

    def _normalize_unit_capacity(self, cap: Dict[str, Any]) -> Dict[str, int]:
        out = {"BOF": 1, "LF": 1, "Casting": 1, "Rolling": 1, "Converter1": 1, "Converter2": 1, "Converter3": 1, "1130": 1, "1131": 1, "1132": 1}
        for unit, value in cap.items():
            out[_safe_str(unit)] = max(1, int(_safe_float(value, 1) or 1))
        return out

    def _normalize_rolling_queue(self, prior: Any) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        if isinstance(prior, dict):
            for unit, heats in prior.items():
                out[_safe_str(unit)] = [_normalize_heat_id(x) for x in _safe_list(heats) if _normalize_heat_id(x)]
        elif isinstance(prior, list):
            for row in prior:
                if not isinstance(row, dict):
                    continue
                unit = _safe_str(row.get("unit") or row.get("rolling_unit") or "Rolling")
                hid = _normalize_heat_id(row.get("heat_id") or row.get("hn"))
                if hid:
                    out.setdefault(unit, []).append(hid)
        return {u: _dedup(v) for u, v in out.items()}

    def _normalize_casting_sequences(self, prior: Any) -> List[Dict[str, Any]]:
        if isinstance(prior, dict) and isinstance(prior.get("groups"), list):
            prior = prior["groups"]
        groups: List[Dict[str, Any]] = []
        if isinstance(prior, dict):
            for unit, heats in prior.items():
                groups.append({"group_id": f"cast_{unit}", "unit": _safe_str(unit), "heat_ids": [_normalize_heat_id(x) for x in _safe_list(heats)], "max_gap": 1.0})
        elif isinstance(prior, list):
            for i, row in enumerate(prior):
                if not isinstance(row, dict):
                    continue
                heats = row.get("heat_ids") or row.get("heats") or row.get("sequence")
                groups.append({
                    "group_id": row.get("group_id") or row.get("cast_id") or f"cast_group_{i+1}",
                    "unit": row.get("unit") or row.get("caster") or row.get("cc") or "Casting",
                    "heat_ids": [_normalize_heat_id(x) for x in _safe_list(heats) if _normalize_heat_id(x)],
                    "max_gap": float(_safe_float(row.get("max_gap") or row.get("epsilon"), 1.0) or 1.0),
                })
        return [g for g in groups if g.get("heat_ids")]

    def _normalize_and_filter(self, raw_segments: List[Dict[str, Any]], ctx: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        repaired, rejected = [], []
        h0, h1 = ctx["horizon_window"]["start_min"], ctx["horizon_window"]["end_min"]
        for idx, raw in enumerate(raw_segments, start=1):
            seg = self._normalize_segment(raw, idx)
            if seg is None:
                bad = copy.deepcopy(raw)
                bad.update({"projection_status": "rejected", "reject_reason": "cannot_parse_segment"})
                rejected.append(bad)
                continue
            if seg["end_min"] <= h0 or seg["start_min"] >= h1:
                seg.update({"projection_status": "rejected", "reject_reason": "outside_horizon"})
                rejected.append(seg)
                continue
            if self.config.clip_to_horizon:
                seg["start_min"] = max(seg["start_min"], h0)
                seg["end_min"] = min(seg["end_min"], h1)
                seg["duration_min"] = seg["end_min"] - seg["start_min"]
            if seg["duration_min"] < self.config.min_duration_min:
                seg.update({"projection_status": "rejected", "reject_reason": "too_short_after_clip"})
                rejected.append(seg)
                continue
            seg = self._repair_unit(seg, ctx)
            if seg.get("projection_status") == "rejected":
                rejected.append(seg)
            else:
                repaired.append(seg)
        return self._sort_segments(repaired), rejected

    def _normalize_segment(self, raw: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        stage = _canonical_stage(raw.get("stage") or raw.get("process") or raw.get("operation"))
        if stage not in {"BOF", "LF", "Casting", "Rolling"}:
            return None
        heat_id = _normalize_heat_id(raw.get("heat_id") or raw.get("canonical_heat_id") or raw.get("hn") or raw.get("id")) or f"UNKNOWN_HEAT_{idx}"
        s = _safe_float(raw.get("start_min") or raw.get("start") or raw.get("t0"), None)
        e = _safe_float(raw.get("end_min") or raw.get("end") or raw.get("t1"), None)
        d = _safe_float(raw.get("duration_min") or raw.get("duration"), None)
        if s is None and e is not None and d is not None:
            s = e - d
        if e is None and s is not None and d is not None:
            e = s + d
        if s is None or e is None:
            return None
        if e <= s:
            if d is None or d <= self.config.min_duration_min:
                return None
            e = s + d
        unit = _safe_str(raw.get("unit") or raw.get("resource") or raw.get("machine") or raw.get("caster") or raw.get("rolling_unit")) or self._default_unit(stage)
        conf = _safe_float(raw.get("confidence"), None)
        if conf is None:
            conf = 0.9 if _looks_calendar_based(raw.get("reason")) else 0.6
        seg = copy.deepcopy(raw)
        seg.update({
            "segment_id": raw.get("segment_id") or raw.get("id") or f"raw_seg_{idx:03d}",
            "heat_id": heat_id,
            "stage": stage,
            "unit": unit,
            "start_min": float(s),
            "end_min": float(e),
            "duration_min": float(e - s),
            "raw_start_min_before_projection": float(s),
            "raw_end_min_before_projection": float(e),
            "raw_duration_min_before_projection": float(e - s),
            "confidence": float(conf),
            "source": raw.get("source") or "model_output",
            "source_kind": raw.get("source_kind") or raw.get("source") or "model_output",
            "is_valid_gantt_prediction": True,
            "projection_status": "candidate",
            "projection_notes": list(_safe_list(raw.get("projection_notes"))),
        })
        return seg

    def _default_unit(self, stage: str) -> str:
        return {"BOF": "BOF", "LF": "LF", "Casting": "Casting", "Rolling": "Rolling"}.get(stage, stage)

    def _repair_unit(self, seg: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
        stage, unit = seg["stage"], _safe_str(seg.get("unit"))
        if self._unit_compatible(stage, unit, ctx["unit_compatibility"].get(stage, [])):
            return seg
        if self.config.allow_unit_inference:
            inferred = self._infer_unit(seg, ctx)
            if inferred:
                seg["raw_unit_before_projection"] = unit
                seg["unit"] = inferred
                seg["unit_inferred_by_projection"] = True
                seg["projection_notes"].append(f"unit_inferred:{unit}->{inferred}")
                return seg
        seg.update({"projection_status": "rejected", "reject_reason": f"incompatible_unit:{stage}:{unit}"})
        return seg

    def _unit_compatible(self, stage: str, unit: str, compat: List[str]) -> bool:
        if unit in compat:
            return True
        u = unit.lower()
        if stage == "BOF":
            return "converter" in u or "ldg" in u or u.startswith("bof")
        if stage == "LF":
            return u.startswith("lf") or "ladle" in u
        if stage == "Casting":
            return "cast" in u or u.startswith("cc") or "caster" in u
        if stage == "Rolling":
            return "roll" in u or bool(re.fullmatch(r"11\d\d", unit))
        return False

    def _infer_unit(self, seg: Dict[str, Any], ctx: Dict[str, Any]) -> Optional[str]:
        stage, hid = seg["stage"], _normalize_heat_id(seg.get("heat_id"))
        for hist in ctx.get("history_tail_operation_segments", []):
            if not isinstance(hist, dict):
                continue
            if _normalize_heat_id(hist.get("heat_id") or hist.get("hn")) == hid and _canonical_stage(hist.get("stage")) == stage:
                u = _safe_str(hist.get("unit"))
                if u:
                    return u
        if stage == "BOF":
            for row in ctx.get("known_future_patch_calendar", []):
                if not isinstance(row, dict):
                    continue
                rh = _normalize_heat_id(row.get("heat_id") or row.get("hn") or row.get("canonical_heat_id"))
                if rh == hid:
                    u = _safe_str(row.get("unit") or row.get("converter") or row.get("main_converter_name"))
                    if u:
                        return u
        compat = ctx["unit_compatibility"].get(stage, [])
        return compat[0] if compat else self._default_unit(stage)

    def _deduplicate_heat_stage(self, segs: List[Dict[str, Any]], ctx: Dict[str, Any]):
        groups = defaultdict(list)
        for s in segs:
            groups[(_normalize_heat_id(s.get("heat_id")), s.get("stage"))].append(s)
        kept, uncertain = [], []
        duplicates = 0
        for key, group in groups.items():
            if len(group) == 1:
                kept.append(group[0])
                continue
            duplicates += 1
            group = sorted(group, key=lambda x: (x.get("confidence", 0), 1 if _looks_calendar_based(x.get("reason")) else 0, x.get("duration_min", 0)), reverse=True)
            best = copy.deepcopy(group[0])
            best["projection_notes"].append(f"duplicate_heat_stage_keep_best:{key}")
            kept.append(best)
            for g in group[1:]:
                x = copy.deepcopy(g)
                x.update({"projection_status": "uncertain", "uncertain_reason": "duplicate_heat_stage_lower_priority"})
                uncertain.append(x)
        return self._sort_segments(kept), uncertain, [], {"step": "deduplicate_heat_stage", "duplicate_groups": duplicates}

    def _project_route_and_waiting(self, segs: List[Dict[str, Any]], ctx: Dict[str, Any]):
        by_heat = defaultdict(list)
        for s in segs:
            by_heat[_normalize_heat_id(s.get("heat_id"))].append(copy.deepcopy(s))
        repaired, uncertain = [], []
        shifts = wait_viol = order_viol = 0
        for hid, chain in by_heat.items():
            chain.sort(key=lambda x: (_stage_rank(x.get("stage")), x.get("start_min", 0)))
            accepted: List[Dict[str, Any]] = []
            for seg in chain:
                prev = None
                for p in reversed(accepted):
                    if _stage_rank(p.get("stage")) < _stage_rank(seg.get("stage")):
                        prev = p
                        break
                if prev is None:
                    accepted.append(seg)
                    continue
                pair = _pair(prev.get("stage"), seg.get("stage"))
                pri = ctx["waiting_priors"].get(pair, {"min": 0.0, "max": 120.0})
                min_s = prev["end_min"] + pri.get("min", 0.0)
                max_s = prev["end_min"] + pri.get("max", 120.0)
                dur = seg["duration_min"]
                if seg["start_min"] < min_s:
                    order_viol += 1
                    shifts += 1
                    seg["raw_start_min_before_route_projection"] = seg["start_min"]
                    seg["raw_end_min_before_route_projection"] = seg["end_min"]
                    seg["start_min"] = min_s
                    seg["end_min"] = min_s + dur
                    seg["route_order_repaired"] = True
                    seg["projection_notes"].append(f"shifted_after_previous_stage:{pair}")
                if seg["start_min"] > max_s:
                    wait_viol += 1
                    if self.config.uncertain_on_waiting_violation and seg.get("confidence", 0) < self.config.high_confidence_threshold:
                        x = copy.deepcopy(seg)
                        x.update({"projection_status": "uncertain", "uncertain_reason": f"waiting_window_exceeded:{pair}"})
                        uncertain.append(x)
                        continue
                    seg["waiting_window_violation_retained"] = True
                accepted.append(seg)
            repaired.extend(accepted)
        return self._sort_segments(repaired), uncertain, [], {"step": "project_route_and_waiting", "route_shifts": shifts, "order_violations": order_viol, "waiting_violations": wait_viol}

    def _project_duration(self, segs: List[Dict[str, Any]], ctx: Dict[str, Any]):
        repaired = []
        projected = 0
        for seg in segs:
            seg = copy.deepcopy(seg)
            p = self._duration_for(seg, ctx)
            dur = seg["end_min"] - seg["start_min"]
            if dur < p["min"] or dur > p["max"]:
                if seg["stage"] == "BOF" and _looks_calendar_based(seg.get("reason")):
                    seg["duration_prior_violation_retained"] = True
                elif self.config.allow_duration_projection:
                    new_dur = min(max(dur, p["min"]), p["max"])
                    if dur < p["min"] * 0.5 or dur > p["max"] * 1.5:
                        new_dur = p["median"]
                    seg["raw_duration_min_before_duration_projection"] = dur
                    seg["end_min"] = seg["start_min"] + new_dur
                    seg["duration_min"] = new_dur
                    seg["duration_prior_projected"] = True
                    projected += 1
            repaired.append(seg)
        return self._sort_segments(repaired), [], [], {"step": "project_duration", "duration_projected": projected}

    def _project_unit_no_overlap(self, segs: List[Dict[str, Any]], ctx: Dict[str, Any]):
        by_unit = defaultdict(list)
        for s in segs:
            by_unit[_safe_str(s.get("unit"))].append(copy.deepcopy(s))
        repaired, rejected = [], []
        shifts = rejects = 0
        h1 = ctx["horizon_window"]["end_min"]
        for unit, group in by_unit.items():
            cap = int(ctx["unit_capacity"].get(unit, 1))
            group.sort(key=lambda x: (x["start_min"], -x.get("confidence", 0), _stage_rank(x.get("stage"))))
            if cap > 1:
                repaired.extend(group)
                continue
            last_end = None
            for seg in group:
                if last_end is not None and seg["start_min"] < last_end:
                    if self._fixed_anchor(seg):
                        seg["unit_overlap_violation_retained"] = True
                    else:
                        dur = seg["duration_min"]
                        seg["raw_start_min_before_unit_projection"] = seg["start_min"]
                        seg["raw_end_min_before_unit_projection"] = seg["end_min"]
                        seg["start_min"] = last_end
                        seg["end_min"] = last_end + dur
                        seg["unit_overlap_repaired"] = True
                        shifts += 1
                if seg["end_min"] > h1:
                    if self.config.clip_to_horizon and seg["start_min"] < h1:
                        seg["end_min"] = h1
                        seg["duration_min"] = seg["end_min"] - seg["start_min"]
                        seg["is_clipped_end"] = True
                    if seg["duration_min"] < self.config.min_duration_min:
                        seg.update({"projection_status": "rejected", "reject_reason": "too_short_after_unit_shift"})
                        rejected.append(seg)
                        rejects += 1
                        continue
                repaired.append(seg)
                last_end = max(last_end or seg["end_min"], seg["end_min"])
        return self._sort_segments(repaired), [], rejected, {"step": "project_unit_no_overlap", "unit_overlap_shifts": shifts, "unit_overlap_rejected": rejects}

    def _project_casting_continuity(self, segs: List[Dict[str, Any]], ctx: Dict[str, Any]):
        groups = ctx.get("casting_sequence_prior", [])
        if not groups:
            return segs, [], [], {"step": "project_casting_continuity", "casting_groups": 0, "repairs": 0, "skipped": True}
        segs = [copy.deepcopy(s) for s in segs]
        idx = {(_normalize_heat_id(s.get("heat_id")), s.get("stage")): s for s in segs if s.get("stage") == "Casting"}
        repairs = 0
        for group in groups:
            unit = _safe_str(group.get("unit") or "Casting")
            max_gap = float(group.get("max_gap", 1.0))
            prev = None
            for hid in group.get("heat_ids", []):
                seg = idx.get((_normalize_heat_id(hid), "Casting"))
                if seg is None:
                    continue
                if unit and seg.get("unit") != unit:
                    seg["raw_unit_before_casting_projection"] = seg.get("unit")
                    seg["unit"] = unit
                    seg["casting_unit_projected"] = True
                    repairs += 1
                if prev is not None:
                    gap = seg["start_min"] - prev["end_min"]
                    if gap < 0 or gap > max_gap:
                        dur = seg["duration_min"]
                        seg["raw_start_min_before_casting_projection"] = seg["start_min"]
                        seg["raw_end_min_before_casting_projection"] = seg["end_min"]
                        seg["start_min"] = prev["end_min"] + max(0.0, min(gap, max_gap))
                        seg["end_min"] = seg["start_min"] + dur
                        seg["casting_continuity_repaired"] = True
                        repairs += 1
                prev = seg
        return self._sort_segments(segs), [], [], {"step": "project_casting_continuity", "casting_groups": len(groups), "repairs": repairs}

    def _project_rolling_queue(self, segs: List[Dict[str, Any]], ctx: Dict[str, Any]):
        rolling = [copy.deepcopy(s) for s in segs if s.get("stage") == "Rolling"]
        others = [copy.deepcopy(s) for s in segs if s.get("stage") != "Rolling"]
        rolling, split_log = self._split_shared_rolling_windows(rolling, ctx)
        by_unit = defaultdict(list)
        for s in rolling:
            by_unit[_safe_str(s.get("unit") or "Rolling")].append(s)
        repaired = []
        queue_repairs = 0
        for unit, group in by_unit.items():
            queue = ctx.get("rolling_queue_prior", {}).get(unit, [])
            order = {h: i for i, h in enumerate(queue)}
            group.sort(key=lambda x: (order.get(_normalize_heat_id(x.get("heat_id")), 10**6), x["start_min"], _normalize_heat_id(x.get("heat_id"))) if queue else (x["start_min"], _normalize_heat_id(x.get("heat_id"))))
            last_end = None
            for seg in group:
                if last_end is not None and seg["start_min"] < last_end:
                    dur = seg["duration_min"]
                    seg["raw_start_min_before_rolling_projection"] = seg["start_min"]
                    seg["raw_end_min_before_rolling_projection"] = seg["end_min"]
                    seg["start_min"] = last_end
                    seg["end_min"] = last_end + dur
                    seg["rolling_queue_repaired"] = True
                    queue_repairs += 1
                repaired.append(seg)
                last_end = max(last_end or seg["end_min"], seg["end_min"])
        log = {"step": "project_rolling_queue", "queue_repairs": queue_repairs, **split_log}
        return self._sort_segments(others + repaired), [], [], log

    def _split_shared_rolling_windows(self, rolling: List[Dict[str, Any]], ctx: Dict[str, Any]):
        groups = defaultdict(list)
        for s in rolling:
            groups[(_safe_str(s.get("unit")), round(s["start_min"], 6), round(s["end_min"], 6))].append(s)
        out = []
        split_groups = split_segments = 0
        for (unit, s, e), group in groups.items():
            if len(group) <= 1:
                out.extend(group)
                continue
            split_groups += 1
            queue = ctx.get("rolling_queue_prior", {}).get(unit, [])
            order = {h: i for i, h in enumerate(queue)}
            group.sort(key=lambda x: (order.get(_normalize_heat_id(x.get("heat_id")), 10**6), _normalize_heat_id(x.get("heat_id"))))
            span = e - s
            slot = span / max(len(group), 1)
            for i, seg in enumerate(group):
                seg = copy.deepcopy(seg)
                ns, ne = s + i * slot, s + (i + 1) * slot
                seg["raw_start_min_before_rolling_split"] = seg["start_min"]
                seg["raw_end_min_before_rolling_split"] = seg["end_min"]
                seg["start_min"] = ns
                seg["end_min"] = ne
                seg["duration_min"] = ne - ns
                seg["rolling_shared_window_split"] = True
                seg["rolling_split_group_size"] = len(group)
                out.append(seg)
                split_segments += 1
        return out, {"rolling_shared_window_groups_split": split_groups, "rolling_shared_window_segments_split": split_segments}

    def _final_check(self, segs: List[Dict[str, Any]], ctx: Dict[str, Any]):
        h0, h1 = ctx["horizon_window"]["start_min"], ctx["horizon_window"]["end_min"]
        repaired, uncertain, rejected = [], [], []
        clipped = 0
        for seg in segs:
            seg = copy.deepcopy(seg)
            if seg["end_min"] <= h0 or seg["start_min"] >= h1:
                seg.update({"projection_status": "rejected", "reject_reason": "outside_horizon_after_projection"})
                rejected.append(seg)
                continue
            old_s, old_e = seg["start_min"], seg["end_min"]
            seg["start_min"] = max(h0, seg["start_min"])
            seg["end_min"] = min(h1, seg["end_min"])
            seg["duration_min"] = seg["end_min"] - seg["start_min"]
            if old_s != seg["start_min"] or old_e != seg["end_min"]:
                seg["final_horizon_clipped"] = True
                clipped += 1
            if seg["duration_min"] < self.config.min_duration_min:
                seg.update({"projection_status": "rejected", "reject_reason": "too_short_final"})
                rejected.append(seg)
            elif seg.get("confidence", 1.0) < self.config.reject_low_confidence_threshold:
                seg.update({"projection_status": "uncertain", "uncertain_reason": "confidence_below_threshold"})
                uncertain.append(seg)
            else:
                repaired.append(seg)
        return self._sort_segments(repaired), uncertain, rejected, {"step": "final_check", "final_clipped": clipped, "final_rejected": len(rejected)}

    def _violations(self, segs: List[Dict[str, Any]], ctx: Dict[str, Any]) -> Dict[str, int]:
        v = {"outside_horizon": 0, "non_positive_duration": 0, "duration_prior": 0, "stage_order": 0, "waiting_window": 0, "unit_overlap": 0, "duplicate_heat_stage": 0, "casting_continuity": 0, "rolling_shared_window": 0, "rolling_queue": 0, "incompatible_unit": 0}
        h0, h1 = ctx["horizon_window"]["start_min"], ctx["horizon_window"]["end_min"]
        for s in segs:
            if s["start_min"] < h0 or s["end_min"] > h1:
                v["outside_horizon"] += 1
            if s["end_min"] <= s["start_min"]:
                v["non_positive_duration"] += 1
            p = self._duration_for(s, ctx)
            if s["duration_min"] < p["min"] or s["duration_min"] > p["max"]:
                v["duration_prior"] += 1
            if not self._unit_compatible(s["stage"], _safe_str(s.get("unit")), ctx["unit_compatibility"].get(s["stage"], [])):
                v["incompatible_unit"] += 1
        counts = defaultdict(int)
        for s in segs:
            counts[(_normalize_heat_id(s.get("heat_id")), s.get("stage"))] += 1
        v["duplicate_heat_stage"] = sum(max(0, c - 1) for c in counts.values())
        by_heat = defaultdict(list)
        for s in segs:
            by_heat[_normalize_heat_id(s.get("heat_id"))].append(s)
        for chain in by_heat.values():
            chain.sort(key=lambda x: _stage_rank(x.get("stage")))
            for a, b in zip(chain, chain[1:]):
                if _stage_rank(a.get("stage")) >= _stage_rank(b.get("stage")):
                    continue
                pri = ctx["waiting_priors"].get(_pair(a.get("stage"), b.get("stage")), {"min": 0, "max": 120})
                gap = b["start_min"] - a["end_min"]
                if gap < pri.get("min", 0):
                    v["stage_order"] += 1
                if gap < pri.get("min", 0) or gap > pri.get("max", 120):
                    v["waiting_window"] += 1
        by_unit = defaultdict(list)
        for s in segs:
            by_unit[_safe_str(s.get("unit"))].append(s)
        for unit, group in by_unit.items():
            if int(ctx["unit_capacity"].get(unit, 1)) > 1:
                continue
            group.sort(key=lambda x: x["start_min"])
            for a, b in zip(group, group[1:]):
                if b["start_min"] < a["end_min"]:
                    v["unit_overlap"] += 1
        roll_groups = defaultdict(int)
        for s in segs:
            if s.get("stage") == "Rolling":
                roll_groups[(_safe_str(s.get("unit")), round(s["start_min"], 6), round(s["end_min"], 6))] += 1
        v["rolling_shared_window"] = sum(max(0, c - 1) for c in roll_groups.values())
        v["casting_continuity"] = self._casting_violations(segs, ctx)
        v["rolling_queue"] = self._rolling_queue_violations(segs, ctx)
        return v

    def _hard_violation_count(self, v: Dict[str, int]) -> int:
        return sum(v.get(k, 0) for k in ["outside_horizon", "non_positive_duration", "stage_order", "unit_overlap", "duplicate_heat_stage", "rolling_shared_window", "incompatible_unit"])

    def _objective(self, segs: List[Dict[str, Any]], ctx: Dict[str, Any]) -> Dict[str, float]:
        terms = {"llm_deviation": 0.0, "duration_prior": 0.0, "stage_order": 0.0, "waiting_window": 0.0, "unit_overlap": 0.0, "casting_continuity": 0.0, "rolling_queue": 0.0, "energy_proxy": 0.0}
        for s in segs:
            rs = _safe_float(s.get("raw_start_min_before_projection"), s["start_min"]) or s["start_min"]
            re = _safe_float(s.get("raw_end_min_before_projection"), s["end_min"]) or s["end_min"]
            terms["llm_deviation"] += (s["start_min"] - rs) ** 2 + (s["end_min"] - re) ** 2
            p = self._duration_for(s, ctx)
            med = max(p.get("median", 1.0), 1e-6)
            terms["duration_prior"] += ((s["duration_min"] - med) / med) ** 2
        v = self._violations(segs, ctx)
        for k in ["stage_order", "waiting_window", "unit_overlap", "casting_continuity", "rolling_queue"]:
            terms[k] = float(v.get(k, 0))
        terms["energy_proxy"] = float(self._production_energy_proxy(segs, ctx).get("proxy_violation_score", 0.0))
        return {k: float(v) for k, v in terms.items()}

    def _weighted_objective(self, terms: Dict[str, float]) -> float:
        w = self.config.objective_weights
        return float(sum(terms.get(k, 0.0) * w.get(k, 1.0) for k in terms))

    def _production_energy_proxy(self, segs: List[Dict[str, Any]], ctx: Dict[str, Any]) -> Dict[str, Any]:
        coeff = _safe_dict(ctx.get("energy_coefficients"))
        if not coeff:
            return {"available": False, "proxy_violation_score": 0.0, "stage_counts": self._stage_counts(segs), "time_buckets": []}
        h0, h1 = ctx["horizon_window"]["start_min"], ctx["horizon_window"]["end_min"]
        bucket = float(_safe_float(coeff.get("bucket_min"), 5.0) or 5.0)
        stage_coeff = _safe_dict(coeff.get("stage_coefficients") or coeff.get("by_stage"))
        limits = _safe_dict(coeff.get("max_load") or coeff.get("proxy_limits"))
        n = max(1, int(math.ceil((h1 - h0) / bucket)))
        buckets, score = [], 0.0
        for i in range(n):
            bs, be = h0 + i * bucket, min(h1, h0 + (i + 1) * bucket)
            active = [s for s in segs if s["start_min"] < be and s["end_min"] > bs]
            proxy = {"power": 0.0, "steam": 0.0, "ldg_byproduct": 0.0}
            for s in active:
                c = _safe_dict(stage_coeff.get(s["stage"]))
                overlap = max(0.0, min(be, s["end_min"]) - max(bs, s["start_min"])) / max(bucket, 1e-9)
                proxy["power"] += float(_safe_float(c.get("power"), 0.0) or 0.0) * overlap
                proxy["steam"] += float(_safe_float(c.get("steam"), 0.0) or 0.0) * overlap
                proxy["ldg_byproduct"] += float(_safe_float(c.get("ldg_byproduct"), 0.0) or 0.0) * overlap
            for k, val in proxy.items():
                lim = _safe_float(limits.get(k), None)
                if lim is not None and val > lim:
                    score += (val - lim) / max(lim, 1e-9)
            buckets.append({"bucket_index": i + 1, "start_min": bs, "end_min": be, "num_active_segments": len(active), "active_stage_counts": self._stage_counts(active), "proxy": {k: round(v, 6) for k, v in proxy.items()}})
        return {"available": True, "proxy_violation_score": round(score, 6), "stage_counts": self._stage_counts(segs), "time_buckets": buckets}

    def _duration_for(self, seg: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, float]:
        stage, unit = seg.get("stage"), _safe_str(seg.get("unit"))
        p = ctx["duration_priors"].get(stage, self.config.default_duration_priors.get(stage))
        if isinstance(p, dict) and unit in p and isinstance(p[unit], dict):
            return p[unit]
        if isinstance(p, dict) and all(k in p for k in ["min", "median", "max"]):
            return p
        return self.config.default_duration_priors.get(stage, {"min": 0.5, "median": 5.0, "max": 60.0})

    def _fixed_anchor(self, seg: Dict[str, Any]) -> bool:
        return bool(seg.get("is_fixed_anchor")) or bool(seg.get("is_history_carryover")) or (seg.get("stage") == "BOF" and _looks_calendar_based(seg.get("reason")) and seg.get("confidence", 0) >= self.config.high_confidence_threshold)

    def _casting_violations(self, segs: List[Dict[str, Any]], ctx: Dict[str, Any]) -> int:
        groups = ctx.get("casting_sequence_prior", [])
        if not groups:
            return 0
        idx = {(_normalize_heat_id(s.get("heat_id")), "Casting"): s for s in segs if s.get("stage") == "Casting"}
        count = 0
        for g in groups:
            prev = None
            max_gap = float(g.get("max_gap", 1.0))
            for hid in g.get("heat_ids", []):
                s = idx.get((_normalize_heat_id(hid), "Casting"))
                if s is None:
                    continue
                if prev is not None:
                    gap = s["start_min"] - prev["end_min"]
                    if gap < 0 or gap > max_gap:
                        count += 1
                prev = s
        return count

    def _rolling_queue_violations(self, segs: List[Dict[str, Any]], ctx: Dict[str, Any]) -> int:
        count = 0
        for unit, queue in ctx.get("rolling_queue_prior", {}).items():
            order = {h: i for i, h in enumerate(queue)}
            group = [s for s in segs if s.get("stage") == "Rolling" and _safe_str(s.get("unit")) == unit and _normalize_heat_id(s.get("heat_id")) in order]
            group.sort(key=lambda x: x["start_min"])
            seq = [order[_normalize_heat_id(s.get("heat_id"))] for s in group]
            for a, b in zip(seq, seq[1:]):
                if a > b:
                    count += 1
        return count

    def _stage_counts(self, segs: List[Dict[str, Any]]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for s in segs:
            st = _canonical_stage(s.get("stage"))
            out[st] = out.get(st, 0) + 1
        return out

    def _finalize_repaired(self, segs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for i, s in enumerate(self._sort_segments(segs), start=1):
            x = copy.deepcopy(s)
            x["projection_id"] = f"pcvp_seg_{i:03d}"
            x["projection_method"] = "LLM-guided Physics-consistent Variational Projection"
            x["projection_status"] = "repaired"
            x["duration_min"] = float(x["end_min"] - x["start_min"])
            x["source"] = x.get("source") or "model_output"
            x["source_kind"] = x.get("source_kind") or x.get("source") or "model_output"
            x["is_valid_gantt_prediction"] = True
            x["projection_notes"] = _dedup(_safe_list(x.get("projection_notes")))
            out.append(x)
        return out

    def _unique(self, segs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out, seen = [], set()
        for s in segs:
            key = (s.get("segment_id"), _normalize_heat_id(s.get("heat_id")), s.get("stage"), _safe_str(s.get("unit")), round(float(_safe_float(s.get("start_min"), 0) or 0), 6), round(float(_safe_float(s.get("end_min"), 0) or 0), 6), s.get("reject_reason") or s.get("uncertain_reason") or "")
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
        return out

    def _sort_segments(self, segs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(segs, key=lambda x: (float(x.get("start_min", 0)), _stage_rank(x.get("stage")), _safe_str(x.get("unit")), _normalize_heat_id(x.get("heat_id"))))

MechanismScheduleProjector = PhysicsConsistentVariationalProjector
MechanismConstrainedScheduleProjector = PhysicsConsistentVariationalProjector
LLMGuidedPhysicsConsistentVariationalProjection = PhysicsConsistentVariationalProjector

__all__ = [
    "ProjectionConfig",
    "PhysicsConsistentVariationalProjector",
    "MechanismScheduleProjector",
    "MechanismConstrainedScheduleProjector",
    "LLMGuidedPhysicsConsistentVariationalProjection",
]

_BasePhysicsConsistentVariationalProjector = PhysicsConsistentVariationalProjector

class PhysicsConsistentVariationalProjector(_BasePhysicsConsistentVariationalProjector):

    _STAGES = ("BOF", "LF", "Casting", "Rolling")

    def _normalize_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        ctx = super()._normalize_context(context)
        raw = _safe_dict(context)

        cands = _safe_dict(raw.get("window_evidence_candidates"))
        stage_bounds = _safe_dict(raw.get("window_stage_evidence_bounds"))

        normalized_cands = {
            "confirmed": [x for x in _safe_list(cands.get("confirmed")) if isinstance(x, dict)],
            "optional": [x for x in _safe_list(cands.get("optional")) if isinstance(x, dict)],
            "route_future_only": [x for x in _safe_list(cands.get("route_future_only")) if isinstance(x, dict)],
        }

        bounds: Dict[str, Dict[str, int]] = {}
        for st in self._STAGES:
            row = _safe_dict(stage_bounds.get(st))
            lower = row.get("lower", 0)
            upper = row.get("upper", lower)
            try:
                lower_i = int(float(lower or 0))
            except Exception:
                lower_i = 0
            try:
                upper_i = int(float(upper if upper is not None else lower_i))
            except Exception:
                upper_i = lower_i

            if upper_i > 10000:
                upper_i = lower_i
            upper_i = max(0, upper_i)
            lower_i = max(0, min(lower_i, upper_i))
            bounds[st] = {"lower": lower_i, "upper": upper_i}

        ctx["window_evidence_candidates"] = normalized_cands
        ctx["window_stage_evidence_bounds"] = bounds
        ctx["window_evidence_enabled"] = bool(any(normalized_cands.values()) or stage_bounds)
        ctx["window_evidence_uses_future_target_labels"] = False
        return ctx

    def _candidate_key2(self, heat_id: Any, stage: Any) -> str:
        return f"{_normalize_heat_id(heat_id)}|{_canonical_stage(stage)}"

    def _seg_key2(self, seg: Dict[str, Any]) -> str:
        return self._candidate_key2(
            seg.get("heat_id") or seg.get("canonical_heat_id") or seg.get("id") or seg.get("hn"),
            seg.get("stage") or seg.get("process") or seg.get("operation"),
        )

    def _candidate_key_sets(self, ctx: Dict[str, Any]) -> Dict[str, set]:
        cands = _safe_dict(ctx.get("window_evidence_candidates"))
        out: Dict[str, set] = {}
        for name in ["confirmed", "optional", "route_future_only"]:
            keys = set()
            for row in _safe_list(cands.get(name)):
                if not isinstance(row, dict):
                    continue
                hid = row.get("heat_id")
                st = row.get("stage")
                if hid and st:
                    keys.add(self._candidate_key2(hid, st))
            out[name] = keys
        out["allowed_main"] = out.get("confirmed", set()) | out.get("optional", set())
        return out

    def _move_to_uncertain(self, seg: Dict[str, Any], reason: str) -> Dict[str, Any]:
        u = copy.deepcopy(seg)
        u["projection_status"] = "uncertain"
        u["source"] = u.get("source") or "model_output_uncertain"
        u["source_kind"] = u.get("source_kind") or "model_output_uncertain"
        u["uncertain_reason"] = reason
        u["is_valid_gantt_prediction"] = False
        return u

    def _pre_filter_by_window_candidates(
        self,
        raw_segments: List[Dict[str, Any]],
        ctx: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        if not ctx.get("window_evidence_enabled"):
            return raw_segments, [], {
                "enabled": False,
                "num_route_future_only_moved": 0,
                "num_outside_candidate_moved": 0,
            }

        keys = self._candidate_key_sets(ctx)
        allowed = keys.get("allowed_main", set())
        route_future = keys.get("route_future_only", set())

        filtered: List[Dict[str, Any]] = []
        uncertain: List[Dict[str, Any]] = []
        n_route = 0
        n_outside = 0

        for seg in raw_segments:
            if not isinstance(seg, dict):
                continue
            stage = _canonical_stage(seg.get("stage") or seg.get("process") or seg.get("operation"))
            key = self._seg_key2(seg)

            if key in route_future and key not in allowed:
                uncertain.append(self._move_to_uncertain(seg, "route_future_only_candidate_not_window_confirmed"))
                n_route += 1
                continue

            if allowed and key not in allowed:
                uncertain.append(self._move_to_uncertain(seg, "outside_confirmed_optional_window_candidate_set"))
                n_outside += 1
                continue

            filtered.append(seg)

        return filtered, uncertain, {
            "enabled": True,
            "num_route_future_only_moved": n_route,
            "num_outside_candidate_moved": n_outside,
        }

    def _apply_stage_upper_bounds(
        self,
        repaired: List[Dict[str, Any]],
        uncertain: List[Dict[str, Any]],
        ctx: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        bounds = _safe_dict(ctx.get("window_stage_evidence_bounds"))
        if not ctx.get("window_evidence_enabled") or not bounds:
            return repaired, uncertain, {
                "enabled": False,
                "over_upper_bound_count": 0,
                "miss_lower_bound_count": 0,
                "num_moved_to_uncertain_by_upper_bound": 0,
                "stage_bounds": {},
            }

        kept: List[Dict[str, Any]] = []
        moved: List[Dict[str, Any]] = []

        by_stage: Dict[str, List[Dict[str, Any]]] = {st: [] for st in self._STAGES}
        other: List[Dict[str, Any]] = []
        for seg in repaired:
            st = _canonical_stage(seg.get("stage") or seg.get("process") or seg.get("operation"))
            if st in by_stage:
                by_stage[st].append(seg)
            else:
                other.append(seg)

        coverage = {}
        over_total = 0
        miss_total = 0
        moved_total = 0

        for st in self._STAGES:
            row = _safe_dict(bounds.get(st))
            lower = int(row.get("lower", 0) or 0)
            upper = int(row.get("upper", lower) if row.get("upper") is not None else lower)
            if upper > 10000:
                upper = lower
            upper = max(0, upper)
            lower = max(0, min(lower, upper))

            segs = by_stage.get(st, [])

            segs_sorted = sorted(
                segs,
                key=lambda x: (
                    -float(_safe_float(x.get("confidence"), 0.0) or 0.0),
                    float(_safe_float(x.get("start_min"), 0.0) or 0.0),
                    -float((_safe_float(x.get("end_min"), 0.0) or 0.0) - (_safe_float(x.get("start_min"), 0.0) or 0.0)),
                ),
            )
            keep_st = segs_sorted[:upper]
            move_st = segs_sorted[upper:]
            kept.extend(keep_st)
            for seg in move_st:
                moved.append(self._move_to_uncertain(seg, f"stage_count_exceeds_window_evidence_upper_bound:{st}"))
            over = max(0, len(segs) - upper)
            miss = max(0, lower - len(keep_st))
            over_total += over
            miss_total += miss
            moved_total += len(move_st)
            coverage[st] = {
                "lower": lower,
                "upper": upper,
                "main_prediction_count": len(keep_st),
                "raw_repaired_count_before_upper_bound": len(segs),
                "over_upper_bound": over,
                "miss_lower_bound": miss,
            }

        kept.extend(other)
        kept = self._sort_segments(kept)
        uncertain = self._unique(uncertain + moved)
        return kept, uncertain, {
            "enabled": True,
            "uses_future_target_labels": False,
            "stage_bounds": coverage,
            "over_upper_bound_count": over_total,
            "miss_lower_bound_count": miss_total,
            "num_moved_to_uncertain_by_upper_bound": moved_total,
            "selection_policy": "evidence_bounded_window_gate",
        }

    def project(self, raw_segments: List[Dict[str, Any]], repair_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        repair_context = _safe_dict(repair_context)
        ctx = self._normalize_context(repair_context)
        raw_segments = [x for x in _safe_list(raw_segments) if isinstance(x, dict)]

        filtered_raw, pre_uncertain, pre_diag = self._pre_filter_by_window_candidates(raw_segments, ctx)

        result = super().project(raw_segments=filtered_raw, repair_context=repair_context)

        repaired = [x for x in _safe_list(result.get("repaired_segments")) if isinstance(x, dict)]
        uncertain = [x for x in _safe_list(result.get("uncertain_segments")) if isinstance(x, dict)] + pre_uncertain
        rejected = [x for x in _safe_list(result.get("rejected_segments")) if isinstance(x, dict)]

        repaired, uncertain, coverage_diag = self._apply_stage_upper_bounds(repaired, uncertain, ctx)

        violations_after = self._violations(repaired, ctx)
        objective_after = self._objective(repaired, ctx)
        energy_proxy = self._production_energy_proxy(repaired, ctx)

        diagnostics = _safe_dict(result.get("repair_diagnostics") or result.get("mechanism_projection_diagnostics"))
        diagnostics["uses_future_target_labels"] = False
        diagnostics["num_raw_segments"] = len(raw_segments)
        diagnostics["num_repaired_segments"] = len(repaired)
        diagnostics["num_uncertain_segments"] = len(self._unique(uncertain))
        diagnostics["num_rejected_segments"] = len(rejected)
        diagnostics["violations_after"] = violations_after
        diagnostics["objective_terms_after"] = objective_after
        diagnostics["objective_total_after"] = self._weighted_objective(objective_after)
        diagnostics["production_energy_proxy"] = energy_proxy
        diagnostics["window_candidate_prefilter"] = pre_diag
        diagnostics["window_stage_evidence_coverage"] = coverage_diag
        diagnostics["projection_policy"] = {
            "draw_predicted_from": "repaired_segments",
            "do_not_use_as_main_prediction": [
                "uncertain_segments",
                "rejected_segments",
                "raw_segments_before_projection",
                "route_future_only_candidates",
                "segments_above_window_stage_upper_bound",
            ],
            "clean_refactor": True,
        }

        repaired = self._finalize_repaired(repaired)
        uncertain = self._unique(uncertain)
        rejected = self._unique(rejected)

        return {
            "repaired_segments": repaired,
            "uncertain_segments": uncertain,
            "rejected_segments": rejected,
            "repair_diagnostics": diagnostics,
            "predicted_operation_segments": repaired,
            "mechanism_projection_diagnostics": diagnostics,
        }

_ScheduleGuidedBaseProjector = PhysicsConsistentVariationalProjector

class PhysicsConsistentVariationalProjector(_ScheduleGuidedBaseProjector):
    def _normalize_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        ctx = super()._normalize_context(context)
        raw = _safe_dict(context)
        sg = _safe_dict(raw.get("schedule_guided_inference"))
        refs = [x for x in _safe_list(raw.get("schedule_reference_candidates")) if isinstance(x, dict)]
        enabled = bool(raw.get("production_conditioning_mode") == "schedule_guided_inference" or sg.get("enabled") or refs)
        if enabled:
            ctx["production_conditioning_mode"] = "schedule_guided_inference"
            ctx["schedule_guided_inference"] = {
                "enabled": True,
                "uses_future_operation_schedule": True,
                "llm_regeneration_required": True,
                "direct_bypass": False,
            }
            ctx["schedule_reference_num_segments"] = len(refs)
        return ctx

    def project(self, raw_segments: List[Dict[str, Any]], repair_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ctx_raw = _safe_dict(repair_context)
        result = super().project(raw_segments=raw_segments, repair_context=repair_context)
        enabled = (
            ctx_raw.get("production_conditioning_mode") == "schedule_guided_inference"
            or bool(_safe_dict(ctx_raw.get("schedule_guided_inference")).get("enabled"))
            or bool(_safe_list(ctx_raw.get("schedule_reference_candidates")))
        )
        if enabled and isinstance(result, dict):
            diag = _safe_dict(result.get("repair_diagnostics") or result.get("mechanism_projection_diagnostics"))
            diag["production_conditioning_mode"] = "schedule_guided_inference"
            diag["schedule_guided_inference"] = {
                "enabled": True,
                "uses_future_operation_schedule": True,
                "llm_regeneration_required": True,
                "direct_bypass": False,
                "schedule_reference_num_segments": len(_safe_list(ctx_raw.get("schedule_reference_candidates"))),
            }
            policy = _safe_dict(diag.get("projection_policy"))
            policy["schedule_guided_inference"] = "PCVP normalizes and checks LLM-regenerated production segments from schedule reference candidates."
            diag["projection_policy"] = policy
            result["repair_diagnostics"] = diag
            result["mechanism_projection_diagnostics"] = diag
            for key in ["predicted_operation_segments", "repaired_segments"]:
                segs = result.get(key)
                if isinstance(segs, list):
                    fixed = []
                    for seg in segs:
                        if isinstance(seg, dict):
                            x = dict(seg)
                            x["source"] = "model_output"
                            x["source_kind"] = x.get("source_kind") or "model_output"
                            x["is_valid_gantt_prediction"] = True
                            fixed.append(x)
                    result[key] = fixed
        return result

_SoftRetentionScheduleGuidedProjectorBase = PhysicsConsistentVariationalProjector

class PhysicsConsistentVariationalProjector(_SoftRetentionScheduleGuidedProjectorBase):

    _SOFT_RETAIN_STAGES = {"BOF", "LF", "Casting", "Rolling"}

    def _normalize_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        ctx = super()._normalize_context(context)
        raw = _safe_dict(context)
        sg = _safe_dict(raw.get("schedule_guided_inference"))
        soft = _safe_dict(raw.get("pcvp_soft_retention_policy"))
        enabled = bool(
            raw.get("production_conditioning_mode") == "schedule_guided_inference"
            or sg.get("enabled")
            or _safe_list(raw.get("schedule_reference_candidates"))
        )
        policy_enabled = bool(soft.get("enabled", True)) if enabled else False
        ctx["pcvp_soft_retention_enabled"] = bool(policy_enabled)
        ctx["pcvp_soft_retention_policy"] = {
            "enabled": bool(policy_enabled),
            "keep_over_upper_bound_segments_as_main": True,
            "keep_route_future_or_outside_candidate_segments_as_main": True,
            "do_not_drop_casting_segments_for_generic_unit_overlap": True,
            "do_not_drop_short_casting_segments": True,
            "policy": "repair_and_annotate_first_soft_retention",
        }
        if policy_enabled:

            for u in ["Casting", "CASTING", "CC", "Caster", "Caster1", "Caster2"]:
                ctx.setdefault("unit_capacity", {})[u] = max(int(ctx.get("unit_capacity", {}).get(u, 1)), 99)
        return ctx

    def _pre_filter_by_window_candidates(
        self,
        raw_segments: List[Dict[str, Any]],
        ctx: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:

        if ctx.get("pcvp_soft_retention_enabled"):
            keys = self._candidate_key_sets(ctx) if hasattr(self, "_candidate_key_sets") else {}
            allowed = keys.get("allowed_main", set()) if isinstance(keys, dict) else set()
            route_future = keys.get("route_future_only", set()) if isinstance(keys, dict) else set()
            annotated: List[Dict[str, Any]] = []
            n_route = 0
            n_outside = 0
            for seg in raw_segments:
                if not isinstance(seg, dict):
                    continue
                x = copy.deepcopy(seg)
                key = self._seg_key2(x) if hasattr(self, "_seg_key2") else ""
                if key in route_future and key not in allowed:
                    x["pcvp_soft_warning"] = _dedup(_safe_list(x.get("pcvp_soft_warning")) + ["route_future_only_candidate_retained_for_schedule_conditioning"])
                    n_route += 1
                elif allowed and key not in allowed:
                    x["pcvp_soft_warning"] = _dedup(_safe_list(x.get("pcvp_soft_warning")) + ["outside_confirmed_optional_candidate_set_retained_for_schedule_conditioning"])
                    n_outside += 1
                annotated.append(x)
            return annotated, [], {
                "enabled": True,
                "soft_retention": True,
                "num_route_future_only_retained": n_route,
                "num_outside_candidate_retained": n_outside,
                "num_route_future_only_moved": 0,
                "num_outside_candidate_moved": 0,
            }
        return super()._pre_filter_by_window_candidates(raw_segments, ctx)

    def _apply_stage_upper_bounds(
        self,
        repaired: List[Dict[str, Any]],
        uncertain: List[Dict[str, Any]],
        ctx: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        if not ctx.get("pcvp_soft_retention_enabled"):
            return super()._apply_stage_upper_bounds(repaired, uncertain, ctx)

        bounds = _safe_dict(ctx.get("window_stage_evidence_bounds"))
        coverage: Dict[str, Dict[str, Any]] = {}
        by_stage: Dict[str, List[Dict[str, Any]]] = {st: [] for st in getattr(self, "_STAGES", ("BOF", "LF", "Casting", "Rolling"))}
        other: List[Dict[str, Any]] = []
        for seg in repaired:
            st = _canonical_stage(seg.get("stage") or seg.get("process") or seg.get("operation"))
            if st in by_stage:
                by_stage[st].append(seg)
            else:
                other.append(seg)

        over_total = 0
        miss_total = 0
        retained_over_total = 0
        kept: List[Dict[str, Any]] = []
        for st in getattr(self, "_STAGES", ("BOF", "LF", "Casting", "Rolling")):
            row = _safe_dict(bounds.get(st))
            lower = int(_safe_float(row.get("lower"), 0) or 0)
            upper = int(_safe_float(row.get("upper"), lower) if row.get("upper") is not None else lower)
            if upper > 10000:
                upper = lower
            upper = max(0, upper)
            lower = max(0, min(lower, upper))
            segs = self._sort_segments(by_stage.get(st, []))
            over = max(0, len(segs) - upper)
            miss = max(0, lower - len(segs))
            over_total += over
            miss_total += miss
            if over > 0:

                for idx, seg in enumerate(segs):
                    if idx >= upper:
                        seg["pcvp_soft_warning"] = _dedup(_safe_list(seg.get("pcvp_soft_warning")) + [f"stage_count_exceeds_window_evidence_upper_bound_retained:{st}"])
                        seg["projection_status"] = seg.get("projection_status") or "soft_retained"
                        retained_over_total += 1
            kept.extend(segs)
            coverage[st] = {
                "lower": lower,
                "upper": upper,
                "main_prediction_count": len(segs),
                "raw_repaired_count_before_upper_bound": len(segs),
                "over_upper_bound": over,
                "miss_lower_bound": miss,
                "soft_retained_over_upper_bound": over,
            }
        kept.extend(other)
        kept = self._sort_segments(kept)
        return kept, self._unique(uncertain), {
            "enabled": True,
            "uses_future_target_labels": False,
            "soft_retention": True,
            "stage_bounds": coverage,
            "over_upper_bound_count": over_total,
            "miss_lower_bound_count": miss_total,
            "num_moved_to_uncertain_by_upper_bound": 0,
            "num_soft_retained_over_upper_bound": retained_over_total,
            "selection_policy": "soft_evidence_bounded_window_gate_repair_annotate_do_not_delete",
        }

    def _soft_normalize_for_main(self, seg: Dict[str, Any], idx: int, ctx: Dict[str, Any], reason: str) -> Optional[Dict[str, Any]]:

        try:
            norm = self._normalize_segment(seg, idx)
        except Exception:
            norm = None
        if norm is None:
            return None
        h0 = ctx["horizon_window"]["start_min"]
        h1 = ctx["horizon_window"]["end_min"]
        if norm["end_min"] <= h0 or norm["start_min"] >= h1:
            return None
        if self.config.clip_to_horizon:
            norm["start_min"] = max(norm["start_min"], h0)
            norm["end_min"] = min(norm["end_min"], h1)
        if norm["end_min"] <= norm["start_min"]:
            return None
        norm["duration_min"] = float(norm["end_min"] - norm["start_min"])
        if norm["duration_min"] < self.config.min_duration_min:

            norm["duration_prior_violation_retained"] = True
            norm["pcvp_soft_warning"] = _dedup(_safe_list(norm.get("pcvp_soft_warning")) + ["short_clipped_segment_retained"])
        norm["projection_status"] = "soft_retained"
        norm["soft_retention_reason"] = reason
        norm["source"] = "model_output"
        norm["source_kind"] = "model_output"
        norm["is_valid_gantt_prediction"] = True
        return norm

    def project(self, raw_segments: List[Dict[str, Any]], repair_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ctx = self._normalize_context(_safe_dict(repair_context))
        result = super().project(raw_segments=raw_segments, repair_context=repair_context)
        if not ctx.get("pcvp_soft_retention_enabled"):
            return result

        repaired = [x for x in _safe_list(result.get("repaired_segments")) if isinstance(x, dict)]
        uncertain = [x for x in _safe_list(result.get("uncertain_segments")) if isinstance(x, dict)]
        rejected = [x for x in _safe_list(result.get("rejected_segments")) if isinstance(x, dict)]

        existing_keys = set()
        for seg in repaired:
            existing_keys.add((
                _normalize_heat_id(seg.get("heat_id")),
                _canonical_stage(seg.get("stage")),
                _safe_str(seg.get("unit")),
                round(float(_safe_float(seg.get("start_min"), 0) or 0), 6),
                round(float(_safe_float(seg.get("end_min"), 0) or 0), 6),
            ))

        restored: List[Dict[str, Any]] = []
        for idx, seg in enumerate(uncertain + rejected, start=100001):
            norm = self._soft_normalize_for_main(seg, idx, ctx, seg.get("uncertain_reason") or seg.get("reject_reason") or "soft_retained_from_diagnostic")
            if norm is None:
                continue
            key = (
                _normalize_heat_id(norm.get("heat_id")),
                _canonical_stage(norm.get("stage")),
                _safe_str(norm.get("unit")),
                round(float(_safe_float(norm.get("start_min"), 0) or 0), 6),
                round(float(_safe_float(norm.get("end_min"), 0) or 0), 6),
            )
            if key in existing_keys:
                continue
            existing_keys.add(key)
            restored.append(norm)

        if restored:
            repaired = self._sort_segments(repaired + restored)

            diag = _safe_dict(result.get("repair_diagnostics") or result.get("mechanism_projection_diagnostics"))
            diag["num_soft_restored_segments_to_main"] = len(restored)
            result["repair_diagnostics"] = diag
            result["mechanism_projection_diagnostics"] = diag

        violations_after = self._violations(repaired, ctx)
        objective_after = self._objective(repaired, ctx)
        energy_proxy = self._production_energy_proxy(repaired, ctx)
        diagnostics = _safe_dict(result.get("repair_diagnostics") or result.get("mechanism_projection_diagnostics"))
        diagnostics["pcvp_soft_retention_enabled"] = True
        diagnostics["soft_retention_policy"] = ctx.get("pcvp_soft_retention_policy")
        diagnostics["num_repaired_segments"] = len(repaired)
        diagnostics["violations_after"] = violations_after
        diagnostics["objective_terms_after"] = objective_after
        diagnostics["objective_total_after"] = self._weighted_objective(objective_after)
        diagnostics["production_energy_proxy"] = energy_proxy
        diagnostics["projection_policy"] = {
            "draw_predicted_from": "repaired_segments_with_soft_retention",
            "do_not_delete_policy": "schedule-guided segments are repaired and annotated; only unparsable or non-overlapping records stay out of main prediction",
            "diagnostic_only": ["uncertain_segments", "rejected_segments", "raw_segments_before_projection"],
            "soft_retention": True,
        }
        repaired = self._finalize_repaired(repaired)
        for seg in repaired:
            seg["source"] = "model_output"
            seg["source_kind"] = seg.get("source_kind") or "model_output"
            seg["is_valid_gantt_prediction"] = True
        result["repaired_segments"] = repaired
        result["predicted_operation_segments"] = repaired
        result["repair_diagnostics"] = diagnostics
        result["mechanism_projection_diagnostics"] = diagnostics
        return result

MechanismScheduleProjector = PhysicsConsistentVariationalProjector
MechanismConstrainedScheduleProjector = PhysicsConsistentVariationalProjector
LLMGuidedPhysicsConsistentVariationalProjection = PhysicsConsistentVariationalProjector
