from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

@dataclass
class EnergyProjectionConfig:

    purchase_eps: float = 1e-6
    nonnegative_floor: float = 0.0

    default_holder_value: float = 0.0

    holder_jump_soft_limit: Dict[str, float] = field(default_factory=lambda: {
        "b": 120.0,
        "c": 60.0,
        "l": 160.0,
    })

    default_holder_risk_bounds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "b": (60.0, 280.0),
        "c": (30.0, 120.0),
        "l": (30.0, 320.0),
    })

    default_holder_physical_bounds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "b": (0.0, 400.0),
        "c": (0.0, 200.0),
        "l": (0.0, 450.0),
    })

    objective_weights: Dict[str, float] = field(default_factory=lambda: {
        "llm_deviation": 1.0,
        "nonnegative": 2.0,
        "holder_physical": 2.0,
        "risk_consistency": 1.5,
        "production_conditioning": 1.0,
        "smoothness": 0.3,
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

def _normalize_heat_id(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).strip().replace("-", "_")
    if not s:
        return ""
    m = re.search(r"([1-9]\d*LDG_\d+)", s, flags=re.I)
    if m:
        return m.group(1).upper()
    parts = s.split("_")
    for i in range(len(parts) - 1):
        if parts[i].upper().endswith("LDG"):
            return f"{parts[i].upper()}_{parts[i + 1]}"
    return s

def _dedup(items: Iterable[Any]) -> List[Any]:
    out: List[Any] = []
    seen = set()
    for item in items:
        key = str(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out

def _risk_list(x: Any) -> List[str]:
    return _dedup([_safe_str(v) for v in _safe_list(x) if _safe_str(v)])

def _gas_aliases(gas: str) -> List[str]:
    gas = gas.lower()
    if gas in {"b", "bfg"}:
        return ["b", "BFG", "bfg"]
    if gas in {"c", "cog"}:
        return ["c", "COG", "cog"]
    if gas in {"l", "ldg"}:
        return ["l", "LDG", "ldg"]
    return [gas]

def _energy_media_alias(media: str) -> List[str]:
    media = media.lower()
    if media in {"s", "steam"}:
        return ["s", "steam", "Steam"]
    if media in {"e", "power", "electric"}:
        return ["e", "power", "Power", "electric"]
    return [media]

def _get_nested_any(obj: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    for k in keys:
        if k in obj:
            return obj[k]
    return None

def _ensure_nested(d: Dict[str, Any], key: str) -> Dict[str, Any]:
    if not isinstance(d.get(key), dict):
        d[key] = {}
    return d[key]

class PhysicsConsistentEnergyProjector:

    def __init__(self, config: Optional[EnergyProjectionConfig] = None) -> None:
        self.config = config or EnergyProjectionConfig()

    def project(
        self,
        raw_prediction: Dict[str, Any],
        repair_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ctx = self._normalize_context(repair_context or {})
        pred = copy.deepcopy(raw_prediction if isinstance(raw_prediction, dict) else {})
        raw_steps = [x for x in _safe_list(pred.get("future_patch_predictions")) if isinstance(x, dict)]

        diagnostics: Dict[str, Any] = {
            "method": "llm_guided_physics_consistent_variational_projection_energy",
            "method_short": "Energy-PCVP",
            "uses_future_target_labels": False,
            "strong_repair_policy": True,
            "weak_delete_policy": True,
            "num_raw_predictions": len(raw_steps),
            "production_conditioning_available": bool(ctx["production_operation_segment_summary"].get("available")),
            "notes": [
                "Raw LLM energy predictions are treated as noisy local observations.",
                "Projection uses history, known future patch context and production operation-segment summary only.",
                "target_future.energy_future must be used only after projection for offline evaluation.",
            ],
        }

        normalized = self._normalize_steps(raw_steps, ctx)
        violations_before = self._compute_violations(normalized, ctx)
        objective_before = self._compute_objective_terms(normalized, ctx)

        repaired = []
        uncertain = []
        rejected = []
        last_holder = self._history_tail_holder_end(ctx)

        for idx, step_pred in enumerate(normalized):
            future_ctx = ctx["future_patch_context"][idx] if idx < len(ctx["future_patch_context"]) else {}
            item, step_uncertain, step_rejected, step_diag = self._repair_one_step(
                step_pred,
                future_ctx=future_ctx,
                ctx=ctx,
                last_holder=last_holder,
            )
            repaired.append(item)
            uncertain.extend(step_uncertain)
            rejected.extend(step_rejected)
            last_holder = self._extract_holder_end(item, fallback=last_holder)
            item.setdefault("energy_projection_step_diagnostics", step_diag)

        violations_after = self._compute_violations(repaired, ctx)
        objective_after = self._compute_objective_terms(repaired, ctx)

        summary = self._repair_summary(pred.get("summary"), repaired, ctx)

        out = copy.deepcopy(pred)
        out["task"] = out.get("task", "energy_horizon_prediction")
        out["mode"] = out.get("mode", "energy_patch_horizon_predictor_gantt_conditioned_pcvp")
        out["horizon"] = len(repaired)
        out["raw_future_patch_predictions_before_energy_projection"] = raw_steps
        out["future_patch_predictions"] = repaired
        out["uncertain_energy_predictions"] = uncertain
        out["rejected_energy_predictions"] = rejected
        out["summary"] = summary

        diagnostics.update({
            "num_repaired_predictions": len(repaired),
            "num_uncertain_predictions": len(uncertain),
            "num_rejected_predictions": len(rejected),
            "violations_before": violations_before,
            "violations_after": violations_after,
            "objective_terms_before": objective_before,
            "objective_terms_after": objective_after,
            "objective_total_before": self._weighted_objective(objective_before),
            "objective_total_after": self._weighted_objective(objective_after),
            "repair_rate": len(repaired) / max(len(raw_steps), 1),
            "uncertain_rate": len(uncertain) / max(len(raw_steps), 1),
            "reject_rate": len(rejected) / max(len(raw_steps), 1),
            "draw_or_evaluate_from": "future_patch_predictions",
        })
        out["mechanism_energy_projection_diagnostics"] = diagnostics

        return {
            "prediction": out,
            "repaired_predictions": repaired,
            "uncertain_predictions": uncertain,
            "rejected_predictions": rejected,
            "repair_diagnostics": diagnostics,
        }

    def _normalize_context(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        ctx = _safe_dict(ctx)
        future_context = _safe_list(ctx.get("future_patch_context"))
        history = _safe_list(ctx.get("energy_history") or ctx.get("history_energy_patches"))
        production_summary = _safe_dict(
            ctx.get("production_operation_segment_summary")
            or ctx.get("operation_segment_summary")
            or {}
        )

        constraints = _safe_dict(ctx.get("constraints"))
        risk_cfg = _safe_dict(constraints.get("risk"))
        holder_risk_bounds = copy.deepcopy(self.config.default_holder_risk_bounds)
        for gas, bounds in risk_cfg.items():
            aliases = _gas_aliases(gas)
            key = aliases[0].lower()[0]
            if isinstance(bounds, (list, tuple)) and len(bounds) >= 2:
                holder_risk_bounds[key] = (float(bounds[0]), float(bounds[1]))

        physical_bounds = copy.deepcopy(self.config.default_holder_physical_bounds)
        if isinstance(ctx.get("holder_physical_bounds"), dict):
            for gas, bounds in ctx["holder_physical_bounds"].items():
                key = _gas_aliases(gas)[0].lower()[0]
                if isinstance(bounds, (list, tuple)) and len(bounds) >= 2:
                    physical_bounds[key] = (float(bounds[0]), float(bounds[1]))

        production_activity_by_step = self._build_activity_by_step(production_summary, future_context)

        return {
            "future_patch_context": future_context,
            "energy_history": history,
            "production_operation_segment_summary": production_summary,
            "production_activity_by_step": production_activity_by_step,
            "constraints": constraints,
            "holder_risk_bounds": holder_risk_bounds,
            "holder_physical_bounds": physical_bounds,
            "stage_energy_coefficients": _safe_dict(ctx.get("stage_energy_coefficients") or ctx.get("energy_coefficients")),
            "raw_context": ctx,
        }

    def _build_activity_by_step(self, production_summary: Dict[str, Any], future_context: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        out: Dict[int, Dict[str, Any]] = {}
        if not production_summary.get("available"):
            return out
        examples = _safe_list(production_summary.get("examples"))
        for idx, fctx in enumerate(future_context, start=1):
            t = _safe_dict(fctx.get("t"))
            p0 = _safe_float(fctx.get("start_min") or t.get("t0"), None)
            p1 = _safe_float(fctx.get("end_min") or t.get("t1"), None)
            stages: Dict[str, int] = {}
            units: Dict[str, int] = {}
            heat_ids: List[str] = []
            active_count = 0
            if p0 is not None and p1 is not None and p1 > p0:
                for seg in examples:
                    if not isinstance(seg, dict):
                        continue
                    s = _safe_float(seg.get("start_min"), None)
                    e = _safe_float(seg.get("end_min"), None)
                    if s is None or e is None or e <= p0 or s >= p1:
                        continue
                    stage = self._canonical_stage(seg.get("stage"))
                    unit = _safe_str(seg.get("unit") or stage, stage)
                    stages[stage] = stages.get(stage, 0) + 1
                    units[unit] = units.get(unit, 0) + 1
                    hid = _normalize_heat_id(seg.get("heat_id"))
                    if hid and hid not in heat_ids:
                        heat_ids.append(hid)
                    active_count += 1

            if active_count == 0 and isinstance(production_summary.get("stage_counts"), dict):
                stages = dict(production_summary.get("stage_counts"))
                active_count = int(sum(_safe_float(v, 0.0) or 0.0 for v in stages.values()))
            out[idx] = {
                "available": True,
                "num_active_operation_segments": active_count,
                "stage_counts": stages,
                "unit_counts": units,
                "heat_ids": heat_ids,
                "activity_level": self._activity_level(active_count, stages),
            }
        return out

    def _activity_level(self, active_count: int, stages: Dict[str, int]) -> str:
        bof = int(_safe_float(stages.get("BOF"), 0.0) or 0)
        rolling = int(_safe_float(stages.get("Rolling"), 0.0) or 0)
        casting = int(_safe_float(stages.get("Casting"), 0.0) or 0)
        score = active_count + bof + 0.5 * rolling + 0.5 * casting
        if score >= 5:
            return "high"
        if score >= 2:
            return "medium"
        if score > 0:
            return "low"
        return "none"

    def _normalize_steps(self, raw_steps: List[Dict[str, Any]], ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
        horizon = len(ctx["future_patch_context"]) or len(raw_steps)
        by_step: Dict[int, Dict[str, Any]] = {}
        by_pid: Dict[str, Dict[str, Any]] = {}
        for i, row in enumerate(raw_steps):
            if not isinstance(row, dict):
                continue
            try:
                st = int(row.get("step"))
                by_step[st] = row
            except Exception:
                pass
            pid = _safe_str(row.get("patch_id") or row.get("id") or row.get("p"))
            if pid:
                by_pid[pid] = row
            else:
                by_pid[f"__idx_{i}"] = row

        out: List[Dict[str, Any]] = []
        for idx in range(horizon):
            step = idx + 1
            fctx = ctx["future_patch_context"][idx] if idx < len(ctx["future_patch_context"]) else {}
            pid = _safe_str(fctx.get("id") or fctx.get("patch_id") or fctx.get("p"))
            row = by_step.get(step) or by_pid.get(pid)
            if row is None and idx < len(raw_steps):
                row = raw_steps[idx]
            if not isinstance(row, dict):
                row = {}
            item = copy.deepcopy(row)
            item["step"] = step
            if pid:
                item["patch_id"] = pid
                item["id"] = pid
                item["p"] = pid
            else:
                item.setdefault("patch_id", _safe_str(item.get("id") or item.get("p") or f"step_{step}"))
                item.setdefault("id", item.get("patch_id"))
                item.setdefault("p", item.get("patch_id"))

            heat_id = _normalize_heat_id(
                fctx.get("hn")
                or _safe_dict(fctx.get("pc")).get("phn")
                or fctx.get("heat_id")
                or item.get("heat_id")
                or item.get("hn")
                or item.get("patch_id")
            )
            item["heat_id"] = heat_id
            item["hn"] = heat_id
            item.setdefault("g", {})
            item.setdefault("st", {})
            item.setdefault("dm", {})
            item.setdefault("ex", {})
            item["rk"] = _risk_list(item.get("rk") or item.get("risk_flags") or item.get("risk_labels"))
            item["raw_energy_prediction_before_projection"] = copy.deepcopy(row)
            out.append(item)
        return out

    def _repair_one_step(
        self,
        item: Dict[str, Any],
        future_ctx: Dict[str, Any],
        ctx: Dict[str, Any],
        last_holder: Dict[str, float],
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        item = copy.deepcopy(item)
        uncertain: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
        diag = {
            "nonnegative_repairs": 0,
            "holder_physical_repairs": 0,
            "risk_repairs": 0,
            "production_conditioning_repairs": 0,
            "smoothness_warnings": 0,
        }

        for gas in ["b", "c", "l"]:
            self._repair_generation(item, gas, diag)
            self._repair_holder(item, gas, ctx, last_holder, diag, uncertain)
        for media in ["s", "e"]:
            self._repair_demand(item, media, diag)
            self._repair_external(item, media, diag)

        prod_available = bool(ctx["production_operation_segment_summary"].get("available"))
        activity = ctx["production_activity_by_step"].get(int(item.get("step", 0)), {})
        if prod_available:
            if item.get("production_activity_basis") in {None, "", "history_only", "unknown"}:
                item["production_activity_basis"] = "schedule_conditioned"
                diag["production_conditioning_repairs"] += 1
            item["operation_segment_summary_used"] = True
            item["operation_schedule_activity"] = activity
            item["response_type"] = self._repair_response_type(item.get("response_type"), activity)
        else:
            item.setdefault("production_activity_basis", "history_only")
            item["operation_segment_summary_used"] = False

        item["rk"] = self._repair_risk_labels(item, ctx, diag)

        item["confidence"] = max(0.0, min(1.0, _safe_float(item.get("confidence"), 0.5) or 0.0))
        item["energy_projection_status"] = "repaired"
        item["energy_projection_method"] = "llm_guided_physics_consistent_variational_projection_energy"
        item["energy_projection_notes"] = _dedup(_safe_list(item.get("energy_projection_notes")))

        return item, uncertain, rejected, diag

    def _repair_generation(self, item: Dict[str, Any], gas: str, diag: Dict[str, Any]) -> None:
        g = _ensure_nested(item, "g")
        obj = None
        for key in _gas_aliases(gas):
            if isinstance(g.get(key), dict):
                obj = g[key]
                if key != gas:
                    g[gas] = obj
                break
        if obj is None:
            obj = {}
            g[gas] = obj
        for k in ["tot", "a", "avg"]:
            if k in obj:
                v = _safe_float(obj.get(k), 0.0) or 0.0
                if v < 0:
                    obj[k] = 0.0
                    diag["nonnegative_repairs"] += 1
        obj.setdefault("tot", max(0.0, _safe_float(obj.get("tot"), 0.0) or 0.0))

    def _repair_holder(self, item: Dict[str, Any], gas: str, ctx: Dict[str, Any], last_holder: Dict[str, float], diag: Dict[str, Any], uncertain: List[Dict[str, Any]]) -> None:
        st = _ensure_nested(item, "st")
        obj = None
        for key in _gas_aliases(gas):
            if isinstance(st.get(key), dict):
                obj = st[key]
                if key != gas:
                    st[gas] = obj
                break
        if obj is None:
            obj = {}
            st[gas] = obj

        lo, hi = ctx["holder_physical_bounds"].get(gas, self.config.default_holder_physical_bounds[gas])
        fallback = last_holder.get(gas, self.config.default_holder_value)
        for k in ["e", "m", "mn", "mx"]:
            if k not in obj:
                continue
            raw = _safe_float(obj.get(k), fallback)
            val = fallback if raw is None else float(raw)
            clipped = min(max(val, lo), hi)
            if clipped != val:
                obj[f"raw_{k}_before_energy_projection"] = val
                obj[k] = clipped
                diag["holder_physical_repairs"] += 1
            else:
                obj[k] = val
        if "e" not in obj:
            obj["e"] = fallback
        if "m" not in obj:
            obj["m"] = obj["e"]

        jump = abs(float(obj.get("e", fallback)) - float(fallback))
        if jump > self.config.holder_jump_soft_limit.get(gas, 999999.0):
            item.setdefault("energy_projection_notes", []).append(f"large_holder_jump_{gas}:{jump:.3f}")
            diag["smoothness_warnings"] += 1
            uncertain.append({
                "step": item.get("step"),
                "patch_id": item.get("patch_id"),
                "field": f"st.{gas}.e",
                "uncertain_reason": "large_holder_jump_retained_after_projection",
                "jump": jump,
            })

    def _repair_demand(self, item: Dict[str, Any], media: str, diag: Dict[str, Any]) -> None:
        dm = _ensure_nested(item, "dm")
        obj = None
        for key in _energy_media_alias(media):
            if isinstance(dm.get(key), dict):
                obj = dm[key]
                if key != media:
                    dm[media] = obj
                break
        if obj is None:
            obj = {}
            dm[media] = obj
        for k in ["tot", "pk", "a", "avg"]:
            if k in obj:
                v = _safe_float(obj.get(k), 0.0) or 0.0
                if v < 0:
                    obj[k] = 0.0
                    diag["nonnegative_repairs"] += 1
        obj.setdefault("tot", max(0.0, _safe_float(obj.get("tot"), 0.0) or 0.0))
        obj.setdefault("pk", max(obj.get("tot", 0.0), _safe_float(obj.get("pk"), 0.0) or 0.0))

    def _repair_external(self, item: Dict[str, Any], media: str, diag: Dict[str, Any]) -> None:
        ex = _ensure_nested(item, "ex")
        obj = None
        for key in _energy_media_alias(media):
            if isinstance(ex.get(key), dict):
                obj = ex[key]
                if key != media:
                    ex[media] = obj
                break
        if obj is None:
            obj = {}
            ex[media] = obj
        for k in ["tot", "a", "avg"]:
            if k in obj:
                v = _safe_float(obj.get(k), 0.0) or 0.0
                if v < 0:
                    obj[k] = 0.0
                    diag["nonnegative_repairs"] += 1
        obj.setdefault("tot", max(0.0, _safe_float(obj.get("tot"), 0.0) or 0.0))

    def _repair_response_type(self, value: Any, activity: Dict[str, Any]) -> str:
        v = _safe_str(value, "unknown").lower()
        if v in {"direct", "indirect", "none"}:
            return v
        level = _safe_str(activity.get("activity_level"), "none")
        if level in {"high", "medium"}:
            return "direct"
        if level == "low":
            return "indirect"
        return "none"

    def _repair_risk_labels(self, item: Dict[str, Any], ctx: Dict[str, Any], diag: Dict[str, Any]) -> List[str]:
        rk = set(_risk_list(item.get("rk")))
        before = set(rk)

        st = _safe_dict(item.get("st"))
        for gas, (low, high) in ctx["holder_risk_bounds"].items():
            obj = _safe_dict(st.get(gas))
            v = _safe_float(obj.get("e") or obj.get("m"), None)
            if v is None:
                continue
            low_label = f"{gas}_low"
            high_label = f"{gas}_high"
            if v < low:
                rk.add(low_label)
                rk.discard(high_label)
            elif v > high:
                rk.add(high_label)
                rk.discard(low_label)
            else:
                rk.discard(low_label)
                rk.discard(high_label)

        ex = _safe_dict(item.get("ex"))
        p_power = _safe_float(_safe_dict(ex.get("e")).get("tot"), 0.0) or 0.0
        p_steam = _safe_float(_safe_dict(ex.get("s")).get("tot"), 0.0) or 0.0
        if p_power > self.config.purchase_eps:
            rk.add("pp")
        else:
            rk.discard("pp")
        if p_steam > self.config.purchase_eps:
            rk.add("ps")
        else:
            rk.discard("ps")

        if rk != before:
            diag["risk_repairs"] += 1
        return sorted(rk)

    def _repair_summary(self, summary: Any, predictions: List[Dict[str, Any]], ctx: Dict[str, Any]) -> Dict[str, Any]:
        out = copy.deepcopy(summary if isinstance(summary, dict) else {})
        prod_available = bool(ctx["production_operation_segment_summary"].get("available"))
        out["production_conditioning_available"] = prod_available
        out["production_conditioning_source"] = "operation_segment_summary" if prod_available else "missing"
        out["production_activity_basis"] = "schedule_conditioned" if prod_available else "history_only"
        out.setdefault("overall_feasibility", "ok")
        out.setdefault("confidence", self._mean([_safe_float(p.get("confidence"), 0.0) or 0.0 for p in predictions]))
        out["holder_trend"] = out.get("holder_trend") or self._holder_trend(predictions)
        out["purchase_trend"] = out.get("purchase_trend") or self._purchase_trend(predictions)
        return out

    def _mean(self, values: List[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    def _holder_trend(self, predictions: List[Dict[str, Any]]) -> Dict[str, str]:
        ans: Dict[str, str] = {}
        for gas, name in [("b", "BFG"), ("c", "COG"), ("l", "LDG")]:
            vals = [_safe_float(_safe_dict(_safe_dict(p.get("st")).get(gas)).get("e"), None) for p in predictions]
            vals = [v for v in vals if v is not None]
            ans[name] = self._trend(vals)
        return ans

    def _purchase_trend(self, predictions: List[Dict[str, Any]]) -> Dict[str, str]:
        power = [_safe_float(_safe_dict(_safe_dict(p.get("ex")).get("e")).get("tot"), None) for p in predictions]
        steam = [_safe_float(_safe_dict(_safe_dict(p.get("ex")).get("s")).get("tot"), None) for p in predictions]
        return {
            "power": self._trend([v for v in power if v is not None]),
            "steam": self._trend([v for v in steam if v is not None]),
        }

    def _trend(self, vals: List[float]) -> str:
        if len(vals) < 2:
            return "unknown"
        delta = vals[-1] - vals[0]
        if abs(delta) < 1e-6:
            return "flat"
        return "up" if delta > 0 else "down"

    def _history_tail_holder_end(self, ctx: Dict[str, Any]) -> Dict[str, float]:
        history = ctx.get("energy_history", [])
        if history:
            return self._extract_holder_end(history[-1], fallback={"b": 0.0, "c": 0.0, "l": 0.0})
        return {"b": 0.0, "c": 0.0, "l": 0.0}

    def _extract_holder_end(self, item: Dict[str, Any], fallback: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        fallback = fallback or {"b": 0.0, "c": 0.0, "l": 0.0}
        st = _safe_dict(item.get("st"))
        out = {}
        for gas in ["b", "c", "l"]:
            obj = _safe_dict(st.get(gas))
            out[gas] = float(_safe_float(obj.get("e") or obj.get("m"), fallback.get(gas, 0.0)) or 0.0)
        return out

    def _compute_violations(self, predictions: List[Dict[str, Any]], ctx: Dict[str, Any]) -> Dict[str, Any]:
        v = {
            "negative_numeric": 0,
            "holder_physical": 0,
            "risk_inconsistency": 0,
            "missing_context_binding": 0,
            "missing_production_conditioning_when_available": 0,
            "large_holder_jump": 0,
        }
        last_holder = self._history_tail_holder_end(ctx)
        prod_available = bool(ctx["production_operation_segment_summary"].get("available"))

        for idx, item in enumerate(predictions):

            if idx < len(ctx["future_patch_context"]):
                fctx = ctx["future_patch_context"][idx]
                pid = _safe_str(fctx.get("id") or fctx.get("patch_id") or fctx.get("p"))
                hid = _normalize_heat_id(fctx.get("hn") or _safe_dict(fctx.get("pc")).get("phn") or fctx.get("heat_id"))
                if pid and _safe_str(item.get("patch_id") or item.get("id") or item.get("p")) != pid:
                    v["missing_context_binding"] += 1
                if hid and _normalize_heat_id(item.get("heat_id") or item.get("hn")) != hid:
                    v["missing_context_binding"] += 1

            if prod_available and item.get("production_activity_basis") in {None, "", "history_only", "unknown"}:
                v["missing_production_conditioning_when_available"] += 1

            for obj in [item.get("g"), item.get("dm"), item.get("ex")]:
                v["negative_numeric"] += self._count_negative_numbers(obj)

            st = _safe_dict(item.get("st"))
            for gas, (lo, hi) in ctx["holder_physical_bounds"].items():
                obj = _safe_dict(st.get(gas))
                for key in ["e", "m", "mn", "mx"]:
                    if key in obj:
                        val = _safe_float(obj.get(key), None)
                        if val is not None and (val < lo or val > hi):
                            v["holder_physical"] += 1
                cur = _safe_float(obj.get("e") or obj.get("m"), None)
                if cur is not None:
                    jump = abs(cur - last_holder.get(gas, cur))
                    if jump > self.config.holder_jump_soft_limit.get(gas, 999999.0):
                        v["large_holder_jump"] += 1
            v["risk_inconsistency"] += self._risk_inconsistency_count(item, ctx)
            last_holder = self._extract_holder_end(item, fallback=last_holder)
        return v

    def _count_negative_numbers(self, obj: Any) -> int:
        if isinstance(obj, dict):
            return sum(self._count_negative_numbers(v) for v in obj.values())
        if isinstance(obj, list):
            return sum(self._count_negative_numbers(v) for v in obj)
        val = _safe_float(obj, None)
        return 1 if val is not None and val < 0 else 0

    def _risk_inconsistency_count(self, item: Dict[str, Any], ctx: Dict[str, Any]) -> int:
        count = 0
        rk = set(_risk_list(item.get("rk")))
        st = _safe_dict(item.get("st"))
        for gas, (low, high) in ctx["holder_risk_bounds"].items():
            obj = _safe_dict(st.get(gas))
            val = _safe_float(obj.get("e") or obj.get("m"), None)
            if val is None:
                continue
            if val < low and f"{gas}_low" not in rk:
                count += 1
            if val > high and f"{gas}_high" not in rk:
                count += 1
        ex = _safe_dict(item.get("ex"))
        power = _safe_float(_safe_dict(ex.get("e")).get("tot"), 0.0) or 0.0
        steam = _safe_float(_safe_dict(ex.get("s")).get("tot"), 0.0) or 0.0
        if power > self.config.purchase_eps and "pp" not in rk:
            count += 1
        if power <= self.config.purchase_eps and "pp" in rk:
            count += 1
        if steam > self.config.purchase_eps and "ps" not in rk:
            count += 1
        if steam <= self.config.purchase_eps and "ps" in rk:
            count += 1
        return count

    def _compute_objective_terms(self, predictions: List[Dict[str, Any]], ctx: Dict[str, Any]) -> Dict[str, float]:
        terms = {
            "llm_deviation": 0.0,
            "nonnegative": 0.0,
            "holder_physical": 0.0,
            "risk_consistency": 0.0,
            "production_conditioning": 0.0,
            "smoothness": 0.0,
        }
        violations = self._compute_violations(predictions, ctx)
        terms["nonnegative"] = float(violations.get("negative_numeric", 0))
        terms["holder_physical"] = float(violations.get("holder_physical", 0))
        terms["risk_consistency"] = float(violations.get("risk_inconsistency", 0))
        terms["production_conditioning"] = float(violations.get("missing_production_conditioning_when_available", 0))
        terms["smoothness"] = float(violations.get("large_holder_jump", 0))

        for item in predictions:
            raw = _safe_dict(item.get("raw_energy_prediction_before_projection"))
            terms["llm_deviation"] += self._numeric_deviation(raw, item)
        return terms

    def _numeric_deviation(self, raw: Dict[str, Any], repaired: Dict[str, Any]) -> float:
        keys = [
            ("g", "b", "tot"), ("g", "c", "tot"), ("g", "l", "tot"),
            ("st", "b", "e"), ("st", "c", "e"), ("st", "l", "e"),
            ("dm", "s", "tot"), ("dm", "e", "tot"),
            ("ex", "s", "tot"), ("ex", "e", "tot"),
        ]
        total = 0.0
        for a, b, c in keys:
            rv = _safe_float(_safe_dict(_safe_dict(raw.get(a)).get(b)).get(c), None)
            pv = _safe_float(_safe_dict(_safe_dict(repaired.get(a)).get(b)).get(c), None)
            if rv is not None and pv is not None:
                total += (pv - rv) ** 2
        return total

    def _weighted_objective(self, terms: Dict[str, float]) -> float:
        return float(sum(float(v) * self.config.objective_weights.get(k, 1.0) for k, v in terms.items()))

    def _canonical_stage(self, x: Any) -> str:
        s = _safe_str(x, "Other").upper().replace(" ", "")
        if "BOF" in s or "CONVERTER" in s:
            return "BOF"
        if s == "LF" or "LADLE" in s:
            return "LF"
        if "CAST" in s or s == "CC":
            return "Casting"
        if "ROLL" in s:
            return "Rolling"
        return "Other"

MechanismEnergyProjector = PhysicsConsistentEnergyProjector
LLMGuidedPhysicsConsistentEnergyProjection = PhysicsConsistentEnergyProjector

__all__ = [
    "EnergyProjectionConfig",
    "PhysicsConsistentEnergyProjector",
    "MechanismEnergyProjector",
    "LLMGuidedPhysicsConsistentEnergyProjection",
]

_ORIGINAL_ENERGY_PCVP_PROJECT = PhysicsConsistentEnergyProjector.project

def _energy_pcvp_project_with_applied_diagnostics(self, raw_prediction, repair_context=None):
    raw = raw_prediction if isinstance(raw_prediction, dict) else {}
    raw_steps = _safe_list(raw.get("future_patch_predictions"))
    result = _ORIGINAL_ENERGY_PCVP_PROJECT(self, raw_prediction, repair_context=repair_context)
    if not isinstance(result, dict):
        return result
    pred = result.get("prediction")
    if isinstance(pred, dict):
        diag = _safe_dict(pred.get("mechanism_energy_projection_diagnostics"))
        diag["applied"] = True
        diag["projection_invoked"] = True
        diag["raw_prediction_missing"] = not bool(raw_steps)
        diag["raw_num_future_patch_predictions"] = len(raw_steps)
        diag["uses_future_target_labels"] = False
        diag.setdefault("strong_repair_policy", True)
        diag.setdefault("weak_delete_policy", True)
        if diag.get("raw_prediction_missing"):
            diag.setdefault("notes", [])
            if isinstance(diag["notes"], list):
                diag["notes"].append("raw future_patch_predictions were missing; context-bound placeholders were repaired by Energy-PCVP")
        pred["mechanism_energy_projection_diagnostics"] = diag
        result["prediction"] = pred
        result["repair_diagnostics"] = diag
    return result

PhysicsConsistentEnergyProjector.project = _energy_pcvp_project_with_applied_diagnostics
MechanismEnergyProjector = PhysicsConsistentEnergyProjector
LLMGuidedPhysicsConsistentEnergyProjection = PhysicsConsistentEnergyProjector

def _pcvp_final_sd(x):
    return x if isinstance(x, dict) else {}

def _pcvp_final_sl(x):
    return x if isinstance(x, list) else ([] if x is None else [x])

def _pcvp_final_float(x, default=0.0):
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

def _pcvp_final_get(obj, path, default=0.0):
    cur = obj
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return _pcvp_final_float(cur, default)

def _pcvp_final_set(obj, path, value):
    cur = obj
    for key in path[:-1]:
        if not isinstance(cur.get(key), dict):
            cur[key] = {}
        cur = cur[key]
    cur[path[-1]] = round(max(0.0, _pcvp_final_float(value, 0.0)), 4)

def _pcvp_final_patch_time(p):
    p = _pcvp_final_sd(p)
    t = _pcvp_final_sd(p.get("t"))
    return (
        _pcvp_final_float(p.get("start_min") or p.get("t0") or t.get("t0") or t.get("s0"), None),
        _pcvp_final_float(p.get("end_min") or p.get("t1") or t.get("t1") or t.get("s1"), None),
    )

def _pcvp_final_canonical_stage(x):
    s = str(x or "").strip().upper().replace(" ", "")
    if "BOF" in s or "CONVERTER" in s or "LDGGENERATION" in s:
        return "BOF"
    if s == "LF" or "LADLE" in s:
        return "LF"
    if "CAST" in s or s == "CC":
        return "Casting"
    if "ROLL" in s:
        return "Rolling"
    return "Other"

def _pcvp_final_build_activity_by_step(self, production_summary, future_context):
    raw_activity = _pcvp_final_sl(_pcvp_final_sd(self._last_normalized_context if hasattr(self, "_last_normalized_context") else {}).get("future_production_activity_by_step"))
    if raw_activity:
        return {int(_pcvp_final_float(x.get("step"), i + 1)): x for i, x in enumerate(raw_activity) if isinstance(x, dict)}

    production_summary = _pcvp_final_sd(production_summary)
    future_context = _pcvp_final_sl(future_context)
    examples = [x for x in _pcvp_final_sl(production_summary.get("examples")) if isinstance(x, dict)]
    prod_available = bool(production_summary.get("available") or production_summary.get("num_segments") or examples or production_summary.get("stage_counts"))
    out = {}
    any_overlap = False
    for idx, fctx in enumerate(future_context, start=1):
        p0, p1 = _pcvp_final_patch_time(fctx)
        stage_overlap = {"BOF": 0.0, "LF": 0.0, "Casting": 0.0, "Rolling": 0.0, "Other": 0.0}
        if p0 is not None and p1 is not None and p1 > p0:
            for seg in examples:
                s = _pcvp_final_float(seg.get("start_min"), None)
                e = _pcvp_final_float(seg.get("end_min"), None)
                if s is None or e is None:
                    continue
                ov = max(0.0, min(p1, e) - max(p0, s))
                if ov <= 0:
                    continue
                any_overlap = True
                stage = _pcvp_final_canonical_stage(seg.get("stage"))
                stage_overlap[stage] = stage_overlap.get(stage, 0.0) + ov
        score = 1.2 * stage_overlap["BOF"] + 0.9 * stage_overlap["LF"] + 0.85 * stage_overlap["Casting"] + 1.3 * stage_overlap["Rolling"]
        out[idx] = {"available": prod_available, "basis": "time_overlap" if score > 0 else ("operation_segment_summary_no_step_overlap" if prod_available else "history_only"), "stage_overlap_min": {k: round(v, 4) for k, v in stage_overlap.items() if abs(v) > 1e-9}, "activity_score": round(score, 4), "activity_level": "high" if score >= 80 else ("medium" if score >= 30 else ("low" if score > 0 else "none"))}
    if prod_available and not any_overlap:
        stage_counts = _pcvp_final_sd(production_summary.get("stage_counts"))
        total_count = sum(_pcvp_final_float(v, 0.0) for v in stage_counts.values())
        for row in out.values():
            row["basis"] = "window_summary_only_no_time_overlap"
            row["window_stage_counts"] = stage_counts
            row["activity_score"] = round(total_count, 4)
            row["activity_level"] = "medium" if total_count > 0 else "none"
    return out

_ORIGINAL_PCVP_NORMALIZE_CONTEXT_FINAL = PhysicsConsistentEnergyProjector._normalize_context

def _pcvp_final_normalize_context(self, ctx):
    ctx0 = _safe_dict(ctx)
    out = _ORIGINAL_PCVP_NORMALIZE_CONTEXT_FINAL(self, ctx0)
    explicit_activity = _pcvp_final_sl(ctx0.get("future_production_activity_by_step"))
    if explicit_activity:
        out["production_activity_by_step"] = {int(_pcvp_final_float(x.get("step"), i + 1)): x for i, x in enumerate(explicit_activity) if isinstance(x, dict)}
    else:
        out["production_activity_by_step"] = _pcvp_final_build_activity_by_step(self, out.get("production_operation_segment_summary", {}), out.get("future_patch_context", []))
    out["semantic_energy_history"] = _pcvp_final_sl(ctx0.get("semantic_energy_history"))
    out["history_transition_summary"] = _pcvp_final_sd(ctx0.get("history_transition_summary"))
    out["energy_numeric_history_reference"] = _pcvp_final_sd(ctx0.get("energy_numeric_history_reference"))
    out["uses_future_target_labels"] = False
    return out

def _pcvp_final_history_last(ctx, key, default=0.0):
    ref = _pcvp_final_sd(ctx.get("energy_numeric_history_reference"))
    last = _pcvp_final_sd(ref.get("last"))
    if key in last:
        return _pcvp_final_float(last.get(key), default)
    hist = _pcvp_final_sl(ctx.get("energy_history"))
    if not hist:
        return default
    h = hist[-1]
    path_map = {
        "g_b_tot": ["g", "b", "tot"], "g_c_tot": ["g", "c", "tot"], "g_l_tot": ["g", "l", "tot"],
        "st_b_e": ["st", "b", "e"], "st_c_e": ["st", "c", "e"], "st_l_e": ["st", "l", "e"],
        "dm_s_tot": ["dm", "s", "tot"], "dm_s_pk": ["dm", "s", "pk"], "dm_e_tot": ["dm", "e", "tot"], "dm_e_pk": ["dm", "e", "pk"],
        "ex_s_tot": ["ex", "s", "tot"], "ex_e_tot": ["ex", "e", "tot"],
    }
    return _pcvp_final_get(h, path_map.get(key, []), default)

def _pcvp_final_apply_mechanism_completion(self, item, ctx, last_holder, diag):
    step = int(_pcvp_final_float(item.get("step"), 0) or 0)
    activity = _pcvp_final_sd(_pcvp_final_sd(ctx.get("production_activity_by_step")).get(step))
    stage_ov = _pcvp_final_sd(activity.get("stage_overlap_min"))
    bof = _pcvp_final_float(stage_ov.get("BOF"), 0.0)
    lf = _pcvp_final_float(stage_ov.get("LF"), 0.0)
    casting = _pcvp_final_float(stage_ov.get("Casting"), 0.0)
    rolling = _pcvp_final_float(stage_ov.get("Rolling"), 0.0)
    prod_available = bool(_pcvp_final_sd(ctx.get("production_operation_segment_summary")).get("available") or _pcvp_final_sd(ctx.get("production_operation_segment_summary")).get("num_segments"))

    g_l = _pcvp_final_get(item, ["g", "l", "tot"], None)
    dm_s = _pcvp_final_get(item, ["dm", "s", "tot"], None)
    dm_e = _pcvp_final_get(item, ["dm", "e", "tot"], None)
    ex_s = _pcvp_final_get(item, ["ex", "s", "tot"], None)
    ex_e = _pcvp_final_get(item, ["ex", "e", "tot"], None)

    hist_g_l = _pcvp_final_history_last(ctx, "g_l_tot", 0.0)
    hist_dm_s = _pcvp_final_history_last(ctx, "dm_s_tot", 0.0)
    hist_dm_e = _pcvp_final_history_last(ctx, "dm_e_tot", 0.0)
    hist_ex_s = _pcvp_final_history_last(ctx, "ex_s_tot", 0.0)
    hist_ex_e = _pcvp_final_history_last(ctx, "ex_e_tot", 0.0)

    ref_g_l = hist_g_l * (1.0 + min(0.85, bof / 80.0))
    ref_dm_s = hist_dm_s * (1.0 + min(0.75, (casting + 0.6 * rolling) / 140.0))
    ref_dm_e = hist_dm_e * (1.0 + min(0.95, (rolling + 0.7 * lf) / 120.0))
    ref_ex_s = max(hist_ex_s, max(0.0, ref_dm_s - 0.55 * ref_g_l) * 0.35)
    ref_ex_e = max(hist_ex_e, max(0.0, ref_dm_e - 0.20 * _pcvp_final_history_last(ctx, "g_b_tot", 0.0)) * 0.55)

    def choose(raw, ref, field):
        raw_v = _pcvp_final_float(raw, None)
        ref_v = _pcvp_final_float(ref, 0.0)
        if raw_v is None or (raw_v <= 1e-9 and ref_v > 1e-9):
            diag["mechanism_numeric_completions"] = diag.get("mechanism_numeric_completions", 0) + 1
            item.setdefault("energy_projection_notes", []).append(f"mechanism_completed_{field}")
            return ref_v

        if prod_available and _pcvp_final_float(activity.get("activity_score"), 0.0) > 0:
            return 0.75 * raw_v + 0.25 * ref_v
        return raw_v

    _pcvp_final_set(item, ["g", "l", "tot"], choose(g_l, ref_g_l, "g.l.tot"))
    _pcvp_final_set(item, ["dm", "s", "tot"], choose(dm_s, ref_dm_s, "dm.s.tot"))
    _pcvp_final_set(item, ["dm", "e", "tot"], choose(dm_e, ref_dm_e, "dm.e.tot"))
    _pcvp_final_set(item, ["ex", "s", "tot"], choose(ex_s, ref_ex_s, "ex.s.tot"))
    _pcvp_final_set(item, ["ex", "e", "tot"], choose(ex_e, ref_ex_e, "ex.e.tot"))

    _pcvp_final_set(item, ["dm", "s", "pk"], max(_pcvp_final_get(item, ["dm", "s", "pk"], 0.0), _pcvp_final_get(item, ["dm", "s", "tot"], 0.0)))
    _pcvp_final_set(item, ["dm", "e", "pk"], max(_pcvp_final_get(item, ["dm", "e", "pk"], 0.0), _pcvp_final_get(item, ["dm", "e", "tot"], 0.0)))

    if _pcvp_final_get(item, ["st", "l", "e"], None) is None or _pcvp_final_get(item, ["st", "l", "e"], 0.0) <= 1e-9:
        prev = _pcvp_final_float(last_holder.get("l"), _pcvp_final_history_last(ctx, "st_l_e", 0.0))
        bal = _pcvp_final_get(item, ["g", "l", "tot"], 0.0) - 0.25 * _pcvp_final_get(item, ["dm", "s", "tot"], 0.0) - 0.12 * _pcvp_final_get(item, ["dm", "e", "tot"], 0.0)
        l_val = min(max(prev + 0.05 * bal, 0.0), 450.0)
        _pcvp_final_set(item, ["st", "l", "e"], l_val)
        _pcvp_final_set(item, ["st", "l", "m"], l_val)
        diag["mechanism_numeric_completions"] = diag.get("mechanism_numeric_completions", 0) + 1

_ORIGINAL_PCVP_REPAIR_ONE_STEP_FINAL = PhysicsConsistentEnergyProjector._repair_one_step

def _pcvp_final_repair_one_step(self, item, future_ctx, ctx, last_holder):

    item = copy.deepcopy(item if isinstance(item, dict) else {})
    pre_diag = {"mechanism_numeric_completions": 0}
    _pcvp_final_apply_mechanism_completion(self, item, ctx, last_holder, pre_diag)
    repaired, uncertain, rejected, diag = _ORIGINAL_PCVP_REPAIR_ONE_STEP_FINAL(self, item, future_ctx, ctx, last_holder)
    diag.update(pre_diag)
    repaired["mechanism_activity_used_for_projection"] = _pcvp_final_sd(_pcvp_final_sd(ctx.get("production_activity_by_step")).get(int(_pcvp_final_float(repaired.get("step"), 0) or 0)))
    repaired.setdefault("energy_projection_notes", [])
    if pre_diag.get("mechanism_numeric_completions"):
        repaired["energy_projection_notes"].append("mechanism_numeric_completion_or_activity_shaping_applied")
    return repaired, uncertain, rejected, diag

PhysicsConsistentEnergyProjector._normalize_context = _pcvp_final_normalize_context
PhysicsConsistentEnergyProjector._repair_one_step = _pcvp_final_repair_one_step
MechanismEnergyProjector = PhysicsConsistentEnergyProjector
LLMGuidedPhysicsConsistentEnergyProjection = PhysicsConsistentEnergyProjector

def _pcvp_v2_get_baseline_for_step(ctx, step):
    baselines = _pcvp_final_sl(ctx.get("energy_baseline_future_predictions"))
    if step <= 0 or step > len(baselines):
        return {}
    b = baselines[step - 1]
    return b if isinstance(b, dict) else {}

def _pcvp_v2_ratio_extreme(v, b):
    v = _pcvp_final_float(v, None)
    b = _pcvp_final_float(b, None)
    if v is None or b is None or b <= 1e-9:
        return False
    r = v / b
    return r > 3.0 or r < 0.20

_ORIGINAL_PCVP_V2_NORMALIZE_CONTEXT = PhysicsConsistentEnergyProjector._normalize_context

def _pcvp_v2_normalize_context(self, ctx):
    out = _ORIGINAL_PCVP_V2_NORMALIZE_CONTEXT(self, ctx)
    ctx0 = _pcvp_final_sd(ctx)
    out["energy_baseline_future_predictions"] = _pcvp_final_sl(ctx0.get("energy_baseline_future_predictions"))
    out["energy_baseline_policy"] = _pcvp_final_sd(ctx0.get("energy_baseline_policy"))
    out["baseline_blend_weight"] = min(max(_pcvp_final_float(ctx0.get("baseline_blend_weight"), 0.20), 0.0), 0.50)
    out["baseline_blend_policy"] = str(ctx0.get("baseline_blend_policy") or "soft_prior_only_for_missing_or_extreme_llm_values")
    out["uses_future_target_labels"] = False
    return out

def _pcvp_v2_apply_baseline_soft_prior(self, item, ctx, diag):
    policy = str(ctx.get("baseline_blend_policy") or "")
    if policy.lower() in {"disabled", "none", "off"}:
        return
    step = int(_pcvp_final_float(item.get("step"), 0) or 0)
    baseline = _pcvp_v2_get_baseline_for_step(ctx, step)
    if not baseline:
        return

    w = min(max(_pcvp_final_float(ctx.get("baseline_blend_weight"), 0.20), 0.0), 0.50)
    paths = [
        ("g", "b", "tot"), ("g", "c", "tot"), ("g", "l", "tot"),
        ("st", "b", "e"), ("st", "b", "m"), ("st", "c", "e"), ("st", "c", "m"), ("st", "l", "e"), ("st", "l", "m"),
        ("dm", "s", "tot"), ("dm", "s", "pk"), ("dm", "e", "tot"), ("dm", "e", "pk"),
        ("ex", "s", "tot"), ("ex", "e", "tot"),
    ]
    applied = 0
    deltas = {}
    for path in paths:
        raw_v = _pcvp_final_get(item, list(path), None)
        base_v = _pcvp_final_get(baseline, list(path), None)
        if base_v is None:
            continue
        field_name = ".".join(path)
        if raw_v is None or (_pcvp_final_float(raw_v, 0.0) <= 1e-9 and _pcvp_final_float(base_v, 0.0) > 1e-9):
            new_v = _pcvp_final_float(base_v, 0.0)
            reason = "missing_or_zero"
        elif _pcvp_v2_ratio_extreme(raw_v, base_v):
            new_v = (1.0 - w) * _pcvp_final_float(raw_v, 0.0) + w * _pcvp_final_float(base_v, 0.0)
            reason = "extreme_deviation_soft_blend"
        else:
            continue
        _pcvp_final_set(item, list(path), new_v)
        applied += 1
        deltas[field_name] = {
            "before": round(_pcvp_final_float(raw_v, 0.0), 4),
            "baseline": round(_pcvp_final_float(base_v, 0.0), 4),
            "after": round(_pcvp_final_float(new_v, 0.0), 4),
            "reason": reason,
        }
    if applied:
        diag["baseline_soft_prior_applied_count"] = diag.get("baseline_soft_prior_applied_count", 0) + applied
        item["baseline_reference_used"] = True
        item["baseline_soft_prior_deltas"] = deltas
        item.setdefault("energy_projection_notes", []).append("baseline_soft_prior_applied_without_target_labels")

_ORIGINAL_PCVP_V2_REPAIR_ONE_STEP = PhysicsConsistentEnergyProjector._repair_one_step

def _pcvp_v2_repair_one_step(self, item, future_ctx, ctx, last_holder):

    item = copy.deepcopy(item if isinstance(item, dict) else {})
    pre_diag = {"baseline_soft_prior_applied_count": 0}
    _pcvp_v2_apply_baseline_soft_prior(self, item, ctx, pre_diag)
    repaired, uncertain, rejected, diag = _ORIGINAL_PCVP_V2_REPAIR_ONE_STEP(self, item, future_ctx, ctx, last_holder)
    if pre_diag.get("baseline_soft_prior_applied_count"):
        diag["baseline_soft_prior_applied_count"] = diag.get("baseline_soft_prior_applied_count", 0) + pre_diag["baseline_soft_prior_applied_count"]
        repaired["baseline_reference_used"] = True
        repaired.setdefault("energy_projection_notes", []).append("baseline_soft_prior_checked")
    else:
        repaired.setdefault("baseline_reference_used", bool(_pcvp_v2_get_baseline_for_step(ctx, int(_pcvp_final_float(repaired.get("step"), 0) or 0))))
    return repaired, uncertain, rejected, diag

PhysicsConsistentEnergyProjector._normalize_context = _pcvp_v2_normalize_context
PhysicsConsistentEnergyProjector._repair_one_step = _pcvp_v2_repair_one_step
MechanismEnergyProjector = PhysicsConsistentEnergyProjector
LLMGuidedPhysicsConsistentEnergyProjection = PhysicsConsistentEnergyProjector
