from __future__ import annotations

import json
import re
import uuid
from typing import Any, Dict, List, Optional

from b_base_chat_model import BaseChatModel
from f_transition_engine import CandidatePlan, EnergyTrajectory

def _extract_first_balanced_json_object(text: str) -> str:

    for start in range(len(text)):
        if text[start] != "{":
            continue

        depth = 0
        in_string = False
        escape = False

        for i in range(start, len(text)):
            ch = text[i]

            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]

    raise ValueError("No complete JSON object was found.")

def _safe_json_loads(text: str) -> Dict[str, Any]:
    text = text.strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    fenced = re.findall(r"```json\s*(.*?)\s*```", text, flags=re.S | re.I)
    for block in fenced:
        try:
            obj = json.loads(block)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue

    if "</think>" in text:
        suffix = text.split("</think>")[-1].strip()
        try:
            obj = json.loads(suffix)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        try:
            candidate = _extract_first_balanced_json_object(suffix)
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    try:
        candidate = _extract_first_balanced_json_object(text)
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    preview = text[:1000]
    raise ValueError(f"Unable to parse JSON from the energy model output. Raw preview: {preview}")

class EnergyAgentAdapter:

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        max_new_tokens: int = 10240,
    ) -> None:
        self.chat_model = BaseChatModel(
            model_path=model_path,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=False,
        )

    def _safe_dict(self, x: Any) -> Dict[str, Any]:
        return x if isinstance(x, dict) else {}

    def _safe_list(self, x: Any) -> List[Any]:
        return x if isinstance(x, list) else []

    def _safe_str(self, x: Any, default: str = "") -> str:
        if x is None:
            return default
        return str(x).strip()

    def _safe_float(self, x: Any, default: float = 0.0) -> float:
        try:
            if x is None:
                return default
            return float(x)
        except Exception:
            return default

    def _normalize_heat_id(self, x: Any) -> str:

        if x is None:
            return ""
        s = str(x).strip()
        if not s:
            return ""
        s = s.replace("-", "_")
        parts = s.split("_")
        for i in range(len(parts) - 1):
            if parts[i].endswith("LDG"):
                return f"{parts[i]}_{parts[i + 1]}"
        return s

    def _dedup_preserve_order(self, items: List[Any]) -> List[Any]:
        out: List[Any] = []
        seen = set()
        for x in items:
            if x is None:
                continue
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    def _strip_for_energy(self, obj: Any) -> Any:

        banned = {
            "_raw_text", "raw_text", "raw_output", "_raw_list_output",
            "model_input", "prompt", "full_prompt", "messages",
            "traceback", "debug", "debug_info", "comments", "commentary",
            "reasoning", "analysis",

            "predicted_operation_segments",
            "raw_predicted_operation_segments",
            "diagnostic_derived_operation_segments",
            "derived_operation_segments_from_patches",
            "fallback_operation_segments",
            "patch_operation_segments",
            "target_operation_segments",
            "production_operation_segments",
            "history_operation_segments",
        }
        if isinstance(obj, dict):
            return {
                k: self._strip_for_energy(v)
                for k, v in obj.items()
                if k not in banned
            }
        if isinstance(obj, list):
            return [self._strip_for_energy(x) for x in obj]
        return obj

    def _time_to_float_for_summary(self, x: Any) -> Optional[float]:
        try:
            if x is None:
                return None
            if isinstance(x, (int, float)):
                return float(x)
            s = str(x).strip()
            if not s:
                return None
            try:
                return float(s)
            except Exception:
                pass
            m = re.search(r"(\d{1,3}):(\d{2})(?::(\d{2}))?", s)
            if not m:
                return None
            hh = int(m.group(1))
            mm = int(m.group(2))
            ss = int(m.group(3) or 0)
            return hh * 60.0 + mm + ss / 60.0
        except Exception:
            return None

    def _canonical_stage_for_summary(self, x: Any) -> str:
        s = self._safe_str(x, "Other").upper().replace(" ", "")
        if "BOF" in s or "CONVERTER" in s:
            return "BOF"
        if s == "LF" or "LADLE" in s:
            return "LF"
        if "CAST" in s or s == "CC":
            return "Casting"
        if "ROLL" in s:
            return "Rolling"
        return "Other"

    def _source_is_valid_model_operation_segment(self, seg: Dict[str, Any], top_source: str = "") -> bool:
        if not isinstance(seg, dict):
            return False
        if seg.get("is_valid_gantt_prediction") is False:
            return False

        source = self._safe_str(
            seg.get("source") or seg.get("source_kind") or seg.get("origin") or top_source,
            "",
        ).lower()
        if not source:

            source = self._safe_str(top_source, "").lower()

        bad_tokens = ["derived", "fallback", "calendar", "patch", "target", "label", "history", "missing"]
        if any(tok in source for tok in bad_tokens):
            return False
        return (
            "model" in source
            or "llm" in source
            or "prediction" in source
            or "predicted" in source
            or source == "model_output"
        )

    def _pick_operation_segments_for_summary(self, production_prediction: Any) -> List[Dict[str, Any]]:
        if not isinstance(production_prediction, dict):
            return []

        if isinstance(production_prediction.get("production_operation_segment_summary"), dict):
            return []

        top_source = self._safe_str(
            production_prediction.get("operation_segment_source")
            or self._safe_dict(production_prediction.get("operation_segment_diagnostics")).get("operation_segment_source")
            or "model_output",
            "model_output",
        )

        candidates = []
        for key in [
            "predicted_operation_segments",
            "model_predicted_operation_segments",
            "valid_predicted_operation_segments",
        ]:
            val = production_prediction.get(key)
            if isinstance(val, list):
                candidates.extend([x for x in val if isinstance(x, dict)])
                break

        return [
            seg for seg in candidates
            if self._source_is_valid_model_operation_segment(seg, top_source=top_source)
        ]

    def _build_operation_segment_summary(self, production_prediction: Any, max_examples: int = 16) -> Dict[str, Any]:

        if not isinstance(production_prediction, dict):
            return {
                "available": False,
                "source": "missing",
                "num_segments": 0,
                "heat_ids": [],
                "stage_counts": {},
                "time_span": {},
                "examples": [],
                "note": "no production operation-segment summary was provided",
            }

        existing = production_prediction.get("production_operation_segment_summary")
        if isinstance(existing, dict):
            out = dict(existing)
            out.setdefault("available", bool(out.get("num_segments", 0)))
            out.setdefault("source", "provided_summary")
            out["full_segments_removed_from_energy_prompt"] = True
            return out

        compact_like_keys = {"num_segments", "stage_counts", "heat_ids", "time_span", "examples"}
        if compact_like_keys & set(production_prediction.keys()) and not isinstance(production_prediction.get("predicted_operation_segments"), list):
            out = dict(production_prediction)
            out.setdefault("available", bool(out.get("num_segments", 0)))
            out.setdefault("source", out.get("source", "provided_summary"))
            out["full_segments_removed_from_energy_prompt"] = True
            return out

        segs = self._pick_operation_segments_for_summary(production_prediction)
        if not segs:
            source = (
                production_prediction.get("operation_segment_source")
                or self._safe_dict(production_prediction.get("operation_segment_diagnostics")).get("operation_segment_source")
                or "missing_model_operation_segments"
            )
            return {
                "available": False,
                "source": source,
                "num_segments": 0,
                "heat_ids": [],
                "stage_counts": {},
                "time_span": {},
                "examples": [],
                "note": "no valid model-output operation segments available for energy conditioning",
            }

        heat_ids: List[str] = []
        stage_counts: Dict[str, int] = {}
        unit_counts: Dict[str, int] = {}
        starts: List[float] = []
        ends: List[float] = []
        examples: List[Dict[str, Any]] = []

        for seg in segs:
            hid = self._normalize_heat_id(seg.get("heat_id") or seg.get("id") or seg.get("hn"))
            if hid and hid not in heat_ids:
                heat_ids.append(hid)

            stage = self._canonical_stage_for_summary(seg.get("stage") or seg.get("process") or seg.get("operation"))
            stage_counts[stage] = stage_counts.get(stage, 0) + 1

            unit = self._safe_str(seg.get("unit") or seg.get("resource") or seg.get("ru") or stage, stage)
            unit_counts[unit] = unit_counts.get(unit, 0) + 1

            s = self._time_to_float_for_summary(seg.get("start_min"))
            if s is None:
                s = self._time_to_float_for_summary(seg.get("start_time"))
            e = self._time_to_float_for_summary(seg.get("end_min"))
            if e is None:
                e = self._time_to_float_for_summary(seg.get("end_time"))
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
                    "duration_min": seg.get("duration_min"),
                    "source_steps": self._safe_list(seg.get("source_steps")),
                })

        return {
            "available": True,
            "source": (
                production_prediction.get("operation_segment_source")
                or self._safe_dict(production_prediction.get("operation_segment_diagnostics")).get("operation_segment_source")
                or "model_output"
            ),
            "num_segments": len(segs),
            "heat_ids": heat_ids[:max_examples],
            "stage_counts": stage_counts,
            "unit_counts": unit_counts,
            "time_span": {
                "start_min": min(starts) if starts else None,
                "end_min": max(ends) if ends else None,
            },
            "examples": examples,
            "full_segments_removed_from_energy_prompt": True,
        }

    def _extract_candidate_plan_core(self, candidate_plan: CandidatePlan) -> Dict[str, Any]:
        raw = candidate_plan.to_dict()

        return {
            "plan_id": raw.get("plan_id"),
            "day_id": raw.get("day_id"),
            "task": raw.get("task"),
            "mode": raw.get("mode"),
            "candidate_summary": raw.get("candidate_summary", {}),
            "production_statistics": raw.get("production_statistics", {}),
            "plan_snapshot": raw.get("plan_snapshot", {}),
        }

    def _extract_predicted_prod_next_state(self, candidate_plan: CandidatePlan) -> Dict[str, Any]:
        raw = candidate_plan.to_dict()
        plan_snapshot = self._safe_dict(raw.get("plan_snapshot"))
        predicted = self._safe_dict(plan_snapshot.get("predicted_prod_next_state_summary"))

        if predicted:
            return predicted

        ps = self._safe_dict(raw.get("production_statistics"))
        return {
            "transition_type": self._safe_dict(raw.get("candidate_summary", {})).get("transition_type", "unknown"),
            "affected_heat_ids": self._safe_list(ps.get("affected_heat_ids")),
            "affected_rolling_units": self._safe_list(ps.get("affected_rolling_units")),
            "predicted_time_changes": self._safe_list(ps.get("predicted_time_changes")),
            "predicted_sequence_changes": self._safe_list(ps.get("predicted_sequence_changes")),
            "stable_heat_ids": self._safe_list(ps.get("stable_heat_ids")),
            "stable_rolling_units": self._safe_list(ps.get("stable_rolling_units")),
            "predicted_local_effects": self._safe_list(ps.get("predicted_local_effects")),
            "constraint_check": self._safe_dict(ps.get("constraint_check")),
            "confidence": self._safe_dict(raw.get("candidate_summary", {})).get("confidence", 0.0),
            "comment": self._safe_dict(raw.get("candidate_summary", {})).get("comment", ""),
        }

    def _extract_energy_focus(self, energy_state: Dict[str, Any]) -> Dict[str, Any]:
        s = self._safe_dict(energy_state.get("s"))
        cf = self._safe_dict(energy_state.get("cf"))
        pi = self._safe_list(energy_state.get("pi"))
        ps = self._safe_list(energy_state.get("ps"))
        mk_top = self._safe_dict(energy_state.get("mk"))

        risky_patches = []
        overlap_patches = []
        purchase_active_patches = []
        stressed_patches = []

        for p in ps:
            rk = self._safe_list(p.get("rk"))
            pc = self._safe_dict(p.get("pc"))
            mk = self._safe_dict(p.get("mk"))

            if rk:
                risky_patches.append({
                    "id": p.get("id"),
                    "hn": p.get("hn"),
                    "sq": p.get("sq"),
                    "rk": rk,
                    "mk": {
                        "ed": self._safe_dict(mk.get("ed")),
                        "pl": self._safe_dict(mk.get("pl")),
                        "lv": mk.get("lv"),
                    },
                })

            if pc.get("ov", False):
                overlap_patches.append({
                    "id": p.get("id"),
                    "hn": p.get("hn"),
                    "sq": p.get("sq"),
                    "oc": pc.get("oc", 0),
                    "mk_lv": mk.get("lv"),
                })

            if "pp" in rk or "ps" in rk:
                purchase_active_patches.append({
                    "id": p.get("id"),
                    "hn": p.get("hn"),
                    "sq": p.get("sq"),
                    "rk": rk,
                    "dep": self._safe_dict(mk.get("ed")),
                })

            if self._safe_str(mk.get("lv")).lower() in {"tight", "stressed"}:
                stressed_patches.append({
                    "id": p.get("id"),
                    "hn": p.get("hn"),
                    "sq": p.get("sq"),
                    "mk": {
                        "hm": self._safe_dict(mk.get("hm")),
                        "ed": self._safe_dict(mk.get("ed")),
                        "pl": self._safe_dict(mk.get("pl")),
                        "lv": mk.get("lv"),
                    },
                })

        risky_patches = sorted(risky_patches, key=lambda x: len(x.get("rk", [])), reverse=True)
        stressed_patches = sorted(stressed_patches, key=lambda x: self._safe_str(self._safe_dict(x.get("mk")).get("lv")), reverse=True)

        return {
            "summary": s,
            "constraints": cf,
            "patch_index": pi,
            "mechanism_summary": mk_top,
            "top_risky_patches": risky_patches[:12],
            "overlap_patches": overlap_patches[:12],
            "purchase_active_patches": purchase_active_patches[:12],
            "top_stressed_patches": stressed_patches[:12],
        }

    def build_empty_rolling_memory(self, day_id: str) -> Dict[str, Any]:
        return {
            "d": day_id,
            "processed_batches": [],
            "processed_core_ids": [],
            "predicted_batches": [],
            "open_risks": [],
            "boundary_state": {
                "last_core_heat_id": None,
                "tail_holder_bias": {},
                "tail_purchase_bias": {},
            },
            "running_totals": {
                "purchase_power_trend": "flat",
                "purchase_steam_trend": "flat",
                "num_predicted_patches": 0,
            },
        }

    def _dedup_dict_list_by_key(self, items: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
        out = []
        seen = set()
        for x in items:
            if not isinstance(x, dict):
                continue
            k = x.get(key)
            if k in seen:
                continue
            seen.add(k)
            out.append(x)
        return out

    def _merge_rolling_memory(
        self,
        rolling_memory: Dict[str, Any],
        memory_update: Dict[str, Any],
        batch_meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        memory = dict(rolling_memory) if isinstance(rolling_memory, dict) else {}

        processed_batches = self._safe_list(memory.get("processed_batches"))
        processed_batches.append({
            "batch_id": batch_meta.get("batch_id"),
            "core_sq_range": batch_meta.get("core_sq_range"),
            "context_sq_range": batch_meta.get("context_sq_range"),
        })
        memory["processed_batches"] = processed_batches

        processed_core_ids = self._safe_list(memory.get("processed_core_ids"))
        processed_core_ids.extend(self._safe_list(memory_update.get("processed_core_ids")))
        memory["processed_core_ids"] = list(dict.fromkeys(processed_core_ids))

        predicted_batches = self._safe_list(memory.get("predicted_batches"))
        predicted_batches.extend(self._safe_list(memory_update.get("new_predicted_patches")))
        memory["predicted_batches"] = self._dedup_dict_list_by_key(predicted_batches, "heat_id")

        open_risks = self._safe_list(memory.get("open_risks"))
        open_risks.extend(self._safe_list(memory_update.get("open_risks")))
        memory["open_risks"] = open_risks[-24:]

        boundary_state = self._safe_dict(memory.get("boundary_state"))
        boundary_state.update(self._safe_dict(memory_update.get("boundary_state")))
        memory["boundary_state"] = boundary_state

        running_totals = self._safe_dict(memory.get("running_totals"))
        running_totals.update(self._safe_dict(memory_update.get("running_totals")))
        memory["running_totals"] = running_totals

        return memory

    def _build_output_schema(self) -> Dict[str, Any]:
        return {
            "task": "energy_next_state_prediction_batch",
            "mode": "energy_state_predictor_batch",
            "trajectory_id": "string",
            "predicted_energy_next_state_summary": {
                "transition_type": "direct_response|indirect_response|stable|unknown",
                "affected_patch_ids": [],
                "affected_heat_ids": [],
                "holder_trend": {
                    "BFG": "up|down|flat|unknown",
                    "COG": "up|down|flat|unknown",
                    "LDG": "up|down|flat|unknown",
                },
                "purchase_trend": {
                    "power": "up|down|flat|unknown",
                    "steam": "up|down|flat|unknown",
                },
                "risk_label_changes": [],
                "predicted_patch_stats": [
                    {
                        "heat_id": "string",
                        "response_type": "direct|indirect|none",
                        "holder_change": {
                            "BFG": "better|same|worse|unknown",
                            "COG": "better|same|worse|unknown",
                            "LDG": "better|same|worse|unknown"
                        },
                        "purchase_change": {
                            "power": "up|down|flat|unknown",
                            "steam": "up|down|flat|unknown"
                        },
                        "risk_change": "increase|decrease|same|unknown",
                        "comment": "string"
                    }
                ],
                "feasible": True,
                "overall_feasibility": "ok|warning|risky",
                "focus": "string",
                "key_risks": [],
                "confidence": 0.0,
                "comment": "string"
            },
            "memory_update": {
                "processed_core_ids": [],
                "new_predicted_patches": [],
                "open_risks": [],
                "boundary_state": {
                    "last_core_heat_id": "string",
                    "tail_holder_bias": {},
                    "tail_purchase_bias": {},
                },
                "running_totals": {
                    "purchase_power_trend": "up|down|flat",
                    "purchase_steam_trend": "up|down|flat",
                    "num_predicted_patches": 0,
                },
            }
        }

    def _normalize_heat_no(self, x: Any) -> str:
        return self._normalize_heat_id(x)

    def _extract_batch_patch_heat_nos(self, batch_ps: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        core_hns: List[str] = []
        context_hns: List[str] = []

        for p in batch_ps:
            if not isinstance(p, dict):
                continue

            hn = self._normalize_heat_no(p.get("hn"))
            if not hn:
                pc = p.get("pc", {})
                if isinstance(pc, dict):
                    hn = self._normalize_heat_no(pc.get("phn"))

            if not hn:
                continue

            context_hns.append(hn)
            if bool(p.get("_is_core", False)):
                core_hns.append(hn)

        def dedupe_keep_order(items: List[str]) -> List[str]:
            out: List[str] = []
            seen = set()
            for x in items:
                if x not in seen:
                    seen.add(x)
                    out.append(x)
            return out

        return {
            "core_heat_nos": dedupe_keep_order(core_hns),
            "context_heat_nos": dedupe_keep_order(context_hns),
        }

    def build_input_from_batch(
        self,
        energy_batch_input: Dict[str, Any],
        candidate_plan: CandidatePlan,
        day_global_focus: Optional[Dict[str, Any]] = None,
        rolling_memory: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        energy_state = self._safe_dict(energy_batch_input.get("energy_state"))
        meta = self._safe_dict(energy_batch_input.get("meta"))
        batch_meta = self._safe_dict(energy_batch_input.get("batch_meta"))

        candidate_core = self._extract_candidate_plan_core(candidate_plan)
        predicted_prod_next_state = self._extract_predicted_prod_next_state(candidate_plan)

        energy_focus = (
            day_global_focus
            if isinstance(day_global_focus, dict)
            else self._extract_energy_focus(energy_state)
        )

        memory = (
            rolling_memory
            if isinstance(rolling_memory, dict)
            else self.build_empty_rolling_memory(
                day_id=self._safe_str(
                    meta.get("d") or energy_state.get("d") or candidate_core.get("day_id")
                )
            )
        )

        batch_ps = self._safe_list(energy_state.get("ps"))
        batch_heat_info = self._extract_batch_patch_heat_nos(batch_ps)

        core_heat_nos = batch_heat_info.get("core_heat_nos", [])
        context_heat_nos = batch_heat_info.get("context_heat_nos", [])

        prod_affected_heat_ids = self._safe_list(predicted_prod_next_state.get("affected_heat_ids"))
        matched_core_heat_nos = [x for x in prod_affected_heat_ids if x in core_heat_nos]
        matched_context_heat_nos = [x for x in prod_affected_heat_ids if x in context_heat_nos]

        prod_effect_on_this_batch = {
            "prod_affected_heat_ids": prod_affected_heat_ids,
            "core_heat_nos": core_heat_nos,
            "context_heat_nos": context_heat_nos,
            "matched_core_heat_nos": matched_core_heat_nos,
            "matched_context_heat_nos": matched_context_heat_nos,
            "has_direct_effect_in_core": len(matched_core_heat_nos) > 0,
            "has_direct_effect_in_context": len(matched_context_heat_nos) > 0,
            "predicted_prod_next_state": predicted_prod_next_state,
        }

        core_mechanism_view = []
        for p in batch_ps:
            if not isinstance(p, dict):
                continue
            if not bool(p.get("_is_core", False)):
                continue

            mk = self._safe_dict(p.get("mk"))
            core_mechanism_view.append({
                "id": p.get("id"),
                "hn": p.get("hn"),
                "sq": p.get("sq"),
                "rk": self._safe_list(p.get("rk")),
                "mk": {
                    "hm": self._safe_dict(mk.get("hm")),
                    "ed": self._safe_dict(mk.get("ed")),
                    "pl": self._safe_dict(mk.get("pl")),
                    "tr": self._safe_dict(mk.get("tr")),
                    "lv": mk.get("lv"),
                },
            })

        return {
            "task_instruction": (
                "You are an energy next-state prediction model for steel plant scheduling. "
                "Your task is NOT to optimize energy. "
                "Your task is NOT to output patch adjustments. "
                "Your task is to predict the next local energy state summary for the current batch, "
                "conditioned on the current energy batch state and the predicted production next-state summary. "
                "Only analyze the core patches in this batch. "
                "Context patches are only for continuity. "
                "Use rolling_memory as the authoritative history from previous batches. "
                "Use mechanism-informed patch features when available, especially holder margin, external dependency, peak-load proxy, and mechanism stress level. "
                "When the predicted production effect does not directly hit the current core patches, "
                "prefer stable or weak indirect response. "
                "Focus on holder trend, purchase trend, risk-label change, and patch-level state change. "
                "Return exactly one JSON object. "
                "Do not output reasoning, analysis, explanation, markdown, or any text outside the JSON."
            ),
            "input": {
                "day_id": meta.get("d") or energy_state.get("d") or candidate_core.get("day_id"),
                "transition_task": {
                    "stage": "experiment_2_energy_state_validation",
                    "goal": "predict_batch_energy_next_state",
                    "principles": [
                        "predict_state_not_adjustment",
                        "respond_to_predicted_production_effect",
                        "preserve_holder_safety",
                        "preserve_patch_consistency",
                        "prefer_stable_if_no_direct_hit"
                    ],
                },
                "production_prediction": {
                    "candidate_plan_meta": candidate_core,
                    "predicted_prod_next_state_summary": predicted_prod_next_state,
                    "prod_effect_on_this_batch": prod_effect_on_this_batch,
                },
                "day_global_focus": energy_focus,
                "rolling_memory": memory,
                "batch_info": batch_meta,
                "energy_state": {
                    "d": energy_state.get("d"),
                    "m": energy_state.get("m"),
                    "s": energy_state.get("s", {}),
                    "cf": energy_state.get("cf", {}),
                    "mk": energy_state.get("mk", {}),
                    "pi": energy_state.get("pi", []),
                    "ps": energy_state.get("ps", []),
                    "meta": energy_state.get("meta", {}),
                },
                "core_mechanism_view": core_mechanism_view,
            },
            "output_schema": self._build_output_schema(),
        }

    def build_messages(self, model_input: Dict[str, Any]) -> List[Dict[str, str]]:
        system_prompt = (
            "You are an energy next-state prediction model.\n"
            "Your job is NOT to optimize energy.\n"
            "Your job is NOT to output patch adjustments.\n"
            "Your job is to predict the next local energy state summary.\n"
            "\n"
            "Hard rules:\n"
            "1. Do NOT output patch_adjustments.\n"
            "2. Do NOT output optimization advice.\n"
            "3. Output only predicted_energy_next_state_summary and memory_update.\n"
            "4. If there is no direct production effect in the core batch, prefer stable or weak indirect response.\n"
            "5. Use patches as state objects, not as adjustment targets.\n"
            "6. Be conservative when uncertain.\n"
            "\n"
            "You must return exactly one JSON object with:\n"
            "- task = energy_next_state_prediction_batch\n"
            "- mode = energy_state_predictor_batch\n"
            "- predicted_energy_next_state_summary = {...}\n"
            "- memory_update = {...}\n"
            "\n"
            "Only output valid JSON.\n"
            "Do not explain outside JSON."
        )
        user_prompt = json.dumps(model_input, ensure_ascii=False, indent=2)
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def call_model(self, model_input: Dict[str, Any]) -> Dict[str, Any]:
        messages = self.build_messages(model_input)
        raw_text = self.chat_model.generate(messages)

        print("\n=== RAW ENERGY MODEL OUTPUT START ===")
        print(raw_text)
        print("=== RAW ENERGY MODEL OUTPUT END ===\n")

        parsed = _safe_json_loads(raw_text)

        if isinstance(parsed, dict):
            parsed["_raw_text"] = raw_text
            return parsed

        if isinstance(parsed, list):
            return {
                "_raw_text": raw_text,
                "_raw_list_output": parsed,
                "task": "energy_next_state_prediction_batch",
                "mode": "energy_state_predictor_batch",
                "predicted_energy_next_state_summary": {
                    "transition_type": "unknown",
                    "affected_patch_ids": [],
                    "affected_heat_ids": [],
                    "holder_trend": {"BFG": "unknown", "COG": "unknown", "LDG": "unknown"},
                    "purchase_trend": {"power": "unknown", "steam": "unknown"},
                    "risk_label_changes": [],
                    "predicted_patch_stats": [],
                    "feasible": True,
                    "overall_feasibility": "ok",
                    "focus": "",
                    "key_risks": [],
                    "confidence": 0.0,
                    "comment": "Model returned list; normalized to empty energy next-state summary.",
                },
                "memory_update": {
                    "processed_core_ids": [],
                    "new_predicted_patches": [],
                    "open_risks": [],
                    "boundary_state": {},
                    "running_totals": {
                        "purchase_power_trend": "flat",
                        "purchase_steam_trend": "flat",
                        "num_predicted_patches": 0,
                    },
                },
            }

        raise TypeError("The parsed energy model output is neither a dict nor a list.")

    def _normalize_predicted_energy_summary(self, raw_output: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(raw_output, dict):
            return {}

        summary = raw_output.get("predicted_energy_next_state_summary")
        if isinstance(summary, dict):
            return summary

        guessed_keys = {
            "transition_type",
            "affected_patch_ids",
            "affected_heat_ids",
            "holder_trend",
            "purchase_trend",
            "risk_label_changes",
            "predicted_patch_stats",
            "feasible",
            "overall_feasibility",
            "focus",
            "key_risks",
            "confidence",
            "comment",
        }
        if any(k in raw_output for k in guessed_keys):
            return {
                "transition_type": raw_output.get("transition_type", "unknown"),
                "affected_patch_ids": self._safe_list(raw_output.get("affected_patch_ids")),
                "affected_heat_ids": self._safe_list(raw_output.get("affected_heat_ids")),
                "holder_trend": self._safe_dict(raw_output.get("holder_trend")),
                "purchase_trend": self._safe_dict(raw_output.get("purchase_trend")),
                "risk_label_changes": self._safe_list(raw_output.get("risk_label_changes")),
                "predicted_patch_stats": self._safe_list(raw_output.get("predicted_patch_stats")),
                "feasible": bool(raw_output.get("feasible", True)),
                "overall_feasibility": self._safe_str(raw_output.get("overall_feasibility", "ok")),
                "focus": self._safe_str(raw_output.get("focus", "")),
                "key_risks": self._safe_list(raw_output.get("key_risks")),
                "confidence": self._safe_float(raw_output.get("confidence", 0.0), 0.0),
                "comment": self._safe_str(raw_output.get("comment", "")),
            }

        return {
            "transition_type": "unknown",
            "affected_patch_ids": [],
            "affected_heat_ids": [],
            "holder_trend": {"BFG": "unknown", "COG": "unknown", "LDG": "unknown"},
            "purchase_trend": {"power": "unknown", "steam": "unknown"},
            "risk_label_changes": [],
            "predicted_patch_stats": [],
            "feasible": True,
            "overall_feasibility": "ok",
            "focus": "",
            "key_risks": [],
            "confidence": 0.0,
            "comment": "No valid predicted_energy_next_state_summary found.",
        }

    def _normalize_memory_update(self, raw_output: Dict[str, Any], predicted_summary: Dict[str, Any]) -> Dict[str, Any]:
        memory_update = raw_output.get("memory_update", {})
        if isinstance(memory_update, dict):
            running_totals = self._safe_dict(memory_update.get("running_totals"))
            if "num_predicted_patches" not in running_totals:
                running_totals["num_predicted_patches"] = len(self._safe_list(predicted_summary.get("predicted_patch_stats")))
                memory_update["running_totals"] = running_totals
            if "new_predicted_patches" not in memory_update:
                memory_update["new_predicted_patches"] = self._safe_list(predicted_summary.get("predicted_patch_stats"))
            return memory_update

        predicted_patch_stats = self._safe_list(predicted_summary.get("predicted_patch_stats"))
        processed_core_ids = []
        for item in predicted_patch_stats:
            if isinstance(item, dict) and item.get("heat_id"):
                processed_core_ids.append(item.get("heat_id"))

        purchase_trend = self._safe_dict(predicted_summary.get("purchase_trend"))

        return {
            "processed_core_ids": processed_core_ids,
            "new_predicted_patches": predicted_patch_stats,
            "open_risks": self._safe_list(predicted_summary.get("key_risks")),
            "boundary_state": {},
            "running_totals": {
                "purchase_power_trend": purchase_trend.get("power", "flat"),
                "purchase_steam_trend": purchase_trend.get("steam", "flat"),
                "num_predicted_patches": len(predicted_patch_stats),
            },
        }

    def parse_output(
        self,
        raw_output: Dict[str, Any],
        candidate_plan: CandidatePlan,
        energy_input: Dict[str, Any],
    ) -> EnergyTrajectory:
        meta = self._safe_dict(energy_input.get("meta"))
        energy_state = self._safe_dict(energy_input.get("energy_state"))

        day_id = (
            meta.get("d")
            or energy_state.get("d")
            or candidate_plan.to_dict().get("day_id")
            or "unknown_day"
        )

        predicted_summary = self._normalize_predicted_energy_summary(raw_output)
        memory_update = self._normalize_memory_update(raw_output, predicted_summary)

        predicted_patch_stats = self._safe_list(predicted_summary.get("predicted_patch_stats"))

        patch_summaries = []
        for item in predicted_patch_stats:
            if isinstance(item, dict):
                patch_summaries.append({
                    "patch_id": item.get("patch_id") or item.get("heat_id"),
                    "optimized_patch_summary": item,
                })

        num_patches = max(len(predicted_patch_stats), 1)
        feasible_flag = bool(predicted_summary.get("feasible", True))
        num_feasible_patches = len(predicted_patch_stats) if feasible_flag else 0
        if not predicted_patch_stats:
            num_feasible_patches = 1 if feasible_flag else 0

        feasible_patch_ratio = (
            float(num_feasible_patches) / float(num_patches)
            if num_patches > 0 else 0.0
        )

        return EnergyTrajectory(
            trajectory_id=raw_output.get("trajectory_id", f"energy_state_pred_{uuid.uuid4().hex[:8]}"),
            plan_id=candidate_plan.plan_id,
            day_id=day_id,
            heat_id=None,
            patch_id=None,
            task=raw_output.get("task", "energy_next_state_prediction_batch"),
            mode=raw_output.get("mode", "energy_state_predictor_batch"),

            optimized_patch_summary=predicted_summary,
            optimized_patch_minute_trajectory=[],

            patch_summaries=patch_summaries,
            num_patches=num_patches,
            num_feasible_patches=num_feasible_patches,
            feasible_patch_ratio=feasible_patch_ratio,
            energy_state_snapshot={
                "predicted_energy_next_state_summary": predicted_summary,
                "memory_update": memory_update,
            },
            raw_output=raw_output,
            meta={"source": "energy_agent_adapter_state_model"},
        )

    def respond_batch(
        self,
        energy_batch_input: Dict[str, Any],
        candidate_plan: CandidatePlan,
        day_global_focus: Optional[Dict[str, Any]] = None,
        rolling_memory: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        model_input = self.build_input_from_batch(
            energy_batch_input=energy_batch_input,
            candidate_plan=candidate_plan,
            day_global_focus=day_global_focus,
            rolling_memory=rolling_memory,
        )
        raw_output = self.call_model(model_input)

        traj = self.parse_output(
            raw_output=raw_output,
            candidate_plan=candidate_plan,
            energy_input=energy_batch_input,
        )

        predicted_summary = self._normalize_predicted_energy_summary(raw_output)
        memory_update = self._normalize_memory_update(raw_output, predicted_summary)

        return {
            "trajectory": traj,
            "raw_output": raw_output,
            "memory_update": memory_update,
        }

    def respond_day_in_batches(
        self,
        energy_state_adapter: Any,
        day_id: str,
        candidate_plan: CandidatePlan,
        core_size: int = 6,
        left_context: int = 1,
        right_context: int = 1,
        stride: Optional[int] = None,
    ) -> Dict[str, Any]:
        full_state = energy_state_adapter.build_energy_state(day_id)
        day_global_focus = energy_state_adapter.build_energy_global_focus(day_id=day_id, full_state=full_state)
        rolling_memory = energy_state_adapter.build_empty_rolling_memory(day_id)

        batch_states = energy_state_adapter.iter_energy_batch_states(
            day_id=day_id,
            core_size=core_size,
            left_context=left_context,
            right_context=right_context,
            stride=stride,
            mode="energy",
            full_state=full_state,
        )

        batch_results = []
        all_predicted_patch_stats: List[Dict[str, Any]] = []

        for batch in batch_states:
            batch_meta = self._safe_dict(batch.get("meta"))

            energy_batch_input = {
                "energy_state": batch,
                "meta": {"d": day_id},
                "batch_meta": {
                    "batch_id": batch_meta.get("batch_id"),
                    "core_sq_range": batch_meta.get("core_sq_range"),
                    "context_sq_range": batch_meta.get("context_sq_range"),
                    "core_ids": batch_meta.get("core_ids", []),
                    "context_ids": batch_meta.get("context_ids", []),
                },
            }

            result = self.respond_batch(
                energy_batch_input=energy_batch_input,
                candidate_plan=candidate_plan,
                day_global_focus=day_global_focus,
                rolling_memory=rolling_memory,
            )

            traj = result["trajectory"]
            raw_output = result["raw_output"]
            memory_update = result["memory_update"]

            rolling_memory = self._merge_rolling_memory(
                rolling_memory=rolling_memory,
                memory_update=memory_update,
                batch_meta=energy_batch_input["batch_meta"],
            )

            predicted_summary = self._normalize_predicted_energy_summary(raw_output)
            predicted_patch_stats = self._safe_list(predicted_summary.get("predicted_patch_stats"))

            batch_ps = self._safe_list(batch.get("ps"))
            core_ids = set(self._safe_list(energy_batch_input["batch_meta"].get("core_ids")))

            hn_to_patch_id: Dict[str, str] = {}
            patch_id_to_hn: Dict[str, str] = {}

            for p in batch_ps:
                if not isinstance(p, dict):
                    continue

                pid = self._safe_str(p.get("id"))
                hn = self._safe_str(p.get("hn"))
                if not hn:
                    hn = self._safe_str(self._safe_dict(p.get("pc")).get("phn"))

                if pid and hn:
                    hn_to_patch_id[hn] = pid
                    patch_id_to_hn[pid] = hn

            filtered_patch_stats: List[Dict[str, Any]] = []

            for x in predicted_patch_stats:
                if not isinstance(x, dict):
                    continue

                item = dict(x)
                short_heat_id = self._safe_str(item.get("heat_id"))
                patch_id = self._safe_str(item.get("patch_id"))

                if not patch_id and short_heat_id:
                    patch_id = hn_to_patch_id.get(short_heat_id, "")
                if not short_heat_id and patch_id:
                    short_heat_id = patch_id_to_hn.get(patch_id, "")

                item["heat_id"] = short_heat_id
                if patch_id:
                    item["patch_id"] = patch_id

                if patch_id and patch_id in core_ids:
                    filtered_patch_stats.append(item)

            all_predicted_patch_stats.extend(filtered_patch_stats)

            batch_results.append({
                "batch_id": energy_batch_input["batch_meta"].get("batch_id"),
                "core_sq_range": energy_batch_input["batch_meta"].get("core_sq_range"),
                "context_sq_range": energy_batch_input["batch_meta"].get("context_sq_range"),
                "trajectory": traj,
                "raw_output": raw_output,
                "memory_after_batch": rolling_memory,
                "filtered_predicted_patch_stats": filtered_patch_stats,
            })

        all_deduped: List[Dict[str, Any]] = []
        seen_all = set()
        for x in all_predicted_patch_stats:
            if not isinstance(x, dict):
                continue
            key = x.get("patch_id") or x.get("heat_id")
            if not key or key in seen_all:
                continue
            seen_all.add(key)
            all_deduped.append(x)

        changed_patch_stats: List[Dict[str, Any]] = []

        for x in all_deduped:
            if not isinstance(x, dict):
                continue

            response_type = self._safe_str(x.get("response_type")).lower()
            risk_change = self._safe_str(x.get("risk_change")).lower()
            holder_change = self._safe_dict(x.get("holder_change"))
            purchase_change = self._safe_dict(x.get("purchase_change"))

            holder_changed = any(
                self._safe_str(holder_change.get(k)).lower() not in {"", "same", "unknown"}
                for k in ["BFG", "COG", "LDG"]
            )
            purchase_changed = any(
                self._safe_str(purchase_change.get(k)).lower() not in {"", "flat", "unknown"}
                for k in ["power", "steam"]
            )

            changed = (
                response_type in {"direct", "indirect"}
                or risk_change not in {"", "same", "unknown"}
                or holder_changed
                or purchase_changed
            )

            if changed:
                changed_patch_stats.append(x)

        affected_patch_ids = [
            x.get("patch_id")
            for x in changed_patch_stats
            if isinstance(x, dict) and x.get("patch_id")
        ]
        affected_heat_ids = [
            x.get("heat_id")
            for x in changed_patch_stats
            if isinstance(x, dict) and x.get("heat_id")
        ]

        holder_votes = {"BFG": [], "COG": [], "LDG": []}
        purchase_power_votes = []
        purchase_steam_votes = []
        risk_label_changes = []

        for x in changed_patch_stats:
            hc = self._safe_dict(x.get("holder_change"))
            pc = self._safe_dict(x.get("purchase_change"))

            for gas in ["BFG", "COG", "LDG"]:
                v = self._safe_str(hc.get(gas))
                if v:
                    holder_votes[gas].append(v)

            pwr = self._safe_str(pc.get("power"))
            stm = self._safe_str(pc.get("steam"))
            if pwr:
                purchase_power_votes.append(pwr)
            if stm:
                purchase_steam_votes.append(stm)

            rc = self._safe_str(x.get("risk_change"))
            if rc and rc not in {"same", "unknown"}:
                risk_label_changes.append({
                    "heat_id": x.get("heat_id"),
                    "patch_id": x.get("patch_id"),
                    "risk_change": rc,
                })

        def _majority_or_default(votes: List[str], default: str) -> str:
            if not votes:
                return default
            cnt: Dict[str, int] = {}
            for v in votes:
                cnt[v] = cnt.get(v, 0) + 1
            return sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

        holder_trend = {
            "BFG": _majority_or_default(holder_votes["BFG"], "same"),
            "COG": _majority_or_default(holder_votes["COG"], "same"),
            "LDG": _majority_or_default(holder_votes["LDG"], "same"),
        }

        running_totals = self._safe_dict(rolling_memory.get("running_totals"))
        purchase_trend = {
            "power": _majority_or_default(purchase_power_votes, running_totals.get("purchase_power_trend", "flat")),
            "steam": _majority_or_default(purchase_steam_votes, running_totals.get("purchase_steam_trend", "flat")),
        }

        final_summary = {
            "transition_type": "day_batch_prediction",

            "affected_patch_ids": affected_patch_ids,
            "affected_heat_ids": affected_heat_ids,
            "holder_trend": holder_trend,
            "purchase_trend": purchase_trend,
            "risk_label_changes": risk_label_changes,

            "predicted_patch_stats": all_deduped,
            "feasible": True,
            "overall_feasibility": "ok",
            "focus": "day_energy_next_state_prediction",
            "key_risks": self._safe_list(rolling_memory.get("open_risks")),
            "confidence": 0.0 if not changed_patch_stats else 0.6,
            "comment": (
                f"day-level batch prediction finished with {len(batch_results)} batches; "
                f"all_core_predictions={len(all_deduped)}, changed_predictions={len(changed_patch_stats)}"
            ),
        }

        return {
            "day_id": day_id,
            "batch_results": batch_results,

            "final_predicted_patch_stats": changed_patch_stats,
            "all_predicted_patch_stats": all_deduped,
            "rolling_memory": rolling_memory,
            "day_global_focus": day_global_focus,
            "final_summary": final_summary,
        }

    def respond(self, energy_input: Dict[str, Any], candidate_plan: CandidatePlan) -> EnergyTrajectory:
        energy_state = self._safe_dict(energy_input.get("energy_state"))
        candidate_core = self._extract_candidate_plan_core(candidate_plan)
        predicted_prod_next_state = self._extract_predicted_prod_next_state(candidate_plan)

        model_input = {
            "task_instruction": (
                "Given the day-level energy patch states and the predicted production next-state summary, "
                "predict the corresponding energy next-state summary. "
                "Only use energy-side state plus production next-state prediction. "
                "Do not optimize energy. "
                "Do not output patch adjustments. "
                "Work at patch/heat level only. "
                "Only output JSON."
            ),
            "input": {
                "day_id": self._safe_dict(energy_input.get("meta")).get("d")
                         or self._safe_dict(energy_input.get("energy_state")).get("d")
                         or candidate_plan.to_dict().get("day_id"),
                "production_prediction": {
                    "candidate_plan_meta": candidate_core,
                    "predicted_prod_next_state_summary": predicted_prod_next_state,
                },
                "energy_state": energy_state,
                "energy_focus": self._extract_energy_focus(energy_state),
            },
            "output_schema": self._build_output_schema(),
        }
        raw_output = self.call_model(model_input)
        return self.parse_output(raw_output, candidate_plan, energy_input)

def _eaa_to_float(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default

def _eaa_get_nested(d, path, default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur

def _eaa_as_list(x):
    return x if isinstance(x, list) else ([] if x is None else [x])

def _eaa_norm_set(x):
    return {str(v).strip() for v in _eaa_as_list(x) if str(v).strip()}

def _eaa_rmse(values):
    if not values:
        return 0.0
    return (sum(v * v for v in values) / len(values)) ** 0.5

def _eaa_numeric_features(p):

    if not isinstance(p, dict):
        return {}

    generation = p.get("g") if isinstance(p.get("g"), dict) else p.get("generation", {})
    storage = p.get("st") if isinstance(p.get("st"), dict) else p.get("storage", {})
    demand = p.get("dm") if isinstance(p.get("dm"), dict) else p.get("demand", {})
    external = p.get("ex") if isinstance(p.get("ex"), dict) else p.get("external", {})

    def gas_total(gas_short, gas_long):
        obj = generation.get(gas_short, generation.get(gas_long, {})) if isinstance(generation, dict) else {}
        if isinstance(obj, dict):
            return obj.get("tot", obj.get("total", obj.get("value")))
        return obj

    def holder_stat(gas_short, gas_long, stat):
        obj = storage.get(gas_short, storage.get(gas_long, {})) if isinstance(storage, dict) else {}
        if isinstance(obj, dict):
            return obj.get(stat)
        return None

    def dm_stat(short, long, stat):
        obj = demand.get(short, demand.get(long, {})) if isinstance(demand, dict) else {}
        if isinstance(obj, dict):
            return obj.get(stat, obj.get({"a": "avg", "tot": "total", "pk": "peak"}.get(stat, stat)))
        return obj if stat in {"a", "tot"} else None

    def ex_stat(short, long, stat):
        obj = external.get(short, external.get(long, {})) if isinstance(external, dict) else {}
        if isinstance(obj, dict):
            return obj.get(stat, obj.get({"a": "avg", "tot": "total"}.get(stat, stat)))
        return obj if stat in {"a", "tot"} else None

    feats = {
        "bfg_gen_total": gas_total("b", "BFG"),
        "cog_gen_total": gas_total("c", "COG"),
        "ldg_gen_total": gas_total("l", "LDG"),
        "bfg_holder_end": holder_stat("b", "BFG", "e"),
        "cog_holder_end": holder_stat("c", "COG", "e"),
        "ldg_holder_end": holder_stat("l", "LDG", "e"),
        "bfg_holder_mean": holder_stat("b", "BFG", "m"),
        "cog_holder_mean": holder_stat("c", "COG", "m"),
        "ldg_holder_mean": holder_stat("l", "LDG", "m"),
        "steam_load_total": dm_stat("s", "steam", "tot"),
        "power_load_total": dm_stat("e", "power", "tot"),
        "steam_load_peak": dm_stat("s", "steam", "pk"),
        "power_load_peak": dm_stat("e", "power", "pk"),
        "purchased_steam_total": ex_stat("s", "steam", "tot"),
        "purchased_power_total": ex_stat("e", "power", "tot"),
    }
    return {k: _eaa_to_float(v, None) for k, v in feats.items() if v is not None}

def _eaa_risk_set(p):
    if not isinstance(p, dict):
        return set()
    for key in ["rk", "risk_flags", "risk_labels", "predicted_risk_flags"]:
        if key in p:
            return _eaa_norm_set(p.get(key))

    rc = str(p.get("risk_change", "")).strip().lower()
    if rc and rc not in {"same", "unknown", "none"}:
        return {rc}
    return set()

def _eaa_strip_for_energy_prompt(self, obj):
    if hasattr(self, "_strip_for_energy"):
        return self._strip_for_energy(obj)
    return obj

def _eaa_future_context(sample):
    input_state = sample.get("input_state", {}) if isinstance(sample, dict) else {}
    ctx = input_state.get("future_patch_context", [])
    return ctx if isinstance(ctx, list) else []

def _eaa_context_patch_id(ctx):
    if not isinstance(ctx, dict):
        return ""
    return str(ctx.get("id") or ctx.get("patch_id") or ctx.get("p") or "").strip()

def _eaa_context_heat_id(self, ctx):
    if not isinstance(ctx, dict):
        return ""
    pc = ctx.get("pc", {}) if isinstance(ctx.get("pc"), dict) else {}
    return self._normalize_heat_id(
        ctx.get("hn")
        or ctx.get("heat_id")
        or ctx.get("canonical_heat_id")
        or pc.get("phn")
        or ctx.get("id")
        or ctx.get("p")
    )

def _eaa_pick_prediction_steps(prediction):
    if not isinstance(prediction, dict):
        return []
    for key in ["future_patch_predictions", "predictions", "energy_future", "predicted_energy_future", "future_patches"]:
        if isinstance(prediction.get(key), list):
            return prediction.get(key)
    return []

def _eaa_build_empty_prediction_from_context(self, ctx, step):
    patch_id = _eaa_context_patch_id(ctx)
    heat_id = _eaa_context_heat_id(self, ctx)
    return {
        "step": step,
        "patch_id": patch_id,
        "id": patch_id,
        "p": patch_id,
        "heat_id": heat_id,
        "hn": heat_id,
        "g": {"b": {"tot": 0.0}, "c": {"tot": 0.0}, "l": {"tot": 0.0}},
        "st": {"b": {"e": 0.0, "m": 0.0}, "c": {"e": 0.0, "m": 0.0}, "l": {"e": 0.0, "m": 0.0}},
        "dm": {"s": {"tot": 0.0, "pk": 0.0}, "e": {"tot": 0.0, "pk": 0.0}},
        "ex": {"s": {"tot": 0.0}, "e": {"tot": 0.0}},
        "rk": [],
        "response_type": "unknown",
        "confidence": 0.0,
        "comment": "fallback_empty_energy_prediction_bound_to_future_context",
    }

def _eaa_normalize_horizon_prediction_with_context(self, parsed, sample, raw_text="", parse_error=None):

    if not isinstance(parsed, dict):
        parsed = {}

    input_state = self._safe_dict(sample.get("input_state")) if isinstance(sample, dict) else {}
    horizon = int(input_state.get("horizon") or sample.get("horizon") or 0)
    future_context = _eaa_future_context(sample)
    if horizon <= 0:
        horizon = len(future_context)

    pred_list = _eaa_pick_prediction_steps(parsed)

    by_step = {}
    by_patch_id = {}
    for idx, pred in enumerate(pred_list):
        if not isinstance(pred, dict):
            continue
        try:
            st = int(pred.get("step"))
            by_step[st] = pred
        except Exception:
            pass
        pid = self._safe_str(pred.get("patch_id") or pred.get("id") or pred.get("p"))
        if pid:
            by_patch_id[pid] = pred
        else:
            by_patch_id[f"__idx_{idx}"] = pred

    normalized = []
    for idx in range(horizon):
        step = idx + 1
        ctx = future_context[idx] if idx < len(future_context) and isinstance(future_context[idx], dict) else {}
        ctx_pid = _eaa_context_patch_id(ctx)
        ctx_hid = _eaa_context_heat_id(self, ctx)

        pred = by_step.get(step) or by_patch_id.get(ctx_pid)
        if pred is None and idx < len(pred_list) and isinstance(pred_list[idx], dict):
            pred = pred_list[idx]
        if not isinstance(pred, dict):
            pred = _eaa_build_empty_prediction_from_context(self, ctx, step)
        else:
            pred = dict(pred)

        rk = pred.get("rk")
        if rk is None:
            for alias in ["risk_flags", "risk_labels", "predicted_risk_flags"]:
                if isinstance(pred.get(alias), list):
                    rk = pred.get(alias)
                    break
        if rk is None:
            rk = []
        pred["rk"] = self._dedup_preserve_order([self._safe_str(x) for x in self._safe_list(rk) if self._safe_str(x)])

        pred["step"] = step
        if ctx_pid:
            pred["patch_id"] = ctx_pid
            pred["id"] = ctx_pid
            pred["p"] = ctx_pid
        else:
            pred.setdefault("patch_id", self._safe_str(pred.get("id") or pred.get("p")))
            pred.setdefault("id", pred.get("patch_id"))
            pred.setdefault("p", pred.get("patch_id"))

        if ctx_hid:
            pred["heat_id"] = ctx_hid
            pred["hn"] = ctx_hid
        else:
            hid = self._normalize_heat_id(pred.get("heat_id") or pred.get("hn") or pred.get("patch_id"))
            pred["heat_id"] = hid
            pred["hn"] = hid

        pred["context_binding"] = {
            "source": "future_patch_context",
            "patch_id_forced": bool(ctx_pid),
            "heat_id_forced": bool(ctx_hid),
        }
        normalized.append(pred)

    out = dict(parsed)
    out["task"] = out.get("task", "energy_horizon_prediction")
    out["mode"] = out.get("mode", "energy_patch_horizon_predictor_gantt_conditioned")
    out["horizon"] = horizon
    out["future_patch_predictions"] = normalized
    out.setdefault("summary", {})
    if raw_text is not None:
        out["_raw_text"] = raw_text
    if parse_error:
        out["_parse_error"] = str(parse_error)
    return out

def _eaa_build_horizon_input(self, sample, production_prediction=None):
    input_state_raw = self._safe_dict(sample.get("input_state"))
    input_state = _eaa_strip_for_energy_prompt(self, input_state_raw)

    production_summary = self._build_operation_segment_summary(production_prediction)
    if not production_summary.get("available") and isinstance(input_state_raw.get("production_operation_segment_summary"), dict):
        production_summary = self._build_operation_segment_summary(input_state_raw.get("production_operation_segment_summary"))
    if not production_summary.get("available") and isinstance(input_state_raw.get("production_prediction"), dict):
        production_summary = self._build_operation_segment_summary(input_state_raw.get("production_prediction"))

    input_state = dict(input_state)

    input_state.pop("production_prediction", None)
    input_state.pop("future_production_states", None)
    input_state.pop("production_patch_prediction", None)
    input_state["production_operation_segment_summary"] = production_summary
    input_state["task_role"] = "energy_horizon_prediction_conditioned_on_operation_schedule"

    horizon = input_state.get("horizon") or sample.get("horizon")

    return {
        "task_instruction": (
            "You are an energy patch-level horizon prediction model for a steel plant production-energy world model. "
            "Given observed historical energy patches, known future patch heat/time context, and a compact summary of "
            "the predicted operation-level production Gantt schedule, predict the future energy states for the next K patches. "
            "This is prediction, not optimization. Do not output energy adjustment advice. "
            "Do not predict production active heats or heat-patch alignment; production schedule information is already summarized "
            "in production_operation_segment_summary. "
            "Future patch identifiers are known: keep step k aligned with future_patch_context[k]. "
            "Do not access target_future. Return only valid JSON."
        ),
        "input": input_state,
        "output_schema": {
            "task": "energy_horizon_prediction",
            "mode": "energy_patch_horizon_predictor_gantt_conditioned",
            "horizon": horizon,
            "future_patch_predictions": [
                {
                    "step": 1,
                    "patch_id": "must equal future_patch_context[step-1].id",
                    "heat_id": "must equal future_patch_context[step-1].hn",
                    "g": {"b": {"tot": 0.0}, "c": {"tot": 0.0}, "l": {"tot": 0.0}},
                    "st": {"b": {"e": 0.0, "m": 0.0}, "c": {"e": 0.0, "m": 0.0}, "l": {"e": 0.0, "m": 0.0}},
                    "dm": {"s": {"tot": 0.0, "pk": 0.0}, "e": {"tot": 0.0, "pk": 0.0}},
                    "ex": {"s": {"tot": 0.0}, "e": {"tot": 0.0}},
                    "rk": [],
                    "response_type": "direct|indirect|none|unknown",
                    "production_activity_basis": "operation_segment_summary|history_only|unknown",
                    "confidence": 0.0,
                    "comment": "string"
                }
            ],
            "summary": {
                "holder_trend": {"BFG": "up|down|flat|unknown", "COG": "up|down|flat|unknown", "LDG": "up|down|flat|unknown"},
                "purchase_trend": {"power": "up|down|flat|unknown", "steam": "up|down|flat|unknown"},
                "overall_feasibility": "ok|warning|risky",
                "production_conditioning_available": bool(production_summary.get("available", False)),
                "confidence": 0.0,
                "comment": "string"
            }
        }
    }

def _eaa_build_horizon_messages(self, model_input):
    system_prompt = (
        "You are an energy patch-level horizon prediction model.\n"
        "Predict future energy patch states from historical energy patches, known future patch context, "
        "and a compact operation-level production Gantt schedule summary.\n"
        "Do NOT optimize. Do NOT output adjustment actions.\n"
        "Do NOT predict production active heats or heat-patch alignment.\n"
        "For each step, keep patch_id and heat_id aligned with future_patch_context.\n"
        "Use production_operation_segment_summary only as functional activity context for energy response.\n"
        "Return exactly one JSON object with task, mode, horizon, future_patch_predictions, and summary.\n"
        "Only output valid JSON."
    )
    user_prompt = json.dumps(model_input, ensure_ascii=False, indent=2)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

def _eaa_predict_horizon(self, sample, production_prediction=None):
    model_input = self.build_horizon_input(sample, production_prediction=production_prediction)
    messages = self.build_horizon_messages(model_input)
    raw_text = ""
    parse_error = None

    try:
        raw_text = self.chat_model.generate(messages)
    except Exception as e:
        parse_error = f"energy_model_generate_failed:{type(e).__name__}:{e}"
        raw_text = ""

    print("\n=== RAW ENERGY HORIZON MODEL OUTPUT START ===")
    print(raw_text)
    print("=== RAW ENERGY HORIZON MODEL OUTPUT END ===\n")

    parsed = {}
    if raw_text and str(raw_text).strip():
        try:
            parsed = _safe_json_loads(raw_text)
        except Exception as e:
            parse_error = f"energy_model_parse_failed:{type(e).__name__}:{e}"
            parsed = {}
    else:
        parse_error = parse_error or "empty_energy_model_output"

    if not isinstance(parsed, dict):
        parsed = {}

    return _eaa_normalize_horizon_prediction_with_context(
        self,
        parsed,
        sample,
        raw_text=raw_text or "",
        parse_error=parse_error,
    )

def _eaa_risk_prf(tp, fp, fn):
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return precision, recall, f1

def _eaa_evaluate_horizon_prediction(self, prediction, target_future):
    pred_list = self._safe_list(prediction.get("future_patch_predictions"))
    tgt_list = self._safe_list(self._safe_dict(target_future).get("energy_future"))

    pred_by_id = {}
    for idx, p in enumerate(pred_list):
        if not isinstance(p, dict):
            continue
        pid = self._safe_str(p.get("patch_id") or p.get("id") or p.get("p"))
        if pid:
            pred_by_id[pid] = p
        else:
            pred_by_id[f"__idx_{idx}"] = p

    abs_errors = {}
    sq_errors = {}
    counts = {}
    risk_tp = risk_fp = risk_fn = 0
    risk_tp_no_ov = risk_fp_no_ov = risk_fn_no_ov = 0
    purchase_tp = purchase_fp = purchase_fn = 0
    holder_tp = holder_fp = holder_fn = 0
    matched = 0
    missing_predictions = 0

    known_context_risks = {"ov"}
    purchase_risks = {"pp", "ps"}
    holder_risks = {"b_low", "b_high", "c_low", "c_high", "l_low", "l_high"}

    for idx, tgt in enumerate(tgt_list):
        if not isinstance(tgt, dict):
            continue
        pid = self._safe_str(tgt.get("id") or tgt.get("patch_id") or tgt.get("p"))
        pred = pred_by_id.get(pid)
        if pred is None and idx < len(pred_list) and isinstance(pred_list[idx], dict):
            pred = pred_list[idx]
        if pred is None:
            missing_predictions += 1
            continue
        matched += 1

        pf = _eaa_numeric_features(pred)
        tf = _eaa_numeric_features(tgt)
        for key, true_v in tf.items():
            if key not in pf:
                continue
            err = _eaa_to_float(pf[key]) - _eaa_to_float(true_v)
            abs_errors[key] = abs_errors.get(key, 0.0) + abs(err)
            sq_errors[key] = sq_errors.get(key, 0.0) + err * err
            counts[key] = counts.get(key, 0) + 1

        pr = _eaa_risk_set(pred)
        tr = _eaa_risk_set(tgt)
        risk_tp += len(pr & tr)
        risk_fp += len(pr - tr)
        risk_fn += len(tr - pr)

        pr_no_ov = pr - known_context_risks
        tr_no_ov = tr - known_context_risks
        risk_tp_no_ov += len(pr_no_ov & tr_no_ov)
        risk_fp_no_ov += len(pr_no_ov - tr_no_ov)
        risk_fn_no_ov += len(tr_no_ov - pr_no_ov)

        pr_purchase = pr & purchase_risks
        tr_purchase = tr & purchase_risks
        purchase_tp += len(pr_purchase & tr_purchase)
        purchase_fp += len(pr_purchase - tr_purchase)
        purchase_fn += len(tr_purchase - pr_purchase)

        pr_holder = pr & holder_risks
        tr_holder = tr & holder_risks
        holder_tp += len(pr_holder & tr_holder)
        holder_fp += len(pr_holder - tr_holder)
        holder_fn += len(tr_holder - pr_holder)

    per_feature_mae = {k: abs_errors[k] / max(counts.get(k, 1), 1) for k in abs_errors}
    per_feature_rmse = {k: (sq_errors[k] / max(counts.get(k, 1), 1)) ** 0.5 for k in sq_errors}
    all_abs = []
    all_sq = []
    for k in abs_errors:
        c = max(counts.get(k, 1), 1)
        all_abs.append(abs_errors[k] / c)
        all_sq.append(sq_errors[k] / c)

    precision, recall, f1 = _eaa_risk_prf(risk_tp, risk_fp, risk_fn)
    precision_no_ov, recall_no_ov, f1_no_ov = _eaa_risk_prf(risk_tp_no_ov, risk_fp_no_ov, risk_fn_no_ov)
    purchase_precision, purchase_recall, purchase_f1 = _eaa_risk_prf(purchase_tp, purchase_fp, purchase_fn)
    holder_precision, holder_recall, holder_f1 = _eaa_risk_prf(holder_tp, holder_fp, holder_fn)

    holder_keys = [k for k in per_feature_mae if "holder" in k]
    purchase_keys = [k for k in per_feature_mae if "purchased" in k]
    demand_keys = [k for k in per_feature_mae if "load" in k]
    generation_keys = [k for k in per_feature_mae if "gen" in k]

    def mean_keys(keys):
        if not keys:
            return 0.0
        return sum(per_feature_mae[k] for k in keys) / len(keys)

    return {
        "num_targets": len(tgt_list),
        "num_predictions": len(pred_list),
        "num_matched": matched,
        "missing_predictions": missing_predictions,
        "coverage": matched / max(len(tgt_list), 1),
        "energy_mae": sum(all_abs) / max(len(all_abs), 1),
        "energy_rmse": _eaa_rmse([x ** 0.5 for x in all_sq]) if all_sq else 0.0,
        "generation_mae": mean_keys(generation_keys),
        "holder_mae": mean_keys(holder_keys),
        "demand_mae": mean_keys(demand_keys),
        "purchase_mae": mean_keys(purchase_keys),
        "purchased_power_mae": per_feature_mae.get("purchased_power_total", 0.0),
        "purchased_steam_mae": per_feature_mae.get("purchased_steam_total", 0.0),
        "risk_precision": precision,
        "risk_recall": recall,
        "risk_f1": f1,
        "risk_tp": risk_tp,
        "risk_fp": risk_fp,
        "risk_fn": risk_fn,
        "risk_precision_excluding_ov": precision_no_ov,
        "risk_recall_excluding_ov": recall_no_ov,
        "risk_f1_excluding_ov": f1_no_ov,
        "risk_tp_excluding_ov": risk_tp_no_ov,
        "risk_fp_excluding_ov": risk_fp_no_ov,
        "risk_fn_excluding_ov": risk_fn_no_ov,
        "purchase_risk_precision": purchase_precision,
        "purchase_risk_recall": purchase_recall,
        "purchase_risk_f1": purchase_f1,
        "holder_limit_risk_precision": holder_precision,
        "holder_limit_risk_recall": holder_recall,
        "holder_limit_risk_f1": holder_f1,
        "per_feature_mae": per_feature_mae,
        "per_feature_rmse": per_feature_rmse,
    }

def _eaa_predict_and_evaluate_horizon(self, sample, production_prediction=None):
    pred = self.predict_horizon(sample, production_prediction=production_prediction)
    target = sample.get("target_future", {})
    metrics = self.evaluate_horizon_prediction(pred, target)
    return {"prediction": pred, "metrics": metrics}

EnergyAgentAdapter.build_horizon_input = _eaa_build_horizon_input
EnergyAgentAdapter.build_horizon_messages = _eaa_build_horizon_messages
EnergyAgentAdapter.predict_horizon = _eaa_predict_horizon
EnergyAgentAdapter.evaluate_horizon_prediction = _eaa_evaluate_horizon_prediction
EnergyAgentAdapter.predict_and_evaluate_horizon = _eaa_predict_and_evaluate_horizon

def _eaa_is_summary_like(obj):
    return isinstance(obj, dict) and (
        isinstance(obj.get("production_operation_segment_summary"), dict)
        or obj.get("available") is not None
        or obj.get("num_segments") is not None
        or isinstance(obj.get("stage_counts"), dict)
    )

def _eaa_find_existing_operation_summary(obj, max_depth=4):
    if max_depth < 0:
        return None
    if not isinstance(obj, dict):
        return None
    val = obj.get("production_operation_segment_summary")
    if isinstance(val, dict):
        return val
    if _eaa_is_summary_like(obj) and not isinstance(obj.get("predicted_operation_segments"), list):
        return obj
    for key in ["prediction", "production_prediction", "plan_snapshot", "candidate_summary", "output", "result"]:
        nested = obj.get(key)
        found = _eaa_find_existing_operation_summary(nested, max_depth=max_depth - 1)
        if isinstance(found, dict):
            return found
    return None

def _eaa_find_operation_segments_recursive(obj, max_depth=4):
    if max_depth < 0:
        return []
    if not isinstance(obj, dict):
        return []
    for key in [
        "predicted_operation_segments",
        "model_predicted_operation_segments",
        "valid_predicted_operation_segments",
        "repaired_segments",
    ]:
        val = obj.get(key)
        if isinstance(val, list):
            return [x for x in val if isinstance(x, dict)]
    for key in ["prediction", "production_prediction", "plan_snapshot", "output", "result"]:
        val = _eaa_find_operation_segments_recursive(obj.get(key), max_depth=max_depth - 1)
        if val:
            return val
    return []

def _eaa_build_operation_segment_summary_pcvp(self, production_prediction, max_examples=16):

    if not isinstance(production_prediction, dict):
        return {
            "available": False,
            "source": "missing",
            "num_segments": 0,
            "heat_ids": [],
            "stage_counts": {},
            "unit_counts": {},
            "time_span": {},
            "examples": [],
            "note": "no production prediction was provided to energy agent",
        }

    existing = _eaa_find_existing_operation_summary(production_prediction)
    if isinstance(existing, dict) and (existing.get("available") or existing.get("num_segments") or existing.get("stage_counts")):
        out = dict(existing)
        out.setdefault("available", bool(out.get("num_segments", 0) or out.get("stage_counts")))
        out.setdefault("source", out.get("source", "provided_summary"))
        out["full_segments_removed_from_energy_prompt"] = True
        out["summary_builder"] = "energy_agent_pcvp_existing_summary"
        return out

    segs = _eaa_find_operation_segments_recursive(production_prediction)
    top_source = self._safe_str(
        production_prediction.get("operation_segment_source")
        or self._safe_dict(production_prediction.get("operation_segment_diagnostics")).get("operation_segment_source")
        or "model_output",
        "model_output",
    )
    segs = [seg for seg in segs if self._source_is_valid_model_operation_segment(seg, top_source=top_source)]
    if not segs:
        source = (
            production_prediction.get("operation_segment_source")
            or self._safe_dict(production_prediction.get("operation_segment_diagnostics")).get("operation_segment_source")
            or "missing_model_operation_segments"
        )
        return {
            "available": False,
            "source": source,
            "num_segments": 0,
            "heat_ids": [],
            "stage_counts": {},
            "unit_counts": {},
            "time_span": {},
            "examples": [],
            "note": "no valid model-output operation segments available for schedule-conditioned energy prediction",
        }

    heat_ids = []
    stage_counts = {}
    unit_counts = {}
    starts = []
    ends = []
    examples = []
    for seg in segs:
        hid = self._normalize_heat_id(seg.get("heat_id") or seg.get("id") or seg.get("hn"))
        if hid and hid not in heat_ids:
            heat_ids.append(hid)
        stage = self._canonical_stage_for_summary(seg.get("stage") or seg.get("process") or seg.get("operation"))
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
        unit = self._safe_str(seg.get("unit") or seg.get("resource") or seg.get("ru") or stage, stage)
        unit_counts[unit] = unit_counts.get(unit, 0) + 1
        s = self._time_to_float_for_summary(seg.get("start_min"))
        if s is None:
            s = self._time_to_float_for_summary(seg.get("start_time"))
        e = self._time_to_float_for_summary(seg.get("end_min"))
        if e is None:
            e = self._time_to_float_for_summary(seg.get("end_time"))
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
                "duration_min": seg.get("duration_min"),
                "source_steps": self._safe_list(seg.get("source_steps")),
            })

    return {
        "available": True,
        "source": top_source or "model_output",
        "num_segments": len(segs),
        "heat_ids": heat_ids[:max_examples],
        "stage_counts": stage_counts,
        "unit_counts": unit_counts,
        "time_span": {
            "start_min": min(starts) if starts else None,
            "end_min": max(ends) if ends else None,
        },
        "examples": examples,
        "full_segments_removed_from_energy_prompt": True,
        "summary_builder": "energy_agent_pcvp_from_model_segments",
    }

def _eaa_build_energy_repair_context_for_agent(self, input_state, production_summary):
    input_state = self._safe_dict(input_state)
    production_summary = self._safe_dict(production_summary)
    base = self._safe_dict(input_state.get("energy_repair_context"))
    base = dict(base)
    base["energy_history"] = input_state.get("energy_history", base.get("energy_history", []))
    base["future_patch_context"] = input_state.get("future_patch_context", base.get("future_patch_context", []))
    base["production_operation_segment_summary"] = production_summary
    base["constraints"] = input_state.get("constraints", base.get("constraints", {}))
    base.setdefault("holder_physical_bounds", {"b": [0.0, 400.0], "c": [0.0, 200.0], "l": [0.0, 450.0]})
    base.setdefault("stage_energy_coefficients", {
        "BOF": {"ldg_generation": 1.0, "steam": 0.5, "power": 0.4},
        "LF": {"power": 0.7, "steam": 0.2},
        "Casting": {"steam": 0.7, "power": 0.3},
        "Rolling": {"power": 1.0, "steam": 0.6},
    })
    return base

def _eaa_build_horizon_input_pcvp(self, sample, production_prediction=None):
    input_state_raw = self._safe_dict(sample.get("input_state"))
    input_state = _eaa_strip_for_energy_prompt(self, input_state_raw)

    production_summary = self._build_operation_segment_summary(production_prediction)
    if not production_summary.get("available") and isinstance(input_state_raw.get("production_operation_segment_summary"), dict):
        production_summary = self._build_operation_segment_summary(input_state_raw.get("production_operation_segment_summary"))
    if not production_summary.get("available") and isinstance(input_state_raw.get("production_prediction"), dict):
        production_summary = self._build_operation_segment_summary(input_state_raw.get("production_prediction"))

    input_state = dict(input_state)
    input_state.pop("production_prediction", None)
    input_state.pop("future_production_states", None)
    input_state.pop("production_patch_prediction", None)
    input_state["production_operation_segment_summary"] = production_summary
    input_state["energy_repair_context"] = _eaa_build_energy_repair_context_for_agent(self, input_state_raw, production_summary)
    input_state["task_role"] = "energy_horizon_prediction_conditioned_on_operation_schedule"
    input_state["production_conditioning_available"] = bool(production_summary.get("available"))

    horizon = input_state.get("horizon") or sample.get("horizon")
    basis = "schedule_conditioned" if production_summary.get("available") else "history_only"

    return {
        "task_instruction": (
            "You are an energy patch-level horizon prediction model for a steel plant production-energy world model. "
            "Given observed historical energy patches, known future patch heat/time context, and a compact summary of "
            "the predicted operation-level production Gantt schedule, predict the future energy states for the next K patches. "
            "This is prediction, not optimization. Do not output energy adjustment advice. "
            "Do not predict production active heats or heat-patch alignment; production schedule information is already summarized "
            "in production_operation_segment_summary. "
            "When production_operation_segment_summary.available is true, every future patch should set production_activity_basis="
            "schedule_conditioned or operation_segment_summary, not history_only. "
            "Future patch identifiers are known: keep step k aligned with future_patch_context[k]. "
            "Do not access target_future. Return only valid JSON."
        ),
        "input": input_state,
        "output_schema": {
            "task": "energy_horizon_prediction",
            "mode": "energy_patch_horizon_predictor_gantt_conditioned",
            "horizon": horizon,
            "future_patch_predictions": [
                {
                    "step": 1,
                    "patch_id": "must equal future_patch_context[step-1].id",
                    "heat_id": "must equal future_patch_context[step-1].hn",
                    "g": {"b": {"tot": 0.0}, "c": {"tot": 0.0}, "l": {"tot": 0.0}},
                    "st": {"b": {"e": 0.0, "m": 0.0}, "c": {"e": 0.0, "m": 0.0}, "l": {"e": 0.0, "m": 0.0}},
                    "dm": {"s": {"tot": 0.0, "pk": 0.0}, "e": {"tot": 0.0, "pk": 0.0}},
                    "ex": {"s": {"tot": 0.0}, "e": {"tot": 0.0}},
                    "rk": [],
                    "response_type": "direct|indirect|none|unknown",
                    "production_activity_basis": basis,
                    "operation_segment_summary_used": bool(production_summary.get("available")),
                    "confidence": 0.0,
                    "comment": "string"
                }
            ],
            "summary": {
                "holder_trend": {"BFG": "up|down|flat|unknown", "COG": "up|down|flat|unknown", "LDG": "up|down|flat|unknown"},
                "purchase_trend": {"power": "up|down|flat|unknown", "steam": "up|down|flat|unknown"},
                "overall_feasibility": "ok|warning|risky",
                "production_conditioning_available": bool(production_summary.get("available", False)),
                "production_conditioning_source": "operation_segment_summary" if production_summary.get("available") else "missing",
                "production_activity_basis": basis,
                "confidence": 0.0,
                "comment": "string"
            }
        }
    }

def _eaa_apply_energy_projection(self, normalized_prediction, sample, model_input):
    try:
        from mechanism_energy_projection import PhysicsConsistentEnergyProjector
    except Exception as e:
        normalized_prediction["mechanism_energy_projection_diagnostics"] = {
            "applied": False,
            "error": f"import_failed:{type(e).__name__}:{e}",
        }
        return normalized_prediction

    input_state = self._safe_dict(model_input.get("input"))
    repair_context = self._safe_dict(input_state.get("energy_repair_context"))
    if not repair_context:
        repair_context = _eaa_build_energy_repair_context_for_agent(
            self,
            self._safe_dict(sample.get("input_state")),
            self._safe_dict(input_state.get("production_operation_segment_summary")),
        )

    projector = PhysicsConsistentEnergyProjector()
    result = projector.project(normalized_prediction, repair_context=repair_context)
    projected = result.get("prediction") if isinstance(result, dict) else None
    if not isinstance(projected, dict):
        normalized_prediction["mechanism_energy_projection_diagnostics"] = {
            "applied": False,
            "error": "projection_returned_non_dict_prediction",
        }
        return normalized_prediction

    diag = self._safe_dict(projected.get("mechanism_energy_projection_diagnostics"))
    diag["applied"] = True
    projected["mechanism_energy_projection_diagnostics"] = diag
    return projected

def _eaa_predict_horizon_pcvp(self, sample, production_prediction=None):
    model_input = self.build_horizon_input(sample, production_prediction=production_prediction)
    messages = self.build_horizon_messages(model_input)
    raw_text = ""
    parse_error = None

    try:
        raw_text = self.chat_model.generate(messages)
    except Exception as e:
        parse_error = f"energy_model_generate_failed:{type(e).__name__}:{e}"
        raw_text = ""

    print("\n=== RAW ENERGY HORIZON MODEL OUTPUT START ===")
    print(raw_text)
    print("=== RAW ENERGY HORIZON MODEL OUTPUT END ===\n")

    parsed = {}
    if raw_text and str(raw_text).strip():
        try:
            parsed = _safe_json_loads(raw_text)
        except Exception as e:
            parse_error = f"energy_model_parse_failed:{type(e).__name__}:{e}"
            parsed = {}
    else:
        parse_error = parse_error or "empty_energy_model_output"

    if not isinstance(parsed, dict):
        parsed = {}

    normalized = _eaa_normalize_horizon_prediction_with_context(
        self,
        parsed,
        sample,
        raw_text=raw_text or "",
        parse_error=parse_error,
    )
    normalized["production_operation_segment_summary"] = self._safe_dict(
        self._safe_dict(model_input.get("input")).get("production_operation_segment_summary")
    )
    projected = _eaa_apply_energy_projection(self, normalized, sample, model_input)
    return projected

def _eaa_evaluate_horizon_prediction_pcvp(self, prediction, target_future):
    metrics = _eaa_evaluate_horizon_prediction(self, prediction, target_future)
    diag = self._safe_dict(prediction.get("mechanism_energy_projection_diagnostics"))
    if diag:
        metrics.update({
            "energy_pcvp_applied": bool(diag.get("applied", True)),
            "energy_pcvp_uses_future_target_labels": bool(diag.get("uses_future_target_labels", False)),
            "energy_pcvp_num_uncertain_predictions": diag.get("num_uncertain_predictions", 0),
            "energy_pcvp_num_rejected_predictions": diag.get("num_rejected_predictions", 0),
            "energy_pcvp_objective_total_before": diag.get("objective_total_before"),
            "energy_pcvp_objective_total_after": diag.get("objective_total_after"),
            "energy_pcvp_production_conditioning_available": bool(diag.get("production_conditioning_available", False)),
            "energy_pcvp_violations_before": diag.get("violations_before", {}),
            "energy_pcvp_violations_after": diag.get("violations_after", {}),
        })
    summary = self._safe_dict(prediction.get("summary"))
    metrics["production_conditioning_available"] = bool(summary.get("production_conditioning_available", False))
    metrics["production_activity_basis"] = summary.get("production_activity_basis")
    return metrics

def _eaa_predict_and_evaluate_horizon_pcvp(self, sample, production_prediction=None):
    pred = self.predict_horizon(sample, production_prediction=production_prediction)
    target = sample.get("target_future", {})
    metrics = self.evaluate_horizon_prediction(pred, target)
    return {"prediction": pred, "metrics": metrics}

EnergyAgentAdapter._build_operation_segment_summary = _eaa_build_operation_segment_summary_pcvp
EnergyAgentAdapter.build_horizon_input = _eaa_build_horizon_input_pcvp
EnergyAgentAdapter.predict_horizon = _eaa_predict_horizon_pcvp
EnergyAgentAdapter.evaluate_horizon_prediction = _eaa_evaluate_horizon_prediction_pcvp
EnergyAgentAdapter.predict_and_evaluate_horizon = _eaa_predict_and_evaluate_horizon_pcvp

import os as _energy_os
import copy as _energy_copy

def _eaa_energy_bool_env(name, default=False):
    val = _energy_os.environ.get(name)
    if val is None:
        return bool(default)
    return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}

def _eaa_energy_int_env(name, default):
    val = _energy_os.environ.get(name)
    if val is None or str(val).strip() == "":
        return int(default)
    try:
        return int(float(val))
    except Exception:
        return int(default)

def _eaa_energy_make_empty_prediction(self, horizon, warning="empty_energy_prediction"):
    return {
        "task": "energy_horizon_prediction",
        "mode": "energy_patch_horizon_predictor_empty",
        "horizon": int(horizon or 0),
        "future_patch_predictions": [],
        "summary": {
            "production_conditioning_available": False,
            "production_activity_basis": "history_only",
            "overall_feasibility": "warning",
            "confidence": 0.0,
            "comment": warning,
        },
        "warning": warning,
    }

def _eaa_energy_call_once(self, model_input, attempt_index=0, retry_feedback=None):
    call_input = _energy_copy.deepcopy(model_input if isinstance(model_input, dict) else {})
    if retry_feedback:
        call_input["retry_feedback"] = retry_feedback
        call_input["task_instruction"] = (
            str(call_input.get("task_instruction", "")) +
            " This is a self-refinement retry. Use only the retry_feedback diagnostics, input history, future_patch_context, "
            "production_operation_segment_summary and mechanism constraints. Do not use target_future labels."
        )

    messages = self.build_horizon_messages(call_input)
    raw_text = ""
    parse_error = None
    try:
        raw_text = self.chat_model.generate(messages)
    except Exception as e:
        parse_error = f"energy_model_generate_failed:{type(e).__name__}:{e}"
        raw_text = ""

    print(f"\n=== RAW ENERGY HORIZON MODEL OUTPUT START attempt={attempt_index} ===")
    print(raw_text)
    print(f"=== RAW ENERGY HORIZON MODEL OUTPUT END attempt={attempt_index} ===\n")

    parsed = {}
    if raw_text and str(raw_text).strip():
        try:
            parsed = _safe_json_loads(raw_text)
        except Exception as e:
            parse_error = f"energy_model_parse_failed:{type(e).__name__}:{e}"
            parsed = {}
    else:
        parse_error = parse_error or "empty_energy_model_output"

    if not isinstance(parsed, dict):
        parsed = {}
        parse_error = parse_error or "parsed_energy_output_is_not_dict"

    return {
        "attempt_index": attempt_index,
        "model_input": call_input,
        "raw_text": raw_text or "",
        "parsed": parsed,
        "parse_error": parse_error,
    }

def _eaa_energy_raw_schema_check(self, parsed, model_input, parse_error=None):
    input_state = self._safe_dict(model_input.get("input"))
    horizon = int(input_state.get("horizon") or model_input.get("horizon") or 0)
    future_ctx = self._safe_list(input_state.get("future_patch_context"))
    if horizon <= 0:
        horizon = len(future_ctx)

    errors = []
    if parse_error:
        errors.append(parse_error)
    if not isinstance(parsed, dict) or not parsed:
        errors.append("parsed_output_missing_or_empty")
        return {
            "valid": False,
            "errors": errors,
            "num_raw_steps": 0,
            "horizon": horizon,
            "missing_steps": list(range(1, horizon + 1)),
        }

    steps = self._safe_list(parsed.get("future_patch_predictions"))
    if not isinstance(parsed.get("future_patch_predictions"), list):
        errors.append("missing_future_patch_predictions")
    if len(steps) < horizon:
        errors.append(f"future_patch_predictions_length_less_than_horizon:{len(steps)}<{horizon}")

    missing_steps = []
    misaligned = []
    by_step = {}
    for i, item in enumerate(steps, start=1):
        if not isinstance(item, dict):
            continue
        try:
            st = int(item.get("step") or i)
        except Exception:
            st = i
        by_step[st] = item

    for k in range(1, horizon + 1):
        item = by_step.get(k)
        if not isinstance(item, dict):
            missing_steps.append(k)
            continue
        if k <= len(future_ctx):
            fctx = self._safe_dict(future_ctx[k - 1])
            expected_pid = self._safe_str(fctx.get("id") or fctx.get("patch_id") or fctx.get("p"))
            got_pid = self._safe_str(item.get("patch_id") or item.get("id") or item.get("p"))
            if expected_pid and got_pid and expected_pid != got_pid:
                misaligned.append({"step": k, "expected_patch_id": expected_pid, "got_patch_id": got_pid})

    if missing_steps:
        errors.append(f"missing_prediction_steps:{missing_steps[:10]}")
    if misaligned:
        errors.append(f"future_patch_context_misaligned:{misaligned[:5]}")

    prod_summary = self._safe_dict(input_state.get("production_operation_segment_summary"))
    prod_available = bool(prod_summary.get("available") or prod_summary.get("num_segments"))
    if prod_available:
        basis_bad = 0
        for item in steps:
            if not isinstance(item, dict):
                continue
            basis = self._safe_str(item.get("production_activity_basis"), "")
            used = item.get("operation_segment_summary_used")
            if basis in {"", "history_only", "unknown"} or used is False:
                basis_bad += 1
        if steps and basis_bad >= max(1, len(steps) // 2):
            errors.append("production_conditioning_missing_in_raw_prediction")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "num_raw_steps": len(steps),
        "horizon": horizon,
        "missing_steps": missing_steps,
        "misaligned_steps": misaligned,
        "production_conditioning_available": prod_available,
    }

def _eaa_energy_project_attempt(self, attempt, sample):
    parsed = attempt.get("parsed") if isinstance(attempt, dict) else {}
    model_input = attempt.get("model_input") if isinstance(attempt, dict) else {}
    schema = _eaa_energy_raw_schema_check(self, parsed, model_input, attempt.get("parse_error"))
    input_state = self._safe_dict(model_input.get("input"))
    horizon = schema.get("horizon", 0)
    if not isinstance(parsed, dict) or not parsed:
        parsed = _eaa_energy_make_empty_prediction(self, horizon, warning="empty_or_invalid_energy_llm_output")

    normalized = _eaa_normalize_horizon_prediction_with_context(
        self,
        parsed,
        sample,
        raw_text=attempt.get("raw_text", ""),
        parse_error=attempt.get("parse_error"),
    )
    normalized["production_operation_segment_summary"] = self._safe_dict(input_state.get("production_operation_segment_summary"))
    normalized["_energy_attempt_schema_check"] = schema
    normalized["_raw_energy_attempt_index"] = attempt.get("attempt_index")

    projected = _eaa_apply_energy_projection(self, normalized, sample, model_input)
    diag = self._safe_dict(projected.get("mechanism_energy_projection_diagnostics"))
    diag["applied"] = bool(diag.get("applied", True))
    diag["attempt_index"] = attempt.get("attempt_index")
    diag["raw_schema_check"] = schema
    projected["mechanism_energy_projection_diagnostics"] = diag
    return projected, schema

def _eaa_energy_prediction_is_valid_for_stop(self, prediction, schema, model_input):
    if not isinstance(prediction, dict):
        return False, ["projection_returned_non_dict"]
    input_state = self._safe_dict(model_input.get("input"))
    horizon = int(input_state.get("horizon") or prediction.get("horizon") or 0)
    if horizon <= 0:
        horizon = len(self._safe_list(input_state.get("future_patch_context")))
    steps = self._safe_list(prediction.get("future_patch_predictions"))
    reasons = []
    if not schema.get("valid"):
        reasons.extend(schema.get("errors", []))
    if len(steps) < horizon:
        reasons.append(f"projected_future_patch_predictions_length_less_than_horizon:{len(steps)}<{horizon}")

    diag = self._safe_dict(prediction.get("mechanism_energy_projection_diagnostics"))
    violations_after = self._safe_dict(diag.get("violations_after"))
    hard_count = int(violations_after.get("missing_context_binding", 0) or 0)

    prod_summary = self._safe_dict(input_state.get("production_operation_segment_summary"))
    if bool(prod_summary.get("available") or prod_summary.get("num_segments")):
        hard_count += int(violations_after.get("missing_production_conditioning_when_available", 0) or 0)
    if hard_count > 0:
        reasons.append(f"energy_pcvp_hard_violations_after:{hard_count}")

    return len(reasons) == 0, reasons

def _eaa_energy_build_retry_feedback(self, attempt_idx, schema, projected, invalid_reasons, model_input):
    input_state = self._safe_dict(model_input.get("input"))
    diag = self._safe_dict(projected.get("mechanism_energy_projection_diagnostics")) if isinstance(projected, dict) else {}
    return {
        "retry_attempt_index": int(attempt_idx),
        "reason_for_retry": invalid_reasons,
        "schema_check": schema,
        "energy_pcvp_violations_before": diag.get("violations_before", {}),
        "energy_pcvp_violations_after": diag.get("violations_after", {}),
        "allowed_information_only": True,
        "uses_future_target_labels": False,
        "do_not_use": [
            "target_future.energy_future",
            "true future energy values",
            "MAE/RMSE/F1 or any offline metric",
        ],
        "must_fix": [
            "Return one JSON object with key future_patch_predictions.",
            "Provide exactly one prediction per future_patch_context step.",
            "Keep step and patch_id aligned with future_patch_context.",
            "If production_operation_segment_summary.available is true, set production_activity_basis to schedule_conditioned and operation_segment_summary_used to true.",
            "Keep generation, demand, purchase and holder values nonnegative and physically plausible.",
            "Make risk labels consistent with holder bounds and purchase values.",
        ],
        "future_patch_context": input_state.get("future_patch_context", []),
        "production_operation_segment_summary": input_state.get("production_operation_segment_summary", {}),
        "local_energy_blocks": input_state.get("local_energy_blocks", []),
    }

def _eaa_energy_extract_last_history_patch(self, model_input):
    hist = self._safe_list(self._safe_dict(model_input.get("input")).get("energy_history"))
    if hist:
        return self._safe_dict(hist[-1])
    return {}

def _eaa_energy_nested_copy(self, src, top, key, subkey=None, default=0.0):
    src = self._safe_dict(src)
    top_obj = self._safe_dict(src.get(top))
    if subkey is None:
        obj = self._safe_dict(top_obj.get(key))
    else:
        obj = self._safe_dict(self._safe_dict(top_obj.get(key)).get(subkey))
    return _energy_copy.deepcopy(obj) if obj else {}

def _eaa_energy_build_rule_fallback_prediction(self, sample, model_input, production_prediction=None):
    input_state = self._safe_dict(model_input.get("input"))
    future_ctx = self._safe_list(input_state.get("future_patch_context"))
    horizon = int(input_state.get("horizon") or sample.get("horizon") or len(future_ctx))
    prod_summary = self._safe_dict(input_state.get("production_operation_segment_summary"))
    last = _eaa_energy_extract_last_history_patch(self, model_input)

    last_g = self._safe_dict(last.get("g"))
    last_st = self._safe_dict(last.get("st"))
    last_dm = self._safe_dict(last.get("dm"))
    last_ex = self._safe_dict(last.get("ex"))

    stage_counts = self._safe_dict(prod_summary.get("stage_counts"))
    prod_available = bool(prod_summary.get("available") or prod_summary.get("num_segments"))
    activity_score = sum(self._safe_float(v, 0.0) for v in stage_counts.values()) if stage_counts else 0.0
    ldg_factor = 1.0 + min(0.20, 0.02 * activity_score)
    power_factor = 1.0 + min(0.15, 0.015 * activity_score)
    steam_factor = 1.0 + min(0.12, 0.012 * activity_score)

    predictions = []
    for idx in range(horizon):
        fctx = self._safe_dict(future_ctx[idx]) if idx < len(future_ctx) else {}
        t = self._safe_dict(fctx.get("t"))
        pid = self._safe_str(fctx.get("id") or fctx.get("patch_id") or fctx.get("p") or f"fallback_step_{idx + 1}")
        heat_id = self._normalize_heat_id(fctx.get("hn") or self._safe_dict(fctx.get("pc")).get("phn") or fctx.get("heat_id") or pid)
        g = _energy_copy.deepcopy(last_g)
        st = _energy_copy.deepcopy(last_st)
        dm = _energy_copy.deepcopy(last_dm)
        ex = _energy_copy.deepcopy(last_ex)

        if prod_available:
            if isinstance(g.get("l"), dict):
                for k in ["tot", "a", "avg"]:
                    if k in g["l"]:
                        g["l"][k] = max(0.0, self._safe_float(g["l"].get(k), 0.0) * ldg_factor)
            if isinstance(dm.get("e"), dict):
                for k in ["tot", "pk", "a", "avg"]:
                    if k in dm["e"]:
                        dm["e"][k] = max(0.0, self._safe_float(dm["e"].get(k), 0.0) * power_factor)
            if isinstance(dm.get("s"), dict):
                for k in ["tot", "pk", "a", "avg"]:
                    if k in dm["s"]:
                        dm["s"][k] = max(0.0, self._safe_float(dm["s"].get(k), 0.0) * steam_factor)

        predictions.append({
            "step": idx + 1,
            "patch_id": pid,
            "id": pid,
            "p": pid,
            "heat_id": heat_id,
            "hn": heat_id,
            "t": {
                "ts": t.get("ts") or fctx.get("start_time"),
                "te": t.get("te") or fctx.get("end_time"),
                "t0": t.get("t0") or fctx.get("start_min"),
                "t1": t.get("t1") or fctx.get("end_min"),
            },
            "g": g,
            "st": st,
            "dm": dm,
            "ex": ex,
            "rk": [],
            "response_type": "direct" if prod_available and activity_score > 0 else "none",
            "production_activity_basis": "schedule_conditioned" if prod_available else "history_only",
            "operation_segment_summary_used": bool(prod_available),
            "source": "rule_assisted_fallback_after_retry_exhausted",
            "confidence": 0.35 if prod_available else 0.25,
            "comment": "retry-exhausted rule-assisted fallback; no target_future labels used",
        })

    return {
        "task": "energy_horizon_prediction",
        "mode": "energy_patch_horizon_predictor_retry_exhausted_rule_fallback",
        "horizon": horizon,
        "future_patch_predictions": predictions,
        "summary": {
            "production_conditioning_available": bool(prod_available),
            "production_conditioning_source": "operation_segment_summary" if prod_available else "missing",
            "production_activity_basis": "schedule_conditioned" if prod_available else "history_only",
            "overall_feasibility": "warning",
            "confidence": 0.35 if prod_available else 0.25,
            "comment": "rule-assisted fallback used after all LLM retry attempts failed; target_future not used",
        },
        "rule_fallback_meta": {
            "rule_fallback_used": True,
            "rule_fallback_uses_future_target_labels": False,
            "rule_fallback_num_raw_steps": len(predictions),
            "rule_fallback_source": "history_tail_plus_future_context_plus_production_operation_segment_summary",
        },
    }

def _eaa_predict_horizon_retry_pcvp_fallback(self, sample, production_prediction=None):
    model_input = self.build_horizon_input(sample, production_prediction=production_prediction)
    input_state = self._safe_dict(model_input.get("input"))
    max_retries = _eaa_energy_int_env(
        "ENERGY_MAX_HORIZON_RETRIES",
        self._safe_dict(input_state.get("energy_retry_guidance_context")).get("max_retry_attempts", 5),
    )
    max_retries = max(0, int(max_retries))
    enable_fallback = _eaa_energy_bool_env(
        "ENERGY_ENABLE_RULE_FALLBACK_AFTER_RETRY",
        self._safe_dict(input_state.get("energy_rule_fallback_context")).get("enabled", True),
    )
    evaluate_fallback = _eaa_energy_bool_env(
        "ENERGY_EVALUATE_RULE_FALLBACK_AS_PREDICTION",
        self._safe_dict(input_state.get("energy_rule_fallback_context")).get("evaluate_as_prediction", True),
    )

    attempts = []
    retry_feedback = None
    selected = None
    selected_schema = None
    invalid_reasons = []
    first_pass_missing = False

    for attempt_idx in range(max_retries + 1):
        attempt = _eaa_energy_call_once(self, model_input, attempt_index=attempt_idx, retry_feedback=retry_feedback)
        projected, schema = _eaa_energy_project_attempt(self, attempt, sample)
        valid, reasons = _eaa_energy_prediction_is_valid_for_stop(self, projected, schema, attempt.get("model_input"))
        if attempt_idx == 0:
            first_pass_missing = (not schema.get("valid")) and any("missing_future_patch_predictions" in str(x) or "empty" in str(x) for x in schema.get("errors", []))
        attempts.append({
            "attempt_index": attempt_idx,
            "parse_error": attempt.get("parse_error"),
            "schema_check": schema,
            "valid_for_stop": valid,
            "invalid_reasons": reasons,
            "num_projected_steps": len(self._safe_list(projected.get("future_patch_predictions"))),
            "pcvp_violations_after": self._safe_dict(self._safe_dict(projected.get("mechanism_energy_projection_diagnostics")).get("violations_after")),
        })
        selected = projected
        selected_schema = schema
        invalid_reasons = reasons
        if valid:
            break
        retry_feedback = _eaa_energy_build_retry_feedback(self, attempt_idx + 1, schema, projected, reasons, attempt.get("model_input"))

    retry_success = bool(selected is not None and _eaa_energy_prediction_is_valid_for_stop(self, selected, selected_schema or {}, model_input)[0])

    fallback_used = False
    if not retry_success and enable_fallback:
        raw_fallback = _eaa_energy_build_rule_fallback_prediction(self, sample, model_input, production_prediction=production_prediction)
        normalized = _eaa_normalize_horizon_prediction_with_context(
            self,
            raw_fallback,
            sample,
            raw_text="",
            parse_error=None,
        )
        normalized["production_operation_segment_summary"] = self._safe_dict(input_state.get("production_operation_segment_summary"))
        selected = _eaa_apply_energy_projection(self, normalized, sample, model_input)
        fallback_used = True

    if selected is None:
        selected = _eaa_energy_make_empty_prediction(self, input_state.get("horizon") or sample.get("horizon"), "all_energy_attempts_failed")
        selected = _eaa_apply_energy_projection(self, selected, sample, model_input)

    retry_meta = {
        "retry_enabled": True,
        "max_horizon_retries": max_retries,
        "num_horizon_attempts": len(attempts),
        "num_horizon_retries_used": max(0, len(attempts) - 1),
        "first_pass_missing": bool(first_pass_missing),
        "retry_success": bool(retry_success),
        "retry_uses_future_target_labels": False,
        "retry_attempts": attempts,
        "final_invalid_reasons_before_fallback": invalid_reasons,
        "rule_fallback_enabled": bool(enable_fallback),
        "rule_fallback_used": bool(fallback_used),
        "rule_fallback_success": bool(fallback_used and len(self._safe_list(selected.get("future_patch_predictions"))) > 0),
        "rule_fallback_evaluate_as_prediction": bool(evaluate_fallback),
        "rule_fallback_uses_future_target_labels": False,
    }

    selected["energy_retry_meta"] = retry_meta
    selected["production_operation_segment_summary"] = self._safe_dict(input_state.get("production_operation_segment_summary"))
    summary = self._safe_dict(selected.get("summary"))
    prod_summary = self._safe_dict(input_state.get("production_operation_segment_summary"))
    prod_available = bool(prod_summary.get("available") or prod_summary.get("num_segments"))
    summary["production_conditioning_available"] = prod_available
    summary["production_activity_basis"] = "schedule_conditioned" if prod_available else "history_only"
    summary["retry_enabled"] = True
    summary["num_horizon_retries_used"] = retry_meta["num_horizon_retries_used"]
    summary["rule_fallback_used"] = bool(fallback_used)
    selected["summary"] = summary

    diag = self._safe_dict(selected.get("mechanism_energy_projection_diagnostics"))
    diag["applied"] = bool(diag.get("applied", True))
    diag["energy_retry_enabled"] = True
    diag["num_horizon_attempts"] = retry_meta["num_horizon_attempts"]
    diag["num_horizon_retries_used"] = retry_meta["num_horizon_retries_used"]
    diag["retry_success"] = retry_meta["retry_success"]
    diag["rule_fallback_used"] = retry_meta["rule_fallback_used"]
    diag["rule_fallback_uses_future_target_labels"] = False
    selected["mechanism_energy_projection_diagnostics"] = diag
    return selected

def _eaa_evaluate_horizon_prediction_retry_pcvp_fallback(self, prediction, target_future):
    metrics = _eaa_evaluate_horizon_prediction_pcvp(self, prediction, target_future)
    retry_meta = self._safe_dict(prediction.get("energy_retry_meta"))
    diag = self._safe_dict(prediction.get("mechanism_energy_projection_diagnostics"))
    metrics.update({
        "energy_retry_enabled": bool(retry_meta.get("retry_enabled", False)),
        "energy_num_horizon_attempts": retry_meta.get("num_horizon_attempts", 1),
        "energy_num_horizon_retries_used": retry_meta.get("num_horizon_retries_used", 0),
        "energy_first_pass_missing": bool(retry_meta.get("first_pass_missing", False)),
        "energy_retry_success": bool(retry_meta.get("retry_success", False)),
        "energy_retry_uses_future_target_labels": bool(retry_meta.get("retry_uses_future_target_labels", False)),
        "energy_rule_fallback_enabled": bool(retry_meta.get("rule_fallback_enabled", False)),
        "energy_rule_fallback_used": bool(retry_meta.get("rule_fallback_used", False)),
        "energy_rule_fallback_success": bool(retry_meta.get("rule_fallback_success", False)),
        "energy_rule_fallback_evaluate_as_prediction": bool(retry_meta.get("rule_fallback_evaluate_as_prediction", True)),
        "energy_rule_fallback_uses_future_target_labels": bool(retry_meta.get("rule_fallback_uses_future_target_labels", False)),
        "energy_pcvp_applied": bool(diag.get("applied", False)),
    })
    return metrics

def _eaa_predict_and_evaluate_horizon_retry_pcvp_fallback(self, sample, production_prediction=None):
    pred = self.predict_horizon(sample, production_prediction=production_prediction)
    target = sample.get("target_future", {})
    metrics = self.evaluate_horizon_prediction(pred, target)
    return {"prediction": pred, "metrics": metrics}

EnergyAgentAdapter.predict_horizon = _eaa_predict_horizon_retry_pcvp_fallback
EnergyAgentAdapter.evaluate_horizon_prediction = _eaa_evaluate_horizon_prediction_retry_pcvp_fallback
EnergyAgentAdapter.predict_and_evaluate_horizon = _eaa_predict_and_evaluate_horizon_retry_pcvp_fallback

def _eaa_final_sd(x):
    return x if isinstance(x, dict) else {}

def _eaa_final_sl(x):
    return x if isinstance(x, list) else ([] if x is None else [x])

def _eaa_final_float(x, default=0.0):
    try:
        if x is None:
            return default
        if isinstance(x, str) and not x.strip():
            return default
        return float(x)
    except Exception:
        return default

def _eaa_final_get(obj, path, default=0.0):
    cur = obj
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return _eaa_final_float(cur, default)

def _eaa_final_set(obj, path, value):
    cur = obj
    for key in path[:-1]:
        if not isinstance(cur.get(key), dict):
            cur[key] = {}
        cur = cur[key]
    cur[path[-1]] = round(max(0.0, _eaa_final_float(value, 0.0)), 4)

def _eaa_final_patch_time(p):
    p = _eaa_final_sd(p)
    t = _eaa_final_sd(p.get("t"))
    return (
        _eaa_final_float(p.get("start_min") or p.get("t0") or t.get("t0") or t.get("s0"), None),
        _eaa_final_float(p.get("end_min") or p.get("t1") or t.get("t1") or t.get("s1"), None),
    )

def _eaa_final_canonical_stage(x):
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

def _eaa_final_build_future_activity(self, future_context, production_summary):
    future_context = _eaa_final_sl(future_context)
    production_summary = _eaa_final_sd(production_summary)
    examples = [x for x in _eaa_final_sl(production_summary.get("examples")) if isinstance(x, dict)]
    prod_available = bool(production_summary.get("available") or production_summary.get("num_segments") or examples or production_summary.get("stage_counts"))
    out = []
    any_overlap = False
    for idx, fctx in enumerate(future_context, start=1):
        p0, p1 = _eaa_final_patch_time(fctx)
        stage_overlap = {"BOF": 0.0, "LF": 0.0, "Casting": 0.0, "Rolling": 0.0, "Other": 0.0}
        units = {}
        heats = []
        if p0 is not None and p1 is not None and p1 > p0:
            for seg in examples:
                s = _eaa_final_float(seg.get("start_min"), None)
                e = _eaa_final_float(seg.get("end_min"), None)
                if s is None or e is None:
                    continue
                ov = max(0.0, min(p1, e) - max(p0, s))
                if ov <= 0:
                    continue
                any_overlap = True
                stage = _eaa_final_canonical_stage(seg.get("stage"))
                stage_overlap[stage] = stage_overlap.get(stage, 0.0) + ov
                unit = str(seg.get("unit") or stage)
                units[unit] = units.get(unit, 0.0) + ov
                hid = self._normalize_heat_id(seg.get("heat_id") or seg.get("hn") or seg.get("id"))
                if hid and hid not in heats:
                    heats.append(hid)
        score = 1.2 * stage_overlap["BOF"] + 0.9 * stage_overlap["LF"] + 0.85 * stage_overlap["Casting"] + 1.3 * stage_overlap["Rolling"]
        row = {
            "step": idx,
            "patch_id": _eaa_final_sd(fctx).get("id") or _eaa_final_sd(fctx).get("patch_id") or _eaa_final_sd(fctx).get("p"),
            "heat_id": self._normalize_heat_id(_eaa_final_sd(fctx).get("hn") or _eaa_final_sd(_eaa_final_sd(fctx).get("pc")).get("phn") or _eaa_final_sd(fctx).get("heat_id")),
            "available": prod_available,
            "basis": "time_overlap" if prod_available and score > 0 else ("operation_segment_summary_no_step_overlap" if prod_available else "history_only"),
            "stage_overlap_min": {k: round(v, 4) for k, v in stage_overlap.items() if abs(v) > 1e-9},
            "unit_overlap_min": {k: round(v, 4) for k, v in units.items()},
            "active_heat_ids": heats[:12],
            "activity_score": round(score, 4),
            "activity_level": "high" if score >= 80 else ("medium" if score >= 30 else ("low" if score > 0 else "none")),
        }
        out.append(row)
    if prod_available and not any_overlap:
        stage_counts = _eaa_final_sd(production_summary.get("stage_counts"))
        total_count = sum(_eaa_final_float(v, 0.0) for v in stage_counts.values())
        for row in out:
            row["basis"] = "window_summary_only_no_time_overlap"
            row["window_stage_counts"] = stage_counts
            row["activity_score"] = round(total_count, 4)
            row["activity_level"] = "medium" if total_count > 0 else "none"
            row["time_alignment_warning"] = "no segment examples overlap future patch time; verify common minute basis"
    return out

def _eaa_final_numeric_history_rows(self, energy_history):
    rows = []
    for p in _eaa_final_sl(energy_history):
        rows.append({
            "g_b_tot": _eaa_final_get(p, ["g", "b", "tot"]),
            "g_c_tot": _eaa_final_get(p, ["g", "c", "tot"]),
            "g_l_tot": _eaa_final_get(p, ["g", "l", "tot"]),
            "st_b_e": _eaa_final_get(p, ["st", "b", "e"]),
            "st_c_e": _eaa_final_get(p, ["st", "c", "e"]),
            "st_l_e": _eaa_final_get(p, ["st", "l", "e"]),
            "dm_s_tot": _eaa_final_get(p, ["dm", "s", "tot"]),
            "dm_s_pk": _eaa_final_get(p, ["dm", "s", "pk"]),
            "dm_e_tot": _eaa_final_get(p, ["dm", "e", "tot"]),
            "dm_e_pk": _eaa_final_get(p, ["dm", "e", "pk"]),
            "ex_s_tot": _eaa_final_get(p, ["ex", "s", "tot"]),
            "ex_e_tot": _eaa_final_get(p, ["ex", "e", "tot"]),
        })
    return rows

def _eaa_final_mean(vals):
    vals = [_eaa_final_float(v, None) for v in vals]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else 0.0

def _eaa_final_trend(vals):
    vals = [_eaa_final_float(v, None) for v in vals]
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return 0.0
    return (vals[-1] - vals[0]) / max(len(vals) - 1, 1)

def _eaa_final_reference_from_history(self, input_state):
    hist = _eaa_final_sl(_eaa_final_sd(input_state).get("energy_history"))
    rows = _eaa_final_numeric_history_rows(self, hist)
    if not rows:
        return {"available": False}
    last = rows[-1]
    ref = {"available": True, "last": {}, "trend_per_step": {}, "mean": {}}
    for key in last.keys():
        seq = [r[key] for r in rows]
        ref["last"][key] = round(last[key], 4)
        ref["trend_per_step"][key] = round(_eaa_final_trend(seq), 4)
        ref["mean"][key] = round(_eaa_final_mean(seq), 4)
    return ref

def _eaa_final_build_repair_context_for_agent(self, input_state, production_summary):
    input_state = self._safe_dict(input_state)
    production_summary = self._safe_dict(production_summary)
    future_context = self._safe_list(input_state.get("future_patch_context"))
    activity = input_state.get("future_production_activity_by_step")
    if not isinstance(activity, list):
        activity = _eaa_final_build_future_activity(self, future_context, production_summary)
    base = self._safe_dict(input_state.get("energy_repair_context"))
    base = dict(base)
    base["energy_history"] = input_state.get("energy_history", base.get("energy_history", []))
    base["future_patch_context"] = future_context
    base["production_operation_segment_summary"] = production_summary
    base["future_production_activity_by_step"] = activity
    base["semantic_energy_history"] = input_state.get("semantic_energy_history", base.get("semantic_energy_history", []))
    base["history_transition_summary"] = input_state.get("history_transition_summary", base.get("history_transition_summary", {}))
    base["energy_numeric_history_reference"] = input_state.get("energy_numeric_history_reference", _eaa_final_reference_from_history(self, input_state))
    base["constraints"] = input_state.get("constraints", base.get("constraints", {}))
    base.setdefault("holder_physical_bounds", {"b": [0.0, 400.0], "c": [0.0, 200.0], "l": [0.0, 450.0]})
    base.setdefault("stage_energy_coefficients", {
        "BOF": {"ldg_generation": 1.0, "steam": 0.5, "power": 0.4},
        "LF": {"power": 0.7, "steam": 0.2},
        "Casting": {"steam": 0.7, "power": 0.3},
        "Rolling": {"power": 1.0, "steam": 0.6},
    })
    base["uses_future_target_labels"] = False
    return base

def _eaa_final_build_horizon_input(self, sample, production_prediction=None):
    input_state_raw = self._safe_dict(sample.get("input_state"))
    input_state = _eaa_strip_for_energy_prompt(self, input_state_raw)

    production_summary = self._build_operation_segment_summary(production_prediction)
    if not production_summary.get("available") and isinstance(input_state_raw.get("production_operation_segment_summary"), dict):
        production_summary = self._build_operation_segment_summary(input_state_raw.get("production_operation_segment_summary"))
    if not production_summary.get("available") and isinstance(input_state_raw.get("production_prediction"), dict):
        production_summary = self._build_operation_segment_summary(input_state_raw.get("production_prediction"))
    if not production_summary.get("available") and isinstance(input_state_raw.get("production_operation_segment_context"), dict):
        production_summary = self._build_operation_segment_summary(input_state_raw.get("production_operation_segment_context"))

    input_state = dict(input_state)
    input_state.pop("production_prediction", None)
    input_state.pop("future_production_states", None)
    input_state.pop("production_patch_prediction", None)
    input_state["production_operation_segment_summary"] = production_summary

    future_context = self._safe_list(input_state.get("future_patch_context"))
    future_activity = input_state.get("future_production_activity_by_step")
    if not isinstance(future_activity, list):
        future_activity = _eaa_final_build_future_activity(self, future_context, production_summary)
    input_state["future_production_activity_by_step"] = future_activity
    input_state["energy_numeric_history_reference"] = input_state.get("energy_numeric_history_reference") or _eaa_final_reference_from_history(self, input_state)
    input_state["energy_repair_context"] = _eaa_final_build_repair_context_for_agent(self, input_state, production_summary)
    input_state["task_role"] = "energy_horizon_prediction_with_mechanism_aware_schedule_conditioning"
    input_state["production_conditioning_available"] = bool(production_summary.get("available") or production_summary.get("num_segments"))
    input_state["production_activity_basis"] = "schedule_conditioned" if input_state["production_conditioning_available"] else "history_only"
    input_state["llm_reasoning_contract"] = {
        "must_output_numeric_future_patch_predictions": True,
        "must_keep_context_ids": True,
        "must_use_step_level_activity_when_available": True,
        "must_not_use_target_future": True,
        "do_not_smooth_all_steps": True,
        "explain_each_step_with_activity_basis_in_comment": True,
    }

    horizon = input_state.get("horizon") or sample.get("horizon") or len(future_context)
    basis = "schedule_conditioned" if input_state["production_conditioning_available"] else "history_only"

    return {
        "task_instruction": (
            "You are an energy patch-level horizon prediction reasoner for a steel plant production-energy world model. "
            "Predict numeric future energy states for the next K patches. This is prediction, not optimization. "
            "Use observed energy_history, semantic_energy_history, history_transition_summary, known future_patch_context, "
            "and future_production_activity_by_step. The production side is represented only by operation-level Gantt segment summary. "
            "Do not predict production active heats or heat-patch alignment. Do not output dispatch advice. "
            "Do not access target_future. "
            "When future_production_activity_by_step shows BOF overlap, infer possible LDG generation response; "
            "when Rolling/LF overlap is high, infer possible power demand and purchased power response; "
            "when Casting/Rolling overlap is high, infer possible steam demand and purchased steam response. "
            "Do not output a flat template sequence unless all history and production activity are truly flat. "
            "For every output step, keep patch_id and heat_id aligned with future_patch_context[step-1]. "
            "Return exactly one valid JSON object."
        ),
        "input": input_state,
        "output_schema": {
            "task": "energy_horizon_prediction",
            "mode": "energy_patch_horizon_predictor_mechanism_aware_gantt_conditioned",
            "horizon": horizon,
            "future_patch_predictions": [
                {
                    "step": 1,
                    "patch_id": "must equal future_patch_context[step-1].id",
                    "heat_id": "must equal future_patch_context[step-1].hn",
                    "g": {"b": {"tot": 0.0}, "c": {"tot": 0.0}, "l": {"tot": 0.0}},
                    "st": {"b": {"e": 0.0, "m": 0.0}, "c": {"e": 0.0, "m": 0.0}, "l": {"e": 0.0, "m": 0.0}},
                    "dm": {"s": {"tot": 0.0, "pk": 0.0}, "e": {"tot": 0.0, "pk": 0.0}},
                    "ex": {"s": {"tot": 0.0}, "e": {"tot": 0.0}},
                    "rk": [],
                    "semantic_transition": {
                        "ldg_generation_trend": "up|flat|down|unknown",
                        "steam_load_trend": "up|flat|down|unknown",
                        "power_load_trend": "up|flat|down|unknown",
                        "purchase_dependency": "low|medium|high|unknown"
                    },
                    "response_type": "direct|indirect|none|unknown",
                    "production_activity_basis": basis,
                    "operation_segment_summary_used": bool(input_state["production_conditioning_available"]),
                    "confidence": 0.0,
                    "comment": "must mention activity evidence or uncertainty"
                }
            ],
            "summary": {
                "holder_trend": {"BFG": "up|down|flat|unknown", "COG": "up|down|flat|unknown", "LDG": "up|down|flat|unknown"},
                "purchase_trend": {"power": "up|down|flat|unknown", "steam": "up|down|flat|unknown"},
                "overall_feasibility": "ok|warning|risky",
                "production_conditioning_available": bool(input_state["production_conditioning_available"]),
                "production_conditioning_source": "operation_segment_summary" if input_state["production_conditioning_available"] else "missing",
                "production_activity_basis": basis,
                "confidence": 0.0,
                "comment": "string"
            }
        }
    }

def _eaa_final_build_horizon_messages(self, model_input):
    system_prompt = (
        "You are a mechanism-aware industrial energy horizon prediction model.\n"
        "You must output numeric future energy patch states as JSON.\n"
        "Use step-level production activity when available; never use target future labels.\n"
        "Avoid generic flat or smoothed sequences. Each step should respond to its own BOF/LF/Casting/Rolling overlap evidence.\n"
        "Keep patch_id and heat_id bound to future_patch_context.\n"
        "Return exactly one JSON object; no markdown, no explanation outside JSON."
    )
    user_prompt = json.dumps(model_input, ensure_ascii=False, indent=2)
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

def _eaa_final_linear_forecast(rows, key, step):
    if not rows:
        return 0.0
    vals = [r.get(key, 0.0) for r in rows]
    last = vals[-1]
    trend = _eaa_final_trend(vals)
    hist_min = min(vals)
    hist_max = max(vals)
    hist_rng = max(hist_max - hist_min, 1.0)
    val = last + step * trend
    return min(max(val, max(0.0, hist_min - 0.50 * hist_rng)), hist_max + 0.75 * hist_rng)

def _eaa_energy_build_rule_fallback_prediction(self, sample, model_input, production_prediction=None):
    input_state = self._safe_dict(model_input.get("input"))
    future_ctx = self._safe_list(input_state.get("future_patch_context"))
    horizon = int(input_state.get("horizon") or sample.get("horizon") or len(future_ctx))
    prod_summary = self._safe_dict(input_state.get("production_operation_segment_summary"))
    prod_available = bool(prod_summary.get("available") or prod_summary.get("num_segments"))
    activity_by_step = input_state.get("future_production_activity_by_step")
    if not isinstance(activity_by_step, list):
        activity_by_step = _eaa_final_build_future_activity(self, future_ctx, prod_summary)
    rows = _eaa_final_numeric_history_rows(self, self._safe_list(input_state.get("energy_history")))
    last_hist = self._safe_list(input_state.get("energy_history"))[-1] if self._safe_list(input_state.get("energy_history")) else {}

    predictions = []
    for idx in range(horizon):
        step = idx + 1
        fctx = self._safe_dict(future_ctx[idx]) if idx < len(future_ctx) else {}
        act = self._safe_dict(activity_by_step[idx]) if idx < len(activity_by_step) else {}
        t = self._safe_dict(fctx.get("t"))
        pid = self._safe_str(fctx.get("id") or fctx.get("patch_id") or fctx.get("p") or f"fallback_step_{step}")
        heat_id = self._normalize_heat_id(fctx.get("hn") or self._safe_dict(fctx.get("pc")).get("phn") or fctx.get("heat_id") or pid)
        activity_score = _eaa_final_float(act.get("activity_score"), 0.0)
        stage_ov = self._safe_dict(act.get("stage_overlap_min"))
        bof = _eaa_final_float(stage_ov.get("BOF"), 0.0)
        lf = _eaa_final_float(stage_ov.get("LF"), 0.0)
        casting = _eaa_final_float(stage_ov.get("Casting"), 0.0)
        rolling = _eaa_final_float(stage_ov.get("Rolling"), 0.0)

        g_b = _eaa_final_linear_forecast(rows, "g_b_tot", step)
        g_c = _eaa_final_linear_forecast(rows, "g_c_tot", step)
        g_l = _eaa_final_linear_forecast(rows, "g_l_tot", step) * (1.0 + min(0.85, bof / 80.0))
        dm_s = _eaa_final_linear_forecast(rows, "dm_s_tot", step) * (1.0 + min(0.75, (casting + 0.6 * rolling) / 140.0))
        dm_e = _eaa_final_linear_forecast(rows, "dm_e_tot", step) * (1.0 + min(0.95, (rolling + 0.7 * lf) / 120.0))
        dm_s_pk = max(dm_s, _eaa_final_linear_forecast(rows, "dm_s_pk", step) * (1.0 + min(0.75, (casting + rolling) / 160.0)))
        dm_e_pk = max(dm_e, _eaa_final_linear_forecast(rows, "dm_e_pk", step) * (1.0 + min(0.95, (rolling + lf) / 140.0)))
        ex_s = max(_eaa_final_linear_forecast(rows, "ex_s_tot", step), max(0.0, dm_s - 0.55 * g_l) * 0.35)
        ex_e = max(_eaa_final_linear_forecast(rows, "ex_e_tot", step), max(0.0, dm_e - 0.20 * g_b) * 0.55)
        st_b = min(max(_eaa_final_linear_forecast(rows, "st_b_e", step), 0.0), 400.0)
        st_c = min(max(_eaa_final_linear_forecast(rows, "st_c_e", step), 0.0), 200.0)
        st_l_prev = _eaa_final_get(last_hist, ["st", "l", "e"])
        st_l = min(max(st_l_prev + 0.05 * (g_l - 0.25 * dm_s - 0.12 * dm_e), 0.0), 450.0)

        rk = []
        if ex_e > 1e-6:
            rk.append("pp")
        if ex_s > 1e-6:
            rk.append("ps")
        if bool(self._safe_dict(fctx.get("pc")).get("ov")):
            rk.append("ov")
        if st_l < 30:
            rk.append("l_low")
        if st_l > 320:
            rk.append("l_high")

        predictions.append({
            "step": step,
            "patch_id": pid,
            "id": pid,
            "p": pid,
            "heat_id": heat_id,
            "hn": heat_id,
            "t": {"ts": t.get("ts") or fctx.get("start_time"), "te": t.get("te") or fctx.get("end_time"), "t0": t.get("t0") or fctx.get("start_min"), "t1": t.get("t1") or fctx.get("end_min")},
            "g": {"b": {"tot": round(g_b, 4)}, "c": {"tot": round(g_c, 4)}, "l": {"tot": round(g_l, 4)}},
            "st": {"b": {"e": round(st_b, 4), "m": round(st_b, 4)}, "c": {"e": round(st_c, 4), "m": round(st_c, 4)}, "l": {"e": round(st_l, 4), "m": round(st_l, 4)}},
            "dm": {"s": {"tot": round(dm_s, 4), "pk": round(dm_s_pk, 4)}, "e": {"tot": round(dm_e, 4), "pk": round(dm_e_pk, 4)}},
            "ex": {"s": {"tot": round(ex_s, 4)}, "e": {"tot": round(ex_e, 4)}},
            "rk": sorted(set(rk)),
            "response_type": "direct" if prod_available and activity_score > 0 else ("indirect" if prod_available else "none"),
            "production_activity_basis": "schedule_conditioned" if prod_available else "history_only",
            "operation_segment_summary_used": bool(prod_available),
            "semantic_transition": {"activity_score": round(activity_score, 4), "stage_overlap_min": stage_ov},
            "source": "mechanism_trend_rule_fallback_after_retry_exhausted",
            "confidence": 0.42 if prod_available else 0.30,
            "comment": "mechanism trend fallback using observed history and predicted operation-segment activity only; no target labels used",
        })

    return {
        "task": "energy_horizon_prediction",
        "mode": "energy_patch_horizon_predictor_mechanism_trend_rule_fallback",
        "horizon": horizon,
        "future_patch_predictions": predictions,
        "summary": {
            "production_conditioning_available": bool(prod_available),
            "production_conditioning_source": "operation_segment_summary" if prod_available else "missing",
            "production_activity_basis": "schedule_conditioned" if prod_available else "history_only",
            "overall_feasibility": "warning",
            "confidence": 0.42 if prod_available else 0.30,
            "comment": "fallback used after retry exhaustion; generated from history trend and step-level production activity; target_future not used",
        },
        "rule_fallback_meta": {
            "rule_fallback_used": True,
            "rule_fallback_uses_future_target_labels": False,
            "rule_fallback_num_raw_steps": len(predictions),
            "rule_fallback_source": "history_trend_plus_future_context_plus_step_level_operation_activity",
        },
    }

EnergyAgentAdapter.build_horizon_input = _eaa_final_build_horizon_input
EnergyAgentAdapter.build_horizon_messages = _eaa_final_build_horizon_messages

_EAA_V2_PRE_BUILD_HORIZON_INPUT = EnergyAgentAdapter.build_horizon_input

def _eaa_v2_build_horizon_input(self, sample, production_prediction=None):
    model_input = _EAA_V2_PRE_BUILD_HORIZON_INPUT(self, sample, production_prediction=production_prediction)
    if not isinstance(model_input, dict):
        return model_input
    inp = model_input.get("input") if isinstance(model_input.get("input"), dict) else {}
    sample_input = sample.get("input_state") if isinstance(sample, dict) and isinstance(sample.get("input_state"), dict) else {}

    if "energy_concrete_history_window" not in inp and isinstance(sample_input.get("energy_concrete_history_window"), list):
        inp["energy_concrete_history_window"] = sample_input.get("energy_concrete_history_window")
    if "energy_baseline_future_predictions" not in inp and isinstance(sample_input.get("energy_baseline_future_predictions"), list):
        inp["energy_baseline_future_predictions"] = sample_input.get("energy_baseline_future_predictions")
    if "energy_baseline_policy" not in inp and isinstance(sample_input.get("energy_baseline_policy"), dict):
        inp["energy_baseline_policy"] = sample_input.get("energy_baseline_policy")

    inp.setdefault("energy_prediction_policy", {})
    inp["energy_prediction_policy"].update({
        "concrete_energy_history_required": True,
        "baseline_is_reference_not_answer": True,
        "do_not_copy_baseline_blindly": True,
        "must_predict_full_numeric_energy_state": True,
        "must_explain_deviation_from_baseline_in_comment": True,
        "target_future_visible": False,
    })

    rc = inp.get("energy_repair_context") if isinstance(inp.get("energy_repair_context"), dict) else {}
    if isinstance(inp.get("energy_baseline_future_predictions"), list):
        rc["energy_baseline_future_predictions"] = inp.get("energy_baseline_future_predictions")
        rc["baseline_blend_weight"] = 0.20
        rc["baseline_blend_policy"] = "soft_prior_only_for_missing_or_extreme_llm_values"
        rc["uses_future_target_labels"] = False
        inp["energy_repair_context"] = rc

    old = str(model_input.get("task_instruction") or "")
    extra = (
        " The input contains energy_concrete_history_window with observed numeric energy values. "
        "It may also contain energy_baseline_future_predictions built only from observed history and predicted production activity. "
        "The baseline is NOT ground truth and must NOT be copied blindly. "
        "Use it as a numeric reference, then infer deviations using future_production_activity_by_step and semantic_energy_history. "
        "Output full numeric future_patch_predictions. "
    )
    if "energy_concrete_history_window" not in old:
        model_input["task_instruction"] = old + extra

    schema = model_input.get("output_schema") if isinstance(model_input.get("output_schema"), dict) else {}
    fps = schema.get("future_patch_predictions")
    if isinstance(fps, list) and fps and isinstance(fps[0], dict):
        fps[0].setdefault("baseline_reference_used", True)
        fps[0].setdefault("delta_from_baseline", {
            "ldg_generation_total": 0.0,
            "ldg_holder_end": 0.0,
            "steam_load_total": 0.0,
            "power_load_total": 0.0,
            "purchased_steam_total": 0.0,
            "purchased_power_total": 0.0,
        })
        fps[0].setdefault("baseline_deviation_reason", "string")
    model_input["input"] = inp
    return model_input

_EAA_V2_PRE_BUILD_HORIZON_MESSAGES = EnergyAgentAdapter.build_horizon_messages

def _eaa_v2_build_horizon_messages(self, model_input):

    user_prompt = json.dumps(model_input, ensure_ascii=False, indent=2)
    system_prompt = (
        "You are a mechanism-aware industrial energy horizon prediction model.\n"
        "You must output numeric future energy patch states as JSON.\n"
        "Use observed concrete energy history values, semantic energy history, and step-level production activity.\n"
        "If energy_baseline_future_predictions is provided, treat it only as a soft numeric reference, not as ground truth.\n"
        "Do not copy the baseline blindly; infer deviations from baseline using BOF/LF/Casting/Rolling activity and energy mechanism.\n"
        "Never use target_future or any future measured energy labels.\n"
        "Avoid generic flat or smoothed sequences. Each step should respond to its own activity evidence.\n"
        "Keep patch_id and heat_id bound to future_patch_context.\n"
        "Return exactly one JSON object; no markdown, no explanation outside JSON."
    )
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

EnergyAgentAdapter.build_horizon_input = _eaa_v2_build_horizon_input
EnergyAgentAdapter.build_horizon_messages = _eaa_v2_build_horizon_messages

from copy import deepcopy as _eaa_stage1_deepcopy

_eaa_stage1_prev_build_horizon_input = EnergyAgentAdapter.build_horizon_input
_eaa_stage1_prev_predict_horizon = EnergyAgentAdapter.predict_horizon


def _eaa_stage1_build_horizon_input(self, sample, production_prediction=None):
    model_input = _eaa_stage1_prev_build_horizon_input(self, sample, production_prediction=production_prediction)
    sample_dict = self._safe_dict(sample)
    sample_input = self._safe_dict(sample_dict.get("input_state"))
    action = self._safe_dict(sample_dict.get("candidate_action") or sample_input.get("candidate_action"))
    if action:
        input_block = self._safe_dict(model_input.get("input"))
        input_block["candidate_action"] = action
        input_block["candidate_energy_action"] = self._safe_dict(action.get("energy_action"))
        input_block["linked_candidate_production_action"] = self._safe_dict(action.get("production_action"))
        input_block["action_conditioned_world_model"] = True
        model_input["input"] = input_block
    return model_input


def _eaa_stage1_predict_horizon(self, sample, production_prediction=None, candidate_action=None):
    working = sample
    if candidate_action is not None:
        working = _eaa_stage1_deepcopy(sample)
        if isinstance(working, dict):
            working["candidate_action"] = candidate_action
            inp = working.setdefault("input_state", {})
            if isinstance(inp, dict):
                inp["candidate_action"] = candidate_action
    return _eaa_stage1_prev_predict_horizon(self, working, production_prediction=production_prediction)


EnergyAgentAdapter.build_horizon_input = _eaa_stage1_build_horizon_input
EnergyAgentAdapter.predict_horizon = _eaa_stage1_predict_horizon
