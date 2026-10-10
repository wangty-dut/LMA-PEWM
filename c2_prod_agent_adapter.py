from __future__ import annotations

import json
import os
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from b_base_chat_model import BaseChatModel
from f_transition_engine import CandidatePlan

try:
    from mechanism_schedule_projection import PhysicsConsistentVariationalProjector
except Exception:
    PhysicsConsistentVariationalProjector = None

def _safe_json_loads(text: str):

    text = text.strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, (dict, list)):
            return obj
    except Exception:
        pass

    fenced = re.findall(r"```json\s*(.*?)\s*```", text, flags=re.S | re.I)
    for block in fenced:
        try:
            obj = json.loads(block)
            if isinstance(obj, (dict, list)):
                return obj
        except Exception:
            continue

    start_arr = text.find("[")
    end_arr = text.rfind("]")
    if start_arr != -1 and end_arr != -1 and end_arr > start_arr:
        candidate = text[start_arr:end_arr + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, (dict, list)):
                return obj
        except Exception:
            pass

    start_obj = text.find("{")
    end_obj = text.rfind("}")
    if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
        candidate = text[start_obj:end_obj + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, (dict, list)):
                return obj
        except Exception:
            pass

    raise ValueError("Unable to parse JSON from the model output.")

class ProductionAgentAdapter:

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        max_new_tokens: int = 4096,
        max_horizon_retries: Optional[int] = None,
        enable_retry_exhausted_rule_fallback: Optional[bool] = None,
        evaluate_rule_fallback_as_prediction: Optional[bool] = None,
    ) -> None:
        self.chat_model = BaseChatModel(
            model_path=model_path,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=False,
        )
        self.physics_projector = (
            PhysicsConsistentVariationalProjector()
            if PhysicsConsistentVariationalProjector is not None else None
        )
        def _env_bool(name: str, default: bool) -> bool:
            val = os.environ.get(name)
            if val is None:
                return default
            return str(val).strip().lower() not in {"0", "false", "no", "off", "none"}

        if max_horizon_retries is None:
            try:
                max_horizon_retries = int(os.environ.get("PROD_MAX_HORIZON_RETRIES", "5"))
            except Exception:
                max_horizon_retries = 5
        self.max_horizon_retries = max(0, int(max_horizon_retries or 0))

        if enable_retry_exhausted_rule_fallback is None:
            enable_retry_exhausted_rule_fallback = _env_bool("PROD_ENABLE_RULE_FALLBACK_AFTER_RETRY", True)
        if evaluate_rule_fallback_as_prediction is None:
            evaluate_rule_fallback_as_prediction = _env_bool("PROD_EVALUATE_RULE_FALLBACK_AS_PREDICTION", True)
        self.enable_retry_exhausted_rule_fallback = bool(enable_retry_exhausted_rule_fallback)
        self.evaluate_rule_fallback_as_prediction = bool(evaluate_rule_fallback_as_prediction)

    def _safe_dict(self, x: Any) -> Dict[str, Any]:
        return x if isinstance(x, dict) else {}

    def _safe_list(self, x: Any) -> List[Any]:
        return x if isinstance(x, list) else []

    def _safe_str(self, x: Any, default: str = "") -> str:
        if x is None:
            return default
        return str(x).strip()

    def _normalize_heat_id(self, x: Any) -> Optional[str]:

        if x is None:
            return None
        s = str(x).strip()
        if not s:
            return None
        s = s.replace("-", "_")
        m = re.search(r"([1-9]\d*LDG_\d+)", s)
        if m:
            return m.group(1)
        parts = s.split("_")
        for i in range(len(parts) - 1):
            if parts[i].endswith("LDG") and parts[i + 1]:
                return f"{parts[i]}_{parts[i + 1]}"
        return s

    def _dedup_preserve_order(self, items: List[Any]) -> List[Any]:
        seen = set()
        out = []
        for x in items:
            if x is None:
                continue
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    def _safe_float(self, x: Any, default: float = 0.0) -> float:
        try:
            if x is None:
                return default
            return float(x)
        except Exception:
            return default

    def _compact_heat_for_agent(self, h: Dict[str, Any]) -> Dict[str, Any]:
        keep = ["id", "f", "sd", "sf", "tb0", "tb1", "tl0", "tl1", "be", "bp", "ru", "mk"]
        out = {k: h[k] for k in keep if k in h and h[k] is not None}

        mk = self._safe_dict(out.get("mk"))
        if mk:
            out["mk"] = {
                "sw": mk.get("sw"),
                "hc": mk.get("hc"),
                "tt": mk.get("tt"),
                "wg": mk.get("wg"),
            }

        return out

    def _compact_rolling_for_agent(self, r: Dict[str, Any]) -> Dict[str, Any]:
        keep = ["u", "gap", "c0", "c1", "t0", "t1", "dur", "p", "hc", "hs", "mk"]
        out = {k: r[k] for k in keep if k in r and r[k] is not None}

        mk = self._safe_dict(out.get("mk"))
        if mk:
            out["mk"] = {
                "pl": mk.get("pl"),
                "as": mk.get("as"),
            }

        return out

    def _compact_peak_for_agent(self, peak_heats: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [{"id": x["id"]} for x in peak_heats if isinstance(x, dict) and x.get("id")]

    def _normalize_prod_state(self, prod_state: Dict[str, Any]) -> Dict[str, Any]:

        if not isinstance(prod_state, dict):
            return {}

        hs = self._safe_list(prod_state.get("hs"))
        rs = self._safe_list(prod_state.get("rs"))
        ph = self._safe_list(prod_state.get("ph"))

        peak_ids = [x.get("id") for x in ph if isinstance(x, dict) and x.get("id")]
        hs_sorted = sorted(
            hs,
            key=lambda x: ((x.get("be", 0.0) or 0.0), (x.get("bp", 0.0) or 0.0)),
            reverse=True,
        )

        focus_ids = list(peak_ids)
        for h in hs_sorted:
            hid = h.get("id")
            if hid and hid not in focus_ids:
                focus_ids.append(hid)
            if len(focus_ids) >= 12:
                break

        heat_map = {h["id"]: h for h in hs if isinstance(h, dict) and h.get("id")}
        focus_heats = [heat_map[hid] for hid in focus_ids if hid in heat_map]

        focus_ru_ids = []
        for h in focus_heats:
            ru = h.get("ru")
            if ru and ru not in focus_ru_ids:
                focus_ru_ids.append(ru)

        rolling_map = {r["u"]: r for r in rs if isinstance(r, dict) and r.get("u")}
        focus_rollings = [rolling_map[uid] for uid in focus_ru_ids if uid in rolling_map][:6]

        out = {
            "d": prod_state.get("d"),
            "m": "production",
            "s": prod_state.get("s", {}),
            "cf": prod_state.get("cf", {}),
            "ph": self._compact_peak_for_agent(ph),
            "hs": [self._compact_heat_for_agent(h) for h in focus_heats],
            "rs": [self._compact_rolling_for_agent(r) for r in focus_rollings],
            "pm": {
                "fh": len(hs),
                "fr": len(rs),
            },
        }

        top_mk = self._safe_dict(prod_state.get("mk"))
        if top_mk:
            out["mk"] = top_mk

        return out

    def _build_state_schema_legend(self) -> Dict[str, Any]:
        return {
            "prod_state_top": {
                "d": "day_id",
                "m": "mode",
                "s": "summary",
                "cf": "constraint_features",
                "ph": "peak_heat_ids",
                "hs": "focus_heat_states",
                "rs": "focus_rolling_states",
                "pm": "prompt_meta",
            },
            "heat_fields": {
                "id": "heat_id",
                "f": "furnace",
                "sd": "sequence_in_day",
                "sf": "sequence_in_furnace",
                "tb0": "bof_start_relative",
                "tb1": "bof_end_relative",
                "tl0": "lf_start_relative",
                "tl1": "lf_end_relative",
                "be": "bof_energy_mwh",
                "bp": "bof_avg_power_mw",
                "ru": "rolling_unit_id",
            },
            "rolling_fields": {
                "u": "rolling_unit_id",
                "gap": "is_gap_segment",
                "c0": "casting_window_start_relative",
                "c1": "casting_window_end_relative",
                "t0": "rolling_start_relative",
                "t1": "rolling_end_relative",
                "dur": "rolling_duration_min",
                "p": "rolling_avg_power_mw",
                "hc": "num_heats_in_unit",
                "hs": "heat_id_list",
            },
            "notes": [
                "Production agent only sees production state.",
                "This is state prediction, not scheduling optimization.",
                "Do not output dispatching actions.",
                "For horizon mode, output operation-level schedule segments when requested.",
                "Only output valid JSON.",
            ]
        }

    def _infer_heat_id_from_pid(self, pid: Any) -> Optional[str]:

        return self._normalize_heat_id(pid)

    def _infer_converter_from_pid(self, pid: Any) -> Optional[str]:
        if pid is None:
            return None
        s = str(pid).strip()
        m = re.search(r"(Converter\d+)", s)
        return m.group(1) if m else None

    def _future_calendar_by_step(self, sample: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
        input_state = self._safe_dict(sample.get("input_state"))
        calendar = self._safe_list(input_state.get("known_future_patch_calendar"))
        out: Dict[int, Dict[str, Any]] = {}
        for i, item in enumerate(calendar, start=1):
            if not isinstance(item, dict):
                continue
            try:
                step = int(item.get("step") or i)
            except Exception:
                step = i
            out[step] = item
        return out

    def _history_heat_ids(self, sample: Dict[str, Any]) -> set:
        input_state = self._safe_dict(sample.get("input_state"))
        ids = set()
        for p in self._safe_list(input_state.get("history_patches")) + [self._safe_dict(input_state.get("anchor_patch"))]:
            if not isinstance(p, dict):
                continue
            for hid in self._safe_list(p.get("active_heat_ids")):
                s = self._normalize_heat_id(hid)
                if s:
                    ids.add(s)
        return ids

    def _recompute_stage_counts(self, active_heats: List[Dict[str, Any]]) -> Dict[str, int]:
        stage_counts = {"BOF": 0, "LF": 0, "Casting": 0, "Rolling": 0}
        for h in active_heats:
            if not isinstance(h, dict):
                continue
            stages = self._safe_list(h.get("stages"))
            stage = self._safe_str(h.get("stage"), "unknown")
            if not stages and stage in stage_counts:
                stages = [stage]
            for st in stages:
                if st in stage_counts:
                    stage_counts[st] += 1
        return stage_counts

    def _make_calendar_heat(self, heat_id: str, stage_hint: str = "BOF") -> Dict[str, Any]:
        stage = stage_hint if stage_hint in {"BOF", "LF", "Casting", "Rolling"} else "BOF"
        return {"id": self._normalize_heat_id(heat_id), "stage": stage, "stages": [stage], "ru": None}

    def _repair_prediction_with_calendar(
        self,
        pred: Dict[str, Any],
        calendar_item: Dict[str, Any],
        history_heat_ids: set,
    ) -> Dict[str, Any]:

        pred = self._safe_dict(pred).copy()
        calendar_item = self._safe_dict(calendar_item)
        if calendar_item:

            for k in ["step", "pi", "pid", "t0", "t1", "canonical_heat_id", "converter"]:
                if calendar_item.get(k) is not None:
                    pred[k] = calendar_item.get(k)

        canonical_heat_id = self._normalize_heat_id(
            calendar_item.get("canonical_heat_id") or
            calendar_item.get("heat_id") or
            calendar_item.get("hn") or
            calendar_item.get("pid") or
            pred.get("canonical_heat_id") or
            pred.get("pid")
        )
        if not canonical_heat_id:
            return pred
        pred["canonical_heat_id"] = canonical_heat_id

        active_heats = [h for h in self._safe_list(pred.get("active_heats")) if isinstance(h, dict)]
        active_heat_ids = [self._normalize_heat_id(x) for x in self._safe_list(pred.get("active_heat_ids")) if self._normalize_heat_id(x)]
        if not active_heat_ids and active_heats:
            active_heat_ids = [self._normalize_heat_id(h.get("id")) for h in active_heats if self._normalize_heat_id(h.get("id"))]

        pred_set = set(active_heat_ids)
        stale_history_copy = bool(pred_set and canonical_heat_id not in pred_set and pred_set.issubset(history_heat_ids))

        if stale_history_copy:

            active_heats = [self._make_calendar_heat(canonical_heat_id, stage_hint="BOF")]
            pred["repair_note"] = "replaced_stale_history_heat_ids_with_calendar_heat_id"
        elif canonical_heat_id not in pred_set:
            active_heats.append(self._make_calendar_heat(canonical_heat_id, stage_hint="BOF"))
            pred["repair_note"] = "added_calendar_heat_id_missing_from_prediction"
        else:

            fixed = []
            seen = set()
            for h in active_heats:
                hid = self._normalize_heat_id(h.get("id"))
                if not hid or hid in seen:
                    continue
                seen.add(hid)
                stage = self._safe_str(h.get("stage"), "unknown")
                stages = self._safe_list(h.get("stages"))
                if hid == canonical_heat_id and stage == "unknown" and not stages:
                    stage = "BOF"
                    stages = ["BOF"]
                fixed.append({"id": hid, "stage": stage, "stages": stages, "ru": h.get("ru")})
            active_heats = fixed

        dedup = []
        seen = set()
        for h in active_heats:
            hid = self._safe_str(h.get("id"))
            if not hid or hid in seen:
                continue
            seen.add(hid)
            stage = self._safe_str(h.get("stage"), "unknown")
            stages = self._safe_list(h.get("stages"))
            if not stages and stage != "unknown":
                stages = [stage]
            dedup.append({"id": hid, "stage": stage, "stages": stages, "ru": h.get("ru")})

        pred["active_heats"] = dedup
        pred["active_heat_ids"] = [h["id"] for h in dedup]
        pred["stage_counts"] = self._recompute_stage_counts(dedup)
        pred["active_rolling_units"] = self._safe_list(pred.get("active_rolling_units"))
        return pred

    def _build_projection_context_summary(self, repair_context: Any) -> Dict[str, Any]:

        ctx = self._safe_dict(repair_context)
        if not ctx:
            return {
                "enabled": False,
                "method": "LLM-guided Physics-consistent Variational Projection",
                "note": "No repair_context found; projection will fall back to normalized model segments.",
            }
        return {
            "enabled": True,
            "method": ctx.get("method", "LLM-guided Physics-consistent Variational Projection"),
            "method_short": ctx.get("method_short", "PCVP"),
            "uses_future_target_labels": bool(ctx.get("uses_future_target_labels", False)),
            "horizon_window": ctx.get("horizon_window"),
            "stage_order": ctx.get("stage_order", ["BOF", "LF", "Casting", "Rolling"]),
            "constraint_families": list(self._safe_dict(ctx.get("constraints")).keys()),
            "num_history_tail_operation_segments": len(self._safe_list(ctx.get("history_tail_operation_segments"))),
            "num_future_calendar_steps": len(self._safe_list(ctx.get("known_future_patch_calendar"))),
            "num_rolling_queue_units": len(self._safe_dict(ctx.get("rolling_queue_prior"))),
            "num_casting_sequence_groups": len(self._safe_list(ctx.get("casting_sequence_prior"))),
            "important_note": "PCVP is applied after LLM generation and does not use target_future labels.",
        }

    def build_horizon_input(self, sample: Dict[str, Any]) -> Dict[str, Any]:

        sample = self._safe_dict(sample)
        input_state = self._safe_dict(sample.get("input_state"))
        meta = self._safe_dict(sample.get("meta"))
        schedule_task = self._safe_dict(input_state.get("operation_schedule_task"))

        return {
            "task_instruction": (
                "You are a production operation-level world model for a steelmaking, continuous-casting and hot-rolling line. "
                "Your only task is to predict operation-level production schedule segments in the future horizon window. "
                "Do not generate patch-level active heat predictions, dispatching actions, or optimization suggestions. "
                "The output must be directly usable for predicted-vs-ground-truth production Gantt comparison. "
                "Use only the anchor context, visible historical operation segments, duration priors, and the known future patch calendar. "
                "Do not use target_future labels. Only output valid JSON."
            ),
            "input": {
                "day_id": sample.get("day_id"),
                "anchor_patch_idx": sample.get("anchor_patch_idx"),
                "history_window": sample.get("history_window"),
                "horizon": sample.get("horizon"),
                "known_future_time_window": input_state.get("known_future_time_window"),
                "known_future_patch_calendar": input_state.get("known_future_patch_calendar"),
                "anchor_context": input_state.get("anchor_context"),
                "operation_schedule_prediction_context": schedule_task,
                "local_schedule_blocks": input_state.get("local_schedule_blocks"),
                "heat_route_skeletons": input_state.get("heat_route_skeletons"),
                "retry_guidance_context": input_state.get("retry_guidance_context"),
                "physics_consistent_projection_context_summary": self._build_projection_context_summary(
                    input_state.get("repair_context")
                ),
                "operation_segment_schema": input_state.get("operation_segment_schema"),
                "meta": {
                    "num_candidate_future_heat_ids": meta.get("num_candidate_future_heat_ids"),
                    "num_visible_history_operation_segments": meta.get("num_visible_history_operation_segments"),
                    "num_local_history_operation_segments": meta.get("num_local_history_operation_segments"),
                    "is_truncated_future": meta.get("is_truncated_future"),
                    "mixed_day_warning": meta.get("mixed_day_warning"),
                    "rolling_unit_fallback_warning": meta.get("rolling_unit_fallback_warning"),
                },
            },
            "required_reasoning_contract": {
                "predicted_operation_segments": [
                    "Output only predicted_operation_segments; do not output future_production_states.",
                    "Include all candidate heats whose operations overlap the future horizon window.",
                    "For each candidate heat, output BOF, LF, Casting and Rolling segments that overlap the known_future_time_window.",
                    "Respect BOF -> LF -> Casting -> Rolling order and avoid impossible overlaps on the same unit.",
                    "Clip segment start/end to the future horizon window.",
                    "Use heat_route_skeletons to decide which stages of each heat can appear in the horizon.",
                    "Use local_schedule_blocks to cover long horizons block by block, but return one combined predicted_operation_segments list.",
                    "Use historical operation segments and duration_priors to infer stage durations and downstream timing.",
                    "Do not output only one BOF segment per future patch; describe full operation-level occupancy over the horizon.",
                    "For Rolling, do not copy a full rolling-unit window to every heat; split shared rolling windows into heat-level Rolling segments by heat order or duration priors.",
                    "Every valid model-predicted segment must set source exactly to model_output.",
                    "Do not mark calendar/fallback/patch-derived/target/history segments as model_output. If timing cannot be inferred, omit the segment rather than using fallback."
                ]
            },
            "output_schema": {
                "task": "production_gantt_horizon_prediction",
                "mode": "production_operation_schedule_world_model_gantt_only",
                "predicted_operation_segments": [
                    {
                        "heat_id": "canonical heat id",
                        "stage": "BOF|LF|Casting|Rolling",
                        "unit": "Converter1|Converter2|Converter3|LF|Casting|rolling unit id",
                        "start_time": "YYYY-MM-DD HH:MM:SS or HH:MM:SS",
                        "end_time": "YYYY-MM-DD HH:MM:SS or HH:MM:SS",
                        "start_min": 0.0,
                        "end_min": 0.0,
                        "duration_min": 0.0,
                        "source_steps": [1, 2],
                        "source": "model_output",
                        "confidence": 0.0,
                        "reason": "history/duration/calendar basis"
                    }
                ]
            }
        }

    def build_horizon_messages(self, model_input: Dict[str, Any]) -> List[Dict[str, str]]:
        system_prompt = (
            "You are a production operation-level world model.\n"
            "Your only task is to predict future operation-level schedule segments for Gantt comparison.\n"
            "Do not predict patch-level active_heat_ids or future_production_states as a separate task.\n"
            "Return exactly one JSON object with keys: task, mode, predicted_operation_segments.\n"
            "\n"
            "Core requirements for predicted_operation_segments:\n"
            "1. Cover the full future horizon window.\n"
            "2. For candidate heats that overlap the window, output all BOF, LF, Casting and Rolling segments that overlap the window.\n"
            "3. Each segment must have heat_id, stage, unit, start_time, end_time, start_min, end_min, duration_min, source_steps, source, confidence.\n"
            "4. The source field of every valid model-predicted operation segment must be exactly model_output.\n"
            "5. Respect the process order BOF -> LF -> Casting -> Rolling and avoid impossible negative durations.\n"
            "6. Use known_future_patch_calendar only for candidate heat identifiers and patch time boundaries; infer operation timing from historical segments and duration priors.\n"
            "7. Do not simply copy the last historical active heats.\n"
            "8. Do not output one BOF segment per future patch only; output multi-heat operation occupancy suitable for a Gantt chart.\n"
            "9. For Rolling, distinguish rolling-unit occupancy from heat-level rolling operation. Do not assign the same full rolling-unit window to every heat.\n"
            "10. If heat-level Rolling times are unavailable, split the rolling-unit window by heat order or duration priors and state the inference basis in reason.\n"
            "11. For long horizons, use heat_route_skeletons and local_schedule_blocks to reason block by block, but return one combined predicted_operation_segments list.\n"
            "12. If this is a retry request, repair only the reported schema/missing-output/mechanism violations; do not start from target labels because they are unavailable.\n"
            "13. The result will pass through a Physics-consistent Variational Projection layer after generation; still output your best mechanism-consistent raw schedule.\n"
            "14. Do not use or mention target_future labels; projection also does not use target labels.\n"
            "\n"
            "Only output valid JSON. Do not explain outside JSON."
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(model_input, ensure_ascii=False, indent=2)},
        ]

    def _call_horizon_model_once(
        self,
        model_input: Dict[str, Any],
        attempt_index: int = 0,
    ) -> Dict[str, Any]:

        messages = self.build_horizon_messages(model_input)
        raw_text = self.chat_model.generate(messages)

        print(f"\n=== RAW PRODUCTION GANTT MODEL OUTPUT START attempt={attempt_index} ===")
        print(raw_text)
        print(f"=== RAW PRODUCTION GANTT MODEL OUTPUT END attempt={attempt_index} ===\n")

        base = {
            "_raw_text": raw_text or "",
            "_attempt_index": int(attempt_index),
            "_retry_request": self._safe_dict(model_input.get("retry_request")),
        }

        if not raw_text or not str(raw_text).strip():
            return {
                **base,
                "task": "production_gantt_horizon_prediction",
                "mode": "production_operation_schedule_empty_fallback",
                "predicted_operation_segments": [],
                "warning": "empty_model_output",
            }

        try:
            parsed = _safe_json_loads(raw_text)
        except Exception as e:
            return {
                **base,
                "task": "production_gantt_horizon_prediction",
                "mode": "production_operation_schedule_parse_fallback",
                "predicted_operation_segments": [],
                "warning": f"json_parse_failed:{type(e).__name__}",
            }

        if isinstance(parsed, dict):
            parsed.update(base)
            return parsed
        if isinstance(parsed, list):
            return {
                **base,
                "task": "production_gantt_horizon_prediction",
                "mode": "production_operation_schedule_world_model_gantt_only",
                "predicted_operation_segments": parsed,
                "_raw_list_output": parsed,
            }
        return {
            **base,
            "task": "production_gantt_horizon_prediction",
            "mode": "production_operation_schedule_type_fallback",
            "predicted_operation_segments": [],
            "warning": "parsed_output_is_not_dict_or_list",
        }

    def call_horizon_model(self, model_input: Dict[str, Any]) -> Dict[str, Any]:

        return self._call_horizon_model_once(model_input, attempt_index=0)

    def _normalize_future_patch_prediction(self, item: Dict[str, Any], fallback_step: int) -> Dict[str, Any]:
        item = self._safe_dict(item)
        active_heats = self._safe_list(item.get("active_heats"))
        heat_ids = self._safe_list(item.get("active_heat_ids"))

        if not heat_ids and active_heats:
            heat_ids = [h.get("id") for h in active_heats if isinstance(h, dict) and h.get("id")]

        norm_heats = []
        for h in active_heats:
            if not isinstance(h, dict):
                continue
            hid = self._normalize_heat_id(h.get("id") or h.get("heat_id") or h.get("heat_no"))
            if not hid:
                continue
            stages = self._safe_list(h.get("stages"))
            stage = self._safe_str(h.get("stage"), "unknown")
            if not stages and stage and stage != "unknown":
                stages = [stage]
            norm_heats.append({
                "id": hid,
                "stage": stage,
                "stages": stages,
                "ru": h.get("ru"),
            })

        existing = {h["id"] for h in norm_heats}
        for hid in heat_ids:
            sid = self._normalize_heat_id(hid)
            if sid and sid not in existing:
                norm_heats.append({"id": sid, "stage": "unknown", "stages": [], "ru": None})
                existing.add(sid)

        stage_counts = self._safe_dict(item.get("stage_counts")) or self._recompute_stage_counts(norm_heats)

        return {
            "step": int(item.get("step") or fallback_step),
            "pi": item.get("pi"),
            "pid": item.get("pid"),
            "t0": item.get("t0"),
            "t1": item.get("t1"),
            "canonical_heat_id": self._normalize_heat_id(item.get("canonical_heat_id") or item.get("heat_id") or item.get("hn") or item.get("pid")),
            "converter": item.get("converter"),
            "active_heat_ids": [h["id"] for h in norm_heats],
            "active_heats": norm_heats,
            "active_rolling_units": self._safe_list(item.get("active_rolling_units")),
            "stage_counts": stage_counts,
            "confidence": self._safe_float(item.get("confidence", 0.0), 0.0),
            "comment": self._safe_str(item.get("comment", "")),
        }

    def _time_to_min_for_segment(self, x: Any) -> Optional[float]:
        if x is None:
            return None
        try:
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
            if m:
                hh = int(m.group(1)); mm = int(m.group(2)); ss = int(m.group(3) or 0)
                return hh * 60 + mm + ss / 60.0
        except Exception:
            return None
        return None

    def _canonical_stage(self, x: Any) -> str:
        s = self._safe_str(x, "Other").strip()
        u = s.upper().replace(" ", "")
        if "BOF" in u or "CONVERTER" in u:
            return "BOF"
        if u == "LF" or "LADLE" in u:
            return "LF"
        if "CAST" in u or u == "CC":
            return "Casting"
        if "ROLL" in u:
            return "Rolling"
        return "Other"
    def _min_to_time_string(self, minute_value: Any) -> Optional[str]:

        try:
            if minute_value is None:
                return None
            total_seconds = int(round(float(minute_value) * 60.0))
            if total_seconds < 0:
                total_seconds = 0
            hh = total_seconds // 3600
            mm = (total_seconds % 3600) // 60
            ss = total_seconds % 60
            return f"{hh:02d}:{mm:02d}:{ss:02d}"
        except Exception:
            return None

    def _source_is_model_output(self, source: Any) -> bool:

        s = self._safe_str(source, "").lower()
        if not s:
            return True
        bad_tokens = ["derived", "fallback", "calendar", "patch", "target", "true", "history", "label"]
        if any(tok in s for tok in bad_tokens):
            return False
        return "model" in s or "prediction" in s or "predicted" in s or s == "model_output"

    def _future_window_bounds(self, future_window: Optional[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float]]:
        fw = self._safe_dict(future_window)
        w0 = self._safe_float(fw.get("t0_min"), None) if fw.get("t0_min") is not None else self._time_to_min_for_segment(fw.get("t0") or fw.get("start_time"))
        w1 = self._safe_float(fw.get("t1_min"), None) if fw.get("t1_min") is not None else self._time_to_min_for_segment(fw.get("t1") or fw.get("end_time"))
        return w0, w1

    def _normalize_operation_segment(
        self,
        seg: Dict[str, Any],
        fallback_idx: int = 0,
        future_window: Optional[Dict[str, Any]] = None,
        require_model_source: bool = False,
    ) -> Optional[Dict[str, Any]]:

        if not isinstance(seg, dict):
            return None

        raw_source = self._safe_str(seg.get("source", ""), "")
        if require_model_source and not self._source_is_model_output(raw_source):
            return None

        hid = self._normalize_heat_id(seg.get("heat_id") or seg.get("id") or seg.get("hn") or seg.get("Heat_No"))
        if not hid:
            return None

        stage = self._canonical_stage(seg.get("stage") or seg.get("process") or seg.get("operation"))
        if stage not in {"BOF", "LF", "Casting", "Rolling"}:
            return None

        unit = seg.get("unit") or seg.get("converter") or seg.get("ru") or seg.get("rolling_unit") or stage
        start_time = seg.get("start_time") or seg.get("start") or seg.get("t0")
        end_time = seg.get("end_time") or seg.get("end") or seg.get("t1")

        start_min = self._safe_float(seg.get("start_min"), None) if seg.get("start_min") is not None else self._time_to_min_for_segment(start_time)
        end_min = self._safe_float(seg.get("end_min"), None) if seg.get("end_min") is not None else self._time_to_min_for_segment(end_time)
        if start_min is None or end_min is None or end_min <= start_min:
            return None

        raw_start_min = float(start_min)
        raw_end_min = float(end_min)

        fw0, fw1 = self._future_window_bounds(future_window)
        clipped_start = False
        clipped_end = False
        if fw0 is not None and fw1 is not None and fw1 > fw0:
            if max(start_min, fw0) >= min(end_min, fw1):
                return None
            new_start = max(start_min, fw0)
            new_end = min(end_min, fw1)
            clipped_start = abs(new_start - start_min) > 1e-9
            clipped_end = abs(new_end - end_min) > 1e-9
            start_min, end_min = new_start, new_end

        source_steps = self._safe_list(seg.get("source_steps"))
        source = "model_output" if require_model_source else (raw_source or "model_output")

        return {
            "segment_id": seg.get("segment_id") or f"seg_{fallback_idx:03d}",
            "heat_id": hid,
            "stage": stage,
            "unit": str(unit) if unit is not None else stage,
            "start_time": self._min_to_time_string(start_min) or start_time,
            "end_time": self._min_to_time_string(end_min) or end_time,
            "start_min": float(start_min),
            "end_min": float(end_min),
            "duration_min": max(0.0, float(end_min) - float(start_min)),
            "raw_start_time": start_time,
            "raw_end_time": end_time,
            "raw_start_min": raw_start_min,
            "raw_end_min": raw_end_min,
            "is_clipped_start": clipped_start,
            "is_clipped_end": clipped_end,
            "source_steps": source_steps,
            "confidence": min(1.0, max(0.0, self._safe_float(seg.get("confidence", 0.0), 0.0))),
            "source": source,
            "source_kind": "model_output" if require_model_source else source,
            "is_valid_gantt_prediction": bool(require_model_source),
            "reason": self._safe_str(seg.get("reason") or seg.get("comment"), ""),
        }

    def _normalize_operation_segments(
        self,
        raw_segments: Any,
        future_window: Optional[Dict[str, Any]] = None,
        require_model_source: bool = False,
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for i, seg in enumerate(self._safe_list(raw_segments), start=1):
            norm = self._normalize_operation_segment(
                seg,
                fallback_idx=i,
                future_window=future_window,
                require_model_source=require_model_source,
            )
            if norm is not None:
                out.append(norm)

        dedup: Dict[Tuple[str, str, str, float, float], Dict[str, Any]] = {}
        for seg in out:
            key = (
                seg.get("heat_id"),
                seg.get("stage"),
                str(seg.get("unit")),
                round(float(seg.get("start_min")), 6),
                round(float(seg.get("end_min")), 6),
            )
            old = dedup.get(key)
            if old is None or self._safe_float(seg.get("confidence"), 0.0) > self._safe_float(old.get("confidence"), 0.0):
                dedup[key] = seg

        out = list(dedup.values())
        out.sort(key=lambda x: (x.get("start_min", 10**12), x.get("unit") or "", x.get("heat_id") or "", x.get("stage") or ""))
        for i, seg in enumerate(out, start=1):
            if not seg.get("segment_id") or str(seg.get("segment_id")).startswith("seg_"):
                seg["segment_id"] = f"pred_seg_{i:03d}" if require_model_source else f"seg_{i:03d}"
        return out

    def _repair_rolling_heat_level_segments(
        self,
        segments: List[Dict[str, Any]],
        sample: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:

        if not segments:
            return [], {
                "num_rolling_split_groups": 0,
                "num_rolling_split_repaired_segments": 0,
            }

        non_rolling: List[Dict[str, Any]] = []
        groups: Dict[Tuple[str, float, float], List[Tuple[int, Dict[str, Any]]]] = {}

        for idx, seg in enumerate(segments):
            if not isinstance(seg, dict):
                continue
            stage = self._canonical_stage(seg.get("stage"))
            if stage != "Rolling":
                non_rolling.append(seg)
                continue

            s = self._safe_float(seg.get("start_min"), None)
            e = self._safe_float(seg.get("end_min"), None)
            if s is None or e is None or e <= s:
                non_rolling.append(seg)
                continue

            unit = self._safe_str(seg.get("unit") or seg.get("rolling_unit") or "Rolling", "Rolling")
            key = (unit, round(float(s), 6), round(float(e), 6))
            groups.setdefault(key, []).append((idx, seg))

        repaired: List[Dict[str, Any]] = list(non_rolling)
        num_groups = 0
        num_repaired_segments = 0

        for (unit, s, e), indexed_group in groups.items():
            if len(indexed_group) <= 1:
                repaired.extend(seg for _, seg in indexed_group)
                continue

            indexed_group = sorted(indexed_group, key=lambda x: x[0])
            n = len(indexed_group)
            slot = (float(e) - float(s)) / max(n, 1)
            if slot <= 0:
                repaired.extend(seg for _, seg in indexed_group)
                continue

            num_groups += 1
            for local_idx, (_, seg) in enumerate(indexed_group):
                ns = float(s) + local_idx * slot
                ne = float(s) + (local_idx + 1) * slot
                new_seg = dict(seg)
                new_seg["raw_start_min_before_rolling_split"] = seg.get("start_min")
                new_seg["raw_end_min_before_rolling_split"] = seg.get("end_min")
                new_seg["raw_start_time_before_rolling_split"] = seg.get("start_time")
                new_seg["raw_end_time_before_rolling_split"] = seg.get("end_time")
                new_seg["start_min"] = float(ns)
                new_seg["end_min"] = float(ne)
                new_seg["duration_min"] = float(ne - ns)
                new_seg["start_time"] = self._min_to_time_string(ns) or seg.get("start_time")
                new_seg["end_time"] = self._min_to_time_string(ne) or seg.get("end_time")
                new_seg["unit"] = unit
                new_seg["rolling_split_repaired"] = True
                new_seg["rolling_split_group_size"] = n
                new_seg["rolling_split_group_index"] = local_idx + 1
                new_seg["repair_note"] = "split_shared_rolling_unit_window_to_heat_level_segment"
                reason = self._safe_str(new_seg.get("reason"), "")
                if "split_shared_rolling_unit_window_to_heat_level_segment" not in reason:
                    new_seg["reason"] = (reason + " | " if reason else "") + "Rolling unit window split to heat-level slot."
                repaired.append(new_seg)
                num_repaired_segments += 1

        repaired.sort(key=lambda x: (
            self._safe_float(x.get("start_min"), 10**12),
            self._safe_str(x.get("unit")),
            self._safe_str(x.get("heat_id")),
            self._safe_str(x.get("stage")),
        ))
        for i, seg in enumerate(repaired, start=1):
            if seg.get("source") == "model_output" and seg.get("is_valid_gantt_prediction", False):
                seg["segment_id"] = f"pred_seg_{i:03d}"
        return repaired, {
            "num_rolling_split_groups": num_groups,
            "num_rolling_split_repaired_segments": num_repaired_segments,
        }

    def _operation_segments_from_future_states(self, future_states: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for p in self._safe_list(future_states):
            if not isinstance(p, dict):
                continue
            step = int(p.get("step") or 0)
            t0, t1 = p.get("t0"), p.get("t1")
            p0, p1 = self._time_to_min_for_segment(t0), self._time_to_min_for_segment(t1)
            if p0 is None or p1 is None or p1 <= p0:
                continue
            for h in self._safe_list(p.get("active_heats")):
                if not isinstance(h, dict):
                    continue
                hid = self._normalize_heat_id(h.get("id") or h.get("heat_id") or h.get("hn"))
                if not hid:
                    continue
                stage = self._canonical_stage(h.get("stage"))
                if stage not in {"BOF", "LF", "Casting", "Rolling"}:
                    continue
                unit = h.get("ru") or p.get("converter") or stage
                rows.append({
                    "segment_id": f"derived_patch_{hid}_{stage}_{step}",
                    "heat_id": hid,
                    "stage": stage,
                    "unit": str(unit) if unit is not None else stage,
                    "start_time": self._min_to_time_string(p0) or t0,
                    "end_time": self._min_to_time_string(p1) or t1,
                    "start_min": float(p0),
                    "end_min": float(p1),
                    "duration_min": float(p1 - p0),
                    "raw_start_time": t0,
                    "raw_end_time": t1,
                    "source_steps": [step] if step else [],
                    "confidence": self._safe_float(p.get("confidence", 0.0), 0.0),
                    "source": "derived_from_future_production_states",
                    "source_kind": "derived_patch_state",
                    "is_valid_gantt_prediction": False,
                })

        rows.sort(key=lambda x: (x.get("heat_id") or "", x.get("stage") or "", x.get("unit") or "", x.get("start_min", 10**12)))
        merged: List[Dict[str, Any]] = []
        for seg in rows:
            if not merged:
                merged.append(seg)
                continue
            last = merged[-1]
            same_key = last.get("heat_id") == seg.get("heat_id") and last.get("stage") == seg.get("stage") and last.get("unit") == seg.get("unit")
            if same_key and seg.get("start_min", 10**12) <= last.get("end_min", -10**12) + 1e-6:
                last["end_min"] = max(last.get("end_min", 0.0), seg.get("end_min", 0.0))
                last["end_time"] = self._min_to_time_string(last["end_min"]) or seg.get("end_time")
                last["duration_min"] = max(0.0, last["end_min"] - last.get("start_min", 0.0))
                last.setdefault("source_steps", [])
                last["source_steps"].extend(seg.get("source_steps", []))
                last["confidence"] = max(self._safe_float(last.get("confidence"), 0.0), self._safe_float(seg.get("confidence"), 0.0))
            else:
                merged.append(seg)
        for i, seg in enumerate(merged, start=1):
            seg["segment_id"] = f"derived_patch_seg_{i:03d}"
        merged.sort(key=lambda x: (x.get("start_min", 10**12), x.get("unit") or "", x.get("heat_id") or ""))
        return merged

    def _interval_iou(self, a0: float, a1: float, b0: float, b1: float) -> float:
        inter = max(0.0, min(a1, b1) - max(a0, b0))
        union = max(a1, b1) - min(a0, b0)
        return inter / union if union > 1e-9 else 0.0

    def _evaluate_operation_segments(self, pred_segments: Any, target_segments: Any) -> Dict[str, Any]:

        pred = self._normalize_operation_segments(pred_segments, require_model_source=True)
        tgt = self._normalize_operation_segments(target_segments, require_model_source=False)

        def _finish(
            tp: int,
            fp: int,
            fn: int,
            precision: float,
            recall: float,
            mean_iou: float,
            start_mae: Optional[float],
            end_mae: Optional[float],
            duration_mae: Optional[float],
            match_pairs: Optional[List[Dict[str, Any]]] = None,
        ) -> Dict[str, Any]:
            f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 1e-12 else 0.0
            pred_n = len(pred)
            tgt_n = len(tgt)
            return {
                "num_target_operation_segments": tgt_n,
                "num_predicted_operation_segments": pred_n,
                "operation_segment_evaluation_eligible": 1 if pred and tgt else 0,
                "operation_segment_precision": float(precision),
                "operation_segment_recall": float(recall),
                "operation_segment_f1": float(f1),
                "operation_segment_mean_iou": float(mean_iou),
                "operation_start_mae": start_mae,
                "operation_end_mae": end_mae,
                "operation_duration_mae": duration_mae,
                "num_operation_segment_matches": int(tp),
                "operation_segment_true_positive_count": int(tp),
                "operation_segment_false_positive_count": int(fp),
                "operation_segment_false_negative_count": int(fn),
                "unmatched_predicted_segment_rate": (fp / pred_n) if pred_n > 0 else 0.0,
                "unmatched_target_segment_rate": (fn / tgt_n) if tgt_n > 0 else 0.0,
                "operation_segment_match_pairs": match_pairs or [],
            }

        if not tgt:
            tp = 0
            fp = len(pred)
            fn = 0
            precision = 1.0 if not pred else 0.0
            recall = 1.0
            mean_iou = 1.0 if not pred else 0.0
            return _finish(tp, fp, fn, precision, recall, mean_iou, None, None, None, [])

        if not pred:
            tp = 0
            fp = 0
            fn = len(tgt)
            return _finish(tp, fp, fn, 0.0, 0.0, 0.0, None, None, None, [])

        candidate_pairs: List[Tuple[float, int, int]] = []
        for pi, p in enumerate(pred):
            for ti, t in enumerate(tgt):
                if p.get("heat_id") != t.get("heat_id"):
                    continue
                if p.get("stage") != t.get("stage"):
                    continue
                iou = self._interval_iou(p["start_min"], p["end_min"], t["start_min"], t["end_min"])
                if iou >= 0.1:
                    candidate_pairs.append((iou, pi, ti))

        candidate_pairs.sort(reverse=True, key=lambda x: x[0])

        matched_pred = set()
        matched_tgt = set()
        matches: List[Tuple[int, int, float]] = []

        for iou, pi, ti in candidate_pairs:
            if pi in matched_pred or ti in matched_tgt:
                continue
            matched_pred.add(pi)
            matched_tgt.add(ti)
            matches.append((pi, ti, iou))

        iou_by_target = {ti: 0.0 for ti in range(len(tgt))}
        start_errs: List[float] = []
        end_errs: List[float] = []
        dur_errs: List[float] = []
        match_pairs: List[Dict[str, Any]] = []

        for pi, ti, iou in matches:
            p = pred[pi]
            t = tgt[ti]
            iou_by_target[ti] = iou
            start_abs = abs(p["start_min"] - t["start_min"])
            end_abs = abs(p["end_min"] - t["end_min"])
            dur_abs = abs((p["end_min"] - p["start_min"]) - (t["end_min"] - t["start_min"]))
            start_errs.append(start_abs)
            end_errs.append(end_abs)
            dur_errs.append(dur_abs)
            match_pairs.append({
                "pred_segment_id": p.get("segment_id"),
                "target_segment_id": t.get("segment_id"),
                "heat_id": t.get("heat_id"),
                "stage": t.get("stage"),
                "iou": iou,
                "start_abs_error_min": start_abs,
                "end_abs_error_min": end_abs,
                "duration_abs_error_min": dur_abs,
            })

        tp = len(matches)
        fp = max(0, len(pred) - tp)
        fn = max(0, len(tgt) - tp)
        precision = tp / max(len(pred), 1)
        recall = tp / max(len(tgt), 1)
        mean_iou = sum(iou_by_target.values()) / max(len(tgt), 1)

        return _finish(
            tp=tp,
            fp=fp,
            fn=fn,
            precision=precision,
            recall=recall,
            mean_iou=mean_iou,
            start_mae=sum(start_errs) / len(start_errs) if start_errs else None,
            end_mae=sum(end_errs) / len(end_errs) if end_errs else None,
            duration_mae=sum(dur_errs) / len(dur_errs) if dur_errs else None,
            match_pairs=match_pairs,
        )

    def _operation_segment_aliases(self, raw_output: Dict[str, Any]) -> Tuple[Optional[str], Any]:

        for key in [
            "predicted_operation_segments",
            "operation_segments",
            "production_operation_segments",
            "operation_schedule",
            "schedule_segments",
        ]:
            val = raw_output.get(key)
            if isinstance(val, list):
                return key, val
        return None, []

    def _derive_future_states_from_operation_segments(
        self,
        segments: List[Dict[str, Any]],
        sample: Optional[Dict[str, Any]] = None,
        calendar_by_step: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:

        sample = self._safe_dict(sample)
        if calendar_by_step is None:
            calendar_by_step = self._future_calendar_by_step(sample)

        out: List[Dict[str, Any]] = []
        for step in sorted(calendar_by_step):
            patch = self._safe_dict(calendar_by_step.get(step))
            p0 = self._time_to_min_for_segment(patch.get("t0_min") if patch.get("t0_min") is not None else patch.get("t0"))
            p1 = self._time_to_min_for_segment(patch.get("t1_min") if patch.get("t1_min") is not None else patch.get("t1"))
            active_heats: List[Dict[str, Any]] = []
            active_rolling_units: List[Any] = []

            if p0 is not None and p1 is not None and p1 > p0:
                seen = set()
                for seg in self._safe_list(segments):
                    if not isinstance(seg, dict):
                        continue
                    s0 = self._safe_float(seg.get("start_min"), None)
                    s1 = self._safe_float(seg.get("end_min"), None)
                    if s0 is None or s1 is None or max(s0, p0) >= min(s1, p1):
                        continue
                    hid = self._normalize_heat_id(seg.get("heat_id"))
                    stage = self._canonical_stage(seg.get("stage"))
                    unit = seg.get("unit")
                    key = (hid, stage, unit)
                    if not hid or stage == "Other" or key in seen:
                        continue
                    seen.add(key)
                    active_heats.append({
                        "id": hid,
                        "stage": stage,
                        "stages": [stage],
                        "ru": unit if stage == "Rolling" else None,
                    })
                    if stage == "Rolling" and unit is not None and unit not in active_rolling_units:
                        active_rolling_units.append(unit)

            out.append({
                "step": step,
                "pi": patch.get("pi"),
                "pid": patch.get("pid"),
                "t0": patch.get("t0"),
                "t1": patch.get("t1"),
                "canonical_heat_id": self._normalize_heat_id(patch.get("canonical_heat_id") or patch.get("pid")),
                "converter": patch.get("converter"),
                "active_heat_ids": [h["id"] for h in active_heats],
                "active_heats": active_heats,
                "active_rolling_units": active_rolling_units,
                "stage_counts": self._recompute_stage_counts(active_heats),
                "confidence": 0.0,
                "comment": "derived_from_predicted_operation_segments_for_downstream_compatibility",
                "source": "derived_from_predicted_operation_segments",
            })
        return out

    def _apply_physics_consistent_projection(
        self,
        normalized_segments: List[Dict[str, Any]],
        sample: Dict[str, Any],
    ) -> Dict[str, Any]:

        input_state = self._safe_dict(sample.get("input_state"))
        repair_context = self._safe_dict(input_state.get("repair_context"))
        if not normalized_segments:
            return {"projected_segments": [], "uncertain_segments": [], "rejected_segments": [], "diagnostics": {"method": "LLM-guided Physics-consistent Variational Projection", "method_short": "PCVP", "projection_applied": False, "skip_reason": "no_normalized_model_segments", "uses_future_target_labels": False}}
        if self.physics_projector is None:
            return {"projected_segments": normalized_segments, "uncertain_segments": [], "rejected_segments": [], "diagnostics": {"method": "LLM-guided Physics-consistent Variational Projection", "method_short": "PCVP", "projection_applied": False, "skip_reason": "mechanism_schedule_projection_module_not_available", "uses_future_target_labels": False}}
        try:
            result = self.physics_projector.project(raw_segments=normalized_segments, repair_context=repair_context)
            return {
                "projected_segments": self._safe_list(result.get("repaired_segments")),
                "uncertain_segments": self._safe_list(result.get("uncertain_segments")),
                "rejected_segments": self._safe_list(result.get("rejected_segments")),
                "diagnostics": self._safe_dict(result.get("repair_diagnostics")),
            }
        except Exception as e:
            return {"projected_segments": normalized_segments, "uncertain_segments": [], "rejected_segments": [], "diagnostics": {"method": "LLM-guided Physics-consistent Variational Projection", "method_short": "PCVP", "projection_applied": False, "projection_failed": True, "error_type": type(e).__name__, "error_message": str(e), "uses_future_target_labels": False}}

    def normalize_horizon_output(self, raw_output: Dict[str, Any], sample: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:

        raw_output = self._safe_dict(raw_output)
        sample = self._safe_dict(sample)

        input_state = self._safe_dict(sample.get("input_state"))
        schedule_task = self._safe_dict(input_state.get("operation_schedule_task"))
        future_window = self._safe_dict(input_state.get("known_future_time_window")) or self._safe_dict(schedule_task.get("future_time_window"))
        calendar_by_step = self._future_calendar_by_step(sample)

        raw_segments_key, raw_segments = self._operation_segment_aliases(raw_output)
        raw_segment_candidates = self._safe_list(raw_segments)
        normalized_segments_before_projection = self._normalize_operation_segments(
            raw_segment_candidates,
            future_window=future_window,
            require_model_source=True,
        )
        rolling_repaired_segments, rolling_repair_meta = self._repair_rolling_heat_level_segments(
            normalized_segments_before_projection,
            sample=sample,
        )

        projection_result = self._apply_physics_consistent_projection(rolling_repaired_segments, sample=sample)
        projected_segments = self._safe_list(projection_result.get("projected_segments"))
        uncertain_segments = self._safe_list(projection_result.get("uncertain_segments"))
        rejected_segments = self._safe_list(projection_result.get("rejected_segments"))
        projection_diagnostics = self._safe_dict(projection_result.get("diagnostics"))
        normalized_segments = projected_segments

        if normalized_segments:
            future_states = self._derive_future_states_from_operation_segments(
                normalized_segments,
                sample=sample,
                calendar_by_step=calendar_by_step,
            )
            segment_source = "model_output_pcvp_projected"
            predicted_operation_segments = normalized_segments
            operation_segment_evaluation_eligible = 1
        else:
            future_states = []
            for step in sorted(calendar_by_step):
                patch = self._safe_dict(calendar_by_step.get(step))
                future_states.append({
                    "step": step,
                    "pi": patch.get("pi"),
                    "pid": patch.get("pid"),
                    "t0": patch.get("t0"),
                    "t1": patch.get("t1"),
                    "canonical_heat_id": self._normalize_heat_id(patch.get("canonical_heat_id") or patch.get("pid")),
                    "converter": patch.get("converter"),
                    "active_heat_ids": [],
                    "active_heats": [],
                    "active_rolling_units": [],
                    "stage_counts": {"BOF": 0, "LF": 0, "Casting": 0, "Rolling": 0},
                    "confidence": 0.0,
                    "comment": "missing_model_operation_segments_empty_placeholder",
                    "source": "empty_placeholder_not_model_prediction",
                })
            segment_source = "missing_model_operation_segments"
            predicted_operation_segments = []
            operation_segment_evaluation_eligible = 0

        invalid_or_rejected_raw_segments = (
            max(0, len(raw_segment_candidates) - len(normalized_segments_before_projection))
            + len(rejected_segments)
        )

        return {
            "task": raw_output.get("task", "production_gantt_horizon_prediction"),
            "mode": raw_output.get("mode", "production_operation_schedule_world_model_gantt_only"),
            "future_production_states": future_states,
            "future_production_states_source": "derived_from_predicted_operation_segments" if normalized_segments else "empty_placeholder",
            "predicted_operation_segments": predicted_operation_segments,
            "raw_predicted_operation_segments_before_projection": normalized_segments_before_projection,
            "rolling_repaired_operation_segments_before_projection": rolling_repaired_segments,
            "uncertain_operation_segments": uncertain_segments,
            "rejected_operation_segments": rejected_segments,
            "mechanism_projection_diagnostics": projection_diagnostics,
            "derived_operation_segments_from_patches": [],
            "fallback_operation_segments": [],
            "operation_segment_source": segment_source,
            "operation_segment_evaluation_eligible": operation_segment_evaluation_eligible,
            "operation_segment_prediction_available": bool(predicted_operation_segments),
            "operation_segment_missing": not bool(predicted_operation_segments),
            "raw_operation_segment_key": raw_segments_key,
            "raw_output": raw_output,
            "repair_meta": {
                "num_calendar_steps": len(calendar_by_step),
                "num_repaired_steps": 0,
                "num_raw_operation_segment_candidates": len(raw_segment_candidates),
                "num_normalized_model_segments_before_projection": len(normalized_segments_before_projection),
                "num_rolling_repaired_segments_before_projection": len(rolling_repaired_segments),
                "num_rejected_or_invalid_raw_operation_segments": invalid_or_rejected_raw_segments,
                "num_predicted_operation_segments": len(predicted_operation_segments),
                "num_uncertain_operation_segments": len(uncertain_segments),
                "num_rejected_operation_segments_by_projection": len(rejected_segments),
                "num_derived_operation_segments": 0,
                "num_rolling_split_groups": self._safe_dict(rolling_repair_meta).get("num_rolling_split_groups", 0),
                "num_rolling_split_repaired_segments": self._safe_dict(rolling_repair_meta).get("num_rolling_split_repaired_segments", 0),
                "physics_consistent_projection_applied": bool(projection_diagnostics.get("num_repaired_segments") is not None),
                "projection_method": projection_diagnostics.get("method", "LLM-guided Physics-consistent Variational Projection"),
                "projection_uses_future_target_labels": bool(projection_diagnostics.get("uses_future_target_labels", False)),
                "projection_violations_before": projection_diagnostics.get("violations_before", {}),
                "projection_violations_after": projection_diagnostics.get("violations_after", {}),
                "operation_segments_used_for_metrics": segment_source.startswith("model_output"),
                "gantt_prediction_protocol": "Only final PCVP-projected predicted_operation_segments with source=model_output are valid model Gantt predictions; raw/uncertain/rejected segments are audit only.",
                "task_reduction_note": "Production prompt removed patch-level heat-alignment prediction and kept only operation-level Gantt prediction followed by physics-consistent variational projection.",
            },
        }

    def _hard_violation_count_from_projection(self, diagnostics: Dict[str, Any]) -> int:
        diag = self._safe_dict(diagnostics)
        after = self._safe_dict(diag.get("violations_after"))
        hard_keys = [
            "outside_horizon",
            "non_positive_duration",
            "stage_order",
            "unit_overlap",
            "duplicate_heat_stage",
            "rolling_shared_window",
            "incompatible_unit",
        ]
        total = 0
        for k in hard_keys:
            total += int(self._safe_float(after.get(k), 0.0))
        return total

    def _summarize_horizon_attempt(
        self,
        raw_output: Dict[str, Any],
        prediction: Dict[str, Any],
        attempt_index: int,
        retry_reasons: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        raw_output = self._safe_dict(raw_output)
        prediction = self._safe_dict(prediction)
        repair_meta = self._safe_dict(prediction.get("repair_meta"))
        projection_diag = self._safe_dict(prediction.get("mechanism_projection_diagnostics"))
        raw_key = prediction.get("raw_operation_segment_key")
        return {
            "attempt_index": int(attempt_index),
            "was_retry": attempt_index > 0,
            "retry_reasons": retry_reasons or [],
            "raw_warning": raw_output.get("warning"),
            "raw_mode": raw_output.get("mode"),
            "raw_segment_key": raw_key,
            "num_raw_operation_segment_candidates": repair_meta.get("num_raw_operation_segment_candidates", 0),
            "num_normalized_model_segments_before_projection": repair_meta.get("num_normalized_model_segments_before_projection", 0),
            "num_projected_predicted_segments": len(self._safe_list(prediction.get("predicted_operation_segments"))),
            "operation_segment_source": prediction.get("operation_segment_source"),
            "operation_segment_prediction_available": bool(prediction.get("operation_segment_prediction_available")),
            "operation_segment_missing": bool(prediction.get("operation_segment_missing")),
            "projection_hard_violations_after": self._hard_violation_count_from_projection(projection_diag),
            "projection_uses_future_target_labels": bool(repair_meta.get("projection_uses_future_target_labels", False)),
        }

    def _horizon_retry_reasons(
        self,
        prediction: Dict[str, Any],
        raw_output: Dict[str, Any],
        attempt_index: int,
    ) -> List[str]:

        reasons: List[str] = []
        pred = self._safe_dict(prediction)
        raw = self._safe_dict(raw_output)
        repair_meta = self._safe_dict(pred.get("repair_meta"))
        projection_diag = self._safe_dict(pred.get("mechanism_projection_diagnostics"))

        if raw.get("warning"):
            reasons.append(str(raw.get("warning")))

        raw_segments = repair_meta.get("num_raw_operation_segment_candidates", 0) or 0
        normalized = repair_meta.get("num_normalized_model_segments_before_projection", 0) or 0
        projected = len(self._safe_list(pred.get("predicted_operation_segments")))

        if raw_segments == 0:
            reasons.append("missing_predicted_operation_segments")
        if raw_segments > 0 and normalized == 0:
            reasons.append("all_raw_segments_failed_schema_or_source_validation")
        if normalized > 0 and projected == 0:
            reasons.append("pcvp_projected_no_valid_segments")
        if pred.get("operation_segment_missing") or not pred.get("operation_segment_prediction_available"):
            reasons.append("operation_segment_prediction_missing_after_normalization")

        hard_after = self._hard_violation_count_from_projection(projection_diag)
        if projected > 0 and hard_after > 0:

            reasons.append(f"remaining_hard_mechanism_violations_after_pcvp:{hard_after}")

        out: List[str] = []
        for r in reasons:
            if r and r not in out:
                out.append(r)

        if attempt_index >= self.max_horizon_retries:
            return []
        return out

    def _build_retry_horizon_input(
        self,
        original_model_input: Dict[str, Any],
        sample: Dict[str, Any],
        previous_raw_output: Dict[str, Any],
        previous_prediction: Dict[str, Any],
        attempt_index: int,
        retry_reasons: List[str],
    ) -> Dict[str, Any]:

        base = json.loads(json.dumps(original_model_input, ensure_ascii=False, default=str))
        input_block = self._safe_dict(base.get("input"))
        retry_guidance = self._safe_dict(input_block.get("retry_guidance_context"))
        repair_meta = self._safe_dict(previous_prediction.get("repair_meta"))
        projection_diag = self._safe_dict(previous_prediction.get("mechanism_projection_diagnostics"))

        base["task_instruction"] = (
            str(base.get("task_instruction", "")) +
            " This is a retry/self-refinement attempt. The previous response did not produce a valid "
            "operation-level Gantt prediction. Regenerate a complete JSON object with predicted_operation_segments "
            "using only the provided schedule context, route skeletons, local blocks, duration priors, and PCVP diagnostics."
        )
        base["retry_request"] = {
            "enabled": True,
            "attempt_index": int(attempt_index),
            "max_retry_attempts": self.max_horizon_retries,
            "retry_reasons": retry_reasons,
            "strict_no_target_labels": True,
            "previous_attempt_summary": self._summarize_horizon_attempt(
                previous_raw_output,
                previous_prediction,
                attempt_index=attempt_index - 1,
                retry_reasons=retry_reasons,
            ),
            "pcvp_diagnostics_available": bool(projection_diag),
            "pcvp_violations_before": projection_diag.get("violations_before", {}),
            "pcvp_violations_after": projection_diag.get("violations_after", {}),
            "num_uncertain_segments": repair_meta.get("num_uncertain_operation_segments", 0),
            "num_rejected_segments": repair_meta.get("num_rejected_operation_segments_by_projection", 0),
            "allowed_feedback": retry_guidance.get("allowed_feedback_to_llm", [
                "schema error",
                "missing required key",
                "PCVP violation diagnostics",
                "future horizon window",
                "known future patch calendar",
                "history operation segments",
                "duration priors",
            ]),
            "forbidden_feedback": retry_guidance.get("forbidden_feedback_to_llm", [
                "target_operation_segments",
                "ground-truth future operation start/end times",
                "IoU/MAE/F1 against target labels",
            ]),
            "required_correction": [
                "Return exactly one JSON object.",
                "Include key predicted_operation_segments.",
                "Each valid segment must have source exactly model_output.",
                "Use heat_route_skeletons and local_schedule_blocks to cover long horizons.",
                "Do not output future_production_states or active_heat_alignment_labels.",
            ],
        }
        base["required_reasoning_contract"]["retry_self_refinement"] = [
            "Repair only the reported schema/missing-output/mechanism-consistency problems.",
            "Do not use target labels; they are not provided.",
            "Do not return an empty predicted_operation_segments list if calendar/carryover/context support at least one operation segment.",
            "If a stage cannot be reliably inferred, omit that stage but still output supported stages.",
        ]
        return base

    def _duration_prior_for_fallback(self, sample: Dict[str, Any], stage: str, unit: Optional[str] = None) -> float:

        input_state = self._safe_dict(sample.get("input_state"))
        op_task = self._safe_dict(input_state.get("operation_schedule_task"))
        priors = self._safe_dict(op_task.get("duration_priors"))
        stage = self._canonical_stage(stage)
        unit = self._safe_str(unit, "")

        stage_unit_stats = self._safe_dict(priors.get("stage_unit_duration_stats_min"))
        if unit:
            for key in [f"{stage}|{unit}", f"{stage}|{unit.replace('Converter', '')}"]:
                stats = self._safe_dict(stage_unit_stats.get(key))
                med = self._safe_float(stats.get("median"), 0.0)
                mean = self._safe_float(stats.get("mean"), 0.0)
                if med > 0:
                    return med
                if mean > 0:
                    return mean

        stage_stats = self._safe_dict(self._safe_dict(priors.get("stage_duration_stats_min")).get(stage))
        med = self._safe_float(stage_stats.get("median"), 0.0)
        mean = self._safe_float(stage_stats.get("mean"), 0.0)
        if med > 0:
            return med
        if mean > 0:
            return mean

        defaults = {"BOF": 15.0, "LF": 18.0, "Casting": 5.0, "Rolling": 15.0}
        return defaults.get(stage, 5.0)

    def _fallback_future_window(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        input_state = self._safe_dict(sample.get("input_state"))
        op_task = self._safe_dict(input_state.get("operation_schedule_task"))
        return self._safe_dict(input_state.get("known_future_time_window")) or self._safe_dict(op_task.get("future_time_window"))

    def _fallback_calendar_rows(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        input_state = self._safe_dict(sample.get("input_state"))
        op_task = self._safe_dict(input_state.get("operation_schedule_task"))
        rows = self._safe_list(input_state.get("known_future_patch_calendar"))
        if not rows:
            rows = self._safe_list(op_task.get("future_calendar"))
        return [x for x in rows if isinstance(x, dict)]

    def _fallback_heat_route_skeletons(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        input_state = self._safe_dict(sample.get("input_state"))
        op_task = self._safe_dict(input_state.get("operation_schedule_task"))
        retry_ctx = self._safe_dict(input_state.get("retry_guidance_context"))
        skeletons = self._safe_list(input_state.get("heat_route_skeletons"))
        if not skeletons:
            skeletons = self._safe_list(op_task.get("heat_route_skeletons"))
        if not skeletons:
            skeletons = self._safe_list(retry_ctx.get("heat_route_skeletons"))
        return [x for x in skeletons if isinstance(x, dict)]

    def _fallback_candidate_heat_ids(self, sample: Dict[str, Any]) -> List[str]:
        input_state = self._safe_dict(sample.get("input_state"))
        op_task = self._safe_dict(input_state.get("operation_schedule_task"))
        ids: List[str] = []
        for x in self._safe_list(op_task.get("candidate_future_heat_ids")):
            hid = self._normalize_heat_id(x)
            if hid and hid not in ids:
                ids.append(hid)
        for row in self._fallback_calendar_rows(sample):
            hid = self._normalize_heat_id(row.get("canonical_heat_id") or row.get("heat_id") or row.get("hn") or row.get("pid"))
            if hid and hid not in ids:
                ids.append(hid)
        for sk in self._fallback_heat_route_skeletons(sample):
            hid = self._normalize_heat_id(sk.get("heat_id"))
            if hid and hid not in ids:
                ids.append(hid)
        return ids

    def _converter_unit_for_fallback(self, converter: Any) -> str:
        s = self._safe_str(converter, "")
        if not s:
            return "BOF"

        m = re.search(r"Converter(\d+)", s)
        if m:
            return f"{m.group(1)}#LDG Generation"
        return s

    def _make_rule_assisted_segment(
        self,
        heat_id: str,
        stage: str,
        unit: str,
        start_min: float,
        end_min: float,
        source_steps: Optional[List[Any]] = None,
        reason: str = "retry_exhausted_rule_assisted_completion",
        confidence: float = 0.45,
    ) -> Dict[str, Any]:
        return {
            "heat_id": self._normalize_heat_id(heat_id),
            "stage": self._canonical_stage(stage),
            "unit": unit,
            "start_time": self._min_to_time_string(start_min),
            "end_time": self._min_to_time_string(end_min),
            "start_min": float(start_min),
            "end_min": float(end_min),
            "duration_min": max(0.0, float(end_min) - float(start_min)),
            "source_steps": self._safe_list(source_steps),
            "source": "model_output_rule_assisted_after_retry",
            "confidence": float(max(0.0, min(1.0, confidence))),
            "reason": reason,
            "rule_assisted_after_retry_exhausted": True,
            "uses_future_target_labels": False,
        }

    def _build_retry_exhausted_rule_fallback_raw_output(
        self,
        sample: Dict[str, Any],
        attempt_records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:

        fw = self._fallback_future_window(sample)
        fw0, fw1 = self._future_window_bounds(fw)
        if fw0 is None or fw1 is None or fw1 <= fw0:
            fw0, fw1 = 0.0, 1.0

        segments: List[Dict[str, Any]] = []
        seen = set()

        def add_segment(seg: Dict[str, Any]) -> None:
            if not seg or not seg.get("heat_id"):
                return
            if seg.get("end_min", 0.0) <= seg.get("start_min", 0.0):
                return
            key = (
                seg.get("heat_id"), seg.get("stage"), seg.get("unit"),
                round(float(seg.get("start_min", 0.0)), 6), round(float(seg.get("end_min", 0.0)), 6),
            )
            if key in seen:
                return
            seen.add(key)
            segments.append(seg)

        calendar_by_heat: Dict[str, List[Dict[str, Any]]] = {}
        for row in self._fallback_calendar_rows(sample):
            hid = self._normalize_heat_id(row.get("canonical_heat_id") or row.get("heat_id") or row.get("hn") or row.get("pid"))
            if hid:
                calendar_by_heat.setdefault(hid, []).append(row)
                s = self._safe_float(row.get("t0_min"), None) if row.get("t0_min") is not None else self._time_to_min_for_segment(row.get("t0") or row.get("ts"))
                e = self._safe_float(row.get("t1_min"), None) if row.get("t1_min") is not None else self._time_to_min_for_segment(row.get("t1") or row.get("te"))
                if s is None:
                    s = fw0
                if e is None or e <= s:
                    e = s + self._duration_prior_for_fallback(sample, "BOF", row.get("converter"))
                s = max(float(s), float(fw0))
                e = min(float(e), float(fw1))
                if e > s:
                    add_segment(self._make_rule_assisted_segment(
                        heat_id=hid,
                        stage="BOF",
                        unit=self._converter_unit_for_fallback(row.get("converter")),
                        start_min=s,
                        end_min=e,
                        source_steps=[row.get("step")] if row.get("step") is not None else [],
                        reason="calendar_supported_bof_after_retry_exhausted",
                        confidence=0.62,
                    ))

        by_heat_end: Dict[str, float] = {}
        by_heat_unit: Dict[str, str] = {}
        for seg in segments:
            hid = seg.get("heat_id")
            by_heat_end[hid] = max(by_heat_end.get(hid, fw0), float(seg.get("end_min", fw0)))
            by_heat_unit[hid] = seg.get("unit") or by_heat_unit.get(hid, "")

        skeletons = self._fallback_heat_route_skeletons(sample)
        candidate_ids = set(self._fallback_candidate_heat_ids(sample))
        for sk in skeletons:
            hid = self._normalize_heat_id(sk.get("heat_id"))
            if not hid or (candidate_ids and hid not in candidate_ids):
                continue
            stages = [self._canonical_stage(x) for x in self._safe_list(sk.get("candidate_stages_in_horizon"))]
            stages = [x for x in stages if x in {"BOF", "LF", "Casting", "Rolling"}]
            if not stages and sk.get("carryover_at_anchor"):
                st = self._canonical_stage(sk.get("anchor_stage"))
                if st in {"BOF", "LF", "Casting", "Rolling"}:
                    stages = [st]
            if not stages:
                continue

            last_end = self._safe_float(sk.get("last_known_end_min"), None)
            current_t = max(fw0, last_end if last_end is not None else by_heat_end.get(hid, fw0))
            if hid in by_heat_end:
                current_t = max(current_t, by_heat_end[hid])

            for stage in stages:

                if stage == "BOF" and hid in calendar_by_heat:
                    continue

                if stage == "LF" and self._duration_prior_for_fallback(sample, "LF") <= 0.5:
                    continue
                dur = self._duration_prior_for_fallback(sample, stage)
                if dur <= 0.5:
                    continue
                start = max(float(current_t), float(fw0))
                end = min(float(start + dur), float(fw1))
                if end <= start:
                    continue
                if stage == "BOF":
                    unit = self._converter_unit_for_fallback((calendar_by_heat.get(hid) or [{}])[0].get("converter"))
                elif stage == "Casting":
                    unit = "Casting"
                elif stage == "Rolling":
                    unit = self._safe_str(sk.get("last_known_unit"), "Rolling") or "Rolling"
                else:
                    unit = stage
                add_segment(self._make_rule_assisted_segment(
                    heat_id=hid,
                    stage=stage,
                    unit=unit,
                    start_min=start,
                    end_min=end,
                    source_steps=sk.get("calendar_steps") or [],
                    reason="heat_route_skeleton_prior_after_retry_exhausted",
                    confidence=0.40 if not sk.get("known_future_anchor") else 0.52,
                ))
                current_t = end

        segments.sort(key=lambda x: (x.get("start_min", 0.0), x.get("heat_id", ""), x.get("stage", "")))
        return {
            "task": "production_gantt_horizon_prediction",
            "mode": "production_operation_schedule_rule_assisted_after_retry_exhausted",
            "predicted_operation_segments": segments,
            "_rule_fallback_generated": True,
            "_rule_fallback_uses_future_target_labels": False,
            "_rule_fallback_num_raw_segments": len(segments),
            "_rule_fallback_reason": "all_llm_retry_attempts_missing_or_invalid",
            "_previous_attempts": attempt_records,
        }

    def _apply_retry_exhausted_rule_fallback(
        self,
        sample: Dict[str, Any],
        attempt_records: List[Dict[str, Any]],
        current_prediction: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        meta = {
            "rule_fallback_enabled": self.enable_retry_exhausted_rule_fallback,
            "rule_fallback_used": False,
            "rule_fallback_success": False,
            "rule_fallback_evaluate_as_prediction": self.evaluate_rule_fallback_as_prediction,
            "rule_fallback_uses_future_target_labels": False,
            "rule_fallback_num_raw_segments": 0,
            "rule_fallback_num_projected_segments": 0,
            "rule_fallback_source": "model_output_rule_assisted_after_retry" if self.evaluate_rule_fallback_as_prediction else "rule_assisted_after_retry_exhausted_diagnostic",
        }
        if not self.enable_retry_exhausted_rule_fallback:
            return current_prediction, {}, meta
        if current_prediction.get("operation_segment_prediction_available"):
            return current_prediction, {}, meta

        fallback_raw = self._build_retry_exhausted_rule_fallback_raw_output(sample, attempt_records)
        meta["rule_fallback_num_raw_segments"] = len(self._safe_list(fallback_raw.get("predicted_operation_segments")))
        if meta["rule_fallback_num_raw_segments"] <= 0:
            return current_prediction, fallback_raw, meta

        fallback_pred = self.normalize_horizon_output(fallback_raw, sample=sample)
        fallback_segments = self._safe_list(fallback_pred.get("predicted_operation_segments"))
        meta["rule_fallback_used"] = True
        meta["rule_fallback_success"] = bool(fallback_segments)
        meta["rule_fallback_num_projected_segments"] = len(fallback_segments)

        if fallback_segments:
            if self.evaluate_rule_fallback_as_prediction:
                fallback_pred["operation_segment_source"] = "model_output_rule_assisted_after_retry"
                fallback_pred["operation_segment_evaluation_eligible"] = 1
                fallback_pred["operation_segment_prediction_available"] = True
                fallback_pred["operation_segment_missing"] = False
            else:
                fallback_pred["operation_segment_source"] = "rule_assisted_after_retry_exhausted_diagnostic"
                fallback_pred["operation_segment_evaluation_eligible"] = 0
                fallback_pred["operation_segment_prediction_available"] = False
                fallback_pred["operation_segment_missing"] = True
            fallback_pred["rule_assisted_after_retry_exhausted"] = True
            fallback_pred["rule_assisted_fallback_operation_segments"] = fallback_segments
            fallback_pred.setdefault("repair_meta", {})
            fallback_pred["repair_meta"].update(meta)
            fallback_pred["repair_meta"].update({
                "operation_segments_used_for_metrics": bool(self.evaluate_rule_fallback_as_prediction),
                "rule_fallback_policy": "after all LLM retry attempts are exhausted; no target labels used",
            })
            return fallback_pred, fallback_raw, meta
        return current_prediction, fallback_raw, meta

    def predict_horizon(self, sample: Dict[str, Any]) -> Dict[str, Any]:

        model_input = self.build_horizon_input(sample)

        attempt_records: List[Dict[str, Any]] = []
        raw_output: Dict[str, Any] = {}
        pred: Dict[str, Any] = {}
        retry_reasons: List[str] = []

        current_input = model_input
        total_attempts = self.max_horizon_retries + 1

        for attempt_index in range(total_attempts):
            raw_output = self._call_horizon_model_once(current_input, attempt_index=attempt_index)
            pred = self.normalize_horizon_output(raw_output, sample=sample)

            retry_reasons = self._horizon_retry_reasons(
                prediction=pred,
                raw_output=raw_output,
                attempt_index=attempt_index,
            )
            attempt_records.append(
                self._summarize_horizon_attempt(
                    raw_output=raw_output,
                    prediction=pred,
                    attempt_index=attempt_index,
                    retry_reasons=retry_reasons,
                )
            )

            if not retry_reasons:
                break

            if attempt_index >= self.max_horizon_retries:
                break

            current_input = self._build_retry_horizon_input(
                original_model_input=model_input,
                sample=sample,
                previous_raw_output=raw_output,
                previous_prediction=pred,
                attempt_index=attempt_index + 1,
                retry_reasons=retry_reasons,
            )

        rule_fallback_meta: Dict[str, Any] = {
            "rule_fallback_enabled": self.enable_retry_exhausted_rule_fallback,
            "rule_fallback_used": False,
            "rule_fallback_success": False,
            "rule_fallback_evaluate_as_prediction": self.evaluate_rule_fallback_as_prediction,
            "rule_fallback_uses_future_target_labels": False,
        }
        if not pred.get("operation_segment_prediction_available"):
            pred, fallback_raw_output, rule_fallback_meta = self._apply_retry_exhausted_rule_fallback(
                sample=sample,
                attempt_records=attempt_records,
                current_prediction=pred,
            )
            if rule_fallback_meta.get("rule_fallback_used"):
                raw_output = fallback_raw_output
                attempt_records.append({
                    "attempt_index": "retry_exhausted_rule_fallback",
                    "was_rule_fallback": True,
                    "operation_segment_prediction_available": bool(pred.get("operation_segment_prediction_available")),
                    "num_projected_segments": len(self._safe_list(pred.get("predicted_operation_segments"))),
                    "uses_future_target_labels": False,
                    "source": pred.get("operation_segment_source"),
                })

        retry_meta = {
            "retry_enabled": self.max_horizon_retries > 0,
            "max_horizon_retries": self.max_horizon_retries,
            "num_attempts": len(attempt_records),
            "num_retries_used": max(0, len(attempt_records) - 1),
            "first_pass_missing": bool(attempt_records and attempt_records[0].get("operation_segment_missing")),
            "final_prediction_available": bool(pred.get("operation_segment_prediction_available")),
            "retry_success": bool(
                attempt_records
                and attempt_records[0].get("operation_segment_missing")
                and pred.get("operation_segment_prediction_available")
            ),
            "attempts": attempt_records,
            "uses_future_target_labels": False,
            "rule_fallback_enabled": bool(rule_fallback_meta.get("rule_fallback_enabled")),
            "rule_fallback_used": bool(rule_fallback_meta.get("rule_fallback_used")),
            "rule_fallback_success": bool(rule_fallback_meta.get("rule_fallback_success")),
            "rule_fallback_evaluate_as_prediction": bool(rule_fallback_meta.get("rule_fallback_evaluate_as_prediction")),
            "rule_fallback_uses_future_target_labels": bool(rule_fallback_meta.get("rule_fallback_uses_future_target_labels", False)),
            "rule_fallback_num_raw_segments": rule_fallback_meta.get("rule_fallback_num_raw_segments", 0),
            "rule_fallback_num_projected_segments": rule_fallback_meta.get("rule_fallback_num_projected_segments", 0),
            "policy": "LLM self-refinement first; after retry exhaustion, optional rule-assisted schedule completion can be applied using prediction-time context only. No target_future, IoU, MAE or F1 is used.",
        }

        pred["retry_meta"] = retry_meta
        pred.setdefault("repair_meta", {})
        pred["repair_meta"].update({
            "retry_enabled": retry_meta["retry_enabled"],
            "num_horizon_attempts": retry_meta["num_attempts"],
            "num_horizon_retries_used": retry_meta["num_retries_used"],
            "first_pass_missing": retry_meta["first_pass_missing"],
            "retry_success": retry_meta["retry_success"],
            "retry_uses_future_target_labels": False,
            "rule_fallback_enabled": retry_meta["rule_fallback_enabled"],
            "rule_fallback_used": retry_meta["rule_fallback_used"],
            "rule_fallback_success": retry_meta["rule_fallback_success"],
            "rule_fallback_evaluate_as_prediction": retry_meta["rule_fallback_evaluate_as_prediction"],
            "rule_fallback_uses_future_target_labels": False,
            "rule_fallback_num_raw_segments": retry_meta["rule_fallback_num_raw_segments"],
            "rule_fallback_num_projected_segments": retry_meta["rule_fallback_num_projected_segments"],
        })
        pred["sample_meta"] = {
            "day_id": sample.get("day_id"),
            "anchor_patch_idx": sample.get("anchor_patch_idx"),
            "history_window": sample.get("history_window"),
            "horizon": sample.get("horizon"),
            "source_meta": self._safe_dict(sample.get("meta")),
        }
        return pred

    def evaluate_horizon_prediction(self, prediction: Dict[str, Any], target_future: Dict[str, Any]) -> Dict[str, Any]:

        pred = self._safe_dict(prediction)
        tf = self._safe_dict(target_future)
        pred_segments = self._safe_list(pred.get("predicted_operation_segments"))
        target_segments = tf.get("operation_segments") or tf.get("production_operation_segments") or []

        metrics = {
            "num_steps": 0,
            "num_target_steps": len(self._safe_list(tf.get("future_patches"))),
            "num_predicted_steps": 0,
            "coverage": 0.0,
            "heat_jaccard": 0.0,
            "heat_exact_match_rate": 0.0,
            "stage_accuracy": 0.0,
            "rolling_unit_jaccard": 0.0,
            "patch_level_metrics_removed": 1.0,
            "production_task_type": "gantt_operation_segment_prediction_only",
        }
        metrics.update(self._evaluate_operation_segments(pred_segments, target_segments))
        metrics["operation_segment_source"] = pred.get("operation_segment_source", "unknown")
        metrics["operation_segment_prediction_available"] = 1.0 if pred_segments else 0.0
        metrics["operation_segment_missing"] = 0.0 if pred_segments else 1.0
        metrics["num_model_predicted_operation_segments"] = len(pred_segments)

        raw_pp = self._safe_dict(pred.get("raw_output")).get("predicted_operation_segments")
        metrics["num_raw_predicted_operation_segments"] = len(self._safe_list(raw_pp)) if raw_pp is not None else len(pred_segments)
        metrics["num_rejected_raw_operation_segments"] = self._safe_float(self._safe_dict(pred.get("repair_meta")).get("num_rejected_or_invalid_raw_operation_segments"), 0.0)
        metrics["num_derived_operation_segments"] = len(self._safe_list(pred.get("derived_operation_segments_from_patches")))
        repair_meta = self._safe_dict(pred.get("repair_meta"))
        projection_diag = self._safe_dict(pred.get("mechanism_projection_diagnostics"))
        metrics["num_rolling_split_repaired_segments"] = self._safe_float(repair_meta.get("num_rolling_split_repaired_segments"), 0.0)
        metrics["num_uncertain_operation_segments"] = self._safe_float(repair_meta.get("num_uncertain_operation_segments"), 0.0)
        metrics["num_rejected_operation_segments_by_projection"] = self._safe_float(repair_meta.get("num_rejected_operation_segments_by_projection"), 0.0)
        metrics["physics_consistent_projection_applied"] = 1.0 if repair_meta.get("physics_consistent_projection_applied") else 0.0
        metrics["projection_uses_future_target_labels"] = 1.0 if repair_meta.get("projection_uses_future_target_labels") else 0.0
        before_v = self._safe_dict(projection_diag.get("violations_before"))
        after_v = self._safe_dict(projection_diag.get("violations_after"))
        metrics["projection_hard_violations_before"] = sum(self._safe_float(v, 0.0) for v in before_v.values()) if before_v else 0.0
        metrics["projection_hard_violations_after"] = sum(self._safe_float(v, 0.0) for v in after_v.values()) if after_v else 0.0
        metrics["projection_objective_total_before"] = self._safe_float(projection_diag.get("objective_total_before"), 0.0)
        metrics["projection_objective_total_after"] = self._safe_float(projection_diag.get("objective_total_after"), 0.0)
        retry_meta = self._safe_dict(pred.get("retry_meta"))
        metrics["retry_enabled"] = 1.0 if retry_meta.get("retry_enabled") else 0.0
        metrics["num_horizon_attempts"] = self._safe_float(retry_meta.get("num_attempts"), 1.0)
        metrics["num_horizon_retries_used"] = self._safe_float(retry_meta.get("num_retries_used"), 0.0)
        metrics["first_pass_missing"] = 1.0 if retry_meta.get("first_pass_missing") else 0.0
        metrics["retry_success"] = 1.0 if retry_meta.get("retry_success") else 0.0
        metrics["retry_uses_future_target_labels"] = 1.0 if retry_meta.get("uses_future_target_labels") else 0.0
        metrics["rule_fallback_enabled"] = 1.0 if retry_meta.get("rule_fallback_enabled") else 0.0
        metrics["rule_fallback_used"] = 1.0 if retry_meta.get("rule_fallback_used") else 0.0
        metrics["rule_fallback_success"] = 1.0 if retry_meta.get("rule_fallback_success") else 0.0
        metrics["rule_fallback_evaluate_as_prediction"] = 1.0 if retry_meta.get("rule_fallback_evaluate_as_prediction") else 0.0
        metrics["rule_fallback_uses_future_target_labels"] = 1.0 if retry_meta.get("rule_fallback_uses_future_target_labels") else 0.0
        metrics["rule_fallback_num_raw_segments"] = self._safe_float(retry_meta.get("rule_fallback_num_raw_segments"), 0.0)
        metrics["rule_fallback_num_projected_segments"] = self._safe_float(retry_meta.get("rule_fallback_num_projected_segments"), 0.0)
        return metrics

    def _normalize_predicted_summary(self, raw_output: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(raw_output, dict):
            return {}

        if isinstance(raw_output.get("predicted_prod_next_state_summary"), dict):
            return raw_output["predicted_prod_next_state_summary"]

        guessed_keys = {
            "transition_type",
            "affected_heat_ids",
            "affected_rolling_units",
            "predicted_time_changes",
            "predicted_sequence_changes",
            "stable_heat_ids",
            "stable_rolling_units",
            "predicted_local_effects",
            "constraint_check",
            "confidence",
            "comment",
        }
        if any(k in raw_output for k in guessed_keys):
            return {
                "transition_type": raw_output.get("transition_type", "unknown"),
                "affected_heat_ids": self._safe_list(raw_output.get("affected_heat_ids")),
                "affected_rolling_units": self._safe_list(raw_output.get("affected_rolling_units")),
                "predicted_time_changes": self._safe_list(raw_output.get("predicted_time_changes")),
                "predicted_sequence_changes": self._safe_list(raw_output.get("predicted_sequence_changes")),
                "stable_heat_ids": self._safe_list(raw_output.get("stable_heat_ids")),
                "stable_rolling_units": self._safe_list(raw_output.get("stable_rolling_units")),
                "predicted_local_effects": self._safe_list(raw_output.get("predicted_local_effects")),
                "constraint_check": self._safe_dict(raw_output.get("constraint_check")),
                "confidence": self._safe_float(raw_output.get("confidence", 0.0), 0.0),
                "comment": self._safe_str(raw_output.get("comment", "")),
            }

        return {
            "transition_type": "unknown",
            "affected_heat_ids": [],
            "affected_rolling_units": [],
            "predicted_time_changes": [],
            "predicted_sequence_changes": [],
            "stable_heat_ids": [],
            "stable_rolling_units": [],
            "predicted_local_effects": [],
            "constraint_check": {
                "process_order_preserved": True,
                "rolling_mapping_stable": True,
                "locality": "unknown",
            },
            "confidence": 0.0,
            "comment": "No valid predicted_prod_next_state_summary found.",
        }

    def parse_output(self, raw_output, prod_input: Dict[str, Any]) -> CandidatePlan:
        meta = self._safe_dict(prod_input.get("meta"))
        day_id = meta.get("d", "unknown_day")

        if isinstance(raw_output, list):
            normalized_output = {
                "task": "production_next_state_prediction",
                "mode": "production_state_predictor",
                "predicted_prod_next_state_summary": {
                    "transition_type": "unknown",
                    "affected_heat_ids": [],
                    "affected_rolling_units": [],
                    "predicted_time_changes": [],
                    "predicted_sequence_changes": [],
                    "stable_heat_ids": [],
                    "stable_rolling_units": [],
                    "predicted_local_effects": [],
                    "constraint_check": {
                        "process_order_preserved": True,
                        "rolling_mapping_stable": True,
                        "locality": "unknown",
                    },
                    "confidence": 0.0,
                    "comment": "Model returned list; normalized by parser.",
                },
                "_raw_list_output": raw_output,
            }
        elif isinstance(raw_output, dict):
            normalized_output = raw_output
        else:
            raise TypeError("The production model output is neither a dict nor a list and cannot be parsed.")

        predicted_summary = self._normalize_predicted_summary(normalized_output)

        affected_heat_ids = self._safe_list(predicted_summary.get("affected_heat_ids"))
        affected_rolling_units = self._safe_list(predicted_summary.get("affected_rolling_units"))

        normalized_time_changes = []
        raw_time_changes = self._safe_list(predicted_summary.get("predicted_time_changes"))

        affected_rolling_unit_set = {str(x) for x in affected_rolling_units}
        heat_set = {str(x) for x in affected_heat_ids}
        heat_time_fields = {"tb0", "tb1", "tl0", "tl1"}
        rolling_time_fields = {"t0", "t1"}

        for item in raw_time_changes:
            if not isinstance(item, dict):
                continue

            field_name = self._safe_str(item.get("field"))
            direction = self._safe_str(item.get("direction", "unknown"))
            magnitude = self._safe_str(item.get("magnitude", "unknown"))

            target_type = self._safe_str(item.get("target_type"))
            target_id = self._safe_str(item.get("target_id"))

            legacy_heat_id = self._safe_str(item.get("heat_id"))
            legacy_rolling_unit_id = self._safe_str(item.get("rolling_unit_id"))

            if not target_id:
                if legacy_heat_id:
                    target_id = legacy_heat_id
                elif legacy_rolling_unit_id:
                    target_id = legacy_rolling_unit_id

            if not target_type:

                if field_name in heat_time_fields:
                    target_type = "heat"
                elif field_name in rolling_time_fields:
                    target_type = "rolling_unit"

                if target_id in affected_rolling_unit_set:
                    target_type = "rolling_unit"
                elif target_id in heat_set:
                    target_type = "heat"

                elif target_id.isdigit():
                    target_type = "rolling_unit"
                elif target_id:
                    target_type = "heat"
                else:
                    target_type = "unknown"

            normalized_time_changes.append({
                "target_type": target_type,
                "target_id": target_id,
                "field": field_name,
                "direction": direction,
                "magnitude": magnitude,
            })

        predicted_summary["predicted_time_changes"] = normalized_time_changes

        candidate_summary = {
            "prediction_style": "next_state_summary",
            "transition_type": predicted_summary.get("transition_type", "unknown"),
            "num_affected_heats": len(affected_heat_ids),
            "num_affected_rolling_units": len(affected_rolling_units),
            "confidence": self._safe_float(predicted_summary.get("confidence", 0.0), 0.0),
            "comment": self._safe_str(predicted_summary.get("comment", "")),
        }

        production_statistics = {
            "num_candidate_actions": 0,
            "affected_heat_ids": affected_heat_ids,
            "affected_rolling_units": affected_rolling_units,
            "predicted_time_changes": normalized_time_changes,
            "predicted_sequence_changes": self._safe_list(predicted_summary.get("predicted_sequence_changes")),
            "stable_heat_ids": self._safe_list(predicted_summary.get("stable_heat_ids")),
            "stable_rolling_units": self._safe_list(predicted_summary.get("stable_rolling_units")),
            "predicted_local_effects": self._safe_list(predicted_summary.get("predicted_local_effects")),
            "constraint_check": self._safe_dict(predicted_summary.get("constraint_check")),
            "candidate_summary": candidate_summary,
        }

        plan_snapshot = {
            "predicted_prod_next_state_summary": predicted_summary
        }

        return CandidatePlan(
            plan_id=normalized_output.get("plan_id", f"prod_state_pred_{uuid.uuid4().hex[:8]}"),
            day_id=day_id,
            heat_id=None,
            task=normalized_output.get("task", "production_next_state_prediction"),
            mode=normalized_output.get("mode", "production_state_predictor"),
            candidate_actions=[],
            candidate_summary=candidate_summary,
            production_statistics=production_statistics,
            plan_snapshot=plan_snapshot,
            raw_output=normalized_output,
            meta={"source": "production_agent_adapter_state_model"},
        )

    def propose(self, prod_input: Dict[str, Any]) -> CandidatePlan:
        model_input = self.build_input(prod_input)
        raw_output = self.call_model(model_input)
        return self.parse_output(raw_output, prod_input)

_BaseProductionAgentAdapter = ProductionAgentAdapter

class ProductionAgentAdapter(_BaseProductionAgentAdapter):

    _STAGES = ("BOF", "LF", "Casting", "Rolling")

    def _candidate_sets_from_sample(self, sample: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        input_state = self._safe_dict(self._safe_dict(sample).get("input_state"))
        cands = self._safe_dict(input_state.get("window_evidence_candidates"))
        return {
            "confirmed": [x for x in self._safe_list(cands.get("confirmed")) if isinstance(x, dict)],
            "optional": [x for x in self._safe_list(cands.get("optional")) if isinstance(x, dict)],
            "route_future_only": [x for x in self._safe_list(cands.get("route_future_only")) if isinstance(x, dict)],
        }

    def _stage_bounds_from_sample(self, sample: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
        input_state = self._safe_dict(self._safe_dict(sample).get("input_state"))
        bounds = self._safe_dict(input_state.get("window_stage_evidence_bounds"))
        out: Dict[str, Dict[str, int]] = {}
        for st in self._STAGES:
            row = self._safe_dict(bounds.get(st))
            try:
                lower = int(row.get("lower", 0) or 0)
            except Exception:
                lower = 0
            try:
                upper = int(row.get("upper", lower) if row.get("upper") is not None else lower)
            except Exception:
                upper = lower

            upper = max(0, min(upper, 10000))
            lower = max(0, min(lower, upper))
            out[st] = {"lower": lower, "upper": upper}
        return out

    def _clean_segment_key2(self, seg: Dict[str, Any]) -> str:
        hid = self._normalize_heat_id(seg.get("heat_id") or seg.get("id") or seg.get("hn")) or ""
        st = self._canonical_stage(seg.get("stage") or seg.get("process") or seg.get("operation"))
        return f"{hid}|{st}"

    def _candidate_key2_sets(self, sample: Dict[str, Any]) -> Dict[str, set]:
        cands = self._candidate_sets_from_sample(sample)
        out: Dict[str, set] = {}
        for name, rows in cands.items():
            s = set()
            for row in rows:
                hid = self._normalize_heat_id(row.get("heat_id")) or ""
                st = self._canonical_stage(row.get("stage"))
                if hid and st:
                    s.add(f"{hid}|{st}")
            out[name] = s
        out["allowed_main"] = out.get("confirmed", set()) | out.get("optional", set())
        return out

    def build_horizon_input(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        model_input = super().build_horizon_input(sample)
        sample = self._safe_dict(sample)
        input_state = self._safe_dict(sample.get("input_state"))
        window_candidates = self._safe_dict(input_state.get("window_evidence_candidates"))
        stage_bounds = self._safe_dict(input_state.get("window_stage_evidence_bounds"))

        input_block = self._safe_dict(model_input.get("input"))
        input_block["window_evidence_candidates"] = window_candidates
        input_block["window_stage_evidence_bounds"] = stage_bounds
        input_block["candidate_selection_policy"] = self._safe_dict(input_state.get("window_evidence_policy"))
        input_block["important_disambiguation"] = {
            "task": "patch_window_operation_occupancy_prediction",
            "not_task": "complete heat-route BOF-LF-Casting-Rolling rollout",
            "confirmed_candidates": "strong evidence; should be covered unless PCVP makes them infeasible",
            "optional_candidates": "weak window evidence; may be selected within upper bounds",
            "route_future_only_candidates": "route-feasible but not window-confirmed; do not put in predicted_operation_segments",
        }
        model_input["input"] = input_block
        model_input["task_instruction"] = (
            "Predict operation-level production occupancy only for the given future patch window. "
            "Use window_evidence_candidates as the candidate universe. "
            "Select/adjust confirmed and optional candidates; do not invent route-completion stages. "
            "Route_future_only candidates must be omitted from predicted_operation_segments or placed in uncertain_operation_segments. "
            "Do not use target_future labels. Return valid JSON only."
        )
        model_input["required_reasoning_contract"] = {
            "predicted_operation_segments": [
                "Use only confirmed and optional window_evidence_candidates for the main predicted_operation_segments list.",
                "confirmed candidates define lower-bound coverage targets.",
                "optional candidates may be included only if their stage upper bound is not exceeded.",
                "route_future_only candidates are diagnostics only; place them in uncertain_operation_segments or omit them.",
                "Do not complete BOF-LF-Casting-Rolling routes by default.",
                "Every main segment must set source exactly to model_output.",
                "The output must be patch-window occupancy, not heat-route rollout.",
            ],
            "uncertain_operation_segments": [
                "Use for route-future-only, low-confidence, or timing-uncertain downstream hypotheses.",
                "These are not main predictions and will not be used as valid Gantt prediction segments.",
            ],
        }
        return model_input

    def build_horizon_messages(self, model_input: Dict[str, Any]) -> List[Dict[str, str]]:
        system_prompt = (
            "You are a production operation-level world model for patch-window Gantt prediction.\n"
            "You must predict only the operations that occupy the specified future patch window.\n"
            "You must not complete whole heat routes by default.\n"
            "Use only confirmed and optional window_evidence_candidates as main prediction candidates.\n"
            "Do not put route_future_only candidates into predicted_operation_segments.\n"
            "Return exactly one JSON object with keys: task, mode, predicted_operation_segments, optional uncertain_operation_segments.\n"
            "Every main predicted segment must have source='model_output'.\n"
            "No target_future labels or offline metrics are available or allowed.\n"
        )
        user_prompt = (
            "Build the JSON prediction from this model input. Follow the candidate selection policy exactly:\n"
            + json.dumps(model_input, ensure_ascii=False, indent=2, default=str)
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _prediction_stage_counts(self, prediction: Dict[str, Any]) -> Dict[str, int]:
        counts = {st: 0 for st in self._STAGES}
        for seg in self._safe_list(self._safe_dict(prediction).get("predicted_operation_segments")):
            if not isinstance(seg, dict):
                continue
            st = self._canonical_stage(seg.get("stage") or seg.get("process") or seg.get("operation"))
            if st in counts:
                counts[st] += 1
        return counts

    def _projection_hard_violations_after(self, prediction: Dict[str, Any]) -> int:
        diag = self._safe_dict(self._safe_dict(prediction).get("mechanism_projection_diagnostics"))
        after = self._safe_dict(diag.get("violations_after"))
        total = 0
        for v in after.values():
            try:
                total += int(float(v))
            except Exception:
                pass
        return total

    def _objective_after(self, prediction: Dict[str, Any]) -> float:
        diag = self._safe_dict(self._safe_dict(prediction).get("mechanism_projection_diagnostics"))
        return self._safe_float(diag.get("objective_total_after"), 0.0)

    def _evidence_selection_metrics(self, prediction: Dict[str, Any], sample: Dict[str, Any]) -> Dict[str, Any]:
        pred = self._safe_dict(prediction)
        bounds = self._stage_bounds_from_sample(sample)
        counts = self._prediction_stage_counts(pred)
        ckey = self._candidate_key2_sets(sample)

        missing = 0 if pred.get("operation_segment_prediction_available") else 1
        hard = self._projection_hard_violations_after(pred)

        over = 0
        lower_miss = 0
        by_stage: Dict[str, Dict[str, int]] = {}
        for st in self._STAGES:
            lower = int(bounds.get(st, {}).get("lower", 0))
            upper = int(bounds.get(st, {}).get("upper", lower))
            c = int(counts.get(st, 0))
            over_st = max(0, c - upper)
            miss_st = max(0, lower - c)
            over += over_st
            lower_miss += miss_st
            by_stage[st] = {"lower": lower, "upper": upper, "count": c, "over": over_st, "miss": miss_st}

        route_future_only_in_main = 0
        outside_candidate_in_main = 0
        for seg in self._safe_list(pred.get("predicted_operation_segments")):
            if not isinstance(seg, dict):
                continue
            key2 = self._clean_segment_key2(seg)
            if key2 in ckey.get("route_future_only", set()) and key2 not in ckey.get("allowed_main", set()):
                route_future_only_in_main += 1
            if ckey.get("allowed_main") and key2 not in ckey.get("allowed_main", set()):
                outside_candidate_in_main += 1

        objective = self._objective_after(pred)
        num_segments = len(self._safe_list(pred.get("predicted_operation_segments")))

        selection_key = (
            int(missing),
            int(hard),
            int(over),
            int(route_future_only_in_main),
            int(outside_candidate_in_main),
            int(lower_miss),
            float(objective),
            -int(num_segments),
        )
        return {
            "missing_prediction": int(missing),
            "hard_violations_after": int(hard),
            "over_upper_bound_count": int(over),
            "route_future_only_in_main_count": int(route_future_only_in_main),
            "outside_candidate_in_main_count": int(outside_candidate_in_main),
            "lower_bound_miss_count": int(lower_miss),
            "objective_total_after": float(objective),
            "num_predicted_segments": int(num_segments),
            "by_stage": by_stage,
            "selection_key": selection_key,
            "uses_future_target_labels": False,
        }

    def _should_early_stop(self, metrics: Dict[str, Any], attempt_index: int, best_metrics: Optional[Dict[str, Any]]) -> bool:
        if metrics.get("missing_prediction", 1) != 0:
            return False
        if metrics.get("hard_violations_after", 99) != 0:
            return False
        if metrics.get("over_upper_bound_count", 99) != 0:
            return False
        if metrics.get("route_future_only_in_main_count", 99) != 0:
            return False
        if metrics.get("outside_candidate_in_main_count", 99) != 0:
            return False
        if metrics.get("lower_bound_miss_count", 99) == 0:
            return True

        if attempt_index >= 1 and best_metrics is not None:
            cur_key = tuple(metrics.get("selection_key"))
            best_key = tuple(best_metrics.get("selection_key"))
            if cur_key >= best_key and metrics.get("hard_violations_after", 0) == 0:
                return True
        return False

    def _build_clean_retry_input(
        self,
        original_model_input: Dict[str, Any],
        sample: Dict[str, Any],
        previous_raw_output: Dict[str, Any],
        previous_prediction: Dict[str, Any],
        previous_metrics: Dict[str, Any],
        attempt_index: int,
    ) -> Dict[str, Any]:
        base = json.loads(json.dumps(original_model_input, ensure_ascii=False, default=str))
        base["task_instruction"] = (
            str(base.get("task_instruction", "")) +
            " Retry only to fix schema/PCVP/evidence-bound issues. Do not use target labels. "
            "Do not increase route-future-only stages in the main prediction."
        )
        base["retry_request"] = {
            "enabled": True,
            "attempt_index": int(attempt_index),
            "strict_no_target_labels": True,
            "previous_attempt_evidence_metrics": {
                k: v for k, v in previous_metrics.items()
                if k not in {"selection_key"}
            },
            "previous_attempt_selection_key": list(previous_metrics.get("selection_key", [])),
            "allowed_feedback": [
                "schema validity",
                "PCVP hard violations",
                "window evidence lower/upper bound counts",
                "route_future_only_in_main_count",
                "outside_candidate_in_main_count",
                "projection objective",
            ],
            "forbidden_feedback": [
                "target_operation_segments",
                "ground truth future operation start/end",
                "IoU, MAE, F1, precision, recall against target",
            ],
            "correction_goal": (
                "Return a cleaner JSON candidate selection within confirmed/optional "
                "window_evidence_candidates while satisfying stage bounds."
            ),
        }
        return base

    def predict_horizon(self, sample: Dict[str, Any]) -> Dict[str, Any]:

        sample = self._safe_dict(sample)
        model_input = self.build_horizon_input(sample)
        max_attempts = max(1, int(self.max_horizon_retries) + 1)

        attempts: List[Dict[str, Any]] = []
        best_pred: Dict[str, Any] = {}
        best_raw: Dict[str, Any] = {}
        best_metrics: Optional[Dict[str, Any]] = None
        current_input = model_input

        for attempt_index in range(max_attempts):
            raw_output = self._call_horizon_model_once(current_input, attempt_index=attempt_index)
            pred = self.normalize_horizon_output(raw_output, sample=sample)
            metrics = self._evidence_selection_metrics(pred, sample)

            attempts.append({
                "attempt_index": attempt_index,
                "operation_segment_prediction_available": bool(pred.get("operation_segment_prediction_available")),
                "num_projected_segments": len(self._safe_list(pred.get("predicted_operation_segments"))),
                "selection_key": list(metrics.get("selection_key", [])),
                "evidence_metrics": {k: v for k, v in metrics.items() if k != "selection_key"},
                "uses_future_target_labels": False,
            })

            if best_metrics is None or tuple(metrics["selection_key"]) < tuple(best_metrics["selection_key"]):
                best_metrics = metrics
                best_pred = pred
                best_raw = raw_output

            if self._should_early_stop(metrics, attempt_index, best_metrics):
                break

            if attempt_index >= max_attempts - 1:
                break

            current_input = self._build_clean_retry_input(
                original_model_input=model_input,
                sample=sample,
                previous_raw_output=raw_output,
                previous_prediction=pred,
                previous_metrics=metrics,
                attempt_index=attempt_index + 1,
            )

        pred = best_pred if isinstance(best_pred, dict) else {}
        raw_output = best_raw if isinstance(best_raw, dict) else {}
        selected_metrics = best_metrics or self._evidence_selection_metrics(pred, sample)

        rule_fallback_meta: Dict[str, Any] = {
            "rule_fallback_enabled": self.enable_retry_exhausted_rule_fallback,
            "rule_fallback_used": False,
            "rule_fallback_success": False,
            "rule_fallback_evaluate_as_prediction": self.evaluate_rule_fallback_as_prediction,
            "rule_fallback_uses_future_target_labels": False,
            "rule_fallback_num_raw_segments": 0,
            "rule_fallback_num_projected_segments": 0,
        }
        if not pred.get("operation_segment_prediction_available"):
            pred, fallback_raw_output, rule_fallback_meta = self._apply_retry_exhausted_rule_fallback(
                sample=sample,
                attempt_records=attempts,
                current_prediction=pred,
            )
            if rule_fallback_meta.get("rule_fallback_used"):
                raw_output = fallback_raw_output
                selected_metrics = self._evidence_selection_metrics(pred, sample)
                attempts.append({
                    "attempt_index": "retry_exhausted_rule_fallback",
                    "was_rule_fallback": True,
                    "operation_segment_prediction_available": bool(pred.get("operation_segment_prediction_available")),
                    "num_projected_segments": len(self._safe_list(pred.get("predicted_operation_segments"))),
                    "uses_future_target_labels": False,
                })

        selected_index = None
        if attempts:
            best_key = list(selected_metrics.get("selection_key", []))
            for item in attempts:
                if item.get("selection_key") == best_key:
                    selected_index = item.get("attempt_index")
                    break

        retry_meta = {
            "retry_enabled": self.max_horizon_retries > 0,
            "early_stop_enabled": True,
            "best_of_n_enabled": True,
            "best_of_n_strategy": "evidence_bounded_lexicographic_pareto",
            "max_horizon_retries": self.max_horizon_retries,
            "num_attempts": len([a for a in attempts if not a.get("was_rule_fallback")]),
            "num_retries_used": max(0, len([a for a in attempts if not a.get("was_rule_fallback")]) - 1),
            "selected_attempt_index": selected_index,
            "selected_attempt_key": list(selected_metrics.get("selection_key", [])),
            "selected_attempt_metrics": {k: v for k, v in selected_metrics.items() if k != "selection_key"},
            "first_pass_missing": bool(attempts and not attempts[0].get("operation_segment_prediction_available")),
            "final_prediction_available": bool(pred.get("operation_segment_prediction_available")),
            "retry_success": bool(attempts and not attempts[0].get("operation_segment_prediction_available") and pred.get("operation_segment_prediction_available")),
            "attempts": attempts,
            "uses_future_target_labels": False,
            **rule_fallback_meta,
            "policy": (
                "Clean candidate selection: deterministic window evidence candidates, "
                "LLM selection/refinement, PCVP projection, early-stop best-of-N. "
                "No target_future or target-based metrics are used."
            ),
        }

        pred["retry_meta"] = retry_meta
        pred["evidence_bounded_selection"] = {k: v for k, v in selected_metrics.items() if k != "selection_key"}
        pred.setdefault("repair_meta", {})
        pred["repair_meta"].update({
            "retry_enabled": retry_meta["retry_enabled"],
            "best_of_n_enabled": True,
            "best_of_n_strategy": retry_meta["best_of_n_strategy"],
            "best_of_n_selected_attempt_index": selected_index,
            "best_of_n_selected_key": retry_meta["selected_attempt_key"],
            "num_horizon_attempts": retry_meta["num_attempts"],
            "num_horizon_retries_used": retry_meta["num_retries_used"],
            "first_pass_missing": retry_meta["first_pass_missing"],
            "retry_success": retry_meta["retry_success"],
            "retry_uses_future_target_labels": False,
            "rule_fallback_enabled": retry_meta["rule_fallback_enabled"],
            "rule_fallback_used": retry_meta["rule_fallback_used"],
            "rule_fallback_success": retry_meta["rule_fallback_success"],
            "rule_fallback_evaluate_as_prediction": retry_meta["rule_fallback_evaluate_as_prediction"],
            "rule_fallback_uses_future_target_labels": False,
            "rule_fallback_num_raw_segments": retry_meta.get("rule_fallback_num_raw_segments", 0),
            "rule_fallback_num_projected_segments": retry_meta.get("rule_fallback_num_projected_segments", 0),
            "evidence_over_upper_bound_count": selected_metrics.get("over_upper_bound_count", 0),
            "evidence_lower_bound_miss_count": selected_metrics.get("lower_bound_miss_count", 0),
            "route_future_only_in_main_count": selected_metrics.get("route_future_only_in_main_count", 0),
            "outside_candidate_in_main_count": selected_metrics.get("outside_candidate_in_main_count", 0),
        })
        pred["sample_meta"] = {
            "day_id": sample.get("day_id"),
            "anchor_patch_idx": sample.get("anchor_patch_idx"),
            "history_window": sample.get("history_window"),
            "horizon": sample.get("horizon"),
            "source_meta": self._safe_dict(sample.get("meta")),
        }
        return pred

def _paa_env_bool_schedule_guided(name: str, default: bool = True) -> bool:
    try:
        v = os.environ.get(name)
        if v is None:
            return bool(default)
        return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}
    except Exception:
        return bool(default)

def _paa_safe_schedule_references(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(sample, dict):
        return []
    input_state = sample.get("input_state") if isinstance(sample.get("input_state"), dict) else {}
    refs = input_state.get("schedule_reference_candidates")
    if isinstance(refs, list) and refs:
        return [x for x in refs if isinstance(x, dict)]
    task = input_state.get("operation_schedule_task") if isinstance(input_state.get("operation_schedule_task"), dict) else {}
    refs = task.get("schedule_reference_candidates")
    if isinstance(refs, list) and refs:
        return [x for x in refs if isinstance(x, dict)]
    return []

def _paa_stage_counts_from_schedule_refs(refs: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"BOF": 0, "LF": 0, "Casting": 0, "Rolling": 0}
    for x in refs:
        st = str(x.get("stage") or "")
        if st in counts:
            counts[st] += 1
    return counts

_paa_previous_build_horizon_input_schedule_guided = ProductionAgentAdapter.build_horizon_input

def _paa_build_horizon_input_schedule_guided(self: ProductionAgentAdapter, sample: Dict[str, Any]) -> Dict[str, Any]:
    model_input = _paa_previous_build_horizon_input_schedule_guided(self, sample)
    if not _paa_env_bool_schedule_guided("PROD_USE_SCHEDULE_GUIDED_INFERENCE", True):
        return model_input

    refs = _paa_safe_schedule_references(sample)
    if not refs:
        return model_input

    input_block = self._safe_dict(model_input.get("input"))
    input_block["production_conditioning_mode"] = "schedule_guided_inference"
    input_block["schedule_reference_candidates"] = refs
    input_block["schedule_reference_stage_counts"] = _paa_stage_counts_from_schedule_refs(refs)
    input_block["schedule_guided_inference_policy"] = {
        "enabled": True,
        "uses_future_operation_schedule": True,
        "llm_must_regenerate_predicted_segments": True,
        "direct_copy_is_not_allowed": True,
        "pcvp_after_llm": True,
        "pcvp_soft_retention_enabled": True,
        "expected_behavior": (
            "Use schedule_reference_candidates as the reference schedule, regenerate a clean JSON list of "
            "predicted_operation_segments, and set source exactly to model_output for regenerated main segments. "
            "PCVP should repair and annotate physically weak segments instead of deleting schedule-guided Casting/Rolling segments."
        ),
    }

    input_block["window_evidence_candidates"] = {
        "confirmed": refs,
        "optional": [],
        "route_future_only": [],
    }
    input_block["window_stage_evidence_bounds"] = {
        st: {"lower": n, "upper": n}
        for st, n in _paa_stage_counts_from_schedule_refs(refs).items()
    }
    model_input["input"] = input_block

    model_input["task_instruction"] = (
        "You are a production operation-level schedule inference model for a patch-level production-energy world model. "
        "A reference production schedule is provided as schedule_reference_candidates. Your task is to regenerate this "
        "reference into a clean predicted_operation_segments JSON for the future horizon, correcting field names, canonicalizing "
        "heat ids/stages/units, and keeping the timing compatible with the horizon. PCVP will be applied after your output. "
        "Do not output dispatching actions or energy predictions. Return valid JSON only."
    )
    model_input["required_reasoning_contract"] = {
        "predicted_operation_segments": [
            "Read schedule_reference_candidates carefully and regenerate them as predicted_operation_segments.",
            "Every regenerated main segment must set source exactly to model_output.",
            "Preserve heat_id, stage, unit, start_min, end_min and duration_min whenever they are already valid.",
            "Do not invent unrelated heats or stages outside schedule_reference_candidates.",
            "If a reference segment has minor formatting issues, repair the format rather than dropping it.",
            "Do not omit Casting/Rolling/LF reference segments merely because their duration is short or their unit label is generic; PCVP will repair/annotate them.",
            "Respect BOF -> LF -> Casting -> Rolling order and single-unit occupancy where applicable, but preserve schedule-reference occupancy evidence for downstream energy conditioning.",
            "Return one combined predicted_operation_segments list for the full horizon.",
        ],
        "uncertain_operation_segments": [
            "Use only if a reference segment is internally inconsistent and cannot be made physically plausible before PCVP.",
        ],
    }
    return model_input

ProductionAgentAdapter.build_horizon_input = _paa_build_horizon_input_schedule_guided

_paa_previous_build_horizon_messages_schedule_guided = ProductionAgentAdapter.build_horizon_messages

def _paa_build_horizon_messages_schedule_guided(self: ProductionAgentAdapter, model_input: Dict[str, Any]) -> List[Dict[str, str]]:
    input_block = self._safe_dict(model_input.get("input"))
    refs = input_block.get("schedule_reference_candidates")
    if _paa_env_bool_schedule_guided("PROD_USE_SCHEDULE_GUIDED_INFERENCE", True) and isinstance(refs, list) and refs:
        system_prompt = (
            "You are a production operation-level schedule inference model.\n"
            "The input contains schedule_reference_candidates for the future horizon.\n"
            "Your job is to regenerate a clean predicted_operation_segments JSON from those references.\n"
            "Do not bypass reasoning by returning prose. Do not output energy predictions or dispatch actions.\n"
            "Every main segment must have source='model_output'.\n"
            "PCVP will check and normalize physical consistency after your output.\n"
            "Return exactly one valid JSON object with keys: task, mode, predicted_operation_segments, optional uncertain_operation_segments.\n"
        )
        user_prompt = (
            "Regenerate the schedule reference candidates into predicted_operation_segments. "
            "Keep the result physically plausible and JSON-only. Model input:\n"
            + json.dumps(model_input, ensure_ascii=False, indent=2, default=str)
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    return _paa_previous_build_horizon_messages_schedule_guided(self, model_input)

ProductionAgentAdapter.build_horizon_messages = _paa_build_horizon_messages_schedule_guided

_paa_previous_predict_horizon_schedule_guided = ProductionAgentAdapter.predict_horizon

def _paa_predict_horizon_schedule_guided(self: ProductionAgentAdapter, sample: Dict[str, Any]) -> Dict[str, Any]:
    pred = _paa_previous_predict_horizon_schedule_guided(self, sample)
    refs = _paa_safe_schedule_references(sample)
    if _paa_env_bool_schedule_guided("PROD_USE_SCHEDULE_GUIDED_INFERENCE", True) and refs and isinstance(pred, dict):
        pred["production_conditioning_mode"] = "schedule_guided_inference"
        pred["schedule_guided_inference_used"] = True
        pred["schedule_guided_inference_uses_future_operation_schedule"] = True
        pred["schedule_reference_num_segments"] = len(refs)
        pred["schedule_guided_inference_policy"] = {
            "llm_regeneration_required": True,
            "pcvp_after_llm": True,
            "direct_known_schedule_bypass": False,
        }
        repair_meta = self._safe_dict(pred.get("repair_meta"))
        repair_meta["production_conditioning_mode"] = "schedule_guided_inference"
        repair_meta["schedule_guided_inference_used"] = True
        repair_meta["schedule_guided_inference_uses_future_operation_schedule"] = True
        repair_meta["schedule_reference_num_segments"] = len(refs)
        repair_meta["pcvp_soft_retention_enabled"] = True
        pred["repair_meta"] = repair_meta
        retry_meta = self._safe_dict(pred.get("retry_meta"))
        retry_meta["schedule_guided_inference_used"] = True
        retry_meta["direct_known_schedule_bypass"] = False
        retry_meta["pcvp_soft_retention_enabled"] = True
        pred["retry_meta"] = retry_meta
        diag = self._safe_dict(pred.get("mechanism_projection_diagnostics"))
        if diag:
            diag["production_conditioning_mode"] = "schedule_guided_inference"
            diag["schedule_guided_inference"] = {
                "enabled": True,
                "uses_future_operation_schedule": True,
                "llm_regeneration_required": True,
                "direct_bypass": False,
                "pcvp_soft_retention_enabled": True,
            }
            pred["mechanism_projection_diagnostics"] = diag
    return pred

ProductionAgentAdapter.predict_horizon = _paa_predict_horizon_schedule_guided

_paa_previous_evaluate_horizon_prediction_schedule_guided = ProductionAgentAdapter.evaluate_horizon_prediction

def _paa_evaluate_horizon_prediction_schedule_guided(
    self: ProductionAgentAdapter,
    prediction: Dict[str, Any],
    target_future: Dict[str, Any],
) -> Dict[str, Any]:
    metrics = _paa_previous_evaluate_horizon_prediction_schedule_guided(self, prediction, target_future)
    if isinstance(prediction, dict) and prediction.get("schedule_guided_inference_used"):
        metrics["production_conditioning_mode"] = "schedule_guided_inference"
        metrics["schedule_guided_inference_used"] = 1.0
        metrics["schedule_guided_inference_uses_future_operation_schedule"] = 1.0
    return metrics

ProductionAgentAdapter.evaluate_horizon_prediction = _paa_evaluate_horizon_prediction_schedule_guided

from copy import deepcopy as _paa_stage1_deepcopy

_paa_stage1_prev_build_horizon_input = ProductionAgentAdapter.build_horizon_input
_paa_stage1_prev_predict_horizon = ProductionAgentAdapter.predict_horizon


def _paa_stage1_build_horizon_input(self, sample):
    model_input = _paa_stage1_prev_build_horizon_input(self, sample)
    sample_dict = self._safe_dict(sample)
    sample_input = self._safe_dict(sample_dict.get("input_state"))
    action = self._safe_dict(sample_dict.get("candidate_action") or sample_input.get("candidate_action"))
    if action:
        input_block = self._safe_dict(model_input.get("input"))
        input_block["candidate_action"] = action
        input_block["action_conditioned_world_model"] = True
        model_input["input"] = input_block
    return model_input


def _paa_stage1_predict_horizon(self, sample, candidate_action=None):
    working = sample
    if candidate_action is not None:
        working = _paa_stage1_deepcopy(sample)
        if isinstance(working, dict):
            working["candidate_action"] = candidate_action
            inp = working.setdefault("input_state", {})
            if isinstance(inp, dict):
                inp["candidate_action"] = candidate_action
    return _paa_stage1_prev_predict_horizon(self, working)


ProductionAgentAdapter.build_horizon_input = _paa_stage1_build_horizon_input
ProductionAgentAdapter.predict_horizon = _paa_stage1_predict_horizon
