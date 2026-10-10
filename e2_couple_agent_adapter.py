from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from b_base_chat_model import BaseChatModel
from f_transition_engine import CandidatePlan, EnergyTrajectory, EvaluationResult

def _extract_first_balanced_json(text: str):

    start_positions = []
    for i, ch in enumerate(text):
        if ch in "{[":
            start_positions.append(i)

    for start in start_positions:
        open_ch = text[start]
        close_ch = "}" if open_ch == "{" else "]"

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
            elif ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    return candidate

    return None

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

    if "</think>" in text:
        suffix = text.split("</think>")[-1].strip()

        try:
            obj = json.loads(suffix)
            if isinstance(obj, (dict, list)):
                return obj
        except Exception:
            pass

        candidate = _extract_first_balanced_json(suffix)
        if candidate:
            try:
                obj = json.loads(candidate)
                if isinstance(obj, (dict, list)):
                    return obj
            except Exception:
                pass

    candidate = _extract_first_balanced_json(text)
    if candidate:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, (dict, list)):
                return obj
        except Exception:
            pass

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

    preview = text[:1200]
    raise ValueError(f"Unable to parse JSON from the coupling model output. Raw preview: {preview}")

class CouplingAgentAdapter:

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        max_new_tokens: int = 1536,
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

    def _extract_predicted_energy_next_state(self, energy_trajectory: EnergyTrajectory) -> Dict[str, Any]:
        predicted = self._safe_dict(energy_trajectory.optimized_patch_summary)
        if predicted:
            return predicted

        snap = self._safe_dict(energy_trajectory.energy_state_snapshot)
        predicted = self._safe_dict(snap.get("predicted_energy_next_state_summary"))
        if predicted:
            return predicted

        return {
            "transition_type": "unknown",
            "affected_patch_ids": [],
            "affected_heat_ids": [],
            "holder_trend": {},
            "purchase_trend": {},
            "risk_label_changes": [],
            "predicted_patch_stats": [],
            "feasible": True,
            "overall_feasibility": "ok",
            "focus": "",
            "key_risks": [],
            "confidence": 0.0,
            "comment": "",
        }

    def build_input(
        self,
        couple_input: Dict[str, Any],
        candidate_plan: CandidatePlan,
        energy_trajectory: EnergyTrajectory,
    ) -> Dict[str, Any]:
        couple_state = self._safe_dict(couple_input.get("couple_state"))
        couple_prompt_state = self._safe_dict(couple_input.get("couple_prompt_state"))
        meta = self._safe_dict(couple_input.get("meta"))
        memory_state = self._safe_dict(couple_input.get("memory_state"))

        predicted_prod_next_state = self._extract_predicted_prod_next_state(candidate_plan)
        predicted_energy_next_state = self._extract_predicted_energy_next_state(energy_trajectory)

        prod_changed = {
            "affected_heat_ids": self._safe_list(predicted_prod_next_state.get("affected_heat_ids")),
            "affected_rolling_units": self._safe_list(predicted_prod_next_state.get("affected_rolling_units")),
            "transition_type": self._safe_str(predicted_prod_next_state.get("transition_type")),
            "predicted_local_effects": self._safe_list(predicted_prod_next_state.get("predicted_local_effects")),
            "constraint_check": self._safe_dict(predicted_prod_next_state.get("constraint_check")),
        }

        energy_changed_patch_stats = []
        for x in self._safe_list(predicted_energy_next_state.get("predicted_patch_stats")):
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

            if (
                response_type in {"direct", "indirect"}
                or risk_change not in {"", "same", "unknown"}
                or holder_changed
                or purchase_changed
            ):
                energy_changed_patch_stats.append(x)

        energy_changed = {
            "affected_patch_ids": self._safe_list(predicted_energy_next_state.get("affected_patch_ids")),
            "affected_heat_ids": self._safe_list(predicted_energy_next_state.get("affected_heat_ids")),
            "transition_type": self._safe_str(predicted_energy_next_state.get("transition_type")),
            "risk_label_changes": self._safe_list(predicted_energy_next_state.get("risk_label_changes")),
            "changed_patch_stats": energy_changed_patch_stats,
            "holder_trend": self._safe_dict(predicted_energy_next_state.get("holder_trend")),
            "purchase_trend": self._safe_dict(predicted_energy_next_state.get("purchase_trend")),
        }

        bridge_mechanism_view = {
            "prompt_mechanism_summary": self._safe_dict(couple_prompt_state.get("me")),
            "prompt_top_bridge": self._safe_list(couple_prompt_state.get("mb")),
            "full_bridge_mechanism": self._safe_dict(couple_state.get("mec")),
        }

        return {
            "task_instruction": (
                "You are a joint production-energy state consistency evaluator. "
                "Your task is NOT to optimize. "
                "Your task is NOT to recommend dispatching decisions. "
                "Your task is to evaluate whether the predicted production next-state changes "
                "and the predicted energy next-state changes are mutually consistent, feasible, "
                "and interpretable as one-step coupled state transition. "
                "Focus on changed entities only. "
                "Ignore unchanged background entities unless they create conflicts. "
                "Use bridge mechanism features when available, especially regime tag, thermal-demand proxy, switch intensity, energy-tightness, and top bridge examples. "
                "Only output JSON."
            ),
            "input": {
                "day_id": meta.get("d"),
                "transition_task": {
                    "stage": "experiment_3_coupled_state_validation",
                    "goal": "evaluate_consistency_of_changed_prod_and_energy_next_states",
                    "principles": [
                        "compare_changed_sets_first",
                        "check_alignment_between_changed_states",
                        "check_risk_transfer_consistency",
                        "check_feasibility_consistency",
                        "ignore_unchanged_background_when_possible"
                    ],
                },
                "couple_state": couple_state,
                "couple_prompt_state": couple_prompt_state,
                "bridge_mechanism_view": bridge_mechanism_view,
                "production_prediction": {
                    "plan_id": candidate_plan.plan_id,
                    "day_id": candidate_plan.day_id,
                    "task": candidate_plan.task,
                    "mode": candidate_plan.mode,
                    "changed_view": prod_changed,
                },
                "energy_prediction": {
                    "trajectory_id": energy_trajectory.trajectory_id,
                    "plan_id": energy_trajectory.plan_id,
                    "day_id": energy_trajectory.day_id,
                    "task": energy_trajectory.task,
                    "mode": energy_trajectory.mode,
                    "changed_view": energy_changed,
                },
                "feedback_memory": self._safe_list(memory_state.get("feedback_memory"))[-5:],
            },
            "output_schema": {
                "task": "joint_state_consistency_prediction",
                "mode": "joint_state_consistency_evaluator",
                "joint_feasibility": True,
                "economic_score": 0.0,
                "risk_score": 0.0,
                "consistency_score": 0.0,
                "coupling_risks": [],
                "major_conflicts": [],
                "violation_list": [],
                "failure_reasons": [],
                "next_state_summary": {
                    "transition_status": "consistent|partially_consistent|inconsistent",
                    "production_effect_summary": "string",
                    "energy_effect_summary": "string",
                    "joint_risk_summary": "string",
                    "state_update_signals": []
                }
            }
        }

    def build_messages(self, model_input: Dict[str, Any]) -> List[Dict[str, str]]:
        system_prompt = (
            "You are a joint state consistency evaluator.\n"
            "Your job is NOT to optimize.\n"
            "Your job is NOT to recommend scheduling or dispatching decisions.\n"
            "Your job is to evaluate whether the changed production next-state and the changed energy next-state are consistent.\n"
            "\n"
            "Hard rules:\n"
            "1. Compare changed entities first.\n"
            "2. Ignore unchanged background entities unless they create a contradiction.\n"
            "3. Do NOT output optimization advice.\n"
            "4. Do NOT output recommended_joint_decisions.\n"
            "5. Do NOT output revision_suggestions.\n"
            "6. Your main deliverable is consistency judgment and next_state_summary.\n"
            "\n"
            "You must return exactly one JSON object with:\n"
            "- task = joint_state_consistency_prediction\n"
            "- mode = joint_state_consistency_evaluator\n"
            "- joint_feasibility\n"
            "- consistency_score\n"
            "- coupling_risks\n"
            "- major_conflicts\n"
            "- violation_list\n"
            "- failure_reasons\n"
            "- next_state_summary\n"
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

        print("\n=== RAW COUPLING MODEL OUTPUT START ===")
        print(raw_text)
        print("=== RAW COUPLING MODEL OUTPUT END ===\n")

        parsed = _safe_json_loads(raw_text)

        if isinstance(parsed, dict):
            parsed["_raw_text"] = raw_text
            return parsed

        if isinstance(parsed, list):
            return {
                "_raw_text": raw_text,
                "_raw_list_output": parsed,
                "task": "joint_state_consistency_prediction",
                "mode": "joint_state_consistency_evaluator",
                "joint_feasibility": True,
                "economic_score": 0.0,
                "risk_score": 0.0,
                "consistency_score": 0.0,
                "coupling_risks": [],
                "major_conflicts": [],
                "violation_list": [],
                "failure_reasons": [],
                "recommended_joint_decisions": [],
                "revision_suggestions": [],
                "next_state_summary": {
                    "transition_status": "partially_consistent",
                    "production_effect_summary": "",
                    "energy_effect_summary": "",
                    "joint_risk_summary": "",
                    "state_update_signals": [],
                    "raw_coupling_items": parsed
                },
            }

        raise TypeError("The parsed coupling model output is neither a dict nor a list.")

    def _normalize_output(self, raw_output: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(raw_output, dict):
            return {}

        normalized = {
            "task": raw_output.get("task", "joint_state_consistency_prediction"),
            "mode": raw_output.get("mode", "joint_state_consistency_evaluator"),
            "joint_feasibility": bool(raw_output.get("joint_feasibility", raw_output.get("feasibility", True))),
            "economic_score": self._safe_float(raw_output.get("economic_score", 0.0), 0.0),
            "risk_score": self._safe_float(raw_output.get("risk_score", 0.0), 0.0),
            "consistency_score": self._safe_float(raw_output.get("consistency_score", 0.0), 0.0),
            "coupling_risks": self._safe_list(raw_output.get("coupling_risks")),
            "major_conflicts": self._safe_list(raw_output.get("major_conflicts")),
            "violation_list": self._safe_list(raw_output.get("violation_list")),
            "failure_reasons": self._safe_list(raw_output.get("failure_reasons")),

            "recommended_joint_decisions": [],
            "revision_suggestions": [],
            "next_state_summary": self._safe_dict(raw_output.get("next_state_summary")),
        }

        if not normalized["next_state_summary"]:
            normalized["next_state_summary"] = {
                "transition_status": "partially_consistent",
                "production_effect_summary": "",
                "energy_effect_summary": "",
                "joint_risk_summary": "",
                "state_update_signals": [],
            }

        return normalized

    def parse_output(
        self,
        raw_output,
        candidate_plan: CandidatePlan,
        couple_input: Dict[str, Any],
    ) -> EvaluationResult:
        meta = self._safe_dict(couple_input.get("meta"))
        day_id = meta.get("d", "unknown_day")

        if isinstance(raw_output, list):
            normalized_output = {
                "task": "joint_state_consistency_prediction",
                "mode": "joint_state_consistency_evaluator",
                "joint_feasibility": True,
                "economic_score": 0.0,
                "risk_score": 0.0,
                "consistency_score": 0.0,
                "coupling_risks": [],
                "major_conflicts": [],
                "violation_list": [],
                "failure_reasons": [],
                "recommended_joint_decisions": [],
                "revision_suggestions": [],
                "next_state_summary": {
                    "transition_status": "partially_consistent",
                    "production_effect_summary": "",
                    "energy_effect_summary": "",
                    "joint_risk_summary": "",
                    "state_update_signals": [],
                    "raw_coupling_items": raw_output,
                },
                "_raw_list_output": raw_output,
            }
        elif isinstance(raw_output, dict):
            normalized_output = self._normalize_output(raw_output)
        else:
            raise TypeError("The coupling model output is neither a dict nor a list and cannot be parsed.")

        return EvaluationResult(
            plan_id=candidate_plan.plan_id,
            day_id=day_id,
            heat_id=None,
            task=normalized_output.get("task", "joint_state_consistency_prediction"),
            mode=normalized_output.get("mode", "joint_state_consistency_evaluator"),
            joint_feasibility=bool(normalized_output.get("joint_feasibility", True)),
            economic_score=float(normalized_output.get("economic_score", 0.0)),
            risk_score=float(normalized_output.get("risk_score", 0.0)),
            consistency_score=float(normalized_output.get("consistency_score", 0.0)),
            coupling_risks=normalized_output.get("coupling_risks", []),
            major_conflicts=normalized_output.get("major_conflicts", []),
            violation_list=normalized_output.get("violation_list", []),
            failure_reasons=normalized_output.get("failure_reasons", []),
            recommended_joint_decisions=normalized_output.get("recommended_joint_decisions", []),
            revision_suggestions=normalized_output.get("revision_suggestions", []),
            next_state_summary=normalized_output.get("next_state_summary", {}),
            raw_output=normalized_output,
            meta={"source": "coupling_agent_adapter_state_model"},
        )

    def evaluate(
        self,
        couple_input: Dict[str, Any],
        candidate_plan: CandidatePlan,
        energy_trajectory: EnergyTrajectory,
    ) -> EvaluationResult:
        model_input = self.build_input(couple_input, candidate_plan, energy_trajectory)
        raw_output = self.call_model(model_input)
        return self.parse_output(raw_output, candidate_plan, couple_input)

def _caa_copy(obj):
    try:
        return json.loads(json.dumps(obj, ensure_ascii=False, default=str))
    except Exception:
        if isinstance(obj, dict):
            return dict(obj)
        if isinstance(obj, list):
            return list(obj)
        return obj

def _caa_list(x):
    return x if isinstance(x, list) else []

def _caa_dict(x):
    return x if isinstance(x, dict) else {}

def _caa_str(x):
    return "" if x is None else str(x).strip()

def _caa_normalize_heat_id(x):

    if x is None:
        return ""
    s = str(x).strip()
    if not s:
        return ""
    s = s.replace("\\", "/").split("/")[-1].strip()
    s = s.replace(" ", "")
    s = s.replace("-", "_")
    s = s.replace(".", "_")

    for prefix in [
        "Converter1_", "Converter2_", "Converter3_", "Converter4_",
        "converter1_", "converter2_", "converter3_", "converter4_",
        "BOF1_", "BOF2_", "BOF3_", "BOF4_",
        "Furnace1_", "Furnace2_", "Furnace3_", "Furnace4_",
    ]:
        if s.startswith(prefix):
            s = s[len(prefix):]

    matches = re.findall(r"[A-Za-z0-9]*LDG_\d+", s, flags=re.I)
    if matches:
        return matches[-1].upper().replace("LDG_", "LDG_")

    parts = [p for p in s.split("_") if p]
    if len(parts) >= 2 and parts[-2].upper().endswith("LDG") and parts[-1].isdigit():
        return f"{parts[-2].upper()}_{parts[-1]}"

    return s

def _caa_strip_for_coupling(obj):

    banned = {
        "_raw_text", "raw_text", "raw_output", "_raw_list_output",
        "model_input", "prompt", "full_prompt", "messages",
        "comment", "comments", "traceback", "debug", "debug_info",
        "exception", "stack", "stacktrace", "reasoning", "analysis",

        "predicted_operation_segments",
        "derived_operation_segments_from_patches",
        "operation_segments",
        "patch_operation_segments",
        "production_operation_segments",
        "production_patch_operation_segments",
        "history_operation_segments",
        "target_operation_segments",
    }
    if isinstance(obj, dict):
        return {
            str(k): _caa_strip_for_coupling(v)
            for k, v in obj.items()
            if k not in banned
        }
    if isinstance(obj, list):
        return [_caa_strip_for_coupling(x) for x in obj]
    return obj

def _caa_pick_operation_segments(production_prediction):

    if not isinstance(production_prediction, dict):
        return []
    for key in [
        "predicted_operation_segments",
        "derived_operation_segments_from_patches",
        "operation_segments",
        "patch_operation_segments",
    ]:
        val = production_prediction.get(key)
        if isinstance(val, list):
            return val
    raw = production_prediction.get("raw_output")
    if isinstance(raw, dict):
        for key in ["predicted_operation_segments", "operation_segments"]:
            val = raw.get(key)
            if isinstance(val, list):
                return val
    return []

def _caa_operation_segment_summary(production_prediction, max_items=24):

    segs = _caa_pick_operation_segments(production_prediction)
    if not segs:
        return {
            "available": False,
            "num_segments": 0,
            "heat_ids": [],
            "stage_counts": {},
            "time_span": {},
            "examples": [],
        }

    heat_ids = []
    stage_counts = {}
    starts = []
    ends = []
    examples = []

    for seg in segs:
        if not isinstance(seg, dict):
            continue
        hid = _caa_normalize_heat_id(
            seg.get("heat_id") or seg.get("id") or seg.get("canonical_heat_id")
        )
        if hid and hid not in heat_ids:
            heat_ids.append(hid)

        stage = _caa_str(seg.get("stage") or seg.get("process_stage") or seg.get("operation"), "unknown")
        stage_counts[stage] = stage_counts.get(stage, 0) + 1

        s = _caa_float(seg.get("start_min"), None)
        e = _caa_float(seg.get("end_min"), None)
        if s is not None:
            starts.append(s)
        if e is not None:
            ends.append(e)

        if len(examples) < max_items:
            examples.append({
                "heat_id": hid,
                "stage": stage,
                "unit": seg.get("unit") or seg.get("resource") or seg.get("lane"),
                "start_min": seg.get("start_min"),
                "end_min": seg.get("end_min"),
                "source_steps": seg.get("source_steps", []),
            })

    return {
        "available": True,
        "num_segments": len(segs),
        "heat_ids": heat_ids[:max_items],
        "stage_counts": stage_counts,
        "time_span": {
            "start_min": min(starts) if starts else None,
            "end_min": max(ends) if ends else None,
        },
        "examples": examples,
    }

def _caa_bool(x, default=False):
    if isinstance(x, bool):
        return x
    if x is None:
        return default
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"true", "yes", "y", "1", "ok", "feasible", "consistent", "aligned"}:
            return True
        if s in {"false", "no", "n", "0", "none", "infeasible", "inconsistent", "unaligned"}:
            return False
    return default

def _caa_float(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default

def _caa_bool01(x):
    if isinstance(x, bool):
        return 1.0 if x else 0.0
    if isinstance(x, str):
        return 1.0 if x.lower() in {"true", "yes", "ok", "feasible", "consistent", "aligned"} else 0.0
    if isinstance(x, (int, float)):
        return 1.0 if x else 0.0
    return 0.0

def _caa_pick_prediction_steps(prediction):
    if not isinstance(prediction, dict):
        return []
    for key in [
        "predicted_coupling_future",
        "joint_future",
        "future_steps",
        "step_predictions",
    ]:
        val = prediction.get(key)
        if isinstance(val, list):
            return val
    evaluation = prediction.get("coupling_horizon_evaluation")
    if isinstance(evaluation, dict):
        for key in ["future_steps", "step_predictions", "predicted_coupling_future"]:
            val = evaluation.get(key)
            if isinstance(val, list):
                return val
    summary = prediction.get("horizon_summary")
    if isinstance(summary, dict):
        val = summary.get("step_predictions")
        if isinstance(val, list):
            return val
    return []

def _caa_step_key(x):
    try:
        return int(x.get("step"))
    except Exception:
        return None

def _caa_set_from_values(values):
    out = set()
    if isinstance(values, list):
        for v in values:
            s = _caa_normalize_heat_id(v)
            if s:
                out.add(s)
    elif values is not None:
        s = _caa_normalize_heat_id(values)
        if s:
            out.add(s)
    return out

def _caa_extract_pred_heat_ids(step_pred):
    if not isinstance(step_pred, dict):
        return set()
    candidates = []
    for key in [
        "aligned_heat_ids",
        "predicted_aligned_heat_ids",
        "active_heat_ids",
        "production_active_heat_ids",
        "affected_heat_ids",
        "heat_ids",
    ]:
        if key in step_pred:
            candidates.extend(_caa_list(step_pred.get(key)))
    if step_pred.get("main_heat_id") is not None:
        candidates.append(step_pred.get("main_heat_id"))
    if step_pred.get("energy_heat_no") is not None:
        candidates.append(step_pred.get("energy_heat_no"))
    return _caa_set_from_values(candidates)

def _caa_extract_pred_risks(step_pred):
    if not isinstance(step_pred, dict):
        return set()
    for key in ["risk_flags", "predicted_risk_flags", "rk", "key_risks"]:
        if key in step_pred:
            return _caa_set_from_values(step_pred.get(key))
    risk_level = _caa_str(step_pred.get("risk_level"))
    if risk_level and risk_level.lower() not in {"none", "normal", "low", "0"}:
        return {risk_level}
    return set()

def _caa_f1(tp, fp, fn):
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-12)
    return p, r, f1

def _caa_normalize_horizon_output(raw_output):
    if isinstance(raw_output, list):
        return {
            "task": "joint_horizon_coupling_prediction",
            "mode": "joint_horizon_coupling_evaluator",
            "predicted_coupling_future": raw_output,
            "horizon_summary": {},
        }
    if not isinstance(raw_output, dict):
        return {
            "task": "joint_horizon_coupling_prediction",
            "mode": "joint_horizon_coupling_evaluator",
            "predicted_coupling_future": [],
            "horizon_summary": {},
        }

    out = dict(raw_output)
    out.setdefault("task", "joint_horizon_coupling_prediction")
    out.setdefault("mode", "joint_horizon_coupling_evaluator")

    if not isinstance(out.get("predicted_coupling_future"), list):
        steps = _caa_pick_prediction_steps(out)
        out["predicted_coupling_future"] = steps if isinstance(steps, list) else []

    if not isinstance(out.get("horizon_summary"), dict):
        out["horizon_summary"] = {}

    return out

def _caa_build_horizon_input(
    self,
    joint_sample,
    production_prediction=None,
    energy_prediction=None,
):
    input_state = _caa_dict(joint_sample.get("input_state"))
    target_hidden_notice = {
        "target_future_is_hidden": True,
        "evaluation_uses_target_future_offline": True,
    }

    return {
        "task_instruction": (
            "You are a production-energy joint horizon coupling evaluator for steel plant world modeling. "
            "Your task is NOT to optimize and NOT to recommend dispatching decisions. "
            "Given historical joint patch states, known future patch context, a production horizon prediction, "
            "and an energy horizon prediction, evaluate whether the future production-energy joint states are "
            "coupled consistently over the prediction horizon. "
            "Return patch-step-level coupling predictions only as JSON. "
            "A compact predicted production operation-segment summary may be provided for time-context only; "
            "do not require or output full Gantt segments in coupling prediction."
        ),
        "input": {
            "sample_id": joint_sample.get("sample_id"),
            "day_id": joint_sample.get("day_id"),
            "anchor_patch_idx": joint_sample.get("anchor_patch_idx"),
            "history_window": joint_sample.get("history_window"),
            "horizon": joint_sample.get("horizon"),
            "joint_history": _caa_list(input_state.get("joint_history")),
            "future_context": _caa_list(input_state.get("future_context")),
            "production_prediction": _caa_strip_for_coupling(production_prediction) if isinstance(production_prediction, dict) else {},
            "production_operation_segment_summary": _caa_operation_segment_summary(production_prediction),
            "energy_prediction": _caa_strip_for_coupling(energy_prediction) if isinstance(energy_prediction, dict) else {},
            "target_policy": target_hidden_notice,
            "evaluation_focus": [
                "heat_patch_alignment",
                "production_energy_response_consistency",
                "risk_transfer_consistency",
                "future_joint_feasibility",
                "locality_of_energy_response",
            ],
        },
        "output_schema": {
            "task": "joint_horizon_coupling_prediction",
            "mode": "joint_horizon_coupling_evaluator",
            "predicted_coupling_future": [
                {
                    "step": 1,
                    "energy_patch_id": "string",
                    "energy_heat_no": "string",
                    "aligned_heat_ids": [],
                    "response_type": "direct|indirect|stable|unknown",
                    "risk_flags": [],
                    "risk_level": "none|low|medium|high|unknown",
                    "feasible": True,
                    "consistency_score": 0.0,
                    "conflicts": [],
                    "comment": "string",
                }
            ],
            "horizon_summary": {
                "mean_consistency": 0.0,
                "feasibility_rate": 0.0,
                "risk_trend": "flat|increasing|decreasing|unknown",
                "major_conflicts": [],
                "comment": "string",
            }
        },
    }

def _caa_build_horizon_messages(self, model_input):
    system_prompt = (
        "You are a joint production-energy coupling evaluator for patch-level horizon prediction.\n"
        "Do NOT optimize. Do NOT recommend scheduling decisions. Do NOT reveal hidden labels.\n"
        "Evaluate future production-energy coupling consistency step by step.\n"
        "Return exactly one valid JSON object with task, mode, predicted_coupling_future and horizon_summary.\n"
        "Do not output markdown or text outside JSON."
    )
    user_prompt = json.dumps(model_input, ensure_ascii=False, indent=2)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

def _caa_call_horizon_model(self, model_input):
    messages = self.build_horizon_messages(model_input)
    try:
        raw_text = self.chat_model.generate(messages)
    except Exception as e:
        return {
            "_raw_text": "",
            "task": "joint_horizon_coupling_prediction",
            "mode": "model_exception_fallback",
            "predicted_coupling_future": [],
            "horizon_summary": {
                "mean_consistency": 0.0,
                "feasibility_rate": 0.0,
                "risk_trend": "unknown",
                "major_conflicts": [f"coupling_model_exception:{type(e).__name__}"],
                "comment": str(e),
            },
        }

    print("\n=== RAW COUPLING HORIZON MODEL OUTPUT START ===")
    print(raw_text)
    print("=== RAW COUPLING HORIZON MODEL OUTPUT END ===\n")

    if not raw_text or not str(raw_text).strip():
        return {
            "_raw_text": raw_text or "",
            "task": "joint_horizon_coupling_prediction",
            "mode": "empty_output_fallback",
            "predicted_coupling_future": [],
            "horizon_summary": {
                "mean_consistency": 0.0,
                "feasibility_rate": 0.0,
                "risk_trend": "unknown",
                "major_conflicts": ["empty_model_output"],
                "comment": "Coupling horizon model returned empty output.",
            },
        }

    try:
        parsed = _safe_json_loads(str(raw_text))
    except Exception as e:
        return {
            "_raw_text": str(raw_text),
            "task": "joint_horizon_coupling_prediction",
            "mode": "json_parse_fallback",
            "predicted_coupling_future": [],
            "horizon_summary": {
                "mean_consistency": 0.0,
                "feasibility_rate": 0.0,
                "risk_trend": "unknown",
                "major_conflicts": [f"json_parse_error:{type(e).__name__}"],
                "comment": str(e),
            },
        }

    if isinstance(parsed, dict):
        parsed["_raw_text"] = raw_text
        return parsed
    if isinstance(parsed, list):
        return {
            "_raw_text": raw_text,
            "task": "joint_horizon_coupling_prediction",
            "mode": "joint_horizon_coupling_evaluator",
            "predicted_coupling_future": parsed,
            "horizon_summary": {},
        }
    return {
        "_raw_text": str(raw_text),
        "task": "joint_horizon_coupling_prediction",
        "mode": "type_fallback",
        "predicted_coupling_future": [],
        "horizon_summary": {
            "mean_consistency": 0.0,
            "feasibility_rate": 0.0,
            "risk_trend": "unknown",
            "major_conflicts": ["parsed_output_not_dict_or_list"],
            "comment": "Coupling horizon model output parsed to unsupported type.",
        },
    }

def _caa_predict_horizon(
    self,
    joint_sample,
    production_prediction=None,
    energy_prediction=None,
):
    model_input = self.build_horizon_input(
        joint_sample=joint_sample,
        production_prediction=production_prediction,
        energy_prediction=energy_prediction,
    )
    raw_output = self.call_horizon_model(model_input)
    return _caa_normalize_horizon_output(raw_output)

def _caa_pick_target_pairs(target_future=None, joint_sample=None):

    source = {}

    if isinstance(joint_sample, dict):
        ct = joint_sample.get("coupling_target")
        if isinstance(ct, dict) and isinstance(ct.get("future_pairs"), list):
            source = ct

        tf = joint_sample.get("target_future")
        if not source and isinstance(tf, dict):
            nested = tf.get("coupling_target")
            if isinstance(nested, dict) and isinstance(nested.get("future_pairs"), list):
                source = nested

    if not source and isinstance(target_future, dict):
        nested = target_future.get("coupling_target")
        if isinstance(nested, dict) and isinstance(nested.get("future_pairs"), list):
            source = nested
        elif isinstance(target_future.get("future_pairs"), list):
            source = target_future

    if source:
        pairs = []
        for item in _caa_list(source.get("future_pairs")):
            if not isinstance(item, dict):
                continue
            active = (
                item.get("canonical_active_heat_ids")
                or item.get("target_active_heat_ids")
                or item.get("production_active_heat_ids")
                or []
            )
            pairs.append({
                "step": int(item.get("step", len(pairs) + 1)),
                "energy_patch_id": item.get("energy_patch_id") or item.get("patch_id"),
                "energy_heat_no": _caa_normalize_heat_id(
                    item.get("canonical_energy_heat_id")
                    or item.get("energy_heat_no")
                    or item.get("energy_heat_id")
                ),
                "production_active_heat_ids": sorted(_caa_set_from_values(active)),
                "aligned": _caa_bool(item.get("is_aligned", item.get("aligned", False))),
                "risk_flags": sorted(_caa_set_from_values(item.get("risk_flags"))),
                "has_risk": _caa_bool(item.get("has_risk", False)),
            })
        return pairs

    target_steps = _caa_list(_caa_dict(target_future).get("future_steps"))
    pairs = []
    for item in target_steps:
        if not isinstance(item, dict):
            continue
        c = _caa_dict(item.get("coupling"))
        pairs.append({
            "step": int(item.get("step", len(pairs) + 1)),
            "energy_patch_id": c.get("energy_patch_id"),
            "energy_heat_no": _caa_normalize_heat_id(
                c.get("canonical_energy_heat_id")
                or c.get("energy_heat_no")
            ),
            "production_active_heat_ids": sorted(_caa_set_from_values(
                c.get("canonical_active_heat_ids")
                or c.get("production_active_heat_ids")
            )),
            "aligned": _caa_bool(c.get("is_aligned", c.get("aligned", False))),
            "risk_flags": sorted(_caa_set_from_values(c.get("risk_flags"))),
            "has_risk": _caa_bool(c.get("has_risk", False)),
        })
    return pairs

def _caa_evaluate_horizon_prediction(
    self,
    prediction,
    target_future=None,
    joint_sample=None,
    production_prediction=None,
    energy_prediction=None,
):

    pred = _caa_normalize_horizon_output(prediction)
    pred_steps = _caa_pick_prediction_steps(pred)

    target_pairs = _caa_pick_target_pairs(
        target_future=target_future,
        joint_sample=joint_sample,
    )

    target_by_step = {}
    for item in target_pairs:
        if isinstance(item, dict):
            step = _caa_step_key(item)
            if step is not None:
                target_by_step[step] = item

    pred_by_step = {}
    for item in pred_steps:
        if isinstance(item, dict):
            step = _caa_step_key(item)
            if step is not None:
                pred_by_step[step] = item

    n = len(target_by_step)
    if n == 0:
        return {
            "num_target_steps": 0,
            "num_predicted_steps": len(pred_by_step),
            "coverage": 0.0,
            "heat_patch_alignment_accuracy": 0.0,
            "risk_precision": 0.0,
            "risk_recall": 0.0,
            "risk_f1": 0.0,
            "risk_precision_excluding_ov": 0.0,
            "risk_recall_excluding_ov": 0.0,
            "risk_f1_excluding_ov": 0.0,
            "purchase_risk_precision": 0.0,
            "purchase_risk_recall": 0.0,
            "purchase_risk_f1": 0.0,
            "holder_limit_risk_precision": 0.0,
            "holder_limit_risk_recall": 0.0,
            "holder_limit_risk_f1": 0.0,
            "mean_consistency": 0.0,
            "feasibility_rate": 0.0,
            "comment": "No coupling target pairs available.",
        }

    covered = 0
    align_correct = 0
    feasible_vals = []
    consistency_vals = []

    risk_tp = risk_fp = risk_fn = 0
    risk_no_ov_tp = risk_no_ov_fp = risk_no_ov_fn = 0
    purchase_tp = purchase_fp = purchase_fn = 0
    holder_tp = holder_fp = holder_fn = 0
    purchase_flags = {"pp", "ps"}
    holder_flags = {"b_low", "b_high", "c_low", "c_high", "l_low", "l_high"}
    missing_steps = []

    for step, tgt in sorted(target_by_step.items()):
        step_pred = pred_by_step.get(step, {})
        has_pred = isinstance(step_pred, dict) and bool(step_pred)
        if has_pred:
            covered += 1
        else:
            missing_steps.append(step)

        tgt_energy_heat = _caa_normalize_heat_id(tgt.get("energy_heat_no"))
        tgt_prod_heats = _caa_set_from_values(tgt.get("production_active_heat_ids"))
        tgt_aligned = _caa_bool(tgt.get("aligned", False))

        pred_heats = _caa_extract_pred_heat_ids(step_pred)

        pred_aligned = None
        if isinstance(step_pred, dict):
            if "aligned" in step_pred:
                pred_aligned = _caa_bool(step_pred.get("aligned"), default=None)
            elif "heat_patch_aligned" in step_pred:
                pred_aligned = _caa_bool(step_pred.get("heat_patch_aligned"), default=None)
            elif "is_aligned" in step_pred:
                pred_aligned = _caa_bool(step_pred.get("is_aligned"), default=None)

        if pred_aligned is None:
            pred_aligned = bool(
                (tgt_energy_heat and tgt_energy_heat in pred_heats)
                or len(pred_heats & tgt_prod_heats) > 0
            )

        if bool(pred_aligned) == bool(tgt_aligned):
            align_correct += 1

        tgt_risks = _caa_set_from_values(tgt.get("risk_flags"))
        pred_risks = _caa_extract_pred_risks(step_pred)

        risk_tp += len(tgt_risks & pred_risks)
        risk_fp += len(pred_risks - tgt_risks)
        risk_fn += len(tgt_risks - pred_risks)

        tgt_no_ov = {x for x in tgt_risks if x != "ov"}
        pred_no_ov = {x for x in pred_risks if x != "ov"}
        risk_no_ov_tp += len(tgt_no_ov & pred_no_ov)
        risk_no_ov_fp += len(pred_no_ov - tgt_no_ov)
        risk_no_ov_fn += len(tgt_no_ov - pred_no_ov)

        tgt_purchase = tgt_risks & purchase_flags
        pred_purchase = pred_risks & purchase_flags
        purchase_tp += len(tgt_purchase & pred_purchase)
        purchase_fp += len(pred_purchase - tgt_purchase)
        purchase_fn += len(tgt_purchase - pred_purchase)

        tgt_holder = tgt_risks & holder_flags
        pred_holder = pred_risks & holder_flags
        holder_tp += len(tgt_holder & pred_holder)
        holder_fp += len(pred_holder - tgt_holder)
        holder_fn += len(tgt_holder - pred_holder)

        if has_pred:
            feasible_vals.append(_caa_bool01(step_pred.get("feasible", True)))
            consistency_vals.append(_caa_float(step_pred.get("consistency_score", 0.0), 0.0))

    rp, rr, rf1 = _caa_f1(risk_tp, risk_fp, risk_fn)
    rp_no_ov, rr_no_ov, rf1_no_ov = _caa_f1(risk_no_ov_tp, risk_no_ov_fp, risk_no_ov_fn)
    pp_p, pp_r, pp_f1 = _caa_f1(purchase_tp, purchase_fp, purchase_fn)
    hp_p, hp_r, hp_f1 = _caa_f1(holder_tp, holder_fp, holder_fn)

    return {
        "num_target_steps": n,
        "num_predicted_steps": len(pred_by_step),
        "coverage": covered / max(n, 1),
        "heat_patch_alignment_accuracy": align_correct / max(n, 1),
        "risk_precision": rp,
        "risk_recall": rr,
        "risk_f1": rf1,
        "risk_precision_excluding_ov": rp_no_ov,
        "risk_recall_excluding_ov": rr_no_ov,
        "risk_f1_excluding_ov": rf1_no_ov,
        "purchase_risk_precision": pp_p,
        "purchase_risk_recall": pp_r,
        "purchase_risk_f1": pp_f1,
        "holder_limit_risk_precision": hp_p,
        "holder_limit_risk_recall": hp_r,
        "holder_limit_risk_f1": hp_f1,
        "risk_no_ov_tp": risk_no_ov_tp,
        "risk_no_ov_fp": risk_no_ov_fp,
        "risk_no_ov_fn": risk_no_ov_fn,
        "purchase_risk_tp": purchase_tp,
        "purchase_risk_fp": purchase_fp,
        "purchase_risk_fn": purchase_fn,
        "holder_limit_risk_tp": holder_tp,
        "holder_limit_risk_fp": holder_fp,
        "holder_limit_risk_fn": holder_fn,
        "mean_consistency": sum(consistency_vals) / max(len(consistency_vals), 1),
        "feasibility_rate": sum(feasible_vals) / max(len(feasible_vals), 1),
        "risk_tp": risk_tp,
        "risk_fp": risk_fp,
        "risk_fn": risk_fn,
        "missing_steps": missing_steps[:20],
    }

def _caa_predict_and_evaluate_horizon(
    self,
    joint_sample,
    production_prediction=None,
    energy_prediction=None,
):
    prediction = self.predict_horizon(
        joint_sample=joint_sample,
        production_prediction=production_prediction,
        energy_prediction=energy_prediction,
    )
    metrics = {}
    if isinstance(joint_sample, dict) and isinstance(joint_sample.get("target_future"), dict):
        metrics = self.evaluate_horizon_prediction(
            prediction=prediction,
            target_future=joint_sample.get("target_future"),
            joint_sample=joint_sample,
            production_prediction=production_prediction,
            energy_prediction=energy_prediction,
        )
    return {
        "prediction": prediction,
        "metrics": metrics,
    }

CouplingAgentAdapter.build_horizon_input = _caa_build_horizon_input
CouplingAgentAdapter.build_horizon_messages = _caa_build_horizon_messages
CouplingAgentAdapter.call_horizon_model = _caa_call_horizon_model
CouplingAgentAdapter.predict_horizon = _caa_predict_horizon
CouplingAgentAdapter.evaluate_horizon_prediction = _caa_evaluate_horizon_prediction
CouplingAgentAdapter.predict_and_evaluate_horizon = _caa_predict_and_evaluate_horizon

def _caa_str(x, default=""):
    if x is None:
        return default
    try:
        s = str(x).strip()
        return s if s else default
    except Exception:
        return default

def _caa_canonical_stage(x):
    s = _caa_str(x, "Other")
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

def _caa_pick_operation_segments(production_prediction):

    if not isinstance(production_prediction, dict):
        return [], "missing"

    source_flag = _caa_str(production_prediction.get("operation_segment_source"), "")

    val = production_prediction.get("predicted_operation_segments")
    if isinstance(val, list) and val:
        if source_flag:
            return val, source_flag
        return val, "predicted_operation_segments"

    raw = production_prediction.get("raw_output")
    if isinstance(raw, dict):
        val = raw.get("predicted_operation_segments")
        if isinstance(val, list) and val:
            return val, "raw_output.predicted_operation_segments"

    val = production_prediction.get("derived_operation_segments_from_patches")
    if isinstance(val, list) and val:
        return val, "derived_from_future_production_states"

    for key in ["operation_segments", "patch_operation_segments"]:
        val = production_prediction.get(key)
        if isinstance(val, list) and val:
            return val, key

    return [], "missing"

def _caa_operation_segment_summary(production_prediction, max_items=16):

    segs, seg_source = _caa_pick_operation_segments(production_prediction)

    summary = {
        "available": False,
        "segment_source": seg_source,
        "trust_level": "none",
        "num_segments": 0,
        "heat_ids": [],
        "stage_counts": {},
        "unit_counts": {},
        "time_span": {},
        "source_counts": {},
        "mean_confidence": None,
        "examples": [],
    }

    if not segs:
        return summary

    heat_ids = []
    stage_counts = {}
    unit_counts = {}
    source_counts = {}
    starts = []
    ends = []
    confs = []
    examples = []

    for seg in segs:
        if not isinstance(seg, dict):
            continue

        hid = _caa_normalize_heat_id(
            seg.get("heat_id") or seg.get("id") or seg.get("canonical_heat_id") or seg.get("hn")
        )
        if hid and hid not in heat_ids:
            heat_ids.append(hid)

        stage = _caa_canonical_stage(seg.get("stage") or seg.get("process_stage") or seg.get("operation"))
        stage_counts[stage] = stage_counts.get(stage, 0) + 1

        unit = _caa_str(seg.get("unit") or seg.get("resource") or seg.get("lane") or seg.get("ru"), "")
        if unit:
            unit_counts[unit] = unit_counts.get(unit, 0) + 1

        s = _caa_float(seg.get("start_min"), None)
        e = _caa_float(seg.get("end_min"), None)
        if s is not None:
            starts.append(s)
        if e is not None:
            ends.append(e)

        conf = _caa_float(seg.get("confidence"), None)
        if conf is not None:
            confs.append(conf)

        src = _caa_str(seg.get("source"), seg_source)
        source_counts[src] = source_counts.get(src, 0) + 1

        if len(examples) < max_items:
            examples.append({
                "heat_id": hid,
                "stage": stage,
                "unit": unit or None,
                "start_min": seg.get("start_min"),
                "end_min": seg.get("end_min"),
                "source_steps": _caa_list(seg.get("source_steps")),
                "confidence": seg.get("confidence"),
                "source": src,
            })

    if "model_output" in seg_source or seg_source in {"predicted_operation_segments", "raw_output.predicted_operation_segments"}:
        trust = "model_output"
    elif "derived" in seg_source:
        trust = "derived_low_trust"
    else:
        trust = "unknown"

    summary.update({
        "available": True,
        "trust_level": trust,
        "num_segments": len(segs),
        "heat_ids": heat_ids[:max_items],
        "stage_counts": stage_counts,
        "unit_counts": unit_counts,
        "time_span": {
            "start_min": min(starts) if starts else None,
            "end_min": max(ends) if ends else None,
        },
        "source_counts": source_counts,
        "mean_confidence": (sum(confs) / max(len(confs), 1)) if confs else None,
        "examples": examples,
    })
    return summary

def _caa_build_horizon_input(
    self,
    joint_sample,
    production_prediction=None,
    energy_prediction=None,
):
    input_state = _caa_dict(joint_sample.get("input_state"))
    target_hidden_notice = {
        "target_future_is_hidden": True,
        "evaluation_uses_target_future_offline": True,
        "full_operation_segments_target_is_hidden": True,
    }

    operation_schedule_context = _caa_dict(input_state.get("production_operation_schedule_context"))
    operation_segment_summary = _caa_operation_segment_summary(production_prediction)

    return {
        "task_instruction": (
            "You are a production-energy joint horizon coupling evaluator for steel plant world modeling. "
            "Your task is NOT to optimize and NOT to recommend dispatching decisions. "
            "Given historical joint patch states, known future patch context, a production horizon prediction, "
            "an optional compact production operation-segment summary, and an energy horizon prediction, "
            "evaluate whether the future production-energy joint states are coupled consistently over the horizon. "
            "Return patch-step-level coupling predictions only as JSON. "
            "Use operation-segment summary only as time-context; do not output Gantt segments and do not require full segments."
        ),
        "input": {
            "sample_id": joint_sample.get("sample_id"),
            "day_id": joint_sample.get("day_id"),
            "anchor_patch_idx": joint_sample.get("anchor_patch_idx"),
            "history_window": joint_sample.get("history_window"),
            "horizon": joint_sample.get("horizon"),
            "joint_history": _caa_list(input_state.get("joint_history")),
            "future_context": _caa_list(input_state.get("future_context")),
            "operation_schedule_context": operation_schedule_context,
            "production_prediction": _caa_strip_for_coupling(production_prediction) if isinstance(production_prediction, dict) else {},
            "production_operation_segment_summary": operation_segment_summary,
            "energy_prediction": _caa_strip_for_coupling(energy_prediction) if isinstance(energy_prediction, dict) else {},
            "target_policy": target_hidden_notice,
            "evaluation_focus": [
                "heat_patch_alignment",
                "operation_time_context_alignment",
                "production_energy_response_consistency",
                "risk_transfer_consistency",
                "future_joint_feasibility",
                "locality_of_energy_response",
            ],
            "operation_segment_policy": {
                "full_gantt_segments_are_not_required_by_coupling": True,
                "if_segment_summary_trust_level_is_derived_low_trust": "use only as weak context, not as proof of production schedule correctness",
                "coupling_output_should_remain_patch_step_level": True,
            },
        },
        "output_schema": {
            "task": "joint_horizon_coupling_prediction",
            "mode": "joint_horizon_coupling_evaluator",
            "predicted_coupling_future": [
                {
                    "step": 1,
                    "energy_patch_id": "string",
                    "energy_heat_no": "string",
                    "aligned_heat_ids": [],
                    "response_type": "direct|indirect|stable|unknown",
                    "risk_flags": [],
                    "risk_level": "none|low|medium|high|unknown",
                    "feasible": True,
                    "consistency_score": 0.0,
                    "schedule_energy_consistency": "consistent|partially_consistent|inconsistent|unknown",
                    "conflicts": [],
                    "comment": "string",
                }
            ],
            "horizon_summary": {
                "mean_consistency": 0.0,
                "feasibility_rate": 0.0,
                "risk_trend": "flat|increasing|decreasing|unknown",
                "schedule_energy_summary": "string",
                "major_conflicts": [],
                "comment": "string",
            }
        },
    }

def _caa_build_horizon_messages(self, model_input):
    system_prompt = (
        "You are a joint production-energy coupling evaluator for patch-level horizon prediction.\n"
        "Do NOT optimize. Do NOT recommend scheduling decisions. Do NOT reveal hidden labels.\n"
        "Evaluate future production-energy coupling consistency step by step.\n"
        "Full production Gantt segments are not part of your output. If a compact operation-segment summary is present, "
        "use it only as time-context for checking whether energy response timing is reasonable.\n"
        "Return exactly one valid JSON object with task, mode, predicted_coupling_future and horizon_summary.\n"
        "Do not output markdown or text outside JSON."
    )
    user_prompt = json.dumps(model_input, ensure_ascii=False, indent=2)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

CouplingAgentAdapter.build_horizon_input = _caa_build_horizon_input
CouplingAgentAdapter.build_horizon_messages = _caa_build_horizon_messages

def _caa_segment_is_valid_model_output_v2(seg):
    if not isinstance(seg, dict):
        return False

    source = _caa_str(seg.get("source") or seg.get("source_kind") or seg.get("operation_segment_source"), "")
    if source == "model_output":
        return True

    valid_flag = seg.get("is_valid_gantt_prediction")
    if isinstance(valid_flag, bool) and valid_flag:
        return True

    lowered = source.lower()
    if any(x in lowered for x in ["derived", "fallback", "calendar", "patch"]):
        return False

    return False

def _caa_segment_is_diagnostic_v2(seg):
    if not isinstance(seg, dict):
        return False
    source = _caa_str(seg.get("source") or seg.get("source_kind") or seg.get("operation_segment_source"), "").lower()
    if any(x in source for x in ["derived", "fallback", "calendar", "patch"]):
        return True
    if seg.get("is_valid_gantt_prediction") is False:
        return True
    return False

def _caa_normalize_segment_for_summary_v2(seg):
    if not isinstance(seg, dict):
        return {}
    hid = _caa_normalize_heat_id(
        seg.get("heat_id") or seg.get("id") or seg.get("canonical_heat_id") or seg.get("hn")
    )
    stage = _caa_canonical_stage(seg.get("stage") or seg.get("process_stage") or seg.get("operation"))
    unit = _caa_str(seg.get("unit") or seg.get("resource") or seg.get("lane") or seg.get("ru"), "")
    start_min = _caa_float(seg.get("start_min"), None)
    end_min = _caa_float(seg.get("end_min"), None)
    return {
        "heat_id": hid,
        "stage": stage,
        "unit": unit or None,
        "start_min": start_min,
        "end_min": end_min,
        "source_steps": _caa_list(seg.get("source_steps")),
        "confidence": seg.get("confidence"),
        "source": _caa_str(seg.get("source") or seg.get("source_kind") or seg.get("operation_segment_source"), ""),
    }

def _caa_collect_operation_segments_v2(production_prediction):

    valid = []
    diagnostic = []
    source_label = "missing_model_operation_segments"
    diagnostic_label = "none"

    if not isinstance(production_prediction, dict):
        return valid, diagnostic, source_label, diagnostic_label

    raw_predicted = production_prediction.get("predicted_operation_segments")
    if isinstance(raw_predicted, list):
        for seg in raw_predicted:
            if _caa_segment_is_valid_model_output_v2(seg):
                valid.append(seg)
            else:
                diagnostic.append(seg)
        if valid:
            source_label = "model_output"
        elif raw_predicted:
            source_label = "predicted_operation_segments_without_model_output_source"

    raw = production_prediction.get("raw_output")
    if not valid and isinstance(raw, dict):
        raw_segments = raw.get("predicted_operation_segments")
        if isinstance(raw_segments, list):
            for seg in raw_segments:
                if _caa_segment_is_valid_model_output_v2(seg):
                    valid.append(seg)
                else:
                    diagnostic.append(seg)
            if valid:
                source_label = "model_output"

    for key in [
        "derived_operation_segments_from_patches",
        "fallback_operation_segments",
        "operation_segments",
        "patch_operation_segments",
        "production_patch_operation_segments",
    ]:
        segs = production_prediction.get(key)
        if isinstance(segs, list) and segs:
            diagnostic.extend(segs)
            if diagnostic_label == "none":
                diagnostic_label = key

    if diagnostic and diagnostic_label == "none":
        diagnostic_label = "non_model_output_segments"

    return valid, diagnostic, source_label, diagnostic_label

def _caa_pick_operation_segments(production_prediction):

    valid, diagnostic, source_label, diagnostic_label = _caa_collect_operation_segments_v2(production_prediction)
    if valid:
        return valid, source_label
    if diagnostic:
        return [], f"missing_model_output_with_diagnostic:{diagnostic_label}"
    return [], source_label

def _caa_operation_segment_summary(production_prediction, max_items=16):

    valid, diagnostic, source_label, diagnostic_label = _caa_collect_operation_segments_v2(production_prediction)

    summary = {
        "available": bool(valid),
        "segment_source": source_label,
        "trust_level": "model_output" if valid else "missing_or_diagnostic_only",
        "num_segments": len(valid),
        "valid_model_segment_count": len(valid),
        "diagnostic_segment_count": len(diagnostic),
        "diagnostic_segment_source": diagnostic_label,
        "rejected_or_non_model_segment_count": len(diagnostic),
        "heat_ids": [],
        "stage_counts": {},
        "unit_counts": {},
        "time_span": {},
        "source_counts": {},
        "mean_confidence": None,
        "examples": [],
        "diagnostic_examples": [],
        "policy": {
            "only_model_output_segments_are_trusted": True,
            "derived_or_fallback_segments_are_diagnostics_only": True,
            "coupling_must_not_treat_diagnostic_segments_as_successful_gantt_prediction": True,
        },
    }

    def _accumulate(segs, example_key, limit, trusted):
        heat_ids = []
        stage_counts = {}
        unit_counts = {}
        source_counts = {}
        starts = []
        ends = []
        confs = []
        examples = []

        for seg in segs:
            if not isinstance(seg, dict):
                continue
            row = _caa_normalize_segment_for_summary_v2(seg)

            hid = row.get("heat_id")
            if hid and hid not in heat_ids:
                heat_ids.append(hid)

            stage = row.get("stage") or "Other"
            stage_counts[stage] = stage_counts.get(stage, 0) + 1

            unit = row.get("unit")
            if unit:
                unit_counts[unit] = unit_counts.get(unit, 0) + 1

            s = row.get("start_min")
            e = row.get("end_min")
            if s is not None:
                starts.append(s)
            if e is not None:
                ends.append(e)

            conf = _caa_float(row.get("confidence"), None)
            if conf is not None:
                confs.append(conf)

            src = row.get("source") or ("model_output" if trusted else "diagnostic")
            source_counts[src] = source_counts.get(src, 0) + 1

            if len(examples) < limit:
                examples.append(row)

        return {
            "heat_ids": heat_ids[:limit],
            "stage_counts": stage_counts,
            "unit_counts": unit_counts,
            "time_span": {
                "start_min": min(starts) if starts else None,
                "end_min": max(ends) if ends else None,
            },
            "source_counts": source_counts,
            "mean_confidence": (sum(confs) / max(len(confs), 1)) if confs else None,
            example_key: examples,
        }

    if valid:
        acc = _accumulate(valid, "examples", max_items, trusted=True)
        summary.update(acc)

    if diagnostic:
        diag_acc = _accumulate(diagnostic, "diagnostic_examples", min(6, max_items), trusted=False)
        summary["diagnostic_summary"] = {
            "heat_ids": diag_acc.get("heat_ids", []),
            "stage_counts": diag_acc.get("stage_counts", {}),
            "time_span": diag_acc.get("time_span", {}),
            "source_counts": diag_acc.get("source_counts", {}),
        }
        summary["diagnostic_examples"] = diag_acc.get("diagnostic_examples", [])

    return summary

def _caa_strip_for_coupling(obj):

    banned = {
        "_raw_text", "raw_text", "raw_output", "_raw_list_output",
        "model_input", "prompt", "full_prompt", "messages",
        "comment", "comments", "traceback", "debug", "debug_info",
        "exception", "stack", "stacktrace", "reasoning", "analysis",
        "predicted_operation_segments",
        "raw_predicted_operation_segments",
        "derived_operation_segments_from_patches",
        "derived_operation_segments",
        "fallback_operation_segments",
        "operation_segments",
        "patch_operation_segments",
        "production_operation_segments",
        "production_patch_operation_segments",
        "history_operation_segments",
        "visible_history_operation_segments",
        "target_operation_segments",
        "target_future",
    }
    if isinstance(obj, dict):
        return {
            str(k): _caa_strip_for_coupling(v)
            for k, v in obj.items()
            if k not in banned
        }
    if isinstance(obj, list):
        return [_caa_strip_for_coupling(x) for x in obj]
    return obj

def _caa_build_horizon_input(
    self,
    joint_sample,
    production_prediction=None,
    energy_prediction=None,
):
    input_state = _caa_dict(joint_sample.get("input_state"))
    operation_schedule_context = _caa_dict(input_state.get("production_operation_schedule_context"))
    operation_segment_summary = _caa_operation_segment_summary(production_prediction)

    target_hidden_notice = {
        "target_future_is_hidden": True,
        "evaluation_uses_target_future_offline": True,
        "full_operation_segments_target_is_hidden": True,
        "coupling_model_must_not_use_hidden_targets": True,
    }

    return {
        "task_instruction": (
            "You are a production-energy joint horizon coupling evaluator for steel plant world modeling. "
            "Your task is NOT to optimize and NOT to recommend dispatching decisions. "
            "Evaluate whether the future production-energy patch states are coupled consistently over the horizon. "
            "A compact production operation-segment summary may be provided only as time-context. "
            "Trust that summary only when trust_level=model_output. "
            "If operation segments are missing or diagnostic-only, report lower confidence rather than treating them as valid Gantt prediction. "
            "Return patch-step-level coupling predictions only as JSON."
        ),
        "input": {
            "sample_id": joint_sample.get("sample_id"),
            "day_id": joint_sample.get("day_id"),
            "anchor_patch_idx": joint_sample.get("anchor_patch_idx"),
            "history_window": joint_sample.get("history_window"),
            "horizon": joint_sample.get("horizon"),
            "joint_history": _caa_list(input_state.get("joint_history")),
            "future_context": _caa_list(input_state.get("future_context")),
            "operation_schedule_context": operation_schedule_context,
            "production_prediction": _caa_strip_for_coupling(production_prediction) if isinstance(production_prediction, dict) else {},
            "production_operation_segment_summary": operation_segment_summary,
            "energy_prediction": _caa_strip_for_coupling(energy_prediction) if isinstance(energy_prediction, dict) else {},
            "target_policy": target_hidden_notice,
            "evaluation_focus": [
                "heat_patch_alignment",
                "operation_time_context_alignment",
                "production_energy_response_consistency",
                "risk_transfer_consistency",
                "future_joint_feasibility",
                "locality_of_energy_response",
            ],
            "operation_segment_policy": {
                "full_gantt_segments_are_not_required_by_coupling": True,
                "valid_gantt_prediction_requires_source_model_output": True,
                "derived_or_fallback_segments_are_diagnostics_only": True,
                "if_operation_segment_summary_available_is_false": (
                    "do not infer successful production schedule prediction; evaluate patch-level coupling only and report confidence limitation"
                ),
                "coupling_output_should_remain_patch_step_level": True,
            },
        },
        "output_schema": {
            "task": "joint_horizon_coupling_prediction",
            "mode": "joint_horizon_coupling_evaluator",
            "predicted_coupling_future": [
                {
                    "step": 1,
                    "energy_patch_id": "string",
                    "energy_heat_no": "string",
                    "aligned_heat_ids": [],
                    "response_type": "direct|indirect|stable|unknown",
                    "risk_flags": [],
                    "risk_level": "none|low|medium|high|unknown",
                    "feasible": True,
                    "consistency_score": 0.0,
                    "schedule_energy_consistency": "consistent|partially_consistent|inconsistent|unknown",
                    "confidence": 0.0,
                    "conflicts": [],
                    "comment": "string",
                }
            ],
            "horizon_summary": {
                "mean_consistency": 0.0,
                "feasibility_rate": 0.0,
                "risk_trend": "flat|increasing|decreasing|unknown",
                "schedule_energy_summary": "string",
                "major_conflicts": [],
                "confidence_limitation": "string",
                "comment": "string",
            },
        },
    }

def _caa_build_horizon_messages(self, model_input):
    system_prompt = (
        "You are a joint production-energy coupling evaluator for patch-level horizon prediction.\n"
        "Do NOT optimize. Do NOT recommend scheduling decisions. Do NOT reveal hidden labels.\n"
        "Evaluate future production-energy coupling consistency step by step.\n"
        "Full production Gantt segments are not part of your output.\n"
        "A compact operation-segment summary may be present. Use it only as timing context, and only trust it when trust_level=model_output.\n"
        "If production operation segments are missing or diagnostic-only, do not claim the production Gantt prediction succeeded.\n"
        "Return exactly one valid JSON object with task, mode, predicted_coupling_future and horizon_summary.\n"
        "Do not output markdown or text outside JSON."
    )
    user_prompt = json.dumps(model_input, ensure_ascii=False, indent=2)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

def _caa_normalize_horizon_step_v2(item, fallback_step):
    item = _caa_dict(item)
    return {
        "step": int(item.get("step") or fallback_step),
        "energy_patch_id": item.get("energy_patch_id") or item.get("patch_id") or item.get("id"),
        "energy_heat_no": _caa_normalize_heat_id(
            item.get("energy_heat_no") or item.get("energy_heat_id") or item.get("hn") or item.get("main_heat_id")
        ),
        "aligned_heat_ids": sorted(_caa_set_from_values(
            item.get("aligned_heat_ids")
            or item.get("predicted_aligned_heat_ids")
            or item.get("active_heat_ids")
            or item.get("production_active_heat_ids")
        )),
        "response_type": _caa_str(item.get("response_type"), "unknown"),
        "risk_flags": sorted(_caa_set_from_values(item.get("risk_flags") or item.get("predicted_risk_flags") or item.get("rk"))),
        "risk_level": _caa_str(item.get("risk_level"), "unknown"),
        "feasible": _caa_bool(item.get("feasible", True), default=True),
        "consistency_score": _caa_float(item.get("consistency_score"), 0.0),
        "schedule_energy_consistency": _caa_str(item.get("schedule_energy_consistency"), "unknown"),
        "confidence": _caa_float(item.get("confidence"), 0.0),
        "conflicts": _caa_list(item.get("conflicts")),
        "comment": _caa_str(item.get("comment"), ""),
        "aligned": item.get("aligned", item.get("is_aligned", item.get("heat_patch_aligned"))),
    }

def _caa_normalize_horizon_output(raw_output):
    if isinstance(raw_output, list):
        steps = raw_output
        out = {
            "task": "joint_horizon_coupling_prediction",
            "mode": "joint_horizon_coupling_evaluator",
            "predicted_coupling_future": steps,
            "horizon_summary": {},
        }
    elif isinstance(raw_output, dict):
        out = dict(raw_output)
        out.setdefault("task", "joint_horizon_coupling_prediction")
        out.setdefault("mode", "joint_horizon_coupling_evaluator")
        if not isinstance(out.get("predicted_coupling_future"), list):
            picked = _caa_pick_prediction_steps(out)
            out["predicted_coupling_future"] = picked if isinstance(picked, list) else []
        if not isinstance(out.get("horizon_summary"), dict):
            out["horizon_summary"] = {}
    else:
        out = {
            "task": "joint_horizon_coupling_prediction",
            "mode": "joint_horizon_coupling_evaluator",
            "predicted_coupling_future": [],
            "horizon_summary": {},
        }

    norm_steps = []
    for i, item in enumerate(_caa_list(out.get("predicted_coupling_future")), start=1):
        if isinstance(item, dict):
            norm_steps.append(_caa_normalize_horizon_step_v2(item, i))
    out["predicted_coupling_future"] = norm_steps

    hs = _caa_dict(out.get("horizon_summary"))
    out["horizon_summary"] = {
        "mean_consistency": _caa_float(hs.get("mean_consistency"), 0.0),
        "feasibility_rate": _caa_float(hs.get("feasibility_rate"), 0.0),
        "risk_trend": _caa_str(hs.get("risk_trend"), "unknown"),
        "schedule_energy_summary": _caa_str(hs.get("schedule_energy_summary"), ""),
        "major_conflicts": _caa_list(hs.get("major_conflicts")),
        "confidence_limitation": _caa_str(hs.get("confidence_limitation"), ""),
        "comment": _caa_str(hs.get("comment"), ""),
    }
    return out

_caa_evaluate_horizon_prediction_base_v2 = _caa_evaluate_horizon_prediction

def _caa_evaluate_horizon_prediction(
    self,
    prediction,
    target_future=None,
    joint_sample=None,
    production_prediction=None,
    energy_prediction=None,
):
    metrics = _caa_evaluate_horizon_prediction_base_v2(
        self,
        prediction=prediction,
        target_future=target_future,
        joint_sample=joint_sample,
        production_prediction=production_prediction,
        energy_prediction=energy_prediction,
    )

    op_summary = _caa_operation_segment_summary(production_prediction)
    metrics.update({
        "production_gantt_prediction_available": bool(op_summary.get("available")),
        "operation_segment_source": op_summary.get("segment_source"),
        "operation_segment_trust_level": op_summary.get("trust_level"),
        "valid_model_operation_segments": int(op_summary.get("valid_model_segment_count") or 0),
        "diagnostic_operation_segments": int(op_summary.get("diagnostic_segment_count") or 0),
        "operation_segment_missing": not bool(op_summary.get("available")),
        "operation_segment_summary_for_coupling": {
            "available": op_summary.get("available"),
            "trust_level": op_summary.get("trust_level"),
            "num_segments": op_summary.get("num_segments"),
            "stage_counts": op_summary.get("stage_counts"),
            "time_span": op_summary.get("time_span"),
            "diagnostic_segment_count": op_summary.get("diagnostic_segment_count"),
        },
    })

    if not op_summary.get("available"):
        metrics["coupling_evaluation_warning"] = (
            "No valid source=model_output production operation segments were available; "
            "coupling metrics remain patch-level and cannot prove Gantt prediction success."
        )

    return metrics

def _caa_predict_and_evaluate_horizon(
    self,
    joint_sample,
    production_prediction=None,
    energy_prediction=None,
):
    prediction = self.predict_horizon(
        joint_sample=joint_sample,
        production_prediction=production_prediction,
        energy_prediction=energy_prediction,
    )
    metrics = {}
    if isinstance(joint_sample, dict) and isinstance(joint_sample.get("target_future"), dict):
        metrics = self.evaluate_horizon_prediction(
            prediction=prediction,
            target_future=joint_sample.get("target_future"),
            joint_sample=joint_sample,
            production_prediction=production_prediction,
            energy_prediction=energy_prediction,
        )
    return {
        "prediction": prediction,
        "metrics": metrics,
        "production_operation_segment_summary": _caa_operation_segment_summary(production_prediction),
    }

CouplingAgentAdapter.build_horizon_input = _caa_build_horizon_input
CouplingAgentAdapter.build_horizon_messages = _caa_build_horizon_messages
CouplingAgentAdapter.call_horizon_model = _caa_call_horizon_model
CouplingAgentAdapter.predict_horizon = _caa_predict_horizon
CouplingAgentAdapter.evaluate_horizon_prediction = _caa_evaluate_horizon_prediction
CouplingAgentAdapter.predict_and_evaluate_horizon = _caa_predict_and_evaluate_horizon

def _caa_risk_set_gof(values):
    out = set()
    if isinstance(values, list):
        iterator = values
    elif values is None:
        iterator = []
    else:
        iterator = [values]
    for v in iterator:
        if v is None:
            continue
        s = str(v).strip()
        if not s:
            continue
        out.add(s)
    return out

def _caa_pick_energy_future_steps_gof(energy_prediction):
    if not isinstance(energy_prediction, dict):
        return []
    for key in ["future_patch_predictions", "energy_future_predictions", "predicted_energy_future", "future_steps", "step_predictions"]:
        val = energy_prediction.get(key)
        if isinstance(val, list):
            return val
    raw = energy_prediction.get("raw_output")
    if isinstance(raw, dict):
        for key in ["future_patch_predictions", "energy_future_predictions", "predicted_energy_future"]:
            val = raw.get(key)
            if isinstance(val, list):
                return val
    return []

def _caa_compact_energy_prediction_gof(energy_prediction, max_steps=24):
    steps = _caa_pick_energy_future_steps_gof(energy_prediction)
    out_steps = []
    risk_hist = {}
    for i, item in enumerate(steps, start=1):
        if not isinstance(item, dict):
            continue
        risks = sorted(_caa_risk_set_gof(item.get("rk") or item.get("risk_flags") or item.get("predicted_risk_flags") or item.get("key_risks")))
        for r in risks:
            risk_hist[r] = risk_hist.get(r, 0) + 1
        if len(out_steps) < max_steps:
            out_steps.append({
                "step": int(item.get("step") or i),
                "patch_id": item.get("patch_id") or item.get("id") or item.get("p"),
                "heat_id": _caa_normalize_heat_id(item.get("heat_id") or item.get("hn") or item.get("energy_heat_no")),
                "response_type": item.get("response_type"),
                "risk_flags": risks,
                "holder_trend": item.get("holder_trend") or item.get("holder_change") or {},
                "purchase_trend": item.get("purchase_trend") or item.get("purchase_change") or {},
                "confidence": item.get("confidence"),
            })
    return {
        "available": bool(steps),
        "num_steps": len(steps),
        "risk_hist": risk_hist,
        "examples": out_steps,
    }

def _caa_functional_future_context_gof(joint_sample):
    input_state = _caa_dict(joint_sample.get("input_state"))
    fc = _caa_dict(input_state.get("functional_coupling_context"))
    future_context = _caa_list(fc.get("future_context"))
    if not future_context:
        future_context = _caa_list(input_state.get("future_context"))
    return future_context

def _caa_build_horizon_input_gantt_functional(
    self,
    joint_sample,
    production_prediction=None,
    energy_prediction=None,
):
    input_state = _caa_dict(joint_sample.get("input_state"))
    operation_schedule_context = (
        _caa_dict(input_state.get("production_schedule_input"))
        or _caa_dict(input_state.get("production_operation_schedule_context"))
    )
    operation_segment_summary = _caa_operation_segment_summary(production_prediction)
    energy_summary = _caa_compact_energy_prediction_gof(energy_prediction)
    future_context = _caa_functional_future_context_gof(joint_sample)

    return {
        "task_instruction": (
            "You are a production-energy functional coupling evaluator for steel plant world modeling. "
            "The production-side task has been reduced to operation-level Gantt schedule prediction only. "
            "Do NOT ask for or infer production patch-level active_heat alignment. "
            "Evaluate whether predicted production operation segments are functionally consistent with predicted energy response over the horizon. "
            "Focus on stage activity, timing, energy response type, risk transfer, and feasibility. "
            "Return step-level functional coupling judgments only as JSON."
        ),
        "input": {
            "sample_id": joint_sample.get("sample_id"),
            "day_id": joint_sample.get("day_id"),
            "anchor_patch_idx": joint_sample.get("anchor_patch_idx"),
            "history_window": joint_sample.get("history_window"),
            "horizon": joint_sample.get("horizon"),
            "future_context": future_context,
            "operation_schedule_context": {
                "future_time_window": operation_schedule_context.get("future_time_window", {}),
                "candidate_future_heat_ids": operation_schedule_context.get("candidate_future_heat_ids", []),
                "required_process_order": operation_schedule_context.get("required_process_order", ["BOF", "LF", "Casting", "Rolling"]),
                "duration_priors": operation_schedule_context.get("duration_priors", {}),
            },
            "production_operation_segment_summary": operation_segment_summary,
            "energy_prediction_summary": energy_summary,
            "production_prediction_meta": _caa_strip_for_coupling(production_prediction) if isinstance(production_prediction, dict) else {},
            "energy_prediction_meta": _caa_strip_for_coupling(energy_prediction) if isinstance(energy_prediction, dict) else {},
            "target_policy": {
                "target_future_is_hidden": True,
                "target_operation_segments_are_hidden": True,
                "evaluation_uses_hidden_targets_offline": True,
            },
            "evaluation_focus": [
                "operation_schedule_energy_response_consistency",
                "stage_activity_to_energy_risk_transfer",
                "timing_consistency_between_operation_segments_and_energy_patches",
                "future_joint_feasibility",
                "confidence_limitation_when_production_segments_missing",
            ],
            "removed_tasks": [
                "heat_patch_alignment_prediction",
                "production_patch_level_state_prediction",
                "coupling_as_patch_active_heat_matching",
            ],
        },
        "output_schema": {
            "task": "joint_horizon_functional_coupling_prediction",
            "mode": "operation_schedule_energy_functional_coupling_evaluator",
            "predicted_coupling_future": [
                {
                    "step": 1,
                    "energy_patch_id": "string",
                    "production_activity_level": "none|low|medium|high|unknown",
                    "dominant_production_stages": ["BOF|LF|Casting|Rolling"],
                    "energy_response_type": "direct|indirect|stable|unknown",
                    "risk_flags": [],
                    "risk_level": "none|low|medium|high|unknown",
                    "schedule_energy_consistency": "consistent|partially_consistent|inconsistent|unknown",
                    "feasible": True,
                    "consistency_score": 0.0,
                    "confidence": 0.0,
                    "conflicts": [],
                    "comment": "string"
                }
            ],
            "horizon_summary": {
                "mean_consistency": 0.0,
                "feasibility_rate": 0.0,
                "risk_trend": "flat|increasing|decreasing|unknown",
                "schedule_energy_summary": "string",
                "major_conflicts": [],
                "confidence_limitation": "string",
                "comment": "string"
            },
        },
    }

def _caa_build_horizon_messages_gantt_functional(self, model_input):
    system_prompt = (
        "You are a functional production-energy coupling evaluator for horizon prediction.\n"
        "The production model outputs operation-level Gantt segments only.\n"
        "Do NOT optimize. Do NOT recommend scheduling decisions. Do NOT reveal hidden labels.\n"
        "Do NOT require heat-patch alignment or production patch-level active_heat predictions.\n"
        "Evaluate whether predicted production operation timing/stage activity is consistent with the predicted energy response and risks.\n"
        "Return exactly one valid JSON object with task, mode, predicted_coupling_future and horizon_summary.\n"
        "Do not output markdown or text outside JSON."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(model_input, ensure_ascii=False, indent=2)},
    ]

def _caa_normalize_horizon_step_gantt_functional(item, fallback_step):
    item = _caa_dict(item)
    risks = sorted(_caa_risk_set_gof(item.get("risk_flags") or item.get("predicted_risk_flags") or item.get("rk") or item.get("key_risks")))
    return {
        "step": int(item.get("step") or fallback_step),
        "energy_patch_id": item.get("energy_patch_id") or item.get("patch_id") or item.get("id"),
        "production_activity_level": _caa_str(item.get("production_activity_level"), "unknown"),
        "dominant_production_stages": _caa_list(item.get("dominant_production_stages") or item.get("production_stages")),
        "energy_response_type": _caa_str(item.get("energy_response_type") or item.get("response_type"), "unknown"),
        "risk_flags": risks,
        "risk_level": _caa_str(item.get("risk_level"), "unknown"),
        "schedule_energy_consistency": _caa_str(item.get("schedule_energy_consistency"), "unknown"),
        "feasible": _caa_bool(item.get("feasible", True), default=True),
        "consistency_score": _caa_float(item.get("consistency_score"), 0.0),
        "confidence": _caa_float(item.get("confidence"), 0.0),
        "conflicts": _caa_list(item.get("conflicts")),
        "comment": _caa_str(item.get("comment"), ""),
    }

def _caa_normalize_horizon_output_gantt_functional(raw_output):
    if isinstance(raw_output, list):
        out = {
            "task": "joint_horizon_functional_coupling_prediction",
            "mode": "operation_schedule_energy_functional_coupling_evaluator",
            "predicted_coupling_future": raw_output,
            "horizon_summary": {},
        }
    elif isinstance(raw_output, dict):
        out = dict(raw_output)
        out.setdefault("task", "joint_horizon_functional_coupling_prediction")
        out.setdefault("mode", "operation_schedule_energy_functional_coupling_evaluator")
        if not isinstance(out.get("predicted_coupling_future"), list):
            picked = _caa_pick_prediction_steps(out)
            out["predicted_coupling_future"] = picked if isinstance(picked, list) else []
        if not isinstance(out.get("horizon_summary"), dict):
            out["horizon_summary"] = {}
    else:
        out = {
            "task": "joint_horizon_functional_coupling_prediction",
            "mode": "operation_schedule_energy_functional_coupling_evaluator",
            "predicted_coupling_future": [],
            "horizon_summary": {},
        }

    out["predicted_coupling_future"] = [
        _caa_normalize_horizon_step_gantt_functional(item, i)
        for i, item in enumerate(_caa_list(out.get("predicted_coupling_future")), start=1)
        if isinstance(item, dict)
    ]
    hs = _caa_dict(out.get("horizon_summary"))
    out["horizon_summary"] = {
        "mean_consistency": _caa_float(hs.get("mean_consistency"), 0.0),
        "feasibility_rate": _caa_float(hs.get("feasibility_rate"), 0.0),
        "risk_trend": _caa_str(hs.get("risk_trend"), "unknown"),
        "schedule_energy_summary": _caa_str(hs.get("schedule_energy_summary"), ""),
        "major_conflicts": _caa_list(hs.get("major_conflicts")),
        "confidence_limitation": _caa_str(hs.get("confidence_limitation"), ""),
        "comment": _caa_str(hs.get("comment"), ""),
    }
    return out

def _caa_predict_horizon_gantt_functional(self, joint_sample, production_prediction=None, energy_prediction=None):
    model_input = self.build_horizon_input(
        joint_sample=joint_sample,
        production_prediction=production_prediction,
        energy_prediction=energy_prediction,
    )
    raw_output = self.call_horizon_model(model_input)
    return _caa_normalize_horizon_output_gantt_functional(raw_output)

def _caa_pick_functional_target_pairs_gof(target_future=None, joint_sample=None):
    sources = []
    if isinstance(joint_sample, dict):
        ct = joint_sample.get("coupling_target")
        if isinstance(ct, dict):
            sources.append(ct)
        tf = joint_sample.get("target_future")
        if isinstance(tf, dict):
            for key in ["functional_coupling_target", "coupling_target"]:
                if isinstance(tf.get(key), dict):
                    sources.append(tf.get(key))
    if isinstance(target_future, dict):
        for key in ["functional_coupling_target", "coupling_target"]:
            if isinstance(target_future.get(key), dict):
                sources.append(target_future.get(key))
        sources.append(target_future)
    for src in sources:
        pairs = src.get("future_pairs") if isinstance(src, dict) else None
        if isinstance(pairs, list):
            return pairs
    return []

def _caa_evaluate_horizon_prediction_gantt_functional(
    self,
    prediction,
    target_future=None,
    joint_sample=None,
    production_prediction=None,
    energy_prediction=None,
):
    pred = _caa_normalize_horizon_output_gantt_functional(prediction)
    pred_steps = _caa_list(pred.get("predicted_coupling_future"))
    pred_by_step = {}
    for item in pred_steps:
        if isinstance(item, dict):
            step = _caa_step_key(item)
            if step is not None:
                pred_by_step[step] = item

    target_pairs = _caa_pick_functional_target_pairs_gof(target_future=target_future, joint_sample=joint_sample)
    target_by_step = {}
    for item in target_pairs:
        if isinstance(item, dict):
            step = _caa_step_key(item)
            if step is not None:
                target_by_step[step] = item

    n = len(target_by_step)
    if n == 0:
        op_summary = _caa_operation_segment_summary(production_prediction)
        return {
            "num_target_steps": 0,
            "num_predicted_steps": len(pred_by_step),
            "coverage": 0.0,
            "functional_coupling_score": 0.0,
            "mean_consistency": 0.0,
            "feasibility_rate": 0.0,
            "risk_precision": 0.0,
            "risk_recall": 0.0,
            "risk_f1": 0.0,
            "production_gantt_prediction_available": bool(op_summary.get("available")),
            "operation_segment_trust_level": op_summary.get("trust_level"),
            "valid_model_operation_segments": int(op_summary.get("valid_model_segment_count") or 0),
            "operation_segment_missing": not bool(op_summary.get("available")),
            "comment": "No functional coupling target pairs available.",
        }

    covered = 0
    feasible_vals = []
    consistency_vals = []
    missing_steps = []
    risk_tp = risk_fp = risk_fn = 0
    activity_consistent = 0

    for step, tgt in sorted(target_by_step.items()):
        p = pred_by_step.get(step, {})
        has_pred = isinstance(p, dict) and bool(p)
        if has_pred:
            covered += 1
            feasible_vals.append(_caa_bool01(p.get("feasible", True)))
            consistency_vals.append(_caa_float(p.get("consistency_score"), 0.0))
        else:
            missing_steps.append(step)

        tgt_risks = _caa_risk_set_gof(tgt.get("risk_flags"))
        pred_risks = _caa_risk_set_gof(p.get("risk_flags") if isinstance(p, dict) else [])
        risk_tp += len(tgt_risks & pred_risks)
        risk_fp += len(pred_risks - tgt_risks)
        risk_fn += len(tgt_risks - pred_risks)

        tgt_activity = _caa_dict(tgt.get("production_operation_activity"))
        tgt_has_activity = int(tgt_activity.get("num_segments") or 0) > 0
        pred_level = _caa_str(p.get("production_activity_level") if isinstance(p, dict) else "unknown", "unknown").lower()
        pred_has_activity = pred_level not in {"", "none", "unknown", "0"}
        if has_pred and bool(pred_has_activity) == bool(tgt_has_activity):
            activity_consistent += 1

    rp, rr, rf1 = _caa_f1(risk_tp, risk_fp, risk_fn)
    op_summary = _caa_operation_segment_summary(production_prediction)
    metrics = {
        "num_target_steps": n,
        "num_predicted_steps": len(pred_by_step),
        "coverage": covered / max(n, 1),
        "functional_coupling_score": sum(consistency_vals) / max(len(consistency_vals), 1),
        "mean_consistency": sum(consistency_vals) / max(len(consistency_vals), 1),
        "feasibility_rate": sum(feasible_vals) / max(len(feasible_vals), 1),
        "production_activity_detection_accuracy": activity_consistent / max(n, 1),
        "risk_precision": rp,
        "risk_recall": rr,
        "risk_f1": rf1,
        "risk_tp": risk_tp,
        "risk_fp": risk_fp,
        "risk_fn": risk_fn,
        "missing_steps": missing_steps[:20],

        "heat_patch_alignment_accuracy": None,
        "heat_patch_alignment_task_removed": True,
        "production_gantt_prediction_available": bool(op_summary.get("available")),
        "operation_segment_source": op_summary.get("segment_source"),
        "operation_segment_trust_level": op_summary.get("trust_level"),
        "valid_model_operation_segments": int(op_summary.get("valid_model_segment_count") or 0),
        "diagnostic_operation_segments": int(op_summary.get("diagnostic_segment_count") or 0),
        "operation_segment_missing": not bool(op_summary.get("available")),
        "operation_segment_summary_for_coupling": {
            "available": op_summary.get("available"),
            "trust_level": op_summary.get("trust_level"),
            "num_segments": op_summary.get("num_segments"),
            "stage_counts": op_summary.get("stage_counts"),
            "time_span": op_summary.get("time_span"),
            "diagnostic_segment_count": op_summary.get("diagnostic_segment_count"),
        },
    }
    if not op_summary.get("available"):
        metrics["coupling_evaluation_warning"] = (
            "No valid source=model_output production operation segments were available; "
            "functional coupling confidence is limited."
        )
    return metrics

def _caa_predict_and_evaluate_horizon_gantt_functional(
    self,
    joint_sample,
    production_prediction=None,
    energy_prediction=None,
):
    prediction = self.predict_horizon(
        joint_sample=joint_sample,
        production_prediction=production_prediction,
        energy_prediction=energy_prediction,
    )
    metrics = {}
    if isinstance(joint_sample, dict) and isinstance(joint_sample.get("target_future"), dict):
        metrics = self.evaluate_horizon_prediction(
            prediction=prediction,
            target_future=joint_sample.get("target_future"),
            joint_sample=joint_sample,
            production_prediction=production_prediction,
            energy_prediction=energy_prediction,
        )
    return {
        "prediction": prediction,
        "metrics": metrics,
        "production_operation_segment_summary": _caa_operation_segment_summary(production_prediction),
        "energy_prediction_summary": _caa_compact_energy_prediction_gof(energy_prediction),
    }

CouplingAgentAdapter.build_horizon_input = _caa_build_horizon_input_gantt_functional
CouplingAgentAdapter.build_horizon_messages = _caa_build_horizon_messages_gantt_functional
CouplingAgentAdapter.predict_horizon = _caa_predict_horizon_gantt_functional
CouplingAgentAdapter.evaluate_horizon_prediction = _caa_evaluate_horizon_prediction_gantt_functional
CouplingAgentAdapter.predict_and_evaluate_horizon = _caa_predict_and_evaluate_horizon_gantt_functional

def _caa_dict_pcvp(x):
    return x if isinstance(x, dict) else {}

def _caa_list_pcvp(x):
    return x if isinstance(x, list) else ([] if x is None else [x])

def _caa_has_positive_num(x):
    try:
        return float(x or 0) > 0
    except Exception:
        return False

def _caa_pcvp_projection_summary_from_production(production_prediction):
    pred = _caa_dict_pcvp(production_prediction)
    diag = _caa_dict_pcvp(pred.get("mechanism_projection_diagnostics") or pred.get("projection_diagnostics"))
    repair_meta = _caa_dict_pcvp(pred.get("repair_meta"))
    if not diag and isinstance(repair_meta.get("mechanism_projection_diagnostics"), dict):
        diag = repair_meta.get("mechanism_projection_diagnostics")
    return {
        "physics_consistent_projection_applied": bool(
            pred.get("physics_consistent_projection_applied")
            or repair_meta.get("physics_consistent_projection_applied")
            or diag.get("method_short") == "PCVP"
            or diag.get("method") == "llm_guided_physics_consistent_variational_projection"
        ),
        "uses_future_target_labels": bool(
            pred.get("projection_uses_future_target_labels")
            or repair_meta.get("projection_uses_future_target_labels")
            or diag.get("uses_future_target_labels", False)
        ),
        "num_raw_segments": diag.get("num_raw_segments") or repair_meta.get("num_raw_segments"),
        "num_repaired_segments": diag.get("num_repaired_segments") or repair_meta.get("num_repaired_segments"),
        "num_uncertain_segments": diag.get("num_uncertain_segments") or repair_meta.get("num_uncertain_segments"),
        "num_rejected_segments": diag.get("num_rejected_segments") or repair_meta.get("num_rejected_segments"),
        "violations_before": diag.get("violations_before") or repair_meta.get("projection_violations_before"),
        "violations_after": diag.get("violations_after") or repair_meta.get("projection_violations_after"),
        "objective_total_before": diag.get("objective_total_before") or repair_meta.get("projection_objective_total_before"),
        "objective_total_after": diag.get("objective_total_after") or repair_meta.get("projection_objective_total_after"),
        "policy": "strong_repair_first_weak_deletion_keep_uncertain_diagnostics",
    }

def _caa_existing_operation_summary_pcvp(production_prediction):
    pred = _caa_dict_pcvp(production_prediction)
    for key in [
        "production_operation_segment_summary",
        "operation_segment_summary",
        "operation_segment_compact_context",
    ]:
        val = pred.get(key)
        if isinstance(val, dict) and (_caa_has_positive_num(val.get("num_segments")) or val.get("available") is True):
            out = dict(val)
            out.setdefault("available", bool(out.get("available", _caa_has_positive_num(out.get("num_segments")))))
            out.setdefault("source", key)
            return out
    return {}

def _caa_segment_is_valid_model_output_pcvp(seg):
    if not isinstance(seg, dict):
        return False
    if seg.get("is_valid_gantt_prediction") is False:
        return False
    status = _caa_str(seg.get("projection_status"), "").lower()
    if status in {"uncertain", "rejected"}:
        return False
    source = _caa_str(seg.get("source") or seg.get("source_kind") or seg.get("operation_segment_source"), "").lower()
    if any(tok in source for tok in ["derived", "fallback", "calendar", "patch", "target", "history", "label", "missing"]):
        return False
    if source == "model_output" or "model" in source or "llm" in source or "predicted" in source:
        return True

    if seg.get("projection_method") == "llm_guided_physics_consistent_variational_projection":
        return True
    return False

def _caa_collect_operation_segments_pcvp(production_prediction):
    pred = _caa_dict_pcvp(production_prediction)
    valid = []
    diagnostic = []
    uncertain = _caa_list(pred.get("uncertain_operation_segments"))
    rejected = _caa_list(pred.get("rejected_operation_segments"))

    raw_predicted = pred.get("predicted_operation_segments")
    if isinstance(raw_predicted, list):
        for seg in raw_predicted:
            if _caa_segment_is_valid_model_output_pcvp(seg):
                valid.append(seg)
            elif isinstance(seg, dict):
                diagnostic.append(seg)

    for key in [
        "raw_predicted_operation_segments",
        "raw_predicted_operation_segments_before_projection",
        "rolling_repaired_operation_segments_before_projection",
        "diagnostic_derived_operation_segments",
        "derived_operation_segments_from_patches",
        "fallback_operation_segments",
        "patch_operation_segments",
        "production_patch_operation_segments",
    ]:
        val = pred.get(key)
        if isinstance(val, list):
            diagnostic.extend([x for x in val if isinstance(x, dict)])

    source_label = "model_output" if valid else "missing_model_operation_segments"
    return valid, diagnostic, uncertain, rejected, source_label

def _caa_operation_segment_summary_pcvp(production_prediction, max_items=16):
    existing = _caa_existing_operation_summary_pcvp(production_prediction)
    projection_summary = _caa_pcvp_projection_summary_from_production(production_prediction)
    if existing:
        existing = dict(existing)
        existing.setdefault("available", bool(existing.get("num_segments", 0)))
        existing.setdefault("trust_level", "model_output_pcvp_repaired" if existing.get("available") else "missing")
        existing["pcvp_projection_summary"] = projection_summary
        existing["policy"] = {
            "only_pcvp_repaired_model_output_segments_are_trusted": True,
            "uncertain_and_rejected_segments_are_diagnostics_only": True,
            "full_segments_removed_from_coupling_prompt": True,
        }
        return existing

    valid, diagnostic, uncertain, rejected, source_label = _caa_collect_operation_segments_pcvp(production_prediction)
    summary = {
        "available": bool(valid),
        "segment_source": source_label,
        "trust_level": "model_output_pcvp_repaired" if valid else "missing_or_diagnostic_only",
        "num_segments": len(valid),
        "valid_model_segment_count": len(valid),
        "diagnostic_segment_count": len(diagnostic),
        "uncertain_segment_count": len([x for x in uncertain if isinstance(x, dict)]),
        "rejected_segment_count": len([x for x in rejected if isinstance(x, dict)]),
        "heat_ids": [],
        "stage_counts": {},
        "unit_counts": {},
        "time_span": {},
        "source_counts": {},
        "mean_confidence": None,
        "examples": [],
        "diagnostic_examples": [],
        "pcvp_projection_summary": projection_summary,
        "policy": {
            "only_pcvp_repaired_model_output_segments_are_trusted": True,
            "uncertain_and_rejected_segments_are_diagnostics_only": True,
            "coupling_must_not_treat_diagnostics_as_successful_gantt_prediction": True,
        },
    }

    heat_ids = []
    starts = []
    ends = []
    confs = []
    for seg in valid:
        row = _caa_normalize_segment_for_summary_v2(seg) if "_caa_normalize_segment_for_summary_v2" in globals() else {}
        if not row:
            row = {
                "heat_id": _caa_normalize_heat_id(seg.get("heat_id") or seg.get("id") or seg.get("hn")),
                "stage": _caa_str(seg.get("stage"), "Other"),
                "unit": seg.get("unit"),
                "start_min": _caa_float(seg.get("start_min"), None),
                "end_min": _caa_float(seg.get("end_min"), None),
                "confidence": seg.get("confidence"),
                "source": seg.get("source") or seg.get("source_kind"),
            }
        hid = row.get("heat_id")
        if hid and hid not in heat_ids:
            heat_ids.append(hid)
        stage = row.get("stage") or "Other"
        unit = row.get("unit") or "unknown"
        summary["stage_counts"][stage] = summary["stage_counts"].get(stage, 0) + 1
        summary["unit_counts"][unit] = summary["unit_counts"].get(unit, 0) + 1
        src = row.get("source") or "model_output"
        summary["source_counts"][src] = summary["source_counts"].get(src, 0) + 1
        if row.get("start_min") is not None:
            starts.append(row.get("start_min"))
        if row.get("end_min") is not None:
            ends.append(row.get("end_min"))
        conf = _caa_float(row.get("confidence"), None)
        if conf is not None:
            confs.append(conf)
        if len(summary["examples"]) < max_items:
            summary["examples"].append(row)

    for seg in diagnostic[: min(6, max_items)]:
        if isinstance(seg, dict):
            try:
                summary["diagnostic_examples"].append(_caa_normalize_segment_for_summary_v2(seg))
            except Exception:
                summary["diagnostic_examples"].append(seg)

    summary["heat_ids"] = heat_ids[:max_items]
    summary["time_span"] = {
        "start_min": min(starts) if starts else None,
        "end_min": max(ends) if ends else None,
    }
    summary["mean_confidence"] = (sum(confs) / max(len(confs), 1)) if confs else None
    return summary

def _caa_compact_energy_prediction_pcvp(energy_prediction, max_steps=24):
    pred = _caa_dict_pcvp(energy_prediction)
    steps = _caa_pick_energy_future_steps_gof(pred) if "_caa_pick_energy_future_steps_gof" in globals() else []
    diag = _caa_dict_pcvp(pred.get("mechanism_energy_projection_diagnostics") or pred.get("energy_projection_diagnostics"))
    risk_hist = {}
    examples = []
    conditioning_available = bool(pred.get("production_conditioning_available") or diag.get("production_conditioning_available"))
    activity_basis = pred.get("production_activity_basis") or diag.get("production_activity_basis") or "unknown"

    for i, item in enumerate(steps, start=1):
        if not isinstance(item, dict):
            continue
        risks = sorted(_caa_risk_set_gof(item.get("rk") or item.get("risk_flags") or item.get("predicted_risk_flags") or item.get("key_risks")))
        for r in risks:
            risk_hist[r] = risk_hist.get(r, 0) + 1
        if item.get("production_activity_basis") and item.get("production_activity_basis") != "history_only":
            conditioning_available = True
            activity_basis = item.get("production_activity_basis")
        if len(examples) < max_steps:
            examples.append({
                "step": int(item.get("step") or i),
                "patch_id": item.get("patch_id") or item.get("id") or item.get("p"),
                "heat_id": _caa_normalize_heat_id(item.get("heat_id") or item.get("hn") or item.get("energy_heat_no")),
                "response_type": item.get("response_type"),
                "risk_flags": risks,
                "holder_trend": item.get("holder_trend") or item.get("holder_change") or {},
                "purchase_trend": item.get("purchase_trend") or item.get("purchase_change") or {},
                "production_activity_basis": item.get("production_activity_basis") or activity_basis,
                "operation_segment_summary_used": bool(item.get("operation_segment_summary_used")),
                "confidence": item.get("confidence"),
            })
    return {
        "available": bool(steps),
        "num_steps": len(steps),
        "risk_hist": risk_hist,
        "examples": examples,
        "production_conditioning_available": conditioning_available,
        "production_activity_basis": activity_basis,
        "energy_pcvp_projection_applied": bool(pred.get("physics_consistent_energy_projection_applied") or diag.get("method_short") == "Energy-PCVP" or diag.get("method")),
        "energy_projection_diagnostics": {
            "num_raw_predictions": diag.get("num_raw_predictions"),
            "num_repaired_predictions": diag.get("num_repaired_predictions"),
            "num_uncertain_predictions": diag.get("num_uncertain_predictions"),
            "num_rejected_predictions": diag.get("num_rejected_predictions"),
            "violations_before": diag.get("violations_before"),
            "violations_after": diag.get("violations_after"),
        },
    }

def _caa_strip_for_coupling_pcvp(obj):
    banned = {
        "_raw_text", "raw_text", "raw_output", "_raw_list_output",
        "model_input", "prompt", "full_prompt", "messages",
        "comment", "comments", "traceback", "debug", "debug_info",
        "exception", "stack", "stacktrace", "reasoning", "analysis",
        "predicted_operation_segments", "raw_predicted_operation_segments",
        "raw_predicted_operation_segments_before_projection",
        "rolling_repaired_operation_segments_before_projection",
        "uncertain_operation_segments", "rejected_operation_segments",
        "diagnostic_derived_operation_segments", "derived_operation_segments_from_patches",
        "fallback_operation_segments", "patch_operation_segments", "target_operation_segments",
        "future_patch_predictions", "raw_future_patch_predictions_before_energy_projection",
        "uncertain_energy_predictions", "rejected_energy_predictions", "target_energy_future",
    }
    if isinstance(obj, dict):
        return {str(k): _caa_strip_for_coupling_pcvp(v) for k, v in obj.items() if k not in banned}
    if isinstance(obj, list):
        return [_caa_strip_for_coupling_pcvp(x) for x in obj]
    return obj

def _caa_build_horizon_input_pcvp(self, joint_sample, production_prediction=None, energy_prediction=None):
    input_state = _caa_dict_pcvp(joint_sample.get("input_state"))
    operation_schedule_context = (
        _caa_dict_pcvp(input_state.get("production_schedule_input"))
        or _caa_dict_pcvp(input_state.get("production_operation_schedule_context"))
    )
    operation_segment_summary = _caa_operation_segment_summary_pcvp(production_prediction)
    energy_summary = _caa_compact_energy_prediction_pcvp(energy_prediction)
    future_context = _caa_functional_future_context_gof(joint_sample) if "_caa_functional_future_context_gof" in globals() else _caa_list_pcvp(input_state.get("future_context"))
    pcvp_contract = _caa_dict_pcvp(input_state.get("pcvp_contract"))

    return {
        "task_instruction": (
            "You are a production-energy functional coupling evaluator for a steel plant world model. "
            "The production prediction has been PCVP-repaired operation-level Gantt segments. "
            "The energy prediction has been PCVP-repaired future energy patch states. "
            "Evaluate whether operation schedule timing and stage activity are functionally consistent with energy response and risk over the horizon. "
            "Do NOT predict heat-patch alignment. Do NOT infer production patch-level active_heat labels. "
            "Do NOT optimize or recommend dispatching decisions. Return JSON only."
        ),
        "input": {
            "sample_id": joint_sample.get("sample_id"),
            "day_id": joint_sample.get("day_id"),
            "anchor_patch_idx": joint_sample.get("anchor_patch_idx"),
            "history_window": joint_sample.get("history_window"),
            "horizon": joint_sample.get("horizon"),
            "future_context": future_context,
            "operation_schedule_context": {
                "future_time_window": operation_schedule_context.get("future_time_window", {}),
                "candidate_future_heat_ids": operation_schedule_context.get("candidate_future_heat_ids", []),
                "required_process_order": operation_schedule_context.get("required_process_order", ["BOF", "LF", "Casting", "Rolling"]),
                "duration_priors": operation_schedule_context.get("duration_priors", {}),
            },
            "production_operation_segment_summary": operation_segment_summary,
            "energy_prediction_summary": energy_summary,
            "pcvp_contract": pcvp_contract,
            "production_prediction_meta": _caa_strip_for_coupling_pcvp(production_prediction) if isinstance(production_prediction, dict) else {},
            "energy_prediction_meta": _caa_strip_for_coupling_pcvp(energy_prediction) if isinstance(energy_prediction, dict) else {},
            "target_policy": {
                "target_future_is_hidden": True,
                "target_operation_segments_are_hidden": True,
                "target_energy_future_is_hidden": True,
                "evaluation_uses_hidden_targets_offline": True,
            },
            "evaluation_focus": [
                "operation_schedule_energy_response_consistency",
                "stage_activity_to_energy_risk_transfer",
                "timing_consistency_between_operation_segments_and_energy_patches",
                "pcvp_projection_feasibility_consistency",
                "confidence_limitation_when_production_or_energy_projection_missing",
            ],
            "removed_tasks": [
                "heat_patch_alignment_prediction",
                "production_patch_level_state_prediction",
                "coupling_as_patch_active_heat_matching",
                "joint_optimization_decision_recommendation",
            ],
            "conditioning_checks": {
                "production_gantt_available": bool(operation_segment_summary.get("available")),
                "energy_conditioned_on_production_schedule": bool(energy_summary.get("production_conditioning_available")),
                "energy_activity_basis": energy_summary.get("production_activity_basis"),
            },
        },
        "output_schema": {
            "task": "joint_horizon_functional_coupling_prediction",
            "mode": "pcvp_operation_schedule_energy_functional_coupling_evaluator",
            "predicted_coupling_future": [
                {
                    "step": 1,
                    "energy_patch_id": "string",
                    "production_activity_level": "none|low|medium|high|unknown",
                    "dominant_production_stages": ["BOF|LF|Casting|Rolling"],
                    "energy_response_type": "direct|indirect|stable|unknown",
                    "risk_flags": [],
                    "risk_level": "none|low|medium|high|unknown",
                    "schedule_energy_consistency": "consistent|partially_consistent|inconsistent|unknown",
                    "pcvp_consistency": "consistent|partially_consistent|inconsistent|unknown",
                    "feasible": True,
                    "consistency_score": 0.0,
                    "confidence": 0.0,
                    "conflicts": [],
                    "comment": "string",
                }
            ],
            "horizon_summary": {
                "mean_consistency": 0.0,
                "feasibility_rate": 0.0,
                "risk_trend": "flat|increasing|decreasing|unknown",
                "schedule_energy_summary": "string",
                "pcvp_consistency_summary": "string",
                "major_conflicts": [],
                "confidence_limitation": "string",
                "comment": "string",
            },
        },
    }

def _caa_build_horizon_messages_pcvp(self, model_input):
    system_prompt = (
        "You are a PCVP-aware functional production-energy coupling evaluator.\n"
        "Production output is operation-level Gantt segments after physics-consistent variational projection.\n"
        "Energy output is future energy patch prediction after physics-consistent variational projection.\n"
        "Do NOT optimize. Do NOT recommend scheduling decisions. Do NOT reveal hidden labels.\n"
        "Do NOT require heat-patch alignment or production patch-level active_heat predictions.\n"
        "Evaluate only whether predicted operation timing/stage activity is functionally consistent with predicted energy response and risks.\n"
        "Return exactly one valid JSON object with task, mode, predicted_coupling_future and horizon_summary.\n"
        "Do not output markdown or text outside JSON."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(model_input, ensure_ascii=False, indent=2)},
    ]

def _caa_normalize_horizon_output_pcvp(raw_output):
    out = _caa_normalize_horizon_output_gantt_functional(raw_output) if "_caa_normalize_horizon_output_gantt_functional" in globals() else _caa_normalize_horizon_output(raw_output)
    out["task"] = "joint_horizon_functional_coupling_prediction"
    out["mode"] = out.get("mode") or "pcvp_operation_schedule_energy_functional_coupling_evaluator"
    if out["mode"] == "operation_schedule_energy_functional_coupling_evaluator":
        out["mode"] = "pcvp_operation_schedule_energy_functional_coupling_evaluator"
    return out

def _caa_predict_horizon_pcvp(self, joint_sample, production_prediction=None, energy_prediction=None):
    model_input = self.build_horizon_input(
        joint_sample=joint_sample,
        production_prediction=production_prediction,
        energy_prediction=energy_prediction,
    )
    raw_output = self.call_horizon_model(model_input)
    return _caa_normalize_horizon_output_pcvp(raw_output)

def _caa_evaluate_horizon_prediction_pcvp(self, prediction, target_future=None, joint_sample=None, production_prediction=None, energy_prediction=None):
    base = _caa_evaluate_horizon_prediction_gantt_functional(
        self,
        prediction=prediction,
        target_future=target_future,
        joint_sample=joint_sample,
        production_prediction=production_prediction,
        energy_prediction=energy_prediction,
    ) if "_caa_evaluate_horizon_prediction_gantt_functional" in globals() else {}
    op_summary = _caa_operation_segment_summary_pcvp(production_prediction)
    energy_summary = _caa_compact_energy_prediction_pcvp(energy_prediction)
    base.update({
        "coupling_task": "operation_schedule_to_energy_response_functional_consistency",
        "heat_patch_alignment_accuracy": None,
        "heat_patch_alignment_task_removed": True,
        "pcvp_coupling_mode": True,
        "production_pcvp_projection_applied": bool(_caa_dict_pcvp(op_summary.get("pcvp_projection_summary")).get("physics_consistent_projection_applied")),
        "energy_pcvp_projection_applied": bool(energy_summary.get("energy_pcvp_projection_applied")),
        "energy_conditioned_on_production_schedule": bool(energy_summary.get("production_conditioning_available")),
        "energy_activity_basis": energy_summary.get("production_activity_basis"),
        "production_operation_segment_summary_for_coupling": {
            "available": op_summary.get("available"),
            "trust_level": op_summary.get("trust_level"),
            "num_segments": op_summary.get("num_segments"),
            "stage_counts": op_summary.get("stage_counts"),
            "time_span": op_summary.get("time_span"),
            "uncertain_segment_count": op_summary.get("uncertain_segment_count"),
            "rejected_segment_count": op_summary.get("rejected_segment_count"),
        },
        "energy_prediction_summary_for_coupling": {
            "available": energy_summary.get("available"),
            "num_steps": energy_summary.get("num_steps"),
            "production_conditioning_available": energy_summary.get("production_conditioning_available"),
            "production_activity_basis": energy_summary.get("production_activity_basis"),
            "risk_hist": energy_summary.get("risk_hist"),
        },
    })
    if op_summary.get("available") and not energy_summary.get("production_conditioning_available"):
        base["coupling_evaluation_warning"] = (
            "Production operation summary is available, but energy prediction does not report schedule conditioning. "
            "Check TransitionEngine Stage 2 and energy_agent_adapter_pcvp.py."
        )
    return base

def _caa_predict_and_evaluate_horizon_pcvp(self, joint_sample, production_prediction=None, energy_prediction=None):
    prediction = self.predict_horizon(
        joint_sample=joint_sample,
        production_prediction=production_prediction,
        energy_prediction=energy_prediction,
    )
    metrics = {}
    if isinstance(joint_sample, dict) and isinstance(joint_sample.get("target_future"), dict):
        metrics = self.evaluate_horizon_prediction(
            prediction=prediction,
            target_future=joint_sample.get("target_future"),
            joint_sample=joint_sample,
            production_prediction=production_prediction,
            energy_prediction=energy_prediction,
        )
    return {
        "prediction": prediction,
        "metrics": metrics,
        "production_operation_segment_summary": _caa_operation_segment_summary_pcvp(production_prediction),
        "energy_prediction_summary": _caa_compact_energy_prediction_pcvp(energy_prediction),
    }

CouplingAgentAdapter.build_horizon_input = _caa_build_horizon_input_pcvp
CouplingAgentAdapter.build_horizon_messages = _caa_build_horizon_messages_pcvp
CouplingAgentAdapter.predict_horizon = _caa_predict_horizon_pcvp
CouplingAgentAdapter.evaluate_horizon_prediction = _caa_evaluate_horizon_prediction_pcvp
CouplingAgentAdapter.predict_and_evaluate_horizon = _caa_predict_and_evaluate_horizon_pcvp

import os as _caa_os_retry_pcvp

def _caa_env_int_retry_pcvp(name, default_value):
    try:
        return max(0, int(_caa_os_retry_pcvp.environ.get(name, str(default_value))))
    except Exception:
        return default_value

def _caa_env_bool_retry_pcvp(name, default_value=True):
    raw = _caa_os_retry_pcvp.environ.get(name, "1" if default_value else "0")
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}

def _caa_float_retry_pcvp(x, default=None):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default

def _caa_time_window_from_future_ctx_retry_pcvp(row):
    row = _caa_dict_pcvp(row) if "_caa_dict_pcvp" in globals() else (row if isinstance(row, dict) else {})
    t = row.get("t") if isinstance(row.get("t"), dict) else {}
    s = _caa_float_retry_pcvp(row.get("start_min"), None)
    e = _caa_float_retry_pcvp(row.get("end_min"), None)
    if s is None:
        s = _caa_float_retry_pcvp(row.get("t0"), None)
    if e is None:
        e = _caa_float_retry_pcvp(row.get("t1"), None)
    if s is None:
        s = _caa_float_retry_pcvp(t.get("t0"), None)
    if e is None:
        e = _caa_float_retry_pcvp(t.get("t1"), None)
    return s, e

def _caa_source_allows_rule_fallback_retry_pcvp(pred):
    pred = pred if isinstance(pred, dict) else {}
    flags = [
        pred.get("rule_fallback_evaluate_as_prediction"),
        pred.get("evaluate_rule_fallback_as_prediction"),
        pred.get("rule_fallback_used") and pred.get("rule_fallback_success"),
    ]
    meta = pred.get("retry_meta") if isinstance(pred.get("retry_meta"), dict) else {}
    flags.extend([
        meta.get("rule_fallback_evaluate_as_prediction"),
        meta.get("rule_fallback_used") and meta.get("rule_fallback_success"),
    ])
    return any(bool(x) for x in flags)

def _caa_segment_is_usable_for_coupling_retry_pcvp(seg, production_prediction):
    if not isinstance(seg, dict):
        return False
    if seg.get("is_valid_gantt_prediction") is False:
        return False
    status = str(seg.get("projection_status") or "").strip().lower()
    if status in {"uncertain", "rejected"}:
        return False
    src = str(seg.get("source") or seg.get("source_kind") or seg.get("operation_segment_source") or "").strip().lower()
    if any(tok in src for tok in ["target", "label", "history"]):
        return False
    if any(tok in src for tok in ["fallback", "rule_assisted", "rule-assisted", "calendar", "patch"]):
        return _caa_source_allows_rule_fallback_retry_pcvp(production_prediction)
    if src == "model_output" or "model" in src or "llm" in src or "predicted" in src:
        return True
    if seg.get("projection_method") == "llm_guided_physics_consistent_variational_projection":
        return True

    return _caa_source_allows_rule_fallback_retry_pcvp(production_prediction)

def _caa_operation_segment_summary_retry_pcvp(production_prediction, max_items=24):
    pred = production_prediction if isinstance(production_prediction, dict) else {}
    existing = pred.get("production_operation_segment_summary")
    if isinstance(existing, dict) and existing.get("num_segments", 0):
        out = dict(existing)
        out.setdefault("available", bool(out.get("num_segments", 0)))
        out.setdefault("trust_level", "provided_summary")
        out["retry_fallback_meta"] = {
            "retry_success": bool(pred.get("retry_success") or _caa_dict_pcvp(pred.get("retry_meta")).get("retry_success")),
            "rule_fallback_used": bool(pred.get("rule_fallback_used") or _caa_dict_pcvp(pred.get("retry_meta")).get("rule_fallback_used")),
            "rule_fallback_evaluate_as_prediction": bool(pred.get("rule_fallback_evaluate_as_prediction") or _caa_dict_pcvp(pred.get("retry_meta")).get("rule_fallback_evaluate_as_prediction")),
            "uses_future_target_labels": False,
        }
        return out

    segs = []
    for key in ["predicted_operation_segments", "model_predicted_operation_segments", "valid_predicted_operation_segments"]:
        val = pred.get(key)
        if isinstance(val, list):
            segs = [x for x in val if _caa_segment_is_usable_for_coupling_retry_pcvp(x, pred)]
            break

    projection_summary = {}
    diag = pred.get("mechanism_projection_diagnostics") or pred.get("mechanism_projection_diagnostics") or pred.get("projection_diagnostics")
    if isinstance(diag, dict):
        projection_summary = {
            "physics_consistent_projection_applied": True,
            "method": diag.get("method"),
            "method_short": diag.get("method_short"),
            "num_repaired_segments": diag.get("num_repaired_segments"),
            "num_uncertain_segments": diag.get("num_uncertain_segments"),
            "num_rejected_segments": diag.get("num_rejected_segments"),
            "violations_before": diag.get("violations_before"),
            "violations_after": diag.get("violations_after"),
        }

    heat_ids, starts, ends, examples, stage_counts, unit_counts = [], [], [], [], {}, {}
    for seg in segs:
        hid = _caa_normalize_heat_id(seg.get("heat_id") or seg.get("id") or seg.get("hn")) if "_caa_normalize_heat_id" in globals() else str(seg.get("heat_id") or "")
        if hid and hid not in heat_ids:
            heat_ids.append(hid)
        stage = str(seg.get("stage") or seg.get("operation") or "Other")
        unit = seg.get("unit") or seg.get("resource") or stage
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
        unit_counts[str(unit)] = unit_counts.get(str(unit), 0) + 1
        s = _caa_float_retry_pcvp(seg.get("start_min"), None)
        e = _caa_float_retry_pcvp(seg.get("end_min"), None)
        if s is not None:
            starts.append(s)
        if e is not None:
            ends.append(e)
        if len(examples) < max_items:
            examples.append({
                "heat_id": hid,
                "stage": stage,
                "unit": unit,
                "start_min": s,
                "end_min": e,
                "confidence": seg.get("confidence"),
                "source": seg.get("source") or seg.get("source_kind"),
            })

    rule_fb = bool(pred.get("rule_fallback_used") or _caa_dict_pcvp(pred.get("retry_meta")).get("rule_fallback_used"))
    trust = "model_output_pcvp_repaired"
    if rule_fb:
        trust = "retry_exhausted_rule_assisted_fallback_pcvp_repaired"
    elif bool(pred.get("retry_success") or _caa_dict_pcvp(pred.get("retry_meta")).get("retry_success")):
        trust = "llm_self_refined_pcvp_repaired"

    return {
        "available": bool(segs),
        "segment_source": "rule_assisted_fallback" if rule_fb else ("model_output" if segs else "missing_model_operation_segments"),
        "trust_level": trust if segs else "missing_or_diagnostic_only",
        "num_segments": len(segs),
        "heat_ids": heat_ids[:max_items],
        "stage_counts": stage_counts,
        "unit_counts": unit_counts,
        "time_span": {"start_min": min(starts) if starts else None, "end_min": max(ends) if ends else None},
        "examples": examples,
        "pcvp_projection_summary": projection_summary,
        "retry_fallback_meta": {
            "retry_success": bool(pred.get("retry_success") or _caa_dict_pcvp(pred.get("retry_meta")).get("retry_success")),
            "num_horizon_retries_used": pred.get("num_horizon_retries_used") or _caa_dict_pcvp(pred.get("retry_meta")).get("num_horizon_retries_used"),
            "rule_fallback_used": rule_fb,
            "rule_fallback_success": bool(pred.get("rule_fallback_success") or _caa_dict_pcvp(pred.get("retry_meta")).get("rule_fallback_success")),
            "rule_fallback_evaluate_as_prediction": _caa_source_allows_rule_fallback_retry_pcvp(pred),
            "uses_future_target_labels": False,
        },
        "policy": {
            "model_output_preferred": True,
            "retry_success_is_valid_model_output": True,
            "rule_assisted_fallback_is_lower_trust_but_allowed_when_explicitly_marked": True,
            "target_labels_never_allowed": True,
        },
    }

def _caa_compact_energy_prediction_retry_pcvp(energy_prediction, max_steps=24):
    base = _caa_compact_energy_prediction_pcvp(energy_prediction, max_steps=max_steps) if "_caa_compact_energy_prediction_pcvp" in globals() else {}
    pred = energy_prediction if isinstance(energy_prediction, dict) else {}
    meta = pred.get("energy_retry_meta") if isinstance(pred.get("energy_retry_meta"), dict) else {}
    if not base:
        steps = pred.get("future_patch_predictions") if isinstance(pred.get("future_patch_predictions"), list) else []
        base = {"available": bool(steps), "num_steps": len(steps), "examples": [], "risk_hist": {}}
    base["energy_retry_fallback_meta"] = {
        "retry_success": bool(pred.get("energy_retry_success") or meta.get("retry_success")),
        "num_horizon_retries_used": pred.get("energy_num_horizon_retries_used") or meta.get("num_horizon_retries_used"),
        "rule_fallback_used": bool(pred.get("energy_rule_fallback_used") or meta.get("rule_fallback_used")),
        "rule_fallback_success": bool(pred.get("energy_rule_fallback_success") or meta.get("rule_fallback_success")),
        "rule_fallback_evaluate_as_prediction": bool(pred.get("energy_rule_fallback_evaluate_as_prediction") or meta.get("rule_fallback_evaluate_as_prediction")),
        "uses_future_target_labels": False,
    }

    if pred.get("energy_pcvp_applied") is not None:
        base["energy_pcvp_projection_applied"] = bool(pred.get("energy_pcvp_applied"))
    if pred.get("production_conditioning_available") is not None:
        base["production_conditioning_available"] = bool(pred.get("production_conditioning_available"))
    if pred.get("production_activity_basis") is not None:
        base["production_activity_basis"] = pred.get("production_activity_basis")
    return base

def _caa_get_future_context_retry_pcvp(joint_sample):
    inp = joint_sample.get("input_state") if isinstance(joint_sample, dict) and isinstance(joint_sample.get("input_state"), dict) else {}
    fc = inp.get("functional_coupling_context") if isinstance(inp.get("functional_coupling_context"), dict) else {}
    if isinstance(fc.get("future_context"), list):
        return fc.get("future_context")
    if isinstance(inp.get("future_context"), list):
        return inp.get("future_context")
    return []

def _caa_dominant_activity_for_step_retry_pcvp(step, future_row, op_summary):
    p0, p1 = _caa_time_window_from_future_ctx_retry_pcvp(future_row)
    stage_counts, active_count = {}, 0
    for seg in op_summary.get("examples", []) if isinstance(op_summary, dict) else []:
        if not isinstance(seg, dict):
            continue
        s = _caa_float_retry_pcvp(seg.get("start_min"), None)
        e = _caa_float_retry_pcvp(seg.get("end_min"), None)
        if s is None or e is None or p0 is None or p1 is None:
            continue
        if e <= p0 or s >= p1:
            continue
        st = str(seg.get("stage") or "Other")
        stage_counts[st] = stage_counts.get(st, 0) + 1
        active_count += 1
    if active_count == 0 and isinstance(op_summary.get("stage_counts"), dict):

        stage_counts = dict(op_summary.get("stage_counts"))
        active_count = int(sum(_caa_float_retry_pcvp(v, 0.0) or 0.0 for v in stage_counts.values()))
    if active_count >= 5:
        level = "high"
    elif active_count >= 2:
        level = "medium"
    elif active_count >= 1:
        level = "low"
    else:
        level = "none"
    dominant = sorted(stage_counts.keys(), key=lambda k: stage_counts.get(k, 0), reverse=True)[:3]
    return level, dominant, active_count

def _caa_validate_coupling_prediction_retry_pcvp(prediction, joint_sample):
    pred = _caa_normalize_horizon_output_pcvp(prediction) if "_caa_normalize_horizon_output_pcvp" in globals() else prediction
    horizon = int((joint_sample or {}).get("horizon") or 0)
    steps = pred.get("predicted_coupling_future") if isinstance(pred, dict) and isinstance(pred.get("predicted_coupling_future"), list) else []
    errors = []
    if horizon > 0 and len(steps) != horizon:
        errors.append({"type": "horizon_length_mismatch", "expected": horizon, "actual": len(steps)})
    seen = set()
    for i, row in enumerate(steps, start=1):
        if not isinstance(row, dict):
            errors.append({"type": "non_dict_step", "index": i})
            continue
        st = int(row.get("step") or i)
        if st in seen:
            errors.append({"type": "duplicate_step", "step": st})
        seen.add(st)
        if not (row.get("energy_patch_id") or row.get("patch_id") or row.get("id")):
            errors.append({"type": "missing_energy_patch_id", "step": st})
        if row.get("consistency_score") is None:
            errors.append({"type": "missing_consistency_score", "step": st})
    if not steps:
        errors.append({"type": "missing_predicted_coupling_future"})
    return {"valid": len(errors) == 0, "errors": errors, "num_steps": len(steps)}

def _caa_build_horizon_input_retry_pcvp(self, joint_sample, production_prediction=None, energy_prediction=None):
    input_state = joint_sample.get("input_state") if isinstance(joint_sample, dict) and isinstance(joint_sample.get("input_state"), dict) else {}
    operation_schedule_context = (
        input_state.get("production_schedule_input")
        if isinstance(input_state.get("production_schedule_input"), dict) else
        input_state.get("production_operation_schedule_context") if isinstance(input_state.get("production_operation_schedule_context"), dict) else {}
    )
    operation_segment_summary = _caa_operation_segment_summary_retry_pcvp(production_prediction)
    energy_summary = _caa_compact_energy_prediction_retry_pcvp(energy_prediction)
    future_context = _caa_get_future_context_retry_pcvp(joint_sample)
    pcvp_contract = input_state.get("pcvp_contract") if isinstance(input_state.get("pcvp_contract"), dict) else {}
    coupling_retry_contract = input_state.get("coupling_retry_fallback_contract") if isinstance(input_state.get("coupling_retry_fallback_contract"), dict) else {}

    return {
        "task_instruction": (
            "You are a PCVP-aware production-energy functional coupling evaluator. "
            "Production and energy predictions may be generated by LLM self-refinement and then repaired by PCVP. "
            "If LLM retry was exhausted, a lower-trust rule-assisted fallback may be explicitly marked. "
            "Evaluate whether operation schedule timing and stage activity are functionally consistent with energy response and risk over the horizon. "
            "Do not predict heat-patch alignment and do not output optimization advice. Return JSON only."
        ),
        "input": {
            "sample_id": joint_sample.get("sample_id") if isinstance(joint_sample, dict) else None,
            "day_id": joint_sample.get("day_id") if isinstance(joint_sample, dict) else None,
            "anchor_patch_idx": joint_sample.get("anchor_patch_idx") if isinstance(joint_sample, dict) else None,
            "history_window": joint_sample.get("history_window") if isinstance(joint_sample, dict) else None,
            "horizon": joint_sample.get("horizon") if isinstance(joint_sample, dict) else None,
            "future_context": future_context,
            "operation_schedule_context": {
                "future_time_window": operation_schedule_context.get("future_time_window", {}),
                "candidate_future_heat_ids": operation_schedule_context.get("candidate_future_heat_ids", []),
                "required_process_order": operation_schedule_context.get("required_process_order", ["BOF", "LF", "Casting", "Rolling"]),
                "duration_priors": operation_schedule_context.get("duration_priors", {}),
            },
            "production_operation_segment_summary": operation_segment_summary,
            "energy_prediction_summary": energy_summary,
            "pcvp_contract": pcvp_contract,
            "coupling_retry_fallback_contract": coupling_retry_contract,
            "production_prediction_meta": _caa_strip_for_coupling_pcvp(production_prediction) if "_caa_strip_for_coupling_pcvp" in globals() and isinstance(production_prediction, dict) else {},
            "energy_prediction_meta": _caa_strip_for_coupling_pcvp(energy_prediction) if "_caa_strip_for_coupling_pcvp" in globals() and isinstance(energy_prediction, dict) else {},
            "target_policy": {
                "target_future_is_hidden": True,
                "target_operation_segments_are_hidden": True,
                "target_energy_future_is_hidden": True,
                "evaluation_uses_hidden_targets_offline": True,
                "retry_and_fallback_use_future_target_labels": False,
            },
            "conditioning_checks": {
                "production_gantt_available": bool(operation_segment_summary.get("available")),
                "production_trust_level": operation_segment_summary.get("trust_level"),
                "production_rule_fallback_used": bool(operation_segment_summary.get("retry_fallback_meta", {}).get("rule_fallback_used")),
                "energy_available": bool(energy_summary.get("available")),
                "energy_conditioned_on_production_schedule": bool(energy_summary.get("production_conditioning_available")),
                "energy_activity_basis": energy_summary.get("production_activity_basis"),
                "energy_rule_fallback_used": bool(energy_summary.get("energy_retry_fallback_meta", {}).get("rule_fallback_used")),
            },
        },
        "output_schema": {
            "task": "joint_horizon_functional_coupling_prediction",
            "mode": "pcvp_retry_operation_schedule_energy_functional_coupling_evaluator",
            "predicted_coupling_future": [
                {
                    "step": 1,
                    "energy_patch_id": "string",
                    "production_activity_level": "none|low|medium|high|unknown",
                    "dominant_production_stages": ["BOF|LF|Casting|Rolling"],
                    "energy_response_type": "direct|indirect|stable|none|unknown",
                    "schedule_energy_response_consistency": 0.0,
                    "risk_transfer_consistency": 0.0,
                    "pcvp_feasible": True,
                    "confidence": 0.0,
                    "conflicts": [],
                    "comment": "string"
                }
            ],
            "horizon_summary": {
                "mean_consistency": 0.0,
                "feasibility_rate": 0.0,
                "risk_trend": "flat|increasing|decreasing|unknown",
                "schedule_energy_summary": "string",
                "pcvp_consistency_summary": "string",
                "major_conflicts": [],
                "confidence_limitation": "string",
                "comment": "string"
            }
        }
    }

def _caa_build_horizon_messages_retry_pcvp(self, model_input):
    system_prompt = (
        "You are a PCVP-aware functional production-energy coupling evaluator.\n"
        "Production output is operation-level Gantt prediction after retry/self-refinement and PCVP projection.\n"
        "Energy output is schedule-conditioned energy horizon prediction after retry/self-refinement and Energy-PCVP projection.\n"
        "A rule-assisted fallback may appear only after retry exhaustion and must be treated as lower-trust but valid when explicitly marked.\n"
        "Do NOT optimize. Do NOT recommend scheduling decisions. Do NOT reveal or use hidden labels.\n"
        "Do NOT require heat-patch alignment or production patch-level active_heat predictions.\n"
        "Return exactly one valid JSON object with task, mode, predicted_coupling_future and horizon_summary.\n"
        "Do not output markdown or text outside JSON."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(model_input, ensure_ascii=False, indent=2)},
    ]

def _caa_build_retry_model_input_retry_pcvp(model_input, attempt_index, validation, previous_prediction):
    new_input = _caa_copy(model_input) if "_caa_copy" in globals() else json.loads(json.dumps(model_input, ensure_ascii=False, default=str))
    new_input["retry_context"] = {
        "retry_attempt": int(attempt_index),
        "reason": "previous coupling output failed schema/horizon validation",
        "validation_errors": validation.get("errors", []),
        "previous_output_summary": {
            "task": previous_prediction.get("task") if isinstance(previous_prediction, dict) else None,
            "mode": previous_prediction.get("mode") if isinstance(previous_prediction, dict) else None,
            "num_predicted_steps": len(previous_prediction.get("predicted_coupling_future", [])) if isinstance(previous_prediction, dict) and isinstance(previous_prediction.get("predicted_coupling_future"), list) else 0,
        },
        "allowed_feedback": [
            "schema_errors",
            "missing step ids",
            "horizon length mismatch",
            "production_operation_segment_summary",
            "energy_prediction_summary",
            "future_context",
            "PCVP diagnostics",
        ],
        "forbidden_feedback": [
            "target_future",
            "coupling_target",
            "offline metrics",
            "accuracy",
            "f1",
            "mae",
            "rmse",
        ],
        "instruction": "Regenerate a complete predicted_coupling_future list with exactly one item per future step. Do not use hidden labels.",
    }
    return new_input

def _caa_rule_fallback_coupling_prediction_retry_pcvp(joint_sample, production_prediction=None, energy_prediction=None, reason="retry_exhausted"):
    horizon = int((joint_sample or {}).get("horizon") or 0)
    future_context = _caa_get_future_context_retry_pcvp(joint_sample)
    if horizon <= 0:
        horizon = len(future_context)
    op_summary = _caa_operation_segment_summary_retry_pcvp(production_prediction)
    energy_summary = _caa_compact_energy_prediction_retry_pcvp(energy_prediction)
    energy_examples = {}
    for row in energy_summary.get("examples", []) if isinstance(energy_summary, dict) else []:
        if isinstance(row, dict):
            try:
                energy_examples[int(row.get("step") or len(energy_examples) + 1)] = row
            except Exception:
                pass

    steps = []
    consistency_values = []
    feasible_count = 0
    for i in range(1, horizon + 1):
        fctx = future_context[i - 1] if i - 1 < len(future_context) and isinstance(future_context[i - 1], dict) else {}
        energy_row = energy_examples.get(i, {})
        patch_id = (
            fctx.get("id") or fctx.get("patch_id") or fctx.get("p")
            or energy_row.get("patch_id") or energy_row.get("id") or f"step_{i}"
        )
        heat_id = _caa_normalize_heat_id(
            fctx.get("hn") or _caa_dict_pcvp(fctx.get("pc")).get("phn") or energy_row.get("heat_id") or patch_id
        ) if "_caa_normalize_heat_id" in globals() else str(fctx.get("hn") or "")
        activity_level, stages, active_count = _caa_dominant_activity_for_step_retry_pcvp(i, fctx, op_summary)
        response = str(energy_row.get("response_type") or "unknown")
        risks = energy_row.get("risk_flags") if isinstance(energy_row.get("risk_flags"), list) else []
        if response.lower() in {"direct", "indirect"} and activity_level != "none":
            consistency = 0.78
        elif response.lower() in {"stable", "none"} and activity_level == "none":
            consistency = 0.82
        elif activity_level != "none" and (risks or response.lower() not in {"", "unknown"}):
            consistency = 0.62
        else:
            consistency = 0.50
        if bool(energy_summary.get("production_conditioning_available")) and activity_level != "none":
            consistency = min(1.0, consistency + 0.08)
        feasible = True
        feasible_count += 1 if feasible else 0
        consistency_values.append(consistency)
        steps.append({
            "step": i,
            "energy_patch_id": patch_id,
            "energy_heat_no": heat_id,
            "production_activity_level": activity_level,
            "dominant_production_stages": stages,
            "num_active_operation_segments": active_count,
            "energy_response_type": response if response else "unknown",
            "risk_flags": risks,
            "schedule_energy_response_consistency": round(consistency, 6),
            "risk_transfer_consistency": round(0.7 if risks and activity_level != "none" else 0.55 if risks else 0.75, 6),
            "pcvp_feasible": feasible,
            "confidence": 0.45,
            "source": "retry_exhausted_rule_assisted_coupling_fallback",
            "uses_future_target_labels": False,
            "conflicts": [] if consistency >= 0.6 else ["weak_schedule_energy_response_evidence"],
            "comment": f"Rule-assisted fallback after {reason}; lower-trust diagnostic coupling completion."
        })

    return {
        "task": "joint_horizon_functional_coupling_prediction",
        "mode": "pcvp_retry_rule_assisted_functional_coupling_fallback",
        "predicted_coupling_future": steps,
        "horizon_summary": {
            "mean_consistency": sum(consistency_values) / max(len(consistency_values), 1),
            "feasibility_rate": feasible_count / max(len(steps), 1),
            "risk_trend": "unknown",
            "schedule_energy_summary": "deterministic lower-trust functional coupling completion from production and energy summaries",
            "pcvp_consistency_summary": "fallback uses PCVP-repaired/fallback-marked production and energy summaries only",
            "major_conflicts": [],
            "confidence_limitation": "rule-assisted fallback was used after coupling LLM retry exhaustion",
            "comment": "No future target labels were used.",
        },
        "coupling_retry_meta": {
            "retry_enabled": True,
            "retry_exhausted": True,
            "rule_fallback_used": True,
            "rule_fallback_success": bool(steps),
            "rule_fallback_uses_future_target_labels": False,
            "fallback_reason": reason,
        },
    }

def _caa_normalize_horizon_output_retry_pcvp(raw_output):
    out = _caa_normalize_horizon_output_pcvp(raw_output) if "_caa_normalize_horizon_output_pcvp" in globals() else raw_output
    if not isinstance(out, dict):
        out = {"predicted_coupling_future": []}
    out["task"] = "joint_horizon_functional_coupling_prediction"
    out.setdefault("mode", "pcvp_retry_operation_schedule_energy_functional_coupling_evaluator")
    if out.get("mode") in {"operation_schedule_energy_functional_coupling_evaluator", "pcvp_operation_schedule_energy_functional_coupling_evaluator"}:
        out["mode"] = "pcvp_retry_operation_schedule_energy_functional_coupling_evaluator"
    return out

def _caa_predict_horizon_retry_pcvp(self, joint_sample, production_prediction=None, energy_prediction=None):
    max_retries = _caa_env_int_retry_pcvp("COUPLING_MAX_HORIZON_RETRIES", 2)
    fallback_enabled = _caa_env_bool_retry_pcvp("COUPLING_ENABLE_RULE_FALLBACK_AFTER_RETRY", True)

    base_input = self.build_horizon_input(
        joint_sample=joint_sample,
        production_prediction=production_prediction,
        energy_prediction=energy_prediction,
    )

    attempts = []
    model_input = base_input
    prediction = None
    validation = {"valid": False, "errors": [{"type": "not_run"}]}

    for attempt in range(max_retries + 1):
        raw_output = self.call_horizon_model(model_input)
        prediction = _caa_normalize_horizon_output_retry_pcvp(raw_output)
        validation = _caa_validate_coupling_prediction_retry_pcvp(prediction, joint_sample)
        attempts.append({
            "attempt": attempt + 1,
            "valid": bool(validation.get("valid")),
            "num_steps": validation.get("num_steps"),
            "errors": validation.get("errors", [])[:8],
            "mode": prediction.get("mode") if isinstance(prediction, dict) else None,
        })
        if validation.get("valid"):
            break
        if attempt < max_retries:
            model_input = _caa_build_retry_model_input_retry_pcvp(base_input, attempt + 1, validation, prediction)

    retry_used = max(0, len(attempts) - 1)
    retry_success = bool(validation.get("valid")) and retry_used > 0
    first_pass_missing = bool(attempts and not attempts[0].get("valid"))

    if not validation.get("valid") and fallback_enabled:
        prediction = _caa_rule_fallback_coupling_prediction_retry_pcvp(
            joint_sample=joint_sample,
            production_prediction=production_prediction,
            energy_prediction=energy_prediction,
            reason="retry_exhausted_or_invalid_output",
        )
        validation = _caa_validate_coupling_prediction_retry_pcvp(prediction, joint_sample)

    if isinstance(prediction, dict):
        meta = prediction.setdefault("coupling_retry_meta", {})
        if not isinstance(meta, dict):
            meta = {}
            prediction["coupling_retry_meta"] = meta
        meta.update({
            "retry_enabled": max_retries > 0,
            "max_horizon_retries": max_retries,
            "num_horizon_attempts": len(attempts),
            "num_horizon_retries_used": retry_used,
            "first_pass_missing_or_invalid": first_pass_missing,
            "retry_success": retry_success,
            "final_valid": bool(validation.get("valid")),
            "validation_errors": validation.get("errors", []),
            "attempts": attempts,
            "rule_fallback_enabled": fallback_enabled,
            "rule_fallback_used": bool(prediction.get("mode") == "pcvp_retry_rule_assisted_functional_coupling_fallback" or meta.get("rule_fallback_used")),
            "rule_fallback_uses_future_target_labels": False,
            "uses_future_target_labels": False,
        })
    return prediction

def _caa_evaluate_horizon_prediction_retry_pcvp(self, prediction, target_future=None, joint_sample=None, production_prediction=None, energy_prediction=None):
    base = _caa_evaluate_horizon_prediction_pcvp(
        self,
        prediction=prediction,
        target_future=target_future,
        joint_sample=joint_sample,
        production_prediction=production_prediction,
        energy_prediction=energy_prediction,
    ) if "_caa_evaluate_horizon_prediction_pcvp" in globals() else {}
    pred = prediction if isinstance(prediction, dict) else {}
    meta = pred.get("coupling_retry_meta") if isinstance(pred.get("coupling_retry_meta"), dict) else {}
    base.update({
        "coupling_retry_enabled": bool(meta.get("retry_enabled")),
        "coupling_num_horizon_attempts": meta.get("num_horizon_attempts"),
        "coupling_num_horizon_retries_used": meta.get("num_horizon_retries_used"),
        "coupling_first_pass_missing_or_invalid": bool(meta.get("first_pass_missing_or_invalid")),
        "coupling_retry_success": bool(meta.get("retry_success")),
        "coupling_rule_fallback_enabled": bool(meta.get("rule_fallback_enabled")),
        "coupling_rule_fallback_used": bool(meta.get("rule_fallback_used")),
        "coupling_rule_fallback_uses_future_target_labels": bool(meta.get("rule_fallback_uses_future_target_labels")),
        "coupling_final_valid": bool(meta.get("final_valid", True)),
    })
    return base

def _caa_predict_and_evaluate_horizon_retry_pcvp(self, joint_sample, production_prediction=None, energy_prediction=None):
    prediction = self.predict_horizon(
        joint_sample=joint_sample,
        production_prediction=production_prediction,
        energy_prediction=energy_prediction,
    )
    metrics = {}
    if isinstance(joint_sample, dict) and isinstance(joint_sample.get("target_future"), dict):
        metrics = self.evaluate_horizon_prediction(
            prediction=prediction,
            target_future=joint_sample.get("target_future"),
            joint_sample=joint_sample,
            production_prediction=production_prediction,
            energy_prediction=energy_prediction,
        )
    return {
        "prediction": prediction,
        "metrics": metrics,
        "production_operation_segment_summary": _caa_operation_segment_summary_retry_pcvp(production_prediction),
        "energy_prediction_summary": _caa_compact_energy_prediction_retry_pcvp(energy_prediction),
    }

CouplingAgentAdapter.build_horizon_input = _caa_build_horizon_input_retry_pcvp
CouplingAgentAdapter.build_horizon_messages = _caa_build_horizon_messages_retry_pcvp
CouplingAgentAdapter.predict_horizon = _caa_predict_horizon_retry_pcvp
CouplingAgentAdapter.evaluate_horizon_prediction = _caa_evaluate_horizon_prediction_retry_pcvp
CouplingAgentAdapter.predict_and_evaluate_horizon = _caa_predict_and_evaluate_horizon_retry_pcvp

def _caa_consistency_label_to_score_final(x, default=None):
    if isinstance(x, (int, float)):
        try:
            return float(x)
        except Exception:
            return default
    s = _caa_str(x, "").strip().lower() if "_caa_str" in globals() else str(x or "").strip().lower()
    if not s:
        return default
    mapping = {
        "consistent": 0.85,
        "high": 0.85,
        "good": 0.80,
        "aligned": 0.85,
        "direct": 0.78,
        "partially_consistent": 0.60,
        "partial": 0.60,
        "medium": 0.60,
        "weak": 0.50,
        "unknown": default,
        "stable": 0.70,
        "inconsistent": 0.20,
        "low": 0.25,
        "conflict": 0.15,
        "none": default,
    }
    return mapping.get(s, default)

def _caa_score_to_consistency_label_final(score):
    try:
        v = float(score)
    except Exception:
        return "unknown"
    if v >= 0.75:
        return "consistent"
    if v >= 0.50:
        return "partially_consistent"
    return "inconsistent"

def _caa_pick_numeric_consistency_final(item, summary_mean=0.0):
    item = item if isinstance(item, dict) else {}
    candidates = [
        item.get("consistency_score"),
        item.get("schedule_energy_response_consistency"),
        item.get("schedule_energy_consistency"),
        item.get("operation_energy_consistency"),
        item.get("production_energy_response_consistency"),
        item.get("functional_coupling_score"),
        item.get("mean_consistency"),
    ]
    for x in candidates:
        val = _caa_consistency_label_to_score_final(x, default=None)
        if val is not None:

            if val != 0.0:
                return max(0.0, min(1.0, float(val)))
    for x in candidates:
        val = _caa_consistency_label_to_score_final(x, default=None)
        if val is not None:
            return max(0.0, min(1.0, float(val)))
    if summary_mean and float(summary_mean) > 0.0:
        return max(0.0, min(1.0, float(summary_mean)))
    return 0.0

def _caa_summary_mean_consistency_final(hs, steps=None):
    hs = hs if isinstance(hs, dict) else {}
    for key in [
        "mean_consistency",
        "functional_coupling_score",
        "schedule_energy_response_consistency",
        "production_energy_response_consistency",
        "operation_energy_functional_coupling_score",
    ]:
        val = _caa_consistency_label_to_score_final(hs.get(key), default=None)
        if val is not None:
            return max(0.0, min(1.0, float(val)))
    vals = []
    for s in _caa_list(steps) if "_caa_list" in globals() else (steps or []):
        if isinstance(s, dict):
            v = _caa_pick_numeric_consistency_final(s, summary_mean=0.0)
            if v is not None:
                vals.append(float(v))
    return sum(vals) / max(len(vals), 1) if vals else 0.0

def _caa_normalize_horizon_step_metric_final(item, fallback_step, summary_mean=0.0):
    item = item if isinstance(item, dict) else {}
    try:
        step = int(item.get("step") or fallback_step)
    except Exception:
        step = int(fallback_step)

    score = _caa_pick_numeric_consistency_final(item, summary_mean=summary_mean)
    label = _caa_str(
        item.get("schedule_energy_consistency")
        or item.get("schedule_energy_consistency_label")
        or item.get("schedule_energy_response_label"),
        "",
    ) if "_caa_str" in globals() else ""
    if not label or label.lower() in {"unknown", "none"}:
        label = _caa_score_to_consistency_label_final(score)

    risks = []
    if "_caa_risk_set_gof" in globals():
        risks = sorted(_caa_risk_set_gof(item.get("risk_flags") or item.get("predicted_risk_flags") or item.get("rk") or item.get("key_risks")))
    else:
        risks = _caa_list(item.get("risk_flags") or item.get("rk")) if "_caa_list" in globals() else []

    feasible_raw = item.get("pcvp_feasible", item.get("feasible", True))
    feasible = _caa_bool(feasible_raw, default=True) if "_caa_bool" in globals() else bool(feasible_raw)

    return {
        "step": step,
        "energy_patch_id": item.get("energy_patch_id") or item.get("patch_id") or item.get("id"),
        "energy_heat_no": _caa_normalize_heat_id(item.get("energy_heat_no") or item.get("energy_heat_id") or item.get("heat_id") or item.get("hn") or item.get("main_heat_id")) if "_caa_normalize_heat_id" in globals() else item.get("energy_heat_no"),
        "production_activity_level": _caa_str(item.get("production_activity_level"), "unknown") if "_caa_str" in globals() else str(item.get("production_activity_level") or "unknown"),
        "dominant_production_stages": _caa_list(item.get("dominant_production_stages") or item.get("production_stages")) if "_caa_list" in globals() else [],
        "energy_response_type": _caa_str(item.get("energy_response_type") or item.get("response_type"), "unknown") if "_caa_str" in globals() else str(item.get("energy_response_type") or item.get("response_type") or "unknown"),
        "response_type": _caa_str(item.get("response_type") or item.get("energy_response_type"), "unknown") if "_caa_str" in globals() else str(item.get("response_type") or item.get("energy_response_type") or "unknown"),
        "risk_flags": risks,
        "risk_level": _caa_str(item.get("risk_level"), "unknown") if "_caa_str" in globals() else str(item.get("risk_level") or "unknown"),
        "schedule_energy_consistency": label,
        "schedule_energy_response_consistency": score,
        "risk_transfer_consistency": _caa_consistency_label_to_score_final(item.get("risk_transfer_consistency"), default=0.0) or 0.0,
        "feasible": feasible,
        "pcvp_feasible": feasible,
        "consistency_score": score,
        "confidence": _caa_float(item.get("confidence"), 0.0) if "_caa_float" in globals() else 0.0,
        "conflicts": _caa_list(item.get("conflicts")) if "_caa_list" in globals() else [],
        "comment": _caa_str(item.get("comment"), "") if "_caa_str" in globals() else "",
    }

def _caa_normalize_horizon_output_retry_pcvp(raw_output):

    if isinstance(raw_output, list):
        out = {
            "task": "joint_horizon_functional_coupling_prediction",
            "mode": "pcvp_retry_operation_schedule_energy_functional_coupling_evaluator",
            "predicted_coupling_future": raw_output,
            "horizon_summary": {},
        }
    elif isinstance(raw_output, dict):
        out = dict(raw_output)
        out.setdefault("task", "joint_horizon_functional_coupling_prediction")
        out.setdefault("mode", "pcvp_retry_operation_schedule_energy_functional_coupling_evaluator")
        if out.get("mode") in {
            "operation_schedule_energy_functional_coupling_evaluator",
            "pcvp_operation_schedule_energy_functional_coupling_evaluator",
        }:
            out["mode"] = "pcvp_retry_operation_schedule_energy_functional_coupling_evaluator"
        if not isinstance(out.get("predicted_coupling_future"), list):
            picked = _caa_pick_prediction_steps(out) if "_caa_pick_prediction_steps" in globals() else []
            out["predicted_coupling_future"] = picked if isinstance(picked, list) else []
        if not isinstance(out.get("horizon_summary"), dict):
            out["horizon_summary"] = {}
    else:
        out = {
            "task": "joint_horizon_functional_coupling_prediction",
            "mode": "pcvp_retry_operation_schedule_energy_functional_coupling_evaluator",
            "predicted_coupling_future": [],
            "horizon_summary": {},
        }

    raw_steps = _caa_list(out.get("predicted_coupling_future")) if "_caa_list" in globals() else []
    raw_hs = out.get("horizon_summary") if isinstance(out.get("horizon_summary"), dict) else {}
    summary_mean = _caa_summary_mean_consistency_final(raw_hs, raw_steps)
    norm_steps = [
        _caa_normalize_horizon_step_metric_final(item, i, summary_mean=summary_mean)
        for i, item in enumerate(raw_steps, start=1)
        if isinstance(item, dict)
    ]

    if norm_steps:
        step_mean = sum(float(s.get("consistency_score", 0.0)) for s in norm_steps) / max(len(norm_steps), 1)
        if step_mean > 0.0:
            summary_mean = step_mean

    feas_vals = [1.0 if bool(s.get("pcvp_feasible", s.get("feasible", True))) else 0.0 for s in norm_steps]
    raw_feas = _caa_consistency_label_to_score_final(raw_hs.get("feasibility_rate"), default=None)
    feas_rate = (sum(feas_vals) / max(len(feas_vals), 1)) if norm_steps else (raw_feas if raw_feas is not None else 0.0)

    out["predicted_coupling_future"] = norm_steps
    out["horizon_summary"] = {
        "mean_consistency": float(summary_mean),
        "feasibility_rate": float(feas_rate),
        "risk_trend": _caa_str(raw_hs.get("risk_trend"), "unknown") if "_caa_str" in globals() else str(raw_hs.get("risk_trend") or "unknown"),
        "schedule_energy_response_consistency": float(summary_mean),
        "schedule_energy_summary": _caa_str(raw_hs.get("schedule_energy_summary"), "") if "_caa_str" in globals() else "",
        "pcvp_consistency_summary": _caa_str(raw_hs.get("pcvp_consistency_summary"), "") if "_caa_str" in globals() else "",
        "major_conflicts": _caa_list(raw_hs.get("major_conflicts")) if "_caa_list" in globals() else [],
        "confidence_limitation": _caa_str(raw_hs.get("confidence_limitation"), "") if "_caa_str" in globals() else "",
        "comment": _caa_str(raw_hs.get("comment"), "") if "_caa_str" in globals() else "",
    }
    out["coupling_metric_normalization"] = {
        "applied": True,
        "reason": "normalized consistency aliases: consistency_score / schedule_energy_response_consistency / schedule_energy_consistency / horizon_summary.mean_consistency",
        "uses_future_target_labels": False,
    }
    return out

def _caa_evaluate_horizon_prediction_retry_pcvp(self, prediction, target_future=None, joint_sample=None, production_prediction=None, energy_prediction=None):
    norm_pred = _caa_normalize_horizon_output_retry_pcvp(prediction)

    try:
        base = _caa_evaluate_horizon_prediction_pcvp(
            self,
            prediction=norm_pred,
            target_future=target_future,
            joint_sample=joint_sample,
            production_prediction=production_prediction,
            energy_prediction=energy_prediction,
        ) if "_caa_evaluate_horizon_prediction_pcvp" in globals() else {}
    except Exception as exc:
        base = {"coupling_metric_base_error": f"{type(exc).__name__}: {exc}"}

    steps = _caa_list(norm_pred.get("predicted_coupling_future")) if "_caa_list" in globals() else []
    hs = norm_pred.get("horizon_summary") if isinstance(norm_pred.get("horizon_summary"), dict) else {}
    step_scores = [float(s.get("consistency_score", 0.0)) for s in steps if isinstance(s, dict)]
    mean_consistency = sum(step_scores) / max(len(step_scores), 1) if step_scores else _caa_summary_mean_consistency_final(hs, steps)
    feasible_vals = [1.0 if bool(s.get("pcvp_feasible", s.get("feasible", True))) else 0.0 for s in steps if isinstance(s, dict)]
    feasibility_rate = sum(feasible_vals) / max(len(feasible_vals), 1) if feasible_vals else _caa_float(hs.get("feasibility_rate"), 0.0)

    horizon = 0
    try:
        horizon = int((joint_sample or {}).get("horizon") or 0)
    except Exception:
        horizon = 0
    target_steps = base.get("num_target_steps")
    try:
        target_steps = int(target_steps)
    except Exception:
        target_steps = 0
    if target_steps <= 0:
        target_steps = horizon or len(steps)
    coverage = len(steps) / max(target_steps, 1) if target_steps else (1.0 if steps else 0.0)
    coverage = min(1.0, max(0.0, coverage))

    meta = norm_pred.get("coupling_retry_meta") if isinstance(norm_pred.get("coupling_retry_meta"), dict) else {}
    base.update({
        "num_target_steps": target_steps,
        "num_predicted_steps": len(steps),
        "coverage": coverage,
        "functional_coupling_score": mean_consistency,
        "operation_energy_functional_coupling_score": mean_consistency,
        "schedule_energy_response_consistency": mean_consistency,
        "mean_consistency": mean_consistency,
        "feasibility_rate": feasibility_rate,
        "coupling_metric_normalization_applied": True,
        "coupling_metric_normalization_uses_future_target_labels": False,
        "coupling_retry_enabled": bool(meta.get("retry_enabled")),
        "coupling_num_horizon_attempts": meta.get("num_horizon_attempts"),
        "coupling_num_horizon_retries_used": meta.get("num_horizon_retries_used"),
        "coupling_first_pass_missing_or_invalid": bool(meta.get("first_pass_missing_or_invalid")),
        "coupling_retry_success": bool(meta.get("retry_success")),
        "coupling_rule_fallback_enabled": bool(meta.get("rule_fallback_enabled")),
        "coupling_rule_fallback_used": bool(meta.get("rule_fallback_used")),
        "coupling_rule_fallback_uses_future_target_labels": bool(meta.get("rule_fallback_uses_future_target_labels")),
        "coupling_final_valid": bool(meta.get("final_valid", True)),
    })
    return base

def _caa_predict_and_evaluate_horizon_retry_pcvp(self, joint_sample, production_prediction=None, energy_prediction=None):
    prediction = self.predict_horizon(
        joint_sample=joint_sample,
        production_prediction=production_prediction,
        energy_prediction=energy_prediction,
    )
    prediction = _caa_normalize_horizon_output_retry_pcvp(prediction)
    metrics = {}
    if isinstance(joint_sample, dict) and isinstance(joint_sample.get("target_future"), dict):
        metrics = self.evaluate_horizon_prediction(
            prediction=prediction,
            target_future=joint_sample.get("target_future"),
            joint_sample=joint_sample,
            production_prediction=production_prediction,
            energy_prediction=energy_prediction,
        )
    return {
        "prediction": prediction,
        "metrics": metrics,
        "production_operation_segment_summary": _caa_operation_segment_summary_retry_pcvp(production_prediction) if "_caa_operation_segment_summary_retry_pcvp" in globals() else {},
        "energy_prediction_summary": _caa_compact_energy_prediction_retry_pcvp(energy_prediction) if "_caa_compact_energy_prediction_retry_pcvp" in globals() else {},
    }

CouplingAgentAdapter.evaluate_horizon_prediction = _caa_evaluate_horizon_prediction_retry_pcvp
CouplingAgentAdapter.predict_and_evaluate_horizon = _caa_predict_and_evaluate_horizon_retry_pcvp

from copy import deepcopy as _caa_stage1_deepcopy

_caa_stage1_prev_build_horizon_input = CouplingAgentAdapter.build_horizon_input
_caa_stage1_prev_predict_horizon = CouplingAgentAdapter.predict_horizon


def _caa_stage1_build_horizon_input(self, joint_sample, production_prediction=None, energy_prediction=None):
    model_input = _caa_stage1_prev_build_horizon_input(
        self,
        joint_sample,
        production_prediction=production_prediction,
        energy_prediction=energy_prediction,
    )
    sample_dict = _caa_dict(joint_sample)
    sample_input = _caa_dict(sample_dict.get("input_state"))
    action = _caa_dict(sample_dict.get("candidate_action") or sample_input.get("candidate_action"))
    if action:
        input_block = _caa_dict(model_input.get("input"))
        input_block["candidate_action"] = action
        input_block["action_conditioned_world_model"] = True
        model_input["input"] = input_block
    return model_input


def _caa_stage1_predict_horizon(self, joint_sample, production_prediction=None, energy_prediction=None, candidate_action=None):
    working = joint_sample
    if candidate_action is not None:
        working = _caa_stage1_deepcopy(joint_sample)
        if isinstance(working, dict):
            working["candidate_action"] = candidate_action
            inp = working.setdefault("input_state", {})
            if isinstance(inp, dict):
                inp["candidate_action"] = candidate_action
    return _caa_stage1_prev_predict_horizon(
        self,
        working,
        production_prediction=production_prediction,
        energy_prediction=energy_prediction,
    )


CouplingAgentAdapter.build_horizon_input = _caa_stage1_build_horizon_input
CouplingAgentAdapter.predict_horizon = _caa_stage1_predict_horizon
