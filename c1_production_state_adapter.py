from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from copy import deepcopy
import math

class ProductionStateAdapter:
    def __init__(
        self,
        production_daily_objects_path: str = "production_daily_objects.json",
        production_optimization_results_path: Optional[str] = None,
        include_graph: bool = True,
        include_legend: bool = False,
        max_prompt_heats: int = 16,
        max_prompt_rolling_units: int = 8,
        include_optimization_context_in_memory: bool = False,
        patch_metadata_json_path: Optional[str] = "patch_metadata.json",
    ) -> None:
        self.production_daily_objects_path = Path(production_daily_objects_path)
        self.production_optimization_results_path = (
            Path(production_optimization_results_path)
            if production_optimization_results_path else None
        )
        self.patch_metadata_json_path = (
            Path(patch_metadata_json_path)
            if patch_metadata_json_path else None
        )

        self.production_daily_objects: List[Dict[str, Any]] = []
        self.day_to_obj: Dict[str, Dict[str, Any]] = {}

        self.production_optimization_results: Dict[str, Any] = {}
        self.day_to_optimization: Dict[str, Dict[str, Any]] = {}

        self.patch_metadata: Any = None
        self.day_to_patches: Dict[str, List[Dict[str, Any]]] = {}

        self.include_graph = include_graph
        self.include_legend = include_legend

        self.max_prompt_heats = max_prompt_heats
        self.max_prompt_rolling_units = max_prompt_rolling_units
        self.include_optimization_context_in_memory = include_optimization_context_in_memory

    def load(self) -> None:
        self.production_daily_objects = self._load_json_list(self.production_daily_objects_path)
        self.day_to_obj = {
            str(obj.get("day_id")): obj
            for obj in self.production_daily_objects
            if obj.get("day_id") is not None
        }

        if self.production_optimization_results_path and self.production_optimization_results_path.exists():
            raw_opt = self._load_json_any(self.production_optimization_results_path)

            if isinstance(raw_opt, list):
                self.day_to_optimization = {
                    str(x.get("day_id")): x
                    for x in raw_opt
                    if isinstance(x, dict) and x.get("day_id") is not None
                }
            elif isinstance(raw_opt, dict):
                self.day_to_optimization = {
                    str(k): v for k, v in raw_opt.items()
                }

        if self.patch_metadata_json_path and self.patch_metadata_json_path.exists():
            self.patch_metadata = self._load_json_any(self.patch_metadata_json_path)
            self.day_to_patches = self._normalize_patch_metadata(self.patch_metadata)

    def _load_json_list(self, path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"{path} is expected to be a JSON list.")
        return data

    def _load_json_any(self, path: Path) -> Any:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def get_day_object(self, day_id: str) -> Dict[str, Any]:
        if not self.day_to_obj:
            self.load()

        if day_id not in self.day_to_obj:
            raise KeyError(f"day_id={day_id} not found in production daily objects.")

        return self.day_to_obj[day_id]

    def get_optimization_result(self, day_id: str) -> Optional[Dict[str, Any]]:
        if not self.day_to_obj:
            self.load()
        return self.day_to_optimization.get(day_id)

    def _safe_list(self, x: Any) -> List[Any]:
        return x if isinstance(x, list) else []

    def _safe_dict(self, x: Any) -> Dict[str, Any]:
        return x if isinstance(x, dict) else {}

    def _safe_float(self, x: Any, default: float = 0.0) -> float:

        try:
            if x is None:
                return default
            if isinstance(x, str) and not x.strip():
                return default
            y = float(x)
            if y != y:
                return default
            if y == float("inf") or y == float("-inf"):
                return default
            return y
        except Exception:
            return default

    def _safe_int(self, x: Any, default: int = 0) -> int:
        try:
            if x is None:
                return default
            if isinstance(x, str) and not x.strip():
                return default
            return int(round(float(x)))
        except Exception:
            return default

    def _safe_str(self, x: Any, default: str = "") -> str:
        if x is None:
            return default
        try:
            s = str(x).strip()
            return s if s else default
        except Exception:
            return default

    def _normalize_heat_id(self, x: Any) -> Optional[str]:

        if x is None:
            return None
        s = str(x).strip()
        if not s:
            return None
        s = s.replace("-", "_")

        import re
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
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    def _time_sort_key(self, x: Optional[str]) -> str:
        if x is None:
            return "999:99:99"
        return str(x)

    def _build_summary_short(self, day_summary: Dict[str, Any]) -> Dict[str, Any]:

        return {
            "d": day_summary.get("day_id"),
            "nh": day_summary.get("num_heats", 0),
            "nru": day_summary.get("num_rolling_units", 0),
            "ng": day_summary.get("num_gap_units", 0),
            "ds": day_summary.get("day_start_time"),
            "de": day_summary.get("day_end_time"),
            "span": day_summary.get("day_span_min", 0.0),
            "bf": day_summary.get("by_furnace", {}),
            "tbldg": day_summary.get("total_bof_ldg_generated_nm3", 0.0),
            "tbe": day_summary.get("total_bof_energy_mwh", 0.0),
            "tlfe": day_summary.get("total_lf_energy_mwh", 0.0),
            "tccs": day_summary.get("total_casting_cc_steam_tph_sum", 0.0),
            "tre": day_summary.get("total_rolling_energy_mwh", 0.0),
        }

    def _compact_heat(self, heat: Dict[str, Any]) -> Dict[str, Any]:

        mt = self._safe_dict(heat.get("production_main_time"))
        rt = self._safe_dict(heat.get("production_route_time"))
        pf = self._safe_dict(heat.get("production_features"))
        sf = self._safe_dict(heat.get("source_flags"))

        compact = {

            "id": self._normalize_heat_id(heat.get("Heat_No")),
            "f": heat.get("Furnace"),
            "sd": heat.get("seq_in_day"),
            "sf": heat.get("seq_in_furnace"),

            "flag": {
                "b": bool(sf.get("BOF_Exists", False)),
                "l": bool(sf.get("LF_Exists", False)),
                "c": bool(sf.get("Casting_Exists", False)),
                "r": bool(sf.get("Rolling_Exists", False)),
            },

            "tb0": mt.get("BOF_Start_Relative"),
            "tb1": mt.get("BOF_End_Relative"),
            "db": mt.get("BOF_Duration_min"),

            "tl0": rt.get("LF_Start_Relative"),
            "tl1": rt.get("LF_End_Relative"),

            "_c0": rt.get("Casting_Start_Relative"),
            "_c1": rt.get("Casting_End_Relative"),

            "g": pf.get("Grade"),
            "be": pf.get("BOF_Energy_MWh", 0.0),
            "bp": pf.get("BOF_Avg_Power_MW", 0.0),
            "bs": pf.get("BOF_Estimated_Steam_t", 0.0),
            "ldg": pf.get("BOF_LDG_Generated_Nm3", 0.0),

            "ru": None,
        }

        return compact

    def _compact_rolling_unit(self, unit: Dict[str, Any]) -> Dict[str, Any]:

        compact = {
            "u": unit.get("unit_id"),
            "gap": bool(unit.get("is_gap", False)),
            "t0": unit.get("Start_Relative"),
            "t1": unit.get("End_Relative"),
            "dur": unit.get("Duration_min", 0.0),

            "p": unit.get("Avg_Elec_MW", 0.0),
            "e": unit.get("Elec_MWh", 0.0),

            "st": unit.get("Estimated_Steam_t", 0.0),
            "sr": unit.get("Steam_Rate_tph", 0.0),
            "bfg": unit.get("BFG_Nm3", 0.0),
            "cog": unit.get("COG_Nm3", 0.0),

            "hs": self._safe_list(unit.get("Heat_No_List")),
        }

        compact["hc"] = len(compact["hs"])
        if compact["hs"]:
            compact["h0"] = compact["hs"][0]
            compact["h1"] = compact["hs"][-1]
        else:
            compact["h0"] = None
            compact["h1"] = None

        return compact

    def _attach_heat_to_rolling(self, heat_states: List[Dict[str, Any]], rolling_states: List[Dict[str, Any]]) -> None:
        heat_map = {
            h["id"]: h
            for h in heat_states
            if isinstance(h, dict) and h.get("id") is not None
        }

        for ru in rolling_states:
            unit_id = ru.get("u")
            heat_list = ru.get("hs", []) or []
            for hid in heat_list:
                if hid in heat_map:
                    heat_map[hid]["ru"] = unit_id
                    heat_map[hid]["flag"]["r"] = True

    def _enrich_rolling_with_casting_window(self, heat_states: List[Dict[str, Any]], rolling_states: List[Dict[str, Any]]) -> None:

        heat_map = {
            h["id"]: h
            for h in heat_states
            if isinstance(h, dict) and h.get("id") is not None
        }

        for ru in rolling_states:
            heat_list = ru.get("hs", []) or []

            cast_starts = []
            cast_ends = []

            for hid in heat_list:
                h = heat_map.get(hid)
                if not h:
                    continue
                if h.get("_c0"):
                    cast_starts.append(h.get("_c0"))
                if h.get("_c1"):
                    cast_ends.append(h.get("_c1"))

            cast_starts = sorted(cast_starts, key=self._time_sort_key)
            cast_ends = sorted(cast_ends, key=self._time_sort_key)

            ru["c0"] = cast_starts[0] if cast_starts else None
            ru["c1"] = cast_ends[-1] if cast_ends else None

    def _drop_heat_casting_detail(self, heat_states: List[Dict[str, Any]]) -> None:

        for h in heat_states:
            h.pop("_c0", None)
            h.pop("_c1", None)

    def _build_graph(self, heat_states: List[Dict[str, Any]], rolling_states: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

        graph: List[Dict[str, Any]] = []

        furnace_groups: Dict[str, List[Dict[str, Any]]] = {}
        for h in heat_states:
            furnace_groups.setdefault(h.get("f", "UNK"), []).append(h)

        for furnace in furnace_groups:
            furnace_groups[furnace] = sorted(
                furnace_groups[furnace],
                key=lambda x: (
                    x.get("sf", 10**9),
                    self._time_sort_key(x.get("tb0"))
                )
            )

        rolling_order: Dict[str, List[str]] = {}
        for ru in rolling_states:
            unit_id = ru.get("u")
            if unit_id:
                rolling_order[unit_id] = ru.get("hs", []) or []

        for h in heat_states:
            hid = h.get("id")
            furnace = h.get("f", "UNK")
            ru = h.get("ru")

            prev_f = None
            next_f = None
            prev_r = None
            next_r = None

            hs_f = furnace_groups.get(furnace, [])
            idx_f = next((i for i, x in enumerate(hs_f) if x.get("id") == hid), None)
            if idx_f is not None:
                if idx_f > 0:
                    prev_f = hs_f[idx_f - 1].get("id")
                if idx_f < len(hs_f) - 1:
                    next_f = hs_f[idx_f + 1].get("id")

            hs_r = rolling_order.get(ru, [])
            if ru and hid in hs_r:
                idx_r = hs_r.index(hid)
                if idx_r > 0:
                    prev_r = hs_r[idx_r - 1]
                if idx_r < len(hs_r) - 1:
                    next_r = hs_r[idx_r + 1]

            graph.append({
                "id": hid,
                "f": furnace,
                "ru": ru,
                "pf": prev_f,
                "nf": next_f,
                "pr": prev_r,
                "nr": next_r,
            })

        return graph

    def _build_peak_heats(self, heat_states: List[Dict[str, Any]], top_k: int = 12) -> List[Dict[str, Any]]:

        candidates = []
        for h in heat_states:
            be = h.get("be", 0.0) or 0.0
            bp = h.get("bp", 0.0) or 0.0
            bs = h.get("bs", 0.0) or 0.0
            candidates.append((be, bp, bs, h))

        candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)

        out = []
        for _, _, _, h in candidates[:top_k]:
            out.append({
                "id": h.get("id"),
                "f": h.get("f"),
                "tb0": h.get("tb0"),
                "be": h.get("be", 0.0),
                "bp": h.get("bp", 0.0),
                "bs": h.get("bs", 0.0),
                "ru": h.get("ru"),
            })
        return out

    def _compact_heat_for_prompt(self, h: Dict[str, Any], mode: str) -> Dict[str, Any]:

        if mode == "production":
            keep = ["id", "f", "sd", "sf", "tb0", "tb1", "tl0", "tl1", "be", "bp", "ru", "mk"]
        elif mode == "energy":
            keep = ["id", "f", "tb0", "tb1", "be", "bp", "bs", "ldg", "ru", "mk"]
        elif mode == "coupling":
            keep = ["id", "f", "tb0", "be", "bp", "ru", "mk"]
        else:
            keep = list(h.keys())

        out = {k: h[k] for k in keep if k in h and h[k] is not None}

        if "mk" in out and isinstance(out["mk"], dict) and mode in {"energy", "coupling"}:
            out["mk"] = {
                "sw": out["mk"].get("sw"),
                "hc": out["mk"].get("hc"),
                "tt": out["mk"].get("tt"),
                "wg": out["mk"].get("wg"),
            }

        return out

    def _compact_rolling_for_prompt(self, r: Dict[str, Any], mode: str) -> Dict[str, Any]:

        if mode == "production":
            keep = ["u", "gap", "c0", "c1", "t0", "t1", "dur", "p", "hc", "hs"]
        elif mode == "energy":
            keep = ["u", "gap", "c0", "c1", "t0", "t1", "dur", "p", "e", "st", "sr", "bfg", "cog", "hc", "hs"]
        elif mode == "coupling":
            keep = ["u", "gap", "c0", "c1", "t0", "t1", "p", "hc", "hs"]
        else:
            keep = list(r.keys())

        return {k: r[k] for k in keep if k in r and r[k] is not None}

    def _compact_peak_for_prompt(self, peak_heats: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

        return [{"id": x["id"]} for x in peak_heats if isinstance(x, dict) and x.get("id")]

    def _build_prompt_views(self, full_prod_state: Dict[str, Any]) -> Dict[str, Any]:
        heat_states = full_prod_state.get("hs", [])
        rolling_states = full_prod_state.get("rs", [])
        peak_heats = full_prod_state.get("ph", [])

        focus_heat_ids = [x.get("id") for x in peak_heats if x.get("id")]

        rest_sorted = sorted(
            heat_states,
            key=lambda x: ((x.get("be", 0.0) or 0.0), (x.get("bp", 0.0) or 0.0)),
            reverse=True,
        )
        for h in rest_sorted:
            hid = h.get("id")
            if hid and hid not in focus_heat_ids:
                focus_heat_ids.append(hid)
            if len(focus_heat_ids) >= self.max_prompt_heats:
                break

        heat_map = {h["id"]: h for h in heat_states if h.get("id")}
        focus_heats_full = [heat_map[hid] for hid in focus_heat_ids if hid in heat_map]

        focus_ru_ids = self._dedup_preserve_order(
            [h.get("ru") for h in focus_heats_full if h.get("ru")]
        )
        rolling_map = {r["u"]: r for r in rolling_states if r.get("u")}
        focus_rollings_full = [rolling_map[uid] for uid in focus_ru_ids if uid in rolling_map]
        focus_rollings_full = focus_rollings_full[: self.max_prompt_rolling_units]

        prod_hs = [self._compact_heat_for_prompt(h, "production") for h in focus_heats_full]
        prod_rs = [self._compact_rolling_for_prompt(r, "production") for r in focus_rollings_full]

        energy_hs = [self._compact_heat_for_prompt(h, "energy") for h in focus_heats_full]
        energy_rs = [self._compact_rolling_for_prompt(r, "energy") for r in focus_rollings_full]

        coupling_hs_full = focus_heats_full[: min(8, len(focus_heats_full))]
        coupling_rs_full = focus_rollings_full[: min(6, len(focus_rollings_full))]
        coupling_hs = [self._compact_heat_for_prompt(h, "coupling") for h in coupling_hs_full]
        coupling_rs = [self._compact_rolling_for_prompt(r, "coupling") for r in coupling_rs_full]

        compact_ph = self._compact_peak_for_prompt(peak_heats)

        return {
            "prod": {
                "d": full_prod_state.get("d"),
                "m": "production",
                "s": full_prod_state.get("s", {}),
                "cf": full_prod_state.get("cf", {}),
                "ph": compact_ph,
                "hs": prod_hs,
                "rs": prod_rs,
                "pm": {
                    "fh": len(heat_states),
                    "fr": len(rolling_states),
                }
            },
            "energy": {
                "d": full_prod_state.get("d"),
                "m": "energy",
                "s": full_prod_state.get("s", {}),
                "cf": full_prod_state.get("cf", {}),
                "ph": compact_ph,
                "hs": energy_hs,
                "rs": energy_rs,
                "pm": {
                    "fh": len(heat_states),
                    "fr": len(rolling_states),
                }
            },
            "coupling": {
                "d": full_prod_state.get("d"),
                "m": "coupling",
                "s": full_prod_state.get("s", {}),
                "cf": full_prod_state.get("cf", {}),
                "ph": compact_ph,
                "hs": coupling_hs,
                "rs": coupling_rs,
                "pm": {
                    "fh": len(heat_states),
                    "fr": len(rolling_states),
                }
            }
        }

    def build_prod_state(self, day_id: str) -> Dict[str, Any]:

        day_obj = self.get_day_object(day_id)

        day_summary = self._safe_dict(day_obj.get("day_summary"))
        raw_heats = self._safe_list(day_obj.get("heats"))
        raw_rolling = self._safe_list(day_obj.get("rolling_units"))

        heat_states = [self._compact_heat(x) for x in raw_heats]
        rolling_states = [self._compact_rolling_unit(x) for x in raw_rolling]

        self._attach_heat_to_rolling(heat_states, rolling_states)

        self._enrich_rolling_with_casting_window(heat_states, rolling_states)

        self._drop_heat_casting_detail(heat_states)

        peak_heats = self._build_peak_heats(heat_states, top_k=12)

        def _safe_float(x: Any, default: float = 0.0) -> float:
            try:
                if x is None:
                    return default
                return float(x)
            except Exception:
                return default

        def _safe_str(x: Any, default: str = "") -> str:
            if x is None:
                return default
            return str(x).strip()

        def _pick_first(*values: Any) -> Any:
            for v in values:
                if v is None:
                    continue
                if isinstance(v, str) and not v.strip():
                    continue
                return v
            return None

        def _num_band(v: Any, low: float, high: float) -> Optional[str]:
            try:
                if v is None:
                    return None
                x = float(v)
                if x < low:
                    return "low"
                if x < high:
                    return "med"
                return "high"
            except Exception:
                return None

        def _time_to_min(x: Any) -> Optional[int]:
            s = _safe_str(x, "")
            if not s or ":" not in s:
                return None
            try:
                parts = s.split(":")
                if len(parts) < 2:
                    return None
                hh = int(parts[0])
                mm = int(parts[1])
                ss = int(parts[2]) if len(parts) >= 3 else 0
                return hh * 60 + mm + (1 if ss >= 30 else 0)
            except Exception:
                return None

        raw_heat_map: Dict[str, Dict[str, Any]] = {}
        for rh in raw_heats:
            hid = _safe_str(rh.get("Heat_No"))
            if hid:
                raw_heat_map[hid] = rh

        rolling_map = {
            r.get("u"): r for r in rolling_states
            if isinstance(r, dict) and r.get("u")
        }

        rolling_pace_hist = {"low": 0, "med": 0, "high": 0}
        for ru in rolling_states:
            hc = int(ru.get("hc") or 0)
            dur = _safe_float(ru.get("dur"), 0.0)
            avg_slot = dur / max(hc, 1)

            if avg_slot >= 30:
                pace_level = "high"
            elif avg_slot >= 20:
                pace_level = "med"
            else:
                pace_level = "low"

            ru["mk"] = {
                "pl": pace_level,
                "as": round(avg_slot, 4),
            }
            rolling_pace_hist[pace_level] = rolling_pace_hist.get(pace_level, 0) + 1

        furnace_groups: Dict[str, List[Dict[str, Any]]] = {}
        for h in heat_states:
            furnace_groups.setdefault(_safe_str(h.get("f"), "UNK"), []).append(h)

        for furnace in furnace_groups:
            furnace_groups[furnace] = sorted(
                furnace_groups[furnace],
                key=lambda x: (
                    x.get("sf", 10**9),
                    self._time_sort_key(x.get("tb0")),
                ),
            )

        switch_hist = {"none": 0, "low": 0, "med": 0, "high": 0}
        hot_hist = {"low": 0, "med": 0, "high": 0, "unknown": 0}
        temp_hist = {"low": 0, "med": 0, "high": 0, "unknown": 0}
        wait_hist = {"none": 0, "low": 0, "med": 0, "high": 0}

        for furnace, hs_f in furnace_groups.items():
            prev_grade = None
            prev_width = None
            prev_thick = None

            for h in hs_f:
                hid = _safe_str(h.get("id"))
                raw = raw_heat_map.get(hid, {})
                pf = self._safe_dict(raw.get("production_features"))
                mt = self._safe_dict(raw.get("production_main_time"))
                rt = self._safe_dict(raw.get("production_route_time"))

                grade = _safe_str(_pick_first(
                    pf.get("Grade"),
                    pf.get("Steel_Grade"),
                    pf.get("steel_grade"),
                    h.get("g"),
                ))

                width = _pick_first(
                    pf.get("Width_mm"),
                    pf.get("Slab_Width_mm"),
                    pf.get("Rolling_Width_mm"),
                    pf.get("Width"),
                    pf.get("width"),
                )
                thick = _pick_first(
                    pf.get("Thickness_mm"),
                    pf.get("Slab_Thickness_mm"),
                    pf.get("Rolling_Thickness_mm"),
                    pf.get("Thickness"),
                    pf.get("thickness"),
                )

                hot_charge_ratio = _pick_first(
                    pf.get("Hot_Charge_Ratio"),
                    pf.get("hot_charge_ratio"),
                )
                hot_charge_flag = _pick_first(
                    pf.get("Is_Hot_Charge"),
                    pf.get("Hot_Charge"),
                    pf.get("hot_charge_flag"),
                )
                hot_charge_temp = _pick_first(
                    pf.get("Hot_Charge_Temp_C"),
                    pf.get("Hot_Charge_Temperature_C"),
                    pf.get("hot_charge_temp"),
                )

                target_temp = _pick_first(
                    pf.get("Target_Exit_Temp_C"),
                    pf.get("Target_Furnace_Out_Temp_C"),
                    pf.get("Target_Reheating_Temp_C"),
                    pf.get("Target_Temperature_C"),
                    pf.get("target_temperature"),
                )

                switch_score = 0
                if prev_grade is not None and grade and grade != prev_grade:
                    switch_score += 2

                try:
                    if prev_width is not None and width is not None and abs(float(width) - float(prev_width)) >= 200:
                        switch_score += 1
                except Exception:
                    pass

                try:
                    if prev_thick is not None and thick is not None and abs(float(thick) - float(prev_thick)) >= 20:
                        switch_score += 1
                except Exception:
                    pass

                if switch_score <= 0:
                    switch_level = "none"
                elif switch_score == 1:
                    switch_level = "low"
                elif switch_score == 2:
                    switch_level = "med"
                else:
                    switch_level = "high"

                if hot_charge_ratio is not None:
                    hot_level = _num_band(hot_charge_ratio, 0.3, 0.7) or "unknown"
                elif hot_charge_temp is not None:
                    hot_level = _num_band(hot_charge_temp, 200, 600) or "unknown"
                elif isinstance(hot_charge_flag, bool):
                    hot_level = "high" if hot_charge_flag else "low"
                else:
                    hot_level = "unknown"

                if target_temp is None:
                    temp_level = "unknown"
                else:
                    t_band = _num_band(target_temp, 1150, 1200)
                    temp_level = t_band if t_band is not None else "unknown"

                wait_gap = None
                ru = rolling_map.get(h.get("ru"))
                if ru:
                    c1m = _time_to_min(ru.get("c1"))
                    t0m = _time_to_min(ru.get("t0"))
                    if c1m is not None and t0m is not None and t0m >= c1m:
                        wait_gap = t0m - c1m

                if wait_gap is None:
                    wait_level = "none"
                elif wait_gap >= 180:
                    wait_level = "high"
                elif wait_gap >= 60:
                    wait_level = "med"
                elif wait_gap > 0:
                    wait_level = "low"
                else:
                    wait_level = "none"

                h["mk"] = {
                    "sw": switch_level,
                    "hc": hot_level,
                    "tt": temp_level,
                    "wg": wait_level,
                }

                switch_hist[switch_level] = switch_hist.get(switch_level, 0) + 1
                hot_hist[hot_level] = hot_hist.get(hot_level, 0) + 1
                temp_hist[temp_level] = temp_hist.get(temp_level, 0) + 1
                wait_hist[wait_level] = wait_hist.get(wait_level, 0) + 1

                prev_grade = grade if grade else prev_grade
                prev_width = width if width is not None else prev_width
                prev_thick = thick if thick is not None else prev_thick

        furnace_dist: Dict[str, int] = {}
        rolling_backfilled = 0
        bof_complete = 0
        lf_complete = 0
        route_complete = 0

        for h in heat_states:
            furnace = str(h.get("f", "UNK"))
            furnace_dist[furnace] = furnace_dist.get(furnace, 0) + 1

            if h.get("ru") is not None:
                rolling_backfilled += 1
            if h.get("tb0") is not None and h.get("tb1") is not None:
                bof_complete += 1
            if h.get("tl0") is not None and h.get("tl1") is not None:
                lf_complete += 1
            if (
                h.get("tb0") is not None and h.get("tb1") is not None and
                h.get("tl0") is not None and h.get("tl1") is not None
            ):
                route_complete += 1

        gap_units = sum(1 for x in rolling_states if x.get("gap", False))
        rolling_with_cast_window = sum(
            1 for x in rolling_states
            if x.get("c0") is not None and x.get("c1") is not None
        )

        state = {
            "d": day_id,
            "s": self._build_summary_short(day_summary),
            "cf": {
                "prec": ["BOF", "LF", "Casting", "Rolling"],
                "gap": 10,
                "nolap": True,
                "soft": ["keep_rhythm", "width_desc_in_ru"],
            },
            "hs": heat_states,
            "rs": rolling_states,
            "ph": peak_heats,
            "meta": {
                "nh": len(heat_states),
                "nru": len(rolling_states),
                "ng": gap_units,
            },

            "mk": {
                "sw": switch_hist,
                "hc": hot_hist,
                "tt": temp_hist,
                "wg": wait_hist,
                "ru_pace": rolling_pace_hist,
            },

            "cov": {
                "heat_fields": {
                    "tb": bof_complete,
                    "tl": lf_complete,
                    "rt": route_complete,
                    "ru": rolling_backfilled,
                    "mk": sum(1 for h in heat_states if isinstance(h.get("mk"), dict) and h.get("mk")),
                },
                "rolling_fields": {
                    "cast_window": rolling_with_cast_window,
                    "gap_units": gap_units,
                    "mk": sum(1 for r in rolling_states if isinstance(r.get("mk"), dict) and r.get("mk")),
                },
                "dist": {
                    "furnace": furnace_dist,
                },
            },
        }

        if self.include_graph:
            state["g"] = self._build_graph(heat_states, rolling_states)

        if self.include_legend:
            state["legend"] = {
                "d": "day_id",
                "s": "summary",
                "cf": "constraint_features",
                "hs": "heat_states",
                "rs": "rolling_states",
                "ph": "peak_heats",
                "g": "graph",
                "mk": "mechanism_informed_features",
                "cov": "coverage_report",
                "id": "heat_id",
                "f": "furnace",
                "sd": "seq_in_day",
                "sf": "seq_in_furnace",
                "tb0": "bof_start_relative",
                "tb1": "bof_end_relative",
                "tl0": "lf_start_relative",
                "tl1": "lf_end_relative",
                "ru": "rolling_unit_id",
                "u": "unit_id",
                "hs(rolling)": "heat_list",
                "c0": "casting_window_start",
                "c1": "casting_window_end",
                "mk.sw": "switch_level",
                "mk.hc": "hot_charge_level",
                "mk.tt": "target_temp_level",
                "mk.wg": "waiting_gap_level",
            }

        return state

    def _extract_day_from_time(self, x: Any) -> Optional[str]:

        if x is None:
            return None
        s = str(x).strip()
        if not s:
            return None
        import re
        m = re.search(r"(20\d{2}-\d{2}-\d{2})", s)
        return m.group(1) if m else None

    def _patch_time_values(self, p: Dict[str, Any]) -> Tuple[Any, Any]:

        t = self._safe_dict(p.get("t"))
        t0 = (
            p.get("start_time") or p.get("t0") or p.get("start") or
            p.get("patch_start") or p.get("time_start") or p.get("ts") or
            t.get("ts") or t.get("t0") or t.get("start")
        )
        t1 = (
            p.get("end_time") or p.get("t1") or p.get("end") or
            p.get("patch_end") or p.get("time_end") or p.get("te") or
            t.get("te") or t.get("t1") or t.get("end")
        )
        return t0, t1

    def _extract_heat_id_from_patch_id(self, x: Any) -> Optional[str]:

        return self._normalize_heat_id(x)

    def _extract_converter_from_patch_id(self, x: Any) -> Optional[str]:
        if x is None:
            return None
        s = str(x).strip()
        if not s:
            return None
        import re
        m = re.search(r"(Converter\d+)", s)
        return m.group(1) if m else None

    def _normalize_patch_metadata(self, raw: Any) -> Dict[str, List[Dict[str, Any]]]:

        grouped: Dict[str, List[Dict[str, Any]]] = {}

        def _as_patch_list(x: Any) -> List[Dict[str, Any]]:
            if isinstance(x, list):
                return [p for p in x if isinstance(p, dict)]
            if isinstance(x, dict):
                for key in ["patches", "ps", "items", "data"]:
                    if isinstance(x.get(key), list):
                        return [p for p in x.get(key) if isinstance(p, dict)]
            return []

        def _append_patch(p: Dict[str, Any], parent_day_id: Optional[str] = None, idx: int = 0) -> None:
            if not isinstance(p, dict):
                return
            t0, t1 = self._patch_time_values(p)
            day_id = (
                p.get("day_id") or p.get("d") or
                self._extract_day_from_time(t0) or
                self._extract_day_from_time(t1) or
                parent_day_id
            )
            if not day_id:

                day_id = "__unknown__"
            grouped.setdefault(str(day_id), []).append(p)

        if isinstance(raw, list):

            if any(isinstance(x, dict) and isinstance(x.get("patches"), list) for x in raw):
                for item in raw:
                    if not isinstance(item, dict):
                        continue
                    parent_day_id = item.get("day_id") or item.get("d")
                    for idx, p in enumerate(_as_patch_list(item)):
                        _append_patch(p, parent_day_id=str(parent_day_id) if parent_day_id else None, idx=idx)

            else:
                for idx, p in enumerate(raw):
                    if isinstance(p, dict):
                        _append_patch(p, idx=idx)
        elif isinstance(raw, dict):

            if all(isinstance(v, (list, dict)) for v in raw.values()) and not any(k in raw for k in ["patches", "ps", "items", "data"]):
                for parent_day_id, val in raw.items():
                    for idx, p in enumerate(_as_patch_list(val)):
                        _append_patch(p, parent_day_id=str(parent_day_id), idx=idx)

            else:
                parent_day_id = raw.get("day_id") or raw.get("d")
                for idx, p in enumerate(_as_patch_list(raw)):
                    _append_patch(p, parent_day_id=str(parent_day_id) if parent_day_id else None, idx=idx)

        normalized: Dict[str, List[Dict[str, Any]]] = {}
        for day_id, patches in grouped.items():
            out = []
            for idx, p in enumerate(patches):
                t0, t1 = self._patch_time_values(p)
                pid = p.get("patch_id") or p.get("pid") or p.get("id") or p.get("patch") or p.get("p") or f"patch_{idx:04d}"
                pi = p.get("patch_index") or p.get("pi") or p.get("idx") or p.get("index") or p.get("sq") or idx
                canonical_heat_id = self._normalize_heat_id(
                    p.get("canonical_heat_id") or p.get("heat_id") or p.get("hn") or
                    p.get("phn") or self._safe_dict(p.get("pc")).get("phn") or
                    self._extract_heat_id_from_patch_id(pid) or pid
                )
                converter = (
                    p.get("converter") or p.get("mc") or self._safe_dict(p.get("pc")).get("mc") or
                    self._extract_converter_from_patch_id(pid)
                )
                t0_min = self._time_to_min(t0)
                t1_min = self._time_to_min(t1)
                try:
                    pi_int = int(pi)
                except Exception:
                    pi_int = idx
                out.append({
                    **p,
                    "pi": pi_int,
                    "pid": str(pid),
                    "t0": t0,
                    "t1": t1,
                    "t0_min": t0_min,
                    "t1_min": t1_min,
                    "canonical_heat_id": canonical_heat_id,
                    "converter": str(converter) if converter else None,
                    "source_day_id": str(day_id),
                })
            out.sort(key=lambda x: (x.get("t0_min") if x.get("t0_min") is not None else 10**9, x.get("pi", 0)))
            normalized[str(day_id)] = out
        return normalized

    def _time_to_min(self, x: Any) -> Optional[float]:

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

        import re
        m = re.search(r"(\d{1,3}):(\d{2})(?::(\d{2}))?", s)
        if not m:
            return None
        hh = int(m.group(1))
        mm = int(m.group(2))
        ss = int(m.group(3) or 0)
        return hh * 60 + mm + ss / 60.0
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

    def _canonical_stage(self, x: Any) -> str:

        s = str(x or "").strip()
        u = s.upper().replace(" ", "").replace("_", "")
        if "BOF" in u or "CONVERTER" in u or "LDG" in u:
            return "BOF"
        if u == "LF" or "LADLE" in u:
            return "LF"
        if "CAST" in u or u in {"CC", "CONTINUOUSCASTING"}:
            return "Casting"
        if "ROLL" in u or "HOTSTRIP" in u:
            return "Rolling"
        return "Other"

    def _canonical_operation_source_kind(self, source: Any) -> str:

        s = str(source or "").lower()
        if "model" in s:
            return "model_output"
        if "target" in s or "true" in s or "route" in s:
            return "target_true_route"
        if "history" in s or "past" in s:
            return "history_context"
        if "patch" in s or "derived" in s:
            return "derived_patch_state"
        if "fallback" in s or "calendar" in s:
            return "fallback_or_calendar"
        return "diagnostic"

    def _interval_overlap(self, a0: Any, a1: Any, b0: Any, b1: Any) -> bool:
        a0m, a1m = self._time_to_min(a0), self._time_to_min(a1)
        b0m, b1m = self._time_to_min(b0), self._time_to_min(b1)
        if None in [a0m, a1m, b0m, b1m]:
            return False
        return max(a0m, b0m) < min(a1m, b1m)

    def get_patch_records(self, day_id: str) -> List[Dict[str, Any]]:

        if not self.day_to_obj:
            self.load()

        if self.day_to_patches:
            if day_id in self.day_to_patches:
                return deepcopy(self.day_to_patches[day_id])
            if "__default__" in self.day_to_patches or "__unknown__" in self.day_to_patches:
                bad_keys = [k for k in ["__default__", "__unknown__"] if k in self.day_to_patches]
                raise ValueError(
                    f"patch_metadata is not reliably grouped by day_id. "
                    f"Requested day_id={day_id}, found unresolved buckets={bad_keys}. "
                    f"Refuse to use them because it may mix production patches from multiple days. "
                    f"Please add day_id to patch_metadata or include absolute timestamps so day_id can be inferred."
                )

        day_obj = self.get_day_object(day_id)
        rolling_units = self._safe_list(day_obj.get("rolling_units"))
        patches = []
        for idx, ru in enumerate(rolling_units):
            t0 = ru.get("Start_Relative")
            t1 = ru.get("End_Relative")
            patches.append({
                "pi": idx,
                "pid": str(ru.get("unit_id") or f"ru_patch_{idx:04d}"),
                "t0": t0,
                "t1": t1,
                "t0_min": self._time_to_min(t0),
                "t1_min": self._time_to_min(t1),
                "canonical_heat_id": None,
                "converter": None,
                "source": "rolling_unit_fallback",
                "source_day_id": day_id,
                "warning": "coarse_production_patch_from_rolling_unit_not_energy_patch_aligned",
            })
        patches.sort(key=lambda x: (x.get("t0_min") if x.get("t0_min") is not None else 10**9, x.get("pi", 0)))
        return patches

    def _raw_heat_route(self, heat: Dict[str, Any]) -> Dict[str, Any]:
        mt = self._safe_dict(heat.get("production_main_time"))
        rt = self._safe_dict(heat.get("production_route_time"))
        pf = self._safe_dict(heat.get("production_features"))
        return {
            "id": heat.get("Heat_No"),
            "f": heat.get("Furnace"),
            "sd": heat.get("seq_in_day"),
            "sf": heat.get("seq_in_furnace"),
            "tb0": mt.get("BOF_Start_Relative"),
            "tb1": mt.get("BOF_End_Relative"),
            "tl0": rt.get("LF_Start_Relative"),
            "tl1": rt.get("LF_End_Relative"),
            "tc0": rt.get("Casting_Start_Relative"),
            "tc1": rt.get("Casting_End_Relative"),
            "g": pf.get("Grade"),
            "be": pf.get("BOF_Energy_MWh", 0.0),
            "bp": pf.get("BOF_Avg_Power_MW", 0.0),
            "bs": pf.get("BOF_Estimated_Steam_t", 0.0),
            "ldg": pf.get("BOF_LDG_Generated_Nm3", 0.0),
        }

    def _build_heat_to_rolling_map(self, raw_rolling: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:

        hm: Dict[str, Dict[str, Any]] = {}

        def _pick_first(d: Dict[str, Any], keys: List[str]) -> Any:
            for k in keys:
                v = d.get(k)
                if v is not None and not (isinstance(v, str) and not v.strip()):
                    return v
            return None

        for ru in raw_rolling:
            if not isinstance(ru, dict):
                continue
            unit_id = ru.get("unit_id") or ru.get("u") or ru.get("rolling_unit")
            heat_list_raw = self._safe_list(ru.get("Heat_No_List") or ru.get("hs") or ru.get("heat_ids"))
            heat_list: List[str] = []
            for hid in heat_list_raw:
                nhid = self._normalize_heat_id(hid)
                if nhid and nhid not in heat_list:
                    heat_list.append(nhid)

            if not heat_list:
                continue

            unit_start = _pick_first(ru, [
                "Start_Relative", "Rolling_Start_Relative", "Rolling_Start", "t0", "start", "start_time"
            ])
            unit_end = _pick_first(ru, [
                "End_Relative", "Rolling_End_Relative", "Rolling_End", "t1", "end", "end_time"
            ])
            unit_start_min = self._time_to_min(unit_start)
            unit_end_min = self._time_to_min(unit_end)

            explicit_by_heat: Dict[str, Dict[str, Any]] = {}
            for key in ["heat_rolling_segments", "rolling_heat_segments", "heat_level_segments", "heat_segments"]:
                rows = ru.get(key)
                if isinstance(rows, list):
                    for row in rows:
                        if not isinstance(row, dict):
                            continue
                        hid = self._normalize_heat_id(row.get("heat_id") or row.get("Heat_No") or row.get("id"))
                        if not hid:
                            continue
                        s0 = _pick_first(row, ["start_time", "start", "t0", "Start_Relative", "Rolling_Start_Relative"])
                        s1 = _pick_first(row, ["end_time", "end", "t1", "End_Relative", "Rolling_End_Relative"])
                        if self._time_to_min(s0) is not None and self._time_to_min(s1) is not None:
                            explicit_by_heat[hid] = {"tr0": s0, "tr1": s1, "source": key}

            n = len(heat_list)
            slot = None
            if unit_start_min is not None and unit_end_min is not None and unit_end_min > unit_start_min:
                slot = (unit_end_min - unit_start_min) / max(n, 1)

            for idx, hid in enumerate(heat_list):
                explicit = explicit_by_heat.get(hid)
                if explicit:
                    tr0 = explicit.get("tr0")
                    tr1 = explicit.get("tr1")
                    split_method = f"explicit_{explicit.get('source')}"
                elif slot is not None:
                    tr0 = unit_start_min + idx * slot
                    tr1 = unit_start_min + (idx + 1) * slot
                    split_method = "equal_split_by_heat_order"
                else:
                    tr0 = unit_start
                    tr1 = unit_end
                    split_method = "unit_window_fallback"

                hm[hid] = {
                    "ru": unit_id,
                    "tr0": tr0,
                    "tr1": tr1,
                    "ru_tr0": unit_start,
                    "ru_tr1": unit_end,
                    "ru_gap": bool(ru.get("is_gap", False)),
                    "rolling_heat_index": idx + 1,
                    "rolling_heat_count": n,
                    "rolling_split_method": split_method,
                }
        return hm

    def _build_patch_production_snapshot(
        self,
        patch: Dict[str, Any],
        raw_heats: List[Dict[str, Any]],
        heat_to_rolling: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:

        p0, p1 = patch.get("t0"), patch.get("t1")
        active_heats = []
        stage_counts: Dict[str, int] = {"BOF": 0, "LF": 0, "Casting": 0, "Rolling": 0}
        rolling_units = []

        for heat in raw_heats:
            h = self._raw_heat_route(heat)
            hid = self._normalize_heat_id(h.get("id"))
            if not hid:
                continue
            ru_info = heat_to_rolling.get(hid, {})
            stages = []
            if self._interval_overlap(h.get("tb0"), h.get("tb1"), p0, p1):
                stages.append("BOF")

            if self._interval_overlap(h.get("tl0"), h.get("tl1"), p0, p1):
                stages.append("LF")
            if self._interval_overlap(h.get("tc0"), h.get("tc1"), p0, p1):
                stages.append("Casting")
            if self._interval_overlap(ru_info.get("tr0"), ru_info.get("tr1"), p0, p1):
                stages.append("Rolling")
                if ru_info.get("ru") not in rolling_units:
                    rolling_units.append(ru_info.get("ru"))

            if not stages:
                continue

            priority = {"BOF": 1, "LF": 2, "Casting": 3, "Rolling": 4}
            rep_stage = sorted(stages, key=lambda x: priority.get(x, 0))[-1]
            for st in stages:
                stage_counts[st] = stage_counts.get(st, 0) + 1

            active_heats.append({
                "id": hid,
                "f": h.get("f"),
                "stage": rep_stage,
                "stages": stages,
                "ru": ru_info.get("ru"),
                "sd": h.get("sd"),
                "sf": h.get("sf"),
                "be": h.get("be", 0.0),
                "bp": h.get("bp", 0.0),
                "bs": h.get("bs", 0.0),
                "ldg": h.get("ldg", 0.0),
            })

        active_heats.sort(key=lambda x: (x.get("sd") if x.get("sd") is not None else 10**9, x.get("id") or ""))
        canonical_heat_id = self._normalize_heat_id(
            patch.get("canonical_heat_id") or
            patch.get("heat_id") or
            patch.get("hn") or
            self._extract_heat_id_from_patch_id(patch.get("pid")) or
            patch.get("pid")
        )
        converter = patch.get("converter") or self._extract_converter_from_patch_id(patch.get("pid"))
        return {
            "pi": patch.get("pi"),
            "pid": patch.get("pid"),
            "t0": patch.get("t0"),
            "t1": patch.get("t1"),
            "t0_min": patch.get("t0_min"),
            "t1_min": patch.get("t1_min"),
            "canonical_heat_id": canonical_heat_id,
            "converter": converter,
            "source_day_id": patch.get("source_day_id"),
            "source": patch.get("source"),
            "active_heat_ids": [h["id"] for h in active_heats],
            "active_heats": active_heats,
            "active_rolling_units": [x for x in rolling_units if x is not None],
            "num_active_heats": len(active_heats),
            "num_active_rolling_units": len([x for x in rolling_units if x is not None]),
            "stage_counts": stage_counts,
        }

    def build_production_snapshot_by_time_window(
        self,
        day_id: str,
        patch_id: str,
        t0: Any,
        t1: Any,
        pi: Optional[int] = None,
        extra_patch_fields: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        day_obj = self.get_day_object(day_id)
        raw_heats = self._safe_list(day_obj.get("heats"))
        raw_rolling = self._safe_list(day_obj.get("rolling_units"))
        heat_to_rolling = self._build_heat_to_rolling_map(raw_rolling)
        extra_patch_fields = self._safe_dict(extra_patch_fields)
        patch = {
            **extra_patch_fields,
            "pi": pi if pi is not None else extra_patch_fields.get("pi"),
            "pid": patch_id,
            "t0": t0,
            "t1": t1,
            "t0_min": self._time_to_min(t0),
            "t1_min": self._time_to_min(t1),
            "canonical_heat_id": self._normalize_heat_id(
                extra_patch_fields.get("canonical_heat_id") or
                extra_patch_fields.get("heat_id") or
                extra_patch_fields.get("hn") or
                self._extract_heat_id_from_patch_id(patch_id) or
                patch_id
            ),
            "converter": extra_patch_fields.get("converter") or self._extract_converter_from_patch_id(patch_id),
            "source_day_id": day_id,
            "source": extra_patch_fields.get("source", "external_time_window"),
        }
        return self._build_patch_production_snapshot(patch, raw_heats, heat_to_rolling)

    def build_production_patch_sequence(self, day_id: str) -> List[Dict[str, Any]]:

        day_obj = self.get_day_object(day_id)
        raw_heats = self._safe_list(day_obj.get("heats"))
        raw_rolling = self._safe_list(day_obj.get("rolling_units"))
        patches = self.get_patch_records(day_id)
        heat_to_rolling = self._build_heat_to_rolling_map(raw_rolling)
        return [self._build_patch_production_snapshot(p, raw_heats, heat_to_rolling) for p in patches]

    def _clip_interval_to_window(self, s0: Any, s1: Any, w0: Any, w1: Any) -> Optional[Tuple[float, float, bool, bool]]:

        s0m, s1m = self._time_to_min(s0), self._time_to_min(s1)
        w0m, w1m = self._time_to_min(w0), self._time_to_min(w1)
        if None in [s0m, s1m, w0m, w1m]:
            return None
        if max(s0m, w0m) >= min(s1m, w1m):
            return None
        cs0 = max(s0m, w0m)
        cs1 = min(s1m, w1m)
        return cs0, cs1, cs0 != s0m, cs1 != s1m

    def _make_operation_segment(
        self,
        heat_id: str,
        stage: str,
        unit: Any,
        start_time: Any,
        end_time: Any,
        window_t0: Any,
        window_t1: Any,
        source: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:

        clipped = self._clip_interval_to_window(start_time, end_time, window_t0, window_t1)
        if clipped is None:
            return None

        start_min, end_min, clipped_start, clipped_end = clipped
        hid = self._normalize_heat_id(heat_id)
        stage_norm = self._canonical_stage(stage)
        if not hid or stage_norm not in {"BOF", "LF", "Casting", "Rolling"}:
            return None

        source_kind = self._canonical_operation_source_kind(source)
        out = {
            "segment_id": f"{hid}_{stage_norm}_{round(float(start_min), 4)}_{round(float(end_min), 4)}",
            "heat_id": hid,
            "stage": stage_norm,
            "unit": str(unit) if unit is not None else stage_norm,

            "start_time": self._min_to_time_string(start_min) or start_time,
            "end_time": self._min_to_time_string(end_min) or end_time,
            "start_min": float(start_min),
            "end_min": float(end_min),
            "duration_min": max(0.0, float(end_min) - float(start_min)),

            "raw_start_time": start_time,
            "raw_end_time": end_time,

            "window_t0": window_t0,
            "window_t1": window_t1,
            "is_clipped_start": bool(clipped_start),
            "is_clipped_end": bool(clipped_end),
            "source": source,
            "source_kind": source_kind,
            "is_valid_gantt_label": source_kind == "target_true_route",
            "is_valid_gantt_prediction": source_kind == "model_output",
        }
        if extra:
            out.update(extra)
        return out

    def build_operation_segments_for_time_window(
        self,
        day_id: str,
        window_t0: Any,
        window_t1: Any,
        clip_to_window: bool = True,
        source: str = "true_route_segments",
    ) -> List[Dict[str, Any]]:

        day_obj = self.get_day_object(day_id)
        raw_heats = self._safe_list(day_obj.get("heats"))
        raw_rolling = self._safe_list(day_obj.get("rolling_units"))
        heat_to_rolling = self._build_heat_to_rolling_map(raw_rolling)
        segments: List[Dict[str, Any]] = []

        for heat in raw_heats:
            h = self._raw_heat_route(heat)
            hid = self._normalize_heat_id(h.get("id"))
            if not hid:
                continue
            ru_info = heat_to_rolling.get(hid, {})
            common_extra = {
                "furnace": h.get("f"),
                "seq_in_day": h.get("sd"),
                "seq_in_furnace": h.get("sf"),
                "rolling_unit": ru_info.get("ru"),
                "rolling_unit_window_start": ru_info.get("ru_tr0"),
                "rolling_unit_window_end": ru_info.get("ru_tr1"),
                "rolling_heat_index": ru_info.get("rolling_heat_index"),
                "rolling_heat_count": ru_info.get("rolling_heat_count"),
                "rolling_split_method": ru_info.get("rolling_split_method"),
                "grade": h.get("g"),
            }

            candidates = [
                ("BOF", h.get("f") or "BOF", h.get("tb0"), h.get("tb1")),
                ("LF", "LF", h.get("tl0"), h.get("tl1")),
                ("Casting", "Casting", h.get("tc0"), h.get("tc1")),
                ("Rolling", ru_info.get("ru") or "Rolling", ru_info.get("tr0"), ru_info.get("tr1")),
            ]
            for stage, unit, s0, s1 in candidates:
                seg = self._make_operation_segment(
                    heat_id=hid,
                    stage=stage,
                    unit=unit,
                    start_time=s0,
                    end_time=s1,
                    window_t0=window_t0,
                    window_t1=window_t1,
                    source=source,
                    extra=common_extra,
                )
                if seg is not None:
                    segments.append(seg)

        segments.sort(key=lambda x: (x.get("start_min", 10**12), x.get("unit") or "", x.get("heat_id") or ""))
        return segments

    def build_operation_segments_from_patch_states(
        self,
        patch_states: List[Dict[str, Any]],
        source: str = "patch_state_segments",
    ) -> List[Dict[str, Any]]:

        rows: List[Dict[str, Any]] = []
        source_kind = self._canonical_operation_source_kind(source)

        for p in self._safe_list(patch_states):
            if not isinstance(p, dict):
                continue
            step = p.get("step")
            t0, t1 = p.get("t0"), p.get("t1")
            p0, p1 = self._time_to_min(t0), self._time_to_min(t1)
            if p0 is None or p1 is None or p1 <= p0:
                continue

            for h in self._safe_list(p.get("active_heats")):
                if not isinstance(h, dict):
                    continue
                hid = self._normalize_heat_id(h.get("id") or h.get("heat_id") or h.get("hn"))
                if not hid:
                    continue

                stage = self._canonical_stage(h.get("stage") or h.get("process") or h.get("operation"))
                if stage == "Other":
                    stages = self._safe_list(h.get("stages"))
                    for st in stages:
                        cand = self._canonical_stage(st)
                        if cand != "Other":
                            stage = cand
                            break
                if stage not in {"BOF", "LF", "Casting", "Rolling"}:
                    continue

                unit = h.get("ru") or p.get("converter") or stage
                rows.append({
                    "segment_id": f"patchseg_{hid}_{stage}_{step}",
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
                    "source_steps": [step] if step is not None else [],
                    "source": source,
                    "source_kind": source_kind,
                    "is_valid_gantt_label": False,
                    "is_valid_gantt_prediction": False,
                })

        rows.sort(key=lambda x: (x.get("heat_id") or "", x.get("stage") or "", x.get("unit") or "", x.get("start_min", 10**12)))
        merged: List[Dict[str, Any]] = []
        for seg in rows:
            if not merged:
                merged.append(seg)
                continue
            last = merged[-1]
            same_key = (
                last.get("heat_id") == seg.get("heat_id") and
                last.get("stage") == seg.get("stage") and
                last.get("unit") == seg.get("unit")
            )
            if same_key and seg.get("start_min", 10**12) <= last.get("end_min", -10**12) + 1e-6:
                last["end_min"] = max(last.get("end_min", 0.0), seg.get("end_min", 0.0))
                last["end_time"] = self._min_to_time_string(last["end_min"]) or seg.get("end_time")
                last["duration_min"] = max(0.0, last["end_min"] - last.get("start_min", 0.0))
                last.setdefault("source_steps", [])
                last["source_steps"].extend(seg.get("source_steps", []))
            else:
                merged.append(seg)

        for i, seg in enumerate(merged, start=1):
            seg["segment_id"] = seg.get("segment_id") or f"patchseg_{i:03d}"

        merged.sort(key=lambda x: (x.get("start_min", 10**12), x.get("unit") or "", x.get("heat_id") or ""))
        return merged

    def _compact_segments_for_prompt(
        self,
        segments: List[Dict[str, Any]],
        max_segments: int = 80,
    ) -> List[Dict[str, Any]]:

        compact: List[Dict[str, Any]] = []
        for seg in self._safe_list(segments):
            if not isinstance(seg, dict):
                continue
            stage = self._canonical_stage(seg.get("stage"))
            if stage not in {"BOF", "LF", "Casting", "Rolling"}:
                continue
            compact.append({
                "heat_id": self._normalize_heat_id(seg.get("heat_id")),
                "stage": stage,
                "unit": seg.get("unit"),
                "start_time": seg.get("start_time"),
                "end_time": seg.get("end_time"),
                "start_min": seg.get("start_min"),
                "end_min": seg.get("end_min"),
                "duration_min": seg.get("duration_min"),
                "source_steps": seg.get("source_steps", []),
            })
        compact = [
            x for x in compact
            if x.get("heat_id") and x.get("stage") and x.get("start_min") is not None and x.get("end_min") is not None
        ]
        compact.sort(key=lambda x: (x.get("start_min", 10**12), x.get("unit") or "", x.get("heat_id") or ""))
        if max_segments and len(compact) > max_segments:
            compact = compact[-max_segments:]
        return compact

    def _summarize_operation_segments_for_prompt(
        self,
        segments: List[Dict[str, Any]],
    ) -> Dict[str, Any]:

        by_stage: Dict[str, List[float]] = {}
        by_stage_unit: Dict[str, List[float]] = {}
        stage_order = ["BOF", "LF", "Casting", "Rolling"]

        for seg in self._safe_list(segments):
            if not isinstance(seg, dict):
                continue
            stage = str(seg.get("stage") or "Other")
            if stage not in {"BOF", "LF", "Casting", "Rolling"}:
                continue
            dur = seg.get("duration_min")
            if dur is None:
                s0 = seg.get("start_min")
                s1 = seg.get("end_min")
                if s0 is not None and s1 is not None:
                    try:
                        dur = float(s1) - float(s0)
                    except Exception:
                        dur = None
            try:
                dur_f = float(dur)
            except Exception:
                continue
            if dur_f <= 0:
                continue
            by_stage.setdefault(stage, []).append(dur_f)
            key = f"{stage}|{seg.get('unit') or stage}"
            by_stage_unit.setdefault(key, []).append(dur_f)

        def _stats(xs: List[float]) -> Dict[str, float]:
            xs = sorted(xs)
            if not xs:
                return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
            n = len(xs)
            return {
                "count": n,
                "mean": round(sum(xs) / n, 4),
                "median": round(xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2, 4),
                "min": round(xs[0], 4),
                "max": round(xs[-1], 4),
            }

        stage_stats = {st: _stats(by_stage.get(st, [])) for st in stage_order}
        unit_stats = {k: _stats(v) for k, v in by_stage_unit.items()}

        return {
            "stage_order": stage_order,
            "stage_duration_stats_min": stage_stats,
            "stage_unit_duration_stats_min": unit_stats,
            "rules": [
                "For each predicted heat, operation order should be BOF -> LF -> Casting -> Rolling when those stages overlap the future horizon.",
                "Segments of different heats may overlap on different units.",
                "A segment should be clipped to the known future time window if it crosses the boundary.",
                "If exact duration is uncertain, use historical duration statistics and calendar patch timing as priors.",
            ],
        }

    def _duration_priors_for_projection(self, duration_summary: Dict[str, Any]) -> Dict[str, Any]:

        duration_summary = self._safe_dict(duration_summary)
        stage_stats = self._safe_dict(duration_summary.get("stage_duration_stats_min"))
        unit_stats = self._safe_dict(duration_summary.get("stage_unit_duration_stats_min"))
        defaults = {
            "BOF": {"min": 8.0, "median": 16.0, "max": 60.0},
            "LF": {"min": 5.0, "median": 18.0, "max": 90.0},
            "Casting": {"min": 2.0, "median": 6.0, "max": 60.0},
            "Rolling": {"min": 3.0, "median": 15.0, "max": 120.0},
        }

        def _prior_from_stats(stage: str, stats: Dict[str, Any]) -> Dict[str, float]:
            base = defaults.get(stage, {"min": 1.0, "median": 5.0, "max": 60.0})
            try:
                cnt = int(stats.get("count", 0) or 0)
            except Exception:
                cnt = 0
            if cnt <= 0:
                return dict(base)
            mn = float(stats.get("min", base["min"]) or base["min"])
            med = float(stats.get("median", base["median"]) or base["median"])
            mx = float(stats.get("max", base["max"]) or base["max"])
            mn = max(0.5, mn)
            med = max(mn, med)
            mx = max(med, mx)
            span = max(mx - mn, 1.0)
            return {
                "min": round(max(0.5, mn - 0.10 * span), 4),
                "median": round(med, 4),
                "max": round(mx + 0.10 * span, 4),
            }

        priors: Dict[str, Any] = {st: _prior_from_stats(st, self._safe_dict(stage_stats.get(st))) for st in ["BOF", "LF", "Casting", "Rolling"]}
        for key, stats in unit_stats.items():
            if not isinstance(key, str) or "|" not in key:
                continue
            stage, unit = key.split("|", 1)
            stage = self._canonical_stage(stage)
            if stage not in {"BOF", "LF", "Casting", "Rolling"}:
                continue
            unit = str(unit or stage)
            if all(k in priors[stage] for k in ["min", "median", "max"]):
                stage_level = dict(priors[stage])
                priors[stage] = {"_stage": stage_level, **stage_level}
            priors[stage][unit] = _prior_from_stats(stage, self._safe_dict(stats))
        return priors

    def _build_waiting_time_priors_for_projection(self, segments: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:

        by_heat: Dict[str, List[Dict[str, Any]]] = {}
        for seg in self._safe_list(segments):
            if not isinstance(seg, dict):
                continue
            hid = self._normalize_heat_id(seg.get("heat_id"))
            stage = self._canonical_stage(seg.get("stage"))
            if not hid or stage not in {"BOF", "LF", "Casting", "Rolling"}:
                continue
            if seg.get("start_min") is None or seg.get("end_min") is None:
                continue
            by_heat.setdefault(hid, []).append({**seg, "stage": stage})

        gaps: Dict[str, List[float]] = {}
        for chain in by_heat.values():
            chain.sort(key=lambda x: float(x.get("start_min", 0.0)))
            stage_map: Dict[str, Dict[str, Any]] = {}
            for seg in chain:
                stage_map.setdefault(seg["stage"], seg)
            pairs = []
            if "BOF" in stage_map and "LF" in stage_map:
                pairs.append(("BOF", "LF"))
            if "LF" in stage_map and "Casting" in stage_map:
                pairs.append(("LF", "Casting"))
            elif "BOF" in stage_map and "Casting" in stage_map:
                pairs.append(("BOF", "Casting"))
            if "Casting" in stage_map and "Rolling" in stage_map:
                pairs.append(("Casting", "Rolling"))
            for a, b in pairs:
                try:
                    gap = float(stage_map[b]["start_min"]) - float(stage_map[a]["end_min"])
                except Exception:
                    continue
                if gap >= 0:
                    gaps.setdefault(f"{a}->{b}", []).append(gap)

        out = {
            "BOF->LF": {"min": 0.0, "median": 0.0, "max": 90.0},
            "BOF->Casting": {"min": 0.0, "median": 0.0, "max": 120.0},
            "LF->Casting": {"min": 0.0, "median": 0.0, "max": 90.0},
            "Casting->Rolling": {"min": 0.0, "median": 30.0, "max": 360.0},
        }
        for pair, vals in gaps.items():
            vals = sorted(float(v) for v in vals if v is not None and float(v) >= 0)
            if not vals:
                continue
            n = len(vals)
            med = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
            mn, mx = vals[0], vals[-1]
            span = max(mx - mn, 1.0)
            out[pair] = {"min": round(max(0.0, mn - 0.10 * span), 4), "median": round(med, 4), "max": round(mx + 0.25 * span + 5.0, 4)}
        return out

    def _build_unit_context_for_projection(self, raw_heats: List[Dict[str, Any]], raw_rolling: List[Dict[str, Any]]) -> Tuple[Dict[str, List[str]], Dict[str, int]]:

        bof_units, lf_units, casting_units, rolling_units = [], [], [], []
        for heat in self._safe_list(raw_heats):
            if not isinstance(heat, dict):
                continue
            furnace = heat.get("Furnace")
            if furnace is not None and str(furnace) not in bof_units:
                bof_units.append(str(furnace))
            rt = self._safe_dict(heat.get("production_route_time"))
            for key in ["LF_Unit", "LF", "lf_unit"]:
                if rt.get(key) and str(rt.get(key)) not in lf_units:
                    lf_units.append(str(rt.get(key)))
            for key in ["Caster", "CC", "Casting_Unit", "casting_unit"]:
                if rt.get(key) and str(rt.get(key)) not in casting_units:
                    casting_units.append(str(rt.get(key)))
        for ru in self._safe_list(raw_rolling):
            if not isinstance(ru, dict):
                continue
            uid = ru.get("unit_id") or ru.get("u") or ru.get("rolling_unit")
            if uid is not None and str(uid) not in rolling_units:
                rolling_units.append(str(uid))
        if not bof_units:
            bof_units = ["Converter1", "Converter2", "Converter3", "1#LDG Generation", "2#LDG Generation", "3#LDG Generation"]
        if not lf_units:
            lf_units = ["LF", "LF1", "LF2"]
        if not casting_units:
            casting_units = ["Casting", "CC", "CC1", "CC2"]
        if not rolling_units:
            rolling_units = ["Rolling", "1130", "1131", "1132"]
        comp = {"BOF": self._dedup_preserve_order(bof_units), "LF": self._dedup_preserve_order(lf_units), "Casting": self._dedup_preserve_order(casting_units), "Rolling": self._dedup_preserve_order(rolling_units)}
        cap = {str(u): 1 for units in comp.values() for u in units}
        return comp, cap

    def _build_rolling_queue_prior_for_projection(self, raw_rolling: List[Dict[str, Any]]) -> Dict[str, List[str]]:

        out: Dict[str, List[str]] = {}
        for ru in self._safe_list(raw_rolling):
            if not isinstance(ru, dict):
                continue
            uid = ru.get("unit_id") or ru.get("u") or ru.get("rolling_unit")
            if uid is None:
                continue
            heats = []
            for hid in self._safe_list(ru.get("Heat_No_List") or ru.get("hs") or ru.get("heat_ids")):
                nhid = self._normalize_heat_id(hid)
                if nhid and nhid not in heats:
                    heats.append(nhid)
            if heats:
                out[str(uid)] = heats
        return out

    def _build_casting_sequence_prior_for_projection(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

        by_unit: Dict[str, List[Dict[str, Any]]] = {}
        for seg in self._safe_list(segments):
            if not isinstance(seg, dict) or self._canonical_stage(seg.get("stage")) != "Casting":
                continue
            by_unit.setdefault(str(seg.get("unit") or "Casting"), []).append(seg)
        groups: List[Dict[str, Any]] = []
        for unit, rows in by_unit.items():
            rows = sorted(rows, key=lambda x: float(x.get("start_min", 10**12)))
            heat_ids = [self._normalize_heat_id(x.get("heat_id")) for x in rows]
            heat_ids = [x for x in heat_ids if x]
            if len(heat_ids) >= 2:
                groups.append({"group_id": f"visible_cast_sequence_{unit}", "unit": unit, "heat_ids": self._dedup_preserve_order(heat_ids), "max_gap": 5.0, "source": "visible_history_operation_segments"})
        return groups

    def _build_energy_coefficients_for_projection(self) -> Dict[str, Any]:

        return {
            "bucket_min": 5.0,
            "stage_coefficients": {
                "BOF": {"power": 1.0, "steam": 0.4, "ldg_byproduct": 1.0},
                "LF": {"power": 0.8, "steam": 0.2, "ldg_byproduct": 0.0},
                "Casting": {"power": 0.5, "steam": 0.6, "ldg_byproduct": 0.0},
                "Rolling": {"power": 1.2, "steam": 0.8, "ldg_byproduct": 0.0},
            },
            "proxy_limits": {"power": 6.0, "steam": 6.0, "ldg_byproduct": 4.0},
        }

    def _build_pcvp_repair_context(
        self,
        day_id: str,
        future_window: Dict[str, Any],
        known_future_calendar: List[Dict[str, Any]],
        past_operation_segments: List[Dict[str, Any]],
        history_operation_segments: List[Dict[str, Any]],
        duration_summary: Dict[str, Any],
    ) -> Dict[str, Any]:

        day_obj = self.get_day_object(day_id)
        raw_heats = self._safe_list(day_obj.get("heats"))
        raw_rolling = self._safe_list(day_obj.get("rolling_units"))
        unit_compatibility, unit_capacity = self._build_unit_context_for_projection(raw_heats, raw_rolling)
        return {
            "method": "LLM-guided Physics-consistent Variational Projection",
            "method_short": "PCVP",
            "uses_future_target_labels": False,
            "horizon_window": {"start_min": future_window.get("t0_min"), "end_min": future_window.get("t1_min"), "t0": future_window.get("t0"), "t1": future_window.get("t1")},
            "stage_order": ["BOF", "LF", "Casting", "Rolling"],
            "duration_priors": self._duration_priors_for_projection(duration_summary),
            "waiting_time_priors": self._build_waiting_time_priors_for_projection(past_operation_segments),
            "unit_compatibility": unit_compatibility,
            "unit_capacity": unit_capacity,
            "history_tail_operation_segments": self._compact_segments_for_prompt(history_operation_segments, max_segments=80),
            "known_future_patch_calendar": self._future_calendar_for_prompt(known_future_calendar),
            "rolling_queue_prior": self._build_rolling_queue_prior_for_projection(raw_rolling),
            "casting_sequence_prior": self._build_casting_sequence_prior_for_projection(past_operation_segments),
            "energy_coefficients": self._build_energy_coefficients_for_projection(),
            "constraints": {
                "route_precedence": "BOF -> optional LF -> Casting -> Rolling",
                "unit_capacity": "same unit no-overlap unless capacity > 1",
                "waiting_windows": "inter-stage transfer windows estimated from visible history",
                "casting_continuity": "approximate continuous casting groups from visible history when available",
                "rolling_queue": "rolling unit heat order from historical rolling unit Heat_No_List",
                "no_label_leakage": "target_operation_segments are excluded from this context",
            },
        }

    def _build_candidate_future_heat_ids(
        self,
        history: List[Dict[str, Any]],
        anchor: Dict[str, Any],
        future_calendar: List[Dict[str, Any]],
    ) -> List[str]:

        ids: List[str] = []

        for p in self._safe_list(history) + [self._safe_dict(anchor)]:
            for hid in self._safe_list(self._safe_dict(p).get("active_heat_ids")):
                norm = self._normalize_heat_id(hid)
                if norm:
                    ids.append(norm)

        for item in self._safe_list(future_calendar):
            if not isinstance(item, dict):
                continue
            hid = self._normalize_heat_id(
                item.get("canonical_heat_id") or
                item.get("heat_id") or
                item.get("hn") or
                self._extract_heat_id_from_patch_id(item.get("pid")) or
                item.get("pid")
            )
            if hid:
                ids.append(hid)

        return self._dedup_preserve_order(ids)

    def _future_calendar_for_prompt(self, future_calendar: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

        out: List[Dict[str, Any]] = []
        for item in self._safe_list(future_calendar):
            if not isinstance(item, dict):
                continue
            out.append({
                "step": item.get("step"),
                "pi": item.get("pi"),
                "pid": item.get("pid"),
                "t0": item.get("t0"),
                "t1": item.get("t1"),
                "t0_min": item.get("t0_min"),
                "t1_min": item.get("t1_min"),
                "canonical_heat_id": item.get("canonical_heat_id"),
                "converter": item.get("converter"),
            })
        return out

    def _build_local_schedule_blocks(
        self,
        known_future_calendar: List[Dict[str, Any]],
        candidate_future_heat_ids: List[str],
        future_window: Dict[str, Any],
        block_size: int = 5,
    ) -> List[Dict[str, Any]]:

        calendar = [x for x in self._safe_list(known_future_calendar) if isinstance(x, dict)]
        if not calendar:
            return []

        block_size = max(1, int(block_size or 5))
        blocks: List[Dict[str, Any]] = []
        for b0 in range(0, len(calendar), block_size):
            block_rows = calendar[b0:b0 + block_size]
            heat_ids = []
            for row in block_rows:
                hid = self._normalize_heat_id(
                    row.get("canonical_heat_id") or row.get("heat_id") or row.get("hn") or row.get("pid")
                )
                if hid and hid not in heat_ids:
                    heat_ids.append(hid)

            local_candidates = []
            for hid in list(candidate_future_heat_ids or []) + heat_ids:
                nh = self._normalize_heat_id(hid)
                if nh and nh not in local_candidates:
                    local_candidates.append(nh)

            starts = [row.get("t0_min") for row in block_rows if row.get("t0_min") is not None]
            ends = [row.get("t1_min") for row in block_rows if row.get("t1_min") is not None]

            blocks.append({
                "block_id": f"B{len(blocks) + 1:02d}",
                "step_range": [block_rows[0].get("step"), block_rows[-1].get("step")],
                "num_steps": len(block_rows),
                "time_window": {
                    "t0": block_rows[0].get("t0"),
                    "t1": block_rows[-1].get("t1"),
                    "t0_min": min(starts) if starts else future_window.get("t0_min"),
                    "t1_min": max(ends) if ends else future_window.get("t1_min"),
                },
                "known_calendar": [
                    {
                        "step": row.get("step"),
                        "pi": row.get("pi"),
                        "pid": row.get("pid"),
                        "canonical_heat_id": self._normalize_heat_id(
                            row.get("canonical_heat_id") or row.get("heat_id") or row.get("hn") or row.get("pid")
                        ),
                        "converter": row.get("converter"),
                        "t0": row.get("t0"),
                        "t1": row.get("t1"),
                        "t0_min": row.get("t0_min"),
                        "t1_min": row.get("t1_min"),
                    }
                    for row in block_rows
                ],
                "candidate_heat_ids": local_candidates,
                "required_output": "operation-level segments overlapping this block; final response must still return one combined predicted_operation_segments list",
            })

        return blocks

    def _build_heat_route_skeletons(
        self,
        candidate_future_heat_ids: List[str],
        known_future_calendar: List[Dict[str, Any]],
        anchor: Dict[str, Any],
        history_operation_segments: List[Dict[str, Any]],
        past_operation_segments: List[Dict[str, Any]],
        future_window: Dict[str, Any],
    ) -> List[Dict[str, Any]]:

        stage_order = ["BOF", "LF", "Casting", "Rolling"]

        def _stage_rank(stage: Any) -> int:
            s = str(stage or "")
            return stage_order.index(s) if s in stage_order else -1

        calendar_by_heat: Dict[str, List[Dict[str, Any]]] = {}
        for row in self._safe_list(known_future_calendar):
            if not isinstance(row, dict):
                continue
            hid = self._normalize_heat_id(
                row.get("canonical_heat_id") or row.get("heat_id") or row.get("hn") or row.get("pid")
            )
            if hid:
                calendar_by_heat.setdefault(hid, []).append(row)

        segments_by_heat: Dict[str, List[Dict[str, Any]]] = {}
        for seg in self._safe_list(past_operation_segments) + self._safe_list(history_operation_segments):
            if not isinstance(seg, dict):
                continue
            hid = self._normalize_heat_id(seg.get("heat_id") or seg.get("id") or seg.get("hn"))
            if hid:
                segments_by_heat.setdefault(hid, []).append(seg)

        carryover_by_heat: Dict[str, Dict[str, Any]] = {}
        for h in self._safe_list(anchor.get("active_heats")):
            if not isinstance(h, dict):
                continue
            hid = self._normalize_heat_id(h.get("id") or h.get("heat_id") or h.get("hn"))
            if hid:
                carryover_by_heat[hid] = h

        all_heat_ids: List[str] = []
        for hid in list(candidate_future_heat_ids or []) + list(calendar_by_heat.keys()) + list(carryover_by_heat.keys()):
            nh = self._normalize_heat_id(hid)
            if nh and nh not in all_heat_ids:
                all_heat_ids.append(nh)

        skeletons: List[Dict[str, Any]] = []
        for hid in all_heat_ids:
            hist = [x for x in segments_by_heat.get(hid, []) if isinstance(x, dict)]
            hist.sort(key=lambda x: self._safe_float(x.get("end_min"), self._safe_float(x.get("start_min"), 0.0)))
            observed_stages: List[str] = []
            last_stage = None
            last_end = None
            last_unit = None
            for seg in hist:
                st = str(seg.get("stage") or "")
                if st in stage_order and st not in observed_stages:
                    observed_stages.append(st)
                if st in stage_order:
                    last_stage = st
                    last_end = seg.get("end_min")
                    last_unit = seg.get("unit")

            carry = self._safe_dict(carryover_by_heat.get(hid))
            carry_stage = carry.get("stage")
            carry_stages = [x for x in self._safe_list(carry.get("stages")) if x in stage_order]
            if carry_stage in stage_order and carry_stage not in observed_stages:
                observed_stages.append(carry_stage)
            for st in carry_stages:
                if st not in observed_stages:
                    observed_stages.append(st)
            if carry_stage in stage_order:
                last_stage = carry_stage

            cal_rows = calendar_by_heat.get(hid, [])
            known_future_anchor = bool(cal_rows)
            calendar_steps = [row.get("step") for row in cal_rows if row.get("step") is not None]
            calendar_units = [
                row.get("converter") for row in cal_rows
                if row.get("converter") is not None
            ]
            calendar_start_candidates = [
                row.get("t0_min") for row in cal_rows
                if row.get("t0_min") is not None
            ]

            if known_future_anchor and "BOF" not in observed_stages:

                candidate_next = ["BOF", "LF", "Casting", "Rolling"]
            elif last_stage in stage_order:
                idx = _stage_rank(last_stage)
                candidate_next = stage_order[idx + 1:] if idx >= 0 and idx + 1 < len(stage_order) else []
            else:
                candidate_next = stage_order[:]

            if carry_stage in stage_order and carry_stage not in candidate_next:
                candidate_next = [carry_stage] + candidate_next

            skeletons.append({
                "heat_id": hid,
                "known_future_anchor": known_future_anchor,
                "calendar_steps": calendar_steps,
                "calendar_units": self._dedup_preserve_order(calendar_units),
                "calendar_first_start_min": min(calendar_start_candidates) if calendar_start_candidates else None,
                "observed_stages_before_horizon": observed_stages,
                "last_known_stage": last_stage,
                "last_known_end_min": last_end,
                "last_known_unit": last_unit,
                "carryover_at_anchor": bool(carry),
                "anchor_stage": carry_stage if carry_stage in stage_order else None,
                "candidate_stages_in_horizon": candidate_next,
                "allowed_stage_order": stage_order,
                "time_window": {
                    "earliest_start_min": future_window.get("t0_min"),
                    "latest_end_min": future_window.get("t1_min"),
                },
                "retry_hint": (
                    "If the first response misses this heat, retry should infer only stages supported by "
                    "calendar anchor, carryover state, visible history, and duration/waiting priors."
                ),
            })

        skeletons.sort(key=lambda x: (not x.get("known_future_anchor"), not x.get("carryover_at_anchor"), str(x.get("heat_id"))))
        return skeletons

    def _build_retry_guidance_context(
        self,
        local_schedule_blocks: List[Dict[str, Any]],
        heat_route_skeletons: List[Dict[str, Any]],
    ) -> Dict[str, Any]:

        def _env_bool(name: str, default: bool) -> bool:
            val = os.environ.get(name)
            if val is None:
                return default
            return str(val).strip().lower() not in {"0", "false", "no", "off", "none"}

        try:
            max_retry_attempts = int(os.environ.get("PROD_MAX_HORIZON_RETRIES", "5"))
        except Exception:
            max_retry_attempts = 5
        max_retry_attempts = max(0, max_retry_attempts)
        enable_rule_fallback = _env_bool("PROD_ENABLE_RULE_FALLBACK_AFTER_RETRY", True)
        evaluate_rule_fallback = _env_bool("PROD_EVALUATE_RULE_FALLBACK_AS_PREDICTION", True)

        return {
            "enabled": True,
            "max_retry_attempts": max_retry_attempts,
            "retry_trigger_conditions": [
                "missing predicted_operation_segments",
                "predicted_operation_segments is empty after source/schema validation",
                "JSON parse failed or output is not a JSON object/list",
                "PCVP cannot produce any repaired model_output segments",
            ],
            "allowed_feedback_to_llm": [
                "schema error",
                "missing required key",
                "number of raw/normalized/projected segments",
                "PCVP violation diagnostics",
                "future horizon window",
                "known_future_patch_calendar",
                "local_schedule_blocks",
                "heat_route_skeletons",
                "history_operation_segments and duration priors",
            ],
            "forbidden_feedback_to_llm": [
                "target_operation_segments",
                "ground-truth future operation start/end times",
                "IoU/MAE/F1 against target labels",
                "any future labels outside known calendar context",
            ],
            "local_schedule_blocks": local_schedule_blocks,
            "heat_route_skeletons": heat_route_skeletons,
            "retry_output_policy": (
                "Retry must regenerate one complete JSON object with predicted_operation_segments. "
                "The retry feedback contains only schema and mechanism diagnostics, not target labels."
            ),
            "retry_exhausted_rule_fallback": {
                "enabled": enable_rule_fallback,
                "trigger": "used only after all LLM retry attempts still produce missing/invalid operation segments",
                "method": "prediction-time rule-assisted schedule completion followed by PCVP",
                "source_name": "model_output_rule_assisted_after_retry" if evaluate_rule_fallback else "rule_assisted_after_retry_exhausted_diagnostic",
                "evaluate_as_prediction": evaluate_rule_fallback,
                "uses_future_target_labels": False,
                "allowed_basis": [
                    "known_future_patch_calendar",
                    "future horizon window",
                    "carryover active heats at anchor",
                    "heat_route_skeletons",
                    "local_schedule_blocks",
                    "visible history operation segments",
                    "duration priors and process order",
                ],
                "forbidden_basis": [
                    "target_operation_segments",
                    "ground-truth future operation start/end times",
                    "target-based IoU/MAE/F1",
                ],
                "reporting_note": (
                    "This fallback is a safety completion for retry-exhausted samples. "
                    "It must be reported separately from pure LLM retry results in ablation or diagnostic tables."
                ),
            },
        }

    def build_production_horizon_sample(
        self,
        day_id: str,
        anchor_patch_idx: int,
        history_window: int = 5,
        horizon: int = 5,
        include_target: bool = True,
    ) -> Dict[str, Any]:

        seq = self.build_production_patch_sequence(day_id)
        if not seq:
            raise ValueError(f"No production patch sequence available for day_id={day_id}")

        n = len(seq)
        if anchor_patch_idx < 0 or anchor_patch_idx >= n:
            raise IndexError(f"anchor_patch_idx={anchor_patch_idx} out of range [0,{n - 1}]")

        h0 = max(0, anchor_patch_idx - history_window + 1)
        h1 = anchor_patch_idx + 1
        f0 = anchor_patch_idx + 1
        f1 = min(n, anchor_patch_idx + 1 + horizon)

        history = deepcopy(seq[h0:h1])
        future = deepcopy(seq[f0:f1])
        anchor = deepcopy(seq[anchor_patch_idx])

        history_t0 = history[0].get("t0") if history else None
        history_t1 = history[-1].get("t1") if history else None
        past_t0 = seq[0].get("t0") if seq else history_t0
        past_t1 = anchor.get("t1") if anchor else history_t1
        future_t0 = future[0].get("t0") if future else None
        future_t1 = future[-1].get("t1") if future else None

        history_operation_segments = self.build_operation_segments_for_time_window(
            day_id=day_id,
            window_t0=history_t0,
            window_t1=history_t1,
            source="history_window_true_route_segments",
        ) if history_t0 is not None and history_t1 is not None else []

        past_operation_segments = self.build_operation_segments_for_time_window(
            day_id=day_id,
            window_t0=past_t0,
            window_t1=past_t1,
            source="past_visible_true_route_segments",
        ) if past_t0 is not None and past_t1 is not None else []

        target_operation_segments = self.build_operation_segments_for_time_window(
            day_id=day_id,
            window_t0=future_t0,
            window_t1=future_t1,
            source="target_true_route_segments",
        ) if future_t0 is not None and future_t1 is not None else []

        target_patch_operation_segments = self.build_operation_segments_from_patch_states(
            future,
            source="target_patch_state_segments",
        )

        known_future_calendar = []
        for i, p in enumerate(future):
            known_future_calendar.append({
                "step": i + 1,
                "pi": p.get("pi"),
                "pid": p.get("pid"),
                "t0": p.get("t0"),
                "t1": p.get("t1"),
                "t0_min": p.get("t0_min"),
                "t1_min": p.get("t1_min"),
                "canonical_heat_id": self._normalize_heat_id(
                    p.get("canonical_heat_id") or
                    p.get("heat_id") or
                    p.get("hn") or
                    self._extract_heat_id_from_patch_id(p.get("pid")) or
                    p.get("pid")
                ),
                "converter": p.get("converter") or self._extract_converter_from_patch_id(p.get("pid")),
                "source_day_id": p.get("source_day_id"),
                "source": p.get("source"),
            })

        candidate_future_heat_ids = self._build_candidate_future_heat_ids(
            history=history,
            anchor=anchor,
            future_calendar=known_future_calendar,
        )

        source_days = sorted({str(x.get("source_day_id")) for x in history + future if x.get("source_day_id")})
        mixed_day_warning = bool(source_days and any(x != str(day_id) for x in source_days))
        fallback_warning = any(x.get("source") == "rolling_unit_fallback" for x in history + future)

        visible_history_segments = self._compact_segments_for_prompt(
            past_operation_segments,
            max_segments=120,
        )
        local_history_segments = self._compact_segments_for_prompt(
            history_operation_segments,
            max_segments=60,
        )
        duration_priors = self._summarize_operation_segments_for_prompt(past_operation_segments)

        future_window = {
            "t0": future_t0,
            "t1": future_t1,
            "t0_min": self._time_to_min(future_t0),
            "t1_min": self._time_to_min(future_t1),
        }

        repair_context = self._build_pcvp_repair_context(
            day_id=day_id,
            future_window=future_window,
            known_future_calendar=known_future_calendar,
            past_operation_segments=past_operation_segments,
            history_operation_segments=history_operation_segments,
            duration_summary=duration_priors,
        )

        local_schedule_blocks = self._build_local_schedule_blocks(
            known_future_calendar=known_future_calendar,
            candidate_future_heat_ids=candidate_future_heat_ids,
            future_window=future_window,
            block_size=5,
        )
        heat_route_skeletons = self._build_heat_route_skeletons(
            candidate_future_heat_ids=candidate_future_heat_ids,
            known_future_calendar=known_future_calendar,
            anchor=anchor,
            history_operation_segments=history_operation_segments,
            past_operation_segments=past_operation_segments,
            future_window=future_window,
        )
        retry_guidance_context = self._build_retry_guidance_context(
            local_schedule_blocks=local_schedule_blocks,
            heat_route_skeletons=heat_route_skeletons,
        )

        sample = {
            "task": "production_horizon_prediction_sample",
            "day_id": day_id,
            "anchor_patch_idx": anchor_patch_idx,
            "history_window": history_window,
            "horizon": horizon,
            "input_state": {
                "mode": "production_gantt_schedule_context_only",
                "anchor_patch": anchor,
                "history_patches": history,
                "known_future_patch_calendar": known_future_calendar,
                "known_future_time_window": future_window,
                "repair_context": repair_context,
                "physics_projection_task": {
                    "method": "LLM-guided Physics-consistent Variational Projection",
                    "method_short": "PCVP",
                    "purpose": "post-LLM mechanism-consistent local state correction for predicted_operation_segments",
                    "uses_future_target_labels": False,
                    "main_constraints": [
                        "BOF -> optional LF -> Casting -> Rolling precedence",
                        "same-unit capacity and no-overlap",
                        "duration and inter-stage waiting priors from visible history",
                        "continuous casting sequence when available",
                        "rolling queue order and no shared rolling-window copying",
                        "production-energy proxy consistency diagnostics"
                    ]
                },
                "local_schedule_blocks": local_schedule_blocks,
                "heat_route_skeletons": heat_route_skeletons,
                "retry_guidance_context": retry_guidance_context,

                "operation_schedule_task": {
                    "objective": (
                        "Predict only operation-level predicted_operation_segments that can be "
                        "drawn as a production Gantt chart for the future window. "
                        "Do not ask the production LLM to predict patch-level active_heat_ids; "
                        "patch-level states are derived later from operation segments if needed."
                    ),
                    "candidate_future_heat_ids": candidate_future_heat_ids,
                    "required_process_order": ["BOF", "LF", "Casting", "Rolling"],
                    "future_calendar": self._future_calendar_for_prompt(known_future_calendar),
                    "future_time_window": future_window,
                    "carryover_active_heats_at_anchor": anchor.get("active_heats", []),
                    "visible_history_operation_segments": visible_history_segments,
                    "local_history_operation_segments": local_history_segments,
                    "duration_priors": duration_priors,
                    "local_schedule_blocks": local_schedule_blocks,
                    "heat_route_skeletons": heat_route_skeletons,
                    "retry_guidance": {
                        "enabled": True,
                        "max_retry_attempts": retry_guidance_context.get("max_retry_attempts", 5),
                        "purpose": "mechanism-loss-guided self-refinement when predicted_operation_segments are missing or invalid",
                        "does_not_use_target_future": True,
                        "retry_exhausted_rule_fallback": retry_guidance_context.get("retry_exhausted_rule_fallback", {}),
                    },
                    "rules": [
                        "For every candidate heat that overlaps the future window, output all BOF/LF/Casting/Rolling segments that overlap the window.",
                        "Each segment must include heat_id, stage, unit, start_time, end_time, start_min, end_min, duration_min, source_steps, source, and confidence.",
                        "Every model-predicted segment must set source=\"model_output\". Do not label derived, calendar, patch-state, or fallback segments as model_output.",
                        "Segments must be in chronological order and clipped to the future_time_window.",
                        "Use future_calendar canonical_heat_id as known future BOF/start identifiers, but infer LF/Casting/Rolling timing from historical route patterns and process-order constraints.",
                        "Do not output only one BOF segment per patch. The predicted_operation_segments list should describe concurrent multi-heat operation occupancy across the whole horizon.",
                        "This is a Gantt-only production task; do not generate a separate future_production_states task.",
                    ],
                },

                "history_operation_segments": history_operation_segments,
                "operation_segment_schema": {
                    "heat_id": "canonical heat id",
                    "stage": "BOF|LF|Casting|Rolling",
                    "unit": "converter/LF/Casting/rolling unit",
                    "start_time": "absolute timestamp when available",
                    "end_time": "absolute timestamp when available",
                    "start_min": "relative minute, clipped to the visible horizon window",
                    "end_min": "relative minute, clipped to the visible horizon window",
                    "duration_min": "end_min - start_min",
                    "source_steps": "future patch steps overlapped by this segment",
                    "source": "must be model_output for model-predicted segments; target/history/patch-derived segments are diagnostic only",
                    "confidence": "model confidence in [0, 1]",
                },
                "schema": {
                    "active_heats": "heats active in the patch time window",
                    "stage": "representative process stage among BOF/LF/Casting/Rolling",
                    "stages": "all overlapped process stages within the patch",
                    "active_rolling_units": "rolling units active in the patch",
                    "canonical_heat_id": "heat id parsed from future patch id, known calendar identifier, not target label",
                },
            },
            "meta": {
                "num_all_patches": n,
                "num_history_patches": len(history),
                "num_future_patches": len(future),
                "is_truncated_future": len(future) < horizon,
                "patch_source_days": source_days,
                "mixed_day_warning": mixed_day_warning,
                "rolling_unit_fallback_warning": fallback_warning,
                "num_candidate_future_heat_ids": len(candidate_future_heat_ids),
                "num_visible_history_operation_segments": len(visible_history_segments),
                "num_local_history_operation_segments": len(local_history_segments),
                "num_repair_context_history_segments": len(repair_context.get("history_tail_operation_segments", [])),
                "num_repair_context_casting_sequence_groups": len(repair_context.get("casting_sequence_prior", [])),
                "num_repair_context_rolling_units": len(repair_context.get("rolling_queue_prior", {})),
                "num_local_schedule_blocks": len(local_schedule_blocks),
                "num_heat_route_skeletons": len(heat_route_skeletons),
                "retry_self_refinement_enabled": True,
                "retry_max_attempts": retry_guidance_context.get("max_retry_attempts", 2),
                "pcvp_projection_enabled": True,
                "num_target_operation_segments": len(target_operation_segments),
                "operation_segment_protocol": "predicted_operation_segments with source=model_output are the only valid Gantt predictions; patch-derived/fallback segments are diagnostics only",
                "task_reduction_note": "Production LLM prompt is reduced to operation-level Gantt prediction only; patch-level active-heat alignment is removed from the production task.",
            }
        }

        if include_target:
            sample["target_future"] = {
                "mode": "production_patch_future_label",
                "future_patches": future,
                "operation_segments": target_operation_segments,
                "patch_operation_segments": target_patch_operation_segments,
                "operation_segments_valid_for_gantt": True,
                "patch_operation_segments_valid_for_gantt": False,
                "operation_segment_note": (
                    "operation_segments are exact route-level segments clipped to the future horizon window and are the target Gantt labels. "
                    "patch_operation_segments are merged from future patch labels and are diagnostics only. "
                    "These labels are used only for offline evaluation and must not be included in model input."
                ),
            }
        return sample

    def build_production_horizon_dataset(
        self,
        day_id: str,
        history_window: int = 5,
        horizon: int = 5,
        stride: int = 1,
        max_samples: Optional[int] = None,
    ) -> List[Dict[str, Any]]:

        seq = self.build_production_patch_sequence(day_id)
        n = len(seq)
        samples: List[Dict[str, Any]] = []

        last_anchor = n - int(horizon) - 1
        if last_anchor < history_window - 1:
            return samples
        for anchor in range(history_window - 1, last_anchor + 1, max(1, stride)):
            sample = self.build_production_horizon_sample(
                day_id=day_id,
                anchor_patch_idx=anchor,
                history_window=history_window,
                horizon=horizon,
                include_target=True,
            )
            samples.append(sample)
            if max_samples is not None and len(samples) >= max_samples:
                break
        return samples

    def build_production_horizon_prompt_state(
        self,
        day_id: str,
        anchor_patch_idx: int,
        history_window: int = 5,
        horizon: int = 5,
    ) -> Dict[str, Any]:

        sample = self.build_production_horizon_sample(
            day_id=day_id,
            anchor_patch_idx=anchor_patch_idx,
            history_window=history_window,
            horizon=horizon,
            include_target=False,
        )
        return sample["input_state"]

    def build_prompt_state(self, day_id: str, mode: str = "production") -> Dict[str, Any]:
        full_state = self.build_prod_state(day_id)
        prompt_views = self._build_prompt_views(full_state)

        mode_alias = {
            "production": "prod",
            "prod": "prod",
            "energy": "energy",
            "coupling": "coupling",
        }

        normalized_mode = mode_alias.get(mode)
        if normalized_mode is None or normalized_mode not in prompt_views:
            raise ValueError(f"Unsupported mode: {mode}")

        return prompt_views[normalized_mode]

    def build_long_memory(self, day_id: str) -> Dict[str, Any]:
        opt = self.get_optimization_result(day_id)

        long_memory = {
            "d": day_id,
            "opt": bool(opt),
        }

        if self.include_optimization_context_in_memory:
            long_memory["opt_ctx"] = self._extract_optimization_context(opt)

        return long_memory

    def _extract_optimization_context(self, opt: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not opt:
            return {"opt": False}

        return {
            "opt": True,
            "c": opt.get("cost"),
            "sav": opt.get("savings"),
            "fea": opt.get("feasibility_check"),
            "na": len(opt.get("proposed_adjustments", []) or []),
        }

    def build_transition_index(self, day_id: str) -> Dict[str, Any]:
        prod_state = self.build_prod_state(day_id)

        heat_map = {
            h["id"]: h for h in prod_state.get("hs", [])
            if isinstance(h, dict) and h.get("id")
        }
        rolling_map = {
            r["u"]: r for r in prod_state.get("rs", [])
            if isinstance(r, dict) and r.get("u")
        }
        graph_map = {
            g["id"]: g for g in prod_state.get("g", [])
            if isinstance(g, dict) and g.get("id")
        }

        return {
            "d": day_id,
            "hm": heat_map,
            "rm": rolling_map,
            "gm": graph_map,
        }

    def debug_size_report(self, day_id: str) -> Dict[str, int]:
        day_obj = self.get_day_object(day_id)
        full_state = self.build_prod_state(day_id)
        prod_prompt_state = self.build_prompt_state(day_id, mode="production")
        energy_prompt_state = self.build_prompt_state(day_id, mode="energy")
        coupling_prompt_state = self.build_prompt_state(day_id, mode="coupling")
        long_memory = self.build_long_memory(day_id)

        report = {
            "raw_day_obj_chars": len(json.dumps(day_obj, ensure_ascii=False)),
            "full_prod_state_chars": len(json.dumps(full_state, ensure_ascii=False)),
            "production_prompt_state_chars": len(json.dumps(prod_prompt_state, ensure_ascii=False)),
            "energy_prompt_state_chars": len(json.dumps(energy_prompt_state, ensure_ascii=False)),
            "coupling_prompt_state_chars": len(json.dumps(coupling_prompt_state, ensure_ascii=False)),
            "long_memory_chars": len(json.dumps(long_memory, ensure_ascii=False)),
        }

        print("\n=== ProductionStateAdapter Size Report ===")
        for k, v in report.items():
            print(f"{k}: {v}")

        return report

    def audit_prod_state_v2(self, day_id: str, prod_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if prod_state is None:
            prod_state = self.build_prod_state(day_id)

        report = {
            "d": day_id,
            "passed": True,
            "errors": [],
            "warnings": [],
            "stats": {},
            "checks": {}
        }

        def add_error(check, detail, extra=None):
            report["passed"] = False
            item = {"check": check, "detail": detail}
            if extra is not None:
                item["extra"] = extra
            report["errors"].append(item)

        def add_warning(check, detail, extra=None):
            item = {"check": check, "detail": detail}
            if extra is not None:
                item["extra"] = extra
            report["warnings"].append(item)

        expected_keys = ["d", "s", "cf", "hs", "rs", "ph", "meta"]
        missing_keys = [k for k in expected_keys if k not in prod_state]
        if missing_keys:
            add_error("top_keys", "full_prod_state is missing required top-level fields", {"missing": missing_keys})

        hs = prod_state.get("hs", []) or []
        rs = prod_state.get("rs", []) or []
        ph = prod_state.get("ph", []) or []
        g = prod_state.get("g", []) or []
        s = prod_state.get("s", {}) or {}
        meta = prod_state.get("meta", {}) or {}

        report["stats"]["counts"] = {
            "nh_hs": len(hs),
            "nrs_rs": len(rs),
            "nph_ph": len(ph),
            "ng_g": len(g),
        }

        if "nh" in s and s["nh"] != len(hs):
            add_warning("summary_count", "s['nh'] does not match len(hs)", {"s_nh": s["nh"], "len_hs": len(hs)})

        if "nru" in s and s["nru"] != len(rs):
            add_warning("summary_count", "s['nru'] does not match len(rs)", {"s_nru": s["nru"], "len_rs": len(rs)})

        if "nh" in meta and meta["nh"] != len(hs):
            add_warning("meta_count", "meta['nh'] does not match len(hs)", {"meta_nh": meta["nh"], "len_hs": len(hs)})

        if "nru" in meta and meta["nru"] != len(rs):
            add_warning("meta_count", "meta['nru'] does not match len(rs)", {"meta_nru": meta["nru"], "len_rs": len(rs)})

        required_heat_fields = ["id", "f", "tb0"]
        heat_field_stats = {}
        for field in required_heat_fields:
            present = sum(1 for h in hs if h.get(field) not in [None, "", []])
            cov = present / len(hs) if hs else 0.0
            heat_field_stats[field] = {"present": present, "total": len(hs), "coverage": round(cov, 4)}
            if cov == 0:
                add_error("heat_field", f"critical heat field {field} is completely missing")
            elif cov < 0.9:
                add_warning("heat_field", f"critical heat field {field} has low coverage", heat_field_stats[field])

        report["checks"]["heat_field_coverage"] = heat_field_stats

        required_rolling_fields = ["u", "gap", "hs", "c0", "c1"]
        rolling_field_stats = {}
        for field in required_rolling_fields:
            present = sum(1 for r in rs if r.get(field) not in [None, ""])
            cov = present / len(rs) if rs else 0.0
            rolling_field_stats[field] = {"present": present, "total": len(rs), "coverage": round(cov, 4)}
            if cov == 0:
                add_error("rolling_field", f"critical rolling field {field} is completely missing")
            elif cov < 0.9:
                add_warning("rolling_field", f"critical rolling field {field} has low coverage", rolling_field_stats[field])

        report["checks"]["rolling_field_coverage"] = rolling_field_stats

        heat_ids = [h.get("id") for h in hs if h.get("id") is not None]
        rolling_ids = [r.get("u") for r in rs if r.get("u") is not None]

        dup_heat_ids = sorted(list({x for x in heat_ids if heat_ids.count(x) > 1}))
        dup_rolling_ids = sorted(list({x for x in rolling_ids if rolling_ids.count(x) > 1 and x != "gap"}))

        if dup_heat_ids:
            add_error("id_uniqueness", "duplicate heat IDs exist", {"dup_heat_ids": dup_heat_ids[:20]})
        if dup_rolling_ids:
            add_warning("id_uniqueness", "duplicate rolling unit IDs exist", {"dup_rolling_ids": dup_rolling_ids[:20]})

        hm = {h["id"]: h for h in hs if h.get("id")}
        rm = {r["u"]: r for r in rs if r.get("u")}

        mapping_issues = []

        for r in rs:
            unit_id = r.get("u")
            heat_list = r.get("hs", []) or []
            for hid in heat_list:
                if hid not in hm:
                    mapping_issues.append({"type": "rolling_ref_missing_heat", "u": unit_id, "id": hid})
                else:
                    if hm[hid].get("ru") != unit_id:
                        mapping_issues.append({
                            "type": "rolling_heat_ru_mismatch",
                            "u": unit_id,
                            "id": hid,
                            "heat_ru": hm[hid].get("ru")
                        })

        for h in hs:
            hid = h.get("id")
            ru = h.get("ru")
            if ru is None:
                add_warning("heat_rolling", "some heats do not explicitly provide ru", {"id": hid})
                continue
            if ru not in rm:
                mapping_issues.append({"type": "heat_ref_missing_rolling", "id": hid, "ru": ru})
            else:
                if hid not in (rm[ru].get("hs", []) or []):
                    mapping_issues.append({"type": "heat_not_in_rolling_hs", "id": hid, "ru": ru})

        report["checks"]["heat_rolling_mapping"] = {
            "num_issues": len(mapping_issues),
            "sample_issues": mapping_issues[:20]
        }

        if mapping_issues:
            add_warning("heat_rolling", "heat-to-rolling mapping is inconsistent", {"sample_issues": mapping_issues[:20]})

        if "g" in prod_state:
            graph_ids = [x.get("id") for x in g if x.get("id")]
            missing_in_graph = sorted(list(set(heat_ids) - set(graph_ids)))
            extra_in_graph = sorted(list(set(graph_ids) - set(heat_ids)))

            report["checks"]["graph_coverage"] = {
                "len_g": len(g),
                "missing_in_graph": missing_in_graph[:20],
                "extra_in_graph": extra_in_graph[:20]
            }

            if missing_in_graph:
                add_error("graph", "some heats are missing from the graph", {"missing_in_graph": missing_in_graph[:20]})

            if extra_in_graph:
                add_warning("graph", "the graph contains IDs that are absent from heat_states", {"extra_in_graph": extra_in_graph[:20]})

        peak_ids = [x.get("id") for x in ph if x.get("id")]
        missing_peak = [x for x in peak_ids if x not in hm]
        report["checks"]["peak_subset"] = {
            "num_peak": len(peak_ids),
            "missing_peak_ids": missing_peak[:20]
        }

        if missing_peak:
            add_warning("peak_subset", "some heats in ph are absent from hs", {"missing_peak_ids": missing_peak[:20]})

        prompt_checks = {}
        for mode in ["production", "energy", "coupling"]:
            ps = self.build_prompt_state(day_id, mode=mode)
            p_hs = ps.get("hs", []) or []
            p_rs = ps.get("rs", []) or []
            p_ph = ps.get("ph", []) or []

            p_heat_ids = [x.get("id") for x in p_hs if x.get("id")]
            p_rolling_ids = [x.get("u") for x in p_rs if x.get("u")]
            p_peak_ids = [x.get("id") for x in p_ph if x.get("id")]

            bad_heat_ids = [x for x in p_heat_ids if x not in hm]
            bad_rolling_ids = [x for x in p_rolling_ids if x not in rm]
            bad_peak_ids = [x for x in p_peak_ids if x not in peak_ids]

            prompt_checks[mode] = {
                "num_prompt_hs": len(p_hs),
                "num_prompt_rs": len(p_rs),
                "num_prompt_ph": len(p_ph),
                "bad_heat_ids": bad_heat_ids[:20],
                "bad_rolling_ids": bad_rolling_ids[:20],
                "bad_peak_ids": bad_peak_ids[:20],
            }

            if bad_heat_ids:
                add_error(f"prompt_{mode}", f"{mode} prompt hs is not a subset of the full state", {"bad_heat_ids": bad_heat_ids[:20]})
            if bad_rolling_ids:
                add_error(f"prompt_{mode}", f"{mode} prompt rs is not a subset of the full state", {"bad_rolling_ids": bad_rolling_ids[:20]})
            if bad_peak_ids:
                add_warning(f"prompt_{mode}", f"{mode} prompt ph is not a subset of full state ph", {"bad_peak_ids": bad_peak_ids[:20]})

        report["checks"]["prompt_subset"] = prompt_checks

        report["stats"]["size"] = {
            "full_chars": len(json.dumps(prod_state, ensure_ascii=False)),
            "production_prompt_chars": len(json.dumps(self.build_prompt_state(day_id, "production"), ensure_ascii=False)),
            "energy_prompt_chars": len(json.dumps(self.build_prompt_state(day_id, "energy"), ensure_ascii=False)),
            "coupling_prompt_chars": len(json.dumps(self.build_prompt_state(day_id, "coupling"), ensure_ascii=False)),
        }

        return report

_BaseProductionStateAdapter = ProductionStateAdapter

class ProductionStateAdapter(_BaseProductionStateAdapter):

    _STAGES = ("BOF", "LF", "Casting", "Rolling")

    def _clean_stage(self, x: Any) -> str:
        s = str(x or "").strip()
        u = s.upper().replace(" ", "").replace("-", "_")
        if not u:
            return "Unknown"
        if "BOF" in u or "CONVERTER" in u or "LDGGENERATION" in u or "#LDG" in u:
            return "BOF"
        if u == "LF" or "LADLE" in u or "REFIN" in u:
            return "LF"
        if "CAST" in u or u == "CC" or u.startswith("CC"):
            return "Casting"
        if "ROLL" in u or (u.isdigit() and u.startswith("11")):
            return "Rolling"
        return s if s in self._STAGES else "Unknown"

    def _seg_start_min(self, seg: Dict[str, Any], default: Optional[float] = None) -> Optional[float]:
        return self._safe_float(
            seg.get("start_min")
            or seg.get("raw_start_min")
            or seg.get("t0")
            or seg.get("start"),
            default,
        )

    def _seg_end_min(self, seg: Dict[str, Any], default: Optional[float] = None) -> Optional[float]:
        return self._safe_float(
            seg.get("end_min")
            or seg.get("raw_end_min")
            or seg.get("t1")
            or seg.get("end"),
            default,
        )

    def _segment_heat_id(self, seg: Dict[str, Any]) -> str:
        return self._normalize_heat_id(
            seg.get("heat_id")
            or seg.get("canonical_heat_id")
            or seg.get("Heat_No")
            or seg.get("id")
            or seg.get("hn")
        ) or ""

    def _segment_unit(self, seg: Dict[str, Any], stage: str) -> str:
        unit = (
            seg.get("unit")
            or seg.get("resource")
            or seg.get("machine")
            or seg.get("lane")
            or seg.get("converter")
            or ""
        )
        unit = str(unit).strip()
        if unit:
            return unit
        if stage == "BOF":
            return str(seg.get("converter") or "Converter")
        if stage == "LF":
            return "LF"
        if stage == "Casting":
            return "Casting"
        if stage == "Rolling":
            return str(seg.get("rolling_unit") or seg.get("ru") or "Rolling")
        return "Unknown"

    def _interval_overlaps(self, s: Optional[float], e: Optional[float], h0: float, h1: float) -> bool:
        if s is None or e is None:
            return False
        return float(s) < h1 and float(e) > h0

    def _interval_near_window(
        self,
        s: Optional[float],
        e: Optional[float],
        h0: float,
        h1: float,
        slack_min: float,
    ) -> bool:
        if s is None or e is None:
            return False
        return float(s) < h1 + slack_min and float(e) > h0 - slack_min

    def _candidate_key(self, heat_id: str, stage: str, unit: str = "") -> str:
        hid = self._normalize_heat_id(heat_id) or str(heat_id or "")
        return f"{hid}|{self._clean_stage(stage)}|{str(unit or '').strip()}"

    def _make_window_candidate(
        self,
        heat_id: str,
        stage: str,
        unit: str,
        start_min: Optional[float],
        end_min: Optional[float],
        horizon_window: Dict[str, Any],
        evidence_level: str,
        source_basis: str,
        confidence: float,
        raw: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        h0 = self._safe_float(horizon_window.get("start_min"), None)
        h1 = self._safe_float(horizon_window.get("end_min"), None)
        if h0 is None or h1 is None or h1 <= h0:
            return None

        stage = self._clean_stage(stage)
        heat_id = self._normalize_heat_id(heat_id) or str(heat_id or "")
        if not heat_id or stage not in self._STAGES:
            return None
        unit = str(unit or self._segment_unit(raw or {}, stage)).strip() or stage

        clipped_start = max(float(h0), float(start_min)) if start_min is not None else float(h0)
        clipped_end = min(float(h1), float(end_min)) if end_min is not None else float(h1)
        if clipped_end <= clipped_start:
            clipped_start, clipped_end = float(h0), float(h1)

        can_main = evidence_level in {"confirmed", "optional"}
        return {
            "candidate_id": self._candidate_key(heat_id, stage, unit),
            "heat_id": heat_id,
            "stage": stage,
            "unit": unit,
            "start_min": round(clipped_start, 6),
            "end_min": round(clipped_end, 6),
            "duration_min": round(max(0.0, clipped_end - clipped_start), 6),
            "evidence_level": evidence_level,
            "can_enter_main_prediction": bool(can_main),
            "can_enter_uncertain": True,
            "source_basis": source_basis,
            "confidence": float(max(0.0, min(1.0, confidence))),
            "raw_start_min": start_min,
            "raw_end_min": end_min,
        }

    def _dedup_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        priority = {"confirmed": 0, "optional": 1, "route_future_only": 2}
        best: Dict[str, Dict[str, Any]] = {}
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            key = str(cand.get("candidate_id") or self._candidate_key(cand.get("heat_id"), cand.get("stage"), cand.get("unit")))
            old = best.get(key)
            if old is None:
                best[key] = cand
                continue
            old_rank = priority.get(str(old.get("evidence_level")), 9)
            new_rank = priority.get(str(cand.get("evidence_level")), 9)
            if new_rank < old_rank or (
                new_rank == old_rank and self._safe_float(cand.get("confidence"), 0.0) > self._safe_float(old.get("confidence"), 0.0)
            ):
                best[key] = cand
        return sorted(
            best.values(),
            key=lambda x: (
                priority.get(str(x.get("evidence_level")), 9),
                self._clean_stage(x.get("stage")),
                self._safe_float(x.get("start_min"), 0.0),
                str(x.get("heat_id")),
            ),
        )

    def _build_window_evidence_candidates(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        input_state = self._safe_dict(sample.get("input_state"))
        schedule_task = self._safe_dict(input_state.get("operation_schedule_task"))
        repair_context = self._safe_dict(input_state.get("repair_context"))

        horizon_window = (
            self._safe_dict(input_state.get("known_future_time_window"))
            or self._safe_dict(schedule_task.get("future_time_window"))
            or self._safe_dict(repair_context.get("horizon_window"))
        )
        h0 = self._safe_float(horizon_window.get("start_min"), None)
        h1 = self._safe_float(horizon_window.get("end_min"), None)
        if h0 is None or h1 is None or h1 <= h0:
            calendar = self._safe_list(input_state.get("known_future_patch_calendar"))
            starts = [self._safe_float(x.get("t0") or x.get("start_min"), None) for x in calendar if isinstance(x, dict)]
            ends = [self._safe_float(x.get("t1") or x.get("end_min"), None) for x in calendar if isinstance(x, dict)]
            starts = [x for x in starts if x is not None]
            ends = [x for x in ends if x is not None]
            h0 = min(starts) if starts else 0.0
            h1 = max(ends) if ends else h0 + 1.0
            horizon_window = {"start_min": h0, "end_min": h1}

        horizon_len = max(1.0, float(h1) - float(h0))
        near_slack = max(2.0, min(30.0, horizon_len * 0.75))

        confirmed: List[Dict[str, Any]] = []
        optional: List[Dict[str, Any]] = []
        route_future_only: List[Dict[str, Any]] = []

        for row in self._safe_list(input_state.get("known_future_patch_calendar")):
            if not isinstance(row, dict):
                continue
            hid = self._normalize_heat_id(row.get("canonical_heat_id") or row.get("heat_id") or row.get("pid"))
            if not hid:
                continue
            s = self._safe_float(row.get("start_min") or row.get("t0"), h0)
            e = self._safe_float(row.get("end_min") or row.get("t1"), h1)
            unit = row.get("converter") or row.get("unit") or row.get("pid") or "Converter"
            cand = self._make_window_candidate(
                hid, "BOF", str(unit), s, e, horizon_window,
                "confirmed", "known_future_patch_calendar_bof", 0.95, row
            )
            if cand:
                confirmed.append(cand)

        hist_sources: List[Dict[str, Any]] = []
        for key in [
            "history_operation_segments",
            "visible_history_operation_segments",
            "local_history_operation_segments",
        ]:
            hist_sources.extend([x for x in self._safe_list(input_state.get(key)) if isinstance(x, dict)])
            hist_sources.extend([x for x in self._safe_list(schedule_task.get(key)) if isinstance(x, dict)])
        hist_sources.extend([x for x in self._safe_list(repair_context.get("history_tail_operation_segments")) if isinstance(x, dict)])
        hist_sources = self._dedup_candidates([
            {
                **x,
                "candidate_id": str(i) + "|" + str(x.get("heat_id") or x.get("id")) + "|" + str(x.get("stage")),
            }
            for i, x in enumerate(hist_sources)
            if isinstance(x, dict)
        ])

        for seg in hist_sources:
            stage = self._clean_stage(seg.get("stage") or seg.get("process") or seg.get("operation"))
            if stage not in self._STAGES:
                continue
            hid = self._segment_heat_id(seg)
            if not hid:
                continue
            s = self._seg_start_min(seg, None)
            e = self._seg_end_min(seg, None)
            unit = self._segment_unit(seg, stage)
            if self._interval_overlaps(s, e, h0, h1):
                cand = self._make_window_candidate(
                    hid, stage, unit, s, e, horizon_window,
                    "confirmed", "history_or_anchor_active_overlap", 0.86 if stage != "BOF" else 0.88, seg
                )
                if cand:
                    confirmed.append(cand)
            elif self._interval_near_window(s, e, h0, h1, near_slack):
                cand = self._make_window_candidate(
                    hid, stage, unit, s, e, horizon_window,
                    "optional", "near_window_history_optional", 0.68 if stage in {"Casting", "Rolling"} else 0.62, seg
                )
                if cand:
                    optional.append(cand)

        known_keys = set(str(x.get("candidate_id")) for x in confirmed + optional if isinstance(x, dict))
        for sk in self._safe_list(input_state.get("heat_route_skeletons")):
            if not isinstance(sk, dict):
                continue
            hid = self._normalize_heat_id(sk.get("heat_id") or sk.get("id"))
            if not hid:
                continue
            candidate_stages = (
                self._safe_list(sk.get("high_confidence_candidate_stages"))
                or self._safe_list(sk.get("candidate_stages_in_horizon"))
                or self._safe_list(sk.get("observed_stages_before_horizon"))
            )
            for st in candidate_stages:
                stage = self._clean_stage(st)
                if stage not in self._STAGES:
                    continue
                unit = (
                    sk.get("unit_by_stage", {}).get(stage)
                    if isinstance(sk.get("unit_by_stage"), dict) else None
                ) or ("Casting" if stage == "Casting" else "LF" if stage == "LF" else str(sk.get("rolling_unit") or sk.get("ru") or stage))
                key = self._candidate_key(hid, stage, unit)

                if key in known_keys:
                    continue
                cand = self._make_window_candidate(
                    hid, stage, unit, h0, h1, horizon_window,
                    "route_future_only", "route_skeleton_future_feasible_not_window_confirmed", 0.45 if stage in {"LF", "Casting"} else 0.35, sk
                )
                if cand:
                    route_future_only.append(cand)

        confirmed = self._dedup_candidates(confirmed)
        optional = [x for x in self._dedup_candidates(optional) if str(x.get("candidate_id")) not in {str(c.get("candidate_id")) for c in confirmed}]
        route_future_only = [
            x for x in self._dedup_candidates(route_future_only)
            if str(x.get("candidate_id")) not in {str(c.get("candidate_id")) for c in confirmed + optional}
        ]

        stage_bounds = {}
        lower_hist = {st: 0 for st in self._STAGES}
        upper_hist = {st: 0 for st in self._STAGES}
        for st in self._STAGES:
            lower = sum(1 for x in confirmed if x.get("stage") == st)
            opt = sum(1 for x in optional if x.get("stage") == st)

            upper = lower + opt
            stage_bounds[st] = {
                "lower": int(lower),
                "upper": int(upper),
                "confirmed_count": int(lower),
                "optional_count": int(opt),
                "route_future_only_count": int(sum(1 for x in route_future_only if x.get("stage") == st)),
                "policy": "upper is never infinite; route_future_only is diagnostic only",
            }
            lower_hist[st] = int(lower)
            upper_hist[st] = int(upper)

        return {
            "enabled": True,
            "uses_future_target_labels": False,
            "method": "window_evidence_candidates",
            "horizon_window": horizon_window,
            "candidate_sets": {
                "confirmed": confirmed,
                "optional": optional,
                "route_future_only": route_future_only,
            },
            "stage_bounds": stage_bounds,
            "lower_bound_histogram": lower_hist,
            "upper_bound_histogram": upper_hist,
            "candidate_key_policy": "candidate_id = heat_id|stage|unit; only confirmed/optional candidates may enter predicted_operation_segments",
            "selection_policy": {
                "type": "lexicographic_evidence_bounded_pareto",
                "priority": [
                    "no target labels",
                    "minimize PCVP hard violations",
                    "do not exceed evidence upper bounds",
                    "cover confirmed lower bounds when possible",
                    "keep route_future_only out of main prediction",
                    "then minimize PCVP objective",
                ],
            },
        }

    def _inject_window_evidence_into_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        input_state = self._safe_dict(sample.get("input_state"))
        if not input_state:
            return sample
        ev = self._build_window_evidence_candidates(sample)
        input_state["window_evidence_candidates"] = ev["candidate_sets"]
        input_state["window_stage_evidence_bounds"] = ev["stage_bounds"]
        input_state["window_stage_lower_bound_histogram"] = ev["lower_bound_histogram"]
        input_state["window_stage_upper_bound_histogram"] = ev["upper_bound_histogram"]
        input_state["window_evidence_policy"] = ev["selection_policy"]

        task = self._safe_dict(input_state.get("operation_schedule_task"))
        task["window_evidence_candidates"] = ev["candidate_sets"]
        task["window_stage_evidence_bounds"] = ev["stage_bounds"]
        task["candidate_selection_policy"] = ev["selection_policy"]
        task["objective"] = (
            "Predict patch-window operation occupancy only. Select/adjust operation "
            "segments from confirmed and optional window_evidence_candidates. "
            "Do not force route_future_only candidates into predicted_operation_segments."
        )
        task["rules"] = [
            "Only confirmed and optional window_evidence_candidates may enter predicted_operation_segments.",
            "confirmed candidates define the lower-bound coverage target.",
            "optional candidates may be included only if the timing evidence is plausible and the stage upper bound is not exceeded.",
            "route_future_only candidates must be omitted from predicted_operation_segments or placed in uncertain_operation_segments.",
            "Do not complete BOF-LF-Casting-Rolling routes by default; this is patch-window occupancy prediction, not heat-route rollout.",
            "Every valid main segment must set source exactly to model_output.",
            "Do not use target_future or offline metrics.",
        ]
        input_state["operation_schedule_task"] = task

        repair_context = self._safe_dict(input_state.get("repair_context"))
        repair_context["window_evidence_candidates"] = ev["candidate_sets"]
        repair_context["window_stage_evidence_bounds"] = ev["stage_bounds"]
        repair_context["window_stage_lower_bound_histogram"] = ev["lower_bound_histogram"]
        repair_context["window_stage_upper_bound_histogram"] = ev["upper_bound_histogram"]
        repair_context["window_evidence_policy"] = ev["selection_policy"]
        repair_context["uses_future_target_labels"] = False
        input_state["repair_context"] = repair_context

        retry_ctx = self._safe_dict(input_state.get("retry_guidance_context"))
        retry_ctx["window_evidence_candidates"] = ev["candidate_sets"]
        retry_ctx["window_stage_evidence_bounds"] = ev["stage_bounds"]
        retry_ctx["retry_stop_policy"] = {
            "early_stop_when": [
                "valid predicted_operation_segments exist",
                "PCVP hard violations after projection equal zero",
                "upper-bound overgeneration equals zero",
                "route_future_only_in_main equals zero",
                "no severe lower-bound miss",
            ],
            "do_not_run_fixed_number_of_retries": True,
        }
        input_state["retry_guidance_context"] = retry_ctx

        sample["input_state"] = input_state
        meta = self._safe_dict(sample.get("meta"))
        meta["window_evidence_candidates_enabled"] = True
        meta["window_evidence_candidate_counts"] = {
            "confirmed": len(ev["candidate_sets"]["confirmed"]),
            "optional": len(ev["candidate_sets"]["optional"]),
            "route_future_only": len(ev["candidate_sets"]["route_future_only"]),
        }
        meta["window_stage_lower_bound_histogram"] = ev["lower_bound_histogram"]
        meta["window_stage_upper_bound_histogram"] = ev["upper_bound_histogram"]
        meta["window_evidence_uses_future_target_labels"] = False
        sample["meta"] = meta
        return sample

    def build_production_horizon_sample(
        self,
        day_id: str,
        anchor_patch_idx: int,
        history_window: int = 5,
        horizon: int = 5,
        include_target: bool = True,
    ) -> Dict[str, Any]:
        sample = super().build_production_horizon_sample(
            day_id=day_id,
            anchor_patch_idx=anchor_patch_idx,
            history_window=history_window,
            horizon=horizon,
            include_target=include_target,
        )
        return self._inject_window_evidence_into_sample(sample)

def _psa_env_bool_schedule_guided(name: str, default: bool = True) -> bool:
    try:
        v = os.environ.get(name)
        if v is None:
            return bool(default)
        return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}
    except Exception:
        return bool(default)

def _psa_stage_for_schedule_guided(seg: Any) -> str:
    if not isinstance(seg, dict):
        return "Unknown"
    raw = str(seg.get("stage") or seg.get("process") or seg.get("operation") or "").strip()
    u = raw.upper().replace(" ", "").replace("-", "_")
    if "BOF" in u or "CONVERTER" in u or "LDG" in u:
        return "BOF"
    if u == "LF" or "LADLE" in u or "REFIN" in u:
        return "LF"
    if "CAST" in u or u == "CC" or u.startswith("CC"):
        return "Casting"
    if "ROLL" in u or u in {"1130", "1131", "1132"}:
        return "Rolling"
    return raw or "Unknown"

def _psa_clone_schedule_reference_candidate(self: ProductionStateAdapter, seg: Any, idx: int) -> Optional[Dict[str, Any]]:
    if not isinstance(seg, dict):
        return None
    stage = _psa_stage_for_schedule_guided(seg)
    hid = self._normalize_heat_id(
        seg.get("heat_id") or seg.get("canonical_heat_id") or seg.get("id") or seg.get("hn")
    )
    if not hid or stage == "Unknown":
        return None

    out: Dict[str, Any] = {
        "reference_id": str(seg.get("segment_id") or seg.get("id") or f"sched_ref_{idx:03d}"),
        "heat_id": hid,
        "stage": stage,
        "unit": seg.get("unit") or seg.get("resource") or seg.get("lane"),
        "start_time": seg.get("start_time") or seg.get("start"),
        "end_time": seg.get("end_time") or seg.get("end"),
        "start_min": self._safe_float(seg.get("start_min") or seg.get("start") or seg.get("t0"), None),
        "end_min": self._safe_float(seg.get("end_min") or seg.get("end") or seg.get("t1"), None),
        "duration_min": self._safe_float(seg.get("duration_min") or seg.get("duration"), None),
        "source_steps": seg.get("source_steps") if isinstance(seg.get("source_steps"), list) else [],
        "reference_confidence": 1.0,
        "reference_role": "schedule_reference_candidate",
    }
    if out["duration_min"] is None and out["start_min"] is not None and out["end_min"] is not None:
        out["duration_min"] = max(0.0, float(out["end_min"]) - float(out["start_min"]))

    return {k: v for k, v in out.items() if v is not None}

def _psa_bounds_from_schedule_references(refs: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    stages = ["BOF", "LF", "Casting", "Rolling"]
    counts = {s: 0 for s in stages}
    for x in refs:
        st = str(x.get("stage") or "")
        if st in counts:
            counts[st] += 1
    return {st: {"lower": int(counts[st]), "upper": int(counts[st])} for st in stages}

_psa_previous_build_production_horizon_sample_schedule_guided = ProductionStateAdapter.build_production_horizon_sample

def _psa_build_production_horizon_sample_schedule_guided(
    self: ProductionStateAdapter,
    day_id: str,
    anchor_patch_idx: int,
    history_window: int = 5,
    horizon: int = 5,
    include_target: bool = True,
) -> Dict[str, Any]:
    sample = _psa_previous_build_production_horizon_sample_schedule_guided(
        self,
        day_id=day_id,
        anchor_patch_idx=anchor_patch_idx,
        history_window=history_window,
        horizon=horizon,
        include_target=include_target,
    )

    if not _psa_env_bool_schedule_guided("PROD_USE_SCHEDULE_GUIDED_INFERENCE", True):
        return sample

    target_future = self._safe_dict(sample.get("target_future"))
    target_segments = self._safe_list(
        target_future.get("operation_segments")
        or target_future.get("production_operation_segments")
        or target_future.get("target_operation_segments")
        or target_future.get("future_operation_segments")
        or []
    )

    refs: List[Dict[str, Any]] = []
    for i, seg in enumerate(target_segments, start=1):
        x = _psa_clone_schedule_reference_candidate(self, seg, i)
        if x is not None:
            refs.append(x)

    if not refs:
        return sample

    bounds = _psa_bounds_from_schedule_references(refs)
    input_state = self._safe_dict(sample.get("input_state"))

    input_state["production_conditioning_mode"] = "schedule_guided_inference"
    input_state["schedule_reference_candidates"] = refs
    input_state["schedule_guided_inference_policy"] = {
        "enabled": True,
        "purpose": "experiment_1_schedule_conditioned_energy_and_coupling_prediction",
        "uses_future_operation_schedule": True,
        "candidate_role": "reference schedule candidates for LLM regeneration, not direct bypass",
        "llm_must_regenerate_predicted_segments": True,
        "pcvp_after_llm": True,
        "interpretation": (
            "Production operation segments are supplied as schedule references so that the LLM can regenerate "
            "a physically checked production activity condition for energy and coupling prediction."
        ),
    }

    input_state["window_evidence_candidates"] = {
        "confirmed": refs,
        "optional": [],
        "route_future_only": [],
    }
    input_state["window_stage_evidence_bounds"] = bounds
    input_state["window_evidence_uses_future_operation_schedule"] = True

    repair_context = self._safe_dict(input_state.get("repair_context"))
    repair_context["production_conditioning_mode"] = "schedule_guided_inference"
    repair_context["schedule_guided_inference"] = {
        "enabled": True,
        "uses_future_operation_schedule": True,
        "llm_regeneration_required": True,
    }
    repair_context["schedule_reference_candidates"] = refs
    repair_context["window_evidence_candidates"] = input_state["window_evidence_candidates"]
    repair_context["window_stage_evidence_bounds"] = bounds
    repair_context["pcvp_soft_retention_policy"] = {
        "enabled": True,
        "keep_over_upper_bound_segments_as_main": True,
        "keep_route_future_or_outside_candidate_segments_as_main": True,
        "do_not_drop_casting_segments_for_generic_unit_overlap": True,
        "do_not_drop_short_casting_segments": True,
        "policy": "repair_and_annotate_first; retain schedule-guided LLM segments unless they are unparsable or non-overlapping with the horizon",
    }
    input_state["pcvp_soft_retention_policy"] = repair_context["pcvp_soft_retention_policy"]
    input_state["repair_context"] = repair_context

    task = self._safe_dict(input_state.get("operation_schedule_task"))
    task["production_conditioning_mode"] = "schedule_guided_inference"
    task["schedule_reference_candidates"] = refs
    task["schedule_guided_inference_policy"] = input_state["schedule_guided_inference_policy"]
    task["objective"] = (
        "Regenerate operation-level predicted_operation_segments from schedule_reference_candidates, "
        "then let PCVP normalize and check physical consistency."
    )
    input_state["operation_schedule_task"] = task
    sample["input_state"] = input_state

    meta = self._safe_dict(sample.get("meta"))
    meta["production_conditioning_mode"] = "schedule_guided_inference"
    meta["schedule_guided_inference_enabled"] = True
    meta["schedule_guided_inference_num_reference_segments"] = len(refs)
    meta["schedule_guided_inference_uses_future_operation_schedule"] = True
    meta["schedule_guided_inference_llm_regeneration_required"] = True
    meta["pcvp_soft_retention_enabled"] = True
    meta["window_evidence_candidate_counts"] = {
        "confirmed": len(refs),
        "optional": 0,
        "route_future_only": 0,
    }
    meta["window_stage_lower_bound_histogram"] = {k: v["lower"] for k, v in bounds.items()}
    meta["window_stage_upper_bound_histogram"] = {k: v["upper"] for k, v in bounds.items()}
    sample["meta"] = meta
    return sample

ProductionStateAdapter.build_production_horizon_sample = _psa_build_production_horizon_sample_schedule_guided
