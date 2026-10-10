from __future__ import annotations

import json
from typing import Dict, Any, List, Optional

from c1_production_state_adapter import ProductionStateAdapter
from d1_energy_state_adapter import EnergyStateAdapter

class JointStateAssembler:
    def __init__(
        self,
        production_daily_objects_path: str = "production_daily_objects.json",
        production_optimization_results_path: Optional[str] = None,
        energy_data_path: str = "data_of_IES.xlsx",
        heat_json_path: str = "heat_segments.json",
        heat_excel_path: str = "heat_segments.xlsx",
        patch_metadata_json_path: str = "patch_metadata.json",
        patch_metadata_excel_path: str = "patch_metadata.xlsx",
        patches_dir: str = "patches",
        include_legends_in_meta: bool = False,
    ) -> None:
        self.production_adapter = ProductionStateAdapter(
            production_daily_objects_path=production_daily_objects_path,
            production_optimization_results_path=production_optimization_results_path,
            include_graph=True,
            include_legend=False,
            include_optimization_context_in_memory=False,
        )

        self.energy_adapter = EnergyStateAdapter(
            energy_data_path=energy_data_path,
            heat_json_path=heat_json_path,
            heat_excel_path=heat_excel_path,
            patch_metadata_json_path=patch_metadata_json_path,
            patch_metadata_excel_path=patch_metadata_excel_path,
            patches_dir=patches_dir,
            include_legend=False,
        )

        self.include_legends_in_meta = include_legends_in_meta

    def load(self, sample_size: Optional[int] = None) -> None:
        self.production_adapter.load()
        self.energy_adapter.load(sample_size=sample_size)

    def _json_len(self, obj: Any) -> int:
        return len(json.dumps(obj, ensure_ascii=False, default=str, separators=(",", ":")))

    def _safe_dict(self, x: Any) -> Dict[str, Any]:
        return x if isinstance(x, dict) else {}

    def _safe_list(self, x: Any) -> List[Any]:
        return x if isinstance(x, list) else []

    def _safe_bool(self, x: Any) -> bool:
        return bool(x)

    def _safe_float(self, x: Any, default: float = 0.0) -> float:
        try:
            if x is None:
                return default
            return float(x)
        except Exception:
            return default

    def _safe_int(self, x: Any, default: int = 0) -> int:
        try:
            if x is None:
                return default
            return int(x)
        except Exception:
            return default

    def _safe_str(self, x: Any, default: str = "") -> str:
        if x is None:
            return default
        return str(x).strip()

    def _round(self, x: float, n: int = 4) -> float:
        try:
            return round(float(x), n)
        except Exception:
            return 0.0

    def _get_production_legend(self) -> Optional[Dict[str, Any]]:
        if hasattr(self.production_adapter, "get_legend"):
            try:
                return self.production_adapter.get_legend()
            except Exception:
                return None
        return None

    def _get_energy_legend(self) -> Optional[Dict[str, Any]]:
        if hasattr(self.energy_adapter, "get_legend"):
            try:
                return self.energy_adapter.get_legend()
            except Exception:
                return None
        return None

    def _build_prod_energy_name_map(self, day_id: str) -> Dict[str, Any]:
        p2e: Dict[str, str] = {}
        e2p: Dict[str, str] = {}

        energy_heat_ids = self.energy_adapter.day_to_heats.get(day_id, []) \
            if hasattr(self.energy_adapter, "day_to_heats") else []

        for energy_heat_id in energy_heat_ids:
            patch_meta = self.energy_adapter.heat_id_to_patch_meta.get(energy_heat_id, {})
            prod_heat_no = self._safe_str(patch_meta.get("production_heat_no"))
            if not prod_heat_no:
                continue
            if prod_heat_no not in p2e:
                p2e[prod_heat_no] = energy_heat_id
            if energy_heat_id not in e2p:
                e2p[energy_heat_id] = prod_heat_no

        return {"p2e": p2e, "e2p": e2p}

    def build_couple_state(
        self,
        day_id: str,
        prod_full_state: Dict[str, Any],
        prod_prompt_state: Dict[str, Any],
        energy_full_state: Dict[str, Any],
        energy_prompt_state: Dict[str, Any],
        name_map: Dict[str, Any],
    ) -> Dict[str, Any]:

        prod_hs = self._safe_list(prod_full_state.get("hs"))
        prod_rs = self._safe_list(prod_full_state.get("rs"))
        prod_graph = self._safe_list(prod_full_state.get("g"))
        prod_summary = self._safe_dict(prod_full_state.get("s"))
        prod_cov = self._safe_dict(prod_full_state.get("cov"))
        prod_cf = self._safe_dict(prod_full_state.get("cf"))
        prod_mk = self._safe_dict(prod_full_state.get("mk"))

        energy_ps = self._safe_list(energy_full_state.get("ps"))
        energy_summary = self._safe_dict(energy_full_state.get("s"))
        energy_cov = self._safe_dict(energy_full_state.get("cov"))
        energy_cf = self._safe_dict(energy_full_state.get("cf"))
        energy_mk = self._safe_dict(energy_full_state.get("mk"))

        e2p = self._safe_dict(name_map.get("e2p"))
        p2e = self._safe_dict(name_map.get("p2e"))

        prod_heat_ids = [x.get("id") for x in prod_hs if isinstance(x, dict) and x.get("id")]
        prod_heat_set = set(prod_heat_ids)

        energy_ids_raw = [x.get("id") for x in energy_ps if isinstance(x, dict) and x.get("id")]
        energy_ids_mapped = [e2p.get(x) for x in energy_ids_raw if e2p.get(x)]
        energy_heat_set = set(energy_ids_mapped)

        same_day = (prod_full_state.get("d") == energy_full_state.get("d") == day_id)

        matched_heat_ids = sorted(list(prod_heat_set & energy_heat_set))
        missing_in_energy = sorted(list(prod_heat_set - energy_heat_set))
        extra_in_energy = sorted(list(energy_heat_set - prod_heat_set))
        match_score = self._round(len(matched_heat_ids) / max(len(prod_heat_set), 1)) if prod_heat_set else 0.0

        prod_heat_map = {
            x.get("id"): x for x in prod_hs
            if isinstance(x, dict) and x.get("id")
        }
        prod_ru_map = {
            x.get("u"): x for x in prod_rs
            if isinstance(x, dict) and x.get("u")
        }
        energy_patch_map = {
            e2p.get(x.get("id")): x for x in energy_ps
            if isinstance(x, dict) and x.get("id") and e2p.get(x.get("id"))
        }

        checked_pairs = 0
        weak_time_pairs = 0
        aligned_pairs = 0
        time_gap_examples = []

        for hid in matched_heat_ids:
            ph = prod_heat_map.get(hid, {})
            ep = energy_patch_map.get(hid, {})

            if not ph or not ep:
                continue

            pt0 = ph.get("tb0")
            pt1 = ph.get("tl1") or ph.get("tb1")
            et0 = self._safe_int(ep.get("t", {}).get("t0"), -1)
            et1 = self._safe_int(ep.get("t", {}).get("t1"), -1)

            checked_pairs += 1

            prod_has_time = (pt0 is not None) and (pt1 is not None)
            energy_has_time = (et0 >= 0) and (et1 >= 0)

            if prod_has_time and energy_has_time:
                aligned_pairs += 1
            else:
                weak_time_pairs += 1
                if len(time_gap_examples) < 8:
                    time_gap_examples.append({
                        "id": hid,
                        "prod_tb0": pt0,
                        "prod_t1": pt1,
                        "energy_t0": et0 if et0 >= 0 else None,
                        "energy_t1": et1 if et1 >= 0 else None,
                    })

        time_consistency = self._round(aligned_pairs / max(checked_pairs, 1)) if checked_pairs else 0.0

        overlap_active = sum(
            1 for p in energy_ps
            if self._safe_dict(p.get("pc")).get("ov", False)
        )
        risky_patches = [
            {
                "id": p.get("id"),
                "pid": e2p.get(p.get("id")),
                "sq": p.get("sq"),
                "rk": self._safe_list(p.get("rk")),
                "ov": self._safe_bool(self._safe_dict(p.get("pc")).get("ov", False)),
            }
            for p in energy_ps
            if self._safe_list(p.get("rk"))
        ]
        risky_patches = sorted(
            risky_patches,
            key=lambda x: (len(x.get("rk", [])), x.get("ov", False)),
            reverse=True,
        )[:8]

        risk_hist = self._safe_dict(energy_cov.get("risk_hist", {}))

        be_max = max([self._safe_float(x.get("be"), 0.0) for x in prod_hs] + [1.0])
        bp_max = max([self._safe_float(x.get("bp"), 0.0) for x in prod_hs] + [1.0])

        sw_weight = {"none": 0.0, "low": 0.5, "med": 1.0, "high": 1.5}
        tt_weight = {"unknown": 0.0, "low": 0.2, "med": 0.6, "high": 1.0}
        hc_weight = {"unknown": 0.0, "high": 0.0, "med": 0.4, "low": 0.8}
        wg_weight = {"none": 0.0, "low": 0.3, "med": 0.7, "high": 1.0}

        dep_weight = {"low": 0.0, "med": 0.6, "high": 1.2}
        peak_weight = {"low": 0.0, "med": 0.5, "high": 1.0}

        bridge_examples = []
        td_scores = []
        tight_scores = []

        high_switch_count = 0
        high_td_count = 0
        tight_patch_count = 0
        high_dep_count = 0

        for hid in matched_heat_ids:
            ph = prod_heat_map.get(hid, {})
            ep = energy_patch_map.get(hid, {})
            if not ph or not ep:
                continue

            pmk = self._safe_dict(ph.get("mk"))
            emk = self._safe_dict(ep.get("mk"))
            ru = prod_ru_map.get(ph.get("ru"), {})
            rumk = self._safe_dict(ru.get("mk"))

            sw = self._safe_str(pmk.get("sw"), "none")
            hc = self._safe_str(pmk.get("hc"), "unknown")
            tt = self._safe_str(pmk.get("tt"), "unknown")
            wg = self._safe_str(pmk.get("wg"), "none")
            pl = self._safe_str(rumk.get("pl"), "low")

            be_norm = self._safe_float(ph.get("be"), 0.0) / max(be_max, 1e-6)
            bp_norm = self._safe_float(ph.get("bp"), 0.0) / max(bp_max, 1e-6)

            thermal_demand_score = (
                be_norm +
                0.5 * bp_norm +
                sw_weight.get(sw, 0.0) +
                tt_weight.get(tt, 0.0) +
                hc_weight.get(hc, 0.0) +
                wg_weight.get(wg, 0.0)
            )

            hm = self._safe_dict(emk.get("hm"))
            holder_score = 0.0
            for gas in ["b", "c", "l"]:
                lv = self._safe_str(self._safe_dict(hm.get(gas)).get("lv"), "normal")
                if lv == "tight":
                    holder_score += 0.8
                elif lv == "breach":
                    holder_score += 1.5

            ed = self._safe_dict(emk.get("ed"))
            dep_lv = self._safe_str(ed.get("lv"), "low")
            dep_score = dep_weight.get(dep_lv, 0.0)

            plv = self._safe_dict(emk.get("pl"))
            peak_score = max(
                peak_weight.get(self._safe_str(plv.get("s"), "low"), 0.0),
                peak_weight.get(self._safe_str(plv.get("e"), "low"), 0.0),
            )

            energy_tight_score = holder_score + dep_score + peak_score + (0.5 if self._safe_bool(self._safe_dict(ep.get("pc")).get("ov", False)) else 0.0)

            td_scores.append(thermal_demand_score)
            tight_scores.append(energy_tight_score)

            if sw == "high":
                high_switch_count += 1
            if thermal_demand_score >= 2.2:
                high_td_count += 1
            if energy_tight_score >= 1.5:
                tight_patch_count += 1
            if dep_lv == "high":
                high_dep_count += 1

            bridge_examples.append({
                "id": hid,
                "eid": ep.get("id"),
                "sq": ep.get("sq"),
                "td": self._round(thermal_demand_score, 4),
                "sw": sw,
                "pc": pl,
                "et": self._round(energy_tight_score, 4),
                "ov": self._safe_bool(self._safe_dict(ep.get("pc")).get("ov", False)),
            })

        bridge_examples = sorted(
            bridge_examples,
            key=lambda x: (x.get("td", 0.0) + x.get("et", 0.0)),
            reverse=True,
        )[:8]

        avg_td = self._round(sum(td_scores) / max(len(td_scores), 1), 4) if td_scores else 0.0
        avg_tight = self._round(sum(tight_scores) / max(len(tight_scores), 1), 4) if tight_scores else 0.0

        if match_score < 0.90:
            regime_tag = "mapping-gap"
        elif tight_patch_count > max(1, len(matched_heat_ids) // 3):
            regime_tag = "energy-tight"
        elif high_switch_count > max(1, len(matched_heat_ids) // 4):
            regime_tag = "switch-transition"
        elif overlap_active > 0:
            regime_tag = "overlap-risk"
        else:
            regime_tag = "normal-steady"

        if avg_tight >= 2.0:
            capacity_limit_level = "high"
        elif avg_tight >= 1.0:
            capacity_limit_level = "med"
        else:
            capacity_limit_level = "low"

        prod_heat_cov = self._safe_dict(prod_cov.get("heat_fields", {}))
        energy_patch_cov = self._safe_dict(energy_cov.get("patch_fields", {}))

        hard_rs = []
        soft_rs = []

        if not same_day:
            hard_rs.append("day_id_mismatch")
        if match_score < 0.90:
            soft_rs.append("prod_energy_mapping_gap")
        if time_consistency < 0.80:
            soft_rs.append("cross_state_time_weak")
        if overlap_active > 0:
            soft_rs.append("energy_overlap_active")
        if risky_patches:
            soft_rs.append("patch_risk_active")

        feasible = len(hard_rs) == 0
        feas_level = "hard_infeasible" if hard_rs else ("soft_feasible" if soft_rs else "feasible")

        cost_proxy = self._round(
            len(missing_in_energy) * 3.0 +
            len(extra_in_energy) * 2.0 +
            overlap_active * 1.0 +
            len(risky_patches) * 1.0 +
            avg_tight,
            4
        )
        econ_score = self._round(1.0 / (1.0 + max(cost_proxy, 0.0)), 6)

        if missing_in_energy:
            focus = "mapping_consistency"
        elif time_consistency < 0.80:
            focus = "time_consistency"
        elif risky_patches:
            focus = "risk_consistency"
        else:
            focus = "state_completeness"

        bn = []
        if missing_in_energy or extra_in_energy:
            bn.append("prod_energy_alignment")
        if time_consistency < 0.80:
            bn.append("time_alignment")
        if overlap_active > 0:
            bn.append("overlap")
        if risky_patches:
            bn.append("risk_binding")

        rf = sorted(set(list(risk_hist.keys())))

        couple_state = {
            "d": day_id,
            "m": "coupling",

            "rel": {
                "same_day": same_day,
                "matched_heat_ids": matched_heat_ids[:24],
                "missing_in_energy": missing_in_energy[:24],
                "extra_in_energy": extra_in_energy[:24],
                "match_score": match_score,
                "time_consistency": time_consistency,
                "checked_pairs": checked_pairs,
                "time_gap_examples": time_gap_examples,
            },

            "repr": {
                "prod_heat_cov": prod_heat_cov,
                "energy_patch_cov": energy_patch_cov,
                "prod_nh": len(prod_heat_set),
                "energy_np": len(energy_ps),
                "prod_nru": len(prod_rs),
                "prod_ngraph": len(prod_graph),
            },

            "rsk": {
                "focus": focus,
                "overlap_active": overlap_active,
                "risk_hist": risk_hist,
                "top_risky_patches": risky_patches,
            },

            "mec": {
                "reg": regime_tag,
                "td": {
                    "avg": avg_td,
                    "high": high_td_count,
                },
                "sw": {
                    "high": high_switch_count,
                    "hist": self._safe_dict(prod_mk.get("sw", {})),
                },
                "et": {
                    "avg": avg_tight,
                    "tight": tight_patch_count,
                    "dep_high": high_dep_count,
                    "energy_hist": self._safe_dict(energy_mk.get("level_hist", {})),
                },
                "cap": {
                    "lv": capacity_limit_level,
                    "holder_tight": self._safe_int(energy_mk.get("holder_tight_patches", 0), 0),
                    "high_peak": self._safe_int(energy_mk.get("high_peak_patches", 0), 0),
                },
                "top_bridge": bridge_examples,
            },

            "s": {
                "same_day": same_day,
                "nh_prod": len(prod_heat_set),
                "nh_energy": len(energy_heat_set),
                "match_h": len(matched_heat_ids),
                "match_score": match_score,
                "xscale": time_consistency,
                "ov": overlap_active,
                "nrk": len(risky_patches),
                "npp": self._safe_int(energy_summary.get("npp", 0), 0),
                "nps": self._safe_int(energy_summary.get("nps", 0), 0),
            },
            "cf": {
                "focus": focus,
                "bn": bn,
                "rf": rf,
                "prec": self._safe_list(prod_cf.get("prec", ["BOF", "LF", "Casting", "Rolling"])),
                "med": self._safe_list(energy_cf.get("med", ["BFG", "COG", "LDG", "Steam", "Power"])),
            },
            "align": {
                "missing_in_energy": missing_in_energy[:24],
                "extra_in_energy": extra_in_energy[:24],
                "top_risky_patches": risky_patches,
                "mapped_prod_to_energy": len(p2e),
                "mapped_energy_to_prod": len(e2p),
                "time_gap_examples": time_gap_examples,
            },
            "fes": {
                "ok": feasible,
                "lv": feas_level,
                "hr": hard_rs,
                "sr": soft_rs,
            },
            "ecs": {
                "cp": cost_proxy,
                "es": econ_score,
            },
            "meta": {
                "prod_graph_size": len(prod_graph),
                "prod_gap_units": self._safe_int(prod_summary.get("ng", 0), 0),
                "prod_rolling_units": self._safe_int(prod_summary.get("nru", 0), 0),
                "prod_by_furnace": self._safe_dict(prod_summary.get("bf", {})),
            },
        }

        return couple_state

    def build_couple_prompt_state(self, couple_state: Dict[str, Any]) -> Dict[str, Any]:
        s = self._safe_dict(couple_state.get("s"))
        cf = self._safe_dict(couple_state.get("cf"))
        align = self._safe_dict(couple_state.get("align"))
        fes = self._safe_dict(couple_state.get("fes"))
        ecs = self._safe_dict(couple_state.get("ecs"))
        mec = self._safe_dict(couple_state.get("mec"))

        td = self._safe_dict(mec.get("td"))
        sw = self._safe_dict(mec.get("sw"))
        et = self._safe_dict(mec.get("et"))
        cap = self._safe_dict(mec.get("cap"))

        return {
            "d": couple_state.get("d"),
            "m": "coupling_prompt",
            "k": {
                "ms": s.get("match_score", 0.0),
                "xs": s.get("xscale", 0.0),
                "ov": s.get("ov", 0),
                "rk": s.get("nrk", 0),
                "pp": s.get("npp", 0),
                "ps": s.get("nps", 0),
                "ok": fes.get("ok", False),
                "lv": fes.get("lv", "unknown"),
                "cp": ecs.get("cp", 0.0),
                "es": ecs.get("es", 0.0),
            },
            "f": cf.get("focus"),
            "bn": self._safe_list(cf.get("bn"))[:6],
            "rf": self._safe_list(cf.get("rf"))[:8],
            "mg": self._safe_list(align.get("missing_in_energy"))[:8],
            "tp": self._safe_list(align.get("top_risky_patches"))[:4],

            "me": {
                "reg": mec.get("reg"),
                "td": td.get("avg", 0.0),
                "sw": sw.get("high", 0),
                "et": et.get("tight", 0),
                "cap": cap.get("lv", "low"),
            },
            "mb": self._safe_list(mec.get("top_bridge"))[:4],
        }

    def build_memory_state(self, day_id: str) -> Dict[str, Any]:
        prod_long_memory = self.production_adapter.build_long_memory(day_id)

        energy_full_state = self.energy_adapter.build_energy_state(day_id)
        energy_global_focus = self.energy_adapter.build_energy_global_focus(
            day_id=day_id,
            full_state=energy_full_state,
        )
        energy_batch_memory = self.energy_adapter.build_empty_rolling_memory(day_id)

        return {
            "prod_long_memory": prod_long_memory,
            "energy_long_memory": {
                "d": day_id,
                "opt": False,
                "global_focus": energy_global_focus,
                "rolling_batch_memory": energy_batch_memory,
            },
            "feedback_memory": [],
        }

    def build_joint_state(
        self,
        day_id: str,
        prod_mode: str = "production",
        energy_mode: str = "energy",
        include_full_state_in_meta: bool = False,
    ) -> Dict[str, Any]:
        prod_full_state = self.production_adapter.build_prod_state(day_id)
        energy_full_state = self.energy_adapter.build_energy_state(day_id)

        prod_prompt_state = self.production_adapter.build_prompt_state(day_id, mode=prod_mode)
        energy_prompt_state = self.energy_adapter.build_prompt_state(day_id, mode=energy_mode)

        name_map = self._build_prod_energy_name_map(day_id)

        couple_state = self.build_couple_state(
            day_id=day_id,
            prod_full_state=prod_full_state,
            prod_prompt_state=prod_prompt_state,
            energy_full_state=energy_full_state,
            energy_prompt_state=energy_prompt_state,
            name_map=name_map,
        )

        couple_prompt_state = self.build_couple_prompt_state(couple_state)
        memory_state = self.build_memory_state(day_id)

        prod_cov = self._safe_dict(prod_full_state.get("cov"))
        energy_cov = self._safe_dict(energy_full_state.get("cov"))
        couple_rel = self._safe_dict(couple_state.get("rel"))

        meta = {
            "d": day_id,
            "prod_mode": prod_mode,
            "energy_mode": energy_mode,

            "size": {
                "prod_full_chars": self._json_len(prod_full_state),
                "prod_prompt_chars": self._json_len(prod_prompt_state),
                "energy_full_chars": self._json_len(energy_full_state),
                "energy_prompt_chars": self._json_len(energy_prompt_state),
                "couple_state_chars": self._json_len(couple_state),
                "couple_prompt_chars": self._json_len(couple_prompt_state),
                "memory_state_chars": self._json_len(memory_state),
            },

            "counts": {
                "prod_heats": len(self._safe_list(prod_full_state.get("hs"))),
                "prod_rolling_units": len(self._safe_list(prod_full_state.get("rs"))),
                "energy_patches": len(self._safe_list(energy_full_state.get("ps"))),
                "energy_patch_index": len(self._safe_list(energy_full_state.get("pi"))),
                "matched_heats": self._safe_int(couple_state.get("s", {}).get("match_h", 0), 0),
            },

            "repr_check": {
                "prod_cov": prod_cov,
                "energy_cov": energy_cov,
                "same_day": couple_rel.get("same_day"),
                "match_score": couple_rel.get("match_score"),
                "time_consistency": couple_rel.get("time_consistency"),
                "checked_pairs": couple_rel.get("checked_pairs"),
            },

            "alignment": {
                "mapped_prod_to_energy": len(name_map.get("p2e", {})),
                "mapped_energy_to_prod": len(name_map.get("e2p", {})),
                "missing_in_energy": len(self._safe_list(couple_state.get("align", {}).get("missing_in_energy"))),
                "extra_in_energy": len(self._safe_list(couple_state.get("align", {}).get("extra_in_energy"))),
            },
        }

        if self.include_legends_in_meta:
            meta["legends"] = {
                "production": self._get_production_legend(),
                "energy": self._get_energy_legend(),
            }

        if include_full_state_in_meta:
            meta["full_state"] = {
                "prod_full_state": prod_full_state,
                "energy_full_state": energy_full_state,
            }

        return {
            "prod_state": prod_prompt_state,
            "energy_state": energy_prompt_state,
            "couple_state": couple_state,
            "couple_prompt_state": couple_prompt_state,
            "memory_state": memory_state,
            "meta": meta,
        }

    def build_transition_index(self, day_id: str) -> Dict[str, Any]:
        prod_ti = self.production_adapter.build_transition_index(day_id)
        energy_ti = self.energy_adapter.build_transition_index(day_id)

        return {
            "d": day_id,
            "pm": prod_ti,
            "em": energy_ti,
        }

    def audit_joint_state(self, day_id: str, joint_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if joint_state is None:
            joint_state = self.build_joint_state(day_id)

        report = {
            "d": day_id,
            "passed": True,
            "errors": [],
            "warnings": [],
            "stats": {},
            "checks": {},
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

        expected_keys = [
            "prod_state",
            "energy_state",
            "couple_state",
            "couple_prompt_state",
            "memory_state",
            "meta",
        ]
        missing_keys = [k for k in expected_keys if k not in joint_state]
        if missing_keys:
            add_error("top_keys", "joint_state is missing required top-level fields", {"missing": missing_keys})

        prod_state = self._safe_dict(joint_state.get("prod_state"))
        energy_state = self._safe_dict(joint_state.get("energy_state"))
        couple_state = self._safe_dict(joint_state.get("couple_state"))
        couple_prompt_state = self._safe_dict(joint_state.get("couple_prompt_state"))
        memory_state = self._safe_dict(joint_state.get("memory_state"))
        meta = self._safe_dict(joint_state.get("meta"))

        if couple_state.get("m") != "coupling":
            add_warning("couple_mode", "couple_state.m is not coupling", {"m": couple_state.get("m")})

        if couple_prompt_state.get("m") != "coupling_prompt":
            add_warning("couple_prompt_mode", "couple_prompt_state.m is not coupling_prompt", {"m": couple_prompt_state.get("m")})

        ds = [prod_state.get("d"), energy_state.get("d"), couple_state.get("d"), meta.get("d")]
        if any(x != day_id for x in ds if x is not None):
            add_error("day_consistency", "day_id is inconsistent across joint_state substates", {"values": ds})

        match_score = self._safe_float(couple_state.get("s", {}).get("match_score", 0.0), 0.0)
        xscale = self._safe_float(couple_state.get("s", {}).get("xscale", 0.0), 0.0)

        if match_score < 0.8:
            add_warning("alignment", "the production heat to energy patch matching rate is low", {"match_score": match_score})
        if xscale < 0.6:
            add_warning("cross_scale", "cross-scale consistency is low", {"xscale": xscale})

        report["stats"]["size"] = {
            "joint_chars": self._json_len(joint_state),
            "prod_state_chars": self._json_len(prod_state),
            "energy_state_chars": self._json_len(energy_state),
            "couple_state_chars": self._json_len(couple_state),
            "couple_prompt_state_chars": self._json_len(couple_prompt_state),
            "memory_state_chars": self._json_len(memory_state),
        }

        report["checks"]["focus"] = {
            "current_focus": couple_state.get("cf", {}).get("focus"),
            "bottlenecks": self._safe_list(couple_state.get("cf", {}).get("bn")),
            "risk_flags": self._safe_list(couple_state.get("cf", {}).get("rf")),
        }

        return report

    def debug_size_report(self, day_id: str) -> Dict[str, int]:
        joint_state = self.build_joint_state(day_id)
        report = {
            "joint_state_chars": self._json_len(joint_state),
            "prod_state_chars": self._json_len(joint_state.get("prod_state", {})),
            "energy_state_chars": self._json_len(joint_state.get("energy_state", {})),
            "couple_state_chars": self._json_len(joint_state.get("couple_state", {})),
            "couple_prompt_state_chars": self._json_len(joint_state.get("couple_prompt_state", {})),
            "memory_state_chars": self._json_len(joint_state.get("memory_state", {})),
        }

        print("\n=== JointStateAssembler Size Report ===")
        for k, v in report.items():
            print(f"{k}: {v}")
        return report

def _jha_copy(obj):
    try:
        return json.loads(json.dumps(obj, ensure_ascii=False, default=str))
    except Exception:
        if isinstance(obj, dict):
            return dict(obj)
        if isinstance(obj, list):
            return list(obj)
        return obj

def _jha_list(x):
    return x if isinstance(x, list) else []

def _jha_dict(x):
    return x if isinstance(x, dict) else {}

def _jha_str(x):
    return "" if x is None else str(x).strip()

def _jha_normalize_heat_id(x):

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

    import re
    matches = re.findall(r"[A-Za-z0-9]*LDG_\d+", s, flags=re.I)
    if matches:
        return matches[-1].upper().replace("LDG_", "LDG_")

    parts = [p for p in s.split("_") if p]
    if len(parts) >= 2 and parts[-2].upper().endswith("LDG") and parts[-1].isdigit():
        return f"{parts[-2].upper()}_{parts[-1]}"

    return s

def _jha_unique_norm_list(values):
    out = []
    seen = set()
    for v in _jha_list(values):
        nv = _jha_normalize_heat_id(v)
        if nv and nv not in seen:
            out.append(nv)
            seen.add(nv)
    return out

def _jha_patch_id_from_energy(p):
    if not isinstance(p, dict):
        return ""
    return _jha_str(
        p.get("id")
        or p.get("patch_id")
        or p.get("pid")
        or p.get("p")
        or p.get("hn")
    )

def _jha_raw_heat_no_from_energy(p):
    if not isinstance(p, dict):
        return ""
    pc = _jha_dict(p.get("pc"))
    return _jha_str(
        p.get("hn")
        or p.get("heat_id")
        or p.get("heat_no")
        or pc.get("phn")
        or pc.get("production_heat_no")
        or pc.get("ph")
        or p.get("id")
    )

def _jha_heat_no_from_energy(p):
    return _jha_normalize_heat_id(_jha_raw_heat_no_from_energy(p))

def _jha_heat_ids_from_prod(p):
    if not isinstance(p, dict):
        return []

    candidates = []

    ids = p.get("active_heat_ids")
    if isinstance(ids, list):
        candidates.extend(ids)

    active = _jha_list(p.get("active_heats"))
    for h in active:
        if isinstance(h, dict):
            candidates.append(
                h.get("id")
                or h.get("heat_id")
                or h.get("heat_no")
                or h.get("hn")
            )
        elif h is not None:
            candidates.append(h)

    for key in ["heat_ids", "affected_heat_ids", "production_active_heat_ids"]:
        if isinstance(p.get(key), list):
            candidates.extend(p.get(key))

    return _jha_unique_norm_list(candidates)

def _jha_stage_hist_from_prod(p):
    if not isinstance(p, dict):
        return {}
    if isinstance(p.get("stage_counts"), dict):
        return p.get("stage_counts")
    hist = {}
    for h in _jha_list(p.get("active_heats")):
        if not isinstance(h, dict):
            continue
        st = _jha_str(h.get("stage"))
        if st:
            hist[st] = hist.get(st, 0) + 1
    return hist

def _jha_compact_prod_patch_for_coupling(p):
    if not isinstance(p, dict):
        return {}
    return {
        "pi": p.get("pi"),
        "pid": p.get("pid"),
        "t0": p.get("t0"),
        "t1": p.get("t1"),
        "active_heat_ids": _jha_heat_ids_from_prod(p),
        "active_rolling_units": _jha_list(p.get("active_rolling_units")),
        "num_active_heats": p.get("num_active_heats", len(_jha_heat_ids_from_prod(p))),
        "stage_counts": _jha_stage_hist_from_prod(p),
    }

def _jha_compact_energy_patch_for_coupling(p, hide_future_energy=False):
    if not isinstance(p, dict):
        return {}
    pc = _jha_dict(p.get("pc"))
    t = _jha_dict(p.get("t"))
    out = {
        "id": p.get("id"),
        "hn": p.get("hn"),
        "p": p.get("p"),
        "sq": p.get("sq"),
        "t": {
            "t0": t.get("t0"),
            "t1": t.get("t1"),
            "dt": t.get("dt"),
        },
        "pc": {
            "mc": pc.get("mc"),
            "phn": pc.get("phn"),
            "ov": pc.get("ov"),
            "oc": pc.get("oc"),
            "oh": pc.get("oh", []),
            "ph": pc.get("ph"),
            "nh": pc.get("nh"),
        },
    }
    if not hide_future_energy:
        out.update({
            "g": p.get("g", {}),
            "st": p.get("st", {}),
            "dm": p.get("dm", {}),
            "ex": p.get("ex", {}),
            "rk": p.get("rk", []),
            "mk": p.get("mk", {}),
        })
    return out

def _jha_build_step_coupling_label(prod_patch, energy_patch, step):

    prod_heat_ids = set(_jha_heat_ids_from_prod(prod_patch))
    energy_heat_raw = _jha_raw_heat_no_from_energy(energy_patch)
    energy_heat_no = _jha_normalize_heat_id(energy_heat_raw)
    energy_patch_id = _jha_patch_id_from_energy(energy_patch)
    risk_flags = sorted(set(_jha_list(_jha_dict(energy_patch).get("rk"))))
    pc = _jha_dict(_jha_dict(energy_patch).get("pc"))

    overlap_heat_ids = _jha_unique_norm_list(pc.get("oh"))
    direct_aligned = bool(energy_heat_no and energy_heat_no in prod_heat_ids)
    overlap_aligned = len(set(overlap_heat_ids) & prod_heat_ids) > 0 if overlap_heat_ids else False
    aligned = bool(direct_aligned or overlap_aligned)

    return {
        "step": int(step),
        "prod_patch_id": _jha_dict(prod_patch).get("pid") or _jha_dict(prod_patch).get("patch_id"),
        "energy_patch_id": energy_patch_id,

        "energy_heat_no": energy_heat_no,
        "energy_heat_raw": energy_heat_raw,
        "canonical_energy_heat_id": energy_heat_no,
        "production_active_heat_ids": sorted(prod_heat_ids),
        "canonical_active_heat_ids": sorted(prod_heat_ids),

        "production_stage_counts": _jha_stage_hist_from_prod(prod_patch),
        "aligned": aligned,
        "is_aligned": aligned,
        "direct_heat_match": direct_aligned,
        "overlap_heat_match": overlap_aligned,
        "expected_response_type": "direct" if direct_aligned else ("indirect" if overlap_aligned else "stable_or_none"),

        "risk_flags": risk_flags,
        "has_risk": len(risk_flags) > 0,
        "overlap": bool(pc.get("ov", False)),
        "overlap_heat_ids": overlap_heat_ids,
    }

def _jha_build_coupling_target_for_horizon(production_target_future, energy_target_future):

    prod_future = _jha_list(production_target_future)
    energy_future = _jha_list(energy_target_future)
    n = min(len(prod_future), len(energy_future))

    future_pairs = []
    for i in range(n):
        p_state = _jha_dict(prod_future[i])
        e_state = _jha_dict(energy_future[i])
        label = _jha_build_step_coupling_label(p_state, e_state, i + 1)

        future_pairs.append({
            "step": i + 1,
            "prod_patch_id": label.get("prod_patch_id"),
            "patch_id": label.get("energy_patch_id"),
            "energy_patch_id": label.get("energy_patch_id"),
            "energy_heat_id": label.get("energy_heat_raw"),
            "energy_heat_no": label.get("energy_heat_no"),
            "canonical_energy_heat_id": label.get("canonical_energy_heat_id"),
            "target_active_heat_ids": label.get("production_active_heat_ids", []),
            "canonical_active_heat_ids": label.get("canonical_active_heat_ids", []),
            "is_aligned": bool(label.get("is_aligned")),
            "aligned": bool(label.get("aligned")),
            "direct_heat_match": bool(label.get("direct_heat_match")),
            "overlap_heat_match": bool(label.get("overlap_heat_match")),
            "expected_response_type": label.get("expected_response_type", "unknown"),
            "risk_flags": label.get("risk_flags", []),
            "has_risk": bool(label.get("has_risk")),
            "overlap": bool(label.get("overlap")),
            "overlap_heat_ids": label.get("overlap_heat_ids", []),
        })

    return {
        "mode": "deterministic_heat_patch_coupling_target",
        "num_future_pairs": len(future_pairs),
        "future_pairs": future_pairs,
        "schema": {
            "is_aligned": "whether canonical energy heat id is within canonical production active heat ids or overlap heat ids",
            "expected_response_type": "direct when energy heat matches active production heat; indirect when overlap heat matches; otherwise stable_or_none",
        },
    }

def _jha_pair_history(prod_history, energy_history):
    n = min(len(prod_history), len(energy_history))
    out = []
    for i in range(n):
        pp = prod_history[i]
        ep = energy_history[i]
        label = _jha_build_step_coupling_label(pp, ep, step=i - n + 1)
        out.append({
            "relative_step": i - n + 1,
            "prod": _jha_compact_prod_patch_for_coupling(pp),
            "energy": _jha_compact_energy_patch_for_coupling(ep, hide_future_energy=False),
            "coupling": label,
        })
    return out

def _jha_build_joint_horizon_sample(
    self,
    day_id,
    anchor_patch_idx,
    history_window=5,
    horizon=5,
    include_target=True,
):

    if not hasattr(self.production_adapter, "build_production_horizon_sample"):
        raise AttributeError(
            "ProductionStateAdapter lacks build_production_horizon_sample(). "
            "Please use the patch-level horizon version of production_state_adapter.py."
        )
    if not hasattr(self.energy_adapter, "build_energy_horizon_sample"):
        raise AttributeError(
            "EnergyStateAdapter lacks build_energy_horizon_sample(). "
            "Please use the patch-level horizon version of energy_state_adapter.py."
        )

    prod_sample = self.production_adapter.build_production_horizon_sample(
        day_id=day_id,
        anchor_patch_idx=anchor_patch_idx,
        history_window=history_window,
        horizon=horizon,
        include_target=include_target,
    )

    production_context = {
        "mode": "production_context_for_energy_horizon",
        "production_input_state": prod_sample.get("input_state", {}),
        "known_future_patch_calendar": _jha_dict(prod_sample.get("input_state")).get("known_future_patch_calendar", []),
    }

    energy_sample = self.energy_adapter.build_energy_horizon_sample(
        day_id=day_id,
        anchor_patch_idx=anchor_patch_idx,
        history_window=history_window,
        horizon=horizon,
        production_context=production_context,
        production_prediction={},
        include_target=include_target,
    )

    prod_input = _jha_dict(prod_sample.get("input_state"))
    energy_input = _jha_dict(energy_sample.get("input_state"))

    prod_history = _jha_list(prod_input.get("history_patches"))
    energy_history = _jha_list(energy_input.get("energy_history"))
    joint_history = _jha_pair_history(prod_history, energy_history)

    prod_future_calendar = _jha_list(prod_input.get("known_future_patch_calendar"))
    energy_future_context = _jha_list(energy_input.get("future_patch_context"))
    n_future_ctx = min(len(prod_future_calendar), len(energy_future_context))
    future_context = []
    for i in range(n_future_ctx):
        future_context.append({
            "step": i + 1,
            "production_calendar": prod_future_calendar[i],
            "energy_patch_context": energy_future_context[i],
        })

    sample = {
        "sample_id": f"{day_id}_joint_anchor_{anchor_patch_idx}_W{history_window}_K{horizon}",
        "task": "production_energy_joint_horizon_prediction_sample",
        "day_id": day_id,
        "anchor_patch_idx": int(anchor_patch_idx),
        "history_window": int(history_window),
        "horizon": int(horizon),
        "input_state": {
            "mode": "production_energy_joint_patch_history",
            "day_id": day_id,
            "anchor_patch_idx": int(anchor_patch_idx),
            "history_window": int(history_window),
            "horizon": int(horizon),
            "production_input": prod_input,
            "energy_input": energy_input,
            "joint_history": joint_history,
            "future_context": future_context,
            "schema": {
                "joint_history": "historical patch-level production-energy coupling observations",
                "future_context": "future patch calendar and heat-time context without future energy labels",
                "target_future": "hidden true future labels used only for offline evaluation",
            },
        },
        "component_samples": {
            "production": prod_sample,
            "energy": energy_sample,
        },
        "meta": {
            "num_history_pairs": len(joint_history),
            "num_future_context": len(future_context),
            "is_truncated_future": len(future_context) < horizon,
        },
    }

    if include_target:
        prod_target_dict = _jha_dict(prod_sample.get("target_future"))
        prod_future = _jha_list(prod_target_dict.get("future_patches"))
        prod_operation_segments = _jha_list(
            prod_target_dict.get("operation_segments") or
            prod_target_dict.get("production_operation_segments") or
            prod_target_dict.get("patch_operation_segments")
        )
        prod_patch_operation_segments = _jha_list(prod_target_dict.get("patch_operation_segments"))
        energy_future = _jha_list(_jha_dict(energy_sample.get("target_future")).get("energy_future"))
        n = min(len(prod_future), len(energy_future))
        future_steps = []
        for i in range(n):
            future_steps.append({
                "step": i + 1,
                "prod": _jha_compact_prod_patch_for_coupling(prod_future[i]),
                "energy": _jha_compact_energy_patch_for_coupling(energy_future[i], hide_future_energy=False),
                "coupling": _jha_build_step_coupling_label(prod_future[i], energy_future[i], i + 1),
            })

        coupling_target = _jha_build_coupling_target_for_horizon(
            production_target_future=prod_future,
            energy_target_future=energy_future,
        )

        sample["coupling_target"] = coupling_target
        sample["target_future"] = {
            "mode": "production_energy_joint_patch_future_label",
            "day_id": day_id,
            "anchor_patch_idx": int(anchor_patch_idx),
            "horizon": int(horizon),
            "production_future": prod_future,
            "production_operation_segments": prod_operation_segments,
            "production_patch_operation_segments": prod_patch_operation_segments,
            "energy_future": energy_future,
            "future_steps": future_steps,
            "coupling_target": coupling_target,
        }

    return sample

def _jha_build_joint_horizon_dataset(
    self,
    day_id,
    history_window=5,
    horizon=5,
    stride=1,
    max_samples=None,
    include_target=True,
):

    if not hasattr(self.energy_adapter, "build_energy_patch_sequence"):
        raise AttributeError("EnergyStateAdapter lacks build_energy_patch_sequence().")
    seq = self.energy_adapter.build_energy_patch_sequence(day_id)
    n = len(seq)
    start = int(history_window) - 1
    end = n - int(horizon) - 1
    samples = []
    if end < start:
        return samples

    anchor = start
    while anchor <= end:
        samples.append(
            self.build_joint_horizon_sample(
                day_id=day_id,
                anchor_patch_idx=anchor,
                history_window=history_window,
                horizon=horizon,
                include_target=include_target,
            )
        )
        if max_samples is not None and len(samples) >= int(max_samples):
            break
        anchor += max(1, int(stride))
    return samples

def _jha_build_joint_horizon_prompt_state(
    self,
    day_id,
    anchor_patch_idx,
    history_window=5,
    horizon=5,
):
    sample = self.build_joint_horizon_sample(
        day_id=day_id,
        anchor_patch_idx=anchor_patch_idx,
        history_window=history_window,
        horizon=horizon,
        include_target=False,
    )
    return sample["input_state"]

def _jha_debug_joint_horizon_sample_report(
    self,
    day_id,
    anchor_patch_idx,
    history_window=5,
    horizon=5,
):
    sample = self.build_joint_horizon_sample(
        day_id=day_id,
        anchor_patch_idx=anchor_patch_idx,
        history_window=history_window,
        horizon=horizon,
        include_target=True,
    )
    report = {
        "sample_id": sample.get("sample_id"),
        "day_id": day_id,
        "anchor_patch_idx": anchor_patch_idx,
        "history_window": history_window,
        "horizon": horizon,
        "input_chars": self._json_len(sample.get("input_state", {})),
        "target_chars": self._json_len(sample.get("target_future", {})),
        "num_history_pairs": len(_jha_list(_jha_dict(sample.get("input_state")).get("joint_history"))),
        "num_future_steps": len(_jha_list(_jha_dict(sample.get("target_future")).get("future_steps"))),
        "num_coupling_target_pairs": len(_jha_list(_jha_dict(sample.get("coupling_target")).get("future_pairs"))),
    }
    print("\n=== Joint Horizon Sample Report ===")
    for k, v in report.items():
        print(f"{k}: {v}")
    return report

JointStateAssembler.build_joint_horizon_sample = _jha_build_joint_horizon_sample
JointStateAssembler.build_joint_horizon_dataset = _jha_build_joint_horizon_dataset
JointStateAssembler.build_joint_horizon_prompt_state = _jha_build_joint_horizon_prompt_state
JointStateAssembler.debug_joint_horizon_sample_report = _jha_debug_joint_horizon_sample_report

def _jha_compact_operation_schedule_context(input_state):

    input_state = _jha_dict(input_state)

    task = _jha_dict(input_state.get("operation_schedule_task"))
    hist_segments = (
        _jha_list(input_state.get("visible_history_operation_segments"))
        or _jha_list(input_state.get("history_operation_segments"))
        or _jha_list(task.get("visible_history_operation_segments"))
        or _jha_list(task.get("history_operation_segments"))
        or _jha_list(task.get("local_history_operation_segments"))
    )
    future_calendar = (
        _jha_list(input_state.get("known_future_patch_calendar"))
        or _jha_list(task.get("future_calendar"))
    )

    def _seg_brief(seg, max_idx):
        if not isinstance(seg, dict):
            return {}
        return {
            "idx": max_idx,
            "heat_id": _jha_normalize_heat_id(
                seg.get("heat_id") or seg.get("id") or seg.get("canonical_heat_id") or seg.get("hn")
            ),
            "stage": seg.get("stage") or seg.get("process_stage") or seg.get("operation"),
            "unit": seg.get("unit") or seg.get("resource") or seg.get("lane") or seg.get("ru"),
            "start_min": seg.get("start_min"),
            "end_min": seg.get("end_min"),
            "duration_min": seg.get("duration_min"),
        }

    hist_brief = []
    for i, seg in enumerate(hist_segments[-24:], start=1):
        b = _seg_brief(seg, i)
        if b:
            hist_brief.append(b)

    future_heat_ids = []
    for p in future_calendar:
        if not isinstance(p, dict):
            continue
        hid = _jha_normalize_heat_id(
            p.get("canonical_heat_id") or p.get("heat_id") or p.get("hn") or p.get("pid")
        )
        if hid and hid not in future_heat_ids:
            future_heat_ids.append(hid)

    stage_order = (
        task.get("required_process_order")
        or task.get("process_order")
        or ["BOF", "LF", "Casting", "Rolling"]
    )

    return {
        "available": bool(task or hist_segments or future_calendar),
        "objective": task.get("objective", "predict future operation-level production schedule segments for Gantt comparison"),
        "required_process_order": stage_order,
        "candidate_future_heat_ids": task.get("candidate_future_heat_ids") or future_heat_ids,
        "future_time_window": task.get("future_time_window", {}),
        "duration_priors": task.get("duration_priors", {}),
        "carryover_active_heats_at_anchor": task.get("carryover_active_heats_at_anchor", []),
        "num_history_operation_segments": len(hist_segments),
        "history_operation_segment_examples": hist_brief,
        "rules": task.get("rules", [
            "Use only history and known future patch calendar in model input.",
            "Do not expose target future operation_segments in prompt.",
            "Operation-level target labels are used only for offline evaluation."
        ]),
    }

def _jha_build_joint_horizon_sample_schedule_v1(
    self,
    day_id,
    anchor_patch_idx,
    history_window=5,
    horizon=5,
    include_target=True,
):

    if not hasattr(self.production_adapter, "build_production_horizon_sample"):
        raise AttributeError(
            "ProductionStateAdapter lacks build_production_horizon_sample(). "
            "Please use the Gantt-enabled production_state_adapter.py."
        )
    if not hasattr(self.energy_adapter, "build_energy_horizon_sample"):
        raise AttributeError(
            "EnergyStateAdapter lacks build_energy_horizon_sample(). "
            "Please use the patch-level horizon version of energy_state_adapter.py."
        )

    prod_sample = self.production_adapter.build_production_horizon_sample(
        day_id=day_id,
        anchor_patch_idx=anchor_patch_idx,
        history_window=history_window,
        horizon=horizon,
        include_target=include_target,
    )

    prod_input = _jha_dict(prod_sample.get("input_state"))
    operation_schedule_context = _jha_compact_operation_schedule_context(prod_input)

    production_context = {
        "mode": "production_context_for_energy_horizon",
        "production_input_state": prod_input,
        "known_future_patch_calendar": _jha_list(prod_input.get("known_future_patch_calendar")),
        "operation_schedule_context": operation_schedule_context,
    }

    energy_sample = self.energy_adapter.build_energy_horizon_sample(
        day_id=day_id,
        anchor_patch_idx=anchor_patch_idx,
        history_window=history_window,
        horizon=horizon,
        production_context=production_context,
        production_prediction={},
        include_target=include_target,
    )

    energy_input = _jha_dict(energy_sample.get("input_state"))

    prod_history = _jha_list(prod_input.get("history_patches"))
    energy_history = _jha_list(energy_input.get("energy_history"))
    joint_history = _jha_pair_history(prod_history, energy_history)

    prod_future_calendar = _jha_list(prod_input.get("known_future_patch_calendar"))
    energy_future_context = _jha_list(energy_input.get("future_patch_context"))
    n_future_ctx = min(len(prod_future_calendar), len(energy_future_context))
    future_context = []
    for i in range(n_future_ctx):
        future_context.append({
            "step": i + 1,
            "production_calendar": prod_future_calendar[i],
            "energy_patch_context": energy_future_context[i],
        })

    sample = {
        "sample_id": f"{day_id}_joint_anchor_{anchor_patch_idx}_W{history_window}_K{horizon}",
        "task": "production_energy_joint_horizon_prediction_sample",
        "day_id": day_id,
        "anchor_patch_idx": int(anchor_patch_idx),
        "history_window": int(history_window),
        "horizon": int(horizon),
        "input_state": {
            "mode": "production_energy_joint_patch_history",
            "day_id": day_id,
            "anchor_patch_idx": int(anchor_patch_idx),
            "history_window": int(history_window),
            "horizon": int(horizon),
            "production_input": prod_input,
            "energy_input": energy_input,
            "joint_history": joint_history,
            "future_context": future_context,
            "production_operation_schedule_context": operation_schedule_context,
            "schema": {
                "joint_history": "historical patch-level production-energy coupling observations",
                "future_context": "future patch calendar and heat-time context without future energy labels",
                "production_operation_schedule_context": "model-visible compact schedule context built from history and future calendar only",
                "target_future": "hidden true future labels used only for offline evaluation",
            },
        },
        "component_samples": {
            "production": prod_sample,
            "energy": energy_sample,
        },
        "meta": {
            "num_history_pairs": len(joint_history),
            "num_future_context": len(future_context),
            "is_truncated_future": len(future_context) < horizon,
            "operation_schedule_context_available": bool(operation_schedule_context.get("available")),
            "operation_schedule_context_chars": len(json.dumps(operation_schedule_context, ensure_ascii=False, default=str)),
        },
    }

    if include_target:
        prod_target_dict = _jha_dict(prod_sample.get("target_future"))
        prod_future = _jha_list(prod_target_dict.get("future_patches"))

        prod_operation_segments = _jha_list(
            prod_target_dict.get("production_operation_segments")
            or prod_target_dict.get("operation_segments")
        )
        prod_patch_operation_segments = _jha_list(
            prod_target_dict.get("production_patch_operation_segments")
            or prod_target_dict.get("patch_operation_segments")
        )

        energy_future = _jha_list(_jha_dict(energy_sample.get("target_future")).get("energy_future"))
        n = min(len(prod_future), len(energy_future))
        future_steps = []
        for i in range(n):
            future_steps.append({
                "step": i + 1,
                "prod": _jha_compact_prod_patch_for_coupling(prod_future[i]),
                "energy": _jha_compact_energy_patch_for_coupling(energy_future[i], hide_future_energy=False),
                "coupling": _jha_build_step_coupling_label(prod_future[i], energy_future[i], i + 1),
            })

        coupling_target = _jha_build_coupling_target_for_horizon(
            production_target_future=prod_future,
            energy_target_future=energy_future,
        )

        sample["coupling_target"] = coupling_target
        sample["target_future"] = {
            "mode": "production_energy_joint_patch_future_label",
            "day_id": day_id,
            "anchor_patch_idx": int(anchor_patch_idx),
            "horizon": int(horizon),
            "production_future": prod_future,
            "production_operation_segments": prod_operation_segments,
            "production_patch_operation_segments": prod_patch_operation_segments,
            "energy_future": energy_future,
            "future_steps": future_steps,
            "coupling_target": coupling_target,
            "operation_segment_schema": {
                "production_operation_segments": "route-level true operation segments clipped to the future horizon window; used only for offline Gantt evaluation",
                "production_patch_operation_segments": "patch-derived true segments; useful for patch-level diagnostics",
            },
        }

        sample["meta"]["num_target_production_operation_segments"] = len(prod_operation_segments)
        sample["meta"]["num_target_production_patch_operation_segments"] = len(prod_patch_operation_segments)

    return sample

JointStateAssembler.build_joint_horizon_sample = _jha_build_joint_horizon_sample_schedule_v1

def _jha_time_to_min_v2(x):
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
        import re

        m = re.search(r"(\d{1,3}):(\d{2})(?::(\d{2}))?", s)
        if m:
            hh = int(m.group(1))
            mm = int(m.group(2))
            ss = int(m.group(3) or 0)
            return hh * 60.0 + mm + ss / 60.0
    except Exception:
        return None
    return None

def _jha_canonical_stage_v2(x):
    s = _jha_str(x)
    u = s.upper().replace(" ", "").replace("_", "")
    if "BOF" in u or "CONVERTER" in u:
        return "BOF"
    if u == "LF" or "LADLE" in u:
        return "LF"
    if "CAST" in u or u == "CC" or "CONTINUOUSCASTING" in u:
        return "Casting"
    if "ROLL" in u:
        return "Rolling"
    return "Other"

def _jha_brief_operation_segment_v2(seg, idx):
    if not isinstance(seg, dict):
        return {}

    heat_id = _jha_normalize_heat_id(
        seg.get("heat_id")
        or seg.get("id")
        or seg.get("canonical_heat_id")
        or seg.get("hn")
        or seg.get("Heat_No")
    )
    stage = _jha_canonical_stage_v2(
        seg.get("stage") or seg.get("process_stage") or seg.get("operation") or seg.get("process")
    )
    start_min = seg.get("start_min")
    end_min = seg.get("end_min")
    if start_min is None:
        start_min = _jha_time_to_min_v2(seg.get("start_time") or seg.get("start") or seg.get("t0"))
    if end_min is None:
        end_min = _jha_time_to_min_v2(seg.get("end_time") or seg.get("end") or seg.get("t1"))

    out = {
        "idx": idx,
        "heat_id": heat_id,
        "stage": stage,
        "unit": seg.get("unit") or seg.get("resource") or seg.get("lane") or seg.get("ru"),
        "start_min": start_min,
        "end_min": end_min,
        "duration_min": seg.get("duration_min"),
    }
    if out["duration_min"] is None and start_min is not None and end_min is not None:
        try:
            out["duration_min"] = max(0.0, float(end_min) - float(start_min))
        except Exception:
            pass
    return {k: v for k, v in out.items() if v is not None and v != ""}

def _jha_patch_calendar_brief_v2(p, idx):
    if not isinstance(p, dict):
        return {}

    heat_id = _jha_normalize_heat_id(
        p.get("canonical_heat_id")
        or p.get("heat_id")
        or p.get("hn")
        or p.get("pid")
        or p.get("id")
    )
    return {
        "step": int(p.get("step") or idx),
        "pi": p.get("pi"),
        "pid": p.get("pid") or p.get("id") or p.get("patch_id"),
        "canonical_heat_id": heat_id,
        "converter": p.get("converter"),
        "t0": p.get("t0") or p.get("start_time") or p.get("start"),
        "t1": p.get("t1") or p.get("end_time") or p.get("end"),
    }

def _jha_operation_segment_collection_summary_v2(segments):
    segs = [x for x in _jha_list(segments) if isinstance(x, dict)]
    heat_ids = []
    stage_counts = {}
    unit_counts = {}
    starts = []
    ends = []

    for seg in segs:
        hid = _jha_normalize_heat_id(
            seg.get("heat_id") or seg.get("id") or seg.get("canonical_heat_id") or seg.get("hn")
        )
        if hid and hid not in heat_ids:
            heat_ids.append(hid)

        stage = _jha_canonical_stage_v2(seg.get("stage") or seg.get("process_stage") or seg.get("operation"))
        stage_counts[stage] = stage_counts.get(stage, 0) + 1

        unit = _jha_str(seg.get("unit") or seg.get("resource") or seg.get("lane") or seg.get("ru"))
        if unit:
            unit_counts[unit] = unit_counts.get(unit, 0) + 1

        s = seg.get("start_min")
        e = seg.get("end_min")
        if s is None:
            s = _jha_time_to_min_v2(seg.get("start_time") or seg.get("start") or seg.get("t0"))
        if e is None:
            e = _jha_time_to_min_v2(seg.get("end_time") or seg.get("end") or seg.get("t1"))
        try:
            if s is not None:
                starts.append(float(s))
            if e is not None:
                ends.append(float(e))
        except Exception:
            pass

    return {
        "num_segments": len(segs),
        "heat_ids": heat_ids[:48],
        "stage_counts": stage_counts,
        "unit_counts": unit_counts,
        "time_span": {
            "start_min": min(starts) if starts else None,
            "end_min": max(ends) if ends else None,
        },
    }

def _jha_compact_operation_schedule_context_v2(input_state):

    input_state = _jha_dict(input_state)

    task = _jha_dict(input_state.get("operation_schedule_task"))
    hist_segments = (
        _jha_list(input_state.get("visible_history_operation_segments"))
        or _jha_list(input_state.get("history_operation_segments"))
        or _jha_list(task.get("visible_history_operation_segments"))
        or _jha_list(task.get("history_operation_segments"))
        or _jha_list(task.get("local_history_operation_segments"))
    )
    future_calendar = (
        _jha_list(input_state.get("known_future_patch_calendar"))
        or _jha_list(task.get("future_calendar"))
        or _jha_list(task.get("known_future_patch_calendar"))
    )

    hist_brief = []

    for i, seg in enumerate(hist_segments[-32:], start=1):
        b = _jha_brief_operation_segment_v2(seg, i)
        if b:
            hist_brief.append(b)

    future_calendar_brief = []
    future_heat_ids = []
    for i, p in enumerate(future_calendar, start=1):
        b = _jha_patch_calendar_brief_v2(p, i)
        if b:
            future_calendar_brief.append(b)
        hid = b.get("canonical_heat_id")
        if hid and hid not in future_heat_ids:
            future_heat_ids.append(hid)

    candidate_future_heat_ids = (
        task.get("candidate_future_heat_ids")
        or task.get("future_heat_ids")
        or future_heat_ids
    )

    stage_order = (
        task.get("required_process_order")
        or task.get("process_order")
        or ["BOF", "LF", "Casting", "Rolling"]
    )

    return {
        "available": bool(task or hist_segments or future_calendar),
        "objective": task.get(
            "objective",
            "predict route-level future operation segments for production Gantt comparison",
        ),
        "required_process_order": stage_order,
        "candidate_future_heat_ids": candidate_future_heat_ids,
        "future_time_window": task.get("future_time_window", {}),
        "duration_priors": task.get("duration_priors", {}),
        "carryover_active_heats_at_anchor": task.get("carryover_active_heats_at_anchor", []),
        "known_future_patch_calendar": future_calendar_brief,
        "num_known_future_calendar_items": len(future_calendar),
        "num_history_operation_segments": len(hist_segments),
        "history_operation_segment_examples": hist_brief,
        "prediction_output_contract": {
            "primary_key": "predicted_operation_segments",
            "valid_prediction_source": "model_output",
            "required_segment_fields": [
                "heat_id", "stage", "unit",
                "start_time", "end_time",
                "start_min", "end_min",
                "duration_min", "source", "confidence",
            ],
            "valid_stages": ["BOF", "LF", "Casting", "Rolling"],
            "rule": "Only segments directly predicted by the production model may be marked source=model_output.",
            "fallback_policy": "calendar-derived or patch-derived segments must not be used as valid model predictions.",
        },
        "rules": task.get("rules", []) + [
            "Use only history and known future patch calendar in model input.",
            "Do not expose target future production_operation_segments in prompt.",
            "Do not generate only patch labels; route-level operation segments are required from the production model.",
            "Target operation segments are used only by offline evaluation and plotting code.",
        ],
    }

def _jha_build_joint_horizon_sample_schedule_v2(
    self,
    day_id,
    anchor_patch_idx,
    history_window=5,
    horizon=5,
    include_target=True,
):

    if not hasattr(self.production_adapter, "build_production_horizon_sample"):
        raise AttributeError(
            "ProductionStateAdapter lacks build_production_horizon_sample(). "
            "Please use the Gantt-enabled production_state_adapter.py."
        )
    if not hasattr(self.energy_adapter, "build_energy_horizon_sample"):
        raise AttributeError(
            "EnergyStateAdapter lacks build_energy_horizon_sample(). "
            "Please use the patch-level horizon version of energy_state_adapter.py."
        )

    prod_sample = self.production_adapter.build_production_horizon_sample(
        day_id=day_id,
        anchor_patch_idx=anchor_patch_idx,
        history_window=history_window,
        horizon=horizon,
        include_target=include_target,
    )

    prod_input = _jha_dict(prod_sample.get("input_state"))
    operation_schedule_context = _jha_compact_operation_schedule_context_v2(prod_input)

    production_context = {
        "mode": "production_context_for_energy_horizon",
        "production_input_state": prod_input,
        "known_future_patch_calendar": _jha_list(prod_input.get("known_future_patch_calendar")),
        "operation_schedule_context": operation_schedule_context,
        "leakage_guard": {
            "contains_target_future_operation_segments": False,
            "target_future_is_hidden": True,
        },
    }

    energy_sample = self.energy_adapter.build_energy_horizon_sample(
        day_id=day_id,
        anchor_patch_idx=anchor_patch_idx,
        history_window=history_window,
        horizon=horizon,
        production_context=production_context,
        production_prediction={},
        include_target=include_target,
    )

    energy_input = _jha_dict(energy_sample.get("input_state"))

    prod_history = _jha_list(prod_input.get("history_patches"))
    energy_history = _jha_list(energy_input.get("energy_history"))
    joint_history = _jha_pair_history(prod_history, energy_history)

    prod_future_calendar = _jha_list(prod_input.get("known_future_patch_calendar"))
    energy_future_context = _jha_list(energy_input.get("future_patch_context"))
    n_future_ctx = min(len(prod_future_calendar), len(energy_future_context))
    future_context = []
    for i in range(n_future_ctx):
        future_context.append({
            "step": i + 1,
            "production_calendar": _jha_patch_calendar_brief_v2(prod_future_calendar[i], i + 1),
            "energy_patch_context": energy_future_context[i],
        })

    sample = {
        "sample_id": f"{day_id}_joint_anchor_{anchor_patch_idx}_W{history_window}_K{horizon}",
        "task": "production_energy_joint_horizon_prediction_sample",
        "day_id": day_id,
        "anchor_patch_idx": int(anchor_patch_idx),
        "history_window": int(history_window),
        "horizon": int(horizon),
        "input_state": {
            "mode": "production_energy_joint_patch_history",
            "day_id": day_id,
            "anchor_patch_idx": int(anchor_patch_idx),
            "history_window": int(history_window),
            "horizon": int(horizon),
            "production_input": prod_input,
            "energy_input": energy_input,
            "joint_history": joint_history,
            "future_context": future_context,
            "production_operation_schedule_context": operation_schedule_context,
            "schema": {
                "joint_history": "historical patch-level production-energy coupling observations",
                "future_context": "future patch calendar and heat-time context without future energy labels",
                "production_operation_schedule_context": "model-visible compact schedule context from history/calendar only",
                "target_future": "hidden true future labels used only for offline evaluation",
            },
            "leakage_guard": {
                "target_future_exposed_to_model": False,
                "target_operation_segments_exposed_to_model": False,
            },
        },
        "component_samples": {
            "production": prod_sample,
            "energy": energy_sample,
        },
        "meta": {
            "num_history_pairs": len(joint_history),
            "num_future_context": len(future_context),
            "is_truncated_future": len(future_context) < int(horizon),
            "operation_schedule_context_available": bool(operation_schedule_context.get("available")),
            "operation_schedule_context_chars": len(json.dumps(operation_schedule_context, ensure_ascii=False, default=str)),
            "gantt_policy": {
                "model_valid_segment_source": "model_output",
                "target_route_segments_key": "target_future.production_operation_segments",
                "target_patch_segments_key": "target_future.production_patch_operation_segments",
                "derived_or_fallback_segments_are_diagnostics_only": True,
            },
        },
    }

    if include_target:
        prod_target_dict = _jha_dict(prod_sample.get("target_future"))
        prod_future = _jha_list(prod_target_dict.get("future_patches"))

        prod_operation_segments = _jha_list(
            prod_target_dict.get("production_operation_segments")
            or prod_target_dict.get("operation_segments")
        )

        prod_patch_operation_segments = _jha_list(
            prod_target_dict.get("production_patch_operation_segments")
            or prod_target_dict.get("patch_operation_segments")
        )

        energy_future = _jha_list(_jha_dict(energy_sample.get("target_future")).get("energy_future"))
        n = min(len(prod_future), len(energy_future))
        future_steps = []
        for i in range(n):
            future_steps.append({
                "step": i + 1,
                "prod": _jha_compact_prod_patch_for_coupling(prod_future[i]),
                "energy": _jha_compact_energy_patch_for_coupling(energy_future[i], hide_future_energy=False),
                "coupling": _jha_build_step_coupling_label(prod_future[i], energy_future[i], i + 1),
            })

        coupling_target = _jha_build_coupling_target_for_horizon(
            production_target_future=prod_future,
            energy_target_future=energy_future,
        )

        route_summary = _jha_operation_segment_collection_summary_v2(prod_operation_segments)
        patch_summary = _jha_operation_segment_collection_summary_v2(prod_patch_operation_segments)

        sample["coupling_target"] = coupling_target
        sample["target_future"] = {
            "mode": "production_energy_joint_patch_future_label",
            "day_id": day_id,
            "anchor_patch_idx": int(anchor_patch_idx),
            "horizon": int(horizon),
            "production_future": prod_future,
            "production_operation_segments": prod_operation_segments,
            "production_patch_operation_segments": prod_patch_operation_segments,
            "energy_future": energy_future,
            "future_steps": future_steps,
            "coupling_target": coupling_target,
            "operation_segment_schema": {
                "production_operation_segments": "route-level true operation segments clipped to the future horizon window; primary target for Gantt comparison",
                "production_patch_operation_segments": "patch-derived true segments; diagnostic only, not the primary Gantt target",
                "model_prediction_key": "production_prediction.predicted_operation_segments",
                "valid_prediction_source": "model_output",
            },
            "operation_target_summary": {
                "route_level": route_summary,
                "patch_level_diagnostic": patch_summary,
            },
        }

        sample["meta"]["num_target_production_operation_segments"] = len(prod_operation_segments)
        sample["meta"]["num_target_production_patch_operation_segments"] = len(prod_patch_operation_segments)
        sample["meta"]["operation_target_summary"] = {
            "route_level_num_segments": route_summary.get("num_segments", 0),
            "patch_level_num_segments": patch_summary.get("num_segments", 0),
            "route_stage_counts": route_summary.get("stage_counts", {}),
            "route_time_span": route_summary.get("time_span", {}),
        }
        if not prod_operation_segments and prod_patch_operation_segments:
            sample["meta"]["operation_target_warning"] = (
                "route-level production_operation_segments are empty; only patch-derived diagnostic target segments exist"
            )

    return sample

JointStateAssembler.build_joint_horizon_sample = _jha_build_joint_horizon_sample_schedule_v2

if __name__ == "__main__":
    assembler = JointStateAssembler(
        production_daily_objects_path="production_daily_objects.json",
        production_optimization_results_path=None,
        energy_data_path="data_of_IES.xlsx",
        heat_json_path="heat_segments.json",
        heat_excel_path="heat_segments.xlsx",
        patch_metadata_json_path="patch_metadata.json",
        patch_metadata_excel_path="patch_metadata.xlsx",
        patches_dir="patches",
        include_legends_in_meta=False,
    )

    assembler.load(sample_size=None)

    example_day_id = list(assembler.production_adapter.day_to_obj.keys())[0]

    joint_state = assembler.build_joint_state(
        day_id=example_day_id,
        prod_mode="production",
        energy_mode="energy",
        include_full_state_in_meta=False,
    )

    transition_index = assembler.build_transition_index(example_day_id)

    print("example_day_id:", example_day_id)
    print("joint_state keys:", list(joint_state.keys()))
    print("prod_state keys:", list(joint_state["prod_state"].keys()))
    print("energy_state keys:", list(joint_state["energy_state"].keys()))
    print("couple_state keys:", list(joint_state["couple_state"].keys()))
    print("couple_prompt_state keys:", list(joint_state["couple_prompt_state"].keys()))
    print("memory_state keys:", list(joint_state["memory_state"].keys()))
    print("transition_index keys:", list(transition_index.keys()))

    assembler.debug_size_report(example_day_id)

    audit = assembler.audit_joint_state(example_day_id)
    print("passed:", audit["passed"])
    print("num_errors:", len(audit["errors"]))
    print("num_warnings:", len(audit["warnings"]))

    print("\n=== couple_prompt_state ===")
    print(json.dumps(joint_state["couple_prompt_state"], ensure_ascii=False, indent=2))

def _jha_safe_json_chars_gof(obj):
    try:
        return len(json.dumps(obj, ensure_ascii=False, default=str, separators=(",", ":")))
    except Exception:
        return 0

def _jha_time_to_min_gof(x):
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
        import re
        m = re.search(r"(\d{1,3}):(\d{2})(?::(\d{2}))?", s)
        if m:
            return int(m.group(1)) * 60.0 + int(m.group(2)) + int(m.group(3) or 0) / 60.0
    except Exception:
        return None
    return None

def _jha_stage_gof(x):
    s = _jha_str(x)
    u = s.upper().replace(" ", "").replace("_", "")
    if "BOF" in u or "CONVERTER" in u:
        return "BOF"
    if u == "LF" or "LADLE" in u:
        return "LF"
    if "CAST" in u or u == "CC" or "CONTINUOUSCASTING" in u:
        return "Casting"
    if "ROLL" in u:
        return "Rolling"
    return "Other"

def _jha_segment_brief_gof(seg, idx):
    if not isinstance(seg, dict):
        return {}
    start_min = seg.get("start_min")
    end_min = seg.get("end_min")
    if start_min is None:
        start_min = _jha_time_to_min_gof(seg.get("start_time") or seg.get("start") or seg.get("t0"))
    if end_min is None:
        end_min = _jha_time_to_min_gof(seg.get("end_time") or seg.get("end") or seg.get("t1"))
    out = {
        "idx": idx,
        "heat_id": _jha_normalize_heat_id(seg.get("heat_id") or seg.get("id") or seg.get("canonical_heat_id") or seg.get("hn") or seg.get("Heat_No")),
        "stage": _jha_stage_gof(seg.get("stage") or seg.get("process_stage") or seg.get("operation") or seg.get("process")),
        "unit": seg.get("unit") or seg.get("resource") or seg.get("lane") or seg.get("ru"),
        "start_min": start_min,
        "end_min": end_min,
        "duration_min": seg.get("duration_min"),
    }
    if out.get("duration_min") is None and start_min is not None and end_min is not None:
        try:
            out["duration_min"] = max(0.0, float(end_min) - float(start_min))
        except Exception:
            pass
    return {k: v for k, v in out.items() if v is not None and v != ""}

def _jha_patch_calendar_brief_gof(p, idx):
    if not isinstance(p, dict):
        return {}
    return {
        "step": int(p.get("step") or idx),
        "pi": p.get("pi"),
        "pid": p.get("pid") or p.get("id") or p.get("patch_id"),
        "canonical_heat_id": _jha_normalize_heat_id(p.get("canonical_heat_id") or p.get("heat_id") or p.get("hn") or p.get("pid") or p.get("id")),
        "converter": p.get("converter"),
        "t0": p.get("t0") or p.get("start_time") or p.get("start"),
        "t1": p.get("t1") or p.get("end_time") or p.get("end"),
        "t0_min": p.get("t0_min") if p.get("t0_min") is not None else _jha_time_to_min_gof(p.get("t0") or p.get("start_time") or p.get("start")),
        "t1_min": p.get("t1_min") if p.get("t1_min") is not None else _jha_time_to_min_gof(p.get("t1") or p.get("end_time") or p.get("end")),
    }

def _jha_operation_schedule_context_gof(input_state):

    input_state = _jha_dict(input_state)
    task = _jha_dict(input_state.get("operation_schedule_task"))
    hist_segments = (
        _jha_list(input_state.get("visible_history_operation_segments"))
        or _jha_list(input_state.get("history_operation_segments"))
        or _jha_list(task.get("visible_history_operation_segments"))
        or _jha_list(task.get("history_operation_segments"))
        or _jha_list(task.get("local_history_operation_segments"))
    )
    future_calendar = (
        _jha_list(input_state.get("known_future_patch_calendar"))
        or _jha_list(task.get("future_calendar"))
        or _jha_list(task.get("known_future_patch_calendar"))
    )

    history_examples = []
    for i, seg in enumerate(hist_segments[-32:], start=1):
        b = _jha_segment_brief_gof(seg, i)
        if b:
            history_examples.append(b)

    future_calendar_brief = []
    future_heat_ids = []
    for i, p in enumerate(future_calendar, start=1):
        b = _jha_patch_calendar_brief_gof(p, i)
        if b:
            future_calendar_brief.append(b)
            hid = b.get("canonical_heat_id")
            if hid and hid not in future_heat_ids:
                future_heat_ids.append(hid)

    return {
        "available": bool(task or hist_segments or future_calendar),
        "task": "production_gantt_only",
        "objective": "predict route-level operation segments for production Gantt comparison",
        "removed_tasks": [
            "production_patch_level_active_heat_prediction",
            "production_heat_patch_alignment_prediction",
        ],
        "required_process_order": task.get("required_process_order") or task.get("process_order") or ["BOF", "LF", "Casting", "Rolling"],
        "candidate_future_heat_ids": task.get("candidate_future_heat_ids") or task.get("future_heat_ids") or future_heat_ids,
        "future_time_window": task.get("future_time_window") or input_state.get("known_future_time_window") or {},
        "known_future_patch_calendar": future_calendar_brief,
        "duration_priors": task.get("duration_priors", {}),
        "carryover_active_heats_at_anchor": task.get("carryover_active_heats_at_anchor", []),
        "num_history_operation_segments": len(hist_segments),
        "history_operation_segment_examples": history_examples,
        "prediction_output_contract": {
            "required_key": "predicted_operation_segments",
            "valid_source": "model_output",
            "forbidden_prediction_outputs": ["future_production_states", "active_heat_alignment_labels"],
            "diagnostic_or_derived_segments_are_not_valid_predictions": True,
        },
    }

def _jha_energy_context_gof(energy_input, max_history=8):
    energy_input = _jha_dict(energy_input)
    hist = []
    for item in _jha_list(energy_input.get("energy_history"))[-max_history:]:
        if not isinstance(item, dict):
            continue
        hist.append({
            "step": item.get("step") or item.get("sq"),
            "patch_id": item.get("id") or item.get("patch_id") or item.get("p"),
            "heat_id": _jha_normalize_heat_id(item.get("hn") or item.get("heat_id") or item.get("canonical_heat_id")),
            "risk_flags": item.get("rk", []),
            "response_type": item.get("response_type"),
        })
    future = []
    for i, item in enumerate(_jha_list(energy_input.get("future_patch_context")), start=1):
        if not isinstance(item, dict):
            continue
        future.append({
            "step": int(item.get("step") or i),
            "patch_id": item.get("id") or item.get("patch_id") or item.get("p"),
            "heat_id": _jha_normalize_heat_id(item.get("hn") or item.get("heat_id") or item.get("canonical_heat_id")),
            "context": item,
        })
    return {
        "history_energy_patch_examples": hist,
        "future_energy_patch_context": future,
        "num_history_energy_patches": len(_jha_list(energy_input.get("energy_history"))),
        "num_future_energy_context": len(future),
    }

def _jha_energy_patch_brief_gof(p, step):
    if not isinstance(p, dict):
        return {"step": step}
    return {
        "step": int(p.get("step") or step),
        "patch_id": p.get("id") or p.get("patch_id") or p.get("p"),
        "heat_id": _jha_normalize_heat_id(p.get("hn") or p.get("heat_id") or p.get("canonical_heat_id")),
        "risk_flags": p.get("rk") or p.get("risk_flags") or [],
        "response_type": p.get("response_type"),
        "feasible": p.get("feasible", True),
    }

def _jha_segment_activity_for_window_gof(segments, t0_min, t1_min):
    out = {
        "num_segments": 0,
        "heat_ids": [],
        "stage_counts": {},
        "unit_counts": {},
    }
    if t0_min is None or t1_min is None or t1_min <= t0_min:
        return out
    for seg in _jha_list(segments):
        if not isinstance(seg, dict):
            continue
        s = seg.get("start_min") if seg.get("start_min") is not None else _jha_time_to_min_gof(seg.get("start_time") or seg.get("t0"))
        e = seg.get("end_min") if seg.get("end_min") is not None else _jha_time_to_min_gof(seg.get("end_time") or seg.get("t1"))
        try:
            s = float(s); e = float(e)
        except Exception:
            continue
        if max(s, float(t0_min)) >= min(e, float(t1_min)):
            continue
        out["num_segments"] += 1
        hid = _jha_normalize_heat_id(seg.get("heat_id") or seg.get("id") or seg.get("canonical_heat_id"))
        if hid and hid not in out["heat_ids"]:
            out["heat_ids"].append(hid)
        st = _jha_stage_gof(seg.get("stage") or seg.get("operation"))
        out["stage_counts"][st] = out["stage_counts"].get(st, 0) + 1
        unit = _jha_str(seg.get("unit") or seg.get("resource") or seg.get("ru"))
        if unit:
            out["unit_counts"][unit] = out["unit_counts"].get(unit, 0) + 1
    return out

def _jha_operation_target_summary_gof(segments):
    segs = [x for x in _jha_list(segments) if isinstance(x, dict)]
    heats = []
    stage_counts = {}
    unit_counts = {}
    starts = []
    ends = []
    for seg in segs:
        hid = _jha_normalize_heat_id(seg.get("heat_id") or seg.get("id") or seg.get("canonical_heat_id"))
        if hid and hid not in heats:
            heats.append(hid)
        st = _jha_stage_gof(seg.get("stage") or seg.get("operation"))
        stage_counts[st] = stage_counts.get(st, 0) + 1
        unit = _jha_str(seg.get("unit") or seg.get("resource") or seg.get("ru"))
        if unit:
            unit_counts[unit] = unit_counts.get(unit, 0) + 1
        s = seg.get("start_min") if seg.get("start_min") is not None else _jha_time_to_min_gof(seg.get("start_time"))
        e = seg.get("end_min") if seg.get("end_min") is not None else _jha_time_to_min_gof(seg.get("end_time"))
        try:
            if s is not None:
                starts.append(float(s))
            if e is not None:
                ends.append(float(e))
        except Exception:
            pass
    return {
        "num_segments": len(segs),
        "heat_ids": heats[:64],
        "stage_counts": stage_counts,
        "unit_counts": unit_counts,
        "time_span": {
            "start_min": min(starts) if starts else None,
            "end_min": max(ends) if ends else None,
        },
    }

def _jha_build_functional_coupling_target_gof(prod_segments, energy_future, future_context):

    pairs = []
    n = max(len(_jha_list(energy_future)), len(_jha_list(future_context)))
    for i in range(n):
        step = i + 1
        ctx = _jha_dict(_jha_list(future_context)[i]) if i < len(_jha_list(future_context)) else {}
        prod_cal = _jha_dict(ctx.get("production_calendar"))
        energy_ctx = _jha_dict(ctx.get("energy_patch_context"))
        t0 = prod_cal.get("t0_min")
        t1 = prod_cal.get("t1_min")
        if t0 is None:
            t0 = _jha_time_to_min_gof(prod_cal.get("t0") or energy_ctx.get("t0") or energy_ctx.get("start_time"))
        if t1 is None:
            t1 = _jha_time_to_min_gof(prod_cal.get("t1") or energy_ctx.get("t1") or energy_ctx.get("end_time"))
        activity = _jha_segment_activity_for_window_gof(prod_segments, t0, t1)
        e_true = _jha_dict(_jha_list(energy_future)[i]) if i < len(_jha_list(energy_future)) else {}
        energy_brief = _jha_energy_patch_brief_gof(e_true, step)
        pairs.append({
            "step": step,
            "energy_patch_id": energy_brief.get("patch_id") or energy_ctx.get("id") or energy_ctx.get("patch_id"),
            "energy_heat_no": energy_brief.get("heat_id") or _jha_normalize_heat_id(energy_ctx.get("hn") or energy_ctx.get("heat_id")),
            "production_operation_activity": activity,
            "risk_flags": energy_brief.get("risk_flags", []),
            "has_risk": bool(energy_brief.get("risk_flags")),
            "feasible": energy_brief.get("feasible", True),
        })
    return {
        "mode": "functional_production_schedule_energy_target",
        "future_pairs": pairs,
        "schema": {
            "production_operation_activity": "true operation-segment activity overlapping the energy patch time window",
            "risk_flags": "true energy risk flags for offline coupling evaluation",
            "note": "This target is hidden from all model inputs.",
        },
    }

def _jha_build_joint_horizon_sample_gantt_functional(
    self,
    day_id,
    anchor_patch_idx,
    history_window=5,
    horizon=5,
    include_target=True,
):

    if not hasattr(self.production_adapter, "build_production_horizon_sample"):
        raise AttributeError("ProductionStateAdapter lacks build_production_horizon_sample().")
    if not hasattr(self.energy_adapter, "build_energy_horizon_sample"):
        raise AttributeError("EnergyStateAdapter lacks build_energy_horizon_sample().")

    prod_sample = self.production_adapter.build_production_horizon_sample(
        day_id=day_id,
        anchor_patch_idx=anchor_patch_idx,
        history_window=history_window,
        horizon=horizon,
        include_target=include_target,
    )
    prod_input = _jha_dict(prod_sample.get("input_state"))
    schedule_context = _jha_operation_schedule_context_gof(prod_input)

    production_context = {
        "mode": "gantt_only_production_context_for_energy",
        "operation_schedule_context": schedule_context,
        "known_future_patch_calendar": schedule_context.get("known_future_patch_calendar", []),
        "future_time_window": schedule_context.get("future_time_window", {}),
        "leakage_guard": {
            "contains_target_future_operation_segments": False,
            "contains_future_production_patch_labels": False,
            "target_future_is_hidden": True,
        },
    }

    energy_sample = self.energy_adapter.build_energy_horizon_sample(
        day_id=day_id,
        anchor_patch_idx=anchor_patch_idx,
        history_window=history_window,
        horizon=horizon,
        production_context=production_context,
        production_prediction={},
        include_target=include_target,
    )
    energy_input = _jha_dict(energy_sample.get("input_state"))
    energy_context = _jha_energy_context_gof(energy_input)

    future_calendar = schedule_context.get("known_future_patch_calendar", [])
    energy_future_context = _jha_list(energy_input.get("future_patch_context"))
    n_future_ctx = min(len(_jha_list(future_calendar)), len(energy_future_context))
    future_context = []
    for i in range(n_future_ctx):
        future_context.append({
            "step": i + 1,
            "production_calendar": _jha_dict(future_calendar[i]),
            "energy_patch_context": energy_future_context[i],
        })

    sample = {
        "sample_id": f"{day_id}_joint_anchor_{anchor_patch_idx}_W{history_window}_K{horizon}",
        "task": "gantt_energy_functional_coupling_horizon_sample",
        "day_id": day_id,
        "anchor_patch_idx": int(anchor_patch_idx),
        "history_window": int(history_window),
        "horizon": int(horizon),
        "input_state": {
            "mode": "gantt_only_production_energy_functional_coupling",
            "day_id": day_id,
            "anchor_patch_idx": int(anchor_patch_idx),
            "history_window": int(history_window),
            "horizon": int(horizon),
            "production_schedule_input": schedule_context,
            "energy_input": energy_input,
            "energy_context_for_coupling": energy_context,
            "functional_coupling_context": {
                "mode": "operation_schedule_to_energy_response",
                "future_context": future_context,
                "description": "Coupling is evaluated between predicted production operation segments and predicted energy response, not by asking production to predict patch active_heat labels.",
            },

            "production_operation_schedule_context": schedule_context,
            "future_context": future_context,
            "schema": {
                "production_schedule_input": "model-visible production Gantt schedule context from history/calendar only",
                "energy_input": "energy model input; target labels are hidden inside component target_future only",
                "functional_coupling_context": "compact future context for functional production-energy coupling",
                "removed_tasks": [
                    "production_patch_level_state_prediction",
                    "production_heat_patch_alignment_prediction",
                    "coupling_heat_patch_alignment_as_primary_task",
                ],
                "target_future": "hidden true future labels used only for offline evaluation",
            },
            "leakage_guard": {
                "target_future_exposed_to_model": False,
                "target_operation_segments_exposed_to_model": False,
                "future_production_patch_labels_exposed_to_model": False,
            },
        },
        "component_samples": {
            "production": prod_sample,
            "energy": energy_sample,
        },
        "meta": {
            "num_future_context": len(future_context),
            "is_truncated_future": len(future_context) < int(horizon),
            "production_task": "operation_level_gantt_prediction_only",
            "energy_task": "energy_horizon_prediction",
            "coupling_task": "functional_operation_schedule_energy_response_consistency",
            "operation_schedule_context_available": bool(schedule_context.get("available")),
            "operation_schedule_context_chars": _jha_safe_json_chars_gof(schedule_context),
            "functional_coupling_context_chars": _jha_safe_json_chars_gof(future_context),
            "gantt_policy": {
                "model_valid_segment_source": "model_output",
                "target_route_segments_key": "target_future.production_operation_segments",
                "derived_or_fallback_segments_are_diagnostics_only": True,
            },
        },
    }

    if include_target:
        prod_target = _jha_dict(prod_sample.get("target_future"))
        prod_future = _jha_list(prod_target.get("future_patches"))
        prod_operation_segments = _jha_list(prod_target.get("production_operation_segments") or prod_target.get("operation_segments"))
        prod_patch_operation_segments = _jha_list(prod_target.get("production_patch_operation_segments") or prod_target.get("patch_operation_segments"))
        energy_future = _jha_list(_jha_dict(energy_sample.get("target_future")).get("energy_future"))
        functional_target = _jha_build_functional_coupling_target_gof(
            prod_segments=prod_operation_segments,
            energy_future=energy_future,
            future_context=future_context,
        )
        sample["coupling_target"] = functional_target
        sample["target_future"] = {
            "mode": "gantt_energy_functional_future_label",
            "day_id": day_id,
            "anchor_patch_idx": int(anchor_patch_idx),
            "horizon": int(horizon),
            "production_future": prod_future,
            "production_operation_segments": prod_operation_segments,
            "production_patch_operation_segments": prod_patch_operation_segments,
            "energy_future": energy_future,
            "functional_coupling_target": functional_target,
            "coupling_target": functional_target,
            "operation_segment_schema": {
                "production_operation_segments": "route-level true operation segments clipped to the future horizon window; primary target for Gantt comparison",
                "production_patch_operation_segments": "patch-derived true segments; diagnostic only",
                "model_prediction_key": "production_prediction.predicted_operation_segments",
                "valid_prediction_source": "model_output",
            },
            "operation_target_summary": {
                "route_level": _jha_operation_target_summary_gof(prod_operation_segments),
                "patch_level_diagnostic": _jha_operation_target_summary_gof(prod_patch_operation_segments),
            },
        }
        sample["meta"].update({
            "num_target_production_operation_segments": len(prod_operation_segments),
            "num_target_production_patch_operation_segments": len(prod_patch_operation_segments),
            "num_target_energy_future_steps": len(energy_future),
            "num_functional_coupling_target_pairs": len(_jha_list(functional_target.get("future_pairs"))),
        })
    return sample

def _jha_build_joint_horizon_dataset_gantt_functional(
    self,
    day_id,
    horizon=5,
    history_window=5,
    stride=1,
    max_samples=None,
    anchor_indices=None,
):
    if anchor_indices:
        anchors = [int(x) for x in anchor_indices]
    else:
        if not hasattr(self.production_adapter, "build_production_patch_sequence"):
            raise AttributeError("ProductionStateAdapter lacks build_production_patch_sequence().")
        seq = self.production_adapter.build_production_patch_sequence(day_id)
        last_anchor = max(history_window - 1, len(seq) - 2)
        anchors = list(range(history_window - 1, last_anchor + 1, max(1, int(stride))))
    samples = []
    for anchor in anchors:
        samples.append(self.build_joint_horizon_sample(
            day_id=day_id,
            anchor_patch_idx=anchor,
            history_window=history_window,
            horizon=horizon,
            include_target=True,
        ))
        if max_samples is not None and len(samples) >= int(max_samples):
            break
    return samples

def _jha_build_joint_horizon_prompt_state_gantt_functional(
    self,
    day_id,
    anchor_patch_idx,
    history_window=5,
    horizon=5,
):
    sample = self.build_joint_horizon_sample(
        day_id=day_id,
        anchor_patch_idx=anchor_patch_idx,
        history_window=history_window,
        horizon=horizon,
        include_target=False,
    )
    return sample.get("input_state", {})

def _jha_debug_joint_horizon_sample_report_gantt_functional(
    self,
    day_id,
    anchor_patch_idx,
    history_window=5,
    horizon=5,
):
    sample = self.build_joint_horizon_sample(
        day_id=day_id,
        anchor_patch_idx=anchor_patch_idx,
        history_window=history_window,
        horizon=horizon,
        include_target=True,
    )
    report = {
        "sample_id": sample.get("sample_id"),
        "task": sample.get("task"),
        "input_chars": self._json_len(sample.get("input_state", {})) if hasattr(self, "_json_len") else _jha_safe_json_chars_gof(sample.get("input_state", {})),
        "target_chars": self._json_len(sample.get("target_future", {})) if hasattr(self, "_json_len") else _jha_safe_json_chars_gof(sample.get("target_future", {})),
        "production_task": _jha_dict(sample.get("meta")).get("production_task"),
        "energy_task": _jha_dict(sample.get("meta")).get("energy_task"),
        "coupling_task": _jha_dict(sample.get("meta")).get("coupling_task"),
        "num_target_production_operation_segments": _jha_dict(sample.get("meta")).get("num_target_production_operation_segments", 0),
        "num_functional_coupling_target_pairs": _jha_dict(sample.get("meta")).get("num_functional_coupling_target_pairs", 0),
    }
    print("\n=== Gantt-only Functional Joint Horizon Sample Report ===")
    for k, v in report.items():
        print(f"{k}: {v}")
    return report

JointStateAssembler.build_joint_horizon_sample = _jha_build_joint_horizon_sample_gantt_functional
JointStateAssembler.build_joint_horizon_dataset = _jha_build_joint_horizon_dataset_gantt_functional
JointStateAssembler.build_joint_horizon_prompt_state = _jha_build_joint_horizon_prompt_state_gantt_functional
JointStateAssembler.debug_joint_horizon_sample_report = _jha_debug_joint_horizon_sample_report_gantt_functional

def _jha_bool_pcvp(x, default=False):
    if isinstance(x, bool):
        return x
    if x is None:
        return default
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"true", "1", "yes", "y", "ok", "available"}:
            return True
        if s in {"false", "0", "no", "n", "missing"}:
            return False
    return default

def _jha_build_pcvp_contract(input_state, schedule_context, energy_input):
    energy_repair_context = _jha_dict(energy_input.get("energy_repair_context"))
    return {
        "method": "LLM-guided Physics-consistent Variational Projection",
        "method_short": "PCVP",
        "production_projection": {
            "state": "applied_after_LLM_production_prediction",
            "input_key": "production_prediction.predicted_operation_segments",
            "output_key": "production_prediction.predicted_operation_segments",
            "diagnostic_keys": [
                "raw_predicted_operation_segments_before_projection",
                "uncertain_operation_segments",
                "rejected_operation_segments",
                "mechanism_projection_diagnostics",
                "repair_meta",
            ],
            "policy": "strong_repair_first_weak_deletion_keep_uncertain_diagnostics",
        },
        "energy_projection": {
            "state": "applied_after_LLM_energy_prediction",
            "input_key": "energy_prediction.future_patch_predictions",
            "output_key": "energy_prediction.future_patch_predictions",
            "diagnostic_keys": [
                "raw_future_patch_predictions_before_energy_projection",
                "uncertain_energy_predictions",
                "rejected_energy_predictions",
                "mechanism_energy_projection_diagnostics",
            ],
            "policy": "project_energy_patch_predictions_to_physical_feasible_space_without_target_future",
            "energy_repair_context_available": bool(energy_repair_context),
        },
        "energy_conditioning_contract": {
            "required_input_from_transition_engine": "production_operation_segment_summary",
            "must_be_true_when_segments_available": "energy_prediction.production_conditioning_available",
            "expected_basis": "schedule_conditioned",
            "forbidden_basis_when_summary_available": "history_only",
        },
        "leakage_guard": {
            "target_operation_segments_exposed_to_model": False,
            "target_energy_future_exposed_to_model": False,
            "target_coupling_pairs_exposed_to_model": False,
            "targets_used_only_for_offline_evaluation": True,
        },
    }

def _jha_build_joint_horizon_sample_pcvp(
    self,
    day_id,
    anchor_patch_idx,
    history_window=5,
    horizon=5,
    include_target=True,
):

    if not hasattr(self.production_adapter, "build_production_horizon_sample"):
        raise AttributeError("ProductionStateAdapter lacks build_production_horizon_sample().")
    if not hasattr(self.energy_adapter, "build_energy_horizon_sample"):
        raise AttributeError("EnergyStateAdapter lacks build_energy_horizon_sample().")

    prod_sample = self.production_adapter.build_production_horizon_sample(
        day_id=day_id,
        anchor_patch_idx=anchor_patch_idx,
        history_window=history_window,
        horizon=horizon,
        include_target=include_target,
    )
    prod_input = _jha_dict(prod_sample.get("input_state"))
    schedule_context = _jha_operation_schedule_context_gof(prod_input)

    production_context_for_energy = {
        "mode": "pcvp_gantt_only_production_context_for_energy",
        "operation_schedule_context": schedule_context,
        "known_future_patch_calendar": schedule_context.get("known_future_patch_calendar", []),
        "future_time_window": schedule_context.get("future_time_window", {}),
        "pcvp_note": (
            "The actual production_operation_segment_summary is attached by "
            "transition_engine.py after production PCVP projection."
        ),
        "leakage_guard": {
            "contains_target_future_operation_segments": False,
            "contains_future_production_patch_labels": False,
            "target_future_is_hidden": True,
        },
    }

    energy_sample = self.energy_adapter.build_energy_horizon_sample(
        day_id=day_id,
        anchor_patch_idx=anchor_patch_idx,
        history_window=history_window,
        horizon=horizon,
        production_context=production_context_for_energy,
        production_prediction={},
        include_target=include_target,
    )
    energy_input = _jha_dict(energy_sample.get("input_state"))
    energy_context = _jha_energy_context_gof(energy_input)

    future_calendar = _jha_list(schedule_context.get("known_future_patch_calendar"))
    energy_future_context = _jha_list(energy_input.get("future_patch_context"))
    n_future_ctx = min(len(future_calendar), len(energy_future_context))
    future_context = []
    for i in range(n_future_ctx):
        future_context.append({
            "step": i + 1,
            "production_calendar": _jha_dict(future_calendar[i]),
            "energy_patch_context": energy_future_context[i],
        })

    pcvp_contract = _jha_build_pcvp_contract(
        input_state=prod_input,
        schedule_context=schedule_context,
        energy_input=energy_input,
    )

    sample = {
        "sample_id": f"{day_id}_joint_anchor_{anchor_patch_idx}_W{history_window}_K{horizon}",
        "task": "pcvp_gantt_energy_functional_coupling_horizon_sample",
        "day_id": day_id,
        "anchor_patch_idx": int(anchor_patch_idx),
        "history_window": int(history_window),
        "horizon": int(horizon),
        "input_state": {
            "mode": "pcvp_gantt_energy_functional_coupling",
            "day_id": day_id,
            "anchor_patch_idx": int(anchor_patch_idx),
            "history_window": int(history_window),
            "horizon": int(horizon),
            "production_schedule_input": schedule_context,
            "production_operation_schedule_context": schedule_context,
            "energy_input": energy_input,
            "energy_context_for_coupling": energy_context,
            "energy_repair_context": _jha_dict(energy_input.get("energy_repair_context")),
            "functional_coupling_context": {
                "mode": "operation_schedule_to_energy_response_pcvp",
                "future_context": future_context,
                "description": (
                    "Coupling is evaluated between PCVP-repaired production "
                    "operation segments and PCVP-repaired energy response. "
                    "It is not a heat-patch alignment prediction task."
                ),
            },
            "pcvp_contract": pcvp_contract,
            "schema": {
                "production_schedule_input": "model-visible production Gantt context from history/calendar only",
                "energy_input": "energy model input; true future energy labels are hidden",
                "energy_repair_context": "allowed physics context for energy PCVP, with no target_future values",
                "functional_coupling_context": "compact future context for schedule-to-energy functional coupling",
                "removed_tasks": [
                    "production_patch_level_state_prediction",
                    "production_heat_patch_alignment_prediction",
                    "coupling_heat_patch_alignment_as_primary_task",
                    "joint_optimization_decision_recommendation",
                ],
                "target_future": "hidden true future labels used only for offline evaluation",
            },
            "leakage_guard": pcvp_contract.get("leakage_guard", {}),
        },
        "component_samples": {
            "production": prod_sample,
            "energy": energy_sample,
        },
        "meta": {
            "num_future_context": len(future_context),
            "is_truncated_future": len(future_context) < int(horizon),
            "production_task": "operation_level_gantt_prediction_with_pcvp_projection",
            "energy_task": "schedule_conditioned_energy_horizon_prediction_with_pcvp_projection",
            "coupling_task": "functional_operation_schedule_energy_response_consistency",
            "operation_schedule_context_available": bool(schedule_context.get("available")),
            "operation_schedule_context_chars": _jha_safe_json_chars_gof(schedule_context),
            "functional_coupling_context_chars": _jha_safe_json_chars_gof(future_context),
            "pcvp_contract_chars": _jha_safe_json_chars_gof(pcvp_contract),
            "gantt_policy": {
                "model_valid_segment_source": "model_output",
                "target_route_segments_key": "target_future.production_operation_segments",
                "derived_or_fallback_segments_are_diagnostics_only": True,
                "pcvp_projection_diagnostics_required": True,
            },
        },
    }

    if include_target:
        prod_target = _jha_dict(prod_sample.get("target_future"))
        prod_future = _jha_list(prod_target.get("future_patches"))
        prod_operation_segments = _jha_list(
            prod_target.get("production_operation_segments")
            or prod_target.get("operation_segments")
        )
        prod_patch_operation_segments = _jha_list(
            prod_target.get("production_patch_operation_segments")
            or prod_target.get("patch_operation_segments")
        )
        energy_future = _jha_list(_jha_dict(energy_sample.get("target_future")).get("energy_future"))
        functional_target = _jha_build_functional_coupling_target_gof(
            prod_segments=prod_operation_segments,
            energy_future=energy_future,
            future_context=future_context,
        )
        sample["coupling_target"] = functional_target
        sample["target_future"] = {
            "mode": "pcvp_gantt_energy_functional_future_label",
            "day_id": day_id,
            "anchor_patch_idx": int(anchor_patch_idx),
            "horizon": int(horizon),
            "production_future": prod_future,
            "production_operation_segments": prod_operation_segments,
            "production_patch_operation_segments": prod_patch_operation_segments,
            "energy_future": energy_future,
            "functional_coupling_target": functional_target,
            "coupling_target": functional_target,
            "operation_segment_schema": {
                "production_operation_segments": "true route-level operation segments clipped to future horizon; hidden from model",
                "production_patch_operation_segments": "patch-derived true segments; diagnostic only",
                "model_prediction_key": "production_prediction.predicted_operation_segments",
                "valid_prediction_source": "model_output",
                "pcvp_prediction_key": "production_prediction.predicted_operation_segments after PCVP projection",
            },
            "operation_target_summary": {
                "route_level": _jha_operation_target_summary_gof(prod_operation_segments),
                "patch_level_diagnostic": _jha_operation_target_summary_gof(prod_patch_operation_segments),
            },
        }
        sample["meta"].update({
            "num_target_production_operation_segments": len(prod_operation_segments),
            "num_target_production_patch_operation_segments": len(prod_patch_operation_segments),
            "num_target_energy_future_steps": len(energy_future),
            "num_functional_coupling_target_pairs": len(_jha_list(functional_target.get("future_pairs"))),
        })

    return sample

def _jha_build_joint_horizon_dataset_pcvp(
    self,
    day_id,
    horizon=5,
    history_window=5,
    stride=1,
    max_samples=None,
    anchor_indices=None,
):

    horizon = int(horizon)
    history_window = int(history_window)
    stride = max(1, int(stride))

    if not hasattr(self.production_adapter, "build_production_patch_sequence"):
        raise AttributeError("ProductionStateAdapter lacks build_production_patch_sequence().")
    seq = self.production_adapter.build_production_patch_sequence(day_id)
    n_seq = len(seq) if isinstance(seq, list) else 0

    min_anchor = history_window - 1
    last_anchor = n_seq - horizon - 1

    errors = []
    if anchor_indices:
        anchors = [int(x) for x in anchor_indices]
        checked_anchors = []
        for anchor in anchors:
            if anchor < min_anchor:
                errors.append({
                    "sample_id": f"{day_id}_joint_anchor_{anchor}_W{history_window}_K{horizon}",
                    "anchor_patch_idx": anchor,
                    "horizon": horizon,
                    "error": "anchor_before_history_window",
                    "error_message": f"anchor={anchor} < minimum valid anchor={min_anchor}",
                })
                continue
            if anchor > last_anchor:
                errors.append({
                    "sample_id": f"{day_id}_joint_anchor_{anchor}_W{history_window}_K{horizon}",
                    "anchor_patch_idx": anchor,
                    "horizon": horizon,
                    "error": "insufficient_future_patches_for_horizon",
                    "error_message": (
                        f"anchor+horizon={anchor + horizon} exceeds last patch index={n_seq - 1}; "
                        f"maximum valid anchor for K={horizon} is {last_anchor}"
                    ),
                })
                continue
            checked_anchors.append(anchor)
        anchors = checked_anchors
    else:
        if n_seq <= 0:
            raise ValueError(f"No production patch sequence available for day_id={day_id}.")
        if last_anchor < min_anchor:
            raise ValueError(
                f"Not enough production patches for exact horizon: day_id={day_id}, "
                f"num_patches={n_seq}, history_window={history_window}, horizon={horizon}."
            )
        anchors = list(range(min_anchor, last_anchor + 1, stride))

    if max_samples is not None:
        anchors = anchors[: int(max_samples)]

    samples = []
    for anchor in anchors:
        try:
            sample = self.build_joint_horizon_sample(
                day_id=day_id,
                anchor_patch_idx=anchor,
                history_window=history_window,
                horizon=horizon,
                include_target=True,
            )
            comp = _jha_dict(sample.get("component_samples"))
            prod_ok = bool(_jha_dict(comp.get("production")))
            energy_ok = bool(_jha_dict(comp.get("energy")))
            if not prod_ok or not energy_ok:
                errors.append({
                    "sample_id": sample.get("sample_id") or f"{day_id}_joint_anchor_{anchor}_W{history_window}_K{horizon}",
                    "anchor_patch_idx": int(anchor),
                    "horizon": horizon,
                    "error": "missing_component_samples_after_build",
                    "error_message": "sample lacks component_samples.production or component_samples.energy",
                    "sample_keys": list(sample.keys()) if isinstance(sample, dict) else [],
                    "component_keys": list(comp.keys()),
                    "has_production": prod_ok,
                    "has_energy": energy_ok,
                })
                continue
            samples.append(sample)
        except Exception as exc:
            errors.append({
                "sample_id": f"{day_id}_joint_anchor_{anchor}_W{history_window}_K{horizon}",
                "day_id": day_id,
                "anchor_patch_idx": int(anchor),
                "history_window": int(history_window),
                "horizon": int(horizon),
                "error": f"build_joint_horizon_sample_failed:{type(exc).__name__}",
                "error_message": str(exc),
            })

    self._last_horizon_sample_build_errors = errors

    if not samples:
        raise ValueError(
            f"No valid PCVP joint horizon samples were built for day_id={day_id}, K={horizon}. "
            f"num_patches={n_seq}, min_anchor={min_anchor}, max_valid_anchor={last_anchor}, "
            f"num_errors={len(errors)}. First errors: {errors[:3]}"
        )

    return samples

def _jha_build_joint_horizon_prompt_state_pcvp(self, joint_sample):
    js = _jha_dict(joint_sample)
    inp = _jha_dict(js.get("input_state"))
    return {
        "sample_id": js.get("sample_id"),
        "task": js.get("task"),
        "mode": inp.get("mode"),
        "day_id": js.get("day_id"),
        "anchor_patch_idx": js.get("anchor_patch_idx"),
        "history_window": js.get("history_window"),
        "horizon": js.get("horizon"),
        "production_schedule_input": inp.get("production_schedule_input"),
        "energy_context_for_coupling": inp.get("energy_context_for_coupling"),
        "functional_coupling_context": inp.get("functional_coupling_context"),
        "pcvp_contract": inp.get("pcvp_contract"),
        "leakage_guard": inp.get("leakage_guard"),
        "target_policy": {
            "target_future_hidden": True,
            "offline_evaluation_only": True,
        },
    }

def _jha_debug_joint_horizon_sample_report_pcvp(self, sample):
    sample = _jha_dict(sample)
    inp = _jha_dict(sample.get("input_state"))
    comp = _jha_dict(sample.get("component_samples"))
    prod = _jha_dict(comp.get("production"))
    energy = _jha_dict(comp.get("energy"))
    target = _jha_dict(sample.get("target_future"))
    return {
        "sample_id": sample.get("sample_id"),
        "task": sample.get("task"),
        "mode": inp.get("mode"),
        "horizon": sample.get("horizon"),
        "production_component_available": bool(prod),
        "energy_component_available": bool(energy),
        "pcvp_contract_available": bool(inp.get("pcvp_contract")),
        "energy_repair_context_available": bool(inp.get("energy_repair_context")),
        "functional_coupling_context_available": bool(_jha_dict(inp.get("functional_coupling_context")).get("future_context")),
        "target_future_available": bool(target),
        "num_target_operation_segments": len(_jha_list(target.get("production_operation_segments"))),
        "num_target_energy_future": len(_jha_list(target.get("energy_future"))),
        "leakage_guard": inp.get("leakage_guard"),
    }

JointStateAssembler.build_joint_horizon_sample = _jha_build_joint_horizon_sample_pcvp
JointStateAssembler.build_joint_horizon_dataset = _jha_build_joint_horizon_dataset_pcvp
JointStateAssembler.build_joint_horizon_prompt_state = _jha_build_joint_horizon_prompt_state_pcvp
JointStateAssembler.debug_joint_horizon_sample_report = _jha_debug_joint_horizon_sample_report_pcvp

def _jha_count_retry_policy_v1(env_name: str, default_value: int) -> int:
    try:
        import os
        return max(0, int(os.environ.get(env_name, str(default_value))))
    except Exception:
        return default_value

def _jha_build_coupling_retry_fallback_contract_v1(horizon: int) -> Dict[str, Any]:
    max_retry = _jha_count_retry_policy_v1("COUPLING_MAX_HORIZON_RETRIES", 2)
    import os
    fallback_enabled = str(os.environ.get("COUPLING_ENABLE_RULE_FALLBACK_AFTER_RETRY", "1")).strip().lower() not in {"0", "false", "no", "off"}
    return {
        "method": "coupling_llm_self_refinement_with_rule_assisted_fallback",
        "max_horizon_retries": max_retry,
        "retry_enabled": max_retry > 0,
        "rule_fallback_after_retry_enabled": bool(fallback_enabled),
        "horizon": int(horizon),
        "allowed_retry_feedback": [
            "schema_errors",
            "missing_predicted_coupling_future",
            "missing_or_duplicate_step_ids",
            "horizon_length_mismatch",
            "production_operation_segment_summary",
            "energy_prediction_summary",
            "pcvp_projection_diagnostics",
            "future_patch_context",
        ],
        "forbidden_feedback": [
            "target_future",
            "coupling_target",
            "target_operation_segments",
            "target_energy_future",
            "offline_metrics",
            "accuracy",
            "f1",
            "mae",
            "rmse",
        ],
        "fallback_policy": {
            "name": "retry_exhausted_rule_assisted_functional_coupling_completion",
            "uses_future_target_labels": False,
            "inputs": [
                "future_context",
                "production_operation_segment_summary",
                "energy_prediction_summary",
                "pcvp_contract",
            ],
            "status": "diagnostic_and_engineering_safety_fallback",
        },
    }

_jha_previous_build_joint_horizon_sample_for_retry_pcvp = JointStateAssembler.build_joint_horizon_sample
_jha_previous_build_joint_horizon_prompt_state_for_retry_pcvp = JointStateAssembler.build_joint_horizon_prompt_state
_jha_previous_debug_joint_horizon_sample_report_for_retry_pcvp = JointStateAssembler.debug_joint_horizon_sample_report

def _jha_build_joint_horizon_sample_retry_pcvp(
    self,
    day_id: str,
    anchor_patch_idx: int,
    history_window: int = 5,
    horizon: int = 1,
    include_target: bool = True,
) -> Dict[str, Any]:
    sample = _jha_previous_build_joint_horizon_sample_for_retry_pcvp(
        self,
        day_id=day_id,
        anchor_patch_idx=anchor_patch_idx,
        history_window=history_window,
        horizon=horizon,
        include_target=include_target,
    )
    if not isinstance(sample, dict):
        return sample

    input_state = sample.setdefault("input_state", {})
    if not isinstance(input_state, dict):
        input_state = {}
        sample["input_state"] = input_state

    retry_contract = _jha_build_coupling_retry_fallback_contract_v1(horizon)
    input_state["coupling_retry_fallback_contract"] = retry_contract
    input_state["coupling_task_contract"] = {
        "task": "operation_schedule_to_energy_response_functional_coupling",
        "production_input_status": "production result is expected to be LLM retry/self-refined + PCVP projected; rule-assisted fallback may be present and must be marked",
        "energy_input_status": "energy result is expected to be schedule-conditioned + Energy-PCVP projected; rule-assisted fallback may be present and must be marked",
        "coupling_output_status": "coupling LLM output may be self-refined; if retry is exhausted, deterministic functional fallback is allowed and marked",
        "target_leakage_guard": {
            "uses_future_target_labels": False,
            "target_future_hidden_from_llm": True,
            "offline_evaluation_only": True,
        },
    }

    meta = sample.setdefault("meta", {})
    if isinstance(meta, dict):
        meta["coupling_retry_enabled"] = bool(retry_contract.get("retry_enabled"))
        meta["coupling_max_horizon_retries"] = retry_contract.get("max_horizon_retries")
        meta["coupling_rule_fallback_after_retry_enabled"] = bool(retry_contract.get("rule_fallback_after_retry_enabled"))
        meta["coupling_retry_fallback_contract_chars"] = _jha_safe_json_chars_gof(retry_contract) if "_jha_safe_json_chars_gof" in globals() else len(json.dumps(retry_contract, ensure_ascii=False, default=str))

    comp = _jha_dict(sample.get("component_samples"))
    if not comp.get("production") or not comp.get("energy"):
        sample.setdefault("_build_warnings", []).append("component_samples missing after retry/pcvp coupling enrichment")
    return sample

def _jha_build_joint_horizon_prompt_state_retry_pcvp(self, joint_sample: Dict[str, Any]) -> Dict[str, Any]:
    base = _jha_previous_build_joint_horizon_prompt_state_for_retry_pcvp(self, joint_sample)
    if not isinstance(base, dict):
        base = {}
    inp = _jha_dict(joint_sample.get("input_state"))
    base["coupling_retry_fallback_contract"] = inp.get("coupling_retry_fallback_contract")
    base["coupling_task_contract"] = inp.get("coupling_task_contract")
    return base

def _jha_debug_joint_horizon_sample_report_retry_pcvp(self, sample: Dict[str, Any]) -> Dict[str, Any]:
    try:
        base = _jha_previous_debug_joint_horizon_sample_report_for_retry_pcvp(self, sample)
    except Exception:
        base = {}
    if not isinstance(base, dict):
        base = {}
    inp = _jha_dict(sample.get("input_state"))
    comp = _jha_dict(sample.get("component_samples"))
    base.update({
        "coupling_retry_fallback_contract_available": bool(inp.get("coupling_retry_fallback_contract")),
        "coupling_retry_enabled": bool(_jha_dict(inp.get("coupling_retry_fallback_contract")).get("retry_enabled")),
        "coupling_rule_fallback_after_retry_enabled": bool(_jha_dict(inp.get("coupling_retry_fallback_contract")).get("rule_fallback_after_retry_enabled")),
        "component_samples_production_available": bool(comp.get("production")),
        "component_samples_energy_available": bool(comp.get("energy")),
    })
    return base

JointStateAssembler.build_joint_horizon_sample = _jha_build_joint_horizon_sample_retry_pcvp
JointStateAssembler.build_joint_horizon_prompt_state = _jha_build_joint_horizon_prompt_state_retry_pcvp
JointStateAssembler.debug_joint_horizon_sample_report = _jha_debug_joint_horizon_sample_report_retry_pcvp


def _jha_world_model_default_action() -> Dict[str, Any]:
    return {
        "action_id": "observed_noop",
        "production_action": {"type": "keep", "target": "production_schedule", "delta_min": 0},
        "energy_action": {"type": "keep", "target": "energy_system", "delta_level": "none"},
        "source": "world_model_training_default",
    }


def _jha_world_model_duration(item: Any, default: float = 1.0) -> float:
    item = _jha_dict(item)
    for key in ["duration_min", "dt", "delta_t"]:
        if item.get(key) is not None:
            try:
                return max(1e-6, float(item.get(key)))
            except Exception:
                pass
    t0 = item.get("t0_min") if item.get("t0_min") is not None else item.get("start_min")
    t1 = item.get("t1_min") if item.get("t1_min") is not None else item.get("end_min")
    if t0 is not None and t1 is not None:
        try:
            return max(1e-6, float(t1) - float(t0))
        except Exception:
            pass
    t = _jha_dict(item.get("t"))
    t0 = t.get("t0") if t.get("t0") is not None else t.get("s0")
    t1 = t.get("t1") if t.get("t1") is not None else t.get("s1")
    if t0 is not None and t1 is not None:
        try:
            return max(1e-6, float(t1) - float(t0))
        except Exception:
            pass
    return float(default)


def _jha_build_world_model_training_record(
    self,
    joint_sample: Dict[str, Any],
    candidate_action: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    sample = _jha_dict(joint_sample)
    comp = _jha_dict(sample.get("component_samples"))
    prod_sample = _jha_dict(comp.get("production"))
    energy_sample = _jha_dict(comp.get("energy"))
    prod_input = _jha_dict(prod_sample.get("input_state"))
    energy_input = _jha_dict(energy_sample.get("input_state"))
    target = _jha_dict(sample.get("target_future"))
    prod_target = _jha_dict(prod_sample.get("target_future"))
    energy_target = _jha_dict(energy_sample.get("target_future"))

    prod_history = _jha_list(prod_input.get("history_patches"))
    energy_history = _jha_list(energy_input.get("energy_history"))
    production_future = _jha_list(target.get("production_future") or prod_target.get("future_patches"))
    energy_future = _jha_list(target.get("energy_future") or energy_target.get("energy_future"))

    functional_target = _jha_dict(
        target.get("functional_coupling_target")
        or target.get("coupling_target")
        or sample.get("coupling_target")
    )
    coupling_future = _jha_list(functional_target.get("future_pairs"))

    history_window = int(sample.get("history_window") or max(len(prod_history), len(energy_history), 1))
    horizon = int(sample.get("horizon") or max(len(production_future), len(energy_future), 1))

    def pad_left(values, n):
        values = list(values[-n:])
        if len(values) < n:
            values = [{} for _ in range(n - len(values))] + values
        return values

    def pad_right(values, n):
        values = list(values[:n])
        if len(values) < n:
            values = values + [{} for _ in range(n - len(values))]
        return values

    prod_history = pad_left(prod_history, history_window)
    energy_history = pad_left(energy_history, history_window)
    production_future = pad_right(production_future, horizon)
    energy_future = pad_right(energy_future, horizon)
    coupling_future = pad_right(coupling_future, horizon)
    coupling_history = [
        {"production": prod_history[i], "energy": energy_history[i]}
        for i in range(history_window)
    ]

    history_delta_t = [
        _jha_world_model_duration(prod_history[i], _jha_world_model_duration(energy_history[i], 1.0))
        for i in range(history_window)
    ]
    future_delta_t = [
        _jha_world_model_duration(production_future[i], _jha_world_model_duration(energy_future[i], 1.0))
        for i in range(horizon)
    ]

    action = _jha_dict(candidate_action)
    if not action:
        action = _jha_dict(sample.get("candidate_action"))
    if not action:
        action = _jha_dict(_jha_dict(sample.get("input_state")).get("candidate_action"))
    if not action:
        action = _jha_world_model_default_action()

    return {
        "sample_id": sample.get("sample_id"),
        "day_id": sample.get("day_id"),
        "anchor_patch_idx": sample.get("anchor_patch_idx"),
        "history_window": history_window,
        "horizon": horizon,
        "production_history": prod_history,
        "energy_history": energy_history,
        "coupling_history": coupling_history,
        "production_future": production_future,
        "energy_future": energy_future,
        "coupling_future": coupling_future,
        "history_delta_t": history_delta_t,
        "future_delta_t": future_delta_t,
        "candidate_action": action,
    }


def _jha_build_world_model_training_samples(
    self,
    day_ids: List[str],
    history_window: int = 5,
    horizon: int = 5,
    stride: int = 1,
    max_samples_per_day: Optional[int] = None,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for day_id in day_ids:
        samples = self.build_joint_horizon_dataset(
            day_id=day_id,
            history_window=history_window,
            horizon=horizon,
            stride=stride,
            max_samples=max_samples_per_day,
        )
        for sample in samples:
            action = _jha_world_model_default_action()
            sample = _jha_copy(sample)
            sample["candidate_action"] = action
            inp = _jha_dict(sample.get("input_state"))
            inp["candidate_action"] = action
            sample["input_state"] = inp
            records.append(sample)
    return records


def _jha_common_day_ids(self) -> List[str]:
    prod_days = list(getattr(self.production_adapter, "day_to_obj", {}).keys())
    energy_days = set(getattr(self.energy_adapter, "day_to_heats", {}).keys())
    return [str(day) for day in prod_days if day in energy_days]


JointStateAssembler.build_world_model_training_record = _jha_build_world_model_training_record
JointStateAssembler.build_world_model_training_samples = _jha_build_world_model_training_samples
JointStateAssembler.get_common_day_ids = _jha_common_day_ids
