from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
from a_data_loader import DataLoader

class EnergyStateAdapter:
    def __init__(
        self,
        energy_data_path: str = "data_of_IES.xlsx",
        heat_json_path: str = "heat_segments.json",
        heat_excel_path: str = "heat_segments.xlsx",
        patch_metadata_json_path: str = "patch_metadata.json",
        patch_metadata_excel_path: str = "patch_metadata.xlsx",
        patches_dir: str = "patches",
        include_legend: bool = False,
    ) -> None:
        self.energy_loader = DataLoader()

        self.energy_data_path = energy_data_path
        self.heat_json_path = Path(heat_json_path)
        self.heat_excel_path = Path(heat_excel_path)
        self.patch_metadata_json_path = Path(patch_metadata_json_path)
        self.patch_metadata_excel_path = Path(patch_metadata_excel_path)
        self.patches_dir = Path(patches_dir)

        self.energy_df: Optional[pd.DataFrame] = None
        self.heat_df: Optional[pd.DataFrame] = None
        self.patch_meta_df: Optional[pd.DataFrame] = None

        self.heat_id_to_row: Dict[str, Dict[str, Any]] = {}
        self.heat_id_to_patch_meta: Dict[str, Dict[str, Any]] = {}
        self.day_to_heats: Dict[str, List[str]] = {}

        self.include_legend = include_legend

    def _safe_int(self, value: Any, default: int = 0) -> int:
        try:
            if value is None or pd.isna(value):
                return default
            return int(round(float(value)))
        except Exception:
            return default

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            if value is None or pd.isna(value):
                return default
            return float(value)
        except Exception:
            return default

    def _safe_str(self, value: Any) -> str:
        if value is None or pd.isna(value):
            return ""
        return str(value).strip()

    def _safe_list(self, value: Any) -> List[Any]:
        return value if isinstance(value, list) else []

    def _normalize_heat_id(self, value: Any) -> str:

        if value is None:
            return ""
        try:
            if pd.isna(value):
                return ""
        except Exception:
            pass

        s = str(value).strip()
        if not s:
            return ""
        s = s.replace("-", "_")

        parts = s.split("_")
        for i in range(len(parts) - 1):
            if parts[i].endswith("LDG"):
                return f"{parts[i]}_{parts[i + 1]}"

        return s

    def _normalize_heat_id_list(self, value: Any) -> List[str]:
        out: List[str] = []
        for x in self._normalize_list_like(value):
            hid = self._normalize_heat_id(x)
            if hid and hid not in out:
                out.append(hid)
        return out

    def _json_load(self, path: Path) -> Any:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _normalize_list_like(self, value: Any) -> List[str]:
        if isinstance(value, list):
            return [self._safe_str(x) for x in value if self._safe_str(x)]
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return []
        s = self._safe_str(value)
        if not s:
            return []
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return [self._safe_str(x) for x in obj if self._safe_str(x)]
        except Exception:
            pass
        return [x.strip() for x in s.split(";") if x.strip()]

    def _parse_summary_dict(self, value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return value
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return {}
        s = self._safe_str(value)
        if not s:
            return {}
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        return {}

    def _json_len(self, obj: Any) -> int:
        return len(json.dumps(obj, ensure_ascii=False, default=str, separators=(",", ":")))

    def _series_stats_short(self, series: List[float]) -> Dict[str, float]:
        if not series:
            return {"s": 0.0, "e": 0.0, "mn": 0.0, "mx": 0.0, "m": 0.0, "de": 0.0}
        start = float(series[0])
        end = float(series[-1])
        min_v = float(min(series))
        max_v = float(max(series))
        mean_v = float(sum(series) / len(series))
        return {
            "s": round(start, 4),
            "e": round(end, 4),
            "mn": round(min_v, 4),
            "mx": round(max_v, 4),
            "m": round(mean_v, 4),
            "de": round(end - start, 4),
        }

    def _integral_total_from_hourly_rate(self, series: List[float]) -> float:
        if not series:
            return 0.0
        return round(float(sum(series) / 60.0), 4)

    def load(self, sample_size: Optional[int] = None) -> None:
        self.energy_df = self.energy_loader.load_and_preprocess_data(
            self.energy_data_path,
            sample_size=sample_size
        )
        self.energy_df = self.energy_df.sort_values("MinuteIndex").reset_index(drop=True)

        self.heat_df = self._load_heat_segments()
        self.patch_meta_df = self._load_patch_metadata()
        self._build_indexes()

    def _load_heat_segments(self) -> pd.DataFrame:
        if self.heat_json_path.exists():
            records = self._json_load(self.heat_json_path)
            df = pd.DataFrame(records)
        elif self.heat_excel_path.exists():
            df = pd.read_excel(self.heat_excel_path)
        else:
            raise FileNotFoundError("Could not find heat_segments.json or heat_segments.xlsx")

        if df.empty:
            raise ValueError("The heat segments file is empty")

        int_cols = [
            "segment_start_minute_index", "segment_end_minute_index",
            "patch_start_minute_index", "patch_end_minute_index",
            "duration_minutes_raw", "patch_duration_minutes",
            "day_index", "global_heat_id", "local_heat_id", "overlap_heat_count",
        ]
        for col in int_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

        if "is_overlapped" in df.columns:
            df["is_overlapped"] = df["is_overlapped"].astype(bool)

        for col in ["overlap_heat_ids", "overlap_converter_names"]:
            if col in df.columns:
                df[col] = df[col].apply(self._normalize_list_like)

        for col in ["start_time", "end_time", "bof_start_time", "bof_end_time"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        sort_cols = [c for c in ["segment_start_minute_index", "converter_name", "production_heat_no"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(by=sort_cols).reset_index(drop=True)

        return df

    def _load_patch_metadata(self) -> pd.DataFrame:
        if self.patch_metadata_json_path.exists():
            records = self._json_load(self.patch_metadata_json_path)
            df = pd.DataFrame(records)
        elif self.patch_metadata_excel_path.exists():
            df = pd.read_excel(self.patch_metadata_excel_path)
        else:
            raise FileNotFoundError("Could not find patch_metadata.json or patch_metadata.xlsx")

        if df.empty:
            raise ValueError("The patch metadata file is empty")

        int_cols = [
            "segment_start_minute_index", "segment_end_minute_index",
            "patch_start_minute_index", "patch_end_minute_index",
            "duration_minutes_raw", "patch_duration_minutes",
            "day_index", "global_heat_id", "local_heat_id", "overlap_heat_count",
        ]
        for col in int_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

        if "is_overlapped" in df.columns:
            df["is_overlapped"] = df["is_overlapped"].astype(bool)

        for col in ["overlap_heat_ids", "overlap_converter_names"]:
            if col in df.columns:
                df[col] = df[col].apply(self._normalize_list_like)

        if "summary" in df.columns:
            df["summary"] = df["summary"].apply(self._parse_summary_dict)

        return df

    def _build_indexes(self) -> None:
        assert self.heat_df is not None
        assert self.patch_meta_df is not None

        self.heat_id_to_row.clear()
        self.heat_id_to_patch_meta.clear()
        self.day_to_heats.clear()

        for _, row in self.heat_df.iterrows():
            row_dict = row.to_dict()
            heat_id = self._safe_str(row_dict.get("heat_id"))
            if not heat_id:
                continue

            self.heat_id_to_row[heat_id] = row_dict

            day_index = self._safe_int(row_dict.get("day_index", 0), 0)
            day_id = self._infer_day_id_from_day_index(day_index)

            self.day_to_heats.setdefault(day_id, [])
            self.day_to_heats[day_id].append(heat_id)

        for _, row in self.patch_meta_df.iterrows():
            row_dict = row.to_dict()
            heat_id = self._safe_str(row_dict.get("heat_id"))
            patch_id = self._safe_str(row_dict.get("patch_id"))

            if heat_id:
                self.heat_id_to_patch_meta[heat_id] = row_dict
            if patch_id and patch_id not in self.heat_id_to_patch_meta:
                self.heat_id_to_patch_meta[patch_id] = row_dict

    def _infer_day_id_from_day_index(self, day_index: int) -> str:
        if self.energy_df is None:
            return f"DayIndex_{day_index}"

        sub = self.energy_df[self.energy_df["DayIndex"] == day_index]
        if not sub.empty and "Time" in sub.columns:
            ts = sub["Time"].iloc[0]
            if pd.notna(ts):
                return pd.Timestamp(ts).strftime("%Y-%m-%d")
        return f"DayIndex_{day_index}"

    def _load_patch_detail_df(self, patch_meta: Dict[str, Any], patch_id: str) -> pd.DataFrame:
        json_file = self._safe_str(patch_meta.get("json_file", ""))
        csv_file = self._safe_str(patch_meta.get("csv_file", ""))

        json_path = Path(json_file) if json_file else None
        csv_path = Path(csv_file) if csv_file else None

        if json_path and json_path.exists():
            patch_json = self._json_load(json_path)
            if isinstance(patch_json, dict):
                return pd.DataFrame(patch_json.get("data", []))
            if isinstance(patch_json, list):
                return pd.DataFrame(patch_json)

        if csv_path and csv_path.exists():
            return pd.read_csv(csv_path)

        fallback_json = self.patches_dir / f"{patch_id}.json"
        fallback_csv = self.patches_dir / f"{patch_id}.csv"

        if fallback_json.exists():
            patch_json = self._json_load(fallback_json)
            if isinstance(patch_json, dict):
                return pd.DataFrame(patch_json.get("data", []))
            if isinstance(patch_json, list):
                return pd.DataFrame(patch_json)

        if fallback_csv.exists():
            return pd.read_csv(fallback_csv)

        raise FileNotFoundError(f"Could not find the patch detail file: patch_id={patch_id}")

    def _build_patch_state_short(self, heat_id: str) -> Dict[str, Any]:
        if not self.heat_id_to_patch_meta:
            self.load()

        patch_meta = self.heat_id_to_patch_meta.get(heat_id)
        if not patch_meta:
            raise KeyError(f"heat_id={heat_id} not found in patch metadata")

        patch_id = self._safe_str(patch_meta.get("patch_id")) or heat_id
        patch_df = self._load_patch_detail_df(patch_meta=patch_meta, patch_id=patch_id)
        if patch_df.empty:
            raise ValueError(f"patch detail is empty for heat_id={heat_id}, patch_id={patch_id}")

        if "MinuteIndex" in patch_df.columns:
            patch_df["MinuteIndex"] = pd.to_numeric(patch_df["MinuteIndex"], errors="coerce").fillna(0).astype(int)
            patch_df = patch_df.sort_values("MinuteIndex").reset_index(drop=True)

        duration_minutes = self._safe_int(
            patch_meta.get("patch_duration_minutes", len(patch_df)),
            len(patch_df)
        )
        duration_minutes = max(duration_minutes, 1)

        summary = patch_meta.get("summary", {})
        if not isinstance(summary, dict):
            summary = {}

        bfg_gen = (
            pd.to_numeric(patch_df.get("1#blast furnace Generation (NKM³/h)", 0.0), errors="coerce").fillna(0.0)
            + pd.to_numeric(patch_df.get("2#blast furnace Generation (NKM³/h)", 0.0), errors="coerce").fillna(0.0)
        ).tolist()

        cog_gen = (
            pd.to_numeric(patch_df.get("1#coke oven Generation (NKM³/h)", 0.0), errors="coerce").fillna(0.0)
            + pd.to_numeric(patch_df.get("2#coke oven Generation (NKM³/h)", 0.0), errors="coerce").fillna(0.0)
        ).tolist()

        ldg_gen = (
            pd.to_numeric(patch_df.get("1#LDG Generation (NKM³/h)", 0.0), errors="coerce").fillna(0.0)
            + pd.to_numeric(patch_df.get("2#LDG Generation (NKM³/h)", 0.0), errors="coerce").fillna(0.0)
            + pd.to_numeric(patch_df.get("3#LDG Generation (NKM³/h)", 0.0), errors="coerce").fillna(0.0)
            + pd.to_numeric(patch_df.get("4#LDG Generation (NKM³/h)", 0.0), errors="coerce").fillna(0.0)
        ).tolist()

        bfg_holder = pd.to_numeric(patch_df.get("BFG holder (NKM³)", 0.0), errors="coerce").fillna(0.0).tolist()
        cog_holder = pd.to_numeric(patch_df.get("COG holder (NKM³)", 0.0), errors="coerce").fillna(0.0).tolist()
        ldg_holder = pd.to_numeric(patch_df.get("LDG holder (NKM³)", 0.0), errors="coerce").fillna(0.0).tolist()

        steam_load = pd.to_numeric(patch_df.get("Total_Steam_Load_tph", 0.0), errors="coerce").fillna(0.0).tolist()
        power_load = pd.to_numeric(patch_df.get("Electric Load (MW)", 0.0), errors="coerce").fillna(0.0).tolist()

        purchased_steam = pd.to_numeric(patch_df.get("Purchased_Steam_tph", 0.0), errors="coerce").fillna(0.0).tolist()
        purchased_power = pd.to_numeric(patch_df.get("Purchased_Power(MW)", 0.0), errors="coerce").fillna(0.0).tolist()

        rk = self._build_patch_risk_flags(summary, patch_meta)

        production_heat_no = self._normalize_heat_id(
            patch_meta.get("production_heat_no")
            or patch_meta.get("heat_no")
            or patch_meta.get("heat_id")
            or patch_meta.get("patch_id")
            or heat_id
        )
        overlap_heat_ids_raw = patch_meta.get("overlap_heat_ids", [])
        overlap_heat_ids = overlap_heat_ids_raw if isinstance(overlap_heat_ids_raw, list) else []
        overlap_heat_nos = []
        for x in overlap_heat_ids:
            hx = self._normalize_heat_id(x)
            if hx and hx not in overlap_heat_nos:
                overlap_heat_nos.append(hx)

        return {
            "id": heat_id,
            "hn": production_heat_no,
            "p": patch_id,
            "sq": None,
            "t": {
                "ts": self._safe_str(patch_meta.get("start_time")),
                "te": self._safe_str(patch_meta.get("end_time")),
                "bs": self._safe_str(patch_meta.get("bof_start_time")),
                "be": self._safe_str(patch_meta.get("bof_end_time")),
                "s0": self._safe_int(patch_meta.get("segment_start_minute_index", 0), 0),
                "s1": self._safe_int(patch_meta.get("segment_end_minute_index", 0), 0),
                "t0": self._safe_int(patch_meta.get("patch_start_minute_index", 0), 0),
                "t1": self._safe_int(patch_meta.get("patch_end_minute_index", 0), 0),
                "dt": duration_minutes,
            },
            "g": {
                "b": {"a": round(sum(bfg_gen) / len(bfg_gen), 4) if bfg_gen else 0.0, "tot": self._integral_total_from_hourly_rate(bfg_gen)},
                "c": {"a": round(sum(cog_gen) / len(cog_gen), 4) if cog_gen else 0.0, "tot": self._integral_total_from_hourly_rate(cog_gen)},
                "l": {"a": round(sum(ldg_gen) / len(ldg_gen), 4) if ldg_gen else 0.0, "tot": self._integral_total_from_hourly_rate(ldg_gen)},
            },
            "st": {
                "b": self._series_stats_short(bfg_holder),
                "c": self._series_stats_short(cog_holder),
                "l": self._series_stats_short(ldg_holder),
            },
            "dm": {
                "s": {
                    "a": round(sum(steam_load) / len(steam_load), 4) if steam_load else 0.0,
                    "tot": self._integral_total_from_hourly_rate(steam_load),
                    "pk": round(max(steam_load), 4) if steam_load else 0.0,
                },
                "e": {
                    "a": round(sum(power_load) / len(power_load), 4) if power_load else 0.0,
                    "tot": self._integral_total_from_hourly_rate(power_load),
                    "pk": round(max(power_load), 4) if power_load else 0.0,
                },
            },
            "ex": {
                "s": {"a": round(sum(purchased_steam) / len(purchased_steam), 4) if purchased_steam else 0.0, "tot": self._integral_total_from_hourly_rate(purchased_steam)},
                "e": {"a": round(sum(purchased_power) / len(purchased_power), 4) if purchased_power else 0.0, "tot": self._integral_total_from_hourly_rate(purchased_power)},
            },
            "pc": {
                "mc": self._safe_str(patch_meta.get("main_converter_name")),
                "phn": production_heat_no,
                "ov": bool(patch_meta.get("is_overlapped", False)),
                "oc": self._safe_int(patch_meta.get("overlap_heat_count", 0), 0),
                "oh": overlap_heat_ids,
                "ohn": overlap_heat_nos,
                "ocn": patch_meta.get("overlap_converter_names", []) if isinstance(patch_meta.get("overlap_converter_names", []), list) else [],
                "ph": None,
                "nh": None,
            },
            "rk": rk,
        }

    def _build_patch_risk_flags(self, summary: Dict[str, Any], patch_meta: Dict[str, Any]) -> List[str]:
        rk: List[str] = []

        if self._safe_float(summary.get("bfg_holder_min", 9999.0), 9999.0) < 60:
            rk.append("b_low")
        if self._safe_float(summary.get("bfg_holder_max", -1.0), -1.0) > 280:
            rk.append("b_high")

        if self._safe_float(summary.get("cog_holder_min", 9999.0), 9999.0) < 30:
            rk.append("c_low")
        if self._safe_float(summary.get("cog_holder_max", -1.0), -1.0) > 120:
            rk.append("c_high")

        if self._safe_float(summary.get("ldg_holder_min", 9999.0), 9999.0) < 30:
            rk.append("l_low")
        if self._safe_float(summary.get("ldg_holder_max", -1.0), -1.0) > 320:
            rk.append("l_high")

        if self._safe_float(summary.get("purchased_power_total", 0.0), 0.0) > 0:
            rk.append("pp")
        if self._safe_float(summary.get("purchased_steam_total", 0.0), 0.0) > 0:
            rk.append("ps")

        if bool(patch_meta.get("is_overlapped", False)):
            rk.append("ov")

        return sorted(set(rk))

    def _build_summary_short(self, patch_states: List[Dict[str, Any]], day_id: str) -> Dict[str, Any]:
        num_heats = len(patch_states)
        num_overlapped = sum(1 for x in patch_states if x.get("pc", {}).get("ov", False))
        num_risky = sum(1 for x in patch_states if len(x.get("rk", [])) > 0)
        num_pp = sum(1 for x in patch_states if "pp" in (x.get("rk", []) or []))
        num_ps = sum(1 for x in patch_states if "ps" in (x.get("rk", []) or []))

        starts = [x.get("t", {}).get("t0") for x in patch_states if x.get("t", {}).get("t0") is not None]
        ends = [x.get("t", {}).get("t1") for x in patch_states if x.get("t", {}).get("t1") is not None]

        return {
            "d": day_id,
            "nh": num_heats,
            "nov": num_overlapped,
            "nrk": num_risky,
            "npp": num_pp,
            "nps": num_ps,
            "t0": min(starts) if starts else None,
            "t1": max(ends) if ends else None,
        }

    def _build_constraint_features(self) -> Dict[str, Any]:
        return {
            "med": ["BFG", "COG", "LDG", "Steam", "Power"],
            "risk": {
                "b": [60, 280],
                "c": [30, 120],
                "l": [30, 320],
            },
            "obj": [
                "predict_generation_storage_demand_purchase",
                "predict_holder_limit_and_purchase_risk",
                "predict_energy_response_to_operation_schedule",
            ],
            "act": [],
        }

    def _build_patch_index(self, patch_states: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for x in patch_states:
            out.append({
                "id": x.get("id"),
                "hn": self._normalize_heat_id(x.get("hn") or self._safe_dict(x.get("pc")).get("phn") or x.get("id")),
                "sq": x.get("sq"),
                "t0": x.get("t", {}).get("t0"),
                "t1": x.get("t", {}).get("t1"),
                "rk": x.get("rk", []),
                "ov": x.get("pc", {}).get("ov", False),
            })
        return out

    def _compact_patch_for_prompt(self, p: Dict[str, Any], mode: str) -> Dict[str, Any]:
        if mode == "energy":
            return {
                "id": p.get("id"),
                "hn": p.get("hn"),
                "p": p.get("p"),
                "sq": p.get("sq"),
                "t": {
                    "ts": p.get("t", {}).get("ts"),
                    "te": p.get("t", {}).get("te"),
                    "t0": p.get("t", {}).get("t0"),
                    "t1": p.get("t", {}).get("t1"),
                    "dt": p.get("t", {}).get("dt"),
                },
                "g": p.get("g", {}),
                "st": p.get("st", {}),
                "dm": p.get("dm", {}),
                "ex": p.get("ex", {}),
                "pc": {
                    "mc": p.get("pc", {}).get("mc"),
                    "ov": p.get("pc", {}).get("ov"),
                    "oc": p.get("pc", {}).get("oc"),
                    "ph": p.get("pc", {}).get("ph"),
                    "nh": p.get("pc", {}).get("nh"),
                },
                "rk": p.get("rk", []),
                "mk": p.get("mk", {}),
            }

        if mode == "coupling":
            mk = p.get("mk", {}) if isinstance(p.get("mk"), dict) else {}
            return {
                "id": p.get("id"),
                "hn": p.get("hn"),
                "sq": p.get("sq"),
                "t": {
                    "t0": p.get("t", {}).get("t0"),
                    "t1": p.get("t", {}).get("t1"),
                    "dt": p.get("t", {}).get("dt"),
                },
                "g": {
                    "b": p.get("g", {}).get("b", {}),
                    "l": p.get("g", {}).get("l", {}),
                },
                "st": p.get("st", {}),
                "dm": p.get("dm", {}),
                "ex": p.get("ex", {}),
                "pc": {
                    "mc": p.get("pc", {}).get("mc"),
                    "ov": p.get("pc", {}).get("ov"),
                    "oc": p.get("pc", {}).get("oc"),
                    "ph": p.get("pc", {}).get("ph"),
                    "nh": p.get("pc", {}).get("nh"),
                },
                "rk": p.get("rk", []),
                "mk": {
                    "ed": mk.get("ed"),
                    "pl": mk.get("pl"),
                    "lv": mk.get("lv"),
                },
            }

        return p

    def _build_prompt_views(self, full_energy_state: Dict[str, Any]) -> Dict[str, Any]:
        patch_states = full_energy_state.get("ps", [])

        energy_ps = [self._compact_patch_for_prompt(x, "energy") for x in patch_states]
        coupling_ps = [self._compact_patch_for_prompt(x, "coupling") for x in patch_states]

        return {
            "energy": {
                "d": full_energy_state.get("d"),
                "m": "energy",
                "s": full_energy_state.get("s", {}),
                "cf": full_energy_state.get("cf", {}),
                "pi": full_energy_state.get("pi", []),
                "ps": energy_ps,
                "meta": full_energy_state.get("meta", {}),
            },
            "coupling": {
                "d": full_energy_state.get("d"),
                "m": "coupling",
                "s": full_energy_state.get("s", {}),
                "cf": full_energy_state.get("cf", {}),
                "pi": full_energy_state.get("pi", []),
                "ps": coupling_ps,
                "meta": full_energy_state.get("meta", {}),
            }
        }

    def build_energy_state(self, day_id: str) -> Dict[str, Any]:

        if not self.heat_id_to_patch_meta:
            self.load()

        heat_ids = list(self.day_to_heats.get(day_id, []))
        if not heat_ids:
            raise KeyError(f"day_id={day_id} not found in energy patch metadata")

        patch_states: List[Dict[str, Any]] = []
        skipped: List[str] = []

        for heat_id in heat_ids:
            try:
                ps = self._build_patch_state_short(heat_id)
                patch_states.append(ps)
            except Exception:
                skipped.append(heat_id)

        if not patch_states:
            raise ValueError(f"no valid patch states for day_id={day_id}")

        patch_states = sorted(
            patch_states,
            key=lambda x: (
                self._safe_int(x.get("t", {}).get("t0", 10**9), 10**9),
                self._safe_str(x.get("id")),
            )
        )

        for i, p in enumerate(patch_states, start=1):
            p["sq"] = i

        for i, p in enumerate(patch_states):
            prev_id = patch_states[i - 1]["id"] if i > 0 else None
            next_id = patch_states[i + 1]["id"] if i < len(patch_states) - 1 else None
            if "pc" not in p or not isinstance(p["pc"], dict):
                p["pc"] = {}
            p["pc"]["ph"] = prev_id
            p["pc"]["nh"] = next_id

        def _q(values: List[float], q: float, default: float = 0.0) -> float:
            if not values:
                return default
            vals = sorted(values)
            idx = int(round((len(vals) - 1) * q))
            idx = max(0, min(idx, len(vals) - 1))
            return float(vals[idx])

        def _level_by_threshold(v: float, low: float, high: float) -> str:
            if v >= high:
                return "high"
            if v >= low:
                return "med"
            return "low"

        steam_pks = [self._safe_float(p.get("dm", {}).get("s", {}).get("pk"), 0.0) for p in patch_states]
        power_pks = [self._safe_float(p.get("dm", {}).get("e", {}).get("pk"), 0.0) for p in patch_states]

        steam_pk_med = _q(steam_pks, 0.5, 0.0)
        steam_pk_hi = _q(steam_pks, 0.75, 0.0)
        power_pk_med = _q(power_pks, 0.5, 0.0)
        power_pk_hi = _q(power_pks, 0.75, 0.0)

        holder_tight_count = 0
        high_dep_count = 0
        high_peak_count = 0
        mech_level_hist = {"normal": 0, "tight": 0, "stressed": 0}

        for p in patch_states:
            st = p.get("st", {}) or {}
            dm = p.get("dm", {}) or {}
            ex = p.get("ex", {}) or {}
            pc = p.get("pc", {}) or {}

            holder_margin = {}
            holder_flags = []

            holder_cfg = {
                "b": (60, 280, 20),
                "c": (30, 120, 10),
                "l": (30, 320, 20),
            }

            for gas, (low_b, high_b, warn_gap) in holder_cfg.items():
                stat = st.get(gas, {}) or {}
                mn = self._safe_float(stat.get("mn"), 0.0)
                mx = self._safe_float(stat.get("mx"), 0.0)

                lo_margin = round(mn - low_b, 4)
                hi_margin = round(high_b - mx, 4)

                if lo_margin < 0 or hi_margin < 0:
                    lv = "breach"
                elif min(lo_margin, hi_margin) < warn_gap:
                    lv = "tight"
                else:
                    lv = "normal"

                holder_margin[gas] = {
                    "lo": lo_margin,
                    "hi": hi_margin,
                    "lv": lv,
                }
                holder_flags.append(lv)

            steam_tot = self._safe_float(dm.get("s", {}).get("tot"), 0.0)
            power_tot = self._safe_float(dm.get("e", {}).get("tot"), 0.0)
            steam_ex = self._safe_float(ex.get("s", {}).get("tot"), 0.0)
            power_ex = self._safe_float(ex.get("e", {}).get("tot"), 0.0)

            steam_dep = round(steam_ex / max(steam_tot, 1e-6), 4) if steam_tot > 0 else 0.0
            power_dep = round(power_ex / max(power_tot, 1e-6), 4) if power_tot > 0 else 0.0
            dep_score = max(steam_dep, power_dep)

            if dep_score >= 0.5:
                dep_level = "high"
            elif dep_score >= 0.2:
                dep_level = "med"
            else:
                dep_level = "low"

            steam_pk = self._safe_float(dm.get("s", {}).get("pk"), 0.0)
            power_pk = self._safe_float(dm.get("e", {}).get("pk"), 0.0)
            steam_peak_level = _level_by_threshold(steam_pk, steam_pk_med, steam_pk_hi)
            power_peak_level = _level_by_threshold(power_pk, power_pk_med, power_pk_hi)

            storage_trend = {
                "b": "up" if self._safe_float(st.get("b", {}).get("de"), 0.0) > 0 else ("down" if self._safe_float(st.get("b", {}).get("de"), 0.0) < 0 else "flat"),
                "c": "up" if self._safe_float(st.get("c", {}).get("de"), 0.0) > 0 else ("down" if self._safe_float(st.get("c", {}).get("de"), 0.0) < 0 else "flat"),
                "l": "up" if self._safe_float(st.get("l", {}).get("de"), 0.0) > 0 else ("down" if self._safe_float(st.get("l", {}).get("de"), 0.0) < 0 else "flat"),
            }

            patch_level = "normal"
            if "breach" in holder_flags or dep_level == "high":
                patch_level = "stressed"
            elif "tight" in holder_flags or dep_level == "med" or bool(pc.get("ov", False)):
                patch_level = "tight"

            p["mk"] = {
                "hm": holder_margin,
                "ed": {
                    "s": steam_dep,
                    "e": power_dep,
                    "lv": dep_level,
                },
                "pl": {
                    "s": steam_peak_level,
                    "e": power_peak_level,
                },
                "tr": storage_trend,
                "lv": patch_level,
            }

            if any(x["lv"] in {"tight", "breach"} for x in holder_margin.values()):
                holder_tight_count += 1
            if dep_level == "high":
                high_dep_count += 1
            if steam_peak_level == "high" or power_peak_level == "high":
                high_peak_count += 1
            mech_level_hist[patch_level] = mech_level_hist.get(patch_level, 0) + 1

        patch_index = self._build_patch_index(patch_states)
        summary = self._build_summary_short(patch_states, day_id)

        has_time = 0
        has_generation = 0
        has_storage = 0
        has_demand = 0
        has_external = 0
        has_risk = 0
        has_mech = 0
        overlapped = 0

        risk_hist: Dict[str, int] = {}
        duration_list: List[int] = []

        for p in patch_states:
            t = p.get("t", {}) or {}
            g = p.get("g", {}) or {}
            st = p.get("st", {}) or {}
            dm = p.get("dm", {}) or {}
            ex = p.get("ex", {}) or {}
            rk = p.get("rk", []) or []
            pc = p.get("pc", {}) or {}

            if t.get("t0") is not None and t.get("t1") is not None:
                has_time += 1
            if g:
                has_generation += 1
            if st:
                has_storage += 1
            if dm:
                has_demand += 1
            if ex:
                has_external += 1
            if rk:
                has_risk += 1
            if p.get("mk"):
                has_mech += 1
            if pc.get("ov", False):
                overlapped += 1

            dt = self._safe_int(t.get("dt", 0), 0)
            if dt > 0:
                duration_list.append(dt)

            for r in rk:
                risk_hist[r] = risk_hist.get(r, 0) + 1

        state = {
            "d": day_id,
            "s": summary,
            "cf": self._build_constraint_features(),
            "ps": patch_states,
            "pi": patch_index,
            "meta": {
                "nh": len(patch_states),
                "nskip": len(skipped),
                "skipped_heat_ids": skipped[:20],
            },

            "mk": {
                "holder_tight_patches": holder_tight_count,
                "high_external_dependency_patches": high_dep_count,
                "high_peak_patches": high_peak_count,
                "level_hist": mech_level_hist,
            },

            "cov": {
                "patch_fields": {
                    "t": has_time,
                    "g": has_generation,
                    "st": has_storage,
                    "dm": has_demand,
                    "ex": has_external,
                    "rk": has_risk,
                    "mk": has_mech,
                },
                "risk_hist": risk_hist,
                "overlap": {
                    "nov": overlapped,
                    "ratio": round(overlapped / max(len(patch_states), 1), 4),
                },
                "duration": {
                    "min": min(duration_list) if duration_list else 0,
                    "max": max(duration_list) if duration_list else 0,
                    "avg": round(sum(duration_list) / max(len(duration_list), 1), 4) if duration_list else 0.0,
                },
            },
        }

        if self.include_legend:
            state["legend"] = {
                "d": "day_id",
                "s": "summary",
                "cf": "constraint_features",
                "ps": "patch_states",
                "pi": "patch_index",
                "mk": "mechanism_informed_features",
                "cov": "coverage_report",
                "id": "energy_patch_heat_id",
                "hn": "production_heat_no",
                "p": "patch_id",
                "sq": "patch_sequence_in_day",
                "t": "time_info",
                "g": "generation",
                "st": "storage",
                "dm": "demand",
                "ex": "external_supply",
                "pc": "patch_context",
                "rk": "risk_flags",
                "mk.hm": "holder_margin",
                "mk.ed": "external_dependency",
                "mk.pl": "peak_load_proxy",
                "mk.tr": "storage_trend",
                "mk.lv": "mechanism_stress_level",
            }

        return state

    def build_prompt_state(self, day_id: str, mode: str = "energy") -> Dict[str, Any]:
        full_state = self.build_energy_state(day_id)
        views = self._build_prompt_views(full_state)

        mode_alias = {
            "energy": "energy",
            "coupling": "coupling",
        }
        normalized_mode = mode_alias.get(mode)
        if normalized_mode is None or normalized_mode not in views:
            raise ValueError(f"Unsupported mode: {mode}")

        return views[normalized_mode]

    def build_energy_global_focus(self, day_id: str, full_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        state = full_state if isinstance(full_state, dict) else self.build_energy_state(day_id)
        ps = self._safe_list(state.get("ps"))

        risky_patches = []
        overlap_patches = []
        purchase_active_patches = []

        for p in ps:
            if not isinstance(p, dict):
                continue
            rk = self._safe_list(p.get("rk"))
            pc = p.get("pc", {}) if isinstance(p.get("pc"), dict) else {}
            item = {
                "id": p.get("id"),
                "sq": p.get("sq"),
                "rk": rk,
                "ov": bool(pc.get("ov", False)),
                "oc": pc.get("oc", 0),
            }

            if rk:
                risky_patches.append(item)
            if bool(pc.get("ov", False)):
                overlap_patches.append(item)
            if "pp" in rk or "ps" in rk:
                purchase_active_patches.append(item)

        risky_patches = sorted(risky_patches, key=lambda x: len(x.get("rk", [])), reverse=True)[:12]
        overlap_patches = sorted(overlap_patches, key=lambda x: (x.get("oc", 0), len(x.get("rk", []))), reverse=True)[:12]
        purchase_active_patches = sorted(purchase_active_patches, key=lambda x: len(x.get("rk", [])), reverse=True)[:12]

        return {
            "d": day_id,
            "summary": state.get("s", {}),
            "constraints": state.get("cf", {}),
            "patch_index": state.get("pi", []),
            "top_risky_patches": risky_patches,
            "overlap_patches": overlap_patches,
            "purchase_active_patches": purchase_active_patches,
        }

    def build_empty_rolling_memory(self, day_id: str) -> Dict[str, Any]:
        return {
            "d": day_id,
            "processed_batches": [],
            "processed_core_ids": [],
            "stable_patch_adjustments": [],
            "open_risks": [],
            "boundary_state": {
                "last_core_heat_id": None,
                "tail_holder_bias": {},
                "tail_purchase_bias": {},
            },
            "running_totals": {
                "purchase_power_trend": "flat",
                "purchase_steam_trend": "flat",
                "num_adjusted_patches": 0,
            },
        }

    def _select_pi_by_ids(self, pi: List[Dict[str, Any]], id_set: set) -> List[Dict[str, Any]]:
        return [x for x in pi if isinstance(x, dict) and x.get("id") in id_set]

    def _slice_core_context_ranges(
        self,
        n: int,
        core_start_idx: int,
        core_size: int,
        left_context: int,
        right_context: int,
    ) -> Tuple[int, int, int, int]:
        core_l = max(core_start_idx, 0)
        core_r = min(core_start_idx + core_size, n)
        ctx_l = max(0, core_l - left_context)
        ctx_r = min(n, core_r + right_context)
        return core_l, core_r, ctx_l, ctx_r

    def build_energy_batch_state(
        self,
        day_id: str,
        core_start_idx: int,
        core_size: int = 6,
        left_context: int = 1,
        right_context: int = 1,
        mode: str = "energy",
        full_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        state = full_state if isinstance(full_state, dict) else self.build_energy_state(day_id)
        ps_full = self._safe_list(state.get("ps"))
        pi_full = self._safe_list(state.get("pi"))
        n = len(ps_full)

        core_l, core_r, ctx_l, ctx_r = self._slice_core_context_ranges(
            n=n,
            core_start_idx=core_start_idx,
            core_size=core_size,
            left_context=left_context,
            right_context=right_context,
        )

        core_ps_full = ps_full[core_l:core_r]
        context_ps_full = ps_full[ctx_l:ctx_r]
        context_id_set = {x.get("id") for x in context_ps_full if isinstance(x, dict) and x.get("id")}
        core_ids = [x.get("id") for x in core_ps_full if isinstance(x, dict) and x.get("id")]
        context_ids = [x.get("id") for x in context_ps_full if isinstance(x, dict) and x.get("id")]
        context_pi = self._select_pi_by_ids(pi_full, context_id_set)

        compacted_ps = [self._compact_patch_for_prompt(x, mode) for x in context_ps_full]

        core_id_set = set(core_ids)
        for item in compacted_ps:
            item["_is_core"] = item.get("id") in core_id_set

        return {
            "d": day_id,
            "m": f"{mode}_batch",
            "s": state.get("s", {}),
            "cf": state.get("cf", {}),
            "pi": context_pi,
            "ps": compacted_ps,
            "meta": {
                **(state.get("meta", {}) if isinstance(state.get("meta"), dict) else {}),
                "batch": True,
                "core_start_idx0": core_l,
                "core_end_idx0_exclusive": core_r,
                "context_start_idx0": ctx_l,
                "context_end_idx0_exclusive": ctx_r,
                "core_sq_range": [core_l + 1, core_r],
                "context_sq_range": [ctx_l + 1, ctx_r],
                "core_ids": core_ids,
                "context_ids": context_ids,
                "core_size": core_size,
                "left_context": left_context,
                "right_context": right_context,
                "total_patches_in_day": n,
            },
        }

    def iter_energy_batch_states(
        self,
        day_id: str,
        core_size: int = 6,
        left_context: int = 1,
        right_context: int = 1,
        stride: Optional[int] = None,
        mode: str = "energy",
        full_state: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        state = full_state if isinstance(full_state, dict) else self.build_energy_state(day_id)
        ps_full = self._safe_list(state.get("ps"))
        n = len(ps_full)

        if stride is None or stride <= 0:
            stride = core_size

        out = []
        batch_id = 0
        start = 0
        while start < n:
            batch = self.build_energy_batch_state(
                day_id=day_id,
                core_start_idx=start,
                core_size=core_size,
                left_context=left_context,
                right_context=right_context,
                mode=mode,
                full_state=state,
            )
            batch["meta"]["batch_id"] = batch_id
            out.append(batch)
            batch_id += 1
            start += stride
        return out

    def build_transition_index(self, day_id: str) -> Dict[str, Any]:
        state = self.build_energy_state(day_id)
        pm = {x["id"]: x for x in state.get("ps", []) if x.get("id")}
        im = {x["id"]: x for x in state.get("pi", []) if x.get("id")}
        return {
            "d": day_id,
            "pm": pm,
            "im": im,
        }

    def audit_energy_state(self, day_id: str, state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if state is None:
            state = self.build_energy_state(day_id)

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

        expected_keys = ["d", "s", "cf", "ps", "pi", "meta"]
        missing_keys = [k for k in expected_keys if k not in state]
        if missing_keys:
            add_error("top_keys", "full_energy_state is missing required top-level fields", {"missing": missing_keys})

        ps = state.get("ps", []) or []
        pi = state.get("pi", []) or []
        s = state.get("s", {}) or {}
        meta = state.get("meta", {}) or {}

        report["stats"]["counts"] = {
            "nps": len(ps),
            "npi": len(pi),
        }

        if "nh" in s and s["nh"] != len(ps):
            add_warning("summary_count", "s['nh'] does not match len(ps)", {"s_nh": s["nh"], "len_ps": len(ps)})

        if "nh" in meta and meta["nh"] != len(ps):
            add_warning("meta_count", "meta['nh'] does not match len(ps)", {"meta_nh": meta["nh"], "len_ps": len(ps)})

        required_patch_fields = ["id", "sq", "t", "g", "st", "dm", "ex", "pc", "rk"]
        patch_field_stats = {}
        for field in required_patch_fields:
            present = sum(1 for x in ps if x.get(field) not in [None, "", []])
            cov = present / len(ps) if ps else 0.0
            patch_field_stats[field] = {"present": present, "total": len(ps), "coverage": round(cov, 4)}
            if cov == 0:
                add_error("patch_field", f"critical patch field {field} is completely missing")
            elif cov < 0.9:
                add_warning("patch_field", f"critical patch field {field} has low coverage", patch_field_stats[field])

        report["checks"]["patch_field_coverage"] = patch_field_stats

        patch_ids = [x.get("id") for x in ps if x.get("id") is not None]
        dup_patch_ids = sorted(list({x for x in patch_ids if patch_ids.count(x) > 1}))
        if dup_patch_ids:
            add_error("id_uniqueness", "duplicate patch heat IDs exist", {"dup_patch_ids": dup_patch_ids[:20]})

        pi_ids = [x.get("id") for x in pi if x.get("id") is not None]
        missing_in_pi = sorted(list(set(patch_ids) - set(pi_ids)))
        extra_in_pi = sorted(list(set(pi_ids) - set(patch_ids)))

        report["checks"]["patch_index_coverage"] = {
            "missing_in_pi": missing_in_pi[:20],
            "extra_in_pi": extra_in_pi[:20]
        }

        if missing_in_pi:
            add_error("patch_index", "some patches are missing from pi", {"missing_in_pi": missing_in_pi[:20]})
        if extra_in_pi:
            add_warning("patch_index", "pi contains IDs that are absent from ps", {"extra_in_pi": extra_in_pi[:20]})

        prompt_checks = {}
        for mode in ["energy", "coupling"]:
            ps_mode = self.build_prompt_state(day_id, mode=mode)
            p_ps = ps_mode.get("ps", []) or []
            p_ids = [x.get("id") for x in p_ps if x.get("id")]
            bad_ids = [x for x in p_ids if x not in patch_ids]

            prompt_checks[mode] = {
                "num_prompt_ps": len(p_ps),
                "bad_ids": bad_ids[:20],
            }

            if bad_ids:
                add_error(f"prompt_{mode}", f"{mode} prompt ps is not a subset of the full state", {"bad_ids": bad_ids[:20]})

        report["checks"]["prompt_subset"] = prompt_checks

        report["stats"]["size"] = {
            "full_chars": len(json.dumps(state, ensure_ascii=False)),
            "energy_prompt_chars": len(json.dumps(self.build_prompt_state(day_id, "energy"), ensure_ascii=False)),
            "coupling_prompt_chars": len(json.dumps(self.build_prompt_state(day_id, "coupling"), ensure_ascii=False)),
        }

        return report

    def debug_size_report(self, day_id: str) -> Dict[str, int]:
        full_state = self.build_energy_state(day_id)
        energy_prompt_state = self.build_prompt_state(day_id, mode="energy")
        coupling_prompt_state = self.build_prompt_state(day_id, mode="coupling")
        batch_prompt_state = self.build_energy_batch_state(day_id=day_id, core_start_idx=0, core_size=6, left_context=1, right_context=1, mode="energy", full_state=full_state)

        report = {
            "full_energy_state_chars": len(json.dumps(full_state, ensure_ascii=False)),
            "energy_prompt_state_chars": len(json.dumps(energy_prompt_state, ensure_ascii=False)),
            "coupling_prompt_state_chars": len(json.dumps(coupling_prompt_state, ensure_ascii=False)),
            "batch_energy_prompt_state_chars": len(json.dumps(batch_prompt_state, ensure_ascii=False)),
        }

        print("\n=== EnergyStateAdapter Size Report ===")
        for k, v in report.items():
            print(f"{k}: {v}")

        return report

def _esa_horizon_copy_dict(obj):
    try:
        return json.loads(json.dumps(obj, ensure_ascii=False, default=str))
    except Exception:
        return dict(obj) if isinstance(obj, dict) else obj

def _esa_pick_patch_key(p):
    if not isinstance(p, dict):
        return ""
    return str(p.get("id") or p.get("p") or p.get("hn") or "").strip()

def _esa_compact_patch_known_context(self, p):

    pc = p.get("pc", {}) if isinstance(p.get("pc"), dict) else {}
    t = p.get("t", {}) if isinstance(p.get("t"), dict) else {}
    hn = self._normalize_heat_id(p.get("hn") or pc.get("phn") or p.get("id") or p.get("p"))
    return {
        "id": p.get("id"),
        "hn": hn,
        "p": p.get("p"),
        "sq": p.get("sq"),
        "t": {
            "ts": t.get("ts"),
            "te": t.get("te"),
            "t0": t.get("t0"),
            "t1": t.get("t1"),
            "dt": t.get("dt"),
        },
        "pc": {
            "mc": pc.get("mc"),
            "phn": self._normalize_heat_id(pc.get("phn") or hn),
            "ov": pc.get("ov"),
            "oc": pc.get("oc"),
            "oh": pc.get("oh", []),
            "ohn": self._normalize_heat_id_list(pc.get("ohn") or pc.get("oh", [])),
            "ocn": pc.get("ocn", []),
            "ph": pc.get("ph"),
            "nh": pc.get("nh"),
        },
    }

def _esa_compact_patch_observed(self, p):

    if hasattr(self, "_compact_patch_for_prompt"):
        try:
            return self._compact_patch_for_prompt(p, mode="energy")
        except Exception:
            pass

    return {
        "id": p.get("id"),
        "hn": self._normalize_heat_id(p.get("hn") or self._safe_dict(p.get("pc")).get("phn") or p.get("id") or p.get("p")),
        "p": p.get("p"),
        "sq": p.get("sq"),
        "t": p.get("t", {}),
        "g": p.get("g", {}),
        "st": p.get("st", {}),
        "dm": p.get("dm", {}),
        "ex": p.get("ex", {}),
        "pc": p.get("pc", {}),
        "rk": p.get("rk", []),
        "mk": p.get("mk", {}),
    }

def _esa_compact_patch_target(self, p):

    return {
        "id": p.get("id"),
        "hn": self._normalize_heat_id(p.get("hn") or self._safe_dict(p.get("pc")).get("phn") or p.get("id") or p.get("p")),
        "p": p.get("p"),
        "sq": p.get("sq"),
        "t": p.get("t", {}),
        "g": p.get("g", {}),
        "st": p.get("st", {}),
        "dm": p.get("dm", {}),
        "ex": p.get("ex", {}),
        "pc": p.get("pc", {}),
        "rk": p.get("rk", []),
        "mk": p.get("mk", {}),
    }

def _esa_build_energy_patch_sequence(self, day_id, full_state=None):

    state = full_state if isinstance(full_state, dict) else self.build_energy_state(day_id)
    patches = list(state.get("ps", []) or [])
    patches = sorted(
        patches,
        key=lambda x: (
            self._safe_int((x.get("t") or {}).get("t0", 10**9), 10**9),
            self._safe_int(x.get("sq", 10**9), 10**9),
            self._safe_str(x.get("id")),
        ),
    )

    out = []
    for i, p in enumerate(patches, start=1):
        q = _esa_horizon_copy_dict(p)
        if isinstance(q, dict):
            q["sq"] = i
        out.append(q)
    return out

def _esa_build_energy_horizon_sample(
    self,
    day_id,
    anchor_patch_idx,
    history_window=5,
    horizon=5,
    production_context=None,
    production_prediction=None,
    include_target=True,
    full_state=None,
):

    if history_window <= 0:
        raise ValueError("history_window must be positive")
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    sequence = self.build_energy_patch_sequence(day_id, full_state=full_state)
    n = len(sequence)
    i = int(anchor_patch_idx)

    if i < 0 or i >= n:
        raise IndexError(f"anchor_patch_idx={anchor_patch_idx} out of range for n={n}")
    if i - history_window + 1 < 0:
        raise ValueError(
            f"Not enough history: anchor={i}, history_window={history_window}. "
            f"Need anchor >= {history_window - 1}."
        )
    if i + horizon >= n:
        raise ValueError(
            f"Not enough future labels: anchor={i}, horizon={horizon}, n={n}. "
            f"Need anchor+horizon < n."
        )

    hist = sequence[i - history_window + 1 : i + 1]
    future = sequence[i + 1 : i + horizon + 1]

    input_state = {
        "task": "energy_patch_horizon_prediction",
        "task_role": "energy_horizon_prediction_only",
        "day_id": day_id,
        "anchor_patch_idx": i,
        "anchor_patch_id": sequence[i].get("id"),
        "history_window": history_window,
        "horizon": horizon,
        "energy_history": [self._compact_patch_observed(p) for p in hist],
        "future_patch_context": [self._compact_patch_known_context(p) for p in future],

        "production_operation_segment_context": production_context if production_context is not None else {},
        "production_operation_segment_summary": production_prediction if production_prediction is not None else {},
        "constraints": self._build_constraint_features(),
        "notes": [
            "future_patch_context contains only known heat/time/overlap context",
            "future energy values, future mechanism summaries and direct future risk labels are hidden from model input",
            "pc.ov in future_patch_context is a known structural overlap flag, not a predicted risk label",
            "production context is limited to operation-level Gantt segment summary",
            "do not perform production heat-patch alignment prediction in the energy model",
            "target_future is used only for offline evaluation",
        ],
    }

    sample = {
        "sample_id": f"{day_id}_energy_anchor_{i}_W{history_window}_K{horizon}",
        "day_id": day_id,
        "anchor_patch_idx": i,
        "history_window": history_window,
        "horizon": horizon,
        "input_state": input_state,
        "meta": {
            "num_day_patches": n,
            "history_sq_range": [hist[0].get("sq"), hist[-1].get("sq")],
            "future_sq_range": [future[0].get("sq"), future[-1].get("sq")],
            "history_patch_ids": [p.get("id") for p in hist],
            "future_patch_ids": [p.get("id") for p in future],
            "future_heat_ids": [self._normalize_heat_id(p.get("hn") or self._safe_dict(p.get("pc")).get("phn") or p.get("id")) for p in future],
        },
    }

    if include_target:
        sample["target_future"] = {
            "day_id": day_id,
            "anchor_patch_idx": i,
            "horizon": horizon,
            "energy_future": [self._compact_patch_target(p) for p in future],
        }

    return sample

def _esa_build_energy_horizon_dataset(
    self,
    day_id,
    history_window=5,
    horizon=5,
    stride=1,
    max_samples=None,
    production_context_provider=None,
    production_prediction_provider=None,
    include_target=True,
    full_state=None,
):

    sequence = self.build_energy_patch_sequence(day_id, full_state=full_state)
    n = len(sequence)
    samples = []
    start = history_window - 1
    end = n - horizon - 1

    if end < start:
        return []

    anchor = start
    while anchor <= end:
        future = sequence[anchor + 1 : anchor + horizon + 1]
        future_context = [self._compact_patch_known_context(p) for p in future]

        prod_ctx = {}
        prod_pred = {}
        if callable(production_context_provider):
            prod_ctx = production_context_provider(anchor, horizon, future_context) or {}
        if callable(production_prediction_provider):
            prod_pred = production_prediction_provider(anchor, horizon, future_context) or {}

        samples.append(
            self.build_energy_horizon_sample(
                day_id=day_id,
                anchor_patch_idx=anchor,
                history_window=history_window,
                horizon=horizon,
                production_context=prod_ctx,
                production_prediction=prod_pred,
                include_target=include_target,
                full_state={"ps": sequence, "d": day_id},
            )
        )
        if max_samples is not None and len(samples) >= int(max_samples):
            break
        anchor += max(1, int(stride))

    return samples

def _esa_build_energy_horizon_prompt_state(self, *args, **kwargs):

    kwargs["include_target"] = False
    sample = self.build_energy_horizon_sample(*args, **kwargs)
    return sample["input_state"]

def _esa_debug_horizon_dataset_report(self, day_id, history_window=5, horizon=5, stride=1, max_samples=3):
    samples = self.build_energy_horizon_dataset(
        day_id=day_id,
        history_window=history_window,
        horizon=horizon,
        stride=stride,
        max_samples=max_samples,
        include_target=True,
    )
    report = {
        "day_id": day_id,
        "history_window": history_window,
        "horizon": horizon,
        "num_samples_preview": len(samples),
        "sample_sizes": [len(json.dumps(s, ensure_ascii=False, default=str)) for s in samples],
        "first_sample_meta": samples[0].get("meta", {}) if samples else {},
    }
    print("\n=== Energy horizon dataset report ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report

EnergyStateAdapter._compact_patch_known_context = _esa_compact_patch_known_context
EnergyStateAdapter._compact_patch_observed = _esa_compact_patch_observed
EnergyStateAdapter._compact_patch_target = _esa_compact_patch_target
EnergyStateAdapter.build_energy_patch_sequence = _esa_build_energy_patch_sequence
EnergyStateAdapter.build_energy_horizon_sample = _esa_build_energy_horizon_sample
EnergyStateAdapter.build_energy_horizon_dataset = _esa_build_energy_horizon_dataset
EnergyStateAdapter.build_energy_horizon_prompt_state = _esa_build_energy_horizon_prompt_state
EnergyStateAdapter.debug_horizon_dataset_report = _esa_debug_horizon_dataset_report

def _esa_build_energy_repair_context(self, input_state):

    input_state = input_state if isinstance(input_state, dict) else {}
    constraints = input_state.get("constraints", {}) if isinstance(input_state.get("constraints"), dict) else {}

    prod_summary = {}
    for key in [
        "production_operation_segment_summary",
        "production_operation_segment_context",
        "operation_segment_summary",
    ]:
        val = input_state.get(key)
        if isinstance(val, dict) and (val.get("available") or val.get("num_segments") or val.get("stage_counts")):
            prod_summary = val
            break

    return {
        "energy_history": input_state.get("energy_history", []),
        "future_patch_context": input_state.get("future_patch_context", []),
        "production_operation_segment_summary": prod_summary,
        "constraints": constraints,
        "holder_physical_bounds": {

            "b": [0.0, 400.0],
            "c": [0.0, 200.0],
            "l": [0.0, 450.0],
        },
        "stage_energy_coefficients": {

            "BOF": {"ldg_generation": 1.0, "steam": 0.5, "power": 0.4},
            "LF": {"power": 0.7, "steam": 0.2},
            "Casting": {"steam": 0.7, "power": 0.3},
            "Rolling": {"power": 1.0, "steam": 0.6},
        },
        "notes": [
            "energy_repair_context is prediction-time information only",
            "target_future.energy_future is intentionally excluded",
            "production enters energy prediction only through operation-level Gantt summary",
        ],
    }

def _esa_build_energy_horizon_sample_pcvp(
    self,
    day_id,
    anchor_patch_idx,
    history_window=5,
    horizon=5,
    production_context=None,
    production_prediction=None,
    include_target=True,
    full_state=None,
):

    if history_window <= 0:
        raise ValueError("history_window must be positive")
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    sequence = self.build_energy_patch_sequence(day_id, full_state=full_state)
    n = len(sequence)
    i = int(anchor_patch_idx)

    if i < 0 or i >= n:
        raise IndexError(f"anchor_patch_idx={anchor_patch_idx} out of range for n={n}")
    if i - history_window + 1 < 0:
        raise ValueError(
            f"Not enough history: anchor={i}, history_window={history_window}. "
            f"Need anchor >= {history_window - 1}."
        )
    if i + horizon >= n:
        raise ValueError(
            f"Not enough future labels: anchor={i}, horizon={horizon}, n={n}. "
            f"Need anchor+horizon < n."
        )

    hist = sequence[i - history_window + 1 : i + 1]
    future = sequence[i + 1 : i + horizon + 1]

    prod_ctx = production_context if isinstance(production_context, dict) else {}
    prod_pred = production_prediction if isinstance(production_prediction, dict) else {}

    input_state = {
        "task": "energy_patch_horizon_prediction",
        "task_role": "energy_horizon_prediction_conditioned_on_operation_schedule",
        "day_id": day_id,
        "anchor_patch_idx": i,
        "anchor_patch_id": sequence[i].get("id"),
        "history_window": history_window,
        "horizon": horizon,
        "energy_history": [self._compact_patch_observed(p) for p in hist],
        "future_patch_context": [self._compact_patch_known_context(p) for p in future],
        "production_operation_segment_context": prod_ctx,
        "production_operation_segment_summary": prod_pred,
        "constraints": self._build_constraint_features(),
        "notes": [
            "future_patch_context contains only known heat/time/overlap context",
            "future energy values, future mechanism summaries and direct future risk labels are hidden from model input",
            "pc.ov in future_patch_context is a known structural overlap flag, not a predicted risk label",
            "production context is limited to operation-level Gantt segment summary",
            "energy prediction should be schedule_conditioned when production_operation_segment_summary.available is true",
            "target_future is used only for offline evaluation",
        ],
    }
    input_state["energy_repair_context"] = self._build_energy_repair_context(input_state)

    sample = {
        "sample_id": f"{day_id}_energy_anchor_{i}_W{history_window}_K{horizon}",
        "day_id": day_id,
        "anchor_patch_idx": i,
        "history_window": history_window,
        "horizon": horizon,
        "input_state": input_state,
        "meta": {
            "num_day_patches": n,
            "history_sq_range": [hist[0].get("sq"), hist[-1].get("sq")],
            "future_sq_range": [future[0].get("sq"), future[-1].get("sq")],
            "history_patch_ids": [p.get("id") for p in hist],
            "future_patch_ids": [p.get("id") for p in future],
            "future_heat_ids": [self._normalize_heat_id(p.get("hn") or self._safe_dict(p.get("pc")).get("phn") or p.get("id")) for p in future],
            "production_conditioning_available_at_sample_build": bool(prod_pred.get("available") or prod_pred.get("num_segments")),
        },
    }

    if include_target:
        sample["target_future"] = {
            "day_id": day_id,
            "anchor_patch_idx": i,
            "horizon": horizon,
            "energy_future": [self._compact_patch_target(p) for p in future],
        }

    return sample

def _esa_build_energy_horizon_dataset_pcvp(
    self,
    day_id,
    history_window=5,
    horizon=5,
    stride=1,
    max_samples=None,
    production_context_provider=None,
    production_prediction_provider=None,
    include_target=True,
    full_state=None,
):
    sequence = self.build_energy_patch_sequence(day_id, full_state=full_state)
    n = len(sequence)
    samples = []
    start = history_window - 1
    end = n - horizon - 1
    if end < start:
        return []

    anchor = start
    while anchor <= end:
        future = sequence[anchor + 1 : anchor + horizon + 1]
        future_context = [self._compact_patch_known_context(p) for p in future]

        prod_ctx = {}
        prod_pred = {}
        if callable(production_context_provider):
            prod_ctx = production_context_provider(anchor, horizon, future_context) or {}
        if callable(production_prediction_provider):
            prod_pred = production_prediction_provider(anchor, horizon, future_context) or {}

        samples.append(
            self.build_energy_horizon_sample(
                day_id=day_id,
                anchor_patch_idx=anchor,
                history_window=history_window,
                horizon=horizon,
                production_context=prod_ctx,
                production_prediction=prod_pred,
                include_target=include_target,
                full_state={"ps": sequence, "d": day_id},
            )
        )
        if max_samples is not None and len(samples) >= int(max_samples):
            break
        anchor += max(1, int(stride))
    return samples

EnergyStateAdapter._build_energy_repair_context = _esa_build_energy_repair_context
EnergyStateAdapter.build_energy_horizon_sample = _esa_build_energy_horizon_sample_pcvp
EnergyStateAdapter.build_energy_horizon_dataset = _esa_build_energy_horizon_dataset_pcvp

if __name__ == "__main__":
    adapter = EnergyStateAdapter(
        energy_data_path="data_of_IES.xlsx",
        heat_json_path="heat_segments.json",
        heat_excel_path="heat_segments.xlsx",
        patch_metadata_json_path="patch_metadata.json",
        patch_metadata_excel_path="patch_metadata.xlsx",
        patches_dir="patches",
        include_legend=False,
    )

    adapter.load(sample_size=None)

    example_day_id = list(adapter.day_to_heats.keys())[0]

    full_state = adapter.build_energy_state(example_day_id)
    energy_prompt_state = adapter.build_prompt_state(example_day_id, mode="energy")
    coupling_prompt_state = adapter.build_prompt_state(example_day_id, mode="coupling")
    batch_prompt_state = adapter.build_energy_batch_state(
        day_id=example_day_id,
        core_start_idx=0,
        core_size=6,
        left_context=1,
        right_context=1,
        mode="energy",
        full_state=full_state,
    )
    transition_index = adapter.build_transition_index(example_day_id)

    print("example_day_id:", example_day_id)
    print("full_energy_state keys:", list(full_state.keys()))
    print("energy prompt keys:", list(energy_prompt_state.keys()))
    print("coupling prompt keys:", list(coupling_prompt_state.keys()))
    print("batch prompt keys:", list(batch_prompt_state.keys()))
    print("transition_index keys:", list(transition_index.keys()))

    adapter.debug_size_report(example_day_id)

    audit = adapter.audit_energy_state(example_day_id)
    print("passed:", audit["passed"])
    print("num_errors:", len(audit["errors"]))
    print("num_warnings:", len(audit["warnings"]))

    print("\n=== errors ===")
    for x in audit["errors"][:20]:
        print(x)

    print("\n=== warnings ===")
    for x in audit["warnings"][:20]:
        print(x)

    print("\n=== stats ===")
    print(json.dumps(audit["stats"], ensure_ascii=False, indent=2))

    print("\n=== batch meta ===")
    print(json.dumps(batch_prompt_state["meta"], ensure_ascii=False, indent=2))

def _esa_energy_bool_env(name, default=False):
    try:
        import os
        val = os.environ.get(name)
        if val is None:
            return bool(default)
        return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}
    except Exception:
        return bool(default)

def _esa_energy_int_env(name, default):
    try:
        import os
        val = os.environ.get(name)
        if val is None or str(val).strip() == "":
            return int(default)
        return int(float(val))
    except Exception:
        return int(default)

def _esa_build_local_energy_blocks_from_context(self, future_patch_context, block_size=None):
    block_size = int(block_size or _esa_energy_int_env("ENERGY_LOCAL_BLOCK_SIZE", 5))
    block_size = max(1, block_size)
    future_patch_context = future_patch_context if isinstance(future_patch_context, list) else []
    blocks = []
    for offset in range(0, len(future_patch_context), block_size):
        chunk = future_patch_context[offset: offset + block_size]
        if not chunk:
            continue
        steps = []
        heat_ids = []
        patch_ids = []
        for j, item in enumerate(chunk, start=offset + 1):
            if not isinstance(item, dict):
                continue
            t = item.get("t") if isinstance(item.get("t"), dict) else {}
            hid = self._normalize_heat_id(
                item.get("hn")
                or (item.get("pc") if isinstance(item.get("pc"), dict) else {}).get("phn")
                or item.get("heat_id")
                or item.get("id")
            )
            pid = self._safe_str(item.get("id") or item.get("patch_id") or item.get("p"))
            if hid and hid not in heat_ids:
                heat_ids.append(hid)
            if pid and pid not in patch_ids:
                patch_ids.append(pid)
            steps.append({
                "step": j,
                "patch_id": pid,
                "heat_id": hid,
                "time_window": {
                    "start_min": item.get("start_min") or t.get("t0"),
                    "end_min": item.get("end_min") or t.get("t1"),
                    "start_time": t.get("ts") or item.get("start_time"),
                    "end_time": t.get("te") or item.get("end_time"),
                },
                "known_overlap": (item.get("pc") if isinstance(item.get("pc"), dict) else {}).get("ov"),
            })
        blocks.append({
            "block_id": f"energy_block_{len(blocks) + 1}",
            "step_range": [offset + 1, offset + len(chunk)],
            "num_steps": len(chunk),
            "patch_ids": patch_ids,
            "heat_ids": heat_ids,
            "steps": steps,
        })
    return blocks

def _esa_build_energy_retry_guidance_context(self, input_state):
    input_state = input_state if isinstance(input_state, dict) else {}
    future_patch_context = input_state.get("future_patch_context") if isinstance(input_state.get("future_patch_context"), list) else []
    production_summary = input_state.get("production_operation_segment_summary") if isinstance(input_state.get("production_operation_segment_summary"), dict) else {}
    return {
        "enabled": True,
        "method": "mechanism-loss-guided energy LLM self-refinement before Energy-PCVP",
        "max_retry_attempts": _esa_energy_int_env("ENERGY_MAX_HORIZON_RETRIES", 5),
        "retry_trigger_conditions": [
            "empty_model_output",
            "json_parse_failed",
            "missing_future_patch_predictions",
            "future_patch_predictions_length_less_than_horizon",
            "future_patch_context_binding_missing_or_misaligned",
            "production_conditioning_missing_when_production_summary_available",
            "Energy-PCVP hard violations remain after projection",
        ],
        "allowed_retry_feedback": [
            "schema errors",
            "missing fields",
            "future horizon window",
            "future_patch_context identifiers and time boundaries",
            "history energy patches",
            "production_operation_segment_summary",
            "Energy-PCVP violation diagnostics",
            "nonnegative, holder-bound and risk-consistency constraints",
        ],
        "forbidden_retry_feedback": [
            "target_future.energy_future",
            "true future energy values",
            "MAE/RMSE against target",
            "risk precision/recall/F1 against target",
            "any ground-truth future labels",
        ],
        "required_output_contract": {
            "required_key": "future_patch_predictions",
            "required_length": int(input_state.get("horizon") or len(future_patch_context) or 0),
            "patch_binding": "step k must correspond to future_patch_context[k-1]",
            "when_production_summary_available": "use production_activity_basis=schedule_conditioned and operation_segment_summary_used=true",
        },
        "production_conditioning_available": bool(production_summary.get("available") or production_summary.get("num_segments")),
        "num_future_patch_context": len(future_patch_context),
        "local_energy_blocks": _esa_build_local_energy_blocks_from_context(self, future_patch_context),
        "uses_future_target_labels": False,
    }

def _esa_build_energy_rule_fallback_context(self, input_state):
    input_state = input_state if isinstance(input_state, dict) else {}
    return {
        "enabled": _esa_energy_bool_env("ENERGY_ENABLE_RULE_FALLBACK_AFTER_RETRY", True),
        "method": "retry-exhausted rule-assisted energy completion",
        "evaluate_as_prediction": _esa_energy_bool_env("ENERGY_EVALUATE_RULE_FALLBACK_AS_PREDICTION", True),
        "uses_future_target_labels": False,
        "trigger": "used only when all LLM retry attempts fail to produce valid future_patch_predictions",
        "allowed_sources": [
            "energy_history tail state",
            "future_patch_context heat/time identifiers",
            "production_operation_segment_summary",
            "energy constraints and holder bounds",
            "history-based carry-forward and trend continuation",
        ],
        "forbidden_sources": [
            "target_future.energy_future",
            "true future energy values",
            "future target risks",
            "offline evaluation metrics",
        ],
        "completion_policy": [
            "bind every output step to future_patch_context",
            "carry forward last observed holder/generation/demand/purchase state",
            "apply small mechanism-consistent response adjustment if production activity is available",
            "send fallback result through Energy-PCVP before evaluation",
            "mark source as rule_assisted_fallback_after_retry_exhausted",
        ],
        "num_history_patches": len(input_state.get("energy_history") if isinstance(input_state.get("energy_history"), list) else []),
        "num_future_patch_context": len(input_state.get("future_patch_context") if isinstance(input_state.get("future_patch_context"), list) else []),
    }

def _esa_build_energy_horizon_sample_retry_pcvp_fallback(
    self,
    day_id,
    anchor_patch_idx,
    history_window=5,
    horizon=5,
    production_context=None,
    production_prediction=None,
    include_target=True,
    full_state=None,
):
    sample = _esa_build_energy_horizon_sample_pcvp(
        self,
        day_id=day_id,
        anchor_patch_idx=anchor_patch_idx,
        history_window=history_window,
        horizon=horizon,
        production_context=production_context,
        production_prediction=production_prediction,
        include_target=include_target,
        full_state=full_state,
    )
    input_state = sample.get("input_state") if isinstance(sample.get("input_state"), dict) else {}
    input_state["energy_retry_guidance_context"] = _esa_build_energy_retry_guidance_context(self, input_state)
    input_state["energy_rule_fallback_context"] = _esa_build_energy_rule_fallback_context(self, input_state)
    input_state["local_energy_blocks"] = input_state["energy_retry_guidance_context"].get("local_energy_blocks", [])
    input_state["energy_prediction_output_contract"] = {
        "required_key": "future_patch_predictions",
        "required_length": int(horizon),
        "valid_prediction_source": "model_output or rule_assisted_fallback_after_retry_exhausted",
        "target_future_not_visible_to_model": True,
        "pcvp_projection_required": True,
    }
    sample["input_state"] = input_state
    sample.setdefault("meta", {})
    sample["meta"]["energy_retry_enabled"] = True
    sample["meta"]["energy_rule_fallback_enabled"] = bool(input_state["energy_rule_fallback_context"].get("enabled"))
    sample["meta"]["energy_retry_max_attempts"] = input_state["energy_retry_guidance_context"].get("max_retry_attempts")
    return sample

def _esa_build_energy_horizon_dataset_retry_pcvp_fallback(
    self,
    day_id,
    history_window=5,
    horizon=5,
    stride=1,
    max_samples=None,
    production_context_provider=None,
    production_prediction_provider=None,
    include_target=True,
    full_state=None,
):
    sequence = self.build_energy_patch_sequence(day_id, full_state=full_state)
    n = len(sequence)
    samples = []
    start = history_window - 1
    end = n - horizon - 1
    if end < start:
        return []

    anchor = start
    while anchor <= end:
        future = sequence[anchor + 1: anchor + horizon + 1]
        future_context = [self._compact_patch_known_context(p) for p in future]

        prod_ctx = {}
        prod_pred = {}
        if callable(production_context_provider):
            prod_ctx = production_context_provider(anchor, horizon, future_context) or {}
        if callable(production_prediction_provider):
            prod_pred = production_prediction_provider(anchor, horizon, future_context) or {}

        try:
            samples.append(
                self.build_energy_horizon_sample(
                    day_id=day_id,
                    anchor_patch_idx=anchor,
                    history_window=history_window,
                    horizon=horizon,
                    production_context=prod_ctx,
                    production_prediction=prod_pred,
                    include_target=include_target,
                    full_state={"ps": sequence, "d": day_id},
                )
            )
        except Exception as e:

            if not hasattr(self, "_last_energy_horizon_sample_build_errors"):
                self._last_energy_horizon_sample_build_errors = []
            self._last_energy_horizon_sample_build_errors.append({
                "day_id": day_id,
                "anchor_patch_idx": anchor,
                "horizon": horizon,
                "error_type": type(e).__name__,
                "error_message": str(e),
            })
        if max_samples is not None and len(samples) >= int(max_samples):
            break
        anchor += max(1, int(stride))
    return samples

EnergyStateAdapter.build_energy_horizon_sample = _esa_build_energy_horizon_sample_retry_pcvp_fallback
EnergyStateAdapter.build_energy_horizon_dataset = _esa_build_energy_horizon_dataset_retry_pcvp_fallback

def _esa_final_sd(x):
    return x if isinstance(x, dict) else {}

def _esa_final_sl(x):
    return x if isinstance(x, list) else ([] if x is None else [x])

def _esa_final_float(x, default=0.0):
    try:
        if x is None:
            return default
        if isinstance(x, str) and not x.strip():
            return default
        return float(x)
    except Exception:
        return default

def _esa_final_get(obj, path, default=0.0):
    cur = obj
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return _esa_final_float(cur, default)

def _esa_final_patch_time(p):
    p = _esa_final_sd(p)
    t = _esa_final_sd(p.get("t"))
    t0 = _esa_final_float(p.get("start_min") or p.get("t0") or t.get("t0") or t.get("s0"), None)
    t1 = _esa_final_float(p.get("end_min") or p.get("t1") or t.get("t1") or t.get("s1"), None)
    return t0, t1

def _esa_final_canonical_stage(x):
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

def _esa_final_numeric_patch(self, p):
    p = _esa_final_sd(p)
    return {
        "id": p.get("id") or p.get("patch_id") or p.get("p"),
        "hn": self._normalize_heat_id(p.get("hn") or _esa_final_sd(p.get("pc")).get("phn") or p.get("heat_id") or p.get("id") or p.get("p")),
        "sq": p.get("sq"),
        "t": p.get("t", {}),
        "g": {
            "b_tot": _esa_final_get(p, ["g", "b", "tot"]),
            "c_tot": _esa_final_get(p, ["g", "c", "tot"]),
            "l_tot": _esa_final_get(p, ["g", "l", "tot"]),
        },
        "st": {
            "b_e": _esa_final_get(p, ["st", "b", "e"]),
            "c_e": _esa_final_get(p, ["st", "c", "e"]),
            "l_e": _esa_final_get(p, ["st", "l", "e"]),
        },
        "dm": {
            "s_tot": _esa_final_get(p, ["dm", "s", "tot"]),
            "s_pk": _esa_final_get(p, ["dm", "s", "pk"]),
            "e_tot": _esa_final_get(p, ["dm", "e", "tot"]),
            "e_pk": _esa_final_get(p, ["dm", "e", "pk"]),
        },
        "ex": {
            "s_tot": _esa_final_get(p, ["ex", "s", "tot"]),
            "e_tot": _esa_final_get(p, ["ex", "e", "tot"]),
        },
        "rk": _esa_final_sl(p.get("rk")),
    }

def _esa_final_trend(values, eps=1e-6):
    vals = [_esa_final_float(v, None) for v in values]
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return "unknown"
    delta = vals[-1] - vals[0]
    span = max(max(vals) - min(vals), 1.0)
    rel = delta / span
    if abs(delta) <= eps or abs(rel) < 0.10:
        return "flat"
    if rel >= 0.45:
        return "strong_up"
    if rel > 0:
        return "up"
    if rel <= -0.45:
        return "strong_down"
    return "down"

def _esa_final_level(v, low, high):
    v = _esa_final_float(v, 0.0)
    ratio = (v - low) / max(high - low, 1.0)
    if ratio < 0.15:
        return "very_low"
    if ratio < 0.35:
        return "low"
    if ratio < 0.65:
        return "medium"
    if ratio < 0.85:
        return "high"
    return "very_high"

def _esa_final_load_state(v):
    v = _esa_final_float(v, 0.0)
    if v < 40:
        return "very_low"
    if v < 80:
        return "low"
    if v < 140:
        return "medium"
    if v < 220:
        return "high"
    return "very_high"

def _esa_final_dependency(purchase, demand):
    purchase = _esa_final_float(purchase, 0.0)
    demand = max(_esa_final_float(demand, 0.0), 1.0)
    r = purchase / demand
    if r < 0.05:
        return "none"
    if r < 0.15:
        return "low"
    if r < 0.35:
        return "medium"
    return "high"

def _esa_final_build_semantic_energy_history(self, energy_history):
    rows = [_esa_final_numeric_patch(self, p) for p in _esa_final_sl(energy_history)]
    out = []
    for idx, r in enumerate(rows):
        out.append({
            "index": idx + 1,
            "patch_id": r.get("id"),
            "heat_id": r.get("hn"),
            "holder_state": {
                "BFG": {"level": _esa_final_level(r["st"]["b_e"], 60.0, 280.0), "value": round(r["st"]["b_e"], 4)},
                "COG": {"level": _esa_final_level(r["st"]["c_e"], 30.0, 120.0), "value": round(r["st"]["c_e"], 4)},
                "LDG": {"level": _esa_final_level(r["st"]["l_e"], 30.0, 320.0), "value": round(r["st"]["l_e"], 4)},
            },
            "demand_state": {
                "steam": _esa_final_load_state(r["dm"]["s_tot"]),
                "power": _esa_final_load_state(r["dm"]["e_tot"]),
            },
            "purchase_dependency": {
                "steam": _esa_final_dependency(r["ex"]["s_tot"], r["dm"]["s_tot"]),
                "power": _esa_final_dependency(r["ex"]["e_tot"], r["dm"]["e_tot"]),
            },
            "risk_flags": r.get("rk", []),
            "numeric_anchor": {
                "ldg_generation_total": round(r["g"]["l_tot"], 4),
                "ldg_holder_end": round(r["st"]["l_e"], 4),
                "steam_load_total": round(r["dm"]["s_tot"], 4),
                "power_load_total": round(r["dm"]["e_tot"], 4),
                "purchased_steam_total": round(r["ex"]["s_tot"], 4),
                "purchased_power_total": round(r["ex"]["e_tot"], 4),
            },
        })
    return out

def _esa_final_history_transition_summary(self, energy_history):
    rows = [_esa_final_numeric_patch(self, p) for p in _esa_final_sl(energy_history)]
    if not rows:
        return {"available": False}
    def seq(section, key):
        return [r[section][key] for r in rows]
    last = rows[-1]
    return {
        "available": True,
        "num_history_patches": len(rows),
        "trend": {
            "ldg_generation_total": _esa_final_trend(seq("g", "l_tot")),
            "ldg_holder_end": _esa_final_trend(seq("st", "l_e")),
            "steam_load_total": _esa_final_trend(seq("dm", "s_tot")),
            "power_load_total": _esa_final_trend(seq("dm", "e_tot")),
            "purchased_steam_total": _esa_final_trend(seq("ex", "s_tot")),
            "purchased_power_total": _esa_final_trend(seq("ex", "e_tot")),
        },
        "last_numeric_anchor": {
            "ldg_generation_total": round(last["g"]["l_tot"], 4),
            "bfg_generation_total": round(last["g"]["b_tot"], 4),
            "cog_generation_total": round(last["g"]["c_tot"], 4),
            "ldg_holder_end": round(last["st"]["l_e"], 4),
            "bfg_holder_end": round(last["st"]["b_e"], 4),
            "cog_holder_end": round(last["st"]["c_e"], 4),
            "steam_load_total": round(last["dm"]["s_tot"], 4),
            "power_load_total": round(last["dm"]["e_tot"], 4),
            "steam_load_peak": round(last["dm"]["s_pk"], 4),
            "power_load_peak": round(last["dm"]["e_pk"], 4),
            "purchased_steam_total": round(last["ex"]["s_tot"], 4),
            "purchased_power_total": round(last["ex"]["e_tot"], 4),
        },
        "note": "numeric anchors are computed only from observed energy_history, not from target_future",
    }

def _esa_final_production_summary_from_input(input_state):
    input_state = _esa_final_sd(input_state)
    for key in ["production_operation_segment_summary", "production_operation_segment_context", "operation_segment_summary"]:
        val = input_state.get(key)
        if isinstance(val, dict) and (val.get("available") or val.get("num_segments") or val.get("stage_counts") or val.get("examples")):
            out = dict(val)
            out.setdefault("available", bool(out.get("num_segments") or out.get("examples") or out.get("stage_counts")))
            return out
    return {"available": False, "num_segments": 0, "stage_counts": {}, "examples": []}

def _esa_final_build_future_production_activity_by_step(self, future_context, production_summary):
    future_context = _esa_final_sl(future_context)
    production_summary = _esa_final_sd(production_summary)
    examples = [x for x in _esa_final_sl(production_summary.get("examples")) if isinstance(x, dict)]
    prod_available = bool(production_summary.get("available") or production_summary.get("num_segments") or examples or production_summary.get("stage_counts"))
    out = []

    any_overlap = False
    for idx, fctx in enumerate(future_context, start=1):
        p0, p1 = _esa_final_patch_time(fctx)
        stage_overlap = {"BOF": 0.0, "LF": 0.0, "Casting": 0.0, "Rolling": 0.0, "Other": 0.0}
        unit_overlap = {}
        heat_ids = []
        if p0 is not None and p1 is not None and p1 > p0:
            for seg in examples:
                s = _esa_final_float(seg.get("start_min"), None)
                e = _esa_final_float(seg.get("end_min"), None)
                if s is None or e is None:
                    continue
                ov = max(0.0, min(p1, e) - max(p0, s))
                if ov <= 0:
                    continue
                any_overlap = True
                stage = _esa_final_canonical_stage(seg.get("stage"))
                stage_overlap[stage] = stage_overlap.get(stage, 0.0) + ov
                unit = str(seg.get("unit") or stage)
                unit_overlap[unit] = unit_overlap.get(unit, 0.0) + ov
                hid = self._normalize_heat_id(seg.get("heat_id") or seg.get("hn") or seg.get("id"))
                if hid and hid not in heat_ids:
                    heat_ids.append(hid)
        score = 1.20 * stage_overlap["BOF"] + 0.90 * stage_overlap["LF"] + 0.85 * stage_overlap["Casting"] + 1.30 * stage_overlap["Rolling"]
        out.append({
            "step": idx,
            "patch_id": _esa_final_sd(fctx).get("id") or _esa_final_sd(fctx).get("patch_id") or _esa_final_sd(fctx).get("p"),
            "heat_id": self._normalize_heat_id(_esa_final_sd(fctx).get("hn") or _esa_final_sd(_esa_final_sd(fctx).get("pc")).get("phn") or _esa_final_sd(fctx).get("heat_id")),
            "available": prod_available,
            "basis": "time_overlap" if prod_available and score > 0 else ("operation_segment_summary_no_step_overlap" if prod_available else "history_only"),
            "stage_overlap_min": {k: round(v, 4) for k, v in stage_overlap.items() if abs(v) > 1e-9},
            "unit_overlap_min": {k: round(v, 4) for k, v in unit_overlap.items()},
            "active_heat_ids": heat_ids[:12],
            "activity_score": round(score, 4),
            "activity_level": "high" if score >= 80 else ("medium" if score >= 30 else ("low" if score > 0 else "none")),
        })

    if prod_available and not any_overlap:
        stage_counts = _esa_final_sd(production_summary.get("stage_counts"))
        total_count = sum(_esa_final_float(v, 0.0) for v in stage_counts.values())
        for row in out:
            row["basis"] = "window_summary_only_no_time_overlap"
            row["window_stage_counts"] = stage_counts
            row["activity_score"] = round(total_count, 4)
            row["activity_level"] = "medium" if total_count > 0 else "none"
            row["time_alignment_warning"] = "no segment example overlaps future_patch_context; check time base if unexpected"
    return out

def _esa_final_build_energy_repair_context(self, input_state):
    input_state = _esa_final_sd(input_state)
    prod_summary = _esa_final_production_summary_from_input(input_state)
    future_context = _esa_final_sl(input_state.get("future_patch_context"))
    energy_history = _esa_final_sl(input_state.get("energy_history"))
    return {
        "energy_history": energy_history,
        "future_patch_context": future_context,
        "production_operation_segment_summary": prod_summary,
        "future_production_activity_by_step": _esa_final_build_future_production_activity_by_step(self, future_context, prod_summary),
        "semantic_energy_history": _esa_final_build_semantic_energy_history(self, energy_history),
        "history_transition_summary": _esa_final_history_transition_summary(self, energy_history),
        "constraints": input_state.get("constraints", {}),
        "holder_physical_bounds": {"b": [0.0, 400.0], "c": [0.0, 200.0], "l": [0.0, 450.0]},
        "stage_energy_coefficients": {
            "BOF": {"ldg_generation": 1.0, "steam": 0.5, "power": 0.4},
            "LF": {"power": 0.7, "steam": 0.2},
            "Casting": {"steam": 0.7, "power": 0.3},
            "Rolling": {"power": 1.0, "steam": 0.6},
        },
        "uses_future_target_labels": False,
        "notes": [
            "repair context uses observed history, known future patch context, and predicted production operation-segment summary only",
            "target_future.energy_future is excluded",
        ],
    }

def _esa_final_enrich_energy_horizon_sample(self, sample):
    if not isinstance(sample, dict):
        return sample
    input_state = sample.get("input_state") if isinstance(sample.get("input_state"), dict) else {}
    prod_summary = _esa_final_production_summary_from_input(input_state)
    energy_history = _esa_final_sl(input_state.get("energy_history"))
    future_context = _esa_final_sl(input_state.get("future_patch_context"))
    activity = _esa_final_build_future_production_activity_by_step(self, future_context, prod_summary)
    semantic_history = _esa_final_build_semantic_energy_history(self, energy_history)
    hist_summary = _esa_final_history_transition_summary(self, energy_history)

    input_state["production_operation_segment_summary"] = prod_summary
    input_state["production_conditioning_available"] = bool(prod_summary.get("available") or prod_summary.get("num_segments"))
    input_state["production_activity_basis"] = "schedule_conditioned" if input_state["production_conditioning_available"] else "history_only"
    input_state["future_production_activity_by_step"] = activity
    input_state["semantic_energy_history"] = semantic_history
    input_state["history_transition_summary"] = hist_summary
    input_state["energy_numeric_history_reference"] = hist_summary.get("last_numeric_anchor", {}) if isinstance(hist_summary, dict) else {}
    input_state["energy_mechanism_reasoning_contract"] = {
        "llm_role": "predict future numeric energy states by mechanism-aware reasoning, not by copying targets",
        "must_use_when_available": ["future_production_activity_by_step", "semantic_energy_history", "history_transition_summary"],
        "numeric_output_required": True,
        "future_target_values_visible": False,
        "forbidden": ["target_future.energy_future", "future measured energy values", "future target risk labels"],
        "anti_smoothing_guidance": [
            "do not output nearly constant purchased power/steam values when production activity changes by step",
            "use BOF overlap as LDG generation evidence",
            "use Rolling/LF overlap as power demand evidence",
            "use Casting/Rolling overlap as steam demand evidence",
            "if step-level activity is unavailable, state the uncertainty in comment and keep production_activity_basis explicit",
        ],
    }
    input_state["energy_repair_context"] = _esa_final_build_energy_repair_context(self, input_state)
    sample["input_state"] = input_state
    sample.setdefault("meta", {})
    sample["meta"]["semantic_energy_state_enabled"] = True
    sample["meta"]["future_production_activity_by_step_enabled"] = True
    sample["meta"]["production_conditioning_available"] = input_state["production_conditioning_available"]
    sample["meta"]["production_activity_basis"] = input_state["production_activity_basis"]
    return sample

_ESA_PRE_FINAL_BUILD_ENERGY_HORIZON_SAMPLE = EnergyStateAdapter.build_energy_horizon_sample

def _esa_final_build_energy_horizon_sample(self, *args, **kwargs):
    sample = _ESA_PRE_FINAL_BUILD_ENERGY_HORIZON_SAMPLE(self, *args, **kwargs)
    return _esa_final_enrich_energy_horizon_sample(self, sample)

EnergyStateAdapter.build_energy_horizon_sample = _esa_final_build_energy_horizon_sample
EnergyStateAdapter.build_energy_repair_context = _esa_final_build_energy_repair_context
EnergyStateAdapter._build_semantic_energy_history = _esa_final_build_semantic_energy_history
EnergyStateAdapter._build_future_production_activity_by_step = _esa_final_build_future_production_activity_by_step
EnergyStateAdapter._build_history_transition_summary = _esa_final_history_transition_summary

def _esa_v2_path_set(obj, path, value):
    cur = obj
    for key in path[:-1]:
        if not isinstance(cur.get(key), dict):
            cur[key] = {}
        cur = cur[key]
    try:
        v = float(value)
    except Exception:
        v = 0.0
    cur[path[-1]] = round(max(0.0, v), 4)

def _esa_v2_history_numeric_rows(self, energy_history):
    rows = []
    for p in _esa_final_sl(energy_history):
        r = _esa_final_numeric_patch(self, p)
        rows.append({
            "patch_id": r.get("id"),
            "heat_id": r.get("hn"),
            "t": r.get("t", {}),
            "g_b_tot": r["g"]["b_tot"],
            "g_c_tot": r["g"]["c_tot"],
            "g_l_tot": r["g"]["l_tot"],
            "st_b_e": r["st"]["b_e"],
            "st_c_e": r["st"]["c_e"],
            "st_l_e": r["st"]["l_e"],
            "dm_s_tot": r["dm"]["s_tot"],
            "dm_s_pk": r["dm"]["s_pk"],
            "dm_e_tot": r["dm"]["e_tot"],
            "dm_e_pk": r["dm"]["e_pk"],
            "ex_s_tot": r["ex"]["s_tot"],
            "ex_e_tot": r["ex"]["e_tot"],
            "rk": r.get("rk", []),
        })
    return rows

def _esa_v2_mean(vals):
    vals = [_esa_final_float(v, None) for v in vals]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else 0.0

def _esa_v2_trend(vals):
    vals = [_esa_final_float(v, None) for v in vals]
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return 0.0

    return (vals[-1] - vals[0]) / max(len(vals) - 1, 1)

def _esa_v2_linear_forecast(rows, key, step):
    if not rows:
        return 0.0
    vals = [_esa_final_float(r.get(key), 0.0) for r in rows]
    last = vals[-1]
    trend = _esa_v2_trend(vals)
    hist_min = min(vals)
    hist_max = max(vals)
    hist_rng = max(hist_max - hist_min, 1.0)
    val = last + step * trend

    lo = max(0.0, hist_min - 0.50 * hist_rng)
    hi = hist_max + 0.75 * hist_rng
    return min(max(val, lo), hi)

def _esa_v2_build_concrete_energy_history_window(self, energy_history):

    out = []
    for idx, r in enumerate(_esa_v2_history_numeric_rows(self, energy_history), start=1):
        out.append({
            "index": idx,
            "patch_id": r.get("patch_id"),
            "heat_id": r.get("heat_id"),
            "t": r.get("t", {}),
            "generation_total": {
                "BFG": round(r["g_b_tot"], 4),
                "COG": round(r["g_c_tot"], 4),
                "LDG": round(r["g_l_tot"], 4),
            },
            "holder_end": {
                "BFG": round(r["st_b_e"], 4),
                "COG": round(r["st_c_e"], 4),
                "LDG": round(r["st_l_e"], 4),
            },
            "demand_total": {
                "steam": round(r["dm_s_tot"], 4),
                "power": round(r["dm_e_tot"], 4),
            },
            "demand_peak": {
                "steam": round(r["dm_s_pk"], 4),
                "power": round(r["dm_e_pk"], 4),
            },
            "purchase_total": {
                "steam": round(r["ex_s_tot"], 4),
                "power": round(r["ex_e_tot"], 4),
            },
            "risk_flags": r.get("rk", []),
        })
    return out

def _esa_v2_build_energy_baseline_future_predictions(self, energy_history, future_context, future_activity):

    rows = _esa_v2_history_numeric_rows(self, energy_history)
    future_context = _esa_final_sl(future_context)
    future_activity = _esa_final_sl(future_activity)
    if not rows:
        return []

    out = []
    last_hist = rows[-1]
    last_l_holder = _esa_final_float(last_hist.get("st_l_e"), 0.0)

    for idx, fctx in enumerate(future_context, start=1):
        act = _esa_final_sd(future_activity[idx - 1]) if idx - 1 < len(future_activity) else {}
        stage_ov = _esa_final_sd(act.get("stage_overlap_min"))
        bof = _esa_final_float(stage_ov.get("BOF"), 0.0)
        lf = _esa_final_float(stage_ov.get("LF"), 0.0)
        casting = _esa_final_float(stage_ov.get("Casting"), 0.0)
        rolling = _esa_final_float(stage_ov.get("Rolling"), 0.0)
        pid = _esa_final_sd(fctx).get("id") or _esa_final_sd(fctx).get("patch_id") or _esa_final_sd(fctx).get("p") or f"baseline_step_{idx}"
        hid = self._normalize_heat_id(_esa_final_sd(fctx).get("hn") or _esa_final_sd(_esa_final_sd(fctx).get("pc")).get("phn") or _esa_final_sd(fctx).get("heat_id") or pid)

        g_b = _esa_v2_linear_forecast(rows, "g_b_tot", idx)
        g_c = _esa_v2_linear_forecast(rows, "g_c_tot", idx)
        g_l = _esa_v2_linear_forecast(rows, "g_l_tot", idx)
        st_b = _esa_v2_linear_forecast(rows, "st_b_e", idx)
        st_c = _esa_v2_linear_forecast(rows, "st_c_e", idx)
        dm_s = _esa_v2_linear_forecast(rows, "dm_s_tot", idx)
        dm_s_pk = _esa_v2_linear_forecast(rows, "dm_s_pk", idx)
        dm_e = _esa_v2_linear_forecast(rows, "dm_e_tot", idx)
        dm_e_pk = _esa_v2_linear_forecast(rows, "dm_e_pk", idx)
        ex_s_hist = _esa_v2_linear_forecast(rows, "ex_s_tot", idx)
        ex_e_hist = _esa_v2_linear_forecast(rows, "ex_e_tot", idx)

        g_l *= (1.0 + min(0.85, bof / 80.0))
        dm_s *= (1.0 + min(0.75, (casting + 0.6 * rolling) / 140.0))
        dm_e *= (1.0 + min(0.95, (rolling + 0.7 * lf) / 120.0))
        dm_s_pk = max(dm_s, dm_s_pk * (1.0 + min(0.75, (casting + rolling) / 160.0)))
        dm_e_pk = max(dm_e, dm_e_pk * (1.0 + min(0.95, (rolling + lf) / 140.0)))

        ex_s = max(ex_s_hist, max(0.0, dm_s - 0.55 * g_l) * 0.35)
        ex_e = max(ex_e_hist, max(0.0, dm_e - 0.20 * g_b) * 0.55)
        st_l = min(max(last_l_holder + 0.05 * (g_l - 0.25 * dm_s - 0.12 * dm_e), 0.0), 450.0)
        last_l_holder = st_l

        item = {
            "step": idx,
            "patch_id": pid,
            "id": pid,
            "p": pid,
            "heat_id": hid,
            "hn": hid,
            "baseline_source": "observed_history_trend_plus_predicted_operation_activity_no_target",
            "baseline_role": "soft_numeric_reference_not_ground_truth",
            "activity_basis": act.get("basis", "unknown"),
            "activity_score": act.get("activity_score", 0.0),
            "g": {}, "st": {}, "dm": {}, "ex": {},
        }
        _esa_v2_path_set(item, ["g", "b", "tot"], g_b)
        _esa_v2_path_set(item, ["g", "c", "tot"], g_c)
        _esa_v2_path_set(item, ["g", "l", "tot"], g_l)
        _esa_v2_path_set(item, ["st", "b", "e"], min(max(st_b, 0.0), 400.0))
        _esa_v2_path_set(item, ["st", "b", "m"], min(max(st_b, 0.0), 400.0))
        _esa_v2_path_set(item, ["st", "c", "e"], min(max(st_c, 0.0), 200.0))
        _esa_v2_path_set(item, ["st", "c", "m"], min(max(st_c, 0.0), 200.0))
        _esa_v2_path_set(item, ["st", "l", "e"], st_l)
        _esa_v2_path_set(item, ["st", "l", "m"], st_l)
        _esa_v2_path_set(item, ["dm", "s", "tot"], dm_s)
        _esa_v2_path_set(item, ["dm", "s", "pk"], dm_s_pk)
        _esa_v2_path_set(item, ["dm", "e", "tot"], dm_e)
        _esa_v2_path_set(item, ["dm", "e", "pk"], dm_e_pk)
        _esa_v2_path_set(item, ["ex", "s", "tot"], ex_s)
        _esa_v2_path_set(item, ["ex", "e", "tot"], ex_e)
        out.append(item)

    return out

_ESA_V2_PREV_ENRICH_SAMPLE = _esa_final_enrich_energy_horizon_sample

def _esa_v2_enrich_energy_horizon_sample(self, sample):
    sample = _ESA_V2_PREV_ENRICH_SAMPLE(self, sample)
    if not isinstance(sample, dict):
        return sample
    input_state = sample.get("input_state") if isinstance(sample.get("input_state"), dict) else {}
    energy_history = _esa_final_sl(input_state.get("energy_history"))
    future_context = _esa_final_sl(input_state.get("future_patch_context"))
    activity = input_state.get("future_production_activity_by_step")
    if not isinstance(activity, list):
        activity = []

    concrete_history = _esa_v2_build_concrete_energy_history_window(self, energy_history)
    baseline_future = _esa_v2_build_energy_baseline_future_predictions(self, energy_history, future_context, activity)

    input_state["energy_concrete_history_window"] = concrete_history
    input_state["energy_baseline_future_predictions"] = baseline_future
    input_state["energy_baseline_policy"] = {
        "enabled": True,
        "source": "observed_history_trend_plus_predicted_operation_activity",
        "uses_future_target_labels": False,
        "role_for_llm": "reference_and_anchor_only; LLM must still infer numeric future states from concrete history and production activity",
        "role_for_pcvp": "soft_prior_for_missing_or_extreme_values_not_hard_replacement",
        "recommended_soft_blend_weight": 0.20,
    }
    input_state.setdefault("energy_mechanism_reasoning_contract", {})
    input_state["energy_mechanism_reasoning_contract"]["must_use_when_available"] = list(dict.fromkeys(
        _esa_final_sl(input_state["energy_mechanism_reasoning_contract"].get("must_use_when_available"))
        + ["energy_concrete_history_window", "energy_baseline_future_predictions"]
    ))
    input_state["energy_mechanism_reasoning_contract"]["baseline_is_not_target"] = True

    rc = input_state.get("energy_repair_context") if isinstance(input_state.get("energy_repair_context"), dict) else {}
    rc["energy_concrete_history_window"] = concrete_history
    rc["energy_baseline_future_predictions"] = baseline_future
    rc["energy_baseline_policy"] = input_state["energy_baseline_policy"]
    rc["baseline_blend_weight"] = 0.20
    rc["baseline_blend_policy"] = "soft_prior_only_for_missing_or_extreme_llm_values"
    rc["uses_future_target_labels"] = False
    input_state["energy_repair_context"] = rc

    sample["input_state"] = input_state
    sample.setdefault("meta", {})
    sample["meta"]["energy_concrete_history_window_enabled"] = True
    sample["meta"]["energy_baseline_future_predictions_enabled"] = True
    sample["meta"]["energy_baseline_uses_future_target_labels"] = False
    return sample

_ESA_V2_PRE_BUILD_ENERGY_HORIZON_SAMPLE = EnergyStateAdapter.build_energy_horizon_sample

def _esa_v2_build_energy_horizon_sample(self, *args, **kwargs):
    sample = _ESA_V2_PRE_BUILD_ENERGY_HORIZON_SAMPLE(self, *args, **kwargs)
    return _esa_v2_enrich_energy_horizon_sample(self, sample)

EnergyStateAdapter.build_energy_horizon_sample = _esa_v2_build_energy_horizon_sample
EnergyStateAdapter._build_concrete_energy_history_window = _esa_v2_build_concrete_energy_history_window
EnergyStateAdapter._build_energy_baseline_future_predictions = _esa_v2_build_energy_baseline_future_predictions
