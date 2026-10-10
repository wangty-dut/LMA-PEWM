from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional, Dict, List

class DataLoader:

    def __init__(self) -> None:

        self.translation_dict: Dict[str, str] = {

            "Time": "Time",
            "Time (min)": "Time (min)",

            "1#blast furnace Generation (BFG)(NKM³/h)": "1#blast furnace Generation (NKM³/h)",
            "2#blast furnace Generation (BFG)(NKM³/h)": "2#blast furnace Generation (NKM³/h)",

            "1#coke oven Generation(COG) (NKM³/h)": "1#coke oven Generation (NKM³/h)",
            "1#coke oven Generation (COG)(NKM³/h)": "1#coke oven Generation (NKM³/h)",
            "2#coke oven Generation(COG) (NKM³/h)": "2#coke oven Generation (NKM³/h)",
            "2#coke oven Generation (COG)(NKM³/h)": "2#coke oven Generation (NKM³/h)",

            "1#LDG Generation (NKM³/h)": "1#LDG Generation (NKM³/h)",
            "2#LDG Generation (NKM³/h)": "2#LDG Generation (NKM³/h)",
            "3#LDG Generation (NKM³/h)": "3#LDG Generation (NKM³/h)",
            "4#LDG Generation (NKM³/h)": "4#LDG Generation (NKM³/h)",

            "BFG holder (NKM³)": "BFG holder (NKM³)",
            "COG holder (NKM³)": "COG holder (NKM³)",
            "LDG holder (NKM³)": "LDG holder (NKM³)",

            "Total_Steam_Load_tph": "Total_Steam_Load_tph",
            "Electric Load (MW)": "Electric Load (MW)",
            "Purchased_Power(MW)": "Purchased_Power(MW)",
            "Purchased_Steam_tph": "Purchased_Steam_tph",

            "2250hot-rolling Consumption(BFG) (NKM³/h)": "2250hot_rolling_BFG",
            "5G_Scold rolling Consumption(BFG) (NKM³/h)": "5G_Scold_rolling_BFG",
            "Heating furnace Consumption(BFG) (NKM³/h)": "Heating_furnace_BFG",
            "Other users Consumption(BFG) (NKM³/h)": "Other_users_BFG",

            "hydrogen production Consumption(COG) (NKM³/h)": "Hydrogen_production_COG",
            "Hydrogen production Consumption(COG) (NKM³/h)": "Hydrogen_production_COG",
            "pelletizing Consumption(COG) (NKM³/h)": "Pelletizing_COG",
            "2250hot-rolling Consumption(COG) (NKM³/h)": "2250hot_rolling_COG",
            "Heating furnace Consumption(COG) (NKM³/h)": "Heating_furnace_COG",
            "Other users Consumption(COG) (NKM³/h)": "Other_users_COG",

            "blast furnace Consumption(LDG) (NKM³/h)": "Blast_furnace_LDG",
            "Gas mixing station Consumption (LDG) (NKM³/h)": "Gas_mixing_station_LDG",
            "Rotary bottom furnace Consumption(LDG) (NKM³/h)": "Rotary_bottom_furnace_LDG",
            "Heating furnace Consumption(LDG) (NKM³/h)": "Heating_furnace_LDG",
            "Other users Consumption(LDG) (NKM³/h)": "Other_users_LDG",

            "Boiler1 Consumption(BFG) (NKM³/h)": "Boiler1_BFG",
            "Boiler1 Consumption(COG) (NKM³/h)": "Boiler1_COG",
            "Boiler1 Consumption(LDG) (NKM³/h)": "Boiler1_LDG",

            "Boiler2 Consumption(BFG) (NKM³/h)": "Boiler2_BFG",
            "Boiler2 Consumption(COG) (NKM³/h)": "Boiler2_COG",
            "Boiler2 Consumption(LDG) (NKM³/h)": "Boiler2_LDG",

            "Boiler3 Consumption(BFG) (NKM³/h)": "Boiler3_BFG",
            "Boiler3 Consumption(COG) (NKM³/h)": "Boiler3_COG",
            "Boiler3 Consumption(LDG) (NKM³/h)": "Boiler3_LDG",

            "Boiler4 Consumption(BFG) (NKM³/h)": "Boiler4_BFG",
            "Boiler4 Consumption(COG) (NKM³/h)": "Boiler4_COG",
            "Boiler4 Consumption(LDG) (NKM³/h)": "Boiler4_LDG",

            "CHP1 Consumption(BFG) (NKM³/h)": "CHP1_BFG",
            "CHP1 Consumption(COG) (NKM³/h)": "CHP1_COG",
            "CHP2 Consumption(BFG) (NKM³/h)": "CHP2_BFG",
            "CHP2 Consumption(COG) (NKM³/h)": "CHP2_COG",

            "Boiler1_Steam_tph (t/h)": "Boiler1_Steam_tph",
            "Boiler2_Steam_tph (t/h)": "Boiler2_Steam_tph",
            "Boiler3_Steam_tph (t/h)": "Boiler3_Steam_tph",
            "Boiler4_Steam_tph (t/h)": "Boiler4_Steam_tph",
            "CHP1_Steam_tph (t/h)": "CHP1_Steam_tph",
            "CHP2_Steam_tph (t/h)": "CHP2_Steam_tph",

            "Boiler1_Power_MW (MW)": "Boiler1_Power_MW",
            "Boiler2_Power_MW (MW)": "Boiler2_Power_MW",
            "Boiler3_Power_MW (MW)": "Boiler3_Power_MW",
            "Boiler4_Power_MW (MW)": "Boiler4_Power_MW",
            "CHP1_Power_MW (MW)": "CHP1_Power_MW",
            "CHP2_Power_MW (MW)": "CHP2_Power_MW",
        }

        self.required_columns_with_defaults: Dict[str, float] = {

            "1#blast furnace Generation (NKM³/h)": 0.0,
            "2#blast furnace Generation (NKM³/h)": 0.0,
            "1#coke oven Generation (NKM³/h)": 0.0,
            "2#coke oven Generation (NKM³/h)": 0.0,
            "1#LDG Generation (NKM³/h)": 0.0,
            "2#LDG Generation (NKM³/h)": 0.0,
            "3#LDG Generation (NKM³/h)": 0.0,
            "4#LDG Generation (NKM³/h)": 0.0,

            "BFG holder (NKM³)": 100.0,
            "COG holder (NKM³)": 50.0,
            "LDG holder (NKM³)": 100.0,

            "Total_Steam_Load_tph": 0.0,
            "Electric Load (MW)": 0.0,
            "Purchased_Power(MW)": 0.0,
            "Purchased_Steam_tph": 0.0,

            "Boiler1_Steam_tph": 0.0,
            "Boiler2_Steam_tph": 0.0,
            "Boiler3_Steam_tph": 0.0,
            "Boiler4_Steam_tph": 0.0,
            "CHP1_Steam_tph": 0.0,
            "CHP2_Steam_tph": 0.0,

            "Boiler1_Power_MW": 0.0,
            "Boiler2_Power_MW": 0.0,
            "Boiler3_Power_MW": 0.0,
            "Boiler4_Power_MW": 0.0,
            "CHP1_Power_MW": 0.0,
            "CHP2_Power_MW": 0.0,
        }

        self.consumption_columns: List[str] = [

            "2250hot_rolling_BFG",
            "5G_Scold_rolling_BFG",
            "Heating_furnace_BFG",
            "Other_users_BFG",

            "Hydrogen_production_COG",
            "Pelletizing_COG",
            "2250hot_rolling_COG",
            "Heating_furnace_COG",
            "Other_users_COG",

            "Blast_furnace_LDG",
            "Gas_mixing_station_LDG",
            "Rotary_bottom_furnace_LDG",
            "Heating_furnace_LDG",
            "Other_users_LDG",

            "Boiler1_BFG", "Boiler1_COG", "Boiler1_LDG",
            "Boiler2_BFG", "Boiler2_COG", "Boiler2_LDG",
            "Boiler3_BFG", "Boiler3_COG", "Boiler3_LDG",
            "Boiler4_BFG", "Boiler4_COG", "Boiler4_LDG",

            "CHP1_BFG", "CHP1_COG",
            "CHP2_BFG", "CHP2_COG",
        ]

    def _normalize_column_name(self, col: str) -> str:

        col = str(col).strip()
        col = col.replace("（", "(").replace("）", ")")
        col = " ".join(col.split())
        return col

    def normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()
        df.columns = [self._normalize_column_name(c) for c in df.columns]
        return df

    def translate_columns_to_english_schema(self, df: pd.DataFrame) -> pd.DataFrame:

        df = self.normalize_columns(df)
        existing = {src: dst for src, dst in self.translation_dict.items() if src in df.columns}
        df = df.rename(columns=existing)
        return df

    def _read_file(self, file_path: str) -> pd.DataFrame:

        if file_path.endswith(".xlsx"):
            return pd.read_excel(file_path)
        if file_path.endswith(".csv"):
            return pd.read_csv(file_path)
        raise ValueError("Unsupported file type. Please use .xlsx or .csv")

    def _process_time_columns(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        if "Time" in df.columns:
            df["Time"] = pd.to_datetime(df["Time"], errors="coerce")

            if df["Time"].isna().any():
                invalid_count = int(df["Time"].isna().sum())
                print(f"Warning: dropping {invalid_count} rows with invalid Time")
                df = df.dropna(subset=["Time"]).reset_index(drop=True)

            if df.empty:
                raise ValueError("All rows were dropped because Time could not be parsed.")

            df["Hour"] = df["Time"].dt.hour.astype(int)
            df["Minute"] = df["Time"].dt.minute.astype(int)

            min_time = df["Time"].min().normalize()
            df["DayIndex"] = (df["Time"].dt.normalize() - min_time).dt.days.astype(int)
            df["MinuteIndex"] = ((df["Time"] - df["Time"].iloc[0]).dt.total_seconds() // 60).astype(int)

        elif "Time (min)" in df.columns:

            dt_series = pd.to_datetime(df["Time (min)"], errors="coerce")

            if dt_series.notna().sum() > 0:
                df["Time"] = dt_series

                if df["Time"].isna().any():
                    invalid_count = int(df["Time"].isna().sum())
                    print(f"Warning: dropping {invalid_count} rows with invalid Time (min)")
                    df = df.dropna(subset=["Time"]).reset_index(drop=True)

                if df.empty:
                    raise ValueError("All rows were dropped because Time (min) could not be parsed.")

                df["Hour"] = df["Time"].dt.hour.astype(int)
                df["Minute"] = df["Time"].dt.minute.astype(int)

                min_time = df["Time"].min().normalize()
                df["DayIndex"] = (df["Time"].dt.normalize() - min_time).dt.days.astype(int)
                df["MinuteIndex"] = ((df["Time"] - df["Time"].iloc[0]).dt.total_seconds() // 60).astype(int)

            else:

                df["Time (min)"] = pd.to_numeric(df["Time (min)"], errors="coerce")

                if df["Time (min)"].isna().any():
                    invalid_count = int(df["Time (min)"].isna().sum())
                    print(f"Warning: dropping {invalid_count} rows with invalid numeric Time (min)")
                    df = df.dropna(subset=["Time (min)"]).reset_index(drop=True)

                df["MinuteIndex"] = df["Time (min)"].astype(int)
                df["Hour"] = ((df["MinuteIndex"] // 60) % 24).astype(int)
                df["Minute"] = (df["MinuteIndex"] % 60).astype(int)
                df["DayIndex"] = (df["MinuteIndex"] // (60 * 24)).astype(int)

        else:
            print("Warning: No Time / Time (min) column found. Using row index as fallback.")
            idx = np.arange(len(df))
            df["MinuteIndex"] = idx.astype(int)
            df["Hour"] = ((idx // 60) % 24).astype(int)
            df["Minute"] = (idx % 60).astype(int)
            df["DayIndex"] = (idx // (24 * 60)).astype(int)

        return df

    def _fill_missing_required_columns(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()
        for col, default_val in self.required_columns_with_defaults.items():
            if col not in df.columns:
                print(f"Warning: missing required column '{col}', filling default value {default_val}.")
                df[col] = default_val
        return df

    def _fill_missing_consumption_columns(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()
        for col in self.consumption_columns:
            if col not in df.columns:
                df[col] = 0.0
        return df

    def _convert_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        numeric_columns = (
            list(self.required_columns_with_defaults.keys())
            + self.consumption_columns
            + ["Hour", "Minute", "DayIndex", "MinuteIndex"]
        )

        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        if "Hour" in df.columns:
            df["Hour"] = df["Hour"].astype(int)
        if "Minute" in df.columns:
            df["Minute"] = df["Minute"].astype(int)
        if "DayIndex" in df.columns:
            df["DayIndex"] = df["DayIndex"].astype(int)
        if "MinuteIndex" in df.columns:
            df["MinuteIndex"] = df["MinuteIndex"].astype(int)

        return df

    def _sample_rows(self, df: pd.DataFrame, sample_size: Optional[int]) -> pd.DataFrame:

        if sample_size is None:
            return df
        if sample_size >= len(df):
            return df
        return df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    def _print_basic_check(self, df: pd.DataFrame) -> None:

        print(f"Final shape: {df.shape}")
        print("Key columns check:")
        print(" - Time exists:", "Time" in df.columns)
        print(" - Time (min) exists:", "Time (min)" in df.columns)
        print(" - Hour exists:", "Hour" in df.columns)
        print(" - Minute exists:", "Minute" in df.columns)
        print(" - DayIndex exists:", "DayIndex" in df.columns)
        print(" - MinuteIndex exists:", "MinuteIndex" in df.columns)
        print(" - Purchased_Power(MW) exists:", "Purchased_Power(MW)" in df.columns)
        print(" - Purchased_Steam_tph exists:", "Purchased_Steam_tph" in df.columns)

        if len(df) > 0:
            print(" - Hour range:", int(df["Hour"].min()), "to", int(df["Hour"].max()))
            print(" - DayIndex range:", int(df["DayIndex"].min()), "to", int(df["DayIndex"].max()))
            print(" - MinuteIndex range:", int(df["MinuteIndex"].min()), "to", int(df["MinuteIndex"].max()))

    def load_and_preprocess_data(self, file_path: str, sample_size: Optional[int] = None) -> pd.DataFrame:

        print(f"Loading data file: {file_path}")

        df = self._read_file(file_path)
        print(f"Raw shape: {df.shape}")

        df = self.translate_columns_to_english_schema(df)
        print("Column normalization done.")

        df = self._process_time_columns(df)
        df = self._fill_missing_required_columns(df)
        df = self._fill_missing_consumption_columns(df)
        df = self._convert_numeric_columns(df)
        df = self._sample_rows(df, sample_size)

        self._print_basic_check(df)
        return df

if __name__ == "__main__":
    loader = DataLoader()
    df = loader.load_and_preprocess_data("data_of_IES.xlsx", sample_size=5)

    print("\nPreview:")
    preview_cols = [c for c in ["Time", "Time (min)", "Hour", "Minute", "DayIndex", "MinuteIndex"] if c in df.columns]
    print(df[preview_cols].head())
