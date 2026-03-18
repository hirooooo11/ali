from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class FeatureConfig:
    label_col: str = "is_trade"
    timestamp_col: str = "context_timestamp"
    id_col: str = "instance_id"

   
    id_like_cols: Tuple[str, ...] = ("user_id", "item_id", "shop_id", "item_brand_id")
    list_cols: Tuple[str, ...] = ("item_category_list", "item_property_list")


class FeaturePipeline:
    def __init__(self, cfg: Optional[FeatureConfig] = None):
        self.cfg = cfg or FeatureConfig()
        self._count_maps: Dict[str, pd.Series] = {}
        self._cross_count_map: Optional[pd.Series] = None

    @staticmethod
    def safe_list(x) -> int:
        if pd.isna(x):
            return 0
        s = str(x).strip()
        if s == "":
            return 0
        if ";" in s:
            return len([t for t in s.split(";") if t != ""])
        if "," in s:
            return len([t for t in s.split(",") if t != ""])
        return 1

    def fit(self, df: pd.DataFrame) -> "FeaturePipeline":
        df = df.copy()
        for col in self.cfg.id_like_cols:
            if col in df.columns:
                self._count_maps[col] = df[col].value_counts(dropna=False)


        if "user_id" in df.columns and "item_id" in df.columns:
            key = df["user_id"].astype(str) + "_" + df["item_id"].astype(str)
            self._cross_count_map = key.value_counts(dropna=False)

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if self.cfg.timestamp_col in df.columns:
            ts = pd.to_datetime(df[self.cfg.timestamp_col], unit="s", errors="coerce")
            df["hour"] = ts.dt.hour.astype("float32")
            df["dayofweek"] = ts.dt.dayofweek.astype("float32")


        for col in self.cfg.list_cols:
            if col in df.columns:
                df[col + "_newn"] = df[col].map(self.safe_list).astype("float32")



        for col, vc in self._count_maps.items():
            if col in df.columns:
                df[col + "_new"] = df[col].map(vc).fillna(0).astype("float32")


        if self._cross_count_map is not None and "user_id" in df.columns and "item_id" in df.columns:
            key = df["user_id"].astype(str) + "_" + df["item_id"].astype(str)
            df["user_item_new"] = key.map(self._cross_count_map).fillna(0).astype("float32")


        drop_cols = []
        for col in list(self.cfg.id_like_cols) + [self.cfg.id_col]:
            if col in df.columns:
                drop_cols.append(col)
        if drop_cols:
            df = df.drop(columns=drop_cols)

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
