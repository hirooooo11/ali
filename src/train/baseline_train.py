import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder



@dataclass
class Config:
    random_seed: int
    train_path: str
    sep: str
    label_col: str
    timestamp_col: str
    id_col: str
    missing_value_sentinel: int
    split_strategy: str
    valid_ratio: float
    numeric_features: List[str]
    categorical_features: List[str]
    time_features: List[str]
    model_params: dict
    artifacts_dir: str
    train_csv: str
    valid_csv: str
    model_path: str
    metrics_path: str


def load_config(cfg_path: str = "configs/baseline.yaml") -> Config:
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return Config(
        random_seed=int(raw["random_seed"]),
        train_path=raw["data"]["train_path"],
        sep=raw["data"]["sep"],
        label_col=raw["data"]["label_col"],
        timestamp_col=raw["data"]["timestamp_col"],
        id_col=raw["data"]["id_col"],
        missing_value_sentinel=int(raw["data"]["missing"]),
        split_strategy=raw["split"]["strategy"],
        valid_ratio=float(raw["split"]["valid_ratio"]),
        numeric_features=list(raw["features"]["numeric"]),
        categorical_features=list(raw["features"]["categorical"]),
        time_features=list(raw["features"]["time_features"]),
        model_params=dict(raw["model"]["params"]),
        artifacts_dir=raw["output"]["artifacts_dir"],
        train_csv=raw["output"]["train_csv"],
        valid_csv=raw["output"]["valid_csv"],
        model_path=raw["output"]["model_path"],
        metrics_path=raw["output"]["metrics_path"],
    )


def ensure_parent_dir(path_str: str) -> None:
    p = Path(path_str)
    p.parent.mkdir(parents=True, exist_ok=True)


def ensure_dir(dir_str: str) -> None:
    Path(dir_str).mkdir(parents=True, exist_ok=True)



def read_raw_data(train_path: str, sep: str) -> pd.DataFrame:
    df = pd.read_csv(train_path, sep=sep, engine="python")
    return df


def add_time_features(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    df = df.copy()

    ts = pd.to_datetime(df[timestamp_col], unit="s", errors="coerce")
    df["hour"] = ts.dt.hour
    df["dayofweek"] = ts.dt.dayofweek  
    return df


def missing_sentinel(
    df: pd.DataFrame, cols: List[str], sentinel: int
) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].replace(sentinel, np.nan)
    return df


def time_split(
    df: pd.DataFrame, timestamp_col: str, valid_ratio: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_sorted = df.sort_values(timestamp_col).reset_index(drop=True)
    n = len(df_sorted)
    n_valid = int(n * valid_ratio)
    n_train = n - n_valid

    train_df = df_sorted.iloc[:n_train].copy()
    valid_df = df_sorted.iloc[n_train:].copy()
    return train_df, valid_df



def build_pipeline(
    numeric_cols: List[str],
    categorical_cols: List[str],
    model_params: dict,
) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    clf = LogisticRegression(
        max_iter=int(model_params.get("max_iter", 200)),
        class_weight=model_params.get("class_weight", None),
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", clf),
        ]
    )
    return pipe


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "logloss": float(log_loss(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
    }
    return metrics



def main():
    cfg = load_config("configs/baseline.yaml")
    np.random.seed(cfg.random_seed)
    df = read_raw_data(cfg.train_path, cfg.sep)
    df = add_time_features(df, cfg.timestamp_col)


    used_numeric = [c for c in cfg.numeric_features + cfg.time_features if c in df.columns]
    used_categorical = [c for c in cfg.categorical_features if c in df.columns]
    used_cols = used_numeric + used_categorical + [cfg.label_col, cfg.timestamp_col, cfg.id_col]
    used_cols = [c for c in used_cols if c in df.columns]
    df = df[used_cols].copy()


    df = missing_sentinel(
        df,
        cols=used_numeric + used_categorical,
        sentinel=cfg.missing_value_sentinel,
    )
    for c in used_numeric:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if cfg.split_strategy == "time":
        train_df, valid_df = time_split(df, cfg.timestamp_col, cfg.valid_ratio)
    else:
        df_shuffled = df.sample(frac=1.0, random_state=cfg.random_seed).reset_index(drop=True)
        n = len(df_shuffled)
        n_valid = int(n * cfg.valid_ratio)
        train_df = df_shuffled.iloc[:-n_valid].copy()
        valid_df = df_shuffled.iloc[-n_valid:].copy()


    X_train = train_df[used_numeric + used_categorical]
    y_train = train_df[cfg.label_col].astype(int).values

    X_valid = valid_df[used_numeric + used_categorical]
    y_valid = valid_df[cfg.label_col].astype(int).values


    pipe = build_pipeline(used_numeric, used_categorical, cfg.model_params)
    pipe.fit(X_train, y_train)


    y_prob = pipe.predict_proba(X_valid)[:, 1]
    metrics = compute_metrics(y_valid, y_prob, threshold=0.5)


    ensure_parent_dir(cfg.artifacts_dir)
    ensure_parent_dir(cfg.train_csv)
    ensure_parent_dir(cfg.valid_csv)
    ensure_parent_dir(cfg.model_path)
    ensure_parent_dir(cfg.metrics_path)

    train_df.to_csv(cfg.train_csv, index=False)
    valid_df.to_csv(cfg.valid_csv, index=False)

    joblib.dump(pipe, cfg.model_path)

    with open(cfg.metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)



    print(f"Train size: {len(train_df)} | Valid size: {len(valid_df)}")
    print("Valid metrics:", metrics)
    print("Saved to:", cfg.artifacts_dir)


if __name__ == "__main__":
    main()
