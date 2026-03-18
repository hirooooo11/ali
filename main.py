import argparse
import datetime
import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


from src.train.baseline_train import (
    load_config, read_raw_data, add_time_features, missing_sentinel,
    time_split, build_pipeline, compute_metrics
)
from src.feature.feature_pipeline import FeaturePipeline, FeatureConfig

def run_train(cfg):
    df = read_raw_data(cfg.train_path, cfg.sep)
    df = add_time_features(df, cfg.timestamp_col)

    potential_features = cfg.numeric_features + cfg.categorical_features + cfg.time_features
    df = missing_sentinel(
        df,
        cols=[c for c in potential_features if c in df.columns],
        sentinel=cfg.missing_value_sentinel,
    )
    for c in cfg.numeric_features:
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


    fp = FeaturePipeline(FeatureConfig(
        label_col=cfg.label_col,
        timestamp_col=cfg.timestamp_col,
        id_col=cfg.id_col,
    ))
    fp.fit(train_df)             
    train_df = fp.transform(train_df)
    valid_df = fp.transform(valid_df)

    used_numeric = [c for c in cfg.numeric_features + cfg.time_features if c in train_df.columns]
    used_categorical = [c for c in cfg.categorical_features if c in train_df.columns]
    
    X_train = train_df[used_numeric + used_categorical]
    y_train = train_df[cfg.label_col].astype(int).values

    X_valid = valid_df[used_numeric + used_categorical]
    y_valid = valid_df[cfg.label_col].astype(int).values


    pipe = build_pipeline(used_numeric, used_categorical, cfg.model_params)
    pipe.fit(X_train, y_train)


    y_prob = pipe.predict_proba(X_valid)[:, 1]
    metrics = compute_metrics(y_valid, y_prob, threshold=0.5)


    train_df.to_csv(cfg.train_csv, index=False)
    valid_df.to_csv(cfg.valid_csv, index=False)
    joblib.dump(pipe, cfg.model_path)
    joblib.dump(fp, Path(cfg.artifacts_dir) / "feature_pipeline.joblib")

    with open(cfg.metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Train size: {len(train_df)} | Valid size: {len(valid_df)}")
    print(f"Valid metrics: {metrics}")
    print(f"All artifacts saved to: {cfg.artifacts_dir}")


def main():
    parser = argparse.ArgumentParser(description="Standardized Training Framework")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml file") 
    parser.add_argument("--mode", type=str, choices=["train", "eval", "predict"], default="train", help="Run mode") 
    args = parser.parse_args()


    cfg = load_config(args.config)
    np.random.seed(cfg.random_seed)


    config_name = Path(args.config).stem  
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"outputs/{timestamp}_{config_name}")
    output_dir.mkdir(parents=True, exist_ok=True)


    cfg.artifacts_dir = str(output_dir)
    cfg.train_csv = str(output_dir / "train.csv")
    cfg.valid_csv = str(output_dir / "valid.csv")
    cfg.model_path = str(output_dir / "model.joblib")
    cfg.metrics_path = str(output_dir / "metrics_valid.json")


    print(f"Config : {args.config}")
    print(f"Mode   : {args.mode}")
    print(f"Outputs: {output_dir}")

    if args.mode == "train":
        run_train(cfg)

if __name__ == "__main__":
    main()