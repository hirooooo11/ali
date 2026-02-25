import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def load_config(cfg_path: str = "configs/baseline.yaml") -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path_str: str) -> None:
    p = Path(path_str)
    p.parent.mkdir(parents=True, exist_ok=True)


def main():
    cfg = load_config("configs/baseline.yaml")

    valid_path = cfg["output"]["valid_csv"]
    model_path = cfg["output"]["model_path"]
    label_col = cfg["data"]["label_col"]


    valid_df = pd.read_csv(valid_path)
    y_true = valid_df[label_col].astype(int).values


    numeric = list(cfg["features"]["numeric"])
    time_feats = list(cfg["features"]["time_features"])
    categorical = list(cfg["features"]["categorical"])

    used_numeric = [c for c in numeric + time_feats if c in valid_df.columns]
    used_categorical = [c for c in categorical if c in valid_df.columns]

    X_valid = valid_df[used_numeric + used_categorical]


    model = joblib.load(model_path)


    y_prob = model.predict_proba(X_valid)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred).tolist()

    metrics = {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "logloss": float(log_loss(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": 0.5,
        "confusion_matrix": {
            "format": "[[TN, FP], [FN, TP]]",
            "value": cm,
        },
        "n_valid": int(len(valid_df)),
        "pos_rate_valid": float(np.mean(y_true)),
    }

    out_path = str(Path(cfg["output"]["artifacts_dir"]) / "metrics_full_valid.json")
    ensure_dir(out_path)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


    print("Valid metrics:", metrics)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
