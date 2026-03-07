import json
import joblib
from pathlib import Path
import sys

def main():
    out_base = Path("outputs")
    valid_dirs = [d for d in out_base.iterdir() if d.is_dir() and (d / "model.joblib").exists()]
    latest_dir = sorted(valid_dirs)[-1]

    pipe = joblib.load(latest_dir / "model.joblib")
    

    preprocessor = pipe.named_steps['preprocess']
    clf = pipe.named_steps['clf']
    

    num_cols = list(preprocessor.transformers_[0][2])
    cat_cols = list(preprocessor.transformers_[1][2])
    

    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_names = cat_encoder.get_feature_names_out(cat_cols)
    
    feature_names = num_cols + list(cat_names)
    coefficients = clf.coef_[0]
    

    importance = [{"feature": n, "weight": float(c)} for n, c in zip(feature_names, coefficients)]
    
    pos_features = sorted([f for f in importance if f["weight"] > 0], key=lambda x: x["weight"], reverse=True)
    neg_features = sorted([f for f in importance if f["weight"] < 0], key=lambda x: x["weight"])
    

    result_json = {
        "top_20_positive": pos_features[:20],
        "top_20_negative": neg_features[:20]
    }
    
    json_path = out_base / "feature_importance.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=4, ensure_ascii=False)
        

if __name__ == "__main__":
    main()