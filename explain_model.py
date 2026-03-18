import json
import pandas as pd
import joblib
from pathlib import Path

def explanation_prompt(top_features: dict, all_cases_data: dict) -> str:
    prompt = f"""
你是一位资深电商业务分析师。请根据以下样本的真实数据，为每个样本分别生成结构化业务解释。
【样本的真实数据】：
{json.dumps(all_cases_data, ensure_ascii=False, indent=2)}

生成要求：
1. 结构化业务解释（每个案例不超过 200 字）。
2. 不允许编造因果关系。
3. 只允许基于输入的特征进行解释。
4. 请一次性输出所有案例的解释，输出格式必须严格如下：


关键特征值：[列出数据中的关键特征]
模型预测概率：[列出预测概率]
简要解释模型为什么这样判断以及特征重要性


"""
    return prompt.strip()

def main():
    with open("outputs/feature_importance.json", "r", encoding="utf-8") as f:
        importance_data = json.load(f)
        top_features = {
            "top_positive": [x["feature"] for x in importance_data.get("top_20_positive", [])[:5]],
            "top_negative": [x["feature"] for x in importance_data.get("top_20_negative", [])[:5]]
        }


    out_base = Path("outputs")
    latest_dir = sorted([d for d in out_base.iterdir() if d.is_dir() and (d / "model.joblib").exists()])[-1]
    pipe = joblib.load(latest_dir / "model.joblib")
    df = pd.read_csv(latest_dir / "valid.csv")


    preprocessor = pipe.named_steps['preprocess']
    num_cols = list(preprocessor.transformers_[0][2])
    cat_cols = list(preprocessor.transformers_[1][2])
    df['real_prob'] = pipe.predict_proba(df[num_cols + cat_cols])[:, 1]


    cases = {
        "1. 高预测概率样本": df[(df['real_prob'] > 0.8) & (df['is_trade'] == 1)].sort_values('real_prob', ascending=False).head(1),
        "2. 低预测概率样本": df[(df['real_prob'] < 0.1) & (df['is_trade'] == 0)].sort_values('real_prob', ascending=True).head(1),
        "3. 误判样本": df[(df['real_prob'] > 0.5) & (df['is_trade'] == 0)].sort_values('real_prob', ascending=False).head(1)
    }

    core_keys = top_features["top_positive"] + top_features["top_negative"]
    all_cases_data = {}
    

    for name, sample in cases.items():
        if not sample.empty:
            prob = sample.iloc[0]['real_prob']
            raw_features = sample.iloc[0].to_dict()
            key_features = {k: v for k, v in raw_features.items() if k in core_keys and pd.notna(v) and v != 0}
            all_cases_data[name] = {
                "模型预测概率": f"{prob:.2%}",
                "关键特征值": key_features
            }
            

    prompt = explanation_prompt(top_features, all_cases_data)
    print(prompt)

if __name__ == "__main__":
    main()