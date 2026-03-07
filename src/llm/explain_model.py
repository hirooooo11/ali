import json
import pandas as pd
import joblib
from pathlib import Path

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


    cases_info = {}
    core_keys = top_features["top_positive"] + top_features["top_negative"]
    

    for name, sample in cases.items():
        if not sample.empty:
            prob = sample.iloc[0]['real_prob']
            raw_features = sample.iloc[0].to_dict()
            key_features = {k: v for k, v in raw_features.items() if k in core_keys and pd.notna(v) and v != 0}
            cases_info[name] = {"预测概率": f"{prob:.2%}", "关键特征值": key_features}
            
            print(f"【{name}】")
            print(f"• 模型预测概率: {prob:.2%}")
            print(f"• 关键特征值: {key_features}\n")


    prompt = f"""
你是一位资深电商业务分析师。请根据以下全局特征重要性和3个样本的真实数据，生成结构化业务解释。

【模型全局特征 (Top5正负向)】：
{json.dumps(top_features, ensure_ascii=False, indent=2)}

【样本真实输入特征与概率】：
{json.dumps(cases_info, ensure_ascii=False, indent=2)}

生成要求：
1. 不允许编造因果关系。
2. 只允许基于输入的特征进行解释。
3. 结构化业务解释（每个不超过200字）。
4. 输出格式固定：
[样本名称]
业务解释：[用大白话解释模型为什么这样判断]
"""
    print(prompt.strip())

if __name__ == "__main__":
    main()