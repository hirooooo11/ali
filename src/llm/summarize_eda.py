import os
import json
from openai import OpenAI


REPORT_DIR = 'D:/ali/reports'
JSON_FILE = os.path.join(REPORT_DIR, 'eda_stats.json')
OUTPUT_FILE = os.path.join(REPORT_DIR, 'eda_report.md')


client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"), 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def read_json():
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def ask_qwen(stats_data):
    prompt = f"""
    这是我用 Python 跑出来的 EDA 统计数据：

    {json.dumps(stats_data, ensure_ascii=False, indent=2)}

    请根据数据，帮我生成一份数据分析报告（Markdown格式），必须包含以下三部分内容：

    1. 数据分析摘要：用简练的语言总结这份数据的整体情况。
    2. 数据质量风险点：请重点分析是否存在缺失值严重、长尾分布、类别爆炸等问题。
    3. Baseline 建设建议：基于上述风险，对我接下来建立 Baseline 模型给出 4 条具体建议。

    请直接输出分析内容，不要包含代码。
    """



    completion = client.chat.completions.create(
        model="qwen-plus", 
        messages=[
            {'role': 'system', 'content': '你是一个专业的数据挖掘专家。'},
            {'role': 'user', 'content': prompt}
        ]
    )
    
    return completion.choices[0].message.content

def save_report(ai_content):

    final_report = f"""

1. 流量趋势图
这是根据时间戳统计的每日样本量：

![每日流量趋势](./figures/daily.png)

2. 智能分析摘要 (By Qwen)
{ai_content}
"""
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(final_report)
    


if __name__ == "__main__":
    data = read_json()
    
    if data:
        analysis_text = ask_qwen(data)
        save_report(analysis_text)