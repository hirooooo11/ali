1. Goal
本次实验的目标是搭建一个逻辑回归的 baseline 模型，用于预测用户是否会发生交易（is_trade）。通过这个简单模型，我们可以了解数据的基本表现，并为后续改进提供参考。
2. Configuration Summary (from baseline.yaml)
2.1 Data
训练数据路径：data/round1_ijcai_18_train_20180301.txt
分隔符：空格（\s+）
标签列：is_trade
时间戳列：context_timestamp
ID 列：instance_id
缺失值标记：-1
2.2 Split
切分策略：按时间顺序划分（time）
验证集比例：20%（valid_ratio: 0.2）
2.3 Features
Numeric:
item_price_level, item_sales_level, item_collected_level, item_pv_level, user_age_level, user_star_level, user_pv_level
Categorical:
item_id, item_brand_id, item_city_id, item_category_list, item_property_list, user_id, user_gender_id, user_city_id, user_occupation_id, context_page_id, shop_id, shop_review_num_level, shop_star_level
Time features:
hour, dayofweek
3. Preprocessing
缺失值处理：将 -1 视为缺失值（根据配置），但具体填充方式（如 numeric 用 median、categorical 用 most_frequent）未在 yaml 或输出中说明 → not provided
编码方式：未明确说明是否使用 OneHot 或其他编码 → not provided
特殊处理：已知 -1 被当作缺失值处理，但预处理细节未完整给出 → not provided
4. Model
Model name: logistic_regression
Key params: max_iter=1000, class_weight='balanced'
Notes about class imbalance: 使用了 class_weight='balanced' 来缓解正负样本不平衡问题（验证集中正样本率仅约 1.59%）
5. Train/Valid Statistics
Train size: 382,511
Valid size: 95,627
Positive rate (valid): 1.586% （即 is_trade=1 的样本占比）
6. Evaluation Results (Valid)
AUC: 0.6666
LogLoss: 0.6119
Accuracy: 0.6722
Precision: 0.0272
Recall: 0.5649
F1: 0.0518
Threshold: 0.5
Confusion matrix（格式：[[TN, FP], [FN, TP]]）: [[63425, 30685], [660, 857]]
解释：模型把很多负样本错判为正样本（FP 很高），导致 precision 很低；但能召回超过一半的正样本（recall ≈ 56.5%），说明在不平衡数据下更偏向“宁可错杀”。
7. Artifacts Saved
模型文件：artifacts/baseline/model.joblib
验证集指标：artifacts/baseline/metrics_valid.json（实际保存名为 metrics_full_valid.json，可能为笔误）
训练/验证数据快照：artifacts/baseline/train.csv 和 artifacts/baseline/valid.csv
8. How to Run
命令未提供 → not provided
（通常应包含类似 python train.py --config baseline.yaml 的指令）
9. Next Steps
尝试更好的特征工程，比如对类别特征做 target encoding。
考虑使用树模型（如 LightGBM）代替逻辑回归，可能更适合这种高维稀疏数据。
调整分类阈值（现在是 0.5），因为正样本太少，可能更低的阈值能提升 F1。
检查是否所有 categorical 特征都合理编码，有些高基数特征（如 user_id）可能需要特殊处理。
加入更多时间相关特征，比如是否周末、是否促销日等。
可视化特征重要性或 AUC 曲线，帮助理解模型弱点。