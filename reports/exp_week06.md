# CTR预测Baseline实验周报

## 📋 实验基本信息

- **实验编号**: baseline_03
- **模型**: Logistic Regression
- **日期**: 2026-02-26
- **数据集**: IJCAI-18 CTR Prediction

---

## 🎯 实验配置概览

### 数据划分
- **划分策略**: 时间序列划分 (Time-based Split)
- **验证集比例**: 20%
- **随机种子**: 42

### 特征工程
**数值特征 (12个)**:
- 商品特征: `item_price_level`, `item_sales_level`, `item_collected_level`, `item_pv_level`
- 用户特征: `user_age_level`, `user_star_level`
- 店铺特征: `shop_review_positive_rate`, `shop_score_service`, `shop_score_delivery`, `shop_score_description`
- 类别特征: `item_category_list_newn`, `item_property_list_newn`

**类别特征 (6个)**:
- `user_gender_id`, `user_occupation_id`, `context_page_id`, `item_city_id`, `shop_review_num_level`, `shop_star_level`

**时间特征 (2个)**:
- `hour`, `dayofweek`

### 模型参数
- **算法**: Logistic Regression
- **最大迭代次数**: 1000
- **类别权重**: balanced
- **分类阈值**: 0.5

---

## 📊 实验结果

| 指标 | 数值 | 评级 | 说明 |
|------|------|------|------|
| **AUC** | 0.6616 | ⚠️ 及格 | 具备基础排序能力 |
| **LogLoss** | 0.6480 | ⚠️ 偏高 | 概率校准待优化 |
| **Accuracy** | 0.6403 | 🟡 参考性弱 | 不平衡数据下意义有限 |
| **Precision** | 0.0266 | 🔴 极低 | 预测正例中仅2.66%真实点击 |
| **Recall** | 0.6098 | 🟡 尚可 | 召回61%真实点击 |
| **F1-Score** | 0.0510 | 🔴 极低 | 综合效果差 |

---

## 🔍 核心问题诊断

1. **样本极度不平衡**: Precision(2.66%)与Recall(60.98%)差距巨大，推断正样本占比<5%
2. **阈值设置不合理**: 默认0.5阈值不适配CTR低转化率场景
3. **特征表达力不足**: 缺乏交叉特征、统计特征、Target Encoding等高级特征
4. **模型容量有限**: LR线性模型难以捕捉复杂非线性交互

---

## 🚀 下一步优化建议

### P0（本周）
- [ ] **阈值调优**: 基于PR曲线搜索最优阈值(建议0.01~0.3)，目标Precision≥5%
- [ ] **补充评估指标**: 增加AUC-PR、GAUC、Calibration Curve

### P1（下周）
- [ ] **交叉特征**: 
  - `user_star_level × item_sales_level`
  - `shop_score_service × item_price_level`
  - `hour × dayofweek`
- [ ] **统计特征** (注意时间穿越):
  - 用户/商品/店铺历史CTR(滑动窗口)
  - Target Encoding (需OOF防过拟合)

### P2（下下周）
- [ ] **模型升级**: 尝试LightGBM/XGBoost，预期AUC+0.05~0.08
- [ ] **深度模型**: Wide&Deep / DeepFM 探索

### P3（长期）
- [ ] 样本策略: 负采样 / Focal Loss / 难例挖掘
- [ ] 特征筛选: 基于重要性移除低效特征
- [ ] 模型融合: Stacking / 时间窗口融合

---

## ⚠️ 风险提示

- 所有统计特征必须基于**历史时间窗口**计算，严禁使用未来信息
- Target Encoding必须采用**Out-of-Fold**策略生成
- 迭代实验需保持**验证集划分一致**，确保指标可比

---

## 📈 目标设定

| 周期 | AUC目标 | Precision目标 | 关键动作 |
|------|---------|---------------|----------|
| 短期(1-2周) | 0.70+ | 0.05+ | 阈值调优+交叉特征 |
| 中期(1月) | 0.75+ | 0.10+ | 树模型+统计特征 |
| 长期(3月) | 0.80+ | 0.15+ | 深度模型+AB测试 |

---

## 📝 本周总结

✅ **完成**: LR baseline全流程打通 + 时间序列验证框架 + 指标评估输出  
💡 **发现**: 样本不平衡严重 / 阈值需重调 / 特征工程空间大 / LR已达瓶颈  
🎯 **下周重点**: 阈值搜索(1d) + 交叉特征(2-3d) + 统计特征(2-3d) + LightGBM实验(2d)

---

> 报告生成: 2026-02-26 | 实验版本: baseline_03 | 下次实验: baseline_04