# Week 5

> 目标 ：在 Week04 逻辑回归 baseline 的基础上，**理清“纯 ID 和 真特征”**，并在 Week05 中优先保留更可泛化、更可能提升转化率的特征；对纯 ID 进行剔除或替代编码（如 count encoding），再评估模型表现。  

---

## 1. Week4

**Week04 模型**：Logistic Regression 
- AUC: **0.6666**
- LogLoss: **0.6119**
- Accuracy: **0.6722**
- Precision: **0.0272**
- Recall: **0.5649**
- F1: **0.0518**
- Confusion matrix: [[63425, 30685], [660, 857]]  
解释：召回率较高，但误报很多，precision 很低。

---

## 2. Week5

### 2.1 特征逻辑澄清：ID 和 真特征
**中文**  
- **纯 ID**：`instance_id`, `user_id`, `item_id`, `shop_id`, `item_brand_id`  
  - 问题：高基数、维度爆炸、容易“记住 ID”，泛化差。  
- **更可泛化的真特征**：价格等级、销量等级、店铺评分、用户画像、时间特征、统计类特征（count/cross-count）、列表长度等。  
- **策略（Week05 采用）**：用 **count encoding / cross-count** 替代 raw ID。
---

## 3. 本周改动

### 3.1 保留
- 时间特征 / Time features：`hour`, `dayofweek`

### 3.2 新增
- **Count Encoding**（频次特征）  
  - `user_id_new`, `item_id_new`, `shop_id_new`, `item_brand_id_new`
- **Cross-count**（交叉频次）  
  - `user_item_new`（用户-商品组合出现次数）
- **List Length**（列表长度特征）  
  - `item_category_list_newn`, `item_property_list_newn`


### 3.3 剔除
- 不再直接把高基数 ID one-hot：`user_id`, `item_id`, `shop_id`, `item_brand_id`  
- 不把 `instance_id` 当特征使用。

---

## 4. 本周结果

**Dataset split**：Time split（与 Week04 保持一致）  
- Train size: **382,511**
- Valid size: **95,627**

**本周跑出来的结果**：
- AUC: **0.5458**
- LogLoss: **0.6729**
- Accuracy: **0.9841**
- Precision: **0.0000**
- Recall: **0.0000**
- F1: **0.0000**
- Threshold: **0.5**

---

## 5. 对比分析

| Metric | Week04 (baseline) | Week05 (ID cleaned + stats) | Change |
|---|---:|---:|---|
| AUC | 0.6666 | 0.5458 | ↓ |
| LogLoss | 0.6119 | 0.6729 | ↑ |
| Accuracy | 0.6722 | 0.9841 | ↑ |
| Precision | 0.0272 | 0.0000 | ↓ |
| Recall | 0.5649 | 0.0000 | ↓ |
| F1 | 0.0518 | 0.0000 | ↓ |

**关键解读**
- Week05 的 **Accuracy 变得非常高（0.984）**，但这是**极度类别不平衡**下的典型假象：只要模型把几乎所有样本都预测为 0（不买），accuracy 就会很高。  
- Precision/Recall/F1 全为 0 说明：在阈值 0.5 下，模型**没有预测出任何正类**（全部判为负类），因此 TP=0。

---

## 6. 为什么这次反而变差

### 6.1 可能原因 1：信息损失
Week04 使用了大量类别特征（包含强区分信号），虽然有高基数问题，但模型还能“抓到一些正样本”。  
Week05 删除了 raw ID 后，**只用 count/cross-count 可能不足以替代原始区分能力**，导致排序能力（AUC）下降。

### 6.2 可能原因 2：特征进入模型的方式过于保守
为了让 Week05 稳定跑通，我把生成后的特征多数当 numeric 使用；如果同时把一些**低基数 categorical（如 gender/occupation/page_id 等）**加入 one-hot，可能会显著改善。

### 6.3 可能原因 3：阈值 0.5 不适用于极不平衡任务
即使模型能区分排序（AUC>0.5），但在阈值 0.5 下仍可能全判负类。  
Week05 需要额外做：**阈值调优**（例如选使 F1 最大的阈值，或根据 PR curve 选点）。

---


## 7. 结论

Week05 按要求完成了“特征逻辑清晰化”和“纯 ID 剔除/替代编码”，但当前实现导致模型在阈值 0.5 下完全不预测正类，AUC 与 F1 等指标明显下降。

---
