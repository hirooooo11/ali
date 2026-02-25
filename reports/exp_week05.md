# Week 5

> **实验目标**：在 Week 04 逻辑回归模型的基础上，首先对存在缺陷的初始 `yaml` 特征配置进行修正，建立可靠的基线模型；随后，严格遵循“一次只加入一类特征”的原则，进行前向特征选择消融实验，评估不同特征组合对转化率预测的实际效果。

---

## 1. 初始 Baseline 配置的修正 (YAML 调整)

复盘 Week 04 的实验时，我们发现了初始 `baseline.yaml` 配置中存在的几个明显不合理之处，并在本周的第一步进行了修正：

### 1.1 剔除高风险特征（防止过拟合与维度爆炸）
- **高基数 ID 特征**：移除了 `user_id`, `item_id`, `shop_id`, `item_brand_id`。这些纯 ID 特征的类别数量极大，直接作为类别特征进行 One-Hot 编码放入逻辑回归，会导致模型维度爆炸并产生严重的过拟合，大幅降低泛化能力。
- **复合字符串特征**：移除了 `item_category_list` 和 `item_property_list`。它们是带有分号的字符串序列（如"颜色:红;尺寸:L"），直接作为单一类别入模会产生海量无效的稀疏矩阵。
- **缺失/无效特征**：移除了数据字典和原数据中不存在的 `user_pv_level` 和 `user_city_id`，以保证代码正常运行。

### 1.2 补充关键连续型特征
结合 EDA 报告和数据字典，我们发现原配置遗漏了多项重要的店铺评分连续型特征。因此，在 Base 配置的 `numeric` 列表中补充了：
- `shop_review_positive_rate` (店铺好评率)
- `shop_score_service` (服务评分)
- `shop_score_delivery` (物流评分)
- `shop_score_description` (描述评分)

**修正后的 Base 初始特征名单**：
- **Numeric**: `item_price_level`, `item_sales_level`, `item_collected_level`, `item_pv_level`, `user_age_level`, `user_star_level`, `shop_review_positive_rate`, `shop_score_service`, `shop_score_delivery`, `shop_score_description`
- **Categorical**: `user_gender_id`, `user_occupation_id`, `context_page_id`, `item_city_id`, `shop_review_num_level`, `shop_star_level`

---

## 2. 实验设计：前向特征选择

为验证通过 `FeaturePipeline` 生成的衍生特征的有效性，本周采用前向特征选择方法进行消融实验：
- 每次只在上一轮表现最优的配置上**新增一类特征**。
- 若验证集 AUC 提升，则**保留**该类特征进入下一轮；
- 若 AUC 下降或持平，则判定为负向/无效特征，**予以剔除**，不带入下一轮测试。

### 实验组别规划：
1. **Exp 1 (Base)**：修正后的基础数值与低基数类别特征（不包含时间与统计特征）。
2. **Exp 2 (+ Time)**：在 Base 基础上，加入时间特征（`hour`, `dayofweek`）。
3. **Exp 3 (+ Count)**：在 Exp 2 基础上，尝试加入 ID 频次统计特征（`user_id_new` 等）。
4. **Exp 4 (+ List Length)**：剔除导致模型效果下降的频次特征，在 Exp 2 基础上加入列表长度特征（`item_category_list_newn` 等）。

---

## 3. 本周实验结果与分析

**Dataset split**：Time split
- Train size: **382,511** | Valid size: **95,627**
- 模型: Logistic Regression (`max_iter=1000`, `class_weight='balanced'`)
- 核心评估指标: **AUC**

| 实验组别 | 包含的特征组合 | 验证集 AUC | LogLoss | 结论与动作 |
| :--- | :--- | :--- | :--- | :--- |
| **Exp 1: Base** | 修正后的基础基线 | 0.6616 | 0.6480 | 确立基础指标（对比 Week05 初期的 0.5458 有显著提升） |
| **Exp 2: + Time** | Base + 时间特征 | **0.6641** | 0.6407 |  **指标提升 (+0.0025)，保留该组特征** |
| **Exp 3: + Count** | Exp 2 + ID频次特征 | 0.6616 | 0.6405 |  指标下降**予以剔除** |
| **Exp 4: + List Length**| Exp 2 + 列表长度特征 | **0.6647** | 0.6384 |  **指标进一步提升 (+0.0006)，保留该组特征** |

### 3.1 正向特征分析 (Exp 2 & Exp 4)
1. **时间特征的有效性**：引入 `hour` 和 `dayofweek` 后，AUC 从 0.6616 提升至 0.6641。这表明用户的点击时间特征（如具体的钟点或星期几）与最终的购买转化率存在相关性。
2. **列表长度特征的有效性**：加入 `_newn` 列表长度特征后，模型达到了本次实验的**最高分 0.6647**。从业务角度解释，商品属性列表的长度通常反映了商品描述的详尽程度，较长的属性列表可能有助于提高用户的信任度，进而对转化率产生正向影响。

### 3.2 负向特征分析 (Exp 3 )
在引入 `user_id_new` 等频次统计特征时，模型触发了 `ConvergenceWarning: lbfgs failed to converge` 警告，且 AUC 回落至 0.6616。
- **量纲差异问题**：频次特征的数值范围较大（部分可达数百上千），而原有的店铺评分等特征值域在 0-1 之间。特征尺度差异过大导致逻辑回归的梯度下降过程难以在 1000 次迭代内收敛。  
- **线性模型的局限**：原始的频次极值对于线性模型而言容易成为噪声，通常需要通过对数转换或分箱、标准化处理后，才能在逻辑回归中发挥较好的作用。

