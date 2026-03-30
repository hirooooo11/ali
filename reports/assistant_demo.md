# 广告转化分析助手示例问答

**1. 数据集整体转化率是多少?**

**价值说明**：用于帮助业务团队了解当前大盘的基础转化水平，为后续模型优化设定基准线。
* **Answer**:
1.89%

* **Sources**:
- eda_report.md (1. 数据分析摘要)
- model_explain.md (3. 风险声明（相关性 ≠ 因果）)







**2. 哪些因素会提高转化概率?**

**价值说明**：用于帮助运营与产品团队明确转化痛点，指导精细化的流量分配与页面展示策略。
* **Answer**: 
影响转化率的因素（基于历史数据的统计关联）包括：

- **店铺维度**：高评价等级（如`shop_review_num_level_21.0`）、异常高星等级（如`shop_star_level_5018.0`）、物流与服务评分（`shop_score_delivery`, `shop_score_service`）呈负向关联；  
- **商品维度**：高价（`item_price_level`）、非优势产地（多个`item_city_id_*`特征）、属性列表过短（`_newn`为正向，说明长度不足可能拉低转化）；  
- **流量来源维度**：特定落地页（如`context_page_id_4017.0`, `4011.0`, `4014.0`等）用户购买意图弱；  
- **用户维度**：特定职业人群（`user_occupation_id_2003.0`）转化意愿低；  
- **时间维度**：`hour`和`dayofweek`对转化有轻微正向提升（AUC微升），表明时段存在影响。

⚠️ 注意：以上均为相关性发现，不等于因果关系；实际优化需通过A/B测试验证。

* **Sources**: 
- model_explain.md (3. 风险声明（相关性 ≠ 因果）)
- exp_week05.md (3.1 正向特征分析 (Exp 2 & Exp 4))







**3. Week5 的特征工程带来了什么改进?**

**价值说明**：向算法与数据团队汇报模型迭代的直接收益，验证特征工程方向的有效性。
* **Answer**:
Week5 的特征工程通过引入时间特征（Exp 2）和列表长度特征（Exp 4），使验证集 AUC 从基线 0.6616 提升至 **0.6647**（+0.0031），LogLoss 同步下降，表明模型判别能力增强、预测更稳健；而 ID 频次特征（Exp 3）未带来增益，已被剔除。

* **Sources**: 
- model_explain.md (TOP20正向影响特征)
- exp_week05.md (3. 本周实验结果与分析)







**4. 模型最重要的特征是什么?**

**价值说明**：用于指导电商选品和流量倾斜策略，明确“爆款”逻辑对转化的核心驱动力。
* **Answer**: 
item_sales_level

* **Sources**: 
- model_explain.md (1. 样本结构化业务解释)
- eda_report.md (3. Baseline 建设建议)
- exp_week04.md (9. 下一步)







**5. 为什么用户活跃度会影响转化?**

**价值说明**：用于启发用户生命周期运营，评估促活手段（如签到、Push）对最终成单的潜在商业价值。
* **Answer**: 
Not enough information

* **Sources**: 
- model_explain.md (TOP20正向影响特征)
- data_dictionary.md (data_dictionary)
- eda_report.md (1. 数据分析摘要)








**6. 哪些商品更容易成交?**

**价值说明**：辅助招商与供应链团队优化商品结构，优先引入和扶持具备高转化潜质的品类。
* **Answer**: 
根据现有信息，**高销量等级（item_sales_level均值11.16，热销商品占主导）、高收藏等级（item_collected_level 0–17）、高浏览量等级（item_pv_level 0–21）和高价格等级（item_price_level 0–17）的商品更可能成交**；但需注意：  
- 数据未提供各特征与成交（is_trade）的直接相关性分析；  
- 标签极度不均衡（转化率仅1.89%），且无交叉分析或显著性检验支持具体商品ID/类目/品牌与成交的强关联。  

→ **不能明确识别“哪些具体商品”更容易成交，仅可推断高热度、高价格等级的商品在样本中更常见于成交场景。**

* **Sources**: 
- data_dictionary.md (data_dictionary)
- eda_report.md (（2）长尾分布风险：显著存在于多个维度)
- exp_week05.md (3.2 负向特征分析 (Exp 3 ))








**7. 为什么评分高的店铺转化率更高?**

**价值说明**：用于建立更科学的商家评级与流量分配机制，激励平台商家主动提升服务质量。
* **Answer**: 
因为高评分店铺（如5013星、5006星）在模型中被识别为显著正向特征，表明用户更信任其服务、描述和发货质量，从而降低决策阻力、提升购买意愿。

* **Sources**: 
- data_dictionary.md (data_dictionary)
- eda_report.md (1. 数据分析摘要)
- model_explain.md (TOP20正向影响特征)







**8. Week6 训练框架做了什么改进?**

**价值说明**：用于记录工程架构与训练流程的演进，为后续的模型复用和团队协作沉淀技术资产。
* **Answer**: 
Not enough information

* **Sources**: 
- model_explain.md (1. 样本结构化业务解释)
- eda_report.md (3. Baseline 建设建议)







**9. 哪些特征对模型帮助不大?**

**价值说明**：用于指导特征库的瘦身与降维，降低服务器计算成本并防止模型过拟合。
* **Answer**: 
- `shop_review_positive_rate`：均值达 0.99，75% 分位为 1.0，取值高度集中于高位，区分能力弱；  
- `user_gender_id=0` 和 `user_occupation_id=2005`：分别占比超 75% 和 62.7%，类别极度集中，信息熵低，原始编码下贡献有限；  
- 原始高基数类别特征（如完整 `item_category_list`、`item_city_id`、`user_id`）：未降维或编码前易导致稀疏性高、噪声大，直接影响模型效能。

* **Sources**: 
- eda_report.md (3. Baseline 建设建议)
- exp_week04.md (9. 下一步)







**10. 如果要提升转化率,运营应该关注什么?**

**价值说明**：直接为一线运营团队提供可落地的策略建议，驱动平台整体业绩增长。
* **Answer**: 
运营应关注**价格竞争力与销量热度的协同优化**：优先推动中低价位（如 item_price_level ≤ 4）且具备一定市场热度（item_sales_level ≥ 15.0）的商品曝光与转化；同时，避免孤立调价或刷销量——需通过 A/B 测试验证因果效果，并结合用户画像、类目特征及时间趋势（如大促日）综合施策。

* **Sources**: 
- model_explain.md (1. 样本结构化业务解释)
- eda_report.md (3. Baseline 建设建议)