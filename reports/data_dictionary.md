| 字段名                       | 建议类型   | 业务含义推测                                                                 |
|:--------------------------|:---------|:-------------------------------------------------------------------------|
| instance_id               | int      | 实例唯一标识符（用于区分不同样本）                                                       |
| item_id                   | category | 商品ID（唯一标识商品）                                                                |
| item_category_list        | category | 商品类目列表（如"家电;手机"，用分号分隔的类目ID组合）                                         |
| item_property_list        | category | 商品属性列表（如"颜色:红色;尺寸:L"，描述商品具体属性）                                         |
| item_brand_id             | category | 商品品牌ID（唯一标识品牌）                                                          |
| item_city_id              | category | 商品所在城市ID（商品发布或库存所在城市）                                                  |
| item_price_level          | category | 商品价格等级（0-17，0为最低价，17为最高价，脱敏后等级）                                      |
| item_sales_level          | category | 商品销量等级（-1表示无数据，0-17表示销量从低到高）                                         |
| item_collected_level      | category | 商品收藏等级（0-17，表示用户收藏热度）                                                |
| item_pv_level             | category | 商品浏览量等级（0-21，表示页面浏览量热度）                                             |
| user_id                   | category | 用户ID（唯一标识用户）                                                              |
| user_gender_id            | category | 用户性别ID（0:男, 1:女, 2:未知, 3:其他）                                           |
| user_age_level            | category | 用户年龄分段等级（-1表示未知，其他为年龄段编码，如0-18岁、19-25岁等）                           |
| user_occupation_id        | category | 用户职业ID（5个类别，如学生、上班族、自由职业等）                                           |
| user_star_level           | category | 用户会员星级等级（-1表示无数据，其他为星级编码，如1-5星）                                     |
| context_id                | int      | 上下文会话ID（如页面会话唯一标识）                                                     |
| context_timestamp         | timestamp | 用户点击发生的时间戳（实际业务时间，需转换为datetime）                                       |
| context_page_id           | category | 页面ID（如首页、列表页、详情页，共20类）                                              |
| predict_category_property | category | 基于用户查询词预测的类目属性（如"女装;连衣裙"，用于广告与用户搜索词的匹配）                    |
| shop_id                   | category | 店铺ID（唯一标识店铺）                                                              |
| shop_review_num_level     | category | 店铺评论数量等级（0-25，0表示无评论，25表示评论量最高）                                       |
| shop_review_positive_rate | float    | 店铺正评率（-1表示无数据，0.0-1.0表示正评比例）                                          |
| shop_star_level           | category | 店铺星级等级（4999≈4.99星，5020≈5.02星，脱敏后编码）                                      |
| shop_score_service        | float    | 店铺服务评分（-1表示无数据，0.0-1.0表示服务评分）                                          |
| shop_score_delivery       | float    | 店铺发货评分（-1表示无数据，0.0-1.0表示发货评分）                                          |
| shop_score_description    | float    | 店铺描述评分（-1表示无数据，0.0-1.0表示描述评分）                                          |
| is_trade                  | category | 是否成交（0:未成交, 1:成交，转化目标）                                              |

