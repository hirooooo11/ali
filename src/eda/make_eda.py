import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import datetime


DATA_DIR = 'D:/ali/data'
REPORT_DIR = 'D:/ali/reports'
IMG_DIR = os.path.join(REPORT_DIR, 'figures')
DATA_FILE = 'round1_ijcai_18_train_20180301.txt'

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False






def load_data():
    path = os.path.join(DATA_DIR, DATA_FILE)
    df = pd.read_csv(path, sep=' ')
    df.replace(-1, pd.NA, inplace=True)
    return df

def analyze_basics(df, stats):
    # Label 分布 
    if 'is_trade' in df.columns:
        counts = df['is_trade'].value_counts()
        stats['label分布'] = {
            '未购买': int(counts.get(0, 0)),
            '购买': int(counts.get(1, 0)),
            '转化率': f"{counts.get(1, 0) / len(df):.4%}"
        }
    
    # 缺失值统计
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    stats['缺失值'] = missing.to_dict()

    # 核心特征分布
    numeric_cols = ['item_price_level', 'item_sales_level', 'shop_review_positive_rate']
    stats['核心特征分布'] = {}
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            desc = df[col].describe()
            
            stats['核心特征分布'][col] = {
                'mean': round(desc['mean'], 2),
                'min': desc['min'],
                'max': desc['max'],
                '75%': desc['75%'] 
            }





def analyze_top(df, stats):
    cat_cols = [
        'user_gender_id',    
        'user_age_level',    
        'user_occupation_id',
        'item_category_list',
        'item_city_id'       
    ]
    
    stats['Top 类别频次'] = {}
    
    for col in cat_cols:
        if col in df.columns:
            top5 = df[col].value_counts().head(5)
            stats['Top 类别频次'][col] = {str(k): int(v) for k, v in top5.items()}






def analyze_time(df, stats):
    if 'context_timestamp' not in df.columns:
        return


    df['dt'] = pd.to_datetime(df['context_timestamp'], unit='s')
    df['date'] = df['dt'].dt.date
    df['hour'] = df['dt'].dt.hour  
    
 
    daily_counts = df['date'].value_counts().sort_index()
    stats['时间分布'] = {
        'start_date': str(daily_counts.index[0]),
        'end_date': str(daily_counts.index[-1]),
        'daily_traffic': {str(k): int(v) for k, v in daily_counts.items()}
    }
    

    plt.figure(figsize=(10, 5))
    daily_counts.plot(kind='bar', color='steelblue')
    plt.title('每日流量趋势 ')
    plt.xlabel('日期')
    plt.ylabel('点击量')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'daily.png'))
    plt.close()



def save_json(stats):
    out_path = os.path.join(REPORT_DIR, 'eda_stats.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=4)







if __name__ == "__main__":
    
    df_data = load_data()
    full_stats = {}
    

    analyze_basics(df_data, full_stats)       
    analyze_top(df_data, full_stats) 
    analyze_time(df_data, full_stats)  
    

    save_json(full_stats)