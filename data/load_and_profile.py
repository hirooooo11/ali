import pandas as pd
import os
import datetime

DATA_DIR = 'D:/ali/data' 
REPORT_DIR = 'D:/ali/reports'
DATA_FILE = 'round1_ijcai_18_train_20180301.txt' 

def format_timestamp(ts):
    try:
        return datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    except:
        return str(ts)

def generate_profile(file_path):
    print(f"正在读取数据: {file_path} ...")
    df = pd.read_csv(file_path, sep=' ')
    sample_size = len(df)  

    stats_list = []
    
    for col in df.columns:
        na_count = df[col].isnull().sum()           
        minus_one_count = (df[col] == -1).sum()     
        missing = na_count + minus_one_count        


        unique = df[col].nunique()
        
        range_info = "无"
        
        if 'timestamp' in col:
            min_ts = df[col].min()
            max_ts = df[col].max()
            min_date = format_timestamp(min_ts)
            max_date = format_timestamp(max_ts)
            range_info = f"{min_date} 至 {max_date}"

        elif pd.api.types.is_numeric_dtype(df[col]) and 'id' not in col.lower():
            range_info = f"[{df[col].min()}, {df[col].max()}]"

        stats = {
            "字段名": col,
            "缺失率": f"{missing/sample_size:.2%}", 
            "类别字段数量": unique,
            "时间字段范围": range_info
        }
        stats_list.append(stats)

    return sample_size, pd.DataFrame(stats_list) 

def save_for_qwen(sample_size, stats_df, output_path): 
    markdown_table = stats_df.to_markdown(index=False)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("数据字段概览\n\n")  
        f.write(f"**样本总量**: {sample_size}\n\n") 
        f.write(markdown_table)


if __name__ == "__main__":
    os.makedirs(REPORT_DIR, exist_ok=True)
    full_input_path = os.path.join(DATA_DIR, DATA_FILE)
    full_output_path = os.path.join(REPORT_DIR, 'data_profile_old.md')
    

    
    result = generate_profile(full_input_path)
    if result is not None:
        sample_size, stats_df = result 
        save_for_qwen(sample_size, stats_df, full_output_path)
