import pandas as pd
from pathlib import Path

# 定义parquet文件路径
parquet_files = Path("working") / "total_dataset.parquet"

# 如果parquet文件不存在，则从CSV文件创建它
if not parquet_files.exists():
    df = pd.DataFrame()
    for p in Path("working").glob("*.csv"):
        df_stock = pd.read_csv(p)
        df = pd.concat([df, df_stock], axis=0)
    df.to_parquet(parquet_files)

# 读取parquet, 取其中2020-01-01之后的数据，按照年份不同，分别保存。文件名为total_2020.parquet, total_2021.parquet, total_2022.parquet等

df = pd.read_parquet(parquet_files)

# 假设数据中有一个日期列，需要先转换为datetime类型
# 根据股票数据特点，很可能是'date'或'trade_date'列作为时间字段
date_columns = ['date', 'trade_date', 'dt', 'tradedate']

# 寻找数据框中存在的日期列
date_col = None
for col in date_columns:
    if col in df.columns:
        date_col = col
        break

if date_col is not None:
    # 尝试将日期列转换为datetime类型
    try:
        df[date_col] = pd.to_datetime(df[date_col])
        
        # 筛选2020-01-01之后的数据
        df_filtered = df[df[date_col] >= '2020-01-01']
        
        # 检查是否有符合条件的数据
        if df_filtered.empty:
            print("没有找到2020-01-01之后的数据")
        else:
            print(f"找到 {len(df_filtered)} 条2020-01-01之后的数据")
            
            # 按年份分组并保存到不同的parquet文件
            for year, group in df_filtered.groupby(df_filtered[date_col].dt.year):
                output_file = f"total_{year}.parquet"
                group.to_parquet(output_file)
                print(f"Saved {output_file} with {len(group)} rows")
                
    except Exception as e:
        print(f"日期转换过程中出现错误: {e}")
        print("请检查数据格式是否正确")
else:
    print(f"在数据中没有找到常见的日期列名: {date_columns}")
    print("以下是数据中的列名:")
    print(df.columns.tolist())
    print("请确认日期列的名称并相应修改代码")