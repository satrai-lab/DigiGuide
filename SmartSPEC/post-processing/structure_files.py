import pandas as pd
import os

# 读取 CSV 文件
input_file = "D:\Programming\Projects\SmartSPEC\models\Drahi-X\V-0602\output\data.csv"  # 替换为你的文件路径
df = pd.read_csv(input_file, parse_dates=["StartDateTime", "EndDateTime"])

# 按时间排序
df = df.sort_values(by="StartDateTime")

# 创建目标目录
output_dir = "trajectories_split_data"
os.makedirs(output_dir, exist_ok=True)

# 按日期分组并写入不同文件
for date, group in df.groupby(df["StartDateTime"].dt.date):
    # 获取月份目录，例如 "2020-01"
    month_folder = os.path.join(output_dir, date.strftime("%m"))
    os.makedirs(month_folder, exist_ok=True)

    # 生成文件名，例如 "2020-01-01.csv"
    file_path = os.path.join(month_folder, f"{date.strftime('%d')}.csv")

    # 再次按时间排序，确保当天数据的时间顺序正确
    group = group.sort_values(by="StartDateTime")

    # 将数据写入 CSV
    group.to_csv(file_path, index=False)

print("数据分割完成，已按月份和日期存储，并保证时间排序。")

import pandas as pd
import os
from datetime import datetime, timedelta

def find_latest_events_in_next_30min(base_time, data_dir="./trajectories_split_data"):
    """
    在数据集中查找当前时间到未来 30 分钟内的所有数据，并只保留每个 PersonID 最后的出现记录。

    :param base_time: datetime 对象，作为查询的基准时间
    :param data_dir: 存储数据的文件夹路径
    :return: 筛选后的 Pandas DataFrame
    """
    # 确保 base_time 是 datetime 类型
    if isinstance(base_time, str):
        base_time = datetime.strptime(base_time, "%Y-%m-%d %H:%M:%S")

    # 计算 30 分钟后的时间
    end_time = base_time + timedelta(minutes=30)

    # 确定对应的 CSV 文件
    file_path = os.path.join(data_dir, base_time.strftime("%m"), base_time.strftime("%d") + ".csv")

    # 如果文件不存在，直接返回空 DataFrame
    if not os.path.exists(file_path):
        print(f"文件 {file_path} 不存在，返回空数据集。")
        return pd.DataFrame()

    # 读取 CSV 文件
    df = pd.read_csv(file_path, parse_dates=["StartDateTime", "EndDateTime"])

    # 筛选出 StartDateTime 在 [base_time, base_time + 30min] 之间的记录
    mask = (df["StartDateTime"] >= base_time) & (df["StartDateTime"] <= end_time)
    filtered_df = df.loc[mask]

    # 按 PersonID 分组，每个 PersonID 只保留 StartDateTime 最大的一行（即最后出现的）
    latest_records = filtered_df.sort_values(by="StartDateTime").groupby("PersonID").last().reset_index()

    return latest_records

# 示例调用
# base_time = "2020-01-01 13:18:00"
# result = find_latest_events_in_next_30min(base_time)
# for index, row in result.iterrows():
#     if 10 <= row.EventID < 144:
#         print(row.EventID)
# # print(result)

