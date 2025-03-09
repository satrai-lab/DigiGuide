import os
import pandas as pd
from datetime import datetime, timedelta


class TrajectoryDataLoader:
    def __init__(self, trajectory_base_path):
        """
        :param trajectory_base_path: 存放12个月份文件夹的根目录
        """
        self.base_path = trajectory_base_path
        # 缓存每个月的数据，键为月份字符串，如 "01", "02", ... "12"
        self.month_data_cache = {}

    def load_month_data(self, month: str) -> pd.DataFrame:
        """
        加载指定月份文件夹下所有 CSV 文件的数据，并合并为一个 DataFrame，
        同时预处理并建立时间索引，加载后的数据会缓存在 self.month_data_cache 中。
        :param month: 月份字符串，例如 "01"
        :return: 预处理后的 DataFrame（索引为去除年份后的时间）
        """
        if month in self.month_data_cache:
            return self.month_data_cache[month]

        month_folder = os.path.join(self.base_path, month)
        csv_files = [os.path.join(month_folder, f) for f in os.listdir(month_folder) if f.endswith(".csv")]

        if not csv_files:
            self.month_data_cache[month] = pd.DataFrame()
            return self.month_data_cache[month]

        df_list = []
        for file_path in csv_files:
            df = pd.read_csv(file_path, parse_dates=["StartDateTime", "EndDateTime"])
            df_list.append(df)

        month_df = pd.concat(df_list, ignore_index=True)

        # 预处理：将 StartDateTime 转换为去除年份的时间
        # 这里我们固定年份为1900，方便后续比较（注意：这假设查询时也采用相同的转换）
        month_df["TimeWithoutYear"] = pd.to_datetime(
            month_df["StartDateTime"].dt.strftime("1900-%m-%d %H:%M:%S"),
            format="1900-%m-%d %H:%M:%S"
        )
        # 根据 TimeWithoutYear 排序，并设置为索引，以便后续利用索引快速切片
        month_df.sort_values(by="TimeWithoutYear", inplace=True)
        month_df.set_index("TimeWithoutYear", inplace=True)

        self.month_data_cache[month] = month_df
        return month_df

    def get_filtered_data(self, query_time: datetime) -> pd.DataFrame:
        """
        根据查询时间，过滤数据并返回每个 PersonID 最后出现的记录。
        此处比较的是月份、日和时分秒（统一转换到固定年份1900），以利用预先构建的时间索引。
        :param query_time: 查询时间（datetime 类型）
        :return: 过滤后的 DataFrame
        """
        # 使用 query_time 中的月份加载对应数据
        month = query_time.strftime("%m")
        df = self.load_month_data(month)

        if df.empty:
            return df

        # 将查询时间转换为统一基准（1900年）的时间
        query_time_key = pd.to_datetime(query_time.strftime("1900-%m-%d %H:%M:%S"), format="1900-%m-%d %H:%M:%S")
        query_time_end_key = query_time_key + timedelta(minutes=30)

        # 利用索引进行时间区间切片（这一步比遍历整列过滤效率高很多）
        filtered_df = df.loc[query_time_key:query_time_end_key].copy()

        # 对过滤后的数据按照 StartDateTime 排序，再按 PersonID 分组，取每组的最后一条记录
        latest_records = filtered_df.sort_values(by="StartDateTime").groupby("PersonID").last().reset_index()
        return latest_records

# 使用示例
# if __name__ == "__main__":
#     # 假设 base_path 是包含12个月份文件夹的目录
#     base_path = "SmartSPEC/post-processing/trajectories_split_data"
#     loader = TrajectoryDataLoader(base_path)
#
#     # 示例：以当前时间作为查询时间
#     now = datetime.now()
#     result_df = loader.get_filtered_data(now)
#     print(result_df)
