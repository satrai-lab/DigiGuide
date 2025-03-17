import os
import pandas as pd
from datetime import datetime, timedelta


class TrajectoryDataLoader:
    def __init__(self, trajectory_base_path):
        """
        :param trajectory_base_path: Root directory containing folders for 12 months
        """
        self.base_path = trajectory_base_path
        # Cache data for each month, with keys as month strings like "01", "02", ... "12"
        self.month_data_cache = {}

    def load_month_data(self, month: str) -> pd.DataFrame:
        """
        Load data from all CSV files in the specified month folder and merge them into a single DataFrame.
        Preprocess and create a time index. The loaded data will be cached in self.month_data_cache.
        :param month: Month string, e.g., "01"
        :return: Preprocessed DataFrame (indexed by time without year)
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

        # Preprocessing: Convert StartDateTime to time without year
        # Here we fix the year to 1900 for easier comparison later (Note: This assumes the same conversion is used during queries)
        month_df["TimeWithoutYear"] = pd.to_datetime(
            month_df["StartDateTime"].dt.strftime("1900-%m-%d %H:%M:%S"),
            format="1900-%m-%d %H:%M:%S"
        )
        # Sort by TimeWithoutYear and set as index for efficient slicing using the index later
        month_df.sort_values(by="TimeWithoutYear", inplace=True)
        month_df.set_index("TimeWithoutYear", inplace=True)

        self.month_data_cache[month] = month_df
        return month_df

    def get_filtered_data(self, query_time: datetime) -> pd.DataFrame:
        """
        Filter data based on the query time and return the last record for each PersonID.
        Here, we compare month, day, hour, minute, and second (all converted to the fixed year 1900) to utilize the pre-built time index.
        :param query_time: Query time (datetime type)
        :return: Filtered DataFrame
        """
        # Load corresponding data using the month from query_time
        month = query_time.strftime("%m")
        df = self.load_month_data(month)

        if df.empty:
            return df

        # Convert query time to a unified baseline (year 1900)
        query_time_key = pd.to_datetime(query_time.strftime("1900-%m-%d %H:%M:%S"), format="1900-%m-%d %H:%M:%S")
        query_time_end_key = query_time_key + timedelta(minutes=30)

        # Slice the time range using the index (this is much more efficient than filtering the entire column)
        filtered_df = df.loc[query_time_key:query_time_end_key].copy()

        # Sort the filtered data by StartDateTime, then group by PersonID and take the last record of each group
        latest_records = filtered_df.sort_values(by="StartDateTime").groupby("PersonID").last().reset_index()
        return latest_records

# Example usage
# if __name__ == "__main__":
#     # Assume base_path is the directory containing folders for 12 months
#     base_path = "SmartSPEC/post-processing/trajectories_split_data"
#     loader = TrajectoryDataLoader(base_path)
#
#     # Example: Use the current time as the query time
#     now = datetime.now()
#     result_df = loader.get_filtered_data(now)
#     print(result_df)
