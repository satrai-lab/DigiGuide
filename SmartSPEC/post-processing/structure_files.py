import pandas as pd
import os

# Read CSV file
input_file = "D:\Programming\Projects\SmartSPEC\models\Drahi-X\V-0602\output\data.csv"  # Replace with your file path
df = pd.read_csv(input_file, parse_dates=["StartDateTime", "EndDateTime"])

# Sort by time
df = df.sort_values(by="StartDateTime")

# Create target directory
output_dir = "trajectories_split_data"
os.makedirs(output_dir, exist_ok=True)

# Group by date and write to different files
for date, group in df.groupby(df["StartDateTime"].dt.date):
    # Get month folder, e.g., "2020-01"
    month_folder = os.path.join(output_dir, date.strftime("%m"))
    os.makedirs(month_folder, exist_ok=True)

    # Generate file name, e.g., "2020-01-01.csv"
    file_path = os.path.join(month_folder, f"{date.strftime('%d')}.csv")

    # Sort by time again to ensure the time order of the day's data is correct
    group = group.sort_values(by="StartDateTime")

    # Write data to CSV
    group.to_csv(file_path, index=False)

print("Data splitting completed, stored by month and date, and time order ensured.")

import pandas as pd
import os
from datetime import datetime, timedelta

def find_latest_events_in_next_30min(base_time, data_dir="./trajectories_split_data"):
    """
    Find all data from the current time to the next 30 minutes in the dataset, and keep only the last occurrence record for each PersonID.

    :param base_time: datetime object, used as the base time for the query
    :param data_dir: Path to the folder where data is stored
    :return: Filtered Pandas DataFrame
    """
    # Ensure base_time is of datetime type
    if isinstance(base_time, str):
        base_time = datetime.strptime(base_time, "%Y-%m-%d %H:%M:%S")

    # Calculate the time 30 minutes later
    end_time = base_time + timedelta(minutes=30)

    # Determine the corresponding CSV file
    file_path = os.path.join(data_dir, base_time.strftime("%m"), base_time.strftime("%d") + ".csv")

    # If the file does not exist, return an empty DataFrame
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist, returning empty dataset.")
        return pd.DataFrame()

    # Read CSV file
    df = pd.read_csv(file_path, parse_dates=["StartDateTime", "EndDateTime"])

    # Filter records where StartDateTime is between [base_time, base_time + 30min]
    mask = (df["StartDateTime"] >= base_time) & (df["StartDateTime"] <= end_time)
    filtered_df = df.loc[mask]

    # Group by PersonID, keep only the record with the maximum StartDateTime for each PersonID
    latest_records = filtered_df.sort_values(by="StartDateTime").groupby("PersonID").last().reset_index()

    return latest_records

# Example call
# base_time = "2020-01-01 13:18:00"
# result = find_latest_events_in_next_30min(base_time)
# for index, row in result.iterrows():
#     if 10 <= row.EventID < 144:
#         print(row.EventID)
# # print(result)

