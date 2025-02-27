import os
import pandas as pd
from constants import POSITION_LIDAR_1_GRID_NO_BB, POSITION_LIDAR_2_GRID_NO_BB, POSITION_LIDAR_3_GRID_NO_BB

def count_lines_in_csv(directory):
    max_lines = 0
    min_lines = float('inf')
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath)
            num_lines = len(df)  # Exclude header
            if num_lines > max_lines:
                max_lines = num_lines
            if num_lines < min_lines:
                min_lines = num_lines
    return max_lines, min_lines

max_lines_lidar1, min_lines_lidar1 = count_lines_in_csv(POSITION_LIDAR_1_GRID_NO_BB)
max_lines_lidar2, min_lines_lidar2 = count_lines_in_csv(POSITION_LIDAR_2_GRID_NO_BB)
max_lines_lidar3, min_lines_lidar3 = count_lines_in_csv(POSITION_LIDAR_3_GRID_NO_BB)

print(f"Max lines in LIDAR 1 CSV files: {max_lines_lidar1}, Min lines: {min_lines_lidar1}")
print(f"Max lines in LIDAR 2 CSV files: {max_lines_lidar2}, Min lines: {min_lines_lidar2}")
print(f"Max lines in LIDAR 3 CSV files: {max_lines_lidar3}, Min lines: {min_lines_lidar3}")
