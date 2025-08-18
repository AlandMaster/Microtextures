import pandas as pd
import os

os.chdir(r"D:\Work\Summer Placement Testbed")

# Replace 'your_csv_file.csv' with the path to your CSV file
csv_file = 'TUI_test'

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file)

# Replace 'your_column_name' with the actual column name you want to analyze
column_name = 'velocity-magnitude'

# Find the maximum value in the specified column and multiply it by 0.99
threshold_value = df[column_name].max() * 0.99

print(threshold_value)

# Iterate through the column to find the first value that exceeds the threshold
for index, row in df.iterrows():
    if row[column_name] > threshold_value:
        # Return the values of the row where the condition is met
        print("Row where the condition is met:")
        print(row)
        # Return the value from the y-coordinate column
        print("y-coordinate value:")
        print(row['    y-coordinate'])
        break
