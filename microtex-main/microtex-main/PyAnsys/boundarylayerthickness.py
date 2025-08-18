import pandas as pd
import os
import sys
from tomlkit import parse


def execute():
    
    #append parent directory to sys.path so that config.toml can be accessed
    sys.path.append("..")

    with open('config.toml', 'r') as file:
        toml_string = file.read()
        config = parse(toml_string)

    fp_csv = (config["filepaths"]["csv"])
    #os.chdir(r"D:\Work\Summer Placement Testbed")

    # Replace 'your_csv_file.csv' with the path to your CSV file
    csv_file = fp_csv + 'PyFluent_probe.csv'

    print(csv_file)

    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Replace 'your_column_name' with the actual column name you want to analyze
    column_name = 'velocity-magnitude'

    # Find the maximum value in the specified column and multiply it by 0.99
    threshold_value = df[column_name].max() * 0.99

    #print(threshold_value)

    # Iterate through the column to find the first value that exceeds the threshold
    for index, row in df.iterrows():
        if row[column_name] > threshold_value:
            # Return the values of the row where the condition is met
            #print("Row where the condition is met:")
            #print(row)
            # Return the value from the y-coordinate column
            #print("y-coordinate value:")
            #print(row['    y-coordinate'])
            return row['    y-coordinate']
        
if __name__=='__main__':
    y_value = execute()
    print(y_value)
