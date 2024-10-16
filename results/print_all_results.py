# print all .csv files in the results folder
import os
import pandas as pd

results_folder = 'results/'
for file in os.listdir(results_folder):
    if file.endswith('.csv'):
        print(f"Loading {file}...")
        df = pd.read_csv(os.path.join(results_folder, file))
        print(df)
        print()  # Add an empty line for better readability