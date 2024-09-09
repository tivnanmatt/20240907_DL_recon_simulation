import pandas as pd


# Load the CSV
filename = '/data/rsna-intracranial-hemorrhage-detection/stage_2_train.csv'
df = pd.read_csv(filename)

# Extract patient IDs by splitting the 'ID' column
df['PatientID'] = df['ID'].apply(lambda x: x.split('_')[1])

# Create a pivot table with PatientID as rows and the disease types as columns
# Each patient will have 6 rows for the 6 different disease types
df['DiseaseType'] = df['ID'].apply(lambda x: x.split('_')[2])
df_pivot = df.pivot_table(index='PatientID', columns='DiseaseType', values='Label', aggfunc='first')

# Reset the index to make PatientID a column again
df_pivot.reset_index(inplace=True)

# Display the first few rows of the new DataFrame
print(df_pivot.head())

# Save it to a new csv file
reformat_filename = 'data/stage_2_train_reformat.csv'
df_pivot.to_csv(reformat_filename, index=False)
print(f'Saved reformatted file to {reformat_filename}')