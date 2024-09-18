import pandas as pd
from sklearn.model_selection import train_test_split
from step0_common_info import dataset_dir

# Load the CSV
filename = f'{dataset_dir}/stage_2_train.csv'
df = pd.read_csv(filename)

# Step 1: Reformat the full metadata for the full dataset
df['PatientID'] = df['ID'].apply(lambda x: x.split('_')[1])
df['DiseaseType'] = df['ID'].apply(lambda x: x.split('_')[2])
df_pivot = df.pivot_table(index='PatientID', columns='DiseaseType', values='Label', aggfunc='first')

# Add 'no_hemorrhage' category
df_pivot['no_hemorrhage'] = (df_pivot.fillna(0).sum(axis=1) == 0).astype(int)

# Remove the "any" column if it exists (not explicitly created in this script, but added to ensure)
df_pivot = df_pivot.loc[:, ~df_pivot.columns.str.contains('any', case=False)]

# Move 'no_hemorrhage' to be the first column after 'PatientID'
df_pivot.reset_index(inplace=True)  # Ensures 'PatientID' is a column
cols = ['PatientID', 'no_hemorrhage'] + [col for col in df_pivot.columns if col not in ['PatientID', 'no_hemorrhage']]
df_pivot = df_pivot[cols]

# Save the full metadata to CSV
full_filename = 'data/metadata_full.csv'
df_pivot.to_csv(full_filename, index=False)
print(f"Saved full metadata to {full_filename}")

# Step 2: Remove cases with more than one hemorrhage type marked as 1
hemorrhage_types = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
df_pivot['hemorrhage_sum'] = df_pivot[hemorrhage_types].sum(axis=1)
df_multiclass = df_pivot[df_pivot['hemorrhage_sum'] <= 1].drop(columns=['hemorrhage_sum'])

# Save the multiclass metadata to CSV
multiclass_filename = 'data/metadata_multiclass.csv'
df_multiclass.to_csv(multiclass_filename, index=False)
print(f"Saved multiclass metadata to {multiclass_filename}")

# --------- Original Student Version of filter_valid_indices ---------

def filter_valid_indices(df_pivot, dicom_dir):
    valid_indices = []
    num_needed_healthy = 500
    num_needed_hemorrhage = 100
    
    count_healthy = 0
    count_hemorrhage = {hem_type: 0 for hem_type in hemorrhage_types[1:]}  # Exclude 'no_hemorrhage'

    for idx, row in df_pivot.iterrows():
        patient_id = row['PatientID']
        dicom_path = f'{dicom_dir}/ID_{patient_id}.dcm'
        try:
            dicom_data = pydicom.dcmread(dicom_path)
            image = dicom_data.pixel_array + float(dicom_data.RescaleIntercept)
            if image.shape[0] == 512 and image.shape[1] == 512:
                is_healthy = row['no_hemorrhage'] == 1
                if is_healthy and count_healthy < num_needed_healthy:
                    valid_indices.append(idx)
                    count_healthy += 1
                    print(f"Added healthy sample {count_healthy}/{num_needed_healthy}")
                elif not is_healthy:
                    for hem_type in hemorrhage_types[1:]:
                        if row[hem_type] == 1 and count_hemorrhage[hem_type] < num_needed_hemorrhage:
                            valid_indices.append(idx)
                            count_hemorrhage[hem_type] += 1
                            print(f"Added {hem_type} sample {count_hemorrhage[hem_type]}/{num_needed_hemorrhage}")
                            break
                if count_healthy >= num_needed_healthy and all(count_hemorrhage[hem_type] >= num_needed_hemorrhage for hem_type in hemorrhage_types[1:]):
                    break
        except Exception as e:
            print(f"Skipping {patient_id}: {e}")
    
    return valid_indices


# -------- New filter_valid_indices function --------

def filter_valid_indices(df_multiclass):
    # Set target counts
    num_needed_healthy = 500
    num_needed_hemorrhage = 100
    
    # Step 1: Extract the first 500 healthy patients (no hemorrhage)
    df_healthy = df_multiclass[df_multiclass['no_hemorrhage'] == 1].head(num_needed_healthy)

    # Step 2: Extract the first 100 patients for each hemorrhage type
    df_hemorrhage = pd.DataFrame()  # Create an empty DataFrame to store hemorrhage cases
    for hem_type in hemorrhage_types:  # Skip 'no_hemorrhage'
        df_type = df_multiclass[df_multiclass[hem_type] == 1].head(num_needed_hemorrhage)
        df_hemorrhage = pd.concat([df_hemorrhage, df_type])

    # Step 3: Combine healthy and hemorrhage cases into one DataFrame
    df_combined = pd.concat([df_healthy, df_hemorrhage])

    return df_combined

# Get valid evaluation indices
df_combined = filter_valid_indices(df_multiclass)

# Step 3: Remove evaluation indices from the main dataset
df_evaluation = df_combined
df_remaining = df_multiclass.drop(df_combined.index)

# Save evaluation CSV file
evaluation_filename = 'data/metadata_evaluation.csv'
df_evaluation.to_csv(evaluation_filename, index=False)
print(f"Saved evaluation file to {evaluation_filename}")

# Step 4: Split the remaining data into training and validation
dataset_size = len(df_remaining)
indices = list(range(dataset_size))

train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)

df_train = df_remaining.iloc[train_indices]
df_val = df_remaining.iloc[val_indices]

# Save training and validation CSV files
train_filename = 'data/metadata_training.csv'
val_filename = 'data/metadata_validation.csv'

df_train.to_csv(train_filename, index=False)
df_val.to_csv(val_filename, index=False)

print(f"Saved training file to {train_filename}")
print(f"Saved validation file to {val_filename}")
