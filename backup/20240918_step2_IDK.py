import os
import pandas as pd
import numpy as np
import pydicom
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset
from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image

class RSNA_Intracranial_Hemorrhage_Dataset(Dataset):
    def __init__(self, csv_file, dicom_dir, transform=None, indices_file='valid_indices.txt'):
        self.metadata = pd.read_csv(csv_file)
        self.metadata = self.metadata.iloc[:50000]
        self.dicom_dir = dicom_dir
        self.transform = transform
        self.hemorrhage_types = ['no_hemorrhage', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
        self.metadata['no_hemorrhage'] = (self.metadata['any'] == 0).astype(int)

        self.indices_file = indices_file

        # Remove all images classified with multiple hemorrhage types
        self.metadata = self.metadata[self.metadata[self.hemorrhage_types].sum(axis=1) <= 1].reset_index(drop=True)
        
        # Check if the indices file already exists
        if os.path.exists(self.indices_file):
            with open(self.indices_file, 'r') as f:
                self.valid_indices = [int(idx) for idx in f.readlines()]
            print(f"Loaded {len(self.valid_indices)} valid indices from {self.indices_file}")
        else:
            self.valid_indices = self.filter_valid_indices()
            with open(self.indices_file, 'w') as f:
                for idx in self.valid_indices:
                    f.write(f"{idx}\n")
            print(f"Saved {len(self.valid_indices)} valid indices to {self.indices_file}")

        self.metadata = self.metadata.loc[self.valid_indices].reset_index(drop=True)
        assert len(self.metadata) == 1000, f"Dataset size after selection is {len(self.metadata)}, expected 1000."

        print("Class counts in the dataset:")
        for hem_type in self.hemorrhage_types:
            count = self.metadata[hem_type].sum()
            print(f"{hem_type}: {count}")

    def filter_valid_indices(self):
        valid_indices = []
        num_needed_healthy = 500
        num_needed_hemorrhage = 100
        
        count_healthy = 0
        count_hemorrhage = {hem_type: 0 for hem_type in self.hemorrhage_types[1:]}  # Exclude 'no_hemorrhage'

        for idx, row in self.metadata.iterrows():
            patient_id = row['PatientID']
            dicom_path = f'{self.dicom_dir}/ID_{patient_id}.dcm'
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
                        for hem_type in self.hemorrhage_types[1:]:
                            if row[hem_type] == 1 and count_hemorrhage[hem_type] < num_needed_hemorrhage:
                                valid_indices.append(idx)
                                count_hemorrhage[hem_type] += 1
                                print(f"Added {hem_type} sample {count_hemorrhage[hem_type]}/{num_needed_hemorrhage}")
                                break
                    if count_healthy >= num_needed_healthy and all(count_hemorrhage[hem_type] >= num_needed_hemorrhage for hem_type in self.hemorrhage_types[1:]):
                        break
            except Exception as e:
                print(f"Skipping {patient_id}: {e}")
        
        return valid_indices

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        # Get the patient ID and corresponding labels
        patient_id = self.metadata.iloc[idx, 0]
        labels = self.metadata.iloc[idx, 1:]
        labels = labels[self.hemorrhage_types]
        
        # One hot encoding: find the hemorrhage type (or no hemorrhage if 0th is 1)
        hemorrhage_class = np.argmax(labels.to_numpy())  # Finds the index of the first 1 (or no hemorrhage if 0th is 1)
        one_hot_labels = np.eye(6)[hemorrhage_class]  # Creates a one-hot vector for 6 classes
        
        patient_dicom_dir = f'{self.dicom_dir}/ID_{patient_id}.dcm'
        dicom_data = pydicom.dcmread(patient_dicom_dir)
        assert float(dicom_data.RescaleSlope) == 1.0, 'RescaleSlope is not 1.0'
        image = dicom_data.pixel_array + float(dicom_data.RescaleIntercept)

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(image).float()
            image.unsqueeze_(0)  # Add a channel dimension

        # now do a 2x2 mean pooling to downsample the image to 256x256
        image = torch.nn.functional.avg_pool2d(image, kernel_size=2)

        return image, torch.tensor(one_hot_labels, dtype=torch.float32)

if __name__ == '__main__':
    dataset = RSNA_Intracranial_Hemorrhage_Dataset('data/stage_2_train_reformat.csv', '/mnt/AXIS02_share/rsna-intracranial-hemorrhage-detection/stage_2_train/')

    # 1. Check the length of the dataset
    print(f"Dataset size: {len(dataset)}")

    # 2. Access a specific item by index
    image, labels = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Labels: {labels}")

