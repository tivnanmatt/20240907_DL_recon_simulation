import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import pydicom

class RSNA_Intracranial_Hemorrhage_Dataset(Dataset):
    def __init__(self, csv_file, dicom_dir, transform=None):
        self.metadata = pd.read_csv(csv_file)
        self.dicom_dir = dicom_dir
        self.transform = transform
        self.hemorrhage_types = ['no_hemorrhage', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
        self.metadata['no_hemorrhage'] = (self.metadata['any'] == 0).astype(int)
        # Remove all images classified with multiple hemorrhage types
        self.metadata = self.metadata[self.metadata[self.hemorrhage_types].sum(axis=1) <= 1].reset_index(drop=True)

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        # Get the patient ID and corresponding labels
        patient_id = self.metadata.iloc[idx, 0]
        labels = self.metadata.iloc[idx, 1:]
        labels = labels[self.hemorrhage_types]
        
        # Exclude multi-hemorrhage cases (if sum of hemorrhage types is greater than 1)
        if labels.sum() > 1:
            return self.__getitem__(idx+1)
        
        # One hot encoding: find the hemorrhage type (or no hemorrhage) and convert it to a one-hot vector
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

        # Handle cases where image is not 512x512
        if image.shape[1] != 512 or image.shape[2] != 512:
            return self.__getitem__(idx+1)
        
        # Select first 1000 patients
        target_healthy = 500
        target_hemorrhage = 100

        selected_indices = []

        no_hemorrhage_patients = self.metadata[self.metadata['no_hemorrhage'] ==1].head(target_healthy)
        selected_indices.extend(no_hemorrhage_patients.index.tolist())

        for hemorrhage_type in self.hemorrhage_types[1:]: 
            hemorrhage_patients = self.metadata[self.metadata[hemorrhage_type] == 1].head(target_hemorrhage)
            selected_indices.extend(hemorrhage_patients.index.tolist())

        # Filter metadata
        self.metadata = self.metadata.loc[selected_indices].reset_index(drop=True)

        # now do a 2x2 mean pooling to downsample the image to 256x256
        image = torch.nn.functional.avg_pool2d(image, kernel_size=2)

        return image, torch.tensor(one_hot_labels, dtype=torch.float32)


# Usage Example
if __name__ == '__main__':
    dataset = RSNA_Intracranial_Hemorrhage_Dataset('data/stage_2_train_reformat.csv', '/data/rsna-intracranial-hemorrhage-detection/stage_2_train/')
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Example of iterating over the dataset
    for images, labels in dataloader:
        print(f'Batch of images shape: {images.shape}')
        print(f'Batch of labels: {labels}')
