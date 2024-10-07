import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import pydicom
from step00_common_info import dataset_dir, dicom_dir

class RSNA_Intracranial_Hemorrhage_Dataset(Dataset):
    def __init__(self, csv_file, dicom_dir=None, transform=None, patch_size=None, expected_size=512):

        if dicom_dir is None:
            dicom_dir = dataset_dir + '/stage_2_train'
        # self.metadata = pd.read_csv('data/metadata_evaluation.csv')
        self.metadata = pd.read_csv(csv_file)
        # Clip metadata to only include the first 10000 rows
        # self.metadata = self.metadata.iloc
        self.dicom_dir = dicom_dir
        self.transform = transform
        self.hemorrhage_types = ['no_hemorrhage', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
        # self.metadata['no_hemorrhage'] = (self.metadata['any'] == 0).astype(int)
        # Remove all images classified with multiple hemorrhage types
        # self.metadata = self.metadata[self.metadata[self.hemorrhage_types].sum(axis=1) <= 1].reset_index(drop=True)
        self.expected_size = expected_size

        self.patch_size = patch_size

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):

        if isinstance(idx, slice):
            start = idx.start
            stop = idx.stop
            step = idx.step

            if start is None:
                start = 0
            if stop is None:
                stop = len(self.metadata)
            if step is None:
                step = 1

            image_list = []
            label_list = []
            for i in range(start, stop, step):
                image, label = self.__getitem__(i)
                image_list.append(image)
                label_list.append(label)
            return torch.concat(image_list, dim=0), torch.concat(label_list, dim=0)

        # Get the patient ID and corresponding labels
        patient_id = self.metadata.iloc[idx, 0]
        labels = self.metadata.iloc[idx, 1:]
        labels = labels[self.hemorrhage_types]
        
        # Exclude multi-hemorrhage cases (if sum of hemorrhage types is greater than 1)
        # if labels.sum() > 1:
        #     return self.__getitem__(idx+1)
        
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

        # Handle cases where image is not the expected size
        if self.expected_size == 512 and (image.shape[1] != 512 or image.shape[2] != 512):
            # Handle this case more gracefully, e.g., by skipping or logging
            # return self.__getitem__((idx + 1) % len(self.metadata))
            print(f"Warning: Image at index {idx} is not 512x512, returning image at index {idx+1}")
            return self.__getitem__(idx + 1)
        
        if self.expected_size == 256 and (image.shape[1] != 256 or image.shape[2] != 256):
            # Handle size check for reconstructed images
            raise ValueError(f"Image at index {idx} is not 256x256")
        
        # Now perform downsampling if the expected size is 512
        if self.expected_size == 512:
            image = torch.nn.functional.avg_pool2d(image, kernel_size=2)
            image = torch.clip(image, -1000, 2000)
        

        # clip to -1000 to 2000
        image = torch.clip(image, -1000, 2000)

        if self.patch_size and self.patch_size < 256:
            # Randomly crop the image to the patch size
            x = np.random.randint(0, 256 - self.patch_size)
            y = np.random.randint(0, 256 - self.patch_size)
            image = image[:, x:x+self.patch_size, y:y+self.patch_size]

        return image, torch.tensor(one_hot_labels, dtype=torch.float32)


# Usage Example
if __name__ == '__main__':
    dataset = RSNA_Intracranial_Hemorrhage_Dataset('data/stage_2_train_reformat.csv', dataset_dir + '/stage_2_train')
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Example of iterating over the dataset
    # for images, labels in dataloader:
        # print(f'Batch of images shape: {images.shape}')
        # print(f'Batch of labels: {labels}')
