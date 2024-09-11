
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

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        
        patient_id = self.metadata.iloc[idx, 0]
        labels = self.metadata.iloc[idx, 1:]
        labels = torch.tensor(labels.to_numpy().astype(float)).float().squeeze()
        patient_dicom_dir = f'{self.dicom_dir}/ID_{patient_id}.dcm'
        dicom_data = pydicom.dcmread(patient_dicom_dir)
        assert float(dicom_data.RescaleSlope) == 1.0, 'RescaleSlope is not 1.0'
        image = dicom_data.pixel_array + float(dicom_data.RescaleIntercept)

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(image).float()
            # if it has less than 512 rows, pad with -1000
            # if image.shape[0] < 512:
            #     pad_left = (512 - image.shape[0]) // 2
            #     pad_right = 512 - image.shape[0] - pad_left
            #     assert pad_left + pad_right + image.shape[0] == 512
            #     image = torch.nn.functional.pad(image, (0, 0, pad_left, pad_right), value=-1000)
            # # if it has less than 512 columns, pad with -1000
            # if image.shape[1] < 512:
            #     pad_top = (512 - image.shape[1]) // 2
            #     pad_bottom = 512 - image.shape[1] - pad_top
            #     assert pad_top + pad_bottom + image.shape[1] == 512
            #     image = torch.nn.functional.pad(image, (pad_top, pad_bottom, 0, 0), value=-1000)
            image.unsqueeze_(0)

        return image, labels

    
if __name__ == '__main__':

        
    from tqdm import tqdm
    import time
    
    num_batches = 128
    batch_size=1
    animateFlag = True

    torch.manual_seed(0)

    dataset = RSNA_Intracranial_Hemorrhage_Dataset( 'data/stage_2_train_reformat.csv', 
                                                    '/data/rsna-intracranial-hemorrhage-detection/stage_2_train/')
    

    print(f'Initial dataset size: {len(dataset)}')
    dataset.metadata = dataset.metadata.rename(columns={'any': 'no_hemorrhage'})

    # OPTIONAL: If you would want to make it a multi-class classification, add:
    hemorrhage_types = ['no_hemorrhage', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
    # This is a bit confusing, as no_hemorrhage = 1 indicates a hemorrhage

    # Remove all images classified with multiple hemorrhage types
    indices_to_remove_multiple = []
    for idx, row in tqdm(dataset.metadata.iterrows(), total=len(dataset.metadata), desc='Checking multiple hemorrhages'):
        if sum([row[hem_type] for hem_type in hemorrhage_types]) > 2:
            indices_to_remove_multiple.append(idx)

    print(f"number of images to remove due to multiple hemorrhage types: {len(indices_to_remove_multiple)}")
    dataset.metadata.drop(indices_to_remove_multiple, inplace=True)
    dataset.metadata.reset_index(drop=True, inplace=True)
    print(f"Dataset size after removing multiple hemorrhages: {len(dataset)}")

    # Remove all image with incorrect 512x512 dimensions
    indices_to_remove_dimensions = []
    for idx, row in tqdm(dataset.metadata.iterrows(), total=len(dataset.metadata), desc='Checking image dimensions'):
        dicom_data = pydicom.dcmread(f'/data/rsna-intracranial-hemorrhage-detection/stage_2_train/ID_{row["PatientID"]}.dcm')
        image = dicom_data.pixel_array + float(dicom_data.RescaleIntercept)

        if image.shape != (512, 512):
            indices_to_remove_dimensions.append(idx)
    
    print(f"Number of images to remove due to incorrect dimensions: {len(indices_to_remove_dimensions)}")
    dataset.metadata.drop(indices_to_remove_dimensions, inplace=True)
    dataset.metadata.reset_index(drop=True, inplace=True)
    print(f"Dataset size after removing non 512x512 images: {len(dataset)}")

    # Remove corrupt images that are not centered
    indices_to_remove_corrupted = []  # We'll store indices of rows to remove
    for idx, row in tqdm(dataset.metadata.iterrows(), total=len(dataset.metadata), desc="Processing images"):
        # Load the DICOM image
        dicom_data = pydicom.dcmread(f'/data/rsna-intracranial-hemorrhage-detection/stage_2_train/ID_{row["PatientID"]}.dcm')
        image = dicom_data.pixel_array + float(dicom_data.RescaleIntercept)

        # Extract edges for variance calculation
        top_edge = image[:2, :]  # top row
        bottom_edge = image[-2:, :]  # bottom row
        left_edge = image[:, :2]  # left edge
        right_edge = image[:, -2:]  # right edge

        # Combine outer edges and compute variance
        outer_edges = torch.cat([top_edge.flatten(), bottom_edge.flatten(), left_edge.flatten(), right_edge.flatten()])
        variance_intensity = outer_edges.var()

        var_threshold = 10  # threshold for corrupted centering #NOTE not sure whether this is a good threshold or not

        # If variance is too high, mark this index for removal
        if variance_intensity > var_threshold:
            indices_to_remove_corrupted.append(idx)

    # Remove the rows corresponding to the corrupted images
    print(f'Number of images to remove due to corrupted centering: {len(indices_to_remove_corrupted)}')
    dataset.metadata.drop(indices_to_remove_corrupted, inplace=True)
    dataset.metadata.reset_index(drop=True, inplace=True)
    print(f'Dataset size after removing corrupted images: {len(dataset)}')

    hemorrhage_counts = {hem_type: dataset.metadata[dataset.metadata[hem_type] == 1].shape[0] for hem_type in hemorrhage_types}

    # Print the counts
    print(f"Class counts before balancing: {hemorrhage_counts}")

    # Find the minimum class size
    min_class_size = min(hemorrhage_counts.values())
    print(f'Minimum class size: {min_class_size}')

    balanced_indices = []

    # Function to randomly sample indices for a given class
    def sample_indices(class_name, condition):
        indices = dataset.metadata[condition].index.tolist()
        return np.random.choice(indices, size=min_class_size, replace=False).tolist()
    
    # Sample indices for each hemorrhage type
    for hem_type in hemorrhage_types:
        sampled_indices = sample_indices(hem_type, dataset.metadata[hem_type] == 1)
        balanced_indices.extend(sampled_indices)

    balanced_metadata = dataset.metadata.loc[balanced_indices].reset_index(drop=True)
    dataset.metadata = balanced_metadata

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataloader_iter = iter(dataloader)

    image_list = []
    label_list = []
    t0 = time.time()
    for i in tqdm(range(num_batches), total=num_batches):
        try:
            images, labels = next(dataloader_iter)

        except StopIteration:
            dataloader_iter = iter(dataloader)
            images, labels = next(dataloader_iter)
        image_list.append(images)
        label_list.append(labels)

    images = torch.cat(image_list, dim=0)
    labels = torch.cat(label_list, dim=0)


    print(f'Number of images for no hemorrhage: {len(dataset.metadata["no_hemorrhage"] == 0)}')
    print(f'Number of images for epidural hemorrhage: {len(dataset.metadata['epidural'])}')
    print(f'Number of images for intraparenchymal hemorrhage: {len(dataset.metadata['intraparenchymal'])}')
    print(f'Number of images for intraventricular hemorrhage: {len(dataset.metadata['intraventricular'])}')
    print(f'Number of images for subarachnoid hemorrhage: {len(dataset.metadata['subarachnoid'])}')
    print(f'Number of images for subdural hemorrhage: {len(dataset.metadata['subdural'])}')

    t1 = time.time()
    print(f'Elapsed time: {t1-t0:.2f} seconds to load {num_batches} batches of 16 images each ({num_batches*batch_size} images total)')


    if animateFlag:
        
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        num_frames = 128

        fig = plt.figure()
        ax = fig.add_subplot(111)
        im = ax.imshow(images[0,0].numpy(), cmap='gray', vmin=0, vmax=80)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im)
        plt.tight_layout()

        def update_img(n):
            im.set_data(images[n,0].numpy())
            # make the title the label
            if labels[n,0] == 0:
                title_str = 'No Hemorrhage'
            else:
                title_str = ''
            if labels[n,1] == 1:
                title_str += ' Epidural Hemorrhage'
            if labels[n,2] == 1:
                title_str += ' Intraparenchymal Hemorrhage'
            if labels[n,3] == 1:
                title_str += ' Intraventricular Hemorrhage'
            if labels[n,4] == 1:
                title_str += ' Subarachnoid Hemorrhage'
            if labels[n,5] == 1:
                title_str += ' Subdural Hemorrhage'
            ax.set_title(title_str)
            return im,

        ani = animation.FuncAnimation(fig, update_img, frames=num_frames, blit=True)

        writer = animation.writers['ffmpeg'](fps=1)

        ani.save('figures/dataset.mp4', writer=writer)



        
