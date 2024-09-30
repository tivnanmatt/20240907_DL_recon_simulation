import os
import numpy as np
import matplotlib.pyplot as plt
from step2_dataset_dataloader import RSNA_Intracranial_Hemorrhage_Dataset
from step0_common_info import dataset_dir, dicom_dir

# Example label map; replace with actual mapping if available
label_map = {
    0: 'no_hemorrhage',
    1: 'epidural',
    2: 'intraparenchymal',
    3: 'intraventricular',
    4: 'subarachnoid',
    5: 'subdural'
}

class ImageComparison:
    def __init__(self, original_dataset, reconstructed_datasets):
        self.original_dataset = original_dataset
        self.reconstructed_datasets = reconstructed_datasets
        
        # Extract unique labels and first indices
        self.labels = [np.argmax(self.original_dataset[i][1].numpy()) for i in range(len(self.original_dataset))]
        self.first_indices = self.get_first_indices_per_label()

        # Define intensity windows
        self.brain_window = (0.0, 80.0)
        self.bone_window = (0, 255)

    def get_first_indices_per_label(self):
        """
        Get the first index for each unique label in the dataset.
        """
        unique_labels = np.unique(self.labels)
        first_indices = {}
        
        for label in unique_labels:
            indices = np.where(np.array(self.labels) == label)[0]
            if len(indices) > 0:
                first_indices[label] = indices[0]  # Get the first index for the label
        
        return first_indices

    def apply_window(self, image, window_min, window_max):
        """
        Apply a window to an image.
        """
        return np.clip(image, window_min, window_max)

    def plot_results(self, output_dir):
        """
        Plot original images and reconstructed images for each label.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        num_reconstructions = len(self.reconstructed_datasets)
        n_labels = len(self.first_indices)

        # Adjust the number of columns (four pairs: original, fbp, mbir, dlr)
        n_cols = num_reconstructions + 1
        fig, axes = plt.subplots(n_labels + 1, n_cols, figsize=(20, 10),
                                gridspec_kw={'wspace': 0.05, 'hspace': 0.3})  # Adjust spacing between columns

        fig.patch.set_facecolor('black')

        # Define titles for each column, which will be centered above the pairs
        column_titles = ['Original'] + list(self.reconstructed_datasets.keys())
        for col_idx, title in enumerate(column_titles):
            ax = axes[0, col_idx]  # Position the title above the first image in each pair
            ax.set_title(title, color='white', fontsize=14, pad=10)
            ax.axis('off')  # Hide the title axes

        for row_idx, (label, index) in enumerate(self.first_indices.items()):
            original_image = self.original_dataset[index][0].numpy().squeeze()  # Remove channel dimension if present

            # Apply windows for brain and bone
            original_brain = self.apply_window(original_image, *self.brain_window)
            original_bone = self.apply_window(original_image, *self.bone_window)

            # Stack original brain and bone images horizontally
            original_pair = np.hstack((original_brain, original_bone))

            # Plot the original paired image
            ax = axes[row_idx + 1, 0]
            ax.imshow(original_pair, cmap='gray')  # Adjust vmin/vmax based on your data range
            ax.axis('off')

            # Plot reconstructed images
            for recon_idx, (key, dataset) in enumerate(self.reconstructed_datasets.items()):
                reconstructed_image = dataset[index][0].numpy().squeeze()

                # Apply brain and bone windows
                reconstructed_brain = self.apply_window(reconstructed_image, *self.brain_window)
                reconstructed_bone = self.apply_window(reconstructed_image, *self.bone_window)

                # Stack reconstructed brain and bone images horizontally
                recon_pair = np.hstack((reconstructed_brain, reconstructed_bone))
                
                # Plot the reconstructed paired image
                ax = axes[row_idx + 1, recon_idx + 1]
                ax.imshow(recon_pair, cmap='gray')  # Adjust these values as needed
                ax.axis('off')

            # Draw a thin white line between columns
            for col in range(1, n_cols):  # Start at column 1, skip the first (Original)
                axes[row_idx + 1, col].axvline(x=0, color='white', linewidth=2)  # Line at the beginning of the column

        # Set black background and save plot
        plt.tight_layout(pad=0, h_pad=0, w_pad=0)
        plt.savefig(os.path.join(output_dir, 'recon_comparison.png'), facecolor=fig.get_facecolor())
        plt.close()


if __name__ == '__main__':
    # Define paths to the datasets
    original_csv_file = 'data/metadata_evaluation.csv'
    original_dicom_dir = dicom_dir

    fbp_dicom_dir = 'data/FBP_reconstructions/'
    mbir_dicom_dir = 'data/MBIR_reconstructions/'
    dlr_dicom_dir = 'data/DLR_reconstructions/'

    # Create dataset instances
    print('Loading datasets...')
    original_dataset = RSNA_Intracranial_Hemorrhage_Dataset(original_csv_file, original_dicom_dir, expected_size=512)
    fbp_dataset = RSNA_Intracranial_Hemorrhage_Dataset(original_csv_file, fbp_dicom_dir, expected_size=256)
    mbir_dataset = RSNA_Intracranial_Hemorrhage_Dataset(original_csv_file, mbir_dicom_dir, expected_size=256)
    dlr_dataset = RSNA_Intracranial_Hemorrhage_Dataset(original_csv_file, dlr_dicom_dir, expected_size=256)

    # Create dictionary of reconstructed datasets
    reconstructed_datasets = {
        'FBP': fbp_dataset,
        'MBIR': mbir_dataset,
        'DLR': dlr_dataset
    }

    # Initialize image comparison
    comparison = ImageComparison(original_dataset, reconstructed_datasets)

    # Plot results
    comparison.plot_results('figures/label_firsts_paired')
