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
    # Add more mappings as necessary
}

class ImageComparison:
    def __init__(self, original_dataset, reconstructed_datasets):
        self.original_dataset = original_dataset
        self.reconstructed_datasets = reconstructed_datasets
        
        # Extract unique labels and first indices
        self.labels = [np.argmax(self.original_dataset[i][1].numpy()) for i in range(len(self.original_dataset))]
        self.first_indices = self.get_first_indices_per_label()

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

    def create_difference_map(self, original, reconstructed):
        """
        Create a difference map between original and reconstructed images.
        """
        return np.abs(original - reconstructed)

    def plot_results(self, output_dir):
        """
        Plot original images, reconstructed images, and difference maps for each label.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        num_reconstructions = len(self.reconstructed_datasets)

        for label, index in self.first_indices.items():
            original_image = self.original_dataset[index][0].numpy()
            original_image = original_image.squeeze()  # Remove channel dimension if present

            # Retrieve the label name from the label_map
            label_name = label_map.get(label, f'Unknown Label {label}')

            fig, axes = plt.subplots(2, num_reconstructions + 1, figsize=(15, 10))
            fig.suptitle(f'Label: {label_name} (ID: {label})')

            # Original Image (leftmost in the first row)
            axes[0, 0].imshow(original_image, cmap='gray')
            axes[0, 0].set_title('Original')
            axes[0, 0].axis('off')

            # Hide remaining areas in the first row
            for i in range(1, num_reconstructions + 1):
                axes[0, i].axis('off')

            # Loop through each reconstruction type
            for i, (key, dataset) in enumerate(self.reconstructed_datasets.items()):
                reconstructed_image = dataset[index][0].numpy()
                reconstructed_image = reconstructed_image.squeeze()

                # Plot reconstructed image
                axes[0, i + 1].imshow(reconstructed_image, cmap='gray')
                axes[0, i + 1].set_title(f'Reconstructed ({key})')
                axes[0, i + 1].axis('off')

                # Create difference map
                difference_map = self.create_difference_map(original_image, reconstructed_image)

                # Plot difference map
                axes[1, i + 1].imshow(difference_map, cmap='hot')
                axes[1, i + 1].set_title(f'Diff Map ({key})')
                axes[1, i + 1].axis('off')

            # Hide unused subplot area in the bottom-left corner
            axes[1, 0].axis('off')

            # Save the plot with label name in the filename
            plt.savefig(os.path.join(output_dir, f'label_{label_name}_comparison.png'))
            plt.close()
        print(f"Comparison images saved to {output_dir}")


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
    comparison.plot_results('figures/comparison_images')
