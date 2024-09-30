import os
import numpy as np
import matplotlib.pyplot as plt
from step2_dataset_dataloader import RSNA_Intracranial_Hemorrhage_Dataset
from step0_common_info import dicom_dir

class PairedImageSaver:
    def __init__(self, original_dataset, reconstructed_datasets, labels, brain_window=(0.0, 80.0), bone_window=(0, 255)):
        """
        Initialize with original dataset and reconstructed datasets, and labels.
        """
        self.original_dataset = original_dataset
        self.reconstructed_datasets = reconstructed_datasets
        self.labels = labels
        self.brain_window = brain_window
        self.bone_window = bone_window

        # Ensure output directory exists
        self.output_dir = 'figures/paired_images'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def apply_window(self, image, window_min, window_max):
        """
        Apply a window to an image.
        """
        return np.clip(image, window_min, window_max)

    def find_first_images_per_label(self):
        """
        Find the first image corresponding to each label.
        """
        label_indices = {}
        for i, (_, label_tensor) in enumerate(self.original_dataset):
            label = np.argmax(label_tensor.numpy())
            if label not in label_indices:
                label_indices[label] = i
            if len(label_indices) == len(self.labels):
                break
        return label_indices

    def save_paired_images(self):
        """
        Save paired brain/bone window images for each label and reconstruction type.
        """
        # Find the first image for each label
        label_indices = self.find_first_images_per_label()

        for label, index in label_indices.items():
            print(f"Saving images for label {label} (index {index})...")

            # Get original and reconstructed images
            original_image = self.original_dataset[index][0].numpy().squeeze()
            reconstructed_images = {key: dataset[index][0].numpy().squeeze() for key, dataset in self.reconstructed_datasets.items()}

            # Apply intensity windows and save images
            self._save_paired_images_for_label(label, original_image, reconstructed_images)

    def _save_paired_images_for_label(self, label, original_image, reconstructed_images):
        """
        Save paired brain and bone images for a given label, with clear separation between methods.
        """
        # Create a single row with 8 columns (2 columns for each method)
        fig, axes = plt.subplots(1, 8, figsize=(20, 5))
        fig.subplots_adjust(hspace=0, wspace=0.1)  # Add horizontal space between methods

        # Process original image
        original_brain = self.apply_window(original_image, *self.brain_window)
        original_bone = self.apply_window(original_image, *self.bone_window)

        # Add a white vertical line between methods
        fig.patch.set_facecolor('white')

        # Display original brain and bone images side by side
        axes[0].imshow(original_brain, cmap='gray')
        axes[0].axis('off')

        axes[1].imshow(original_bone, cmap='gray')
        axes[1].axis('off')

        # Process and display reconstructed images (FBP, MBIR, DLR)
        for i, (key, image) in enumerate(reconstructed_images.items()):
            brain_window_image = self.apply_window(image, *self.brain_window)
            bone_window_image = self.apply_window(image, *self.bone_window)

            # Display brain window image in even-indexed columns (0, 2, 4, ...)
            axes[2 * (i + 1)].imshow(brain_window_image, cmap='gray')
            axes[2 * (i + 1)].axis('off')

            # Display bone window image in odd-indexed columns (1, 3, 5, ...)
            axes[2 * (i + 1) + 1].imshow(bone_window_image, cmap='gray')
            axes[2 * (i + 1) + 1].axis('off')

        # Add titles for each method spanning both brain and bone images
        method_titles = ['Original', 'FBP', 'MBIR', 'DLR']
        for i, title in enumerate(method_titles):
            axes[2 * i].annotate(title, (0.5, 1.05), xycoords='axes fraction', ha='center', fontsize=12, weight='bold')

        # Save the image
        save_path = os.path.join(self.output_dir, f'label_{label}_paired_images.png')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(f'Saved image to {save_path}')

if __name__ == '__main__':
    # Load the datasets and the labels (adapt to your environment)
    original_dataset = RSNA_Intracranial_Hemorrhage_Dataset('data/metadata_evaluation.csv', dicom_dir, expected_size=512)
    fbp_dataset = RSNA_Intracranial_Hemorrhage_Dataset('data/metadata_evaluation.csv', 'data/FBP_reconstructions/', expected_size=256)
    mbir_dataset = RSNA_Intracranial_Hemorrhage_Dataset('data/metadata_evaluation.csv', 'data/MBIR_reconstructions/', expected_size=256)
    dlr_dataset = RSNA_Intracranial_Hemorrhage_Dataset('data/metadata_evaluation.csv', 'data/DLR_reconstructions/', expected_size=256)

    reconstructed_datasets = {
        'FBP': fbp_dataset,
        'MBIR': mbir_dataset,
        'DLR': dlr_dataset
    }

    # Define the labels (you can replace this with actual label names if available)
    labels = list(range(6))  # Assuming 6 labels (0 to 5)

    # Initialize and save paired images
    paired_image_saver = PairedImageSaver(original_dataset, reconstructed_datasets, labels)
    paired_image_saver.save_paired_images()
