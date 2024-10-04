import os
import numpy as np
import matplotlib.pyplot as plt
from step2_dataset_dataloader import RSNA_Intracranial_Hemorrhage_Dataset
from step0_common_info import dicom_dir
import matplotlib.gridspec as gridspec

class PairedImageSaver:
    def __init__(self, original_dataset, reconstructed_datasets, labels, brain_window=(0.0, 80.0), bone_window=(-1000, 2000)):
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

        self.label_names = {
            0: 'no_hemorrhage',
            1: 'epidural',
            2: 'intraparenchymal',
            3: 'intraventricular',
            4: 'subarachnoid',
            5: 'subdural'
        }

    def apply_window(self, image, window_min, window_max):
        """
        Apply a window to an image.
        """
        return np.clip(image, window_min, window_max)

    def find_first_images_per_label(self, specific_indices=None):
        """
        Find the first image corresponding to each label, or use specific indices if provided.
        
        Args:
            specific_indices (dict, optional): A dictionary mapping labels to specific indices.
                                            If provided, will use these indices for each label.
                                            Example: {0: 14, 1: 23, 2: 7}
        Returns:
            dict: A dictionary mapping each label to an image index.
        """
        label_indices = {}

        # If specific indices are provided, use them
        if specific_indices:
            for label, index in specific_indices.items():
                if label in self.labels:
                    label_indices[label] = index
            return label_indices

        # Otherwise, find the first image for each label
        for i, (_, label_tensor) in enumerate(self.original_dataset):
            label = np.argmax(label_tensor.numpy())
            if label not in label_indices:
                label_indices[label] = i
            if len(label_indices) == len(self.labels):
                break

        return label_indices
    
    def get_first_10_indices_per_label(self):
        """
        Get the first 10 indices for each unique label in the original dataset.
        """
        unique_labels = np.unique([np.argmax(label_tensor.numpy()) for _, label_tensor in self.original_dataset])
        label_indices = {}
        
        # Iterate through the dataset to gather indices
        for label in unique_labels:
            indices = np.where(np.array([np.argmax(label_tensor.numpy()) for _, label_tensor in self.original_dataset]) == label)[0]
            print(f'Label: {label}, Indices: {indices}')  # Debugging output
            if len(indices) > 0:
                label_indices[label] = indices[:10]  # Get the first 10 indices for the label
        
        return label_indices

    def save_first_10_images_per_label(self):
        """
        Save the first 10 images per label into corresponding folders with brain and bone window images.
        """
        label_indices = self.get_first_10_indices_per_label()
        
        for label, indices in label_indices.items():
            label_name = self.label_names.get(label, f'label_{label}')
            label_dir = os.path.join(self.output_dir, label_name)
            if not os.path.exists(label_dir):
                os.makedirs(label_dir)

            for i, idx in enumerate(indices):
                # Access the original image
                image = self.original_dataset[idx][0].numpy().squeeze()  

                # Apply the brain and bone windows
                brain_image = self.apply_window(image, *self.brain_window)
                bone_image = self.apply_window(image, *self.bone_window)

                # Save brain image
                plt.imshow(brain_image, cmap='gray')
                plt.axis('off')
                save_brain_path = os.path.join(label_dir, f'brain_image_{i + 1}.png')
                plt.savefig(save_brain_path, bbox_inches='tight', pad_inches=0)
                plt.close()
                print(f'Saved {save_brain_path}')

                # Save bone image
                plt.imshow(bone_image, cmap='gray')
                plt.axis('off')
                save_bone_path = os.path.join(label_dir, f'bone_image_{i + 1}.png')
                plt.savefig(save_bone_path, bbox_inches='tight', pad_inches=0)
                plt.close()
                print(f'Saved {save_bone_path}')
    
    def save_paired_images(self):
        """
        Save vertically stacked brain/bone window images for each label and reconstruction type.
        """
        specific_indices = {
            0: 6,
            1: 502,
            2: 600,
            3: 702,
            4: 805,
            5: 903
        }
        label_indices = self.find_first_images_per_label(specific_indices=specific_indices)

        # Create a figure for 9 rows and 8 columns (1 extra column for window labels)
        fig = plt.figure(figsize=(15, 20))
        gs = gridspec.GridSpec(9, 8, height_ratios=[0.02] + [0.1, 0.1] * 4, width_ratios=[0.1] + [0.1] + [1] * 6)
        fig.subplots_adjust(hspace=0.0, wspace=0.0)
        fig.patch.set_facecolor('black')

        # Title for the header row
        recon_titles = ['', 'no_hemorrhage', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
        for i, title in enumerate(recon_titles):
            ax_title = fig.add_subplot(gs[0, i + 1])  # Adjusted index for titles
            ax_title.annotate(title, (0.5, 0.5), xycoords='axes fraction', ha='center', fontsize=12, color='white', weight='bold')  
            ax_title.axis('off')

        # Define y-labels for each reconstruction method
        recon_methods = ['Original', 'FBP', 'MBIR', 'DLR']
        row_mapping = {0: 'Original', 2: 'FBP', 4: 'MBIR', 6: 'DLR'}

        # Add reconstruction type names rotated sideways, centered across the brain/bone rows
        for recon_idx, recon_method in row_mapping.items():
            ax_recon_label = fig.add_subplot(gs[recon_idx + 1:recon_idx + 3, 0])  # Span two rows for brain and bone images
            ax_recon_label.text(0.5, 0.5, recon_method, fontsize=16, color='white', ha='center', va='center', rotation=90, weight='bold')
            ax_recon_label.axis('off')

        # Add brain and bone window labels in the extra column
        for recon_idx in row_mapping.keys():
            ax_window_label_brain = fig.add_subplot(gs[recon_idx + 1, 1])  # Extra column for brain window
            ax_window_label_brain.text(0.5, 0.5, f'[0, 80] HU', fontsize=10, color='white', ha='center', va='center', rotation=90)
            ax_window_label_brain.axis('off')

            ax_window_label_bone = fig.add_subplot(gs[recon_idx + 2, 1])  # Extra column for bone window
            ax_window_label_bone.text(0.5, 0.5, f'[-1000, 2000] HU', fontsize=10, color='white', ha='center', va='center', rotation=90)
            ax_window_label_bone.axis('off')

        for label_idx, (label, index) in enumerate(label_indices.items()):
            print(f"Saving images for label {label} (index {index})...")

            original_image = self.original_dataset[index][0].numpy().squeeze()
            reconstructed_images = {key: dataset[index][0].numpy().squeeze() for key, dataset in self.reconstructed_datasets.items()}

            # Apply intensity windows for the original images
            original_brain = self.apply_window(original_image, *self.brain_window)
            original_bone = self.apply_window(original_image, *self.bone_window)

            for recon_idx, recon_method in row_mapping.items():
                # Define brain and bone images for each reconstruction type
                if recon_method == 'Original':
                    brain_image = original_brain
                    bone_image = original_bone
                else:
                    brain_image = self.apply_window(reconstructed_images[recon_method], *self.brain_window)
                    bone_image = self.apply_window(reconstructed_images[recon_method], *self.bone_window)


                crop = 16

                # Add brain images
                ax_brain = fig.add_subplot(gs[recon_idx + 1, label_idx + 2])  # Adjusted index for brain images
                ax_brain.imshow(brain_image[crop:-crop, crop:-crop], cmap='gray')
                ax_brain.axis('off')

                # Add bone images
                ax_bone = fig.add_subplot(gs[recon_idx + 2, label_idx + 2])  # Adjusted index for bone images
                ax_bone.imshow(bone_image[crop:-crop, crop:-crop], cmap='gray')
                ax_bone.axis('off')

            # line_y_position = recon_idx + 3.5  # Position for the line after the bone image row
            # plt.axhline(y=line_y_position, color='white', linewidth=2)

        # Save the figure
        save_path = os.path.join(self.output_dir, 'full_page_figurev3.png')
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

    output_dir = 'figures/label_examples'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # paired_image_saver.save_first_10_images_per_label()

