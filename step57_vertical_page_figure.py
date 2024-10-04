import os
import numpy as np
import matplotlib.pyplot as plt
from step02_dataset_dataloader import RSNA_Intracranial_Hemorrhage_Dataset
from step00_common_info import dicom_dir
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
        # Find the first image for each label
        # label_indices = self.find_first_images_per_label()
        specific_indices = {
            0: 6,
            1: 502,
            2: 600,
            3: 702,
            4: 805,
            5: 903
        }
        label_indices = paired_image_saver.find_first_images_per_label(specific_indices=specific_indices)

        # Create a figure to save all labels in one image
        num_labels = len(label_indices)
        fig = plt.figure(figsize=(15, 19))  # Increased figure size for larger images

        # Create a gridspec for the new layout
        total_rows = num_labels * 2 + 1
        gs = gridspec.GridSpec(total_rows, 6, height_ratios=[0.05] + [0.2] * (num_labels * 2), width_ratios=[0.2, 0.5, 1, 1, 1, 1])
        fig.subplots_adjust(hspace=0.2, wspace=0.2)  # Reduce spaces between subplots for optimization
        fig.patch.set_facecolor('black')

        # Add titles for the columns: label, brain/bone, original, fbp, mbir, dlr
        column_titles = ['Window', 'Original', 'FBP', 'MBIR', 'DLR']
        for i, title in enumerate(column_titles):
            ax_title = fig.add_subplot(gs[0, i + 1])  # Shift titles to the right by 1 since "Label" is removed
            ax_title.annotate(title, (0.5, 0.5), xycoords='axes fraction', ha='center', fontsize=16, color='white', weight='bold')
            ax_title.axis('off')

        # Define y-labels for each label
        y_labels = {
            0: 'no_hemorrhage',
            1: 'epidural',
            2: 'intraparenchymal',
            3: 'intraventricular',
            4: 'subarachnoid',
            5: 'subdural'
        }

        # Iterate over each label and plot corresponding images
        for label_idx, (label, index) in enumerate(label_indices.items()):
            print(f"Saving images for label {label} (index {index})...")

            # Get original and reconstructed images
            original_image = self.original_dataset[index][0].numpy().squeeze()
            reconstructed_images = {key: dataset[index][0].numpy().squeeze() for key, dataset in self.reconstructed_datasets.items()}

            # Apply intensity windows for the original images
            original_brain = self.apply_window(original_image, *self.brain_window)
            original_bone = self.apply_window(original_image, *self.bone_window)

            # Add y-label for the current label (rotated 90 degrees)
            ax_label = fig.add_subplot(gs[label_idx * 2 + 1:label_idx * 2 + 3, 0])  # Span across both brain and bone rows
            ax_label.text(0.5, 0.5, y_labels[label_idx], fontsize=14, color='white', ha='center', va='center', rotation=90)  # Centered label
            ax_label.axis('off')

            # Plot brain and bone images under each reconstruction type
            for i, key in enumerate(['Original', 'FBP', 'MBIR', 'DLR']):
                if key == 'Original':
                    brain_image = original_brain
                    bone_image = original_bone
                else:
                    brain_image = self.apply_window(reconstructed_images[key], *self.brain_window)
                    bone_image = self.apply_window(reconstructed_images[key], *self.bone_window)

                # Plot brain images
                ax_brain = fig.add_subplot(gs[label_idx * 2 + 1, i + 2])
                ax_brain.imshow(brain_image, cmap='gray')
                ax_brain.axis('off')

                # Plot bone images
                ax_bone = fig.add_subplot(gs[label_idx * 2 + 2, i + 2])
                ax_bone.imshow(bone_image, cmap='gray')
                ax_bone.axis('off')

            # Add "Brain [0, 80] HU" and "Bone [-1000, 2000] HU" to the second column
            ax_brain_bone_label = fig.add_subplot(gs[label_idx * 2 + 1, 1])  # Brain label
            ax_brain_bone_label.text(0.5, 0.5, '[0, 80] HU', fontsize=12, color='white', ha='center', va='center')
            ax_brain_bone_label.axis('off')

            ax_bone_bone_label = fig.add_subplot(gs[label_idx * 2 + 2, 1])  # Bone label
            ax_bone_bone_label.text(0.5, 0.5, '[-1000, 2000] HU', fontsize=12, color='white', ha='center', va='center')
            ax_bone_bone_label.axis('off')

            # Add a horizontal white line after the bone row
            line_ax = fig.add_subplot(gs[label_idx * 2 + 2, :])
            line_ax.axhline(y=0, color='white', linewidth=1)  # White horizontal line
            line_ax.axis('off')  # Hide the axis for the line subplot

        # Save the figure
        save_path = os.path.join(self.output_dir, 'all_labels_paired_optimized.png')
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

