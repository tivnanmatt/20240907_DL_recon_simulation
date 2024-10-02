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
        Save vertically stacked brain/bone window images for each label and reconstruction type.
        """
        # Find the first image for each label
        label_indices = self.find_first_images_per_label()

        # Create a figure to save all labels in one image
        num_labels = len(label_indices)
        fig = plt.figure(figsize=(15, 18))  # Adjusted for more vertical space

        # Create a gridspec for the new layout
        total_rows = num_labels * 2 + 3  # 2 rows per label (brain and bone), plus rows for PSNR and SSIM
        gs = gridspec.GridSpec(total_rows, 7, height_ratios=[0.1] * (num_labels * 2) + [0.1, 0.1, 0.1])  # Rows for images, PSNR, SSIM
        fig.subplots_adjust(hspace=0.4, wspace=0.1)
        fig.patch.set_facecolor('black')

        # Add titles for the columns: label, brain/bone, original, fbp, mbir, dlr
        column_titles = ['Label', 'Window', 'Original', 'FBP', 'MBIR', 'DLR']
        for i, title in enumerate(column_titles):
            ax_title = fig.add_subplot(gs[0, i])  # Titles in the first row
            ax_title.annotate(title, (0.5, 0.5), xycoords='axes fraction', ha='center', fontsize=12, color='white', weight='bold')
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

            # Add y-label for the current label
            ax_label = fig.add_subplot(gs[label_idx * 2 + 1, 0])  # Labels appear in the first column
            ax_label.text(0.5, 0.5, y_labels[label_idx], fontsize=10, color='white', ha='center', va='center')
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
            ax_brain_bone_label.text(0.5, 0.5, '[0, 80] HU', fontsize=10, color='white', ha='center', va='center')
            ax_brain_bone_label.axis('off')

            ax_bone_bone_label = fig.add_subplot(gs[label_idx * 2 + 2, 1])  # Bone label
            ax_bone_bone_label.text(0.5, 0.5, '[-1000, 2000] HU', fontsize=10, color='white', ha='center', va='center')
            ax_bone_bone_label.axis('off')

        # # Add PSNR and SSIM rows
        # psnr_labels = ['PSNR', '', '21.9898', '24.6508', '30.2313']
        # for i, label in enumerate(psnr_labels):
        #     ax_psnr = fig.add_subplot(gs[-2, i])  # Use the second last row for PSNR
        #     ax_psnr.annotate(label, (0.5, 0.5), xycoords='axes fraction', ha='center', fontsize=10, color='white', weight='bold')
        #     ax_psnr.axis('off')

        # ssim_labels = ['SSIM', '', '0.5060', '0.6032', '0.7571']
        # for i, label in enumerate(ssim_labels):
        #     ax_ssim = fig.add_subplot(gs[-1, i])  # Use the last row for SSIM
        #     ax_ssim.annotate(label, (0.5, 0.5), xycoords='axes fraction', ha='center', fontsize=10, color='white', weight='bold')
        #     ax_ssim.axis('off')

        # Save the figure
        save_path = os.path.join(self.output_dir, 'all_labels_paired_vertical_psnr_ssim.png')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(f'Saved image to {save_path}')


    
    # def save_paired_images(self):
    #     """
    #     Save paired brain/bone window images for each label and reconstruction type.
    #     """
    #     # Find the first image for each label
    #     label_indices = self.find_first_images_per_label()

    #     # Create a figure to save all labels in one image
    #     num_labels = len(label_indices)
    #     fig = plt.figure(figsize=(18, 13))  # Slightly wider to accommodate the y-label column

    #     # Create a gridspec with extra rows for PSNR and SSIM
    #     total_rows = num_labels + 4  # 1 for titles, 1 for Brain/Bone, num_labels for each label, 1 for PSNR, 1 for SSIM
    #     gs = gridspec.GridSpec(total_rows, 9, height_ratios=[0.1, 0.1] + [1] * num_labels + [0.1, 0.1])  # Adjusted height ratios
    #     fig.subplots_adjust(hspace=0.2, wspace=0.1)  # Adjust space between images
    #     fig.patch.set_facecolor('black')

    #     # Add method titles in the first row, spanning across brain and bone columns
    #     method_titles = ['Original', 'FBP', 'MBIR', 'DLR']
    #     for i, title in enumerate(method_titles):
    #         ax_title = fig.add_subplot(gs[0, 2 * i + 1: 2 * i + 3])  # Span across the brain and bone columns
    #         ax_title.annotate(title, (0.5, 0.5), xycoords='axes fraction', ha='center', fontsize=12, color='white', weight='bold')
    #         ax_title.axis('off')  # Turn off the axis

    #     # Add Brain and Bone titles in the second row
    #     for i in range(len(method_titles)):
    #         ax_brain_title = fig.add_subplot(gs[1, 2 * i + 1])  # Brain title
    #         ax_brain_title.annotate('[0, 80] HU', (0.5, 0.5), xycoords='axes fraction', ha='center', fontsize=10, color='white', weight='bold')
    #         ax_brain_title.axis('off')

    #         ax_bone_title = fig.add_subplot(gs[1, 2 * i + 2])  # Bone title
    #         ax_bone_title.annotate('[-1000, 2000] HU', (0.5, 0.5), xycoords='axes fraction', ha='center', fontsize=10, color='white', weight='bold')
    #         ax_bone_title.axis('off')

    #     # Define y-labels for each row
    #     y_labels = {
    #         0: 'no_hemorrhage',
    #         1: 'epidural',
    #         2: 'intraparenchymal',
    #         3: 'intraventricular',
    #         4: 'subarachnoid',
    #         5: 'subdural'
    #     }

    #     for row, (label, index) in enumerate(label_indices.items()):
    #         print(f"Saving images for label {label} (index {index})...")

    #         # Get original and reconstructed images
    #         original_image = self.original_dataset[index][0].numpy().squeeze()
    #         reconstructed_images = {key: dataset[index][0].numpy().squeeze() for key, dataset in self.reconstructed_datasets.items()}

    #         # Apply intensity windows
    #         original_brain = self.apply_window(original_image, *self.brain_window)
    #         original_bone = self.apply_window(original_image, *self.bone_window)

    #         # Display y-label for the current row in the first column
    #         ax_ylabel = fig.add_subplot(gs[row + 2, 0])  # Y-label column shifted down by 2 rows
    #         ax_ylabel.text(0.5, 0.5, y_labels[row], fontsize=10, color='white', weight='bold', ha='center', va='center')
    #         ax_ylabel.axis('off')  # Hide the axis

    #         # Display original brain and bone images
    #         ax_brain = fig.add_subplot(gs[row + 2, 1])  # First column for original brain
    #         ax_brain.imshow(original_brain, cmap='gray')
    #         ax_brain.axis('off')

    #         ax_bone = fig.add_subplot(gs[row + 2, 2])  # Second column for original bone
    #         ax_bone.imshow(original_bone, cmap='gray')
    #         ax_bone.axis('off')

    #         # Process and display reconstructed images (FBP, MBIR, DLR)
    #         for i, (key, image) in enumerate(reconstructed_images.items()):
    #             brain_window_image = self.apply_window(image, *self.brain_window)
    #             bone_window_image = self.apply_window(image, *self.bone_window)

    #             # Display brain window image in even-indexed columns (4, 6, 8, ...)
    #             ax_brain_recon = fig.add_subplot(gs[row + 2, 3 + i * 2])  # Adjusted column index for brain images
    #             ax_brain_recon.imshow(brain_window_image, cmap='gray')
    #             ax_brain_recon.axis('off')

    #             # Display bone window image in odd-indexed columns (5, 7, 9, ...)
    #             ax_bone_recon = fig.add_subplot(gs[row + 2, 4 + i * 2])  # Adjusted column index for bone images
    #             ax_bone_recon.imshow(bone_window_image, cmap='gray')
    #             ax_bone_recon.axis('off')

    #     # Add vertical lines between columns for better separation
    #     for row in range(num_labels):
    #         for col in range(1, 9, 2):  # Columns 1, 3, 5, 7 for vertical lines
    #             ax_vertical_line = fig.add_subplot(gs[row + 2, col])
    #             ax_vertical_line.axvline(x=0, color='white', linewidth=0.5)
    #             ax_vertical_line.axis('off')  # Hide the axis

    #     # Add horizontal line between images and PSNR/SSIM rows
    #     # plt.axhline(y=num_labels + 2.5, color='white', linewidth=1)  # Adjust y-value for the horizontal line

    #     # PSNR row
    #     psnr_labels = ['PSNR', '', '', '21.9898', '14.5637', '24.6508', '16.1179', '30.2313', '22.1455']
    #     for i, label in enumerate(psnr_labels):
    #         ax_psnr = fig.add_subplot(gs[num_labels + 2, i])  # Use the last row for PSNR
    #         ax_psnr.annotate(label, (0.5, 0.5), xycoords='axes fraction', ha='center', fontsize=10, color='white', weight='bold')
    #         ax_psnr.axis('off')

    #     # SSIM row
    #     ssim_labels = ['SSIM', '', '', '0.5060', '0.5156', '0.6032', '0.6108', '0.7571', '0.7585']
    #     for i, label in enumerate(ssim_labels):
    #         ax_ssim = fig.add_subplot(gs[num_labels + 3, i])  # Use the second last row for SSIM
    #         ax_ssim.annotate(label, (0.5, 0.5), xycoords='axes fraction', ha='center', fontsize=10, color='white', weight='bold')
    #         ax_ssim.axis('off')

    #     # Save the image
    #     save_path = os.path.join(self.output_dir, 'all_labels_paired_psnr_ssim.png')
    #     plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    #     plt.close(fig)
    #     print(f'Saved image to {save_path}')


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

