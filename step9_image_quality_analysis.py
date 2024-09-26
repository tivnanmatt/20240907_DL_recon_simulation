
import os
import pandas as pd
import numpy as np
import pydicom
from skimage.metrics import structural_similarity as ssim
from skimage.util import img_as_float
import torch
from PIL import Image
from step2_dataset_dataloader import RSNA_Intracranial_Hemorrhage_Dataset
from step0_common_info import dataset_dir, dicom_dir
import matplotlib.pyplot as plt

class ImageQualityAnalysis:
    def __init__(self, original_dataset, reconstructed_datasets, max_pixel_value=255):
        """
        Initialize with original and reconstructed datasets. Ensure all datasets are compatible.
        """
        print('Initializing Image Quality Analysis...')
        self.original_dataset = original_dataset
        self.reconstructed_datasets = reconstructed_datasets
        self.max_pixel_value = max_pixel_value

        # Define intensity windows
        self.brain_window = (0.0, 80.0)
        self.bone_window = (0, 255)

        self.inf_psnr_indices = []
        self.inf_psnr_images = []
        
        # Ensure all datasets are the same size and compatible for comparison
        self.check_data_compatibility()
        
        # Extract labels
        self.labels = [np.argmax(self.original_dataset[i][1].numpy()) for i in range(len(self.original_dataset))]
        print('Labels extracted: {self.labels[:5]}')

        # Calculate metrics
        self.metrics_per_reconstruction = self.calculate_metrics_for_all_reconstructions()
        print('Image quality metrics calculated.')

    def check_data_compatibility(self):
        """
        Ensure the original and all reconstructed datasets have the same shape and number of images.
        """
        num_images = len(self.original_dataset)
        print(f"Number of images in original dataset: {num_images}")
        for key, dataset in self.reconstructed_datasets.items():
            assert len(dataset) == num_images, f"{key} dataset must have the same number of images as the original dataset"
        print("All datasets have the same number of images.")

    def apply_window(self, image, window_min, window_max):
        """
        Apply a window to an image.
        """
        return np.clip(image, window_min, window_max)

    def calculate_metrics_for_all_reconstructions(self):
        """
        Compute RMSE, SSIM, and PSNR for each type of reconstruction dataset.
        """
        metrics = {}
        num_images = len(self.original_dataset)
        print(f"Calculating metrics for {num_images} images...")

        for key, dataset in self.reconstructed_datasets.items():
            print(f"Calculating metrics for {key} dataset...")
            rmse_values_brain = []
            ssim_values_brain = []
            psnr_values_brain = []
            rmse_values_bone = []
            ssim_values_bone = []
            psnr_values_bone = []
            
            for i in range(num_images):
                if i % 100 == 0:
                    print(f"Processing image {i+1}/{num_images} in {key} dataset...")    

                # Get the original and reconstructed images
                original_image = self.original_dataset[i][0].numpy()
                resampled_image = dataset[i][0].numpy()
                
                # Normalize images to float
                original_image = img_as_float(original_image).squeeze()
                resampled_image = img_as_float(resampled_image).squeeze()
                
                # Apply intensity windows
                original_brain = self.apply_window(original_image, *self.brain_window)
                resampled_brain = self.apply_window(resampled_image, *self.brain_window)
                original_bone = self.apply_window(original_image, *self.bone_window)
                resampled_bone = self.apply_window(resampled_image, *self.bone_window)
                
                # Calculate metrics for brain window
                rmse_brain = np.sqrt(np.mean((original_brain - resampled_brain) ** 2))
                rmse_values_brain.append(rmse_brain)
                
                ssim_brain = ssim(original_brain, resampled_brain, data_range=1.0)
                ssim_values_brain.append(ssim_brain)
                
                mse_brain = np.mean((original_brain - resampled_brain) ** 2)
                psnr_brain = 10 * np.log10((self.max_pixel_value ** 2) / mse_brain) if mse_brain > 0 else float('inf')
                
                # Check for infinite PSNR values
                if psnr_brain == float('inf'):
                    self.inf_psnr_indices.append(i)
                    self.inf_psnr_images.append((original_brain, original_bone, resampled_brain, resampled_bone))
                
                psnr_values_brain.append(psnr_brain)
                
                # Calculate metrics for bone window
                rmse_bone = np.sqrt(np.mean((original_bone - resampled_bone) ** 2))
                rmse_values_bone.append(rmse_bone)
                
                ssim_bone = ssim(original_bone, resampled_bone, data_range=1)
                ssim_values_bone.append(ssim_bone)
                
                mse_bone = np.mean((original_bone - resampled_bone) ** 2)
                psnr_bone = 10 * np.log10((self.max_pixel_value ** 2) / mse_bone) if mse_bone > 0 else float('inf')
                psnr_values_bone.append(psnr_bone)
            
            print(f"Metrics calculated for {key} dataset.")
            metrics[key] = {
                'RMSE_brain': np.array(rmse_values_brain),
                'SSIM_brain': np.array(ssim_values_brain),
                'PSNR_brain': np.array(psnr_values_brain),
                'RMSE_bone': np.array(rmse_values_bone),
                'SSIM_bone': np.array(ssim_values_bone),
                'PSNR_bone': np.array(psnr_values_bone)
            }
        
        return metrics
    
    def plot_inf_psnr_images(self, output_dir):
        """
        Plot images where PSNR in brain window is infinite.
        """
        print(f"Plotting images with infinite PSNR in brain window...")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for idx, (orig_brain, orig_bone, recon_brain, recon_bone) in zip(self.inf_psnr_indices, self.inf_psnr_images):
            plt.figure(figsize=(12, 6))
            plt.subplot(2, 2, 1)
            plt.title(f"Original Brain {idx}")
            plt.imshow(orig_brain, cmap='gray')
            plt.axis('off')
            
            plt.subplot(2, 2, 2)
            plt.title(f"Reconstructed Brain {idx}")
            plt.imshow(recon_brain, cmap='gray')
            plt.axis('off')

            plt.subplot(2, 2, 3)
            plt.title(f"Original Bone {idx}")
            plt.imshow(orig_bone, cmap='gray')
            plt.axis('off')
            
            plt.subplot(2, 2, 4)
            plt.title(f"Reconstructed Bone {idx}")
            plt.imshow(recon_bone, cmap='gray')
            plt.axis('off')
            
            plt.savefig(os.path.join(output_dir, f'image_{idx}_inf_psnr.png'))
            plt.close()
        print(f"Images with infinite PSNR saved to {output_dir}")

    def save_metrics_to_csv(self, csv_dir):
        """
        Save RMSE, SSIM, and PSNR metrics for each reconstruction method to separate CSV files.
        """
        print(f"Saving metrics to CSV files in {csv_dir}...")
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)
        
        for key, metric_data in self.metrics_per_reconstruction.items():
            data = {
                'Label': self.labels,
                'RMSE_brain': metric_data['RMSE_brain'],
                'SSIM_brain': metric_data['SSIM_brain'],
                'PSNR_brain': metric_data['PSNR_brain'],
                'RMSE_bone': metric_data['RMSE_bone'],
                'SSIM_bone': metric_data['SSIM_bone'],
                'PSNR_bone': metric_data['PSNR_bone']
            }
            df = pd.DataFrame(data)
            df.to_csv(os.path.join(csv_dir, f'{key}_metrics.csv'), index=False)
            print(f"Metrics saved to {key}_metrics.csv")

    def calculate_average_metrics(self):
        """
        Calculate average RMSE, SSIM, and PSNR per reconstruction type,
        ignoring rows with infinite PSNR values, and average per class.
        """
        averages_per_reconstruction = {}
        averages_per_class = {}

        for key, metric_data in self.metrics_per_reconstruction.items():
            print(f"Calculating averages for {key} dataset...")

            # Filter out infinite PSNR values for PSNR calculations only
            valid_psnr_brain = np.isfinite(metric_data['PSNR_brain'])
            valid_psnr_bone = np.isfinite(metric_data['PSNR_bone'])
            
            # Filter only PSNR values based on valid PSNR values
            filtered_psnr_brain = metric_data['PSNR_brain'][valid_psnr_brain]
            filtered_psnr_bone = metric_data['PSNR_bone'][valid_psnr_bone]

            # Keep RMSE and SSIM as they are without filtering
            rmse_brain = metric_data['RMSE_brain']
            ssim_brain = metric_data['SSIM_brain']
            rmse_bone = metric_data['RMSE_bone']
            ssim_bone = metric_data['SSIM_bone']

            # Calculate averages for each reconstruction type
            averages_per_reconstruction[key] = {
                'Avg_RMSE_brain': np.mean(rmse_brain),
                'Avg_SSIM_brain': np.mean(ssim_brain),
                'Avg_PSNR_brain': np.mean(filtered_psnr_brain),
                'Avg_RMSE_bone': np.mean(rmse_bone),
                'Avg_SSIM_bone': np.mean(ssim_bone),
                'Avg_PSNR_bone': np.mean(filtered_psnr_bone)
            }

            # Now calculate averages per class (label)
            unique_labels = np.unique(self.labels)
            averages_per_class[key] = {}

            for label in unique_labels:
                # Get indices where the labels match the current label
                class_indices = np.where(np.array(self.labels) == label)[0]

                # Brain metrics for the current class
                class_rmse_brain = rmse_brain[class_indices]
                class_ssim_brain = ssim_brain[class_indices]
                
                # Bone metrics for the current class
                class_rmse_bone = rmse_bone[class_indices]
                class_ssim_bone = ssim_bone[class_indices]

                # Filter PSNR indices only where valid PSNR exists
                class_psnr_brain_indices = class_indices[valid_psnr_brain[class_indices]]
                class_psnr_bone_indices = class_indices[valid_psnr_bone[class_indices]]

                # Get PSNR values for the current class using valid PSNR indices
                class_psnr_brain = filtered_psnr_brain[np.isin(np.arange(len(filtered_psnr_brain)), class_psnr_brain_indices)]
                class_psnr_bone = filtered_psnr_bone[np.isin(np.arange(len(filtered_psnr_bone)), class_psnr_bone_indices)]

                # Store the averages per class
                averages_per_class[key][f'Label_{label}'] = {
                    'Avg_RMSE_brain': np.mean(class_rmse_brain),
                    'Avg_SSIM_brain': np.mean(class_ssim_brain),
                    'Avg_PSNR_brain': np.mean(class_psnr_brain) if len(class_psnr_brain) > 0 else None,
                    'Avg_RMSE_bone': np.mean(class_rmse_bone),
                    'Avg_SSIM_bone': np.mean(class_ssim_bone),
                    'Avg_PSNR_bone': np.mean(class_psnr_bone) if len(class_psnr_bone) > 0 else None
                }

        return averages_per_reconstruction, averages_per_class

    def save_averages_to_csv(self, csv_dir):
        """
        Save the average RMSE, SSIM, and PSNR per reconstruction and per class to CSV.
        """
        print(f"Saving average metrics to CSV files in {csv_dir}...")
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)

        averages_per_reconstruction, averages_per_class = self.calculate_average_metrics()

        # Save averages per reconstruction type
        df_reconstruction = pd.DataFrame(averages_per_reconstruction).T
        df_reconstruction.to_csv(os.path.join(csv_dir, 'average_metrics_per_reconstruction.csv'), index=True)
        print(f"Averages per reconstruction type saved to 'average_metrics_per_reconstruction.csv'")

        # Save averages per class
        for key, class_averages in averages_per_class.items():
            df_class = pd.DataFrame(class_averages).T
            df_class.to_csv(os.path.join(csv_dir, f'{key}_average_metrics_per_class.csv'), index=True)
            print(f"Averages per class saved for {key} to '{key}_average_metrics_per_class.csv'")

    def __call__(self):
        """
        Allows the class to be called as a function to return the metrics dictionary.
        """
        return self.metrics_per_reconstruction

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

    # Initialize the image quality analysis
    analysis = ImageQualityAnalysis(original_dataset, reconstructed_datasets)

    # Save metrics to CSV
    analysis.save_metrics_to_csv('image_quality_metrics')

    # Plot images with infinite PSNR
    analysis.plot_inf_psnr_images('images_with_inf_psnr')

    # Save average metrics to CSV
    analysis.save_averages_to_csv('average_metrics')

    # Print metrics (optional)
    metrics = analysis()
    for key, metric_data in metrics.items():
        print(f"{key} Metrics:")
        print(f"Average RMSE (brain window): {metric_data['RMSE_brain'].mean()}")
        print(f"Average SSIM (brain window): {metric_data['SSIM_brain'].mean()}")
        print(f"Average PSNR (brain window): {metric_data['PSNR_brain'].mean()}")
        print(f"Average RMSE (bone window): {metric_data['RMSE_bone'].mean()}")
        print(f"Average SSIM (bone window): {metric_data['SSIM_bone'].mean()}")
        print(f"Average PSNR (bone window): {metric_data['PSNR_bone'].mean()}")