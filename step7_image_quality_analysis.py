
import os
import pandas as pd
import numpy as np
import pydicom
from skimage.metrics import structural_similarity as ssim
import torch
from PIL import Image
from step2_dataset_dataloader import RSNA_Intracranial_Hemorrhage_Dataset
from step4_iterative_reconstruction import (
    forward_project,
    back_project,
    pinv_recon,
    LinearLogLikelihood,
    QuadraticSmoothnessLogPrior,
    HuberPenalty,
    iterative_reconstruction_gradient_descent,
    HU_to_attenuation,
    plot_reconstructions
)

class image_quality_analysis:
    def __init__(self, original_data, resampled_data, hem_types, max_pixel_value=255):
        """
        Initialize with two datasets: original and resampled. Ensure the datasets are compatible.
        """
        self.original_data = original_data
        self.resampled_data = resampled_data
        self.hem_types = hem_types
        self.max_pixel_value = max_pixel_value  # Maximum possible pixel value, typically 255 for 8-bit images.
        
        # Ensure the datasets are the same size and compatible for comparison
        assert original_data.shape == resampled_data.shape, "Datasets must have the same shape"
        
        # Calculate RMSE, SSIM, and PSNR
        self.rmse_per_image, self.ssim_per_image, self.psnr_per_image = self.calculate_metrics_per_image()
    
    def calculate_metrics_per_image(self):
        """
        Compute the Root Mean Squared Error (RMSE), Structural Similarity Index (SSIM), and Peak Signal-to-Noise Ratio (PSNR)
        for each image between the original and resampled datasets.
        Returns three arrays: RMSE values, SSIM values, and PSNR values where each element corresponds to an individual image.
        """
        num_images = self.original_data.shape[0]
        rmse_values = []
        ssim_values = []
        psnr_values = []
        
        for i in range(num_images):
            # Flatten each image before calculating RMSE
            original_flat = self.original_data[i].flatten()
            resampled_flat = self.resampled_data[i].flatten()
            
            # Calculate RMSE for the current image
            rmse = np.sqrt(np.mean((original_flat - resampled_flat) ** 2))
            rmse_values.append(rmse)
            
            # Calculate SSIM for the current image
            original_image = self.original_data[i]
            resampled_image = self.resampled_data[i]
            ssim_value = ssim(original_image, resampled_image, data_range=resampled_image.max() - resampled_image.min())
            ssim_values.append(ssim_value)
            
            # Calculate PSNR for the current image
            mse = np.mean((original_flat - resampled_flat) ** 2)
            if mse == 0:
                psnr = float('inf')  # If MSE is zero, PSNR is infinite (perfect reconstruction)
            else:
                psnr = 10 * np.log10((self.max_pixel_value ** 2) / mse)
            psnr_values.append(psnr)
        
        return np.array(rmse_values), np.array(ssim_values), np.array(psnr_values)
    
    def save_metrics_to_csv(self, csv_file):
        """
        Save RMSE, SSIM, and PSNR metrics along with hemorrhage types to a CSV file.
        """
        # Ensure that hem_types have the same length as the number of images
        assert len(self.hem_types) == self.original_data.shape[0], "Hemorrhage types length must match the number of images"
        
        # Prepare data for CSV
        data = {
            'hem_type': self.hem_types,
            'RMSE': self.rmse_per_image,
            'SSIM': self.ssim_per_image,
            'PSNR': self.psnr_per_image
        }
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False)
    
    def __call__(self):
        """
        Allows the class to be called as a function to return the RMSE, SSIM, and PSNR arrays.
        """
        return self.rmse_per_image, self.ssim_per_image, self.psnr_per_image
    
if __name__ == '__main__':
    # Define paths to the datasets
    original_csv_file = 'data/stage_2_train_reformat.csv'
    original_dicom_dir = '/mnt/AXIS02_share/rsna-intracranial-hemorrhage-detection/stage_2_train/'

    resampled_csv_file = 'data/stage_2_train_reformat_resampled.csv'  # Placeholder path for resampled data
    resampled_dicom_dir = '/mnt/AXIS02_share/rsna-intracranial-hemorrhage-detection/stage_2_train_resampled/'  # Placeholder path

    # Create dataset instances
    original_dataset = RSNA_Intracranial_Hemorrhage_Dataset(original_csv_file, original_dicom_dir)
    resampled_dataset = RSNA_Intracranial_Hemorrhage_Dataset(resampled_csv_file, resampled_dicom_dir)

    # Extract data from datasets
    original_images = np.stack([original_dataset[i][0].numpy() for i in range(len(original_dataset))])  # Convert tensor to numpy array
    resampled_images = np.stack([resampled_dataset[i][0].numpy() for i in range(len(resampled_dataset))])  # Convert tensor to numpy array
    hem_types = [np.argmax(original_dataset[i][1].numpy()) for i in range(len(original_dataset))]  # Extract labels

    # Initialize the image quality analysis
    analysis = image_quality_analysis(original_images, resampled_images, hem_types)

    # Save metrics to CSV
    analysis.save_metrics_to_csv('image_quality_metrics.csv')

    # Print the metrics (optional)
    rmse, ssim, psnr = analysis()