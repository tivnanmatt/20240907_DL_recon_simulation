import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def load_metrics(csv_dir, reconstruction_types):
    """
    Load metrics from CSV files.
    """
    metrics = {}
    for recon_type in reconstruction_types:
        file_path = os.path.join(csv_dir, f'{recon_type}_metrics.csv')
        if os.path.exists(file_path):
            print(f"Loading metrics from {file_path}...")
            metrics[recon_type] = pd.read_csv(file_path)
        else:
            print(f"File {file_path} not found.")
    return metrics

def plot_histograms(metrics, output_dir):
    """
    Plot histograms for RMSE, PSNR, and SSIM metrics, skipping files with infinite PSNR.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for recon_type, data in metrics.items():
        plt.figure(figsize=(18, 5))

        # Filter out infinite PSNR values
        psnr_brain = data['PSNR_brain'].replace([np.inf, -np.inf], np.nan).dropna()
        psnr_bone = data['PSNR_bone'].replace([np.inf, -np.inf], np.nan).dropna()

        # Plot RMSE histogram
        plt.subplot(1, 3, 1)
        plt.hist(data['RMSE_brain'], bins=50, alpha=0.7, label='Brain Window')
        plt.hist(data['RMSE_bone'], bins=50, alpha=0.7, label='Bone Window')
        plt.title(f'{recon_type} RMSE Histogram')
        plt.xlabel('RMSE')
        plt.ylabel('Frequency')
        plt.legend()

        # Plot PSNR histogram
        plt.subplot(1, 3, 2)
        plt.hist(psnr_brain, bins=50, alpha=0.7, label='Brain Window')
        plt.hist(psnr_bone, bins=50, alpha=0.7, label='Bone Window')
        plt.title(f'{recon_type} PSNR Histogram')
        plt.xlabel('PSNR')
        plt.ylabel('Frequency')
        plt.legend()

        # Plot SSIM histogram
        plt.subplot(1, 3, 3)
        plt.hist(data['SSIM_brain'], bins=50, alpha=0.7, label='Brain Window')
        plt.hist(data['SSIM_bone'], bins=50, alpha=0.7, label='Bone Window')
        plt.title(f'{recon_type} SSIM Histogram')
        plt.xlabel('SSIM')
        plt.ylabel('Frequency')
        plt.legend()

        # Save the figure
        save_path = os.path.join(output_dir, f'{recon_type}_histograms.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Saved histograms to {save_path}")

if __name__ == '__main__':
    # Define the directories
    metrics_csv_dir = 'image_quality_metrics'  # Directory where metrics CSV files are saved
    histograms_output_dir = 'histograms'       # Directory to save histogram plots

    # Define reconstruction types
    reconstruction_types = ['FBP', 'MBIR', 'DLR']

    # Load metrics
    metrics = load_metrics(metrics_csv_dir, reconstruction_types)

    # Plot and save histograms
    plot_histograms(metrics, histograms_output_dir)