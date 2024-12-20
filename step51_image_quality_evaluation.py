import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # Import seaborn for violin plots
import numpy as np

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

        # Filter out infinite values
        initial_count = len(data)
        data = data.replace([np.inf, -np.inf], np.nan).dropna()
        final_count = len(data)
        print(f"Data for {recon_type} after filtering: {initial_count - final_count} rows dropped.")

        # Adjust spacing between subplots
        plt.subplots_adjust(wspace=0.4)  # Increase horizontal space between subplots

        # Plot PSNR histogram
        plt.subplot(1, 2, 1)
        ax1 = plt.gca()
        ax1.hist(data['PSNR_brain'], bins=50, alpha=0.7, label='Brain Window')
        ax1.hist(data['PSNR_bone'], bins=50, alpha=0.7, label='Bone Window')

        # Combine legends
        ax1.legend(loc='upper right')

        ax1.set_title(f'{recon_type} PSNR Histogram')
        ax1.set_xlabel('PSNR')
        ax1.set_ylabel('Frequency')
        
        # Plot SSIM histogram
        plt.subplot(1, 2, 2)
        ax3 = plt.gca()
        ax3.hist(data['SSIM_brain'], bins=50, alpha=0.7, label='Brain Window')
        ax3.hist(data['SSIM_bone'], bins=50, alpha=0.7, label='Bone Window')

        # Combine legends
        ax3.legend(loc='upper left')

        ax3.set_title(f'{recon_type} SSIM Histogram')
        # ax3.set_ylim(0, )
        ax3.set_xlabel('SSIM')
        ax3.set_ylabel('Frequency')

        # Save the figure
        save_path = os.path.join(output_dir, f'{recon_type}_histograms.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Saved histograms to {save_path}")


def prepare_data_for_violin(metrics):
    """
    Prepare the combined DataFrame for violin plots.
    """
    violin_data = []
    for recon_type, data in metrics.items():
        for metric in ['SSIM', 'RMSE', 'PSNR']:
            melted_data = pd.melt(data, 
                                  value_vars=[f'{metric}_brain', f'{metric}_bone'],
                                  var_name='Window', 
                                  value_name='Value')
            melted_data['Reconstruction_Type'] = recon_type
            melted_data['Metric'] = metric
            violin_data.append(melted_data)

    # Concatenate all data into a single DataFrame
    combined_data = pd.concat(violin_data, ignore_index=True)
    return combined_data

def plot_violin_plots(combined_data, output_dir):
    """
    Plot violin plots comparing reconstruction types for each metric, distinguishing bone/brain window.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create a violin plot for each metric
    for metric in combined_data['Metric'].unique():
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        
        # Filter data for the current metric
        metric_data = combined_data[combined_data['Metric'] == metric]

        # Plot violin plots
        sns.violinplot(x='Reconstruction_Type', y='Value', hue='Window', data=metric_data, split=True, ax=ax)
        ax.set_title(f'{metric} Comparison Across Reconstruction Types')
        ax.set_xlabel('Reconstruction Type')
        ax.set_ylabel(metric)
        ax.legend(title='Window')
        
        # Save the figure
        save_path = os.path.join(output_dir, f'{metric}_violin_plots.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Saved {metric} violin plots to {save_path}")

if __name__ == '__main__':
    # Define the directories
    metrics_csv_dir = 'image_quality_metrics'  # Directory where metrics CSV files are saved
    violin_plots_output_dir = 'figures/violin_plots'   # Directory to save violin plots
    histograms_output_dir = 'figures/histograms' 

    # Define reconstruction types
    reconstruction_types = ['FBP', 'MBIR', 'DLR']

    # Load metrics
    metrics = load_metrics(metrics_csv_dir, reconstruction_types)

    # Prepare data for violin plots
    combined_data = prepare_data_for_violin(metrics)

    # Plot and save violin plots
    plot_violin_plots(combined_data, violin_plots_output_dir)

    # Make histograms
    plot_histograms(metrics, histograms_output_dir)
