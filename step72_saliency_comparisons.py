import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from captum.attr import IntegratedGradients
from step00_common_info import dicom_dir
from step02_dataset_dataloader import RSNA_Intracranial_Hemorrhage_Dataset
from step04_cnn_classifier import SupervisedClassifier, load_classifier   
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import openpyxl
import seaborn as sns

def apply_window(image, window_min, window_max):
    """Apply a window to an image."""
    return np.clip(image, window_min, window_max)

# Normalize the saliency maps to the range [-1, 1]
def normalize_saliency(saliency):
    saliency_min = saliency.min()
    saliency_max = saliency.max()
    # Normalize to the range [-1, 1]
    return (saliency - saliency_min) / (saliency_max - saliency_min) 

def compute_integrated_gradients(model, inputs, target_class, device):
    """
    Compute Integrated Gradients for the given input using the model.
    """
    model.eval()

    max_pixel_value = 2000  # or whatever max value is appropriate for your images
    min_pixel_value = -1000
    
    # Create a random noise baseline
    baseline = (torch.rand_like(inputs) * (max_pixel_value - min_pixel_value) + min_pixel_value).to(device)
    baseline = torch.zeros_like(inputs).to(device)

    # Initialize Integrated Gradients object
    ig = IntegratedGradients(model)
    
    # Compute the integrated gradients
    attributions = ig.attribute(inputs, target=target_class, baselines=baseline, n_steps=10)
    
    # Get the maximum across channels
    attributions = torch.max(attributions, dim=1).values.abs()
    
    return attributions

def compute_saliency_maps(model, inputs, target_class):
    """Compute saliency maps for the given input using the model."""
    model.eval()
    inputs.requires_grad = True
    
    # Forward pass
    output = model(inputs)
    
    # Zero the gradients
    model.zero_grad()

    # Get the score for the target class
    score = output[0][target_class]
    
    # Backward pass
    score.backward()

    # Get the saliency map
    saliency, _ = torch.max(inputs.grad.data.abs(), dim=1) # 'Normal Vanilla'
    # saliency = inputs.grad.data # Taking the absolute value of the gradients
    return saliency

def generate_guided_backprop(model, input_image, target_class):
    """Generate guided backpropagation saliency map."""
    model.eval()
    input_image.requires_grad = True
    output = model(input_image)
    
    # Zero the gradients
    model.zero_grad()

    # Get the score for the target class
    score = output[0][target_class]
    
    # Backward pass
    score.backward()

    # Get the guided backpropagation saliency map
    guided_gradients = torch.relu(input_image.grad.data)
    return guided_gradients.squeeze().cpu().numpy()

def plot_patient_analysis(model, original_dataset, fbp_dataset, mbir_dataset, dlr_dataset, case_index, device):
    """Plot the analysis for a specific patient."""
    # Define the brain window limits
    window_min, window_max = 0, 80

    # Load specific images from the datasets
    inputs, label = original_dataset[case_index]
    fbp_img, _ = fbp_dataset[case_index]
    mbir_img, _ = mbir_dataset[case_index]
    dlr_img, _ = dlr_dataset[case_index]

    # Move images to device
    inputs = inputs.unsqueeze(0).to(device)  # Add a batch dimension
    fbp_img = fbp_img.unsqueeze(0).to(device)
    mbir_img = mbir_img.unsqueeze(0).to(device)
    dlr_img = dlr_img.unsqueeze(0).to(device)
    label = label.to(device)

    model.eval()
    
    # Get predicted labels
    with torch.no_grad():
        original_predictions = model(inputs)
        fbp_predictions = model(fbp_img)
        mbir_predictions = model(mbir_img)
        dlr_predictions = model(dlr_img)

        original_predicted_label = np.argmax(F.softmax(original_predictions, dim=1).cpu().numpy(), axis=1).item()
        fbp_predicted_label = np.argmax(F.softmax(fbp_predictions, dim=1).cpu().numpy(), axis=1).item()
        mbir_predicted_label = np.argmax(F.softmax(mbir_predictions, dim=1).cpu().numpy(), axis=1).item()
        dlr_predicted_label = np.argmax(F.softmax(dlr_predictions, dim=1).cpu().numpy(), axis=1).item()

    # Normalize images
    img = inputs[0].permute(1, 2, 0).detach().cpu().numpy()
    img = apply_window(img, window_min, window_max)
    img_normalized = (img - img.min()) / (img.max() - img.min())

    fbp_img_normalized = fbp_img[0].permute(1, 2, 0).detach().cpu().numpy()
    fbp_img_normalized = apply_window(fbp_img_normalized, window_min, window_max)
    fbp_img_normalized = (fbp_img_normalized - fbp_img_normalized.min()) / (fbp_img_normalized.max() - fbp_img_normalized.min())

    mbir_img_normalized = mbir_img[0].permute(1, 2, 0).detach().cpu().numpy()
    mbir_img_normalized = apply_window(mbir_img_normalized, window_min, window_max)
    mbir_img_normalized = (mbir_img_normalized - mbir_img_normalized.min()) / (mbir_img_normalized.max() - mbir_img_normalized.min())

    dlr_img_normalized = dlr_img[0]. permute(1, 2, 0).detach().cpu().numpy()
    dlr_img_normalized = apply_window(dlr_img_normalized, window_min, window_max)
    dlr_img_normalized = (dlr_img_normalized - dlr_img_normalized.min()) / (dlr_img_normalized.max() - dlr_img_normalized.min())

    # Compute saliency maps
    saliency_original = compute_saliency_maps(model, inputs, original_predicted_label)
    saliency_fbp = compute_saliency_maps(model, fbp_img, fbp_predicted_label)
    saliency_mbir = compute_saliency_maps(model, mbir_img, mbir_predicted_label)
    saliency_dlr = compute_saliency_maps(model, dlr_img, dlr_predicted_label)

    # Normalize the saliency maps to the range [-1, 1]
    def normalize_saliency(saliency):
        saliency_min = saliency.min()
        saliency_max = saliency.max()
        # Normalize to the range [-1, 1]
        return 2 * (saliency - saliency_min) / (saliency_max - saliency_min) - 1

    saliency_original = normalize_saliency(saliency_original)
    saliency_fbp = normalize_saliency(saliency_fbp)
    saliency_mbir = normalize_saliency(saliency_mbir)
    saliency_dlr = normalize_saliency(saliency_dlr)

    # Compute guided backpropagation saliency maps
    scaler = MinMaxScaler()
    guided_original = generate_guided_backprop(model, inputs, original_predicted_label)
    guided_fbp = generate_guided_backprop(model, fbp_img, fbp_predicted_label)
    guided_mbir = generate_guided_backprop(model, mbir_img, mbir_predicted_label)
    guided_dlr = generate_guided_backprop(model, dlr_img, dlr_predicted_label)
    guided_original_normalized = scaler.fit_transform(guided_original.reshape(-1, 1)).reshape(guided_original.shape)
    guided_fbp_normalized = scaler.fit_transform(guided_fbp.reshape(-1, 1)).reshape(guided_fbp.shape)
    guided_mbir_normalized = scaler.fit_transform(guided_mbir.reshape(-1, 1)).reshape(guided_mbir.shape)
    guided_dlr_normalized = scaler.fit_transform(guided_dlr.reshape(-1, 1)).reshape(guided_dlr.shape)

    # Compute Integrated Gradients
    ig_original = compute_integrated_gradients(model, inputs, original_predicted_label, device)
    ig_fbp = compute_integrated_gradients(model, fbp_img, fbp_predicted_label, device)
    ig_mbir = compute_integrated_gradients(model, mbir_img, mbir_predicted_label, device)
    ig_dlr = compute_integrated_gradients(model, dlr_img, dlr_predicted_label, device)

    # Normalize Integrated Gradients
    ig_original = normalize_saliency(ig_original)
    ig_fbp = normalize_saliency(ig_fbp)
    ig_mbir = normalize_saliency(ig_mbir)
    ig_dlr = normalize_saliency(ig_dlr)

    # Create a figure with 4 rows and 5 columns
    fig, axes = plt.subplots(nrows=4, ncols=8, figsize=(20, 20), facecolor='black', gridspec_kw={'width_ratios': [0.05, 1, 1, 1, 1, 1, 1,1]})

    row_labels = ['Original', 'FBP', 'MBIR', 'DLR']

    crop = 16
    # First column: Original, FBP, MBIR, DLR images
    axes[0, 1].imshow(img_normalized[crop:-crop, crop:-crop], cmap='gray')
    axes[1, 1].imshow(fbp_img_normalized[crop:-crop, crop:-crop], cmap='gray')
    axes[2, 1].imshow(mbir_img_normalized[crop:-crop, crop:-crop], cmap='gray')
    axes[3, 1].imshow(dlr_img_normalized[crop:-crop, crop:-crop], cmap='gray')

    # Second column: Vanilla gradients
    axes[0, 2].imshow(saliency_original.squeeze().detach().cpu().numpy()[crop:-crop, crop:-crop], cmap='hot')
    axes[1, 2].imshow(saliency_fbp.squeeze().detach().cpu().numpy()[crop:-crop, crop:-crop], cmap='hot')
    axes[2, 2].imshow(saliency_mbir.squeeze().detach().cpu().numpy()[crop:-crop, crop:-crop], cmap='hot')
    axes[3, 2].imshow(saliency_dlr.squeeze().detach().cpu().numpy()[crop:-crop, crop:-crop], cmap='hot')

    # Third column: Overlay of vanilla gradients on images
    axes[0, 3].imshow(img_normalized[crop:-crop, crop:-crop], cmap='gray')
    axes[0, 3].imshow(saliency_original.squeeze().detach().cpu().numpy()[crop:-crop, crop:-crop], cmap='hot', alpha=0.5)
    axes[1, 3].imshow(fbp_img_normalized[crop:-crop, crop:-crop], cmap='gray')
    axes[1, 3].imshow(saliency_fbp.squeeze().detach().cpu().numpy()[crop:-crop, crop:-crop], cmap='hot', alpha=0.5)
    axes[2, 3].imshow(mbir_img_normalized[crop:-crop, crop:-crop], cmap='gray')
    axes[2, 3].imshow(saliency_mbir.squeeze().detach().cpu().numpy()[crop:-crop, crop:-crop], cmap='hot', alpha=0.5)
    axes[3, 3].imshow(dlr_img_normalized[crop:-crop, crop:-crop], cmap='gray')
    axes[3, 3].imshow(saliency_dlr.squeeze().detach().cpu().numpy()[crop:-crop, crop:-crop], cmap='hot', alpha=0.5)


    # Fourth column: Guided backpropagation
    axes[0, 4].imshow(guided_original_normalized[crop:-crop, crop:-crop], cmap='hot')
    axes[1, 4].imshow(guided_fbp_normalized[crop:-crop, crop:-crop], cmap='hot')
    axes[2, 4].imshow(guided_mbir_normalized[crop:-crop, crop:-crop], cmap='hot')
    axes[3, 4].imshow(guided_dlr_normalized[crop:-crop, crop:-crop], cmap='hot')

    # Fifth column: Overlay of guided backpropagation on images
    axes[0, 5].imshow(img_normalized[crop:-crop, crop:-crop], cmap='gray')
    axes[0, 5].imshow(guided_original_normalized[crop:-crop, crop:-crop], cmap='hot', alpha=0.5)
    axes[1, 5].imshow(fbp_img_normalized[crop:-crop, crop:-crop], cmap='gray')
    axes[1, 5].imshow(guided_fbp_normalized[crop:-crop, crop:-crop], cmap='hot', alpha=0.5)
    axes[2, 5].imshow(mbir_img_normalized[crop:-crop, crop:-crop], cmap='gray')
    axes[2, 5].imshow(guided_mbir_normalized[crop:-crop, crop:-crop], cmap='hot', alpha=0.5)
    axes[3, 5].imshow(dlr_img_normalized[crop:-crop, crop:-crop], cmap='gray')
    axes[3, 5].imshow(guided_dlr_normalized[crop:-crop, crop:-crop], cmap='hot', alpha=0.5)

    # sixth column: Integrated gradients
    axes[0, 6].imshow(ig_original.squeeze().detach().cpu().numpy()[crop:-crop, crop:-crop], cmap='hot')
    axes[1, 6].imshow(ig_fbp.squeeze().detach().cpu().numpy()[crop:-crop, crop:-crop], cmap='hot')
    axes[2, 6].imshow(ig_mbir.squeeze().detach().cpu().numpy()[crop:-crop, crop:-crop], cmap='hot')
    axes[3, 6].imshow(ig_dlr.squeeze().detach().cpu().numpy()[crop:-crop, crop:-crop], cmap='hot')

    # Seventh column: Overlay of integrated gradients on images
    axes[0, 7].imshow(img_normalized[crop:-crop, crop:-crop], cmap='gray')
    axes[0, 7].imshow(ig_original.squeeze().detach().cpu().numpy()[crop:-crop, crop:-crop], cmap='hot', alpha=0.5)
    axes[1, 7].imshow(fbp_img_normalized[crop:-crop, crop:-crop], cmap='gray')
    axes[1, 7].imshow(ig_fbp.squeeze().detach().cpu().numpy()[crop:-crop, crop:-crop], cmap='hot', alpha=0.5)
    axes[2, 7].imshow(mbir_img_normalized[crop:-crop, crop:-crop], cmap='gray')
    axes[2, 7].imshow(ig_mbir.squeeze().detach().cpu().numpy()[crop:-crop, crop:-crop], cmap='hot', alpha=0.5)
    axes[3, 7].imshow(dlr_img_normalized[crop:-crop, crop:-crop], cmap='gray')
    axes[3, 7].imshow(ig_dlr.squeeze().detach().cpu().numpy()[crop:-crop, crop:-crop], cmap='hot', alpha=0.5)


    # Add row labels in the narrow column
    for i, label in enumerate(row_labels):
        # Create a centered text axis for each label
        text_axis = fig.add_subplot(4, 6, i * 6 + 1)
        text_axis.text(0, 0.5, label, va='center', ha='center', rotation='vertical',
                       fontsize=20, color='white', weight='bold', transform=text_axis.transAxes)
        text_axis.axis('off')  # Hide axis for text label

    # Remove axis labels
    for i in range(4):
        for j in range(5):
            axes[i, j].axis('off')

    # Adjust spacing between images
    plt.subplots_adjust(wspace=0.1, hspace=-0.5)        

    # Save the figure
    plt.savefig(f'figures/saliency/analysis/patient_{case_index}_analysis.png', dpi=300)
    plt.close()

def compute_saliency_metrics(model, dataset, device):
    model.eval()
    saliency_metrics = []

    for idx in range(len(dataset)):
        inputs, label = dataset[idx]
        inputs = inputs.unsqueeze(0).to(device)
        label = label.to(device)
        
        # Get the actual class label
        if label.dim() == 1:  # One-hot encoded vector
            label = torch.argmax(label).item()  # Get the index of the class
        else:
            raise ValueError("Label tensor is not in the expected one-hot encoded format.")

        with torch.no_grad():
            output = model(inputs)
            predicted_label = np.argmax(F.softmax(output, dim=1).cpu().numpy(), axis=1).item()
            confidence_score = F.softmax(output, dim=1)[0, predicted_label].item()

        saliency_vanilla = compute_saliency_maps(model, inputs, predicted_label)
        saliency_vanilla_normalized = normalize_saliency(saliency_vanilla)

        guided_backprop = generate_guided_backprop(model, inputs, predicted_label)
        guided_backprop_normalized = normalize_saliency(guided_backprop)

        saliency_metrics.append({
            'label': label,
            'predicted_label': predicted_label,
            'confidence_score': confidence_score,
            'mean_saliency_vanilla': saliency_vanilla_normalized.mean().item(),
            'variance_saliency_vanilla': saliency_vanilla_normalized.var().item(),
            'mean_guided_backprop': guided_backprop_normalized.mean().item(),
            'variance_guided_backprop': guided_backprop_normalized.var().item()
            })

    return saliency_metrics

def save_metrics_to_excel(key, metrics, filename='results/saliency_metrics{key}.csv'):
    df = pd.DataFrame(metrics)
    df.to_csv(filename, index=False)


def plot_boxplots(saliency_metrics, dataset_name):
    labels = []
    confidence_scores = []
    mean_saliency_vanilla = []
    variance_saliency_vanilla = []
    mean_guided_backprop = []
    variance_guided_backprop = []

    for metric in saliency_metrics:
        labels.append(metric['label'])
        confidence_scores.append(metric['confidence_score'])
        mean_saliency_vanilla.append(metric['mean_saliency_vanilla'])
        variance_saliency_vanilla.append(metric['variance_saliency_vanilla'])
        mean_guided_backprop.append(metric['mean_guided_backprop'])
        variance_guided_backprop.append(metric['variance_guided_backprop'])

    df = pd.DataFrame({
        'label': labels,
        'confidence_score': confidence_scores,
        'mean_saliency_vanilla': mean_saliency_vanilla,
        'variance_saliency_vanilla': variance_saliency_vanilla,
        'mean_guided_backprop': mean_guided_backprop,
        'variance_guided_backprop': variance_guided_backprop
        }) 
    
    # Melt the DataFrame to long format for Seaborn
    df_melted = df.melt(id_vars='label', value_vars=['mean_saliency_vanilla', 'mean_guided_backprop',],
                        var_name='Metric', value_name='Score')
    
    plt.figure(figsize=(15, 10))
    sns.boxplot(x='label', y='Score', hue='Metric', data=df_melted, palette='Set2')

    plt.title(f'Boxplots of Metrics by Label for {dataset_name} Dataset')
    plt.xlabel('Label')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f'figures/saliency/boxplots/boxplots_{dataset_name}.png', dpi=300)
    plt.close()

def compute_saliency_distribution(saliency_metrics, dataset_name):

    vanilla_saliency = []
    guided_backprop = []

    for metric in saliency_metrics:
        vanilla_saliency.append(metric['mean_saliency_vanilla'])
        guided_backprop.append(metric['mean_guided_backprop'])

    vanilla_saliency = np.array(vanilla_saliency)
    guided_backprop = np.array(guided_backprop)

    # Plot the distribution of saliency metrics
    plt.figure(figsize=(15, 10))

    sns.histplot(vanilla_saliency, bins=20, color='skyblue', alpha=0.5, label='Vanilla Saliency')
    sns.kdeplot(vanilla_saliency, color='blue', label='Vanilla Saliency')    
    plt.title(f'Distribution of Vanilla Saliency Scores for {dataset_name} Dataset')
    plt.xlabel('Saliency Score')
    plt.ylabel('Frequency')
    plt.legend(loc='best')
    plt.savefig(f'figures/saliency/boxplots/distribution_vanilla_{dataset_name}.png', dpi=300)
    plt.close()

    plt.figure(figsize=(15, 10))
    sns.histplot(guided_backprop, bins=20, color='lightcoral', alpha=0.5, label='Guided Backpropagation')
    sns.kdeplot(guided_backprop, color='red', label='Vanilla Saliency')    
    plt.title(f'Distribution of Guided Backpropagation Scores for {dataset_name} Dataset')
    plt.xlabel('Saliency Score')
    plt.ylabel('Frequency')
    plt.legend(loc='best')
    plt.savefig(f'figures/saliency/boxplots/distribution_gbp_{dataset_name}.png', dpi=300)
    plt.close()

# Example usage: loading data and generating analysis
if __name__ == "__main__":
    output_folder = 'figures/saliency/analysis'  # Directory to save analysis figures

    os.makedirs(output_folder, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the pre-trained model and weights
    model = SupervisedClassifier().to(device)
    load_classifier(model)

    # Create dataset instances
    original_csv_file = 'data/metadata_evaluation.csv'
    original_dicom_dir = dicom_dir

    fbp_dicom_dir = 'data/FBP_reconstructions/'
    mbir_dicom_dir = 'data/MBIR_reconstructions/'
    dlr_dicom_dir = 'data/DLR_reconstructions/'

    original_dataset = RSNA_Intracranial_Hemorrhage_Dataset(original_csv_file, original_dicom_dir, expected_size=512)
    fbp_dataset = RSNA_Intracranial_Hemorrhage_Dataset(original_csv_file, fbp_dicom_dir, expected_size=256)
    mbir_dataset = RSNA_Intracranial_Hemorrhage_Dataset(original_csv_file, mbir_dicom_dir, expected_size=256)
    dlr_dataset = RSNA_Intracranial_Hemorrhage_Dataset(original_csv_file, dlr_dicom_dir, expected_size=256)

    # Analyze patient with index 702
    plot_patient_analysis(model, original_dataset, fbp_dataset, mbir_dataset, dlr_dataset, 812, device)
    print("Analysis figure saved in 'figures/saliency/v3' directory.")

        # Compute and plot saliency metrics separately for each dataset
    datasets = {
        'original': original_dataset,
        'fbp': fbp_dataset,
        'mbir': mbir_dataset,
        'dlr': dlr_dataset
    }

    for dataset_name, dataset in datasets.items():
        saliency_metrics = compute_saliency_metrics(model, dataset, device)
        save_metrics_to_excel(dataset_name, saliency_metrics)  # Save metrics for each dataset
        plot_boxplots(saliency_metrics, dataset_name)  # Plot boxplots for each dataset
        print(f"Boxplot for {dataset_name} dataset saved.")
        compute_saliency_distribution(saliency_metrics, dataset_name)  # Plot saliency distributions
        print(f"Distributions for {dataset_name} dataset saved.")