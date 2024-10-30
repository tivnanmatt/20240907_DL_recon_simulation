import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import DataLoader
from step02_dataset_dataloader import RSNA_Intracranial_Hemorrhage_Dataset
from step04_cnn_classifier import SupervisedClassifierObserver, load_classifier
from step00_common_info import dicom_dir
import shap

# Define label names
label_names = [
    "no_hemorrhage", 
    "epidural", 
    "intraparenchymal", 
    "intraventricular", 
    "subarachnoid", 
    "subdural"
]

# Ensure figures/cases directory exists
cases_dir = 'figures/shap'
if not os.path.exists(cases_dir):
    os.makedirs(cases_dir)

def compute_shap_values(model, dataset, num_samples=10):
    # Create a SHAP explainer
    background_images = dataset[:num_samples][0].clone()  # Use the first `num_samples` images as background
    
    # Ensure background_images has the correct shape
    if background_images.dim() == 4 and background_images.size(1) == 100:
        background_images = background_images[:, 0:1, :, :]  # Keep only the first channel
    elif background_images.dim() == 3:
        background_images = background_images.unsqueeze(1)  # Shape: [num_samples, 1, height, width]
    
    # Move background_images to the same device as the model
    device = next(model.parameters()).device
    background_images = background_images.to(device)

    background_images = background_images.clone().detach()  # Detach to create a new tensor
    background_images.requires_grad = True 

    explainer = shap.DeepExplainer(model, background_images)
    
    # Compute SHAP values
    shap_values = explainer.shap_values(background_images.clone())
    
    return shap_values

def plot_images_and_shap_maps(shap_results, indices, title_prefix):
    num_datasets = len(shap_results)
    num_images = len(indices)
    
    plt.figure(figsize=(16, 8 * num_images))
    
    for i, (key, result) in enumerate(shap_results.items()):
        dataset = result['dataset']
        shap_values = result['shap_values']
        
        for j in range(num_images):
            idx = indices[j]
            image = dataset[idx][0].numpy().squeeze()
            shap_image = shap_values[idx][0]  # Get the SHAP values for the specific image
            
            # Normalize the SHAP values
            shap_image_normalized = (shap_image - shap_image.min()) / (shap_image.max() - shap_image.min())
            
            # Plot original image
            plt.subplot(num_images * 2, num_datasets, i + 1 + j * num_datasets)
            plt.imshow(image, cmap='gray')
            plt.title(f'{key} Image (Case {idx})')
            plt.axis('off')

            # Plot SHAP values
            plt.subplot(num_images * 2, num_datasets, i + 1 + (j + num_images) * num_datasets)
            plt.imshow(shap_image_normalized, cmap='hot')
            plt.title(f'{key} SHAP Values (Case {idx})')
            plt.axis('off')

    plt.tight_layout()
    plt.savefig(f'{cases_dir}/{title_prefix}_combined_shap_maps.png', dpi=300)
    plt.close()

def compute_shap_values_for_datasets(model, datasets, num_samples=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Ensure the model is on the correct device
    shap_results = {}
    for key, dataset in datasets.items():
        shap_values = compute_shap_values(model, dataset, num_samples)
        shap_results[key] = {
            'dataset': dataset,
            'shap_values': shap_values
        }
    return shap_results

# Main code
if __name__ == "__main__":
    original_csv_file = 'data/metadata_evaluation.csv'
    original_dicom_dir = dicom_dir

    fbp_dicom_dir = 'data/FBP_reconstructions/'
    mbir_dicom_dir = 'data/MBIR_reconstructions/'
    dlr_dicom_dir = 'data/DLR_reconstructions/'

    print('Loading datasets...')
    original_dataset = RSNA_Intracranial_Hemorrhage_Dataset(original_csv_file, original_dicom_dir, expected_size=512)
    fbp_dataset = RSNA_Intracranial_Hemorrhage_Dataset(original_csv_file, fbp_dicom_dir, expected_size=256)
    mbir_dataset = RSNA_Intracranial_Hemorrhage_Dataset(original_csv_file, mbir_dicom_dir, expected_size=256)
    dlr_dataset = RSNA_Intracranial_Hemorrhage_Dataset(original_csv_file, dlr_dicom_dir, expected_size=256)

    datasets = {
        'Original': original_dataset,
        'FBP': fbp_dataset,
        'MBIR': mbir_dataset,
        'DLR': dlr_dataset
    }

    observer = SupervisedClassifierObserver(verbose=True, batch_size=1)
    observer.model = load_classifier(observer.model, 'weights/supervised_classifier_resnet50_weights_09102024.pth')

    print('Evaluating datasets...')
    results = {}
    for key in datasets.keys():
        accuracy, ground_truths, predictions = observer.evaluate(DataLoader(datasets[key], batch_size=1, shuffle=False), num_patients=len(datasets[key]))
        results[key] = {
            'accuracy': accuracy,
            'ground_truths': ground_truths,
            'predictions': predictions
        }
        print(f'{key} accuracy: {accuracy}')

    # Extract case indices from the filenames in the 'figures/cases/v3' directory
    all_case_indices = list(range(1, 5)) + list(range(501, 505)) + list(range(601, 605)) + list(range(701, 705)) + list(range(801, 805)) + list(range(901, 905))  # Assuming all datasets have the same number of cases

    print(f'Found {len(all_case_indices)} cases to evaluate.')

    # Compute SHAP values for the original dataset
    shap_results = compute_shap_values_for_datasets(observer.model, datasets)

    # Plot SHAP values for specific cases
    plot_images_and_shap_maps(shap_results, all_case_indices, title_prefix='Combined_SHAP')

    print('Finished plotting all cases!')