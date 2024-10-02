import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import re
from step0_common_info import dicom_dir
from step2_dataset_dataloader import RSNA_Intracranial_Hemorrhage_Dataset
from step3_cnn_classifier import SupervisedClassifier, load_classifier   

def apply_window(image, window_min, window_max):
    """
    Apply a window to an image.
    """
    return np.clip(image, window_min, window_max)

def compute_saliency_maps(model, inputs, target_class):
    """
    Compute saliency maps for the given input using the model.
    """
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
    saliency, _ = torch.max(inputs.grad.data.abs(), dim=1)
    return saliency

def generate_and_save_saliency_images(model, original_dataset, fbp_dataset, mbir_dataset, dlr_dataset, device, output_folder, specific_cases):
    # Define the brain window limits
    window_min, window_max = 0, 80

    # Set the model to evaluation mode
    model.eval()

    for case_index in specific_cases:
        # Load specific images from the datasets
        inputs, label = original_dataset[case_index]
        fbp_img, _ = fbp_dataset[case_index]
        mbir_img, _ = mbir_dataset[case_index]
        dlr_img, _ = dlr_dataset[case_index]

        # Move images to device
        inputs = inputs.unsqueeze(0).to(device)  # Add a batch dimension
        fbp_img = fbp_img.unsqueeze(0).to(device)  # Add a batch dimension
        mbir_img = mbir_img.unsqueeze(0).to(device)  # Add a batch dimension
        dlr_img = dlr_img.unsqueeze(0).to(device)  # Add a batch dimension
        label = label.to(device)

        with torch.no_grad():
            # Get model predictions for each input
            original_predictions = model(inputs)
            fbp_predictions = model(fbp_img)
            mbir_predictions = model(mbir_img)
            dlr_predictions = model(dlr_img)

            # Apply softmax to get probabilities
            original_pred_probs = F.softmax(original_predictions, dim=1).cpu().numpy()
            fbp_pred_probs = F.softmax(fbp_predictions, dim=1).cpu().numpy()
            mbir_pred_probs = F.softmax(mbir_predictions, dim=1).cpu().numpy()
            dlr_pred_probs = F.softmax(dlr_predictions, dim=1).cpu().numpy()

            # Get predicted labels
            original_predicted_label = np.argmax(original_pred_probs, axis=1).item()
            fbp_predicted_label = np.argmax(fbp_pred_probs, axis=1).item()
            mbir_predicted_label = np.argmax(mbir_pred_probs, axis=1).item()
            dlr_predicted_label = np.argmax(dlr_pred_probs, axis=1).item()

            # Get true label index from one-hot encoding
            true_label = torch.argmax(label).item()

        # Convert images to NumPy and normalize them
        img = inputs[0].permute(1, 2, 0).detach().cpu().numpy()
        img = apply_window(img, window_min, window_max)  # Apply brain window
        img_normalized = (img - img.min()) / (img.max() - img.min())

        # Normalize FBP, MBIR, and DLR images
        fbp_img_normalized = fbp_img[0].permute(1, 2, 0).detach().cpu().numpy()
        fbp_img_normalized = apply_window(fbp_img_normalized, window_min, window_max)
        mbir_img_normalized = mbir_img[0].permute(1, 2, 0).detach().cpu().numpy()
        mbir_img_normalized = apply_window(mbir_img_normalized, window_min, window_max)
        dlr_img_normalized = dlr_img[0].permute(1, 2, 0).detach().cpu().numpy()
        dlr_img_normalized = apply_window(dlr_img_normalized, window_min, window_max)

        # Normalize using numpy operations
        fbp_img_normalized = (fbp_img_normalized - fbp_img_normalized.min()) / (fbp_img_normalized.max() - fbp_img_normalized.min())
        mbir_img_normalized = (mbir_img_normalized - mbir_img_normalized.min()) / (mbir_img_normalized.max() - mbir_img_normalized.min())
        dlr_img_normalized = (dlr_img_normalized - dlr_img_normalized.min()) / (dlr_img_normalized.max() - dlr_img_normalized.min())

        # Compute saliency maps
        saliency_original = compute_saliency_maps(model, inputs, true_label)
        saliency_fbp = compute_saliency_maps(model, fbp_img, fbp_predicted_label)
        saliency_mbir = compute_saliency_maps(model, mbir_img, mbir_predicted_label)
        saliency_dlr = compute_saliency_maps(model, dlr_img, dlr_predicted_label)

        # Convert saliency maps to numpy and remove first dimension
        saliency_original = saliency_original.squeeze().detach().cpu().numpy()
        saliency_fbp = saliency_fbp.squeeze().detach().cpu().numpy()
        saliency_mbir = saliency_mbir.squeeze().detach().cpu().numpy()
        saliency_dlr = saliency_dlr.squeeze().detach().cpu().numpy()

        # Create a figure with two rows
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))

        # Plot original image with true label
        axes[0, 0].imshow(img_normalized, cmap='gray')
        axes[0, 0].set_title(f'Original Image\nTrue Label: {true_label}', fontsize=12)
        axes[0, 0].axis('off')

        # Plot FBP image with predicted label
        axes[0, 1].imshow(fbp_img_normalized, cmap='gray')
        axes[0, 1].set_title(f'FBP Prediction: {fbp_predicted_label}', fontsize=12)
        axes[0, 1].axis('off')

        # Plot MBIR image with predicted label
        axes[0, 2].imshow(mbir_img_normalized, cmap='gray')
        axes[0, 2].set_title(f'MBIR Prediction: {mbir_predicted_label}', fontsize=12)
        axes[0, 2].axis('off')

        # Plot DLR image with predicted label
        axes[0, 3].imshow(dlr_img_normalized, cmap='gray')
        axes[0, 3].set_title(f'DLR Prediction: {dlr_predicted_label}', fontsize=12)
        axes[0, 3].axis('off')

        # Overlay saliency maps on the original image
        axes[1, 0].imshow(img_normalized, cmap='gray')
        axes[1, 0].imshow(saliency_original, cmap='hot', alpha=0.5)
        axes[1, 0].set_title(f'Saliency Map on Original\nTrue Label: {true_label}', fontsize=12)
        axes[1, 0].axis('off')

        axes[1, 1].imshow(fbp_img_normalized, cmap='gray')
        axes[1, 1].imshow(saliency_fbp, cmap='hot', alpha=0.5)
        axes[1, 1].set_title(f'Saliency Map on FBP\nPred: {fbp_predicted_label}', fontsize=12)
        axes[1, 1].axis('off')

        axes[1, 2].imshow(mbir_img_normalized, cmap='gray')
        axes[1, 2].imshow(saliency_mbir, cmap='hot', alpha=0.5)
        axes[1, 2].set_title(f'Saliency Map on MBIR\nPred: {mbir_predicted_label}', fontsize=12)
        axes[1, 2].axis('off')

        axes[1, 3].imshow(dlr_img_normalized, cmap='gray')
        axes[1, 3].imshow(saliency_dlr, cmap='hot', alpha=0.5)
        axes[1, 3].set_title(f'Saliency Map on DLR\nPred: {dlr_predicted_label}', fontsize=12)
        axes[1, 3].axis('off')

        plt.tight_layout()
        plt.savefig(f"{output_folder}/saliency_index{case_index}.png", dpi=300)
        plt.close()



def get_case_indices_from_folder(folder_path):
    case_indices = []
    for filename in os.listdir(folder_path):
        match = re.search(r'_case_(\d+)', filename)  # Regex to extract case number
        if match:
            case_indices.append(int(match.group(1)))
    return sorted(set(case_indices))  # Return unique and sorted indices


# Example usage: loading data and generating saliency images
if __name__ == "__main__":
    output_folder = 'figures/saliency'  # Directory to save Saliency images

    os.makedirs(output_folder, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the pre-trained model and weights
    model = SupervisedClassifier().to(device)
    load_classifier(model)

    # Create dataset instances
    print('Loading datasets...')
    original_csv_file = 'data/metadata_evaluation.csv'
    original_dicom_dir = dicom_dir

    fbp_dicom_dir = 'data/FBP_reconstructions/'
    mbir_dicom_dir = 'data/MBIR_reconstructions/'
    dlr_dicom_dir = 'data/DLR_reconstructions/'

    original_dataset = RSNA_Intracranial_Hemorrhage_Dataset(original_csv_file, original_dicom_dir, expected_size=512)
    fbp_dataset = RSNA_Intracranial_Hemorrhage_Dataset(original_csv_file, fbp_dicom_dir, expected_size=256)
    mbir_dataset = RSNA_Intracranial_Hemorrhage_Dataset(original_csv_file, mbir_dicom_dir, expected_size=256)
    dlr_dataset = RSNA_Intracranial_Hemorrhage_Dataset(original_csv_file, dlr_dicom_dir, expected_size=256)

    # Get all case indices from the figures/cases folder
    case_folder = 'figures/cases'
    specific_cases = get_case_indices_from_folder(case_folder)

    # Generate Saliency images for all cases
    print("Generating Saliency images for all cases...")
    generate_and_save_saliency_images(model, original_dataset, fbp_dataset, mbir_dataset, dlr_dataset, device, output_folder, specific_cases)
    print(f"Saliency images saved in '{output_folder}'.")
