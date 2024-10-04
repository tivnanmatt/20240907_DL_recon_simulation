import os
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
from step00_common_info import dicom_dir
from step02_dataset_dataloader import RSNA_Intracranial_Hemorrhage_Dataset
from step04_cnn_classifier import SupervisedClassifier, load_classifier   
import numpy as np
import torch.nn.functional as F
import re

def apply_window(image, window_min, window_max):
    """
    Apply a window to an image.
    """
    return np.clip(image, window_min, window_max)

def generate_and_save_gradcam_images(model, original_dataset, fbp_dataset, mbir_dataset, dlr_dataset, device, output_folder, specific_cases):
    target_layer = model.resnet.layer4[2].conv3  # Reference the target layer for Grad-CAM
    gradcam = GradCAM(model=model, target_layers=[target_layer])
    model.eval()

    # Define the brain window limits
    window_min, window_max = 0, 80

    for case_index in specific_cases:
        # Load specific images from the datasets
        inputs, label = original_dataset[case_index]
        fbp_img, _ = fbp_dataset[case_index]
        mbir_img, _ = mbir_dataset[case_index]
        dlr_img, _ = dlr_dataset[case_index]

        # Move images to device
        inputs = inputs.unsqueeze(0).to(device)
        fbp_img = fbp_img.unsqueeze(0).to(device)
        mbir_img = mbir_img.unsqueeze(0).to(device)
        dlr_img = dlr_img.unsqueeze(0).to(device)
        label = label.to(device)  # No need to unsqueeze here as it should be the correct shape

        # Enable gradient tracking for inputs
        inputs.requires_grad = True
        fbp_img.requires_grad = True
        mbir_img.requires_grad = True
        dlr_img.requires_grad = True

        # Generate Grad-CAM heatmaps for the original image
        grayscale_cam_original = gradcam(input_tensor=inputs)

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

        # Generate Grad-CAM heatmaps for FBP, MBIR, and DLR
        fbp_grayscale_cam = gradcam(input_tensor=fbp_img)
        mbir_grayscale_cam = gradcam(input_tensor=mbir_img)
        dlr_grayscale_cam = gradcam(input_tensor=dlr_img)

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

        # Plot Grad-CAM overlays
        axes[1, 0].imshow(show_cam_on_image(img_normalized, grayscale_cam_original[0, :]))
        axes[1, 0].set_title(f'Grad-CAM on Original\nTrue Label: {true_label}', fontsize=12)
        axes[1, 0].axis('off')

        axes[1, 1].imshow(show_cam_on_image(fbp_img_normalized, fbp_grayscale_cam[0, :]))
        axes[1, 1].set_title(f'Grad-CAM on FBP\nPred: {fbp_predicted_label}', fontsize=12)
        axes[1, 1].axis('off')

        axes[1, 2].imshow(show_cam_on_image(mbir_img_normalized, mbir_grayscale_cam[0, :]))
        axes[1, 2].set_title(f'Grad-CAM on MBIR\nPred: {mbir_predicted_label}', fontsize=12)
        axes[1, 2].axis('off')

        axes[1, 3].imshow(show_cam_on_image(dlr_img_normalized, dlr_grayscale_cam[0, :]))
        axes[1, 3].set_title(f'Grad-CAM on DLR\nPred: {dlr_predicted_label}', fontsize=12)
        axes[1, 3].axis('off')

        plt.tight_layout()
        plt.savefig(f"{output_folder}/gradcam_index{case_index}.png", dpi=300)
        plt.close()

def get_case_indices_from_folder(folder_path):
    case_indices = []
    for filename in os.listdir(folder_path):
        match = re.search(r'_case_(\d+)', filename)  # Regex to extract case number
        if match:
            case_indices.append(int(match.group(1)))
    return sorted(set(case_indices))  # Return unique and sorted indices


# Example usage: loading data and generating Grad-CAM images
if __name__ == "__main__":
    output_folder = 'figures/gradcam'  # Directory to save Grad-CAM images

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

    # Generate Grad-CAM images for all cases
    print("Generating Grad-CAM images for all cases...")
    generate_and_save_gradcam_images(model, original_dataset, fbp_dataset, mbir_dataset, dlr_dataset, device, output_folder, specific_cases)
    print(f"Grad-CAM images saved in '{output_folder}'.")