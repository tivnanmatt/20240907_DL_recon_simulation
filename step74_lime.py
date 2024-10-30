# Has a problem with gray scale images

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import re
from lime import lime_image
from skimage.segmentation import mark_boundaries
from step00_common_info import dicom_dir
from step02_dataset_dataloader import RSNA_Intracranial_Hemorrhage_Dataset
from step04_cnn_classifier import SupervisedClassifier, load_classifier   

def apply_window(image, window_min, window_max):
    """
    Apply a window to an image.
    """
    return np.clip(image, window_min, window_max)

def compute_lime_explanation(model, image, true_label):
    """
    Compute LIME explanation for the given image using the model.
    """
    # Convert grayscale image to RGB by repeating the channel
    if image.ndim == 2:  # Check if the image is grayscale
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)  # Convert to (H, W, 3)
    elif image.ndim == 3 and image.shape[2] == 1:  # Check if the image has a single channel
        image = np.repeat(image, 3, axis=2)  # Convert to (H, W, 3)

    # Check if the image is now RGB
    print(f"Image shape for LIME: {image.shape}")  # Debugging line

    explainer = lime_image.LimeImageExplainer()
    
    # Define a prediction function for LIME
    def predict_fn(images):
        images = torch.tensor(images).permute(0, 3, 1, 2).float().to(device)  # Change to (N, C, H, W)
        with torch.no_grad():
            logits = model(images)
            return F.softmax(logits, dim=1).cpu().numpy()

    # Generate LIME explanation
    explanation = explainer.explain_instance(
        image,
        predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )
    
    return explanation

def generate_and_save_lime_images(model, original_dataset, fbp_dataset, mbir_dataset, dlr_dataset, device, output_folder, specific_cases):
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

        # Compute LIME explanations
        lime_original = compute_lime_explanation(model, img_normalized, original_predicted_label)
        lime_fbp = compute_lime_explanation(model, fbp_img[0].permute(1, 2, 0).detach().cpu().numpy (), fbp_predicted_label)
        lime_mbir = compute_lime_explanation(model, mbir_img[0].permute(1, 2, 0).detach().cpu().numpy(), mbir_predicted_label)
        lime_dlr = compute_lime_explanation(model, dlr_img[0].permute(1, 2, 0).detach().cpu().numpy(), dlr_predicted_label)

        # Create a figure with two rows
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        plt.suptitle(f'Case {case_index} - True label: {true_label}', fontsize=16)

        # Plot original image with true label
        axes[0, 0].imshow(img_normalized, cmap='gray')
        axes[0, 0].set_title(f'Original Prediction: {original_predicted_label}', fontsize=12)
        axes[0, 0].axis('off')

        # Plot FBP image with predicted label
        axes[0, 1].imshow(fbp_img[0].permute(1, 2, 0).detach().cpu().numpy(), cmap='gray')
        axes[0, 1].set_title(f'FBP Prediction: {fbp_predicted_label}', fontsize=12)
        axes[0, 1].axis('off')

        # Plot MBIR image with predicted label
        axes[0, 2].imshow(mbir_img[0].permute(1, 2, 0).detach().cpu().numpy(), cmap='gray')
        axes[0, 2].set_title(f'MBIR Prediction: {mbir_predicted_label}', fontsize=12)
        axes[0, 2].axis('off')

        # Plot DLR image with predicted label
        axes[0, 3].imshow(dlr_img[0].permute(1, 2, 0).detach().cpu().numpy(), cmap='gray')
        axes[0, 3].set_title(f'DLR Prediction: {dlr_predicted_label}', fontsize=12)
        axes[0, 3].axis('off')

        # Overlay LIME explanations on the original image
        temp, mask = lime_original.get_image_and_mask(original_predicted_label, positive_only=True, num_features=5, hide_rest=False)
        axes[1, 0].imshow(mark_boundaries(temp, mask))
        axes[1, 0].axis('off')

        temp, mask = lime_fbp.get_image_and_mask(fbp_predicted_label, positive_only=True, num_features=5, hide_rest=False)
        axes[1, 1].imshow(mark_boundaries(temp, mask))
        axes[1, 1].axis('off')

        temp, mask = lime_mbir.get_image_and_mask(mbir_predicted_label, positive_only=True, num_features=5, hide_rest=False)
        axes[1, 2].imshow(mark_boundaries(temp, mask))
        axes[1, 2].axis('off')

        temp, mask = lime_dlr.get_image_and_mask(dlr_predicted_label, positive_only=True, num_features=5, hide_rest=False)
        axes[1, 3].imshow(mark_boundaries(temp, mask))
        axes[1, 3].axis('off')

        plt.tight_layout()
        plt.savefig(f"{output_folder}/lime_index{case_index}.png", dpi=300)
        plt.close()

# Example usage: loading data and generating LIME images
if __name__ == "__main__":
    output_folder = 'figures/saliency/lime'  # Directory to save LIME images

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
    case_folder = 'figures/cases/v2'
    # specific_cases = get_case_indices_from_folder(case_folder)
    specific_cases = list(range(1, 5)) + list(range(501, 505)) + list(range(601, 605)) + list(range(701, 705)) + list(range(801, 805)) + list(range(901, 905))

    # Generate LIME images for all cases
    print("Generating LIME images for all cases...")
    generate_and_save_lime_images(model, original_dataset, fbp_dataset, mbir_dataset, dlr_dataset, device, output_folder, specific_cases)
    print(f"LIME images saved in '{output_folder}'.")