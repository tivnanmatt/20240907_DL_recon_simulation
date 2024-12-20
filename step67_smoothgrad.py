import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import captum
from captum.attr import (
    SmoothGrad,
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
)
from step00_common_info import dicom_dir
from step02_dataset_dataloader import RSNA_Intracranial_Hemorrhage_Dataset
from step04_cnn_classifier import SupervisedClassifier, load_classifier

def apply_window(image, window_min, window_max):
    """
    Apply a window to an image.
    """
    return np.clip(image, window_min, window_max)

def compute_smoothgrad(model, inputs, target_class, device, std_dev=0.1, n_samples=50):
    """
    Compute SmoothGrad for the given input using the model.
    """
    model.eval()
    
    # Initialize SmoothGrad object
    smoothgrad = SmoothGrad(model)
    
    # Compute the smoothgrad
    attributions = smoothgrad.attribute(inputs, target=target_class, n_samples=n_samples, std_dev=std_dev)
    
    # Get the maximum across channels
    attributions = torch.max(attributions, dim=1).values
    
    return attributions

def generate_and_save_sg_images(model, original_dataset, fbp_dataset, mbir_dataset, dlr_dataset, device, output_folder, specific_cases):
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

        # Move images to device and add a batch dimension
        inputs = inputs.unsqueeze(0).to(device)
        fbp_img = fbp_img.unsqueeze(0).to(device)
        mbir_img = mbir_img.unsqueeze(0).to(device)
        dlr_img = dlr_img.unsqueeze(0).to(device)
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
        dlr_img_normalized = ( dlr_img_normalized - dlr_img_normalized.min()) / (dlr_img_normalized.max() - dlr_img_normalized.min())

        # Compute SmoothGrad
        sg_original = compute_smoothgrad(model, inputs, original_predicted_label, device)
        sg_fbp = compute_smoothgrad(model, fbp_img, fbp_predicted_label, device)
        sg_mbir = compute_smoothgrad(model, mbir_img, mbir_predicted_label, device)
        sg_dlr = compute_smoothgrad(model, dlr_img, dlr_predicted_label, device)

        sg_original = (sg_original - sg_original.min()) / (sg_original.max() - sg_original.min())
        sg_fbp = (sg_fbp - sg_fbp.min()) / (sg_fbp.max() - sg_fbp.min())
        sg_mbir = (sg_mbir - sg_mbir.min()) / (sg_mbir.max() - sg_mbir.min())
        sg_dlr = (sg_dlr - sg_dlr.min()) / (sg_dlr.max() - sg_dlr.min())

        # Convert attributions to numpy arrays
        sg_original = sg_original.squeeze().detach().cpu().numpy()
        sg_fbp = sg_fbp.squeeze().detach().cpu().numpy()
        sg_mbir = sg_mbir.squeeze().detach().cpu().numpy()
        sg_dlr = sg_dlr.squeeze().detach().cpu().numpy()

        # Create a figure with two rows
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        plt.suptitle(f'Case {case_index} - True label: {true_label}', fontsize=16)

        # Plot original image with true label
        axes[0, 0].imshow(img_normalized, cmap='gray')
        axes[0, 0].set_title(f'Original Prediction: {original_predicted_label}', fontsize=12)
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

        # Overlay SmoothGrad heatmaps on the original image
        axes[1, 0].imshow(img_normalized, cmap='gray')
        axes[1, 0].imshow(sg_original, cmap='hot', alpha=0.5)
        axes[1, 0].axis('off')

        axes[1, 1].imshow(fbp_img_normalized, cmap='gray')
        axes[1, 1].imshow(sg_fbp, cmap='hot', alpha=0.5)
        axes[1, 1].axis('off')

        axes[1, 2].imshow(mbir_img_normalized, cmap='gray')
        axes[1, 2].imshow(sg_mbir, cmap='hot', alpha=0.5)
        axes[1, 2].axis('off')

        axes[1, 3].imshow(dlr_img_normalized, cmap='gray')
        axes[1, 3].imshow(sg_dlr, cmap='hot', alpha=0.5)
        axes[1, 3].axis('off')

        plt.tight_layout()
        plt.savefig(f"{output_folder}/sg_case_{case_index}.png", dpi=300)
        plt.close()

# Example usage
if __name__ == "__main__":
    output_folder = 'figures/saliency/smoothgrad'
    os.makedirs(output_folder, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the pre-trained model and weights
    model = SupervisedClassifier().to(device)
    load_classifier(model)

    original_csv_file = 'data/metadata_evaluation.csv'
    original_dicom_dir = dicom_dir

    fbp_dicom_dir = 'data/FBP_reconstructions/'
    mbir_dicom_dir = 'data/MBIR_reconstructions/'
    dlr_dicom_dir = 'data/DLR_reconstructions/'

    # Load datasets
    original_dataset = RSNA_Intracranial_Hemorrhage_Dataset (original_csv_file, original_dicom_dir, expected_size=512)
    fbp_dataset = RSNA_Intracranial_Hemorrhage_Dataset(original_csv_file, fbp_dicom_dir, expected_size=256)
    mbir_dataset = RSNA_Intracranial_Hemorrhage_Dataset(original_csv_file, mbir_dicom_dir, expected_size=256)
    dlr_dataset = RSNA_Intracranial_Hemorrhage_Dataset(original_csv_file, dlr_dicom_dir, expected_size=256)

    # Specific cases to visualize
    specific_cases = list(range(1, 5)) + list(range(501, 505)) + list(range(601, 605)) + list(range(701, 705)) + list(range(801, 805)) + list(range(901, 905))

    # Generate and save SmoothGrad images
    generate_and_save_sg_images(model, original_dataset, fbp_dataset, mbir_dataset, dlr_dataset, device, output_folder, specific_cases)