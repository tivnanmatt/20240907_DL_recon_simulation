import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from step00_common_info import dicom_dir
from step02_dataset_dataloader import RSNA_Intracranial_Hemorrhage_Dataset
from step04_cnn_classifier import SupervisedClassifier, load_classifier, SupervisedClassifierObserver   

def apply_window(image, window_min, window_max):
    """Apply a window to an image."""
    return np.clip(image, window_min, window_max)

def occlusion_sensitivity(model, image, target_class=None, stride=8, occlusion_size=16):
    """Compute the occlusion sensitivity map for a given image."""
    model.eval()
    
    # Ensure the image is a PyTorch tensor
    if isinstance(image, np.ndarray):
        image = torch.tensor(image).float().to(device)  # Convert to tensor

    # Check if the image is in the shape (H, W, C)
    if image.dim() == 3 and image.size(2) == 1:  # (H, W, 1)
        image = image.squeeze(-1)  # Remove the last dimension (now it will be (H, W))

    if image.dim() == 2:  # If it's (H, W)
        image = image.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions (1, 1, H, W)

    elif image.dim() == 3:  # If it's (C, H, W)
        image = image.unsqueeze(0)  # Add batch dimension (1, C, H, W)

    # Get the prediction for the original image
    with torch.no_grad():
        original_prediction = F.softmax(model(image), dim=1).detach().cpu().numpy()

    if target_class is None:
        target_class = np.argmax(original_prediction)  # Use the predicted class if not specified

    sensitivity_map = np.zeros(image.shape[2:])  # Create an empty sensitivity map

    # Loop over the image with the specified stride
    for y in range(0, image.shape[2], stride):
        for x in range(0, image.shape[3], stride):
            # Create a copy of the image and occlude the region
            occluded_image = image.clone()
            occluded_image[0, 0, y:y+occlusion_size, x:x+occlusion_size] = 0  # Set the occluded area to zero
            
            # Get the prediction for the occluded image
            with torch.no_grad():
                occluded_prediction = F.softmax(model(occluded_image), dim=1).detach().cpu().numpy()

            # Calculate the change in prediction for the target class
            original_score = original_prediction[0, target_class]
            occluded_score = occluded_prediction[0, target_class]
            sensitivity_map[y:y+occlusion_size, x:x+occlusion_size] = np.abs(original_score - occluded_score)

    return sensitivity_map

def generate_and_save_occlusion_images(model, original_dataset, fbp_dataset, mbir_dataset, dlr_dataset, device, output_folder, specific_cases):
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

        fbp_img_normalized = fbp_img[0].permute(1, 2, 0).detach().cpu().numpy()
        fbp_img_normalized = apply_window(fbp_img_normalized, window_min, window_max)
        fbp_img_normalized = (fbp_img_normalized - fbp_img_normalized.min()) / (fbp_img_normalized.max() - fbp_img_normalized.min())

        mbir_img_normalized = mbir_img[0].permute(1, 2, 0).detach().cpu().numpy()
        mbir_img_normalized = apply_window(mbir_img_normalized, window_min, window_max)
        mbir_img_normalized = (mbir_img_normalized - mbir_img_normalized.min()) / (mbir_img_normalized.max() - mbir_img_normalized.min())

        dlr_img_normalized = dlr_img[0]. permute(1, 2, 0).detach().cpu().numpy()
        dlr_img_normalized = apply_window(dlr_img_normalized, window_min, window_max)
        dlr_img_normalized = (dlr_img_normalized - dlr_img_normalized.min()) / (dlr_img_normalized.max() - dlr_img_normalized.min())

       # Compute occlusion sensitivity maps
        target_class_original = original_predicted_label
        occlusion_original = occlusion_sensitivity(model, img_normalized, target_class=target_class_original, stride=8, occlusion_size=16)
        occlusion_fbp = occlusion_sensitivity(model, fbp_img_normalized, target_class=fbp_predicted_label, stride=8, occlusion_size=16)
        occlusion_mbir = occlusion_sensitivity(model, mbir_img_normalized, target_class=mbir_predicted_label, stride=8, occlusion_size=16)
        occlusion_dlr = occlusion_sensitivity(model, dlr_img_normalized, target_class=dlr_predicted_label, stride=8, occlusion_size=16)

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

        # Plot occlusion sensitivity maps
        axes[1, 0].imshow(occlusion_original, cmap='hot')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(occlusion_fbp, cmap='hot')
        axes[1, 1].axis('off')

        axes[1, 2].imshow(occlusion_mbir, cmap='hot')
        axes[1, 2].axis('off')

        axes[1, 3].imshow(occlusion_dlr, cmap='hot')
        axes[1, 3].axis('off')

        plt.tight_layout()
        plt.savefig(f"{output_folder}/patient_{case_index}_analysis_occlusion.png", dpi=300)
        plt.close()

# Example usage: loading data and generating occlusion sensitivity images
if __name__ == "__main__":
    output_folder = 'figures/saliency/analysis'  # Directory to save occlusion sensitivity images

    os.makedirs(output_folder, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the pre-trained model and weights
    observer = SupervisedClassifierObserver()
    # load_classifier(model)
    observer.model = load_classifier(observer.model, 'weights/supervised_classifier_resnet50_weights_09102024.pth')

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
    # specific_cases = list(range(1, 5)) + list(range(501, 505)) + list(range(601, 605)) + list(range(701, 705)) + list(range(801, 805)) + list(range(901, 905))
    specific_cases = [6, 502, 520, 600, 684, 702, 725, 805, 812, 903, 931]

    # Generate occlusion sensitivity images for all cases
    print("Generating occlusion sensitivity images for all cases...")
    generate_and_save_occlusion_images(observer.model, original_dataset, fbp_dataset, mbir_dataset, dlr_dataset, device, output_folder, specific_cases)
    print(f"Occlusion sensitivity images saved in '{output_folder}'.")