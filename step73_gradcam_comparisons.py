import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import DataLoader
from step02_dataset_dataloader import RSNA_Intracranial_Hemorrhage_Dataset
from step04_cnn_classifier import SupervisedClassifierObserver, load_classifier, SupervisedClassifier
from step00_common_info import dicom_dir
from pytorch_grad_cam import GradCAMPlusPlus, GradCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import cv2
import torch.nn.functional as F


# Define label names
label_names = [
    "no_hemorrhage", 
    "epidural", 
    "intraparenchymal", 
    "intraventricular", 
    "subarachnoid", 
    "subdural"
]

# Apply windowing
def apply_window(image, window_min, window_max):
    return np.clip(image, window_min, window_max)

# Ensure figures/cases directory exists
cases_dir = 'figures/gradcam/v3'
if not os.path.exists(cases_dir):
    os.makedirs(cases_dir)

def normalize(image):
    min_val = np.min(image)
    max_val = np.max(image)
    return (image - min_val) / (max_val - min_val)

# Grad-CAM function
def generate_grad_cam_map(model, input_tensor, target_layers, pred_label_idx):
    cam_reg = GradCAM(model=model, target_layers=target_layers, reshape_transform=None)
    cam_plus = GradCAMPlusPlus(model=model, target_layers=target_layers, reshape_transform=None)
    cam_full = FullGrad(model=model, target_layers=target_layers, reshape_transform=None)
    targets = [ClassifierOutputTarget(pred_label_idx)]
    grayscale_cam_reg = cam_reg(input_tensor=input_tensor, targets=targets)
    grayscale_cam_plus = cam_plus(input_tensor=input_tensor, targets=targets)
    grayscale_cam_full = cam_full(input_tensor=input_tensor, targets=targets)
    return grayscale_cam_reg[0], grayscale_cam_plus[0], grayscale_cam_full[0]

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

   # Generate Grad-CAM maps
    target_layers = [model.resnet.layer4[-1]]  # Update this according to your model's architecture
    gradcam_reg, gradcam_plus, fullgrad = generate_grad_cam_map(model, inputs, target_layers, original_predicted_label)
    fbp_gradcam_reg, fbp_gradcam_plus, fbp_fullgrad = generate_grad_cam_map(model, fbp_img, target_layers, fbp_predicted_label)
    mbir_gradcam_reg, mbir_gradcam_plus, mbir_fullgrad = generate_grad_cam_map(model, mbir_img, target_layers, mbir_predicted_label)
    dlr_gradcam_reg, dlr_gradcam_plus, dlr_fullgrad = generate_grad_cam_map(model, dlr_img, target_layers, dlr_predicted_label)

    # Normalize the saliency maps to the range [-1, 1]
    def normalize_saliency(saliency):
        saliency_min = saliency.min()
        saliency_max = saliency.max()
        # Normalize to the range [-1, 1]
        return 2 * (saliency - saliency_min) / (saliency_max - saliency_min) - 1

    gradcam_reg = normalize_saliency(gradcam_reg)
    gradcam_plus = normalize_saliency(gradcam_plus)
    fullgrad = normalize_saliency(fullgrad)
    
    fbp_gradcam_reg = normalize_saliency(fbp_gradcam_reg)
    fbp_gradcam_plus = normalize_saliency(fbp_gradcam_plus)
    fbp_fullgrad = normalize_saliency(fbp_fullgrad)

    mbir_gradcam_reg = normalize_saliency(mbir_gradcam_reg)
    mbir_gradcam_plus = normalize_saliency(mbir_gradcam_plus)
    mbir_fullgrad = normalize_saliency(mbir_fullgrad)

    dlr_gradcam_reg = normalize_saliency(dlr_gradcam_reg)
    dlr_gradcam_plus = normalize_saliency(dlr_gradcam_plus)
    dlr_fullgrad = normalize_saliency(dlr_fullgrad)

    # Create a figure with 4 rows and 5 columns
    fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(20, 20), facecolor='black', gridspec_kw={'width_ratios': [0.05, 1, 1, 1, 1]})

    row_labels = ['Original', 'FBP', 'MBIR', 'DLR']

    crop = 16
    # First column: Original, FBP, MBIR, DLR images
    axes[0, 1].imshow(img_normalized[crop:-crop, crop:-crop], cmap='gray')
    axes[1, 1].imshow(fbp_img_normalized[crop:-crop, crop:-crop], cmap='gray')
    axes[2, 1].imshow(mbir_img_normalized[crop:-crop, crop:-crop], cmap='gray')
    axes[3, 1].imshow(dlr_img_normalized[crop:-crop, crop:-crop], cmap='gray')

    # Second column: Regular gradcam
    axes[0, 2].imshow(img_normalized[crop:-crop, crop:-crop], cmap='gray')
    axes[0, 2].imshow(gradcam_reg[crop:-crop, crop:-crop], cmap='hot', alpha=0.5)
    axes[1, 2].imshow(fbp_img_normalized[crop:-crop, crop:-crop], cmap='gray')
    axes[1, 2].imshow(fbp_gradcam_reg[crop:-crop, crop:-crop], alpha=0.5, cmap='hot')
    axes[2, 2].imshow(mbir_img_normalized[crop:-crop, crop:-crop], cmap='gray')
    axes[2, 2].imshow(mbir_gradcam_reg[crop:-crop, crop:-crop], alpha=0.5, cmap='hot')
    axes[3, 2].imshow(dlr_img_normalized[crop:-crop, crop:-crop], cmap='gray')
    axes[3, 2].imshow(dlr_gradcam_reg[crop:-crop, crop:-crop], alpha=0.5, cmap='hot')

    # third column: Gradcam ++
    axes[0, 3].imshow(img_normalized[crop:-crop, crop:-crop], cmap='gray')
    axes[0, 3].imshow(gradcam_plus[crop:-crop, crop:-crop], alpha=0.5, cmap='hot')  
    axes[1, 3].imshow(fbp_img_normalized[crop:-crop, crop:-crop], cmap='gray')
    axes[1, 3].imshow(fbp_gradcam_plus[crop:-crop, crop:-crop], alpha=0.5, cmap='hot')
    axes[2, 3].imshow(mbir_img_normalized[crop:-crop, crop:-crop], cmap='gray')
    axes[2, 3].imshow(mbir_gradcam_plus[crop:-crop, crop:-crop], alpha=0.5, cmap='hot')
    axes[3, 3].imshow(dlr_img_normalized[crop:-crop, crop:-crop], cmap='gray')
    axes[3, 3].imshow(dlr_gradcam_plus[crop:-crop, crop:-crop], alpha=0.5, cmap='hot')

    # Fourth column: Fullgrad
    axes[0, 4].imshow(img_normalized[crop:-crop, crop:-crop], cmap='gray')
    axes[0, 4].imshow(fullgrad[crop:-crop, crop:-crop], alpha=0.5, cmap='hot')
    axes[1, 4].imshow(fbp_img_normalized[crop:-crop, crop:-crop], cmap='gray')
    axes[1, 4].imshow(fbp_fullgrad[crop:-crop, crop:-crop], alpha=0.5, cmap='hot')
    axes[2, 4].imshow(mbir_img_normalized[crop:-crop, crop:-crop], cmap='gray')
    axes[2, 4].imshow(mbir_fullgrad[crop:-crop, crop:-crop], alpha=0.5, cmap='hot')
    axes[3, 4].imshow(dlr_img_normalized[crop:-crop, crop:-crop], cmap='gray')
    axes[3, 4].imshow(dlr_fullgrad[crop:-crop, crop:-crop], alpha=0.5, cmap='hot')

    # Add row labels in the narrow column
    for i, label in enumerate(row_labels):
        # Create a centered text axis for each label
        text_axis = fig.add_subplot(4, 6, i * 6 + 1)
        text_axis.text(0, 0.5, label, va='center', ha='center', rotation='vertical',
                       fontsize=20, color='white', weight='bold', transform=text_axis.transAxes)
        text_axis.axis('off')  # Hide axis for text label

    # Add column titles
    column_titles = ['Reconstructions', 'Grad-CAM', 'Grad-CAM++', 'FullGrad']
    for j, title in enumerate(column_titles):
        axes[0, j + 1].set_title(title, fontsize=20, color='white', weight='bold')

    # Remove axis labels
    for i in range(4):
        for j in range(5):
            axes[i, j].axis('off')

    # Adjust spacing between images
    plt.subplots_adjust(wspace=0.1, hspace=0.1)        

    # Save the figure
    plt.savefig(f'figures/saliency/analysis/patient_{case_index}_analysis_grad.png', dpi=300)
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