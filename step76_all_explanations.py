import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from step00_common_info import dicom_dir
from step02_dataset_dataloader import RSNA_Intracranial_Hemorrhage_Dataset
from step04_cnn_classifier import SupervisedClassifier, load_classifier   

from torch.utils.data import DataLoader
from pytorch_grad_cam import GradCAMPlusPlus, GradCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import cv2
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

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


# Define label names
label_names = [
    "no_hemorrhage", 
    "epidural", 
    "intraparenchymal", 
    "intraventricular", 
    "subarachnoid", 
    "subdural"
]

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
    # Prepare the figure with 5 columns and 8 rows
    
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

    target_class_original = original_predicted_label
    occlusion_original = occlusion_sensitivity(model, img_normalized, target_class=target_class_original, stride=8, occlusion_size=16)
    occlusion_fbp = occlusion_sensitivity(model, fbp_img_normalized, target_class=fbp_predicted_label, stride=8, occlusion_size=16)
    occlusion_mbir = occlusion_sensitivity(model, mbir_img_normalized, target_class=mbir_predicted_label, stride=8, occlusion_size=16)
    occlusion_dlr = occlusion_sensitivity(model, dlr_img_normalized, target_class=dlr_predicted_label, stride=8, occlusion_size=16)

    fig, axes = plt.subplots(
    7, 4, 
    figsize=(30, 40), 
    facecolor='black', 
    gridspec_kw={
        'width_ratios': [1, 1, 1, 1],
        'height_ratios': [1, 1, 1, 1, 1, 1, 1]  # Modify values here to adjust row height
    }
)
    
    crop = 16 

    row_labels = ['Original', 'Occlusion Sensitivity', 'Vanilla Gradients', 'Guided Backpropagation', 'Gradcam', ' Gradcam++', 'FullGrad']
    # Add row labels in the narrow column
    for i, label in enumerate(row_labels):
        # Create a centered text axis for each label
        text_axis = fig.add_subplot(7, 1, i + 1)
        text_axis.text(0, 0.5, label, va='center', ha='center', rotation='vertical',
                       fontsize=24, color='white', weight='bold', transform=text_axis.transAxes)
        text_axis.axis('off')  # Hide axis for text label

    # Add column titles
    column_titles = ['Original', 'FBP', 'MBIR', 'DLR']
    for j, title in enumerate(column_titles):
        axes[0, j].set_title(title, fontsize=24, color='white', weight='bold')

    # Display original images
    axes[0, 0].imshow(img_normalized[crop:-crop, crop:-crop], cmap='gray')
    axes[0, 1].imshow(fbp_img_normalized[crop:-crop, crop:-crop], cmap='gray')
    axes[0, 2].imshow(mbir_img_normalized[crop:-crop, crop:-crop], cmap='gray')
    axes[0, 3].imshow(dlr_img_normalized[crop:-crop, crop:-crop], cmap='gray')

    # Occlusion sensitivity
    axes[1, 0].imshow(occlusion_original, cmap='hot')
    axes[1, 1].imshow(occlusion_fbp, cmap='hot')
    axes[1, 2].imshow(occlusion_mbir, cmap='hot')
    axes[1, 3].imshow(occlusion_dlr, cmap='hot')

    # Third column: Overlay of vanilla gradients on images
    axes[2, 0].imshow(img_normalized[crop:-crop, crop:-crop], cmap='gray')
    axes[2, 0].imshow(saliency_original.squeeze().detach().cpu().numpy()[crop:-crop, crop:-crop], cmap='hot', alpha=0.5)
    axes[2, 1].imshow(fbp_img_normalized[crop:-crop, crop:-crop], cmap='gray')
    axes[2, 1].imshow(saliency_fbp.squeeze().detach().cpu().numpy()[crop:-crop, crop:-crop], cmap='hot', alpha=0.5)
    axes[2, 2].imshow(mbir_img_normalized[crop:-crop, crop:-crop], cmap='gray')
    axes[2, 2].imshow(saliency_mbir.squeeze().detach().cpu().numpy()[crop:-crop, crop:-crop], cmap='hot', alpha=0.5)
    axes[2, 3].imshow(dlr_img_normalized[crop:-crop, crop:-crop], cmap='gray')
    axes[2, 3].imshow(saliency_dlr.squeeze().detach().cpu().numpy()[crop:-crop, crop:-crop], cmap='hot', alpha=0.5)
    
    # Fifth column: Overlay of guided backpropagation on images
    axes[3, 0].imshow(img_normalized[crop:-crop, crop:-crop], cmap='gray')
    axes[3, 0].imshow(guided_original_normalized[crop:-crop, crop:-crop], cmap='hot', alpha=0.5)
    axes[3, 1].imshow(fbp_img_normalized[crop:-crop, crop:-crop], cmap='gray')
    axes[3, 1].imshow(guided_fbp_normalized[crop:-crop, crop:-crop], cmap='hot', alpha=0.5)
    axes[3, 2].imshow(mbir_img_normalized[crop:-crop, crop:-crop], cmap='gray')
    axes[3, 2].imshow(guided_mbir_normalized[crop:-crop, crop:-crop], cmap='hot', alpha=0.5)
    axes[3, 3].imshow(dlr_img_normalized[crop:-crop, crop:-crop], cmap='gray')
    axes[3, 3].imshow(guided_dlr_normalized[crop:-crop, crop:-crop], cmap='hot', alpha=0.5)

    # Second column: Regular gradcam
    axes[4, 0].imshow(img_normalized[crop:-crop, crop:-crop], cmap='gray')
    axes[4, 0].imshow(gradcam_reg[crop:-crop, crop:-crop], cmap='hot', alpha=0.5)
    axes[4, 1].imshow(fbp_img_normalized[crop:-crop, crop:-crop], cmap='gray')
    axes[4, 1].imshow(fbp_gradcam_reg[crop:-crop, crop:-crop], alpha=0.5, cmap='hot')
    axes[4, 2].imshow(mbir_img_normalized[crop:-crop, crop:-crop], cmap='gray')
    axes[4, 2].imshow(mbir_gradcam_reg[crop:-crop, crop:-crop], alpha=0.5, cmap='hot')
    axes[4, 3].imshow(dlr_img_normalized[crop:-crop, crop:-crop], cmap='gray')
    axes[4, 3].imshow(dlr_gradcam_reg[crop:-crop, crop:-crop], alpha=0.5, cmap='hot')
    
    # third column: Gradcam ++
    axes[5, 0].imshow(img_normalized[crop:-crop, crop:-crop], cmap='gray')
    axes[5, 0].imshow(gradcam_plus[crop:-crop, crop:-crop], alpha=0.5, cmap='hot')  
    axes[5, 1].imshow(fbp_img_normalized[crop:-crop, crop:-crop], cmap='gray')
    axes[5, 1].imshow(fbp_gradcam_plus[crop:-crop, crop:-crop], alpha=0.5, cmap='hot')
    axes[5, 2].imshow(mbir_img_normalized[crop:-crop, crop:-crop], cmap='gray')
    axes[5, 2].imshow(mbir_gradcam_plus[crop:-crop, crop:-crop], alpha=0.5, cmap='hot')
    axes[5, 3].imshow(dlr_img_normalized[crop:-crop, crop:-crop], cmap='gray')
    axes[5, 3].imshow(dlr_gradcam_plus[crop:-crop, crop:-crop], alpha=0.5, cmap='hot')

    # Fourth column: Fullgrad
    axes[6, 0].imshow(img_normalized[crop:-crop, crop:-crop], cmap='gray')
    axes[6, 0].imshow(fullgrad[crop:-crop, crop:-crop], alpha=0.5, cmap='hot')
    axes[6, 1].imshow(fbp_img_normalized[crop:-crop, crop:-crop], cmap='gray')
    axes[6, 1].imshow(fbp_fullgrad[crop:-crop, crop:-crop], alpha=0.5, cmap='hot')
    axes[6, 2].imshow(mbir_img_normalized[crop:-crop, crop:-crop], cmap='gray')
    axes[6, 2].imshow(mbir_fullgrad[crop:-crop, crop:-crop], alpha=0.5, cmap='hot')
    axes[6, 3].imshow(dlr_img_normalized[crop:-crop, crop:-crop], cmap='gray')
    axes[6, 3].imshow(dlr_fullgrad[crop:-crop, crop:-crop], alpha=0.5, cmap='hot')


    # Hide axes for empty plots
    for ax in axes.flatten():
        ax.axis('off')

    # Adjust spacing between images
    plt.subplots_adjust(wspace=0.01, hspace=0.01)        

    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'patient_{case_index}_analysis_all_v2.png'))
    plt.close(fig)

def normalize_saliency(saliency):
    if isinstance(saliency, np.ndarray):
        saliency = torch.tensor(saliency)

    saliency_min = saliency.min()
    saliency_max = saliency.max()
    
    if saliency_max == saliency_min:
        return torch.zeros_like(saliency)  # or return saliency if you want to leave it unchanged
    
    # Normalize the saliency map
    return (saliency - saliency_min) / (saliency_max - saliency_min)

def compute_saliency_metrics(model, dataset, device):
    print("Computing saliency metrics...")
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

def compute_gradcam_metrics(model, dataset, device, target_layers):
    print("Computing Grad-CAM metrics...")
    model.eval()
    gradcam_metrics = []

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

        # Generate Grad-CAM maps
        grad_cam_reg, grad_cam_plus, grad_cam_full = generate_grad_cam_map(model, inputs, target_layers, predicted_label)

        # Normalize Grad-CAM maps
        grad_cam_reg_normalized = normalize_saliency(grad_cam_reg)
        grad_cam_plus_normalized = normalize_saliency(grad_cam_plus)
        grad_cam_full_normalized = normalize_saliency(grad_cam_full)

        # Append metrics to the list
        gradcam_metrics.append({
            'mean_gradcam_reg': grad_cam_reg_normalized.mean().item(),
            'variance_gradcam_reg': grad_cam_reg_normalized.var().item(),
            'mean_gradcam_plus': grad_cam_plus_normalized.mean().item(),
            'variance_gradcam_plus': grad_cam_plus_normalized.var().item(),
            'mean_gradcam_full': grad_cam_full_normalized.mean().item(),
            'variance_gradcam_full': grad_cam_full_normalized.var().item(),
        })

    return gradcam_metrics    

def compute_occlusion_metrics(model, dataset, device, stride=8, occlusion_size=16):
    print("Computing occlusion metrics...")
    model.eval()
    occlusion_metrics = []

    for idx in range(len(dataset)):
        inputs, label = dataset[idx]
        inputs = inputs.unsqueeze(0).to(device)  # Ensure inputs are batched
        label = label.to(device)
        
        # Get the actual class label
        if label.dim() == 1:  # One-hot encoded vector
            label = torch.argmax(label).item()  # Get the index of the class
        else:
            raise ValueError("Label tensor is not in the expected one-hot encoded format.")

        # Compute occlusion sensitivity map
        sensitivity_map = occlusion_sensitivity(model, inputs[0], target_class=None, stride=stride, occlusion_size=occlusion_size)

        # Normalize sensitivity map
        sensitivity_map_normalized = normalize_saliency(sensitivity_map)

        # Append metrics to the list
        occlusion_metrics.append({
            'mean_occlusion': sensitivity_map_normalized.mean(),
            'variance_occlusion': sensitivity_map_normalized.var(),
        })

    return occlusion_metrics

def save_metrics_to_csv(saliency_metrics, gradcam_metrics, occlusion_metrics, filename='results/combined_metrics.csv'):
    # Create DataFrames from the metrics
    df_saliency = pd.DataFrame(saliency_metrics)
    df_gradcam = pd.DataFrame(gradcam_metrics)
    df_occlusion = pd.DataFrame(occlusion_metrics)

    # Combine all metrics into one DataFrame
    combined_metrics = pd.concat([df_saliency, df_gradcam, df_occlusion], axis=1)

    # Save the combined DataFrame to an Excel file
    combined_metrics.to_csv(filename, index=False)

    print(f"Metrics saved to '{filename}'.")

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

    # Analyze patient with index 812
    plot_patient_analysis(model, original_dataset, fbp_dataset, mbir_dataset, dlr_dataset, 520, device)
    print("Analysis figure saved in 'figures/saliency/v3' directory.")

        # Compute and plot saliency metrics separately for each dataset
    datasets = {
        'original': original_dataset,
        'fbp': fbp_dataset,
        'mbir': mbir_dataset,
        'dlr': dlr_dataset
    }


# for dataset_name, dataset in datasets.items():
#     # Reinitialize the lists for each dataset to prevent accumulation
#     saliency_metrics = compute_saliency_metrics(model, dataset, device)
#     gradcam_metrics = compute_gradcam_metrics(model, dataset, device, target_layers=[model.resnet.layer4[-1]])
#     occlusion_metrics = compute_occlusion_metrics(model, dataset, device)

#     # Save each dataset's metrics to its CSV file
#     save_metrics_to_csv(saliency_metrics, gradcam_metrics, occlusion_metrics, filename=f'results/combined_metrics_{dataset_name}.csv')
#     print(f"Metrics saved to 'results/combined_metrics_{dataset_name}.csv'.")