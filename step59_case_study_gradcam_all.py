import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import DataLoader
from step02_dataset_dataloader import RSNA_Intracranial_Hemorrhage_Dataset
from step04_cnn_classifier import SupervisedClassifierObserver, load_classifier
from step00_common_info import dicom_dir
from pytorch_grad_cam import GradCAMPlusPlus, GradCAM, FullGrad, AblationCAM, ScoreCAM, EigenCAM, LayerCAM, EigenGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import cv2
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import (
    show_cam_on_image, deprocess_image, preprocess_image)
from pytorch_grad_cam.metrics.cam_mult_image import CamMultImageConfidenceChange
from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget

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
cases_dir = 'figures/gradcam_all_cases'
if not os.path.exists(cases_dir):
    os.makedirs(cases_dir)

# Apply windowing
def apply_window(image, window_min, window_max):
    return np.clip(image, window_min, window_max)

def normalize(image):
    min_val = np.min(image)
    max_val = np.max(image)
    return (image - min_val) / (max_val - min_val)

# Grad-CAM function
def generate_grad_cam_map(model, input_tensor, target_layers, pred_label_idx):
    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=None)
    targets = [ClassifierOutputTarget(pred_label_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    
    return grayscale_cam[0]

# Plot cases with Grad-CAM
def plot_cases_with_gradcam(indices, model, target_layer, datasets, predictions, true_labels, title_prefix):
    dataset_names = ['Original', 'FBP', 'MBIR', 'DLR']

    for idx in indices:
        plt.figure(figsize=(16, 8), facecolor='black')

        brain_window = (0, 80)
        true_label = np.argmax(true_labels[idx])
        plt.suptitle(f'Case {idx} - True label: {label_names[true_label]}', color='white', fontsize=24)

        for i, dataset_name in enumerate(dataset_names):
            dataset = datasets[dataset_name]
            prediction = predictions[dataset_name]['predictions'][idx]
            image = dataset[idx][0].numpy().squeeze()

            # Apply brain window
            image_brain = apply_window(image, *brain_window)
            image_brain = normalize(image_brain)

            # Convert grayscale to RGB for Grad-CAM overlay
            image_rgb = np.stack((image_brain, image_brain, image_brain), axis=-1)

            # Plot brain window images without Grad-CAM
            plt.subplot(2, 4, i + 1)
            plt.imshow(image_brain, cmap='gray')
            plt.title(f'{dataset_name} Image', color='white', fontsize=18)
            plt.axis('off')

            # Generate Grad-CAM saliency map
            input_tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0)
            saliency_map = generate_grad_cam_map(model, input_tensor, target_layer, pred_label_idx=np.argmax(prediction))

            # Resize saliency map and overlay it
            saliency_map_res = cv2.resize(saliency_map, (image.shape[1], image.shape[0]))
            saliency_overlay_brain = show_cam_on_image(image_rgb, saliency_map_res, use_rgb=True)

            # Plot brain window images with Grad-CAM overlay
            plt.subplot(2, 4, i + 5)
            plt.imshow(saliency_overlay_brain, cmap='gray')
            plt.title(f'{dataset_name} prediction: \n{label_names[np.argmax(prediction)]}', color='white', fontsize=18)
            plt.axis('off')

        plt.tight_layout()

        # Save the figure
        plt.savefig(f'{cases_dir}/{title_prefix}_case_{idx}.png', dpi=300, facecolor='black')
        plt.close()

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

    # Plot all cases for Original, FBP, MBIR, and DLR
    plot_cases_with_gradcam(
        list(range(1, 5)) + list(range(501, 505)) + list(range(601, 605)) + list(range(701, 705)) + list(range(801, 805)) + list(range(901, 905)),
        observer.model,
        target_layer=[observer.model.resnet.layer4[2].conv3],  # Last layer of ResNet50
        datasets=datasets,
        predictions=results,
        true_labels=results['Original']['ground_truths'],
        title_prefix='All'
    )

    print('Finished plotting all cases!')
