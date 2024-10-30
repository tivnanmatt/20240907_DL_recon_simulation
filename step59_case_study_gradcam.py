import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import DataLoader
from step02_dataset_dataloader import RSNA_Intracranial_Hemorrhage_Dataset
from step04_cnn_classifier import SupervisedClassifierObserver, load_classifier
from step00_common_info import dicom_dir
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import cv2

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
cases_dir = 'figures/gradcam/v3'
if not os.path.exists(cases_dir):
    os.makedirs(cases_dir)

# Function to extract case indices from filenames
def extract_case_indices(cases_dir):
    case_files = os.listdir(cases_dir)
    case_indices = []
    for filename in case_files:
        # Assuming filenames follow the pattern 'some_name_case_X.png'
        if 'case_' in filename:
            try:
                idx = int(filename.split('case_')[1].split('.')[0])  # Extract the number after 'case_' and before '.png'
                case_indices.append(idx)
            except ValueError:
                continue  # Skip files that don't match the pattern
    return case_indices

# Apply windowing
def apply_window(image, window_min, window_max):
    return np.clip(image, window_min, window_max)

def normalize(image):
    min_val = np.min(image)
    max_val = np.max(image)
    return (image - min_val) / (max_val - min_val)

# Grad-CAM function
def generate_grad_cam_map(model, input_tensor, target_layers, pred_label_idx):
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers, reshape_transform=None)
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
    selected_case_indices = extract_case_indices('figures/cases/v3')

    print(f'Found {len(selected_case_indices)} cases to evaluate.')

    # Plot selected cases for Original, FBP, MBIR, and DLR
    plot_cases_with_gradcam(
        selected_case_indices,
        observer.model,
        target_layer=[observer.model.resnet.layer4[2].conv3],  # Last layer of ResNet50
        datasets=datasets,
        predictions=results,
        true_labels=results['Original']['ground_truths'],
        title_prefix='Selected'
    )

    print('Finished plotting selected cases!')



# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import torch
# import torch.nn.functional as F
# from torchvision import models, transforms
# from torch.utils.data import DataLoader
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
# from torch.utils.data import DataLoader
# from step02_dataset_dataloader import RSNA_Intracranial_Hemorrhage_Dataset
# from step04_cnn_classifier import SupervisedClassifierObserver, load_classifier
# from step00_common_info import dicom_dir

# # Ensure figures/cases directory exists
# cases_dir = 'figures/gradcam/v2'
# if not os.path.exists(cases_dir):
#     os.makedirs(cases_dir)

# # Define label names
# label_names = [
#     "no_hemorrhage", 
#     "epidural", 
#     "intraparenchymal", 
#     "intraventricular", 
#     "subarachnoid", 
#     "subdural"
# ]

# def load_results(csv_file, dicom_dir, expected_size):
#     dataset = RSNA_Intracranial_Hemorrhage_Dataset(csv_file, dicom_dir, expected_size=expected_size)
#     data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
#     return dataset, data_loader

# def identify_cases(ground_truths, original_predictions, fbp_predictions, mbir_predictions, dlr_predictions):
#     ground_truths = np.array(ground_truths)

#     # Softmax for probabilities and argmax for labels
#     pred_probs_original = np.exp(original_predictions) / np.sum(np.exp(original_predictions), axis=1, keepdims=True)
#     pred_labels_original = np.argmax(pred_probs_original, axis=1)

#     pred_probs_fbp = np.exp(fbp_predictions) / np.sum(np.exp(fbp_predictions), axis=1, keepdims=True)
#     pred_labels_fbp = np.argmax(pred_probs_fbp, axis=1)

#     pred_probs_mbir = np.exp(mbir_predictions) / np.sum(np.exp(mbir_predictions), axis=1, keepdims=True)
#     pred_labels_mbir = np.argmax(pred_probs_mbir, axis=1)

#     pred_probs_dlr = np.exp(dlr_predictions) / np.sum(np.exp(dlr_predictions), axis=1, keepdims=True)
#     pred_labels_dlr = np.argmax(pred_probs_dlr, axis=1)

#     true_labels = np.argmax(ground_truths, axis=1)

#     # Correct indices for each reconstruction
#     correct_original_indices = np.where(pred_labels_original == true_labels)[0]
#     correct_fbp_indices = np.where(pred_labels_fbp == true_labels)[0]
#     correct_mbir_indices = np.where(pred_labels_mbir == true_labels)[0]
#     correct_dlr_indices = np.where(pred_labels_dlr == true_labels)[0]

#     # Find cases where only Original, FBP, MBIR, or DLR is correct
#     original_correct_not_fbp_mbir_dlr = np.setdiff1d(correct_original_indices, np.union1d(np.union1d(correct_fbp_indices, correct_mbir_indices), correct_dlr_indices))
#     fbp_correct_not_original_mbri_dlri = np.setdiff1d(correct_fbp_indices, np.union1d(np.union1d(correct_original_indices, correct_mbir_indices), correct_dlr_indices))
#     mbir_correct_not_original_fbpi_dlri = np.setdiff1d(correct_mbir_indices, np.union1d(np.union1d(correct_original_indices, correct_fbp_indices), correct_dlr_indices))
#     dlri_correct_not_original_fbpi_mbri = np.setdiff1d(correct_dlr_indices, np.union1d(np.union1d(correct_original_indices, correct_fbp_indices), correct_mbir_indices))

#     return original_correct_not_fbp_mbir_dlr, fbp_correct_not_original_mbri_dlri, mbir_correct_not_original_fbpi_dlri, dlri_correct_not_original_fbpi_mbri

# def apply_window(image, window_min, window_max):
#     """
#     Apply a window to an image.
#     """
#     return np.clip(image, window_min, window_max)

# # Function to generate Grad-CAM heatmap
# def generate_grad_cam(model, img_tensor, target_layer):
#     model.eval()
#     output = model(img_tensor)
#     pred_class = output.argmax(dim=1).item()

#     model.zero_grad()
#     output[0, pred_class].backward()

#     gradients = target_layer.grad
#     activations = target_layer

#     pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

#     for i in range(activations.size(1)):
#         activations[:, i, :, :] *= pooled_gradients[i]

#     heatmap = torch.mean(activations, dim=1).squeeze()
#     heatmap = np.maximum(heatmap.detach().numpy(), 0)
#     heatmap /= np.max(heatmap)

#     return heatmap

# # Function to plot cases with Grad-CAM overlay
# def plot_cases_with_grad_cam(indices, original_dataset, fbp_dataset, mbir_dataset, dlr_dataset, 
#                              original_predictions, fbp_predictions, mbir_predictions, dlr_predictions, true_labels, title_prefix):
    
#     observer = SupervisedClassifierObserver(verbose=True, batch_size=1)
#     observer.model = load_classifier(observer.model, 'weights/supervised_classifier_resnet50_weights_09102024.pth')
#     model = observer.model
#     target_layer = model.resnet.layer4[2].conv3

#     crop = 16  # Define the cropping size
#     for idx in indices:
#         plt.figure(figsize=(16, 10), facecolor='black')

#         # Set the window parameters
#         brain_window = (0.0, 80.0)
#         bone_window = (-1000, 2000)

#         # True label for the current case
#         true_label = np.argmax(true_labels[idx])

#         # Main title: True label
#         plt.suptitle(f'Case {idx} - True label: {label_names[true_label]}', color='white', fontsize=24)

#         # Function to plot images with Grad-CAM overlay
#         def plot_image_with_grad_cam(dataset, predictions, subplot_index, title):
#             img, _ = dataset[idx]
#             img = Image.fromarray(img.numpy().squeeze(), mode='L')  # Convert tensor to PIL Image

#             img_tensor = torch.tensor(img).unsqueeze(0)

#             heatmap = generate_grad_cam(model, img_tensor, target_layer)
#             heatmap = Image.fromarray(np.uint8(255 * heatmap)).resize(img.size, Image.ANTIALIAS)
#             heatmap = np.array(heatmap)

#             superimposed_img = np.array(img) * 0.5 + heatmap * 0.5
#             superimposed_img = Image.fromarray(np.uint8(superimposed_img))

#             plt.subplot(2, 4, subplot_index)
#             plt.imshow(superimposed_img[crop:-crop, crop:-crop], cmap='gray')
#             pred = np.argmax(predictions[idx])
#             plt.title(f'{title} prediction: \n{label_names[pred]}', color='white', fontsize=18)
#             plt.axis('off')
#             plt.ylabel('[0, 80] HU', color='white')

#             plt.subplot(2, 4, subplot_index + 4)
#             windowed_image = apply_window(np.array(img), *bone_window)
#             plt.imshow(windowed_image[crop:-crop, crop:-crop], cmap='gray', vmin=bone_window, vmax=bone_window)
#             plt.axis('off')
#             plt.ylabel('[-1000, 2000] HU', color='white')

#         # Plot Original image
#         plot_image_with_grad_cam(original_dataset, original_predictions, 1, 'Original')

#         # Plot FBP image
#         plot_image_with_grad_cam(fbp_dataset, fbp_predictions, 2, 'FBP')

#         # Plot MBIR image
#         plot_image_with_grad_cam(mbir_dataset, mbir_predictions, 3, 'MBIR')

#         # Plot DLR image
#         plot_image_with_grad_cam(dlr_dataset, dlr_predictions, 4, 'DLR')

#         plt.tight_layout()

#         # Save the figure
#         plt.savefig(f'{cases_dir}/{title_prefix}_case_{idx}.png', dpi=300, facecolor='black')
#         plt.close()

# # Main code to run the analysis
# if __name__ == "__main__":
#     # CSV and DICOM directories
#     original_csv_file = 'data/metadata_evaluation.csv'
#     original_dicom_dir = dicom_dir

#     fbp_dicom_dir = 'data/FBP_reconstructions/'
#     mbir_dicom_dir = 'data/MBIR_reconstructions/'
#     dlr_dicom_dir = 'data/DLR_reconstructions/'

#     # Create dataset instances
#     print('Loading datasets...')
#     original_dataset = RSNA_Intracranial_Hemorrhage_Dataset(original_csv_file, original_dicom_dir, expected_size=512)
#     fbp_dataset = RSNA_Intracranial_Hemorrhage_Dataset(original_csv_file, fbp_dicom_dir, expected_size=256)
#     mbir_dataset = RSNA_Intracranial_Hemorrhage_Dataset(original_csv_file, mbir_dicom_dir, expected_size=256)
#     dlr_dataset = RSNA_Intracranial_Hemorrhage_Dataset(original_csv_file, dlr_dicom_dir, expected_size=256)

#     datasets = {
#         'FBP': fbp_dataset,
#         'MBIR': mbir_dataset,
#         'DLR': dlr_dataset,
#         'Original': original_dataset
#     }

#     # Initialize observer and load classifier
#     observer = SupervisedClassifierObserver(verbose=True, batch_size=1)
#     observer.model = load_classifier(observer.model, 'weights/supervised_classifier_resnet50_weights_09102024.pth')

#     # Load predictions
#     print('Evaluating datasets...')
#     results = {}
#     for key in ['Original', 'FBP', 'MBIR', 'DLR']:
#         accuracy, ground_truths, predictions = observer.evaluate(DataLoader(datasets[key], batch_size=1, shuffle=False), num_patients=len(datasets[key]))
#         results[key] = {
#             'accuracy': accuracy,
#             'ground_truths': ground_truths,
#             'predictions': predictions
#         }

#     ground_truths = results['Original']['ground_truths']

#     # Use FBP ground_truths for case identification as all datasets share the same ground truth
#     original_correct_not_fbp_mbir_dlr, fbp_correct_not_original_mbir_dlr, mbir_correct_not_original_fbp_dlr, dlr_correct_not_original_fbp_mbir = identify_cases(
#         ground_truths,
#         results['Original']['predictions'],
#         results['FBP']['predictions'],
#         results['MBIR']['predictions'],
#         results['DLR']['predictions']
#     )

#     # Plot cases with Grad-CAM overlay
#     print('Plotting cases with Grad-CAM...')
#     plot_cases_with_grad_cam(original_correct_not_fbp_mbir_dlr, datasets['Original'], datasets['FBP'], datasets['MBIR'], datasets['DLR'],
#                              results['Original']['predictions'], results['FBP']['predictions'], results['MBIR']['predictions'], results['DLR']['predictions'], 
#                              ground_truths, 'Original_Correct_Not_FBP_MBIR_DLR')

#     plot_cases_with_grad_cam(fbp_correct_not_original_mbir_dlr, datasets['Original'], datasets['FBP'], datasets['MBIR'], datasets['DLR'], 
#                              results['Original']['predictions'], results['FBP']['predictions'], results['MBIR']['predictions'], results['DLR']['predictions'], 
#                              ground_truths, 'FBP_Correct_Not_Original_MBIR_DLR')

#     plot_cases_with_grad_cam(mbir_correct_not_original_fbp_dlr, datasets['Original'], datasets['FBP'], datasets['MBIR'], datasets['DLR'], 
#                              results['Original']['predictions'], results['FBP']['predictions'], results['MBIR']['predictions'], results['DLR']['predictions'], 
#                              ground_truths, 'MBIR_Correct_Not_Original_FBP_DLR')

#     plot_cases_with_grad_cam(dlr_correct_not_original_fbp_mbir, datasets['Original'], datasets['FBP'], datasets['MBIR'], datasets['DLR'], 
#                              results['Original']['predictions'], results['FBP']['predictions'], results['MBIR']['predictions'], results['DLR']['predictions'], 
#                              ground_truths, 'DLR_Correct_Not_Original_FBP_MBIR')
# import os
# import torch
# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.image import show_cam_on_image
# import matplotlib.pyplot as plt
# from step00_common_info import dicom_dir
# from step02_dataset_dataloader import RSNA_Intracranial_Hemorrhage_Dataset
# from step04_cnn_classifier import SupervisedClassifier, load_classifier   
# import numpy as np
# import torch.nn.functional as F
# import re

# def apply_window(image, window_min, window_max):
#     """
#     Apply a window to an image.
#     """
#     return np.clip(image, window_min, window_max)

# def generate_and_save_gradcam_images(model, original_dataset, fbp_dataset, mbir_dataset, dlr_dataset, device, output_folder, specific_cases):
#     target_layer = model.resnet.layer4[2].conv3  # Reference the target layer for Grad-CAM
#     gradcam = GradCAM(model=model, target_layers=[target_layer])
#     model.eval()

#     # Define the brain window limits
#     window_min, window_max = 0, 80

#     for case_index in specific_cases:
#         # Load specific images from the datasets
#         inputs, label = original_dataset[case_index]
#         fbp_img, _ = fbp_dataset[case_index]
#         mbir_img, _ = mbir_dataset[case_index]
#         dlr_img, _ = dlr_dataset[case_index]

#         # Move images to device
#         inputs = inputs.unsqueeze(0).to(device)
#         fbp_img = fbp_img.unsqueeze(0).to(device)
#         mbir_img = mbir_img.unsqueeze(0).to(device)
#         dlr_img = dlr_img.unsqueeze(0).to(device)
#         label = label.to(device)  # No need to unsqueeze here as it should be the correct shape

#         # Enable gradient tracking for inputs
#         inputs.requires_grad = True
#         fbp_img.requires_grad = True
#         mbir_img.requires_grad = True
#         dlr_img.requires_grad = True

#         # Generate Grad-CAM heatmaps for the original image
#         grayscale_cam_original = gradcam(input_tensor=inputs)

#         with torch.no_grad():
#             # Get model predictions for each input
#             original_predictions = model(inputs)
#             fbp_predictions = model(fbp_img)
#             mbir_predictions = model(mbir_img)
#             dlr_predictions = model(dlr_img)

#             # Apply softmax to get probabilities
#             original_pred_probs = F.softmax(original_predictions, dim=1).cpu().numpy()
#             fbp_pred_probs = F.softmax(fbp_predictions, dim=1).cpu().numpy()
#             mbir_pred_probs = F.softmax(mbir_predictions, dim=1).cpu().numpy()
#             dlr_pred_probs = F.softmax(dlr_predictions, dim=1).cpu().numpy()

#             # Get predicted labels
#             original_predicted_label = np.argmax(original_pred_probs, axis=1).item()
#             fbp_predicted_label = np.argmax(fbp_pred_probs, axis=1).item()
#             mbir_predicted_label = np.argmax(mbir_pred_probs, axis=1).item()
#             dlr_predicted_label = np.argmax(dlr_pred_probs, axis=1).item()

#             # Get true label index from one-hot encoding
#             true_label = torch.argmax(label).item()

#         # Convert images to NumPy and normalize them
#         img = inputs[0].permute(1, 2, 0).detach().cpu().numpy()
#         img = apply_window(img, window_min, window_max)  # Apply brain window
#         img_normalized = (img - img.min()) / (img.max() - img.min())

#         # Normalize FBP, MBIR, and DLR images
#         fbp_img_normalized = fbp_img[0].permute(1, 2, 0).detach().cpu().numpy()
#         fbp_img_normalized = apply_window(fbp_img_normalized, window_min, window_max)
#         mbir_img_normalized = mbir_img[0].permute(1, 2, 0).detach().cpu().numpy()
#         mbir_img_normalized = apply_window(mbir_img_normalized, window_min, window_max)
#         dlr_img_normalized = dlr_img[0].permute(1, 2, 0).detach().cpu().numpy()
#         dlr_img_normalized = apply_window(dlr_img_normalized, window_min, window_max)   

#         # Normalize using numpy operations
#         fbp_img_normalized = (fbp_img_normalized - fbp_img_normalized.min()) / (fbp_img_normalized.max() - fbp_img_normalized.min())
#         mbir_img_normalized = (mbir_img_normalized - mbir_img_normalized.min()) / (mbir_img_normalized.max() - mbir_img_normalized.min())
#         dlr_img_normalized = (dlr_img_normalized - dlr_img_normalized.min()) / (dlr_img_normalized.max() - dlr_img_normalized.min())

#         # Generate Grad-CAM heatmaps for FBP, MBIR, and DLR
#         fbp_grayscale_cam = gradcam(input_tensor=fbp_img)
#         mbir_grayscale_cam = gradcam(input_tensor=mbir_img)
#         dlr_grayscale_cam = gradcam(input_tensor=dlr_img)

#         # Create a figure with two rows
#         fig, axes = plt.subplots(2, 4, figsize=(16, 8))

#         # Plot original image with true label
#         axes[0, 0].imshow(img_normalized, cmap='gray')
#         axes[0, 0].set_title(f'Original Image\nTrue Label: {true_label}', fontsize=12)
#         axes[0, 0].axis('off')

#         # Plot FBP image with predicted label
#         axes[0, 1].imshow(fbp_img_normalized, cmap='gray')
#         axes[0, 1].set_title(f'FBP Prediction: {fbp_predicted_label}', fontsize=12)
#         axes[0, 1].axis('off')

#         # Plot MBIR image with predicted label
#         axes[0, 2].imshow(mbir_img_normalized, cmap='gray')
#         axes[0, 2].set_title(f'MBIR Prediction: {mbir_predicted_label}', fontsize=12)
#         axes[0, 2].axis('off')

#         # Plot DLR image with predicted label
#         axes[0, 3].imshow(dlr_img_normalized, cmap='gray')
#         axes[0, 3].set_title(f'DLR Prediction: {dlr_predicted_label}', fontsize=12)
#         axes[0, 3].axis('off')

#         # Plot Grad-CAM overlays
#         axes[1, 0].imshow(show_cam_on_image(img_normalized, grayscale_cam_original[0, :]))
#         axes[1, 0].set_title(f'Grad-CAM on Original\nTrue Label: {true_label}', fontsize=12)
#         axes[1, 0].axis('off')

#         axes[1, 1].imshow(show_cam_on_image(fbp_img_normalized, fbp_grayscale_cam[0, :]))
#         axes[1, 1].set_title(f'Grad-CAM on FBP\nPred: {fbp_predicted_label}', fontsize=12)
#         axes[1, 1].axis('off')

#         axes[1, 2].imshow(show_cam_on_image(mbir_img_normalized, mbir_grayscale_cam[0, :]))
#         axes[1, 2].set_title(f'Grad-CAM on MBIR\nPred: {mbir_predicted_label}', fontsize=12)
#         axes[1, 2].axis('off')

#         axes[1, 3].imshow(show_cam_on_image(dlr_img_normalized, dlr_grayscale_cam[0, :]))
#         axes[1, 3].set_title(f'Grad-CAM on DLR\nPred: {dlr_predicted_label}', fontsize=12)
#         axes[1, 3].axis('off')

#         plt.tight_layout()
#         plt.savefig(f"{output_folder}/gradcam_index{case_index}.png", dpi=300)
#         plt.close()

# def get_case_indices_from_folder(folder_path):
#     case_indices = []
#     for filename in os.listdir(folder_path):
#         match = re.search(r'_case_(\d+)', filename)  # Regex to extract case number
#         if match:
#             case_indices.append(int(match.group(1)))
#     return sorted(set(case_indices))  # Return unique and sorted indices


# # Example usage: loading data and generating Grad-CAM images
# if __name__ == "__main__":
#     output_folder = 'figures/gradcam'  # Directory to save Grad-CAM images

#     os.makedirs(output_folder, exist_ok=True)

#      = torch.devicedevice('cuda' if torch.cuda.is_available() else 'cpu')

#     # Load the pre-trained model and weights
#     model = SupervisedClassifier().to(device)
#     load_classifier(model)

#     # Create dataset instances
#     print('Loading datasets...')
#     original_csv_file = 'data/metadata_evaluation.csv'
#     original_dicom_dir = dicom_dir

#     fbp_dicom_dir = 'data/FBP_reconstructions/'
#     mbir_dicom_dir = 'data/MBIR_reconstructions/'
#     dlr_dicom_dir = 'data/DLR_reconstructions/'

#     original_dataset = RSNA_Intracranial_Hemorrhage_Dataset(original_csv_file, original_dicom_dir, expected_size=512)
#     fbp_dataset = RSNA_Intracranial_Hemorrhage_Dataset(original_csv_file, fbp_dicom_dir, expected_size=256)
#     mbir_dataset = RSNA_Intracranial_Hemorrhage_Dataset(original_csv_file, mbir_dicom_dir, expected_size=256)
#     dlr_dataset = RSNA_Intracranial_Hemorrhage_Dataset(original_csv_file, dlr_dicom_dir, expected_size=256)

#     # Get all case indices from the figures/cases folder
#     case_folder = 'figures/cases'
#     specific_cases = get_case_indices_from_folder(case_folder)

#     # Generate Grad-CAM images for all cases
#     print("Generating Grad-CAM images for all cases...")
#     generate_and_save_gradcam_images(model, original_dataset, fbp_dataset, mbir_dataset, dlr_dataset, device, output_folder, specific_cases)
#     print(f"Grad-CAM images saved in '{output_folder}'.")