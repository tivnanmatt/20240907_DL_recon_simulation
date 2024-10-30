import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from step00_common_info import dicom_dir
import numpy as np
import os
from step02_dataset_dataloader import RSNA_Intracranial_Hemorrhage_Dataset
from step04_cnn_classifier import SupervisedClassifierObserver, load_classifier

label_names = [
    "no_hemorrhage", 
    "epidural", 
    "intraparenchymal", 
    "intraventricular", 
    "subarachnoid", 
    "subdural"
]

def apply_window(image, window_min, window_max):
    return np.clip(image, window_min, window_max)

def plot_cases(indices, original_dataset, fbp_dataset, mbir_dataset, dlr_dataset, 
               original_predictions, fbp_predictions, mbir_predictions, dlr_predictions, true_labels, title_prefix, cases_dir):
    crop = 16  # Define the cropping size

    if not os.path.exists(cases_dir):
        os.makedirs(cases_dir)

    for idx in indices:
        plt.figure(figsize=(16, 10), facecolor='black')

        # Set the window parameters
        brain_window = (0.0, 80.0)
        bone_window = (-1000, 2000)

        # True label for the current case
        true_label = np.argmax(true_labels[idx])
        true_label = label_names[true_label]

        # Main title: True label
        plt.suptitle(f'Case {idx} - True label: {true_label}', color='white', fontsize=24)

        # Plot Original image
        plt.subplot(2, 4, 1)
        original_image = original_dataset[idx][0].numpy().squeeze()
        original_image_brain = apply_window(original_image, *brain_window)
        plt.imshow(original_image_brain[crop:-crop, crop:-crop], cmap='gray', vmin=brain_window[0], vmax=brain_window[1])
        original_pred = np.argmax(original_predictions[idx])
        original_pred_label = label_names[original_pred]
        plt.title(f'Original prediction: \n{original_pred_label}', color='white', fontsize=18)
        plt.axis('off')
        plt.ylabel('[0, 80] HU', color='white')

        plt.subplot(2, 4, 5)
        original_image_bone = apply_window(original_image, *bone_window)
        plt.imshow(original_image_bone[crop:-crop, crop:-crop], cmap='gray', vmin=bone_window[0], vmax=bone_window[1])
        plt.axis('off')
        plt.ylabel('[-1000, 2000] HU', color='white')

        # Plot FBP image
        plt.subplot(2, 4, 2)
        fbp_image = fbp_dataset[idx][0].numpy().squeeze()
        fbp_image_brain = apply_window(fbp_image, *brain_window)
        fbp_pred = np.argmax(fbp_predictions[idx])
        fbp_pred_label = label_names[fbp_pred]
        plt.imshow(fbp_image_brain[crop:-crop, crop:-crop], cmap='gray', vmin=brain_window[0], vmax=brain_window[1])
        plt.title(f'FBP prediction: \n{fbp_pred_label}', color='white', fontsize=18)
        plt.axis('off')
        plt.ylabel('[0, 80] HU', color='white')

        plt.subplot(2, 4, 6)
        fbp_image_bone = apply_window(fbp_image, *bone_window)
        plt.imshow(fbp_image_bone[crop:-crop, crop:-crop], cmap='gray', vmin=bone_window[0], vmax=bone_window[1])
        plt.axis('off')
        plt.ylabel('[-1000, 2000] HU', color='white')

        # Plot MBIR image
        plt.subplot(2, 4, 3)
        mbir_image = mbir_dataset[idx][0].numpy().squeeze()
        mbir_image_brain = apply_window(mbir_image, *brain_window)
        mbir_pred = np.argmax(mbir_predictions[idx])
        mbir_pred_label = label_names[mbir_pred]
        plt.imshow(mbir_image_brain[crop:-crop, crop:-crop], cmap='gray', vmin=brain_window[0], vmax=brain_window[1])
        plt.title(f'MBIR prediction: \n{mbir_pred_label}', color='white', fontsize=18)
        plt.axis('off')
        plt.ylabel('[0, 80] HU', color='white')

        plt .subplot(2, 4, 7)
        mbir_image_bone = apply_window(mbir_image, *bone_window)
        plt.imshow(mbir_image_bone[crop:-crop, crop:-crop], cmap='gray', vmin=bone_window[0], vmax=bone_window[1])
        plt.axis('off')
        plt.ylabel('[-1000, 2000] HU', color='white')

        # Plot DLR image
        plt.subplot(2, 4, 4)
        dlr_image = dlr_dataset[idx][0].numpy().squeeze()
        dlr_image_brain = apply_window(dlr_image, *brain_window)
        dlr_pred = np.argmax(dlr_predictions[idx])
        dlr_pred_label = label_names[dlr_pred]
        plt.imshow(dlr_image_brain[crop:-crop, crop:-crop], cmap='gray', vmin=brain_window[0], vmax=brain_window[1])
        plt.title(f'DLR prediction: \n{dlr_pred_label}', color='white', fontsize=18)
        plt.axis('off')
        plt.ylabel('[0, 80] HU', color='white')

        plt.subplot(2, 4, 8)
        dlr_image_bone = apply_window(dlr_image, *bone_window)
        plt.imshow(dlr_image_bone[crop:-crop, crop:-crop], cmap='gray', vmin=bone_window[0], vmax=bone_window[1])
        plt.axis('off')
        plt.ylabel('[-1000, 2000] HU', color='white')

        plt.tight_layout()

        # Save the figure
        plt.savefig(f'{cases_dir}/{title_prefix}_case_{idx}.png', dpi=600, facecolor='black')
        plt.close()

# Main code to run the analysis
if __name__ == "__main__":
    # CSV and DICOM directories
    original_csv_file = 'data/metadata_evaluation.csv'
    original_dicom_dir = dicom_dir

    fbp_dicom_dir = 'data/FBP_reconstructions/'
    mbir_dicom_dir = 'data/MBIR_reconstructions/'
    dlr_dicom_dir = 'data/DLR_reconstructions/'

    cases_dir = 'figures/zooms'  # Directory to save the plots

    # Create dataset instances
    print('Loading datasets...')
    original_dataset = RSNA_Intracranial_Hemorrhage_Dataset(original_csv_file, original_dicom_dir, expected_size=512)
    fbp_dataset = RSNA_Intracranial_Hemorrhage_Dataset(original_csv_file, fbp_dicom_dir, expected_size=256)
    mbir_dataset = RSNA_Intracranial_Hemorrhage_Dataset(original_csv_file, mbir_dicom_dir, expected_size=256)
    dlr_dataset = RSNA_Intracranial_Hemorrhage_Dataset(original_csv_file, dlr_dicom_dir, expected_size=256)

    print(f"Original Dataset Size: {len(original_dataset)}")
    print(f"FBP Dataset Size: {len(fbp_dataset)}")
    print(f"MBIR Dataset Size: {len(mbir_dataset)}")
    print(f"DLR Dataset Size: {len(dlr_dataset)}")

    # Initialize the observer
    observer = SupervisedClassifierObserver()
    observer.model = load_classifier(observer.model, 'weights/supervised_classifier_resnet50_weights_09102024.pth')
    print("Model loaded successfully.")

    # Evaluating the datasets
    results = {}
    for key, dataset in zip(['Original', 'FBP', 'MBIR', 'DLR'], 
                            [original_dataset, fbp_dataset, mbir_dataset, dlr_dataset]):
        try:
            print(f'Evaluating {key} dataset...')
            accuracy, ground_truths, predictions = observer.evaluate(DataLoader(dataset, batch_size=1, shuffle=False), num_patients=len(dataset))
            results[key] = {
                'accuracy': accuracy,
                'ground_truths': ground_truths,
                'predictions': predictions
            }
            # print(f"{key} Predictions Shape: {predictions.shape}")  # Check the shape of predictions
            # print(f"{key} Predictions: {predictions}")  # Check the contents of predictions

            if predictions.size == 0:
                print(f"No predictions returned for {key}.")
        except Exception as e:
            print(f"Error evaluating {key}: {e}")

    # Identifying cases where predictions differ
    if all(key in results for key in ['Original', 'FBP', 'MBIR', 'DLR']):
        original_predictions = results['Original']['predictions']
        fbp_predictions = results['FBP']['predictions']
        mbir_predictions = results['MBIR']['predictions']
        dlr_predictions = results['DLR']['predictions']
        ground_truths = results['Original']['ground_truths']

        # Define the specific indices you want to plot
        specific_indices = [520, 684, 725, 812, 931]

        # Call the plot_cases function with the specific indices
        plot_cases(
            indices=specific_indices,
            original_dataset=original_dataset,
            fbp_dataset=fbp_dataset,
            mbir_dataset=mbir_dataset,
            dlr_dataset=dlr_dataset,
            original_predictions=original_predictions,
            fbp_predictions=fbp_predictions,
            mbir_predictions=mbir_predictions,
            dlr_predictions=dlr_predictions,
            true_labels=ground_truths,
            title_prefix='specific_cases',
            cases_dir=cases_dir
        )