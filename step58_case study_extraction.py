import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from step02_dataset_dataloader import RSNA_Intracranial_Hemorrhage_Dataset
from step04_cnn_classifier import SupervisedClassifierObserver, load_classifier
from step00_common_info import dicom_dir

# # Define label names
label_names = [
    "no_hemorrhage", 
    "epidural", 
    "intraparenchymal", 
    "intraventricular", 
    "subarachnoid", 
    "subdural"
]

# Ensure figures/cases directory exists
cases_dir = 'figures/cases/v5'
if not os.path.exists(cases_dir):
    os.makedirs(cases_dir)

def load_results(csv_file, dicom_dir, expected_size):
    dataset = RSNA_Intracranial_Hemorrhage_Dataset(csv_file, dicom_dir, expected_size=expected_size)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    return dataset, data_loader

def identify_cases(ground_truths, original_predictions, fbp_predictions, mbir_predictions, dlr_predictions):
    ground_truths = np.array(ground_truths)

    # Softmax for probabilities and argmax for labels
    pred_labels_original = np.argmax(original_predictions, axis=1)
    pred_labels_fbp = np.argmax(fbp_predictions, axis=1)
    pred_labels_mbir = np.argmax(mbir_predictions, axis=1)
    pred_labels_dlr = np.argmax(dlr_predictions, axis=1)

    true_labels = np.argmax(ground_truths, axis=1)

    # Correct indices for each reconstruction
    correct_original_indices = np.where(pred_labels_original == true_labels)[0]
    correct_fbp_indices = np.where(pred_labels_fbp == true_labels)[0]
    correct_mbir_indices = np.where(pred_labels_mbir == true_labels)[0]
    correct_dlr_indices = np.where(pred_labels_dlr == true_labels)[0]

    # Find cases where only Original, FBP, MBIR, or DLR is correct
    original_correct_not_fbp_mbir_dlr = np.setdiff1d(correct_original_indices, np.union1d(np.union1d(correct_fbp_indices, correct_mbir_indices), correct_dlr_indices))
    fbp_correct_not_original_mbri_dlri = np.setdiff1d(correct_fbp_indices, np.union1d(np.union1d(correct_original_indices, correct_mbir_indices), correct_dlr_indices))
    mbir_correct_not_original_fbpi_dlri = np.setdiff1d(correct_mbir_indices, np.union1d(np.union1d(correct_original_indices, correct_fbp_indices), correct_dlr_indices))
    dlri_correct_not_original_fbpi_mbri = np.setdiff1d(correct_dlr_indices, np.union1d(np.union1d(correct_original_indices, correct_fbp_indices), correct_mbir_indices))
    original_and_dlr_correct_not_fbp_mbir = np.setdiff1d(np.intersect1d(correct_original_indices, correct_dlr_indices), np.union1d(correct_fbp_indices, correct_mbir_indices))
    original_and_fbp_correct_not_mbir_dlr = np.setdiff1d(np.intersect1d(correct_original_indices, correct_fbp_indices), np.union1d(correct_mbir_indices, correct_dlr_indices))
    original_and_mbir_correct_not_fbp_dlr = np.setdiff1d(np.intersect1d(correct_original_indices, correct_mbir_indices), np.union1d(correct_fbp_indices, correct_dlr_indices))

    return original_correct_not_fbp_mbir_dlr, fbp_correct_not_original_mbri_dlri, mbir_correct_not_original_fbpi_dlri, dlri_correct_not_original_fbpi_mbri, original_and_dlr_correct_not_fbp_mbir, original_and_fbp_correct_not_mbir_dlr, original_and_mbir_correct_not_fbp_dlr

def apply_window(image, window_min, window_max):
    return np.clip(image, window_min, window_max)

def plot_cases(indices, original_dataset, fbp_dataset, mbir_dataset, dlr_dataset, 
               original_predictions, fbp_predictions, mbir_predictions, dlr_predictions, true_labels, title_prefix):
    crop = 16  # Define the cropping size
    for idx in indices:
        plt.figure(figsize=(16, 10), facecolor='black')

        # Set the window parameters
        brain_window = (0.0, 80.0)
        bone_window = (-1000, 2000)

        # True label for the current case
        true_label = np.argmax(true_labels[idx])  # true_labels is now extracted dynamically
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

        plt.subplot(2, 4, 7)
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
        plt.savefig(f'{cases_dir}/{title_prefix}_case_{idx}.png', dpi=300, facecolor='black')
        plt.close()



# Main code to run the analysis
if __name__ == "__main__":
    # CSV and DICOM directories
    original_csv_file = 'data/metadata_evaluation.csv'
    original_dicom_dir = dicom_dir

    fbp_dicom_dir = 'data/FBP_reconstructions/'
    mbir_dicom_dir = 'data/MBIR_reconstructions/'
    dlr_dicom_dir = 'data/DLR_reconstructions/'

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

        # Call identify_cases function
        original_correct_not_fbp_mbir_dlr, fbp_correct_not_original_mbri_dlri, mbir_correct_not_original_fbpi_dlri, \
             dlri_correct_not_original_fbpi_mbri, original_and_dlr_correct_not_fbp_mbir, original_and_fbp_correct_not_mbir_dlr, \
                 original_and_mbir_correct_not_fbp_dlr = identify_cases(ground_truths, original_predictions, fbp_predictions, mbir_predictions, dlr_predictions)

        # Plot cases based on the indices returned
        plot_cases(original_correct_not_fbp_mbir_dlr, original_dataset, fbp_dataset, mbir_dataset, dlr_dataset, 
                   original_predictions, fbp_predictions, mbir_predictions, dlr_predictions, ground_truths, "original_correct_not_fbp_mbir_dlr")

        plot_cases(fbp_correct_not_original_mbri_dlri, original_dataset, fbp_dataset, mbir_dataset, dlr_dataset, 
                   original_predictions, fbp_predictions, mbir_predictions, dlr_predictions, ground_truths, "fbp_correct_not_original_mbri_dlri")

        plot_cases(mbir_correct_not_original_fbpi_dlri, original_dataset, fbp_dataset, mbir_dataset, dlr_dataset, 
                   original_predictions, fbp_predictions, mbir_predictions, dlr_predictions, ground_truths, "mbir_correct_not_original_fbpi_dlri")

        plot_cases(dlri_correct_not_original_fbpi_mbri, original_dataset, fbp_dataset, mbir_dataset, dlr_dataset, 
                   original_predictions, fbp_predictions, mbir_predictions, dlr_predictions, ground_truths, "dlri_correct_not_original_fbpi_mbri")
        
        plot_cases(original_and_dlr_correct_not_fbp_mbir, original_dataset, fbp_dataset, mbir_dataset, dlr_dataset, 
                   original_predictions, fbp_predictions, mbir_predictions, dlr_predictions, ground_truths, "original_and_dlr_correct_not_fbp_mbir")
        
        plot_cases(original_and_fbp_correct_not_mbir_dlr, original_dataset, fbp_dataset, mbir_dataset, dlr_dataset, 
                   original_predictions, fbp_predictions, mbir_predictions, dlr_predictions, ground_truths, "original_and_fbp_correct_not_mbir_dlr")
        
        plot_cases(original_and_mbir_correct_not_fbp_dlr, original_dataset, fbp_dataset, mbir_dataset, dlr_dataset, 
                   original_predictions, fbp_predictions, mbir_predictions, dlr_predictions, ground_truths, "original_and_mbir_correct_not_fbp_dlr")
    else:
        print("Not all datasets have results. Cannot proceed with case identification.")


# import numpy as np
# import matplotlib.pyplot as plt
# import os
# from torch.utils.data import DataLoader
# from step02_dataset_dataloader import RSNA_Intracranial_Hemorrhage_Dataset
# from step04_cnn_classifier import SupervisedClassifierObserver, load_classifier
# from step00_common_info import dicom_dir

# # Ensure figures/cases directory exists
# cases_dir = 'figures/cases/v3'
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
#     # pred_probs_original = np.exp(original_predictions) / np.sum(np.exp(original_predictions), axis=1, keepdims=True)
#     pred_labels_original = np.argmax(original_predictions, axis=1)

#     # pred_probs_fbp = np.exp(fbp_predictions) / np.sum(np.exp(fbp_predictions), axis=1, keepdims=True)
#     pred_labels_fbp = np.argmax(fbp_predictions, axis=1)

#     # pred_probs_mbir = np.exp(mbir_predictions) / np.sum(np.exp(mbir_predictions), axis=1, keepdims=True)
#     pred_labels_mbir = np.argmax(mbir_predictions, axis=1)

#     # pred_probs_dlr = np.exp(dlr_predictions) / np.sum(np.exp(dlr_predictions), axis=1, keepdims=True)
#     pred_labels_dlr = np.argmax(dlr_predictions, axis=1)

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

# def plot_cases(indices, original_dataset, fbp_dataset, mbir_dataset, dlr_dataset, 
#                original_predictions, fbp_predictions, mbir_predictions, dlr_predictions, true_labels, title_prefix):
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

#         # Plot Original image
#         plt.subplot(2, 4, 1)
#         original_image = original_dataset[idx][0].numpy().squeeze()
#         original_image_brain = apply_window(original_image, *brain_window)
#         plt.imshow(original_image_brain[crop:-crop, crop:-crop], cmap='gray', vmin=brain_window[0], vmax=brain_window[1])
#         original_pred = np.argmax(original_predictions[idx])
#         plt.title(f'Original prediction: \n{label_names[original_pred]}', color='white', fontsize=18)
#         plt.axis('off')
#         plt.ylabel('[0, 80] HU', color='white')

#         plt.subplot(2, 4, 5)
#         original_image_bone = apply_window(original_image, *bone_window)
#         plt.imshow(original_image_bone[crop:-crop, crop:-crop], cmap='gray', vmin=bone_window[0], vmax=bone_window[1])
#         plt.axis('off')
#         plt.ylabel('[-1000, 2000] HU', color='white')

#         # Plot FBP image
#         plt.subplot(2, 4, 2)
#         fbp_image = fbp_dataset[idx][0].numpy().squeeze()
#         fbp_image_brain = apply_window(fbp_image, *brain_window)
#         fbp_pred = np.argmax(fbp_predictions[idx])
#         plt.imshow(fbp_image_brain[crop:-crop, crop:-crop], cmap='gray', vmin=brain_window[0], vmax=brain_window[1])
#         plt.title(f'FBP prediction: \n{label_names[fbp_pred]}', color='white', fontsize=18)
#         plt.axis('off')
#         plt.ylabel('[0, 80] HU', color='white')

#         plt.subplot(2, 4, 6)
#         fbp_image_bone = apply_window(fbp_image, *bone_window)
#         plt.imshow(fbp_image_bone[crop:-crop, crop:-crop], cmap='gray', vmin=bone_window[0], vmax=bone_window[1])
#         plt.axis('off')
#         plt.ylabel('[-1000, 2000] HU', color='white')

#         # Plot MBIR image
#         plt.subplot(2, 4, 3)
#         mbir_image = mbir_dataset[idx][0].numpy().squeeze()
#         mbir_image_brain = apply_window(mbir_image, *brain_window)
#         mbir_pred = np.argmax(mbir_predictions[idx])
#         plt.imshow(mbir_image_brain[crop:-crop, crop:-crop], cmap='gray', vmin=brain_window[0], vmax=brain_window[1])
#         plt.title(f'MBIR prediction: \n{label_names[mbir_pred]}', color='white', fontsize=18)
#         plt.axis('off')
#         plt.ylabel('[0, 80] HU', color='white')

#         plt.subplot(2, 4, 7)
#         mbir_image_bone = apply_window(mbir_image, *bone_window)
#         plt.imshow(mbir_image_bone[crop:-crop, crop:-crop], cmap='gray', vmin=bone_window[0], vmax=bone_window[1])
#         plt.axis('off')
#         plt.ylabel('[-1000, 2000] HU', color='white')

#         # Plot DLR image
#         plt.subplot(2, 4, 4)
#         dlr_image = dlr_dataset[idx][0].numpy().squeeze()
#         dlr_image_brain = apply_window(dlr_image, *brain_window)
#         dlr_pred = np.argmax(dlr_predictions[idx])
#         plt.imshow(dlr_image_brain[crop:-crop, crop:-crop], cmap='gray', vmin=brain_window[0], vmax=brain_window[1])
#         plt.title(f'DLR prediction: \n{label_names[dlr_pred]}', color='white', fontsize=18)
#         plt.axis('off')
#         plt.ylabel('[0, 80] HU', color='white')

#         plt.subplot(2, 4, 8)
#         dlr_image_bone = apply_window(dlr_image, *bone_window)
#         plt.imshow(dlr_image_bone[crop:-crop, crop:-crop], cmap='gray', vmin=bone_window[0], vmax=bone_window[1])
#         plt.axis('off')
#         plt.ylabel('[-1000, 2000] HU', color='white')

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

#     # Use FBP ground_truths for case identification as all datasets share the same ground truth
#     original_correct_not_fbp_mbir_dlr, fbp_correct_not_original_mbir_dlr, mbir_correct_not_original_fbp_dlr, dlr_correct_not_original_fbp_mbir = identify_cases(
#         results['Original']['ground_truths'],
#         results['Original']['predictions'],
#         results['FBP']['predictions'],
#         results['MBIR']['predictions'],
#         results['DLR']['predictions']
#     )

#     # Plot cases
#     print('Plotting cases...')
#         # Original is correct, others are not
#     plot_cases(original_correct_not_fbp_mbir_dlr, datasets['Original'], datasets['FBP'], datasets['MBIR'], datasets['DLR'],
#                results['Original']['predictions'], results['FBP']['predictions'], results['MBIR']['predictions'], results['DLR']['predictions'], 
#                ground_truths, 'Original_Correct_Not_FBP_MBIR_DLR')

#     # FBP is correct, others are not
#     plot_cases(fbp_correct_not_original_mbir_dlr, datasets['Original'], datasets['FBP'], datasets['MBIR'], datasets['DLR'], 
#                results['Original']['predictions'], results['FBP']['predictions'], results['MBIR']['predictions'], results['DLR']['predictions'], 
#                ground_truths, 'FBP_Correct_Not_Original_MBIR_DLR')

#     # MBIR is correct, others are not
#     plot_cases(mbir_correct_not_original_fbp_dlr, datasets['Original'], datasets['FBP'], datasets['MBIR'], datasets['DLR'], 
#                results['Original']['predictions'], results['FBP']['predictions'], results['MBIR']['predictions'], results['DLR']['predictions'], 
#                ground_truths, 'MBIR_Correct_Not_Original_FBP_DLR')

#     # DLR is correct, others are not
#     plot_cases(dlr_correct_not_original_fbp_mbir, datasets['Original'], datasets['FBP'], datasets['MBIR'], datasets['DLR'], 
#                results['Original']['predictions'], results['FBP']['predictions'], results['MBIR']['predictions'], results['DLR']['predictions'], 
#                ground_truths, 'DLR_Correct_Not_Original_FBP_MBIR')

#     print("Analysis complete. Plots saved to 'figures/cases' directory.")


# # Ensure figures/cases directory exists
# cases_dir = 'figures/cases/v2'
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

# def identify_cases(ground_truths, fbp_predictions, mbir_predictions, dlr_predictions):
#     ground_truths = np.array(ground_truths)

#     # FBP
#     pred_probs_fbp = np.exp(fbp_predictions) / np.sum(np.exp(fbp_predictions), axis=1, keepdims=True)
#     pred_labels_fbp = np.argmax(pred_probs_fbp, axis=1)

#     # MBIR
#     pred_probs_mbir = np.exp(mbir_predictions) / np.sum(np.exp(mbir_predictions), axis=1, keepdims=True)
#     pred_labels_mbir = np.argmax(pred_probs_mbir, axis=1)

#     # DLR
#     pred_probs_dlr = np.exp(dlr_predictions) / np.sum(np.exp(dlr_predictions), axis=1, keepdims=True)
#     pred_labels_dlr = np.argmax(pred_probs_dlr, axis=1)

#     true_labels = np.argmax(ground_truths, axis=1)

#     # Correct indices for each reconstruction
#     correct_fbp_indices = np.where(pred_labels_fbp == true_labels)[0]
#     correct_mbri_indices = np.where(pred_labels_mbir == true_labels)[0]
#     correct_dlri_indices = np.where(pred_labels_dlr == true_labels)[0]

#     # Find cases where only FBP, MBIR, or DLR is correct
#     fbp_correct_not_mbri_dlri = np.setdiff1d(correct_fbp_indices, np.union1d(correct_mbri_indices, correct_dlri_indices))
#     mbir_correct_not_fbpi_dlri = np.setdiff1d(correct_mbri_indices, np.union1d(correct_fbp_indices, correct_dlri_indices))
#     dlri_correct_not_fbpi_mbri = np.setdiff1d(correct_dlri_indices, np.union1d(correct_fbp_indices, correct_mbri_indices))

#     return fbp_correct_not_mbri_dlri, mbir_correct_not_fbpi_dlri, dlri_correct_not_fbpi_mbri

# def apply_window(image, window_min, window_max):
#     """
#     Apply a window to an image.
#     """
#     return np.clip(image, window_min, window_max)

# def plot_cases(indices, original_dataset, fbp_dataset, mbir_dataset, dlr_dataset, 
#                fbp_predictions, mbir_predictions, dlr_predictions, true_labels, title_prefix):
#     for idx in indices:
#         plt.figure(figsize=(16, 10), facecolor='black')  # Set background color to black

#         # Set the window parameters
#         brain_window = (0.0, 80.0)
#         bone_window = (-1000, 2000)

#         # Original image with true label (Brain)
#         plt.subplot(2, 4, 1)
#         original_image = original_dataset[idx][0].numpy().squeeze()
#         original_image_brain = apply_window(original_image, *brain_window)
#         plt.imshow(original_image_brain, cmap='gray', vmin=brain_window[0], vmax=brain_window[1])
#         true_label = np.argmax(true_labels[idx])
#         plt.title(f'True: {label_names[true_label]}', color='white')
#         plt.axis('off')
#         plt.ylabel('[0, 80] HU', color='white')

#         # Original image with true label (Bone)
#         plt.subplot(2, 4, 5)
#         original_image_bone = apply_window(original_image, *bone_window)
#         plt.imshow(original_image_bone, cmap='gray', vmin=bone_window[0], vmax=bone_window[1])
#         # plt.title('Original Bone', color='white')
#         plt.axis('off')
#         plt.ylabel('[-1000, 2000] HU', color='white')

#         # FBP reconstruction (Brain)
#         plt.subplot(2, 4, 2)
#         fbp_image = fbp_dataset[idx][0].numpy().squeeze()
#         fbp_image_brain = apply_window(fbp_image, *brain_window)
#         fbp_pred = np.argmax(fbp_predictions[idx])
#         plt.imshow(fbp_image_brain, cmap='gray', vmin=brain_window[0], vmax=brain_window[1])
#         plt.title(f'FBP: {label_names[fbp_pred]}', color='white')
#         plt.axis('off')
#         plt.ylabel('[0, 80] HU', color='white')

#         # FBP reconstruction (Bone)
#         plt.subplot(2, 4, 6)
#         fbp_image_bone = apply_window(fbp_image, *bone_window)
#         plt.imshow(fbp_image_bone, cmap='gray', vmin=bone_window[0], vmax=bone_window[1])
#         # plt.title('FBP Bone', color='white')
#         plt.axis('off')
#         plt.ylabel('[-1000, 2000] HU', color='white')

#         # MBIR reconstruction (Brain)
#         plt.subplot(2, 4, 3)
#         mbir_image = mbir_dataset[idx][0].numpy().squeeze()
#         mbir_image_brain = apply_window(mbir_image, *brain_window)
#         mbir_pred = np.argmax(mbir_predictions[idx])
#         plt.imshow(mbir_image_brain, cmap='gray', vmin=brain_window[0], vmax=brain_window[1])
#         plt.title(f'MBIR: {label_names[mbir_pred]}', color='white')
#         plt.axis('off')
#         plt.ylabel('[0, 80] HU', color='white')

#         # MBIR reconstruction (Bone)
#         plt.subplot(2, 4, 7)
#         mbir_image_bone = apply_window(mbir_image, *bone_window)
#         plt.imshow(mbir_image_bone, cmap='gray', vmin=bone_window[0], vmax=bone_window[1])
#         # plt.title('MBIR Bone', color='white')
#         plt.axis('off')
#         plt.ylabel('[-1000, 2000] HU', color='white')

#         # DLR reconstruction (Brain)
#         plt.subplot(2, 4, 4)
#         dlr_image = dlr_dataset[idx][0].numpy().squeeze()
#         dlr_image_brain = apply_window(dlr_image, *brain_window)
#         dlr_pred = np.argmax(dlr_predictions[idx])
#         plt.imshow(dlr_image_brain, cmap='gray', vmin=brain_window[0], vmax=brain_window[1])
#         plt.title(f'DLR: {label_names[dlr_pred]}', color='white')
#         plt.axis('off')
#         plt.ylabel('[0, 80] HU', color='white')

#         # DLR reconstruction (Bone)
#         plt.subplot(2, 4, 8)
#         dlr_image_bone = apply_window(dlr_image, *bone_window)
#         plt.imshow(dlr_image_bone, cmap='gray', vmin=bone_window[0], vmax=bone_window[1])
#         # plt.title('DLR Bone', color='white')
#         plt.axis('off')
#         plt.ylabel('[-1000, 2000] HU', color='white')

#         plt.suptitle(f'{title_prefix} Case {idx}', color='white')
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
#     observer.model = load_classifier(observer.model, 'weights/supervised_classifier_resnet50_weights.pth')

#     # Load predictions
#     print('Evaluating datasets...')
#     results = {}
#     for key in ['FBP', 'MBIR', 'DLR']:
#         accuracy, ground_truths, predictions = observer.evaluate(DataLoader(datasets[key], batch_size=1, shuffle=False), num_patients=len(datasets[key]))
#         results[key] = {
#             'accuracy': accuracy,
#             'ground_truths': ground_truths,
#             'predictions': predictions
#         }

#     # Use FBP ground_truths for case identification as all datasets share the same ground truth
#     fbp_correct_not_mbri_dlri, mbir_correct_not_fbpi_dlri, dlri_correct_not_fbpi_mbri = identify_cases(
#         results['FBP']['ground_truths'],
#         results['FBP']['predictions'],
#         results['MBIR']['predictions'],
#         results['DLR']['predictions']
#     )

#     # Plot cases
#     print('Plotting cases...')
#     plot_cases(fbp_correct_not_mbri_dlri, datasets['Original'], datasets['FBP'], datasets['MBIR'], datasets['DLR'], 
#                results['FBP']['predictions'], results['MBIR']['predictions'], results['DLR']['predictions'], 
#                results['FBP']['ground_truths'], 'FBP_Correct_Not_MBIR_DLR')

#     plot_cases(mbir_correct_not_fbpi_dlri, datasets['Original'], datasets['FBP'], datasets['MBIR'], datasets['DLR'], 
#                results['FBP']['predictions'], results['MBIR']['predictions'], results['DLR']['predictions'], 
#                results['FBP']['ground_truths'], 'MBIR_Correct_Not_FBP_DLR')

#     plot_cases(dlri_correct_not_fbpi_mbri, datasets['Original'], datasets['FBP'], datasets['MBIR'], datasets['DLR'], 
#                results['FBP']['predictions'], results['MBIR']['predictions'], results['DLR']['predictions'], 
#                results['FBP']['ground_truths'], 'DLR_Correct_Not_FBP_MBIR')

#     print("Analysis complete. Plots saved to 'figures/cases' directory.")