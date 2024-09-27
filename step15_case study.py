import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from step2_dataset_dataloader import RSNA_Intracranial_Hemorrhage_Dataset
from step3_cnn_classifier import SupervisedClassifierObserver, load_classifier
from step0_common_info import dicom_dir

# Ensure figures/cases directory exists
cases_dir = 'figures/cases'
if not os.path.exists(cases_dir):
    os.makedirs(cases_dir)

# Define label names
label_names = [
    "no_hemorrhage", 
    "epidural", 
    "intraparenchymal", 
    "intraventricular", 
    "subarachnoid", 
    "subdural"
]

def load_results(csv_file, dicom_dir, expected_size):
    dataset = RSNA_Intracranial_Hemorrhage_Dataset(csv_file, dicom_dir, expected_size=expected_size)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    return dataset, data_loader

def identify_cases(ground_truths, fbp_predictions, mbir_predictions, dlr_predictions):
    ground_truths = np.array(ground_truths)

    # FBP
    pred_probs_fbp = np.exp(fbp_predictions) / np.sum(np.exp(fbp_predictions), axis=1, keepdims=True)
    pred_labels_fbp = np.argmax(pred_probs_fbp, axis=1)

    # MBIR
    pred_probs_mbir = np.exp(mbir_predictions) / np.sum(np.exp(mbir_predictions), axis=1, keepdims=True)
    pred_labels_mbir = np.argmax(pred_probs_mbir, axis=1)

    # DLR
    pred_probs_dlr = np.exp(dlr_predictions) / np.sum(np.exp(dlr_predictions), axis=1, keepdims=True)
    pred_labels_dlr = np.argmax(pred_probs_dlr, axis=1)

    true_labels = np.argmax(ground_truths, axis=1)

    # Correct indices for each reconstruction
    correct_fbp_indices = np.where(pred_labels_fbp == true_labels)[0]
    correct_mbri_indices = np.where(pred_labels_mbir == true_labels)[0]
    correct_dlri_indices = np.where(pred_labels_dlr == true_labels)[0]

    # Find cases where only FBP, MBIR, or DLR is correct
    fbp_correct_not_mbri_dlri = np.setdiff1d(correct_fbp_indices, np.union1d(correct_mbri_indices, correct_dlri_indices))
    mbir_correct_not_fbpi_dlri = np.setdiff1d(correct_mbri_indices, np.union1d(correct_fbp_indices, correct_dlri_indices))
    dlri_correct_not_fbpi_mbri = np.setdiff1d(correct_dlri_indices, np.union1d(correct_fbp_indices, correct_mbri_indices))

    return fbp_correct_not_mbri_dlri, mbir_correct_not_fbpi_dlri, dlri_correct_not_fbpi_mbri

def plot_cases(indices, original_dataset, fbp_dataset, mbir_dataset, dlr_dataset, fbp_predictions, mbir_predictions, dlr_predictions, true_labels, title_prefix):
    for idx in indices:
        plt.figure(figsize=(16, 10))

        # Original image with true label
        original_image = original_dataset[idx][0].numpy().squeeze()  # Load original image
        true_label = np.argmax(true_labels[idx])
        plt.subplot(1, 5, 1)
        plt.imshow(original_image, cmap='gray')
        plt.title(f'True: {label_names[true_label]}')
        plt.axis('off')

        # FBP reconstruction
        plt.subplot(1, 5, 2)
        fbp_image = fbp_dataset[idx][0].numpy().squeeze()  # Load FBP image
        fbp_pred = np.argmax(fbp_predictions[idx])
        plt.imshow(fbp_image, cmap='gray')
        plt.title(f'FBP: {label_names[fbp_pred]}')
        plt.axis('off')

        # MBIR reconstruction
        plt.subplot(1, 5, 3)
        mbir_image = mbir_dataset[idx][0].numpy().squeeze()  # Load MBIR image
        mbir_pred = np.argmax(mbir_predictions[idx])
        plt.imshow(mbir_image, cmap='gray')
        plt.title(f'MBIR: {label_names[mbir_pred]}')
        plt.axis('off')

        # DLR reconstruction
        plt.subplot(1, 5, 4)
        dlr_image = dlr_dataset[idx][0].numpy().squeeze()  # Load DLR image
        dlr_pred = np.argmax(dlr_predictions[idx])
        plt.imshow(dlr_image, cmap='gray')
        plt.title(f'DLR: {label_names[dlr_pred]}')
        plt.axis('off')

        plt.suptitle(f'{title_prefix} Case {idx}')
        plt.tight_layout()

        # Save the figure
        plt.savefig(f'{cases_dir}/{title_prefix}_case_{idx}.png', dpi=600)
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

    datasets = {
        'FBP': fbp_dataset,
        'MBIR': mbir_dataset,
        'DLR': dlr_dataset,
        'Original': original_dataset
    }

    # Initialize observer and load classifier
    observer = SupervisedClassifierObserver(verbose=True, batch_size=1)
    observer.model = load_classifier(observer.model, 'weights/supervised_classifier_resnet50_weights.pth')

    # Load predictions
    print('Evaluating datasets...')
    results = {}
    for key in ['FBP', 'MBIR', 'DLR']:
        accuracy, ground_truths, predictions = observer.evaluate(DataLoader(datasets[key], batch_size=1, shuffle=False), num_patients=len(datasets[key]))
        results[key] = {
            'accuracy': accuracy,
            'ground_truths': ground_truths,
            'predictions': predictions
        }

    # Use FBP ground_truths for case identification as all datasets share the same ground truth
    fbp_correct_not_mbri_dlri, mbir_correct_not_fbpi_dlri, dlri_correct_not_fbpi_mbri = identify_cases(
        results['FBP']['ground_truths'],
        results['FBP']['predictions'],
        results['MBIR']['predictions'],
        results['DLR']['predictions']
    )

    # Plot cases
    print('Plotting cases...')
    plot_cases(fbp_correct_not_mbri_dlri, datasets['Original'], datasets['FBP'], datasets['MBIR'], datasets['DLR'], 
               results['FBP']['predictions'], results['MBIR']['predictions'], results['DLR']['predictions'], 
               results['FBP']['ground_truths'], 'FBP_Correct_Not_MBIR_DLR')

    plot_cases(mbir_correct_not_fbpi_dlri, datasets['Original'], datasets['FBP'], datasets['MBIR'], datasets['DLR'], 
               results['FBP']['predictions'], results['MBIR']['predictions'], results['DLR']['predictions'], 
               results['FBP']['ground_truths'], 'MBIR_Correct_Not_FBP_DLR')

    plot_cases(dlri_correct_not_fbpi_mbri, datasets['Original'], datasets['FBP'], datasets['MBIR'], datasets['DLR'], 
               results['FBP']['predictions'], results['MBIR']['predictions'], results['DLR']['predictions'], 
               results['FBP']['ground_truths'], 'DLR_Correct_Not_FBP_MBIR')

    print("Analysis complete. Plots saved to 'figures/cases' directory.")
