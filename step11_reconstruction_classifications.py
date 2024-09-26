import torch
from torch.utils.data import DataLoader
import pandas as pd
from step2_dataset_dataloader import RSNA_Intracranial_Hemorrhage_Dataset
from step3_cnn_classifier import SupervisedClassifierObserver, load_classifier
from step0_common_info import dataset_dir, dicom_dir
from sklearn.metrics import roc_curve, auc
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def compute_ovr_auc(ground_truths, predictions, labels, dataset_name):
    ground_truths_bin = label_binarize(np.argmax(ground_truths, axis=1), classes=np.arange(6))
    predictions_prob = torch.softmax(torch.tensor(predictions), dim=1).numpy()

    ovr_auc_results = {}
    plt.figure(figsize=(6,6))
    for i, label in enumerate(labels):
        fpr, tpr, _ = roc_curve(ground_truths_bin[:, i], predictions_prob[:, i])
        auc_ovr = auc(fpr, tpr)
        ovr_auc_results[label] = auc_ovr
        print(f"OvR AUC for {label}: {auc_ovr:.4f}")

        # Plotting the ROC curve
        plt.plot(fpr, tpr, label=f'{label} (AUC = {auc_ovr:.4f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'One-vs-Rest ROC Curve for {dataset_name}')
    plt.legend(loc='lower right')
    plt.savefig(f'figures/ovr_auc_roc_{dataset_name}.png', dpi=300)
    plt.close()

    # Save the results to a csv file
    df = pd.DataFrame(ovr_auc_results.items(), columns=['Hemorrhage Type', 'AUC'])
    df.to_csv(f'results/ovr_auc_{dataset_name}.csv', index=False)
    print(f"One-vs-Rest AUC results saved to 'results/ovr_auc_{dataset_name}.csv'")

def compute_ovo_auc(ground_truths, predictions, labels, dataset_name):
    ground_truths_bin = np.argmax(ground_truths, axis=1)
    predictions_prob = torch.softmax(torch.tensor(predictions), dim=1).numpy()

    ovo_auc_results = {}
    class_pairs = list(combinations(range(6), 2))

    plt.figure(figsize=(6,6))
    for pair in class_pairs:
        pair_idx = np.where(np.isin(ground_truths_bin, pair))[0]
        if len(pair_idx) == 0:
            continue
        binary_gt = (ground_truths_bin[pair_idx] == pair[1]).astype(int)
        binary_pred = predictions_prob[pair_idx][:, pair[1]]

        fpr, tpr, _ = roc_curve(binary_gt, binary_pred)
        auc_ovo = auc(fpr, tpr)
        ovo_auc_results[f"{labels[pair[0]]}_vs_{labels[pair[1]]}"] = auc_ovo
        print(f"OvO AUC for {labels[pair[0]]} vs {labels[pair[1]]}: {auc_ovo:.4f}")

        plt.plot(fpr, tpr, label=f'{labels[pair[0]]} vs {labels[pair[1]]} (AUC = {auc_ovo:.4f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'One-vs-One ROC Curve for {dataset_name}')
    plt.legend(loc='best', fontsize=5.8)
    plt.savefig(f'figures/ovo_auc_roc_{dataset_name}.png', dpi=300)
    plt.close()

    # Save the results to a csv file
    df = pd.DataFrame(ovo_auc_results.items(), columns=['Hemorrhage Type Pair', 'AUC'])
    df.to_csv(f'results/ovo_auc_{dataset_name}.csv', index=False)
    print(f"One-vs-One AUC results saved to 'results/ovo_auc_{dataset_name}.csv'")

def plot_confusion_matrix(ground_truths, predictions, labels, dataset_name):
    ground_truths_bin = np.argmax(ground_truths, axis=1)
    predicted_labels = np.argmax(predictions, axis=1)
    
    cm = confusion_matrix(ground_truths_bin, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    plt.xticks(rotation=45, ha='right')  # Rotate the x-axis labels by 45 degrees
    # plt.yticks(rotation=0)
    
    plt.figure(figsize=(10, 10))
    disp.plot(cmap=plt.cm.OrRd, values_format='d')
    plt.title(f'Confusion Matrix for {dataset_name}')
    plt.savefig(f'figures/confusion_matrix_{dataset_name}.png', dpi=300)
    plt.close()
    print(f"Confusion matrix for {dataset_name} saved to 'figures/confusion_matrix_{dataset_name}.png'")

def evaluate_reconstructed_datasets(observer, dicom_dirs, csv_file, batch_size=128):
    results = {}
    for dataset_name, dicom_dir in dicom_dirs.items():
        print(f"Loading dataset: {dataset_name} from {dicom_dir}")
        dataset = RSNA_Intracranial_Hemorrhage_Dataset(csv_file, dicom_dir, expected_size=256)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Evaluating {dataset_name} dataset...")
        accuracy, ground_truths, predictions = observer.evaluate(data_loader, num_patients=len(dataset))
        results[dataset_name] = {
            'accuracy': accuracy,
            'ground_truths': ground_truths,
            'predictions': predictions
        }
        print(f"Accuracy for {dataset_name}: {accuracy * 100:.2f}%")
        
        # Compute and save OvR AUC
        compute_ovr_auc(ground_truths, predictions, observer.labels, dataset_name)
        
        # Compute and save OvO AUC
        compute_ovo_auc(ground_truths, predictions, observer.labels, dataset_name)

        # Compute and plot confusion matrix
        plot_confusion_matrix(ground_truths, predictions, observer.labels, dataset_name)
    
    return results

if __name__ == "__main__":
    csv_file = 'data/metadata_evaluation.csv'  # CSV file for labels
    batch_size = 128
    device_ids = [0, 1]
    multiGPU_flag = True

    dicom_dirs = {
        'FBP_reconstructions': 'data/FBP_reconstructions',
        'MBIR_reconstructions': 'data/MBIR_reconstructions',
        'DLR_reconstructions': 'data/DLR_reconstructions'
    }

    observer = SupervisedClassifierObserver(verbose=True, batch_size=batch_size)

    if multiGPU_flag:
        observer.model = torch.nn.DataParallel(observer.model, device_ids=device_ids)

    try:
        observer.model = load_classifier(observer.model, 'weights/supervised_classifier_resnet50_weights.pth')
        print("Model weights loaded successfully.")
    except FileNotFoundError:
        print("Weights file not found. Please ensure the model is trained and weights are saved.")

    results = evaluate_reconstructed_datasets(observer, dicom_dirs, csv_file, batch_size=batch_size)

