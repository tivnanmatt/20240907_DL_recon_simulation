import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from step02_dataset_dataloader import RSNA_Intracranial_Hemorrhage_Dataset
from step04_cnn_classifier import SupervisedClassifierObserver, load_classifier
from step00_common_info import dicom_dir

def generate_error_heatmap(original, predicted_label, true_label):
    # Create an error heatmap based on the predicted and true labels
    error_map = np.zeros(original.shape, dtype=np.float32)

    true_class = np.argmax(true_label)
    
    # Check if the predicted label does not match the true label
    if predicted_label != true_class:
        error_map[original > 0] = 1  # Assuming original is a binary mask or has relevant features
    
    return error_map

def visualize_results(original_images, fbp_images, mbir_images, dlr_images, heatmaps, index, save_path=None):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Display original images
    axes[0, 0].imshow(np.squeeze(original_images[index]), cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 1].imshow(np.squeeze(fbp_images[index]), cmap='gray')
    axes[0, 1].set_title('FBP')
    axes[0, 2].imshow(np.squeeze(mbir_images[index]), cmap='gray')
    axes[0, 2].set_title('MBIR')
    axes[0, 3].imshow(np.squeeze(dlr_images[index]), cmap='gray')
    axes[0, 3].set_title('DLR')
    
    # Display heatmaps (limit to 4)
    for i in range(min(4, len(heatmaps))):  # Limit to 4 heatmaps
        axes[1, i].imshow(np.squeeze(heatmaps[i]), cmap='hot', alpha=0.5)  # Overlay heatmap
        axes[1, i].set_title(f'Error Map - {["Original", "FBP", "MBIR", "DLR"][i]}')
        
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()
if __name__ == "__main__":
    # Load datasets
    original_csv_file = 'data/metadata_evaluation.csv'
    original_dicom_dir = dicom_dir
    
    original_dataset = RSNA_Intracranial_Hemorrhage_Dataset(original_csv_file, original_dicom_dir, expected_size=512)
    fbp_dataset = RSNA_Intracranial_Hemorrhage_Dataset(original_csv_file, 'data/FBP_reconstructions/', expected_size=256)
    mbir_dataset = RSNA_Intracranial_Hemorrhage_Dataset(original_csv_file, 'data/MBIR_reconstructions/', expected_size=256)
    dlr_dataset = RSNA_Intracranial_Hemorrhage_Dataset(original_csv_file, 'data/DLR_reconstructions/', expected_size=256)

    # Load the classifier
    observer = SupervisedClassifierObserver(verbose=True, batch_size=1)
    observer.model = load_classifier(observer.model, 'weights/supervised_classifier_resnet50_weights_09102024.pth')
    observer.model.eval()

    datasets = {
        'Original': original_dataset,
        'FBP': fbp_dataset,
        'MBIR': mbir_dataset,
        'DLR': dlr_dataset
    }

    indices_to_analyze = [14, 15, 16]  # List of indices to analyze
for index_to_analyze in indices_to_analyze:
    heatmaps = []
    original_images = []
    fbp_images = []
    mbir_images = []
    dlr_images = []
    
    for key, dataset in datasets.items():
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        for images, labels in dataloader:
            images = images.to(observer.device)
            predictions = observer.model(images)
            predicted_labels = torch.argmax(predictions, dim=1).cpu().numpy()
            true_labels = labels.cpu().numpy()
            
            # Store the original images for visualization
            if key == 'Original':
                original_images.append(images.cpu().numpy()[0])
            elif key == 'FBP':
                fbp_images.append(images.cpu().numpy()[0])
            elif key == 'MBIR':
                mbir_images.append(images.cpu().numpy()[0])
            elif key == 'DLR':
                dlr_images.append(images.cpu().numpy()[0])
            
            # Generate error heatmap
            heatmap = generate_error_heatmap(images.cpu().numpy()[0], predicted_labels[0], true_labels[0])
            heatmaps.append(heatmap)

    # Visualize results for the specified index and save the plot
    visualize_results(original_images, fbp_images, mbir_images, dlr_images, heatmaps, index_to_analyze, save_path=f'figures/error_propagation/error_heatmaps_{index_to_analyze}.png')