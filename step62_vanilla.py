import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from step02_dataset_dataloader import RSNA_Intracranial_Hemorrhage_Dataset
from step04_cnn_classifier import SupervisedClassifierObserver, load_classifier
from step00_common_info import dicom_dir

def apply_window(image, window_min, window_max):
    return np.clip(image, window_min, window_max)

class VanillaGradients:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.device = next(self.model.parameters()).device

    def generate_explanation(self, input_image, target_class):
        input_image.requires_grad_()
        self.model.zero_grad()
        
        output = self.model(input_image)
        output[0, target_class].backward()

        gradients = input_image.grad.data.abs()
        return gradients.squeeze().cpu().numpy()

    def explain(self, input_image, target_class):
        explanation = self.generate_explanation(input_image, target_class)
        return explanation

def plot_vanilla_gradients(case_indices, datasets, observer, target_class, save_dir='figures/saliency/vanilla'):
    os.makedirs(save_dir, exist_ok=True)
    observer.model.eval()

    # Loop through each case index
    for case_index in case_indices:
        # Create a figure with 4 columns (for each dataset) and 2 rows (for original and gradients)
        fig, axs = plt.subplots(2, 4, figsize=(20, 10))

        for j, dataset in enumerate(datasets):
            # Create a data loader for the dataset
            data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
            # Get the specific case index image
            input_image, _ = dataset[case_index]  # Fetch the image at the current case index
            input_image = apply_window(input_image.unsqueeze(0), 0, 80).to(observer.device)  # Add batch dimension

            # Plot original image
            axs[0, j].imshow(input_image.squeeze().detach().cpu().numpy(), cmap='gray')
            axs[0, j].set_title(f'Original Image - {dataset.__class__.__name__}')
            axs[0, j].axis('off')

            # Compute and plot Vanilla Gradients
            explainer = VanillaGradients(observer.model)
            explanation = explainer.explain(input_image, target_class)
            axs[1, j].imshow(explanation, cmap='hot')
            axs[1, j].set_title('Vanilla Gradients')
            axs[1, j].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'vanilla_gradients_case_{case_index}.png'))  # Save each case separately
        plt.close()

if __name__ == "__main__":
    # CSV and DICOM directories
    original_csv_file = 'data/metadata_evaluation.csv'
    original_dicom_dir = dicom_dir

    # Create dataset instances
    print('Loading datasets...')
    original_dataset = RSNA_Intracranial_Hemorrhage_Dataset(original_csv_file, original_dicom_dir, expected_size=512)
    fbp_dataset = RSNA_Intracranial_Hemorrhage_Dataset(original_csv_file, 'data/FBP_reconstructions/', expected_size=256)
    mbir_dataset = RSNA_Intracranial_Hemorrhage_Dataset(original_csv_file, 'data/MBIR_reconstructions/', expected_size=256)
    dlr_dataset = RSNA_Intracranial_Hemorrhage_Dataset(original_csv_file, 'data/DLR_reconstructions/', expected_size=256)

    observer = SupervisedClassifierObserver(verbose=True, batch_size=1)
    observer.model = load_classifier(observer.model, 'weights/supervised_classifier_resnet50_weights_09102024.pth')
    observer.model = observer.model.to(observer.device)
    observer.model.eval()

    case_indices = list(range(1, 5)) + list(range(501, 505)) + list(range(601, 605)) + list(range(701, 705)) + list(range(801, 805)) + list(range(901, 905))
    datasets = [original_dataset, fbp_dataset, mbir_dataset, dlr_dataset]
    target_class = 0  # Replace with the target class you're interested in

    plot_vanilla_gradients(case_indices, datasets, observer, target_class)