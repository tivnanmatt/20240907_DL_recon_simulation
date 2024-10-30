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

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.gradients = None
        self.activations = None
        self.register_hooks()

    def register_hooks(self):
        def forward_hook(module, input, output):
            # Clone the output to avoid modifying the view
            self.activations = output.detach().clone()

        def backward_hook(module, grad_in, grad_out):
            # Clone the gradient to avoid modifying the view
            self.gradients = grad_out[0].detach().clone()
            

        # Hook into the last convolutional layer
        last_conv_layer = self.model.resnet.layer4[2].conv3  # Adjust based on your architecture
        last_conv_layer.register_forward_hook(forward_hook)
        last_conv_layer.register_full_backward_hook(backward_hook)

    def generate_explanation(self, input_image, target_class):
        input_image.requires_grad_()
        self.model.zero_grad()

        output = self.model(input_image)
        print("Model output:", output)
        print("Score for target class:", output[0, target_class].item())

        output[0, target_class].backward()

        gradients = self.gradients
        activations = self.activations

        print("Gradients shape:", gradients.shape)
        print("Gradients values:", gradients)
        print("Activations shape:", activations.shape)
        print("Activations values:", activations)

        weights = torch.mean(gradients, dim=(1, 2), keepdim=True)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = torch.relu(cam)

        if cam.max() == 0:
            print("Warning: CAM max is zero, normalization will fail.")
            return np.zeros_like(cam.squeeze().cpu().numpy())  # Return a zero array if CAM max is zero

        cam = cam / (cam.max() + 1e-5)  # Normalize the CAM
        return cam.squeeze().cpu().numpy()

    def explain(self, input_image, target_class):
        explanation = self.generate_explanation(input_image, target_class)
        return explanation
    
def plot_gradcam_for_predictions(datasets, observer, results, case_indices, save_dir='figures/saliency/gradcam'):
    os.makedirs(save_dir, exist_ok=True)

    # Iterate over case indices
    for case_index in case_indices:
        # Create a figure for a 2-row layout (1 row for original images, 1 row for Grad-CAM results)
        fig, axs = plt.subplots(2, 4, figsize=(16, 8))  # 2 rows, 4 columns
        fig.subplots_adjust(hspace=0.4)  # Adjust space between rows

        # Iterate over datasets and plot images and Grad-CAM results based on case_index
        for j, (key, data) in enumerate(results.items()):
            predictions = data['predictions']
            ground_truths = data['ground_truths']

            # Ensure case_index is valid for the dataset
            if case_index < len(datasets[key]):
                # Get the original image
                input_image, _ = datasets[key][case_index]  # Get the image for the current case
                input_image = input_image.unsqueeze(0).to(observer.device)  # Add batch dimension and move to device

                # Ensure predictions[case_index] is a tensor
                predictions_tensor = torch.tensor(predictions[case_index]).to(observer.device) if isinstance(predictions[case_index], np.ndarray) else predictions[case_index].to(observer.device)

                # Convert one-hot encoded prediction to class index
                target_class = torch.argmax(predictions_tensor).item()  # Convert to class index

                # Generate Grad-CAM explanation
                explainer = GradCAM(observer.model)
                explanation = explainer.explain(input_image, target_class)

                # Plot original image in the first row
                axs[0, j].imshow(input_image.squeeze().detach().cpu().numpy(), cmap='gray')
                axs[0, j].set_title(f'{key} Image (Predicted: {target_class})')
                axs[0, j].axis('off')

                # Plot Grad-CAM result in the second row
                axs[1, j].imshow(explanation, cmap='hot', vmin=0, vmax=1)
                axs[1, j].set_title(f'Grad-CAM - {key} (Class {target_class})')
                axs[1, j].axis('off')

        # Save the figure for the current case index
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'gradcam_case_{case_index}.png'))  # Save the figure with a unique filename
        plt.close()  # Close the figure to free up memory

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

    # Now plot Grad-CAM for each prediction
    plot_gradcam_for_predictions(datasets, observer, results, case_indices)

    # plot_gradcam(case_indices, datasets, observer, target_class)