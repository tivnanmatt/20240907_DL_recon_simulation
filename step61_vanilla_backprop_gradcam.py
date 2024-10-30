import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from step02_dataset_dataloader import RSNA_Intracranial_Hemorrhage_Dataset
from step04_cnn_classifier import SupervisedClassifierObserver, load_classifier
from step00_common_info import dicom_dir

# Is giving out of cuda memory error

def apply_window(image, window_min, window_max):
    return np.clip(image, window_min, window_max)

class BaseExplainer:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.device = next(self.model.parameters()).device

    def generate_explanation(self, input_image, target_class):
        input_image = input_image.to(self.device)
        raise NotImplementedError

    def visualize(self, input_image, explanation, title):
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(input_image.squeeze().cpu().numpy(), cmap='gray')
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(explanation, cmap='hot')
        plt.title(title)
        plt.axis('off')

        plt.tight_layout()
        plt.show()

class VanillaGradients(BaseExplainer):
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

class GuidedBackprop(BaseExplainer):
    def __init__(self, model):
        super().__init__(model)
        self.forward_relu_outputs = []
        self.update_relus()

    def update_relus(self):
        def relu_backward_hook_function(module, grad_in, grad_out):
            # Clone the output of the forward layer to avoid modifying views
            corresponding_forward_output = self.forward_relu_outputs[-1].clone()  
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            
            # Clone the gradient before modifying it
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0].clone(), min=0.0)  
            del self.forward_relu_outputs[-1]
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            self.forward_relu_outputs.append(ten_out.clone())

        for module in self.model.modules():
            if isinstance(module, torch.nn.ReLU):
                module.register_full_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)

    def generate_explanation(self, input_image, target_class):
        input_image.requires_grad_()
        self.model.zero_grad()
        
        output = self.model(input_image)
        output[0, target_class].backward()

        gradients = input_image.grad.data
        return gradients.squeeze().cpu().numpy()

    def explain(self, input_image, target_class):
        explanation = self.generate_explanation(input_image, target_class)
        return explanation

class GradCAM(BaseExplainer):
    def __init__(self, model):
        super().__init__(model)
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
        output[0, target_class].backward()

        gradients = self.gradients
        activations = self.activations

        # Ensure gradients and activations are not None
        if gradients is None or activations is None:
            raise ValueError("Gradients or activations are None. Check your hooks.")

        weights = torch.mean(gradients, dim=(1, 2), keepdim=True)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam / (cam.max() + 1e-5)  # Normalize the CAM
        return cam.squeeze().cpu().numpy()

    def explain(self, input_image, target_class):
        explanation = self.generate_explanation(input_image, target_class)
        return explanation

def plot_cases_with_gradcam(case_indices, datasets, observer, target_class, save_dir='figures/gradcam/vanilla/v1'):
    os.makedirs(save_dir, exist_ok=True)
    observer.model.eval()

    # Create a figure with 4 columns and as many rows as the length of case_indices times the number of datasets (4 rows per case)
    fig, axs = plt.subplots(len(case_indices) * len(datasets), 4, figsize=(20, len(case_indices) * 5))
    
    # Iterate over case indices
    for i, case_index in enumerate(case_indices):
        # Iterate over datasets
        for j, dataset in enumerate(datasets):
            # Create the data loader for the current dataset
            data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
            input_image, _ = next(iter(data_loader))
            
            # Apply the windowing function to the input image
            input_image = apply_window(input_image, 0, 80)
            input_image = input_image.to(observer.device)
            
            # Calculate the row index for the current case and dataset
            row_index = i * len(datasets) + j
            
            # Original Image
            axs[row_index, 0].imshow(input_image.squeeze().cpu().numpy(), cmap='gray')
            axs[row_index, 0].axis('off')
            if j == 0:
                axs[row_index, 0].set_title('Original Image')

            # Get the model's prediction for this image
            with torch.no_grad():
                output = observer.model(input_image)
                predicted_class = torch.argmax(output, dim=1).item()
            
            # Use the predicted class as the target class
            target_class = predicted_class
            
            # Vanilla Gradients
            explainer_vanilla = VanillaGradients(observer.model)
            explanation_vanilla = explainer_vanilla.explain(input_image, target_class)
            axs[row_index, 1].imshow(explanation_vanilla, cmap='hot')
            axs[row_index, 1].axis('off')
            if j == 0:
                axs[row_index, 1].set_title('Vanilla Gradients')
            
            # Guided Backpropagation
            explainer_guided_backprop = GuidedBackprop(observer.model)
            explanation_guided_backprop = explainer_guided_backprop.explain(input_image, target_class)
            axs[row_index, 2].imshow(explanation_guided_backprop, cmap='hot')
            axs[row_index, 2].axis('off')
            if j == 0:
                axs[row_index, 2].set_title('Guided Backpropagation')
            
            # Grad-CAM
            explainer_gradcam = GradCAM(observer.model)
            explanation_gradcam = explainer_gradcam.explain(input_image, target_class)
            axs[row_index, 3].imshow(explanation_gradcam, cmap='hot')
            axs[row_index, 3].axis('off')
            if j == 0:
                axs[row_index, 3].set_title('Grad-CAM')
        
        # Free GPU memory after each case to avoid memory build-up
        torch.cuda.empty_cache()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'gradcam_cases_{case_index}.png'))  # Save the figure


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

    observer = SupervisedClassifierObserver(verbose=True, batch_size=1)
    observer.model = load_classifier(observer.model, 'weights/supervised_classifier_resnet50_weights_09102024.pth')
    observer.model = observer.model.to(observer.device)
    observer.model.eval()

    case_indices = list(range(1, 5)) + list(range(501, 505)) + list(range(601, 605)) + list(range(701, 705)) + list(range(801, 805)) + list(range(901, 905))
    datasets = [original_dataset, fbp_dataset, mbir_dataset, dlr_dataset]
    target_class = 0  # Replace with the target class you're interested in
    save_dir = 'figures/gradcam/vanilla/v1'  # or any other directory you want to save to

    plot_cases_with_gradcam(case_indices, datasets, observer, target_class, save_dir)