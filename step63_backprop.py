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

class GuidedBackprop:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.forward_relu_outputs = []
        self.update_relus()

    def update_relus(self):
        def relu_backward_hook_function(module, grad_in, grad_out):
            corresponding_forward_output = self.forward_relu_outputs[-1].clone()  
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            
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

def plot_guided_backprop(case_indices, datasets, observer, target_class, save_dir='figures/saliency/backprop'):
    os.makedirs(save_dir, exist_ok=True)
    observer.model.eval()

    # Create a figure with 4 rows and 2 columns
    fig, axs = plt.subplots(len(case_indices), 4, figsize=(20, 5 * len(case_indices)))

    for i, case_index in enumerate(case_indices):
        for j, dataset in enumerate(datasets):
            data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
            input_image, _ = next(iter(data_loader))
            input_image = apply_window(input_image, 0, 80).to(observer.device)

            explainer = GuidedBackprop(observer.model)
            explanation = explainer.explain(input_image, target_class)

            # Plot original image in the first two columns
            axs[i, j].imshow(input_image.squeeze().detach().cpu().numpy(), cmap='gray')
            axs[i, j].set_title(f'Original Image - {dataset.__class__.__name__} (Case {case_index})')
            axs[i, j].axis('off')

            # Plot guided backpropagation result in the last two columns
            if j < 2:  # Only plot guided backpropagation for the first two datasets
                axs[i, j + 2].imshow(explanation, cmap='hot')
                axs[i, j + 2].set_title('Guided Backpropagation')
                axs[i, j + 2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'guided_backprop_all_cases.png'))  # Save the entire figure
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

    plot_guided_backprop(case_indices, datasets, observer, target_class)