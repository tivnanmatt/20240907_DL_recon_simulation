# NOT WORKING YET

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from step2_dataset_dataloader import RSNA_Intracranial_Hemorrhage_Dataset
from step3_cnn_classifier import SupervisedClassifierObserver, load_classifier
import pandas as pd

class EvaluationScript:
    def __init__(self, model_path, batch_size=32, results_dir='results'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.observer = SupervisedClassifierObserver(batch_size=batch_size)
        
        # Load the pretrained model
        self.observer.model = load_classifier(self.observer.model, model_path)
        self.observer.model = self.observer.model.to(self.device)
        self.batch_size = batch_size
        self.results_dir = results_dir

        # Ensure the results directory exists
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def prepare_dataloader(self, folder_path):
        """Prepare the DataLoader for the given folder path."""
        transform = transforms.Compose([
            transforms.Grayscale(),  # If images are grayscale
            transforms.Resize((256, 256)),  # Resize to the expected input size for the network
            transforms.ToTensor()
        ])
        
        dataset = datasets.ImageFolder(root=folder_path, transform=transform)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        return loader

    def save_results(self, folder_name, accuracy, ground_truths, predictions):
        """Save evaluation results to CSV files."""
        results_path = os.path.join(self.results_dir, f'{folder_name}_results.csv')

        # Save accuracy
        with open(results_path, 'w') as file:
            file.write(f'Accuracy: {accuracy * 100:.2f}%\n')

        # Save AUC results
        ovr_auc_results = self.observer.compute_ovr_auc(ground_truths, predictions)
        ovo_auc_results = self.observer.compute_ovo_auc(ground_truths, predictions)

        # Save One-vs-Rest AUC results
        ovr_auc_df = pd.DataFrame(ovr_auc_results.items(), columns=['Hemorrhage Type', 'AUC'])
        ovr_auc_df.to_csv(os.path.join(self.results_dir, f'{folder_name}_ovr_auc.csv'), index=False)

        # Save One-vs-One AUC results
        ovo_auc_df = pd.DataFrame(ovo_auc_results.items(), columns=['Hemorrhage Type Pair', 'AUC'])
        ovo_auc_df.to_csv(os.path.join(self.results_dir, f'{folder_name}_ovo_auc.csv'), index=False)

        print(f'Results saved for {folder_name}.')

    def evaluate_folder(self, folder_path):
        """Evaluate model on images from a folder and save results."""
        folder_name = os.path.basename(folder_path)
        loader = self.prepare_dataloader(folder_path)
        accuracy, ground_truths, predictions = self.observer.evaluate(loader)
        print(f'Evaluation Results for {folder_path}:')
        print(f'Accuracy: {accuracy * 100:.2f}%')

        # Save results to files
        self.save_results(folder_name, accuracy, ground_truths, predictions)

if __name__ == "__main__":
    # Define paths to the image folders and the model
    model_path = 'weights/supervised_classifier_resnet50_weights.pth'
    folder_paths = ['data/DLR_reconstruction', 'data/FBP_reconstruction', 'data/MBIR_reconstruction']

    # Initialize evaluation script
    evaluator = EvaluationScript(model_path=model_path, batch_size=32, results_dir='results')
    
    # Loop through each folder and evaluate
    for folder in folder_paths:
        print(f"Evaluating images in {folder}...")
        evaluator.evaluate_folder(folder)