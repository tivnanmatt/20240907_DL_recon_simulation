# NOT WORKING YET

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import os
from step2_dataset_dataloader import RSNA_Intracranial_Hemorrhage_Dataset
from step3_cnn_classifier import SupervisedClassifierObserver, load_classifier
import pandas as pd
from step0_common_info import dataset_dir
from pydicom import dcmread
from PIL import Image
import numpy as np
from sklearn.preprocessing import label_binarize

class DicomDataset(Dataset):
    """Custom Dataset for loading DICOM files."""
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.dicom_files = [f for f in os.listdir(folder_path) if f.endswith('.dcm')]

    def __len__(self):
        return len(self.dicom_files)

    def __getitem__(self, idx):
        dicom_file = self.dicom_files[idx]
        dicom_path = os.path.join(self.folder_path, dicom_file)

        # Read DICOM file
        dicom_data = dcmread(dicom_path)

        # Check if DICOM metadata has the wrong image size and fix it
        if dicom_data.Rows != 256 or dicom_data.Columns != 256:
            print(f"Fixing DICOM header for {dicom_file}: changing Rows and Columns to 256.")
            dicom_data.Rows = 256
            dicom_data.Columns = 256

        # Ensure that the number of bytes matches the expected pixel size (256x256)
        expected_pixel_bytes = 256 * 256 * dicom_data.BitsAllocated // 8
        actual_pixel_bytes = len(dicom_data.PixelData)
        if actual_pixel_bytes != expected_pixel_bytes:
            raise ValueError(f"Pixel data size mismatch for {dicom_file}: "
                             f"{actual_pixel_bytes} vs {expected_pixel_bytes} bytes.")

        # Now safely access the pixel array
        image = dicom_data.pixel_array
        image = Image.fromarray(image).convert("L")  # Convert to grayscale PIL image
        
        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)

        # TODO: Assign label based on your logic
        # Placeholder label: you need to replace this with your actual label logic
        label = 0  # Replace with actual label logic (e.g., using a CSV or file name)

        return image, label

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
        """Prepare a DataLoader for the DICOM dataset."""
        transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize to the expected input size for the network
            transforms.ToTensor()
        ])
        
        dataset = DicomDataset(folder_path=folder_path, transform=transform)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        return loader

    def save_results(self, folder_name, accuracy, ground_truths, predictions):
        """Save evaluation results to CSV files."""
        results_path = os.path.join(self.results_dir, f'{folder_name}_results.csv')

        # Save accuracy
        with open(results_path, 'w') as file:
            file.write(f'Accuracy: {accuracy * 100:.2f}%\n')

        # Convert ground_truths and predictions to numpy arrays
        ground_truths = np.array(ground_truths)
        predictions = np.array(predictions)

        # Ensure ground_truths are in one-hot encoded format if necessary
        if ground_truths.ndim == 1:
            num_classes = np.max(ground_truths) + 1
            ground_truths_bin = label_binarize(ground_truths, classes=np.arange(num_classes))
        else:
            ground_truths_bin = ground_truths

        # Check if predictions are probabilities
        if predictions.ndim == 1:
            raise ValueError("Predictions should be probabilities, not class labels")
        
        # Save AUC results
        ovr_auc_results = self.observer.compute_ovr_auc(ground_truths_bin, predictions)
        ovo_auc_results = self.observer.compute_ovo_auc(ground_truths_bin, predictions)

        # Save One-vs-Rest AUC results
        ovr_auc_df = pd.DataFrame(ovr_auc_results.items(), columns=['Hemorrhage Type', 'AUC'])
        ovr_auc_df.to_csv(os.path.join(self.results_dir, f'{folder_name}_ovr_auc.csv'), index=False)

        # Save One-vs-One AUC results
        ovo_auc_df = pd.DataFrame(ovo_auc_results.items(), columns=['Hemorrhage Type Pair', 'AUC'])
        ovo_auc_df.to_csv(os.path.join(self.results_dir, f'{folder_name}_ovo_auc.csv'), index=False)

        print(f'Results saved for {folder_name}.')

    def evaluate_folder(self, folder_path):
        """Evaluate model on DICOM images from a folder and save results."""
        folder_name = os.path.basename(folder_path)
        loader = self.prepare_dataloader(folder_path)
        accuracy, ground_truths, predictions = self.observer.evaluate(loader)

        # Ensure predictions are in the correct format
        predictions = np.array(predictions)
        if predictions.ndim == 1 or predictions.shape[1] != len(np.unique(ground_truths)):
            # Assuming predictions are class labels and need to be converted to probabilities
            num_classes = len(np.unique(ground_truths))
            predictions_prob = np.zeros((len(predictions), num_classes))
            for i, pred in enumerate(predictions):
                predictions_prob[i, pred] = 1  # Dummy conversion, adjust based on actual model output
            predictions = predictions_prob
        else:
            # Predictions are already probabilities
            predictions = torch.tensor(predictions)
            predictions = torch.nn.functional.softmax(predictions, dim=1).numpy()

        print(f'Evaluation Results for {folder_path}:')
        print(f'Accuracy: {accuracy * 100:.2f}%')

        # Save results to files
        self.save_results(folder_name, accuracy, ground_truths, predictions)

if __name__ == "__main__":
    # Define paths to the DICOM folders and the model
    model_path = 'weights/supervised_classifier_resnet50_weights.pth'
    folder_paths = ['data/DLR_reconstructions', 'data/FBP_reconstructions', 'data/MBIR_reconstructions']

    # Initialize evaluation script
    evaluator = EvaluationScript(model_path=model_path, batch_size=32, results_dir='results')
    
    # Loop through each folder and evaluate
    for folder in folder_paths:
        print(f"Evaluating DICOM images in {folder}...")
        evaluator.evaluate_folder(folder)