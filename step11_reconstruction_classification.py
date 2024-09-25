# NOT WORKING YET, but not seeing it right now

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
from torchvision.transforms import ToTensor
from sklearn.preprocessing import label_binarize

class DicomDataset(Dataset):
    """Custom Dataset for loading DICOM files."""
    def __init__(self, folder_path, labels_csv=None, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.dicom_files = [f for f in os.listdir(folder_path) if f.endswith('.dcm')]

        # Load labels if provided
        self.labels = self.load_labels(labels_csv) if labels_csv else [0] * len(self.dicom_files)

    def load_labels(self, labels_csv):
        """Load labels from a CSV file."""
        df = pd.read_csv(labels_csv)
        file_labels = dict(zip(df['filename'], df['label']))  # Adjust based on your CSV structure
        labels = [file_labels.get(f, 0) for f in self.dicom_files]  # Default to 0 if file not found
        return labels

    def __len__(self):
        return len(self.dicom_files)

    def __getitem__(self, idx):
        dicom_file = self.dicom_files[idx]
        dicom_path = os.path.join(self.folder_path, dicom_file)

        dicom_data = dcmread(dicom_path)

        if dicom_data.Rows != 256 or dicom_data.Columns != 256:
            dicom_data.Rows = 256
            dicom_data.Columns = 256

        expected_pixel_bytes = 256 * 256 * dicom_data.BitsAllocated // 8
        actual_pixel_bytes = len(dicom_data.PixelData)
        if actual_pixel_bytes != expected_pixel_bytes:
            raise ValueError(f"Pixel data size mismatch for {dicom_file}: "
                            f"{actual_pixel_bytes} vs {expected_pixel_bytes} bytes.")

        image = dicom_data.pixel_array
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("L")

        # Debugging print statements
        print(f"Type of image before transformation: {type(image)}")

        if self.transform:
            image = self.transform(image)
            print(f"Type of image after transformation: {type(image)}")

        if image.ndimension() == 3 and image.shape[0] != 1:
            image = image.unsqueeze(0)

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return image, label

class EvaluationScript:
    def __init__(self, model_path, csv_file, batch_size=32, results_dir='results'):
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

        self.csv_file = csv_file

    def prepare_dataloader(self, folder_path):
        """Prepare a DataLoader for the dataset in the specified folder."""
        transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize to the expected input size for the network
            transforms.ToTensor()
        ])
        
        dataset = RSNA_Intracranial_Hemorrhage_Dataset(
            csv_file=self.csv_file,
            dicom_dir=folder_path,
            transform=transform
        )
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

        # Ensure ground_truths are in one-hot encoded format
        if ground_truths.ndim == 1:
            num_classes = 6  # Adjust based on the number of classes
            ground_truths_bin = label_binarize(ground_truths, classes=np.arange(num_classes))
        else:
            ground_truths_bin = ground_truths

        # Ensure predictions are probabilities
        if predictions.ndim == 1:
            num_classes = 6  # Adjust based on the number of classes
            predictions_prob = np.zeros((len(predictions), num_classes))
            for i, pred in enumerate(predictions):
                predictions_prob[i, pred] = 1
            predictions = predictions_prob
        else:
            # Predictions are already probabilities
            predictions = torch.tensor(predictions)
            predictions = F.softmax(predictions, dim=1).numpy()

        print(f'Evaluation Results for {folder_name}:')
        print(f'Accuracy: {accuracy * 100:.2f}%')

        # Save results to files
        ovr_auc_results = self.observer.compute_ovr_auc(ground_truths_bin, predictions)
        ovo_auc_results = self.observer.compute_ovo_auc(ground_truths_bin, predictions)

        ovr_auc_df = pd.DataFrame(ovr_auc_results.items(), columns=['Hemorrhage Type', 'AUC'])
        ovr_auc_df.to_csv(os.path.join(self.results_dir, f'{folder_name}_ovr_auc.csv'), index=False)

        ovo_auc_df = pd.DataFrame(ovo_auc_results.items(), columns=['Hemorrhage Type Pair', 'AUC'])
        ovo_auc_df.to_csv(os.path.join(self.results_dir, f'{folder_name}_ovo_auc.csv'), index=False)

        print(f'Results saved for {folder_name}.')

    def evaluate_folder(self, folder_path):
        """Evaluate model on images from a specific folder and save results."""
        folder_name = os.path.basename(folder_path)
        loader = self.prepare_dataloader(folder_path)
        accuracy, ground_truths, predictions = self.observer.evaluate(loader)

        # Ensure predictions are in the correct format
        predictions = np.array(predictions)  # Ensure predictions is a numpy array

        # Check dimensions to decide how to handle the predictions
        if predictions.ndim == 1:  # If predictions are class indices
            num_classes = 6
            predictions_prob = np.zeros((len(predictions), num_classes))
            for i, pred in enumerate(predictions):
                predictions_prob[i, pred] = 1
            predictions = predictions_prob
        elif predictions.ndim == 2 and predictions.shape[1] == num_classes:  # If already probabilities
            predictions = predictions  # No change needed
        else:
            raise ValueError("Unexpected shape of predictions array")

        print(f'Evaluation Results for {folder_path}:')
        print(f'Accuracy: {accuracy * 100:.2f}%')

        # Save results to files
        self.save_results(folder_name, accuracy, ground_truths, predictions)

if __name__ == "__main__":
    # Define paths to the CSV file, model, and dataset folders
    model_path = 'weights/supervised_classifier_resnet50_weights.pth'
    csv_file = 'data/metadata_evaluation.csv'
    folder_paths = ['data/DLR_reconstructions', 'data/FBP_reconstructions', 'data/MBIR_reconstructions']

    # Initialize evaluation script
    evaluator = EvaluationScript(model_path=model_path, csv_file=csv_file, batch_size=32, results_dir='results')
    
    # Loop through each folder and evaluate
    for folder in folder_paths:
        print(f"Evaluating images in {folder}...")
        evaluator.evaluate_folder(folder)