import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from step02_dataset_dataloader import RSNA_Intracranial_Hemorrhage_Dataset
from step04_cnn_classifier import SupervisedClassifierObserver, load_classifier
from step00_common_info import dicom_dir
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean

class FeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.features = []

    def hook(self, module, input, output):
        # Store the output features
        self.features.append(output.detach())

    def register_hook(self):
        # Hook into the last convolutional layer
        last_conv_layer = self.model.resnet.layer4[2].conv3  # Adjust based on your architecture
        last_conv_layer.register_forward_hook(self.hook)

    def get_features(self, input_image):
        self.features = []  # Reset features
        with torch.no_grad():
            self.model(input_image)
        return [feature.cpu().numpy() for feature in self.features]  # Return features as numpy arrays

def calculate_feature_drift(features_dict):
    drift_results = {}
    for class_label in features_dict.keys():
        original_features = features_dict[class_label]['Original']
        drift_results[class_label] = {}
        
        for key in ['FBP', 'MBIR', 'DLR']:
            if key in features_dict[class_label]:
                drift_results[class_label][key] = {
                    'cosine_similarity': cosine_similarity(original_features, features_dict[class_label][key]),
                    'euclidean_distance': euclidean(original_features.flatten(), features_dict[class_label][key].flatten())
                }
    return drift_results

def plot_drift_results(drift_results):
    for class_label, metrics in drift_results.items():
        plt.figure(figsize=(10, 5))
        x_labels = list(metrics.keys())
        cosine_vals = [metrics[key]['cosine_similarity'].mean() for key in x_labels]
        euclidean_vals = [metrics[key]['euclidean_distance'] for key in x_labels]

        plt.bar(x_labels, cosine_vals, color='b', alpha=0.6, label='Cosine Similarity')
        plt.plot(x_labels, euclidean_vals, color='r', marker='o', label='Euclidean Distance')
        
        plt.title(f'Feature Drift Analysis for Class: {class_label}')
        plt.ylabel('Similarity / Distance')
        plt.legend()
        plt.show()

def save_images(datasets, case_indices, save_dir='figures/feature_drift'):
    os.makedirs(save_dir, exist_ok=True)

    for dataset_name, dataset in datasets.items():
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        for case_index in case_indices:
            if case_index < len(dataset):
                image, label = dataset[case_index]  # Get the image and its label
                image = image.squeeze().cpu().numpy()  # Convert to numpy array (remove batch dimension)
                # Save the image
                plt.imsave(os.path.join(save_dir, f'{dataset_name}_case_{case_index}.png'), image, cmap='gray')
                print(f'Saved: {dataset_name}_case_{case_index}.png')

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

    datasets = {
        'Original': original_dataset,
        'FBP': fbp_dataset,
        'MBIR': mbir_dataset,
        'DLR': dlr_dataset
    }

    feature_extractor = FeatureExtractor(observer.model)
    feature_extractor.register_hook()

    # Store features for each class across datasets
    features_dict = {i: {} for i in range(4)}  # Assuming class labels are 0, 1, 2, 3 for one-hot encoding

    for key, dataset in datasets.items():
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        for images, labels in dataloader:
            images = images.to(observer.device)
            features = feature_extractor.get_features(images)
            for label in labels:
                label = label.cpu().numpy()  # Convert to numpy array
                one_hot_label = np.argmax(label)  # Get the index of the class with the highest score
                if one_hot_label not in features_dict:
                    features_dict[one_hot_label] = {}
                features_dict[one_hot_label][key] = features[0]  # Store features for this label and dataset

    # Calculate drift results
    drift_results = calculate_feature_drift(features_dict)
    
    # Plot the drift results
    plot_drift_results(drift_results)

    case_indices = list(range(1, 5)) + list(range(501, 505)) + list(range(601, 605)) + list(range(701, 705)) + list(range(801, 805)) + list(range(901, 905))

    # Save specified images
    save_images(datasets, case_indices)