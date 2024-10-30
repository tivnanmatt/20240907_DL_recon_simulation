# from step00_common_info import dicom_dir
# from step02_dataset_dataloader import RSNA_Intracranial_Hemorrhage_Dataset
# from step04_cnn_classifier import SupervisedClassifierObserver, load_classifier

# import torch
# import matplotlib.pyplot as plt
# import numpy as np
# from torch.utils.data import DataLoader
# from sklearn.manifold import TSNE
# # import umap.umap_ as umap
# from sklearn.preprocessing import StandardScaler
# import seaborn as sns
# import os
# from sklearn.preprocessing import MinMaxScaler
# from scipy import stats
# import pandas as pd

# label_names = [
#     "no_hemorrhage", 
#     "epidural", 
#     "intraparenchymal", 
#     "intraventricular", 
#     "subarachnoid", 
#     "subdural"
# ]

# # Function to extract features from the dataset using the model
# def extract_features(model, dataloader):
#     model.eval()  # Set model to evaluation mode
#     features = []
#     labels = []
#     with torch.no_grad():
#         for data in dataloader:
#             images, targets = data
#             images = images.cuda()  # Move images to GPU if available
#             output = model(images)  # Extract features from the model
#             features.append(output.cpu().numpy())  # Move to CPU and append
#             labels.append(targets.cpu().numpy())
#     return np.concatenate(features), np.concatenate(labels)

# # Function to perform t-SNE/UMAP and visualize the results
# # def visualize_feature_space(features, labels, method='tsne', dataset_name='Original', save_dir='./figures'):
# #     # Create directory to save results if it doesn't exist
# #     if not os.path.exists(save_dir):
# #         os.makedirs(save_dir)

# #     scaler = StandardScaler()
# #     features_scaled = scaler.fit_transform(features)
    
# #     if method == 'tsne':
# #         reducer = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)
# #     # elif method == 'umap':
# #     #     reducer = umap.UMAP(n_components=2, random_state=42)
# #     # else:
# #     #     raise ValueError("Method must be either 'tsne' or 'umap'")
    
# #     reduced_features = reducer.fit_transform(features_scaled)
# #     class_labels = np.argmax(labels, axis=1)
# #     class_label_names = [label_names[label] for label in class_labels]

# #     palette = ['#e6194B',  # Bright red
# #                       '#3cb44b',  # Bright green
# #                       '#ffe119',  # Yellow
# #                       '#4363d8',  # Bright blue
# #                       '#f58231',  # Orange
# #                       '#253342']  # Purple

# #     # Plot the embeddings
# #     plt.figure(figsize=(10, 8))
# #     sns.scatterplot(x=reduced_features[:, 0], y=reduced_features[:, 1], hue=class_label_names, palette=palette, s=60, alpha=0.7, edgecolor='k')
# #     plt.title(f'{dataset_name} Dataset - {method.upper()} Feature Space')
# #     plt.xlabel('Component 1')
# #     plt.ylabel('Component 2')
# #     plt.legend(loc='best', title='Classes', fontsize='small')

#     # Save the image with a descriptive name
#     # image_path = os.path.join(save_dir, f'{dataset_name}_feature_space_{method}.png')
#     # plt.savefig(image_path)
#     # print(f"Saved {method.upper()} visualization for {dataset_name} dataset at {image_path}")

#     # plt.close()  # Close the plot to free memory

# from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
# from sklearn.preprocessing import MinMaxScaler
# from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting

# def statistical_comparison(features, labels, dataset_names):
#     # Prepare a DataFrame to store feature distributions
#     feature_data = pd.DataFrame(features)
#     feature_data['label'] = np.argmax(labels, axis=1)

#     # Create a dictionary to hold results
#     results = {}

#     for label_index in range(len(label_names)):
#         # Filter data for the current label
#         current_label_data = feature_data[feature_data['label'] == label_index]

#         # Perform statistical tests for each feature across datasets
#         dataset_results = {}
#         for column in current_label_data.columns[:-1]:  # Exclude the label column
#             groups = [current_label_data[current_label_data['dataset'] == dataset][column] for dataset in dataset_names]
#             # Perform Kruskal-Wallis test
#             stat, p_value = stats.kruskal(*groups)
#             dataset_results[column] = {'statistic': stat, 'p_value': p_value}

#         results[label_names[label_index]] = dataset_results

#     return results

# # Updated function to support 3D plotting and normalization
# def visualize_feature_space(features, labels, method='tsne', dataset_name='Original', save_dir='./figures', n_components=2, normalize=True):
#     # Create directory to save results if it doesn't exist
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)

#     scaler = StandardScaler()
#     features_scaled = scaler.fit_transform(features)
    
#     if method == 'tsne':
#         reducer = TSNE(n_components=n_components, random_state=42)
    
#     reduced_features = reducer.fit_transform(features_scaled)

#     if normalize:
#         # Normalize the t-SNE components to range [0, 1]
#         min_max_scaler = MinMaxScaler()
#         reduced_features = min_max_scaler.fit_transform(reduced_features)

#     class_labels = np.argmax(labels, axis=1)
#     class_label_names = [label_names[label] for label in class_labels]

#     # Custom color palette with highly distinguishable colors
#     custom_palette = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4']

#     if n_components == 3:
#         # 3D Plot
#         fig = plt.figure(figsize=(10, 8))
#         ax = fig.add_subplot(111, projection='3d')
        
#         # Scatter plot in 3D
#         for idx, class_name in enumerate(label_names):
#             indices = [i for i, name in enumerate(class_label_names) if name == class_name]
#             ax.scatter(reduced_features[indices, 0], reduced_features[indices, 1], reduced_features[indices, 2],
#                        label=class_name, s=60, alpha=0.8, edgecolor='k', color=custom_palette[idx])
        
#         ax.set_title(f'{dataset_name} Dataset - {method.upper()} 3D Feature Space')
#         ax.set_xlabel('Component 1 (normalized)')
#         ax.set_ylabel('Component 2 (normalized)')
#         ax.set_zlabel('Component 3 (normalized)')
#         ax.legend(loc='best', title='Classes', fontsize='small')

#     else:
#         # 2D Plot (fallback for 2D t-SNE)
#         plt.figure(figsize=(10, 8))
#         sns.scatterplot(x=reduced_features[:, 0], y=reduced_features[:, 1], hue=class_label_names, 
#                         palette=custom_palette, s=60, alpha=0.8, edgecolor='k')
#         plt.title(f'{dataset_name} Dataset - {method.upper()} Feature Space')
#         plt.xlabel('Component 1 (normalized)')
#         plt.ylabel('Component 2 (normalized)')
#         plt.legend(loc='best', title='Classes', fontsize='small')

#     # Save the plot
#     image_path = os.path.join(save_dir, f'{dataset_name}_feature_space_{method}_{n_components}D.png')
#     plt.savefig(image_path)
#     print(f"Saved {method.upper()} {n_components}D visualization for {dataset_name} dataset at {image_path}")

#     plt.close()  # Close the plot to free memory
#   # Close the plot to free memory


# if __name__ == "__main__":
#     original_csv_file = 'data/metadata_evaluation.csv'
#     original_dicom_dir = dicom_dir

#     fbp_dicom_dir = 'data/FBP_reconstructions/'
#     mbir_dicom_dir = 'data/MBIR_reconstructions/'
#     dlr_dicom_dir = 'data/DLR_reconstructions/'

#     print('Loading datasets...')
#     original_dataset = RSNA_Intracranial_Hemorrhage_Dataset(original_csv_file, original_dicom_dir, expected_size=512)
#     fbp_dataset = RSNA_Intracranial_Hemorrhage_Dataset(original_csv_file, fbp_dicom_dir, expected_size=256)
#     mbir_dataset = RSNA_Intracranial_Hemorrhage_Dataset(original_csv_file, mbir_dicom_dir, expected_size=256)
#     dlr_dataset = RSNA_Intracranial_Hemorrhage_Dataset(original_csv_file, dlr_dicom_dir, expected_size=256)

#     datasets = {
#         'Original': original_dataset,
#         'FBP': fbp_dataset,
#         'MBIR': mbir_dataset,
#         'DLR': dlr_dataset
#     }

#     observer = SupervisedClassifierObserver(verbose=True, batch_size=1)
#     observer.model = load_classifier(observer.model, 'weights/supervised_classifier_resnet50_weights_09102024.pth')

#     print('Extracting features and visualizing...')
#     for key, dataset in datasets.items():
#         dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
#         features, labels = extract_features(observer.model, dataloader)
#         n_components = 2
        
#         # Visualize using t-SNE
#         visualize_feature_space(features, labels, method='tsne', dataset_name=key, save_dir='./figures/tsne', n_components=n_components, normalize=True)
        
#         # Visualize using UMAP
#         # visualize_feature_space(features, labels, method='umap', dataset_name=key, save_dir='./figures/umap')

from step00_common_info import dicom_dir
from step02_dataset_dataloader import RSNA_Intracranial_Hemorrhage_Dataset
from step04_cnn_classifier import SupervisedClassifierObserver, load_classifier

import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns
import os
import pandas as pd
from scipy import stats

label_names = [
    "no_hemorrhage", 
    "epidural", 
    "intraparenchymal", 
    "intraventricular", 
    "subarachnoid", 
    "subdural"
]

# Function to extract features from the dataset using the model
def extract_features(model, dataloader):
    model.eval()  # Set model to evaluation mode
    features = []
    labels = []
    with torch.no_grad():
        for data in dataloader:
            images, targets = data
            images = images.cuda()  # Move images to GPU if available
            output = model(images)  # Extract features from the model
            features.append(output.cpu().numpy())  # Move to CPU and append
            labels.append(targets.cpu().numpy())
    return np.concatenate(features), np.concatenate(labels)

# Function to perform t-SNE and visualize the results
def visualize_feature_space(features, labels, method='tsne', dataset_name='Original', save_dir='./figures', n_components=2, normalize=True):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    if method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=42)
    
    reduced_features = reducer.fit_transform(features_scaled)

    if normalize:
        min_max_scaler = MinMaxScaler()
        reduced_features = min_max_scaler.fit_transform(reduced_features)

    class_labels = np.argmax(labels, axis=1)
    class_label_names = [label_names[label] for label in class_labels]

    custom_palette = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4']

    if n_components == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        for idx, class_name in enumerate(label_names):
            indices = [i for i, name in enumerate(class_label_names) if name == class_name]
            ax.scatter(reduced_features[indices, 0], reduced_features[indices, 1], reduced_features[indices, 2],
                       label=class_name, s=60, alpha=0.8, edgecolor='k', color=custom_palette[idx])
        
        ax.set_title(f'{dataset_name} Dataset - {method.upper()} 3D Feature Space')
        ax.set_xlabel('Component 1 (normalized)')
        ax.set_ylabel('Component 2 (normalized)')
        ax.set_zlabel('Component 3 (normalized)')
        ax.legend(loc='best', title='Classes', fontsize='small')

    else:
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=reduced_features[:, 0], y=reduced_features[:, 1], hue=class_label_names, 
                        palette=custom_palette, s=60, alpha=0.8, edgecolor='k')
        plt.title(f'{dataset_name} Dataset - {method.upper()} Feature Space')
        plt.xlabel('Component 1 (normalized)')
        plt.ylabel('Component 2 (normalized)')
        plt.legend(loc='best', title='Classes', fontsize='small')

    image_path = os.path.join(save_dir, f'{dataset_name}_feature_space_{method}_{n_components}D.png')
    plt.savefig(image_path)
    print(f"Saved {method.upper()} {n_components}D visualization for {dataset_name} dataset at {image_path}")
    plt.close()  # Close the plot to free memory

# Function for statistical comparison of features across datasets
def statistical_comparison(features, labels, dataset_names):
    feature_data = pd.DataFrame(features)
    num_features_per_dataset = len(features)//len(dataset_names)
    dataset_names_repeated = np.concatenate([[name] * num_features_per_dataset for name in dataset_names])
    feature_data['label'] = np.argmax(labels, axis=1)
    feature_data['dataset'] = dataset_names_repeated

    results = {}

    for label_index in range(len(label_names)):
        current_label_data = feature_data[feature_data['label'] == label_index]
        dataset_results = {}
        
        for column in current_label_data.columns[:-2]:  # Ex clue label and dataset columns
            groups = [current_label_data[current_label_data['dataset'] == dataset][column] for dataset in dataset_names]
            stat, p_value = stats.kruskal(*groups)
            dataset_results[column] = {'statistic': stat, 'p_value': p_value}

        results[label_names[label_index]] = dataset_results

    return results

# Function to visualize feature distributions
def visualize_feature_distributions(feature_data, dataset_names, comparison_results):
    for label_index in range(len(label_names)):
        current_label_data = feature_data[feature_data['label'] == label_index]
        
        plt.figure(figsize=(12, 6))
        for column in current_label_data.columns[:-2]:  # Exclude label and dataset columns
            sns.boxplot(data=current_label_data, x='dataset', y=column)
            plt.title(f'Distribution of {column} for {label_names[label_index]}')
            plt.ylabel(column)
            plt.xlabel('Dataset')
            plt.xticks(rotation=45)

            # Add p-values to the plot
            if label_names[label_index] in comparison_results:
                for dataset in dataset_names:
                    if column in comparison_results[label_names[label_index]]:
                        p_value = comparison_results[label_names[label_index]][column]['p_value']
                        plt.annotate(f'p = {p_value:.4f}', 
                                     xy=(dataset_names.index(dataset), current_label_data[column].max()), 
                                     xytext=(0, 10), 
                                     textcoords='offset points', 
                                     ha='center', 
                                     fontsize=10, 
                                     color='black')

            plt.tight_layout()
            plt.savefig(f'./figures/tsne/v2/distribution_{label_names[label_index]}_{column}v2.png')
            plt.close()
if __name__ == "__main__":
    original_csv_file = 'data/metadata_evaluation.csv'
    original_dicom_dir = dicom_dir

    fbp_dicom_dir = 'data/FBP_reconstructions/'
    mbir_dicom_dir = 'data/MBIR_reconstructions/'
    dlr_dicom_dir = 'data/DLR_reconstructions/'

    print('Loading datasets...')
    original_dataset = RSNA_Intracranial_Hemorrhage_Dataset(original_csv_file, original_dicom_dir, expected_size=512)
    fbp_dataset = RSNA_Intracranial_Hemorrhage_Dataset(original_csv_file, fbp_dicom_dir, expected_size=256)
    mbir_dataset = RSNA_Intracranial_Hemorrhage_Dataset(original_csv_file, mbir_dicom_dir, expected_size=256)
    dlr_dataset = RSNA_Intracranial_Hemorrhage_Dataset(original_csv_file, dlr_dicom_dir, expected_size=256)

    datasets = {
        'Original': original_dataset,
        'FBP': fbp_dataset,
        'MBIR': mbir_dataset,
        'DLR': dlr_dataset
    }

    observer = SupervisedClassifierObserver(verbose=True, batch_size=1)
    observer.model = load_classifier(observer.model, 'weights/supervised_classifier_resnet50_weights_09102024.pth')

    print('Extracting features and visualizing...')
    all_features = {}
    all_labels = {}

    for key, dataset in datasets.items():
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        features, labels = extract_features(observer.model, dataloader)
        
        all_features[key] = features
        all_labels[key] = labels

    features_combined = np.vstack([all_features[key] for key in datasets.keys()])
    labels_combined = np.vstack([all_labels[key] for key in datasets.keys()])

    dataset_names = list(datasets.keys())
    dataset_labels = np.concatenate([[key] * len(all_features[key]) for key in dataset_names])

    feature_data_combined = pd.DataFrame(features_combined)
    feature_data_combined['dataset'] = dataset_labels
    feature_data_combined['label'] = np.argmax(labels_combined, axis=1)

    for key, dataset in datasets.items():
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        features, labels = extract_features(observer.model, dataloader)
        
        n_components = 2
        visualize_feature_space(features, labels, method='tsne', dataset_name=key, save_dir='./figures/tsne/v2', n_components=n_components, normalize=True)

    comparison_results = statistical_comparison(features_combined, labels_combined, dataset_names)
    print("Statistical Comparison Results:")
    for label, result in comparison_results.items():
        print(f"Statistical Comparison Results for {label}:")
        for feature, stats in result.items():
            print(f"Feature: {feature}, Statistic: {stats['statistic']}, p-value: {stats['p_value']}")

    # visualize_feature_distributions(feature_data_combined, dataset_names)
    visualize_feature_distributions(feature_data_combined, dataset_names, comparison_results)