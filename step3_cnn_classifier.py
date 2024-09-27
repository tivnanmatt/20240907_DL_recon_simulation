import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from step2_dataset_dataloader import RSNA_Intracranial_Hemorrhage_Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, auc
from sklearn.preprocessing import label_binarize
import numpy as np
from tqdm import tqdm
from torch.utils.data import Subset
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
import pandas as pd
from torch_ema import ExponentialMovingAverage

# Get the 'tab20' colormap
cmap = plt.get_cmap('tab20')

# Extract even-indexed and odd-indexed colors
colors_even = cmap.colors[::2]  # Get all even indices
colors_odd = cmap.colors[1::2]  # Get all odd indices

# Combine even and odd colors
custom_colors = colors_even + colors_odd

# Set the custom color cycle as the default
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=custom_colors)

class SupervisedClassifier(nn.Module):
    def __init__(self):
        super(SupervisedClassifier, self).__init__()
        # Load pretrained ResNet50 model
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)

        # Modify the first conv layer to accept 1 channel instead of 3
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Replace the final fully connected layer
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 512),  # Keep output size from resnet's fc
            nn.BatchNorm1d(512),
            nn.PReLU(),
            nn.Linear(512, 6)  # 6 output classes for multi-class classification
        )

    def forward(self, x):
        # print(f"Input batch size: {x.size()} on {x.device}")
        return self.resnet(x)
    

class SupervisedClassifierObserver:
    def __init__(self, device=None, verbose=False, batch_size=32):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = SupervisedClassifier().to(self.device)
        self.verbose = verbose
        self.batch_size = batch_size
        self.labels = ['no_hemorrhage', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']

    def train(self, train_loader, val_loader=None, test_loader=None, verbose=False, num_epochs=20, num_iterations_train=100, num_iterations_val=10, lr=2e-4):
        criterion = nn.CrossEntropyLoss()  # Loss function for multi-class classification
        optimizer = optim.Adam(self.model.parameters(), lr=lr)  # Optimizer
        ema = ExponentialMovingAverage(self.model.parameters(), decay=0.95)  # Exponential moving average for stabilizing training

        # Training loop
        self.model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            train_loader_iter = iter(train_loader)

            for _ in tqdm(range(num_iterations_train)):
                try:
                    images, labels = next(train_loader_iter)
                except StopIteration:
                    train_loader_iter = iter(train_loader)
                    images, labels = next(train_loader_iter)

                images = images.to(self.device)
                labels = labels.to(self.device)  # CrossEntropy expects LongTensor for labels

                max_noise_std = 1000.0
                noise_std = torch.rand(images.size(0), 1, 1, 1).to(images.device) * max_noise_std
                noise = torch.randn_like(images) * noise_std

                images_noisy = images + noise


                optimizer.zero_grad()
                # outputs = self.model(images) # non-robust classifier
                outputs = self.model(images_noisy) # robust classifier
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                ema.update()  # Update EMA

                running_loss += loss.item()

            print(f"Epoch {epoch + 1}, Training Loss: {running_loss / num_iterations_train}")

            if val_loader:
                val_loss = self.validate(val_loader, num_iterations_val=num_iterations_val)
                print(f"Validation Loss after Epoch {epoch + 1}: {val_loss}")

            # Save the trained model weights
            # torch.save(self.model.state_dict(), 'weights/supervised_classifier_resnet50_weights.pth')
            save_classifier(self.model, 'weights/supervised_classifier_resnet50_weights.pth')


            if test_loader is not None:

                print("Running evaluation...")
                import time
                t0 = time.time()

                results = observer.evaluate(test_loader, num_patients=len(test_dataset))
                observer.print_evaluation(results)

                print("Time taken for evaluation:", time.time() - t0)

    def validate(self, val_loader, num_iterations_val=10):
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        self.model.eval()

        val_loader_iter = iter(val_loader)
        with torch.no_grad():
            for _ in tqdm(range(num_iterations_val)):
                try:
                    images, labels = next(val_loader_iter)
                except StopIteration:
                    break

                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

        avg_loss = total_loss / num_iterations_val
        return avg_loss

    def evaluate(self, test_loader, num_patients=256):
        test_loader_iter = iter(test_loader)

        all_ground_truths = []
        all_predictions_gpu = []
        total = 0
        correct = 0

        self.model.eval()

        with torch.no_grad():
            for images, labels in tqdm(test_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                all_ground_truths.append(labels)

                outputs = self.model(images)
                all_predictions_gpu.append(outputs)
                _, predicted = torch.max(outputs.data, 1)
                _, true_labels = torch.max(labels, 1)
                total += labels.size(0)
                correct += (predicted == true_labels).sum().item()

        accuracy = correct / total
        print(f'Accuracy: {accuracy * 100:.2f}%')

        all_predictions = torch.cat(all_predictions_gpu).cpu().numpy()
        all_ground_truths = torch.cat(all_ground_truths).cpu().numpy()

        return accuracy, all_ground_truths, all_predictions
    
    def get_classifier_output(self, data_loader):
        self.model.eval()
        all_predictions = []
        all_probabilities = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(data_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.append(predicted.cpu().numpy())
                all_probabilities.append(probabilities.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        all_predictions = np.concatenate(all_predictions)
        all_probabilities = np.concatenate(all_probabilities)
        all_labels = np.concatenate(all_labels)

        return all_labels, all_predictions, all_probabilities


    def compute_ovr_auc(self, ground_truths, predictions):
        # Binarize ground_truths for One-vs-Rest (OvR)
        ground_truths_bin = label_binarize(np.argmax(ground_truths, axis=1), classes=np.arange(6))
        predictions_prob = torch.softmax(torch.tensor(predictions), dim=1).numpy()

        ovr_auc_results = {}
        plt.figure(figsize=(6,6))
        for i, label in enumerate(self.labels):
            fpr, tpr, _ = roc_curve(ground_truths_bin[:, i], predictions_prob[:, i])
            auc_ovr = auc(fpr, tpr)
            ovr_auc_results[label] = auc_ovr
            print(f"OvR AUC for {label}: {auc_ovr:.4f}")

            # Plotting the ROC curve
            plt.plot(fpr,tpr,label=f'{label} (AUC = {auc_ovr:.4f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('One-vs-Rest ROC Curve')
        plt.legend(loc='lower right')
        plt.savefig('figures/ovr_auc_roc_{}.png'.format(batch_size), dpi=300)
        plt.close()

        # Save the results to a csv file
        df = pd.DataFrame(ovr_auc_results.items(), columns=['Hemorrhage Type', 'AUC'])
        df.to_csv('results/ovr_auc_{}.csv'.format(batch_size), index=False)

        print(f"One-vs-Rest AUC results saved to 'results/ovr_auc.csv'")
        
        pd.DataFrame(ovr_auc_results.items(), columns=['Hemorrhage Type', 'AUC']).to_csv('results/ovr_auc.csv', index=False)


    def compute_ovo_auc(self, ground_truths, predictions):
        ground_truths_bin = np.argmax(ground_truths, axis=1)
        predictions_prob = torch.softmax(torch.tensor(predictions), dim=1).numpy()

        ovo_auc_results = {}
        class_pairs = list(combinations(range(6), 2))

        plt.figure(figsize=(6,6))
        for pair in class_pairs:
            # Select only the samples where the ground truth is one of the two classes
            pair_idx = np.where(np.isin(ground_truths_bin, pair))[0]
            if len(pair_idx) == 0:
                continue
            binary_gt = (ground_truths_bin[pair_idx] == pair[1]).astype(int)
            binary_pred = predictions_prob[pair_idx][:, pair[1]]  # Use softmax probabilities

            fpr, tpr, _ = roc_curve(binary_gt, binary_pred)
            auc_ovo = auc(fpr, tpr)
            ovo_auc_results[f"{self.labels[pair[0]]}_vs_{self.labels[pair[1]]}"] = auc_ovo
            print(f"OvO AUC for {self.labels[pair[0]]} vs {self.labels[pair[1]]}: {auc_ovo:.4f}")

            # Plotting the ROC curve
            plt.plot(fpr,tpr, label=f'{self.labels[pair[0]]} vs {self.labels[pair[1]]} (AUC = {auc_ovo:.4f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('One-vs-One ROC Curve')
        plt.legend(loc='best', fontsize=5.8)
        plt.savefig('figures/ovo_auc_roc_{}.png'.format(batch_size), dpi=300)
        plt.close()

        # Save the results to a csv file
        df = pd.DataFrame(ovo_auc_results.items(), columns=['Hemorrhage Type Pair', 'AUC'])
        df.to_csv('results/ovo_auc_{}.csv'.format(batch_size), index=False)

        print(f"One-vs-One AUC results saved to 'results/ovo_auc.csv'")


    def print_evaluation(self, results):
        accuracy, ground_truths, predictions = results
        print(f'Overall Accuracy: {accuracy * 100:.2f}%')

        # Compute One-vs-Rest (OvR) AUC
        self.compute_ovr_auc(ground_truths, predictions)

        # Compute One-vs-One (OvO) AUC
        self.compute_ovo_auc(ground_truths, predictions)


def save_classifier(model, path='weights/supervised_classifier_resnet50_weights.pth'):
        # try:
        # Check if model is wrapped in DataParallel
        #     if isinstance(model, torch.nn.DataParallel):
        #         # Save state_dict from the underlying module
        #         state_dict = model.module.state_dict()
        #     else:
        #         # Save state_dict from the model directly
        #         state_dict = model.state_dict()
            
        #     # Save the state_dict to the specified path
        #     torch.save(state_dict, path)
        #     print(f"Model weights saved successfully to {path}")
    
        # except Exception as e:

        #     print(f"Error saving model weights: {e}")
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)
    
def load_classifier(model, path='weights/supervised_classifier_resnet50_weights.pth'):
    # try: 
    #     state_dict = torch.load(path, weights_only=True)
    #     print("State dict loaded successfully.")

    #     print("State dict keys:", state_dict.keys())

    #     if isinstance(model, torch.nn.DataParallel):
    #         model.module

    #     model.load_state_dict(state_dict, strict=False)

    #     print("Model weights loaded successfully.")
    #     return model
    
    # except Exception as e:
    #     print("Error loading model weights:", e)
    
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(torch.load(path))
    else:
        model.load_state_dict(torch.load(path))
    return model

# Example usage
if __name__ == "__main__":
    train_flag = False
    load_flag = True
    multiGPU_flag = True
    device_ids = [0,1]
    batch_size = 128
    num_epochs = 100
    num_iterations_train = 10
    num_iterations_val = 1
    lr = 1e-4

    from step0_common_info import dicom_dir
    
    train_dataset = RSNA_Intracranial_Hemorrhage_Dataset(
            'data/metadata_training.csv',
            dicom_dir)
    print('Train dataset size:', len(train_dataset))

    val_dataset = RSNA_Intracranial_Hemorrhage_Dataset(
            'data/metadata_validation.csv',
            dicom_dir)
    print('Validation dataset size:', len(val_dataset))
    
    test_dataset = RSNA_Intracranial_Hemorrhage_Dataset(
            'data/metadata_evaluation.csv',
            dicom_dir)
    print('Test dataset size:', len(test_dataset))

    # print("Train class distribution:", train_dataset.metadata['subdural'].sum())
    # print("Val class distribution:", val_dataset.metadata['subdural'].sum())
    # print("Test class distribution:", test_dataset.metadata['subdural'].sum())

    def compute_sample_weights(metadata, hemorrhage_types):
        class_counts = metadata[hemorrhage_types].sum(axis=0).to_numpy()
        class_weights = 1.0 / class_counts
        sample_weights_matrix = metadata[hemorrhage_types].to_numpy() * class_weights
        sample_weights = sample_weights_matrix.sum(axis=1)
        return sample_weights

    sample_weights = compute_sample_weights(train_dataset.metadata, train_dataset.hemorrhage_types)
    train_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dataset), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)

    sample_weights = compute_sample_weights(val_dataset.metadata, val_dataset.hemorrhage_types)
    val_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(val_dataset), replacement=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    observer = SupervisedClassifierObserver(verbose=True, batch_size=batch_size)

    if multiGPU_flag:
        observer.model = torch.nn.DataParallel(observer.model, device_ids=device_ids)
    

    if load_flag:
        try:
            observer.model = load_classifier(observer.model, 'weights/supervised_classifier_resnet50_weights.pth')
            print("Model weights loaded successfully.")
        except FileNotFoundError:
            print("Weights file not found. Training from scratch.")

    if train_flag:
        # Train the model
        observer.train(train_loader, val_loader=val_loader, test_loader=test_loader, num_epochs=num_epochs, num_iterations_train=num_iterations_train, num_iterations_val=num_iterations_val, lr=lr)

    

    # # Obtain classifier output for the test set
    all_labels, all_predictions, all_probabilities = observer.get_classifier_output(test_loader)

    print("Evaluating the model on the test set...")
    accuracy, ground_truths, predictions = observer.evaluate(test_loader)
    print(f"Final Test Accuracy: {accuracy * 100:.2f}%")
    
    # # Example of how to print or use these outputs
    # print("Classifier outputs:")
    # print("Labels: ", all_labels)
    # print("Predictions: ", all_predictions, 'Length:', len(all_predictions))
    # print("Probabilities: ", all_probabilities)