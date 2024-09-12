import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
# from nih_chest_xray_reader import NIHChestXrayDataset
from step2_dataset_dataloader import RSNA_Intracranial_Hemorrhage_Dataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

# Notes to self:
# - There is a significant class imbalance among various forms of hemmorhages. Class weights should be 
#   employed to compensate for this mismatch and to allow the neural network to train more aggressively 
#   on infrequently observed hemorrhages. Another option would be construct a dataset with equal sizes
# - Changed it to a multi-class classification to simplify the task. This requires to adjust:
#   . Remove Sigmoid activation
#   . Change BCELoss() to CrossEntropyLosS()
#   . Remove label float conversion and make integers
#   . Use precision/accuracy or recall as evaluation?

class SupervisedClassifier(nn.Module):
    def __init__(self):
        super(SupervisedClassifier, self).__init__()
        # Define a deeper CNN model for multi-class classification with more channels, BatchNorm, and PReLU
        
        dropout_rate = 0.5

        self.conv_layers = nn.Sequential(
            # First block
            nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=2),  # Down convolution with stride
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),  # Down convolution with stride
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Dropout(dropout_rate),

            
            # Second block
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),  # Down convolution with stride
            nn.BatchNorm2d(256),
            nn.PReLU(),

            # Third block
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2),  # Down convolution with stride
            nn.BatchNorm2d(512),
            nn.PReLU(),
            
            # Fourth block
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.PReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.PReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.PReLU(),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1, stride=2),  # Down convolution with stride
            nn.BatchNorm2d(1024),
            nn.PReLU()
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(1024 * 16 * 16, 1024), 
            nn.BatchNorm1d(1024),          
            nn.PReLU(),
            nn.Linear(1024, 512),          
            nn.BatchNorm1d(512),          
            nn.PReLU(),
            nn.Linear(512, 6),  
            # nn.Sigmoid() 
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc_layers(x)
        return x

class SupervisedClassifierObserver:
    def __init__(self, device=None, verbose=False, batch_size=32):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SupervisedClassifier().to(self.device)
        self.verbose = verbose
        self.batch_size = batch_size
        self.labels = ['no_hemorrhage', 'epidural' , 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural'] 

    def train(self, train_dataset, val_dataset=None, verbose=False, num_epochs=20, num_iterations_train=100, num_iterations_val=10):
        # Use Binary Cross-Entropy loss for multi-label classification
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=2e-4)

        # Do not use num_workers yet, was 16
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True) if val_dataset else None

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
                labels = labels.to(self.device).long()

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch {epoch + 1}, Training Loss: {running_loss / num_iterations_train}")

            if val_loader:
                val_loss = self.validate(val_loader, num_iterations_val=num_iterations_val)
                print(f"Validation Loss after Epoch {epoch + 1}: {val_loss}")

            # Save the trained model weights
            torch.save(self.model.state_dict(), 'supervised_classifier_weights.pth')

    def validate(self, eval_loader, num_iterations_val=10):
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss() # Switch to cross-entropy-loss
        self.model.eval()

        eval_loader_iter = iter(eval_loader)
        with torch.no_grad():
            for _ in tqdm(range(num_iterations_val)):
                try:
                    images, labels = next(eval_loader_iter)
                except StopIteration:
                    break

                images = images.to(self.device)
                labels = labels.to(self.device).float()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

        avg_loss = total_loss / num_iterations_val
        return avg_loss

    def evaluate(self, eval_dataset, num_patients=256):
        eval_loader = DataLoader(eval_dataset, batch_size=self.batch_size, shuffle=False) # num_workers=16
        eval_loader_iter = iter(eval_loader)

        all_ground_truths = []
        all_predictions_gpu = []

        self.model.eval()
        with torch.no_grad():
            for _ in tqdm(range(num_patients // self.batch_size)):
                images, labels = next(eval_loader_iter)
                images = images.to(self.device)
                labels = labels.to(self.device).long()
                all_ground_truths.append(labels)

                outputs = self.model(images)  # Keep outputs on GPU for now
                all_predictions_gpu.append(outputs)
                _, predicted = torch.max(outputs.data, 1)  # Get the index of the max log-probability
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f'Accuracy: {accuracy * 100:.2f}%')

        all_predictions = torch.cat(all_predictions_gpu).cpu().numpy()  # Convert predictions to numpy
        all_ground_truths = torch.cat(all_ground_truths).cpu().numpy()  # Convert ground truths to numpy

        results = np.zeros((len(self.labels), 5))  # For TP, FP, FN, TN, AUC
        for i, label in enumerate(self.labels):
            y_true = all_ground_truths[:, i]
            y_scores = all_predictions[:, i]

            if len(np.unique(y_true)) < 2:
                if self.verbose:
                    print(f"Skipping AUC calculation for '{label}' due to insufficient data.")
                auc = np.nan
            else:
                auc = roc_auc_score(y_true, y_scores)

            results[i, 4] = auc

            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            threshold = thresholds[np.argmax(tpr - fpr)]  # Optimal threshold based on Youden's J statistic
            predictions = (y_scores >= threshold).astype(int)

            results[i, 0] = np.sum((predictions == 1) & (y_true == 1))  # TP
            results[i, 1] = np.sum((predictions == 1) & (y_true == 0))  # FP
            results[i, 2] = np.sum((predictions == 0) & (y_true == 1))  # FN
            results[i, 3] = np.sum((predictions == 0) & (y_true == 0))  # TN

        return accuracy, results

    def print_evaluation(self, results, filename=None):
        if filename:
            np.savetxt(filename, results, fmt='%.4f', delimiter=',', header="TP,FP,FN,TN,AUC", comments='')

        print(f"{'Disease':<20}{'TP':<10}{'FP':<10}{'FN':<10}{'TN':<10}{'AUC':<10}")
        for i, label in enumerate(self.labels):
            tp, fp, fn, tn, auc = results[i]
            print(f"{label:<20}{int(tp):<10}{int(fp):<10}{int(fn):<10}{int(tn):<10}{auc:<10.4f}")


# Example usage
if __name__ == "__main__":
    train_flag = True
    load_flag = True
#  csv_file, dicom_dir, transform=None):
    if train_flag:

        full_dataset = RSNA_Intracranial_Hemorrhage_Dataset( 'data/stage_2_train_reformat.csv', 
                                                '/data/rsna-intracranial-hemorrhage-detection/stage_2_train/')



        # Get the indices for the dataset
        dataset_size = len(full_dataset)
        indices = list(range(dataset_size))

        # First split: train (70%), and the rest (30%) which will be split into val and eval
        train_indices, temp_indices = train_test_split(indices, test_size=0.3, random_state=42)

        # Second split: Split the temp (30%) into validation and test sets (15% each)
        val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)

        # Create subsets
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, test_indices)

        observer = SupervisedClassifierObserver(verbose=True, batch_size=64)

        if load_flag:
            try:
                observer.model.load_state_dict(torch.load('supervised_classifier_weights.pth'))
                print("Model weights loaded successfully.")
            except FileNotFoundError:
                print("Weights file not found. Training from scratch.")

        # Train the model with the validation dataset
        observer.train(train_dataset, val_dataset=val_dataset, num_epochs=1, num_iterations_train=100, num_iterations_val=10)

    observer = SupervisedClassifierObserver(verbose=True, batch_size=64)
    observer.model.load_state_dict(torch.load('supervised_classifier_weights.pth'))

    eval_dataset = Subset(full_dataset, val_indices)

    results = observer.evaluate(eval_dataset, num_patients=128)
    observer.print_evaluation(results, filename="supervised_classifier_results.csv")