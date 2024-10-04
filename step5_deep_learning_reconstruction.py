import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from tqdm import tqdm
from step2_dataset_dataloader import RSNA_Intracranial_Hemorrhage_Dataset
from torch.utils.data import DataLoader, WeightedRandomSampler
from step4_iterative_reconstruction import HU_to_attenuation, attenuation_to_HU, CTProjector
from torch_ema import ExponentialMovingAverage
from diffusers import UNet2DModel
import matplotlib.pyplot as plt
import os
import time

# Device handling function (CPU/GPU)
def get_device(device_input):
    if isinstance(device_input, list):
        device_ids = device_input
        device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")
    else:
        device_ids = None
        device = torch.device(device_input)
    return device, device_ids


# Checking the device of a tensor
def check_tensor_device(tensor, name="Tensor"):
    print(f"{name} is on {tensor.device}")

# Updated DeepLearningReconstructor with registered buffer
class DeepLearningReconstructor(nn.Module):
    def __init__(self):
        super(DeepLearningReconstructor, self).__init__()

        # Model architecture parameters (UNet2D)
        block_out_channels = (32, 64, 128, 256)
        layers_per_block = 4

        self.unet = UNet2DModel(
            sample_size=None,
            in_channels=32, # input channel dimension
            out_channels=1, # output single channel
            center_input_sample=False, 
            time_embedding_type='positional',
            freq_shift=0,
            flip_sin_to_cos=True,
            down_block_types=('DownBlock2D', 'DownBlock2D', 'AttnDownBlock2D', 'AttnDownBlock2D'),
            up_block_types=('UpBlock2D', 'UpBlock2D', 'AttnUpBlock2D', 'AttnUpBlock2D'),
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            mid_block_scale_factor=1,
            downsample_padding=1,
            downsample_type='conv',
            upsample_type='conv',
            dropout=0.0,
            act_fn='silu',
            attention_head_dim=None,
            norm_num_groups=4,
            attn_norm_num_groups=None,
            norm_eps=1e-05,
            resnet_time_scale_shift='default',
            add_attention=True, # Add attention mechanism in Unet
            class_embed_type=None,
            num_class_embeds=None,
            num_train_timesteps=None
        )

        # Register the singular values in pseudo-inverse reconstruction
        self.singular_values_list = torch.linspace(0, 3000, 32).long()

    # Forward pass through the model
    def forward(self, x_tilde):
        t = torch.zeros(x_tilde.shape[0], device=x_tilde.device) # Time input
        x_hat = self.unet(x_tilde, t)[0] # Run the Unet model
        return x_hat


# Loss closure module
class CTReconstructionLossClosure(nn.Module):
    def __init__(self, projector, reconstructor, patch_size=256, brain_weight=0.95):
        super(CTReconstructionLossClosure, self).__init__()
        self.projector = projector # Forward CT projector
        self.reconstructor = reconstructor # Deep learning reconstructor
        self.patch_size = patch_size # Size of image patches for processing
        self.brain_weight = brain_weight # Weight for brain region in loss
        self.criterion = nn.MSELoss() # Mean squared error loss

    def forward(self, phantom_batch):
        phantom = phantom_batch.float() # Ensure phantom is float
        batch_size = phantom.shape[0]

        device = phantom.device

        # Create brain mask to isolate relevant regions (0 - 80 HU)
        brain_mask = torch.logical_and(phantom > 0.0, phantom < 80.0)

        # Convert phantom to linear attenuation coefficients
        phantom = HU_to_attenuation(phantom)

        # Forward project the phantom to generate sinogram
        sinogram = self.projector.forward_project(phantom)

        singular_values_list = self.reconstructor.singular_values_list

        # Perform pseudo-inverse reconstruction using singular values
        x_tilde_components = self.projector.pseudoinverse_reconstruction(
            sinogram, singular_values_list
        )

        # Sum the components to obtain the pseudo-inverse reconstruction
        pseudoinverse = torch.sum(x_tilde_components, dim=1, keepdim=True)

        # Extract patches
        patch_size = self.patch_size
        pseudoinverse_patches = torch.zeros(batch_size, 32, patch_size, patch_size, device=device)
        phantom_patches = torch.zeros(batch_size, 1, patch_size, patch_size, device=device)
        brain_mask_patches = torch.zeros(batch_size, 1, patch_size, patch_size, dtype=torch.bool, device=device)

        # Extract patches of size patch_size for each image in the batch
        for i in range(batch_size):
            if patch_size == 256:
                iRow = 0
                iCol = 0
            else:
                iRow = np.random.randint(0, 256 - patch_size) # Random cropping
                iCol = np.random.randint(0, 256 - patch_size)
            pseudoinverse_patches[i] = x_tilde_components[i, :, iRow:iRow+patch_size, iCol:iCol+patch_size]
            phantom_patches[i] = phantom[i, :, iRow:iRow+patch_size, iCol:iCol+patch_size]
            brain_mask_patches[i] = brain_mask[i, :, iRow:iRow+patch_size, iCol:iCol+patch_size]

        # Reconstruct the patches using the deep learning reconstructor
        reconstruction_patches = self.reconstructor(pseudoinverse_patches)

        # Convert both reconstructed and ground truth to HU units for loss computation
        phantom_patches = attenuation_to_HU(phantom_patches)
        reconstruction_patches = attenuation_to_HU(reconstruction_patches)

        if patch_size < 256:
            # Extract center region
            # Crops the center region of the patches to avoid edge artifacts
            patch_margin = patch_size // 4
            patch_margin = np.clip(patch_margin, 0, 32) # Clip to prevent negative margins

            phantom_patches = phantom_patches[:, :, patch_margin:-patch_margin, patch_margin:-patch_margin]
            reconstruction_patches = reconstruction_patches[:, :, patch_margin:-patch_margin, patch_margin:-patch_margin]
            brain_mask_patches = brain_mask_patches[:, :, patch_margin:-patch_margin, patch_margin:-patch_margin]

        # Compute loss (weighted by brain regions and background regions)
        loss = (1 - self.brain_weight) * self.criterion(reconstruction_patches, phantom_patches)
        if torch.any(brain_mask_patches):
            loss += self.brain_weight * self.criterion(
                reconstruction_patches[brain_mask_patches],
                phantom_patches[brain_mask_patches]
            )

        return loss
    


# Training the deep learning reconstructor
def train_model(projector, 
                reconstructor, 
                train_loader, 
                val_loader=None, 
                num_epochs=100, 
                num_iterations_train=100, 
                num_iterations_val=10, 
                lr=1e-4,
                patch_size=256, 
                device_input='cuda'):
    
    device, device_ids = get_device(device_input)

    # If using multiple devices, set default device to device_ids[0]
    if device_ids:
        default_device = torch.device(f'cuda:{device_ids[0]}')
    else:
        default_device = device

    # Move projector and reconstructor to default device
    projector = projector.to(default_device)
    reconstructor = reconstructor.to(default_device)

    # Create loss closure and move it to default device
    loss_closure = CTReconstructionLossClosure(projector, reconstructor, patch_size=patch_size).to(default_device)

    # Wrap loss closure with DataParallel if multiple devices are specified
    if device_ids:
        loss_closure = torch.nn.DataParallel(loss_closure, device_ids=device_ids)

    optimizer = Adam(reconstructor.parameters(), lr=lr)

    # If the optimizer was saved, load it
    try:
        optimizer.load_state_dict(torch.load('weights/deep_learning_reconstructor_optimizer.pth'))
        print("Optimizer loaded successfully.")
    except FileNotFoundError:
        print("No optimizer state found. Starting from scratch.")

    # Exponential moving average for model parameters
    ema = ExponentialMovingAverage(reconstructor.parameters(), decay=0.95)

    # Convert the training loader into an iterator for dynamic fetching
    train_loader_iter = iter(train_loader)
    
    # Loop over epochs
    for epoch in range(num_epochs):
        reconstructor.train()   # Set the reconstructor to training mode
        train_loss = 0

        for _ in tqdm(range(num_iterations_train)):
            try:
                phantom_batch, _ = next(train_loader_iter) # Fetch the next batch
            except StopIteration:
                train_loader_iter = iter(train_loader)
                phantom_batch, _ = next(train_loader_iter)

            # Calculate the batch size
            batch_size = phantom_batch.shape[0]

            # Move phantom_batch to default device
            phantom_batch = phantom_batch.to(default_device)

            # Zero the gradients before each optimization step
            optimizer.zero_grad()

            # Compute the loss using the loss closure (reconstruction error)
            loss = loss_closure(phantom_batch)
            loss = loss.mean()  # In case of DataParallel
            loss.backward() # Backpropagate the loss
            optimizer.step() # Update model parameters
            ema.update() # Update EMA to smooth parameter updates

            train_loss += loss.item() # Accumulate the training loss for this batch

        # Report RMSE
        print(f'Epoch {epoch + 1}/{num_epochs}, Training RMSE (HU): {np.sqrt(train_loss / num_iterations_train)}')

        # Validation loop using loss closure
        if val_loader is not None:
            reconstructor.eval()
            val_loss = 0 
            val_loader_iter = iter(val_loader)

            with torch.no_grad(): # Disable gradient calculation for validation
                for _ in tqdm(range(num_iterations_val)):
                    try:
                        phantom_batch, _ = next(val_loader_iter)
                    except StopIteration:
                        val_loader_iter = iter(val_loader)
                        phantom_batch, _ = next(val_loader_iter)

                    batch_size = phantom_batch.shape[0]
                    phantom_batch = phantom_batch.to(default_device)

                    # Compute validation loss using the loss closure
                    loss = loss_closure(phantom_batch)
                    loss = loss.mean()  # In case of DataParallel
                    val_loss += loss.item()

            # Report Validation RMSE
            print(f'Validation RMSE (HU): {np.sqrt(val_loss / num_iterations_val)}')

        # Save the optimizer state after each epoch
        torch.save(optimizer.state_dict(), 'weights/deep_learning_reconstructor_optimizer.pth')

        # Save the model after each epoch
        save_reconstructor(reconstructor, 'weights/deep_learning_reconstructor.pth')

# Save the reconstructor model's state dict
def save_reconstructor(reconstructor, filename):
    if isinstance(reconstructor, torch.nn.DataParallel):
        torch.save(reconstructor.module.state_dict(), filename)
    else:
        torch.save(reconstructor.state_dict(), filename)

# Load the reconstructor model's state dict
def load_reconstructor(reconstructor, filename):
    if isinstance(reconstructor, torch.nn.DataParallel):
        reconstructor.module.load_state_dict(torch.load(filename))
    else:
        reconstructor.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))

# Evaluation function
def evaluate_reconstructor(projector, reconstructor, test_loader, num_iterations=1, device_input='cuda'):
    """
        Evaluate the reconstructor model on the test set and save the results.
    
    Args:
        projector: Object representing the CT projector (used for generating sinograms).
        reconstructor: Deep learning reconstructor model to be evaluated.
        test_loader: DataLoader for the test dataset.
        num_iterations: Number of test images to process (default is 1).
        device_input: Device configuration (either 'cuda' or 'cpu').
    """
    
    device, device_ids = get_device(device_input)
    reconstructor.eval()
    test_loader_iter = iter(test_loader)


    # Create the figures directory if it doesn't exist
    os.makedirs('figures', exist_ok=True)

    # Loop through the test images
    for i in tqdm(range(num_iterations)):
        try:
            phantom, _ = next(test_loader_iter)
        except StopIteration:
            test_loader_iter = iter(test_loader)
            phantom, _ = next(test_loader_iter)

        phantom = phantom.to(device).float()

        brain_mask = torch.logical_and(phantom > 0.0, phantom < 80.0)

        phantom = HU_to_attenuation(phantom)

        t0 = time.time()

        sinogram = projector.forward_project(phantom)

        t1 = time.time()
        print(f'Elapsed time to forward project = {t1 - t0:.4f}s')

        t0 = time.time()
        x_tilde_components = projector.pseudoinverse_reconstruction(
            sinogram, reconstructor.singular_values_list.to(device)
        )
        pseudoinverse = torch.sum(x_tilde_components, dim=1, keepdim=True)
        t1 = time.time()
        print(f'Elapsed time to pseudo-inverse reconstruct = {t1 - t0:.4f}s')

        t0 = time.time()
        reconstruction = reconstructor(x_tilde_components)
        t1 = time.time()
        print(f'Elapsed time for deep learning reconstruction = {t1 - t0:.4f}s')

        # Convert to HU units for visualization
        phantom_HU = attenuation_to_HU(phantom)
        pseudoinverse_HU = attenuation_to_HU(pseudoinverse)
        reconstruction_HU = attenuation_to_HU(reconstruction)

        # Save brain window figures
        plot_reconstructions(
            vmin=0.0,
            vmax=80.0,
            filename=f'DLR_batch_{i}_brain.png',
            phantom=phantom_HU,
            sinogram=sinogram,
            pinv_reconstruction=pseudoinverse_HU,
            reconstruction=reconstruction_HU
        )
        # Save bone window figures
        plot_reconstructions(
            vmin=-1000.0,
            vmax=2000.0,
            filename=f'DLR_batch_{i}_bone.png',
            phantom=phantom_HU,
            sinogram=sinogram,
            pinv_reconstruction=pseudoinverse_HU,
            reconstruction=reconstruction_HU
        )

# Visualization function of reconstructed images: phantom, sinogram, pseudo-inverse, and deep learning
def plot_reconstructions(vmin, vmax, filename, phantom, sinogram, pinv_reconstruction, reconstruction):
    plt.figure(figsize=(24, 6))
    plt.subplot(1, 4, 1)
    plt.imshow(phantom.cpu().numpy()[0, 0, :, :], cmap='gray', vmin=vmin, vmax=vmax)
    plt.title('Phantom')
    plt.axis('off')
    plt.subplot(1, 4, 2)
    plt.imshow(sinogram.cpu().numpy()[0, 0, :, :], cmap='gray')
    plt.gca().set_aspect('auto')
    plt.title('Sinogram')
    plt.axis('off')
    plt.subplot(1, 4, 3)
    plt.imshow(pinv_reconstruction.detach().cpu().numpy()[0, 0, :, :], cmap='gray', vmin=vmin, vmax=vmax)
    plt.title('Pseudo-inverse Reconstruction')
    plt.axis('off')
    plt.subplot(1, 4, 4)
    plt.imshow(reconstruction.detach().cpu().numpy()[0, 0, :, :], cmap='gray', vmin=vmin, vmax=vmax)
    plt.title('Deep Learning Reconstruction')
    plt.axis('off')
    plt.savefig(f'./figures/{filename}', dpi=300)
    plt.close('all')

# Main script
def main():

    train_flag = True  # Set to True to enable training
    evaluate_flag = True  # Set to True to run evaluation after training
    load_flag = True # Load pre-trained model if available
    device_input = [0, 1, 2, 3]  # For multi-GPU
    # device_input = 'cuda'  # For single GPU
    batch_size = 16
    num_epochs = 5  # Adjust as needed
    num_iterations_train = 100
    num_iterations_val = 1
    num_iterations_test = 10  # Number of test images to process during evaluation
    patch_size = 256

    device, device_ids = get_device(device_input)

    from step0_common_info import dicom_dir

    train_dataset = RSNA_Intracranial_Hemorrhage_Dataset(
        'data/metadata_training.csv',
        dicom_dir
    )

    val_dataset = RSNA_Intracranial_Hemorrhage_Dataset(
        'data/metadata_validation.csv',
        dicom_dir
    )

    test_dataset = RSNA_Intracranial_Hemorrhage_Dataset(
        'data/metadata_evaluation.csv',
        dicom_dir
    )

    # Fuction to compute sample weights for handling class imbalance in hemorrhage types
    def compute_sample_weights(metadata, hemorrhage_types):
        class_counts = metadata[hemorrhage_types].sum(axis=0).to_numpy() # Count the number of samples per hemorrhage type
        class_weights = 1.0 / class_counts # Compute weights as the inverse of the class counts
        sample_weights_matrix = metadata[hemorrhage_types].to_numpy() * class_weights # weight each sample by the class weights
        sample_weights = sample_weights_matrix.sum(axis=1) # Sum the weights across hemorrhage types
        return sample_weights

    # Compute and assign sample weights for the training dataset
    sample_weights = compute_sample_weights(train_dataset.metadata, train_dataset.hemorrhage_types)
    train_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dataset), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=32)

    # Compute and assign sample weights for the validation dataset
    sample_weights = compute_sample_weights(val_dataset.metadata, val_dataset.hemorrhage_types)
    val_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(val_dataset), replacement=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)

    # Test loader does not need weighted sampling, just shuffle the data for evaluation
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    projector = CTProjector()
    reconstructor = DeepLearningReconstructor()

    if load_flag:
        try:
            print("Loading pre-trained reconstructor weights.")
            load_reconstructor(reconstructor, 'weights/deep_learning_reconstructor.pth')
            print("Model loaded successfully.")
        except FileNotFoundError:
            print("No pre-trained model found. Starting from scratch.")

    if train_flag:
        num_epochs_per_save = 20
        _num_run = num_epochs // num_epochs_per_save
        _num_run += 1 if num_epochs % num_epochs_per_save > 0 else 0
        for i in range(_num_run):
            _num_epochs = num_epochs_per_save if (num_epochs - i * num_epochs_per_save) >= num_epochs_per_save else num_epochs - i * num_epochs_per_save
            train_model(
                projector,
                reconstructor,
                train_loader,
                val_loader=None,
                num_epochs=_num_epochs,
                num_iterations_train=num_iterations_train,
                num_iterations_val=num_iterations_val,
                lr=1e-4,
                patch_size=patch_size,
                device_input=device_input
            )

            save_reconstructor(reconstructor, 'weights/deep_learning_reconstructor.pth')

            if evaluate_flag:
                # Evaluate the reconstructor
                evaluate_reconstructor(
                    projector,
                    reconstructor,
                    test_loader,
                    num_iterations=num_iterations_test,
                    device_input=device_input
                )

if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Time taken: {time.time() - t0:.2f} seconds")







