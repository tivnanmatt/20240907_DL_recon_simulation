# step5_deep_learning_reconstruction.py

import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from tqdm import tqdm
from step2_dataset_dataloader import RSNA_Intracranial_Hemorrhage_Dataset
from torch.utils.data import DataLoader, WeightedRandomSampler
from step4_iterative_reconstruction import HU_to_attenuation, attenuation_to_HU, CTProjector
from torch_ema import ExponentialMovingAverage
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from diffusers import UNet2DModel

import matplotlib.pyplot as plt  # Import matplotlib for plotting
import os  # For creating directories and saving files

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

import time
t0 = time.time()

class DeepLearningReconstructor(nn.Module):
    def __init__(self):
        super(DeepLearningReconstructor, self).__init__()

        # Simple CNN architecture as before
        # def conv_block(in_channels, out_channels):
        #     return nn.Sequential(
        #         nn.Conv2d(in_channels, out_channels, 3, padding=1),
        #         nn.BatchNorm2d(out_channels),
        #         nn.SiLU()
        #     )
        
        # self.model = nn.Sequential(
        #     conv_block(1, 64),
        #     conv_block(64, 128),
        #     conv_block(128, 64),
        #     conv_block(64, 32),
        #     nn.Conv2d(32, 1, 3, padding=1)
        # )

        # self.model = nn.Sequential(
        #     conv_block(1, 16),
        #     conv_block(16, 16),
        #     nn.Conv2d(16, 1, 3, padding=1)
        # )

        # block_out_channels = (128, 256, 512, 1024)
        block_out_channels = (32, 64, 128, 256)
        # block_out_channels = (16, 32, 64, 128)
        # block_out_channels = (4, 8, 16, 32)
        
        layers_per_block = 4
        # layers_per_block = 2



        # identity
        # self.head = nn.Identity() 

        self.unet = UNet2DModel(
            sample_size=None,
            in_channels=1,  # 32 components from the pseudo-inverse
            out_channels=1,  # Final reconstructed image
            center_input_sample=False,
            time_embedding_type='positional',
            freq_shift=0,
            flip_sin_to_cos=True,
            down_block_types=('DownBlock2D', 'DownBlock2D', 'AttnDownBlock2D', 'AttnDownBlock2D'),
            up_block_types=('UpBlock2D', 'UpBlock2D', 'AttnUpBlock2D', 'AttnUpBlock2D'),
            block_out_channels=block_out_channels   ,
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
            add_attention=True,
            class_embed_type=None,
            num_class_embeds=None,
            num_train_timesteps=None
        )

        class LambdaLayer(nn.Module):
            def __init__(self, lambd):
                super(LambdaLayer, self).__init__()
                self.lambd = lambd
            def forward(self, x):
                return self.lambd(x)

        # conv 32 -> 1
        # self.tail = nn.Conv2d(32, 1, 3, padding=1)


        self.model = nn.Sequential(
            LambdaLayer(lambda x: self.unet(x, torch.zeros(x.shape[0], device=x.device))[0]),
        )


    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.unet = self.unet.to(*args, **kwargs)
        self.model = self.model.to(*args, **kwargs)
        return self

    def forward(self, x_tilde):
        # x_hat = x_tilde + self.unet(x_tilde)
        # t = torch.zeros(x_tilde.shape[0], device=x_tilde.device)
        # x_hat = x_tilde + self.unet(x_tilde,t)[0]
        # x_hat = self.model(x_tilde)
        x_hat = x_tilde + self.model(x_tilde)
        return x_hat
        

def train_model(projector, 
                reconstructor, 
                train_loader, 
                val_loader=None, 
                num_epochs=100, 
                num_iterations_train=100, 
                num_iterations_val=10, 
                lr=1e-4,
                patch_size = 256, 
                device=device):
    assert isinstance(projector, CTProjector) or (isinstance(projector, torch.nn.DataParallel) and isinstance(projector.module, CTProjector))
    assert isinstance(reconstructor, DeepLearningReconstructor) or (isinstance(reconstructor, torch.nn.DataParallel) and isinstance(reconstructor.module, DeepLearningReconstructor))

    optimizer = Adam(reconstructor.parameters(), lr=lr)

    # If the optimizer was saved, load it
    try:
        optimizer.load_state_dict(torch.load('weights/deep_learning_reconstructor_optimizer.pth'))
        print("Optimizer loaded successfully.")
    except FileNotFoundError:
        print("No optimizer state found. Starting from scratch.")

    ema = ExponentialMovingAverage(reconstructor.parameters(), decay=0.995)
    criterion = nn.MSELoss()

    train_loader_iter = iter(train_loader)
    
    for epoch in range(num_epochs):
        reconstructor.train()
        train_loss = 0

        for _ in tqdm(range(num_iterations_train)):
            try:
                phantom, _ = next(train_loader_iter)
            except StopIteration:
                train_loader_iter = iter(train_loader)
                phantom, _ = next(train_loader_iter)

            phantom = phantom.to(device).float()

            brain_mask = torch.logical_and(phantom > 0.0, phantom < 80.0)

            phantom = HU_to_attenuation(phantom)

            # Simulate forward projection and sinogram with Poisson noise
            if isinstance(projector, torch.nn.DataParallel):
                sinogram = projector.module.forward_project(phantom)
            else:
                sinogram = projector.forward_project(phantom)
            
            # I0 = 1e10
            # photon_counts = I0 * torch.exp(-sinogram)
            # photon_counts = torch.poisson(photon_counts)
            # noisy_sinogram = -torch.log((photon_counts + 1) / I0)

            # Forward pass: reconstruct using U-Net
            optimizer.zero_grad()

            # Step 1: Get components from the pseudo-inverse reconstruction
            if isinstance(projector, torch.nn.DataParallel):
                x_tilde_components = projector.module.pseudoinverse_reconstruction(sinogram, projector.module.singular_values_list)
            else:
                x_tilde_components = projector.pseudoinverse_reconstruction(sinogram, projector.singular_values_list)
            
            pseudoinverse = torch.sum(x_tilde_components, dim=1, keepdim=True)
            # reconstruction = reconstructor(pseudoinverse)
            
            pseudoinverse_patches = torch.zeros(pseudoinverse.shape[0], 1, patch_size, patch_size, device=device)
            phantom_patches = torch.zeros(pseudoinverse.shape[0], 1, patch_size, patch_size, device=device)
            brain_mask_patches = torch.zeros(pseudoinverse.shape[0], 1, patch_size, patch_size, dtype=torch.bool, device=device)
            for i in range(pseudoinverse.shape[0]):
                # iRow = np.random.randint(64, 256-128)
                # iCol = np.random.randint(64, 256-128)
                if patch_size == 256:
                    iRow = 0
                    iCol = 0
                else:
                    iRow = np.random.randint(0, 256-patch_size)
                    iCol = np.random.randint(0, 256-patch_size)
                pseudoinverse_patches[i] = pseudoinverse[i, :, iRow:iRow+patch_size, iCol:iCol+patch_size]
                phantom_patches[i] = phantom[i, :, iRow:iRow+patch_size, iCol:iCol+patch_size]
                brain_mask_patches[i] = brain_mask[i, :, iRow:iRow+patch_size, iCol:iCol+patch_size]
            
            reconstruction_patches = reconstructor(pseudoinverse_patches)

            phantom = attenuation_to_HU(phantom)
            reconstruction_patches = attenuation_to_HU(reconstruction_patches)
            pseudoinverse = attenuation_to_HU(pseudoinverse)

            phantom_patches = attenuation_to_HU(phantom_patches)
            pseudoinverse_patches = attenuation_to_HU(pseudoinverse_patches)

            # Calculate MSE loss
            brain_weight = 0.99
            loss = (1 - brain_weight) * criterion(reconstruction_patches, phantom_patches)
            if torch.any(brain_mask):
                loss += brain_weight * criterion(reconstruction_patches[brain_mask_patches], phantom_patches[brain_mask_patches])
            loss.backward()
            optimizer.step()
            ema.update()

            train_loss += loss.item()

        # Report RMSE
        print(f'Epoch {epoch + 1}/{num_epochs}, Training RMSE (HU): {np.sqrt(train_loss / num_iterations_train)}')

        if val_loader:
            reconstructor.eval()
            val_loss = 0
            val_loader_iter = iter(val_loader)

            with torch.no_grad():
                for _ in tqdm(range(num_iterations_val)):
                    try:
                        phantom, _ = next(val_loader_iter)
                    except StopIteration:
                        val_loader_iter = iter(val_loader)
                        phantom, _ = next(val_loader_iter)

                    phantom = phantom.to(device).float()

                    brain_mask = torch.logical_and(phantom > 0.0, phantom < 80.0)

                    phantom = HU_to_attenuation(phantom)

                    if isinstance(projector, torch.nn.DataParallel):
                        sinogram = projector.module.forward_project(phantom)
                    else:
                        sinogram = projector.forward_project(phantom)
                    
                    # I0 = 1e10
                    # photon_counts = I0 * torch.exp(-sinogram)
                    # photon_counts = torch.poisson(photon_counts)
                    # noisy_sinogram = -torch.log((photon_counts + 1) / I0)

                    if isinstance(projector, torch.nn.DataParallel):
                        x_tilde_components = projector.module.pseudoinverse_reconstruction(sinogram, projector.module.singular_values_list)
                    else:
                        x_tilde_components = projector.pseudoinverse_reconstruction(sinogram, projector.singular_values_list)
                    
                    pseudoinverse = torch.sum(x_tilde_components, dim=1, keepdim=True)
                    # reconstruction = reconstructor(pseudoinverse)
                    reconstruction = reconstructor(phantom)

                    phantom = attenuation_to_HU(phantom)
                    reconstruction = attenuation_to_HU(reconstruction)
                    pseudoinverse = attenuation_to_HU(pseudoinverse)

                    brain_weight = 0.99
                    loss = (1 - brain_weight) * criterion(reconstruction, phantom)
                    if torch.any(brain_mask):
                        loss += brain_weight * criterion(reconstruction[brain_mask], phantom[brain_mask])
                    
                    val_loss += loss.item()
                    
            print(f'Validation RMSE (HU): {np.sqrt(val_loss / num_iterations_val)}')

        # Save the model after each epoch
        # save_reconstructor(reconstructor, 'weights/deep_learning_reconstructor.pth')

def save_reconstructor(reconstructor, filename):
    if isinstance(reconstructor, torch.nn.DataParallel):
        torch.save(reconstructor.module.state_dict(), filename)
    else:
        torch.save(reconstructor.state_dict(), filename)

def load_reconstructor(reconstructor, filename):
    if isinstance(reconstructor, torch.nn.DataParallel):
        reconstructor.module.load_state_dict(torch.load(filename))
    else:
        reconstructor.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))

# Add the evaluation function
def evaluate_reconstructor(projector, reconstructor, test_loader, num_iterations=1, device=device):
    reconstructor.eval()
    test_loader_iter = iter(test_loader)
    
    # Create the figures directory if it doesn't exist
    os.makedirs('figures', exist_ok=True)
    
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

        # Simulate forward projection and sinogram with Poisson noise
        # I0 = 1e10
        # t0 = time.time()
        sinogram = projector.forward_project(phantom)
        # photon_counts = I0 * torch.exp(-sinogram)
        # photon_counts = torch.poisson(photon_counts)
        # sinogram_noisy = -torch.log((photon_counts + 1) / I0)

        t1 = time.time()
        print(f'Elapsed time to forward project = {t1 - t0:.4f}s')

        # Pseudo-inverse reconstruction
        t0 = time.time()
        x_tilde_components = projector.pseudoinverse_reconstruction(sinogram, projector.singular_values_list)
        pseudoinverse = torch.sum(x_tilde_components, dim=1, keepdim=True)
        t1 = time.time()
        print(f'Elapsed time to pseudo-inverse reconstruct = {t1 - t0:.4f}s')

        # Deep Learning Reconstruction
        t0 = time.time()
        # reconstruction = reconstructor(phantom)
        reconstruction = reconstructor(pseudoinverse)
        t1 = time.time()
        print(f'Elapsed time for deep learning reconstruction = {t1 - t0:.4f}s')

        # Convert to HU units for visualization
        phantom_HU = attenuation_to_HU(phantom)
        pseudoinverse_HU = attenuation_to_HU(pseudoinverse)
        reconstruction_HU = attenuation_to_HU(reconstruction)

        # Save figures
        plot_reconstructions(
            vmin=0.0,
            vmax=80.0,
            filename=f'DLR_batch_{i}_brain.png',
            phantom=phantom_HU,
            sinogram=sinogram,
            pinv_reconstruction=pseudoinverse_HU,
            reconstruction=reconstruction_HU
        )
        plot_reconstructions(
            vmin=-1000.0,
            vmax=2000.0,
            filename=f'DLR_batch_{i}_bone.png',
            phantom=phantom_HU,
            sinogram=sinogram,
            pinv_reconstruction=pseudoinverse_HU,
            reconstruction=reconstruction_HU
        )

def plot_reconstructions(vmin, vmax, filename, phantom, sinogram, pinv_reconstruction, reconstruction):
    # This function is similar to the one in step4_iterative_reconstruction.py
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
    load_flag = True
    multiGPU_flag = False  # Set to True if using multiple GPUs
    device_ids = [0,1,2]
    batch_size = 4
    num_epochs = 100  # Adjust as needed
    num_iterations_train = 50
    num_iterations_val = 1
    num_iterations_test = 10  # Number of test images to process during evaluation
    patch_size = 256

    from step0_common_info import dicom_dir
    
    train_dataset = RSNA_Intracranial_Hemorrhage_Dataset(
            'data/metadata_training.csv',
            dicom_dir)
    
    val_dataset = RSNA_Intracranial_Hemorrhage_Dataset(
            'data/metadata_validation.csv',
            dicom_dir)
    
    test_dataset = RSNA_Intracranial_Hemorrhage_Dataset(
            'data/metadata_evaluation.csv',
            dicom_dir)

    def compute_sample_weights(metadata, hemorrhage_types):
        class_counts = metadata[hemorrhage_types].sum(axis=0).to_numpy()
        class_weights = 1.0 / class_counts
        sample_weights_matrix = metadata[hemorrhage_types].to_numpy() * class_weights
        sample_weights = sample_weights_matrix.sum(axis=1)
        return sample_weights

    sample_weights = compute_sample_weights(train_dataset.metadata, train_dataset.hemorrhage_types)
    train_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dataset), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=16)

    sample_weights = compute_sample_weights(val_dataset.metadata, val_dataset.hemorrhage_types)
    val_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(val_dataset), replacement=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)  # Batch size of 1 for evaluation

    projector = CTProjector().to(device)
    reconstructor = DeepLearningReconstructor().to(device)

    if multiGPU_flag:
        projector = torch.nn.DataParallel(projector, device_ids=device_ids)
        reconstructor = torch.nn.DataParallel(reconstructor, device_ids=device_ids)

    if load_flag:
        try:
            print("Loading pre-trained reconstructor weights.")
            load_reconstructor(reconstructor, 'weights/deep_learning_reconstructor.pth')
            print("Model loaded successfully.")
        except FileNotFoundError:
            print("No pre-trained model found. Starting from scratch.")

    if train_flag:
        train_model(
            projector, 
            reconstructor, 
            train_loader, 
            val_loader=None, 
            num_epochs=num_epochs, 
            num_iterations_train=num_iterations_train,
            num_iterations_val=num_iterations_val,
            lr=1e-4, 
            patch_size = patch_size,
            device=device
        )

        save_reconstructor(reconstructor, 'weights/deep_learning_reconstructor.pth')


    if evaluate_flag:
        # Evaluate the reconstructor
        evaluate_reconstructor(
            projector, 
            reconstructor, 
            test_loader, 
            num_iterations=num_iterations_test, 
            device=device
        )

if __name__ == "__main__":
    main()

print(f"Time taken: {time.time() - t0:.2f} seconds")
