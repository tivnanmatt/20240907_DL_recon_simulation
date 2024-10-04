import os
# visible devices 3
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from tqdm import tqdm
from step02_dataset_dataloader import RSNA_Intracranial_Hemorrhage_Dataset
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from step10_iterative_reconstruction import HU_to_attenuation, attenuation_to_HU, CTProjector, LinearLogLikelihood
from step12_diffusion_training import HU_to_SU, SU_to_HU, get_device
from torch_ema import ExponentialMovingAverage
from diffusers import UNet2DModel
import matplotlib.pyplot as plt
import time
import deepspeed

import torch.distributed as dist
import torch.multiprocessing as mp

from step12_diffusion_training import UnconditionalDiffusionModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PseudoinverseDiffusionBridge(UnconditionalDiffusionModel):
    def __init__(self, projector):
        super(PseudoinverseDiffusionBridge, self).__init__()

        assert isinstance(projector, CTProjector), 'projector must be an instance of CTProjector'
        self.projector = projector
        # self.projector.null_space = lambda x: self.projector.null_space(x, singular_values=[3000])

    def sample_x_t_given_x_0(self, x_0, t):
        x_t =  super(PseudoinverseDiffusionBridge, self).sample_x_t_given_x_0(x_0, t)
        x_t = x_0 + self.projector.null_space(x_t - x_0)
        return x_t
    
    def sample_x_t_plus_dt_given_x_t(self, x_t, t, dt):
        x_t_plus_dt = super(PseudoinverseDiffusionBridge, self).sample_x_t_plus_dt_given_x_t(x_t, t, dt)
        x_t_plus_dt = x_t + self.projector.null_space(x_t_plus_dt - x_t)
        return x_t_plus_dt
    
    def sample_x_t_plus_delta_t_given_x_t(self, x_t, t, delta_t):
        x_t_plus_delta_t = super(PseudoinverseDiffusionBridge, self).sample_x_t_plus_delta_t_given_x_t(x_t, t, delta_t)
        x_t_plus_delta_t = x_t + self.projector.null_space(x_t_plus_delta_t - x_t)
        return x_t_plus_delta_t
    
    def predict_x_0_given_x_t(self, x_t, t):
        x_0 = super(PseudoinverseDiffusionBridge, self).predict_x_0_given_x_t(x_t, t)
        # x_0 = x_t + self.projector.null_space(x_0 - x_t)
        return x_0        

    def predict_score_given_x_t(self, x_t, t):
        score = super(PseudoinverseDiffusionBridge, self).predict_score_given_x_t(x_t, t)
        # score = self.projector.null_space(score)
        return score
    
    def sample_x_t_minus_dt_given_x_t(self, x_t, t, dt, mode='ode'):
        x_t_minus_dt = super(PseudoinverseDiffusionBridge, self).sample_x_t_minus_dt_given_x_t(x_t, t, dt, mode)
        # x_t_minus_dt = x_t + self.projector.null_space(x_t_minus_dt - x_t)
        return x_t_minus_dt

def evaluate_diffusion_model(pseudoinverse_diffusion_bridge, test_loader, num_samples=1, noise_variance=1.0):
    assert isinstance(pseudoinverse_diffusion_bridge, PseudoinverseDiffusionBridge)
    assert isinstance(pseudoinverse_diffusion_bridge.projector, CTProjector)

    # measurements = None # Placeholder for measurements
    # linear_log_likelihood = LinearLogLikelihood(measurements, projector, noise_variance=noise_variance)

    # diffusion_model.eval()
    test_loader_iter = iter(test_loader)

    with torch.no_grad():
        for i in range(num_samples):
            x_0, _ = next(test_loader_iter)

            x_0 = x_0.to(device)
            x_0_HU = x_0.clone()  # Keep original for display

            # set the measurements for diffusion posterior sampling model
            sinogram = pseudoinverse_diffusion_bridge.projector.forward_project(HU_to_attenuation(x_0))

            # this is only for initialization, 
            # run the pseudoinverse on the sinogram,
            # then convert to SU so we can runn the diffusion forward process
            pseudoinverse = pseudoinverse_diffusion_bridge.projector.pseudoinverse_reconstruction(
                                                                        sinogram, 
                                                                        singular_values=[3000])
            pseudoinverse_HU = attenuation_to_HU(pseudoinverse)
            pseudoinverse_SU = HU_to_SU(pseudoinverse_HU)

            # also convert the original image to SU
            x_0 = HU_to_SU(x_0)

            # sample the forward diffusion process, p(x_t|x_0) at time t
            t = torch.tensor([1.0], device=device)
            x_t = pseudoinverse_diffusion_bridge.sample_x_t_given_x_0(pseudoinverse_SU, t)
            # x_t = pseudoinverse_diffusion_brdige.sample_x_t_given_x_0(x_0, t)

            num_steps = 128
            timesteps = (torch.linspace(1.0, 0.0, num_steps + 1).to(device)**2.0) * t.item()

            x_0_hat = pseudoinverse_diffusion_bridge.sample_x_0_given_x_t(x_t, 
                                                                          t, 
                                                                          mode='sde', 
                                                                          timesteps=timesteps)

            x_0_hat = SU_to_HU(x_0_hat)
            x_t = SU_to_HU(x_t)

            # Plotting
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))

            im0 = axs[0].imshow(x_0_HU[0, 0].cpu().numpy(), cmap='gray', vmin=-1000, vmax=2000)
            axs[0].set_title('Original Image')
            axs[0].axis('off')
            fig.colorbar(im0, ax=axs[0])

            im1 = axs[1].imshow(x_t[0, 0].cpu().numpy(), cmap='gray', vmin=-1000, vmax=2000)
            axs[1].set_title('Noisy Image at t=1.0')
            axs[1].axis('off')
            fig.colorbar(im1, ax=axs[1])

            im2 = axs[2].imshow(x_0_hat[0, 0].cpu().numpy(), cmap='gray', vmin=-1000, vmax=2000)
            axs[2].set_title('Reconstructed Image')
            axs[2].axis('off')
            fig.colorbar(im2, ax=axs[2])

            plt.savefig(f'./figures/PDB_batch_{i}_bone.png')

            
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))

            im0 = axs[0].imshow(x_0_HU[0, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=80)
            axs[0].set_title('Original Image')
            axs[0].axis('off')
            fig.colorbar(im0, ax=axs[0])

            im1 = axs[1].imshow(x_t[0, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=80)
            axs[1].set_title('Noisy Image at t=1.0')
            axs[1].axis('off')
            fig.colorbar(im1, ax=axs[1])

            im2 = axs[2].imshow(x_0_hat[0, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=80)
            axs[2].set_title('Reconstructed Image')
            axs[2].axis('off')
            fig.colorbar(im2, ax=axs[2])

            plt.savefig(f'./figures/PDB_batch_{i}_brain.png')

            plt.close('all')

class PDBLossClosure(nn.Module):
    def __init__(self, 
                 pseudoinverse_diffusion_bridge, 
                 patch_size=256, 
                 brain_weight=0.99, 
                 T=1.0):
        super(PDBLossClosure, self).__init__()
        self.pseudoinverse_diffusion_bridge = pseudoinverse_diffusion_bridge
        self.patch_size = patch_size
        self.brain_weight = brain_weight
        self.T = T
        self.criterion = nn.MSELoss()

    def forward(self, x_0):

        assert isinstance(self.pseudoinverse_diffusion_bridge, 
                          UnconditionalDiffusionModel), \
                        'diffusion_model must be an instance of UnconditionalDiffusionModel'

        assert isinstance(x_0, torch.Tensor), 'x_0 must be a torch.Tensor'
        assert len(x_0.shape) == 4, \
                'x_0 must be a 4D tensor with shape (batch_size, 1, patch_size, patch_size)'
        assert x_0.shape[1] == 1, 'x_0 must have 1 channel'

        x_0 = x_0.float()
        batch_size = x_0.shape[0]
        device = x_0.device

        brain_mask = torch.logical_and(x_0 > 0.0, x_0 < 80.0)

        x_0 = HU_to_SU(x_0)

        t = torch.rand(batch_size, 1, device=device)**2.0 * self.T

        x_t = self.pseudoinverse_diffusion_bridge.sample_x_t_given_x_0(x_0, t)

        # extract patches
        if self.patch_size < 256:
            seed = np.random.randint(0, 100000)
            x_0 = self.extract_patches(x_0, batch_size, device, seed)
            x_t = self.extract_patches(x_t, batch_size, device, seed)
            brain_mask = self.extract_patches(brain_mask, batch_size, device, seed).to(torch.bool)
            
        x_0_hat = self.pseudoinverse_diffusion_bridge.predict_x_0_given_x_t(x_t, t)

        x_0 = SU_to_HU(x_0)
        x_0_hat = SU_to_HU(x_0_hat)

        res = x_0 - x_0_hat

        # Extract center region
        patch_margin = self.patch_size // 4
        patch_margin = np.clip(patch_margin, 0, 32)
        x_0 = x_0[:, :, patch_margin:-patch_margin, patch_margin:-patch_margin]
        x_0_hat = x_0_hat[:, :, patch_margin:-patch_margin, patch_margin:-patch_margin]
        brain_mask = brain_mask[:, :, patch_margin:-patch_margin, patch_margin:-patch_margin]
        res = res[:, :, patch_margin:-patch_margin, patch_margin:-patch_margin]

        loss_weights = 1/(t + 1e-6)
        loss_weights = batch_size * loss_weights / loss_weights.sum()
        loss_weights = torch.sqrt(loss_weights).view(-1, 1, 1, 1)

        loss = (1 - self.brain_weight) * self.criterion(res * loss_weights, res * 0)
        if torch.any(brain_mask):
            loss += self.brain_weight * self.criterion((res * loss_weights)[brain_mask], res[brain_mask] * 0)

        return loss

    def extract_patches(self, data, batch_size, device, seed=None):
        if seed is not None:
            np.random.seed(seed)
        iRows = np.random.randint(0, 256 - self.patch_size, batch_size)
        iCols = np.random.randint(0, 256 - self.patch_size, batch_size)
        patch_size = self.patch_size
        patches = torch.zeros(batch_size, 1, patch_size, patch_size, device=device)
        for i in range(batch_size):
            iRow = iRows[i]
            iCol = iCols[i]
            patches[i] = data[i, :, iRow:iRow + patch_size, iCol:iCol + patch_size]
        return patches

def train_model(loss_closure, train_loader, num_epochs=100, num_iterations_train=100, lr=1e-4):
    optimizer = torch.optim.Adam(loss_closure.parameters(), lr=lr)
    ema = ExponentialMovingAverage(loss_closure.parameters(), decay=0.995)

    train_loader_iter = iter(train_loader)
    for epoch in range(num_epochs):
        loss_closure.train()
        train_loss = 0

        for _ in tqdm(range(num_iterations_train)):
            try:
                x_0, _ = next(train_loader_iter)
            except StopIteration:
                train_loader_iter = iter(train_loader)
                x_0, _ = next(train_loader_iter)

            x_0 = x_0.to(device)

            optimizer.zero_grad()
            loss = loss_closure(x_0).mean()  # Mean across GPUs if DataParallel
            loss.backward()
            optimizer.step()
            ema.update()

            train_loss += loss.item()

        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {np.sqrt(train_loss / num_iterations_train)}')

def train_diffusion_model_with_deepspeed(loss_closure, train_loader, num_epochs=100, num_iterations_train=100, lr=1e-4):
    # Set up DeepSpeed configuration
    deepspeed_config = {
        "train_batch_size": 1,  # Adjust as necessary
        "gradient_accumulation_steps": 1,
        "zero_optimization": {
            "stage": 2,  # Use ZeRO Stage 2
        },
    }

    # Initialize DeepSpeed
    model, optimizer, _, _ = deepspeed.initialize(
        model=loss_closure,
        optimizer=torch.optim.Adam(loss_closure.parameters(), lr=lr),
        config=deepspeed_config
    )

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for _ in tqdm(range(num_iterations_train)):
            x_0, _ = next(iter(train_loader))
            x_0 = x_0.to(model.device)

            loss = model(x_0)  # Forward pass
            model.backward(loss)  # Backward pass
            model.step()  # Update weights

            train_loss += loss.item()

        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss / num_iterations_train}')

def compute_sample_weights(metadata, hemorrhage_types):
    class_counts = metadata[hemorrhage_types].sum(axis=0).to_numpy()
    class_weights = 1.0 / class_counts
    sample_weights_matrix = metadata[hemorrhage_types].to_numpy() * class_weights
    sample_weights = sample_weights_matrix.sum(axis=1)
    return sample_weights

def save_diffusion_model_weights(diffusion_model, save_path):
    torch.save(diffusion_model.state_dict(), save_path)

def load_diffusion_model_weights(diffusion_model, load_path):
    state_dict = torch.load(load_path, map_location='cpu')
    diffusion_model.load_state_dict(state_dict)

def main():
    # print('AT LEAST I STARTED')
    train_flag = True
    load_flag = False
    use_deepspeed = False  # Set this flag to control whether to use DeepSpeed
    # device_input = [0,1,2,3] if use_deepspeed else [0]  # Use multiple GPUs for DeepSpeed, single GPU otherwise
    batch_size = 32
    num_epochs = 10
    num_iterations_train = 100
    num_iterations_val = 5
    lr = 1e-4
    patch_size = 64

    from step00_common_info import dicom_dir

    train_dataset = RSNA_Intracranial_Hemorrhage_Dataset(
            'data/metadata_training.csv',
            dicom_dir)
    
    val_dataset = RSNA_Intracranial_Hemorrhage_Dataset(
            'data/metadata_validation.csv',
            dicom_dir)
    
    test_dataset = RSNA_Intracranial_Hemorrhage_Dataset(
            'data/metadata_evaluation.csv',
            dicom_dir)

    sample_weights = compute_sample_weights(train_dataset.metadata, train_dataset.hemorrhage_types)
    train_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dataset), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=16)

    val_loader = None

    torch.manual_seed(1)
    inds = np.arange(len(test_dataset))
    test_dataset = Subset(test_dataset, inds)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    projector = CTProjector()
    pseudoinverse_diffusion_bridge = PseudoinverseDiffusionBridge(projector)
    loss_closure = PDBLossClosure(pseudoinverse_diffusion_bridge, patch_size)

    if load_flag:
        try:
            load_diffusion_model_weights(pseudoinverse_diffusion_bridge, 'weights/PDB_weights.pth')
            print("Diffusion model weights loaded successfully.")
        except:
            print("Diffusion model weights not found. Training from scratch.")

    if train_flag:
        if use_deepspeed:
            # Set MASTER_ADDR and MASTER_PORT programmatically
            # os.environ['MASTER_ADDR'] = 'localhost'  # Or replace with the master node's address
            # os.environ['MASTER_PORT'] = '12355'      # Some unused port on your machine
            # os.environ['WORLD_SIZE'] = str(len(device_input))

            # Train the model using DeepSpeed
            train_diffusion_model_with_deepspeed(loss_closure, train_loader, num_epochs, num_iterations_train, lr)

        else:

            pseudoinverse_diffusion_bridge = pseudoinverse_diffusion_bridge.to(device)
            # Train the model normally on a single GPU
            train_model(loss_closure, train_loader, num_epochs, num_iterations_train, lr)


    save_diffusion_model_weights(pseudoinverse_diffusion_bridge, 'weights/PDB_weights.pth')

    # Move the diffusion model to the device
    pseudoinverse_diffusion_bridge.to(device)
    pseudoinverse_diffusion_bridge.eval()
    # Evaluate the diffusion model
    evaluate_diffusion_model(pseudoinverse_diffusion_bridge, test_loader, num_samples=10)

if __name__ == "__main__":
    main()
