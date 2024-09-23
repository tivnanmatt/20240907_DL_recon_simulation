import torch
import torch.nn as nn
import numpy as np

from diffusers import UNet2DModel

from torch_ema import ExponentialMovingAverage

from tqdm import tqdm

from step2_dataset_dataloader import RSNA_Intracranial_Hemorrhage_Dataset

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import WeightedRandomSampler
from diffusers import AutoencoderKL, UNet2DConditionModel

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

import os

import matplotlib.pyplot as plt  # Import matplotlib for plotting

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cuda:2'

class LatentDiffusionModel(nn.Module):
    def __init__(self, log_likelihood, log_likelihood_weight=1.0):

        super(LatentDiffusionModel, self).__init__()

        block_out_channels = (16, 32, 64, 128)
        
        layers_per_block = 2

        self.unet = UNet2DModel(
            sample_size=64,
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
        self.log_likelihood = log_likelihood
        self.log_likelihood_weight = log_likelihood_weight

    def encode(self, x):
        # x = x.repeat(1, 4, 1, 1)
        #     return self.vae.encode(x).latent_dist.mean
        return x
    
    def decode(self, z):
        # x = self.vae.decode(z).sample
        # x = torch.mean(x, dim=1, keepdim=True)
        return z
    
    def sample_z_t_given_z_0(self, z_0, t):
        assert isinstance(t, torch.Tensor)
        return z_0 + torch.sqrt(t.view(-1,1,1,1)) * torch.randn_like(z_0)
    
    def sample_z_t_plus_dt_given_z_t(self, z_t, t, dt):
        assert isinstance(t, torch.Tensor)
        assert isinstance(dt, torch.Tensor)
        return z_t + torch.sqrt(dt) * torch.randn_like(z_t)
    
    def sample_z_t_plus_delta_t_given_z_t(self, z_t, t, delta_t):
        assert isinstance(t, torch.Tensor)
        return z_t + torch.sqrt(delta_t) * torch.randn_like(z_t)
    
    def predict_z_0_given_z_t(self, z_t, t):
        assert isinstance(t, torch.Tensor)
        # encoder_hidden_states = torch.zeros((z_t.size(0), 64, 768)).to(device)  # Adjust dimensions as needed
        # return z_t -  self.unet(z_t, t.view(-1))[0]* torch.sqrt(t.view(-1,1,1,1))
        # return z_t -  self.unet(z_t, t.view(-1))[0]
        
        sigma_data = 0.5
        def c_skip(t):
            return (sigma_data**2) / (t + sigma_data**2)
        
        def c_out(t):
            return sigma_data*torch.sqrt(t)/torch.sqrt(t + sigma_data**2)
        
        
        # return self.unet(z_t, t.view(-1))[0]
        return c_skip(t) * z_t + c_out(t) * self.unet(z_t, t.view(-1))[0]

    def predict_score_given_z_t(self, z_t, t):
        mu = self.predict_z_0_given_z_t(z_t, t)
        sigma2 = t
        return -(z_t - mu) / sigma2
    
    def sample_z_t_minus_dt_given_z_t(self, z_t, t, dt, mode='sde', y=None):
        assert mode in ['sde', 'ode']
        assert isinstance(t, torch.Tensor)
        assert isinstance(dt, torch.Tensor)

        prior_score = self.predict_score_given_z_t(z_t, t)
        
        if y is None:
            score = prior_score
        else:
            # z_0_hat = self.predict_z_0_given_z_t(z_t, t)
            # this will re run the network so lets just undo the transformation from score to mean
            sigma2 = t
            z_0_hat = z_t + sigma2 * prior_score
            x_0_hat = self.decode(z_0_hat)
            log_likelihood = self.log_likelihood(y, x_0_hat)
            likelihood_score = self.log_likelihood_weight * torch.autograd.grad(log_likelihood, z_t, create_graph=False)[0]
            posterior_score = prior_score + likelihood_score
            score = posterior_score

        # flip the sign to apply the standard Anderson formula
        dt = -dt

        if mode == 'sde':
            return z_t - dt * score + torch.sqrt(torch.abs(dt)) * torch.randn_like(z_t)
        elif mode == 'ode':
            return z_t - dt * 0.5 * score

    def sample_z_t_minus_delta_t_given_z_t(self, z_t, t, delta_t, mode='sde', num_steps=16, y=None, timesteps=None):
        assert mode in ['sde', 'ode']
        if timesteps is None:
            timesteps = torch.linspace(t.item(), t.item() - delta_t.item(), num_steps+1).to(device)
        x = z_t
        for i in range(num_steps):
            print(f'Sampling step {i}/{num_steps}, t={timesteps[i]}, isnan={torch.isnan(x).any()}')
            dt = timesteps[i] - timesteps[i+1]
            dt = dt.unsqueeze(0).to(device)
            if mode == 'sde':
                x = self.sample_z_t_minus_dt_given_z_t(x, timesteps[i].unsqueeze(0), dt, y=y)
            else:
                x = self.sample_z_t_minus_dt_given_z_t(x, timesteps[i].unsqueeze(0), dt, mode='ode', y=y)
        return x
    
    def sample_z_0_given_z_t(self, z_t, t, mode='sde', num_steps=1024, y=None, timesteps=None):
        delta_t = t.clone()
        return self.sample_z_t_minus_delta_t_given_z_t(z_t, t, delta_t, mode=mode, num_steps=num_steps, y=y, timesteps=timesteps)

def HU_to_SU(x):
    return x / 1000

def SU_to_HU(x):
    return x * 1000

def train_diffusion_model(
            diffusion_model, 
            train_loader, 
            val_loader=None, 
            time_sampler=None, 
            T=0.01, 
            num_epochs=100, 
            num_iterations_train=100,
            num_iterations_val=10, 
            lr=2e-4):
    
    assert isinstance(diffusion_model, LatentDiffusionModel)

    optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=lr)
    ema = ExponentialMovingAverage(diffusion_model.parameters(), decay=0.995)  # Exponential moving average for stabilizing training
    criterion = nn.MSELoss()

    if time_sampler is None:
        time_sampler = lambda batch_size: T * (torch.rand(batch_size, 1)**2.0)

    for iEpoch in range(num_epochs):
        diffusion_model.train()
        train_loader_iter = iter(train_loader)
        train_loss = 0
        for iIteration in tqdm(range(num_iterations_train)):
            # print(f'Iteration {iIteration}, Memory Allocated: {torch.cuda.memory_allocated() / (1024 ** 3)} GB')
            try:
                x_0, _ = next(train_loader_iter)
            except StopIteration:
                train_loader_iter = iter(train_loader)
                x_0,_ = next(train_loader_iter)
            x_0 = x_0.to(device)
            
            brain_mask = torch.logical_and(x_0 > 0.0, x_0 < 80.0)

            x_0 = HU_to_SU(x_0)

            # encode the input image into the latent space
            z_0 = diffusion_model.encode(x_0)

            # t = time_sampler(x_0.size(0)).to(device)

            # forward diffusion
            t = time_sampler(x_0.size(0)).to(device)
            z_t = diffusion_model.sample_z_t_given_z_0(z_0, t)
            
            # reverse diffusion predictor
            z_0_hat = z_0.clone()
            z_0_hat = diffusion_model.predict_z_0_given_z_t(z_t, t)           

            # reverse diffusion decoder
            x_0_hat = diffusion_model.decode(z_0_hat)
            
            x_0 = SU_to_HU(x_0)
            x_0_hat = SU_to_HU(x_0_hat)


            res = x_0 - x_0_hat
            loss_weights = 1/(t + 1e-6)
            loss_weights = x_0.size(0)*loss_weights / loss_weights.sum()
            loss_weights = torch.sqrt(loss_weights).view(-1, 1, 1, 1)

            brain_weight = 0.95
            loss = (1-brain_weight)*criterion(res*loss_weights, res*0)
            if torch.any(brain_mask):
                loss += brain_weight*criterion((res*loss_weights)[brain_mask], res[brain_mask]*0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update()

            train_loss += loss.item()
        
        train_loss /= num_iterations_train
        train_loss = np.sqrt(train_loss) # RMSE loss
        print(f'Epoch {iEpoch}, Training Loss: {train_loss}')

        if val_loader is not None:
            diffusion_model.eval()
            val_loader_iter = iter(val_loader)
            val_loss = 0
            with torch.no_grad():
                for i in tqdm(range(num_iterations_val)):
                    try:
                        x_0, _ = next(val_loader_iter)
                    except StopIteration:
                        val_loader_iter = iter(val_loader)
                        x_0,_ = next(val_loader_iter)
                    x_0 = x_0.to(device)
                    brain_mask = torch.logical_and(x_0 > 0.0, x_0 < 80.0)

                    x_0 = HU_to_SU(x_0)
                    z_0 = diffusion_model.encode(x_0)

                    t = time_sampler(x_0.size(0)).to(device)*.01

                    z_t = diffusion_model.sample_z_t_given_z_0(z_0, t)
                    z_0_hat = diffusion_model.predict_z_0_given_z_t(z_t, t)
                    x_0_hat = diffusion_model.decode(z_0_hat)

                    x_0 = SU_to_HU(x_0)
                    x_0_hat = SU_to_HU(x_0_hat)


                    res = x_0 - x_0_hat
                    loss_weights = 1/(t + 1e-6)
                    loss_weights = x_0.size(0)*loss_weights / loss_weights.sum()
                    loss_weights = torch.sqrt(loss_weights).view(-1, 1, 1, 1)

                    loss = (1-brain_weight)*criterion(res*loss_weights, 0*res)
                    if torch.any(brain_mask):
                        loss += brain_weight*criterion((res*loss_weights)[brain_mask], 0*res[brain_mask])

                    val_loss += loss.item()

            val_loss /= num_iterations_val
            val_loss = np.sqrt(val_loss) # RMSE loss
            print(f'Epoch {iEpoch}, Validation Loss: {val_loss}')

def compute_sample_weights(metadata, hemorrhage_types):
    class_counts = metadata[hemorrhage_types].sum(axis=0).to_numpy()
    class_weights = 1.0 / class_counts
    sample_weights_matrix = metadata[hemorrhage_types].to_numpy() * class_weights
    sample_weights = sample_weights_matrix.sum(axis=1)
    return sample_weights

def save_diffusion_model_weights(diffusion_model, save_path):
    torch.save(diffusion_model.state_dict(), save_path)

def load_diffusion_model_weights(diffusion_model, load_path):
    diffusion_model.load_state_dict(torch.load(load_path))

def evaluate_diffusion_model(diffusion_model, test_loader, num_samples=5):
    assert isinstance(diffusion_model, LatentDiffusionModel)
    diffusion_model.eval()
    test_loader_iter = iter(test_loader)
    with torch.no_grad():
        for i in range(num_samples):
            x_0, _ = next(test_loader_iter)
            if i >= 5:
                break  # Display first 5 samples

            x_0 = x_0.to(device)
            x_0_HU = x_0.clone()  # Keep original for display

            x_0 = HU_to_SU(x_0)
            z_0 = diffusion_model.encode(x_0)

            t = torch.tensor([0.01], device=device)

            z_t = diffusion_model.sample_z_t_given_z_0(z_0, t)
            
            num_steps = 32
            timesteps = (torch.linspace(1.0, 0.0, num_steps+1).to(device)**2.0)*t.item()

            z_0_hat = diffusion_model.sample_z_0_given_z_t(z_t, t, mode='sde', num_steps=num_steps, timesteps=timesteps)

            x_0_hat = diffusion_model.decode(z_0_hat)
            x_0_hat = SU_to_HU(x_0_hat)

            # Noisy image at t=1.0
            x_t = diffusion_model.decode(z_t)
            x_t = SU_to_HU(x_t)

            # Plotting
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

            plt.savefig(f'evaluation_{i}.png')

def main():

    train_flag = True
    load_flag = True
    multiGPU_flag = False
    device_ids = [1,2,3]
    batch_size = 64
    num_epochs = 10
    num_iterations_train = 100
    num_iterations_val = 5
    lr = 1e-4

    from step0_common_info import dicom_dir

    train_dataset = RSNA_Intracranial_Hemorrhage_Dataset(
            'data/metadata_training.csv',
            dicom_dir,
            patch_size=64)
    
    val_dataset = RSNA_Intracranial_Hemorrhage_Dataset(
            'data/metadata_validation.csv',
            dicom_dir,
            patch_size=64)
    
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=8)

    sample_weights = compute_sample_weights(val_dataset.metadata, val_dataset.hemorrhage_types)
    val_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(val_dataset), replacement=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    def log_likelihood(y, x):
        return -torch.nn.functional.mse_loss(x, y, reduction='none').sum(dim=(1,2,3))

    # Initialize the diffusion model
    diffusion_model = LatentDiffusionModel(log_likelihood=log_likelihood, log_likelihood_weight=1.0).to(device)

    if multiGPU_flag:
        diffusion_model.unet = torch.nn.DataParallel(diffusion_model.unet, device_ids=device_ids)

    if load_flag:
        # Load the diffusion model's weights
        try:
            load_diffusion_model_weights(diffusion_model, 'weights/diffusion_model_weights.pth')
            print("Diffusion model weights loaded successfully.")
        except:
            print("Diffusion model weights not found. Training from scratch.")

    if train_flag:

        # Train the diffusion model
        train_diffusion_model(
            diffusion_model,
            train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            num_iterations_train=num_iterations_train,
            num_iterations_val=num_iterations_val,
            lr=lr
        )

        print("Training complete. Saving diffusion model weights.")
        save_diffusion_model_weights(diffusion_model, 'weights/diffusion_model_weights.pth')

    from step5_deep_learning_reconstruction import CTProjector
    projector = CTProjector()

    # Evaluate the diffusion model
    evaluate_diffusion_model(diffusion_model, test_loader, projector)

if __name__ == "__main__":
    main()


