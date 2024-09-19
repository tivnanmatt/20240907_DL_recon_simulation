
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

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

class DiffusionModel(nn.Module):
    def __init__(self):

        super(DiffusionModel, self).__init__()

        block_out_channels = (128, 256, 512, 1024)
        # block_out_channels = (16, 32, 64, 128)

        layers_per_block = 4
        # # layers_per_block = 2

        self.unet = UNet2DModel(
            sample_size=256,
            in_channels=2,  # 32 components from the pseudo-inverse
            out_channels=1,  # Final reconstructed image
            center_input_sample=False,
            time_embedding_type='positional',
            freq_shift=0,
            flip_sin_to_cos=True,
            down_block_types=('DownBlock2D', 'AttnDownBlock2D', 'AttnDownBlock2D', 'AttnDownBlock2D'),
            up_block_types=('AttnUpBlock2D', 'AttnUpBlock2D', 'AttnUpBlock2D', 'UpBlock2D'),
            block_out_channels=block_out_channels   ,
            layers_per_block=layers_per_block,
            mid_block_scale_factor=1,
            downsample_padding=1,
            downsample_type='conv',
            upsample_type='conv',
            dropout=0.0,
            act_fn='silu',
            attention_head_dim=None,
            norm_num_groups=1,
            attn_norm_num_groups=None,
            norm_eps=1e-05,
            resnet_time_scale_shift='default',
            add_attention=True,
            class_embed_type=None,
            num_class_embeds=None,
            num_train_timesteps=None
        )
        


    def sample_x_t_given_x_0(self, x_0, t):
        assert isinstance(t, torch.Tensor)
        return x_0 + torch.sqrt(t.view(-1,1,1,1)) * torch.randn_like(x_0)
    
    def sample_x_t_plus_dt_given_x_t(self, x_t, t, dt):
        assert isinstance(t, torch.Tensor)
        assert isinstance(dt, torch.Tensor)
        return x_t + torch.sqrt(dt) * torch.randn_like(x_t)
    
    def sample_x_t_plus_delta_t_given_x_t(self, x_t, t, delta_t):
        assert isinstance(t, torch.Tensor)
        return x_t + torch.sqrt(delta_t) * torch.randn_like(x_t)
    
    def estimate_x_0_given_x_t(self, x_t, t, x_tilde=None):
        assert isinstance(t, torch.Tensor)
        if x_tilde is None:
            x_tilde = x_t*0.0
        # concat on channel dim
        x_in = torch.cat([x_t, t.view(-1,1,1,1).expand(-1,1,256,256)], dim=1)
        return x_t - self.unet(x_in, t.view(-1))[0] * torch.sqrt(t.view(-1,1,1,1))

    def estimate_score_given_x_t(self, x_t, t, x_tilde=None):
        mu = self.estimate_x_0_given_x_t(x_t, t, x_tilde)
        sigma2 = t
        return -(x_t - mu) / sigma2
    
    def sample_x_t_minus_dt_given_x_t(self, x_t, t, dt, mode='sde', x_tilde=None):
        assert mode in ['sde', 'ode']
        assert isinstance(t, torch.Tensor)
        assert isinstance(dt, torch.Tensor)
        if mode == 'sde':
            return x_t - dt * self.estimate_score_given_x_t(x_t, t, x_tilde) + torch.sqrt(dt) * torch.randn_like(x_t)
        elif mode == 'ode':
            return x_t - dt * 0.5 * self.estimate_score_given_x_t(x_t, t, x_tilde)

    def sample_x_minus_delta_t_given_x_t(self, x_t, t, delta_t, mode='sde', num_steps=16):
        assert mode in ['sde', 'ode']
        timesteps = torch.linspace(t, t - delta_t, num_steps+1)
        x = x_t
        for i in range(num_steps):
            dt = timesteps[i] - timesteps[i+1]
            if mode == 'sde':
                x = self.sample_x_t_minus_dt_given_x_t(x, timesteps[i], dt)
            else:
                x = self.sample_x_t_minus_dt_given_x_t(x, timesteps[i], dt, mode='ode')

        return x



def HU_to_SU(x):
    return x / 1000.0

def SU_to_HU(x):
    return x * 1000.0
    
def train_diffusion_model(
            diffusion_model, 
            train_loader, 
            val_loader=None, 
            time_sampler=None, 
            T=1.0, 
            num_epochs=100, 
            num_iterations_train=100,
            num_iterations_val=10, 
            lr=2e-4):
    
    assert isinstance(diffusion_model, DiffusionModel) or (isinstance(diffusion_model, torch.nn.DataParallel) and isinstance(diffusion_model.module, DiffusionModel))

    optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=lr)
    # ema = ExponentialMovingAverage(self.parameters(), decay=0.95)  # Exponential moving average for stabilizing training
    criterion = nn.MSELoss()

    if time_sampler is None:
        time_sampler = lambda batch_size: T * (torch.rand(batch_size, 1)**2.0)

    for iEpoch in range(num_epochs):
        diffusion_model.train()
        train_loader_iter = iter(train_loader)
        train_loss = 0
        for iIteration in tqdm(range(num_iterations_train)):
            try:
                x_0, _ = next(train_loader_iter)
            except StopIteration:
                train_loader_iter = iter(train_loader)
                x_0,_ = next(train_loader_iter)
            x_0 = x_0.to(device)

            brain_mask = torch.logical_and(x_0 > 0, x_0 < 80)
            x_0 = HU_to_SU(x_0)
            t = time_sampler(x_0.size(0)).to(device)

            if isinstance(diffusion_model, torch.nn.DataParallel):
                x_t = diffusion_model.module.sample_x_t_given_x_0(x_0, t)
                x_0_hat = diffusion_model.module.estimate_x_0_given_x_t(x_t, t)
            else:
                x_t = diffusion_model.sample_x_t_given_x_0(x_0, t)
                x_0_hat = diffusion_model.estimate_x_0_given_x_t(x_t, t)
            
            x_0 = SU_to_HU(x_0)
            x_t = SU_to_HU(x_t)
            x_0_hat = SU_to_HU(x_0_hat)
            brain_weight = 0.9
            loss = (1-brain_weight)*criterion(x_0_hat, x_0)
            if torch.any(brain_mask):
                loss += brain_weight*criterion(x_t[brain_mask], x_0[brain_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # ema.update()
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

                    brain_mask = torch.logical_and(x_0 > 0, x_0 < 80)
                    x_0 = HU_to_SU(x_0)
                    t = time_sampler(x_0.size(0)).to(device)
                       
                    if isinstance(diffusion_model, torch.nn.DataParallel):
                        x_t = diffusion_model.module.sample_x_t_given_x_0(x_0, t)
                        x_0_hat = diffusion_model.module.estimate_x_0_given_x_t(x_t, t)
                    else:   
                        x_t = diffusion_model.sample_x_t_given_x_0(x_0, t)
                        x_0_hat = diffusion_model.estimate_x_0_given_x_t(x_t, t)
                    x_0 = SU_to_HU(x_0)
                    x_t = SU_to_HU(x_t)
                    x_0_hat = SU_to_HU(x_0_hat)
                    
                    loss = (1-brain_weight)*criterion(x_0_hat, x_0)
                    if torch.any(brain_mask):
                        loss += brain_weight*criterion(x_t[brain_mask], x_0[brain_mask])
                    val_loss += loss.item()

            val_loss /= num_iterations_val
            val_loss = np.sqrt(val_loss) # RMSE loss
            print(f'Epoch {iEpoch}, Validation Loss: {val_loss}')

            save_diffusion_model_weights(diffusion_model, 'weights/diffusion_model_weights.pth')



def compute_sample_weights(metadata, hemorrhage_types):
    class_counts = metadata[hemorrhage_types].sum(axis=0).to_numpy()
    class_weights = 1.0 / class_counts
    sample_weights_matrix = metadata[hemorrhage_types].to_numpy() * class_weights
    sample_weights = sample_weights_matrix.sum(axis=1)
    return sample_weights


def save_diffusion_model_weights(diffusion_model, save_path):
    if isinstance(diffusion_model.unet, torch.nn.DataParallel):
        torch.save(diffusion_model.unet.module.state_dict(), save_path)
    else:
        torch.save(diffusion_model.unet.state_dict(), save_path)

def load_diffusion_model_weights(diffusion_model, load_path):
    if isinstance(diffusion_model.unet, torch.nn.DataParallel):
        diffusion_model.unet.module.load_state_dict(torch.load(load_path))
    else:
        diffusion_model.unet.load_state_dict(torch.load(load_path))

def main():

    train_flag = True
    load_flag = True
    multiGPU_flag = True
    device_ids = [0,1,2,3]
    batch_size = 2
    num_epochs = 100
    num_iterations_train = 30
    num_iterations_val = 5
    lr = 1e-4

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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)

    sample_weights = compute_sample_weights(val_dataset.metadata, val_dataset.hemorrhage_types)
    val_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(val_dataset), replacement=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the diffusion model
    diffusion_model = DiffusionModel().to(device)

    if multiGPU_flag:
        # projector = torch.nn.DataParallel(projector, device_ids=device_ids)
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

if __name__ == "__main__":
    main()