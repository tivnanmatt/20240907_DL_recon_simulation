
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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cuda:2'

class LatentDiffusionModel(nn.Module):
    def __init__(self, log_likelihood, log_likelihood_weight=1.0):

        super(LatentDiffusionModel, self).__init__()

        self.vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(device)      
        self.unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet").to(device)
        
        self.log_likelihood = log_likelihood
        self.log_likelihood_weight = log_likelihood_weight

    def encode(self, x):
        x = x.repeat(1, 3, 1, 1)
        return self.vae.encode(x).latent_dist.mean
    
    def decode(self, z):
        x = self.vae.decode(z).sample
        x = torch.mean(x, dim=1, keepdim=True)
        return x
    
    def sample_z_t_given_z_0(self, z_0, t):
        return z_0 + torch.sqrt(t.view(-1,1,1)) * torch.randn_like(z_0)

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
        encoder_hidden_states = torch.zeros((z_t.size(0), 64, 768)).to(device)  # Adjust dimensions as needed
        return z_t -  self.unet(z_t, t.view(-1), encoder_hidden_states)[0]* torch.sqrt(t.view(-1,1,1,1))

    def predict_score_given_z_t(self, z_t, t):
        mu = self.predict_z_0_given_z_t(z_t, t)
        sigma2 = t
        return -(z_t - mu) / sigma2
    
    def sample_z_t_minus_dt_given_z_t(self, z_t, t, dt, mode='sde',y=None):
        assert mode in ['sde', 'ode']
        assert isinstance(t, torch.Tensor)
        assert isinstance(dt, torch.Tensor)

        prior_score = self.predict_score_given_z_t(z_t, t)
        
        if y is not None:
            score = prior_score
        else:
            z_0_hat = self.predict_z_0_given_z_t(z_t, t)
            x_0_hat = self.decode(z_0_hat)
            log_likelihood = self.log_likelihood(y, x_0_hat)
            likelihood_score = self.log_likelihood_weight * torch.autograd.grad(log_likelihood, z_t, create_graph=False)[0]
            posterior_score = prior_score + likelihood_score
            score = posterior_score

        if mode == 'sde':
            return z_t - dt * score + torch.sqrt(dt) * torch.randn_like(z_t)
        elif mode == 'ode':
            return z_t - dt * 0.5 * score

    def sample_z_t_minus_delta_t_given_z_t(self, z_t, t, delta_t, mode='sde', num_steps=16, y=None):
        assert mode in ['sde', 'ode']
        timesteps = torch.linspace(t, t - delta_t, num_steps+1)
        x = z_t
        for i in range(num_steps):
            dt = timesteps[i] - timesteps[i+1]
            if mode == 'sde':
                x = self.sample_z_t_minus_dt_given_z_t(x, timesteps[i], dt, y=y)
            else:
                x = self.sample_z_t_minus_dt_given_z_t(x, timesteps[i], dt, mode='ode', y=y)
        return x
    
    def sample_z_0_given_z_t(self, z_t, t, mode='sde', num_steps=1024, y=None):
        return self.sample_z_t_minus_delta_t_given_z_t(z_t, t, t, mode=mode, num_steps=num_steps, y=y)



def HU_to_SU(x):
    return (x-500) / 1500

def SU_to_HU(x):
    return x * 1500 + 500
    
def train_diffusion_model(
            diffusion_model, 
            train_loader, 
            val_loader=None, 
            time_sampler=None, 
            T=0.0001, 
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
            z_0_hat[1:] = diffusion_model.predict_z_0_given_z_t(z_t[1:], t[1:])           

            # reverse diffusion decoder
            x_0_hat = diffusion_model.decode(z_0_hat)
            
            x_0 = SU_to_HU(x_0)
            x_0_hat = SU_to_HU(x_0_hat)

            brain_weight = 0.95
            loss = (1-brain_weight)*criterion(x_0_hat, x_0)
            if torch.any(brain_mask):
                loss += brain_weight*criterion(x_0_hat[brain_mask], x_0[brain_mask])

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

                    t = time_sampler(x_0.size(0)).to(device)

                    z_t = diffusion_model.sample_z_t_given_z_0(z_0, t)
                    z_0_hat = diffusion_model.predict_z_0_given_z_t(z_t, t)
                    x_0_hat = diffusion_model.decode(z_0_hat)

                    x_0 = SU_to_HU(x_0)
                    x_0_hat = SU_to_HU(x_0_hat)

                    loss = (1-brain_weight)*criterion(x_0_hat, x_0)
                    if torch.any(brain_mask):
                        loss += brain_weight*criterion(x_0_hat[brain_mask], x_0[brain_mask])

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

def main():

    train_flag = True
    load_flag = True
    multiGPU_flag = False
    device_ids = [1,2,3]
    batch_size = 4
    num_epochs = 10
    num_iterations_train = 100
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=8)

    sample_weights = compute_sample_weights(val_dataset.metadata, val_dataset.hemorrhage_types)
    val_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(val_dataset), replacement=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    def log_likelihood(y, x):
        return -torch.nn.functional.mse_loss(x, y, reduction='none').sum(dim=(1,2,3))

    # Initialize the diffusion model
    diffusion_model = LatentDiffusionModel(log_likelihood=log_likelihood, log_likelihood_weight=1.0).to(device)

    if multiGPU_flag:
        # projector = torch.nn.DataParallel(projector, device_ids=device_ids)
        diffusion_model.unet = torch.nn.DataParallel(diffusion_model.unet, device_ids=device_ids)
        diffusion_model.vae = torch.nn.DataParallel(diffusion_model.vae, device_ids=device_ids)

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


if __name__ == "__main__":
    main()