




import os
# visible device 3
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
import torch.nn as nn
import numpy as np
from diffusers import UNet2DModel
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm
from step2_dataset_dataloader import RSNA_Intracranial_Hemorrhage_Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from diffusers import AutoencoderKL, UNet2DConditionModel
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from step4_iterative_reconstruction import CTProjector, LinearLogLikelihood, HU_to_attenuation, attenuation_to_HU
import matplotlib.pyplot as plt  # Import matplotlib for plotting

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class UnconditionalDiffusionModel(nn.Module):
    def __init__(self):
        super(UnconditionalDiffusionModel, self).__init__()
        
        block_out_channels = (32, 64, 128, 256)
        layers_per_block = 4

        # block_out_channels = (128, 256, 512, 1024)
        # layers_per_block = 4
        
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
    
    def predict_x_0_given_x_t(self, x_t, t):
        assert isinstance(t, torch.Tensor)
        
        sigma_data = 0.5
        def c_skip(t):
            return (sigma_data**2) / (t + sigma_data**2)
        
        def c_out(t):
            return sigma_data*torch.sqrt(t)/torch.sqrt(t + sigma_data**2)
        

        # # I am skeptical about the EDM preconditioning, switching to a direct denoiser

        def c_skip(t):
            return torch.ones_like(t).to(torch.float32)
        
        def c_out(t):
            return torch.sqrt(t)
        
        return c_skip(t).view(-1,1,1,1) * x_t + c_out(t).view(-1,1,1,1) * self.unet(x_t, t.view(-1))[0]

    def predict_score_given_x_t(self, x_t, t):
        mu = self.predict_x_0_given_x_t(x_t, t)
        sigma2 = t.view(-1,1,1,1)
        score = -(x_t - mu) / sigma2
        return score
    
    def sample_x_t_minus_dt_given_x_t(self, x_t, t, dt, mode='sde'):
        assert mode in ['sde', 'ode']
        assert isinstance(t, torch.Tensor)
        assert isinstance(dt, torch.Tensor)

        score = self.predict_score_given_x_t(x_t, t)

        # flip the sign to apply the standard Anderson formula
        dt = -dt

        if mode == 'sde':
            return x_t - dt * score + torch.sqrt(torch.abs(dt)) * torch.randn_like(x_t)
        elif mode == 'ode':
            return x_t - dt * 0.5 * score

    def sample_x_t_minus_delta_t_given_x_t(self, x_t, t, delta_t, mode='sde', num_steps=None, timesteps=None):
        assert mode in ['sde', 'ode']
        assert (timesteps is not None) or (num_steps is not None), 'Either timesteps or num_steps must be provided'
        if timesteps is None:
            timesteps = torch.linspace(t.item(), t.item() - delta_t.item(), num_steps+1).to(device)
        x = x_t
        for i in range(len(timesteps) - 1 ):
            print(f'Sampling step {i}/{len(timesteps) - 1}, t={timesteps[i]}, isnan={torch.isnan(x).any()}')
            dt = timesteps[i] - timesteps[i+1]
            dt = dt.unsqueeze(0).to(device)
            x = self.sample_x_t_minus_dt_given_x_t(x, timesteps[i].unsqueeze(0), dt, mode=mode)
        return x
    
    def sample_x_0_given_x_t(self, x_t, t, mode='sde', num_steps=1024, timesteps=None):
        delta_t = t.clone()
        return self.sample_x_t_minus_delta_t_given_x_t(x_t, t, delta_t, mode=mode, num_steps=num_steps, timesteps=timesteps)


class DiffusionPosteriorSampling(UnconditionalDiffusionModel):
    def __init__(self, diffusion_model, log_likelihood):
        nn.Module.__init__(self)
        assert isinstance(diffusion_model, UnconditionalDiffusionModel)
        assert isinstance(log_likelihood, LinearLogLikelihood)
        self.diffusion_model = diffusion_model
        self.log_likelihood = log_likelihood

    def sample_x_t_minus_dt_given_x_t(self, x_t, t, dt, mode='sde'):
        assert mode in ['sde', 'ode']
        assert isinstance(t, torch.Tensor)
        assert isinstance(dt, torch.Tensor)
        assert isinstance(self.diffusion_model, UnconditionalDiffusionModel)
        assert isinstance(self.log_likelihood, LinearLogLikelihood)

        # compute the prior score
        prior_score = self.diffusion_model.predict_score_given_x_t(x_t, t)
        
        # applie Tweedie's formula to compute the posterior mean, x_0_hat
        sigma2 = t.view(-1, 1, 1, 1)
        x_0_hat = x_t + sigma2 * prior_score

        # this ignores the jacobian of x_0_hat w.r.t. x_t, 
        # but it can be shown to be a nearly scalar matrix, 
        # https://arxiv.org/html/2407.12956v1, Appendix A, Figure 11
        # so it can be wrapped into one hyperparameter, 
        # the measurement variance of the likelihood
        # defined outside this model
        # the negative sign here is because LinearLogLikelihood is actually a negative log likelihood... TODO: fix this
        _x_0_hat_SU = x_0_hat
        _x_0_hat_HU = SU_to_HU(_x_0_hat_SU)
        _x_0_hat_attenuation = HU_to_attenuation(_x_0_hat_HU)

        _x_t_SU = x_t
        _x_t_HU = SU_to_HU(_x_t_SU)
        _x_t_attenuation = HU_to_attenuation(_x_t_HU)
        _likelihood_score_attenuation = -self.log_likelihood.gradient(_x_0_hat_attenuation)
        # _likelihood_score_attenuation = -self.log_likelihood.gradient(_x_t_attenuation)
        _likelihood_score_HU = attenuation_to_HU(_likelihood_score_attenuation, scaleOnly=True)
        _likelihood_score_SU = HU_to_SU(_likelihood_score_HU)
        likelihood_score = _likelihood_score_SU

                

        # the good work of Rev. Thomas Bayes
        # prior_score *= 0.0
        posterior_score = prior_score + likelihood_score
        print('DEBUG: prior_score:', prior_score.norm().item(), ', likelihood_score:', likelihood_score.norm().item(), 'posterior_score:', posterior_score.norm().item())

        # flip the sign to apply the standard Anderson formula
        dt = -dt

        if mode == 'sde':
            return x_t - dt * posterior_score + torch.sqrt(torch.abs(dt)) * torch.randn_like(x_t)
        elif mode == 'ode':
            return x_t - dt * 0.5 * posterior_score




class DiffusionLossClosure(nn.Module):
    def __init__(self, diffusion_model, patch_size=256, brain_weight=0.95, T=1.0):
        super(DiffusionLossClosure, self).__init__()
        self.diffusion_model = diffusion_model
        self.patch_size = patch_size
        self.brain_weight = brain_weight
        self.T = T
        self.criterion = nn.MSELoss()

    def forward(self, x_0):

        assert isinstance(self.diffusion_model, UnconditionalDiffusionModel), 'diffusion_model must be an instance of UnconditionalDiffusionModel'

        assert isinstance(x_0, torch.Tensor), 'x_0 must be a torch.Tensor'
        assert len(x_0.shape) == 4, 'x_0 must be a 4D tensor with shape (batch_size, 1, patch_size, patch_size)'
        assert x_0.shape[1] == 1, 'x_0 must have 1 channel'


        x_0 = x_0.float()
        batch_size = x_0.shape[0]
        device = x_0.device

        brain_mask = torch.logical_and(x_0 > 0.0, x_0 < 80.0)

        x_0 = HU_to_SU(x_0)

        t = torch.rand(batch_size, 1, device=device)**2.0 * self.T

        x_t = self.diffusion_model.sample_x_t_given_x_0(x_0, t)

        x_0_hat = self.diffusion_model.predict_x_0_given_x_t(x_t, t)

        x_0 = SU_to_HU(x_0)
        x_0_hat = SU_to_HU(x_0_hat)

        res = x_0 - x_0_hat




        # Extract patches
        patch_size = self.patch_size
        
        # Extract center region
        patch_margin = patch_size // 4
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
    


    def extract_patches(self, data, batch_size, device):
        patch_size = self.patch_size
        patches = torch.zeros(batch_size, 1, patch_size, patch_size, device=device)
        for i in range(batch_size):
            iRow = np.random.randint(0, 256 - patch_size)
            iCol = np.random.randint(0, 256 - patch_size)
            patches[i] = data[i, :, iRow:iRow + patch_size, iCol:iCol + patch_size]
        return patches










def HU_to_SU(x):
    return x / 1000.0

def SU_to_HU(x):
    return x * 1000.0


def evaluate_diffusion_model(diffusion_model, projector, test_loader, num_samples=1, noise_variance=1.0):
    assert isinstance(diffusion_model, UnconditionalDiffusionModel)
    assert isinstance(projector, CTProjector)

    measurements = None # Placeholder for measurements
    linear_log_likelihood = LinearLogLikelihood(measurements, projector, noise_variance=noise_variance)
    
    diffusion_posterior_sampling = DiffusionPosteriorSampling(diffusion_model, linear_log_likelihood)
    
    diffusion_model.eval()
    test_loader_iter = iter(test_loader)

    with torch.no_grad():
        for i in range(num_samples):
            x_0, _ = next(test_loader_iter)

            x_0 = x_0.to(device)
            x_0_HU = x_0.clone()  # Keep original for display

            # set the measurements for diffusion posterior sampling model
            sinogram = projector.forward_project(HU_to_attenuation(x_0))
            diffusion_posterior_sampling.log_likelihood.measurements = sinogram

            # this is only for initialization, 
            # run the pseudoinverse on the sinogram,
            # then convert to SU so we can runn the diffusion forward process
            pseudoinverse = projector.pseudoinverse_reconstruction(sinogram, singular_values=[3000])
            pseudoinverse_HU = attenuation_to_HU(pseudoinverse)
            pseudoinverse_SU = HU_to_SU(pseudoinverse_HU)

            # also convert the original image to SU
            x_0 = HU_to_SU(x_0)

            # sample the forward diffusion process, p(x_t|x_0) at time t
            t = torch.tensor([1.0], device=device)
            # x_t = diffusion_model.sample_x_t_given_x_0(pseudoinverse_SU, t)
            x_t = diffusion_model.sample_x_t_given_x_0(x_0, t)

            num_steps = 128
            timesteps = (torch.linspace(1.0, 0.0, num_steps + 1).to(device)**2.0) * t.item()

            x_0_hat = diffusion_posterior_sampling.sample_x_0_given_x_t(x_t, t, mode='ode', timesteps=timesteps)

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

            plt.savefig(f'./figures/DPS_batch_{i}_bone.png')

            
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

            plt.savefig(f'./figures/DPS_batch_{i}_brain.png')


            plt.close('all')


# def train_diffusion_model(
#             diffusion_model, 
#             train_loader, 
#             val_loader=None, 
#             time_sampler=None, 
#             T=1.0, 
#             num_epochs=100, 
#             num_iterations_train=100,
#             num_iterations_val=10, 
#             lr=2e-4):
    
#     assert isinstance(diffusion_model, UnconditionalDiffusionModel)

#     optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=lr)
#     ema = ExponentialMovingAverage(diffusion_model.parameters(), decay=0.995)  # Exponential moving average for stabilizing training
#     criterion = nn.MSELoss()

#     if time_sampler is None:
#         time_sampler = lambda batch_size: T * (torch.rand(batch_size, 1)**2.0)
    
#     train_loader_iter = iter(train_loader)
#     for iEpoch in range(num_epochs):
#         diffusion_model.train()
#         train_loss = 0
#         for iIteration in tqdm(range(num_iterations_train)):
#             # print(f'Iteration {iIteration}, Memory Allocated: {torch.cuda.memory_allocated() / (1024 ** 3)} GB')
#             try:
#                 x_0, _ = next(train_loader_iter)
#             except StopIteration:
#                 train_loader_iter = iter(train_loader)
#                 x_0,_ = next(train_loader_iter)
#             x_0 = x_0.to(device)
            
#             brain_mask = torch.logical_and(x_0 > 0.0, x_0 < 80.0)

#             x_0 = HU_to_SU(x_0)


#             # t = time_sampler(x_0.size(0)).to(device)

#             # forward diffusion
#             t = time_sampler(x_0.size(0)).to(device)
#             x_t = diffusion_model.sample_x_t_given_x_0(x_0, t)
            
#             # reverse diffusion predictor
#             x_0_hat = x_0.clone()
#             x_0_hat = diffusion_model.predict_x_0_given_x_t(x_t, t)           

                
#             x_0 = SU_to_HU(x_0)
#             x_0_hat = SU_to_HU(x_0_hat)


#             res = x_0 - x_0_hat
#             loss_weights = 1/(t + 1e-6)
#             loss_weights = x_0.size(0)*loss_weights / loss_weights.sum()
#             loss_weights = torch.sqrt(loss_weights).view(-1, 1, 1, 1)

#             brain_weight = 0.95
#             loss = (1-brain_weight)*criterion(res*loss_weights, res*0)
#             if torch.any(brain_mask):
#                 loss += brain_weight*criterion((res*loss_weights)[brain_mask], res[brain_mask]*0)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             ema.update()

#             train_loss += loss.item()
        
#         train_loss /= num_iterations_train
#         train_loss = np.sqrt(train_loss) # RMSE loss
#         print(f'Epoch {iEpoch}, Training Loss: {train_loss}')

#         if val_loader is not None:
#             diffusion_model.eval()
#             val_loader_iter = iter(val_loader)
#             val_loss = 0
#             with torch.no_grad():
#                 for i in tqdm(range(num_iterations_val)):
#                     try:
#                         x_0, _ = next(val_loader_iter)
#                     except StopIteration:
#                         val_loader_iter = iter(val_loader)
#                         x_0,_ = next(val_loader_iter)
#                     x_0 = x_0.to(device)
#                     brain_mask = torch.logical_and(x_0 > 0.0, x_0 < 80.0)

#                     x_0 = HU_to_SU(x_0)

#                     t = time_sampler(x_0.size(0)).to(device)*.01

#                     x_t = diffusion_model.sample_x_t_given_x_0(x_0, t)
#                     x_0_hat = diffusion_model.predict_x_0_given_x_t(x_t, t)

#                     x_0 = SU_to_HU(x_0)
#                     x_0_hat = SU_to_HU(x_0_hat)

#                     res = x_0 - x_0_hat

#                     loss_weights = 1/(t + 1e-6)
#                     loss_weights = x_0.size(0)*loss_weights / loss_weights.sum()
#                     loss_weights = torch.sqrt(loss_weights).view(-1, 1, 1, 1)

#                     loss = (1-brain_weight)*criterion(res*loss_weights, 0*res)
#                     if torch.any(brain_mask):
#                         loss += brain_weight*criterion((res*loss_weights)[brain_mask], 0*res[brain_mask])

#                     val_loss += loss.item()

#             val_loss /= num_iterations_val
#             val_loss = np.sqrt(val_loss) # RMSE loss
#             print(f'Epoch {iEpoch}, Validation Loss: {val_loss}')








# Device handling function
def get_device(device_input):
    if isinstance(device_input, list):
        device_ids = device_input
        device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")
    else:
        device_ids = None
        device = torch.device(device_input)
    return device, device_ids








def train_diffusion_model_parallel(diffusion_model, train_loader, val_loader=None, 
                                   num_epochs=100, num_iterations_train=100, num_iterations_val=10, 
                                   lr=1e-4, patch_size=256, device_input='cuda'):
    
    device, device_ids = get_device(device_input)
    diffusion_model.to(device)

    loss_closure = DiffusionLossClosure(diffusion_model, patch_size).to(device)

    # Multi-GPU handling
    if device_ids:
        diffusion_model = torch.nn.DataParallel(diffusion_model, device_ids=device_ids)
        loss_closure = torch.nn.DataParallel(loss_closure, device_ids=device_ids)

    optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=lr)
    ema = ExponentialMovingAverage(diffusion_model.parameters(), decay=0.995)

    train_loader_iter = iter(train_loader)
    for epoch in range(num_epochs):
        diffusion_model.train()
        train_loss = 0

        for iIteration in tqdm(range(num_iterations_train)):
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

        # if val_loader is not None:
            # evaluate_diffusion_model_parallel(diffusion_model, val_loader, num_iterations_val, device)




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

    train_flag = False
    load_flag = True
    device_input = [0,1,2,3]
    batch_size = 64
    num_epochs = 10
    num_iterations_train = 100
    num_iterations_val = 5
    lr = 1e-4
    patch_size = 64

    from step0_common_info import dicom_dir

    train_dataset = RSNA_Intracranial_Hemorrhage_Dataset(
            'data/metadata_training.csv',
            dicom_dir,
            patch_size=patch_size)
    
    val_dataset = RSNA_Intracranial_Hemorrhage_Dataset(
            'data/metadata_validation.csv',
            dicom_dir,
            patch_size=patch_size)
    
    test_dataset = RSNA_Intracranial_Hemorrhage_Dataset(
            'data/metadata_evaluation.csv',
            dicom_dir)

    # def compute_sample_weights(metadata, hemorrhage_types):
    #     class_counts = metadata[hemorrhage_types].sum(axis=0).to_numpy()
    #     class_weights = 1.0 / class_counts
    #     sample_weights_matrix = metadata[hemorrhage_types].to_numpy() * class_weights
    #     sample_weights = sample_weights_matrix.sum(axis=1)
    #     return sample_weights

    sample_weights = compute_sample_weights(train_dataset.metadata, train_dataset.hemorrhage_types)
    train_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dataset), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=16)

    sample_weights = compute_sample_weights(val_dataset.metadata, val_dataset.hemorrhage_types)
    val_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(val_dataset), replacement=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)

    torch.manual_seed(1)
    inds = np.arange(len(test_dataset))
    test_dataset = Subset(test_dataset, inds)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Initialize the diffusion model
    diffusion_model = UnconditionalDiffusionModel().to(device)
    # projector = CTProjector().to(device)

    if load_flag:
        # Load the diffusion model's weights
        try:
            load_diffusion_model_weights(diffusion_model, 'weights/diffusion_model_weights.pth')
            print("Diffusion model weights loaded successfully.")
        except:
            print("Diffusion model weights not found. Training from scratch.")

    if train_flag:

        # # Train the diffusion model
        # train_diffusion_model(
        #     diffusion_model,
        #     train_loader,
        #     val_loader=None,
        #     num_epochs=num_epochs,
        #     num_iterations_train=num_iterations_train,
        #     num_iterations_val=num_iterations_val,
        #     lr=lr
        # )

        train_diffusion_model_parallel(
            diffusion_model, train_loader, val_loader=None, 
            num_epochs=num_epochs, num_iterations_train=num_iterations_train, num_iterations_val=num_iterations_val, 
            lr=lr, patch_size=patch_size, device_input=device_input
        )

        print("Training complete. Saving diffusion model weights.")
        save_diffusion_model_weights(diffusion_model, 'weights/diffusion_model_weights.pth')


    # Evaluate the diffusion model
    projector = CTProjector().to(device)
    evaluate_diffusion_model(diffusion_model, projector, test_loader, num_samples=10)












if __name__ == "__main__":
    main()

















































