import os
# visible device 3
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
import torch.nn as nn
import numpy as np
from diffusers import UNet2DModel
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm
from step02_dataset_dataloader import RSNA_Intracranial_Hemorrhage_Dataset
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torch.utils.data.sampler import Sampler
from diffusers import AutoencoderKL, UNet2DConditionModel
from sklearn.model_selection import train_test_split
from step10_iterative_reconstruction import CTProjector, LinearLogLikelihood, HU_to_attenuation, attenuation_to_HU
import matplotlib.pyplot as plt  # Import matplotlib for plotting
from step03_histogram_equalization import HU_to_SU, SU_to_HU
import deepspeed
import torch.distributed as dist
import torch.multiprocessing as mp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def setup_distributed():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

class WeightedDistributedSampler(Sampler):
    def __init__(self, weights, num_samples, num_replicas=None, rank=None, replacement=True, seed=0):
        if num_replicas is None:
            num_replicas = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()

        self.weights = weights
        self.num_samples = num_samples
        self.num_replicas = num_replicas
        self.rank = rank
        self.replacement = replacement
        self.seed = seed

        self.num_samples_per_replica = int(np.ceil(num_samples / self.num_replicas))
        self.total_samples = self.num_samples_per_replica * self.num_replicas
        
        self.total_samples = torch.as_tensor(self.total_samples, dtype=torch.int64)
        self.weights = torch.as_tensor(self.weights, dtype=torch.float32)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.rank)

        indices = torch.multinomial(self.weights, self.total_samples, self.replacement, generator=g).tolist()
        indices = indices[self.rank:self.total_samples:self.num_replicas]

        return iter(indices)

    def __len__(self):
        return self.num_samples_per_replica

class UnconditionalDiffusionModel(nn.Module):
    def __init__(self):
        super(UnconditionalDiffusionModel, self).__init__()
        
        block_out_channels = (32, 64, 128, 256)
        layers_per_block = 4
        
        self.unet = UNet2DModel(
            sample_size=None,
            in_channels=1,
            out_channels=1,
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
        return self.unet(x_t, t.view(-1))[0]

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

        # Compute the prior score
        prior_score = self.diffusion_model.predict_score_given_x_t(x_t, t)
        
        # Convert x_t from SU to HU, then to attenuation units
        # zero gradient
        x_t.requires_grad_(True)
        x_t_HU = SU_to_HU(x_t)
        x_t_attenuation = HU_to_attenuation(x_t_HU)

        # Compute log likelihood
        log_likelihood_value = self.log_likelihood.forward(x_t_attenuation)
        log_likelihood_sum = log_likelihood_value.sum()
        x_t.grad = None
        log_likelihood_sum.backward()
        likelihood_gradient = x_t.grad

        # Posterior score
        posterior_score = prior_score + likelihood_gradient

        # Flip the sign to apply the standard Anderson formula
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
        # we want to detach here so it does not backpropagate through the diffusion model
        x_0 = x_0.detach()

        t = torch.rand(batch_size, 1, device=device)**2.0 * self.T

        x_t = self.diffusion_model.sample_x_t_given_x_0(x_0, t)

        x_0_hat = self.diffusion_model.predict_x_0_given_x_t(x_t, t)

        # x_0 = SU_to_HU(x_0)
        # x_0_hat = SU_to_HU(x_0_hat)

        res = x_0 - x_0_hat

        # Extract patches
        patch_size = self.patch_size

        if patch_size < 256:
            # Extract center region
            patch_margin = patch_size // 4
            patch_margin = np.clip(patch_margin, 0, 32)
            x_0 = x_0[:, :, patch_margin:-patch_margin, patch_margin:-patch_margin]
            x_0_hat = x_0_hat[:, :, patch_margin:-patch_margin, patch_margin:-patch_margin]
            brain_mask = brain_mask[:, :, patch_margin:-patch_margin, patch_margin:-patch_margin]
            res = res[:, :, patch_margin:-patch_margin, patch_margin:-patch_margin]

        loss = self.criterion(res, res * 0)

        # loss_weights = 1/(t + 1e-6)
        # loss_weights = batch_size * loss_weights / loss_weights.sum()
        # loss_weights = torch.sqrt(loss_weights).view(-1, 1, 1, 1)

        # loss = (1 - self.brain_weight) * self.criterion(res * loss_weights, res * 0)
        # if torch.any(brain_mask):
        #     loss += self.brain_weight * self.criterion((res * loss_weights)[brain_mask], res[brain_mask] * 0)

        return loss

def evaluate_diffusion_model(diffusion_model, projector, test_loader, num_samples=1, noise_variance=1.0):
    assert isinstance(diffusion_model, UnconditionalDiffusionModel)
    assert isinstance(projector, CTProjector)

    diffusion_model.eval()
    test_loader_iter = iter(test_loader)

    with torch.no_grad():
        for i in range(num_samples):
            x_0, _ = next(test_loader_iter)

            x_0 = x_0.to(device)
            x_0_HU = x_0.clone()

            sinogram = projector.forward_project(HU_to_attenuation(x_0))
            measurements = sinogram
            linear_log_likelihood = LinearLogLikelihood(measurements, projector, noise_variance=noise_variance)
            diffusion_posterior_sampling = DiffusionPosteriorSampling(diffusion_model, linear_log_likelihood)

            pseudoinverse = projector.pseudoinverse_reconstruction(sinogram, singular_values=[3000])
            pseudoinverse_HU = attenuation_to_HU(pseudoinverse)
            pseudoinverse_SU = HU_to_SU(pseudoinverse_HU)

            x_0 = HU_to_SU(x_0)

            t = torch.tensor([0.01], device=device)
            x_t = diffusion_model.sample_x_t_given_x_0(pseudoinverse_SU, t)

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
            loss = loss_closure(x_0).mean()
            loss.backward()
            optimizer.step()
            ema.update()

            train_loss += loss.item()

        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {np.sqrt(train_loss / num_iterations_train)}')

def train_diffusion_model_with_deepspeed(loss_closure, train_loader, num_epochs=100, num_iterations_train=100, lr=1e-4):
    deepspeed_config = {
        "train_batch_size": 16,
        "gradient_accumulation_steps": 1,
        "zero_optimization": {
            "stage": 2,
        },
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": lr,
                "betas": [0.9, 0.999],
                "eps": 1e-8
            }
        },
        "ema": {
            "enabled": True,
            "ema_decay": 0.995,
            "ema_fp32": True
        },
        "fp16": {
            "enabled": False
        }
    }

    model, optimizer, _, _ = deepspeed.initialize(
        model=loss_closure,
        optimizer=torch.optim.Adam(loss_closure.parameters(), lr=lr),
        config=deepspeed_config
    )

    train_loader_iter = iter(train_loader)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for _ in tqdm(range(num_iterations_train)):
            try:
                x_0, _ = next(train_loader_iter)
            except StopIteration:
                train_loader_iter = iter(train_loader)
                x_0, _ = next(train_loader_iter)
            x_0 = x_0.to(model.device)

            loss = model(x_0)
            model.backward(loss)
            model.step()

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
    train_flag = True
    load_flag = True
    use_deepspeed = False  # Set to False for non-DeepSpeed training
    batch_size = 4
    num_epochs = 40
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

    if use_deepspeed:
        local_rank = setup_distributed()
    else:
        local_rank = 0
    
    torch.manual_seed(42 + local_rank)
    sample_weights = compute_sample_weights(train_dataset.metadata, train_dataset.hemorrhage_types)

    if use_deepspeed:
        deepspeed.init_distributed()
        train_sampler = WeightedDistributedSampler(weights=sample_weights, num_samples=len(train_dataset), replacement=True)
    else:
        train_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dataset), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=16)

    val_loader = None

    torch.manual_seed(1)
    inds = np.arange(len(test_dataset))
    test_dataset = Subset(test_dataset, inds)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    projector = CTProjector()
    diffusion_model = UnconditionalDiffusionModel()
    loss_closure = DiffusionLossClosure(diffusion_model, patch_size)

    if load_flag:
        try:
            load_diffusion_model_weights(diffusion_model, 'weights/diffusion_model_weights.pth')
            print("Diffusion model weights loaded successfully.")
        except:
            print("Diffusion model weights not found. Training from scratch.")

    if train_flag:
        if use_deepspeed:
            train_diffusion_model_with_deepspeed(loss_closure, train_loader, num_epochs, num_iterations_train, lr)
        else:
            diffusion_model = diffusion_model.to(device)
            loss_closure = loss_closure.to(device)
            train_model(loss_closure, train_loader, num_epochs, num_iterations_train, lr)

    if use_deepspeed:
        if torch.distributed.get_rank() == 0:
            save_diffusion_model_weights(diffusion_model, 'weights/diffusion_model_weights.pth')
    else:
        save_diffusion_model_weights(diffusion_model, 'weights/diffusion_model_weights.pth')

    diffusion_model.to(device)
    diffusion_model.eval()
    evaluate_diffusion_model(diffusion_model, projector, test_loader, num_samples=10)

if __name__ == "__main__":
    main()
