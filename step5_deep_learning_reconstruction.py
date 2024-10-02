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

# Device handling function
def get_device(device_input):
    if isinstance(device_input, list):
        device_ids = device_input
        device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")
    else:
        device_ids = None
        device = torch.device(device_input)
    return device, device_ids



def check_tensor_device(tensor, name="Tensor"):
    print(f"{name} is on {tensor.device}")



# Updated DeepLearningReconstructor with registered buffer
class DeepLearningReconstructor(nn.Module):
    def __init__(self):
        super(DeepLearningReconstructor, self).__init__()

        block_out_channels = (32, 64, 128, 256)
        layers_per_block = 4

        self.unet = UNet2DModel(
            sample_size=None,
            in_channels=32,
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

        self.singular_values_list = torch.linspace(0, 3000, 32).long()

    def forward(self, x_tilde):
        t = torch.zeros(x_tilde.shape[0], device=x_tilde.device)
        x_hat = self.unet(x_tilde, t)[0]
        return x_hat


# Loss closure module
class CTReconstructionLossClosure(nn.Module):
    def __init__(self, projector, reconstructor, patch_size=256, brain_weight=0.95):
        super(CTReconstructionLossClosure, self).__init__()
        self.projector = projector
        self.reconstructor = reconstructor
        self.patch_size = patch_size
        self.brain_weight = brain_weight
        self.criterion = nn.MSELoss()

    def forward(self, phantom_batch):
        phantom = phantom_batch.float()
        batch_size = phantom.shape[0]

        device = phantom.device

        brain_mask = torch.logical_and(phantom > 0.0, phantom < 80.0)

        phantom = HU_to_attenuation(phantom)

        sinogram = self.projector.forward_project(phantom)

        singular_values_list = self.reconstructor.singular_values_list

        x_tilde_components = self.projector.pseudoinverse_reconstruction(
            sinogram, singular_values_list
        )

        pseudoinverse = torch.sum(x_tilde_components, dim=1, keepdim=True)

        # Extract patches
        patch_size = self.patch_size
        pseudoinverse_patches = torch.zeros(batch_size, 32, patch_size, patch_size, device=device)
        phantom_patches = torch.zeros(batch_size, 1, patch_size, patch_size, device=device)
        brain_mask_patches = torch.zeros(batch_size, 1, patch_size, patch_size, dtype=torch.bool, device=device)

        for i in range(batch_size):
            if patch_size == 256:
                iRow = 0
                iCol = 0
            else:
                iRow = np.random.randint(0, 256 - patch_size)
                iCol = np.random.randint(0, 256 - patch_size)
            pseudoinverse_patches[i] = x_tilde_components[i, :, iRow:iRow+patch_size, iCol:iCol+patch_size]
            phantom_patches[i] = phantom[i, :, iRow:iRow+patch_size, iCol:iCol+patch_size]
            brain_mask_patches[i] = brain_mask[i, :, iRow:iRow+patch_size, iCol:iCol+patch_size]

        reconstruction_patches = self.reconstructor(pseudoinverse_patches)

        phantom_patches = attenuation_to_HU(phantom_patches)
        reconstruction_patches = attenuation_to_HU(reconstruction_patches)

        # Extract center region
        patch_margin = patch_size // 4
        patch_margin = np.clip(patch_margin, 0, 32)
        phantom_patches = phantom_patches[:, :, patch_margin:-patch_margin, patch_margin:-patch_margin]
        reconstruction_patches = reconstruction_patches[:, :, patch_margin:-patch_margin, patch_margin:-patch_margin]
        brain_mask_patches = brain_mask_patches[:, :, patch_margin:-patch_margin, patch_margin:-patch_margin]

        # Compute loss
        loss = (1 - self.brain_weight) * self.criterion(reconstruction_patches, phantom_patches)
        if torch.any(brain_mask_patches):
            loss += self.brain_weight * self.criterion(
                reconstruction_patches[brain_mask_patches],
                phantom_patches[brain_mask_patches]
            )

        return loss
    



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

    ema = ExponentialMovingAverage(reconstructor.parameters(), decay=0.95)

    train_loader_iter = iter(train_loader)
    
    for epoch in range(num_epochs):
        reconstructor.train()
        train_loss = 0

        for _ in tqdm(range(num_iterations_train)):
            try:
                phantom_batch, _ = next(train_loader_iter)
            except StopIteration:
                train_loader_iter = iter(train_loader)
                phantom_batch, _ = next(train_loader_iter)

            batch_size = phantom_batch.shape[0]

            # Move phantom_batch to default device
            phantom_batch = phantom_batch.to(default_device)

            optimizer.zero_grad()
            loss = loss_closure(phantom_batch)
            loss = loss.mean()  # In case of DataParallel
            loss.backward()
            optimizer.step()
            ema.update()

            train_loss += loss.item()

        # Report RMSE
        print(f'Epoch {epoch + 1}/{num_epochs}, Training RMSE (HU): {np.sqrt(train_loss / num_iterations_train)}')

        # Validation loop using loss closure
        if val_loader is not None:
            reconstructor.eval()
            val_loss = 0
            val_loader_iter = iter(val_loader)

            with torch.no_grad():
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

        # Save the optimizer state
        torch.save(optimizer.state_dict(), 'weights/deep_learning_reconstructor_optimizer.pth')

        # Save the model after each epoch
        save_reconstructor(reconstructor, 'weights/deep_learning_reconstructor.pth')

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

# Evaluation function
def evaluate_reconstructor(projector, reconstructor, test_loader, num_iterations=1, device_input='cuda'):
    device, device_ids = get_device(device_input)
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

    train_flag = False  # Set to True to enable training
    evaluate_flag = True  # Set to True to run evaluation after training
    load_flag = True
    device_input = [0, 1, 2, 3]  # For multi-GPU
    # device_input = 'cuda'  # For single GPU
    batch_size = 36
    num_epochs = 2  # Adjust as needed
    num_iterations_train = 5
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

    def compute_sample_weights(metadata, hemorrhage_types):
        class_counts = metadata[hemorrhage_types].sum(axis=0).to_numpy()
        class_weights = 1.0 / class_counts
        sample_weights_matrix = metadata[hemorrhage_types].to_numpy() * class_weights
        sample_weights = sample_weights_matrix.sum(axis=1)
        return sample_weights

    sample_weights = compute_sample_weights(train_dataset.metadata, train_dataset.hemorrhage_types)
    train_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dataset), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=32)

    sample_weights = compute_sample_weights(val_dataset.metadata, val_dataset.hemorrhage_types)
    val_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(val_dataset), replacement=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)

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
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import WeightedRandomSampler
from diffusers import AutoencoderKL, UNet2DConditionModel

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from step4_iterative_reconstruction import CTProjector, LinearLogLikelihood, HU_to_attenuation, attenuation_to_HU



import matplotlib.pyplot as plt  # Import matplotlib for plotting

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class UnconditionalDiffusionModel(nn.Module):
    def __init__(self):

        super(UnconditionalDiffusionModel, self).__init__()

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
        
        return c_skip(t).view(-1,1,1,1) * x_t + c_out(t).view(-1,1,1,1) * self.unet(x_t, t.view(-1))[0]

    def predict_score_given_x_t(self, x_t, t):
        mu = self.predict_x_0_given_x_t(x_t, t)
        sigma2 = t.view(-1,1,1,1)
        return -(x_t - mu) / sigma2
    
    def sample_x_t_minus_dt_given_x_t(self, x_t, t, dt, mode='sde', likelihood_score_fn=None):
        assert mode in ['sde', 'ode']
        assert isinstance(t, torch.Tensor)
        assert isinstance(dt, torch.Tensor)

        prior_score = self.predict_score_given_x_t(x_t, t)
        
        if likelihood_score_fn is None:
            score = prior_score
        else:
            sigma2 = t.view(-1,1,1,1)
            x_0_hat = x_t + sigma2 * prior_score
            likelihood_score = likelihood_score_fn(x_0_hat)
            # likelihood_score = likelihood_score_fn(x_t)
            
            posterior_score = prior_score + likelihood_score
            score = posterior_score
            # score = likelihood_score

        # flip the sign to apply the standard Anderson formula
        dt = -dt

        if mode == 'sde':
            return x_t - dt * score + torch.sqrt(torch.abs(dt)) * torch.randn_like(x_t)
        elif mode == 'ode':
            return x_t - dt * 0.5 * score

    def sample_x_t_minus_delta_t_given_x_t(self, x_t, t, delta_t, mode='sde', num_steps=16, likelihood_score_fn=None, timesteps=None):
        assert mode in ['sde', 'ode']
        if timesteps is None:
            timesteps = torch.linspace(t.item(), t.item() - delta_t.item(), num_steps+1).to(device)
        x = x_t
        for i in range(num_steps):
            print(f'Sampling step {i}/{num_steps}, t={timesteps[i]}, isnan={torch.isnan(x).any()}')
            dt = timesteps[i] - timesteps[i+1]
            dt = dt.unsqueeze(0).to(device)
            if mode == 'sde':
                x = self.sample_x_t_minus_dt_given_x_t(x, timesteps[i].unsqueeze(0), dt, likelihood_score_fn=likelihood_score_fn)
            else:
                x = self.sample_x_t_minus_dt_given_x_t(x, timesteps[i].unsqueeze(0), dt, mode='ode', likelihood_score_fn=likelihood_score_fn)
        return x
    
    def sample_x_0_given_x_t(self, x_t, t, mode='sde', num_steps=1024, likelihood_score_fn=None, timesteps=None):
        delta_t = t.clone()
        return self.sample_x_t_minus_delta_t_given_x_t(x_t, t, delta_t, mode=mode, num_steps=num_steps, likelihood_score_fn=likelihood_score_fn, timesteps=timesteps)

def HU_to_SU(x):
    return x / 1000

def SU_to_HU(x):
    return x * 1000

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
    
    assert isinstance(diffusion_model, UnconditionalDiffusionModel)

    optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=lr)
    ema = ExponentialMovingAverage(diffusion_model.parameters(), decay=0.995)  # Exponential moving average for stabilizing training
    criterion = nn.MSELoss()

    if time_sampler is None:
        time_sampler = lambda batch_size: T * (torch.rand(batch_size, 1)**2.0)
    
    train_loader_iter = iter(train_loader)
    for iEpoch in range(num_epochs):
        diffusion_model.train()
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


            # t = time_sampler(x_0.size(0)).to(device)

            # forward diffusion
            t = time_sampler(x_0.size(0)).to(device)
            x_t = diffusion_model.sample_x_t_given_x_0(x_0, t)
            
            # reverse diffusion predictor
            x_0_hat = x_0.clone()
            x_0_hat = diffusion_model.predict_x_0_given_x_t(x_t, t)           

                
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

                    t = time_sampler(x_0.size(0)).to(device)*.01

                    x_t = diffusion_model.sample_x_t_given_x_0(x_0, t)
                    x_0_hat = diffusion_model.predict_x_0_given_x_t(x_t, t)

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

def evaluate_diffusion_model(diffusion_model, test_loader, projector, num_samples=5):
    assert isinstance(diffusion_model, UnconditionalDiffusionModel)
    diffusion_model.eval()
    test_loader_iter = iter(test_loader)


    assert isinstance(projector, CTProjector)

    with torch.no_grad():
        for i in range(num_samples):
            x_0, _ = next(test_loader_iter)
            if i >= 5:
                break  # Display first 5 samples

            x_0 = x_0.to(device)
            x_0_HU = x_0.clone()  # Keep original for display

            sinogram = projector.forward_project(HU_to_attenuation(x_0))
            pseudoinverse = projector.pseudoinverse_reconstruction(sinogram, singular_values=[3000])
            pseudoinverse_HU = attenuation_to_HU(pseudoinverse)
            pseudoinverse_SU = HU_to_SU(pseudoinverse_HU)


            log_likelihood = LinearLogLikelihood(sinogram, projector, noise_variance=0.1)
            # log_likelihood = LinearLogLikelihood(sinogram, projector, noise_variance=1.0)

            def likelihood_score_fn(x):
                x = SU_to_HU(x)
                x = HU_to_attenuation(x)
                likelihood_score = -log_likelihood.gradient(x)
                likelihood_score = attenuation_to_HU(likelihood_score, scaleOnly=True)
                likelihood_score = HU_to_SU(likelihood_score)
                return likelihood_score

            x_0 = HU_to_SU(x_0)

            t = torch.tensor([0.1], device=device)

            # x_t = diffusion_model.sample_x_t_given_x_0(x_0, t)
            x_t = diffusion_model.sample_x_t_given_x_0(pseudoinverse_SU, t)
            
            num_steps = 1024
            timesteps = (torch.linspace(1.0, 0.0, num_steps+1).to(device)**2.0)*t.item()

            x_0_hat = diffusion_model.sample_x_0_given_x_t(x_t, t, mode='sde', num_steps=num_steps, timesteps=timesteps, likelihood_score_fn=likelihood_score_fn)

            x_0_hat = SU_to_HU(x_0_hat)

            # Noisy image at t=1.0
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

            plt.savefig(f'evaluation_{i}_bone.png')

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

            plt.savefig(f'evaluation_{i}_brain.png')

            plt.close('all')



def main():

    train_flag = False
    load_flag = True
    multiGPU_flag = False
    device_ids = [1,2,3]
    batch_size = 32
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

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    def log_likelihood(y, x):
        return -torch.nn.functional.mse_loss(x, y, reduction='none').sum(dim=(1,2,3))

    # Initialize the diffusion model
    diffusion_model = UnconditionalDiffusionModel().to(device)

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
            val_loader=None,
            num_epochs=num_epochs,
            num_iterations_train=num_iterations_train,
            num_iterations_val=num_iterations_val,
            lr=lr
        )

        print("Training complete. Saving diffusion model weights.")
        save_diffusion_model_weights(diffusion_model, 'weights/diffusion_model_weights.pth')

    projector = CTProjector().to(device)

    # Evaluate the diffusion model
    evaluate_diffusion_model(diffusion_model, test_loader, projector)

if __name__ == "__main__":
    main()