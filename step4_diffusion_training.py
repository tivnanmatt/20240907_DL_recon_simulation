
import torch
import torch.nn as nn
import numpy as np

from diffusers import UNet2DModel

from torch_ema import ExponentialMovingAverage

from tqdm import tqdm

from step2_dataset_dataloader import RSNA_Intracranial_Hemorrhage_Dataset

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class DiffusionModel(nn.Module):
    def __init__(self):

        super(DiffusionModel, self).__init__()
        
        self.unet = UNet2DModel(
            sample_size=256,
            in_channels=1,
            out_channels=1,
            center_input_sample=False,
            time_embedding_type='positional',
            freq_shift=0,
            flip_sin_to_cos=True,
            down_block_types=('DownBlock2D', 'AttnDownBlock2D', 'AttnDownBlock2D', 'AttnDownBlock2D'),
            up_block_types=('AttnUpBlock2D', 'AttnUpBlock2D', 'AttnUpBlock2D', 'UpBlock2D'),
            block_out_channels=(32, 64, 128, 256),
            layers_per_block=2,
            mid_block_scale_factor=1,
            downsample_padding=1,
            downsample_type='conv',
            upsample_type='conv',
            dropout=0.0,
            act_fn='silu',
            attention_head_dim=None,
            norm_num_groups=32,
            attn_norm_num_groups=None,
            norm_eps=1e-05,
            resnet_time_scale_shift='default',
            add_attention=True,
            class_embed_type=None,
            num_class_embeds=None,
            num_train_timesteps=None
        )

    def HU_to_SU(self,x):
        return x / 1000.0
    
    def SU_to_HU(self,x):
        return x * 1000.0

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
    
    def estimate_x_0_given_x_t(self, x_t, t):
        assert isinstance(t, torch.Tensor)
        return x_t - self.unet(x_t, t.view(-1))[0] * torch.sqrt(t.view(-1,1,1,1))
    
    def estimate_score_given_x_t(self, x_t, t):
        mu = self.estimate_x_0_given_x_t(x_t, t)
        sigma2 = t
        return -(x_t - mu) / sigma2
    
    def sample_x_t_minus_dt_given_x_t(self, x_t, t, dt, mode='sde'):
        assert mode in ['sde', 'ode']
        assert isinstance(t, torch.Tensor)
        assert isinstance(dt, torch.Tensor)
        if mode == 'sde':
            return x_t - dt * self.estimate_score_given_x_t(x_t, t) + torch.sqrt(dt) * torch.randn_like(x_t)
        else:
            return x_t - dt * 0.5 * self.estimate_score_given_x_t(x_t, t)

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
    
    def train_diffusion_model(  self, 
                train_loader, 
                val_loader=None, 
                time_sampler=None, 
                T=0.0001, 
                num_epochs=100, 
                num_iterations_train=100,
                num_iterations_val=10, 
                lr=2e-4,
                device='cuda'):
        
        self.to(device)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        ema = ExponentialMovingAverage(self.parameters(), decay=0.95)  # Exponential moving average for stabilizing training
        criterion = nn.MSELoss()

        if time_sampler is None:
            time_sampler = lambda batch_size: T * (torch.rand(batch_size, 1)**2.0)

        for iEpoch in range(num_epochs):
            self.train()
            train_loader_iter = iter(train_loader)
            train_loss = 0
            for iIteration in tqdm(range(num_iterations_train)):
                try:
                    x_0 = next(train_loader_iter).to(device)
                except StopIteration:
                    train_loader_iter = iter(train_loader)
                    x_0 = next(train_loader_iter).to(device)

                x_0 = self.HU_to_SU(x_0)
                t = time_sampler(x_0.size(0)).to(device)
                x_t = self.sample_x_t_given_x_0(x_0, t)
                x_0_hat = self.estimate_x_0_given_x_t(x_t, t)
                x_0 = self.SU_to_HU(x_0)
                x_t = self.SU_to_HU(x_t)
                x_0_hat = self.SU_to_HU(x_0_hat)
                loss = criterion(x_0_hat, x_0)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ema.update()
                train_loss += loss.item()
            
            train_loss /= num_iterations_train
            train_loss = np.sqrt(train_loss) # RMSE loss
            print(f'Epoch {iEpoch}, Training Loss: {train_loss}')

            if val_loader is not None:
                self.eval()
                val_loader_iter = iter(val_loader)
                val_loss = 0
                with torch.no_grad():
                    for i in tqdm(range(num_iterations_val)):
                        try:
                            x_0 = next(val_loader_iter).to(device)
                        except StopIteration:
                            val_loader_iter = iter(val_loader)
                            x_0 = next(val_loader_iter).to(device)
                        x_0 = self.HU_to_SU(x_0)
                        t = time_sampler(x_0.size(0)).to(device)
                        x_t = self.sample_x_t_given_x_0(x_0, t)
                        x_0_hat = self.estimate_x_0_given_x_t(x_t, t)
                        x_0 = self.SU_to_HU(x_0)
                        x_t = self.SU_to_HU(x_t)
                        x_0_hat = self.SU_to_HU(x_0_hat)
                        loss = criterion(x_0_hat, x_0)
                        val_loss += loss.item()
                val_loss /= num_iterations_val
                val_loss = np.sqrt(val_loss) # RMSE loss
                print(f'Epoch {iEpoch}, Validation Loss: {val_loss}')

                torch.save(self.state_dict(), 'weights/diffusion_model_weights.pth')




def compute_sample_weights(metadata, hemorrhage_types):
    class_counts = metadata[hemorrhage_types].sum(axis=0).to_numpy()
    class_weights = 1.0 / class_counts
    sample_weights_matrix = metadata[hemorrhage_types].to_numpy() * class_weights
    sample_weights = sample_weights_matrix.sum(axis=1)
    return sample_weights


def main():

    # Dataset paths
    csv_file = 'data/stage_2_train_reformat.csv'
    image_folder = '/data/rsna-intracranial-hemorrhage-detection/stage_2_train/'

    loadFlag=True
    trainFlag=True

    # Hyperparameters
    batch_size = 8
    num_epochs = 50
    num_iterations_train = 10
    num_iterations_val = 10
    lr = 2e-4
    
    # Load the dataset
    full_dataset = RSNA_Intracranial_Hemorrhage_Dataset(csv_file, image_folder)

    # Split dataset into train, validation, and test sets
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    train_indices, temp_indices = train_test_split(indices, test_size=0.3, random_state=42)
    val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    # Compute sample weights for balanced training
    sample_weights = compute_sample_weights(full_dataset.metadata, full_dataset.hemorrhage_types)
    train_sampler = torch.utils.data.WeightedRandomSampler(weights=sample_weights[train_indices], num_samples=len(train_indices), replacement=True)
    val_sampler = torch.utils.data.WeightedRandomSampler(weights=sample_weights[val_indices], num_samples=len(val_indices), replacement=True)

    def image_only_collate_fn(batch):
        # Extract only images from the batch
        images = [item[0] for item in batch]
        return torch.stack(images)

    # Dataloaders with custom collate function to return only images
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=image_only_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, collate_fn=image_only_collate_fn)

    # Initialize the diffusion model
    diffusion_model = DiffusionModel().to('cuda' if torch.cuda.is_available() else 'cpu')

    if loadFlag:
        # Load the diffusion model's weights
        try:
            diffusion_model.load_state_dict(torch.load('weights/diffusion_model_weights.pth'))
            print("Diffusion model weights loaded successfully.")
        except:
            print("Diffusion model weights not found. Training from scratch.")

    if trainFlag:

        # Train the diffusion model
        diffusion_model.train_diffusion_model(
            train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            num_iterations_train=num_iterations_train,
            num_iterations_val=num_iterations_val,
            lr=lr
        )

    # Save the diffusion model's weights
    torch.save(diffusion_model.state_dict(), 'weights/diffusion_model_weights.pth')
    print("Diffusion model weights saved successfully.")

if __name__ == "__main__":
    main()