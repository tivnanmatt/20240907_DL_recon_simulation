import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from tqdm import tqdm
from step2_dataset_dataloader import RSNA_Intracranial_Hemorrhage_Dataset
from step4_iterative_reconstruction import HU_to_attenuation, attenuation_to_HU
from torch_ema import ExponentialMovingAverage
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from diffusers import UNet2DModel

# visible devices are 1,2,3
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class DeepLearningReconstructor(nn.Module):
    def __init__(self):
        super(DeepLearningReconstructor, self).__init__()

        block_out_channels = (128, 256, 512, 1024)
        # block_out_channels = (16, 32, 64, 128)

        
        # layers_per_block = 4
        layers_per_block = 2

        self.unet = UNet2DModel(
            sample_size=256,
            in_channels=32,  # 32 components from the pseudo-inverse
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
            norm_num_groups=4,
            attn_norm_num_groups=None,
            norm_eps=1e-05,
            resnet_time_scale_shift='default',
            add_attention=True,
            class_embed_type=None,
            num_class_embeds=None,
            num_train_timesteps=None
        )
        # self.U = torch.load('weights/U.pt')
        # self.S = torch.load('weights/S.pt')
        # self.V = torch.load('weights/V.pt')
        # make these parameters
        self.U = nn.Parameter(torch.load('weights/U.pt'))
        self.S = nn.Parameter(torch.load('weights/S.pt'))
        self.V = nn.Parameter(torch.load('weights/V.pt'))

        # make sure they are not trainable
        self.U.requires_grad = False
        self.S.requires_grad = False
        self.V.requires_grad = False

        self.condition_number = 1e2

        self.idxNull = self.S < torch.max(self.S)/ self.condition_number
        index_min_singular_value = torch.max(torch.where(~self.idxNull)[0])

        # Ensure U, S, V are loaded on the correct device (e.g., GPU or CPU)
        self.singular_values_list = torch.linspace(0, index_min_singular_value, 33)[1:].to(torch.int32)

    def forward_project(self, image):
        """
        Simulates the forward projection of the image to generate a sinogram (batch-based).
        """
        assert isinstance(image, torch.Tensor)
        assert len(image.shape) == 4  # Expecting batch, channel, height, width
        batch_size = image.shape[0]
        assert image.shape[1] == 1
        assert image.shape[2] == 256
        assert image.shape[3] == 256

        # Flatten image to 2D for projection
        x = image.view(batch_size, 256 * 256)
        VT_x = torch.tensordot(x, self.V.T, dims=([1],[1])).view(batch_size, self.S.shape[0])
        S_VT_x = self.S.view(1, -1) * VT_x
        sinogram = torch.tensordot(S_VT_x, self.U, dims=([1],[1])).view(batch_size, 1, 72, 375)
        return sinogram

    def back_project(self, sinogram):
        """
        Inverse transformation from sinogram back to image space (batch-based).
        """
        assert isinstance(sinogram, torch.Tensor)
        assert len(sinogram.shape) == 4  # Expecting batch, channel, height, width
        batch_size = sinogram.shape[0]
        assert sinogram.shape[1] == 1
        assert sinogram.shape[2] == 72
        assert sinogram.shape[3] == 375

        # Flatten sinogram to 2D for back projection
        y = sinogram.view(batch_size, 72 * 375)
        UT_y = torch.tensordot(y, self.U.T, dims=([1],[1])).view(batch_size, self.S.shape[0])
        S_UT_y = self.S.view(1, -1) * UT_y
        V_S_UT_y = torch.tensordot(S_UT_y, self.V.T, dims=([1],[1])).view(batch_size, 1, 256, 256)
        AT_y = V_S_UT_y
        return AT_y

    def pseudoinverse_reconstruction(self, sinogram, singular_values=None):
        """
        Performs the pseudo-inverse reconstruction using a list of singular values (batch-based).
        """
        assert isinstance(sinogram, torch.Tensor)
        assert len(sinogram.shape) == 4  # Expecting batch, channel, height, width
        batch_size = sinogram.shape[0]
        assert sinogram.shape[1] == 1
        assert sinogram.shape[2] == 72
        assert sinogram.shape[3] == 375

        # Flatten sinogram to 2D for reconstruction
        y = sinogram.view(batch_size, 72 * 375)

        x_tilde_components = []

        # Handle the singular values and perform the reconstruction
        invS = 1.0 / self.S
        for i in range(len(singular_values)):
            if i == 0:
                sv_min = 0
            else:
                sv_min = singular_values[i - 1]
            sv_max = singular_values[i]
            # sv_min = 0
            sv_max = singular_values[i]

            _U = self.U[:, sv_min:sv_max]
            _S = self.S[sv_min:sv_max]
            _V = self.V[:, sv_min:sv_max]


            idxNull = _S < 1e-4*torch.max(self.S)
            _invS = torch.zeros_like(_S)
            _invS[~idxNull] = 1.0 / _S[~idxNull]
            
            UT_y = torch.tensordot(y, _U.T, dims=([1],[1])).view(batch_size, _S.shape[0])
            S_UT_y = _invS.view(1, -1) * UT_y
            V_S_UT_y = torch.tensordot(S_UT_y, _V, dims=([1],[1])).view(batch_size, 1, 256, 256)

            x_tilde_components.append(V_S_UT_y)

        x_tilde_components = torch.cat(x_tilde_components, dim=1)

        return x_tilde_components

    def forward(self, sinogram, return_psuedoinverse=False):
        """
        Forward pass of the model, which performs:
        1. Pseudo-inverse reconstruction (breaking sinogram into 32 components).
        2. Image processing via U-Net.
        """
        # Step 1: Get 32 components from the pseudo-inverse reconstruction
        x_tilde_components = self.pseudoinverse_reconstruction(sinogram, self.singular_values_list)
        x_tilde = torch.sum(x_tilde_components, dim=1, keepdim=True)
        
        # Step 2: Pass these components through the U-Net for final reconstruction
        t =  torch.zeros([x_tilde_components.shape[0]]).to(x_tilde_components.device)
        x_hat = self.unet(x_tilde_components, t)[0]

        if return_psuedoinverse:
            return x_hat, x_tilde
        else:
            return x_hat

    
    def train_model(self, train_loader, val_loader=None, num_epochs=100, num_iterations_train=100, num_iterations_val=10, lr=2e-4, device='cuda'):
        self.to(device)
        optimizer = Adam(self.parameters(), lr=lr)


        # if the optimizer was saved, load it
        try:
            optimizer.load_state_dict(torch.load('weights/deep_learning_reconstructor_optimizer.pth'))
            print("Optimizer loaded successfully.")
        except FileNotFoundError:
            print("No optimizer state found. Starting from scratch.")





        # ema = ExponentialMovingAverage(self.parameters(), decay=0.99)
        criterion = nn.MSELoss()
        

        for epoch in range(num_epochs):
            self.train()
            train_loss = 0
            train_loader_iter = iter(train_loader)

            for _ in tqdm(range(num_iterations_train)):
                try:
                    phantom, _ = next(train_loader_iter)
                except StopIteration:
                    train_loader_iter = iter(train_loader)
                    phantom, _ = next(train_loader_iter)

                phantom = phantom.to(device).float()
                phantom[phantom < -1000.0] = -1000.0

                brain_mask = torch.logical_and(phantom > 0.0, phantom < 80.0)

                phantom = HU_to_attenuation(phantom)

                # Simulate forward projection and sinogram with Poisson noise
                sinogram = self.forward_project(phantom)  # Updated to use class method
                I0 = 1e5
                photon_counts = I0 * torch.exp(-sinogram)
                photon_counts = torch.poisson(photon_counts)
                noisy_sinogram = -torch.log((photon_counts + 1) / I0)

                # Forward pass: reconstruct using U-Net
                optimizer.zero_grad()
                reconstruction, pseudoinverse = self.forward(noisy_sinogram, return_psuedoinverse=True)

                phantom = attenuation_to_HU(phantom)
                reconstruction = attenuation_to_HU(reconstruction)
                pseudoinverse = attenuation_to_HU(pseudoinverse)

                # Calculate MSE loss
                loss = 0.05*criterion(reconstruction, phantom)
                if torch.any(brain_mask):
                    loss += 0.95*criterion(reconstruction[brain_mask], phantom[brain_mask])
                loss.backward()
                optimizer.step()
                # ema.update()

                train_loss += loss.item()


            # report RMSE
            print(f'Epoch {epoch + 1}/{num_epochs}, Training RMSE (HU): {np.sqrt(train_loss / num_iterations_train)}')

            if val_loader:
                self.eval()
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
                        phantom[phantom < -1000.0] = -1000.0

                        brain_mask = torch.logical_and(phantom > 0.0, phantom < 80.0)

                        phantom = HU_to_attenuation(phantom)

                        sinogram = self.forward_project(phantom)  # Updated to use class method
                        I0 = 1e5
                        photon_counts = I0 * torch.exp(-sinogram)
                        photon_counts = torch.poisson(photon_counts)
                        noisy_sinogram = -torch.log((photon_counts + 1) / I0)

                        reconstruction, pseudoinverse = self.forward(noisy_sinogram, return_psuedoinverse=True)

                        phantom = attenuation_to_HU(phantom)
                        reconstruction = attenuation_to_HU(reconstruction)
                        pseudoinverse = attenuation_to_HU(pseudoinverse)

                        loss = 0.05*criterion(reconstruction, phantom)
                        if torch.any(brain_mask):
                            loss += 0.95*criterion(reconstruction[brain_mask], phantom[brain_mask])

                        val_loss += loss.item()
                        

                print(f'Validation RMSE (HU): {np.sqrt(val_loss / num_iterations_val)}')

            # # Save the model after each epoch
            torch.save(self.state_dict(), 'weights/deep_learning_reconstructor.pth')
            # Save the optimizer state
            torch.save(optimizer.state_dict(), 'weights/deep_learning_reconstructor_optimizer.pth')
        
        # Save the model after training
        torch.save(self.state_dict(), 'weights/deep_learning_reconstructor.pth')
        # Save the optimizer state
        torch.save(optimizer.state_dict(), 'weights/deep_learning_reconstructor_optimizer.pth')

# Main script
def main():

    batch_size = 4
    load_flag = True
    train_flag = True

    csv_file = 'data/stage_2_train_reformat.csv'
    dicom_dir = '../../data/rsna-intracranial-hemorrhage-detection/stage_2_train/'
    full_dataset = RSNA_Intracranial_Hemorrhage_Dataset(csv_file, dicom_dir)

    # Split dataset into train, validation, and test sets
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    train_indices, temp_indices = train_test_split(indices, test_size=0.3, random_state=42)
    val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    reconstructor = DeepLearningReconstructor()

    train_flag = True
    load_flag = True

    if load_flag:
        try:
            reconstructor.load_state_dict(torch.load('weights/deep_learning_reconstructor.pth'))
            print("Model loaded successfully.")
        except FileNotFoundError:
            print("No pre-trained model found. Starting from scratch.")


    # reconstructor.unet = torch.nn.DataParallel(reconstructor.unet, device_ids=[1,2,3])
    # reconstructor.U = torch.nn.DataParallel(reconstructor.U, device_ids=[1,2,3])
    # reconstructor.S = torch.nn.DataParallel(reconstructor.S, device_ids=[1,2,3])
    # reconstructor.V = torch.nn.DataParallel(reconstructor.V, device_ids=[1,2,3])

    if train_flag:
        reconstructor.train_model(train_loader, 
                                  val_loader=val_loader, 
                                  num_epochs=100, 
                                  num_iterations_train=100, 
                                  num_iterations_val=10, 
                                  lr=2e-4, 
                                  device='cuda')

        # Save the trained model
        torch.save(reconstructor.state_dict(), 'weights/deep_learning_reconstructor.pth')

if __name__ == "__main__":
    main()
