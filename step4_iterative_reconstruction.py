# step4_iterative_reconstruction.py

import torch
import torch.nn as nn

from torch_ema import ExponentialMovingAverage

from tqdm import tqdm

from step2_dataset_dataloader import RSNA_Intracranial_Hemorrhage_Dataset

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import WeightedRandomSampler

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

import time

import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CTProjector(nn.Module):
    def __init__(self):
        super(CTProjector, self).__init__()
        self.U = torch.load('weights/U.pt')
        self.S = torch.load('weights/S.pt')
        self.V = torch.load('weights/V.pt')

        # Ensure U, S, V are loaded on the correct device (e.g., GPU or CPU)
        self.U = self.U.to(device)
        self.S = self.S.to(device)
        self.V = self.V.to(device)

        # Make sure they are not trainable
        self.U.requires_grad = False
        self.S.requires_grad = False
        self.V.requires_grad = False

        self.condition_number = 1e2

        self.idxNull = self.S < torch.max(self.S) / self.condition_number
        index_min_singular_value = torch.max(torch.where(~self.idxNull)[0])

        self.singular_values_list = torch.linspace(0, index_min_singular_value, 33)[1:].to(torch.int32)

    def to(self, device):
        self.U = self.U.to(device)
        self.S = self.S.to(device)
        self.V = self.V.to(device)
        return super(CTProjector, self).to(device)

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
        VT_x = torch.tensordot(x, self.V.T, dims=([1], [1])).view(batch_size, self.S.shape[0])
        S_VT_x = self.S.view(1, -1) * VT_x
        sinogram = torch.tensordot(S_VT_x, self.U, dims=([1], [1])).view(batch_size, 1, 72, 375)
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
        UT_y = torch.tensordot(y, self.U.T, dims=([1], [1])).view(batch_size, self.S.shape[0])
        S_UT_y = self.S.view(1, -1) * UT_y
        V_S_UT_y = torch.tensordot(S_UT_y, self.V, dims=([1], [1])).view(batch_size, 1, 256, 256)
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
        if singular_values is None:
            singular_values = [self.S.shape[0]]

        for i in range(len(singular_values)):
            if i == 0:
                sv_min = 0
            else:
                sv_min = singular_values[i - 1]
            sv_max = singular_values[i]

            _U = self.U[:, sv_min:sv_max]
            _S = self.S[sv_min:sv_max]
            _V = self.V[:, sv_min:sv_max]

            idxNull = _S < 1e-4 * torch.max(self.S)
            _invS = torch.zeros_like(_S)
            _invS[~idxNull] = 1.0 / _S[~idxNull]

            UT_y = torch.tensordot(y, _U.T, dims=([1], [1])).view(batch_size, _S.shape[0])
            S_UT_y = _invS.view(1, -1) * UT_y
            V_S_UT_y = torch.tensordot(S_UT_y, _V, dims=([1], [1])).view(batch_size, 1, 256, 256)

            x_tilde_components.append(V_S_UT_y)

        x_tilde_components = torch.cat(x_tilde_components, dim=1)

        return x_tilde_components

class ReconstructionLossTerm(torch.nn.Module):
    def __init__(self):
        super(ReconstructionLossTerm, self).__init__()

    def forward(self, image):
        raise NotImplementedError
    
    def gradient(self, image):
        raise NotImplementedError
    
    def hessian(self, image, image_input):
        raise NotImplementedError

class LinearLogLikelihood(ReconstructionLossTerm):
    def __init__(self, measurements, projector, noise_variance=1.0):
        super(LinearLogLikelihood, self).__init__()
        self.measurements = measurements
        self.noise_variance = noise_variance
        self.projector = projector

    def forward(self, image, sinogram=None):
        if sinogram is None:
            sinogram = self.projector.forward_project(image)
        residual = sinogram - self.measurements
        loss = 0.5 * torch.sum(residual**2) / self.noise_variance
        return loss

    def gradient(self, image, sinogram=None):
        if sinogram is None:
            sinogram = self.projector.forward_project(image)
        residual = sinogram - self.measurements
        gradient = self.projector.back_project(residual) / self.noise_variance
        return gradient

    def hessian(self, image, image_input):
        return self.projector.back_project(self.projector.forward_project(image_input)) / self.noise_variance

class QuadraticSmoothnessLogPrior(ReconstructionLossTerm):
    def __init__(self, beta=1.0):
        super(QuadraticSmoothnessLogPrior, self).__init__()
        self.beta = beta

    def laplacian(self, image):
        laplacian = torch.zeros_like(image)
        laplacian += 4 * image
        laplacian -= torch.roll(image, 1, 2)
        laplacian -= torch.roll(image, -1, 2)
        laplacian -= torch.roll(image, 1, 3)
        laplacian -= torch.roll(image, -1, 3)
        return laplacian

    def forward(self, image, laplacian=None):
        if laplacian is None:
            laplacian = self.laplacian(image)
        loss = 0.5 * self.beta * torch.sum(laplacian**2)
        return loss

    def gradient(self, image, laplacian=None):
        if laplacian is None:
            laplacian = self.laplacian(image)
        gradient = self.beta * laplacian
        return gradient

    def hessian(self, image, image_input):
        hessian = self.beta * self.laplacian(image_input)
        return hessian

def iterative_reconstruction_gradient_descent(image_init, loss_terms, num_iterations=100, step_size=1.0, verbose=True):
    image = image_init.clone().detach().requires_grad_(True)
    prev_gradient_norm = 0.0
    for iteration in tqdm(range(num_iterations)):
        gradient = torch.zeros_like(image)
        for loss_term in loss_terms:
            gradient += loss_term.gradient(image)
        image = image - step_size * gradient
        gradient_norm = torch.norm(gradient)
        if gradient_norm >= prev_gradient_norm:
            step_size = step_size * 0.5
        else:
            step_size = step_size * 1.05

        if step_size < 1e-6:
            break

        if verbose:
            print('Iteration %d, gradient norm = %f, step size = %f' % (iteration, gradient_norm, step_size))
            prev_gradient_norm = gradient_norm
    return image

def HU_to_attenuation(image, scaleOnly=False):
    if scaleOnly:
        return (image) * 0.1 / 1000.0
    else:
        return (image + 1000.0) * 0.1 / 1000.0

def attenuation_to_HU(image, scaleOnly=False):
    if scaleOnly:
        return image * 1000.0 / 0.1
    else:
        return image * 1000.0 / 0.1 - 1000.0

def plot_reconstructions(vmin, vmax, filename, phantom, sinogram, pinv_reconstruction, reconstruction_quadratic):
    phantom = attenuation_to_HU(phantom)
    pinv_reconstruction = attenuation_to_HU(pinv_reconstruction)
    reconstruction_quadratic = attenuation_to_HU(reconstruction_quadratic)

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
    plt.title('Filtered Backprojection Reconstruction')
    plt.axis('off')
    plt.subplot(1, 4, 4)
    plt.imshow(reconstruction_quadratic.detach().cpu().numpy()[0, 0, :, :], cmap='gray', vmin=vmin, vmax=vmax)
    plt.title('Quadratic Penalized Likelihood Iterative Reconstruction')
    plt.axis('off')
    plt.savefig(f'./figures/{filename}', dpi=300)
    plt.close('all')

def main():

    batch_size = 1
    num_patients = 10

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

    # Initialize projector
    projector = CTProjector().to(device)


    train_loader_iter = iter(train_loader)
    # Simulate CT measurements and iterative reconstruction
    for i in tqdm(range(num_patients)):

        try:
            phantom, _ = next(train_loader_iter)
        except StopIteration:
            train_loader_iter = iter(train_loader)
            phantom, _ = next(train_loader_iter)

        phantom = phantom.to(device).float()

        phantom = HU_to_attenuation(phantom)
            
        print(f"Processing batch {i+1}/{num_patients}")

        # Simulate forward projection and sinogram with Poisson noise
        # I0 = 1e10
        t0 = time.time()
        sinogram = projector.forward_project(phantom)
        # photon_counts = I0 * torch.exp(-sinogram)
        # photon_counts = torch.poisson(photon_counts)
        # sinogram = -torch.log((photon_counts + 1) / I0)
        t1 = time.time()
        print(f'Elapsed time to forward project = {t1 - t0:.4f}s')

        # Pseudo-inverse reconstruction
        t0 = time.time()
        pinv_reconstruction = projector.pseudoinverse_reconstruction(sinogram, singular_values=[3000])
        pinv_reconstruction = torch.sum(pinv_reconstruction, dim=1, keepdim=True)
        t1 = time.time()
        print(f'Elapsed time to pseudo-inverse reconstruct = {t1 - t0:.4f}s')

        # Quadratic Penalized Likelihood Iterative Reconstruction
        log_likelihood = LinearLogLikelihood(sinogram, projector, noise_variance=1.0)

        t0 = time.time()
        reconstruction_quadratic = iterative_reconstruction_gradient_descent(
            pinv_reconstruction.clone(),
            [log_likelihood, QuadraticSmoothnessLogPrior(beta=10.0)],
            num_iterations=500,
            step_size=1e-2,
            verbose=True
        )
        t1 = time.time()
        print(f'Elapsed time for quadratic penalized likelihood reconstruction = {t1 - t0:.4f}s')

        # Save figures
        plot_reconstructions(
            0.0, 80.0, f'MBIR_batch_{i}_brain.png',
            phantom, sinogram, pinv_reconstruction, reconstruction_quadratic
        )
        plot_reconstructions(
            -1000.0, 2000.0, f'MBIR_batch_{i}_bone.png',
            phantom, sinogram, pinv_reconstruction, reconstruction_quadratic
        )

if __name__ == "__main__":
    main()
