
import torch
import torch.nn as nn

from diffusers import UNet2DModel

from torch_ema import ExponentialMovingAverage

from tqdm import tqdm

from step2_dataset_dataloader import RSNA_Intracranial_Hemorrhage_Dataset

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

import time

import matplotlib.pyplot as plt

# make it so that device 3 is the only visible GPU
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="2"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def forward_project(image):
    assert isinstance(image, torch.Tensor)
    assert len(image.shape) == 4
    batch_size = image.shape[0]
    assert image.shape[1] == 1
    assert image.shape[2] == 256
    assert image.shape[3] == 256
    x = x.view(batch_size, 256*256)
    VT_x = torch.tensordot(V.T, x, dims=1).view(batch_size, 72*375)
    S_VT_x = S.view(1, 72*375) * VT_x
    sinogram = torch.tensordot(U, S_VT_x, dims=1).view(batch_size, 1, 72, 375)
    return sinogram

def back_project(sinogram):
    assert isinstance(sinogram, torch.Tensor)
    assert len(sinogram.shape) == 4
    batch_size = sinogram.shape[0]
    assert sinogram.shape[1] == 1
    assert sinogram.shape[2] == 72
    assert sinogram.shape[3] == 375
    y = sinogram.view(batch_size, 72*375)
    UT_y = torch.tensordot(U.T, y, dims=1).view(batch_size, 256*256)
    x = torch.tensordot(V, UT_y, dims=1).view(batch_size, 1, 256, 256)
    return x

def pinv_recon(sinogram, singular_values=None):
    assert isinstance(sinogram, torch.Tensor)
    assert len(sinogram.shape) == 4
    batch_size = sinogram.shape[0]
    assert sinogram.shape[1] == 1
    assert sinogram.shape[2] == 72
    assert sinogram.shape[3] == 375
    y = sinogram.view(batch_size, 72*375)
    UT_y = torch.tensordot(U.T, y, dims=1).view(batch_size, 256*256)
    invS = 1.0 / S
    if singular_values is not None:
        invS[:, singular_values:] = 0.0
    S_UT_y = UT_y * invS
    x = torch.tensordot(V, S_UT_y, dims=1).view(batch_size, 1, 256, 256)
    return x




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
    def __init__(self, measurements, noise_variance=1.0):
        super(LinearLogLikelihood, self).__init__()
        self.measurements = measurements
        self.noise_variance = noise_variance
    def forward(self, image, sinogram=None):
        if sinogram is None:
            sinogram = forward_project(image)
        residual = sinogram - self.measurements
        loss = 0.5 * torch.sum(residual**2) / self.noise_variance
        return loss
    def gradient(self, image, sinogram=None):
        if sinogram is None:
            sinogram = forward_project(image)
        residual = sinogram - self.measurements
        gradient = back_project(residual) / self.noise_variance
        return gradient
    def hessian(self, image, image_input):
        return back_project(forward_project(image_input)) / self.noise_variance
    
class QuadraticSmoothnessLogPrior(ReconstructionLossTerm):
    def __init__(self, beta=1.0):
        super(QuadraticSmoothnessLogPrior, self).__init__()
        self.beta = beta
    def laplacian(self, image):
        laplacian = torch.zeros_like(image)
        laplacian += 4*image
        laplacian -= torch.roll(image, 1, 0)
        laplacian -= torch.roll(image, -1, 0)
        laplacian -= torch.roll(image, 1, 1)
        laplacian -= torch.roll(image, -1, 1)
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
    def hessian(self, image,image_input):
        hessian = self.beta * self.laplacian(image_input)
        return hessian

class HuberPenalty(ReconstructionLossTerm):
    def __init__(self, beta=1.0, delta=1.0):
        super(HuberPenalty, self).__init__()
        self.beta = beta
        self.delta = delta

    def laplacian(self, image):
        laplacian = torch.zeros_like(image)
        laplacian += 4 * image
        laplacian -= torch.roll(image, 1, 0)
        laplacian -= torch.roll(image, -1, 0)
        laplacian -= torch.roll(image, 1, 1)
        laplacian -= torch.roll(image, -1, 1)
        return laplacian

    def forward(self, image, laplacian=None):
        if laplacian is None:
            laplacian = self.laplacian(image)
        
        abs_laplacian = torch.abs(laplacian)
        
        # Apply quadratic penalty for |x| <= delta and linear penalty for |x| > delta
        quadratic_region = abs_laplacian <= self.delta
        linear_region = abs_laplacian > self.delta
        
        # Quadratic part: 0.5 * beta * x^2
        quadratic_loss = 0.5 * self.beta * laplacian[quadratic_region] ** 2
        
        # Linear part: beta * delta * (|x| - 0.5 * delta)
        linear_loss = self.beta * self.delta * (abs_laplacian[linear_region] - 0.5 * self.delta)
        
        # Total loss: sum of quadratic and linear regions
        total_loss = torch.sum(quadratic_loss) + torch.sum(linear_loss)
        
        return total_loss

    def gradient(self, image, laplacian=None):
        if laplacian is None:
            laplacian = self.laplacian(image)
        
        abs_laplacian = torch.abs(laplacian)
        
        # Apply gradients according to the two regions
        quadratic_region = abs_laplacian <= self.delta
        linear_region = abs_laplacian > self.delta
        
        gradient = torch.zeros_like(laplacian)
        
        # Gradient for quadratic region: beta * x
        gradient[quadratic_region] = self.beta * laplacian[quadratic_region]
        
        # Gradient for linear region: beta * delta * sign(x)
        gradient[linear_region] = self.beta * self.delta * torch.sign(laplacian[linear_region])
        
        return gradient

    def hessian(self, image, image_input):
        abs_laplacian = torch.abs(image_input)
        
        # Hessian is defined only for quadratic region since it's 0 for linear region
        quadratic_region = abs_laplacian <= self.delta
        
        hessian = torch.zeros_like(image_input)
        
        # Hessian for quadratic region is beta (since second derivative of x^2 is constant)
        hessian[quadratic_region] = self.beta
        
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
        return  (image)*0.1/1000.0
    else:
        return  (image + 1000.0)*0.1/1000.0

def attenuation_to_HU(image, scaleOnly=False):
    if scaleOnly:
        return  image*1000.0/0.1
    else:
        return  image*1000.0/0.1 - 1000.0

def plot_reconstructions(vmin, vmax, filename, phantom, sinogram, pinv_reconstruction, reconstruction_quadratic, reconstruction_huber):
    phantom = attenuation_to_HU(phantom)
    pinv_reconstruction = attenuation_to_HU(pinv_reconstruction)
    reconstruction_quadratic = attenuation_to_HU(reconstruction_quadratic)
    reconstruction_huber = attenuation_to_HU(reconstruction_huber)

    plt.figure(figsize=(36,6))
    plt.subplot(1,5,1)
    plt.imshow(phantom.cpu().numpy(), cmap='gray', vmin=vmin, vmax=vmax)
    plt.title('Phantom')
    plt.axis('off')
    plt.subplot(1,5,2)
    plt.imshow(sinogram.cpu().numpy(), cmap='gray')
    plt.gca().set_aspect('auto')
    plt.title('Sinogram')
    plt.axis('off')
    plt.subplot(1,5,3)
    plt.imshow(pinv_reconstruction.detach().cpu().numpy(), cmap='gray', vmin=vmin, vmax=vmax)
    plt.title('Filtered Backprojection Reconstruction')
    plt.axis('off')
    plt.subplot(1,5,4)
    plt.imshow(reconstruction_quadratic.detach().cpu().numpy(), cmap='gray', vmin=vmin, vmax=vmax)
    plt.title('Quadratic Penalized Likelihood Iterative Reconstruction')
    plt.axis('off')
    plt.subplot(1,5,5)
    plt.imshow(reconstruction_huber.cpu().detach().numpy(), cmap='gray', vmin=vmin, vmax=vmax)
    plt.title('Huber Penalized Likelihood Iterative Reconstruction')
    plt.axis('off')
    plt.savefig(f'./figures/{filename}', dpi=300)
    plt.close('all')

    phantom = HU_to_attenuation(phantom)
    pinv_reconstruction = HU_to_attenuation(pinv_reconstruction)
    reconstruction_quadratic = HU_to_attenuation(reconstruction_quadratic)
    reconstruction_huber = HU_to_attenuation(reconstruction_huber)

def main():
    # Dataset paths
    csv_file = 'data/stage_2_train_reformat.csv'
    image_folder = '../../data/rsna-intracranial-hemorrhage-detection/stage_2_train/'
    
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

    # Dataloaders
    batch_size = 1
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


    num_iterations = 100

    train_loader_iter = iter(train_loader)

    # Simulate CT measurements and iterative reconstruction
    for i in tqdm(range(num_iterations)):

        try:
            phantom, _ = next(train_loader_iter)
        except StopIteration:
            train_loader_iter = iter(train_loader)
            phantom, _ = next(train_loader_iter)

        phantom = phantom.to(device).view(256,256).float()
        phantom[phantom < -1000.0] = -1000.0
        phantom = HU_to_attenuation(phantom)
            
        print(f"Processing batch {i+1}/{num_iterations}")

        # Simulate forward projection and sinogram with Poisson noise
        I0 = 1e5
        t0 = time.time()
        sinogram = forward_project(phantom)
        photon_counts = I0 * torch.exp(-sinogram)
        photon_counts = torch.poisson(photon_counts)
        sinogram = -torch.log((photon_counts + 1) / I0)
        t1 = time.time()
        print(f'Elapsed time to forward project = {t1 - t0:.4f}s')

        # Pseudo-inverse reconstruction
        t0 = time.time()
        pinv_reconstruction = pinv_recon(sinogram, singular_values=1000)
        t1 = time.time()
        print(f'Elapsed time to pseudo-inverse reconstruct = {t1 - t0:.4f}s')

        # Quadratic Penalized Likelihood Iterative Reconstruction
        log_likelihood = LinearLogLikelihood(sinogram, noise_variance=1.0)

        pinv_reconstruction = iterative_reconstruction_gradient_descent(pinv_reconstruction*0, 
                                                                             [log_likelihood], 
                                                                             num_iterations=500, 
                                                                             step_size=1e-2, 
                                                                             verbose=True)        
        
        log_prior_quadratic = QuadraticSmoothnessLogPrior(beta=50.0)
        t0 = time.time()
        reconstruction_quadratic = iterative_reconstruction_gradient_descent(pinv_reconstruction*0, 
                                                                             [log_likelihood, log_prior_quadratic], 
                                                                             num_iterations=500, 
                                                                             step_size=1e-2, 
                                                                             verbose=True)
        t1 = time.time()
        print(f'Elapsed time for quadratic penalized likelihood reconstruction = {t1 - t0:.4f}s')

        # Huber Penalized Likelihood Iterative Reconstruction
        log_prior_huber = HuberPenalty(beta=500.0, delta=5e-5)
        t0 = time.time()
        reconstruction_huber = iterative_reconstruction_gradient_descent(pinv_reconstruction*0, 
                                                                        [log_likelihood, log_prior_huber], 
                                                                        num_iterations=500, 
                                                                        step_size=1e-2, 
                                                                        verbose=True)
        t1 = time.time()
        print(f'Elapsed time for Huber penalized likelihood reconstruction = {t1 - t0:.4f}s')

        # Save figures
        plot_reconstructions(0.0, 80.0, f'reconstructions_batch_{i}_brain.png', phantom, sinogram, pinv_reconstruction, reconstruction_quadratic, reconstruction_huber)
        plot_reconstructions(-200.0, 1000.0, f'reconstructions_batch_{i}_bone.png', phantom, sinogram, pinv_reconstruction, reconstruction_quadratic, reconstruction_huber)

        # if i == 1:  # limit to 2 batches for demonstration
        #     break

if __name__ == "__main__":

        
    t0 = time.time()
    U = torch.load('weights/U.pt')
    t1 = time.time()
    print('Elapsed time to load U.pt = %f' % (t1 - t0))

    t0 = time.time()
    S = torch.load('weights/S.pt')
    t1 = time.time()
    print('Elapsed time to load S.pt = %f' % (t1 - t0))

    t0 = time.time()
    V = torch.load('weights/V.pt')
    t1 = time.time()
    print('Elapsed time to load V.pt = %f' % (t1 - t0))

    t0 = time.time()
    U = U.to(device)
    S = S.to(device)
    V = V.to(device)
    t1 = time.time()
    print('Elapsed time to move to GPU = %f' % (t1 - t0))

    main()