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
        # Call the initializer of the parent class (nn.Module)
        super(CTProjector, self).__init__()
        U = torch.load('weights/U.pt')
        S = torch.load('weights/S.pt')
        V = torch.load('weights/V.pt')

        # register these as buffers
        self.register_buffer('U', U)
        self.register_buffer('S', S)
        self.register_buffer('V', V)

        # Ensure U, S, V are loaded on the correct device (e.g., GPU or CPU)
        # self.U = self.U.to(device)
        # self.S = self.S.to(device)
        # self.V = self.V.to(device)

        # Prevent the loaded matrices from being updated during training
        # This ensures these tensors are used only for forward pass computation, not for backpropagation
        # Make sure they are not trainable
        self.U.requires_grad = False
        self.S.requires_grad = False
        self.V.requires_grad = False

        # Condition number for truncating singular values. Will be used to identify 'small' singular
        # values that can be ignored during pseudo-inverse reconstruction, as they are likely due to noise.
        self.condition_number = 1e2

        # Identify the singular values that are not 'small' based on the condition number
        # The condition checks if the singular value 'S' is smaller than the maximum singular value divided by the condition number
        # This creates a boolean mask (self.idxNull) that can be used to filter out 'small' singular values
        self.idxNull = self.S < torch.max(self.S) / self.condition_number
        
        # Find the index of the largest singular value that is still considered significant.
        # torch.where(~self.idxNull)[0] returns the indices of the non-zero elements in the boolean mask
        # torch.max() returns the maximum value in the tensor
        index_min_singular_value = torch.max(torch.where(~self.idxNull)[0])

        # Create a list of equally spaced singular value indics, from 1 to the index of the 
        # largest valid singular value. This is used to restrict or select specific singular values
        # for pseudo-inverse reconstruction. torch.linspace generates 33 evenly spaced points from 0 to the
        # maximum index of the valid singular values. The first element is removed to avoid using 0 singular values.
        self.singular_values_list = torch.linspace(0, index_min_singular_value, 33)[1:].to(torch.int32)

    # def to(self, device):
    #     super(CTProjector, self).to(device)
    #     # Ensure that all buffers are moved to the correct device
    #     self.U = self.U.to(device)
    #     self.S = self.S.to(device)
    #     self.V = self.V.to(device)
    #     return self
    
    def forward_project(self, image):
        """
        Simulates the forward projection of the image to generate a sinogram (batch-based).
        Args:
            image (torch.Tensor): The input image tensor of shape (batch-based, 1, 256, 256).

        Returns:
            sinogram (torch.Tensor): The simulated sinogram tensor of shape (batch-based, 1, 72, 375).
            - 72: Number of angles for projection
            - 375: Projection length (based on geometry)
        """
        assert isinstance(image, torch.Tensor) # Ensure that the input image is a tensor
        assert len(image.shape) == 4  # Expecting batch, channel, height, width
        batch_size = image.shape[0]
        assert image.shape[1] == 1 # Validate that the input image tensor has 1 channel (grayscale)
        assert image.shape[2] == 256 # Validate that the input image tensor has a height of 256
        assert image.shape[3] == 256 # Validate that the input image tensor has a width of 256

        # Flatten image to 2D for projection
        # The input image is reshaped from (batch_size, 1, 256, 256) to (batch_size, 256*256).
        # This step prepares the image for matrix operations by converting the 2D spatial dimensions into a 1D vector
        x = image.view(batch_size, 256 * 256)

        # Compute VT_x = V^T * x
        # The transpose of the right singular vectors (V) is multiplied by the flattened image tensor (x).
        # This simulates applying a projection matrix (V) to the input image in the transformed space.
        # VT_x is reshaped to (batch_size, 256) to match the shape of the singular values (S).
        VT_x = torch.tensordot(x, self.V.T, dims=([1], [1])).view(batch_size, self.S.shape[0])
        
        # Apply the singular values to the transformed image
        # Multiply each value in VT_x by the corresponding singular value in S.
        # This step simulates scaling the transformed data by the singular values to account for the 
        # magnitude of the projections
        S_VT_x = self.S.view(1, -1) * VT_x
        
        # Perform a tensor dot product between the scaled projections (S_VT_x) and the matrix U.
        # This simulates transforming the data back to the sinogram space.
        # The result is reshaped to (batch_size, 1, 72, 375) to match the expected sinogram shape.
        sinogram = torch.tensordot(S_VT_x, self.U, dims=([1], [1])).view(batch_size, 1, 72, 375)
        
        # Return the sinogram, which is the forward projeciton of the input image.
        return sinogram

    def back_project(self, sinogram):
        """
        Inverse transformation from sinogram back to image space (batch-based).
        Args:
            sinogram (torch.Tensor): The input sinogram tensor of shape (batch-based, 1, 72, 375)

        Returns:
            AT_y (torch.Tensor): The reconstructed image tensor of shape (batch_size, 1, 256, 256)
        """
        assert isinstance(sinogram, torch.Tensor) # Ensure that the input sinogram is a PyTorch tensor
        assert len(sinogram.shape) == 4  # Expecting batch, channel, height, width
        batch_size = sinogram.shape[0]
        assert sinogram.shape[1] == 1
        assert sinogram.shape[2] == 72
        assert sinogram.shape[3] == 375

        # Flatten sinogram to 2D for back projection. The input sinogram tensor is reshaped from (batch_size, 1, 72, 375) to (batch_size, 72*375).
        # This prepares the sinogram for matrix operations by converting the 2D projection data into a 1D vector.
        y = sinogram.view(batch_size, 72 * 375)

        # Compute UT_y = U^T * y. The transpose of the left singular vectors (U) is multiplied by the flattened sinogram tensor (y).
        # This steps applies the inverse projection matrix (U^T), transforming the sinogram back towrad the original image space.
        # UT_y is reshaped to (batch_size, number of singular values) to match the shape of the singular values (S).
        UT_y = torch.tensordot(y, self.U.T, dims=([1], [1])).view(batch_size, self.S.shape[0])
       
        # Apply the pseudo-inverse of the singular values to the transformed sinogram. 
        # Multiply each value in UT_y by the corresponding singular value in S.
        # This step scales the data in the transformed space by the singular values, simulating the inverse of the forward projection.
        S_UT_y = self.S.view(1, -1) * UT_y
        
        # Apply the final transformation (V matrix)
        # Perform a tensor dot product between the scaled inverse projections (S_UT_y) and the matrix V.
        # This step transforms the data from the singular value space back to the original image space.
        # The result is reshaped to (batch_size, 1, 256, 256) to match the original image shape.
        V_S_UT_y = torch.tensordot(S_UT_y, self.V, dims=([1], [1])).view(batch_size, 1, 256, 256)
        
        # The final reconstructed image is stored in AT_y and returned to the caller
        AT_y = V_S_UT_y
        return AT_y

    def pseudoinverse_reconstruction(self, sinogram, singular_values=None):
        """
        Performs the pseudo-inverse reconstruction using a list of singular values (batch-based).
        Args:
            sinogram (torch.Tensor): The input sinogram tensor of shape (batch-based, 1, 72, 375)
            singular_values (list): A list of indices specifying ranges of singular values to use for the reconstruction.
        
        Returns:
            x_tilde_components (torch.Tensor): The reconstructed image components concatenated together based 
                on the pseudo-inverse reconstruction.  
        
        """
        assert isinstance(sinogram, torch.Tensor)
        assert len(sinogram.shape) == 4  # Expecting batch, channel, height, width
        batch_size = sinogram.shape[0]
        assert sinogram.shape[1] == 1
        assert sinogram.shape[2] == 72
        assert sinogram.shape[3] == 375

        # Flatten sinogram to 2D for reconstruction
        # The input sinogram tensor is reshaped from (batch_size, 1, 72, 375) to (batch_size, 72*375).
        # This prepares the sinogram for matrix operations by converting the 2D projection data into a 1D vector.
        y = sinogram.view(batch_size, 72 * 375)

        x_tilde_components = []

        # Handle the singular values and perform the reconstruction
        # If no singular values are provided, the full set of singular values is used.
        # This means that by default, all singular values are used for the reconstruction.
        if singular_values is None:
            singular_values = [self.S.shape[0]]

        # Loop over the provided range of singular values for reconstruction.
        # This allows reconstruction based on different portions of the singular value spectrum.
        for i in range(len(singular_values)):
            # Set the lower (sv_min) and upper (sv_max) boundary for the current range of singular values
            if i == 0:
                sv_min = 0
            else:
                sv_min = singular_values[i - 1]
            sv_max = singular_values[i]

            # Extract the U, S, and V matrices based on the current range of singular values
            _U = self.U[:, sv_min:sv_max]
            _S = self.S[sv_min:sv_max]
            _V = self.V[:, sv_min:sv_max]

            # Calculate the pseudo-inverse of the singular values. Identify very small singular values
            # and set their inverse to zeroto avoid division by near-zero values
            idxNull = _S < 1e-4 * torch.max(self.S) # Singular values below a threshold
            _invS = torch.zeros_like(_S)            # Initialize the inverse singular values
            _invS[~idxNull] = 1.0 / _S[~idxNull]    # Invert only the non-zero singular values

            # Compute the back-projected image component for the current singular value range.
            # Perform a tensor dot product between the flattened sinogram and the transpose of _U
            # This transforms the sinogram back toward the image space for the current singular vlaue range.
            UT_y = torch.tensordot(y, _U.T, dims=([1], [1])).view(batch_size, _S.shape[0])
            
            # Multiply by the pseudo-inverse singular values (_invS) to apply the inverse transformation
            S_UT_y = _invS.view(1, -1) * UT_y

            # Perform a tensor dot product between the transformed data and _V to project back to image space
            # This yields the reconstructed component for the current singular value range.
            V_S_UT_y = torch.tensordot(S_UT_y, _V, dims=([1], [1])).view(batch_size, 1, 256, 256)

            # Append the reconstructed component to the list of components. Each component corresponds
            # to a different range of singular values used for the reconstruction.
            x_tilde_components.append(V_S_UT_y)

        # Concatenate all components along the channel dimension to form the full reconstructed image
        x_tilde_components = torch.cat(x_tilde_components, dim=1)

        return x_tilde_components
    
    def inverse_hessian(self, image, meas_var=None, reg=None):
        """
        Compute the inverse Hessian of the image using the projector.
        Args:
            image (torch.Tensor): The input image tensor of shape (batch-based, 1, 256, 256)
            reg (float): Regularization parameter to stabilize the inverse Hessian computation
        
        Returns:
            inverse_hessian (torch.Tensor): The computed inverse Hessian of the input image
        """
        # Apply V^T to the image
        VT_x = torch.tensordot(image.view(image.shape[0], 256 * 256), self.V.T, dims=([1], [1])).view(image.shape[0], self.S.shape[0])

        if meas_var is None:
            meas_var = 1.0

        # Define the inverse of singular values squared
        if reg is None:
            invS2 = 1.0 / (self.S**2 / meas_var)
        else:
            invS2 = 1.0 / ((self.S**2 / meas_var) + reg)

        # Apply the inverse of singular values squared to the transformed image
        invS2_VT_x = invS2.view(1, -1) * VT_x

        # Apply V to the result
        V_invS2_VT_x = torch.tensordot(invS2_VT_x, self.V, dims=([1], [1])).view(image.shape[0], 1, 256, 256)

        return V_invS2_VT_x
    
    def null_space(self, image, singular_values=None):
        """
        Compute the null space of the image using V VT.
        Args:
            image (torch.Tensor): The input image tensor of shape (batch-based, 1, 256, 256)
            singular_values (list): A list of indices specifying ranges of singular values to use for the reconstruction.

        Returns:
            null_space (torch.Tensor): The computed null space of the input image
        """
        if singular_values is None:
            singular_values = [self.S.shape[0]]

        assert isinstance(singular_values, list)
        assert len(singular_values) ==1, 'for now only one singular value range is supported'
        sv_min = 0
        sv_max = singular_values[0]


        # print("DEBUG: self.V.device", self.V.device)
        # print("DEBUG: image.device", image.device)
        VT_x = torch.tensordot(image.view(image.shape[0], 256 * 256), self.V.T, dims=([1], [1])).view(image.shape[0], self.S.shape[0])
        range_space_transfer = torch.zeros_like(self.S)
        range_space_transfer[:sv_max] = 1.0 
        VT_x = VT_x * range_space_transfer
        range_space_x = torch.tensordot(VT_x, self.V, dims=([1], [1])).view(image.shape[0], 1, 256, 256)
        null_space_x = image - range_space_x

        return null_space_x

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
        """
        Initializes the LinearLogLikelihood loss term.
        Args:
            measurements (torch.Tensor): The measured sinogram tensor of shape (batch-based, 1, 72, 375) 
                that the image is being compared against.
            projector (CTProjector): The CTProjector module used for forward and back projection.
            noise_variance (float): The variance of the Gaussian noise in the measurements. Default is 1.0.
        """
        
        super(LinearLogLikelihood, self).__init__()
        self.measurements = measurements        # Store the measured sinograms (ground-truth projections)
        self.noise_variance = noise_variance    # Store the noise variance
        self.projector = projector              # Store the projector module that provides forward and back projection

    def forward(self, image, sinogram=None):
        """
        Computes the linear log-likelihood loss for a given image.
        Args:
            image (torch.Tensor): The input image to be reconstructed of shape (batch-based, 1, 256, 256).
            sinogram (torch.Tensor): The sinogram corresponding to the input image. If None, the sinogram 
                is computed using the forward projector.
        
        Returns:
            loss (torch.Tensor): The computed linear log-likelihood loss for the input image. Measures how well the 
                forward projection of the image matches the measured sinogram (self.measurement)
        """
        if sinogram is None:
            sinogram = self.projector.forward_project(image)
        residual = sinogram - self.measurements # Calculate the difference between computed sinogram and measured
        
        # Compute the loss as the squared L2-norm of the residual, scaled by the noise variance.
        # This represents the discrepancy between the forward projection of the image and the measured sinogram.
        loss = 0.5 * torch.sum(residual**2) / self.noise_variance
        return loss

    def gradient(self, image, sinogram=None):
        """
        Computes the gradient of the log-likelihood loss with respect to the gradient.

        Args:
            image (torch.Tensor): The input image to be reconstructed of shape (batch-based, 1, 256, 256).
            sinogram (torch.Tensor): The sinogram corresponding to the input image. If None, the sinogram 
                is computed using the forward projector.
        
        Returns:
            gradient (torch.Tensor): The computed gradient of the log-likelihood loss with respect to the input image.
                The gradient represents the direction of steepest descent for the optimization problem.
        """
        if sinogram is None:
            sinogram = self.projector.forward_project(image)
        residual = sinogram - self.measurements
       
        # Compute the gradient by back-projecting the residual using the backprojector.
        # This step computes how the error in the sinogram propagates back to the image space.
        gradient = self.projector.back_project(residual) / self.noise_variance
        return gradient

    def hessian(self, image, image_input):
        """
        Compute the Hessian (second derivative) of the log-likelihood loss with respect to the input image.
        Specifically, it computes the Hessian-vector product.

        Args:
            image (torch.Tensor): The input image to be reconstructed of shape (batch-based, 1, 256, 256).
            image_input (torch.Tensor): The input image tensor used to compute the Hessian-vector product.

        Returns:
            hessian (torch.Tensor): The computed Hessian-vector product for the input image.
        """
        # Forward project the input image to obtain the sinogram, back project the sinogram 
        # back into the image space. This is equivalent to applying the Hessian operator in the image domain. 
        # Return the result of the Hessian-vector product devidid by the noise variance to adjust the scaling of the Hessian.
        return self.projector.back_project(self.projector.forward_project(image_input)) / self.noise_variance
    
    def inverse_hessian(self, image, image_input, reg=None):
        """
        Compute the inverse Hessian (second derivative) of the log-likelihood loss with respect to the input image.
        Specifically, it computes the inverse Hessian-vector product.

        Args:
            image (torch.Tensor): The input image to be reconstructed of shape (batch-based, 1, 256, 256).
            image_input (torch.Tensor): The input image tensor used to compute the inverse Hessian-vector product.

        Returns:
            inverse_hessian (torch.Tensor): The computed inverse Hessian-vector product for the input image.
        """
        # Compute the inverse Hessian-vector product by applying the forward projector to the input image
        # and then back projecting the result back into the image space. This is equivalent to applying the inverse
        # Hessian operator in the image domain. Return the result of the inverse Hessian-vector product.
        return self.projector.inverse_hessian(image_input, meas_var=self.noise_variance, reg=reg)


class QuadraticSmoothnessLogPrior(ReconstructionLossTerm):
    """
    This class implements a quadratic smoothness log-prior. It encourages smoothness in
    the reconstructed image by penalizing the squared Laplacian of the image, i.e. penalizing
    large differences between neighboring pixels values. The smoothness is enforced using a Laplacian operator.

    Args:
        beta (float): The regularization parameter that controls the strength of the smoothness penalty.
    """
    def __init__(self, beta=1.0):
        super(QuadraticSmoothnessLogPrior, self).__init__()
        self.beta = beta

    def laplacian(self, image):
        """
        Compute the discrete Laplacian of the input image, which is a measure of second-order
        spatial derivatives.
        
        Args:  
            image (torch.Tensor): The input image tensor of shape (batch-based, 1, 256, 256).

        Returns:
            laplacian (torch.Tensor): The computed Laplacian of the input image
        """
        laplacian = torch.zeros_like(image)
        laplacian += 4 * image  # Add 4 times the value of each pixel to the Laplacian

        # Subtract the value of the neighboring pixels
        laplacian -= torch.roll(image, 1, 2)    # Shift up
        laplacian -= torch.roll(image, -1, 2)   # Shift down
        laplacian -= torch.roll(image, 1, 3)    # Shift left    
        laplacian -= torch.roll(image, -1, 3)   # Shift right
        return laplacian

    def forward(self, image, laplacian=None):
        """
        Computes the smoothness prior loss. This loss penalizes large second-order derivatives
        (non-smooth regions) in the image using Laplacian operator.

        Args:
            image (torch.Tensor): The input image tensor of shape (batch-based, 1, 256, 256).
            laplacian (torch.Tensor): The precomputed Laplacian of the input image. If None, the Laplacian
                is computed using the laplacian() method.

        Returns:
            loss (torch.Tensor): The computed smoothness prior loss for the input image.
        """
        if laplacian is None:
            laplacian = self.laplacian(image)
        
        # Compute the quadratic smootheness loss as 0.5 * beta * sum(laplacian^2)
        loss = 0.5 * self.beta * torch.sum(laplacian**2)
        return loss

    def gradient(self, image, laplacian=None):
        """
        Computes the gradient of the smoothness prior loss with respect to the input image.
        This is used to guide optimization steps during image reconstruction.

        Args:
            image (torch.Tensor): The input image tensor of shape (batch-based, 1, 256, 256).
            laplacian (torch.Tensor): The precomputed Laplacian of the input image. If None, the Laplacian
                is computed using the laplacian() method.

        Returns:
            gradient (torch.Tensor): The computed gradient of the smoothness prior loss with respect to the input image.
        """
        if laplacian is None:
            laplacian = self.laplacian(image)
        gradient = self.beta * laplacian
        return gradient

    def hessian(self, image, image_input):
        """ 
        Computes the Hessian-vector product, which is the second-order derivative
        of the smoothness prior loss with respect to the input image.

        Args:
            image (torch.Tensor): the current image
            image_input (torch.Tensor): the input image tensor used to compute the Hessian-vector product

        Returns:
            hessian (torch.Tensor): the computed Hessian-vector product for the input image,
            which is beta times the Laplacian of the image_input.
        """
        hessian = self.beta * self.laplacian(image_input)
        return hessian

def iterative_reconstruction_gradient_descent(image_init, loss_terms, num_iterations=100, step_size=1.0, verbose=True):
    """
    Performs iterative reconstruction using gradient descent. The image is optimized by
    minimizing a set of loss terms over a fixed number of iterations or until convergence.

    Args:
        image_init (torch.Tensor): The initial image tensor for reconstruction of shape (batch-based, 1, 256, 256).
        loss_terms (list): A list of ReconstructionLossTerm objects representing the loss terms to optimize. Each loss term must implement a gradient method
        num_iterations (int): The number of iterations to run the optimization. Default is 100
        step_size (float): The step size for the gradient descent update. Default is 1.0
        verbose (bool): If True, prints the gradient norm and step size at each iteration.

    Returns:
        image (torch.Tensor): The optimized image tensor after gradient descent with the specified number of iterations.
    """
    
    # Clone the initial image and detach it from the computation graph.
    # Enable gradient computation for the image to allow optimization.
    image = image_init.clone().detach().requires_grad_(True)
    
    # Store the norm of the gradient from the previous iteration to monitor convergence.
    prev_gradient_norm = 0.0

    for iteration in tqdm(range(num_iterations)):
        gradient = torch.zeros_like(image)

        # Loop through each loss term and accumulate the gradient contributions from each loss term
        for loss_term in loss_terms:
            gradient += loss_term.gradient(image)
        
        # Update the image by subtracting the step size times the gradient
        # x_new = x_old - step+size*gradient
        image = image - step_size * gradient

        # Compute the norm (magnitude) of the gradient to monitor convergence
        gradient_norm = torch.norm(gradient)

        # Dynamic step size adjustment based on the gradient norm.
        # If the gradient norm increases, reduce the step size by half
        if gradient_norm >= prev_gradient_norm:
            step_size = step_size * 0.5
        else:
            step_size = step_size * 1.05
        
        # If the step size becomes too small, terminate iteration
        if step_size < 1e-6:
            break

        if verbose:
            print('Iteration %d, gradient norm = %f, step size = %f' % (iteration, gradient_norm, step_size))
            prev_gradient_norm = gradient_norm
    
    # Return the final reconstructed image
    return image

def HU_to_attenuation(image, scaleOnly=False):
    """
    μ = (HU + 1000) * 0.1 / 1000 
    """
    if scaleOnly:
        return (image) * 0.1 / 1000.0
    else:
        return (image + 1000.0) * 0.1 / 1000.0

def attenuation_to_HU(image, scaleOnly=False):
    """
    HU = μ * 1000 / 0.1 - 1000
    """
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
    """
    Main function to run CT image reconstruction pipeline. 
    """
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
        """
        Compute sample weights for data balancing during training.

        Args:
            metadata (pd.DataFrame): The metadata dataframe containing the hemorrhage types.
            hemorrhage_types (list): The list of hemorrhage types to consider for balancing.
        
        Returns:
            sample_weights (np.array): The computed sample weights for each sample in the dataset.
        """
        class_counts = metadata[hemorrhage_types].sum(axis=0).to_numpy()
        class_weights = 1.0 / class_counts # Inverse class frequency for balancing
        sample_weights_matrix = metadata[hemorrhage_types].to_numpy() * class_weights # Compute weight matrix
        sample_weights = sample_weights_matrix.sum(axis=1) # Sum weights per sample
        return sample_weights

    # Compute sample weights and create a sampler for the training set
    sample_weights = compute_sample_weights(train_dataset.metadata, train_dataset.hemorrhage_types)
    train_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dataset), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)

    # Compute sample weights and create a smapler for the validation set
    sample_weights = compute_sample_weights(val_dataset.metadata, val_dataset.hemorrhage_types)
    val_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(val_dataset), replacement=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)

    # Create a data loader for the test set without sampling
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize projector
    projector = CTProjector().to(device)

    # Create an iterator for the training data loader
    train_loader_iter = iter(train_loader)

    # Simulate CT measurements and iterative reconstruction
    for i in tqdm(range(num_patients)):

        try:
            phantom, _ = next(train_loader_iter) # Try to get the next batch from the training data iterator
        except StopIteration:
            # If the iterator is exhausted, reinitialize and get the next batch
            train_loader_iter = iter(train_loader)
            phantom, _ = next(train_loader_iter)

        phantom = phantom.to(device).float()

        phantom = HU_to_attenuation(phantom)
            
        print(f"Processing batch {i+1}/{num_patients}")

        # Simulate forward projection and sinogram with Poisson noise
        # I0 = 1e10
        t0 = time.time()
        sinogram = projector.forward_project(phantom) # Generate sinograms from the phantom
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
            pinv_reconstruction.clone(), # Start from the pseudo-inverse reconstruction
            [log_likelihood, QuadraticSmoothnessLogPrior(beta=10.0)], # Add a smoothness prior
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
