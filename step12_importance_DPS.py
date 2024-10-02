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

from step6_diffusion_training import UnconditionalDiffusionModel, HU_to_SU, SU_to_HU, load_diffusion_model_weights

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ImportanceDiffusionPosteriorSampling(UnconditionalDiffusionModel):
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
        t = t.view(-1, 1, 1, 1)
        sigma2 = t
        x_0_hat = x_t + sigma2 * prior_score

        # this ignores the jacobian of x_0_hat w.r.t. x_t, 
        # but it can be shown to be a nearly scalar matrix, 
        # https://arxiv.org/html/2407.12956v1, Appendix A, Figure 11
        # so it can be wrapped into one hyperparameter, 
        # the measurement variance of the likelihood
        # defined outside this model
        # the negative sign here is because LinearLogLikelihood is actually a negative log likelihood... TODO: fix this
        
        # likelihood_score = -self.log_likelihood.gradient(x_0_hat) 
        
        _var_x_HU = 0*t + 100.0 # 0.1 std = 100 HU
        _std_x_HU = torch.sqrt(_var_x_HU)
        _std_x_SU = HU_to_SU(_std_x_HU)
        _std_x_attenuation = HU_to_attenuation(_std_x_HU, scaleOnly=True)
        _var_x_attenuation = _std_x_attenuation**2
        _var_x_SU = _std_x_SU**2

        _t_var_SU = t
        _t_std_SU = torch.sqrt(_t_var_SU)
        _t_std_HU = SU_to_HU(_t_std_SU)
        _t_std_attenuation = HU_to_attenuation(_t_std_HU, scaleOnly=True)
        _t_var_attenuation = _t_std_attenuation**2
        _t_var_HU = _t_std_HU**2

        _x_t_SU = x_t
        _x_t_HU = SU_to_HU(_x_t_SU)
        _x_t_attenuation = HU_to_attenuation(_x_t_HU)

        _x_0_hat_SU = x_0_hat
        _x_0_hat_HU = SU_to_HU(_x_0_hat_SU)
        _x_0_hat_attenuation = HU_to_attenuation(_x_0_hat_HU)


        


        # all of these are in attenuatin units now
        var_x = _var_x_attenuation
        var_t = _t_var_attenuation
        x_t = _x_t_attenuation
        x_0_hat = _x_0_hat_attenuation

        # define the diffusion posterior mean and variance
        diffusion_posterior_mean = (var_x/ (var_t+var_x))*x_t + (var_t/(var_t+var_x))*x_0_hat
        diffusion_posterior_var = (var_t*var_x)/(var_t+var_x)
        
        # get the likelihood gradient in SU
        likelihood_gradient = -self.log_likelihood.gradient(None, 
                                                            sinogram=0.0)
        diffusion_posterior_gradient = diffusion_posterior_mean/diffusion_posterior_var
        total_gradient = likelihood_gradient + diffusion_posterior_gradient

        # apply the inverse hessian, with regularization
        reconstruction_posterior_mean = self.log_likelihood.inverse_hessian(None, 
                                                                            total_gradient, 
                                                                            reg=1/diffusion_posterior_var)
        
        # define the residual from diffusion posterior mean to reconstruction posterior mean and apply weights
        _residual = reconstruction_posterior_mean - diffusion_posterior_mean

        # apply weights to get the likelihood score
        _residual -= (1/diffusion_posterior_gradient)*self.log_likelihood.inverse_hessian(None, 
                                                                                         _residual, 
                                                                                         reg=1/diffusion_posterior_var)
        _residual = (1/diffusion_posterior_var)*_residual
        likelihood_score = _residual.clone()


        _likelihood_score_attenuation = likelihood_score.clone()
        _likelihood_score_HU = attenuation_to_HU(_likelihood_score_attenuation, scaleOnly=True)
        _likelihood_score_SU = HU_to_SU(_likelihood_score_HU)
        likelihood_score = _likelihood_score_SU.clone()

        # likelihood_score *= 0.5*torch.norm(prior_score)/torch.norm(likelihood_score)
        likelihood_score = 1e-7*likelihood_score

        # the good work of Rev. Thomas Bayes
        posterior_score = prior_score + likelihood_score
        print('DEBUG: prior_score:', prior_score.norm().item(), ', likelihood_score:', likelihood_score.norm().item(), 'posterior_score:', posterior_score.norm().item())

        # fix this crime
        x_t = _x_t_SU

        # flip the sign to apply the standard Anderson formula
        dt = -dt

        # the good work of Rev. Thomas Bayes
        posterior_score = prior_score + likelihood_score

        if mode == 'sde':
            return x_t - dt * posterior_score + torch.sqrt(torch.abs(dt)) * torch.randn_like(x_t)
        elif mode == 'ode':
            return x_t - dt * 0.5 * posterior_score



def evaluate_diffusion_model(diffusion_model, test_loader, num_samples=10):
    assert isinstance(diffusion_model, UnconditionalDiffusionModel)


    projector = CTProjector().to(device)
    measurements = None # Placeholder for measurements
    linear_log_likelihood = LinearLogLikelihood(measurements, projector, noise_variance=0.1)
    
    diffusion_posterior_sampling = ImportanceDiffusionPosteriorSampling(diffusion_model, linear_log_likelihood)
    
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
            t = torch.tensor([0.01], device=device)
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

            plt.savefig(f'./figures/IDPS_batch_{i}_bone.png')


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

            plt.savefig(f'./figures/IDPS_batch_{i}_brain.png')


            plt.close('all')

def load_diffusion_model_weights(diffusion_model, load_path):
    diffusion_model.load_state_dict(torch.load(load_path))

def main():

    from step0_common_info import dicom_dir
    
    test_dataset = RSNA_Intracranial_Hemorrhage_Dataset(
            'data/metadata_evaluation.csv',
            dicom_dir)

    # random seed
    torch.manual_seed(0)
    inds = np.arange(len(test_dataset))
    test_dataset = Subset(test_dataset, inds)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Initialize the diffusion model
    diffusion_model = UnconditionalDiffusionModel().to(device)

    # Load the diffusion model's weights
    try:
        load_diffusion_model_weights(diffusion_model, 'weights/diffusion_model_weights.pth')
        print("Diffusion model weights loaded successfully.")
    except:
        raise Exception("Diffusion model weights not found.")


    # Evaluate the diffusion model
    evaluate_diffusion_model(diffusion_model, test_loader, num_samples=10)

if __name__ == "__main__":
    main()









































