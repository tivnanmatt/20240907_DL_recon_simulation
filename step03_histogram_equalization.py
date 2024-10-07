import torch
from torch.utils.data import DataLoader
from step02_dataset_dataloader import RSNA_Intracranial_Hemorrhage_Dataset, dicom_dir
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_flag = False
if train_flag:
    if __name__ == '__main__':
        train_dataset = RSNA_Intracranial_Hemorrhage_Dataset(
                    'data/metadata_training.csv',
                    dicom_dir)


        train_loader = DataLoader(train_dataset, batch_size=1, num_workers=32)

        train_loader_iter = iter(train_loader)

        num_training_images = 10000
        # training_images = torch.zeros(num_training_images, 1, 256, 256)
        hist = torch.zeros(num_training_images, 3001)
        bin_centers = torch.linspace(-1000, 2000, 3001)
        for i in tqdm(range(num_training_images)):
            training_image = next(train_loader_iter)[0]
            # get the histogram of the image
            hist[i] = torch.histc(training_image.view(-1), bins=3001, min=-1000, max=2000)
            
        HU_histogram = torch.sum(hist, dim=0)
        HU_histogram = HU_histogram / torch.sum(HU_histogram)
        cumulative_hist = torch.cumsum(HU_histogram, dim=0)

        from matplotlib import pyplot as plt

        plt.figure()
        plt.plot(bin_centers.detach().numpy(), HU_histogram.detach().numpy())
        plt.title('Histogram of training images')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.savefig('tmp.png')

        plt.figure()
        plt.plot(bin_centers.detach().numpy(), cumulative_hist.detach().numpy())
        plt.title('Cumulative Histogram of training images')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.savefig('tmp.png')

        # save the histograms
        torch.save(HU_histogram, './weights/HU_histogram.pt')

HU_bin_centers = torch.linspace(-1000, 2000, 3001)
HU_histogram = torch.load('./weights/HU_histogram.pt')
HU_cumulative_hist = torch.cumsum(HU_histogram[500:]/torch.sum(HU_histogram[500:]), dim=0)
HU_cumulative_hist = HU_cumulative_hist 

linear_region = torch.zeros(500)
HU_cumulative_hist = torch.cat([linear_region, HU_cumulative_hist])

linear_mix_weight = 0.3
HU_cumulative_hist = (1-linear_mix_weight)*HU_cumulative_hist + linear_mix_weight*torch.linspace(0, 1, 3001)


plt.figure()
plt.plot(HU_bin_centers.detach().numpy(), HU_cumulative_hist.detach().numpy())
plt.title('Cumulative Histogram of training images')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.savefig('tmp.png')

def HU_to_SU(image):

    _HU_bin_centers = HU_bin_centers.to(image.device)
    _HU_cumulative_hist = HU_cumulative_hist.to(image.device)

    # Use searchsorted to find the bin index for each pixel
    idx = torch.searchsorted(_HU_bin_centers[:-2], image.clamp(min=-1000, max=2000))

    # Handle values below the minimum
    below_min_mask = image < _HU_bin_centers[0]
    idx[below_min_mask] = 0  # Set index to the first bin

    # Handle values above the maximum
    above_max_mask = image >= _HU_bin_centers[-1]
    idx[above_max_mask] = len(_HU_bin_centers) - 2  # Set index to the last valid bin

    # Get the lower and upper bounds for HU and SU
    lower_bound_HU = _HU_bin_centers[idx]
    upper_bound_HU = _HU_bin_centers[idx + 1]
    lower_bound_SU = _HU_cumulative_hist[idx]
    upper_bound_SU = _HU_cumulative_hist[idx + 1]

    # Linearly interpolate to find the corresponding SU values
    image_SU = lower_bound_SU + (upper_bound_SU - lower_bound_SU) * \
               ((image - lower_bound_HU) / (upper_bound_HU - lower_bound_HU))

    return image_SU

def SU_to_HU(image_SU):

    _HU_bin_centers = HU_bin_centers.to(image_SU.device)
    _HU_cumulative_hist = HU_cumulative_hist.to(image_SU.device)

    # Use searchsorted to find the bin index for each pixel
    idx = torch.searchsorted(_HU_cumulative_hist[:-2], image_SU.clamp(min=0, max=1))

    # Handle values below the minimum
    below_min_mask = image_SU < _HU_cumulative_hist[0]
    idx[below_min_mask] = 0  # Set index to the first bin

    # Handle values above the maximum
    above_max_mask = image_SU >= _HU_cumulative_hist[-1]
    idx[above_max_mask] = len(_HU_cumulative_hist) - 2  # Set index to the last valid bin

    # Get the lower and upper bounds for SU and HU
    lower_bound_SU = _HU_cumulative_hist[idx]
    upper_bound_SU = _HU_cumulative_hist[idx + 1]
    lower_bound_HU = _HU_bin_centers[idx]
    upper_bound_HU = _HU_bin_centers[idx + 1]

    # Linearly interpolate to find the corresponding HU values
    image_HU = lower_bound_HU + (upper_bound_HU - lower_bound_HU) * \
               ((image_SU - lower_bound_SU) / (upper_bound_SU - lower_bound_SU))

    return image_HU

if __name__ == '__main__':

    train_dataset = RSNA_Intracranial_Hemorrhage_Dataset(
                'data/metadata_training.csv',
                dicom_dir)
    

    image, label = train_dataset[0:1]

    # image = image.to(device)
    # image = image + torch.randn_like(image) * 3000.0

    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(image[0].cpu(), cmap='gray',vmin=-1000, vmax=2000)
    axs[0].set_title('Original Image')

    axs[1].imshow(HU_to_SU(image)[0].cpu(), cmap='gray', vmin=0, vmax=1)
    axs[1].set_title('Histogram Equalized Image')

    axs[2].imshow(SU_to_HU(HU_to_SU(image))[0].cpu(), cmap='gray', vmin=-1000, vmax=2000)
    axs[2].set_title('Back to HU')

    plt.savefig('tmp.png')


    