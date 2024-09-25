# step8_reconstruct_evaluation.py

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import pydicom
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import pydicom.uid 

from step0_common_info import dicom_dir, dataset_dir
from step2_dataset_dataloader import RSNA_Intracranial_Hemorrhage_Dataset
from step4_iterative_reconstruction import (
    CTProjector,
    iterative_reconstruction_gradient_descent,
    HU_to_attenuation,
    attenuation_to_HU,
    LinearLogLikelihood,
    QuadraticSmoothnessLogPrior,
)
from step5_deep_learning_reconstruction import DeepLearningReconstructor, load_reconstructor

def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load test dataset
    test_dataset = RSNA_Intracranial_Hemorrhage_Dataset(
        'data/metadata_evaluation.csv',
        dicom_dir
    )

    # Initialize CTProjector
    projector = CTProjector().to(device)

    # Initialize Deep Learning Reconstructor and load weights
    reconstructor = DeepLearningReconstructor().to(device)
    load_reconstructor(reconstructor, 'weights/deep_learning_reconstructor.pth')

    # Directories for saving reconstructed images
    FBP_dir = 'data/FBP_reconstructions'
    MBIR_dir = 'data/MBIR_reconstructions'
    DLR_dir = 'data/DLR_reconstructions'

    # Create directories if they do not exist
    os.makedirs(FBP_dir, exist_ok=True)
    os.makedirs(MBIR_dir, exist_ok=True)
    os.makedirs(DLR_dir, exist_ok=True)

    for idx in tqdm(range(len(test_dataset))):
        # Get image and label
        image, label = test_dataset[idx]
        # Get patient ID
        patient_id = test_dataset.metadata.iloc[idx]['PatientID']
        # Get the path to the original DICOM file
        dicom_path = os.path.join(dicom_dir, f'ID_{patient_id}.dcm')
        # Read the original DICOM file
        dicom_data = pydicom.dcmread(dicom_path)

        # Move image to device and preprocess
        image = image.to(device).float()
        image[image < -1000.0] = -1000.0
        image = image.view(256, 256)  # Remove channel dimension
        image = HU_to_attenuation(image)

        # Simulate forward projection
        image_batch = image.unsqueeze(0).unsqueeze(0)  # Shape: (1,1,256,256)
        sinogram = projector.forward_project(image_batch)  # Shape: (1,1,72,375)
        sinogram = sinogram.squeeze(0)  # Remove batch dimension

        # Simulate noisy sinogram
        I0 = 1e5
        photon_counts = I0 * torch.exp(-sinogram)
        photon_counts = torch.poisson(photon_counts)
        noisy_sinogram = -torch.log((photon_counts + 1) / I0)

        # Pseudoinverse reconstruction (FBP)
        sinogram = sinogram.unsqueeze(0)
        pinv_reconstruction = projector.pseudoinverse_reconstruction(sinogram, singular_values=[3000])
        pinv_reconstruction = torch.sum(pinv_reconstruction, dim=1, keepdim=True)
        # sinogram = sinogram.squeeze(0)  # Remove batch dimension

        # MBIR reconstruction
        log_likelihood = LinearLogLikelihood(noisy_sinogram, projector, noise_variance=1.0)
        log_prior_quadratic = QuadraticSmoothnessLogPrior(beta=50.0)
        mbir_reconstruction = iterative_reconstruction_gradient_descent(
            pinv_reconstruction.clone(),
            [log_likelihood, log_prior_quadratic],
            num_iterations=100,
            step_size=1e-2,
            verbose=False
        )

        # Deep Learning Reconstruction (DLR)
        x_tilde_components = projector.pseudoinverse_reconstruction(
            noisy_sinogram.unsqueeze(0), projector.singular_values_list
        )
        pseudoinverse = torch.sum(x_tilde_components, dim=1, keepdim=True).to(device)
        reconstruction = reconstructor(pseudoinverse)
        reconstruction = reconstruction.squeeze(0)  # Remove batch dimension

        # Convert images back to HU units
        pinv_reconstruction_HU = attenuation_to_HU(pinv_reconstruction).squeeze()
        mbir_reconstruction_HU = attenuation_to_HU(mbir_reconstruction).squeeze()
        dlr_reconstruction_HU = attenuation_to_HU(reconstruction).squeeze()

        # Check the shape to ensure they are 256x256
                # Check the shape to ensure they are 256x256
        pinv_reconstruction_HU = pinv_reconstruction_HU.squeeze()  # Remove all singleton dimensions
        mbir_reconstruction_HU = mbir_reconstruction_HU.squeeze()  # Remove all singleton dimensions
        dlr_reconstruction_HU = dlr_reconstruction_HU.squeeze() 
        
        assert pinv_reconstruction_HU.shape == (256, 256), f"Unexpected shape for pinv_reconstruction: {pinv_reconstruction_HU.shape}"
        assert mbir_reconstruction_HU.shape == (256, 256), f"Unexpected shape for mbir_reconstruction: {mbir_reconstruction_HU.shape}"
        assert dlr_reconstruction_HU.shape == (256, 256), f"Unexpected shape for dlr_reconstruction: {dlr_reconstruction_HU.shape}"

        # Convert reconstructed images to numpy arrays
        pinv_reconstruction_HU_np = pinv_reconstruction_HU.detach().cpu().numpy()
        mbir_reconstruction_HU_np = mbir_reconstruction_HU.detach().cpu().numpy()
        dlr_reconstruction_HU_np = dlr_reconstruction_HU.detach().cpu().numpy()

        # Get RescaleSlope and RescaleIntercept from original DICOM file
        RescaleSlope = float(dicom_data.RescaleSlope)
        RescaleIntercept = float(dicom_data.RescaleIntercept)

        # Convert HU values to pixel values
        pinv_pixel_data = np.round(
            (pinv_reconstruction_HU_np - RescaleIntercept) / RescaleSlope
        ).astype(np.int16)
        mbir_pixel_data = np.round(
            (mbir_reconstruction_HU_np - RescaleIntercept) / RescaleSlope
        ).astype(np.int16)
        dlr_pixel_data = np.round(
            (dlr_reconstruction_HU_np - RescaleIntercept) / RescaleSlope
        ).astype(np.int16)

        import copy
        
        # Create deep copies of the original DICOM data
        dicom_pinv = copy.deepcopy(dicom_data)
        dicom_mbir = copy.deepcopy(dicom_data)
        dicom_dlr = copy.deepcopy(dicom_data)

        # Update pixel data
        dicom_pinv.PixelData = pinv_pixel_data.tobytes()
        dicom_mbir.PixelData = mbir_pixel_data.tobytes()
        dicom_dlr.PixelData = dlr_pixel_data.tobytes()

        for dicom_obj, pixel_data in zip([dicom_pinv, dicom_mbir, dicom_dlr], 
                                         [pinv_pixel_data, mbir_pixel_data, dlr_pixel_data]):
            dicom_obj.Rows = pixel_data.shape[0]  # Set to 256
            dicom_obj.Columns = pixel_data.shape[1]  # Set to 256
            # dicom_obj.BitsAllocated = 16
            # dicom_obj.BitsStored = 16
            # dicom_obj.HighBit = 15
            # dicom_obj.PixelRepresentation = 1  # Signed integer for HU values

        # Save the DICOM files
        dicom_pinv.save_as(os.path.join(FBP_dir, f'ID_{patient_id}.dcm'))
        dicom_mbir.save_as(os.path.join(MBIR_dir, f'ID_{patient_id}.dcm'))
        dicom_dlr.save_as(os.path.join(DLR_dir, f'ID_{patient_id}.dcm'))
    


if __name__ == "__main__":
    main()

# import torch
# import torch.nn as nn
# import numpy as np
# from torch.utils.data import DataLoader
# import pydicom
# import os
# from tqdm import tqdm

# from step0_common_info import dicom_dir, dataset_dir
# from step2_dataset_dataloader import RSNA_Intracranial_Hemorrhage_Dataset
# from step4_iterative_reconstruction import (
#     CTProjector,
#     iterative_reconstruction_gradient_descent,
#     HU_to_attenuation,
#     attenuation_to_HU,
#     LinearLogLikelihood,
#     QuadraticSmoothnessLogPrior,
# )
# from step5_deep_learning_reconstruction import DeepLearningReconstructor, load_reconstructor

# def main():
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'

#     # Load test dataset
#     test_dataset = RSNA_Intracranial_Hemorrhage_Dataset(
#         'data/metadata_evaluation.csv',
#         dicom_dir
#     )

#     # Initialize CTProjector
#     projector = CTProjector().to(device)

#     # Initialize Deep Learning Reconstructor and load weights
#     reconstructor = DeepLearningReconstructor().to(device)
#     load_reconstructor(reconstructor, 'weights/deep_learning_reconstructor.pth')

#     # Directories for saving reconstructed images
#     FBP_dir = 'data/FBP_reconstructions'
#     MBIR_dir = 'data/MBIR_reconstructions'
#     DLR_dir = 'data/DLR_reconstructions'

#     # Create directories if they do not exist
#     os.makedirs(FBP_dir, exist_ok=True)
#     os.makedirs(MBIR_dir, exist_ok=True)
#     os.makedirs(DLR_dir, exist_ok=True)

#     for idx in tqdm(range(len(test_dataset))):
#         # Get image and label
#         image, label = test_dataset[idx]
#         # Get patient ID
#         patient_id = test_dataset.metadata.iloc[idx]['PatientID']
#         # Get the path to the original DICOM file
#         dicom_path = os.path.join(dicom_dir, f'ID_{patient_id}.dcm')
#         # Read the original DICOM file
#         dicom_data = pydicom.dcmread(dicom_path)

#         # Move image to device and preprocess
#         image = image.to(device).float()
#         image[image < -1000.0] = -1000.0
#         image = image.view(256, 256)  # Remove channel dimension
#         image = HU_to_attenuation(image)

#         # Simulate forward projection
#         image_batch = image.unsqueeze(0).unsqueeze(0)  # Shape: (1,1,256,256)
#         sinogram = projector.forward_project(image_batch)  # Shape: (1,1,72,375)
#         sinogram = sinogram.squeeze(0)  # Remove batch dimension

#         # Simulate noisy sinogram
#         I0 = 1e5
#         photon_counts = I0 * torch.exp(-sinogram)
#         photon_counts = torch.poisson(photon_counts)
#         noisy_sinogram = -torch.log((photon_counts + 1) / I0)

#         # Pseudoinverse reconstruction (FBP)
#         sinogram = sinogram.unsqueeze(0)
#         pinv_reconstruction = projector.pseudoinverse_reconstruction(sinogram, singular_values=[3000])
#         pinv_reconstruction = torch.sum(pinv_reconstruction, dim=1, keepdim=True)

#         # MBIR reconstruction
#         log_likelihood = LinearLogLikelihood(noisy_sinogram, projector, noise_variance=1.0)
#         log_prior_quadratic = QuadraticSmoothnessLogPrior(beta=50.0)
#         mbir_reconstruction = iterative_reconstruction_gradient_descent(
#             pinv_reconstruction.clone(),
#             [log_likelihood, log_prior_quadratic],
#             num_iterations=100,
#             step_size=1e-2,
#             verbose=False
#         )

#         # Deep Learning Reconstruction (DLR)
#         x_tilde_components = projector.pseudoinverse_reconstruction(
#             noisy_sinogram.unsqueeze(0), projector.singular_values_list
#         )
#         pseudoinverse = torch.sum(x_tilde_components, dim=1, keepdim=True).to(device)
#         reconstruction = reconstructor(pseudoinverse)
#         reconstruction = reconstruction.squeeze(0)  # Remove batch dimension

#         # Convert images back to HU units
#         pinv_reconstruction_HU = attenuation_to_HU(pinv_reconstruction).squeeze()
#         mbir_reconstruction_HU = attenuation_to_HU(mbir_reconstruction).squeeze()
#         dlr_reconstruction_HU = attenuation_to_HU(reconstruction).squeeze()

#         # Check the shape to ensure they are 256x256
#         pinv_reconstruction_HU = pinv_reconstruction_HU.squeeze()  # Remove all singleton dimensions
#         mbir_reconstruction_HU = mbir_reconstruction_HU.squeeze()  # Remove all singleton dimensions
#         dlr_reconstruction_HU = dlr_reconstruction_HU.squeeze() 
        
#         assert pinv_reconstruction_HU.shape == (256, 256), f"Unexpected shape for pinv_reconstruction: {pinv_reconstruction_HU.shape}"
#         assert mbir_reconstruction_HU.shape == (256, 256), f"Unexpected shape for mbir_reconstruction: {mbir_reconstruction_HU.shape}"
#         assert dlr_reconstruction_HU.shape == (256, 256), f"Unexpected shape for dlr_reconstruction: {dlr_reconstruction_HU.shape}"

#         # Convert reconstructed images to numpy arrays
#         pinv_reconstruction_HU_np = pinv_reconstruction_HU.detach().cpu().numpy()
#         mbir_reconstruction_HU_np = mbir_reconstruction_HU.detach().cpu().numpy()
#         dlr_reconstruction_HU_np = dlr_reconstruction_HU.detach().cpu().numpy()

#         # Get RescaleSlope and RescaleIntercept from original DICOM file
#         RescaleSlope = float(dicom_data.RescaleSlope)
#         RescaleIntercept = float(dicom_data.RescaleIntercept)

#         # Convert HU values to pixel values
#         pinv_pixel_data = np.round(
#             (pinv_reconstruction_HU_np - RescaleIntercept) / RescaleSlope
#         ).astype(np.int16)
#         mbir_pixel_data = np.round(
#             (mbir_reconstruction_HU_np - RescaleIntercept) / RescaleSlope
#         ).astype(np.int16)
#         dlr_pixel_data = np.round(
#             (dlr_reconstruction_HU_np - RescaleIntercept) / RescaleSlope
#         ).astype(np.int16)

#         # Check the shape of pixel data to confirm it is 256x256
#         assert pinv_pixel_data.shape == (256, 256), f"Unexpected shape for pinv_pixel_data: {pinv_pixel_data.shape}"
#         assert mbir_pixel_data.shape == (256, 256), f"Unexpected shape for mbir_pixel_data: {mbir_pixel_data.shape}"
#         assert dlr_pixel_data.shape == (256, 256), f"Unexpected shape for dlr_pixel_data: {dlr_pixel_data.shape}"

#         # Create copies of the original DICOM data
#         dicom_pinv = dicom_data.copy()
#         dicom_mbir = dicom_data.copy()
#         dicom_dlr = dicom_data.copy()

#         # Update pixel data and DICOM tags
#         for dicom_obj, pixel_data in zip([dicom_pinv, dicom_mbir, dicom_dlr], 
#                                          [pinv_pixel_data, mbir_pixel_data, dlr_pixel_data]):
#             # dicom_obj.PixelData = pixel_data.tobytes()  # Convert pixel data to bytes
#             dicom_obj.Rows = pixel_data.shape[0]  # Set to 256
#             dicom_obj.Columns = pixel_data.shape[1]  # Set to 256
#             dicom_obj.BitsAllocated = 16
#             dicom_obj.BitsStored = 16
#             dicom_obj.HighBit = 15
#             dicom_obj.PixelRepresentation = 1  # Signed integer for HU values

#         # Save the DICOM files
#         dicom_pinv.save_as(os.path.join(FBP_dir, f'ID_{patient_id}.dcm'))
#         dicom_mbir.save_as(os.path.join(MBIR_dir, f'ID_{patient_id}.dcm'))
#         dicom_dlr.save_as(os.path.join(DLR_dir, f'ID_{patient_id}.dcm'))

# if __name__ == "__main__":
#     main()