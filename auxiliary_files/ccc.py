import numpy as np
import matplotlib.pyplot as plt 
import SimpleITK as sitk
import sys
import functions
from SimpleITK import ResampleImageFilter, sitkNearestNeighbor, Transform
from scipy.stats import pearsonr


def input_volumes(volume_path: str, seg_path: str):

    """
    reads the image and resamples the segmentation to the image's size and spacing 
    for later computations

    Args:
        volume_path (str): Path to the image (e.g. the 24h image).
        seg_path (str): Path to the segmentation of the VOIs.

    Returns:
        - Array of the image provided.
        - Array of the resampled segmentation.

    """
        
    image= functions.read_dicom(volume_path)
    image_seg= functions.read_dicom(seg_path) 
    image_spacing= image.GetSpacing()

    def resample_seg(image, image_seg):

        resampler = ResampleImageFilter()
        resampler.SetReferenceImage(image)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetTransform(Transform())
        resampled_seg_img = resampler.Execute(image_seg)

        print('resampled segmentation:')
        print(resampled_seg_img.GetSize())
        print(resampled_seg_img.GetSpacing())
        print()

        return resampled_seg_img 

    resampled_seg= resample_seg(image, image_seg)  
    resampled_seg_np= sitk.GetArrayFromImage(resampled_seg)
    np_image= sitk.GetArrayFromImage(image)/(image_spacing[0]*image_spacing[1]*image_spacing[2]*(1e-3)) #image will be returned in MBq/mL

    return np_image, resampled_seg_np

def resample_to_target(moving_path, reference_path):
    """
    Resample `moving` image to match the voxel grid of `reference`,
    but keep moving image's origin (so no shifting).
    """
    moving = sitk.ReadImage(moving_path)
    reference = sitk.ReadImage(reference_path)

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(reference.GetSize())                
    resampler.SetOutputSpacing(moving.GetSpacing())    
    resampler.SetOutputDirection(moving.GetDirection())
    resampler.SetOutputOrigin(moving.GetOrigin())        
    resampler.SetInterpolator(sitk.sitkLinear)

    resampled = resampler.Execute(moving)
    return sitk.GetArrayFromImage(resampled)


def ccc(y_true, y_pred):

    """
    This function calculates the Concordance Correlation Coefficient (CCC). It uses the mean,
    variance and pearson coefficient of two distributions to determine the CCC.  

    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)

    rho, _ = pearsonr(y_true, y_pred) #pearson coefficient determination
    ccc = (2 * rho * np.sqrt(var_true * var_pred)) / (
        var_true + var_pred + (mean_true - mean_pred) ** 2) #thesis equation 4.8 
    
    return ccc


def calc(original_path, seg_original_path, simulated_path):

    """
    This function
    
    """

    or_volume, res_or_seg= input_volumes(original_path, seg_original_path)
    sim_volume = resample_to_target(simulated_path, original_path)
    sim_img= functions.read_dicom(simulated_path)
    sim_volume = sitk.GetArrayFromImage(sim_img)
    
    unique_values= np.unique(res_or_seg)[np.unique(res_or_seg) != 0]

    ccc_scores = {}

    for value in unique_values: 

        voi_mask_or= np.where(res_or_seg == value, res_or_seg/value, 0)

        #original
        total_voi_or= or_volume*voi_mask_or
        total_voi_or_norm= total_voi_or/np.max(total_voi_or)

        #simulated
        total_voi_sim= sim_volume*voi_mask_or
        total_voi_sim_norm= total_voi_sim/np.max(total_voi_sim)

        ccc_scores[value] = ccc(total_voi_or_norm, total_voi_sim_norm)

    return ccc_scores

    


