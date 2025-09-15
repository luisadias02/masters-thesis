import sys
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import os 
import functions
from SimpleITK import ResampleImageFilter, sitkNearestNeighbor, Transform

def input_volumes(volume_path: str, seg_path: str):

    """
    reads the image and resamples the segmentation to the image's size and spacing 
    for later computations

    Args:
        volume_path (str): Path to the image (e.g., 24h image).
        seg_path (str): Path to the segmentation of the VOIs.

    Returns:
        - Array of the image provided.
        - Array of the resampled segmentation.

    """
        
    image= functions.read_dicom(volume_path)
    print('image matrix:', image.GetSize())
    image_seg= functions.read_dicom(seg_path) 
    print('initial segmentation matrix:', image_seg.GetSize())  
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

def product(np_image: np.ndarray, np_seg: np.ndarray):

    """
    returns the multiplication of masks for the values of the segmentation (each VOI).
    segmentation and image must have the same dimensions

    Args:
        np_image (np.ndarray): image array.
        np_seg (np.ndarray): resampled segmentation array.

    Returns:
        - dictionary of the masks corresponding to each unique value

    """

    unique_values= np.unique(np_seg)
    masks= {}
    for value in unique_values: 
        if value==0: #the remainer of the body
            voi_mask= np.where(np_seg==0, 1, 0)
        else:
            voi_mask= np.where(np_seg == value, np_seg/value, 0) #binary mask

        total_voi= np_image*voi_mask
        masks[value]= total_voi

    return masks

def map_computation(dict_cumulated: dict, masks: dict, volume_path: str, output_path: str, index):

    """
    computes the 3D cumulated activity map.

    Args:
        dict_cumulated (dict): dictionary of the total cumulated activity per VOI.
        masks (dict): dictionary of the VOIs
        volume_path (str): image path to retrieve the metadata info for volume saving.

    Returns:
        - the created map path
    
    """

    keys= {1:'spleen', 2:'right kidney', 3:'left kidney', 5:'liver', 21:'bladder', 22:'tumor1', 23:'tumor2', 0:'remaining'} #segmentation keys - if TotalSegmentator is used

    size= sitk.GetArrayFromImage(functions.read_dicom(volume_path)).shape
    total_array = np.zeros((size[0], size[1], size[2]))
    for key, value in keys.items():
        conv_voi= (masks[key]/np.sum(masks[key]))*dict_cumulated[value][index] #index 0 equals the original image cumulated activities
        total_array += conv_voi                                            #index 1 equals the simulated image cumulated activities

    total_image= sitk.GetImageFromArray(total_array)
    total_image.SetDirection(functions.read_dicom(volume_path).GetDirection())
    total_image.SetOrigin(functions.read_dicom(volume_path).GetOrigin())
    total_image.SetSpacing(functions.read_dicom(volume_path).GetSpacing())

    sitk.WriteImage(total_image, output_path)
    print('TIA map saved in:', output_path)

    return output_path

def resampling_2p21(tia_map_path:str, output_path: str):

    """
    resamples the 3D cumulated activity map to a 2.21mm voxel size.

    Args:
        tia_map_path (str): path for the cumulated activity map to resample.
        output_path (str): path for the resampled image.
    """

    original_tia_image= functions.read_dicom(tia_map_path)
    original_spacing= original_tia_image.GetSpacing()
    print('original spacing:', original_spacing)
    original_size= original_tia_image.GetSize()
    print('original size:', original_size)

    new_spacing= (2.21, 2.21, 2.21)
    new_size = [
    int(round(osz * ospc / nspc))
    for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(original_tia_image.GetOrigin())
    resampler.SetOutputDirection(original_tia_image.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(sitk.Transform()) 

    resampled_tia_map= resampler.Execute(original_tia_image)

    print('resampled image:')
    print(resampled_tia_map.GetSize())
    print(resampled_tia_map.GetSpacing())

    sitk.WriteImage(resampled_tia_map, output_path)
    print('Resampled TIA map saved in:', output_path)










