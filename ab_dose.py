import sys 
import SimpleITK as sitk
import numpy as np 
import os
import functions
import matplotlib.pyplot as plt
from SimpleITK import ResampleImageFilter, sitkNearestNeighbor, Transform
from scipy import signal
from matplotlib import rcParams

def read_images(activity_map_path:str, seg_path:str):

    """
    reads and returns the image, segmentation and resampled segmentation.
    """

    tia_map_image= functions.read_dicom(activity_map_path)
    seg_image= functions.read_dicom(seg_path)

    resampled_seg= read_resample(seg_path, activity_map_path)

    return tia_map_image, seg_image, resampled_seg

def read_resample(seg_path, recon_path):

    def read_dicom(path):
        image= functions.read_dicom(path)
        print(image.GetSize())
        print(image.GetSpacing())
        return image
    
    print('spect matrix:')
    image_recon= read_dicom(recon_path)
    print('initial segmentation matrix:')
    image_seg= read_dicom(seg_path)  

    def resample_seg(image_recon, image_seg):

        resampler = ResampleImageFilter()
        resampler.SetReferenceImage(image_recon)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetTransform(Transform())
        resampled_seg_img = resampler.Execute(image_seg)

        print('resampled segmentation:')
        print(resampled_seg_img.GetSize())
        print(resampled_seg_img.GetSpacing())
        print()

        return resampled_seg_img 

    resampled_seg= resample_seg(image_recon, image_seg) 

    return sitk.GetArrayFromImage(resampled_seg)

def kernel_computation():

    """
    Computes the kernel of S-Values. The returned kernel is (11,11,11).
    """

    data = np.loadtxt("s_values_lu177_2p21.txt", delimiter='\t', skiprows=2) 
    # this kernel corresponds to the 2.21 mm voxel size, for 177Lu in soft tissue.

    s_values = data[:, 3]
    dim = int(round(len(s_values) ** (1/3)))
    s_values_img = s_values.reshape((dim, dim, dim))

    s_values_all = np.zeros((2*dim - 1, 2*dim - 1, 2*dim - 1))

    for i in range(2*dim - 1):
        ii = abs(i - (dim - 1))
        for j in range(2*dim - 1):
            jj = abs(j - (dim - 1))
            for k in range(2*dim - 1):
                kk = abs(k - (dim - 1))
                s_values_all[i, j, k] = s_values_img[ii, jj, kk]

    return s_values_all

def convolution(tia_map: sitk.Image, kernel: np.array, path:str, A0_administered:float):

    """
    performs the convolution of the TIA map and the Dose Voxel Kernel, via FFT convolution.

    Returns:
        - the absorbed dose map (in mGy/MBq)
    """

    image_spacing= tia_map.GetSpacing()
    np_tia_map = sitk.GetArrayFromImage(tia_map)*(image_spacing[0]*image_spacing[1]*image_spacing[2]*(1e-3))/A0_administered# normalization to the administred activity
    print('TIA map shape:', np_tia_map.shape)

    ab_dose_map = signal.fftconvolve(np_tia_map, kernel, mode='same') # the result of the convolution has the 
    print('Dose map shape:', ab_dose_map.shape)                       # same dimensions of the TIA map

    ab_dose_image= sitk.GetImageFromArray(ab_dose_map)
    ab_dose_image.SetSpacing(tia_map.GetSpacing())
    ab_dose_image.SetOrigin(tia_map.GetOrigin())
    ab_dose_image.SetDirection(tia_map.GetDirection())
    sitk.WriteImage(ab_dose_image, path)
    print('Dose map saved in:', path)

    return ab_dose_map

def statistics(ab_dose_map:np.array, resampled_seg:np.array, path:str):

    """
    Computes the dose statistics from the generated absorbed dose maps. 
    """
        
    values= np.unique(resampled_seg)
    keys= {1:'spleen', 2:'right kidney', 3:'left kidney', 5:'liver', 22:'tumor1', 23:'tumor2'} #p1 segmentation keys 

    with open(path, 'w') as f:
        for value in values[1:]:
            if value != 21 and value != 28:
                f.write(f'Segmentation value: {value} ({keys[value]})\n')
                    
                spect_foreground = ab_dose_map > 0
                mask = (resampled_seg == value) & spect_foreground
                conv_value = ab_dose_map[mask]

                mean_dose = np.mean(conv_value) 
                std_dose = np.std(conv_value) 
                min_dose = np.min(conv_value) 
                max_dose = np.max(conv_value) 

                n_overlap_voxels = np.sum(mask)
                f.write(f'  Overlapping voxels: {n_overlap_voxels}\n')
                f.write(f'  Mean: {np.round(mean_dose, 6)} mGy/MBq\n')
                f.write(f'  Std: {np.round(std_dose, 6)} mGy/MBq\n')
                f.write(f'  Min: {np.round(min_dose, 6)} mGy/MBq\n')
                f.write(f'  Max: {np.round(max_dose, 6)} mGy/MBq\n\n')


    print(f"Statistics saved to {path}")








