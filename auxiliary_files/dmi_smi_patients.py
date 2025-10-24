import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os 


def ct_image(ct_path:str): 

    reader= sitk.ImageFileReader()
    reader.SetFileName(ct_path)
    ct_image= reader.Execute()

    size= ct_image.GetSize()
    pixel_spacing= ct_image.GetSpacing()

    print('ct image size:', size)
    print('ct pixel spacing:', pixel_spacing)

    #curvas de calibração (ver excel hu_calib.xlsl):
    #para x< 127.34355:
    # density= 0.00102430x + 1.02419109

    # para x >= 127.34355:
    # density= 0.000503556 + 1.090504592

    ct_array = sitk.GetArrayFromImage(ct_image).astype(np.float32)  
    spacing = ct_image.GetSpacing()
    origin = ct_image.GetOrigin()
    direction = ct_image.GetDirection()

    threshold = 127.34355
    density = np.where(
        ct_array < threshold,
        1000* (0.00102430 * ct_array + 1.02419109),  
        1000*(0.000503556 * ct_array + 1.090504592)
    )

    hu_image = sitk.GetImageFromArray(density)
    hu_image.SetSpacing(spacing)
    hu_image.SetOrigin(origin)
    hu_image.SetDirection(direction)

    return hu_image

def source_imagem(pet_path:str): 

    reader= sitk.ImageFileReader()
    reader.SetFileName(pet_path)
    source_image= reader.Execute()

    size= source_image.GetSize()
    pixel_spacing= source_image.GetSpacing()

    print('original source image size:', size)
    print('original source pixel spacing:', pixel_spacing)

    #RESAMPLING DA SOURCE IMAGE
    def resample():
        new_spacing = (pixel_spacing[0], pixel_spacing[1], 1.0) #1.0 IS THE CT Z DIMENSION

        z = 636 #CT NUMBER OF SLICES

        old_z_extent = size[2] * pixel_spacing[2]
        new_z_spacing = old_z_extent / z

        new_spacing = (pixel_spacing[0], pixel_spacing[1], new_z_spacing)

        new_size = (size[0],size[1],z)
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(source_image.GetDirection())
        resampler.SetOutputOrigin(source_image.GetOrigin())
        resampler.SetInterpolator(sitk.sitkLinear) 

        resampled_img = resampler.Execute(source_image)
        print('resampled source dimensions:')
        print('resampled image size:', resampled_img.GetSize())
        print('resampled image spacing:', resampled_img.GetSpacing())
        resampled_img.SetDirection(source_image.GetDirection())
        resampled_img.SetOrigin(source_image.GetOrigin())  

        sitk.WriteImage(resampled_img, "E:\\tese_mestrado\\patients_study\\p8\\resampled_6d.nii")

        return resampled_img

    source_image= resample()

    return source_image

def save_smi_dmi(ct_array, source_array):

    output_dmi= 'E:\\tese_mestrado\\patients_study\\p8\\p8_spect_ct_2.dmi'
    dmi= ct_array.astype(np.int16)
    dmi.astype('<i2').tofile(output_dmi)
    print('phantom saved in:', output_dmi)

    output_smi= 'E:\\tese_mestrado\\patients_study\\p8\\p8_6d_2.smi'
    smi= source_array.astype(np.int16)
    smi.astype('<i2').tofile(output_smi)
    print('source saved in:', output_smi)















