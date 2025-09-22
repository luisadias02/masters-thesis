import SimpleITK as sitk
import functions
import lmfit_TAC
import tia_maps
import ab_dose
import os

"""
This file contains the main workflow for dosimetry for 177Lu patients. 
It involves the computation of the TACs, followed by the generation of the cumulated activity maps. 
Absorbed dose is then determined by FFT convolution with a Dose Voxel Kernel (VDK).

"""

def input_files(original_path:str, simind_path:str):
    img_or= functions.read_dicom(original_path)
    img_sim= functions.read_dicom(simind_path)
    return img_or, img_sim

def tac_computation(dict_vois:dict, img_24h: sitk.Image, activity_admin:float, output_path:str):
    df, dicts_cumulated= lmfit_TAC.total_computation(dict_vois, activity_admin, output_path)
    print(dicts_cumulated)
    dict_rescaled= lmfit_TAC.dict_conversion(dicts_cumulated, 
                                            pixel_size= img_24h.GetSpacing())
    
    return dict_rescaled

def tia_maps_generation(volume_path:str, seg_path:str, dict_rescaled:dict, output_path:str, sim=False):

    np_image, np_seg= tia_maps.input_volumes(volume_path, seg_path)
    dict_masks= tia_maps.product(np_image, np_seg)
    if sim==False:
        output_tia= os.path.join(output_path, 'tia_map_or.nrrd')
        output_image= tia_maps.map_computation(dict_rescaled, dict_masks, volume_path, output_tia, index=0)
        tia_maps.resampling_2p21(output_image, os.path.join(output_path, 'tia_map_or_2p21.nrrd') )

    else:
        output_tia= os.path.join(output_path, 'tia_map_sim.nrrd')
        output_image= tia_maps.map_computation(dict_rescaled, dict_masks, volume_path, output_tia, index=1)
        tia_maps.resampling_2p21(output_image, os.path.join(output_path, 'tia_map_sim_2p21.nrrd'))

def abdose_calc(tia_path:str, seg_path:str, output_map_path:str, output_path_statistics:str, A0_administered:float):

    tia_map, seg_image, resampled_seg= ab_dose.read_images(tia_path, seg_path)
    kernel= ab_dose.kernel_computation()
    abdose_map= ab_dose.convolution(tia_map, kernel, output_map_path, A0_administered)
    ab_dose.statistics(abdose_map, resampled_seg, output_path_statistics)


def main(dict_vois:dict, original_24h_path:str, simulated_24h_path:str, seg_path:str, output_path:str, administered_activity:float):

    """
    This is the main function, responsible for executing the entire dosimetry workflow. To accurately compare the real and
    simulated absorbed doses the volumes need to be aligned between each other and with the segmentation.

    Args:
    - dict_vois (dict): dictionary with the real and simulated activities per VOI for each time-point. 
    - original_24_path (str): path to the original 24h .nii or .nrrd file
    - original_24_path (str): path to the simulated 24h .nii or .nrrd file
    - seg_path (str): path to the aligned segmentation 
    - output_path (str): path to the main folder to save the files (all the files will be saved in this folder)
    - administered_activity (float): injected activity (MBq)

    """

    img_or, img_sim= input_files(original_24h_path, simulated_24h_path)

    dict_rescaled= tac_computation(dict_vois, img_or, administered_activity, output_path)

    tia_maps_generation(original_24h_path, seg_path, dict_rescaled, output_path, sim=False)
    tia_maps_generation(simulated_24h_path, seg_path, dict_rescaled, output_path, sim=True)

    tia_or_path= os.path.join(output_path, 'tia_map_or_2p21.nrrd')    
    tia_sim_path= os.path.join(output_path, 'tia_map_sim_2p21.nrrd')

    abdose_or= abdose_calc(tia_or_path, seg_path, os.path.join(output_path, 'abdose_map_or.nrrd'), os.path.join(output_path, 'dose_statistics_or.txt'), administered_activity)
    abdose_sim=abdose_calc(tia_sim_path, seg_path, os.path.join(output_path, 'abdose_map_sim.nrrd'), os.path.join(output_path, 'dose_statistics_sim.txt'), administered_activity)

    ab_dose.setup_latex()
    ab_dose.dvh(os.path.join(output_path, 'abdose_map_or.nrrd'), os.path.join(output_path, 'abdose_map_sim.nrrd'), seg_path, administered_activity )

    






















