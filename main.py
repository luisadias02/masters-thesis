import SimpleITK as sitk
import functions
import lmfit_TAC
import tia_maps
import os


def input_files(original_path:str, simind_path:str):
    img_or= functions.read_dicom(original_path)
    img_sim= functions.read_dicom()

    return img_or, img_sim

def tac_computation(dict_vois:dict, img_24h: sitk.Image):
    df, dicts_cumulated= lmfit_TAC.total_computation(dict_vois)
    print(dicts_cumulated)
    dict_rescaled= lmfit_TAC.dict_conversion(dicts_cumulated, 
                                            pixel_size= img_24h.GetSpacing())
    
    return dict_rescaled

def tia_maps_generation(volume_path, seg_path, dict_rescaled, output_path:str, sim=False):

    np_image, np_seg= tia_maps.input_volumes(volume_path, seg_path)
    dict_masks= tia_maps.product(np_image, np_seg)
    if sim==False:
        output_tia= os.path.join(output_path, 'tia_map_or.nrrd')
        output_image= tia_maps.computation(dict_rescaled, dict_masks, volume_path, output_tia, index=0)
        tia_maps.resampling_2p21(output_image, os.path.join(output_path, 'tia_map_or_2p21.nrrd') )

    else:
        output_tia= os.path.join(output_path, 'tia_map_sim.nrrd')
        output_image= tia_maps.computation(dict_rescaled, dict_masks, volume_path, output_tia, index=1)
        tia_maps.resampling_2p21(output_image, os.path.join(output_path, 'tia_map_sim_2p21.nrrd'))


def main(dict_vois:dict, original_24h_path:str, simulated_24h_path:str, seg_path:str, output_path:str):

    img_or, img_sim= input_files(original_24h_path, simulated_24h_path)

    dict_rescaled= tac_computation(dict_vois, img_or)

    tia_maps_generation(original_24h_path, seg_path, dict_rescaled, output_path, sim=False)
    tia_maps_generation(simulated_24h_path, seg_path, dict_rescaled, output_path, sim=True)





















