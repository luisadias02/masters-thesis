import sys 
import SimpleITK as sitk
import numpy as np 
import os
import functions
import matplotlib.pyplot as plt
from SimpleITK import ResampleImageFilter, sitkNearestNeighbor, Transform
from scipy import signal
from matplotlib import rcParams
import matplotlib as mpl

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
    performs the convolution of the TIA map and the Dose Voxel Kernel (DVK), via FFT convolution.

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
    Computes the dose statistics from the generated normalized absorbed dose maps. 
    """
        
    values= np.unique(resampled_seg)
    keys= {1:'spleen', 2:'right kidney', 3:'left kidney', 5:'liver', 22:'tumor1', 23:'tumor2'} #segmentation keys, if TotalSegmentator is used

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

    return ab_dose_map



def dvh(abdose_or_path: str, abdose_sim_path: str, seg_path: str, injected_activity: float):

    abdose_or = sitk.GetArrayFromImage(functions.read_dicom(abdose_or_path)) * injected_activity  # mGy
    abdose_sim = sitk.GetArrayFromImage(functions.read_dicom(abdose_sim_path)) * injected_activity
     
    resampled_seg_or = read_resample(seg_path, abdose_or_path)
    resampled_seg_sim = read_resample(seg_path, abdose_sim_path)
    
    keys = {1:'spleen', 2:'right kidney', 3:'left kidney', 5:'liver', 22:'tumour 1', 23:'tumour 2'} #segmentation keys, if TotalSegmentator is used
    xlimits = {
    1: (0, 4100),   # spleen
    2: (0, 9000),   # right kidney
    3: (0, 7000),   # left kidney
    5: (0, 3000)  # liver
}
    real_color = (155/255, 70/255, 31/255) 
    simulated_color = (66/255, 53/255, 76/255)  

    valid_organs = []
    for value in np.unique(resampled_seg_or):
        if value != 21 and value != 0 and value != 28 and value in keys:
            mask_or = (resampled_seg_or == value)
            mask_sim = (resampled_seg_sim == value)
            dose_or = abdose_or[mask_or]
            dose_sim = abdose_sim[mask_sim]
            if len(dose_or) > 0 and len(dose_sim) > 0:
                valid_organs.append(value)
    
    # Calculate subplot layout
    n_organs = len(valid_organs)
    if n_organs == 0:
        return
    
    ncols = 3  
    nrows = (n_organs + ncols - 1) // ncols  
    
    # Create subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5*nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1) 
    
    axes_flat = axes.flatten()
    
    for idx, value in enumerate(valid_organs):
        ax = axes_flat[idx]
        
        mask_or = (resampled_seg_or == value)
        mask_sim = (resampled_seg_sim == value)
        dose_or = abdose_or[mask_or]
        dose_sim = abdose_sim[mask_sim]
        
        dose_bins = np.linspace(0, max(dose_or.max(), dose_sim.max()), 200)
        hist_or, bin_edges = np.histogram(dose_or, bins=dose_bins)
        cum_or = np.cumsum(hist_or[::-1])[::-1]
        rel_or = cum_or / cum_or[0] if cum_or[0] > 0 else cum_or
        hist_sim, _ = np.histogram(dose_sim, bins=dose_bins)
        cum_sim = np.cumsum(hist_sim[::-1])[::-1]
        rel_sim = cum_sim / cum_sim[0] if cum_sim[0] > 0 else cum_sim
        
        ax.plot(bin_edges[:-1], rel_or, 'k-',color= real_color,  linewidth=2.5, label=r'Original')
        ax.plot(bin_edges[:-1], rel_sim, 'k--',color= simulated_color, linewidth=2.5, label=r'Simulated')
       
        ax.set_xlabel(r'Dose (mGy)')
        ax.set_title(rf'{keys[value].title()}')
        
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.legend(frameon=True, fancybox=False, edgecolor='black', 
                 facecolor='white', framealpha=1)
        
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
            spine.set_color('k')
        
        ax.tick_params(axis='both', which='major', labelsize=18, width=1.2, length=6, colors='k')
        ax.tick_params(axis='both', which='minor', width=1, length=3, colors='k')

        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_color('k')
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_color('k')
        
        if value in xlimits:
            ax.set_xlim(xlimits[value])
        else:
            ax.set_xlim(0, max(dose_or.max(), dose_sim.max()) * 1.05)
        ax.set_ylim(0, 1.05)
    
    for idx in range(n_organs, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    plt.tight_layout()

    plt.savefig('dvh_plots_p2_1.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.show()


def setup_latex():
    """Use this if you don't have LaTeX installed but want similar styling"""
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 18,
        "mathtext.fontset": "cm",  
        "axes.labelsize": 18,
        "axes.titlesize": 22,
        "legend.fontsize": 16,
        "xtick.labelsize": 30,
        "ytick.labelsize": 30,
        "figure.figsize": [6, 4],
        "axes.linewidth": 1.2,
        "grid.linewidth": 0.8,
        "lines.linewidth": 3,
        "text.color": "k",       
        "xtick.color": "k",         
        "ytick.color": "k",        
        "axes.edgecolor": "k",     
    })














