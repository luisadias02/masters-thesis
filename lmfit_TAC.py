import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
import pandas as pd
from scipy.integrate import quad
import os

#A0_administered = 6783 #MBq for p1
#A0_administered = 6810 #MBq for p2


def triexponential_tac(t:float, k1:float, k2:float, k3:float, A2:float, A3:float):
    """
    Triexponential TAC model (k1, k2, k3, A1, A2, A3 > 0)
    """
    A1 = A2 + A3  
    return -A1 * np.exp(-k1 * t) + A2 * np.exp(-k2 * t) + A3 * np.exp(-k3 * t)

def prepare_data(dict_vois:dict):
    """
    prepares the data from the dict, where the first value correspondes to the activity 
    and the second to the std

    """

    times = np.array(list(dict_vois.keys()))
    activities = np.array([dict_vois[t][0] for t in times])
    uncertainties = np.array([dict_vois[t][1] for t in times])
    return times, activities, uncertainties

def triexponential_tac_computation(dict_vois:dict, plot=True, voi=None):

    """
    tri-exponential fit to the data points using the lmfit (non-linear least squares minimization) 
    Bounds and initial guesses were assigned to each VOI. 
    Activity in the dictionary must be in MBq, and the time points in hours.

    Returns:
    - The parameters k1, k2, k3, A1, A2, A3 that best fits the data points.

    """

    t_data, y_data, sigma = prepare_data(dict_vois)

    model = Model(triexponential_tac)

    params = model.make_params(
        k1=1, k2=0.5, k3= 0.01, A2= y_data[1]*0.7, A3= y_data[2]*0.5)
    
    params['k1'].min = 1e-6
    params['k2'].min = 5e-6
    params['k3'].min = 1e-7
    params['A2'].min = 1e-3
    params['A3'].min = 1e-3
    params['k1'].max = 2
    params['k2'].max = 0.5
    params['k3'].max = 0.1

    if voi == 'liver':
        params['A2'].max =  y_data[1]*3
        params['A3'].max =  y_data[1]*3

    elif voi == 'spleen' or voi=='remaining' or voi=='bladder' :
        params['A2'].max =  y_data[1]
        params['A3'].max =  y_data[1]

    elif voi == 'right kidney':
        params['A2'].max =  y_data[1]*1.3
        params['A3'].max =  y_data[1]*1.3
    
    elif voi == 'left kidney':
        params['A2'].max =  y_data[1]*2
        params['A3'].max =  y_data[1]*2

    else:
        params['A2'].max = np.inf
        params['A3'].max = np.inf

    result = model.fit(y_data, params, t=t_data, weights=1/sigma)

    if plot:
        plt.errorbar(t_data[:-1], y_data[:-1], yerr=sigma[:-1], fmt='o', label='Data')
        t_fit = np.linspace(min(t_data), 135, 200)
        y_fit = model.eval(result.params, t=t_fit)
        plt.plot(t_fit, y_fit, label='Fit')
        plt.xlabel('Time')
        plt.ylabel('Activity')
        plt.grid()
        plt.legend()
        plt.title('Triexponential Fit with lmfit')
        plt.show()

    return result

def extract_values(result):

    k1 = result.params['k1'].value
    k2 = result.params['k2'].value
    A2 = result.params['A2'].value
    k3 = result.params['k3'].value
    A3 = result.params['A3'].value
    
    return k1, k2, k3, A2, A3 

def integration(k1:float, k2:float, k3:float, A1:float, A2:float, A3:float):

    """
    integrates the function to infinity, using equation 2.10 from the thesis.

    Returns:
    - The time integrated activity in MBq*h
    """

    return (-A1/k1) + (A2/k2) + (A3/k3)

def calculate_r2(y_real, y_predicted):
    """
    Calculates R-squared to evaluate the fit
    """
    ss_res = np.sum((y_real - y_predicted) ** 2) 
    ss_tot = np.sum((y_real - np.mean(y_real)) ** 2) 
    r2 = 1 - (ss_res / ss_tot)
    return r2

def total_computation(dict_total:dict, A0_administered:float, output_path:str):

    """
    performs the fit to each VOI, for real and simulated (SIMIND) activity data and 
    time points. 

    Args:
        dict_total (dict):  dictionary whose keys are the VOIs name and the value 
                            is another dictionary with the activity in each time point.
        
        A0_administered (float): injected activity (MBq)

        output_path (str): path to the folder where output files are saved

    Returns:
    - The data frame containing k1, k2, k3, A1, A2, A3 fitted for each VOI;
    - A dictionary of the cumulated activities (in MBq*s)
    """

    results=[]
    dicts_cumulated= {}
    with open(os.path.join(output_path, 'cumulated_activities.txt'), 'w') as f:

        for voi in dict_total.keys():

            original= dict_total[voi][0]
            simulated= dict_total[voi][1]

            t_data_or, y_data_or, sigma_or = prepare_data(original)
            t_data_sim, y_data_sim, sigma_sim = prepare_data(simulated)

            time_points = np.array([0, 4, 24, 144, 200]) #for plotting
            t_fit = np.linspace(time_points.min(), time_points.max(), 100)

            result_or= triexponential_tac_computation(original, plot=False, voi= voi)
            result_sim= triexponential_tac_computation(simulated, plot=False, voi= voi)

            k1_or, k2_or, k3_or, A2_or, A3_or = extract_values(result_or)  
            A1_or= A2_or + A3_or
            activity_or= integration(k1_or, k2_or, k3_or, A1_or, A2_or, A3_or)

            k1_sim, k2_sim, k3_sim, A2_sim, A3_sim  = extract_values(result_sim) 
            A1_sim= A2_sim + A3_sim
            activity_sim= integration(k1_sim, k2_sim, k3_sim, A1_sim, A2_sim, A3_sim)

            fit_or = -A1_or * np.exp(-k1_or * t_fit) + A2_or * np.exp(-k2_or * t_fit) + A3_or * np.exp(-k3_or * t_fit)
            fit_sim = -A1_sim * np.exp(-k1_sim * t_fit) + A2_sim * np.exp(-k2_sim * t_fit) + A3_sim * np.exp(-k3_sim * t_fit)

            y_pred_or = -A1_or * np.exp(-k1_or * t_data_or[:-1]) + A2_or * np.exp(-k2_or * t_data_or[:-1]) + A3_or * np.exp(-k3_or * t_data_or[:-1])
            y_pred_sim = -A1_sim * np.exp(-k1_sim * t_data_sim[:-1]) + A2_sim * np.exp(-k2_sim * t_data_sim[:-1]) + A3_sim * np.exp(-k3_sim * t_data_sim[:-1])
            r2_or = calculate_r2(y_data_or[:-1], y_pred_or)
            r2_sim = calculate_r2(y_data_sim[:-1], y_pred_sim)

            results.append({
                'VOI': voi,
                'k1 original': float(k1_or),
                'k2 original': float(k2_or),
                'k3 original': float(k3_or),
                'A1 original': float(A1_or),
                'A2 original': float(A2_or),
                'A3 original': float(A3_or),
                'R2 original': float(r2_or),
                'k1 simulated': float(k1_sim),
                'k2 simulated': float(k2_sim),
                'k3 simulated': float(k3_sim),
                'A1 simulated': float(A1_sim),
                'A2 simulated': float(A2_sim),
                'A3 simulated': float(A3_sim),
                'R2 simulated': float(r2_sim),
            })

            plt.style.use('seaborn-v0_8-whitegrid')
            plt.rcParams.update({
                'font.size': 14,
                'axes.labelsize': 16,
                'axes.titlesize': 18,
                'legend.fontsize': 12,
                'xtick.labelsize': 12,
                'ytick.labelsize': 12,
                'lines.linewidth': 2,
                'lines.markersize': 8
            })

            soft_orange = '#fc8d62'   
            soft_green = '#005000'     

            fig, ax = plt.subplots(figsize=(7, 5))

            ax.errorbar(
                t_data_or[:-1], 
                100 * y_data_or[:-1] / A0_administered,
                yerr=100 * sigma_or[:-1] / A0_administered,
                fmt='o', 
                color=soft_green,
                ecolor=soft_green,
                elinewidth=1.5,
                capsize=4,
                #label=f'{voi} original data'
            )

            ax.errorbar(
                t_data_sim[:-1], 
                100 * y_data_sim[:-1] / A0_administered,
                yerr=100 * sigma_sim[:-1] / A0_administered,
                fmt='s',
                color=soft_orange,
                ecolor=soft_orange,
                elinewidth=1.5,
                capsize=4,
                label=f'{voi} simulated data'
            )

            ax.plot(
                t_fit, 
                100 * fit_or / A0_administered, 
                '-', 
                color=soft_green,
                label=f'{voi} original fit'
            )

            ax.plot(
                t_fit, 
                100 * fit_sim / A0_administered, 
                '--',
                color=soft_orange,
                label=f'{voi} simulated fit'
            )

            ax.set_title(f'TAC Fit for {voi}')
            ax.set_xlabel('Time post-injection (h)')
            ax.set_ylabel('% of Injected Activity')

            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.set_axisbelow(True)

            plt.tight_layout()
            plt.show()

            
            print(voi)
            print('Original activity:', round(activity_or, 2), 'MBq*h')
            print('Simulated activity:', round(activity_sim, 2), 'MBq*h')
            print('Difference(%):', round(100*((activity_sim-activity_or)/activity_or), 2))

            f.write(voi)
            f.write(f' original: {round(activity_or, 2)} MBq*h \n')
            f.write(f' simulated: {round(activity_sim, 2)} MBq*h \n')
            f.write(f' difference (%): {round(100*((activity_sim-activity_or)/activity_or), 2)} \n\n')


            dicts_cumulated[voi] = [(activity_or) * 3600, (activity_sim) * 3600] #CONVERSION FROM MBq*h TO MBq*s

    df = pd.DataFrame(results)
    print('\n', df)
    print('\n')
    print('returned dict of cumulated activity in MBq*s')

    return df, dicts_cumulated

def dict_conversion(dict_total:dict, pixel_size:tuple):

    """
    conversion of the integrated activity values from MBq*s to MBq*s/mL
    
    """

    pixel_volume= (pixel_size[0]*pixel_size[1]*pixel_size[2])*(1e-3) #ml conversion

    dict_rescaled= dict_total.copy()
    for key, value in dict_total.items():
        dict_rescaled[key] = [v / (pixel_volume) for v in value]
    print('returning cumulated activities in MBq*s/mL')

    return dict_rescaled

























