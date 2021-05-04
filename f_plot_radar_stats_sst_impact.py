#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 20:07:52 2021

@author: claudia
"""


# importing necessary libraries
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams
import matplotlib
import netCDF4 as nc4
from netCDF4 import Dataset
import glob
import os.path
import pandas as pd
import numpy as np
import xarray as xr
import scipy.integrate as integrate
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from scipy.signal import chirp, find_peaks, peak_widths, peak_prominences


def f_plot_2dhist(hmax, dict_input):

    strTitle = dict_input['title']
    cbarstr = dict_input['xlabel']
    bins = dict_input['bins']
    xvar = dict_input['xvar']
    yvar = dict_input['yvar']
    xmin = dict_input['xmin']
    xmax = dict_input['xmax']
    
    # plot 2d histogram figure 
    i_good = (~np.isnan(xvar) * ~np.isnan(yvar))
    hst, xedge, yedge = np.histogram2d(xvar[i_good], yvar[i_good], bins=bins)
    xcenter = (xedge[:-1] + xedge[1:])*0.5
    ycenter = (yedge[:-1] + yedge[1:])*0.5
    hst = hst.T
    hst[hst==0] = np.nan
    
    
    hmin         = 100.
    hmax         = 2500.
    labelsizeaxes = 18
    fontSizeTitle = 18
    fontSizeX     = 18
    fontSizeY     = 18
    cbarAspect    = 10
    fontSizeCbar  = 16
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,8))
    rcParams['font.sans-serif'] = ['Tahoma']
    matplotlib.rcParams['savefig.dpi'] = 100
    plt.gcf().subplots_adjust(bottom=0.15)
    fig.tight_layout()
    ax = plt.subplot(1,1,1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    matplotlib.rc('xtick', labelsize=labelsizeaxes)  # sets dimension of ticks in the plots
    matplotlib.rc('ytick', labelsize=labelsizeaxes)  # sets dimension of ticks in the plots
    cax = ax.pcolormesh(xcenter, ycenter, hst, cmap=dict_input['cmap'])
    ax.set_ylim(100.,hmax)                                               # limits of the y-axesn  cmap=plt.cm.get_cmap("viridis", 256)
    ax.set_xlim(xmin, xmax)                                 # limits of the x-axes
    ax.set_title(strTitle, fontsize=fontSizeTitle, loc='left')
    ax.set_xlabel(cbarstr, fontsize=fontSizeX)
    ax.set_ylabel("Height [m]", fontsize=fontSizeY)
    cbar = fig.colorbar(cax, orientation='vertical', shrink=0.75)
    cbar.set_label(label='Occurrences', size=fontSizeCbar)
    cbar.ax.tick_params(labelsize=labelsizeaxes)
    # Turn on the frame for the twin axis, but then hide all
    # but the bottom spine
    
    fig.tight_layout()
    fig.savefig('{path}{varname}_{info}_{dataset}_2d_CFAD.png'.format(**dict_input), bbox_inches='tight')
def f_calc_tot_cloud_fraction(matrix):
    '''
    function to calculate the total cloud fraction of a matrix (time, height)
    

    Parameters
    ----------
    matrix : ndarray (time, height)
        DESCRIPTION. reflectivity values

    Returns
    -------
    cloud fraction ndarray(time)

    '''
    #defining ndarray to contain cloud fraction
    cloud_fraction = []
    N_tot = matrix.shape[0]
    for ind_height in range(matrix.shape[1]):
        cloud_fraction.append(np.sum(~np.isnan(matrix[:,ind_height]))/N_tot)


    return(np.asarray(cloud_fraction))


# reading radar file list
file_list = np.sort(glob.glob("/Volumes/Extreme SSD/ship_motion_correction_merian/corrected_data/eurec4a_intake/*.nc"))


# set the processing mode keyword for the data you want to be processed
processing_mode = 'case_study' # 'all_campaign' #

# setting time window to be checked
if processing_mode == 'case_study':
    string_out = '20200202_20200203'
    sst_flag_file = "20200202_20200203_sst_flag10_90_perc.nc"
    t_start = datetime(2020, 2, 2, 0, 0, 0)
    t_end = datetime(2020, 2, 3, 23, 59, 59)
    x_max_cb = 2000.
    x_min_cb = 400.
    file_list = file_list[14:16]

else:
    string_out = '20200119_20200219'
    sst_flag_file = "20200119_20200219_sst_flag10_90_perc.nc"
    t_start = datetime(2020, 1, 19, 0, 0, 0)
    t_end = datetime(2020, 2, 19, 23, 59, 59)
    x_max_cb = 4500.
    x_min_cb = 0.





# combining all radar data in one dataset
radar_data = xr.open_mfdataset(file_list,
                               concat_dim = 'time',
                               data_vars = 'minimal',
                              )
#%%
# selecting data in the time window of the surface anomaly
data_sliced = radar_data.sel(time=slice(t_start, t_end))


# reading sst data
path_sst_flag = "/Volumes/Extreme SSD/work/006_projects/001_Prec_Trade_Cycle/\
SST_impact_work/"
sst_flag = xr.open_dataset(path_sst_flag+sst_flag_file)

# setting path for figures
path_fig = "/Volumes/Extreme SSD/work/006_projects/001_Prec_Trade_Cycle/\
SST_impact_work/plots/"

#%%
# interpolating sst data at 1 s resolution to the 10 s res of the wind lidar
sst_data_interp = sst_flag.interp(time=data_sliced['time'].values)

# merging the interpolated dataset and the wind lidar dataset
data_merged = xr.merge([data_sliced, sst_data_interp])


#%%
# calculating cloud fraction for cold and warm anomaly
cloud_fraction_cold = f_calc_tot_cloud_fraction(Ze_cold)
cloud_fraction_warm = f_calc_tot_cloud_fraction(Ze_warm)


hmin         = 100.
hmax         = 2200.
labelsizeaxes = 18
fontSizeTitle = 18
fontSizeX     = 18
fontSizeY     = 18
cbarAspect    = 10
fontSizeCbar  = 16

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,12))
rcParams['font.sans-serif'] = ['Tahoma']
matplotlib.rcParams['savefig.dpi'] = 100
plt.gcf().subplots_adjust(bottom=0.15)
fig.tight_layout()
ax = plt.subplot(1,1,1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
matplotlib.rc('xtick', labelsize=labelsizeaxes)  # sets dimension of ticks in the plots
matplotlib.rc('ytick', labelsize=labelsizeaxes)  # sets dimension of ticks in the plots
ax.plot(cloud_fraction_cold, data_merged.height.values, label='cloud fraction on cold SST', color='b', linewidth=4.0)
ax.plot(cloud_fraction_warm, data_merged.height.values, label='cloud fraction on warm SST', color='r', linewidth=4.0)
ax.legend(frameon=False, fontsize=fontSizeY)
ax.set_ylim(100.,hmax)                                               # limits of the y-axesn  cmap=plt.cm.get_cmap("viridis", 256)
#ax.set_xlim(xmin, xmax)                                 # limits of the x-axes
ax.set_title('Cloud fraction for 2-3 Feb 2020', fontsize=fontSizeTitle, loc='left')
ax.set_ylabel("Height [m]", fontsize=fontSizeY)
ax.set_xlabel("Hydrometeor fraction []", fontsize=fontSizeY)

# Turn on the frame for the twin axis, but then hide all
# but the bottom spine
fig.tight_layout()
fig.savefig(path_fig+'__cloud_fractions_2_3_feb.png', bbox_inches='tight')

#%%
# plot CFAD for all variables for cold and warm anomaly.
var_plot = ['ze_cold', 'Ze_warm', 'Vd_cold', 'Vd_warm', 'Sw_cold', 'Sw_warm']

for ivar, var in enumerate(var_plot):
    
    print('plotting CFAD for ', var)
    if var == 'ze_cold':
        dict_input = {'xvar':(10.*np.log10(Ze_cold)).flatten(), 
                      'yvar':(np.ones((len(time_cold),1))*np.array([data_merged.height.values])).flatten(), 
                      'xlabel':'Ze [dBz]',
                      'title':'Reflectivity over cold SST anomaly',
                      'varname':'Ze_cold',
                      'info':'cold', 
                      'path':path_fig,
                      'bins':[100,50],
                      'cmap':'viridis',
                      'xmax':20., 
                      'xmin':-50.,  
                      'dataset':processing_mode,
                      }
    if var == 'Ze_warm':
        dict_input = {'xvar':(10.*np.log10(Ze_warm)).flatten(), 
                      'yvar':(np.ones((len(time_warm),1))*np.array([data_merged.height.values])).flatten(), 
                      'xlabel':'Ze [dBz]',
                      'title':'Reflectivity over warm SST anomaly',
                      'varname':'Ze_warm',
                      'info':'warm', 
                      'path':path_fig,
                      'bins':[100,50],
                      'cmap':'viridis',
                      'xmax':20., 
                      'xmin':-50.,
                      'dataset':processing_mode,
                      }
    if var == 'Vd_cold':
        dict_input = {'xvar':Vd_cold.flatten(), 
                      'yvar':(np.ones((len(time_cold),1))*np.array([data_merged.height.values])).flatten(), 
                      'xlabel':'Vd [m$s^{-1}$]',
                      'title':'Mean Doppler velocity over cold SST anomaly',
                      'varname':'Vd_cold',
                      'info':'cold', 
                      'path':path_fig,
                      'bins':[100,50],
                      'cmap':'viridis',
                      'xmax':4., 
                      'xmin':-6.,
                      'dataset':processing_mode,
                      }
    if var == 'Vd_warm':
        dict_input = {'xvar':Vd_warm.flatten(), 
                      'yvar':(np.ones((len(time_warm),1))*np.array([data_merged.height.values])).flatten(), 
                      'xlabel':'Vd [m$s^{-1}$]',
                      'title':'Mean Doppler velocity over warm SST anomaly',
                      'varname':'Vd_warm',
                      'info':'warm', 
                      'path':path_fig,
                      'bins':[100,50],
                      'cmap':'viridis',
                      'xmax':4., 
                      'xmin':-6.,
                      'dataset':processing_mode,
                      }
    if var == 'Sw_cold':
        dict_input = {'xvar':Sw_cold.flatten(), 
                      'yvar':(np.ones((len(time_cold),1))*np.array([data_merged.height.values])).flatten(), 
                      'xlabel':'Sw [m$s^{-1}$]',
                      'title':'Spectral width over cold SST anomaly',
                      'varname':'Sw_cold',
                      'info':'cold', 
                      'path':path_fig,
                      'bins':[100,50],
                      'cmap':'viridis',
                       'xmax':3., 
                      'xmin':0.,
                      'dataset':processing_mode,
                      }
    if var == 'Sw_warm':
        dict_input = {'xvar':Sw_warm.flatten(), 
                      'yvar':(np.ones((len(time_warm),1))*np.array([data_merged.height.values])).flatten(), 
                      'xlabel':'Sw [m$s^{-1}$]',
                      'title':'Spectral width over warm SST anomaly',
                      'varname':'Sw_warm',
                      'info':'warm', 
                      'path':path_fig,
                      'bins':[100,50],
                      'cmap':'viridis',
                      'xmax':3., 
                      'xmin':0.,
                      'dataset':processing_mode,
                      }
        

    
    f_plot_2dhist(2500, dict_input)
    print('plot done')
    print('********************************')
    
    
#%%

# reading arthus data
path_arthus = '/Volumes/Extreme SSD/work/006_projects/001_Prec_Trade_Cycle/data_arthus/'
arthus_files = np.sort(glob.glob(path_arthus+"/*.cdf"))

# removing time duplicates in the two datasets
arthus_1 = xr.open_dataset(arthus_files[0])
arthus_2 = xr.open_dataset(arthus_files[1])
_, index = np.unique(arthus_1['Time'], return_index=True)
arthus_1 = arthus_1.isel(Time=index)
_, index = np.unique(arthus_2['Time'], return_index=True)
arthus_2 = arthus_2.isel(Time=index)

# merging them to create one single arthus data file
arthus_data = xr.merge([arthus_1, arthus_2])

# interpolating sst data at 1 s resolution to the 10 s res of the wind lidar
sst_arthus_interp = sst_flag.interp(time=arthus_data['Time'].values)

# merging the interpolated dataset and the wind lidar dataset
data_arthus_merged = xr.merge([arthus_data, sst_arthus_interp])


# selecting data in the lowest 300 m and from 500 to 800 m
data_surf = data_arthus_merged.sel(Height=slice(0., 300.))
data_cloud = data_arthus_merged.sel(Height=slice(300., 700.))

# calculating mean over height for both datasets
data_surf_mean = data_surf.mean(dim='Height', skipna=True)
data_cloud_mean = data_cloud.mean(dim='Height', skipna=True)

LH_surf = data_surf_mean['Latent_heat_flux'].values
LH_subcloud = data_cloud_mean['Latent_heat_flux'].values
flag = data_surf_mean['flag'].values

i_cold = np.where(flag == 1)
i_warm = np.where(flag == 2)

LH_cold_surf = LH_surf[i_cold]
LH_warm_surf = LH_surf[i_warm]

LH_cold_subcloud = LH_subcloud[i_cold]
LH_warm_subcloud = LH_subcloud[i_warm]

#%%

# plotting histogram of vertical wind averaged in the first 300 m
labelsizeaxes = 12
fontSTitle = 12
fontSizeX = 12
fontSizeY = 12
cbarAspect = 10
fontSizeCbar = 12
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
rcParams['font.sans-serif'] = ['Tahoma']
matplotlib.rcParams['savefig.dpi'] = 100
plt.gcf().subplots_adjust(bottom=0.15)
fig.tight_layout()
ax = plt.subplot(1, 2, 1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
matplotlib.rc('xtick', labelsize=labelsizeaxes)  # sets dimension of ticks
matplotlib.rc('ytick', labelsize=labelsizeaxes)  # sets dimension of ticks
ax.hist(LH_warm_surf, bins=5, color='red', label='LHF on warm sst percentile', histtype='step')
ax.hist(LH_cold_surf, bins=5, color='blue', label='LHF on cold sst percentile', histtype='step')
ax.legend(frameon=False)
ax.set_xlim([-50.,250.])
ax.set_title('LHF between 0 and 300m for :'+string_out, fontsize=fontSTitle, loc='left')
ax.set_xlabel("LHF [$Wm^{-2}$]", fontsize=fontSizeX)
ax.set_ylabel("occurrences [#]", fontsize=fontSizeY)

ax = plt.subplot(1, 2, 2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
matplotlib.rc('xtick', labelsize=labelsizeaxes)  # sets dimension of ticks
matplotlib.rc('ytick', labelsize=labelsizeaxes)  # sets dimension of ticks
ax.hist(LH_warm_subcloud, bins=5, color='red', label='LHF on warm sst percentile', histtype='step')
ax.hist(LH_cold_subcloud, bins=5, color='blue', label='LHF  on cold sst percentile', histtype='step')
ax.legend(frameon=False)
ax.set_xlim([-50.,250.])

ax.set_title('LHF between 300 and 700m histograms for :'+string_out, fontsize=fontSTitle, loc='left')
ax.set_xlabel("LHF [$Wm^{-2}$]", fontsize=fontSizeX)
ax.set_ylabel("occurrences [#]", fontsize=fontSizeY)
fig.tight_layout()
fig.savefig(path_fig+string_out+'_LHF_histogram.png', format='png')



