#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 10:42:54 2020
@ date; 14 april 2021
@author: Claudia Acquistapace
@goal: elaborate stats of main variables of interest (vertical wind,
cloud fraction, integrated water vapor, w, LH, SH, TKE
as a function of height (e.g. 600-1000 m LH in the range of cloud formation)
"""

# importing necessary libraries
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib
import numpy as np
import xarray as xr
from datetime import datetime
import matplotlib.dates as mdates
import glob

# set the processing mode keyword for the data you want to be processed
processing_mode = 'case_study' # 'all_campaign''all_campaign' #

# setting time window to be checked
if processing_mode == 'case_study':
    string_out = '20200202_20200203'
    sst_flag_file = "20200202_20200203_sst_flag10_90_perc.nc"
    t_start = datetime(2020, 2, 2, 0, 0, 0)
    t_end = datetime(2020, 2, 3, 23, 59, 59)
    x_max_cb = 2000.
    x_min_cb = 400.

else:
    string_out = '20200119_20200219'
    sst_flag_file = "20200119_20200219_sst_flag10_90_perc.nc"
    t_start = datetime(2020, 1, 19, 0, 0, 0)
    t_end = datetime(2020, 2, 19, 23, 59, 59)
    x_max_cb = 4500.
    x_min_cb = 0.


# reading sst data
path_sst_flag = "/Volumes/Extreme SSD/work/006_projects/001_Prec_Trade_Cycle/\
SST_impact_work/"
sst_flag = xr.open_dataset(path_sst_flag+sst_flag_file)

#%%

# read wind lidar data
path_wind_lidar = "/Volumes/Extreme SSD/work/006_projects/001_Prec_Trade_Cycle/data_wind_lidar/"
wind_lidar_file = "wind_lidar_eurec4a.nc"
path_fig = "/Volumes/Extreme SSD/work/006_projects/001_Prec_Trade_Cycle/\
SST_impact_work/plots/"
wind_lidar = xr.open_dataset(path_wind_lidar+wind_lidar_file)

# selecting time interval to extract from wind lidar dataset
wind_lidar_slice = wind_lidar.sel(time=slice(t_start, t_end))

# interpolating sst data at 1 s resolution to the 10 s res of the wind lidar
sst_data_interp = sst_flag.interp(time=wind_lidar_slice['time'].values)

# merging the interpolated dataset and the wind lidar dataset
data_merged = xr.merge([wind_lidar_slice, sst_data_interp])

# selecting data in the lowest 300 m and from 500 to 800 m
data_surf = data_merged.sel(height=slice(0., 300.))
data_cloud = data_merged.sel(height=slice(500., 1000.))

# calculating mean over height for both datasets
data_surf_mean = data_surf.mean(dim='height', skipna=True)
w_surf = data_surf_mean['w'].values
cb_surf = data_surf_mean['cb'].values
flag = data_surf_mean['flag'].values


# selecting cold and warm dataset indeces
i_cold = np.where(flag == 1)
i_warm = np.where(flag == 2)

# reading wind and cloud base
w_cold = w_surf[i_cold]
w_warm = w_surf[i_warm]
cb_cold = cb_surf[i_cold]
cb_warm = cb_surf[i_warm]



# plotting histogram of vertical wind averaged in the first 300 m
labelsizeaxes    = 32
fontSizeTitle    = 32
fontSizeX        = 32
fontSizeY        = 32
cbarAspect       = 10
fontSizeCbar     = 32
fig, axs = plt.subplots(1, 2, figsize=(24,14), constrained_layout=True)

# setting dates formatter 
#[a.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) for a in axs[:].flatten()]
matplotlib.rc('xtick', labelsize=32)  # sets dimension of ticks in the plots
matplotlib.rc('ytick', labelsize=32)  # sets dimension of ticks in the plots
grid            = True

#axs[0].plot(time_mrr, profile_mrr, color='white')
axs[0].spines["top"].set_visible(False)
axs[0].spines["right"].set_visible(False)
axs[0].get_xaxis().tick_bottom()
axs[0].get_yaxis().tick_left()
axs[0].spines["bottom"].set_linewidth(2)
axs[0].spines["left"].set_linewidth(2)
axs[0].tick_params(which='minor', length=7, width=3)
axs[0].tick_params(which='major', length=7, width=3)
axs[0].hist(w_warm, bins=30, color='red', label='w on warm sst percentile', histtype='step', lw=3)
axs[0].hist(w_cold, bins=30, color='blue', label='w on cold sst percentile', histtype='step', lw=3)
axs[0].legend(frameon=False, fontsize=24)
axs[0].set_xlim([-2.,2.])
axs[0].set_title('w histograms for :'+string_out, fontsize=32, loc='left')
axs[0].set_xlabel("w [ms$^{-1}$]", fontsize=fontSizeX)
axs[0].set_ylabel("occurrences [#]", fontsize=fontSizeY)


axs[1].spines["top"].set_visible(False)
axs[1].spines["right"].set_visible(False)
axs[1].get_xaxis().tick_bottom()
axs[1].get_yaxis().tick_left()
axs[1].spines["bottom"].set_linewidth(2)
axs[1].spines["left"].set_linewidth(2)
axs[1].tick_params(which='minor', length=7, width=3)
axs[1].tick_params(which='major', length=7, width=3)
axs[1].hist(cb_warm, bins=30, color='red', label='cb height on warm sst percentile', histtype='step', lw=3)
axs[1].hist(cb_cold, bins=30, color='blue', label='cb height on cold sst percentile', histtype='step', lw=3)
axs[1].set_xlim([x_min_cb,x_max_cb])
axs[1].legend(frameon=False, fontsize=24)
axs[1].set_title('cloud base histograms for :'+string_out, fontsize=32, loc='left')
axs[1].set_xlabel("clooud base height [m]", fontsize=fontSizeX)
axs[1].set_ylabel("occurrences [#]", fontsize=fontSizeY)

fig.tight_layout()
fig.savefig(path_fig+string_out+'_w_cb_histogram.png', format='png')
#%%

# reading arthus data
path_arthus = '/Volumes/Extreme SSD/work/006_projects/001_Prec_Trade_Cycle/data_arthus/case_1/'
arthus_files = np.sort(glob.glob(path_arthus+"/LHF*.cdf"))

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
# reading water vapor mixing ratio data from Diego
path_arthus = '/Volumes/Extreme SSD/work/006_projects/001_Prec_Trade_Cycle/data_arthus/case_1/'
arthus_wvmr_files = np.sort(glob.glob(path_arthus+"/ARTHUS_WVMR_*.cdf"))

# removing time duplicates in the two datasets
arthus_wvmr_1 = xr.open_dataset(arthus_wvmr_files[0])
arthus_wvmr_2 = xr.open_dataset(arthus_wvmr_files[1])

# removing double time stamps in time arrays
_, index = np.unique(arthus_wvmr_1['Time'], return_index=True)
arthus_wvmr_1 = arthus_wvmr_1.isel(Time=index)
_, index = np.unique(arthus_wvmr_2['Time'], return_index=True)
arthus_wvmr_2 = arthus_wvmr_2.isel(Time=index)

# merging them to create one single arthus data file
arthus_data = xr.merge([arthus_wvmr_1, arthus_wvmr_2])


# resampling arthus data on time array of SST
arthus_sst = arthus_data.reindex({'Time':sst_flag.time.values}, method=None)

# merging the sst dataset and the arthus dataset
data_arthus_merged = xr.merge([arthus_sst, sst_flag])


# selecting cold and warm time stamps
i_cold = np.where(data_arthus_merged.flag.values == 1)
i_warm = np.where(data_arthus_merged.flag.values == 2)

wvmr_cold = data_arthus_merged.Water_vapor_mixing_ratio.values[i_cold,:]
wvmr_warm = data_arthus_merged.Water_vapor_mixing_ratio.values[i_warm,:]
#%%

# reading and merging ARTHUS files for T
arthus_T_files = np.sort(glob.glob(path_arthus+"/ARTHUS_T_*.cdf"))


# removing time duplicates in the two datasets
arthus_T_1 = xr.open_dataset(arthus_T_files[0])
arthus_T_2 = xr.open_dataset(arthus_T_files[1])

# removing double time stamps in time arrays
_, index = np.unique(arthus_T_1['Time'], return_index=True)
arthus_T_1 = arthus_T_1.isel(Time=index)
_, index = np.unique(arthus_T_2['Time'], return_index=True)
arthus_T_2 = arthus_T_2.isel(Time=index)

# merging them to create one single arthus data file
arthus_T_data = xr.merge([arthus_T_1, arthus_T_2])

# resampling arthus data on time array of SST
arthus_T_sst = arthus_T_data.reindex({'Time':sst_flag.time.values}, method=None)

# merging the sst dataset and the arthus dataset
data_arthus_T_merged = xr.merge([arthus_T_sst, sst_flag])

# selecting cold and warm time stamps
i_cold = np.where(data_arthus_T_merged.flag.values == 1)
i_warm = np.where(data_arthus_T_merged.flag.values == 2)

wvmr_cold = data_arthus_T_merged.Temperature.values[i_cold,:]
wvmr_warm = data_arthus_T_merged.Temperature.values[i_warm,:]


#%%
fig, axs = plt.subplots(3, 1, figsize=(16,24), sharex=True, constrained_layout=True)
# set here the variable from ds to plot, its color map and its min and max values
#data_arthus_merged.Water_vapor_mixing_ratio.plot(x='Time', y='Height', cmap="seismic", vmin=0., vmax=100.)

#axs[0].plot(time_mrr, profile_mrr, color='white')
axs[0].spines["top"].set_visible(False)
axs[0].spines["right"].set_visible(False)
axs[0].get_xaxis().tick_bottom()
axs[0].get_yaxis().tick_left()
axs[0].spines["bottom"].set_linewidth(2)
axs[0].spines["left"].set_linewidth(2)
axs[0].tick_params(which='minor', length=7, width=3)
axs[0].tick_params(which='major', length=7, width=3)
axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%d %H')) 
mesh = axs[0].pcolormesh(data_arthus_merged.Time.values, data_arthus_merged.Height.values, data_arthus_merged.Water_vapor_mixing_ratio.values.T, cmap='viridis', vmin=0., vmax=100.)
cbar = fig.colorbar(mesh, ax=axs[0])
cbar.set_label(label='WVMR',  size=32)
axs[0].set_ylabel('Height [m]', fontsize=32)
#axs[0].set_xlabel('Time UTC [hh:mm]', fontsize=32)
#axs[0].set_ylim(0, 3000)


axs[1].spines["top"].set_visible(False)
axs[1].spines["right"].set_visible(False)
axs[1].get_xaxis().tick_bottom()
axs[1].get_yaxis().tick_left()
axs[1].spines["bottom"].set_linewidth(2)
axs[1].spines["left"].set_linewidth(2)
axs[1].tick_params(which='minor', length=7, width=3)
axs[1].tick_params(which='major', length=7, width=3)
axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%d %H')) 

mesh = axs[1].pcolormesh(data_arthus_T_merged.Time.values, data_arthus_T_merged.Height.values, data_arthus_T_merged.Temperature.values.T, cmap='jet', vmin=290., vmax=300.)
cbar = fig.colorbar(mesh, ax=axs[1])
cbar.set_label(label='Temperature [K]',  size=32)
axs[1].set_ylabel('Height [m]', fontsize=32)
#axs[1].set_xlabel('Time UTC [hh:mm]', fontsize=32)


axs[2].spines["top"].set_visible(False)
axs[2].spines["right"].set_visible(False)
axs[2].get_xaxis().tick_bottom()
axs[2].get_yaxis().tick_left()
axs[2].spines["bottom"].set_linewidth(2)
axs[2].spines["left"].set_linewidth(2)
axs[2].tick_params(which='minor', length=7, width=3)
axs[2].tick_params(which='major', length=7, width=3)
axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%d %H')) 
sst = data_arthus_T_merged.sst.values
time = data_arthus_T_merged.Time.values
#axs[2].plot(data_arthus_T_merged.Time.values, data_arthus_T_merged.flag.values)
sst_cold = sst[i_cold]
time_cold = time[i_cold]
sst_warm = sst[i_warm]
time_warm = time[i_warm]
axs[2].plot(data_arthus_T_merged.Time.values, data_arthus_T_merged.sst.values)

axs[2].scatter(time_cold, sst_cold, color='blue', marker="o", s=100, label='cold patch')
axs[2].scatter(time_warm, sst_warm, color='red', marker="o", s=100, label='warm patch')
axs[2].set_xlabel('Time UTC [dd hh]', fontsize=32)
axs[2].set_ylabel('SST [$^{\circ}$C]', fontsize=32)
axs[2].legend(frameon=False, fontsize=32)
fig.savefig(path_fig+string_out+'_WVMR_FLAG.png', format='png')

#%%%
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
labelsizeaxes    = 32
fontSizeTitle    = 32
fontSizeX        = 32
fontSizeY        = 32
cbarAspect       = 10
fontSizeCbar     = 32
fig, axs = plt.subplots(1, 2, figsize=(24,14), constrained_layout=True)

# setting dates formatter 
#[a.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) for a in axs[:].flatten()]
matplotlib.rc('xtick', labelsize=32)  # sets dimension of ticks in the plots
matplotlib.rc('ytick', labelsize=32)  # sets dimension of ticks in the plots
grid            = True

#axs[0].plot(time_mrr, profile_mrr, color='white')
axs[0].spines["top"].set_visible(False)
axs[0].spines["right"].set_visible(False)
axs[0].get_xaxis().tick_bottom()
axs[0].get_yaxis().tick_left()
axs[0].spines["bottom"].set_linewidth(2)
axs[0].spines["left"].set_linewidth(2)
axs[0].tick_params(which='minor', length=7, width=3)
axs[0].tick_params(which='major', length=7, width=3)
axs[0].hist(LH_warm_surf, bins=5, color='red', label='LHF on warm sst percentile', histtype='step', lw=3)
axs[0].hist(LH_cold_surf, bins=5, color='blue', label='LHF on cold sst percentile', histtype='step', lw=3)
axs[0].legend(frameon=False, fontsize=24)
axs[0].set_xlim([-50.,250.])
axs[0].set_title('LHF between 0 and 300m', fontsize=32, loc='left')
axs[0].set_xlabel("LHF [Wm$^{-2}$]", fontsize=fontSizeX)
axs[0].set_ylabel("occurrences [#]", fontsize=fontSizeY)


axs[1].spines["top"].set_visible(False)
axs[1].spines["right"].set_visible(False)
axs[1].get_xaxis().tick_bottom()
axs[1].get_yaxis().tick_left()
axs[1].spines["bottom"].set_linewidth(2)
axs[1].spines["left"].set_linewidth(2)
axs[1].tick_params(which='minor', length=7, width=3)
axs[1].tick_params(which='major', length=7, width=3)
axs[1].hist(LH_warm_subcloud, bins=5, color='red', label='LHF on warm sst percentile', histtype='step', lw=3)
axs[1].hist(LH_cold_subcloud, bins=5, color='blue', label='LHF  on cold sst percentile', histtype='step', lw=3)
axs[1].set_xlim([-50.,250.])
axs[1].legend(frameon=False, fontsize=24)
axs[1].set_title('LHF between 300 and 700m', fontsize=32, loc='left')
axs[1].set_xlabel("LHF [Wm$^{-2}$]", fontsize=fontSizeX)
axs[1].set_ylabel("occurrences [#]", fontsize=fontSizeY)

fig.tight_layout()
fig.savefig(path_fig+string_out+'_LHF_histogram.png', format='png')



#%%
