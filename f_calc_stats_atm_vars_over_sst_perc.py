#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
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


# reading sst data
path_sst_flag = "/Volumes/Extreme SSD/work/006_projects/001_Prec_Trade_Cycle/\
SST_impact_work/"
sst_flag = xr.open_dataset(path_sst_flag+sst_flag_file)

# %%

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
flag = data_surf_mean['flag_tsg'].values


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
# %%

# reading arthus data for latent heat flux
path_arthus = '/Volumes/Extreme SSD/work/006_projects/001_Prec_Trade_Cycle/data_arthus/case_1/'
arthus_files = np.sort(glob.glob(path_arthus+"/*_LHF.cdf"))
arthus_files_SHF = np.sort(glob.glob(path_arthus+"/*_SHF.cdf"))

# removing time duplicates in the two datasets
arthus_1 = xr.open_dataset(arthus_files[0])
arthus_2 = xr.open_dataset(arthus_files[1])
_, index = np.unique(arthus_1['Time'], return_index=True)
arthus_1 = arthus_1.isel(Time=index)
_, index = np.unique(arthus_2['Time'], return_index=True)
arthus_2 = arthus_2.isel(Time=index)


# removing time duplicates in the two datasets
arthus_1_SHF = xr.open_dataset(arthus_files_SHF[0])
arthus_2_SHF = xr.open_dataset(arthus_files_SHF[1])
_, index = np.unique(arthus_1_SHF['Time'], return_index=True)
arthus_1_SHF = arthus_1_SHF.isel(Time=index)
_, index = np.unique(arthus_2_SHF['Time'], return_index=True)
arthus_2_SHF = arthus_2_SHF.isel(Time=index)


# merging them to create one single arthus data file
arthus_data = xr.merge([arthus_1, arthus_2])
arthus_data = arthus_data.rename_dims({'Time':'time'})
arthus_data = arthus_data.rename_dims({'Height':'height'})
arthus_data = arthus_data.rename({'Time':'time'})
arthus_data = arthus_data.rename({'Height':'height'})


# interpolating sst data at 1 s resolution to the 10 s res of the wind lidar
sst_arthus_interp = sst_flag.interp(time=arthus_data['time'].values, method='nearest')

# merging the interpolated dataset and the wind lidar dataset
data_arthus_merged = xr.merge([arthus_data, sst_arthus_interp])

# merging them to create one single arthus data file
arthus_data_SHF = xr.merge([arthus_1_SHF, arthus_2_SHF])
arthus_data_SHF = arthus_data_SHF.rename_dims({'Time':'time'})
arthus_data_SHF = arthus_data_SHF.rename_dims({'Height':'height'})
arthus_data_SHF = arthus_data_SHF.rename({'Time':'time'})
arthus_data_SHF = arthus_data_SHF.rename({'Height':'height'})


# interpolating sst data at 1 s resolution to the 10 s res of the wind lidar
sst_arthus_interp_SHF = sst_flag.interp(time=arthus_data_SHF['time'].values, method='nearest')

# merging the interpolated dataset and the wind lidar dataset
data_arthus_merged_SHF = xr.merge([arthus_data_SHF, sst_arthus_interp_SHF])
# %%

# plotting latent and sensible heat fluxes
fig, axs = plt.subplots(3, 1, figsize=(24,16), sharex=True, constrained_layout=True)
axs[0].spines["top"].set_visible(False)
axs[0].spines["right"].set_visible(False)
axs[0].get_xaxis().tick_bottom()
axs[0].get_yaxis().tick_left()
axs[0].spines["bottom"].set_linewidth(2)
axs[0].spines["left"].set_linewidth(2)
axs[0].tick_params(which='minor', length=7, width=3)
axs[0].tick_params(which='major', length=7, width=3)
axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%d %H')) 
mesh = axs[0].pcolormesh(data_arthus_merged_SHF.time.values, data_arthus_merged_SHF.height.values, data_arthus_merged_SHF.Sensible_heat_flux.values.T, cmap='inferno', vmin=0., vmax=100.)
cbar = fig.colorbar(mesh, ax=axs[0])
cbar.set_label(label='Sensible Heat Flux ',  size=32)
axs[0].set_ylabel('Height [m]', fontsize=32)
axs[0].set_ylim(200,1500)
#axs[0].set_xlim(sliced_ds.time.values[0], sliced_ds.time.values[-1])


axs[1].spines["top"].set_visible(False)
axs[1].spines["right"].set_visible(False)
axs[1].get_xaxis().tick_bottom()
axs[1].get_yaxis().tick_left()
axs[1].spines["bottom"].set_linewidth(2)
axs[1].spines["left"].set_linewidth(2)
axs[1].tick_params(which='minor', length=7, width=3)
axs[1].tick_params(which='major', length=7, width=3)
axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%d %H')) 
mesh = axs[1].pcolormesh(data_arthus_merged.time.values, data_arthus_merged.height.values, data_arthus_merged.Latent_heat_flux.values.T, cmap='viridis', vmin=-5-0., vmax=500.)
cbar = fig.colorbar(mesh, ax=axs[1])
cbar.set_label(label='Latent heat flux [$^{\circ}$K]',  size=32)
axs[1].set_ylabel('Height [m]', fontsize=32)
axs[1].set_ylim(200,1500)
#axs[1].set_xlim(sliced_ds.time.values[0], sliced_ds.time.values[-1])



axs[2].spines["top"].set_visible(False)
axs[2].spines["right"].set_visible(False)
axs[2].get_xaxis().tick_bottom()
axs[2].get_yaxis().tick_left()
axs[2].spines["bottom"].set_linewidth(2)
axs[2].spines["left"].set_linewidth(2)
axs[2].tick_params(which='minor', length=7, width=3)
axs[2].tick_params(which='major', length=7, width=3)
axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%d %H')) 
axs[2].plot(sst_flag.time.values, sst_flag.sst_tsg.values, color='black', linewidth=4)
axs[2].scatter(timetsg_90s, tsg_90s, color='red', label='> 90$^{th}$ percentile', marker="o", s=100)
axs[2].scatter(timetsg_10s, tsg_10s, color='blue', label='<=10$^{th}$ percentile', marker="o", s=100)


axs[2].set_ylabel('SST TSG [$^{\circ}$C]', fontsize=32)
axs[2].set_ylim(26.,28.)
axs[2].set_xlabel('Time UTC [dd hh]', fontsize=32)
axs[2].legend(frameon=False, loc='upper left',fontsize=32)
fig.savefig(path_fig+string_out+'_fluxes_quicklooks_arthus.png', format='png')


# %%

# reading water vapor mixing ratio data from Diego
path_arthus = '/Volumes/Extreme SSD/work/006_projects/001_Prec_Trade_Cycle/data_arthus/case_1/'
arthus_wvmr_file_1 = path_arthus+"/20200202_000008_MRme_10s_97m.cdf"
arthus_wvmr_file_2 = path_arthus+"/20200203_000005_MRme_10s_97m.cdf"

# removing time duplicates in the two datasets
arthus_wvmr_1 = xr.open_dataset(arthus_wvmr_file_1)
arthus_wvmr_2 = xr.open_dataset(arthus_wvmr_file_2)

# removing double time stamps in time arrays
_, index = np.unique(arthus_wvmr_1['Time'], return_index=True)
arthus_data_wvmr_1 = arthus_wvmr_1.isel(Time=index)
_, index = np.unique(arthus_wvmr_2['Time'], return_index=True)
arthus_data_wvmr_2 = arthus_wvmr_2.isel(Time=index)

arthus_all_wvmr = xr.merge([arthus_data_wvmr_1, arthus_data_wvmr_2])

arthus_T_file_1 = path_arthus+"/20200202_000008_Tme_10s_97m.cdf"
arthus_T_file_2 = path_arthus+"/20200203_000005_Tme_10s_97m.cdf"

# removing time duplicates in the two datasets
arthus_T_1 = xr.open_dataset(arthus_T_file_1)
arthus_T_2 = xr.open_dataset(arthus_T_file_2)

# removing double time stamps in time arrays
_, index = np.unique(arthus_T_1['Time'], return_index=True)
arthus_data_T_1 = arthus_T_1.isel(Time=index)
_, index = np.unique(arthus_T_2['Time'], return_index=True)
arthus_data_T_2 = arthus_T_2.isel(Time=index)

arthus_all_T = xr.merge([arthus_data_T_1, arthus_data_T_2])

# %%


# defining flag for data points
flag = sst_flag.flag_tsg.values
sst = sst_flag.sst_tsg.values
time_sst = sst_flag.time.values
tsg_10s = sst[np.where(flag == 1)[0]]
timetsg_10s = time_sst[np.where(flag == 1)[0]]
tsg_90s = sst[np.where(flag == 2)[0]]
timetsg_90s = time_sst[np.where(flag == 2)[0]]

# plot of the WVMR data for the case study
fig, axs = plt.subplots(3, 1, figsize=(24,16), sharex=True, constrained_layout=True)
axs[0].spines["top"].set_visible(False)
axs[0].spines["right"].set_visible(False)
axs[0].get_xaxis().tick_bottom()
axs[0].get_yaxis().tick_left()
axs[0].spines["bottom"].set_linewidth(2)
axs[0].spines["left"].set_linewidth(2)
axs[0].tick_params(which='minor', length=7, width=3)
axs[0].tick_params(which='major', length=7, width=3)
axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%d %H')) 
mesh = axs[0].pcolormesh(arthus_all_wvmr.Time.values, arthus_all_wvmr.Height.values, arthus_all_wvmr.Water_vapor_mixing_ratio.values.T, cmap='inferno', vmin=0., vmax=25.)
cbar = fig.colorbar(mesh, ax=axs[0])
cbar.set_label(label='WVMR [gkg$^{-1}$]',  size=32)
axs[0].set_ylabel('Height [m]', fontsize=32)
axs[0].set_ylim(200,1500)
#axs[0].set_xlim(sliced_ds.time.values[0], sliced_ds.time.values[-1])


axs[1].spines["top"].set_visible(False)
axs[1].spines["right"].set_visible(False)
axs[1].get_xaxis().tick_bottom()
axs[1].get_yaxis().tick_left()
axs[1].spines["bottom"].set_linewidth(2)
axs[1].spines["left"].set_linewidth(2)
axs[1].tick_params(which='minor', length=7, width=3)
axs[1].tick_params(which='major', length=7, width=3)
axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%d %H')) 
mesh = axs[1].pcolormesh(arthus_all_T.Time.values, arthus_all_T.Height.values, arthus_all_T.Temperature.values.T, cmap='viridis', vmin=290., vmax=310.)
cbar = fig.colorbar(mesh, ax=axs[1])
cbar.set_label(label='Temperature [$^{\circ}$K]',  size=32)
axs[1].set_ylabel('Height [m]', fontsize=32)
axs[1].set_ylim(200,1500)
#axs[1].set_xlim(sliced_ds.time.values[0], sliced_ds.time.values[-1])



axs[2].spines["top"].set_visible(False)
axs[2].spines["right"].set_visible(False)
axs[2].get_xaxis().tick_bottom()
axs[2].get_yaxis().tick_left()
axs[2].spines["bottom"].set_linewidth(2)
axs[2].spines["left"].set_linewidth(2)
axs[2].tick_params(which='minor', length=7, width=3)
axs[2].tick_params(which='major', length=7, width=3)
axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%d %H')) 
axs[2].plot(sst_flag.time.values, sst_flag.sst_tsg.values, color='black', linewidth=4)
axs[2].scatter(timetsg_90s, tsg_90s, color='red', label='> 90$^{th}$ percentile', marker="o", s=100)
axs[2].scatter(timetsg_10s, tsg_10s, color='blue', label='<=10$^{th}$ percentile', marker="o", s=100)


axs[2].set_ylabel('SST TSG [$^{\circ}$C]', fontsize=32)
axs[2].set_ylim(26.,28.)
axs[2].set_xlabel('Time UTC [dd hh]', fontsize=32)
axs[2].legend(frameon=False, loc='upper left',fontsize=32)
fig.savefig(path_fig+string_out+'_quicklooks_arthus.png', format='png')




#%%%
# interpolating sst data at 1 s resolution to the 10 s res of the wind lidar
sst_arthus_interp = sst_flag.interp(time=arthus_data['time'].values)

# merging the interpolated dataset and the wind lidar dataset
data_arthus_merged = xr.merge([arthus_data, sst_arthus_interp])


LHF = data_arthus_merged['Latent_heat_flux'].values
flag = data_surf_mean['flag_tsg'].values

i_cold = np.where(flag == 1)
i_warm = np.where(flag == 2)

LH_cold = LHF[i_cold, :]
LH_warm = LHF[i_warm, :]

flag = data_surf['flag_tsg'].values

i_cold = np.where(flag == 1)
i_warm = np.where(flag == 2)

LH_cold_surf = LHF[i_cold[0], 0:6]
LH_warm_surf = LHF[i_warm[0], 0:6]

LH_cold_subcloud = LHF[i_cold[0], 7:27]
LH_warm_subcloud = LHF[i_warm[0], 7:27]

LH_cold_subcloud[LH_cold_subcloud < -700] = np.nan
LH_cold_subcloud[LH_cold_subcloud > 700] = np.nan
LH_warm_subcloud[LH_warm_subcloud < -700] = np.nan
LH_warm_subcloud[LH_warm_subcloud > 700] = np.nan

LH_cold_surf[LH_cold_surf < -700] = np.nan
LH_cold_surf[LH_cold_surf > 700] = np.nan
LH_warm_surf[LH_warm_surf < -700] = np.nan
LH_warm_surf[LH_warm_surf > 700] = np.nan


LH_cold[LH_cold < -700] = np.nan
LH_cold[LH_cold > 700] = np.nan
LH_warm[LH_warm < -700] = np.nan
LH_warm[LH_warm > 700] = np.nan

# %%

# plotting histogram of vertical wind averaged in the first 300 m
labelsizeaxes    = 32
fontSizeTitle    = 32
fontSizeX        = 32
fontSizeY        = 32
cbarAspect       = 10
fontSizeCbar     = 32
fig, axs = plt.subplots(1, 3, figsize=(24,14), constrained_layout=True)

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
axs[0].hist(LH_warm.flatten(), bins=30, color='red', label='LHF on warm sst percentile', histtype='step', lw=3)
axs[0].hist(LH_cold.flatten(), bins=30, color='blue', label='LHF on cold sst percentile', histtype='step', lw=3)
axs[0].legend(frameon=False, fontsize=24)
axs[0].set_xlim([-500.,500.])
axs[0].set_title('LHF', fontsize=32, loc='left')
axs[0].set_xlabel("LHF [Wm$^{-2}$]", fontsize=fontSizeX)
axs[0].set_ylabel("occurrences [#]", fontsize=fontSizeY)



#axs[0].plot(time_mrr, profile_mrr, color='white')
axs[1].spines["top"].set_visible(False)
axs[1].spines["right"].set_visible(False)
axs[1].get_xaxis().tick_bottom()
axs[1].get_yaxis().tick_left()
axs[1].spines["bottom"].set_linewidth(2)
axs[1].spines["left"].set_linewidth(2)
axs[1].tick_params(which='minor', length=7, width=3)
axs[1].tick_params(which='major', length=7, width=3)
axs[1].hist(LH_warm_surf.flatten(), bins=30, color='red', label='LHF on warm sst percentile', histtype='step', lw=3)
axs[1].hist(LH_cold_surf.flatten(), bins=30, color='blue', label='LHF on cold sst percentile', histtype='step', lw=3)
axs[1].legend(frameon=False, fontsize=24)
axs[1].set_xlim([0.,250.])
axs[1].set_title('LHF between 0 and 300m', fontsize=32, loc='left')
axs[1].set_xlabel("LHF [Wm$^{-2}$]", fontsize=fontSizeX)
axs[1].set_ylabel("occurrences [#]", fontsize=fontSizeY)

#axs[0].plot(time_mrr, profile_mrr, color='white')
axs[2].spines["top"].set_visible(False)
axs[2].spines["right"].set_visible(False)
axs[2].get_xaxis().tick_bottom()
axs[2].get_yaxis().tick_left()
axs[2].spines["bottom"].set_linewidth(2)
axs[2].spines["left"].set_linewidth(2)
axs[2].tick_params(which='minor', length=7, width=3)
axs[2].tick_params(which='major', length=7, width=3)
axs[2].hist(LH_warm_subcloud.flatten(), bins=60, color='red', label='LHF on warm sst percentile', histtype='step', lw=3)
axs[2].hist(LH_cold_subcloud.flatten(), bins=60, color='blue', label='LHF on cold sst percentile', histtype='step', lw=3)
axs[2].legend(frameon=False, fontsize=24)
axs[2].set_xlim([-750.,750.])
axs[2].set_title('LHF above 300m', fontsize=32, loc='left')
axs[2].set_xlabel("LHF [Wm$^{-2}$]", fontsize=fontSizeX)
axs[2].set_ylabel("occurrences [#]", fontsize=fontSizeY)


fig.tight_layout()
fig.savefig(path_fig+string_out+'_LHF_histogram.png', format='png')



# %%

# reading horizontal wind and speed data
path_Doppler_lidar = '/Volumes/Extreme SSD/work/006_projects/001_Prec_Trade_Cycle/doppler_lidar/case_1/'
file_1_speed = '20200129_DL_Wind_Speed.nc'
speed_data_1 = xr.open_dataset(path_Doppler_lidar+file_1_speed)
file_2_speed = '20200130_DL_Wind_Speed.nc'
speed_data_2 = xr.open_dataset(path_Doppler_lidar+file_2_speed)



file_1_dir = '20200129_DL_Wind_Direction.nc'
dir_data_1 = xr.open_dataset(path_Doppler_lidar+file_1_dir)

file_2_dir = '20200130_DL_Wind_Direction.nc'
dir_data_2 = xr.open_dataset(path_Doppler_lidar+file_2_dir)

speed_data = xr.merge([speed_data_1, speed_data_2])
dir_data = xr.merge([dir_data_1, dir_data_2])


# %%

# restricting data to the lowest 1500 m
speed_sel = speed_data.sel(Height=slice(0., 1500.))
dir_sel = dir_data.sel(Height=slice(0., 1500.))


# defining flag for data points
flag = sst_flag.flag_tsg.values
sst = sst_flag.sst_tsg.values
time_sst = sst_flag.time.values
tsg_10s = sst[np.where(flag == 1)[0]]
timetsg_10s = time_sst[np.where(flag == 1)[0]]
tsg_90s = sst[np.where(flag == 2)[0]]
timetsg_90s = time_sst[np.where(flag == 2)[0]]

#speed =  speed_data.Horizontal_Wind_Speed.values[:-1,:]
#direction = dir_data.Horizontal_Wind_Direction.values[:-1]

# plot of the Doppler lidar data for the case study
fig, axs = plt.subplots(3, 1, figsize=(24,16), constrained_layout=True)
axs[0].spines["top"].set_visible(False)
axs[0].spines["right"].set_visible(False)
axs[0].get_xaxis().tick_bottom()
axs[0].get_yaxis().tick_left()
axs[0].spines["bottom"].set_linewidth(2)
axs[0].spines["left"].set_linewidth(2)
axs[0].tick_params(which='minor', length=7, width=3)
axs[0].tick_params(which='major', length=7, width=3)
axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%d %H')) 
mesh = axs[0].pcolormesh(speed_data.Time.values, speed_data.Height.values, speed_data.Horizontal_Wind_Speed.values.T, cmap='inferno', vmin=0., vmax=18.)
cbar = fig.colorbar(mesh, ax=axs[0])
cbar.set_label(label='Wind speed [ms$^{-1}$]',  size=32)
axs[0].set_ylabel('Height [m]', fontsize=32)
axs[0].set_ylim(50,1500)
#axs[0].set_xlim(sst_flag.time.values[0], sst_flag.time.values[-1])


axs[1].spines["top"].set_visible(False)
axs[1].spines["right"].set_visible(False)
axs[1].get_xaxis().tick_bottom()
axs[1].get_yaxis().tick_left()
axs[1].spines["bottom"].set_linewidth(2)
axs[1].spines["left"].set_linewidth(2)
axs[1].tick_params(which='minor', length=7, width=3)
axs[1].tick_params(which='major', length=7, width=3)
axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%d %H')) 
mesh = axs[1].pcolormesh(dir_data.Time.values, dir_data.Height.values,  dir_data.Horizontal_Wind_Direction.values.T, cmap='viridis', vmin=0., vmax=200.)
cbar = fig.colorbar(mesh, ax=axs[1])
cbar.set_label(label='Horizontal wind direction [$^{\circ}$]',  size=32)
axs[1].set_ylabel('Height [m]', fontsize=32)
axs[1].set_ylim(50,1500)
#axs[1].set_xlim(sst_flag.time.values[0], sst_flag.time.values[-1])



axs[2].spines["top"].set_visible(False)
axs[2].spines["right"].set_visible(False)
axs[2].get_xaxis().tick_bottom()
axs[2].get_yaxis().tick_left()
axs[2].spines["bottom"].set_linewidth(2)
axs[2].spines["left"].set_linewidth(2)
axs[2].tick_params(which='minor', length=7, width=3)
axs[2].tick_params(which='major', length=7, width=3)
axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%d %H')) 
axs[2].plot(sst_flag.time.values, sst_flag.sst_tsg.values, color='black', linewidth=4)
axs[2].scatter(timetsg_90s, tsg_90s, color='red', label='> 90$^{th}$ percentile', marker="o", s=100)
axs[2].scatter(timetsg_10s, tsg_10s, color='blue', label='<=10$^{th}$ percentile', marker="o", s=100)


axs[2].set_ylabel('SST TSG [$^{\circ}$C]', fontsize=32)
axs[2].set_ylim(26.,28.5)
axs[2].set_xlabel('Time UTC [dd hh]', fontsize=32)
axs[2].legend(frameon=False, loc='upper left',fontsize=32)
fig.savefig(path_fig+string_out+'_quicklooks_Doppler_lidar.png', format='png')

# %%
