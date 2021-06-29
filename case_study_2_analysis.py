#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 18:17:24 2021
read case study 28/29 Jan
@author: claudia
"""

import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib
import netCDF4 as nc4
import numpy as np
import xarray as xr
from datetime import datetime
import matplotlib.dates as mdates

# set percentiles values to use
perc_vals = [10, 90] #  'all_campaign' #
perc_string = str(perc_vals)[1:3]+'_'+str(perc_vals)[5:7]


# paths and filenames
ship_data = "/Volumes/Extreme SSD/ship_motion_correction_merian/ship_data/new/"
ship_file = "ship_dataset_allvariables.nc"
path_fig = "/Volumes/Extreme SSD/work/006_projects/001_Prec_Trade_Cycle/\
SST_impact_work/plots/"
path_out = "/Volumes/Extreme SSD/work/006_projects/001_Prec_Trade_Cycle/\
SST_impact_work/"

string_out = '20200129'
t_start = datetime(2020, 1, 29, 0, 0, 0)
t_end = datetime(2020, 1, 29, 23, 59, 59)

# opening ship data and reading sst
dataset = xr.open_dataset(ship_data+ship_file)

# slicing dataset for the selected time interval and extracting sst
sliced_ds = dataset.sel(time=slice(t_start, t_end))
sst = sliced_ds['SST'].values
time_sst = sliced_ds['time'].values


# ccalculating 10th percentile
perc_10th = np.percentile(sst, 10.)
perc_90th = np.percentile(sst, 90.)


# finding indeces of sst values smaller than sst_perc_low and sst_perc_high
i_sst_low = np.where(sst <= perc_10th)[0]
i_sst_high = np.where(sst >= perc_90th)[0]

print(len(i_sst_low), len(i_sst_high))
#%%
flag_arr = np.zeros(len(sst))
flag_arr[i_sst_low] = 1
flag_arr[i_sst_high] = 2


# plotting sst histogram and sst flag obtained from sst time serie
labelsizeaxes = 25
fontSTitle = 25
fontSizeX = 25
fontSizeY = 25
cbarAspect = 25
fontSizeCbar = 25
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 8))
rcParams['font.sans-serif'] = ['Tahoma']
matplotlib.rcParams['savefig.dpi'] = 100
plt.gcf().subplots_adjust(bottom=0.15)
fig.tight_layout()
ax = plt.subplot(1, 1, 1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.spines["bottom"].set_linewidth(2)
ax.spines["left"].set_linewidth(2)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %H'))

ax.tick_params(which='minor', length=7, width=3)
ax.tick_params(which='major', length=7, width=3)
matplotlib.rc('xtick', labelsize=labelsizeaxes)  # sets dimension of ticks
matplotlib.rc('ytick', labelsize=labelsizeaxes)  # sets dimension of ticks
ax.scatter(time_sst, flag_arr, color='red', marker = 'o')
ax.set_ylim([0., 2.])
ax2 = ax.twinx()
ax2.spines["top"].set_visible(False)
#ax2.spines["right"].set_visible(False)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d %H'))

ax2.set_ylim([26., 28.])
ax2.set_ylabel('SST [$^{\circ}$C]', fontsize=fontSizeX)
ax2.plot(time_sst,  sst, color='blue')
ax2.axhline(perc_10th, 0., 1, color='black', linestyle=':', label="10th percentile")
ax2.axhline(perc_90th, 0., 1, color='green', linestyle=':', label="90th percentile")
ax.set_title('time series of sst and flag for: '+string_out, fontsize=fontSTitle, loc='left')
ax.set_xlabel("time [dd hh]", fontsize=fontSizeX)
ax.set_ylabel("flag []", fontsize=fontSizeY)
ax2.legend(frameon=False, fontsize=fontSizeX)
fig.tight_layout()
fig.savefig(path_fig+string_out+'_sst_flag_'+perc_string+'_perc.png', format='png')



# saving flag, sst, and time array of the slice in ncdf
dims = ['time']
coords = {"time": time_sst}
sst_array = xr.DataArray(dims=dims, coords=coords, data=sst,
attrs = {'long_name':'sea surface temperature at -3 m from sea surface',
'units': 'Degrees Celsius'})
flag_array = xr.DataArray(dims=dims, coords=coords, data=flag_arr,
 attrs={'long_name':'flag indicating sst range: == 1 sst < sst perc low , ==2 for\
 sst > sst perc high', 'units': 'Pa'})
variables         = {'sst':sst_array,
                     'flag':flag_array}
global_attributes = {'created_by':'Claudia Acquistapace',
                     'created_on':str(datetime.now()),
                     'comment':'sst flag '}
sst_flag_dataset      = xr.Dataset(data_vars = variables,
                              coords = coords,
                              attrs = global_attributes)
sst_flag_dataset.to_netcdf(path_out+string_out+'_sst_flag'+perc_string+'_perc.nc')

#%%
        
sst_flag = xr.open_dataset(path_out+string_out+'_sst_flag'+perc_string+'_perc.nc')

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
#axs[1].set_xlim([x_min_cb,x_max_cb])
axs[1].legend(frameon=False, fontsize=24)
axs[1].set_title('cloud base histograms for :'+string_out, fontsize=32, loc='left')
axs[1].set_xlabel("clooud base height [m]", fontsize=fontSizeX)
axs[1].set_ylabel("occurrences [#]", fontsize=fontSizeY)

fig.tight_layout()
fig.savefig(path_fig+string_out+'_w_cb_histogram.png', format='png')



#%%

# reading arthus data
path_arthus = '/Volumes/Extreme SSD/work/006_projects/001_Prec_Trade_Cycle/data_arthus/case_2/'
arthus_file = path_arthus+"/LHF.cdf"

# removing time duplicates in the two datasets
arthus_1 = xr.open_dataset(arthus_file)
_, index = np.unique(arthus_1['Time'], return_index=True)
arthus_data = arthus_1.isel(Time=index)



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
axs[1].set_xlim([-80.,250.])
axs[1].legend(frameon=False, fontsize=24)
axs[1].set_title('LHF between 300 and 700m', fontsize=32, loc='left')
axs[1].set_xlabel("LHF [Wm$^{-2}$]", fontsize=fontSizeX)
axs[1].set_ylabel("occurrences [#]", fontsize=fontSizeY)

fig.tight_layout()
fig.savefig(path_fig+string_out+'_LHF_histogram.png', format='png')


#%%

# reading water vapor mixing ratio data from Diego
path_arthus = '/Volumes/Extreme SSD/work/006_projects/001_Prec_Trade_Cycle/data_arthus/case_2/'
arthus_wvmr_file = path_arthus+"/20200129_000009_WVMR_gr_10s_50m.cdf"

# removing time duplicates in the two datasets
arthus_wvmr_1 = xr.open_dataset(arthus_wvmr_file)

# removing double time stamps in time arrays
_, index = np.unique(arthus_wvmr_1['Time'], return_index=True)
arthus_data_wvmr = arthus_wvmr_1.isel(Time=index)

# resampling arthus data on time array of SST
arthus_sst = arthus_data_wvmr.reindex({'Time':sst_flag.time.values}, method=None)

# merging the sst dataset and the arthus dataset
data_arthus_merged_wvmr = xr.merge([arthus_sst, sst_flag])
wvmr = data_arthus_merged_wvmr.Water_vapor_mixing_ratio.values


# selecting cold and warm time stamps
i_cold = np.where(data_arthus_merged_wvmr.flag.values == 1)
i_warm = np.where(data_arthus_merged_wvmr.flag.values == 2)

wvmr_cold = data_arthus_merged_wvmr.Water_vapor_mixing_ratio.values[i_cold,:]
wvmr_warm = data_arthus_merged_wvmr.Water_vapor_mixing_ratio.values[i_warm,:]
#%%
fig, axs = plt.subplots(2, 1, figsize=(16,24), sharex=True, constrained_layout=True)
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
sst = data_arthus_merged.sst.values
time = data_arthus_merged.Time.values
#axs[2].plot(data_arthus_T_merged.Time.values, data_arthus_T_merged.flag.values)
sst_cold = sst[i_cold]
time_cold = time[i_cold]
sst_warm = sst[i_warm]
time_warm = time[i_warm]
axs[1].plot(data_arthus_merged.Time.values, data_arthus_merged.sst.values)

axs[1].scatter(time_cold, sst_cold, color='blue', marker="o", s=100, label='cold patch')
axs[1].scatter(time_warm, sst_warm, color='red', marker="o", s=100, label='warm patch')
axs[1].set_xlabel('Time UTC [dd hh]', fontsize=32)
axs[1].set_ylabel('SST [$^{\circ}$C]', fontsize=32)
axs[1].legend(frameon=False, fontsize=32)
fig.savefig(path_fig+string_out+'_WVMR_FLAG.png', format='png')


#%%
# radar data analysis

# reading radar file list
file_name = "/Volumes/Extreme SSD/ship_motion_correction_merian/corrected_data/daily_files_intake/daily_files/20200129_wband_radar_msm_eurec4a_intake.nc"


# set the processing mode keyword for the data you want to be processed
processing_mode = 'case_study' # 'all_campaign' #

# setting time window to be checked
x_max_cb = 2000.
x_min_cb = 400.


radar_data = xr.open_dataset(file_name)


# selecting data in the time window of the surface anomaly
data_sliced = radar_data.sel(time=slice(t_start, t_end))


# interpolating sst data at 1 s resolution to the 10 s res of the wind lidar
sst_data_interp = sst_flag.interp(time=data_sliced['time'].values)

# merging the interpolated dataset and the wind lidar dataset
data_merged = xr.merge([data_sliced, sst_data_interp])
data_merged.radar_reflectivity.plot(x='time', y='height', cmap="seismic", vmin=-60., vmax=20.)


# assigning flag as a coordinate
data_merged = data_merged.swap_dims({"time":"flag"})

# selecting data for cold and warm patches
data_cold = data_merged.sel({'flag':1})
data_cold.radar_reflectivity.plot(x='time', y='height', cmap="seismic", vmin=-60., vmax=20.)



data_warm = data_merged.sel({'flag':2})
data_warm.radar_reflectivity.plot(x='time', y='height', cmap="seismic", vmin=-60., vmax=20.)

Ze_cold = data_cold.radar_reflectivity.values
Ze_warm = data_warm.radar_reflectivity.values
Vd_cold = data_cold.mean_doppler_velocity.values
Vd_warm = data_warm.mean_doppler_velocity.values
Sw_cold = data_cold.spectral_width.values
Sw_warm = data_warm.spectral_width.values


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
    labelsizeaxes = 32
    fontSizeTitle = 32
    fontSizeX     = 32
    fontSizeY     = 32
    cbarAspect    = 32
    fontSizeCbar  = 32
    
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
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    ax.tick_params(which='minor', length=7, width=3)
    ax.tick_params(which='major', length=7, width=3)
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

# calculating cloud fraction for cold and warm anomaly
cloud_fraction_cold = f_calc_tot_cloud_fraction(Ze_cold)
cloud_fraction_warm = f_calc_tot_cloud_fraction(Ze_warm)

#%%
hmin         = 100.
hmax         = 2200.
labelsizeaxes = 32
fontSizeTitle = 32
fontSizeX     = 32
fontSizeY     = 32
cbarAspect    = 32
fontSizeCbar  = 32
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
ax.spines["bottom"].set_linewidth(2)
ax.spines["left"].set_linewidth(2)
ax.tick_params(which='minor', length=7, width=3)
ax.tick_params(which='major', length=7, width=3)
matplotlib.rc('xtick', labelsize=labelsizeaxes)  # sets dimension of ticks in the plots
matplotlib.rc('ytick', labelsize=labelsizeaxes)  # sets dimension of ticks in the plots
ax.plot(cloud_fraction_cold, data_merged.height.values, label='cold SST', color='b', linewidth=4.0)
ax.plot(cloud_fraction_warm, data_merged.height.values, label='warm SST', color='r', linewidth=4.0)
ax.legend(frameon=False, fontsize=fontSizeY)
ax.set_ylim(100.,hmax)                                               # limits of the y-axesn  cmap=plt.cm.get_cmap("viridis", 256)
#ax.set_xlim(xmin, xmax)                                 # limits of the x-axes
ax.set_title('Cloud fraction :'+string_out, fontsize=fontSizeTitle, loc='left')
ax.set_ylabel("Height [m]", fontsize=fontSizeY)
ax.set_xlabel("Hydrometeor fraction []", fontsize=fontSizeY)

# Turn on the frame for the twin axis, but then hide all
# but the bottom spine
fig.tight_layout()
fig.savefig(path_fig+string_out+'_cloud_fraction.png', bbox_inches='tight')

#%%


time_cold = data_cold.time.values
time_warm = data_warm.time.values
# plot CFAD for all variables for cold and warm anomaly.
var_plot = ['ze_cold', 'Ze_warm', 'Vd_cold', 'Vd_warm', 'Sw_cold', 'Sw_warm']

for ivar, var in enumerate(var_plot):
    
    print('plotting CFAD for ', var)
    if var == 'ze_cold':
        dict_input = {'xvar':Ze_cold.flatten(), 
                      'yvar':(np.ones((len(time_cold),1))*np.array([data_merged.height.values])).flatten(), 
                      'xlabel':'Ze [dBz]',
                      'title':'Reflectivity over cold SST anomaly',
                      'varname':'Ze_cold',
                      'info':'cold_'+string_out, 
                      'path':path_fig,
                      'bins':[100,50],
                      'cmap':'viridis',
                      'xmax':20., 
                      'xmin':-50.,  
                      'dataset':processing_mode,
                      }
    if var == 'Ze_warm':
        dict_input = {'xvar':Ze_warm.flatten(), 
                      'yvar':(np.ones((len(time_warm),1))*np.array([data_merged.height.values])).flatten(), 
                      'xlabel':'Ze [dBz]',
                      'title':'Reflectivity over warm SST anomaly',
                      'varname':'Ze_warm',
                      'info':'warm_'+string_out ,
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
                      'info':'cold_'+string_out,
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
                      'info':'warm_'+string_out,
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
                      'info':'cold_'+string_out,
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
                      'info':'warm_'+string_out ,
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
    
