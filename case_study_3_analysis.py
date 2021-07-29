#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 19:19:11 2021
code for case study 3 20200123
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

string_out = '20200123'
t_start = datetime(2020, 1, 23, 0, 0, 0)
t_end = datetime(2020, 1, 23, 23, 59, 59)


# opening ship data and reading sst
dataset = xr.open_dataset(path_out+string_out+'_sst_flag10_90_perc.nc')


# slicing dataset for the selected time interval and extracting sst
sst = dataset['sst_tsg'].values
time_sst = dataset['time'].values
flag_arr = dataset['flag_tsg'].values


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
sst_data_interp = dataset.interp(time=wind_lidar_slice['time'].values)

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
fig.savefig(path_fig+string_out+'_w_cb_histogram_tsg.png', format='png')



#%%

# radar data analysis

# reading radar file list
path_radar_data = '/Volumes/Extreme SSD/ship_motion_correction_merian/corrected_data/wband_daily_with_DOI/corrected_comments/'
file_name = path_radar_data+"20200123_wband_radar_msm_eurec4a_intake.nc"


# set the processing mode keyword for the data you want to be processed
processing_mode = 'case_study' # 'all_campaign' #

# setting time window to be checked
x_max_cb = 2000.
x_min_cb = 400.


radar_data = xr.open_dataset(file_name)

# interpolating sst data at 1 s resolution to the 10 s res of the wind lidar
sst_data_interp = dataset.interp(time=radar_data['time'].values)


# merging the interpolated dataset and the wind lidar dataset
data_merged = xr.merge([radar_data, sst_data_interp])
data_merged.radar_reflectivity.plot(x='time', y='height', cmap="seismic", vmin=-60., vmax=20.)


# assigning flag as a coordinate
data_merged = data_merged.swap_dims({"time":"flag_tsg"})

#%%
# selecting data for cold and warm patches
data_cold = data_merged.sel({'flag_tsg':1})
data_cold.radar_reflectivity.plot(x='time', y='height', cmap="seismic", vmin=-60., vmax=20.)
#%%


data_warm = data_merged.sel({'flag_tsg':2})
data_warm.radar_reflectivity.plot(x='time', y='height', cmap="seismic", vmin=-60., vmax=20.)
#%%
Ze_cold = data_cold.radar_reflectivity.values
Ze_warm = data_warm.radar_reflectivity.values
Vd_cold = data_cold.mean_doppler_velocity.values
Vd_warm = data_warm.mean_doppler_velocity.values
Sw_cold = data_cold.spectral_width.values
Sw_warm = data_warm.spectral_width.values

#%%


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
    


#%%


# reading horizontal wind and speed data
path_Doppler_lidar = '/Volumes/Extreme SSD/work/006_projects/001_Prec_Trade_Cycle/doppler_lidar/case_3/'
file_1_speed = '20200123_DL_Wind_Speed.nc'
speed_data = xr.open_dataset(path_Doppler_lidar+file_1_speed)

file_1_dir = '20200123_DL_Wind_Direction.nc'
dir_data = xr.open_dataset(path_Doppler_lidar+file_1_dir)


# restricting data to the lowest 1500 m
speed_sel = speed_data.sel(Height=slice(0., 1500.))
dir_sel = dir_data.sel(Height=slice(0., 1500.))


# defining flag for data points
flag = dataset.flag_tsg.values
sst = dataset.sst_tsg.values
time_sst = dataset.time.values
tsg_10s = sst[np.where(flag == 1)[0]]
timetsg_10s = time_sst[np.where(flag == 1)[0]]
tsg_90s = sst[np.where(flag == 2)[0]]
timetsg_90s = time_sst[np.where(flag == 2)[0]]

speed =  speed_data.Horizontal_Wind_Speed.values[:-1,:]
direction = dir_data.Horizontal_Wind_Direction.values[:-1]

# plot of the Doppler lidar data for the case study
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
mesh = axs[0].pcolormesh(speed_data.Time.values[:-1], speed_data.Height.values, speed.T, cmap='inferno', vmin=0., vmax=5.)
cbar = fig.colorbar(mesh, ax=axs[0])
cbar.set_label(label='Wind speed [ms$^{-1}$]',  size=32)
axs[0].set_ylabel('Height [m]', fontsize=32)
axs[0].set_ylim(50,1500)
axs[0].set_xlim(dataset.time.values[0], dataset.time.values[-1])


axs[1].spines["top"].set_visible(False)
axs[1].spines["right"].set_visible(False)
axs[1].get_xaxis().tick_bottom()
axs[1].get_yaxis().tick_left()
axs[1].spines["bottom"].set_linewidth(2)
axs[1].spines["left"].set_linewidth(2)
axs[1].tick_params(which='minor', length=7, width=3)
axs[1].tick_params(which='major', length=7, width=3)
axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%d %H')) 
mesh = axs[1].pcolormesh(dir_data.Time.values[:-1], dir_data.Height.values, direction.T, cmap='viridis', vmin=0., vmax=360.)
cbar = fig.colorbar(mesh, ax=axs[1])
cbar.set_label(label='Horizontal wind direction [$^{\circ}$]',  size=32)
axs[1].set_ylabel('Height [m]', fontsize=32)
axs[1].set_ylim(50,1500)
axs[1].set_xlim(dataset.time.values[0], dataset.time.values[-1])



axs[2].spines["top"].set_visible(False)
axs[2].spines["right"].set_visible(False)
axs[2].get_xaxis().tick_bottom()
axs[2].get_yaxis().tick_left()
axs[2].spines["bottom"].set_linewidth(2)
axs[2].spines["left"].set_linewidth(2)
axs[2].tick_params(which='minor', length=7, width=3)
axs[2].tick_params(which='major', length=7, width=3)
axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%d %H')) 
axs[2].plot(dataset.time.values, dataset.sst_tsg.values, color='black', linewidth=4)
axs[2].scatter(timetsg_90s, tsg_90s, color='red', label='> 90$^{th}$ percentile', marker="o", s=100)
axs[2].scatter(timetsg_10s, tsg_10s, color='blue', label='<=10$^{th}$ percentile', marker="o", s=100)


axs[2].set_ylabel('SST TSG [$^{\circ}$C]', fontsize=32)
axs[2].set_ylim(27.,28.5)
axs[2].set_xlabel('Time UTC [dd hh]', fontsize=32)
axs[2].legend(frameon=False, loc='upper left',fontsize=32)
fig.savefig(path_fig+string_out+'_quicklooks_Doppler_lidar.png', format='png')


#%%



# interpolating sst data at 1 s resolution to the 10 s res of the wind lidar
sst_data_interp = sliced_ds.interp(time=speed_data['Time'].values, method='nearest')

# merging the interpolated dataset and the wind lidar dataset
speed_merged = xr.merge([speed_sel, sst_data_interp])
dir_merged = xr.merge([dir_sel, sst_data_interp])

#%%
# selecting cold and warm dataset indeces
flag = speed_merged['flag_tsg'].values
i_cold = np.where(flag == 1)
i_warm = np.where(flag == 2)

# reading speed for cold, and warm
speed = speed_merged['Horizontal_Wind_Speed'].values
speed_cold = speed[i_cold, :]
speed_warm = speed[i_warm, :]

dir_wind = dir_merged.Horizontal_Wind_Direction.values
dir_cold = dir_wind[i_cold, :]
dir_warm = dir_wind[i_warm, :]

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
axs[0].hist(speed_cold.flatten(), bins=30, color='red', label='w on warm sst percentile', histtype='step', lw=3)
axs[0].hist(speed_warm.flatten(), bins=30, color='blue', label='w on cold sst percentile', histtype='step', lw=3)
axs[0].legend(frameon=False, fontsize=24)
axs[0].set_xlim([0., 15.])
axs[0].set_title(' Wind speed histograms for :'+string_out, fontsize=32, loc='left')
axs[0].set_xlabel("Horizontal Wind Speed [ms$^{-1}$]", fontsize=fontSizeX)
axs[0].set_ylabel("occurrences [#]", fontsize=fontSizeY)


axs[1].spines["top"].set_visible(False)
axs[1].spines["right"].set_visible(False)
axs[1].get_xaxis().tick_bottom()
axs[1].get_yaxis().tick_left()
axs[1].spines["bottom"].set_linewidth(2)
axs[1].spines["left"].set_linewidth(2)
axs[1].tick_params(which='minor', length=7, width=3)
axs[1].tick_params(which='major', length=7, width=3)
axs[1].hist(dir_cold.flatten(), bins=30, color='red', label='cb height on warm sst percentile', histtype='step', lw=3)
axs[1].hist(dir_warm.flatten(), bins=30, color='blue', label='cb height on cold sst percentile', histtype='step', lw=3)
#axs[1].set_xlim([x_min_cb,x_max_cb])
axs[1].legend(frameon=False, fontsize=24)
axs[1].set_title('cloud base histograms for :'+string_out, fontsize=32, loc='left')
axs[1].set_xlabel('Horizontal wind direction [$^{\circ}$]', fontsize=fontSizeX)
axs[1].set_ylabel("occurrences [#]", fontsize=fontSizeY)

fig.tight_layout()
fig.savefig(path_fig+string_out+'_speed_dir_wind_histogram_tsg.png', format='png')
