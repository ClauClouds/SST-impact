#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 17:19:28 2021
@ goal; code to check for other case studies for merian ship track
lat/lon coordinates of possible new cases: Merian: (52.5W;7.5N); (57W;13.5N)
@author: claudia
"""


# importing necessary libraries
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib
import netCDF4 as nc4
import numpy as np
import xarray as xr
from datetime import datetime
import glob
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


#%%
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

# reading radar file list
file_list = np.sort(glob.glob("/Volumes/Extreme SSD/ship_motion_correction_merian/corrected_data/eurec4a_intake/*.nc"))

processing_mode = 'case_study_2'

# opening ship data and reading sst
dataset = xr.open_dataset(ship_data+ship_file)

# setting time window to be checked
if processing_mode == 'case_study_1':
    string_out = '20200128_20200131'
    t_start = datetime(2020, 1, 27, 0, 0, 0)
    t_end = datetime(2020, 1, 31, 23, 59, 59)
    ymin = 27.
    ymax = 28.
    x_max_cb = 2500.
    x_min_cb = 0.
    file_list=file_list[8:13]
elif processing_mode == 'case_study_2':
    string_out = '20200120_20200121'
    t_start = datetime(2020, 1, 21, 10, 0, 0)
    t_end = datetime(2020, 1, 22, 00, 0, 0)
    ymin = 27.
    ymax = 28.
    x_max_cb = 2500.
    x_min_cb = 400.
    file_list=file_list[1:3]
else:
    string_out = '20200208_20200213'
    t_start = datetime(2020, 2, 8, 0, 0, 0)
    t_end = datetime(2020, 2, 13, 23, 59, 59) 
    ymin = 26.
    ymax = 28.
    x_max_cb = 2500.
    x_min_cb = 400.  
    file_list=file_list[20:26]
# slicing dataset for the selected time interval and extracting sst
sliced = dataset.sel(time=slice(t_start, t_end))
sst = sliced['SST'].values
time_sst = sliced['time'].values




print('max sst ', np.nanmax(sst))
print('min sst ', np.nanmin(sst))

# calculating sst distribution and percentiles
percentiles_sst = np.nanpercentile(sst, perc_vals)
print('sst percentiles {}'.format(percentiles_sst))

# selecting values of sst corresponding to percentiles thresholds of min and
# max percentiles to use as thresholds
perc_low = percentiles_sst[0]
perc_high = percentiles_sst[-1]

# finding indeces of sst values smaller than sst_perc_low and sst_perc_high
i_sst_low = np.where(sst <= perc_low)[0]
i_sst_high = np.where(sst >= perc_high)[0]

# generating flag to identify the sst < thr_low ( flag == 1) and sst > trh_high
# flag == 2.
print(len(sst[i_sst_low]))
print(len(time_sst[i_sst_high]))
flag_arr = np.zeros(len(sst))
flag_arr[i_sst_low] = 1
flag_arr[i_sst_high] = 2

 
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
sst_flag      = xr.Dataset(data_vars = variables,
                              coords = coords,
                              attrs = global_attributes)

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

i_cold = np.where(flag == 1)
i_warm = np.where(flag == 2)
w_cold = w_surf[i_cold]
w_warm = w_surf[i_warm]
cb_cold = cb_surf[i_cold]
cb_warm = cb_surf[i_warm]


print(w_cold)
print(w_warm)


# plotting histogram of vertical wind averaged in the first 300 m and cloud base height
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
ax.hist(w_warm, bins=30, color='red', label='w on warm sst percentile', histtype='step', density=True)
ax.hist(w_cold, bins=30, color='blue', label='w on cold sst percentile', histtype='step', density=True)
ax.legend(frameon=False)
ax.set_xlim([-2.,2.])

ax.set_title('w histograms for :'+string_out, fontsize=fontSTitle, loc='left')
ax.set_xlabel("w [$ms^{-1}$]", fontsize=fontSizeX)
ax.set_ylabel("occurrences [#]", fontsize=fontSizeY)

ax = plt.subplot(1, 2, 2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
matplotlib.rc('xtick', labelsize=labelsizeaxes)  # sets dimension of ticks
matplotlib.rc('ytick', labelsize=labelsizeaxes)  # sets dimension of ticks
ax.hist(cb_warm, bins=30, color='red', label='cb height on warm sst percentile', histtype='step', density=True)
ax.hist(cb_cold, bins=30, color='blue', label='cb height on cold sst percentile', histtype='step', density=True)
ax.set_xlim([x_min_cb,x_max_cb])
ax.legend(frameon=False)
ax.set_title('cloud base histograms for :'+string_out, fontsize=fontSTitle, loc='left')
ax.set_xlabel("clooud base height [m]", fontsize=fontSizeX)
ax.set_ylabel("occurrences [#]", fontsize=fontSizeY)
fig.tight_layout()
fig.savefig(path_fig+string_out+'_w_cb_histogram.png', format='png')



# plotting sst histogram and sst flag obtained from sst time serie
labelsizeaxes = 12
fontSTitle = 12
fontSizeX = 12
fontSizeY = 12
cbarAspect = 10
fontSizeCbar = 12
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
rcParams['font.sans-serif'] = ['Tahoma']
matplotlib.rcParams['savefig.dpi'] = 100
plt.gcf().subplots_adjust(bottom=0.15)
fig.tight_layout()
ax = plt.subplot(1, 1, 1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
matplotlib.rc('xtick', labelsize=labelsizeaxes)  # sets dimension of ticks
matplotlib.rc('ytick', labelsize=labelsizeaxes)  # sets dimension of ticks
ax.set_ylim([ymin, ymax])
ax.plot(time_sst,  sst, color='blue')
ax.set_title('time series of sst for: '+string_out, fontsize=fontSTitle, loc='left')
ax.set_xlabel("time ", fontsize=fontSizeX)
ax.set_ylabel("sst [degrees $^{\circ}C$]", fontsize=fontSizeY)
fig.tight_layout()
fig.savefig(path_fig+string_out+'_sst_'+perc_string+'.png', format='png')










# combining all radar data in one dataset
radar_data = xr.open_mfdataset(file_list,
                               concat_dim = 'time',
                               data_vars = 'minimal',
                              )

# selecting data in the time window of the surface anomaly
data_sliced = radar_data.sel(time=slice(t_start, t_end))

# interpolating sst data at 1 s resolution to the 10 s res of the wind lidar
sst_data_interp = sst_flag.interp(time=data_sliced['time'].values)

# merging the interpolated dataset and the wind lidar dataset
data_merged = xr.merge([data_sliced, sst_data_interp])


# calculating data on cold and warm anomaly
data_cold =  data_merged.where(data_merged.flag == 1)
data_warm =  data_merged.where(data_merged.flag == 2)
time_cold = data_cold.time.values
time_warm = data_warm.time.values

Ze_cold = data_cold.radar_reflectivity.values
Ze_warm = data_warm.radar_reflectivity.values
Vd_cold = data_cold.mean_doppler_velocity.values
Vd_warm = data_warm.mean_doppler_velocity.values
Sw_cold = data_cold.spectral_width.values
Sw_warm = data_warm.spectral_width.values
 
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
ax.set_title('Cloud fraction for '+processing_mode, fontsize=fontSizeTitle, loc='left')
ax.set_ylabel("Height [m]", fontsize=fontSizeY)
ax.set_xlabel("Hydrometeor fraction []", fontsize=fontSizeY)

# Turn on the frame for the twin axis, but then hide all
# but the bottom spine
fig.tight_layout()
fig.savefig(path_fig+'__cloud_fractions+'+string_out+'.png', bbox_inches='tight')



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
    