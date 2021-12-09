#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 10:42:54 2020
@ date; 14 april 2021
@author: Claudia Acquistapace
@goal: this script has the goal of deriving sst distribution for a given ship
track and calculate its percentiles. It then assigns a flag for SST
values below or above a given threshold percentiles.
"""

# importing necessary libraries
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib
import netCDF4 as nc4
import numpy as np
import xarray as xr
from datetime import datetime
import matplotlib.dates as mdates
import pandas as pd

def f_closest(array,value):
    '''
    # closest function
    #---------------------------------------------------------------------------------
    # date :  16.10.2017
    # author: Claudia Acquistapace
    # goal: return the index of the element of the input array that in closest to the value provided to the function
    '''
    import numpy as np
    idx = (np.abs(array-value)).argmin()
    return idx  

# set the processing mode keyword for the data you want to be processed
processing_mode = 'case_study_1' #  'all_campaign' #

# set percentiles values to use
perc_vals = [10, 90] 
perc_string = str(perc_vals)[1:3]+'_'+str(perc_vals)[5:7]


# paths and filenames
ship_data = "/Volumes/Extreme SSD/ship_motion_correction_merian/ship_data/new/"
ship_file = "ship_dataset_allvariables.nc"
path_fig = "/Volumes/Extreme SSD/work/006_projects/001_Prec_Trade_Cycle/\
SST_impact_work/plots/"
path_out = "/Volumes/Extreme SSD/work/006_projects/001_Prec_Trade_Cycle/\
SST_impact_work/"

tsg_file = "/Volumes/Extreme SSD/work/006_projects/001_Prec_Trade_Cycle/tsg_sst_data/tsg/nc/msm_089_1_tsg.nc"

#%%
        
    
# opening ship data and reading sst
dataset = xr.open_dataset(ship_data+ship_file)
tsg_data = xr.open_dataset(tsg_file)
string_out = '20200202_20200203'
t_start = datetime(2020, 2, 2, 0, 0, 0)
t_end = datetime(2020, 2, 3, 23, 59, 59)

    
print(t_start, t_end)
    
# slicing dataset for the selected time interval and extracting sst
sliced_ds = dataset.sel(time=slice(t_start, t_end))
sst = sliced_ds['SST'].values
time_sst = pd.to_datetime(sliced_ds['time'].values)

# slicing tsg datase t for the selected time interval and extracting sst
sliced_tsg_ds = tsg_data.sel(TIME=slice(t_start, t_end))
tsg_sst = sliced_tsg_ds['TEMP'].values
tsg_time_sst = sliced_tsg_ds['TIME'].values
tsg_flag = sliced_tsg_ds['TEMP_QC'].values

# averaging together the sst of the different sst sensors for tsg
temp0 = sliced_tsg_ds.TEMP[:,0].values
temp1 = sliced_tsg_ds.TEMP[:,1].values
temp_merqctsg = temp0
temp_merqctsg[np.isnan(temp0)] = temp1[np.isnan(temp0)]


# ccalculating 10th percentile for ship sst
perc_10th = np.percentile(sst, 10.)
perc_90th = np.percentile(sst, 90.)

# ccalculating 10th percentile for tsg sst
tsg_perc_10th = np.nanpercentile(temp_merqctsg, 10.)
tsg_perc_90th = np.nanpercentile(temp_merqctsg, 90.)

print(tsg_perc_10th, tsg_perc_90th)

# finding indeces of sst values smaller than sst_perc_low and sst_perc_high
i_sst_low = np.where(sst < perc_10th)[0]
i_sst_high = np.where(sst >= perc_90th)[0]

# finding indeces of sst values smaller than sst_perc_low and sst_perc_high for tsg
i_sst_low_tsg = np.where(temp_merqctsg < tsg_perc_10th)[0]
i_sst_high_tsg = np.where(temp_merqctsg >= tsg_perc_90th)[0]


#%%
# generating flag to identify the sst < thr_low ( flag == 1) and sst > trh_high
# flag == 2.

ship_10s = sst[i_sst_low]
time_10s = time_sst[i_sst_low]
ship_90s = sst[i_sst_high]
time_90s = time_sst[i_sst_high]

tsg_10s = temp_merqctsg[i_sst_low_tsg]
timetsg_10s = tsg_time_sst[i_sst_low_tsg]
tsg_90s = temp_merqctsg[i_sst_high_tsg]
timetsg_90s = tsg_time_sst[i_sst_high_tsg]

flag_sst_ship = np.zeros(len(sst))
flag_sst_ship[i_sst_low] = 1
flag_sst_ship[i_sst_high] = 2

flag_sst_tsg = np.zeros(len(tsg_time_sst))
flag_sst_tsg[i_sst_low_tsg] = 1
flag_sst_tsg[i_sst_high_tsg] = 2


#%%
# plotting sst histogram and sst flag obtained from sst time serie
labelsizeaxes = 25
fontSTitle = 25
fontSizeX = 25
fontSizeY = 25
cbarAspect = 25
fontSizeCbar = 25
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 10))
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
ax.spines["top"].set_visible(False)
#ax2.spines["right"].set_visible(False)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %H'))

ax.set_ylim([26., 28.])
ax.set_ylabel('SST [$^{\circ}$C]', fontsize=fontSizeX)
ax.scatter(time_sst,  sst, color='black')
ax.scatter(time_10s,  ship_10s, color='blue', label='< 10th perc')
ax.scatter(time_90s,  ship_90s, color='red', label='>= 90th perc')

ax.plot(tsg_time_sst, temp_merqctsg, label='tsg', color='green')
ax.plot(timetsg_10s,  tsg_10s, color='blue')
ax.plot(timetsg_90s,  tsg_90s, color='red')

ax.axhline(perc_10th, 0., 1, color='black', linestyle=':', label="10th percentile")
ax.axhline(perc_90th, 0., 1, color='green', linestyle=':', label="90th percentile")
ax.set_title('time series of sst and flag for: '+string_out, fontsize=fontSTitle, loc='left')
ax.set_xlabel("time [dd hh]", fontsize=fontSizeX)
ax.set_ylabel("SST [$^{\circ}$C]", fontsize=fontSizeY)
ax.legend(frameon=False, fontsize=fontSizeX, loc='upper right')
fig.tight_layout()
fig.savefig(path_fig+string_out+'_sst_flag_'+perc_string+'_perc.png', format='png')

#%%
# saving flag, sst, and time array of the slice in ncdf
dims_ship = ['time_sst']
dims_tsg =  ['time']
coords_ship = {"time_sst": time_sst}
coords_tsg = {'time':tsg_time_sst}

sst_array = xr.DataArray(dims=dims_ship, coords=coords_ship, data=sst,
attrs = {'long_name':'sea surface temperature at -3 m from sea surface',
'units': 'Degrees Celsius'})
flag_sst_ship_array = xr.DataArray(dims=dims_ship, coords=coords_ship, data=flag_sst_ship,
 attrs={'long_name':'flag indicating sst range: == 1 sst < sst perc low , ==2 for\
 sst > sst perc high', 'units': ''})
 
sst_tsg = xr.DataArray(dims=dims_tsg, coords=coords_tsg, data=temp_merqctsg,
attrs = {'long_name':'sea surface temperature at -6 m from sea surface taken from TSG',
'units': 'Degrees Celsius'})
flag_sst_tsg_array = xr.DataArray(dims=dims_tsg, coords=coords_tsg, data=flag_sst_tsg,
 attrs={'long_name':'flag indicating sst range: == 1 sst < sst perc low , ==2 for\
 sst > sst perc high', 'units': ''})
variables         = {'sst_tsg':sst_tsg,
                     'flag_tsg':flag_sst_tsg_array}
global_attributes = {'created_by':'Claudia Acquistapace',
                     'created_on':str(datetime.now()),
                     'comment':'sst flag '}
sst_flag_dataset      = xr.Dataset(data_vars = variables,
                              coords = coords_tsg,
                              attrs = global_attributes)
sst_flag_dataset.to_netcdf(path_out+string_out+'_sst_flag'+perc_string+'_perc.nc')
