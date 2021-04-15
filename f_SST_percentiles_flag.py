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

# set the processing mode keyword for the data you want to be processed
processing_mode = 'case_study' #  'all_campaign' #

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

# opening ship data and reading sst
dataset = xr.open_dataset(ship_data+ship_file)

# setting time window to be checked
if processing_mode == 'case_study':
    string_out = '20200202_20200203'
    t_start = datetime(2020, 2, 2, 0, 0, 0)
    t_end = datetime(2020, 2, 3, 23, 59, 59)
else:
    string_out = '20200119_20200219'
    t_start = datetime(2020, 1, 19, 0, 0, 0)
    t_end = datetime(2020, 2, 19, 23, 59, 59)

# slicing dataset for the selected time interval and extracting sst
slice = dataset.sel(time=slice(t_start, t_end))
sst = slice['SST'].values
time_sst = slice['time'].values

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
ax.scatter(time_sst, flag_arr, color='red', marker = 'o')
ax.set_ylim([0., 2.])
ax2 = ax.twinx()
ax2.set_ylim([26., 28.])
ax2.plot(time_sst,  sst, color='blue')
ax2.axhline(perc_low, 0., 1, color='black', linestyle=':')
ax2.axhline(perc_high, 0., 1, color='green', linestyle=':')
ax.set_title('time series of sst and flag for: '+string_out, fontsize=fontSTitle, loc='left')
ax.set_xlabel("time ", fontsize=fontSizeX)
ax.set_ylabel("flag []", fontsize=fontSizeY)
fig.tight_layout()
fig.savefig(path_fig+string_out+'_sst_flag_'+perc_string+'_perc.png', format='png')


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
ax.hist(sst, bins=10, color='red')
ax.set_title('sst histogram for :'+string_out, fontsize=fontSTitle, loc='left')
ax.set_xlabel("sst [$^\circ$C]", fontsize=fontSizeX)
ax.set_ylabel("occurrences [#]", fontsize=fontSizeY)
fig.tight_layout()
fig.savefig(path_fig+string_out+'_sst_histogram.png', format='png')

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
