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
ax.hist(w_warm, bins=30, color='red', label='w on warm sst percentile', histtype='step')
ax.hist(w_cold, bins=30, color='blue', label='w on cold sst percentile', histtype='step')
ax.legend(frameon=False)
ax.set_xlim([-2.,2.])

ax.set_title('w histograms for :'+string_out, fontsize=fontSTitle, loc='left')
ax.set_xlabel("w [$ms^{-1}$]", fontsize=fontSizeX)
ax.set_ylabel("occurrences [$\%$]", fontsize=fontSizeY)

ax = plt.subplot(1, 2, 2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
matplotlib.rc('xtick', labelsize=labelsizeaxes)  # sets dimension of ticks
matplotlib.rc('ytick', labelsize=labelsizeaxes)  # sets dimension of ticks
ax.hist(cb_warm, bins=30, color='red', label='cb height on warm sst percentile', histtype='step')
ax.hist(cb_cold, bins=30, color='blue', label='cb height on cold sst percentile', histtype='step')
ax.set_xlim([x_min_cb,x_max_cb])
ax.legend(frameon=False)
ax.set_title('cloud base histograms for :'+string_out, fontsize=fontSTitle, loc='left')
ax.set_xlabel("clooud base height [m]", fontsize=fontSizeX)
ax.set_ylabel("occurrences [$\%$]", fontsize=fontSizeY)
fig.tight_layout()
fig.savefig(path_fig+string_out+'_w_cb_histogram.png', format='png')
