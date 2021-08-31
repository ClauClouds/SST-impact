#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 10:42:54 2020
@ date; 31 august 2021
@author: Claudia Acquistapace
@goal: plot radiosonde quantities for the case study of the 2nd february using data from l'atalante
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

from warnings import warn
import numpy as np
import pandas as pd

"""
The ``atmosphere`` module contains methods to calculate relative and
absolute airmass and to determine pressure from altitude or vice versa.
"""

APPARENT_ZENITH_MODELS = ('simple', 'kasten1966', 'kastenyoung1989',
                          'gueymard1993', 'pickering2002')
TRUE_ZENITH_MODELS = ('youngirvine1967', 'young1994')
AIRMASS_MODELS = APPARENT_ZENITH_MODELS + TRUE_ZENITH_MODELS

def pres2alt(pressure):
    '''
    Determine altitude from site pressure.

    Parameters
    ----------
    pressure : numeric
        Atmospheric pressure. [Pa]

    Returns
    -------
    altitude : numeric
        Altitude above sea level. [m]

    Notes
    ------
    The following assumptions are made

    ============================   ================
    Parameter                      Value
    ============================   ================
    Base pressure                  101325 Pa
    Temperature at zero altitude   288.15 K
    Gravitational acceleration     9.80665 m/s^2
    Lapse rate                     -6.5E-3 K/m
    Gas constant for air           287.053 J/(kg K)
    Relative Humidity              0%
    ============================   ================

    References
    -----------
    .. [1] "A Quick Derivation relating altitude to air pressure" from
       Portland State Aerospace Society, Version 1.03, 12/22/2004.
    '''

    alt = 44331.5 - 4946.62 * pressure ** (0.190263)

    return alt


path_RS = '/Volumes/Extreme SSD/work/006_projects/001_Prec_Trade_Cycle/radiosondes_atalante/case_1/'
file_list_RS = np.sort(glob.glob(path_RS+'EUREC4A_Atalante_Meteomodem-RS_L1-*.nc'))
#np.sort(glob.glob(path_files+'*.nc'))

# calculating total number of soundings
n_soundings = len(file_list_RS)

DS_list = []

for ind_file in range(n_soundings):

    # reading data from file
    data_RS = xr.open_dataset(file_list_RS[ind_file])

    # dropping pressure duplicates
    DS = data_RS.to_dataframe()
    DS_clean = DS.drop_duplicates('p', keep='first')
    data_RS = xr.Dataset.from_dataframe(DS_clean)

    # defining pressure and height grid for the first file
    if ind_file == 0:
        # building pressure grid fro all data
        pressure = data_RS.p.values[:,0]
        pressure_grid = np.arange(data_RS.p.values[0,0], data_RS.p.values[-1,0], -20.)

        # calculating corresponding height array based on the conversion formula
        height = []
        for ind_p in range(len(pressure_grid)):
            height.append(pres2alt(pressure_grid[ind_p]))

    # removing values smaller than maxima at the surface
    if np.argmax(pressure) !=0:
        data_RS = data_RS.sel(level=slice(np.argmax(pressure), data_RS.level.values[-1]))
        print('cut profile ')

    # assigning vertical coordinate to radiosonde data based on conversion formula from pressure
    data_RS_good = data_RS.assign_coords({'level':data_RS.p.values[:,0]})

    # interpolating the pressure on the pressure equidistant levels and the fixed height
    data_RS_interp = data_RS_good.reindex(level=pressure_grid, method='nearest')

    # adding data to list
    DS_list.append(data_RS_interp)

print(len(DS_list))
strasuka
fig, ax = plt.subplots()
# set here the variable from ds to plot, its color map and its min and max values
#plt.plot(data_RS.rh.values[:,0], data_RS.p.values[:,0], color='blue', label='original')
#plt.plot(data_RS_good.rh.values[:,0], data_RS_good.p.values[:,0], color='red', label='p')
plt.plot(data_RS_interp.rh.values[:,0], height, color='green', label='interp')
ax.set_title("relative humidity profile : ")
#ax.set_xlim(time_min, time_max)
#ax.set_ylim(0, 4500);
ax.legend()
fig.savefig(path_RS+'rh_profile.png', format='png')

fig, ax = plt.subplots()
plt.plot(data_RS_good.p.values[:,0])
fig.savefig(path_RS+'P-original.png', format='png')

fig, ax = plt.subplots()
plt.plot(data_RS_interp.p.values[:,0])
fig.savefig(path_RS+'P-interp.png', format='png')
strasuka

#V_windS_hour = V_windS_hour.interp(height=heightRad)

#data_RS = data_RS.assign_coords({"height": height_grid})

print(data_RS)

#data_RS['height'] = height
print(np.ediff1d(height))
print(height[0][0], height[0][-1])
