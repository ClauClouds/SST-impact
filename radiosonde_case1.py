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


data_RS = xr.open_mfdataset(file_list_RS)

print(data_RS)

fig, ax = plt.subplots()
# set here the variable from ds to plot, its color map and its min and max values
data_RS.rh.plot(y='p')
ax.set_title("relative humidity profile : ")
#ax.set_xlim(time_min, time_max)
#ax.set_ylim(0, 4500);
fig.savefig(path_RS+'rh_profile.png', format='png')
