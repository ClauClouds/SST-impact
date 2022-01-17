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
from myFunctions import lcl
from myFunctions import f_closest
from warnings import warn
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from scipy import interpolate
import custom_color_palette as ccp
from matplotlib import rcParams
import matplotlib.ticker as ticker

# reading radiosondes
path_RS = '/Volumes/Extreme SSD/work/006_projects/001_Prec_Trade_Cycle/radiosondes_atalante/case_1/'
file_list_RS = [path_RS+'EUREC4A_Atalante_Vaisala-RS_L2_v3.0.0.nc', path_RS+'EUREC4A_Atalante_Meteomodem-RS_L2_v3.0.0.nc']
path_out = '/Volumes/Extreme SSD/work/006_projects/001_Prec_Trade_Cycle/post_processed_data/'
# +
# read radiosonde from vaisala
data_RS_vaisala = xr.open_dataset(file_list_RS[0])

# read time of the launch for all radiosondes
time_launch = pd.to_datetime(data_RS_vaisala.launch_time.values)
sounding_id = data_RS_vaisala.sounding.values

# find indeces for the cold patch
ind = np.where((time_launch > datetime(2020,2,2,0,0,0)) * (time_launch < datetime(2020,2,3,23,59,59)))[0]
start = sounding_id[ind[0]]
end = sounding_id[ind[-1]]

# selecting data_RS
data_vaisala_sel = data_RS_vaisala.sel(sounding=slice(start, end))


# +
# read radiosonde from meteogram
data_RS_meteo = xr.open_dataset(file_list_RS[1])

# read time of the launch for all radiosondes
time_launch = pd.to_datetime(data_RS_meteo.launch_time.values)
sounding_id = data_RS_meteo.sounding.values

# find indeces for the cold patch
ind = np.where((time_launch > datetime(2020,2,2,0,0,0)) * (time_launch < datetime(2020,2,3,23,59,59)))[0]
start = sounding_id[ind[0]]
end = sounding_id[ind[-1]]

# selecting data_RS
data_meteo_sel = data_RS_meteo.sel(sounding=slice(start, end))


# +
# merging the two selected datasets
data_patch = xr.merge([data_vaisala_sel, data_meteo_sel])


# re-ordering files in temporal order
sounding_id = pd.to_datetime(data_patch.launch_time.values)

# re-ordering and saving order of indeces in ind_sorted
soundings_strings_sorted = np.sort(sounding_id)
ind_sorted = np.argsort(sounding_id)

# building legend strings
n_soundings = len(soundings_strings_sorted)
legend_string = []
for ind_file in range(n_soundings):
    legend_string.append(str(soundings_strings_sorted[ind_file])[8:10]+' - '+str(soundings_strings_sorted[ind_file])[11:13]+':'+str(soundings_strings_sorted[ind_file])[14:16]+' UTC')


# assigning launch_time as main coordinate instead of sounding string
data_swap = data_patch.swap_dims({"sounding": "launch_time"})

# sorting by launch time the radiosonde profiles.
data_swap = data_swap.reindex(launch_time=sorted(data_swap.launch_time.values))#("launch_time", ascending=True)
data_swap.p.plot(x="launch_time", y="alt")
# -
# calculating thermodinamic Parameters


# +
# defining matrices of data
wvmr = data_swap.mr.values
p = data_swap.p.values
rh = data_swap.rh.values
ta = data_swap.ta.values
wdir = data_swap.wdir.values
wspd = data_swap.wspd.values
q = data_swap.q.values
theta_ = data_swap.theta.values
lat = data_swap.lat.values
lon = data_swap.lon.values


# ------------------------------------------------------------------
# calculate LTS index for lower tropospheric stability (Wood and Bretherton, 2006), potential and virtual potential temperature
# ------------------------------------------------------------------
Pthr = 700 * 100. # Pressure level of 700 Hpa used as a reference in Pa
LTS = np.zeros((n_soundings))
height = data_swap.alt.values
dim_height = len(height)
theta_matrix = np.zeros((n_soundings, dim_height))
theta_v_matrix = np.zeros((n_soundings, dim_height))
theta_matrix.fill(np.nan)
theta_v_matrix.fill(np.nan)
PBLheight = np.zeros((n_soundings))
EIS = np.zeros((n_soundings))
z_lcl = np.zeros((n_soundings))
z_lcl.fill(np.nan)
theta_700 = np.zeros((n_soundings))
theta_Surf = np.zeros((n_soundings))

for ind_file in range(n_soundings):
    p_col = data_swap.p.values[ind_file, :]
    t_col = data_swap.ta.values[ind_file, :]
    rh_col = data_swap.rh.values[ind_file, :]
    h_col = height
    wvmr_col = data_swap.mr.values[ind_file, :]
    wspd_col = data_swap.wspd.values[ind_file, :]
    wdir_col = data_swap.wdir.values[ind_file, :]
    theta_col = data_swap.theta.values[ind_file, :]
    # calculating surface values by removing nans
    ind_real = np.where(~np.isnan(p_col))
    P_surf = p_col[ind_real[0][0]]
    T_surf = t_col[ind_real[0][0]]
    RH_surf = rh_col[ind_real[0][0]]
    Theta_surf = theta_col[ind_real[0][0]]
    # ------------------------------------------------------------------------------
    # calculating LCL height
    # important: provide pressure in Pascals, T in K, RH in 70.3
    #---------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------   
    z_lcl[ind_file] = round(lcl(np.array(P_surf),np.array(T_surf),np.array(RH_surf)), 2)
    #print(z_lcl[ind_file])
    
    # ------------------------------------------------------------------------------
    # calculating PBL height
    # ------------------------------------------------------------------------------
    g = 9.8                                                # gravity constant
    Rithreshold = 0.25                                     # Threshold values for Ri
    Rithreshold2 = 0.2
    Ri = np.zeros((dim_height))                       # Richardson number matrix
    PBLheightArr = []
    RiCol = np.zeros((dim_height))

    # calculating richardson number matrix using the bulk Ri method for radiosondes (https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2012JD018143)
    # calculating pressure level at free troposphere
    indP700 = np.where(p_col >= Pthr)[0][-1]
    #print(indP700)
    
    # calculating potential temperature in K
    #theta = []
    Cp = 1004.
    Rl = 287.
    #for ind in range(len(p_col)):
    #    theta.append(t_col[ind]*((100000./(p_col[ind]))**(Rl/Cp)))
    #theta_matrix[ind_file,:] = np.asarray(theta)

    # calculating LTS index for the profile
    LTS[ind_file] = (theta_col[indP700] - Theta_surf)
    
    theta_700[ind_file] = theta_col[indP700]
    theta_Surf[ind_file] = Theta_surf
    #print((theta_col[indP700] - Theta_surf))
    
    
    # calculating profiles of virtual potential temperature
    Theta_v = []
    Rd = 287.058  # gas constant for dry air [Kg-1 K-1 J]
    for indHeight in range(len(p_col)):
        k = Rd*(1-0.23*wvmr_col[indHeight])/Cp
        Tv =  (1 + 0.61*wvmr_col[indHeight])*t_col[indHeight] # calculation of virtual temperature with approximated formula

        Theta_v.append( (1 + 0.61 * wvmr_col[indHeight]) * t_col[indHeight] * (1000./(p_col[indHeight]/100.))**k)
    theta_v_matrix[ind_file,:] = np.asarray(Theta_v)


    # defining surface variables that are needed to calculate Ri
    thetaS = theta_v_matrix[ind_file, ind_real[0][0]]
    #print(np.shape(theta_v_matrix))

    zs = h_col[ind_real[0][0]]                                         # height of the surface reference
    u_s = - wspd_col[ind_real[0][0]] * np.sin(wdir_col[ind_real[0][0]])
    v_s = - wspd_col[ind_real[0][0]] * np.cos(wdir_col[ind_real[0][0]])
    #print(thetaS, zs, u_s, v_s)
    # calculation of wind componends
    for iHeight in range(dim_height):

        # calculating denominator
        # 1_calculating wind components using formulas from https://confluence.ecmwf.int/pages/viewpage.action?pageId=133262398
        u_col = - wspd_col[iHeight] * np.sin(wdir_col[iHeight])
        v_col = - wspd_col[iHeight] * np.cos(wdir_col[iHeight])
        den = (u_col-u_s)**2 + (v_col-v_s)**2
        if den == 0:
            Ri[iHeight] = 0.
        else:
            Ri[iHeight] = (1/den) * (g/thetaS) * (theta_v_matrix[ind_file,iHeight]-thetaS)*(h_col[iHeight]-zs)

    # find index in height where Ri > Rithreshold
    RiCol = Ri[:]
    #print(RiCol)
    #print(np.where(RiCol > Rithreshold2)[0][:])
    #print(len(np.where(RiCol > Rithreshold)[0][:]))
    if len(np.where(RiCol > Rithreshold)[0][:]) != 0:
        PBLheight[ind_file] = h_col[np.where(RiCol > Rithreshold)[0][0]]
        #print('pbl height = ',h_col[np.where(RiCol > Rithreshold)[0][0]])
    else:
        PBLheight[ind_file] = 0.
        #print('pbl height = ',0)
    #print(PBLheight[ind_file])
    
    #------------------------------------------------------------------
    # calculate EIS index for lower tropospheric stability (Wood and Bretherton, 2006) for observations
    # ------------------------------------------------------------------
    g = 9.8 # gravitational constant [ms^-2]
    Cp = 1005.7 # specific heat at constant pressure of air [J K^-1 Kg^-1]
    Lv = 2.50 * 10**6 # latent heat of vaporization ( or condensation ) of water [J/kg] or entalphy
    R = 8.314472 # gas constant for dry air [J/ molK]
    epsilon = 0.622 # ratio of the gas constants of dry air and water vapor
    gamma_d = g / Cp    # dry adiabatic lapse rate in K/m
    Rv = 461.5 # gas constant for water vapor [J/(Kg K)]
    Rd = 287.047  # gas constant for dry air  [J/(Kg K)]
    # calculating saturation vapor Pressure in Pa
    e0                 = 611 # pa
    T0                 = 273.15 # K
    es                 = np.zeros((dim_height))
    for ind_height in range(dim_height):
        es[ind_height] = (e0 * np.exp(Lv/Rv*(T0**(-1)-t_col[ind_height]**(-1))))

    # ---- calculating saturation mixing ratio
    ws = np.zeros((dim_height))#mpcalc.saturation_mixing_ratio(P, T)
    ws.fill(np.nan)
    for indHeight in range(dim_height):
        ws[indHeight] = (epsilon * (es[indHeight]/(p_col[indHeight] - es[indHeight]))) # saturation water vapor mixing ratio kg/Kg

    # calculation of adiabatic moist gradient
    gamma_moist = np.zeros((dim_height))

    # calculating moist adiabatic lapse rate
    for indHeight in range(len(height)):
        num = 1 + (Lv * ws[indHeight])/ (Rd * t_col[indHeight])
        den = 1 + (Lv**2 * ws[indHeight])/ (Rv * Cp * t_col[indHeight]**2)
        gamma_moist[indHeight] = gamma_d * (num / den)

    # calculating moist gradient and the height of the free troposphere level at 700 Hpa
    gamma_moist_700 = gamma_moist[indP700]
    z_700 = h_col[indP700]

    # finding closest radiosonde observation to lcl and calculating moist adiabatic lapse rate there
    ind_lcl = f_closest(h_col, z_lcl[ind_file])
    gamma_lcl = gamma_moist[ind_lcl]

    # calculating EIS using formula from Wood and Bretherton, 2006 https://doi.org/10.1175/JCLI3988.1
    #EIS.append(LTS - gamma_moist_700*z_700 + gamma_lcl*z_lcl[ind_file])
    EIS[ind_file] = LTS[ind_file] - gamma_moist[indP700]*z_700 + gamma_moist[ind_lcl]*z_lcl[ind_file]
    #print(EIS[ind_file])
    #print('EIS obtained from the Wood and Bretherton formula:')
    
# saving theta_v to ncdf for calculation with arthus raman lidar in NCDF
dims           = ['time','height']
coords         = {"time":data_swap.launch_time.values, "height":data_swap.alt.values}
theta_v           = xr.DataArray(dims=dims, coords=coords, data=theta_v_matrix,
                 attrs={'long_name':'Virtual potential temperature',
                        'units':'K'})
variables         = {'theta_v':theta_v}
Theta_v_RS      = xr.Dataset(data_vars = variables,
                       coords = coords)
Theta_v_RS.to_netcdf(path_out+'theta_v_RS.nc')

STRASUKA
# +
# reading cloud base time series from ceilometer atalante
#ceilofiles = np.sort(glob.glob('/Volumes/Extreme SSD/work/006_projects/001_Prec_Trade_Cycle/ceilometer_atalante/*_000.nc'))
ceilo_path = '/Volumes/Extreme SSD/work/006_projects/001_Prec_Trade_Cycle/ceilometer_atalante/'
ceilofiles = [ceilo_path+'20200202_Atalante_CHM188105_000.nc', ceilo_path+'20200203_Atalante_CHM188105_000.nc']
print(ceilofiles)
ceilometer = xr.open_mfdataset(ceilofiles)


# reading tsg file
tsg_file = "/Volumes/Extreme SSD/work/006_projects/001_Prec_Trade_Cycle/tsg_sst_data/tsg/nc/msm_089_1_tsg.nc"

# opening ship data and reading sst
tsg_data = xr.open_dataset(tsg_file)

# identifying time stamps of sst corresponding to time stamps of radiosondes
t_start = datetime(2020, 2, 2, 0, 0, 0)
t_end = datetime(2020, 2, 3, 23, 59, 59)

# slicing tsg datase t for the selected time interval and extracting sst
sliced_tsg_ds = tsg_data.sel(TIME=slice(t_start, t_end))
tsg_sst = sliced_tsg_ds['TEMP'].values
tsg_time_sst = sliced_tsg_ds['TIME'].values
tsg_flag = sliced_tsg_ds['TEMP_QC'].values

# averaging together the sst of the different sst sensors for tsg
temp0 = sliced_tsg_ds.TEMP[:,0].values
temp1 = sliced_tsg_ds.TEMP[:,1].values
sst_tsg = temp0
sst_tsg[np.isnan(temp0)] = temp1[np.isnan(temp0)]


# reading radiosonde time array
datetime_RS = []
tsg_datetime_RS = []
tsg_sst_0_RS = []
tsg_sst_1_RS = []
cc = []
cb_arr = []

for ind_file in range(n_soundings):

    # reading launching time from string
    dd = legend_string[ind_file][0:2]
    hh = legend_string[ind_file][5:7]
    mm = legend_string[ind_file][8:10]

    datetime_RS.append(datetime(2020, 2, int(dd), int(hh), int(mm), 0))
    #print(datetime(2020, 2, int(dd), int(hh), int(mm), 0))

    # selecting closest time in tsg
    tsg_sel = sliced_tsg_ds.sel(TIME=datetime(2020, 2, int(dd), int(hh), int(mm), 0), method='nearest')
    

    if np.isnan(tsg_sel.TEMP.values).all():
        ind_add = 2
        tsg_sel = sliced_tsg_ds.sel(TIME=datetime(2020, 2, int(dd), int(hh), int(mm)+ind_add, 0), method='nearest')
        

    #print(pd.to_datetime(tsg_sel.TIME.values))
    tsg_datetime_RS.append(pd.to_datetime(tsg_sel.TIME.values))
    #print(tsg_sel.TEMP[0].values, tsg_sel.TEMP[0].values, tsg_sel.TEMP[1].values)
    tsg_sst_0_RS.append(np.nanmean(tsg_sel.TEMP.values))
    #print(np.nanmean(tsg_sel.TEMP.values))
    
    # calculating cloud base cloud fraction around the time stamp selected
    time_int_start = datetime(2020, 2, int(dd), int(hh), int(mm), 0) - dt.timedelta(minutes=7)
    time_int_end = datetime(2020, 2, int(dd), int(hh), int(mm), 0) + dt.timedelta(minutes=7)
    # selecting cloud bases counted in the 30 min around the selected time
    ceilo_slice = ceilometer.sel(time=slice(time_int_start, time_int_end))
    # count number of values that are not nan
    cb = ceilo_slice.cbh.values[:,0]
    cb = cb.astype('float')
    cb[cb == -1] = np.nan
    n_clouds = np.count_nonzero(~np.isnan(cb))
    cc.append(n_clouds/len(cb))
    cb_arr.append(np.nanmean(cb))

tsg_sst_0_RS = np.asarray(tsg_sst_0_RS)
sst_tsg_RS = tsg_sst_0_RS
print(sst_tsg_RS)

# +
# plot multipanel with all profiles
colors_soundings = plt.cm.brg(np.linspace(0, 1, n_soundings))

# plot of variable profiles with their own heights
fig, axs = plt.subplots(3, 3, figsize=(24,20), constrained_layout=True)
grid = True
matplotlib.rc('xtick', labelsize=36)  # sets dimension of ticks in the plots
matplotlib.rc('ytick', labelsize=36) # sets dimension of ticks in the plots
labelsizeaxes   = 26
fontSizeTitle   = 26
fontSizeX       = 26
fontSizeY       = 26
cbarAspect      = 26
fontSizeCbar    = 26
text_x = np.concatenate((np.repeat(0.75, 26), np.repeat(0.9,25)))
text_y = np.concatenate((np.arange(1,0,-0.038), np.arange(1,0,-0.04)))

# sets dimension of ticks in the plots
# plotting W-band radar variables and color bars
for ind_file in range(n_soundings):
    #plot p
    axs[0,0].plot(tsg_time_sst, sst_tsg, color='grey', label='sst tsg')
    axs[0,0].scatter(tsg_datetime_RS, sst_tsg_RS, color=colors_soundings, s=120, marker='o', label='radiosonde time stamps')
    #axs[0,0].legend()
    #axs[0,0].text(text_x[ind_file], \
    #              text_y[ind_file], \
    #              legend_string[ind_file], \
    #              color=colors_soundings[ind_file], \
    #              fontsize=14, \
    #              horizontalalignment='center', \
    #              verticalalignment='center', \
    #              transform=axs[0,0].transAxes)
    
    #plot t
    axs[1,0].plot(ta[ind_file,:], height, color=colors_soundings[ind_file], linestyle=':', rasterized=True)
    axs[1,0].set_xlim(280.,300.)
    
    #plot rh
    axs[2,0].plot(rh[ind_file,:]*100, height, color=colors_soundings[ind_file], linestyle=':', rasterized=True)
    
    #plot wdir
    axs[0,1].plot(wdir[ind_file,:], height, color=colors_soundings[ind_file], linestyle=':', rasterized=True)
    axs[0,1].set_xlim(0.,200.)
    
    #plot wspd
    axs[1,1].plot(wspd[ind_file,:], height, color=colors_soundings[ind_file], linestyle=':', rasterized=True)
    axs[1,1].set_xlim(0.,20.)
    
    #plot lat/lon
    axs[2,1].scatter(lon[ind_file,:], lat[ind_file,:], s=20, marker='o', color=colors_soundings[ind_file], linestyle=':', rasterized=True)
    
    # plot q
    axs[0,2].plot(q[ind_file,:], height, color=colors_soundings[ind_file], linestyle=':', rasterized=True)
    #axs[1,2].set_xlim(0.,20.)
    
    # plot theta
    axs[1,2].plot(theta_[ind_file,:], height, color=colors_soundings[ind_file], linestyle=':', rasterized=True)
    axs[1,2].set_xlim(295.,325.)

    # plot virtual pot temp
    axs[2,2].plot(theta_v_matrix[ind_file,:], height, color=colors_soundings[ind_file], linestyle=':', rasterized=True)
    axs[2,2].set_xlim(295.,325.)

count_0 = 0
for ax, l in zip(axs[:,0].flatten(), ['(a) SST [$^{\circ}$C]',  '(b) Temperature [K]', '(c) Relative Humidity [%]']):
    ax.text(-0.05, 1.05, l,  fontweight='black', fontsize=26, transform=ax.transAxes)
    #ax.
    if count_0 == 0:
        ax.set_ylabel('SST [C]', fontsize=fontSizeX)
        ax.set_xlabel('Time [UTC]', fontsize=fontSizeX)        
        ax.set_ylim(26., 28.)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
        ax.set_xlim(t_start, t_end)
    else:
        ax.set_ylabel('Height [m]', fontsize=fontSizeX)
        ax.set_ylim(100., 4500.)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["left"].set_linewidth(3)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=3))
    ax.tick_params(which='minor', length=7, width=3)
    ax.tick_params(which='major', length=7, width=3)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=3))
    ax.tick_params(axis='both', labelsize=26)
    count_0=count_0+1
for ax, l in zip(axs[:,2].flatten(), ['(g) Specific humidity (Kg Kg$^{-1}$)',  '(h) Potential temperature [K]', '(i) Virtual potential temperature [K]']):
    ax.text(-0.05, 1.05, l,  fontweight='black', fontsize=26, transform=ax.transAxes)
    #ax.
    ax.set_ylabel('Height [m]', fontsize=fontSizeX)
    ax.set_ylim(100., 4500.)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["left"].set_linewidth(3)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=3))
    ax.tick_params(which='minor', length=7, width=3)
    ax.tick_params(which='major', length=7, width=3)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=3))
    ax.tick_params(axis='both', labelsize=26)
    
count = 0
for ax, l in zip(axs[:,1].flatten(), ['(d) Wind Direction [$^{\circ}$]',  '(e) Wind speed [ms$^{-1}$]', '(f) Lat/Lon [$^{\circ}$]']):
    ax.text(-0.05, 1.05, l,  fontweight='black', fontsize=26, transform=ax.transAxes)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["left"].set_linewidth(3)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=3))
    ax.tick_params(which='minor', length=7, width=3)
    ax.tick_params(which='major', length=7, width=3)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=3))
    ax.tick_params(axis='both', labelsize=26)
        
    
    if (count <= 1):
        ax.set_ylabel('Height [m]', fontsize=fontSizeX)
        ax.set_ylim(100., 4500.)
    else:
        ax.set_xlabel('Longitude [$^{\circ}$]', fontsize=fontSizeX)
        #ax.set_ylim(0., 2500.)
        ax.set_ylabel('Latitude [$^{\circ}$]', fontsize=fontSizeX)
    count=count+1
fig.savefig(path_RS+'Figure_sounding_variables_L2.png')

# +
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 6), constrained_layout=True)
rcParams['font.sans-serif'] = ['Tahoma']
matplotlib.rcParams['savefig.dpi'] = 100
axs[0].spines["top"].set_visible(False)
axs[0].spines["right"].set_visible(False)
axs[0].get_xaxis().tick_bottom()
axs[0].get_yaxis().tick_left()
axs[0].spines["bottom"].set_linewidth(2)
axs[0].spines["left"].set_linewidth(2)
axs[0].scatter(cc, sst_tsg_RS, color=colors_soundings, s=120, marker='o')
axs[0].set_ylabel('SST [$^{\circ}C$]', fontsize=fontSizeX)
axs[0].set_xlabel('cloud cover []', fontsize=fontSizeX)

axs[1].spines["top"].set_visible(False)
axs[1].spines["right"].set_visible(False)
axs[1].get_xaxis().tick_bottom()
axs[1].get_yaxis().tick_left()
axs[1].spines["bottom"].set_linewidth(2)
axs[1].spines["left"].set_linewidth(2)
axs[1].scatter(cc, LTS, color=colors_soundings, s=120, marker='o')
axs[1].set_ylabel('LTS []', fontsize=fontSizeX)
axs[1].set_xlabel('cloud cover []', fontsize=fontSizeX)
for ax, l in zip(axs[:].flatten(), ['(a) ',  '(b) ']):
    ax.text(-0.05, 1.05, l,  fontweight='black', fontsize=20, transform=ax.transAxes)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
    ax.tick_params(which='minor', length=5, width=2)
    ax.tick_params(which='major', length=7, width=3)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
fig.savefig(path_RS+'scatter_SST_LTS_CC.png', format='png')

# +
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 6), constrained_layout=True)
rcParams['font.sans-serif'] = ['Tahoma']
matplotlib.rcParams['savefig.dpi'] = 100
axs[0].spines["top"].set_visible(False)
axs[0].spines["right"].set_visible(False)
axs[0].get_xaxis().tick_bottom()
axs[0].get_yaxis().tick_left()
axs[0].spines["bottom"].set_linewidth(2)
axs[0].spines["left"].set_linewidth(2)
axs[0].scatter(sst_tsg_RS, theta_Surf, color=colors_soundings, s=120, marker='o')
axs[0].set_xlabel('SST [$^{\circ}C$]', fontsize=fontSizeX)
axs[0].set_ylabel('Potential temperature [K]', fontsize=fontSizeX)

axs[1].spines["top"].set_visible(False)
axs[1].spines["right"].set_visible(False)
axs[1].get_xaxis().tick_bottom()
axs[1].get_yaxis().tick_left()
axs[1].spines["bottom"].set_linewidth(2)
axs[1].spines["left"].set_linewidth(2)
axs[1].set_ylabel('Potential temperature [K]', fontsize=fontSizeX)
axs[1].set_xlabel('SST [$^{\circ}C$]', fontsize=fontSizeX)
axs[1].set_ylim(310., 316.)
axs[1].scatter(sst_tsg_RS, theta_700, color=colors_soundings, s=120, marker='o')

for ax, l in zip(axs[:].flatten(), ['(a) theta at surface',  '(b) theta at 700 Hpa']):
    ax.text(-0.05, 1.05, l,  fontweight='black', fontsize=20, transform=ax.transAxes)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
    ax.tick_params(which='minor', length=5, width=2)
    ax.tick_params(which='major', length=7, width=3)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
fig.savefig(path_RS+'scatter_theta_SST.png', format='png')
# -

# scatter plots of sst radiosonde versus EIS, LTS, LCL, PBLheight


# +
# composite figure for potential temperature and virtual potential temperature
fig, axs = plt.subplots(2, 2, figsize=(12,8), sharex=True, constrained_layout=True)
grid = True
matplotlib.rc('xtick', labelsize=36)  # sets dimension of ticks in the plots
matplotlib.rc('ytick', labelsize=36) # sets dimension of ticks in the plots
labelsizeaxes   = 26
fontSizeTitle   = 26
fontSizeX       = 26
fontSizeY       = 26
cbarAspect      = 26
fontSizeCbar    = 26

import custom_color_palette as ccp
from matplotlib import rcParams
import matplotlib.ticker as ticker


    #plot p
axs[0,0].scatter(sst_tsg_RS, PBLheight, color=colors_soundings, s=80, marker='o', rasterized=True)
#axs[0].set_xlim(290.,325.)
axs[0,0].set_ylabel('PBL height [m]', fontsize=fontSizeX)
axs[0,0].set_ylim(100., 1800.)


#axs[0].text(text_x[ind_file], text_y[ind_file], legend_string[ind_file], color=colors_soundings[ind_file], fontsize=14, horizontalalignment='center', verticalalignment='center', transform=axs[0].transAxes)
#plot t
axs[0,1].scatter(sst_tsg_RS, z_lcl, color=colors_soundings, s=80, marker='o', rasterized=True)
axs[0,1].set_ylabel('LCL [m]', fontsize=fontSizeX)
axs[0,1].set_ylim(100., 1000.)

#axs[1].set_xlim(300.,325.)
#plot rh
axs[1,0].scatter(sst_tsg_RS, EIS, color=colors_soundings, s=80, marker='o', rasterized=True)
axs[1,0].set_ylabel('EIS []', fontsize=fontSizeX)
axs[1,0].set_ylim(-3., 10.)

axs[1,1].scatter(sst_tsg_RS, LTS, color=colors_soundings, s=80, marker='o', rasterized=True)
axs[1,1].set_ylabel('LTS []', fontsize=fontSizeX)
axs[1,1].set_ylim(10., 20.)


for ax, l in zip(axs[:].flatten(), ['(a) ',  '(b) ',  '(c) ',  '(d) ']):
    ax.text(-0.05, 1.05, l,  fontweight='black', fontsize=20, transform=ax.transAxes)
    #ax.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["left"].set_linewidth(3)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
    ax.tick_params(which='minor', length=5, width=2)
    ax.tick_params(which='major', length=7, width=3)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
    ax.tick_params(axis='both', labelsize=26)
fig.savefig(path_RS+'scatter_parameters_sst.png')
# -

# reading cloud base slice for plotting
ceilo_slice = ceilometer.sel(time=slice(t_start, t_end))
cb_plot = ceilo_slice.cbh.values[:,0]
cb_plot = cb_plot.astype('float')
cb_plot[cb_plot == -1] = np.nan
time_cb_plot = ceilo_slice.time.values

# +
# composite figure for time series
fig, axs = plt.subplots(5, 1, figsize=(10,15), sharex=True, constrained_layout=True)
grid = True
matplotlib.rc('xtick', labelsize=36)  # sets dimension of ticks in the plots
matplotlib.rc('ytick', labelsize=36) # sets dimension of ticks in the plots
labelsizeaxes   = 23
fontSizeTitle   = 23
fontSizeX       = 23
fontSizeY       = 23
cbarAspect      = 26
fontSizeCbar    = 26

axs[0].plot(tsg_time_sst, sst_tsg, color='grey', label='sst tsg')
axs[0].scatter(tsg_datetime_RS, sst_tsg_RS, color=colors_soundings, s=120, marker='o', label='radiosonde time stamps')
axs[0].set_ylabel('SST [$^{\circ}$C]', fontsize=fontSizeX)
axs[0].legend()
    #plot p
#axs[0].set_xlim(290.,325.)
axs[1].scatter(tsg_datetime_RS, PBLheight, color=colors_soundings, s=120, marker='o', rasterized=True)
axs[1].set_ylim(0., 1000.)
axs[1].set_ylabel(' PBL height [m]', fontsize=fontSizeX)
#axs[1].set_ylim(100., 1800.)

#axs[0].text(text_x[ind_file], text_y[ind_file], legend_string[ind_file], color=colors_soundings[ind_file], fontsize=14, horizontalalignment='center', verticalalignment='center', transform=axs[0].transAxes)
#plot t
axs[2].scatter(tsg_datetime_RS, z_lcl, color=colors_soundings, s=120, marker='o', rasterized=True)
axs[2].set_ylabel('LCL [m]', fontsize=fontSizeX)
axs[2].set_ylim(500., 1000.)

#axs[1].set_xlim(300.,325.)
#plot rh
axs[4].scatter(tsg_datetime_RS, EIS, color=colors_soundings, s=120, marker='o', rasterized=True, label='EIS')
axs[4].scatter(tsg_datetime_RS, LTS, color=colors_soundings, s=120, marker='+', rasterized=True, label='LTS')
axs[4].legend(frameon=False)
axs[4].set_ylabel('stability indeces []', fontsize=fontSizeX)
axs[4].set_ylim(-3., 20.)

axs[3].scatter(time_cb_plot, cb_plot, color='orange',  s=60, marker=7, label='cloud base')
axs[3].set_ylabel('Cloud base height [m]', fontsize=fontSizeX)
axs[3].set_ylim(0., 1500.)


for ax, l in zip(axs[:].flatten(), ['(a) ',  '(b) ',  '(c) ',  '(d) ', '(e) ']):
    ax.text(-0.05, 1.05, l,  fontweight='black', fontsize=16, transform=ax.transAxes)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %H'))
    ax.set_xlim(t_start, t_end)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["left"].set_linewidth(3)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
    ax.tick_params(which='minor', length=5, width=2)
    ax.tick_params(which='major', length=7, width=3)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
    ax.tick_params(axis='both', labelsize=26)
    ax.grid(True, which='both', color='grey', linestyle=':')
fig.savefig(path_RS+'parameters_time_series.png')
# -

print(height)

# creating dataset with all the data
# saving data in ncdf file
dims              = ['sst','height']
coords         = {"sst":sst_tsg_RS, "height":height}
theta           = xr.DataArray(dims=dims, coords=coords, data=theta_,
                 attrs={'long_name':'Potential temperature',
                        'units':'K'})
theta_v           = xr.DataArray(dims=dims, coords=coords, data=theta_v_matrix,
                 attrs={'long_name':'Virtual potential temperature',
                        'units':'K'})
temperature  = xr.DataArray(dims=dims, coords=coords, data=ta,
                 attrs={'long_name':'Air temperature',
                        'units':'K'})
relative_humidity = xr.DataArray(dims=dims, coords=coords, data=rh,
                 attrs={'long_name':'Relative humidity',
                        'units':'%'})
wind_speed = xr.DataArray(dims=dims, coords=coords, data=wspd,
                 attrs={'long_name':'Horizontal wind speed',
                        'units':'ms-1'})
wind_direction = xr.DataArray(dims=dims, coords=coords, data=wdir,
                 attrs={'long_name':'Horizontal wind direction',
                        'units':'ms-1'})
water_vapor_mixing_ratio = xr.DataArray(dims=dims, coords=coords, data=wvmr,
                 attrs={'long_name':'Water vapor mixing ratio',
                        'units':'kg kg-1'})
pressure = xr.DataArray(dims=dims, coords=coords, data=p,
                 attrs={'long_name':'pressure',
                        'units':'Pa'})
latitude = xr.DataArray(dims=dims, coords=coords, data=lat,
                 attrs={'long_name':'Latitude',
                        'units':'degrees'})
longitude = xr.DataArray(dims=dims, coords=coords, data=lon,
                 attrs={'long_name':'Longitude',
                        'units':'degrees'})
pblh = xr.DataArray(dims=['sst'], coords={'sst':sst_tsg_RS}, data=PBLheight,
                 attrs={'long_name':'Planetary boundary layer height calculated using bulk Richardson number',
                        'units':'meters'})
lifting_condensation_level = xr.DataArray(dims=['sst'], coords={'sst':sst_tsg_RS}, data=z_lcl,
                 attrs={'long_name':'lifting condensation level calculated using Rumps formula',
                        'units':'meters'})
cloud_base_height = xr.DataArray(dims=['sst'], coords={'sst':sst_tsg_RS}, data=cb_arr,
                 attrs={'long_name':'mean cloud base extracted from ceilometer from Atalante',
                        'units':'meters'})
cloud_fraction = xr.DataArray(dims=['sst'], coords={'sst':sst_tsg_RS}, data=cc,
                 attrs={'long_name':'mean cloud base cloud fraction extracted from ceilometer from Atalante averaged on 15 min around the selected time',
                        'units':'meters'})
LTS_index = xr.DataArray(dims=['sst'], coords={'sst':sst_tsg_RS}, data=LTS,
                 attrs={'long_name':'Lower tropospheric stability index from Wood and Bretherton, 2006',
                        'units':'K'})
EIS_index = xr.DataArray(dims=['sst'], coords={'sst':sst_tsg_RS}, data=EIS,
                 attrs={'long_name':'Lower tropospheric stability index from Wood and Bretherton, 2006',
                        'units':'K'})

variables         = {'lts':LTS_index,
                     'eis':EIS_index,
                     'cf':cloud_fraction,
                     'cbh':cloud_base_height,
                     'lcl':lifting_condensation_level,
                     'pblh':pblh,
                     'longitude':longitude,
                     'latitude':latitude,
                     'pressure':pressure,
                     'wvmr':water_vapor_mixing_ratio,
                     'wind_dir':wind_direction,
                     'wind_speed':wind_speed,
                     'rh':relative_humidity*100.,
                     'ta':temperature,
                     'theta_v':theta_v,
                     'theta':theta,
                             }

RS_atalante_Data      = xr.Dataset(data_vars = variables,
                       coords = coords)
RS_atalante_Data_new = RS_atalante_Data.reindex(sst=sorted(RS_atalante_Data.sst.values))
RS_atalante_Data_new.to_netcdf(path_out+'radiosondes_atalante_binned_sst.nc')


#RS_atalante_Data_new.pressure.plot()
RS_atalante_Data_new.theta.plot(x='sst', y='height')

# +
# plot of variables sorted as a function of sst
fig, axs = plt.subplots(3, 3, figsize=(24,20), constrained_layout=True)
grid = True
matplotlib.rc('xtick', labelsize=26)  # sets dimension of ticks in the plots
matplotlib.rc('ytick', labelsize=26) # sets dimension of ticks in the plots
# check the link https://matplotlib.org/stable/tutorials/text/text_props.html for more info on types
font = {'family' : 'Tahoma',
        'weight' : 'normal',
        'size'   : 26}
matplotlib.rc('font', **font) 
labelsizeaxes   = 26
fontSizeTitle   = 26
fontSizeX       = 26
fontSizeY       = 26
cbarAspect      = 26
fontSizeCbar    = 26
RS_atalante_Data_new.theta.plot(x='sst', y='height', ax=axs[0,0], cmap="viridis", vmin=300., vmax=320.)
axs[0,0].set_ylim(0.,5000.)

RS_atalante_Data_new.theta_v.plot(x='sst', y='height', ax=axs[0,1], cmap="viridis", vmin=300., vmax=320.)
axs[0,1].set_ylim(0.,5000.)

RS_atalante_Data_new.wvmr.plot(x='sst', y='height', ax=axs[0,2], cmap="viridis", vmin=0., vmax=.02)
axs[0,2].set_ylim(0.,5000.)

RS_atalante_Data_new.rh.plot(x='sst', y='height', ax=axs[1,0], cmap="viridis", vmin=0., vmax=100.)
axs[1,0].set_ylim(0.,5000.)

RS_atalante_Data_new.wind_speed.plot(x='sst', y='height', ax=axs[1,1], cmap="viridis", vmin=0., vmax=20.)
axs[1,1].set_ylim(0.,5000.)

RS_atalante_Data_new.wind_dir.plot(x='sst', y='height', ax=axs[1,2], cmap="viridis", vmin=0., vmax=200.)
axs[1,2].set_ylim(0.,5000.)

RS_atalante_Data_new.ta.plot(x='sst', y='height', ax=axs[2,0], cmap="viridis", vmin=280., vmax=300.)
axs[2,0].set_ylim(0.,5000.)

RS_atalante_Data_new.eis.plot(marker='o', ax=axs[2,1], label='EIS')
RS_atalante_Data_new.lts.plot(marker='v', ax=axs[2,1], label='LTS')
axs[2,1].legend(frameon=False)

RS_atalante_Data_new.pblh.plot(marker='o', ax=axs[2,2], label='pblh')
RS_atalante_Data_new.cbh.plot(marker='v', ax=axs[2,2], label='cbh')
RS_atalante_Data_new.lcl.plot(marker='P', ax=axs[2,2], label='lcl')

axs[2,2].legend(frameon=False)
fig.savefig(path_RS+'fields_sortedby_sst.png')
# -
#%%
# building binned array of sst
bin_size = 0.25
binned_sst = np.round(np.arange(np.nanmin(sst_tsg_RS),np.nanmax(sst_tsg_RS), bin_size),1)
print(len(binned_sst))
binned_sst

#defining color palette for subsequent plots
colors_binned_sst = plt.cm.seismic(np.linspace(0, 1, len(binned_sst)))


# calculating mean and std properties for each bin of sst
list_binned_datasets = []
list_test = []
for ind_sst in range(len(binned_sst)-1):
    
    # slicing the bin of sst selected
    sliced_data_SST = RS_atalante_Data_new.sel(sst=slice(binned_sst[ind_sst], binned_sst[ind_sst+1]))
    
    # saving the slice in a list
    list_binned_datasets.append(sliced_data_SST)
    
    # calculating quantiles of the slice
    list_test.append(sliced_data_SST.quantile([0, 0.25, 0.5, 0.75, 1], dim="sst", skipna=True))



binned_sst

# storing the new variables in the file
# RRData['calibration_constant']  = data['calibration_constant']
# RS_atalante_Data.attrs                    = global_attributes
# RS_atalante_Data['time'].attrs          = {'units':'seconds since 1970-01-01 00:00:00'}
# RS_atalante_Data.to_netcdf(pathOutData+date+'_'+hour+'_preprocessedClau_4Albert.nc')

# +
print(binned_sst)
#generate data arrays for boxplots of scalar quantities

def f_generate_list_binned_var(var_list, var_name_string):
    output_list = []
    n_elements_list = len(var_list)
    for i,el in enumerate(var_list):
        output_list.append(el[var_name_string].values)
    return(output_list)

lts_binned = f_generate_list_binned_var(list_binned_datasets, 'lts')
eis_binned = f_generate_list_binned_var(list_binned_datasets, 'eis')
lcl_binned = f_generate_list_binned_var(list_binned_datasets, 'lcl')
pblh_binned = f_generate_list_binned_var(list_binned_datasets, 'pblh')
cbh_binned = f_generate_list_binned_var(list_binned_datasets, 'cbh')
cf_binned = f_generate_list_binned_var(list_binned_datasets, 'cf')

# calculate label marks for bins
sst_bin_label = []
for ind in range(len(binned_sst)-1):
    sst_bin_label.append(round((binned_sst[ind]+binned_sst[ind+1])/2,2))
    
    
# removing nans from the cloud base height data
cbh_binned_nonans = []
for ind,el in enumerate(cbh_binned):
    cbh_binned_test = el
    cbh_binned_nonans.append(cbh_binned_test[~np.isnan(cbh_binned_test)])
sst_bin_label

# +
# composite figure variables binned in sst 
fig, axs = plt.subplots(3, 2, figsize=(16,10), sharex=True, constrained_layout=True)
grid = True
matplotlib.rc('xtick', labelsize=36)  # sets dimension of ticks in the plots
matplotlib.rc('ytick', labelsize=36) # sets dimension of ticks in the plots
val = 22
labelsizeaxes   = val
fontSizeTitle   = val
fontSizeX       = val
fontSizeY       = val
cbarAspect      = val
fontSizeCbar    = val
green_diamond = dict(markerfacecolor='b', marker='D')

axs[0,0].boxplot(lcl_binned, positions=sst_bin_label, notch=False, flierprops=green_diamond)
#axs[0].set_xlim(290.,325.)
axs[0,1].boxplot(lts_binned, positions=sst_bin_label, notch=False, flierprops=green_diamond)
axs[1,0].boxplot(cf_binned, positions=sst_bin_label, notch=False, flierprops=green_diamond)
axs[2,0].boxplot(pblh_binned, positions=sst_bin_label, notch=False, flierprops=green_diamond)
axs[2,1].boxplot(eis_binned, positions=sst_bin_label, notch=False, flierprops=green_diamond)
axs[1,1].boxplot(cbh_binned_nonans, positions=sst_bin_label, notch=False, flierprops=green_diamond)

axs[0,0].set_ylabel('lcl [m]', fontsize=fontSizeX)
axs[0,0].set_ylim(500., 1000.)
axs[0,1].set_ylabel('lts [K]', fontsize=fontSizeX)
axs[0,1].set_ylim(10., 20.)
axs[1,0].set_ylabel('cloud fraction []', fontsize=fontSizeX)
axs[1,0].set_ylim(0., 1.)
axs[1,1].set_ylabel('cbh [m]', fontsize=fontSizeX)
axs[1,1].set_ylim(500., 1500.)
axs[2,0].set_ylabel('pblh[m]', fontsize=fontSizeX)
axs[2,0].set_ylim(0., 1500.)
axs[2,1].set_ylabel('eis [K]', fontsize=fontSizeX)
axs[2,0].set_xlabel('SST [$^{\circ}$C]', fontsize=fontSizeX)
axs[2,1].set_xlabel('SST [$^{\circ}$C]', fontsize=fontSizeX)

count = 0
for ax, l in zip(axs[:].flatten(), ['(a) ',  '(b) ',  '(c) ',  '(d) ', '(e) ', '(f) ']):
    ax.text(-0.05, 1.1, l,  fontweight='black', fontsize=val, transform=ax.transAxes)
    ax.set_xlim(binned_sst[0]-0.1, binned_sst[-1]+0.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["left"].set_linewidth(3)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
    ax.tick_params(which='minor', length=5, width=2)
    ax.tick_params(which='major', length=7, width=3)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
    ax.tick_params(axis='both', labelsize=26)
    #ax.grid(True, which='both', color='grey', linestyle=':')
    count = count+1
fig.savefig(path_RS+'scatter_binned_sst.png')

# +
list_std_binned_datasets = []
list_mean_binned_datasets = []

# calculate mean profile and std
for ind,el in enumerate(list_binned_datasets):
    # calculating mean and standard deviation
    list_mean_binned_datasets.append(el.mean(dim='sst', skipna=True))
    list_std_binned_datasets.append(el.std(dim='sst', skipna=True))
list_mean_binned_datasets[0]
# -

fig, axs = plt.subplots(2, 4, figsize=(16,10), sharey=True, constrained_layout=True)
import matplotlib.font_manager as font_manager
grid = True
matplotlib.rc('xtick', labelsize=36)  # sets dimension of ticks in the plots
matplotlib.rc('ytick', labelsize=36) # sets dimension of ticks in the plots
labelsizeaxes   = 16
fontSizeTitle   = 16
fontSizeX       = 16
fontSizeY       = 16
cbarAspect      = 16
fontSizeCbar    = 16
labels =[]
for ind, el in enumerate(sst_bin_label):
    labels.append('SST = '+str(el))
print(labels)
font = font_manager.FontProperties(family='Tahoma',
                                   weight='light',
                                   style='normal', size=12)
for ind_dataset,DS in enumerate(list_mean_binned_datasets):
    
    #plot dataset mean and std of the radiosondes
    axs[0,0].plot(DS['theta'].values, DS['height'].values, color=colors_binned_sst[ind_dataset], label=labels[ind_dataset], rasterized=True)
    axs[0,0].legend(frameon=False, prop=font)
    axs[1,0].plot(DS['theta_v'].values, DS['height'].values, color=colors_binned_sst[ind_dataset], label=labels[ind_dataset], rasterized=True)
    axs[0,1].plot(DS['wind_speed'].values, DS['height'].values, color=colors_binned_sst[ind_dataset], label=labels[ind_dataset], rasterized=True)
    axs[1,1].plot(DS['wind_dir'].values, DS['height'].values, color=colors_binned_sst[ind_dataset], label=labels[ind_dataset], rasterized=True)
    axs[0,2].plot(DS['wvmr'].values, DS['height'].values, color=colors_binned_sst[ind_dataset], label=labels[ind_dataset], rasterized=True)
    axs[1,2].plot(DS['pressure'].values, DS['height'].values, color=colors_binned_sst[ind_dataset], label=labels[ind_dataset], rasterized=True)
    axs[0,3].plot(DS['rh'].values, DS['height'].values, color=colors_binned_sst[ind_dataset], label=labels[ind_dataset], rasterized=True)
    axs[1,3].plot(DS['ta'].values, DS['height'].values, color=colors_binned_sst[ind_dataset], label=labels[ind_dataset], rasterized=True)


    #axs[0].set_xlim(290.,325.)
axs[0,0].set_xlim(295., 325.)
axs[0,0].set_xlabel('Potential temp [$^{\circ}$K]', fontsize=fontSizeX)
axs[1,0].set_xlim(295., 325.)
axs[1,0].set_xlabel('Virtual potential temp [$^{\circ}$K]', fontsize=fontSizeX)
axs[0,1].set_xlabel('Wind speed [ms$^{-1}$]', fontsize=fontSizeX)
axs[1,1].set_xlabel('Wind dir [$^{\circ}$]', fontsize=fontSizeX)
axs[1,1].set_xlim(0., 200.)
axs[1,3].set_xlim(280., 300.)
axs[0,3].set_xlabel('Rel hum. [%]', fontsize=fontSizeX)
axs[0,2].set_xlabel('water vap. mix. ratio [kgkg$^{-1}$]', fontsize=fontSizeX)
axs[1,2].set_xlabel('Pressure [Pa]', fontsize=fontSizeX)
axs[1,3].set_xlabel('Air Temp [$^{\circ}$K]', fontsize=fontSizeX)
#axs[1,2].set_xlim(50000, 1000020)
for ax, l in zip(axs[:].flatten(), ['(a) ',  '(b) ',  '(c) ',  '(d) ', '(e) ', '(f) ', '(g) ', '(h) ']):
    ax.text(-0.05, 1.1, l,  fontweight='black', fontsize=22, transform=ax.transAxes)
    ax.set_ylim(0., 4500.)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["left"].set_linewidth(3)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
    ax.tick_params(which='minor', length=5, width=2)
    ax.tick_params(which='major', length=7, width=3)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
    ax.tick_params(axis='both', labelsize=26)
fig.savefig(path_RS+'profiles_binned_sst.png')
