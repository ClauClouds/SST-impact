#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 20:28:50 2021

@author: j
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 12:53:16 2021

@author: j
"""

import pandas as pd
import custom_color_palette as ccp #for custum colormaps
import matplotlib as mpl
import matplotlib.pyplot as plt
import eurec4a
import GOES
import custom_color_palette as ccp
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cf
from cartopy.feature import NaturalEarthFeature
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
import pyproj as pyproj
from pyresample import utils
from pyresample.geometry import SwathDefinition
from pyresample.kd_tree import resample_nearest
import datetime as dt
from datetime import datetime, timezone
import cftime
import scipy.stats as stats # for Mann_whitney test
from decimal import *

#%% LOAD the data
#list of GOES data

path = '/home/j/Documents/article_SST/GOES_data/'
#######àà choose the case study ########################
case_study = 3; # 1 = 23 Jan, 2=29-30 Jan, 3=2-3 Feb
########################################################

if case_study == 1:
    date = '23 Jan LT' 
    flist = GOES.locate_files(path+'COD/', 'OR_ABI-L2-CODF*.nc','20200123-040000', '20200124-040400',
                          use_parameter='scan_start_time')
    # get the info about the server where the data are and the catalogn of the data\
        #with the eurec4a package 
    cat = eurec4a.get_intake_catalog()
    lst=list(cat) #to get the all the stuff that the catalogn contains
    #from the catalog take the GOES16 13 channel data for the 23 Jan 
    # ds23_13 = cat.satellites.GOES16.latlongrid(channel=13, date="2020-01-23").to_dask()
    # ds02_13 = cat.satellites.GOES16.latlongrid(channel=13, date="2020-02-02").to_dask()
    #beacuse there is some porblem with the download of the data get the satellite data directly
    Merian_track = xr.open_dataset("https://observations.ipsl.fr/thredds/dodsC/EUREC4A/PRODUCTS/TRACKS/EUREC4A_tracks_MS-Merian_v1.0.nc")
    
    #Merian track for the selcted temporal interval
    Merian_track_int = Merian_track.sel(time=slice("2020-01-23T04:00:00",\
    "2020-01-24T04:02:00"))
       
    #this is the area coverd by the Merian during 29 and 30 of Feb
    domain = [min(Merian_track_int.lon.data)-0.2, max(Merian_track_int.lon.data)+0.2,
              min(Merian_track_int.lat.data)-0.2, max(Merian_track_int.lat.data)+0.2] 
    

    sst_mur23 = xr.open_dataset(
    '/home/j/Desktop/PROJ_WORK_Thesis/SST_MUR_Remi/20200123090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.nc')
    sst_mur = sst_mur23.sel(lon=slice(domain[0],domain[1]),lat=slice(domain[2],domain[3])) 
    sst = sst_mur;
    sst_data = sst.analysed_sst.data
    sst_data = sst_data[0,:,:]
    sst_data = sst_data-273.15
    sst_lon, sst_lat = np.meshgrid(sst.lon.data, sst.lat.data)
    
elif case_study == 2:
    date = '29-30 Jan LT' 
    flist = GOES.locate_files(path+'COD/', 'OR_ABI-L2-CODF*.nc','20200129-040000', '20200131-040400',
                          use_parameter='scan_start_time')
    Merian_track = xr.open_dataset("https://observations.ipsl.fr/thredds/dodsC/EUREC4A/PRODUCTS/TRACKS/EUREC4A_tracks_MS-Merian_v1.0.nc")
    
    #Merian track for the selcted temporal interval
    Merian_track_int = Merian_track.sel(time=slice("2020-01-29T04:00:00",\
    "2020-01-31T04:02:00"))
       
    #this is the area coverd by the Merian during 29 and 30 of Feb
    domain = [min(Merian_track_int.lon.data)-0.2, max(Merian_track_int.lon.data)+0.2,
              min(Merian_track_int.lat.data)-0.2, max(Merian_track_int.lat.data)+0.2] 
    
    # sst MUR
    sst_mur = xr.open_dataset(
        '/home/j/Desktop/PROJ_WORK_Thesis/SST_MUR_Remi/SST_MUR_2020-01-29.nc')
    sst_mur_grid = xr.open_dataset(
        '/home/j/Desktop/PROJ_WORK_Thesis/SST_MUR_Remi/SST_MUR_Grid.nc')
    
    sst_mur = sst_mur.assign_coords(lon=("lon", sst_mur_grid.X_SST_MUR))
    sst_mur = sst_mur.assign_coords(lat=("lat", sst_mur_grid.Y_SST_MUR))
    sst_mur = sst_mur.rename({'X': 'lon','Y': 'lat'})
    
    sst_mur = sst_mur.sel(lon=slice(domain[0],domain[1]), lat=slice(domain[2],domain[3]))
    
    sst = sst_mur;
    sst_data = sst.sst_MUR.data
    
    sst_lon, sst_lat = np.meshgrid(sst.lon.data, sst.lat.data)

elif case_study == 3:
    date = '2-3 Feb LT' 
    flist = GOES.locate_files(path+'COD/', 'OR_ABI-L2-CODF*.nc','20200202-040000', '20200204-040400',
                          use_parameter='scan_start_time')
    Merian_track = xr.open_dataset("https://observations.ipsl.fr/thredds/dodsC/EUREC4A/PRODUCTS/TRACKS/EUREC4A_tracks_MS-Merian_v1.0.nc")
    
    #Merian track for the selcted temporal interval
    Merian_track_int = Merian_track.sel(time=slice("2020-02-02T04:00:00",\
    "2020-02-04T04:02:00"))
       
    #this is the area coverd by the Merian during 29 and 30 of Feb
    #domain = [min(Merian_track_int.lon.data)-0.2, max(Merian_track_int.lon.data)+0.2,
             # min(Merian_track_int.lat.data)-0.2, max(Merian_track_int.lat.data)+0.2] 
    domain = [-55.5, -52, 5.5, 8.5]; # limits choosen with Claudia, Agostino etc
    
    # sst MUR
    sst_mur = xr.open_dataset(
        '/home/j/Desktop/PROJ_WORK_Thesis/SST_MUR_Remi/SST_MUR_2020-02-02.nc')
    sst_mur_grid = xr.open_dataset(
        '/home/j/Desktop/PROJ_WORK_Thesis/SST_MUR_Remi/SST_MUR_Grid.nc')
    
    sst_mur = sst_mur.assign_coords(lon=("lon", sst_mur_grid.X_SST_MUR))
    sst_mur = sst_mur.assign_coords(lat=("lat", sst_mur_grid.Y_SST_MUR))
    sst_mur = sst_mur.rename({'X': 'lon','Y': 'lat'})
    
    sst_mur = sst_mur.sel(lon=slice(domain[0],domain[1]), lat=slice(domain[2],domain[3]))
    
    sst = sst_mur;
    sst_data = sst.sst_MUR.data
    
    sst_lon, sst_lat = np.meshgrid(sst.lon.data, sst.lat.data)


# Creates a grid map with cylindrical equidistant projection (equirectangular projection) and 2 km of spatial resolution.
LonCenCyl, LatCenCyl = GOES.create_gridmap(domain, PixResol=2.0)

# Calculates the parameters for reprojection
# For this we need install the pyproj and pyresample packages. Try with pip install pyproj and pip install pyresample.
Prj = pyproj.Proj('+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 +x_0=0 +y_0=0 +a=6378.137 +b=6378.137 +units=km')
AreaID = 'cyl'
AreaName = 'cyl'
ProjID = 'cyl'
Proj4Args = '+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 +x_0=0 +y_0=0 +a=6378.137 +b=6378.137 +units=km'

ny, nx = LonCenCyl.data.shape
SW = Prj(LonCenCyl.data.min(), LatCenCyl.data.min())
NE = Prj(LonCenCyl.data.max(), LatCenCyl.data.max())
area_extent = [SW[0], SW[1], NE[0], NE[1]]

# Creates area for reproject
AreaDef = utils.get_area_def(AreaID, AreaName, ProjID, Proj4Args, nx, ny, area_extent)

# reproject SST data to equirectangular gridmap ...
SwathDef = SwathDefinition(lons=sst_lon, lats=sst_lat)
SSTCyl = resample_nearest(SwathDef, sst_data, AreaDef, radius_of_influence=6000,
                          fill_value=np.nan, epsilon=3, reduce_data=True)
  
# .. and save SST data as xarray
SST = xr.DataArray(SSTCyl[::-1,:], coords=[LatCenCyl.data[::-1,0], 
                                           LonCenCyl.data[0,:]], dims=['lat', 'lon'])

# SSTup =SST.where(SST.lon>=-53.5)

# # find where the SST is lower then a threshold (26.5 to begin with)

# mask_cold_sst = SST.data <= 26.5
# mask_sst_upwind = SSTup.data >= 26.6;
# mask_sstg27 = SST.data >= 27.0
# mask_sst = ~np.isnan(SST.data)
# %% calculate cloud fraction outside and over the SST (cold)
#inizialize some variables
CFin=np.empty(len(flist))
CFin[:] = np.nan
CFout=np.empty(len(flist))
CFout[:] = np.nan
time = dt.datetime(1,1,1)
time = np.array([time + dt.timedelta(hours=i*0) for i in range(len(flist))])
cloud_thr = 294;
product_switch = 0 # 1 --> use single channel, 0--> use clear sky porduct
mask = 1
CC = np.empty([SST.data.shape[0],SST.data.shape[1],len(flist)])
BCMall = xr.DataArray(CC, dims=("lat", "lon", "time"), coords={"lat": SST.lat.data, 
                                   "lon": SST.lon.data, "time":time })

for k in np.arange(0,flist.__len__()):
   
    file = flist[k]
    ds = GOES.open_dataset(file)
    time[k]= ds.variable('t').data
    # ds = GOES.open_dataset(path +
    #     'OR_ABI-L2-CMIPF-M6C02_G16_s20200331650132_e20200331659440_c20200331659520.nc')
    
    if  product_switch == 1: 
        # get image with the coordinates of Center of their pixels. The
        CMI, LonCen, LatCen = ds.image('CMI', lonlat='center', domain=domain)
      
        # reproject GOES data to equirectangular gridmap ...
        SwathDef = SwathDefinition(lons=LonCen.data, lats=LatCen.data)
        CMICyl = resample_nearest(SwathDef, CMI.data, AreaDef, radius_of_influence=6000,
                                  fill_value=np.nan, epsilon=3, reduce_data=True)  
        # ... and save GOES data as xarray
        CMI = xr.DataArray(CMICyl[::-1,:], coords=[LatCenCyl.data[::-1,0], 
                                     LonCenCyl.data[0,:]], dims=['lat', 'lon'])
        # select the value of CMI above the cold SST
        CMIcold = CMI.where(mask_cold_sst == True)
        cloudyCMIcold = CMIcold.data <= cloud_thr 
        # calculate cloud fraction over cold water
        CFin[k] = np.count_nonzero(cloudyCMIcold)/np.count_nonzero(~np.isnan(CMIcold.data))
        
        # due opzioni per calcolarsi il fuori
        #1 tutto il resto (meno la terra, a quello serve la moltiplica)
        CMIout = CMI.where(mask_sst * (~mask_cold_sst) == True )
        #2 tutta la sst maggiore di 27 °
        #CMIout = CMI.where(mask_sstg27 == True)
        #3 zona upwind (defnita come una zona a est della sst fredda)
    
        # find where there are clouds
        cloudyCMIout = CMIout.data <= cloud_thr 
        
        # cloud fraction outside the cold water
        CFout[k] = np.count_nonzero(cloudyCMIout)/np.count_nonzero(~np.isnan(CMIout.data))
        print(k)
        
        
    elif product_switch == 0:
        # same thing but for the clear sky mask (BCM)
        BCM, LonCen, LatCen = ds.image('COD', lonlat='center', domain=domain)
        # reproject GOES data to equirectangular gridmap ...
        SwathDef = SwathDefinition(lons=LonCen.data, lats=LatCen.data)
        BCMCyl = resample_nearest(SwathDef, BCM.data, AreaDef, radius_of_influence=6000,
                                  fill_value=np.nan, epsilon=3, reduce_data=True)  
        # ... and save GOES data as xarray
        BCM = xr.DataArray(BCMCyl[::-1,:], coords=[LatCenCyl.data[::-1,0], 
                                     LonCenCyl.data[0,:]], dims=['lat', 'lon'])
       
        # # select cold and warm areas 
        # BCMcold = BCM.where(mask_cold_sst == True)
        # cloudyBCMcold = BCMcold.data == 1
        # # calculate cloud fraction over cold water
        # CFin[k] = np.count_nonzero(cloudyBCMcold)/np.count_nonzero(
        #     ~np.isnan(BCMcold))
               
        # # tre opzioni per calcolarsi il fuori
        # if mask == 1:
        # #1 tutto il resto (meno la terra, a quello serve la moltiplica)
        #         BCMout = BCM.where(mask_sst * (~mask_cold_sst) == True )
        # #2 tutta la sst maggiore di 27 °
        # elif mask == 2: 
        #         BCMout = BCM.where(mask_sstg27 == True)
        # elif mask == 3: 
        # #3 zona upwind (defnita come una zona a est della sst fredda)
        #         BCMout = BCM.where(mask_sst_upwind == True)
        
        # # find where there are clouds
        # cloudyBCMout = BCMout.data == 1 
        
        # # cloud fraction outside the cold water
        # CFout[k] = np.count_nonzero(cloudyBCMout)/np.count_nonzero(
        #     ~np.isnan(BCMout)) 
        
        # save each cloud image of 0 and 1 as a numpy array
        # CC[:,:,k] = BCM.data
        # save the all xarray with time and lat lon coordinates these two operation are reduntad
        BCMall[:,:,k]=BCM
        
        print(k)
for k in  np.arange(0,flist.__len__()):
    BCMall.time.data[k].replace(tzinfo = timezone.utc)
    
#%% save the BCMall to a netcdf

BCMnc = BCMall
for k in  np.arange(0,flist.__len__()):
    BCMnc.time.data[k] = BCMnc.time.data[k].timestamp()
new_filename = date + '_CODall_GOES.nc'
print ('saving to ', new_filename)
BCMall.to_netcdf(path = path + new_filename)
BCMall.close()
print ('finished saving')


#%% figure for video 

#load the BCM all saved to netcdf (this is to save the time to run the previous section)
BCMall = xr.open_dataarray(path + date + '_CODall_GOES.nc')
#Barbados Time
time = dt.datetime(1,1,1)
time = np.array([time + dt.timedelta(hours=i*0) for i in range(len(flist))])
for k in  np.arange(0,flist.__len__()):
   time[k] = datetime.fromtimestamp(int(BCMall.time.data[k]))
timeBA = time-dt.timedelta(hours=4)

save_path = path + '/figures/GOES/video/'
import matplotlib.dates as mdates
myFmt = mdates.DateFormatter('%d-%H')
from matplotlib import rcParams
matplotlib.rcParams.update({'font.size': 16})

save_path = '/home/j/Documents/article_SST/figures/GOES/'
mycmap = matplotlib.colors.ListedColormap(['white', 'grey'])

levels = plt.MaxNLocator(nbins=10).tick_values(0, 50)

# pick the desired colormap, sensible levels, and define a normalization
# instance which takes data values and translates those into levels.
cmap = plt.get_cmap('cool')
norm = matplotlib.colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)

for k in np.arange(144,288):
    fig, ax = plt.subplots(figsize=(14,8), dpi=300)

    #plot SST
    cs = plt.contour(sst_mur.lon.data, sst_mur.lat.data, sst_data, 
                 cmap = 'viridis')
    cs.clabel(fontsize=12, inline=1, fmt='%.2f', colors='k')
    
    css = plt.pcolormesh(BCMall.lon.data, BCMall.lat.data, BCMall[:,:,k].data, cmap = cmap,
                 alpha=0.75,vmin=0 ,vmax=50, norm = norm)
    cbar = fig.colorbar(css, orientation='vertical')
    cbar.set_label(label='COD at 640 nm',size=12)
    cbar.ax.tick_params(labelsize=12)

    
    #plot merian track 
    plt.plot(Merian_track_int.lon, Merian_track_int.lat, color = 'r')
    plt.plot(Merian_track_int.lon[k*10], Merian_track_int.lat[k*10], 'o', color = 'magenta')
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(timeBA[k].strftime('%d %b %H:%M') + ' LT (UTC-4): ' + 
              'GOES cloud optical depth at 640 nm and SST MUR for 23 Jan',
              fontsize=14, loc='left')
    
    #save the figure
    plt.savefig(save_path + 'COD_' + timeBA[k].strftime('%d_%b_%H:%M') + '_LT'+'.png',
                bbox_inches='tight',dpi=300)
   # plt.show()
    plt.close()
    
     
#%% figure: 2-dimensional probability density function: Cloud optical depth vs SST

#load the BCM all saved to netcdf (this is to save the time to run the previous section)
BCMall = xr.open_dataarray(path + date + '_CODall_GOES.nc')

SST_repeat = np.repeat(SST.data[:,:,np.newaxis], BCMall.shape[2], axis=2)

save_path = '/home/j/Documents/article_SST/figures/GOES/'
plt.rcParams['font.size'] = '16'
fig, ax = plt.subplots(figsize=(14,8), dpi=300)
plt.hist2d(BCMall.data.flatten(order='C'),SST_repeat.flatten(order='C'),
           range=[[BCMall.min().data, 10], [SST.min().data,SST.max().data]],
           bins = 20, cmap='viridis')
plt.title(date + ' 2D histogram cloud optical depth at 640 nm vs SST', loc='left')
plt.colorbar(label='counts')
plt.xlabel('Cloud Optical Depth')
plt.ylabel('SST [°C]')
#save the figure
plt.savefig(save_path + date + '_COD'  + '_LT'+'.png',
                bbox_inches='tight',dpi=300)
