# %%
"""
Created on Sat Mar 20 09:43:09 2021

@author: j

this program read GOES files (cloud mask or a specific channel) and produce an estimate of Cloud Fraction outside and inside an SST contour (the value set for is the 26.5 contour). A good part of this code have been readadapted from personal communication with the developer of GOES toolbox (joaohenry23@gmail.com). To use this porgram the GOES package need to be install first. See the instruction at: https://github.com/joaohenry23/GOES
Due to my inexperience in using python is quite porbable that part of this code can be ameliorated. In comparison to the non minimal version this contain only the essential part

"""

import GOES #install this pacage from: https://github.com/joaohenry23/GOES
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

domain = [-55.5,-52.0, 5.9, 8.0]
path = '/home/j/Desktop/PROJ_WORK_Thesis/GOES_data/' # path to the GOES file are
save_path='/home/j/Desktop/PROJ_WORK_Thesis/figures/GOES/' # path to save the image to

#get the list of GOES file in path
flist = GOES.locate_files(path, 'OR_ABI-L2-ACMF*.nc',
                          '20200202-040000', '20200204-040400')

# get MErian and TALANTe track from EUREC4a intake
Merian_track = xr.open_dataset(\
"https://observations.ipsl.fr/thredds/dodsC/EUREC4A/PRODUCTS/TRACKS/EUREC4A_tracks_MS-Merian_v1.0.nc")
Atalante_track = xr.open_dataset(\
"https://observations.ipsl.fr/thredds/dodsC/EUREC4A/PRODUCTS/TRACKS/EUREC4A_tracks_Atalante_v1.0.nc")
# slice the tracks
Merian_track2_3 = Merian_track.sel(time=slice("2020-02-02T00:00:00",\
 "2020-02-04T00:00:00"));
Atalante_track2_3 = Atalante_track.sel(time=slice("2020-02-02T00:00:00",\
 "2020-02-04T00:00:00"))

#read local SST file to xarray dataset 
sst_mur3 = xr.open_dataset(
    '/home/j/Desktop/PROJ_WORK_Thesis/SST_MUR_Remi/SST_MUR_2020-02-03.nc')

sst_mur3 = sst_mur3.assign_coords(lon=("lon", sst_mur_grid.X_SST_MUR))
sst_mur3 = sst_mur3.assign_coords(lat=("lat", sst_mur_grid.Y_SST_MUR))
sst_mur3 = sst_mur3.rename({'X': 'lon','Y': 'lat'})

sst = sst_mur3
sst_lon, sst_lat = np.meshgrid(sst.lon.data, sst.lat.data)
sst_data = sst.sst_MUR.data

# Creates a grid map with cylindrical equidistant projection (equirectangular projection) and 2 km of spatial resolution
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
SSTup =SST.where(SST.lon>=-53.25)

# find where the SST is lower then a threshold (26.5 to begin with)

mask_cold_sst = SST.data <= 26.5
mask_sst_upwind = SSTup.data >= 26.6;
mask_sstg27 = SST.data >= 27.0
mask_sst = ~np.isnan(SST.data)

# %% calculate cloud fraction outside and over the SST (cold)
time = dt.datetime(1,1,1)
time = np.array([time + dt.timedelta(hours=i*0) for i in range(len(flist))])
BCMall = xr.DataArray(CC, dims=("lat", "lon", "time"), coords={"lat": SST.lat.data, 
                                   "lon": SST.lon.data, "time":time })

# this for cycle can be run only the first time and then save the result BCMall to a netcdf to be uploaded every other time
for k in np.arange(0,flist.__len__()):
   
    file = flist[k]
    ds = GOES.open_dataset(file)
    time[k]= ds.variable('t').data 

   # get image with the coordinates of Center of their pixels
    BCM, LonCen, LatCen = ds.image('BCM', lonlat='center', domain=domain)

    # reproject GOES data to equirectangular gridmap ...
    SwathDef = SwathDefinition(lons=LonCen.data, lats=LatCen.data)
    BCMCyl = resample_nearest(SwathDef, BCM.data, AreaDef, radius_of_influence=6000,
                              fill_value=np.nan, epsilon=3, reduce_data=True)  
    # ... and save GOES data as xarray
    BCM = xr.DataArray(BCMCyl[::-1,:], coords=[LatCenCyl.data[::-1,0], 
                                 LonCenCyl.data[0,:]], dims=['lat', 'lon'])
    # save the all xarray with time and lat lon coordinates
    BCMall[:,:,k]=BCM        
    print(k)
        
        
for k in  np.arange(0,flist.__len__()):
    BCMall.time.data[k].replace(tzinfo = timezone.utc)

# %% save the BCMall to a netcdf
BCMnc = BCMall
for k in  np.arange(0,flist.__len__()):
    BCMnc.time.data[k] = BCMnc.time.data[k].timestamp()
new_filename = '2_3_Feb_BCMall_GOES.nc'
print ('saving to ', new_filename)
BCMall.to_netcdf(path = path + new_filename)
BCMall.close()
print ('finished saving')

# %%
#load the BCM all saved to netcdf (this is to save the time to run the previous section)
BCMall = xr.open_dataarray(path + '2_3_Feb_BCMall_GOES.nc')
CC = BCMall.data

# calcolo cloud cover totale 
cc = CC.sum(axis=2)/CC[1,1,:].size

# select cloud cover outside and inside of the SST la cloud cover fuori e dentro la sst
ccIn = cc[mask_cold_sst]
ccOut = cc[mask_sst * (~mask_cold_sst)]
BCMallUpwind = BCMall.sel(lon=slice(-53.5, domain[1]))
ccUpwind = BCMallUpwind.data.sum(axis=2)/BCMallUpwind.data[1,1,:].size

cc_re = cc.flatten(order='C')
ccIn_re = ccIn.flatten(order='C')
ccOut_re = ccOut.flatten(order='C')
ccUpwind_re = ccUpwind.flatten(order='C')

# figure: 2d histogram CC vs SST
plt.hist2d(cc.flatten(order='C'),SST.data.flatten(order='C'),
           range=[[0, 0.4], [26,28.5]], cmap='viridis')
plt.title('2-3 Feb 2D histogram cloud cover vs SST')
plt.colorbar(label='counts')
plt.xlabel('Cloud Cover')
plt.ylabel('SST')

#convert to Barbados time    
timeBA =BCMall.time.data-dt.timedelta(hours=4) 

# %% Figures

import matplotlib.dates as mdates
myFmt = mdates.DateFormatter('%d-%H')
from matplotlib import rcParams
matplotlib.rcParams.update({'font.size': 20})


# Figure 1: subplot of the area of interest and histogram of cloud cover
fig, ax = plt.subplots(nrows=2, dpi=300,figsize=(12, 15))
plt.sca(ax[0])
plt.contourf(BCMall.lon.data, BCMall.lat.data, cc, cmap = 'Blues', 
               levels = np.arange(0,0.701,0.1))
plt.colorbar()
cs = plt.contour(SST.lon.data, SST.lat.data, SST.data, 
                levels = [26.3 ,26.5, 27, 27.3, 27.6],  colors ='k' )
cs.clabel(fontsize=15, inline=1, fmt='%.1f', colors='k')
cs1 = plt.contour(SST.lon.data, SST.lat.data, SST.data, 
                levels = [26.3 ,26.5, 27, 27.3,27.6],  cmap='autumn_r' )
plt.title('Cloud cover from 02/02 00:00 to 03/02 23:59 LT and SST_MUR for 03/02 ',
          loc='left', fontsize=20)
plt.grid()
plt.xlabel('Longitude')
plt.ylabel('Latitude')

plt.sca(ax[1])
bins = np.arange(0,0.4,0.05)
plt.hist(ccIn,alpha=0.2,color = 'blue',density=True,
           rwidth=0.8,bins=bins, label='over cold SST (SST<=26.5)')
plt.hist(ccOut,alpha=0.2, color = 'red',density=True,
           rwidth=1, bins=bins, label='outside (SST>26.5)')
plt.hist(ccUpwind_re,alpha=0.2, color = 'green',density=True,
           rwidth=1, bins=bins, label='upwind (lon>-53.5)')
plt.legend(framealpha=0.2)
plt.title(
    'Normalized cloud cover distribution over and outside the cold SST',
    fontsize=18)
    
     
  
    


