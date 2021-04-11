"""
Created on Sat Mar 20 09:43:09 2021

@author: j

this program read GOES files (cloud mask or a specific channel) and produce an estimate of Cloud Fraction outside and inside an SST contour (the value set for is the 26.5 contour). A good part of this code have been readadapted from personal communication with the developer of GOES toolbox (joaohenry23@gmail.com). To use this porgram the GOES package need to be install first. See the instruction at: https://github.com/joaohenry23/GOES
Due to my inexperience in using python is quite porbable that part of this code can be ameliorated. 

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
sst_mur1 = xr.open_dataset(
    '/home/j/Desktop/PROJ_WORK_Thesis/SST_MUR_Remi/SST_MUR_2020-02-01.nc')
sst_mur2 = xr.open_dataset(
    '/home/j/Desktop/PROJ_WORK_Thesis/SST_MUR_Remi/SST_MUR_2020-02-02.nc')
sst_mur3 = xr.open_dataset(
    '/home/j/Desktop/PROJ_WORK_Thesis/SST_MUR_Remi/SST_MUR_2020-02-03.nc')
sst_mur4 = xr.open_dataset(
    '/home/j/Desktop/PROJ_WORK_Thesis/SST_MUR_Remi/SST_MUR_2020-02-04.nc')
sst_mur5 = xr.open_dataset(
    '/home/j/Desktop/PROJ_WORK_Thesis/SST_MUR_Remi/SST_MUR_2020-02-05.nc')
sst_mur6 = xr.open_dataset(
    '/home/j/Desktop/PROJ_WORK_Thesis/SST_MUR_Remi/SST_MUR_2020-02-06.nc')
sst_mur_grid = xr.open_dataset(
    '/home/j/Desktop/PROJ_WORK_Thesis/SST_MUR_Remi/SST_MUR_Grid.nc')

sst_mur1 = sst_mur1.assign_coords(lon=("lon", sst_mur_grid.X_SST_MUR))
sst_mur1 = sst_mur1.assign_coords(lat=("lat", sst_mur_grid.Y_SST_MUR))
sst_mur1 = sst_mur1.rename({'X': 'lon','Y': 'lat'})

sst_mur2 = sst_mur2.assign_coords(lon=("lon", sst_mur_grid.X_SST_MUR))
sst_mur2 = sst_mur2.assign_coords(lat=("lat", sst_mur_grid.Y_SST_MUR))
sst_mur2 = sst_mur2.rename({'X': 'lon','Y': 'lat'})

sst_mur3 = sst_mur3.assign_coords(lon=("lon", sst_mur_grid.X_SST_MUR))
sst_mur3 = sst_mur3.assign_coords(lat=("lat", sst_mur_grid.Y_SST_MUR))
sst_mur3 = sst_mur3.rename({'X': 'lon','Y': 'lat'})

sst_mur4 = sst_mur4.assign_coords(lon=("lon", sst_mur_grid.X_SST_MUR))
sst_mur4 = sst_mur4.assign_coords(lat=("lat", sst_mur_grid.Y_SST_MUR))
sst_mur4 = sst_mur4.rename({'X': 'lon','Y': 'lat'})

sst_mur5 = sst_mur5.assign_coords(lon=("lon", sst_mur_grid.X_SST_MUR))
sst_mur5 = sst_mur5.assign_coords(lat=("lat", sst_mur_grid.Y_SST_MUR))
sst_mur5 = sst_mur5.rename({'X': 'lon','Y': 'lat'})

sst_mur6 = sst_mur6.assign_coords(lon=("lon", sst_mur_grid.X_SST_MUR))
sst_mur6 = sst_mur6.assign_coords(lat=("lat", sst_mur_grid.Y_SST_MUR))
sst_mur6 = sst_mur6.rename({'X': 'lon','Y': 'lat'})

# figure: comparison between SST MUR
levs =[26.5]
plt.figure(dpi=300)
plt.contour(sst_mur1.lon,sst_mur1.lat,sst_mur1.sst_MUR.data,levels = levs,
            colors='red',label='1 Feb', alpha=0.7)
plt.contour(sst_mur2.lon,sst_mur2.lat,sst_mur2.sst_MUR.data,levels = levs,
            colors='blue',label='2 Feb', alpha=0.7)
plt.contour(sst_mur3.lon,sst_mur3.lat,sst_mur3.sst_MUR.data,levels = levs,
            colors='green',label='3 Feb', alpha=0.7)
plt.contour(sst_mur4.lon,sst_mur4.lat,sst_mur4.sst_MUR.data,levels = levs,
            colors='grey',label='4 Feb', alpha=0.7)
plt.contour(sst_mur5.lon,sst_mur5.lat,sst_mur5.sst_MUR.data,levels = levs,
            colors='magenta',label='5 Feb', alpha=0.7)
plt.contour(sst_mur6.lon,sst_mur6.lat,sst_mur6.sst_MUR.data,levels = levs,
            colors='orange',label='6 Feb', alpha=0.7)
plt.text(0.9,0.9,
         'red : 1 Feb, \n blue: 2 Feb \n green: 3 Feb \n grey: 4 Feb \n magenta: 5 Feb \n orange: 6 Feb',
         horizontalalignment='right',
     verticalalignment='top', transform = ax.transAxes)
plt.legend(["1 Feb", "2 Feb","3 Feb","4 Feb", "5 Feb","6 Feb"])
plt.title('26.5 SST contours for a 1-6 Feb')
plt.grid()

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

# this for cycle can be run only the first time and then save the result BCMall to a netcdf to be uploaded every other time
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
        BCM, LonCen, LatCen = ds.image('BCM', lonlat='center', domain=domain)
        
        # reproject GOES data to equirectangular gridmap ...
        SwathDef = SwathDefinition(lons=LonCen.data, lats=LatCen.data)
        BCMCyl = resample_nearest(SwathDef, BCM.data, AreaDef, radius_of_influence=6000,
                                  fill_value=np.nan, epsilon=3, reduce_data=True)  
        # ... and save GOES data as xarray
        BCM = xr.DataArray(BCMCyl[::-1,:], coords=[LatCenCyl.data[::-1,0], 
                                     LonCenCyl.data[0,:]], dims=['lat', 'lon'])
        
        BCMcold = BCM.where(mask_cold_sst == True)
        cloudyBCMcold = BCMcold.data == 1
        # calculate cloud fraction over cold water
        CFin[k] = np.count_nonzero(cloudyBCMcold)/np.count_nonzero(
            ~np.isnan(BCMcold))
               
        # tre opzioni per calcolarsi il fuori
        if mask == 1:
        #1 tutto il resto (meno la terra, a quello serve la moltiplica)
                BCMout = BCM.where(mask_sst * (~mask_cold_sst) == True )
        #2 tutta la sst maggiore di 27 °
        elif mask == 2: 
                BCMout = BCM.where(mask_sstg27 == True)
        elif mask == 3: 
        #3 zona upwind (defnita come una zona a est della sst fredda)
                BCMout = BCM.where(mask_sst_upwind == True)
        
        # find where there are clouds
        cloudyBCMout = BCMout.data == 1 
        
        # cloud fraction outside the cold water
        CFout[k] = np.count_nonzero(cloudyBCMout)/np.count_nonzero(
            ~np.isnan(BCMout)) 
        
        # save each cloud image of 0 and 1 ans a numpy array
        CC[:,:,k] = BCM.data
        # save the all xarray with time and lat lon coordinates these two operation are reduntad
        BCMall[:,:,k]=BCM        
        print(k)
        
        
for k in  np.arange(0,flist.__len__()):
    BCMall.time.data[k].replace(tzinfo = timezone.utc)
    
#%% save the BCMall to a netcdf
BCMnc = BCMall
for k in  np.arange(0,flist.__len__()):
    BCMnc.time.data[k] = BCMnc.time.data[k].timestamp()
new_filename = '2_3_Feb_BCMall_GOES.nc'
print ('saving to ', new_filename)
BCMall.to_netcdf(path = path + new_filename)
BCMall.close()
print ('finished saving')

#%%   
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

#%% Figures 

import matplotlib.dates as mdates
myFmt = mdates.DateFormatter('%d-%H')
from matplotlib import rcParams
matplotlib.rcParams.update({'font.size': 20})


# Figure 1: subplot of the area of interest and histogram of cloud conver 
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
         # histtype='barstacked', rwidth=0.8)


# Figure 2: figure for video
fts = 18
mycmap = matplotlib.colors.ListedColormap(['white', 'blue'])
for k in np.arange(208,288): #(0,flist.__len__()):
    fig = plt.figure('maps', figsize=(12,8), dpi=300)
    ax = fig.add_axes([0.1,0.1,0.8,0.8], projection=ccrs.PlateCarree())
    ax.outline_patch.set_linewidth(0.3)
    countries = cf.NaturalEarthFeature(category='cultural', name='admin_0_countries', 
                              alpha=0.3, scale='50m', facecolor='grey')
    ax.add_feature(countries, edgecolor='black', linewidth=0.5)
    cs = plt.contour(SST.lon.data, SST.lat.data, SST.data, 
                levels = [26.5],  colors ='k', alpha=0.7, linewidths = 3)
    cs.clabel(fontsize=14, inline=1, fmt='%.1f', colors='k')
    # plt.contourf(SST.lon.data, SST.lat.data, SST.data, 
    #             levels = [26.2, 26.5, 27, 27.5, 28],  cmap='coolwarm', alpha=0.7 )
    plt.contourf(BCMall.lon.data, BCMall.lat.data, BCMall[:,:,k].data, 
                 cmap=mycmap,alpha=0.7 )
    plt.title(timeBA[k].strftime('%d %b %H:%M') + ' LT (UTC-4): ' + 'clouds from clear sky mask',
              fontsize=fts, loc='left')
    ax.set_aspect('equal', adjustable='box') 
    plt.grid()
    ax.set_extent([domain[0]+360.0, domain[1]+360.0, domain[2], domain[3]], crs=ccrs.PlateCarree())
    #set xticks
    dx = 0.5
    xticks = np.arange(domain[0], domain[1], dx)
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter(dateline_direction_label=True))
    ax.set_xlabel('Longitude', color='black', fontsize=fts, labelpad=3.0)
    #set yticks
    dy = 0.5
    yticks = np.arange(domain[2], domain[3], dy)
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.set_ylabel('Latitude', color='black', fontsize=fts, labelpad=3.0)
     # Sets tick characteristics
    ax.tick_params(left=True, right=True, bottom=True, top=True,
                   labelleft=True, labelright=False, labelbottom=True, labeltop=False,
                   length=0.0, width=0.05, labelsize=14.0, labelcolor='black')   
    # Sets grid characteristics
    ax.gridlines(xlocs=xticks, ylocs=yticks, alpha=0.8, color='gray',
                 draw_labels=False, linewidth=0.25, linestyle='--')
    #save the figure
    plt.savefig(save_path + 'ACM_' + timeBA[k].strftime('%d_%b_%H:%M') + '_LT'+'.png',
                bbox_inches='tight',dpi=300)
    plt.show()
    plt.close()
   


# Figure 3: figure timesereis of Cloud fraction and SST areas (probably not useful)
fig, ax = plt.subplots(nrows=2, dpi=300,figsize=(12, 15))
plt.sca(ax[0])
plt.plot(timeBA, CFin,'b.', label='CF over the cold SST (<=26.5)')
plt.plot(timeBA, CFout,'r.',label='CF over the warm SST (>=27)')
if  product_switch == 1: 
    plt.title('Cloud fraction upwind and over the cold SST (cloud = BT<294K)',
              loc='left')
elif product_switch == 0: 
     plt.title('CF upwind and over the cold SST (cloud from clear sky mask)',
              loc='left')
plt.xlabel('Local Time (UTC-4)')
ax[0].xaxis.set_major_formatter(myFmt)
plt.grid()
plt.legend()
plt.sca(ax[1])
plt.contour(SST.lon.data, SST.lat.data, SST.data, colors = 'black',
                levels = [26.5,26.6])
plt.contourf(SSTup.lon.data, SSTup.lat.data, SSTup.data,
                levels = [23.5,26.5,27,30], cmap='bwr')
plt.contourf(SST.lon.data, SST.lat.data, SST.data,
                levels = [25.5,26.5], cmap='winter')
plt.title('the two SST based regions')
ax[1].set_aspect('equal', adjustable='box') 
#plt.subplots_adjust(bottom=0., right=0.8, top=0.9)
plt.grid()
   
    
     
  
    

    
