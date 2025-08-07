import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import netCDF4 as nc
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
import pyproj
from pyproj import Proj
import pyresample

def r2_score_multi(y_pred: np.ndarray, y_true: np.ndarray):
    """Calculated the r-squared score between 2 arrays of values

    :param y_pred: predicted array
    :param y_true: "truth" array
    :return: r-squared metric
    """
    return r2_score(y_pred.flatten(), y_true.flatten())

def filterData():

    data = nc.Dataset('/ec/res4/scratch/sbjb/sfx_data/CERISE_Land_Pv2_dev02/climate/PGD.nc', mode='r')    
    xx = data["XX"][:]    
    yy = data["YY"][:]    
    data.close()

    return xx,yy

def sfx2areadef(lat0,lon0,latori,lonori,xx,yy):

    print(latori.data)

    proj2 = "+proj=lcc +lat_1=%.2f +lat_2=%.2f +lat_0=%.2f +lon_0=%.2f +units=m +ellps=WGS84 +no_defs" % (lat0,lat0,lat0,lon0)
    p2 = pyproj.Proj(proj2,preserve_units=False)
    origo = p2(lonori.data,latori.data)
    print(origo)
    extent = origo + (origo[0] + xx[-1,-1], origo[1]+yy[-1,-1])
    area_def = pyresample.geometry.AreaDefinition("id2","hei2","lcc",proj2,xx.shape[1],yy.shape[0],extent)

    return area_def

def getSFXgrid():

    pgd = nc.Dataset('/ec/res4/scratch/sbjb/sfx_data/CERISE_Land_Pv2_dev02/climate/PGD.nc','r')    
    lon0 = pgd["LON0"][0]
    lat0 = pgd["LAT0"][0]
    lonc = pgd["LONORI"][0]
    latc = pgd["LATORI"][0]

    dx = pgd["DX"][:]
    dy = pgd["DY"][:]

    xx = pgd["XX"][:]
    yy = pgd["YY"][:]
    areadef =  sfx2areadef(lat0,lon0,latc,lonc,xx,yy)

    return areadef, xx, yy


def createDataBase():

    #ds = xr.open_dataset('/ec/res4/scratch/sbjb/Projects/CERISE/ObsOpData/GNN/10GHz_static/Tuning_data/Graphs_20170828.h5', engine='netcdf4', chunks={"phony_dim_0": 512}) 

    all_dates = pd.date_range(start="2017-06-02", end="2018-06-01", freq='D')

    # Filter out the first 10 days of each month
    #filtered_dates = [date for date in all_dates if date.day <= 10] # Validation data
    filtered_dates = [date for date in all_dates if date.day > 10] # Training data

    # Convert to list of strings for display
    list_dates = [date.strftime('%Y%m%d') for date in filtered_dates]

    counter = 0

    for date_task in list_dates:

        print(date_task)
        
        try:
            ds = xr.open_dataset('/ec/res4/scratch/sbjb/Projects/CERISE/ObsOpData/GNN/36GHz_static/Tuning_data_patch_fix/Graphs_36_5_%s.h5'%date_task, engine='netcdf4') 
        except FileNotFoundError:
            continue

        ds = ds.assign_coords(
            phony_dim_0=np.arange(ds.sizes['phony_dim_0']), 
            phony_dim_1=np.arange(ds.sizes['phony_dim_1']), 
            phony_dim_2=np.arange(ds.sizes['phony_dim_2'])  
        )

        #filtered_data = ds.where(ds['FRAC_NATURE'] == 1.)

        #nan_count_x = filtered_data.isnull().sum(dim="phony_dim_1")
        nan_count_x = ds.isnull().sum(dim="phony_dim_1")

        #dd = filtered_data.where(nan_count_x < 40, drop=True)
        dd = ds.where(nan_count_x <= 1, drop=True)

        ds_mean = dd.mean(dim="phony_dim_1")

        ds_non_nan_values = ds_mean.where(~np.isnan(ds_mean), drop=True)
        no_zs = ds_non_nan_values

        #no_zs = ds_non_nan_values.where(ds_non_nan_values['ZS'] > 00.0)
        #no_zs = ds_non_nan_values.where(ds_non_nan_values['LAI_ga'] < 3.0)
        #no_zs = no_zs.where(no_zs['WGI1_ga'] < 0.01)
        #no_zs = no_zs.where(no_zs['COVER006'] < 0.1)
        #no_zs = no_zs.where(no_zs['DSN_T_ISBA'] > 0.01)

        if counter == 0:
            no_zs.to_zarr("/ec/res4/scratch/sbjb/Projects/CERISE/ObsOpData/NN/36GHz/Zarr/train_36GHz_data.zarr", mode="w")
        else:
            no_zs.to_zarr("/ec/res4/scratch/sbjb/Projects/CERISE/ObsOpData/NN/36GHz/Zarr/train_36GHz_data.zarr", mode="a", append_dim="phony_dim_0")

        counter +=1 

        #no_zs.to_zarr("pred.zarr", mode="w")
        #no_zs = no_zs.where(no_zs['TG1_ga'] > 275.0)
        #print(np.shape(no_zs["xx"].values))
        #plt.figure()
        #plt.scatter(no_zs["xx"].values, no_zs["yy"].values, c=no_zs["WG1_ga"].values)
        #plt.colorbar()
        #plt.show()


def main():

    createDataBase()

if __name__ == "__main__":

    main()
