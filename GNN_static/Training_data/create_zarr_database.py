import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import datetime
import time
import os, sys, glob
import netCDF4 as nc
import numpy.ma as ma
import zarr
import pandas as pd
import scipy
import h5py
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
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


def xgmodel():

    ds = xr.open_zarr("/ec/res4/scratch/sbjb/Projects/CERISE/ObsOpData/NN/18GHz/Zarr/train_18GHz_data.zarr", chunks={"pony_dim_0":512})
    ds_val = xr.open_zarr("/ec/res4/scratch/sbjb/Projects/CERISE/ObsOpData/NN/18GHz/Zarr/val_18GHz_data.zarr", chunks={"pony_dim_0":512})

    feat_lst = [
        "ZS",        
        "FRAC_LAND_AND_SEA_WATER", 
        "Distance_to_footprint_center",
        "TG1_ga",
        "TG2_ga",
        "WG1_ga",
        "WG2_ga",
        "WGI1_ga",
        "WGI2_ga",
        "TS_ISBA",        
        "LAI_ga",
        "HSN_VEG1_ga",
        "HSN_VEG6_ga",
        "HSN_VEG12_ga",
        "RSN_VEG1_ga",
        "RSN_VEG6_ga",
        "RSN_VEG12_ga",         
        "WSN_T_ISBA",
        "DSN_T_ISBA"        
    ]

    targ_lst = [
        "AMSR2_BT18.7V"
    ]

    feats_ds = ds[feat_lst].to_array().astype("float32").transpose()
    target_ds0 = ds[targ_lst].to_array().astype("float32").transpose()

    feats_ds_val = ds_val[feat_lst].to_array().astype("float32").transpose()
    target_ds_val = ds_val[targ_lst].to_array().astype("float32").transpose()

    # Training data
    X = feats_ds.values
    y = target_ds0.values

    msk1 = ~np.isnan(X).any(axis=1)
    msk2 = ~np.isnan(y).any(axis=1)
    
    msk_both = np.logical_and(msk1,msk2)

    #X_t = X[msk_both,:]
    #y_t = y[msk_both,:]
    X_train = X[msk_both,:]
    y_train = y[msk_both,:]

    # Validation data
    Xval = feats_ds_val.values
    yval = target_ds_val.values

    msk1_val = ~np.isnan(Xval).any(axis=1)
    msk2_val = ~np.isnan(yval).any(axis=1)
    
    msk_both_val = np.logical_and(msk1_val,msk2_val)

    X_val = Xval[msk_both_val,:]
    y_val = yval[msk_both_val,:]

    #X_train, X_val, y_train, y_val = train_test_split(X_t, y_t, test_size=0.2, random_state=42)

    print(np.shape(X_train))
    print(np.shape(y_train))
    
    model = xgb.XGBRegressor(
        n_estimators=256,
        tree_method="hist",
        objevtive=mean_absolute_error,
        #multi_strategy="multi_output_tree",
        learning_rate=0.3,
        eval_metric=r2_score_multi,
        #eval_metric="rmse",
        subsample=0.6,
    )
    
    xgb_model = model

    print("Fitting XGB model...")
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    model.get_booster().feature_names = feat_lst

    feature_important = model.get_booster().get_score(importance_type='weight')
    keys = list(feature_important.keys())   
    values = list(feature_important.values())

    pickle.dump(xgb_model, open("/ec/res4/scratch/sbjb/Projects/CERISE/ObsOpData/XGB/18GHz/xgb_train", "wb"))

    data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)
    plt.figure()
    data.nlargest(42, columns="score").plot(kind='barh', figsize = (20,10)) 
    plt.savefig("test2_year.png")
    plt.close()


def predict():

    xgb_model_loaded = pickle.load(open("/ec/res4/scratch/sbjb/Projects/CERISE/ObsOpData/XGB/18GHz/xgb_train", "rb"))
    #ds = xr.open_zarr("/ec/res4/scratch/sbjb/Projects/CERISE/ObsOpData/GNN/10GHz_static/eval_method/test_data.zarr", chunks={"pony_dim_0":512})
    #ds = xr.open_dataset('/ec/res4/scratch/sbjb/Projects/CERISE/ObsOpData/GNN/10GHz_static/Tuning_data/Graphs_20170802.h5', engine='netcdf4') 
    
    feat_lst = [
        "ZS",        
        "FRAC_LAND_AND_SEA_WATER", 
        "Distance_to_footprint_center",
        "TG1_ga",
        "TG2_ga",
        "WG1_ga",
        "WG2_ga",
        "WGI1_ga",
        "WGI2_ga",
        "TS_ISBA",        
        "LAI_ga",
        "HSN_VEG1_ga",
        "HSN_VEG6_ga",
        "HSN_VEG12_ga",
        "RSN_VEG1_ga",
        "RSN_VEG6_ga",
        "RSN_VEG12_ga",         
        "WSN_T_ISBA",
        "DSN_T_ISBA"        
    ]   

    targ_lst = [
        "AMSR2_BT18.7V"
    ]

    areadef, xx, yy = getSFXgrid()    
    mlons, mlats = areadef.get_lonlats()

    all_dates = pd.date_range(start="2017-06-02", end="2018-06-01", freq='D')

    # Filter out the first 10 days of each month
    filtered_dates = [date for date in all_dates if date.day <= 10] # Validation data    

    # Convert to list of strings for display
    list_dates = [date.strftime('%Y%m%d') for date in filtered_dates]

    counter = 0

    path_output = "/ec/res4/scratch/sbjb/Projects/CERISE/ObsOpData/XGB/18GHz/Predictions/v01/"

    for date_task in list_dates:        
        
        try:
            ds = xr.open_dataset('/ec/res4/scratch/sbjb/Projects/CERISE/ObsOpData/GNN/18GHz_static/Tuning_data_patch_fix/Graphs_18_7_%s.h5'%date_task, engine='netcdf4') 
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
        dd = ds.where(nan_count_x < 12, drop=True)

        ds_mean = dd.mean(dim="phony_dim_1")

        ds_non_nan_values = ds_mean.where(~np.isnan(ds_mean), drop=True)
        no_zs = ds_non_nan_values

        #print(np.shape(no_zs["DSN_T_ISBA"].values))
        #plt.figure()
        #plt.scatter(no_zs["xx"].values,no_zs["yy"].values, c=no_zs["DSN_T_ISBA"].values )
        #plt.colorbar()
        #plt.savefig("dsn_t_isba.png")        
        #plt.close()


        #no_zs = ds_non_nan_values.where(ds_non_nan_values['ZS'] > 00.0)
        #no_zs = ds_non_nan_values.where(ds_non_nan_values['LAI_ga'] < 3.0)
        #no_zs = no_zs.where(no_zs['WGI1_ga'] < 0.01)
        #no_zs = no_zs.where(no_zs['COVER006'] < 0.1)
        #no_zs = no_zs.where(no_zs['DSN_T_ISBA'] > 0.01)

        feats_arr = no_zs[feat_lst].to_array().astype("float32").values.transpose()    
    
        obs_arr = no_zs[targ_lst].to_array().astype("float32").values.transpose()

        print(np.shape(obs_arr))

        #xx = no_zs["xx"].values
        #yy = no_zs["yy"].values
    
        preds = xgb_model_loaded.predict(feats_arr)
    
        orig_def = pyresample.geometry.SwathDefinition(lons=no_zs["AMSR2_lon"].values, lats=no_zs["AMSR2_lat"].values)
        targ_def = areadef 
        TB = pyresample.kd_tree.resample_nearest(orig_def, preds, targ_def,  radius_of_influence=5000.0, fill_value=np.nan)   
        #TB = ma.masked_values(TB, -999)                  

        TB_ob = pyresample.kd_tree.resample_nearest(orig_def, obs_arr[:,0], targ_def,  radius_of_influence=5000.0, fill_value=np.nan)   
        #TB_ob = ma.masked_values(TB_ob, np.nan)                  

        # Write to file

        output_filename = path_output + "Predictions_" + date_task + ".nc"
        with nc.Dataset(str(output_filename), "w", format = "NETCDF4") as output_netcdf:
            x = output_netcdf.createDimension("x", 800)
            y = output_netcdf.createDimension("y", 1000)
            #
            pred = output_netcdf.createVariable("pred", "d", ("y","x"))
            pred.units = "Kelvin" 
            pred.standard_name = "projection_coordinates"
            pred[:,:] = TB[:,:]

            ob = output_netcdf.createVariable("obs", "d", ("y","x"))
            ob.units = "Kelvin" 
            ob.standard_name = "projection_coordinates"
            ob[:,:] = TB_ob[:,:]


        del(no_zs)
        del(ds)

#            Outputs = vars()
#            
#            Outputs[var] = output_netcdf.createVariable(var, "d", (var))
#                Outputs[var].units = "meters" 
#                Outputs[var].standard_name = "projection_" + var + "_coordinates"
#                Outputs[var] = np.copy(self.Surfex_coord[var])
#            #
#            for var in ["lat", "lon"]:
#                Outputs[var] = output_netcdf.createVariable(var, "d", ("y", "x"))
#                if var == "lat":
#                    Outputs[var].standard_name = "latitude"
#                    Outputs[var].unit = "degrees_north"
#                else:
#                    Outputs[var].standard_name = "longitude"
#                    Outputs[var].units = "degrees_east"
#                Outputs[var][:,:] = np.copy(self.Surfex_coord[var])
#            #
#            for var in Gridded_targets:
#                Outputs["Target_" + var] = output_netcdf.createVariable("Target_" + var, "d", ("y", "x"))
#                Outputs["Target_" + var].units = "Kelvins"
#                Outputs["Target_" + var].standard_name = "Brightness temperature"
#                Outputs["Target_" + var][:,:] = np.copy(Gridded_targets[var])
#            #
#            for var in Gridded_predictions:
#                Outputs["Prediction_" + var] = output_netcdf.createVariable("Prediction_" + var, "d", ("y", "x"))
#                Outputs["Prediction_" + var].units = "Kelvins"
#                Outputs["Prediction_" + var].standard_name = "Brightness temperature"
#                Outputs["Prediction_" + var][:,:] = np.copy(Gridded_predictions[var])
            #            


def main():

    #createDataBase()
    #xgmodel()
    predict()


if __name__ == "__main__":

    main()
