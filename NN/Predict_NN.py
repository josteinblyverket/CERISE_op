
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
import torch
from NeuralNets import NeuralNet
from util import getSFXgrid
import pyresample
from DataLoaderPredict import loadToPredict
import netCDF4 as nc
import pandas as pd

def prediction(X_pred, y_target, ds):

    model_path = "/ec/res4/scratch/sbjb/Projects/CERISE/ObsOpData/NN/18GHz/Models/v01/mlp_sensitivity_test.pth"

    loaded_weights = torch.load("/ec/res4/scratch/sbjb/Projects/CERISE/ObsOpData/NN/18GHz/Zarr/norm_weights_n.pth")

    # Extract mean and std
    xmean = loaded_weights["xmean"]
    xstd = loaded_weights["xstd"]
    ymean = loaded_weights["ymean"]
    ystd = loaded_weights["ystd"]    

    #model = SimpleNet()
    model = NeuralNet()
    model.load_state_dict(torch.load(model_path))

    model.eval()

    with torch.no_grad():
        # Forward pass to get predictions
        predictions = model(X_pred)
        

    print("Predictions:", predictions)
    unorm_predictions = (predictions * ystd) + ymean
    unorm_obs = (y_target * ystd) + ymean

    numpy_data = unorm_predictions.numpy()
    numpy_obs = unorm_obs.numpy()
    
#    fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns
#
#    # Plot on the first subplot
#    im1 = axes[0].scatter(ds["xx"].values, ds["yy"].values, c=numpy_data)
#    axes[0].set_title('MLP')    
#    axes[0].legend()
#
#    # Plot on the second subplot
#    im2 = axes[1].scatter(ds["xx"].values, ds["yy"].values, c=y_target)
#    axes[1].set_title('Obs')    
#    axes[1].legend()    
#
#    # Adjust layout and display the plot
#    fig.colorbar(im1, ax=axes[1], orientation='vertical')
#
#    plt.tight_layout()
#    plt.savefig("myfig3.png")
#    plt.close()

    return numpy_data, numpy_obs


def runPredictions(feat_lst, targ_lst):

    all_dates = pd.date_range(start="2018-06-02", end="2018-10-01", freq='D')

    # Filter out the first 10 days of each month
    #filtered_dates = [date for date in all_dates if date.day <= 10] # Validation data    
    filtered_dates = [date for date in all_dates] # Test data

    # Convert to list of strings for display
    list_dates = [date.strftime('%Y%m%d') for date in filtered_dates]

    counter = 0

    path_output = "/ec/res4/scratch/sbjb/Projects/CERISE/ObsOpData/NN/18GHz/Predictions/v03/"
    #ds = xr.open_dataset('/ec/res4/scratch/sbjb/Projects/CERISE/ObsOpData/GNN/10GHz_static/Training_patch_fix/old/Graphs_20170703.h5', engine='netcdf4') 
    areadef, xx, yy = getSFXgrid()    
       
    targ_def = areadef 

    for date_task in list_dates:
        
        try:
            ds = xr.open_dataset('/ec/res4/scratch/sbjb/Projects/CERISE/ObsOpData/GNN/18GHz_static/Test_data/Graphs_18_7_%s.h5'%date_task, engine='netcdf4') 
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
        dd = ds.where(nan_count_x < 12, drop=True) # 40, 12 and 1

        ds_mean = dd.mean(dim="phony_dim_1")

        ds_non_nan_values = ds_mean.where(~np.isnan(ds_mean), drop=True)
        no_zs = ds_non_nan_values
        
        x_pred, y_target, ds = loadToPredict(feat_lst, targ_lst, no_zs)

        pred, targ = prediction(x_pred, y_target, no_zs)

        orig_def = pyresample.geometry.SwathDefinition(lons=no_zs["AMSR2_lon"].values, lats=no_zs["AMSR2_lat"].values)

        print(np.shape(pred))
        print(np.shape(targ))

        TBV = pyresample.kd_tree.resample_nearest(orig_def, pred[:,0], targ_def,  radius_of_influence=5000.0, fill_value=np.nan)       
        TB_obV = pyresample.kd_tree.resample_nearest(orig_def, targ[:,0], targ_def,  radius_of_influence=5000.0, fill_value=np.nan)   
        #TBV = pyresample.kd_tree.resample_nearest(orig_def, pred[:], targ_def,  radius_of_influence=5000.0, fill_value=np.nan)       
        #TB_obV = pyresample.kd_tree.resample_nearest(orig_def, targ[:], targ_def,  radius_of_influence=5000.0, fill_value=np.nan)   

        #TBH = pyresample.kd_tree.resample_nearest(orig_def, pred[:,1], targ_def,  radius_of_influence=5000.0, fill_value=np.nan)       
        #TB_obH = pyresample.kd_tree.resample_nearest(orig_def, targ[:,1], targ_def,  radius_of_influence=5000.0, fill_value=np.nan)   

        output_filename = path_output + "Predictions_" + date_task + ".nc"
        with nc.Dataset(str(output_filename), "w", format = "NETCDF4") as output_netcdf:
            x = output_netcdf.createDimension("x", 800)
            y = output_netcdf.createDimension("y", 1000)
            #
            pred = output_netcdf.createVariable("predV", "d", ("y","x"))
            pred.units = "Kelvin" 
            pred.standard_name = "projection_coordinates"
            pred[:,:] = TBV[:,:]

            ob = output_netcdf.createVariable("obsV", "d", ("y","x"))
            ob.units = "Kelvin" 
            ob.standard_name = "projection_coordinates"
            ob[:,:] = TB_obV[:,:]

            #predH = output_netcdf.createVariable("predH", "d", ("y","x"))
            #predH.units = "Kelvin" 
            #predH.standard_name = "projection_coordinates"
            #predH[:,:] = TBH[:,:]
#
            #obH = output_netcdf.createVariable("obsH", "d", ("y","x"))
            #obH.units = "Kelvin" 
            #obH.standard_name = "projection_coordinates"
            #obH[:,:] = TB_obH[:,:]

        del(no_zs)
        del(ds)

def main():

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
        #"AMSR2_BT18.7H"
    ]
    
    runPredictions(feat_lst, targ_lst)   
    

if __name__ == "__main__":
    
    main()