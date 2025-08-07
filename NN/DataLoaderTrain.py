import xarray as xr
import torch
import numpy as np

def loadDataSubSample(feat_lst, targ_lst):

    # Open up the Zarr data
    ds1 = xr.open_zarr("/ec/res4/scratch/sbjb/Projects/CERISE/ObsOpData/NN/18GHz/Zarr/train_18GHz_data.zarr")
    ds_val = xr.open_zarr("/ec/res4/scratch/sbjb/Projects/CERISE/ObsOpData/NN/18GHz/Zarr/val_18GHz_data.zarr")

    # Inspect the concatenated dataset and see available model variables
    print(ds1)

    #ds = ds1.dropna(dim='N_DIM', how='all').dropna(dim='time', how='all')
       
    feats_ds = ds1[feat_lst].to_array().astype("float32").transpose()
    target_ds0 = ds1[targ_lst].to_array().astype("float32").transpose()

    feats_ds_val = ds_val[feat_lst].to_array().astype("float32").transpose()
    target_ds0_val = ds_val[targ_lst].to_array().astype("float32").transpose()

    # Pre-process data
   
    X = feats_ds.values
    y = target_ds0.values    

    Xval = feats_ds_val.values
    yval = target_ds0_val.values    
    
    print(np.shape(X))
    print(np.shape(y))    
    
    #idx = (y[:,0] < 350.0) & (y[:,0] > 150.0) & (X[:,-2] < 1.0) & (X[:,-2] >= 0.0)
    #idn = np.where(idx)[0][:]    

    X_train_np = X#[idn,:]
    y_train_np = y#[idn,:]     

    X_val_np = Xval
    y_val_np = yval

#    print(np.min(X_train_np))
#    print(np.max(X_train_np))
#
#    print(np.min(y_train_np))
#    print(np.max(y_train_np))   

    X_train = torch.from_numpy(X_train_np)
    y_train = torch.from_numpy(y_train_np)

    X_val = torch.from_numpy(X_val_np)
    y_val = torch.from_numpy(y_val_np)
    
    # Data that does not have 0 as min should have a different value for nans
    Xnan = torch.nan_to_num(X_train, nan=0.0)
    ynan = torch.nan_to_num(y_train, nan=0.0)

    Xnanval = torch.nan_to_num(X_val, nan=0.0)
    ynanval = torch.nan_to_num(y_val, nan=0.0)

    xmean = Xnan.mean(dim=0,keepdim=True)     
    ymean = ynan.mean(dim=0,keepdim=True)

    xstd = Xnan.mean(dim=0,keepdim=True)
    ystd = ynan.mean(dim=0,keepdim=True)

    norm_weights = {
        "xmean": xmean,
        "ymean": ymean,
        "xstd": xstd,
        "ystd": ystd,
    }

    torch.save(norm_weights, "/ec/res4/scratch/sbjb/Projects/CERISE/ObsOpData/NN/18GHz/Zarr/norm_weights_n_dual_pol.pth")

    Xnorm = (Xnan - xmean)/(xstd)
    Ynorm = (ynan - ymean)/(ystd)   

    Xnorm_val = (Xnanval - xmean)/(xstd)
    Ynorm_val = (ynanval - ymean)/(ystd)   

    return Xnorm, Ynorm, Xnorm_val, Ynorm_val


def loadData(feat_lst, targ_lst):

    # Open up the Zarr data
    ds1 = xr.open_zarr("/ec/res4/scratch/sbjb/Projects/CERISE/ObsOpData/NN/18GHz/Zarr/train_18GHz_data.zarr")

    # Inspect the concatenated dataset and see available model variables
    print(ds1)

    #ds = ds1.dropna(dim='N_DIM', how='all').dropna(dim='time', how='all')
       
    feats_ds = ds1[feat_lst].to_array().astype("float32").transpose()
    target_ds0 = ds1[targ_lst].to_array().astype("float32").transpose()

    # Pre-process data
   
    X = feats_ds.values
    y = target_ds0.values    
    
#    print(np.shape(X))
#    print(np.shape(y))    
    
    #idx = (y[:,0] < 350.0) & (y[:,0] > 150.0) & (X[:,-2] < 1.0) & (X[:,-2] >= 0.0)
    #idn = np.where(idx)[0][:]    

    X_train_np = X#[idn,:]
    y_train_np = y#[idn,:]     

#    print(np.min(X_train_np))
#    print(np.max(X_train_np))
#
#    print(np.min(y_train_np))
#    print(np.max(y_train_np))   

    X_train = torch.from_numpy(X_train_np)
    y_train = torch.from_numpy(y_train_np)
    
    Xnan = torch.nan_to_num(X_train, nan=0.0)
    ynan = torch.nan_to_num(y_train, nan=0.0)

    xmean = Xnan.mean(dim=0,keepdim=True)     
    ymean = ynan.mean(dim=0,keepdim=True)

    xstd = Xnan.mean(dim=0,keepdim=True)
    ystd = ynan.mean(dim=0,keepdim=True)


    norm_weights = {
        "xmean": xmean,
        "ymean": ymean,
        "xstd": xstd,
        "ystd": ystd,
    }

    torch.save(norm_weights, "/ec/res4/scratch/sbjb/Projects/CERISE/ObsOpData/NN/18GHz/norm_weights.pth")

    Xnorm = (Xnan - xmean)/(xstd)
    Ynorm = (ynan - ymean)/(ystd)    

    return Xnorm, Ynorm
