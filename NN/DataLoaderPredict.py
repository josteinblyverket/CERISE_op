
import xarray as xr
import torch
import numpy as np


#def loadToPredict(feat_lst, targ_lst, no_zs, dx):
def loadToPredict(feat_lst, targ_lst, no_zs):
    
    feats_arr = no_zs[feat_lst].to_array().astype("float32").values.transpose()    
    obs_arr = no_zs[targ_lst].to_array().astype("float32").values.transpose()

    # Pre-process data

    X = feats_arr
    y = obs_arr
        
    X_train_np = X[:,:]
    y_train_np = y[:,:]    

    print(np.min(X_train_np))
    print(np.max(X_train_np))

    print(np.min(y_train_np))
    print(np.max(y_train_np))
    print("Shape")
    print(np.shape(X_train_np))   

    X_pred = torch.from_numpy(X_train_np)
    y_target = torch.from_numpy(y_train_np)

    Xnan = torch.nan_to_num(X_pred, nan=0.0)
    ynan = torch.nan_to_num(y_target, nan=0.0)

    #Xnan[:,5] += dx

    loaded_weights = torch.load("/ec/res4/scratch/sbjb/Projects/CERISE/ObsOpData/NN/18GHz/Zarr/norm_weights_n.pth")

    # Extract mean and std
    xmean = loaded_weights["xmean"]
    xstd = loaded_weights["xstd"]
    ymean = loaded_weights["ymean"]
    ystd = loaded_weights["ystd"]

    #print("First guess values")
    #non_zero_indices = np.nonzero(Xnan[:,3])[0]   

    #print(Xnan[non_zero_indices[0],:])
    
    #Xnorm = (Xnan[non_zero_indices[0],:] - xmean)/(xstd)
    #Ynorm = (ynan[non_zero_indices[0],:] - ymean)/(ystd)    

    Xnorm = (Xnan - xmean)/(xstd)
    Ynorm = (ynan - ymean)/(ystd)        

    return Xnorm, Ynorm, no_zs