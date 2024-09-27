#!/usr/bin/env python
# coding: utf-8

# In[11]:


import os
import sys
import time
import h5py
import netCDF4
import datetime
import numpy as np
import tensorflow as tf
from numpy.lib.stride_tricks import as_strided
#
tf.keras.mixed_precision.set_global_policy("mixed_float16")
#print("GPUs available: ", tf.config.list_physical_devices('GPU'))
#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#print(physical_devices)
#tf.config.experimental.set_memory_growth(physical_devices[0], True)


# # Constants

# In[12]:


model_name = "CNN"
experiment = "v13_20_epochs"
stride = "1"
#
function_path = "/lustre/storeB/users/cyrilp/CERISE/Scripts/Patch_CNN/Models/" + model_name + "/"
sys.path.insert(0, function_path)
from CNN import *
#
date_min_test = "20230601"
date_max_test = "20230601"
hours_AMSR2 = "H03"
domain_size = (250, 250)
stride_prediction = 1
#
paths = {}
paths["base"] = "/lustre/storeB/project/nwp/H2O/wp3/Deep_learning_predictions/"
paths["training"] = paths["base"] + "Training_data/"
paths["model"] = paths["base"] + "Patches/Models/" + model_name + "/" + experiment + "/"
paths["normalization_stats"] = paths["base"] + "Normalization/"
paths["predictions_netCDF"] = paths["base"] + "Patches/Predictions/" + model_name + "/netCDF/" + experiment + "_stride_" + stride + "/"
paths["prediction_scores"] = paths["base"] + "Patches/Predictions/" + model_name + "/score/"
#
for var in paths:
    if os.path.isdir(paths[var]) == False:
        os.system("mkdir -p " + paths[var])
#
file_normalization = paths["normalization_stats"] + "Stats_normalization_20200901_20220531.h5"
file_model_weights = paths["model"] + model_name + ".h5"


# # Model parameters

# In[13]:


list_targets = ["tb18_v", "tb36_v"]
#
predictors = {}
predictors["constants"] = ["ZS", "PATCHP1", "PATCHP2", "FRAC_LAND_AND_SEA_WATER"]
predictors["ISBA"] = ["Q2M_ISBA", "DSN_T_ISBA", "LAI_ga", "TS_ISBA", "PSN_ISBA"]
predictors["TG"] = [1, 2]
predictors["WG"] = [1, 2]
predictors["WGI"] = [1, 2]
predictors["WSN_VEG"] = [1, 2]
predictors["RSN_VEG"] = [1, 2]
predictors["HSN_VEG"] = [1, 2]
predictors["SNOWTEMP"] = [1, 12]
predictors["SNOWLIQ"] = [1, 12]
#
list_predictors = predictors["constants"] + predictors["ISBA"]
for pred in predictors:
    if (pred != "constants") and (pred != "ISBA"):
        for lay in predictors[pred]:
            var_name = pred + str(lay) + "_ga"
            list_predictors = list_predictors + [var_name]
#
model_params = {"list_predictors": list_predictors,
                "list_targets": list_targets,
                "patch_dim": (5, 5),
                "batch_size": 512,
                "conv_filters": [32, 64, 128, 64],
                "dense_width": [32, 16],
                "activation": "relu",
                "kernel_initializer": "he_normal",
                "batch_norm": True,
                "pooling_type": "Average",
                "dropout": 0,
                }


# # Make list dates function

# In[14]:


def make_list_dates(date_min, date_max):
    current_date = datetime.datetime.strptime(date_min, "%Y%m%d")
    end_date = datetime.datetime.strptime(date_max, "%Y%m%d")
    list_dates = []
    while current_date <= end_date:
        date_str = current_date.strftime("%Y%m%d")
        filename = paths["training"] + date_str[0:4] + "/" + date_str[4:6] + "/" + "Dataset_" + date_str + hours_AMSR2 + ".nc"
        if os.path.isfile(filename):
            list_dates.append(date_str)
        current_date = current_date + datetime.timedelta(days = 1)
    return(list_dates)


# # Standardization data

# In[15]:


def load_normalization_stats(file_normalization):
    normalization_stats = {}
    hf = h5py.File(file_normalization, "r")
    for var in hf:
        normalization_stats[var] = hf[var][()]
    hf.close()
    return(normalization_stats)


# # Extract evaluation data

# In[16]:


def extract_eval_data(date_task, list_targets, hours_AMSR2, paths):
    Eval_data = {}
    filename_training = paths["training"] + date_task[0:4] + "/" + date_task[4:6] + "/" + "Dataset_" + date_task + hours_AMSR2 + ".nc"
    nc = netCDF4.Dataset(filename_training)
    for var in list_targets:
        var_data = nc.variables[var][:,:]
        if np.shape(var_data.mask) == np.shape(var_data):
            var_data[var_data.mask == True] = np.nan
        Eval_data[var] = np.copy(var_data)
    #
    Eval_data["FRAC_SEA"] = nc.variables["FRAC_SEA"][:,:]
    Eval_data["FRAC_WATER"] = nc.variables["FRAC_WATER"][:,:]
    nc.close()
    return(Eval_data)


# # Make predictions function

# In[17]:


class make_predictions():
    def __init__(self, date_task, model, model_params, normalization_stats, paths, hours_AMSR2, domain_size, stride_prediction):
        self.date_task = date_task
        self.model = model
        self.model_params = model_params
        self.normalization_stats = normalization_stats
        self.paths = paths
        self.hours_AMSR2 = hours_AMSR2
        self.domain_size = domain_size
        self.stride_prediction = stride_prediction
        self.patch_dim = model_params["patch_dim"][0]
    #
    def normalize(self, var, var_data):
        norm_var_data = (var_data - self.normalization_stats[var + "_min"]) / (self.normalization_stats[var + "_max"] - self.normalization_stats[var + "_min"])
        return(norm_var_data)
    #
    def unnormalize(self, var, norm_var_data):
        unnorm_var = norm_var_data * (self.normalization_stats[var + "_max"] - self.normalization_stats[var + "_min"]) + self.normalization_stats[var + "_min"]
        return(unnorm_var)
    #
    def load_data_full_grid(self):
        X = np.full((1, *self.domain_size, len(self.model_params["list_predictors"])), np.nan)
        filename = self.paths["training"] + self.date_task[0:4] + "/" + self.date_task[4:6] + "/" + "Dataset_" + self.date_task + self.hours_AMSR2 + ".nc"
        nc = netCDF4.Dataset(filename, "r")
        for v, var in enumerate(self.model_params["list_predictors"]):
            var_data = nc.variables[var][:,:]
            if np.shape(var_data.mask) == np.shape(var_data):
                var_data[var_data.mask == True] = self.normalization_stats[var + "_min"]
            var_data[np.isnan(var_data) == True] = self.normalization_stats[var + "_min"]
            X[0,:,:,v] = self.normalize(var, var_data)
        nc.close()
        return(X)
    #
    def make_patches(self, dataset):
        # dataset should have the following shape: (n_samples (1), y_dim, x_dim, n_variables)
        n_samples, y_dim, x_dim, num_vars = np.shape(dataset)
        # Calculate the shape of the new array
        new_shape = (1,
                    (y_dim - self.patch_dim) // self.stride_prediction + 1,
                    (x_dim - self.patch_dim) // self.stride_prediction + 1,
                    self.patch_dim,
                    self.patch_dim,
                    num_vars)
        # Calculate the strides of the new array
        new_strides = (dataset.strides[0],
                       dataset.strides[1] * self.stride_prediction,
                       dataset.strides[2] * self.stride_prediction,
                       dataset.strides[1],
                       dataset.strides[2],
                       dataset.strides[3])
        # Use as_strided to create the new array
        patches = np.squeeze(as_strided(dataset, shape = new_shape, strides = new_strides), axis = 0)
        patches = patches.reshape(-1, self.patch_dim, self.patch_dim, num_vars)
        return(patches)
    #
    def grid_patch_prediction(self, patch_predictions):
        idx = 0
        sum_pred = np.zeros(self.domain_size)
        N_pred = np.zeros(self.domain_size)
        #
        for i in range(self.domain_size[0] - self.patch_dim + 1):
            for j in range(self.domain_size[1] - self.patch_dim + 1):
                for di in range(self.patch_dim):
                    for dj in range(self.patch_dim):
                        sum_pred[i + di, j + dj] = sum_pred[i + di, j + dj] + patch_predictions[idx]
                        N_pred[i + di, j + dj] = N_pred[i + di, j + dj] + 1
                idx = idx + 1
        N_pred[N_pred == 0] = 1
        Pred = sum_pred / N_pred
        return(Pred)
    #
    def make_prediction(self):
        X_full_grid = self.load_data_full_grid()
        X_patches = self.make_patches(X_full_grid)
        #
        model_predictions = np.squeeze(self.model.predict(X_patches))
        Pred = {}
        for v, var in enumerate(self.model_params["list_targets"]):
            if len(self.model_params["list_targets"]) == 1:
                var_data = self.unnormalize(var, model_predictions[:])
            else:
                var_data = self.unnormalize(var, model_predictions[:,v])
            #
            Pred[var] = self.grid_patch_prediction(var_data)
        #
        return(Pred)
    #
    def __call__(self):
        Predictions = self.make_prediction()
        return(Predictions)


# # Save predictions in netCDF files

# In[18]:


def save_predictions_in_netCDF(date_task, Pred_data, Eval_data, list_targets, paths, stride_prediction):
    file_output = paths["predictions_netCDF"] + "Predictions_" + date_task + ".nc"
    output_netcdf = netCDF4.Dataset(file_output, "w", format = "NETCDF4")
    Outputs = vars()
    #
    dimensions = ["x", "y"]
    for di in dimensions:
        Outputs[di] = output_netcdf.createDimension(di, 250)
    #
    for var in list_targets:
        Outputs["target_" + var] = output_netcdf.createVariable("target_" + var, "d", ("y", "x"))
        Outputs["target_" + var][:,:] = Eval_data[var]
    #
    for var in Pred_data:
        Outputs["pred_" + var] = output_netcdf.createVariable("pred_" + var, "d", ("y", "x"))
        Outputs["pred_" + var][:,:] = Pred_data[var]
    #
    output_netcdf.close()


# # Verification

# In[19]:


class verification():
    def __init__(self, date_task, date_min, date_max, Pred_data, Eval_data, paths, experiment, stride):
        self.date_task = date_task
        self.date_min = date_min
        self.date_max = date_max
        self.Pred_data = Pred_data
        self.Eval_data = Eval_data
        self.paths = paths
        self.experiment = experiment
        self.stride = stride
    #
    def Calculate_scores(self):
        Scores = {}
        for var in Pred_data:
            Scores["MAE_" + var] = np.nanmean(np.abs(self.Pred_data[var] - self.Eval_data[var]))
            Scores["RMSE_" + var] = np.sqrt(np.nanmean(np.square(self.Pred_data[var] - self.Eval_data[var])))
            #
            idx_land = self.Eval_data["FRAC_SEA"] == 0
            idx_land_excl_lakes = (self.Eval_data["FRAC_SEA"] + self.Eval_data["FRAC_WATER"] < 0.5)
            Scores["Land_MAE_" + var] = np.nanmean(np.abs(self.Pred_data[var][idx_land == True] - self.Eval_data[var][idx_land == True]))
            Scores["Land_RMSE_" + var] = np.sqrt(np.nanmean(np.square(self.Pred_data[var][idx_land == True] - self.Eval_data[var][idx_land == True])))
            Scores["Land_excl_lakes_MAE_" + var] = np.nanmean(np.abs(self.Pred_data[var][idx_land_excl_lakes == True] - self.Eval_data[var][idx_land_excl_lakes == True]))
            Scores["Land_excl_lakes_RMSE_" + var] = np.sqrt(np.nanmean(np.square(self.Pred_data[var][idx_land_excl_lakes == True] - self.Eval_data[var][idx_land_excl_lakes == True])))
        return(Scores)
    #
    def save_scores(self, Scores):
        output_file = self.paths["prediction_scores"] + "Scores_" + self.date_min + "_" + self.date_max + "_" + self.experiment + "_stride_" + self.stride + ".txt"
        #
        if self.date_task == self.date_min:
            if os.path.isfile(output_file) == True:
                os.system("rm " + output_file)
            #
            header = "date"
            for var in Scores:
                header = header + "\t" + var
            #
            output = open(output_file, "a")
            output.write(header + "\n")
            output.close()
        #
        scores_str = self.date_task
        for var in Scores:
            scores_str = scores_str + "\t" + str(Scores[var])
        #
        output = open(output_file, "a")
        output.write(scores_str + "\n")
        output.close()
    #
    def __call__(self):
        Scores = self.Calculate_scores()
        self.save_scores(Scores)


# # Data processing

# In[20]:


t0 = time.time()
#
list_dates = make_list_dates(date_min_test, date_max_test)
normalization_stats = load_normalization_stats(file_normalization)
model = CNN(**model_params).make_model()
model.load_weights(file_model_weights)
for date_task in list_dates:
    print(date_task)
    Eval_data = extract_eval_data(date_task, list_targets, hours_AMSR2, paths)
    Pred_data = make_predictions(date_task, model, model_params, normalization_stats, paths, hours_AMSR2, domain_size, stride_prediction)()
    save_predictions_in_netCDF(date_task, Pred_data, Eval_data, list_targets, paths, stride_prediction)
    #verification(date_task, date_min_test, date_max_test, Pred_data, Eval_data, paths, experiment, stride)()
#
tf = time.time()
print("Computing time: ", tf - t0)

