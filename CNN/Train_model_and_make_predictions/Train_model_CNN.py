#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import json
import time
import h5py
import pickle
import netCDF4
import datetime
import numpy as np
import tensorflow as tf
tf.keras.mixed_precision.set_global_policy("mixed_float16")
print("GPUs available: ", tf.config.list_physical_devices('GPU'))
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0], True)
#
t0 = time.time()


# # Constants

# In[ ]:


model_name = "CNN"
experiment = "v15_20_epochs_without_snow_min"
function_path = "/lustre/storeB/users/cyrilp/CERISE/Scripts/Patch_CNN/Models/" + model_name + "/"
sys.path.insert(0, function_path)
from Data_generator import *
from CNN import *
#
Number_of_samples_training = 1034319
Number_of_samples_validation = 441461
#
paths = {}
paths["base"] = "/lustre/storeB/project/nwp/H2O/wp3/Deep_learning_predictions/"
paths["training"] = paths["base"] + "Patches/Training_patches/"
paths["validation"] = paths["base"] + "Patches/Validation_patches/" 
paths["normalization_stats"] = paths["base"] + "Normalization/"
paths["checkpoints"] = paths["base"] + "Patches/Models/" + model_name + "/" + experiment +"/Checkpoints/"
paths["output"] = paths["base"] + "Patches/Models/" + model_name + "/" + experiment + "/"
#
for var in paths:
    if os.path.isdir(paths[var]) == False:
        os.system("mkdir -p " + paths[var])
#
file_normalization = paths["normalization_stats"] + "Stats_normalization_20200901_20220531.h5"
file_checkpoints = paths["checkpoints"] + "Checkpoints.h5"
if os.path.isfile(file_checkpoints) == True:
    os.system("rm " + file_checkpoints)


# # Model parameters

# In[ ]:


list_targets = ["tb18_v", "tb36_v"]
#
predictors = {}
predictors["constants"] = ["ZS", "PATCHP1", "PATCHP2", "FRAC_LAND_AND_SEA_WATER"]
predictors["ISBA"] = ["Q2M_ISBA", "DSN_T_ISBA", "LAI_ga", "TS_ISBA", "PSN_ISBA"]
predictors["TG"] = [1, 2]
predictors["WG"] = [1, 2]
predictors["WGI"] = [1, 2]
predictors["WSN_VEG"] = [1, 2]
#predictors["RSN_VEG"] = [1, 2]
#predictors["HSN_VEG"] = [1, 2]
#predictors["SNOWTEMP"] = [1, 12]
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
#
compile_params = {"initial_learning_rate": 0.005, 
                  "decay_steps": 6630,
                  "decay_rate": 0.5,
                  "staircase": True,
                  "n_epochs": 20,
                  }
#
model_and_compile_params = {**model_params, **compile_params}


# # Save model parameters

# In[1]:


def save_model_parameters(model_and_compile_params, model_history, paths = paths):
    file_model_parameters = paths["output"] + "Model_parameters.txt"
    file_model_training_history = paths["output"] + "Training_history.pkl"
    #
    if os.path.isfile(file_model_parameters) == True:
        os.system("rm " + file_model_parameters)
    if os.path.isfile(file_model_training_history) == True:
        os.system("rm " + file_model_training_history)
    #
    pickle.dump(model_history.history, open(file_model_training_history, "wb"))
    with open(file_model_parameters, "w") as output_file:
        output_file.write(json.dumps(model_and_compile_params))


# # Standardization data

# In[ ]:


def load_normalization_stats(file_normalization):
    normalization_stats = {}
    hf = h5py.File(file_normalization, "r")
    for var in hf:
        normalization_stats[var] = hf[var][()]
    hf.close()
    return(normalization_stats)


# # Loss function and metrics

# In[ ]:


def MSE_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    MSE = tf.reduce_mean(tf.square(y_true - y_pred))
    return(MSE)

def RMSE(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    RMSE_score = tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))
    return(RMSE_score)


# # Data generator

# In[ ]:


normalization_stats = load_normalization_stats(file_normalization)
list_train_IDs = np.arange(Number_of_samples_training) 
list_valid_IDs = np.arange(Number_of_samples_validation) 
#
params_train = {"list_predictors": model_params["list_predictors"],
                "list_targets": model_params["list_targets"],
                "list_IDs": list_train_IDs,
                "normalization_stats": normalization_stats,
                "batch_size": model_params["batch_size"],
                "path_data": paths["training"],
                "dim": model_params["patch_dim"],
                "shuffle": False,
                }
#
params_valid = {"list_predictors": model_params["list_predictors"],
                "list_targets": model_params["list_targets"],
                "list_IDs": list_valid_IDs,
                "normalization_stats": normalization_stats,
                "batch_size": model_params["batch_size"],
                "path_data": paths["validation"],
                "dim": model_params["patch_dim"],
                "shuffle": False,
                }
#
train_generator = Data_generator(**params_train)
valid_generator = Data_generator(**params_valid)


# # Data processing

# In[ ]:


lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate = compile_params["initial_learning_rate"],
    decay_steps = compile_params["decay_steps"],
    decay_rate = compile_params["decay_rate"],
    staircase = compile_params["staircase"])
#
opt = tf.keras.optimizers.Adam(learning_rate = lr_schedule)
loss = MSE_loss
metrics = RMSE
#
model = CNN(**model_params).make_model()
print(type(model))
print(model.summary())
model.compile(loss = loss, metrics = metrics, optimizer = opt)
print("Model compiled")
#
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath = file_checkpoints, save_weights_only = True, monitor = 'val_loss', mode = 'min', verbose = 2, save_best_only = True)
#
model_history = model.fit(train_generator, validation_data = valid_generator, epochs = compile_params["n_epochs"], verbose = 2, callbacks = [checkpoint])
print("Model fitted")
#
filename_model = model_name + ".h5"
model.save_weights(paths["output"] + filename_model)
#
save_model_parameters(model_and_compile_params, model_history, paths = paths)
#
t1 = time.time()
dt = t1 - t0
print("Computing time: " + str(dt) + " seconds")

