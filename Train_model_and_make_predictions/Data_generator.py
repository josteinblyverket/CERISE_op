#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import netCDF4
import tensorflow as tf
import numpy as np
tf.keras.utils.set_random_seed(1234)


# In[ ]:


class Data_generator(tf.keras.utils.Sequence):
    def __init__(self, list_predictors, list_targets, list_IDs, normalization_stats, batch_size, path_data, dim, shuffle):
        self.list_predictors = list_predictors
        self.list_targets = list_targets
        self.list_IDs = list_IDs
        self.normalization_stats = normalization_stats
        self.batch_size = batch_size
        self.path_data = path_data
        self.dim = dim
        self.shuffle = shuffle
        self.n_predictors = len(list_predictors)
        self.n_targets = len(list_targets)
        self.on_epoch_end()
    #
    def __len__(self): # Number of batches per epoch
        return int(np.ceil(len(self.list_IDs)) / self.batch_size)  
    #
    def __getitem__(self, index): # Generate one batch of data
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        # Find list of IDs
        list_IDs_batch = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_batch)
        return(X, y)
    #
    def on_epoch_end(self): # Updates indexes after each epoch
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            rng = np.random.default_rng()
            rng.shuffle(self.indexes)
    #
    def normalize(self, var, var_data):
        norm_data = (var_data - self.normalization_stats[var + "_min"]) / (self.normalization_stats[var + "_max"] - self.normalization_stats[var + "_min"])
        return(norm_data)
    #
    def __data_generation(self, list_IDs_batch): # Generates data containing batch_size samples
        #
        # Initialization
        X = np.full((self.batch_size, *self.dim, self.n_predictors), np.nan)
        y = np.full((self.batch_size, self.n_targets), np.nan)
        # Generate data
        file_ID = self.path_data + "Patches.nc"
        nc = netCDF4.Dataset(file_ID, "r")
        #
        for v, var in enumerate(self.list_predictors):
            var_data = nc.variables[var][list_IDs_batch,:,:]
            if np.shape(var_data.mask) == np.shape(var_data):
                var_data[var_data.mask == True] = self.normalization_stats[var + "_min"]
            var_data[np.isnan(var_data) == True] = self.normalization_stats[var + "_min"]
            X[:,:,:,v] = self.normalize(var, var_data)
        #
        for v, var in enumerate(self.list_targets):
            var_data = nc.variables[var][list_IDs_batch,:,:]
            if np.shape(var_data.mask) == np.shape(var_data):
                var_data[var_data.mask == True] = np.nan
            var_data = np.nanmean(var_data, axis = (1, 2))
            y[:,v] = self.normalize(var, var_data)
        #
        nc.close()
        return(X, y)

