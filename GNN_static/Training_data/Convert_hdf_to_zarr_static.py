#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import h5py
import zarr
import random
import numpy as np


# # Constants

# In[2]:


AMSR2_frequency = "18GHz"
chunk_size = 512
subsampling_output = 1
date_min = "20220901"
date_max = "20230531"
#
paths = {}
paths["data"] = "/lustre/storeB/project/nwp/H2O/wp3/Deep_learning_predictions/Training_data_GNN/" + AMSR2_frequency + "_static/" 
#
hdf_filename = paths["data"] + "Graphs_" + date_min + "_" + date_max + "_subsampling_0.h5"


# # Convert hdf to zarr

# In[3]:


class ConvertHDFToZarr:
    def __init__(self, hdf_filename, date_min, date_max, chunk_size, subsampling_output, paths):
        self.hdf_filename = hdf_filename
        self.date_min = date_min
        self.date_max = date_max
        self.chunk_size = chunk_size
        self.subsampling_output = subsampling_output
        self.paths = paths
        self.zarr_filename = self.paths["data"] + "Graphs_" + date_min + "_" + date_max + "_subsampling_" + str(self.subsampling_output) + ".zarr"

    def get_list_IDs(self):
        with h5py.File(self.hdf_filename, "r") as hdf:
            N_samples = len(hdf["AMSR2_xx"][()])
            list_IDs = np.arange(0, N_samples, self.subsampling_output)
        shuffle_list_IDs = random.sample(list(list_IDs), len(list_IDs))
        return shuffle_list_IDs

    def convert_to_zarr_dataset(self):
        shuffle_list_IDs = self.get_list_IDs()
        with h5py.File(self.hdf_filename, "r") as hdf_file:
            zarr_root = zarr.open(self.zarr_filename, mode = "w")

            for var, var_data in hdf_file.items():
                print(var)
                subsampled_shuffled_data = np.take(var_data, shuffle_list_IDs, axis = 0)
                zarr_root.create_dataset(var, 
                                         data = subsampled_shuffled_data, 
                                         shape = subsampled_shuffled_data.shape, 
                                         dtype = subsampled_shuffled_data.dtype,
                                         chunks = self.chunk_size)
            
    def __call__(self):
        self.convert_to_zarr_dataset()


# # Data processing

# In[4]:


t0 = time.time()
ConvertHDFToZarr(hdf_filename = hdf_filename, date_min = date_min, date_max = date_max, chunk_size = chunk_size, subsampling_output = subsampling_output, paths = paths)()
tf = time.time()
print("Computing time: ", tf - t0)
