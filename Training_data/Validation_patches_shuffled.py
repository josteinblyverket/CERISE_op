#!/usr/bin/env python
# coding: utf-8

# In[235]:


import time
import os
import random
import netCDF4
import datetime
import numpy as np


# # Constants

# In[236]:


paths = {}
paths["data"] = "/lustre/storeB/project/nwp/H2O/wp3/Deep_learning_predictions/Patches/Training_data_7x7_patches/"
paths["output"] = "/lustre/storeB/project/nwp/H2O/wp3/Deep_learning_predictions/Patches/Validation_patches_7x7/"
#
date_min = "20220901"
date_max = "20230601"
#
target_var_cleaning = "tb18_v"
patch_size = 7


# # Create Patch file

# In[239]:


class create_patch_file():
    def __init__(self, date_min, date_max, paths):
        self.date_min = date_min
        self.date_max = date_max
        self.paths = paths
    #
    def make_list_dates(self):
        current_date = datetime.datetime.strptime(self.date_min, "%Y%m%d")
        end_date = datetime.datetime.strptime(self.date_max, "%Y%m%d")
        list_dates = []
        while current_date <= end_date:
            date_str = current_date.strftime("%Y%m%d")
            filename = self.paths["data"] + date_str[0:4] + "/" + date_str[4:6] + "/Dataset_" + date_str +  ".nc"
            if os.path.isfile(filename) == True:
                list_dates.append(date_str)
            current_date = current_date + datetime.timedelta(days = 1)
        return(list_dates)
    #
    def read_data(self, date_task):
        Dataset = {}
        filename = self.paths["data"] + date_task[0:4] + "/" + date_task[4:6] + "/" + "Dataset_" + date_task + ".nc"
        nc = netCDF4.Dataset(filename, "r")
        for var in nc.variables:
            Dataset[var] = nc.variables[var][:,:]
        nc.close()
        return(Dataset)
    #
    def concatenate_patches(self):
        list_dates = self.make_list_dates()
        for di, date_task in enumerate(list_dates):
            Dataset = self.read_data(date_task)
            #
            if di == 0:
                Patches = Dataset.copy()
            else:
                for var in Dataset:
                    Patches[var] = np.concatenate((Patches[var], Dataset[var]), axis = 0)
        return(Patches)        
    #
    def __call__(self):
        Patches = self.concatenate_patches()
        for v, var in enumerate(Patches):
            if v == 0:
                Number_of_samples = np.shape(Patches[var][:,:,:])[0]
        return(Patches, Number_of_samples)


# # Shuffle and clean dataset

# In[ ]:


class shuffle_and_clean_dataset():
    def __init__(self, Patches, target_var_cleaning):
        self.Patches = Patches
        self.target_var_cleaning = target_var_cleaning
    #
    def clean_data(self):
        Clean_dataset = {}
        idx_nan = np.isnan(np.nanmean(self.Patches[self.target_var_cleaning][:,:,:], axis = (1,2)))
        Number_of_valid_samples = np.sum(idx_nan == False)
        for var in self.Patches:
            Clean_dataset[var] = self.Patches[var][idx_nan == False,:,:]
        return(Clean_dataset, Number_of_valid_samples)
    #
    def shuffle_patches(self, Dataset, Number_of_valid_samples):
        Shuffled_dataset = {}
        list_IDs_shuffled = random.sample(range(0, Number_of_valid_samples), Number_of_valid_samples)
        for var in Dataset:
            Shuffled_dataset[var] = Dataset[var][list_IDs_shuffled,:,:]
        return(Shuffled_dataset)
    #
    def __call__(self):
        Clean_dataset, Number_of_valid_samples = self.clean_data()
        Shuffled_dataset = self.shuffle_patches(Clean_dataset, Number_of_valid_samples)
        return(Shuffled_dataset, Number_of_valid_samples)


# # Write netCDF

# In[ ]:


def write_netCDF(Patches, Number_of_samples, paths, patch_size):
    output_file = paths["output"] + "Patches.nc"
    output_netcdf = netCDF4.Dataset(output_file, "w", format = "NETCDF4")
    #
    ID_patch = output_netcdf.createDimension("ID_patch", Number_of_samples)
    x = output_netcdf.createDimension("x", patch_size)
    y = output_netcdf.createDimension("y", patch_size)
    #
    Outputs = vars()
    for var in Patches:
        Outputs[var] = output_netcdf.createVariable(var, "d", ("ID_patch", "y", "x"))
        Outputs[var][:,:,:] = Patches[var]
    #
    output_netcdf.close()


# # Data processing

# In[240]:


t0 = time.time()
Patches, Number_of_samples = create_patch_file(date_min, date_max, paths)()
print("Concatenate patches done, Number of samples: ", Number_of_samples)
Shuffled_dataset, Number_of_valid_samples = shuffle_and_clean_dataset(Patches, target_var_cleaning)()
print("Shuffle patches done, Number of samples: ", Number_of_valid_samples)
write_netCDF(Shuffled_dataset, Number_of_valid_samples, paths, patch_size)
#
tf = time.time()
print("Computing time: ", tf - t0)

