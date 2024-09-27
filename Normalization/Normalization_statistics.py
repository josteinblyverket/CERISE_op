#!/usr/bin/env python
# coding: utf-8

# In[26]:


import os
import h5py
import netCDF4
import datetime
import numpy as np
import time


# # Constants

# In[27]:


date_min = "20200901"
date_max = "20220531"
#
paths = {}
paths["data"] = "/lustre/storeB/project/nwp/H2O/wp3/Deep_learning_predictions/Training_data/"
paths["output"] = "/lustre/storeB/project/nwp/H2O/wp3/Deep_learning_predictions/Normalization/"
#
list_variables = {}
list_variables["geolocation"] = ["time", "x", "y", "lat", "lon"]
#
hours_AMSR2 = "H03"


# # extract_dataset function
# 
#     date_min: earliest date to consider
#     date_max: latest date to consider
#     paths: paths from the "Constants" section

# In[28]:


def extract_dataset(date_min, date_max, hours_AMSR2 = hours_AMSR2, paths = paths):
    current_date = datetime.datetime.strptime(date_min, "%Y%m%d")
    end_date = datetime.datetime.strptime(date_max, "%Y%m%d")
    dataset = []
    while current_date <= end_date:
        cdate = current_date.strftime("%Y%m%d")
        filename = paths["data"] + cdate[0:4] + "/" + cdate[4:6] + "/" + "Dataset_" + cdate + hours_AMSR2 + ".nc"
        if os.path.isfile(filename):
            dataset.append(filename)
        current_date = current_date + datetime.timedelta(days = 1)
    return(dataset)


# # normalization_and_standardization_stats
#    
#     dataset: dataset created using the function "extract dataset"
#     list_variables: list_variables from the Constants section

# In[29]:


class normalization_and_standardization_stats():
    def __init__(self, dataset, list_variables):
        self.dataset = dataset
        self.list_variables = list_variables
    #
    def extract_variables(self):
        self.list_variables["data"] = []
        nc = netCDF4.Dataset(self.dataset[0], "r")
        for var in nc.variables:
            if (var in self.list_variables["geolocation"]) == False:
                self.list_variables["data"].append(var)
        nc.close()
        return(list_variables["data"])
    #
    def extract_stats(self, variables_to_normalize):
        Stats = {}
        for var in variables_to_normalize:
            print(var)
            for i, fi in enumerate(self.dataset):
                nc = netCDF4.Dataset(fi, "r")
                field = nc.variables[var][:,:]
                field[field.mask == True] = np.nan
                field_flat = np.ndarray.flatten(field)
                if i == 0:
                    field_conc = np.copy(field_flat)
                else:
                    field_conc = np.hstack((field_conc, field_flat))
                nc.close()
            #
            Stats[var + "_min"] = np.nanmin(field_conc)
            Stats[var + "_max"] = np.nanmax(field_conc)
            Stats[var + "_std"] = np.nanstd(field_conc)
            Stats[var + "_mean"] = np.nanmean(field_conc)
        #
        return(Stats)
    #
    def __call__(self):
        variables_to_normalize = self.extract_variables()
        Stats = self.extract_stats(variables_to_normalize)
        return(Stats)


# write_hdf5 function
# 
#     Stats: output of the function "extract_stats"
#     date_min: date_min from the "Constants" section
#     date_max: date_max from the "Constants" section
#     frequency: frequency from the "Constants" section
#     paths: paths from the "Constants" section

# In[30]:


def write_hdf5(Stats, date_min = date_min, date_max = date_max, paths = paths):
    filename = paths["output"] + "Stats_normalization_" + date_min + "_" + date_max + ".h5"
    hf = h5py.File(filename, 'w')
    for var in Stats: 
        hf.create_dataset(var, data = Stats[var])
    hf.close()


# Data processing

# In[31]:


t0 = time.time()
dataset = extract_dataset(date_min, date_max, paths = paths)
print("len(dataset)", len(dataset))
Stats = normalization_and_standardization_stats(dataset, list_variables)()
write_hdf5(Stats, date_min = date_min, date_max = date_max, paths = paths)
tf = time.time()
print("Computing time", tf-t0)


# In[ ]:




