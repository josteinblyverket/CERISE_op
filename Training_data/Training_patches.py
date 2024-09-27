#!/usr/bin/env python
# coding: utf-8

# In[11]:


import time
import os
import netCDF4
import datetime
import numpy as np


# # Constants

# In[12]:


SGE_TASK_ID = 3
#
paths = {}
paths["training"] = "/lustre/storeB/project/nwp/H2O/wp3/Deep_learning_predictions/Training_data/"
paths["output"] = "/lustre/storeB/project/nwp/H2O/wp3/Deep_learning_predictions/Patches/Training_data/"
#
date_min = "20200901" 
date_max = "20230601"
#
params_patch = {"patch_size": 5,
                 "min_fraction_land": 0.5,
                 "min_coverage_AMSR2": 1,
                }


# # List dates

# In[13]:


def make_list_dates(date_min, date_max, paths):
    current_date = datetime.datetime.strptime(date_min, "%Y%m%d")
    end_date = datetime.datetime.strptime(date_max, "%Y%m%d")
    list_dates = []
    while current_date <= end_date:
        date_str = current_date.strftime("%Y%m%d")
        filename = paths["training"] + date_str[0:4] + "/" + date_str[4:6] + "/Dataset_" + date_str +  "H03.nc"
        if os.path.isfile(filename) == True:
            list_dates.append(date_str)
        current_date = current_date + datetime.timedelta(days = 1)
    return(list_dates)


# # Make patches

# In[14]:


class make_patches():
    def __init__(self, date_task, params_patch, paths):
        self.date_task = date_task
        self.params_patch = params_patch
        self.paths = paths
        self.surfex_dims = (250, 250)
    #
    def load_data(self):
        Dataset = {}
        filename = self.paths["training"] + self.date_task[0:4] + "/" + self.date_task[4:6] + "/Dataset_" + self.date_task +  "H03.nc"
        nc = netCDF4.Dataset(filename, "r")
        for var in nc.variables:
            Dataset[var] = nc.variables[var][:,:]
        nc.close()
        return(Dataset)
    #
    def extract_patch(self, ymin, xmin, Dataset):
        Patch = {}
        ymax = ymin + self.params_patch["patch_size"]
        xmax = xmin + self.params_patch["patch_size"]
        #
        coverage_AMSR2 = 1 - np.sum(Dataset["tb36_v"][ymin:ymax, xmin:xmax].mask == True) / (self.params_patch["patch_size"]**2)
        if coverage_AMSR2 >= self.params_patch["min_coverage_AMSR2"]:
            frac_water = np.sum(Dataset["FRAC_SEA"][ymin:ymax, xmin:xmax] + Dataset["FRAC_WATER"][ymin:ymax, xmin:xmax]) / (self.params_patch["patch_size"]**2)
            if frac_water <= self.params_patch["min_fraction_land"]:
                for var in Dataset:
                    Patch[var] = np.expand_dims(Dataset[var][ymin:ymax, xmin:xmax], axis = 0)
        return(Patch)
    #
    def write_netCDF(self, All_patches, IDs_patch):
        path_output = self.paths["output"] + self.date_task[0:4] + "/" + self.date_task[4:6] + "/" 
        if os.path.exists(path_output) == False:
            os.system("mkdir -p " + path_output)    
        output_filename = path_output + "Dataset_" + self.date_task + ".nc"  
        if os.path.isfile(output_filename):
            os.system("rm " + output_filename)
        output_netcdf = netCDF4.Dataset(output_filename, "w", format = "NETCDF4")
        #
        ID_patch = output_netcdf.createDimension("ID_patch", len(IDs_patch))
        x = output_netcdf.createDimension("x", self.params_patch["patch_size"])
        y = output_netcdf.createDimension("y", self.params_patch["patch_size"])
        #
        Outputs = vars()  
        for var in All_patches:
            Outputs[var] = output_netcdf.createVariable(var, "d", ("ID_patch", "y", "x"))
            Outputs[var][:,:,:] = All_patches[var]
        #
        output_netcdf.close() 
    #
    def __call__(self):
        Dataset = self.load_data()
        ID_patch = 0
        #
        for ymin in range(0, self.surfex_dims[0] - self.params_patch["patch_size"], self.params_patch["patch_size"]):
            for xmin in range(0, self.surfex_dims[1] - self.params_patch["patch_size"], self.params_patch["patch_size"]):
                Patch = self.extract_patch(ymin, xmin, Dataset)
                if len(Patch) > 0:
                    ID_patch = ID_patch + 1
                    #
                    if ID_patch == 1:
                        All_patches = Patch
                    else:
                        for var in Patch:
                            All_patches[var] = np.concatenate((All_patches[var], Patch[var]), axis = 0)
        #
        IDs_patch = np.arange(ID_patch) + 1
        self.write_netCDF(All_patches, IDs_patch)


# # Data processing

# In[15]:


t0 = time.time()
list_dates = make_list_dates(date_min, date_max, paths)
print("number of dates", len(list_dates))
date_task = list_dates[SGE_TASK_ID -1]
make_patches(date_task, params_patch, paths)()
tf = time.time()
print("Computing time", tf - t0)

