#!/usr/bin/env python
# coding: utf-8

import os
import time
import h5py
import datetime
import numpy as np
import pandas as pd


# # Constants

SGE_TASK_ID = 6
#
date_min = "20141001"
date_max = "20141101"
subsampling = 0
#
paths = {}
#paths["daily_data"] = "/ec/res4/scratch/sbjb/sfx_data/CERISE_Land_Pv2_dev01/archive/"
paths["daily_data"] = "/ec/res4/scratch/sbjb/Projects/CERISE/ObsOpData/GNN/18GHz_static/pan-Arctic/Graphs/"
#paths["output"] = "/ec/res4/scratch/sbjb/Projects/CERISE/ObsOpData/GNN/18GHz_static/Training/"
paths["output"] = "/ec/res4/scratch/sbjb/Projects/CERISE/ObsOpData/GNN/18GHz_static/pan-Arctic/Graphs_conc/"
#
AMSR2_frequencies = ["6.9", "7.3", "10.7", "18.7", "23.8", "36.5"]
AMSR2_frequency_task = AMSR2_frequencies[SGE_TASK_ID - 1]
print(AMSR2_frequency_task)

# # List dates

def make_list_dates(date_min, date_max):
    current_date = datetime.datetime.strptime(date_min, "%Y%m%d")
    end_date = datetime.datetime.strptime(date_max, "%Y%m%d")
    list_dates = []
    while current_date <= end_date:
        date_str = current_date.strftime("%Y%m%d")
        list_dates.append(date_str)
        current_date = current_date + datetime.timedelta(days = 1)
    return(list_dates)

# # Read data

class concatenate_graphs_from_multiple_dates():
    def __init__(self, N_Graph_IDs, date_task, date_min, date_max, subsampling, AMSR2_frequency_task, paths):
        self.N_Graph_IDs = N_Graph_IDs
        self.date_task = date_task
        self.date_min = date_min
        self.date_max = date_max
        self.subsampling = subsampling
        self.AMSR2_frequency_task = AMSR2_frequency_task
        self.paths = paths
    #
    def concatenate_graphs(self):
        #path_data = self.paths["daily_data"] + self.AMSR2_frequency_task.split('.')[0] + "GHz_static/" + self.date_task[0:4] + "/" + self.date_task[4:6] + "/"
        path_data = self.paths["daily_data"] #+ self.date_task[0:4] + "/" + self.date_task[4:6] + "/" + self.date_task[6:8] + "/"+ "03" + "/" + "000" + "/"
        filename_data = path_data + "Graphs_18_7_" + self.date_task + ".h5"
        filename_output = self.paths["output"] + "/Graphs_" + self.date_min + "_" + self.date_max + "_val_subsampling_patch_fix_" + str(self.subsampling) + ".h5"
        #filename_output = self.paths["output"] + "/Graphs_" + self.date_min + "_" + self.date_max + "_subsampling_patch_fix_" + str(self.subsampling) + ".h5"
        #
        with h5py.File(filename_output, "a") as hdf_output:
            if os.path.isfile(filename_data) == True:
                with h5py.File(filename_data, "r") as hdf_data:
                    for var in hdf_data.keys():
                        print("var")
                        print(var)                        
                        var_data = hdf_data[var][()]
                        print("dim")                        
                        print(var_data.ndim)
                        print(filename_data)
                        if var not in hdf_output:
                            if var_data.ndim == 1:
                                maxshape = (None,)
                            elif var_data.ndim == 2:
                                maxshape = (None, var_data.shape[1])
                            elif var_data.ndim == 3:
                                maxshape = (None, var_data.shape[1], var_data.shape[2])                            

                            hdf_output.create_dataset(var, data = var_data, maxshape = maxshape)
                        else:
                            print("inside else")
                            print(var)
                            var_data_output = hdf_output[var]
                            print(var_data_output.ndim)
                            if var_data_output.ndim == 1:
                                var_data_output.resize((var_data_output.shape[0] + var_data.shape[0]), axis = 0)
                            elif var_data_output.ndim == 2:
                                var_data_output.resize((var_data_output.shape[0] + var_data.shape[0]), axis = 0)
                            elif var_data_output.ndim == 3:
                                var_data_output.resize((var_data_output.shape[0] + var_data.shape[0]), axis = 0)
                            var_data_output[-var_data.shape[0]:] = var_data
    #
    def __call__(self):
        Graph_ID_output = self.concatenate_graphs()
        return(Graph_ID_output)

# # Data processing

t0 = time.time()
N_Graph_ID_cum = 0
#list_dates = make_list_dates(date_min, date_max)

#print(list_dates)
date_min = "20141001"
date_max = "20141101"

all_dates = pd.date_range(start="2014-10-01", end="2014-11-01", freq='D')
# Filter out the first 10 days of each month
filtered_dates = [date for date in all_dates if date.day <= 10] # Validation data
#filtered_dates = [date for date in all_dates if date.day > 10] # Training data

# Convert to list of strings for display
list_dates = [date.strftime('%Y%m%d') for date in filtered_dates]

print(list_dates)

for di, date_task in enumerate(list_dates):
    print(date_task)
    #try:
    N_Graph_ID_output = concatenate_graphs_from_multiple_dates(N_Graph_IDs = N_Graph_ID_cum, 
                                                                date_task = date_task, 
                                                                date_min = date_min, 
                                                                date_max = date_max, 
                                                                subsampling = subsampling,
                                                                AMSR2_frequency_task = AMSR2_frequency_task, 
                                                                paths = paths)()
    N_Graph_ID_cum = N_Graph_ID_output
    #except:
    #    pass
#
print(N_Graph_ID_cum, type(N_Graph_ID_cum))
tf = time.time()
print("Computing time: ", tf - t0)