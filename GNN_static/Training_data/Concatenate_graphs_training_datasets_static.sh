#!/bin/bash -f
#$ -N Co_training_data_GNN_CERISE
#$ -l h_rt=72:00:00
#$ -S /bin/bash
#$ -pe shmem-1 1
#$ -l h_rss=50G,mem_free=50G,h_data=50G
#$ -q researchlong-r8.q
#$ -t 4-4
##$ -j y
##$ -m ba
#$ -o /home/cyrilp/Documents/OUT/OUT_$JOB_NAME.$JOB_ID_$TASK_ID
#$ -e /home/cyrilp/Documents/ERR/ERR_$JOB_NAME.$JOB_ID_$TASK_ID
##$ -R y
##$ -r y

source /modules/rhel8/conda/install/etc/profile.d/conda.sh
conda activate production-08-2024

echo "Got $NSLOTS slots for job $SGE_TASK_ID."

cat > "/home/cyrilp/Documents/PROG/Co_training_data_GNN_CERISE_""$SGE_TASK_ID"".py" << EOF
###################################################################################################
#!/usr/bin/env python
# coding: utf-8

# In[25]:


import os
import time
import h5py
import datetime
import numpy as np


# # Constants

# In[26]:


#
date_min = "20200901"
date_max = "20220531"
subsampling = 0
#
paths = {}
paths["daily_data"] = "/lustre/storeB/project/nwp/H2O/wp3/Deep_learning_predictions/Training_data_GNN/"
#
AMSR2_frequencies = ["6.9", "7.3", "10.7", "18.7", "23.8", "36.5"]
AMSR2_frequency_task = AMSR2_frequencies[$SGE_TASK_ID - 1]


# # List dates

# In[27]:


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

# In[28]:


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
        path_data = self.paths["daily_data"] + self.AMSR2_frequency_task.split('.')[0] + "GHz_static/" + self.date_task[0:4] + "/" + self.date_task[4:6] + "/"
        filename_data = path_data + "Graphs_" + self.date_task + ".h5"
        filename_output = self.paths["daily_data"] + self.AMSR2_frequency_task.split('.')[0] + "GHz_static/Graphs_" + self.date_min + "_" + self.date_max + "_subsampling_" + str(self.subsampling) + ".h5"
        #
        with h5py.File(filename_output, "a") as hdf_output:
            if os.path.isfile(filename_data) == True:
                with h5py.File(filename_data, "r") as hdf_data:
                    for var in hdf_data.keys():
                        var_data = hdf_data[var][()]
                        if var not in hdf_output:
                            if var_data.ndim == 1:
                                maxshape = (None,)
                            elif var_data.ndim == 2:
                                maxshape = (None, var_data.shape[1])
                            elif var_data.ndim == 3:
                                maxshape = (None, var_data.shape[1], var_data.shape[2])
                            hdf_output.create_dataset(var, data = var_data, maxshape = maxshape)
                        else:
                            var_data_output = hdf_output[var]
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

# In[29]:


t0 = time.time()
N_Graph_ID_cum = 0
list_dates = make_list_dates(date_min, date_max)
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
###################################################################################################
EOF
python3 "/home/cyrilp/Documents/PROG/Co_training_data_GNN_CERISE_""$SGE_TASK_ID"".py"
