#!/usr/bin/env python
# coding: utf-8

# In[125]:


import time
import os
import sys
import h5py
import pandas
import datetime
import logging
import torch
import torchsummary
import torch_geometric
import numpy as np


# In[126]:


t0 = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check if a GPU is available
print("Using device: "  + str(device))


# # Constants

# In[ ]:


experiment_name = "v6"
AMSR2_frequency = "18.7"
date_model = "20250303"
#
function_path = "/lustre/storeB/users/cyrilp/CERISE/Scripts/GNN/Model_static/" + experiment_name + "/"
sys.path.insert(0, function_path)
from Data_generator_GNN import *
from GNN_GAT import *
#
paths = {}
paths["training"] = "/lustre/storeB/project/nwp/H2O/wp3/Deep_learning_predictions/Training_data_GNN/" + AMSR2_frequency.split('.')[0] + "GHz_static/"
paths["normalization"] = "/lustre/storeB/project/nwp/H2O/wp3/Deep_learning_predictions/Normalization/"
paths["output"] = "/lustre/storeB/project/nwp/H2O/wp3/Deep_learning_predictions/GNN/Models_static/" + experiment_name + "/"
#
filename_normalization = paths["normalization"] + "Stats_normalization_20200901_20220531.h5"
#
for var in paths:
    if os.path.isdir(paths[var]) == False:
        os.system("mkdir -p " + paths[var])
#
AMSR2_all_frequencies = ["6.9", "7.3", "10.7", "18.7", "23.8", "36.5"]
AMSR2_all_footprint_radius = np.array([35 + 62, 35 + 62, 24 + 42, 14 + 22, 11 + 19, 7 + 12]) * 0.25 * 1000  # 0.5 * mean diameter (0.5 * (major + minor)), *1000 => km to meters
AMSR2_footprint_radius = AMSR2_all_footprint_radius[AMSR2_all_frequencies.index(AMSR2_frequency)]


# # Model parameters

# In[ ]:


date_min_train = "20200901"
date_max_train = "20220531"
date_min_valid = "20220901"
date_max_valid = "20230531"
subsampling = "1"
#
def he_normal_init(weight):
    torch.nn.init.kaiming_normal_(weight, mode = "fan_out", nonlinearity = "relu")
weight_initializer = he_normal_init
#
activation = torch.nn.ReLU()
kernel_initializer = "he_normal"
shuffle = True
conv_filers = [32, 64, 32]
batch_size = 512
batch_normalization = True
attention_heads = 4
#
predictors = {}
predictors["constants"] = ["ZS", "PATCHP1", "PATCHP2", "FRAC_LAND_AND_SEA_WATER", "Distance_to_footprint_center"]
predictors["atmosphere"] = ["lwe_thickness_of_atmosphere_mass_content_of_water_vapor"]
#predictors["ISBA"] = ["Q2M_ISBA", "DSN_T_ISBA", "LAI_ga", "TS_ISBA", "PSN_ISBA"]
predictors["ISBA"] = ["LAI_ga", "DSN_T_ISBA", "WSN_T_ISBA"]
predictors["TG"] = [1, 2]
predictors["WG"] = [1, 2]
predictors["WGI"] = [1, 2]
#predictors["WSN_VEG"] = [1, 6, 12]
predictors["RSN_VEG"] = [1, 6, 12]
predictors["HSN_VEG"] = [1, 6, 12]
predictors["SNOWTEMP"] = [1, 6, 12]
predictors["SNOWLIQ"] = [1, 6, 12]


# # Training parameters

# In[ ]:


compile_params = {"loss_function": torch.nn.MSELoss(),    # Loss function
                  "initial_learning_rate": 0.000078125,   # Initial learning rate
                  "step_size": 5,                         # Define how many epochs are computed before the learning rate is decreased.
                  "gamma": 0.5,                           # Define the factor used to decrease the learning rate. If 0.25, the learning rate is divided by 4.
                  "n_epochs": 30,                         # Number of epochs used for training the model
                  }


# # Make model parameters

# In[ ]:


class make_parameters():
    def __init__(self, paths, filename_normalization, AMSR2_frequency, AMSR2_footprint_radius, predictors, activation, weight_initializer, conv_filters, batch_size, batch_normalization, attention_heads, shuffle, date_min_train, date_max_train, date_min_valid, date_max_valid, subsampling):
        self.paths = paths
        self.filename_normalization = filename_normalization
        self.AMSR2_frequency = AMSR2_frequency
        self.AMSR2_footprint_radius = AMSR2_footprint_radius 
        self.predictors = predictors
        self.activation = activation
        self.weight_initializer = weight_initializer
        self.conv_filters = conv_filters
        self.batch_size = batch_size
        self.batch_normalization = batch_normalization
        self.attention_heads = attention_heads
        self.shuffle = shuffle
        self.date_min_train = date_min_train
        self.date_max_train = date_max_train
        self.date_min_valid = date_min_valid
        self.date_max_valid = date_max_valid
        self.subsampling = subsampling
        self.filename_train = self.paths["training"] + "Graphs_" + self.date_min_train + "_" + self.date_max_train + "_subsampling_" + self.subsampling + ".zarr"
        self.filename_valid = self.paths["training"] + "Graphs_" + self.date_min_valid + "_" + self.date_max_valid + "_subsampling_" + self.subsampling + ".zarr"
    #
    def make_list_predictors(self):
        list_predictors = predictors["constants"] + predictors["atmosphere"] + predictors["ISBA"]
        for pred in predictors:
            if (pred != "constants") and (pred != "atmosphere") and (pred != "ISBA"):
                for lay in predictors[pred]:
                    var_name = pred + str(lay) + "_ga"
                    list_predictors = list_predictors + [var_name]
        return(list_predictors)
    #
    def make_list_targets(self):
        list_targets = ["AMSR2_BT" + self.AMSR2_frequency + "H", "AMSR2_BT" + self.AMSR2_frequency + "V"]
        return(list_targets)
    #
    def load_normalization_stats(self):
        normalization_stats = {}
        with h5py.File(self.filename_normalization) as hdf:
            for var in hdf:
                normalization_stats[var] = hdf[var][()]
        return(normalization_stats)
    #
    def make_model_parameters(self, list_predictors, list_targets):
        model_params = {"list_predictors": list_predictors,
                        "list_targets": list_targets,
                        "activation": self.activation,
                        "weight_initializer": self.weight_initializer,
                        "conv_filters": self.conv_filters,
                        "batch_normalization": self.batch_normalization,
                        "heads": self.attention_heads,
                        }
        return(model_params)
    #
    def make_data_generator_parameters(self, filename_data, list_predictors, list_targets):
        data_generator_params = {"filename_data": filename_data,
                                 "footprint_radius": self.AMSR2_footprint_radius,
                                 "list_predictors": list_predictors,
                                 "list_targets": list_targets,
                                 "normalization_stats": self.load_normalization_stats(),
                                 "batch_size": self.batch_size,
                                 "paths": self.paths}
        return(data_generator_params)
    #
    def create_data_loader(self, data_generator_params):
        dataset = Data_generator_GNN(**data_generator_params)
        return(dataset)
    #
    def __call__(self):
        list_predictors = self.make_list_predictors()
        list_targets = self.make_list_targets()
        model_params = self.make_model_parameters(list_predictors, list_targets)
        params_train = self.make_data_generator_parameters(self.filename_train, list_predictors, list_targets)
        params_valid = self.make_data_generator_parameters(self.filename_valid, list_predictors, list_targets)
        train_loader = self.create_data_loader(params_train)
        valid_loader = self.create_data_loader(params_valid)
        return(model_params, train_loader, valid_loader)


# # Training model

# In[ ]:


class train_model():
    def __init__(self, AMSR2_frequency, model, train_loader, valid_loader, compile_params, paths, filename_normalization, device, initial_time, date_model):
        self.AMSR2_frequency = AMSR2_frequency
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.compile_params = compile_params
        self.paths = paths
        self.hdf_normalization = h5py.File(filename_normalization)
        self.device = device
        self.initial_time = initial_time 
        self.date_model = date_model
        self.filename_training_stats = paths["output"] + "Training_statistics_" + self.AMSR2_frequency.split('.')[0] + "GHz_" + self.date_model + ".txt"
        self.filename_model = paths["output"] + "GNN_model_" + self.AMSR2_frequency.split('.')[0] + "GHz.pth"
        self.loss_function = compile_params["loss_function"]
        self.optimizer =  torch.optim.Adam(self.model.parameters(), lr = self.compile_params["initial_learning_rate"])
        self.scheduler = self.learning_rate_scheduler()
    #
    def learning_rate_scheduler(self): 
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size = self.compile_params["step_size"], gamma = self.compile_params["gamma"])
        return(scheduler)
    #
    def load_stats_previous_training(self):
        df = pandas.read_csv(self.filename_training_stats, delimiter = "\t")
        best_validation_loss = np.array(df["Best_validation_loss "])[-1]
        N_epochs = int(np.array(df["Epoch"])[-1])
        return(best_validation_loss, N_epochs)
    #
    def write_training_statistics(self, epoch, average_train_loss, average_valid_loss, best_valid_loss, current_learning_rate, N_epochs):
        te = time.time()
        if epoch == N_epochs:
            self.t_previous = self.initial_time
        #
        time_epoch = round(te - self.t_previous)
        self.t_previous = te
        #
        stats_str = str(epoch + 1) + "\t" + \
                    str(average_train_loss) + "\t" + \
                    str(average_valid_loss) + "\t" + \
                    str(best_valid_loss) + "\t" + \
                    str(current_learning_rate) + "\t" + \
                    str(time_epoch) + "\n"
        #
        with open(self.filename_training_stats, "a") as stats_file:
            stats_file.write(stats_str)
    #
    def save_best_model(self, model, average_valid_loss, best_valid_loss, opt, epoch):
        print("Validation loss improved from " + str(best_valid_loss) + " to " + str(average_valid_loss) + ". Model saved.")
        saved_stats = {"epoch": epoch + 1,
                       "model_state_dict": model.state_dict(),
                       "optimizer_state_dict": opt.state_dict(),
                       "val_loss": average_valid_loss}
        #
        torch.save(saved_stats, self.filename_model)
    #
    def training_loop(self):
        best_valid_loss, N_epochs = self.load_stats_previous_training()
        scaler = torch.amp.GradScaler()
        #
        for epoch in range(N_epochs, self.compile_params["n_epochs"]):
            print("epoch", epoch)
            #
            # Training
            #
            self.model.train()  # Set model to training mode
            train_loss = 0
            total_samples_train = 0
            #
            for batch_train in self.train_loader:
                data_batch = batch_train.to(self.device)
                x, y, a, batch = data_batch.x, data_batch.y, data_batch.edge_index, data_batch.batch
                #
                self.optimizer.zero_grad()                       # Clear old gradients, and avoid accumulating gradients from previous batches
                with torch.amp.autocast(device_type = "cuda"):   # This is used for mixed precision training
                    output = self.model(x, a, batch)             # Make predictions
                    loss = self.loss_function(output, y)         # Compute loss
                scaler.scale(loss).backward()                    # Backward pass, compute new gradients
                scaler.step(self.optimizer)                      # Update weights based on new gradients
                scaler.update()                                  # Adjust the scaling factor for the next iteration 
                #
                train_loss += loss.item() * y.size(0)
                total_samples_train += y.size(0)
            #
            average_train_loss = train_loss / total_samples_train
            #
            # Validation
            #
            self.model.eval()
            valid_loss = 0
            total_samples_valid = 0
            #
            with torch.no_grad(), torch.amp.autocast(device_type = "cuda"):
                for batch_valid in self.valid_loader:
                    data_batch = batch_valid.to(self.device)
                    x, y, a, batch = data_batch.x, data_batch.y, data_batch.edge_index, data_batch.batch
                    #
                    output = self.model(x, a, batch)                             # Make predictions
                    loss = self.loss_function(output, y)                         # Compute loss
                    valid_loss += loss.item() * y.size(0)                        # Accumulate loss over several batches with weighting the losses according to the number of graphs within the batch
                    total_samples_valid += y.size(0)                             # Compute the total number of graphs from the beginning of the epoch 
            #
            average_valid_loss = valid_loss / total_samples_valid                # Compute the mean loss for the epoch
            current_learning_rate = self.optimizer.param_groups[0]['lr']         # Extract the current learning rate
            self.scheduler.step()  
            #
            if average_valid_loss < best_valid_loss:
                self.save_best_model(model = self.model, 
                                     average_valid_loss = average_valid_loss, 
                                     best_valid_loss = best_valid_loss, 
                                     opt = self.optimizer, 
                                     epoch = epoch)
                #
                best_valid_loss = average_valid_loss
            #
            self.write_training_statistics(epoch = epoch, 
                                           average_train_loss = average_train_loss, 
                                           average_valid_loss = average_valid_loss,
                                           best_valid_loss = best_valid_loss, 
                                           current_learning_rate = current_learning_rate,
                                           N_epochs = N_epochs)
        #
        return(self.model)
    #
    def __call__(self):
        trained_model = self.training_loop()
        return(trained_model)


# # Data processing

# In[ ]:


model_params, train_loader, valid_loader =  make_parameters(paths = paths, 
                                                            filename_normalization = filename_normalization, 
                                                            AMSR2_frequency = AMSR2_frequency, 
                                                            AMSR2_footprint_radius = AMSR2_footprint_radius,
                                                            predictors = predictors, 
                                                            activation = activation, 
                                                            weight_initializer = weight_initializer, 
                                                            conv_filters = conv_filers, 
                                                            batch_size = batch_size, 
                                                            batch_normalization = batch_normalization,
                                                            attention_heads = attention_heads,
                                                            shuffle = shuffle, 
                                                            date_min_train = date_min_train, 
                                                            date_max_train = date_max_train, 
                                                            date_min_valid = date_min_valid, 
                                                            date_max_valid = date_max_valid,
                                                            subsampling = subsampling)()
#
GNN_model = GNN_GAT(**model_params).to(device)
checkpoint = torch.load(paths["output"] + "GNN_model_" + AMSR2_frequency.split('.')[0] + "GHz.pth", weights_only = False) 
GNN_model.load_state_dict(checkpoint["model_state_dict"])
#
trained_model = train_model(AMSR2_frequency = AMSR2_frequency,
                            model = GNN_model, 
                            train_loader = train_loader, 
                            valid_loader = valid_loader, 
                            compile_params = compile_params, 
                            paths = paths,
                            filename_normalization = filename_normalization,
                            device = device,
                            initial_time = t0,
                            date_model = date_model)()
#
tf = time.time()
print("Computing time", tf - t0)

