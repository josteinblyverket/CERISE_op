#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time
import zarr
import pandas
import random
import torch
import torch_geometric
import numpy as np


# In[ ]:


class Data_generator_GNN(torch.utils.data.Dataset):
    def __init__(self, filename_data, footprint_radius, list_predictors, list_targets, normalization_stats, batch_size, paths):
        self.zarr_dataset = zarr.open(filename_data, "r")
        self.footprint_radius = footprint_radius
        self.list_predictors = list_predictors
        self.list_targets = list_targets
        self.normalization_stats = normalization_stats
        self.batch_size = batch_size
        self.paths = paths
        self.list_IDs = self.generate_list_IDs()

    def generate_list_IDs(self):
        Number_of_graphs = self.zarr_dataset["xx"].shape[0]
        idx = np.arange(0, Number_of_graphs, self.batch_size)
        shuffle_idx = random.sample(list(idx), len(idx))
        return shuffle_idx

    def __len__(self):
        return len(self.list_IDs)

    def normalize(self, var, var_data):
        if var == "Distance_matrix":
            norm_data = 1 - var_data / (self.footprint_radius * 2)
        elif var == "Distance_to_footprint_center":
            norm_data = var_data / self.footprint_radius
        else:
            norm_data = (var_data - self.normalization_stats[var + "_min"]) / (self.normalization_stats[var + "_max"] - self.normalization_stats[var + "_min"])
        return norm_data

    def __getitem__(self, index):
        start_id = self.list_IDs[index]
        end_id = start_id + self.batch_size
        n_nodes = 25  # The number of nodes is 25

        x_chunk = np.stack([self.zarr_dataset[pred][start_id:end_id, :] for pred in self.list_predictors], axis=-1)
        y_chunk = np.stack([self.zarr_dataset[targ][start_id:end_id] for targ in self.list_targets], axis=-1)
        adj_chunk = self.zarr_dataset["Distance_matrix"][start_id:end_id]

        x_chunk = np.nan_to_num(x_chunk, nan=0.0)
        for i, pred in enumerate(self.list_predictors):
            x_chunk[:, :, i] = self.normalize(pred, x_chunk[:, :, i])

        y_chunk = np.nan_to_num(y_chunk, nan=0.0)
        for i, targ in enumerate(self.list_targets):
            y_chunk[:, i] = self.normalize(targ, y_chunk[:, i])

        adj_matrix = self.normalize("Distance_matrix", adj_chunk)

        batch_data = []
        for i in range(self.batch_size):
            sample_id = start_id + i

            # Normalize adjacency matrix
            a = adj_matrix[i]
            edge_index, edge_weight = torch_geometric.utils.dense_to_sparse(torch.tensor(a, dtype=torch.float32))

            x = x_chunk[i,:,:]
            y = np.expand_dims(y_chunk[i,:], axis = 0)
            # Create Data object
            data = torch_geometric.data.Data(
                x=torch.tensor(x, dtype=torch.float32),
                edge_index=edge_index,
                edge_attr=edge_weight,
                y=torch.tensor(y, dtype=torch.float32),
                num_nodes=n_nodes,
                sample_id=torch.tensor(sample_id, dtype=torch.float32),
            )
            batch_data.append(data)

        return torch_geometric.data.Batch.from_data_list(batch_data)

