#!/usr/bin/env python
# coding: utf-8

import h5py
import torch
import torch_geometric
import numpy as np

class Data_generator_GNN_prediction(torch.utils.data.Dataset):
    def __init__(self, filename_data, footprint_radius, list_predictors, normalization_stats, n_nodes, batch_size: int=1):
        self.hdf = h5py.File(filename_data, "r")
        self.footprint_radius = footprint_radius
        self.list_predictors = list_predictors
        self.normalization_stats = normalization_stats
        self.list_IDs = self.generate_list_IDs()
        self.batch_size = int(batch_size)
        self.n_nodes = n_nodes

    def generate_list_IDs(self):
        Number_of_graphs = self.hdf["xx"][()].shape[0]
        print("Number_of_graphs", Number_of_graphs)
        return np.arange(0, Number_of_graphs)

    def __len__(self):
        return np.ceil(len(self.list_IDs)/self.batch_size).astype("int")

    def normalize(self, var, var_data):
        if var == "Distance_matrix":
            norm_data = 1 - var_data / (self.footprint_radius * 2)
        elif var == "Distance_to_footprint_center":
            norm_data = var_data / self.footprint_radius
        else:
            norm_data = (var_data - self.normalization_stats[var + "_min"]) / (self.normalization_stats[var + "_max"] - self.normalization_stats[var + "_min"])
        return norm_data

    def __getitem__(self, index):
        """ return batch number index """
        print("index", index, self.batch_size)
        start_id = self.list_IDs[index*self.batch_size]
        end_id = min(start_id + self.batch_size, self.list_IDs[-1]+1)
        n_nodes = self.n_nodes  # The number of nodes is 25 TODO dot constant
        print("gettitem", start_id, end_id)
        x_chunk = np.stack([self.hdf[pred][start_id:end_id,:] for pred in self.list_predictors], axis = -1).astype("float32")
        adj_chunk = self.hdf["Distance_matrix"][start_id:end_id].astype("float32")

        x_chunk = np.nan_to_num(x_chunk, nan=0.0)
        for i, pred in enumerate(self.list_predictors):
            x_chunk[:,:,i] = self.normalize(pred, x_chunk[:,:,i])
        
        adj_matrix = self.normalize("Distance_matrix", adj_chunk)
        
        #a = np.ones((n_nodes, n_nodes), dtype="float32")
        batch_data = []
        #for i in range(len(self.list_IDs)):
        for i in range(end_id - start_id):
            sample_id = i

            # Normalize adjacency matrix
            a = adj_matrix[i]
            edge_index, edge_weight = torch_geometric.utils.dense_to_sparse(torch.tensor(a, dtype = torch.float32))

            x = x_chunk[i,:,:]
            # Create Data object
            data = torch_geometric.data.Data(
                x = torch.tensor(x, dtype = torch.float32),
                edge_index = edge_index,
                edge_attr = edge_weight,
                num_nodes = n_nodes,
                sample_id = torch.tensor(sample_id, dtype = torch.float32),
            )
            batch_data.append(data)
        batch_data = torch_geometric.data.Batch.from_data_list(batch_data)
        return batch_data

