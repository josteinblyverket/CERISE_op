#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[ ]:


class CNN():
    def __init__(self, list_predictors, list_targets, patch_dim, batch_size, conv_filters, dense_width, activation, kernel_initializer, batch_norm, pooling_type, dropout):
        self.list_predictors = list_predictors
        self.list_targets = list_targets
        self.patch_dim = patch_dim
        self.batch_size = batch_size
        self.conv_filters = conv_filters
        self.dense_width = dense_width
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.batch_norm = batch_norm
        self.pooling_type = pooling_type
        self.dropout = dropout
        self.n_predictors = len(list_predictors)
        self.n_targets = len(list_targets)
    #
    def conv_block(self, x, num_filters, padding):
        x = tf.keras.layers.Conv2D(num_filters, kernel_size = (3,3), padding = padding, kernel_initializer = self.kernel_initializer)(x)
        if self.batch_norm == True:
            x = tf.keras.layers.BatchNormalization(axis = 3)(x)
        x = tf.keras.layers.Activation(self.activation)(x)
        #
        return(x)
    #
    def dense_block(self, x, width, activ, dropout):
        x = tf.keras.layers.Dense(width, kernel_initializer = self.kernel_initializer)(x)
        #
        if self.batch_norm == True:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activ)(x)
        #
        if dropout > 0:
            x = tf.keras.layers.Dropout(dropout)(x)
        #
        return(x)
    #
    def make_model(self): 
        inputs = tf.keras.layers.Input(shape = (*self.patch_dim, self.n_predictors))
        f1 = self.conv_block(inputs, self.conv_filters[0], padding = "same")
        f2 = self.conv_block(f1, self.conv_filters[1], padding = "same")
        f3 = self.conv_block(f2, self.conv_filters[2], padding = "valid")
        f4 = tf.keras.layers.Flatten()(f3)
        f5 = self.dense_block(f4, width = self.dense_width[0], activ = self.activation, dropout = self.dropout)
        f6 = self.dense_block(f5, width = self.dense_width[1], activ = self.activation, dropout = self.dropout)
        outputs = self.dense_block(f6, width = self.n_targets, activ = "linear", dropout = 0)
        CNN_model = tf.keras.Model(inputs, outputs, name = "CNN")
        return(CNN_model)

