#
# Andrew Geiss, Feb 15th 2022
# Edited by William Yik
#
import numpy as np
import tensorflow as tf
from keras.layers import *
from keras.activations import *

class ApplyPosWeight(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        self.weight = self.add_weight(trainable=True)
    
    def call(self, inputs):
        return sigmoid(self.weight) * inputs

def RandDense(prev_layer, out_shape, layer_count_range, layer_size_range, param_lim, high_skips=False):
    # prev_layer is expected to have shape (None, x)
    n_in = prev_layer.shape[1]
    n_out = out_shape
    
    param_count = float('inf')

    while param_count not in range(*param_lim):
        
        if layer_count_range[0] == layer_count_range[1]:
            n_layers = layer_count_range[0]
        else:
            n_layers = np.random.randint(*layer_count_range)

        n_neurons = int(np.random.randint(*layer_size_range))
        if high_skips:
            adj_mat = build_adj_mat(n_layers, high_skips=True)
        else:
            adj_mat = build_adj_mat(n_layers, high_skips=False)
        param_count = count_params(n_in, n_out, n_layers, n_neurons, adj_mat)

    if n_neurons > n_in:
        chan_pad = relu(prev_layer)
        chan_pad = Dense(n_neurons-n_in)(chan_pad)
        layers = [concatenate([prev_layer, chan_pad])]
    else:
        activation = relu(prev_layer)
        layers = [Dense(n_neurons)(activation)]

    #iterate over each layer
    for i in range(0, n_layers):
        
        #gather inputs:
        inputs = [layers[ind] for ind in list(np.where(adj_mat[i,:])[0])]
        if len(inputs)>1:
            if i == n_layers-1:
                inputs = average(inputs)
            else:
                inputs = [ApplyPosWeight()(inp) for inp in inputs]
                inputs = add(inputs)
        else:
            inputs = inputs[0]
            
        #create layer:
        if i < n_layers-1:
            x = relu(inputs)
            x = Dense(n_neurons)(x)
            layers.append(x)
        else:
            out_layer = Dense(n_out, activation='linear')(inputs)

    return out_layer, adj_mat, param_count

def build_adj_mat(n_layers, high_skips=False):
    #generate a random acyclic adjacency matrix:
    adj = np.zeros((n_layers,n_layers))
    x,y = np.tril_indices(n_layers)
    edges = np.zeros(x.shape)
    if high_skips:
        n_active = np.random.randint(int(x.shape[0]/2),x.shape[0]+1)
    else:
        n_active = np.random.randint(0,x.shape[0]+1)
    edges[np.random.choice(np.arange(len(edges)), n_active, replace=False)] = 1
    adj[x,y] = edges
    
    #make sure each node has at least one inbound and one outbound edge:
    for i in range(n_layers):
        if np.max(adj[i,:]) == 0:
            adj[i,np.random.randint(0,i+1)] = 1
        if np.max(adj[:,i]) == 0:
            adj[np.random.randint(i,n_layers),i] = 1
            
    return adj

def count_params(n_in, n_out, n_layers, n_neurons, adj_mat):
    count = (n_in+1)*(n_neurons-n_in)
    for i in range(n_layers):
        insz = n_neurons
        
        if i == n_layers-1:
            outsz = n_out
        else:
            outsz = n_neurons
        count += insz*outsz + outsz
    return int(count)