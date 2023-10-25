# Based on code from ClimateBench (Watson-Parris et al.)
import numpy as np
import pandas as pd
import xarray as xr
data_path = './drive/My Drive/Colab Notebooks/USC Random NN/data/train_val/'

len_historical = 165
slider = 10

# Utilities for normalizing the input data
def normalize(data, var, meanstd_dict):
    mean = meanstd_dict[var][0]
    std = meanstd_dict[var][1]
    return (data - mean)/std

def unnormalize(data, var, meanstd_dict):
    mean = meanstd_dict[var][0]
    std = meanstd_dict[var][1]
    return data * std + mean

# Functions for reshaping the data 
def input_for_training(X_train_xr): 
    X_train_np =  X_train_xr.to_array().transpose('time', 'latitude', 'longitude', 'variable').data
    return X_train_np 

def output_for_training(Y_train_xr, var): 
    Y_train_np = Y_train_xr[var].data
    return Y_train_np

def create_training_data(simus, var_to_predict='tas'):
    X_train = []
    Y_train = []

    for i, simu in enumerate(simus):

        input_name = 'inputs_' + simu + '.nc'
        output_name = 'outputs_' + simu + '.nc'
        
        # load inputs
        input_xr = xr.open_dataset(data_path + input_name)
        
        # load outputs
        output_xr = xr.open_dataset(data_path + output_name).mean(dim='member')
        output_xr = output_xr.assign({"pr": output_xr.pr * 86400,
                                      "pr90": output_xr.pr90 * 86400}).rename({'lon':'longitude', 
                                                                               'lat': 'latitude'}).transpose('time','latitude', 'longitude').drop(['quantile'])
    
        # Append to list 
        X_train.append(input_xr)
        Y_train.append(output_xr)
    
    # Compute mean/std of each variable for the whole dataset
    meanstd_inputs = {}
    
    for var in ['CO2', 'CH4', 'SO2', 'BC']:
        # To not take the historical data into account several time we have to slice the scenario datasets
        # and only keep the historical data once (in the first ssp index 0 in the simus list)
        '''
        array = np.concatenate([X_train[i][var].data for i in [0, 3, 4]] + 
                               [X_train[i][var].sel(time=slice(len_historical, None)).data for i in range(1, 3)])
        '''
        array = np.concatenate([train[var].data for train in X_train])
        #print((array.mean(), array.std()))
        meanstd_inputs[var] = (array.mean(), array.std())
    
    # normalize input data 
    X_train_norm = [] 
    for i, train_xr in enumerate(X_train): 
        for var in ['CO2', 'CH4', 'SO2', 'BC']: 
            var_dims = train_xr[var].dims
            train_xr=train_xr.assign({var: (var_dims, normalize(train_xr[var].data, var, meanstd_inputs))}) 
        X_train_norm.append(train_xr)

    X_train_all = np.concatenate([input_for_training(X_train_norm[i]) for i in range(len(simus))], axis = 0)
    Y_train_all = np.concatenate([output_for_training(Y_train[i], var_to_predict) for i in range(len(simus))], axis=0)
    return X_train_all, Y_train_all, meanstd_inputs
