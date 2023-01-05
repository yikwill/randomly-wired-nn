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
def input_for_training(X_train_xr, skip_historical=False, len_historical=None): 
    
    X_train_np =  X_train_xr.to_array().transpose('time', 'latitude', 'longitude', 'variable').data

    time_length = X_train_np.shape[0]
    # If we skip historical data, the first sequence created has as last element the first scenario data point
    if skip_historical:
        X_train_to_return = np.array([X_train_np[i:i+slider] for i in range(len_historical-slider+1, time_length-slider+1)])
    # Else we just go through the whole dataset historical + scenario (does not matter in the case of 'hist-GHG' and 'hist_aer')
    else:
        X_train_to_return = np.array([X_train_np[i:i+slider] for i in range(0, time_length-slider+1)])
    
    return X_train_to_return 


def output_for_training(Y_train_xr, var, skip_historical=False, len_historical=None): 
    Y_train_np = Y_train_xr[var].data
    
    time_length = Y_train_np.shape[0]
    
    # If we skip historical data, the first sequence created has as target element the first scenario data point
    if skip_historical:
        Y_train_to_return = np.array([[Y_train_np[i+slider-1]] for i in range(len_historical-slider+1, time_length-slider+1)])
    # Else we just go through the whole dataset historical + scenario (does not matter in the case of 'hist-GHG' and 'hist_aer')
    else:
        Y_train_to_return = np.array([[Y_train_np[i+slider-1]] for i in range(0, time_length-slider+1)])
    
    return Y_train_to_return

def create_training_data(simus, var_to_predict='tas', time_slider=None):
    X_train = []
    Y_train = []

    for i, simu in enumerate(simus):

        input_name = 'inputs_' + simu + '.nc'
        output_name = 'outputs_' + simu + '.nc'
    
        # Just load hist data in these cases 'hist-GHG' and 'hist-aer'
        if 'hist' in simu:
            # load inputs 
            input_xr = xr.open_dataset(data_path + input_name)
                
            # load outputs                                                             
            output_xr = xr.open_dataset(data_path + output_name).mean(dim='member')
            output_xr = output_xr.assign({"pr": output_xr.pr * 86400,
                                          "pr90": output_xr.pr90 * 86400}).rename({'lon':'longitude', 
                                                                                   'lat': 'latitude'}).transpose('time','latitude', 'longitude').drop(['quantile'])
        
        # Concatenate with historical data in the case of scenario 'ssp126', 'ssp370' and 'ssp585'
        else:
            # load inputs 
            input_xr = xr.open_mfdataset([data_path + 'inputs_historical.nc', 
                                        data_path + input_name]).compute()
                
            # load outputs                                                             
            output_xr = xr.concat([xr.open_dataset(data_path + 'outputs_historical.nc').mean(dim='member'),
                                   xr.open_dataset(data_path + output_name).mean(dim='member')],
                                   dim='time').compute()
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
        array = np.concatenate([X_train[i][var].data for i in [0, 3, 4]] + 
                               [X_train[i][var].sel(time=slice(len_historical, None)).data for i in range(1, 3)])
        #print((array.mean(), array.std()))
        meanstd_inputs[var] = (array.mean(), array.std())
    
    # normalize input data 
    X_train_norm = [] 
    for i, train_xr in enumerate(X_train): 
        for var in ['CO2', 'CH4', 'SO2', 'BC']: 
            var_dims = train_xr[var].dims
            train_xr=train_xr.assign({var: (var_dims, normalize(train_xr[var].data, var, meanstd_inputs))}) 
        X_train_norm.append(train_xr)

    X_train_all = np.concatenate([input_for_training(X_train_norm[i], skip_historical=(i<2), len_historical=len_historical) for i in range(len(simus))], axis = 0)
    Y_train_all = np.concatenate([output_for_training(Y_train[i], var_to_predict, skip_historical=(i<2), len_historical=len_historical) for i in range(len(simus))], axis=0)
    return X_train_all, Y_train_all, meanstd_inputs
