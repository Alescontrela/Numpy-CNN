'''
Description: Utility methods for a Convolutional Neural Network

Author: Alejandro Escontrela
Version: V.1.
Date: June 12th, 2018
'''
import numpy as np

def initializeFilterNormal(num_filters, filter_dims,input_depth, SCALE = 1):
    '''
    Initialize filter for convolutional step using a normal distribution with a specified standard deviation.
    
    Inputs:
    |--num_filters: number of desired filters for the convolutional layer
    |--filter_dims: dimensions of the filter
    |--input_depth: depth of the input
    
    Outputs:
    |--filter
    
    NOTE: input_depth parameter must be equal to the depth of the input image filters are being applied to
    '''
    
    stddev = SCALE *1 /np.sqrt(input_depth*filter_dims*filter_dims)
    
    return np.random.normal(loc = 0, scale = stddev, size = (num_filters, input_depth, filter_dims, filter_dims))

def initializeWeightsXavier(num_neurons, num_inputs):
    '''
    Initialize the weights for fully connected step using Xavier initialization. Var(W) = 2/num_neurons
    
    inputs:
    |--num_neurons: the number of outputs of this dense layer
    |--num_inputs: input_depth
    
    outputs:
    |--weights
    '''
    return np.random.randn(num_neurons, num_inputs).astype(np.float32) * 0.01

def nanargmax(arr):
    '''
    Find the maximum value of the input array that isn't of type np.nan
    
    inputs: 
    |--arr: input array to find the max value indices of
    
    outputs:
    |--multi_idx: (a,b) ordered pair of max value's index
    '''
    idx = np.nanargmax(arr, axis = None)
    multi_idx = np.unravel_index(idx, (arr.shape))
    
    return multi_idx