'''
Description: forward operations for a convolutional neural network

Author: Alejandro Escontrela
Version: 1.0
Date: June 12th, 2018
'''
import numpy as np

def relu(X):
    '''
    pass input through Rectified Linear Unit activation function
    
    inputs:
    |--X: matrix to pass through nonlinearity
    
    outputs:
    |--X: activated input
    '''
    X[X<=0] = 0
    return X

def maxpool(X, f, s):
    '''
    maxpooling operation
    
    inputs:
    |--X: input to be downsampled
    |--f: kernel size to use when downsampling
    |--s: stride to use
    
    outputs:
    |--downsampled matrix
    '''
    (l, w, w) = X.shape
    w1 = int((w - f)/s) + 1
    out = np.zeros((l, w1, w1))
    for jj in range(l):
        curr_x = out_x = 0
        while curr_x + f <= w:
            curr_y = out_y = 0
            while curr_y + f <= w:
                out[jj, out_x, out_y] = np.max(X[jj, curr_x:curr_x+f, curr_y:curr_y+f])
                curr_y += s
                out_y += 1
            curr_x += s
            out_x += 1
    
    return out

def softmax(z):
    '''
    Softmax operation on dense layer outputs
    
    inputs:
    |--z: prediction to take soft maximum of
    
    outputs:
    |--softmax predictions
    '''
    out = np.exp(z)
    return (out/np.sum(out))

def logloss(label, probs):
    '''
    Categorical Cross-entropy loss operation
    
    inputs:
    |--label: correct outputs
    |--probs: softmax predictions output by network
    
    outputs:
    |--loss
    '''
    return -np.sum(label * np.log(probs))

