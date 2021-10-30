'''
    Helper functions for working with signal data.

*******************************************************************************
MIT License

Copyright (c) 2021 Ibrahim Jubran, Ernesto Evgeniy Sanches Shayda, 
                   Ilan Newman, Dan Feldman

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*******************************************************************************
'''

####################################### NOTES #################################
# - Please cite our paper when using the code: 
#             "Coresets for Decision Trees of Signals" (NeurIPS'21) 
#             Ibrahim Jubran, Ernesto Evgeniy Sanches Shayda, 
#             Ilan Newman, Dan Feldman
#
###############################################################################


import numpy as np

def create_full_slice(X):
    '''
    Returns an indexing boolean array for X full of True values
    '''
    return np.arange(len(X))

def take(myslice, start, end, sortidx):
    '''
    Changes myslice to include existing values only from start to end when 
    sorted by sortidx
    '''
    new_slice = myslice.copy()
    new_slice[sortidx[:start]] = False
    new_slice[sortidx[end:]] = False
    return new_slice

def get_dim_sort(X, dim):
    '''
    Returns an indexing array that would sort X on elements on dim axis
    '''
    return np.argsort(X[:, dim])

def get_original_A_shape(changes):
    '''
    Returns a shape that sparse data would have had if it was represented
    by a dense matrix A. On each axis the shape equals to a number of unique
    coordinate values
    '''
    return tuple(changes_on_axis.sum() for changes_on_axis in changes)

def slice_center(X, myslice):
    '''
    Returns an n-dimensional vector: coordinate of block center
    '''
    return X[myslice].mean(axis=0)

def cost(A):
    '''
    Returns sum of squared errors cost of approximating a tensor with its mean
    '''
    if len(A) == 0:
        return 0
    return ((A - A.mean()) ** 2).sum()

def std_dev_valid(Y, valid):
    ''' 
    Returns sum of squared errors cost of approximating a masked slice valid 
    of data A with its mean.
    '''
    data = Y[valid] 
    return cost(data)   

def get_best_dimensions(X, ndims):
    '''
    Returns indices of estimated ndims best dimensions of X for coreset 
    construction.
    Will be added in the extension of the paper for higher-dimensional data.
    For now, selecting first two dimensions to avoid exponential dependency
    of the running time on the dimensionality of the data.
    '''
    return list(range(ndims))