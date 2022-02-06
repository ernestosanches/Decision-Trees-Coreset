'''
    Helper functions for loading datasets.

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
from PIL import Image
from sklearn.datasets import (fetch_california_housing, make_circles,
                              make_moons, make_blobs)
from sklearn.preprocessing import StandardScaler

def convert_to_sklearn(A):
    '''
    Creates a dataset X, Y from a dense data or an image A.
        X: dense coordinates in the matrix A.
        Y: value in the matrix A at the corresponding coordinates.
    '''
    X = np.concatenate([np.expand_dims(x, axis=-1)
                        for x in np.meshgrid(*map(range, A.shape))],
                       axis = -1).reshape((-1, A.ndim))
    Y = A.T.flatten()
    return X, Y

def scale_data(X, Y):
    '''
    Applies standard scaling to X, Y.
    '''
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    Y = scaler.fit_transform(Y.reshape((-1,1)))[:,0]
    return X, Y

def quantize_data(X, Y, bins):
    '''
    Applies quantization to the values of X into a given number of bins
    to allow sparse storage of the signal data.
    '''
    X = (X - X.min()) / (X.max() - X.min())
    X *= bins - 1
    X = np.asarray(X, dtype=int)
    unique_idx = np.unique(X, axis=0, return_index=True)[-1]
    return X[unique_idx, :], Y[unique_idx]

def get_circles(n_points, n_bins, factor=0.4, random_state=1):
    ''' Generaties an artificial dataset consisting of circles '''
    X, Y = make_circles(n_points, noise=0.2, factor=factor,
                        random_state=random_state)
    X, Y = quantize_data(X, Y, n_bins)
    return X, Y

def get_moons(n_points, n_bins, random_state=1):
    ''' Generaties an artificial dataset consisting of moons '''
    X, Y = make_moons(n_points, noise=0.2,
                      random_state=random_state)
    X, Y = quantize_data(X, Y, n_bins)
    return X, Y

def get_blobs(n_points, n_bins, random_state=1):
    ''' Generaties an artificial dataset consisting of blobs '''    
    X, Y = make_blobs(
        n_points, cluster_std=[1.0, 2.0, 0.5],
        random_state=random_state)
    X, Y = quantize_data(X, Y, n_bins)
    return X, Y

def get_image_data(filepath, ymin=None, ymax=None, xmin=None, xmax=None,
                   sklearn_format=True):
    '''
    Returns dense image data
    '''
    A = np.asarray(Image.open(filepath).convert('L'))

    if sklearn_format:
        X, Y = convert_to_sklearn(A[ymin:ymax, xmin:xmax])
        return X, Y
    else:
        return A

def get_california_housing():
    '''
    Returns sklearn California House price prediction dataset.
    The task is to predict missing values in the features matrix X.
    The targets Y are not used.
    - The returned value consists of all combination of indices i, j 
      in the matrix X and actual values to be predicted in the vector Y.
    '''
    X, Y_discarded = fetch_california_housing(return_X_y=True)
    X, Y_discarded = scale_data(X, Y_discarded)
    X, Y = convert_to_sklearn(X)
    return X, Y        
