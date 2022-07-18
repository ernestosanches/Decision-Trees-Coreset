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

import os
import numpy as np
import pandas as pd
from glob import iglob
from enum import Enum
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

class DatasetType(Enum):
    RECONSTRUCTION = 1 # missing values reconstruction
    REGRESSION = 2 # simple regression

def convert_to_dataset_type(X, Y, n_bins, dataset_type):
    '''
        The dataset is transformed into one of the two forms:
        1. Missing values reconstruction task.
            - The task is to predict missing values in the features matrix X.
            - The original targets Y are not used.
            - The returned value consists of all combination of indices i, j
              in the matrix X and actual values to be predicted in the vector Y.
        2. Regression task on an input signal:
            - The task is to predict the original value Y given two input
              features that are arranged as a signal using data quantization.
    '''
    def count_unique_values_in_columns(data):
        data_sorted = np.sort(data, axis=0)
        return (data_sorted[1:, :] != data_sorted[:-1, :]).sum(axis=0) + 1

    X, Y = scale_data(X, Y)
    if n_bins is not None:
        X, Y = quantize_data(X, Y, n_bins)
    if dataset_type == DatasetType.RECONSTRUCTION:
        # discarding original Y and creating new Y for the task of predicting
        # image(signal) coordinates --> image(signal) value
        X, Y = convert_to_sklearn(X)
    elif dataset_type == DatasetType.REGRESSION:
        # selecting 2 features with most unique valurs to build a signal
        # can use PCA instead as well
        uniques_per_column = count_unique_values_in_columns(X)
        best_columns_idx = np.argsort(uniques_per_column)[:-3:-1]
        X = X[:, best_columns_idx]
    else:
        raise ValueError(dataset_type)
    return X, Y

def get_california_housing(
        n_bins=None, dataset_type=DatasetType.RECONSTRUCTION):
    '''
        Returns sklearn California House price prediction dataset.
    '''
    X, Y = fetch_california_housing(return_X_y=True)
    return convert_to_dataset_type(X, Y, n_bins, dataset_type)

def get_gesture_phase(n_bins=None):
    def read_multiple_csv(filepath_pattern):
        all_files = iglob(filepath_pattern)
        df_from_each_file = map(pd.read_csv, all_files)
        return pd.concat(df_from_each_file, ignore_index=True)
    # this is a classification dataset, so only the RECONSTRUCTION task
    # is supported
    dataset_type=DatasetType.RECONSTRUCTION
    filepath_pattern = "data/gesture_phase_dataset/*_raw.csv"
    data = read_multiple_csv(filepath_pattern).dropna()
    X, Y = data.values[:, :-2], np.zeros(len(data)) # not using the Y
    return convert_to_dataset_type(X, Y, n_bins, dataset_type)

def get_air_quality(
        n_bins=None, dataset_type=DatasetType.RECONSTRUCTION):
    filepath = "data/AirQualityUCI/AirQualityUCI.csv"
    data = (pd
            .read_csv(filepath, sep=';', decimal=',')
            .drop(["Date", "Time"], axis=1)
            .dropna(how="all")
            .dropna(how="any", subset=['CO(GT)'])
            .fillna(0))
    X, Y = data.values[:, 1:], data.values[:, 0]
    return convert_to_dataset_type(X, Y, n_bins, dataset_type)
