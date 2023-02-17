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
import timeit
####################################### NOTES #################################
# - Please cite our paper when using the code:
#             "Coresets for Decision Trees of Signals" (NeurIPS'21)
#             Ibrahim Jubran, Ernesto Evgeniy Sanches Shayda,
#             Ilan Newman, Dan Feldman
#
###############################################################################

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from glob import iglob
from enum import Enum
from PIL import Image
from sklearn.datasets import (fetch_california_housing, make_circles,
                              make_moons, make_blobs)
from sklearn.preprocessing import StandardScaler
from line_profiler_pycharm import profile


def convert_to_sklearn(A):
    '''
    Creates a dataset X, Y from a dense data or an image A.
        X: dense coordinates in the matrix A.
        Y: value in the matrix A at the corresponding coordinates.
    '''
    '''
    # Alon Latman
    Function takes a multi-dimensional array A and flattens it into a two-dimensional array of coordinates and corresponding
    values.
    The np.meshgrid function is used to generate a grid of coordinates for each dimension of A. The *map(range, A.shape)
    part of the code generates a sequence of the dimensions of A, and the range function is applied to each element of 
    the sequence to generate a sequence of numbers from 0 to the corresponding dimension of A. The np.meshgrid function 
    then generates a set of coordinate arrays for each dimension, using these sequences as input.
    The np.expand_dims function is used to add an additional dimension to each of the coordinate arrays, so that they can
    be concatenated together using np.concatenate. The axis=-1 parameter specifies that the new dimension should be added
    at the end of the array.
    the flatten method is used to flatten the values of A into a 1D array, and the resulting arrays are returned as
    a finite ordered list.
    '''
    # uses multi-treading
    # the original code runs O(n), which means that it will increase linearly with the size of the input data.
    # with ThreadPoolExecutor() as executor:
    #     X_ = [executor.submit(np.expand_dims, x, axis=-1) for x in np.meshgrid(*map(range, A.shape))]
    #     X = np.concatenate([future.result() for future in X_], axis=-1).reshape((-1, A.ndim))
    #     Y_ = executor.submit(A.T.flatten)
    #     Y = Y_.result()

    X = np.concatenate([np.expand_dims(x, axis=-1)
                        for x in np.meshgrid(*map(range, A.shape))],
                       axis = -1).reshape((-1, A.ndim))
    Y = A.T.flatten()
    return X, Y


def scale_data(X, Y):
    '''
    Applies standard scaling to X, Y.
    '''
    '''
    # Alon Latman
    Function performing standardization on two input datasets, X and Y. Standardization is a way to
    transform data so that it has zero mean and unit variance.
    The StandardScaler class is a preprocessing step that can be used to standardize a dataset. The fit_transform method
    fits the scaler to the data, then standardizes it. The scaler is first fit to the X dataset, then the X dataset is
    transformed (standardized) using the scaler. The scaler is then fit to the Y dataset, and the Y dataset is
    transformed (standardized) using the scaler. Finally, the transformed Y dataset is reshaped into a column vector and
    the first column is selected, then returned along with the transformed X dataset.
    '''
    # with_mean= False could help
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    Y = scaler.fit_transform(Y.reshape((-1, 1)))[:, 0]
    return X, Y


def quantize_data(X, Y, bins):
    '''
    Applies quantization to the values of X into a given number of bins
    to allow sparse storage of the signal data.
    '''
    '''
    #Alon Latman
    Function takes a multi-dimensional array X and a 1D array Y, and bins the data in X into a specified number of bins.
    The bins parameter specifies the number of bins to use.
    The function first normalizes the data in X by subtracting the minimum value and dividing by the range (maximum value
    minus minimum value). 
    This scales the data so that it is between 0 and 1. The function then scales the data to the number of bins specified
    by the bins parameter, by multiplying by bins - 1.
    The function then converts the data in X to integer values, using np.asarray and the dtype parameter. It then uses
    np.unique to find the unique rows of X and their indices. The function returns the unique rows of X and Y, using the
    indices of the unique rows to index the original arrays.
    '''

    X = (X - X.min()) / (X.max() - X.min())
    X *= bins - 1
    X = np.asarray(X, dtype=int)
    unique_idx = np.unique(X, axis=0, return_index=True)[-1]
    return X[unique_idx, :], Y[unique_idx]


def get_circles(n_points, n_bins, factor=0.4, random_state=1):
    ''' Generaties an artificial dataset consisting of circles '''
    '''
    # Alon Latman
    This function generates an artificial dataset consisting of circles. It does this by calling make_circles,
    which generates synthetic data with two classes that are arranged in circles The "n_points" parameter specifies the 
    number of points to generate, and the factor parameter controls the size of the circles.
    The random_state parameter determines the random seed used to generate the data.
    After generating the data, the get_circles function calls the quantize_data function to bin the data into a specified
    number of bins.
    The "n_bins" parameter specifies the number of bins to use, and the X and Y arguments represent the data and labels,
    respectively.
    function returns the binned data and labels as a finite ordered list.
    '''

    X, Y = make_circles(n_points, noise=0.2, factor=factor,
                        random_state=random_state)
    X, Y = quantize_data(X, Y, n_bins)
    return X, Y


def get_moons(n_points, n_bins, random_state=1):
    ''' Generaties an artificial dataset consisting of moons '''
    '''
    #Alon Latman
    Function generating synthetic two-dimensional classification data using the make_moons function. 
    The make_moons function generates a two-dimensional dataset with two classes shaped in the form of two interleaving 
    half circles. The number of points in the dataset is specified by the n_points parameter, and the amount of noise in
    the data is controlled by the noise parameter. The random_state parameter controls the random number generator used 
    to generate the data, allowing the data to be reproducible.
    '''
    X, Y = make_moons(n_points, noise=0.2,
                      random_state=random_state)
    X, Y = quantize_data(X, Y, n_bins)
    return X, Y


def get_blobs(n_points, n_bins, random_state=1):
    ''' Generaties an artificial dataset consisting of blobs '''
    '''
    #Alon Latman
    Function generating synthetic two-dimensional classification data using the make_blobs.
    The make_blobs function generates a two-dimensional dataset with a specified number of classes, where each class is
    a Gaussian cluster. The number of points in the dataset is specified by the n_points parameter, and the standard
    deviation of each cluster can be controlled using the cluster_std parameter. The random_state parameter controls the
    random number generator used to generate the data, allowing the data to be reproducible.
    The generated data is then passed to a function called quantize_data, which appears to be another function in the
    code. 
    The purpose of this function is not clear from the provided code snippet. 
    The quantize_data function is called with the X and Y datasets and the n_bins parameter, and the resulting
    transformed datasets are returned.
    '''
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
    '''
    #Alon Latman
    Function loading image data from a file specified by the filepath parameter and returning the image
    data as a NumPy array. 
    The image is read using the Image.open function from the Python Pillow library and is converted to grayscale using
    the convert method.
    The image data could be cropped using the ymin, ymax, xmin, and xmax parameters, which specify the minimum and maximum
    row and column indices to include in the cropped image.
    The sklearn_format parameter controls the format of the returned image data. If sklearn_format is True, the image
    data is passed to a function called convert_to_sklearn, which appears to be another function in the code.  
    The convert_to_sklearn function is called with the cropped image data, and the resulting transformed data is returned.
    If sklearn_format is False, the original image data is returned as a NumPy array.
    '''
    A = np.asarray(Image.open(filepath).convert('L'))

    if sklearn_format:
        X, Y = convert_to_sklearn(A[ymin:ymax, xmin:xmax])
        return X, Y
    else:
        return A


class DatasetType(Enum):
    RECONSTRUCTION = 1  # missing values reconstruction
    REGRESSION = 2  # simple regression


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
    '''
    #Alon Latman
    Function performing preprocessing on input datasets X and Y and returning the preprocessed data.
    The n_bins and dataset_type parameters control the type of preprocessing applied to the data.
    The scale_data function, is called with the input datasets X and Y.
    If n_bins is not None, the quantize_data function is called with the input datasets X and Y and the n_bins parameter, 
    and the resulting transformed datasets are returned.
    If dataset_type is equal to the value DatasetType.RECONSTRUCTION, the convert_to_sklearn function is called with the
    transformed X dataset, and the resulting transformed data is returned as the X dataset. The original Y dataset is
    discarded, and a new Y dataset is created for the task of predicting image (signal) coordinates from image (signal) 
    values.
    If dataset_type is equal to the value DatasetType.REGRESSION, the input X dataset is transformed by selecting the
    two columns with the most unique values and returning them as the X dataset. The original Y dataset is returned as
    is.
    If dataset_type has any other value, a ValueError is raised.
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
    '''
    #Alon Latman
    Function is loading a public dataset for predicting house prices in California and returning the data in
    a specified format.
    The n_bins and dataset_type parameters control the type of preprocessing applied to the data.
    The California housing dataset is loaded using the fetch_california_housing function, 
    which returns the input features (X) and target values (Y) as separate NumPy arrays.
    The convert_to_dataset_type function, which appears to be another function in the code, is called with the input
    datasets X and Y, the n_bins parameter, and the dataset_type parameter. 
    The convert_to_dataset_type function returns the transformed datasets X and Y.
    '''
    X, Y = fetch_california_housing(return_X_y=True)
    return convert_to_dataset_type(X, Y, n_bins, dataset_type)


def get_gesture_phase(n_bins=None):
    '''
    #Alon Latman
    This function is loading a dataset from multiple CSV files and returning it in a specified format.
    The dataset consists of a collection of time series data and is being used for the task of reconstruction.
    The function begins by defining an inner function read_multiple_csv that takes a filepath pattern as input and
    returns a concatenated DataFrame containing the data from all CSV files that match the pattern.
    The function then uses the iglob function from the glob module to generate a list of file paths that match the
    pattern, and the pd.read_csv function to read the data from each file into a DataFrame. The DataFrames are
    then concatenated into a single DataFrame using the pd.concat function.
    the function sets the dataset_type variable to DatasetType.RECONSTRUCTION and the filepath_pattern variable to
    "data/gesture_phase_dataset/*_raw.csv".
    It then calls the read_multiple_csv function with the filepath_pattern as input and assigns the returned DataFrame
    to the data variable.
    The function then removes any rows with missing values from the data DataFrame using the dropna method.
    Finally, the function splits the data DataFrame into the X and Y variables, with X containing the time series data
    and Y being a zero array with the same length as X. The function then calls the convert_to_dataset_type function
    with the X, Y, n_bins, and dataset_type variables as input and returns the result.
    '''

    def read_multiple_csv(filepath_pattern):
        all_files = iglob(filepath_pattern)
        df_from_each_file = map(pd.read_csv, all_files)
        return pd.concat(df_from_each_file, ignore_index=True)

    # this is a classification dataset, so only the RECONSTRUCTION task
    # is supported
    dataset_type = DatasetType.RECONSTRUCTION
    filepath_pattern = "data/gesture_phase_dataset/*_raw.csv"
    data = read_multiple_csv(filepath_pattern).dropna()
    X, Y = data.values[:, :-2], np.zeros(len(data))  # not using the Y
    return convert_to_dataset_type(X, Y, n_bins, dataset_type)


def get_air_quality(
        n_bins=None, dataset_type=DatasetType.RECONSTRUCTION):
    '''
    #Alon Latman
    This function is loading a dataset from a single CSV file and returning it in a specified format.
    The dataset consists of air quality data, including measurements of various gases and meteorological variables,
    and is being used for the task of reconstruction.
    The function begins by setting the filepath variable to the location of the CSV file containing the data and then
    uses the pd.read_csv function to read the data into a DataFrame.
    The sep and decimal arguments are used to specify the delimiter and decimal character used in the CSV file,
    respectively.
    the function removes the "Date" and "Time" columns from the DataFrame using the drop method and removes any
    rows with all missing values using the dropna method. It also removes any rows with missing values in the 'CO(GT)'
    column using the subset argument of the dropna method. Finally, the function fills any remaining missing values with
    zeros using the fillna method.
    The function then splits the data in the DataFrame into the X and Y variables, with X containing the measurements of
    the various gases and meteorological variables and Y containing the CO(GT) measurement. The function then calls the
    convert_to_dataset_type function with the X, Y, n_bins, and dataset_type variables as input and returns the result.
    '''
    filepath = "data/AirQualityUCI/AirQualityUCI.csv"
    data = (pd
            .read_csv(filepath, sep=';', decimal=',')
            .drop(["Date", "Time"], axis=1)
            .dropna(how="all")
            .dropna(how="any", subset=['CO(GT)'])
            .fillna(0))
    X, Y = data.values[:, 1:], data.values[:, 0]
    return convert_to_dataset_type(X, Y, n_bins, dataset_type)
