'''
    Optimized partitioning algorithms for 2D signal data.
    Generalized n-dimensional algorithms are found in partition_recursive.py

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
from sys import float_info
from coreset.utils.formats import SparseData
from coreset.utils.partition_recursive import balanced_partition_1d

class Slices:
    ''' Collection of one-dimensional slices of 2D data.
        self.cols - list of column indices from original data
        self.counts - list containing information about each column '''
    def __init__(self, ymin=None, ymax=None, cols=[], counts=[]):
        self.ymin, self.ymax, self.cols = ymin, ymax, cols
        self.counts = counts

    def copy(self):
        assert(self.ymin is not None)
        return Slices(self.ymin, self.ymax, np.copy(self.cols),
                      self.counts)

    def reset_cols(self):
        self.cols = []
        self.counts = []

    def append(self, col, count):
        self.cols.append(col)
        self.counts.append(count)


class Partition:
    ''' Partition algorithms optimized for 2D signal data. '''
    def __init__(self, unique_columns, target_variance):
        self.unique_columns = sorted(unique_columns)
        self.target_variance = max(target_variance, float_info.epsilon)
        self.column_to_idx = {
            col : idx for (idx, col) in enumerate(unique_columns) }
        self.idx_to_column = {
            idx : col for (idx, col) in enumerate(unique_columns) }
        self.reset()

    def  add_row(self, X, Y):
        ''' Adds a slice of the data to the partitioning algorithm '''
        # X contains sparse data; all points are from same row
        if self.slices is None:
            self.slices = Slices()
        else:
            self.prev_slices = self.slices.copy()

        self.slices.reset_cols()
        if self.slices.ymin is None:
            self.slices.ymin = X[0][0]
        self.slices.ymax = X[0][0]

        j = 0
        n = len(X)
        # running mean and variance calulation state variables
        ys, y2s, ns = 0, 0, 0
        # for all sparse points in row
        for i in range(n):
            # column of the current point
            col = self.column_to_idx[X[i][1]]
            # value of the current point
            value = Y[i]
            # There may be missing columns because of sparsity.
            # Processing all columns until the current one
            while j < col:
                # updating state variables according to the missing columns
                ys, y2s, ns = self.process_column(j, ys, y2s, ns)
                j += 1
            # updating state variables according to the current column
            ys, y2s, ns = self.add_element(col, value, ys, y2s, ns)
            j = col + 1
        # After the last point in a row, processing remaining
        # missing sparse columns
        while j <= self.column_to_idx[self.unique_columns[-1]]:
            ys, y2s, ns = self.process_column(j, ys, y2s, ns)
            j += 1

    def process_column(self, col, ys, y2s, ns):
        ''' Updates internal parameters after adding a new data '''
        # Note: self.ys, self.y2s, self.ns contain column state variables;
        # arguments ys, y2s, ns contain current slice state variables
        # Adding current column to the slice
        ys, y2s, ns = (
            ys + self.ys[col], y2s + self.y2s[col], ns + self.ns[col])
        # Checking slice variance
        variance = Partition.calculate_variance(ys, y2s, ns)
        if variance > self.target_variance:
            # Extra checks for over condition. If over the required variance,
            # Using previous state of the slice before adding the column.
            is_prev_exists = self.slices.cols and (
                self.slices.cols[-1] != self.idx_to_column[col])
            if is_prev_exists:
                # creating a single slice; count of slices is 1
                self.slices.append(self.idx_to_column[col], 1)
                ys, y2s, ns = self.ys[col], self.y2s[col], self.ns[col]
            else:
                if col != len(self.unique_columns) - 1:
                    # if slice contains a single column (no previous slice),
                    # splitting it vertically to achieve target variance.
                    # Therefore, one or more slices are added:
                    #   count = variance / target_variance
                    self.slices.append(
                        self.idx_to_column[col + 1],
                        int(np.ceil(variance / self.target_variance)))
                    ys, y2s, ns = 0, 0, 0
        if col == len(self.unique_columns) - 1:
            # if last column, appending current slice ending at the
            # last column (np.inf as bound). Also if variance is over
            # target variance, splitting the slice:
            #   count = variance / target_variance
            variance = Partition.calculate_variance(ys, y2s, ns)
            self.slices.append(
                np.inf, int(np.ceil(variance / self.target_variance)))

        return ys, y2s, ns

    def add_element(self, col, value, ys, y2s, ns):
        ''' Adds a single element of the data to the partitioning algorithm '''
        # updating the running mean and variance state
        self.ys[col] += value
        self.y2s[col] += value ** 2
        self.ns[col] += 1
        # still running processing of the columns for possible splitting
        return self.process_column(col, ys, y2s, ns)

    @staticmethod
    def calculate_variance(ys, y2s, ns):
        ''' Calculates variance using a one-pass streaming algorithm '''
        # Using state variables to calculate final mean and variance
        if ns <= 1:
            return 0
        mean = ys / ns
        variance = y2s - ns * mean ** 2
        return variance

    @staticmethod
    def concatenate(X, Y, slices, gamma, sigma):
        ''' Unites slices and repartitions them using 1-dimensional
            partitioning algorithm '''
        result = []
        # for each horizontal full slice
        for s in slices:
            i = - np.inf
            # all rows of the slice
            rows_idx = np.logical_and(X[:,0] >= s.ymin, X[:,0] <= s.ymax)
            # selecting all slice rows
            Xs, Ys = X[rows_idx], Y[rows_idx]
            # iterating over groups of columns in a slice (vertical sub-slices)
            for j, count in zip(s.cols, s.counts):
                # group of columns in the subslice
                cols_idx = np.logical_and(Xs[:,1] >= i, Xs[:,1] < j)
                # selecting a complete subslice
                d = SparseData(Xs[cols_idx], Ys[cols_idx])
                if np.sum(cols_idx) > 0:
                    if count == 1:
                        # if the subslice had low variance
                        result.append(d)
                    else:
                        # if the column subslice had to be split due to
                        # exceeding the target variance - running the 1D
                        # optimal partitioning algorithm for splitting it
                        result.extend(balanced_partition_1d(d, gamma, sigma))
                i = j
        return result


    def reset(self):
        ''' Empties all counters of the algorithm '''
        self.slices, self.prev_slices = None, None
        self.ys = np.zeros_like(self.unique_columns)
        self.y2s = np.zeros_like(self.unique_columns)
        self.ns = np.zeros_like(self.unique_columns)


def balanced_partition_2d(data, gamma, sigma):
    ''' Balanced partition algorithm optimized for 2D signal data. '''
    n, d = data.get_shape()
    needed = 1 / (gamma ** (d-1))
    target_variance = gamma ** d * sigma

    if n <= needed:
        # no point in algorithmic splitting as we will get single points anyway
        return data.get_single_points_split()

    sortidx = np.lexsort(np.rot90(data.X))
    X, Y = data.X[sortidx], data.Y[sortidx]
    i = 0
    unique_columns = np.unique(X[:, 1])
    partition = Partition(unique_columns, target_variance)
    result = []
    while i < n:
        curr_row = X[i][0]
        j = i
        while (j < n) and (X[j][0] == curr_row):
            j += 1
        # Adding rows while not meeting the splitting condition
        partition.add_row(X[i:j], Y[i:j])
        # Condition: total number of subslices exceeds the target
        if np.sum(partition.slices.counts) > needed:
            # split must be done; checjing whether to use current or prev slice
            if partition.prev_slices is not None:
                result.append(partition.prev_slices)
                partition.reset()
                partition.add_row(X[i:j], Y[i:j])
            else:
                result.append(partition.slices)
                partition.reset()
        # same checks for end of data; as in the internal algorithm
        # that does subslice splitting. Here if end of the data is reached
        # adding remaining slices anyway.
        if j == n and partition.slices is not None:
            result.append(partition.slices)
        i = j
    # Concatenate performs final union of slices and balanced splitting
    return Partition.concatenate(X, Y, result, gamma, sigma)
