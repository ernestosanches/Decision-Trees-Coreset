'''
    Optimized partitioning algorithms for high-dimensional signal data.
    Optimized 2-dimensional algorithms are found in partition_2d.py

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
from coreset.utils.running_stats import RunningStats

def get_splits_condition(data,
                         func_on_slice, condition_on_slice, dim,
                         allow_over_condition, calculate_stats,
                         verbose=False):
    '''
    Splits data into subslices.
    Parameters:
        data, dim: input data
        func_on_slice: data --> list of new_slices
            Function for further splitting a current slice.
            Can be used to recursively split on the remaining dimensions.
        condition_on_slice: (data, return_from_func, stats) -> bool
            Function returns a boolean split condition, after reaching
            such condition a new slice should start.
        allow_over_condition: allows to have slices with last element
            included, which made the slice to not satisfy the condition.
        calculate_stats: keeps calculating running_stats of each slice
    Returns:
        list of slices resulting from splitting the data based on condition.
    '''
    n, d = data.get_shape()
    if n == 1:
        # only one element is in the data, returning a single slice
        # containing it
        return [data]
    if func_on_slice is None:
        # No operations are done on a slice if the function is None.
        # Transforming a slice into a single-element list containing
        # itself, for use in the list.extend() method
        func_on_slice = lambda data: [data]
    # Ensuring that the data is sorted according to the current dimension.
    data.set_curr_dim_and_sort(dim, stable=True)
    n = data.size_on_dim()
    # Iterating over blocks of elements of same value on given dimension
    idx_start = idx_end = 0
    is_curr_valid = is_prev_valid = False
    result = []
    rs = RunningStats() if calculate_stats else None
    while idx_end < n:
        # Current set of blocks of data
        curr_data = data.get_slice(idx_start, idx_end + 1)
        if calculate_stats:
            incremented_data = data.get_slice(idx_end, idx_end + 1)
            rs.add_slice(incremented_data.get_Y())
        # Applying a tranformation on the currently considered data
        func_result_curr = func_on_slice(curr_data)
        # checking condition
        condition = condition_on_slice(curr_data, func_result_curr, rs)
        is_prev_valid = is_curr_valid
        is_curr_valid = (condition is not None)
        # helper variable to correctly determine over condition
        curr_increment_len = 1
        if condition:
            if (allow_over_condition or idx_end == idx_start or
                not is_prev_valid):
                # if it is a slice of size 1, adding even if condition
                # just happened but not if it is the last index
                # (adding later in this case)
                if idx_end + 1 != n:
                    result.extend(func_result_curr)
                    idx_start = idx_end + 1
                    if calculate_stats:
                        rs.clear()
            else:
                # previous index had false condition, adding it
                #prev_slice = take(myslice, idx_start, idx_end , sortidx[dim])
                prev_data = data.get_slice(idx_start, idx_end)
                func_result_prev = func_on_slice(prev_data)
                result.extend(func_result_prev)
                idx_start = idx_end
                if calculate_stats:
                    rs.clear()
                curr_increment_len = 0
        if idx_end + curr_increment_len == n and idx_start != n:
            # this is the last index, adding a slice anyway
            if is_curr_valid:
                result.extend(func_result_curr)
        idx_end += curr_increment_len
    return result

def get_splits_valid(data, k, dim):
    ''' Splits the data using get_splits_condition where the condition
        is a maximum number of valid elements in a slice '''
    '''
    # Alon Latman
    This function define a function called get_splits_valid that takes three arguments: data, k, and dim. 
    The get_splits_valid function defines another function called condition_on_slice which takes two arguments: data and 
    ret, and a variable stats. 
    The condition_on_slice function returns None if the sum of the valid column in the data DataFrame is equal to 0, 
    otherwise it returns a boolean value indicating whether the sum of the valid column is greater than v/k.
    The get_splits_valid function then defines a variable v which is equal to the sum of the valid column in the data 
    DataFrame. 
    It defines a variable func_on_slice as None and calls the get_splits_condition function, passing in the data, 
    func_on_slice, condition_on_slice, dim, allow_over_condition=False, and calculate_stats=False arguments.
    The get_splits_valid function then iterates through the slices returned by the get_splits_condition function and 
    calls the filter_valid method on each slice. Finally, the get_splits_valid function returns the list of slices.
    '''
    def condition_on_slice(data, ret, stats):
        valid_count = data.valid.sum()
        return None if valid_count == 0 else (valid_count > v / k)
    v = data.valid.sum()
    func_on_slice = None
    slices = get_splits_condition(data,
                                func_on_slice, condition_on_slice, dim,
                                allow_over_condition=False,
                                calculate_stats=False)
    for s in slices:
        s.filter_valid()
    return slices

def get_splits_variance(data, target_variance, dim):
    ''' Splits the data using get_splits_condition where the condition
        is a maximum variance of a slice'''
    '''
    # Alon Latman
    Function called get_splits_variance that takes two arguments: data and target_variance. 
    The get_splits_variance function defines another function called condition_on_slice which takes three arguments: 
    data, ret, and stats. 
    The condition_on_slice function returns a boolean value indicating whether the s field of the stats object is 
    greater than the target_variance argument.
    The get_splits_variance function then defines a variable func_on_slice as None and calls the get_splits_condition 
    function, passing in the data, func_on_slice, condition_on_slice, dim, allow_over_condition=False, and 
    calculate_stats=True arguments. 
    The get_splits_variance function returns the value returned by the get_splits_condition function.
    '''
    def condition_on_slice(data, ret, stats):
        return stats.s > target_variance
    func_on_slice = None
    return get_splits_condition(data,
                                func_on_slice, condition_on_slice, dim,
                                allow_over_condition=False,
                                calculate_stats=True)


def bicriteria(data, k):
    ''' Bicriteria approximation of an optimal k-segmentation '''
    '''
    # Alon Latman
    Function called bicriteria that takes two arguments: data and k. The function first initializes some variables: n 
    and d are set to the shape of the data DataFrame, result is set to an empty list, and total_variance is set to 0. 
    It then calls the init_uid and init_valid methods on the data DataFrame.
    The function then enters a while loop that continues as long as there are any True values in the valid column of the 
    data DataFrame. Inside the loop, it initializes the Q list to contain the data DataFrame, and then iterates over the 
    dimensions in range(d). 
    For each dimension, it initializes the P list to be empty and then iterates over the current blocks in Q. 
    For each block, it calls the get_splits_valid function to partition the block into sub-blocks, and then adds the 
    sub-blocks to the P list. After all blocks have been partitioned, it sets Q to be equal to P.
    After the loop over dimensions has completed, the function computes the variance of all blocks in Q and adds the 
    blocks with the smallest half of variances to the result list. 
    It also marks all of these blocks as invalid by setting their corresponding values in the valid column to 0, 
    and increments total_variance by the variance of the block.
    The bicriteria function returns the result list and the total_variance value.
    '''
    n, d = data.get_shape()
    result = []
    total_variance = 0
    data.init_uid()
    data.init_valid()
    # While there are valid items; on each iteration half items are invalidated
    while np.any(data.valid):
        # Partition A into k^d block with equal number of valid items
        # Go thtough each dimension and partition only 1 dimension at a time
        Q = [data]
        for dim in range(d):
            P = []
            # Take all current blocks for partitioning on dimension dim
            for data_slice in Q:
                new_slices = get_splits_valid(
                    data_slice, k, dim)
                P.extend(new_slices)
            Q = P
        # compute variance of all blocks and add smallest half to Result
        # mark all such blocks as invalid
        # can use dense std_dev_valid
        variances_raw = [(data_slice.std_dev_valid(), data_slice.uid)
                         for data_slice in Q]
        variances = sorted(variances_raw, key=lambda x: x[0])
        for variance, uid_slice in variances[:max(1, len(variances) // 2)]:
            data.valid[uid_slice] = 0
            result.append(uid_slice)
            total_variance += variance

    return result, total_variance


def balanced_partition_1d(data, gamma, sigma):
    ''' Balanced partition algorithm on a single dimension
        If data is high-dimensional, splits are done on the last dimension in
        the shape of the data. Splits on last dimension are done according
        to condition of a maximum allowed variance of a slice '''
    n, d = data.get_shape()
    target_variance = gamma ** d * sigma
    dim = d-1
    new_slices = get_splits_variance(data, target_variance, dim)
    return new_slices


def balanced_partition(data, gamma, sigma, d_start=0):
    ''' Balanced partition algorithm on high-dimensional data.
        Splits on all dimensions are done according
        to condition of a maximum allowed valid elements in a slice.
        Only on the last dimension the 1-d splitting is done based on a
        condition of maximum variance of a slice '''
    n, d = data.get_shape()
    needed = 1 / (gamma ** (d-d_start-1))

    if n <= needed:
        # no point in algorithmic splitting as we will get single points anyway
        return data.get_single_points_split()

    if d_start == d - 1:
        return balanced_partition_1d(data, gamma, sigma)

    func_on_slice = lambda data: balanced_partition(
        data, gamma, sigma, d_start + 1)
    condition_on_slice = (lambda data, ret, stats:
                          len(ret) > needed)
    new_slices = get_splits_condition(
        data, func_on_slice, condition_on_slice, d_start,
        allow_over_condition=False, calculate_stats=False)
    return new_slices
