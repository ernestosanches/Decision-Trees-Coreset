'''
    Coreset construction for Decision Trees of Signals

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


from sklearn.tree import DecisionTreeRegressor
from coreset.utils.formats import CoresetData
from coreset.utils.partition_2d import balanced_partition_2d
from coreset.utils.partition_recursive import (
    balanced_partition, bicriteria)

def dt_coreset(data, k, epsilon, 
               use_caratheodory=True, use_exact_bicriteria=False,
               verbose=False, return_slices=False):
    '''
    Returns an epsilon-coreset for k-segmentation of the data

    Parameters:
        use_exact_bicriteria:
            Proposed approximate bicriteria algorithm can be turned off 
            by setting the parameter use_exact_bicriteria=True. In this case, 
            a single DecisionTreeRegressor is trained on the data to provide 
            better approximation at the cost of a longer time to construct
            the coreset.
            Note, that if coreset will be used to train forests consisting 
            of hundreds of trees, or to tune hyperparameters over hundreds
            of possible values, constructing a single decision tree
            will not affect the overall running time much.
        use_caratheodory:
            After computing balanced partition, a 4-points 1-mean Caratheodory 
            coreset for each partition cell is computed if use_caratheodory is 
            is True. Otherwise, a random 4-points sample is computed for each 
            cell.
            use_caratheodory=True should be used both in theory and practice,
            unless experimenting with the coreset itself.
        return_slices:
            If return_slices is false, a united coreset for the whole data 
            is returned. If return_slices is true, raw cells of balanced 
            partition are returned.
            return_slices=False should be used, unless experimenting with the 
            coreset.
    Return value:       
        The function returns two coresets: original small coreset, and 
        a smoothed version of it with duplicated points. Original coreset
        can be used with DecisionTreeRegressor that has a custom Fitting-loss
        algorithm implemented as a cost function, as described in the paper. 
        To avoid modification of the class from sklearn library, a smoothed 
        coreset can be used instead.
    '''
    n, d = data.X.shape

    if use_exact_bicriteria:
        m = DecisionTreeRegressor(max_leaf_nodes=k)
        X, Y = data.to_sklearn()
        m.fit(X, Y)
        pred = m.predict(X)
        sigma_approx = ((Y - pred) ** 2).sum()
        bicriteria_segments_count = k
    else:
        segmentation_approx, sigma_approx = bicriteria(data, k)
        bicriteria_segments_count = len(segmentation_approx)

    # Note: Theoretical values for alpha, beta are too pessimistic.
    # Practically, the coreset works with less-restricted partitioning
    #   alpha = k * np.log(n)
    #   beta = k * np.log(n) ** 2
    alpha, beta = 1, 1
    # Note: Theoretical value for gamma is too pessimistic:    
    #   gamma = epsilon ** 2 / (beta * k)
    gamma = epsilon / beta 
    sigma = sigma_approx / alpha
    
    if verbose:
        print("bicriteria: epsilon={} sigma={}".format(
            epsilon, sigma_approx))
        print("bicriteria segments :", bicriteria_segments_count)
        print(("balanced partition: " + 
               "alpha={:.5f}, beta={:.5f}," +
               "gamma={:.5f}, sigma={:.5f}:").format(
            alpha, beta, gamma, sigma))

    fast_coreset = (data.X.ndim == 2) # for 2D data
    if fast_coreset:
        slices = balanced_partition_2d(data, gamma, sigma)
    else:
        slices = balanced_partition(data, gamma, sigma)

    if verbose:
        print("balanced partition segments: {}".format(
              len(list(filter(lambda s: len(s.X) > 0, slices)))))

    if return_slices:
        return slices
 
    if use_caratheodory:
        coreset = CoresetData.concatenate(
            [data.get_caratheodory_coreset(duplicate=False)
             for data in slices if len(data.X)])
        coreset_dup = CoresetData.concatenate(
            [data.get_caratheodory_coreset(duplicate=True)
             for data in slices if len(data.X)]) 
    else:
        coreset_size = 4
        coreset = CoresetData.concatenate(
            [data.get_random_coreset(coreset_size) for data in slices])
        coreset_dup = coreset

    return coreset, coreset_dup
