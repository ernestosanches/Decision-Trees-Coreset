'''
    Helper classes for working with different data formats.
    Overview:
        - Data of dimensionality d can be a signal or non-signal.
        -- Signal means that full data is represented by a dense
           d-dimensional tensor, and each entry in this tensor has a value.
        -- Non-signal means that data is represented as a list of d-dimensional
           vectors. Vectors may not cover the whole space.

        - Data can be sparse or dense.
        -- Dense means that set of possible values is discrete and uniform.
           i'th coordinate can take values from 0 to n_i. There are no
           missing values.
        -- Sparse means that set of possible values is discrete, but
           non-uniform. i'th coordinate can take values from a predefined
           set of possible values. This set of values is determined from
           the data (such as by calling np.unique(X) to get unique values)

    Supported formats:
        1. SparseData - sparse signal data. Data is signal, but the
        discrete set of possible values is determined by the unique values
        on each axis in the input data.
        For example, 2-dimensional input data
            X = [[1,3], [2,3], [3,4]]
            Y = [10, 11, 12]
        will result in a signal defined over grid:
            x_1 in {1,2,3}
            x_2 in {3,4}
        where there is a defined value of Y in positions
            {1,3} - y=10, {2,3} - y=11, {3,4} - y=12
        and there is no defined value of Y in positions
            {1,4}, {2,4}, {3,3}

        2. CoresetData - a weighted subset of the original data,
           such as SparseData.

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
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from coreset.utils.booster import Fast_Caratheodory
from lightgbm import LGBMRegressor


class SparseData:
    ''' Sparse signal data '''
    def __init__(self, X, Y, uid=None, valid=None):
        self.X, self.Y, self.uid, self.valid = X, Y, uid, valid
        self.weights = None
        self.curr_dim = None
        self.set_curr_dim_and_sort(0)

    def init_valid(self):
        ''' Initializes and valid mask to define all data as valid '''
        self.valid = np.ones(len(self.X), dtype=bool)

    def init_uid(self):
        ''' Initializes indices after sorting the data on first dimension '''
        self.set_curr_dim_and_sort(0, stable=True)
        self.uid = np.arange(len(self.Y))

    def get_data(self):
        ''' Placeholder method to override in child classes'''
        return self

    def filter_valid(self):
        ''' Removes all elements marked as non-valid '''
        keep = self.valid
        self.X, self.Y, self.uid, self.valid = (
            self.X[keep], self.Y[keep], self.uid[keep], self.valid[keep])

    def split(self, test_size=0.25):
        ''' Train-test split '''
        idx_train, idx_test = train_test_split(range(len(self.Y)),
                                               test_size=test_size)
        return self.split_idx(idx_train, idx_test)

    def split_idx(self, idx_train, idx_test):
        ''' Train-test split given indices of items in train and test sets '''
        X_train, X_test, Y_train, Y_test = (
            self.X[idx_train], self.X[idx_test],
            self.Y[idx_train], self.Y[idx_test])
        uid_train, uid_test = ((None, None) if self.uid is None else
                               (self.uid[idx_train], self.uid[idx_test]))
        valid_train, valid_test = ((None, None) if self.valid is None else
                               (self.valid[idx_train], self.valid[idx_test]))
        return (SparseData(X_train, Y_train, uid_train, valid_train),
                SparseData(X_test, Y_test, uid_test, valid_test))

    def split_image(self, test_size, H, W, square_size):
        ''' Split image into train and test sets using random squares.
            Squares of size (H,W) are randomly placed on image
            to determine the hidden testing set of test_size '''
        N = H * W
        test_mask = np.zeros((H, W), dtype=bool)
        while test_mask.sum() < N * test_size:
            x0, x1 = np.random.randint(0, H), np.random.randint(0, W)
            test_mask[x0:x0+square_size, x1:x1+square_size] = 1
            #test_mask[:, x1:x1+square_size] = 1
        train_mask = np.logical_not(test_mask)
        data_train, data_test = self.split_idx(train_mask.T.flatten(), test_mask.T.flatten())
        return data_train, data_test

    def fit_dt(self, sample_weight, k, ModelClass=DecisionTreeRegressor):
        ''' Fits a decision tree with predefined parameters.
            A single changeable parameter: number of leaves = k '''
        model = ModelClass(max_leaf_nodes=k)
        return model.fit(self.X, self.Y, sample_weight=sample_weight)

    def fit_lgb(self, sample_weight, k, n_estimators=100):
        ''' Fits a gradient boosted trees model with predefined parameters.
            Goal is to produce a k-segmentation.
            n_estimator trees are fitted each with k/n_estimators leaves '''
        model = LGBMRegressor(num_leaves=k//n_estimators,
                              n_estimators=n_estimators)
        #model = ModelClass(max_depth=20, max_leaf_nodes=k)
        return model.fit(self.X, self.Y)


    def fit_rf(self, sample_weight, k, n_estimators=100):
        ''' Fits a random forest model with predefined parameters.
            Goal is to produce a k-segmentation.
            n_estimator trees are fitted each with k/n_estimators leaves '''
        model = RandomForestRegressor(max_leaf_nodes = k//n_estimators,
                                      n_estimators=n_estimators)
        return model.fit(self.X, self.Y)

    def predict_dt(self, model):
        ''' Applies the model on currently stored data '''
        return model.predict(self.X)

    def mse(self, y_pred):
        ''' Calculates MSE error given model predictions '''
        return mean_squared_error(self.Y, y_pred)

    def std_dev_valid(self):
        ''' Calculates standard deviation of Y, considering only
            items marked by the valid mask '''
        y_valid = self.Y[self.valid]
        return np.std(y_valid)

    def to_sklearn(self):
        ''' Converts the data into an (X,Y) pair '''
        return self.X, self.Y

    def get_sample(self, size):
        ''' Returns a random sample of stored data '''
        is_replace = (size > len(self.Y))
        idx_sample = np.random.choice(len(self.Y), size, replace=is_replace)
        X_s, Y_s, uid_s, valid_s = (
            self.X[idx_sample], self.Y[idx_sample],
            None if self.uid is None else self.uid[idx_sample],
            None if self.valid is None else self.valid[idx_sample])
        return SparseData(X_s, Y_s, uid_s, valid_s)

    def get_shape(self):
        ''' Returns stored data shape '''
        return self.X.shape

    def set_curr_dim_and_sort(self, dim, stable=False):
        ''' Sorts stored data on a gived dimension index.
            Optionally applies a stable sorting algorithm to prevent
            reshuffling of the data on subsequent sorts on other dimensions '''
        if self.curr_dim == dim:
            return # already sorted by this dimension
        self.curr_dim = dim
        sort_idx = np.argsort(self.X[:, dim],
                              kind="stable" if stable else None)
        self.X, self.Y, self.uid, self.valid = (
            self.X[sort_idx], self.Y[sort_idx],
            None if self.uid is None else self.uid[sort_idx],
            None if self.valid is None else self.valid[sort_idx])
        n = len(self.Y)
        # stores indices of elements where the data on given dimension
        # changes compared to the previous element, for algorithms speedup
        # This variable stores changes on the self.curr_dim dimension
        self.changes_on_dim = ([0] +
                               [i for i in range(1, n)
                                if self.X[i, dim] != self.X[i-1, dim]])

    def size_on_dim(self):
        ''' Returns number of unique values on given dimension '''
        return len(self.changes_on_dim)

    def idx_to_block(self, idx):
        ''' Returns first element in the data when idx'th change
            occurs on the self.curr_dim dimension '''
        size = self.size_on_dim()
        n = len(self.Y)
        return n if idx >= size else self.changes_on_dim[idx]

    def get_slice(self, start_idx, end_idx):
        ''' Returns a subset of data, consisting on blocks of
            non-changing elements on the dimension self.curr_dim,
            given starting and ending indices of such blocks. '''
        start_idx = self.idx_to_block(start_idx)
        end_idx = self.idx_to_block(end_idx)
        return SparseData(
            self.X[start_idx : end_idx, :], self.Y[start_idx : end_idx],
            None if self.uid is None else self.uid[start_idx : end_idx],
            None if self.valid is None else self.valid[start_idx : end_idx])

    def get_single_points_split(self):
        ''' Returns a set of slices containing single blocks of
            non-changing elements on dimension self.curr_dim '''
        n = len(self.Y)
        return [self.get_slice(i, i+1) for i in range(n)]

    def get_Y(self):
        ''' Returns the Y values of the data '''
        return self.Y

    def get_random_coreset(self, size):
        ''' Returns a random subset of data of given size.
            Data is reweighed according to the ratio between
            the full and given sizes. '''
        sample = self.get_sample(size)
        n = len(self.Y)
        return CoresetData(sample.X, sample.Y,
                           np.full(size, n / size), size)

    def get_caratheodory_coreset(self, duplicate=False):
        ''' Computes a Caratheory coreset using algorithm from utils/booster.py
            Optionally duplicates the coreset data using a smoothing algorithm
            described in the paper.
            -- Duplication is only needed to use the coreset
            without implementing a custom Coreset-Cost algorithm
            (as described in the paper) inside sklearn's Decision Tree model.
        '''
        X_r = []
        Y_r = []
        w_r = []
        c_size = 4
        sortidx = np.lexsort(np.rot90(self.X))
        x = self.X[sortidx]
        y = self.Y[sortidx]
        y_col = y.reshape((-1, 1))

        # [y, y**2, 1] matrix used to compute the Caratheodory coreset
        xy = np.hstack([y_col, y_col**2, np.ones_like(y_col)])
        # Computes the coreset
        c_weights = Fast_Caratheodory(xy, np.ones(len(xy)), c_size)
        filled_weight = 0 # total weight of filled points so far
        for i in range(len(c_weights)):
            c_weight = c_weights[i]
            if not duplicate:
                if c_weight > 0:
                    X_r.append(x[i:i+1, :])
                    Y_r.append(y[i])
                    w_r.append(c_weight)
            else:
                # Applies filling algorithm described in the paper
                # to be able to use original DecisionTree algorithms
                # without cost function modification.
                while c_weight > 0 and filled_weight < len(y):
                    j = int(filled_weight)
                    X_r.append(x[j:j+1, :])
                    Y_r.append(y[i])
                    if c_weight > 1:
                        curr_weight = 1
                    else:
                        curr_weight = c_weight
                    w_r.append(curr_weight)
                    c_weight -= curr_weight
                    filled_weight += curr_weight
        X_tilda = np.vstack(X_r)
        Y_tilda = np.array(Y_r)
        weights = np.array(w_r)
        return CoresetData(X_tilda, Y_tilda, weights, c_size)


class CoresetData:
    ''' Defines a subset of data
        Apart from the data itself, contains information about
        weights and a displayed size of the coreset '''
    def __init__(self, X, Y, weights, size_partition):
        self.X, self.Y, self.weights, self.size_partition = (
            X, Y, weights, size_partition)

    def get_data(self):
        ''' Returns the underlying data '''
        return SparseData(self.X, self.Y, do_sort=False)

    def fit_dt(self, k):
        ''' Fits the decision tree on the data '''
        data, weights = self.get_data(), self.weights
        return data.fit_dt(weights, k)

    def fit_lgb(self, k):
        ''' Fits the Gradient Boosted Trees model on the data '''
        data, weights = self.get_data(), self.weights
        return data.fit_lgb(weights, k)

    def fit_rf(self, k):
        ''' Fits the Random Forest model on the data '''
        data, weights = self.get_data(), self.weights
        return data.fit_rf(weights, k)

    @staticmethod
    def concatenate(iterable, keep_size=False):
        ''' Combines a collection of coresets into a single coreset '''
        X, Y, weights, size_partition = zip(*[
            (coreset.X, coreset.Y, coreset.weights,
             coreset.size_partition if keep_size else 1)
            for coreset in iterable])
        return CoresetData(
            np.vstack(X), np.concatenate(Y),
            np.concatenate(weights), np.sum(size_partition))

    @property
    def size(self):
        assert(len(self.X) == len(self.Y) == len(self.weights))
        return len(self.X)
