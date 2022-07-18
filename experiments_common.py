'''
    Common functions for running the experiments.

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
from sklearn.metrics import mean_squared_error

def evaluate_on_full_data(X_train, Y_train, X_test, Y_test, k):
    ''' Model training on full data to compare with coreset results,
        returns mean squared error on the testing set after training
        on the full training dataset '''
    model = DecisionTreeRegressor(max_leaf_nodes=k)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    return mean_squared_error(Y_test, Y_pred)

def evaluate_on_coreset(coreset, X_train, Y_train, X_test, Y_test, k):
    ''' Model training on coreset, returns mean squared error on the
        testing set after training on a subset of the training dataset '''
    X_coreset, Y_coreset, weights = coreset.X, coreset.Y, coreset.weights
    model_coreset = DecisionTreeRegressor(max_leaf_nodes=k)
    model_coreset.fit(X_coreset, Y_coreset, sample_weight=weights)
    Y_pred_coreset = model_coreset.predict(X_test)
    return mean_squared_error(Y_test, Y_pred_coreset)
