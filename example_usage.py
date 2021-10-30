'''
    This is an example code that shows how to construct the coreset, 
    train a model on coreset data, and evaluate the results.

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


from coreset.decision_tree import dt_coreset
from coreset.utils.formats import SparseData
from data.datasets import scale_data, get_circles
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    # Data
    X, Y = get_circles(50000, 300)
    X, Y = scale_data(X, Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
    data_train = SparseData(X_train, Y_train)
    
    # Coreset construction for different epsilons.
    # Note: in practice we get much smaller error than the given epsilon.
    # Tune the epsilon to get the desired coreset size and check for practical 
    # error on the validation set.
    epsilons = [0.04, 0.07, 0.1, 0.2] 
    k = 20
    coreset_verbose = False # True for printing additional information
    
    # Use_exact_bicriteria = True will result in training a single decision
    # tree on full data for evaluating problem complexity (approximating 
    # the optimal cost, called OPT, on given data).
    # Otherwise, OPT is evaluated using (alpha, beta)-bicriteria approximation.
    # In practice when using coreset for hyperparameters tuning, where
    # hundreds of hyperparameters are checked, training a single decision 
    # tree doesn't increases much the computational time, but allows to obtain
    # better coreset, increasing overall accuracy.
    use_exact_bicriteria_values=[False, True]
    
    for use_exact_bicriteria in use_exact_bicriteria_values:
        print("\nConstructing coresets using exact bicriteria: {}\n".format(
            use_exact_bicriteria))
        for epsilon in epsilons:
            print("\nConstructing coreset for epsilon = {}\n".format(epsilon))
            coreset, coreset_dup = dt_coreset(
                data_train, k, epsilon, verbose=coreset_verbose,
                use_exact_bicriteria=use_exact_bicriteria)
                
            # Using the smoothed coreset (coreset_dup). To use the original 
            # coreset, sklearn DecisionTreeRegressor class must be modified 
            # to work with the Fitting-Loss algorithm as the cost function,
            # as described in the supplementary materials of the paper.
            X_coreset, Y_coreset, weights = (
                coreset_dup.X, coreset_dup.Y, coreset_dup.weights)
            
            # Model training on coreset
            model_coreset = DecisionTreeRegressor(max_leaf_nodes=k)
            model_coreset.fit(X_coreset, Y_coreset, sample_weight=weights)
            Y_pred_coreset = model_coreset.predict(X_test)
            error_coreset = mean_squared_error(Y_test, Y_pred_coreset)
        
            # Model training on full data to compare errors
            model = DecisionTreeRegressor(max_leaf_nodes=k)
            model.fit(X_train, Y_train)
            Y_pred = model.predict(X_test)
            error_full = mean_squared_error(Y_test, Y_pred)
        
            # printing the results
            print(("Testing error when training on 100% of data " + 
                  "({} examples): {:.5f}").format(
                len(X_train), error_full))
            print(("Testing error when training on {:.2f}% of data " + 
                  "(coreset of {} examples): {:.5f}").format(
                coreset.size / float(len(X_train)) * 100, coreset.size, 
                error_coreset))
