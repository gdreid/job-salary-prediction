'''
Created on 2016

@author: Graham Reid

Only real difference between this and the lasoo model is that the linear lasoo
model does not log scale the data. If you look at the data as a histogram, you
will see that it is highly skewed towards low paying jobs.
'''

import scipy as sp
import sys
from scipy.sparse import hstack
import pickle
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
import numpy as np
from scipy.sparse import *
import matplotlib.pyplot as plt

#previously used model_data_binary_bigrams.pk
with open('../../data/data_arrays/data_binary_bigram.pk', 'rb') as input:
    data = pickle.load(input).tocsr()
    salaries = pickle.load(input)

scale_fac = 1.0/50000.0
salaries = np.exp(salaries)*scale_fac
test_valid_size = 0.2

data_train, data_test_valid, salaries_train, salaries_test_valid = \
train_test_split(data, salaries, test_size=test_valid_size, random_state=42)

data_valid, data_test, salaries_valid, salaries_test = train_test_split(
    data_test_valid, salaries_test_valid, test_size=0.5, random_state=42)

print 'data_test_valid shape:', data_test_valid.shape
print 'data_train shape:',      data_train.shape
print 'data_valid shape:',      data_valid.shape
print 'data_test shape:',       data_test.shape

print 'fitting model'

'''
regularization parameter alpha for l1 regularization
'''

alpha = 1.0e-5
warm_start = True
it_since_min = 0
delta_it = 10
min_error =  float("inf")
i = 0

it_array   = np.array([])
valid_array = np.array([])

plt.ion()
fig=plt.figure()


print 'alpha: ', alpha

feature_selector = Lasso(alpha=alpha, fit_intercept=True, max_iter=1, \
warm_start=warm_start, positive=False, tol=0.0)

while it_since_min < delta_it:
    i += 1

    print 'fitting'
    sys.stdout.flush()

    feature_selector.fit(data_train, salaries_train)

    print 'predicting'
    sys.stdout.flush()

    salaries_pred = feature_selector.predict(data_valid)/scale_fac
    error = np.average(np.abs(salaries_valid/scale_fac - salaries_pred))
    average_salary = np.average(salaries/scale_fac)
    coeff = feature_selector.coef_
    intercept = feature_selector.intercept_

    it_array = np.append(it_array, [i])
    valid_array = np.append(valid_array, [error])

    plt.clf()
    plt.plot(it_array, valid_array)
    plt.xlabel('iteration count')
    plt.ylabel('mean validation error')
    plt.title('Lasso (linear) Model Selection alpha = ' + str(alpha))
    plt.pause(0.001)
    fig.savefig('../../plots/lasoo_linear_model_selection_' + str(alpha) + '_.pdf')

    print 'it: ', i, 'average error: ', error, " alpha: ", alpha, " non zero: ", \
    np.count_nonzero(coeff)
    sys.stdout.flush()

    if error < min_error :
        min_error = error
        it_since_min = 0

        output_path = '../../models/lasoo_linear_best_' + str(alpha) + '_' '.pk' 

        with open (output_path, 'wb') as output:
	    pickle.dump(feature_selector, output, pickle.HIGHEST_PROTOCOL)
    else :
        it_since_min += 1

