'''
Created on 2013-03-23

@author: Graham Reid
'''

import scipy as sp
from scipy.sparse import *
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import  GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor
from sklearn.cross_validation import train_test_split
import numpy as np
import project_utils

data_path = '../data/model_data_binary.pk'
feature_selector_path = '../models/unigrams/linear/lasoo_100_0.0001_6944.45330344_0.2.pk'
information_gain_path = '../models/unigrams/information_gain/information_gain.pk'
output_path = '../models/RFR_unigram_0.0001_'
coeff_to_load = 800
information_to_load = 1200
for count in range(1):
    features = []

    with open(data_path, 'rb') as input:
        data_old = pickle.load(input).tocsc()
        print type(data_old), data_old.shape
        salaries = pickle.load(input)
        
    with open(feature_selector_path, 'rb') as input:
        feature_selector = pickle.load(input)
    with open(information_gain_path, 'rb') as input:
        information_gain = pickle.load(input)
    
    coeff = np.abs(feature_selector.coef_)
    coeff_order = np.argsort(coeff)[::-1] 
    for i in coeff_order:
        if (coeff[i] > 0 and len(features) < coeff_to_load):
            features.append(i)
    
    print 'feature len: ', len(features)
    information_gain_order = np.argsort(information_gain)[::-1]
    for i in information_gain_order:
        if (len(features) < coeff_to_load + information_to_load and information_gain[i] > 0 and project_utils.contains(features, i) == False):
            features.append(i)
        
    print 'feature len: ', len(features)
    data = [data_old.getcol(features[0])]
    for i in features[1:]:
        data.append(data_old.getcol(i))
    
    data_new = project_utils.create_sparse_hor(data).tocsr()
    print data_new.shape
    print 'features selected, reduced data array created'
    
    test_size = 0.1
    data_train, data_test, salaries_train, salaries_test = train_test_split(data_new, salaries, test_size=test_size, random_state=42)
    salaries_train = np.exp(salaries_train)
    salaries_test = np.exp(salaries_test)
    
    print data_train.shape
    print data_test.shape
    
    n_trees = 50
    max_depth = None
    for j in range(1):
        print 'n_trees ', n_trees
        print 'max_depth ', max_depth
        learning_rate = 0.1
        print 'fitting'

        reg = RandomForestRegressor(n_estimators=n_trees, criterion='mse', max_depth=max_depth, max_features='sqrt', bootstrap=False, n_jobs=1)

        reg.fit(data_train.toarray(), salaries_train)
        print 'predicting'
        salaries_pred = reg.predict(data_test.toarray())
        error = np.average(np.abs(salaries_test - salaries_pred))
        print 'RFR Error: ', error#, ' learning rate: ', learning_rate
        with open (output_path + str(n_trees) + '_' + str(coeff_to_load) + '_' + str(information_to_load) + '_' + str(max_depth) + '_' + str(test_size) + '_' + str(error) + '.pk', 'wb') as output:
            pickle.dump(reg, output, pickle.HIGHEST_PROTOCOL)
    
