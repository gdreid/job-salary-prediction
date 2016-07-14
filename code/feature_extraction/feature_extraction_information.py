'''
Created on 2016

@author: Graham Reid

Here we compute the information gain as a possible metric for fitting a model.
Information gain is defined as:

IG(t) = H(t) - H(x|c)
IG(t) = p(c_i)*log(p(c_i)) + p(t)*p(c_i|t)*log(p(c_i|t)) 
      + p(!t)*p(c_i|!t)*log(p(c_i|!t))

Note there is a big error in how I am doing this. Haven't worked out the bug,
but it is resulting in negative information gains
'''

import scipy as sp
import sys
import pickle
import numpy as np
from scipy.sparse import *

with open('../../data/data_arrays/data_short_binary_bigram.pk', 'rb') as input:
    data = pickle.load(input).tocsr()
    salaries = pickle.load(input)

test_valid_size = 0.3

data_train, data_test_valid, salaries_train, salaries_test_valid = \
train_test_split(data, salaries, test_size=test_valid_size, random_state=42)

data_valid, data_test, salaries_valid, salaries_test = train_test_split(
    data_test_valid, salaries_test_valid, test_size=0.5, random_state=42)

print 'data_test_valid shape:', data_test_valid.shape
print 'data_train shape:',      data_train.shape
print 'data_valid shape:',      data_valid.shape
print 'data_test shape:',       data_test.shape

salaries_train = np.array(salaries_train)
salaries_test = np.array(salaries_test)
salaries_valid = np.array(salaries_valid)

order = np.argsort(salaries_train)
divisions = 2
num_features = data_train.shape[1]
num_samples = data_train.shape[0]
division_len = num_samples/divisions+1
counts = np.array([np.zeros(num_features) for i in range(divisions)])

data_train = data_train[order,:]
salaries_train = salaries_train[order]


'''
Here we are splitting the data into 'division' categories.
we will use these categories to determine features with maximal
information gain relative to other features

at the end of this step, each row of the counts array has the counts of each
feature in the given percentile
'''
count = 0
for i in xrange(num_samples):
    if (i%1000 == 0) :
        print i/1000
    div = min(i/division_len, divisions-1)
    counts[div] += data_train.getrow(i).toarray()[0]

print 'counts loaded'

'total number of counts'
total = np.zeros(num_features)

'initialize the information gain array'
information_gain = np.zeros(num_features)
for i in range(divisions):
    total += counts[i]
frac = total / float(num_samples)
frac_counts = counts / float(division_len)

'''
Entropy of a boolean variable is just
-p(x=1)*log(p(x=1)) -p(x=0)*log(p(x=0)) 
'''
def entropy(frac):
    ent = -(frac*np.log2(frac + 1e-15) + (1.0-frac)*np.log2((1.0-frac)+1e-15))
    return ent



entropy_feature = entropy(frac)

temp1 = np.zeros((divisions, num_features))
temp1[:,:] = 1.0/float(divisions)*frac_counts[:,:]*np.log2(1.0/(frac_counts[:,:]+1e-10))

temp2 = np.zeros((divisions, num_features))
temp2[:,:] = 1.0/float(divisions)*(1.0-frac_counts[:,:])*np.log2(1.0/(1.0-frac_counts[:,:]+1e-10))

cond_entropy = np.zeros(num_features)
for i in xrange(divisions):
    cond_entropy[:] += temp1[i,:] + temp2[i,:]

information_gain = entropy_feature - cond_entropy
print temp1[1,:]
print temp2[1,:]
print entropy_feature

print information_gain
print np.min(information_gain)
