'''
Created on 2016

@author: Graham Reid

Looks up the weights of the various features and translates them back into text.
Since our data is composed of the output of multiple vectorizers, it is
necessary to deconstruct the data into the various field arrays prior to using
the vectorizers to decode the sparse representation
'''

import scipy as sp
import sys
import pickle
import numpy as np

number_of_words = 2000

vectorizer_location = '../../data/vectorizers/vectorizers_binary_bigram.pk'
model_location = '../../models/lasoo/lasoo_model_5e-06_5592.15783974.pk'
output_location = '../../data/weighted_words/weighted_words.pk'

title_array = []
location_array = []
description_array = []
contract_type_array = []
contract_time_array = []
company_array = []
category_array = []
source_array = []

'''
put the best features in the best arrays (those that predict high salaries)
'''

best_title_array = []
best_location_array = []
best_description_array = []
best_contract_type_array = []
best_contract_time_array = []
best_company_array = []
best_category_array = []
best_source_array = []

best_array = [best_title_array, best_location_array, best_description_array, \
best_contract_type_array, best_contract_time_array, best_company_array, \
best_category_array, best_source_array]

'''
bag of words of best features
'''

best_bag_array = []
best_bag_weight_array = []

'''
put the worst features in the best arrays (those that predict low salaries)
'''

worst_title_array = []
worst_location_array = []
worst_description_array = []
worst_contract_type_array = []
worst_contract_time_array = []
worst_company_array = []
worst_category_array = []
worst_source_array = []

worst_array = [worst_title_array, worst_location_array, worst_description_array, \
worst_contract_type_array, worst_contract_time_array, worst_company_array, \
worst_category_array, worst_source_array]

'''
bag of words of worst features
'''

worst_bag_array = []
worst_bag_weight_array = []

#data_array = []
#salary_array = []

print 'loading vectorizers'
with open(vectorizer_location, 'rb') as input:
    title_vectorizer = pickle.load(input)
    description_vectorizer = pickle.load(input)
    location_vectorizer = pickle.load(input)
    contract_time_vectorizer = pickle.load(input)
    contract_type_vectorizer = pickle.load(input)
    company_vectorizer = pickle.load(input)
    category_vectorizer = pickle.load(input)
    source_vectorizer = pickle.load(input)
print 'loaded vectorizers'

print 'loading model'
with open(model_location, 'rb') as input:
    model = pickle.load(input)
print 'loaded model'

title_num = len(title_vectorizer.get_feature_names())
description_num = len(description_vectorizer.get_feature_names())
location_num = len(location_vectorizer.get_feature_names())
contract_time_num = len(contract_time_vectorizer.get_feature_names())
contract_type_num = len(contract_type_vectorizer.get_feature_names())
company_num = len(company_vectorizer.get_feature_names())
category_num = len(category_vectorizer.get_feature_names())
source_num = len(source_vectorizer.get_feature_names())

'''
locations corresponding to the start of each vectorizer's data
'''

loc1 = 0
loc2 = loc1 + title_num
loc3 = loc2 + description_num
loc4 = loc3 + location_num
loc5 = loc4 + contract_time_num
loc6 = loc5 + contract_type_num
loc7 = loc6 + company_num
loc8 = loc7 + category_num
loc9 = loc8 + source_num

'''
setting up the arrays for decoding the training data
'''

loc_array = np.array([loc1,loc2,loc3,loc4,loc5,loc6,loc7,loc8,loc9])

vec_array = ([title_vectorizer.get_feature_names(), \
description_vectorizer.get_feature_names(), \
location_vectorizer.get_feature_names(), \
contract_time_vectorizer.get_feature_names(), \
contract_type_vectorizer.get_feature_names(), \
company_vectorizer.get_feature_names(), \
category_vectorizer.get_feature_names(), \
source_vectorizer.get_feature_names()])

descrip_array = ['(title)', '(description)', '(location)', '(contract_time)', \
'(contract_type)', '(company)', '(category)', '(source)']

coeff = model.coef_
nonzero = coeff[np.abs(coeff) > 0]
coeff_argsort = np.argsort(np.abs(coeff))[::-1]
print np.count_nonzero(nonzero)
print coeff[coeff_argsort]
print coeff_argsort
print loc_array

number_of_words = min(np.count_nonzero(nonzero), number_of_words)
for ngram in xrange(0,number_of_words):
    word = coeff_argsort[ngram]
    i = -1
    while (word - loc_array[i+1]) >= 0:
        #print i+1, word - loc_array[i+1], loc_array[i+1]
        i += 1

    idx = word - loc_array[i]
    print 'weight:', coeff[word], '\tfield:', descrip_array[i], '\tword:', \
    (vec_array[i])[idx]
    
    if coeff[word] < 0:
        worst_array[i].append([str((vec_array[i])[idx]), coeff[word]]) 
        worst_bag_array.append(str((vec_array[i])[idx]))
        worst_bag_weight_array.append(coeff[word])
    else:
        best_array[i].append([str((vec_array[i])[idx]), coeff[word]]) 
        best_bag_array.append(str((vec_array[i])[idx]))
        best_bag_weight_array.append(coeff[word])  

for i in xrange(0, len(descrip_array)) :
    print 'best ' + descrip_array[i] + ': '
    print best_array[i]
    print ''
print ''

for i in xrange(0, len(descrip_array)) :
    print 'worst ' + descrip_array[i] + ': '
    print worst_array[i]
    print ''
print ''

'''
dump the data
'''

with open(output_location, 'wb') as output:
    pickle.dump(best_bag_array, output, pickle.HIGHEST_PROTOCOL)
    pickle.dump(best_bag_weight_array, output, pickle.HIGHEST_PROTOCOL)
    pickle.dump(worst_bag_array, output, pickle.HIGHEST_PROTOCOL)
    pickle.dump(worst_bag_weight_array, output, pickle.HIGHEST_PROTOCOL)
    


