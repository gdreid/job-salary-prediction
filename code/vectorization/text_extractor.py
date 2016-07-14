'''
Created on 2016

@author: Graham Reid

Builds a 2-gram vectorizer using scikit learn count vectorizer. Only really
interesting thing here is that I didn't concatenate all of the fields together.
This helps to preserve context.
'''

import random
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
import csv
import string

trainPath = '../../data/data_raw.csv'
dataFile = open(trainPath)
reader = csv.reader(dataFile)
headers = reader.next()

target_index = headers.index('SalaryNormalized')
title_index = headers.index('Title')
description_index = headers.index('FullDescription')
location_index = headers.index('LocationRaw')
contract_type_index = headers.index('ContractType')
contract_time_index = headers.index('ContractTime')
company_index = headers.index('Company')
category_index = headers.index('Category')
source_index = headers.index('SourceName')

file_len = 0
for line in reader:
    file_len = file_len +1

dataFile.seek(0)
reader.next()
salary_array = []

title_array = []
location_array = []
description_array = []
contract_type_array = []
contract_time_array = []
company_array = []
category_array = []
source_array = []

title_train_array = []
location_train_array = []
description_train_array = []
contract_type_train_array = []
contract_time_train_array = []
company_train_array = []
category_train_array = []
source_train_array = []


def format_string(field) :
   return field.lower().translate(string.maketrans("",""), string.punctuation)

read_fraction = 1.0
training_indices = np.random.randint(0, file_len, int(file_len*read_fraction))

print 'reading data'

index = 0
for line in reader:
    salary_array.append(np.log(float(line[target_index])))
    
    title_array.append(format_string(line[title_index]))

    description_array.append(format_string(line[description_index]))

    location_array.append(format_string(line[location_index]))

    contract_type_array.append(format_string(line[contract_type_index]))

    contract_time_array.append(format_string(line[contract_time_index]))

    company_array.append(format_string(line[company_index]))

    category_array.append(format_string(line[category_index]))

    source_array.append(format_string(line[source_index]))

    index = index + 1
    
'''
for anything larger than unigrams, descriptions might be too large to be loaded 
into memory all at once. Need to use some smaller read_fraction of documents
'''
for i in training_indices:
    title_train_array.append(title_array[i])
    description_train_array.append(description_array[i])
    location_train_array.append(location_array[i])
    contract_time_train_array.append(contract_time_array[i])
    contract_type_train_array.append(contract_type_array[i])
    company_train_array.append(company_array[i])
    category_train_array.append(category_array[i])
    source_train_array.append(source_array[i])

print 'creating vectorizers'

'''
word must be present in at least this fraction of the docments to be 
vectorized (removes one-time mispellings, etc)
'''

fraction = 1.0/10000.0 

title_vectorizer = CountVectorizer(binary = True, strip_accents='ascii', 
    min_df = int(fraction*file_len), ngram_range = (1,2))
description_vectorizer = CountVectorizer(binary = True, strip_accents='ascii', 
    min_df = int(fraction*file_len*read_fraction), ngram_range = (1,2))
location_vectorizer = CountVectorizer(binary = True, strip_accents='ascii', 
    min_df = int(fraction*file_len), ngram_range = (1,2))
contract_time_vectorizer = CountVectorizer(binary = True, strip_accents='ascii', 
    min_df = int(fraction*file_len), ngram_range = (1,2))
contract_type_vectorizer = CountVectorizer(binary = True, strip_accents='ascii', 
    min_df = int(fraction*file_len), ngram_range = (1,2))
company_vectorizer = CountVectorizer(binary = True, strip_accents='ascii', 
    min_df = int(fraction*file_len), ngram_range = (1,2))
category_vectorizer = CountVectorizer(binary = True, strip_accents='ascii', 
    min_df = int(fraction*file_len), ngram_range = (1,2))
source_vectorizer = CountVectorizer(binary = True, strip_accents='ascii', 
    min_df = int(fraction*file_len))

title_vectorizer.fit(title_array)
title_count_array = title_vectorizer.transform(title_array)
print 'title fit, shape: ', title_count_array.shape

description_vectorizer.fit(description_train_array)
description_count_array = description_vectorizer.transform(description_array)
print 'description fit, shape: ', description_count_array.shape

location_vectorizer.fit(location_array)
location_count_array = location_vectorizer.transform(location_array)
print 'location fit, shape: ', location_count_array.shape

contract_time_vectorizer.fit(contract_time_array)
contract_time_count_array = contract_time_vectorizer.transform(contract_time_array)
print 'contract time fit, shape: ', contract_time_count_array.shape

contract_type_vectorizer.fit(contract_type_array)
contract_type_count_array = contract_type_vectorizer.transform(contract_type_array)
print 'contract type fit, shape: ', contract_type_count_array.shape

company_vectorizer.fit(company_array)
company_count_array = company_vectorizer.transform(company_array)
print 'company fit, shape: ', company_count_array.shape

category_vectorizer.fit(category_array)
category_count_array = category_vectorizer.transform(category_array)
print 'category fit, shape: ', category_count_array.shape

source_vectorizer.fit(source_array)
source_count_array = source_vectorizer.transform(source_array)
print 'source fit, shape: ', source_count_array.shape

data_array = hstack([title_count_array, description_count_array, 
    location_count_array, contract_time_count_array, contract_type_count_array,
    company_count_array, category_count_array, source_count_array])

print 'data stacked'

with open('../../data/data_arrays/data_binary_bigram.pk', 'wb') as output:
    pickle.dump(data_array, output, pickle.HIGHEST_PROTOCOL)
    pickle.dump(salary_array, output, pickle.HIGHEST_PROTOCOL)
    
with open('../../data/vectorizers/vectorizers_binary_bigram.pk', 'wb') as output:
    pickle.dump(title_vectorizer, output, pickle.HIGHEST_PROTOCOL)
    pickle.dump(description_vectorizer, output, pickle.HIGHEST_PROTOCOL)
    pickle.dump(location_vectorizer, output, pickle.HIGHEST_PROTOCOL)
    pickle.dump(contract_time_vectorizer, output, pickle.HIGHEST_PROTOCOL)
    pickle.dump(contract_type_vectorizer, output, pickle.HIGHEST_PROTOCOL)
    pickle.dump(company_vectorizer, output, pickle.HIGHEST_PROTOCOL)
    pickle.dump(category_vectorizer, output, pickle.HIGHEST_PROTOCOL)
    pickle.dump(source_vectorizer, output, pickle.HIGHEST_PROTOCOL)

print 'data_array read and written'
print 'data_array shape: ', data_array.shape
