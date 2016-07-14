'''
Created on 2013-02-22

@author: Graham Reid
'''

import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
import csv
import string
import sys

if len(sys.argv) != 5:
    print 'usage data_partitioner <file name> <output dir> <valid fraction> <test fraction>'
    sys.exit(0)

file_path  = sys.argv[1]
output_dir = sys.argv[2]
valid_frac = float(sys.argv[3])
test_frac  = float(sys.argv[4])

data_file = open(file_path)
reader = csv.reader(data_file)
headers = reader.next()

file_len = 0
for line in reader:
    file_len = file_len +1

np.random.seed(42)
order_param = np.random.random(file_len)
ordering    = np.argsort(order_param)

data_file.seek(0)
reader.next()

data_raw     = [reader.next() for i in xrange(file_len)]
data_ordered = data_raw[ordering,:]

test_end  = max(int(test_frac * file_len),1)
valid_end = test_end + max(int(valid_frac*file_len),1)

data_test    = data_ordered[0:test_end,         :]
data_valid   = data_ordered[test_end:valid_end, :]
data_train   = data_ordered[valid_end:,         :]
data_retrain = data_ordered[test_end:,          :]

def write_csv(file_name, data_array) :
    with open(file_name, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for row in data_array:
    	    writer.writerow(row)

write_csv(output_dir+'/data_test.csv',    data_test);
write_csv(output_dir+'/data_valid.csv',   data_valid);
write_csv(output_dir+'/data_train.csv',   data_train);
write_csv(output_dir+'/data_retrain.csv', data_retrain);

