import sys

import numpy as np
np.random.seed(5)
np.set_printoptions(threshold=sys.maxsize)
from sklearn.model_selection import train_test_split

import pandas as pd

import argparse

parser = argparse.ArgumentParser(description='Read csv file to run RFR')
parser.add_argument('csv_file', metavar='csv_filename', type=str, help='csv data file path')
args = parser.parse_args()

# Read data file 
choi_df = pd.read_csv(args.csv_file,encoding = "ISO-8859-1")
choi_df = choi_df.fillna(0.0)
data = np.asarray(choi_df.values.tolist())

# Record list of variables
variables = choi_df.columns.tolist()[:-1]

# Ready to do operations
independent_vars = data[:,:-1].astype('double') # Last column is dependent
dependent_vars = data[:,-1].astype('double')

num_rows = np.shape(independent_vars)[0]
num_vars = np.shape(independent_vars)[1]

# Do some randomization
idx = np.random.choice(np.arange(num_vars), num_vars, replace=False)
shuffled_vars = independent_vars[:,idx].astype('double') # Last column is dependent

# Shuffling variable names
new_variables = []
for i in range(len(variables)):
    id_val = idx[i]
    new_variables.append(variables[id_val])

# Make 40 folds
# Split the data into training and testing sets

num_folds = 40
for fold in range(num_folds):
    shuffled_vars_train, shuffled_vars_test, dependent_vars_train, dependent_vars_test = train_test_split(shuffled_vars, dependent_vars.reshape(-1,1), train_size=0.8, random_state=fold)

    train_data = np.concatenate((shuffled_vars_train,dependent_vars_train),axis=-1)
    test_data = np.concatenate((shuffled_vars_test,dependent_vars_test),axis=-1)

    # Save the file
    np.savetxt('folds/train_'+f'{fold:02}'+'.csv',train_data,delimiter=',')
    np.savetxt('folds/test_'+f'{fold:02}'+'.csv',test_data,delimiter=',')
    
    