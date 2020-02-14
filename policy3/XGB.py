import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost import XGBRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

import sys
np.set_printoptions(threshold=sys.maxsize)

np.random.seed(5)

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

from scipy.stats import mode

num_folds = 40
importance_tracker = np.zeros(shape=(num_folds,len(variables)),dtype='double')
ranking_tracker = np.zeros(shape=(num_folds,len(variables)),dtype='int')
print_ranking = False
plot_ranking = True
hpo = True

for fold in range(num_folds):

    # Split the data into training and testing sets
    shuffled_vars_train, shuffled_vars_test, dependent_vars_train, dependent_vars_test = train_test_split(shuffled_vars, dependent_vars.reshape(-1,1), train_size=0.8, random_state=fold)

    preproc_X = Pipeline([('minmax', MinMaxScaler(feature_range=(0, 1))), ('stdscaler', StandardScaler())])
    preproc_y = Pipeline([('minmax', MinMaxScaler(feature_range=(0, 1))), ('stdscaler', StandardScaler())])

    train_X_p = preproc_X.fit_transform(shuffled_vars_train)#.as_matrix()
    train_y_p = preproc_y.fit_transform(dependent_vars_train.reshape(-1,1))#.as_matrix()
    # test_X_p = preproc_X.transform(shuffled_vars_test)#.as_matrix()
    # test_y_p = preproc_y.transform(dependent_vars_test)#.as_matrix()    

    model = XGBRegressor()
    model.fit(train_X_p, train_y_p[:,0])
    importances = model.feature_importances_

    # Tracking relative importance
    importance_tracker[fold,:] = importances[:]

    # Tracking ranking
    indices = np.argsort(importances)[::-1]
    for f in range(shuffled_vars_train.shape[1]):
        ranking_tracker[fold,int(indices[f])] = f+1

if plot_ranking:
    importances = np.sum(importance_tracker,axis=0)/num_folds
    indices = np.argsort(importances)[::-1]
    # Plot individual feature importance
    plt.figure(figsize=(12,10))
    x = np.arange(num_vars)
    plt.barh(x,width=importances[indices])
    plt.yticks(x, [new_variables[indices[f]] for f in range(num_vars)])

    plt.ylabel('Feature',fontsize=24)
    plt.xlabel('Relative decisiveness',fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    filename = args.csv_file.strip('.csv') + '.png'
    plt.savefig(filename)
    plt.close()

    # Box plot
    modal_list = []
    for f in range(len(variables)):
        print('Feature is:',new_variables[f])
        print('The modal value for importance ranking for this feature is:',mode(ranking_tracker[:,f]).mode)
        modal_list.append(mode(ranking_tracker[:,f]).mode[0])

    modal_indices = np.argsort(modal_list)[::-1]


    import seaborn as sns
    data_to_plot = []
    for f in range(num_vars):
        data_to_plot.append(ranking_tracker[:,modal_indices[f]])

    plt.figure(figsize=(18,10))
    box_plot = sns.boxplot(data=data_to_plot)
    box_plot.set_xticklabels([new_variables[modal_indices[f]] for f in range(num_vars)],fontsize=20)
    box_plot.set_ylabel('Rankings',fontsize=20)
    # box_plot.set_ylim((0,6.0))

    ax = box_plot.axes
    lines = ax.get_lines()
    categories = ax.get_xticks()

    for cat in categories:
        # every 4th line at the interval of 6 is median line
        # 0 -> p25 1 -> p75 2 -> lower whisker 3 -> upper whisker 4 -> p50 5 -> upper extreme value
        y = round(lines[4+cat*6].get_ydata()[0],1) 

        ax.text(
            cat, 
            y, 
            y, 
            ha='center', 
            va='center', 
            fontweight='bold', 
            size=20,
            color='white',
            bbox=dict(facecolor='#445A64'))

    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xticks(rotation=90)
    plt.yticks(fontsize=20)
    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    plt.tight_layout()
    filename = args.csv_file.strip('.csv') + '_RD.png'
    plt.savefig(filename)
    # plt.show()