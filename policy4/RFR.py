import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

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

scaler = StandardScaler()
data = scaler.fit_transform(data)

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

num_forests = 40
importance_tracker = np.zeros(shape=(num_forests,len(variables)),dtype='int')
print_ranking = False
plot_ranking = True
show_tree = False
hpo = True
r2_val = 0
mae_val = 0

for r_state in range(num_forests):

    # Finding best depth
    if hpo:
        mae_test_best = 100.0
        best_depth = 4

        # Split the data into training and testing sets
        shuffled_vars_train, shuffled_vars_test, dependent_vars_train, dependent_vars_test = train_test_split(shuffled_vars, dependent_vars.reshape(-1,1), train_size=0.9, random_state=r_state)

        for depth in range(4,15):
            
            forest = RandomForestRegressor(max_depth=depth, random_state=1,n_estimators=100)
            forest.fit(shuffled_vars_train, dependent_vars_train[:,0])
            importances = forest.feature_importances_
            std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                         axis=0)

            mae_test = mean_absolute_error(dependent_vars_test,forest.predict(shuffled_vars_test))
            mae_train = mean_absolute_error(dependent_vars_train,forest.predict(shuffled_vars_train))

            if mae_test < mae_test_best:
                best_depth = depth
                mae_test_best = mae_test


        print('Optimal depth:',best_depth)
        print('Optimal test_mae:',mae_test_best)
        mae_val = mae_val + mae_test_best

    # Split the data into training and testing sets
    shuffled_vars_train, shuffled_vars_test, dependent_vars_train, dependent_vars_test = train_test_split(shuffled_vars, dependent_vars.reshape(-1,1), train_size=0.9, random_state=r_state)

    forest = RandomForestRegressor(max_depth=best_depth, random_state=1,n_estimators=100)
    forest.fit(shuffled_vars_train, dependent_vars_train[:,0])
    
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)


    indices = np.argsort(importances)[::-1]
    for f in range(shuffled_vars_train.shape[1]):
        importance_tracker[r_state,int(indices[f])] = f+1


    if print_ranking:
        from sklearn.metrics import mean_absolute_error
        mae_test = mean_absolute_error(dependent_vars_test,forest.predict(shuffled_vars_test))
        print('Test mean mean_absolute_error:', mae_test)
        mae_train = mean_absolute_error(dependent_vars_train,forest.predict(shuffled_vars_train))
        print('Train mean mean_absolute_error:', mae_train)

        for f in range(shuffled_vars_train.shape[1]):
            print("%d. feature %d (%f) : %s" % (f + 1, indices[f], importances[indices[f]], new_variables[indices[f]]))

    if plot_ranking:
        # Plot individual feature importance
        plt.figure(figsize=(12,10))
        x = np.arange(num_vars)
        plt.barh(x,width=importances[indices])
        plt.yticks(x, [new_variables[indices[f]] for f in range(num_vars)])

        plt.ylabel('Independent variable',fontsize=24)
        plt.xlabel('Relative decisiveness',fontsize=24)
        plt.tick_params(axis='both', which='major', labelsize=18)
        plt.tight_layout()
        plt.savefig('Ranking_tree_'+str(r_state)+'.png')
        plt.close()

    if show_tree:
        # For visualization
        # Extract single tree
        estimator = forest.estimators_[9]
        from sklearn.tree import export_graphviz
        # Export as dot file
        export_graphviz(estimator, out_file='Figure_1.dot', 
                        feature_names = variables,
                        class_names = ['output'],
                        rounded = True, proportion = False, 
                        precision = 2, filled = True)
        # Convert to png using system command (requires Graphviz)
        from subprocess import call
        call(['dot', '-Tpng', 'Figure_1.dot', '-o', 'Figure_'+str(r_state)+'.png', '-Gdpi=300'])


modal_list = []
for f in range(len(variables)):
    print('Feature is:',new_variables[f])
    print('The modal value for importance ranking for this feature is:',mode(importance_tracker[:,f]).mode)
    modal_list.append(mode(importance_tracker[:,f]).mode[0])

modal_indices = np.argsort(modal_list)[::-1]

if plot_ranking:
    # Plot individual feature importance
    plt.figure(figsize=(12,10))
    x = np.arange(num_vars)
    plt.barh(x,width=[mode(importance_tracker[:,modal_indices[f]]).mode[0] for f in range(num_vars)])
    plt.yticks(x, [new_variables[modal_indices[f]] for f in range(num_vars)])

    plt.ylabel('Independent variable',fontsize=24)
    plt.xlabel('Relative decisiveness',fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tight_layout()
    plt.savefig('Modal_ranking.png')
    plt.close()

    import seaborn as sns
    data_to_plot = []
    for f in range(num_vars):
        data_to_plot.append(importance_tracker[:,modal_indices[f]])

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
    plt.show()

print('Average test MAE:',mae_val/num_forests)