import numpy as np
np.random.seed(5)
import matplotlib.pyplot as plt

key_list = ['lm','rg','svm','gp','knn','dt','br','etr','rfr','abr','gbr','xgb','dl']
metric_list = ['r2','rho','evs','mae','rmse']
num_methods = len(key_list)
num_metrics = 5

metric_matrix = np.zeros(shape=(40,num_methods,num_metrics))

for fold in range(40):
    method = 0
    for key in key_list:
        fname = 'metric_'+key+'_'+f'{fold:02}'
        temp_load = np.loadtxt('results/'+fname+'.csv',delimiter=',',usecols=[1,2,3,4,5],skiprows=1)

        metric_matrix[fold,method,0] = temp_load[0]
        metric_matrix[fold,method,1] = temp_load[1]
        metric_matrix[fold,method,2] = temp_load[2]
        metric_matrix[fold,method,3] = temp_load[3]
        metric_matrix[fold,method,4] = temp_load[4]

        method = method + 1

# Box plots
show_box_plots = False
if show_box_plots:
    import seaborn as sns
    for metric in range(num_metrics):

        data_to_plot = []
        for method in range(0,num_methods):
            data_to_plot.append(metric_matrix[:,method,metric])

        box_plot = sns.boxplot(data=data_to_plot)
        box_plot.set_xticklabels(key_list,fontsize=20)
        box_plot.set_ylabel(metric_list[metric],fontsize=20)

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
        plt.tight_layout()
        plt.show()

# Write out average and std of metric_matrix
metric_means = np.zeros(shape=(num_metrics,num_methods-1))
metric_stds = np.zeros(shape=(num_metrics,num_methods-1))

for metric in range(num_metrics):
    for method in range(num_methods-1):
        metric_means[metric,method] = np.mean(metric_matrix[:,method,metric],axis=0)
        metric_stds[metric,method] = np.std(metric_matrix[:,method,metric],axis=0)

# Save metric matrices
np.savetxt('Metric_Means.csv',metric_means,fmt='%1.2e',delimiter=',')
np.savetxt('Metric_stds.csv',metric_stds,fmt='%1.2e',delimiter=',')