# Comparing different methods for regression on dataset

## Dependencies
This code is tested with numpy 1.18.1, pandas 0.24.2, sklearn 0.21.2 and matplotlib 3.1.0, xgboost 0.9

## Running code
Step 1: Ensure you have a `folds/` and `results/` directory in this folder. Then running `python make_folds filename.csv` will split your dataset into 40 folds and save them within `folds/`. We are assuming that the `filename.csv` file has only one dependant variable (the last column).

Step 2: Run `python 03_runRegression.py results/` will run a 40-fold cross-validated fits and save metrics and predictions within `results/`.

Step 3: Finally `python parse_results.py` will generate `Metric_Means.csv` and `Metric_stds.csv` to analyze the different metrics of the predictions (i.e., R2, \rho, etc.)
