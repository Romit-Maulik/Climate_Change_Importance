# Determining feature importance for actionable climate change mitigation policies

## Dependencies
These codes is tested with numpy 1.18.1, pandas 0.24.2, sklearn 0.21.2 and matplotlib 3.1.0, xgboost 0.9

## Running method comparisons
Code for comparing different methods and resulting metrics can be found in readme markdown files within the `Method_Comparison` folder.

## Running XGB code
The `XGB.py` code reads in a `*.csv` file where the last column is the _only_ dependant variable and the first row consists of variable names. The code can be run using:

`python XGB.py my_csv_filename.csv`

## Feature importance: Public support for federal intervention in climate change mitigation

Note: Consult `Variables.docx` for variable explanations

![Relative decisiveness](https://github.com/Romit-Maulik/Climate_Change_Importance/blob/master/federal/federal.png "Relative decisiveness")
![Fold Rankings](https://github.com/Romit-Maulik/Climate_Change_Importance/blob/master/federal/federal_RD.png "Fold ranking boxplots")
