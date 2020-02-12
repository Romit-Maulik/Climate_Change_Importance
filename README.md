# Determining feature importance for actionable climate change mitigation policies

## Dependencies
This code is tested with numpy 1.18.1, pandas 0.24.2, sklearn 0.21.2 and matplotlib 3.1.0

## Running code
The code reads in a `*.csv` file where the last column is the _only_ dependant variable and the first row consists of variable names. The code can be run using:

`python RFR.py my_csv_filename.csv`

## Note:
Generate individual decision tress can be controlled using the boolean variable `show_tree` (line 54). However, this would require the installation of graphviz on your system. These results are obtained using dot - graphviz version 2.40.1.
