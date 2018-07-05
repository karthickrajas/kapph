# Kapph

## Installation

```
conda install git pip
pip install git+https://github.com/karthickrajas/kapph.git
```


## Description
This package is being built for applying pre processing functionality for datasets.

## printCols

Print the list of columns with the datatype and the number of missing values in the column. Returns the output as a dataframe.
Features to be added yet.

**Usage**
```
pp.printCols(data)
```
**Output**
```
  Column Name  Number of NA values     Type
0     Country                    0.0   Object
1   Purchased                    0.0   Object
2         Age                    1.0  Numeric
3      Salary                    1.0  Numeric
```
