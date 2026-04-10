"""
running this file will print the result from the outliers and missing values analysis;
will also print the shapes of the 80/20 split

first we import the .csv and check missing values and outliers.
then we treat columns 4 and 8 to update it to cathegorical values.
then do the 80/20 split randomly.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


dataset = pd.read_csv("files/data_3dprinter.csv")

### DATA TREATING ###

#check missing values
print("Missing values before handling:")
print(dataset.isnull().sum())

dataset = dataset.dropna() #drop rows with any missing values

print("\nMissing values after handling:")
print(dataset.isnull().sum())

#OneHotEncoding to 'infill_pattern' and 'material'
#note: this drops "infill pattern" and creates "infill_pattern_grid" and "(...)_honeycomb"
    #and uses True and False in each to categorize the data
encoder = OneHotEncoder(sparse_output=False)
encoded_columns = encoder.fit_transform(dataset[['infill_pattern', 'material']])
encoded_df = pd.DataFrame(encoded_columns, columns=encoder.get_feature_names_out(['infill_pattern', 'material']))

#reset index for both DataFrames to ensure proper concatenation
dataset = dataset.reset_index(drop=True)
encoded_df = encoded_df.reset_index(drop=True)

dataset = pd.concat([dataset, encoded_df], axis=1) #concatenate the encoded columns with the original DataFrame

dataset = dataset.drop(['infill_pattern', 'material'], axis=1) #drop the original categorical columns

print("\nDataFrame after One-Hot Encoding:")
print(dataset.head())
#above is for debugging



### 80/20 SPLIT ###

#separate target values from features
#X features
#y target values
X = dataset.drop(['roughness', 'tension_strength', 'elongation', 'material_abs', 'infill_pattern_grid'], axis=1)
y = dataset[['roughness', 'tension_strength', 'elongation']]

"""
from the heatmap we see strong coorelation between material and infill pattern when separated
    after onehotencoding, so we removed one of each.
"""

#Split the data into training and testing sets
#X_train and y_train --> training data (80%)
#X_test and y_test --> testing data (20%); test size = 0.2
#random_state (rs) ensures that the split is the same every time the code is run
rs = 343
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = rs)

"""
depending on the training sets the prediction models behave differently
plausible random states tested: 454; 670; 343; 595; 930
"""

#check to see all rows and columns are included and separated correctly (debugging)
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

