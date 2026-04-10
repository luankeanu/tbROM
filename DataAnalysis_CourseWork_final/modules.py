"""
first we setup a dataframe to store all the values for each module to make comparison and analyis easier later

also a function to determine the %E average for each variable for further analysis (see below)

we setup a pandas dataframe with a column for each stored variable. each module will write itself into this dataframe
    filling all the columns in its own row.

8 modules were used (where SVR is used twice with different kernels):
    linear regression
    decision tree regressor
    random forest regressor
    SVR (kernels: linear and rbf)
    KNN regressor
    ridge, lasso, elastic net regressors

each module has its own separate .py file to facilitate hyperparameter changes
"""

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler


### IMPORT READ.PY ###

from read import * 
#the * part adds all variables from read into modules.py

### MODELS ###

from models.decisiontree import *
from models.elasticnet import *
from models.KNN import *
from models.KNN_selfwritten import train_knn as train_knn_custom
from models.lassoreg import *
from models.linreg_selfwritten import train_linear_regression as train_linear_regression_custom
from models.linreg import *
from models.polyreg import *
from models.randomforest import *
from models.ridgereg import *
from models.SVR import *



def train_models(X_train, y_train, X_test, y_test):
    """Trains all the regression models and returns a dictionary of the trained models."""
    trained_models = {}

    #initialize StandardScaler
    scaler = StandardScaler()

    #fit the scaler on the training data and transform both training and test data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #train Linear Regression (original)
    lr_model, lr_metrics = train_linear_regression(X_train_scaled, y_train, X_test_scaled, y_test) # Use scaled data (NumPy arrays)
    results_df.loc[len(results_df)] = lr_metrics
    trained_models['Linear Regression'] = lr_model

    #train Linear Regression (Custom)
    lr_model_custom, lr_metrics_custom = train_linear_regression_custom(X_train, y_train, X_test, y_test) # Your custom model doesn't use scaled data based on your code
    results_df.loc[len(results_df)] = lr_metrics_custom
    trained_models['Linear Regression (Custom)'] = lr_model_custom 

    #train Polynomial Regression
    poly_reg_model, poly_reg_metrics = train_polynomial_regression(X_train_scaled, y_train, X_test_scaled, y_test) # Pass degree for clarity, even if it's fixed internally
    results_df.loc[len(results_df)] = poly_reg_metrics
    trained_models['Polynomial Regression (Degree 3)'] = poly_reg_model # Use the key matching the Module name metric

    #train Decision Tree [selected features]
    features_dt = ['layer_height', 'nozzle_temperature', 'print_speed'] #make sure this matches analyse.py
    dt_model, dt_metrics = train_decision_tree(X_train, y_train, X_test, y_test, features_to_use=features_dt) # DTs are less sensitive to scaling
    results_df.loc[len(results_df)] = dt_metrics
    trained_models['Decision Tree Regressor'] = dt_model

    #train Random Forest [selected features]
    features_rf = ['layer_height', 'nozzle_temperature', 'print_speed'] #make sure this matches analyse.py
    rf_model, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test, features_to_use=features_rf) # RFs are less sensitive to scaling
    results_df.loc[len(results_df)] = rf_metrics
    trained_models['Random Forest Regressor (Selected Features)'] = rf_model

    #train SVR (Linear)
    svr_linear_model, svr_linear_metrics = train_svr_linear(X_train_scaled, y_train, X_test_scaled, y_test) # Use scaled data
    results_df.loc[len(results_df)] = svr_linear_metrics
    trained_models['Support Vector Regressor (Linear)'] = svr_linear_model

    #train SVR (RBF)
    svr_rbf_model, svr_rbf_metrics = train_svr_rbf(X_train_scaled, y_train, X_test_scaled, y_test) # Use scaled data
    results_df.loc[len(results_df)] = svr_rbf_metrics
    trained_models['Support Vector Regressor (RBF)'] = svr_rbf_model

    #train KNN (original)
    knn_model_original, knn_metrics_original = train_knn(X_train_scaled, y_train, X_test_scaled, y_test) 
    results_df.loc[len(results_df)] = knn_metrics_original
    trained_models['KNN Regressor'] = knn_model_original

    #train KNN (custom)
    knn_model_custom, knn_metrics_custom = train_knn_custom(X_train_scaled, y_train, X_test_scaled, y_test) 
    results_df.loc[len(results_df)] = knn_metrics_custom
    trained_models['KNN Regressor (Custom)'] = knn_model_custom

    #train Ridge Regression
    features_ridge = ['layer_height', 'nozzle_temperature', 'material_pla', 'wall_thickness', 'infill_pattern_honeycomb', 'bed_temperature'] #make sure this matches analyse.py
    ridge_model, ridge_metrics = train_ridge(X_train_scaled[:, [X_train.columns.get_loc(col) for col in features_ridge]], y_train, X_test_scaled[:, [X_test.columns.get_loc(col) for col in features_ridge]], y_test) # Use scaled data
    results_df.loc[len(results_df)] = ridge_metrics
    trained_models['Ridge Regression'] = ridge_model

    #train Lasso Regression
    features_lasso = ['layer_height', 'nozzle_temperature', 'material_pla', 'wall_thickness', 'infill_pattern_honeycomb', 'bed_temperature'] #make sure this matches analyse.py
    lasso_model, lasso_metrics = train_lasso(X_train_scaled[:, [X_train.columns.get_loc(col) for col in features_lasso]], y_train, X_test_scaled[:, [X_test.columns.get_loc(col) for col in features_lasso]], y_test, alpha=0.01, max_iter=10000, tol=0.0001) # Use scaled data with placeholder hyperparameters
    results_df.loc[len(results_df)] = lasso_metrics
    trained_models['Lasso Regression'] = lasso_model

    #train Elastic Net
    features_en = ['layer_height', 'nozzle_temperature', 'material_pla', 'wall_thickness', 'infill_pattern_honeycomb', 'bed_temperature'] #make sure this matches analyse.py
    elastic_net_model, elastic_net_metrics = train_elastic_net(X_train_scaled[:, [X_train.columns.get_loc(col) for col in features_en]], y_train, X_test_scaled[:, [X_test.columns.get_loc(col) for col in features_en]], y_test, alpha=0.001, l1_ratio=0.3, max_iter=10000, tol=0.0001) # Use scaled data with placeholder hyperparameters
    results_df.loc[len(results_df)] = elastic_net_metrics
    trained_models['Elastic Net'] = elastic_net_model

    return trained_models



### DATAFRAME ###

#initialize an empty DataFrame 'results_df' to store the results of each module
results_df = pd.DataFrame(columns=[
    "Module name",
    "R2 (Test)",
    "R2 (Train)",
    "MSE (Test)",
    "MSE (Train)",
    "RMSE (Test)",
    "RMSE (Train)",
    "MAE (Test)",
    "MAE (Train)",

    "R2 roughness (Test)",
    "R2 roughness (Train)",
    "MSE roughness (Test)",
    "MSE roughness (Train)",
    "RMSE roughness (Test)",
    "RMSE roughness (Train)",
    "MAE roughness (Test)",
    "MAE roughness (Train)",
    "roughness %E avg (Test)",
    "roughness %E avg (Train)",

    "R2 Tensile Strength (Test)",
    "R2 Tensile Strength (Train)",
    "MSE Tensile Strength (Test)",
    "MSE Tensile Strength (Train)",
    "RMSE Tensile Strength (Test)",
    "RMSE Tensile Strength (Train)",
    "MAE Tensile Strength (Test)",
    "MAE Tensile Strength (Train)",
    "Tensile Strength %E avg (Test)",
    "Tensile Strength %E avg (Train)",
    
    "R2 elongation (Test)",
    "R2 elongation (Train)",
    "MSE elongation (Test)",
    "MSE elongation (Train)",
    "RMSE elongation (Test)",
    "RMSE elongation (Train)",
    "MAE elongation (Test)",
    "MAE elongation (Train)",
    "elongation %E avg (Test)",
    "elongation %E avg (Train)"
])




### PRINT ###

test_indices = X_test.index.tolist()
train_indices = X_train.index.tolist()
print("\nIndices of rows in the 20% test set:", test_indices)
print("Indices of rows in the 80% training set:", train_indices)


