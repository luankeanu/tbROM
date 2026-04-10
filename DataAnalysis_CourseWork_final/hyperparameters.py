"""
we wrote loops to test the hyper parameters of each model.
all of them are inside comments and can be taken off of coment form to be used.     
"""

from read import *
from modules import *
import csv

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np

from models.decisiontree import *
from models.elasticnet import *
from models.KNN import *
from models.lassoreg import *
from models.linreg import *
from models.polyreg import *
from models.randomforest import *
from models.ridgereg import *
from models.SVR import *



#scale the data (important for KNN and SVR)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


best_r2 = -1.0



### DECISION TREE HYPER PARAMETER CALIBRATION ###

max_depth_range = [None] + list(range(1, 21))  #none + from 1 to 20 (inclusive)
min_samples_split_range = list(range(2, 21)) #from 2 to 20 (inclusive)
min_samples_leaf_range = list(range(1, 11))  #from 1 to 10 (inclusive)r


best_hyperparameters = {}
all_results = []

for max_depth in max_depth_range:
    for min_samples_split in min_samples_split_range:
        for min_samples_leaf in min_samples_leaf_range:

            #train Decision Tree with the current hyperparameters
            dt_model, dt_metrics = train_decision_tree(
                X_train, y_train, X_test, y_test,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf
            )

            #eval performance (using overall R-squared on the test set)
            current_r2 = dt_metrics['R2 (Test)']
            all_results.append({
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'R2 (Test)': current_r2,
                'MSE (Test)': dt_metrics['MSE (Test)']
            })

            #update best performance if current is better
            if current_r2 > best_r2:
                best_r2 = current_r2
                best_hyperparameters = {
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf
                }

print("\n--- Optimal Hyperparameters Found ---")
print("Best R2 Score (Test):", best_r2)
print("Best Hyperparameters:", best_hyperparameters)

#save to .csv
with open('dt_hyperparameter_tuning_results.csv', 'w', newline='') as csvfile:
    fieldnames = ['max_depth', 'min_samples_split', 'min_samples_leaf', 'R2 (Test)', 'MSE (Test)']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_results)




### RANDOM FOREST HYPER PARAMETER CALIBRATION ###

"""
n_estimators_range_rf = list(range(1, 301))  #from 1 to 20 (inclusive)
max_depth_range_rf = [None] + list(range(1, 16)) #none + from 1 to 20 (inclusive)
min_samples_split_range_rf = list(range(2, 11)) #from 2 to 10 (inclusive)
min_samples_leaf_range_rf = list(range(1, 6)) #1 to 5 (inclusive)
"""




n_estimators_range_rf = list(range(1, 301))  #from 1 to 20 (inclusive)
max_depth_range_rf = [None]  #none + from 1 to 20 (inclusive)
min_samples_split_range_rf = [4] #from 2 to 10 (inclusive)
min_samples_leaf_range_rf = [1] #1 to 5 (inclusive)



best_hyperparameters_rf = {}
all_results_rf = []

features_rf = ['layer_height', 'nozzle_temperature', 'print_speed'] # Define the features to use

for n_estimators in n_estimators_range_rf:
    for max_depth in max_depth_range_rf:
        for min_samples_split in min_samples_split_range_rf:
            for min_samples_leaf in min_samples_leaf_range_rf:
                print(f"Trying RF: n_estimators={n_estimators}, max_depth={max_depth}, min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}")

                #train Random Forest with the current hyperparameters AND selected features
                rf_model, rf_metrics = train_random_forest(
                    X_train, y_train, X_test, y_test,
                    features_to_use=features_rf, # Pass the features to use
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf
                )

                #eval performance (using overall R-squared on the test set)
                current_r2_rf = rf_metrics['R2 (Test)']
                all_results_rf.append({
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'R2 (Test)': current_r2_rf,
                    'MSE (Test)': rf_metrics['MSE (Test)']
                })

                #update best performance if current is better
                if current_r2_rf > best_r2:
                    best_r2 = current_r2_rf
                    best_hyperparameters_rf = {
                        'n_estimators': n_estimators,
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split,
                        'min_samples_leaf': min_samples_leaf
                    }

print("\n--- Optimal Hyperparameters Found for Random Forest (Selected Features) ---")
print("Best R2 Score (Test):", best_r2)
print("Best Hyperparameters:", best_hyperparameters_rf)

#save to .csv
with open('rf_hyperparameter_tuning_results_selected_features.csv', 'w', newline='') as csvfile:
    fieldnames = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'R2 (Test)', 'MSE (Test)']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_results_rf)





### LASSO REGRESSION HYPER PARAMETER CALIBRATION ###



alpha_range_lasso = [0.001, 0.01, 0.1, 1, 10, 100] 
best_hyperparameters_lasso = {}


max_iter_lasso = 10000
tolerance_lasso = 0.001

for alpha in alpha_range_lasso:
    lasso_model, lasso_metrics = train_lasso(X_train, y_train, X_test, y_test, alpha=alpha, max_iter=max_iter_lasso, tol=tolerance_lasso)
    current_r2 = lasso_metrics['R2 (Test)']
    if current_r2 > best_r2:
        best_r2 = current_r2
        best_hyperparameters_lasso = {'alpha': alpha, 'max_iter': max_iter_lasso, 'tol': tolerance_lasso} 


print("\n--- Optimal Hyperparameters Found for Lasso Regression ---")
print("Best R2 Score (Test):", best_r2)
print("Best Hyperparameters:", best_hyperparameters_lasso)




### RIDGE REGRESSION HYPER PARAMETER CALIBRATION ###



print("\n--- Tuning Ridge Regression Hyperparameters ---")
param_grid_ridge = {
    'alpha': [0.001, 0.01, 0.1, 1, 10, 100]
}
grid_search_ridge = GridSearchCV(Ridge(), param_grid_ridge, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search_ridge.fit(X_train_scaled, y_train)
best_hyperparameters_ridge = grid_search_ridge.best_params_
best_score_ridge = grid_search_ridge.best_score_
print("Best Score (Negative MSE):", best_score_ridge)
print("Best Hyperparameters:", best_hyperparameters_ridge)



### ELASTIC NET HYPER PARAMETER CALIBRATION ###


alpha_range_elastic_net = [0.001, 0.01, 0.1, 1, 10, 100]
l1_ratio_range = [0.1, 0.3, 0.5, 0.7, 0.9]
best_hyperparameters_elastic_net = {}


max_iter_elastic_net = 10000
tolerance_elastic_net = 0.001

for alpha in alpha_range_elastic_net:
    for l1_ratio in l1_ratio_range:
        elastic_net_model, elastic_net_metrics = train_elastic_net(
            X_train, y_train, X_test, y_test, alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter_elastic_net, tol=tolerance_elastic_net
        )
        current_r2 = elastic_net_metrics['R2 (Test)']
        if current_r2 > best_r2:
            best_r2 = current_r2
            best_hyperparameters_elastic_net = {'alpha': alpha, 'l1_ratio': l1_ratio, 'max_iter': max_iter_elastic_net, 'tol': tolerance_elastic_net} 
print("\n--- Optimal Hyperparameters Found for Elastic Net ---")
print("Best R2 Score (Test):", best_r2)
print("Best Hyperparameters:", best_hyperparameters_elastic_net)




### KNN HYPER PARAMETER CALIBRATION ###


print("\n--- Tuning KNN Hyperparameters ---")
param_grid_knn = {
    'n_neighbors': list(range(1, 21)),
    'weights': ['uniform', 'distance'],
    'p': [1, 2]
}
grid_search_knn = GridSearchCV(KNeighborsRegressor(), param_grid_knn, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search_knn.fit(X_train_scaled, y_train)
best_hyperparameters_knn = grid_search_knn.best_params_
best_score_knn = grid_search_knn.best_score_
print("Best Score (Negative MSE):", best_score_knn)
print("Best Hyperparameters:", best_hyperparameters_knn)



### SVR (Linear and RBF Kernels) HYPER PARAMETER CALIBRATION ###


print("\n--- Tuning SVR (Linear) Hyperparameters ---")
param_grid_svr_linear = {
    'estimator__C': [0.1, 1, 10, 100],
    'estimator__epsilon': [0.01, 0.1, 1]
}
grid_search_svr_linear = GridSearchCV(MultiOutputRegressor(SVR(kernel='linear')), param_grid_svr_linear, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search_svr_linear.fit(X_train_scaled, y_train)
best_hyperparameters_svr_linear = grid_search_svr_linear.best_params_
best_score_svr_linear = grid_search_svr_linear.best_score_
print("Best Score (Negative MSE):", best_score_svr_linear)
print("Best Hyperparameters:", best_hyperparameters_svr_linear)


print("\n--- Tuning SVR (RBF) Hyperparameters ---")
param_grid_svr_rbf = {
    'estimator__C': [0.1, 1, 10, 100],
    'estimator__epsilon': [0.01, 0.1, 1],
    'estimator__gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
}
grid_search_svr_rbf = GridSearchCV(MultiOutputRegressor(SVR(kernel='rbf')), param_grid_svr_rbf, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search_svr_rbf.fit(X_train_scaled, y_train)
best_hyperparameters_svr_rbf = grid_search_svr_rbf.best_params_
best_score_svr_rbf = grid_search_svr_rbf.best_score_
print("Best Score (Negative MSE):", best_score_svr_rbf)
print("Best Hyperparameters:", best_hyperparameters_svr_rbf)




### POLYNOMIAL REGRESSION HYPER PARAMETER CALIBRATION ###


print("\n--- Tuning Polynomial Regression Hyperparameters ---")


pipeline_poly = make_pipeline(
    PolynomialFeatures(include_bias=False), 
    LinearRegression()
)

param_grid_poly = {
    'polynomialfeatures__degree': [2, 3, 4] 

}
scoring_poly = 'neg_mean_squared_error' 

grid_search_poly = GridSearchCV(pipeline_poly, param_grid_poly, cv=5, scoring=scoring_poly, n_jobs=-1)

grid_search_poly.fit(X_train_scaled, y_train)


best_hyperparameters_poly = grid_search_poly.best_params_
best_score_poly = grid_search_poly.best_score_ 

print("Best Score (Negative MSE):", best_score_poly)
print("Best Hyperparameters:", best_hyperparameters_poly)
