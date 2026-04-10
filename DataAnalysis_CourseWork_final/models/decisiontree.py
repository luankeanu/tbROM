import pandas as pd
import sklearn
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from utils import calculate_average_percentage_error

from sklearn.tree import DecisionTreeRegressor


def train_decision_tree(X_train, y_train, X_test, y_test, features_to_use=None, max_depth = 3, min_samples_split = 2, min_samples_leaf = 1):
    if features_to_use is not None:
        X_train_subset = X_train[features_to_use]
        X_test_subset = X_test[features_to_use]
    else:
        X_train_subset = X_train
        X_test_subset = X_test

    decision_tree_model = DecisionTreeRegressor(random_state=42, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    decision_tree_model.fit(X_train_subset, y_train)

    #predictions test and training sets
    y_pred_decision_tree_test = decision_tree_model.predict(X_test_subset)
    y_pred_decision_tree_train = decision_tree_model.predict(X_train_subset)

    #overall R2, MSE, RMSE, MAE
    r2_decision_tree_test = r2_score(y_test, y_pred_decision_tree_test)
    r2_decision_tree_train = r2_score(y_train, y_pred_decision_tree_train)

    mse_decision_tree_test = mean_squared_error(y_test, y_pred_decision_tree_test)
    mse_decision_tree_train = mean_squared_error(y_train, y_pred_decision_tree_train)

    rmse_decision_tree_test = np.sqrt(mse_decision_tree_test)
    rmse_decision_tree_train = np.sqrt(mse_decision_tree_train)

    mae_decision_tree_test = mean_absolute_error(y_test, y_pred_decision_tree_test)
    mae_decision_tree_train = mean_absolute_error(y_train, y_pred_decision_tree_train)

    #R2, MSE, RMSE, MAE, and %E avg for each target variable (Test set)
    r2_roughness_dt_test = r2_score(y_test['roughness'], y_pred_decision_tree_test[:, 0])
    mse_roughness_dt_test = mean_squared_error(y_test['roughness'], y_pred_decision_tree_test[:, 0])
    rmse_roughness_dt_test = np.sqrt(mse_roughness_dt_test)
    mae_roughness_dt_test = mean_absolute_error(y_test['roughness'], y_pred_decision_tree_test[:, 0])
    roughness_pe_avg_dt_test = calculate_average_percentage_error(y_test['roughness'], y_pred_decision_tree_test[:, 0])

    r2_tensile_strength_dt_test = r2_score(y_test['tension_strength'], y_pred_decision_tree_test[:, 1])
    mse_tensile_strength_dt_test = mean_squared_error(y_test['tension_strength'], y_pred_decision_tree_test[:, 1])
    rmse_tensile_strength_dt_test = np.sqrt(mse_tensile_strength_dt_test)
    mae_tensile_strength_dt_test = mean_absolute_error(y_test['tension_strength'], y_pred_decision_tree_test[:, 1])
    tensile_strength_pe_avg_dt_test = calculate_average_percentage_error(y_test['tension_strength'], y_pred_decision_tree_test[:, 1])

    r2_elongation_dt_test = r2_score(y_test['elongation'], y_pred_decision_tree_test[:, 2])
    mse_elongation_dt_test = mean_squared_error(y_test['elongation'], y_pred_decision_tree_test[:, 2])
    rmse_elongation_dt_test = np.sqrt(mse_elongation_dt_test)
    mae_elongation_dt_test = mean_absolute_error(y_test['elongation'], y_pred_decision_tree_test[:, 2])
    elongation_pe_avg_dt_test = calculate_average_percentage_error(y_test['elongation'], y_pred_decision_tree_test[:, 2])

    #R2, MSE, RMSE, MAE, and %E avg for each target variable (Train set)
    r2_roughness_dt_train = r2_score(y_train['roughness'], y_pred_decision_tree_train[:, 0])
    mse_roughness_dt_train = mean_squared_error(y_train['roughness'], y_pred_decision_tree_train[:, 0])
    rmse_roughness_dt_train = np.sqrt(mse_roughness_dt_train)
    mae_roughness_dt_train = mean_absolute_error(y_train['roughness'], y_pred_decision_tree_train[:, 0])
    roughness_pe_avg_dt_train = calculate_average_percentage_error(y_train['roughness'], y_pred_decision_tree_train[:, 0])

    r2_tensile_strength_dt_train = r2_score(y_train['tension_strength'], y_pred_decision_tree_train[:, 1])
    mse_tensile_strength_dt_train = mean_squared_error(y_train['tension_strength'], y_pred_decision_tree_train[:, 1])
    rmse_tensile_strength_dt_train = np.sqrt(mse_tensile_strength_dt_train)
    mae_tensile_strength_dt_train = mean_absolute_error(y_train['tension_strength'], y_pred_decision_tree_train[:, 1])
    tensile_strength_pe_avg_dt_train = calculate_average_percentage_error(y_train['tension_strength'], y_pred_decision_tree_train[:, 1])

    r2_elongation_dt_train = r2_score(y_train['elongation'], y_pred_decision_tree_train[:, 2])
    mse_elongation_dt_train = mean_squared_error(y_train['elongation'], y_pred_decision_tree_train[:, 2])
    rmse_elongation_dt_train = np.sqrt(mse_elongation_dt_train)
    mae_elongation_dt_train = mean_absolute_error(y_train['elongation'], y_pred_decision_tree_train[:, 2])
    elongation_pe_avg_dt_train = calculate_average_percentage_error(y_train['elongation'], y_pred_decision_tree_train[:, 2])

    metrics = {
        "Module name": "Decision Tree Regressor (Selected Features)",
        "R2 (Test)": r2_decision_tree_test,
        "R2 (Train)": r2_decision_tree_train,
        "MSE (Test)": mse_decision_tree_test,
        "MSE (Train)": mse_decision_tree_train,
        "RMSE (Test)": rmse_decision_tree_test,
        "RMSE (Train)": rmse_decision_tree_train,
        "MAE (Test)": mae_decision_tree_test,
        "MAE (Train)": mae_decision_tree_train,

        "R2 roughness (Test)": r2_roughness_dt_test,
        "R2 roughness (Train)": r2_roughness_dt_train,
        "MSE roughness (Test)": mse_roughness_dt_test,
        "MSE roughness (Train)": mse_roughness_dt_train,
        "RMSE roughness (Test)": rmse_roughness_dt_test,
        "RMSE roughness (Train)": rmse_roughness_dt_train,
        "MAE roughness (Test)": mae_roughness_dt_test,
        "MAE roughness (Train)": mae_roughness_dt_train,
        "roughness %E avg (Test)": roughness_pe_avg_dt_test,
        "roughness %E avg (Train)": roughness_pe_avg_dt_train,

        "R2 Tensile Strength (Test)": r2_tensile_strength_dt_test,
        "R2 Tensile Strength (Train)": r2_tensile_strength_dt_train,
        "MSE Tensile Strength (Test)": mse_tensile_strength_dt_test,
        "MSE Tensile Strength (Train)": mse_tensile_strength_dt_train,
        "RMSE Tensile Strength (Test)": rmse_tensile_strength_dt_test,
        "RMSE Tensile Strength (Train)": rmse_tensile_strength_dt_train,
        "MAE Tensile Strength (Test)": mae_tensile_strength_dt_test,
        "MAE Tensile Strength (Train)": mae_tensile_strength_dt_train,
        "Tensile Strength %E avg (Test)": tensile_strength_pe_avg_dt_test,
        "Tensile Strength %E avg (Train)": tensile_strength_pe_avg_dt_train,

        "R2 elongation (Test)": r2_elongation_dt_test,
        "R2 elongation (Train)": r2_elongation_dt_train,
        "MSE elongation (Test)": mse_elongation_dt_test,
        "MSE elongation (Train)": mse_elongation_dt_train,
        "RMSE elongation (Test)": rmse_elongation_dt_test,
        "RMSE elongation (Train)": rmse_elongation_dt_train,
        "MAE elongation (Test)": mae_elongation_dt_test,
        "MAE elongation (Train)": mae_elongation_dt_train,
        "elongation %E avg (Test)": elongation_pe_avg_dt_test,
        "elongation %E avg (Train)": elongation_pe_avg_dt_train,
    }
    return decision_tree_model, metrics