import pandas as pd
import sklearn
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from utils import calculate_average_percentage_error 
from sklearn.multioutput import MultiOutputRegressor

from sklearn.svm import SVR

def train_svr_rbf(X_train, y_train, X_test, y_test, C=100, epsilon=0.01, gamma='scale'):
    """Trains an SVR with RBF kernel and specified hyperparameters."""
    svr_rbf = SVR(kernel='rbf', C=C, epsilon=epsilon, gamma=gamma)
    multi_output_svr_rbf = MultiOutputRegressor(svr_rbf)
    multi_output_svr_rbf.fit(X_train, y_train)

    #predictions test and training sets
    y_pred_svr_rbf_test = multi_output_svr_rbf.predict(X_test)
    y_pred_svr_rbf_train = multi_output_svr_rbf.predict(X_train)

    #overall R2, MSE, RMSE, MAE
    r2_svr_rbf_test = r2_score(y_test, y_pred_svr_rbf_test)
    r2_svr_rbf_train = r2_score(y_train, y_pred_svr_rbf_train)

    mse_svr_rbf_test = mean_squared_error(y_test, y_pred_svr_rbf_test)
    mse_svr_rbf_train = mean_squared_error(y_train, y_pred_svr_rbf_train)

    rmse_svr_rbf_test = np.sqrt(mse_svr_rbf_test)
    rmse_svr_rbf_train = np.sqrt(mse_svr_rbf_train)

    mae_svr_rbf_test = mean_absolute_error(y_test, y_pred_svr_rbf_test)
    mae_svr_rbf_train = mean_absolute_error(y_train, y_pred_svr_rbf_train)

    #R2, MSE, RMSE, MAE, and %E avg for each target variable (Test set)
    r2_roughness_svr_rbf_test = r2_score(y_test['roughness'], y_pred_svr_rbf_test[:, 0])
    mse_roughness_svr_rbf_test = mean_squared_error(y_test['roughness'], y_pred_svr_rbf_test[:, 0])
    rmse_roughness_svr_rbf_test = np.sqrt(mse_roughness_svr_rbf_test)
    mae_roughness_svr_rbf_test = mean_absolute_error(y_test['roughness'], y_pred_svr_rbf_test[:, 0])
    roughness_pe_avg_svr_rbf_test = calculate_average_percentage_error(y_test['roughness'], y_pred_svr_rbf_test[:, 0])

    r2_tensile_strength_svr_rbf_test = r2_score(y_test['tension_strength'], y_pred_svr_rbf_test[:, 1])
    mse_tensile_strength_svr_rbf_test = mean_squared_error(y_test['tension_strength'], y_pred_svr_rbf_test[:, 1])
    rmse_tensile_strength_svr_rbf_test = np.sqrt(mse_tensile_strength_svr_rbf_test)
    mae_tensile_strength_svr_rbf_test = mean_absolute_error(y_test['tension_strength'], y_pred_svr_rbf_test[:, 1])
    tensile_strength_pe_avg_svr_rbf_test = calculate_average_percentage_error(y_test['tension_strength'], y_pred_svr_rbf_test[:, 1])

    r2_elongation_svr_rbf_test = r2_score(y_test['elongation'], y_pred_svr_rbf_test[:, 2])
    mse_elongation_svr_rbf_test = mean_squared_error(y_test['elongation'], y_pred_svr_rbf_test[:, 2])
    rmse_elongation_svr_rbf_test = np.sqrt(mse_elongation_svr_rbf_test)
    mae_elongation_svr_rbf_test = mean_absolute_error(y_test['elongation'], y_pred_svr_rbf_test[:, 2])
    elongation_pe_avg_svr_rbf_test = calculate_average_percentage_error(y_test['elongation'], y_pred_svr_rbf_test[:, 2])

    #R2, MSE, RMSE, MAE, and %E avg for each target variable (Train set)
    r2_roughness_svr_rbf_train = r2_score(y_train['roughness'], y_pred_svr_rbf_train[:, 0])
    mse_roughness_svr_rbf_train = mean_squared_error(y_train['roughness'], y_pred_svr_rbf_train[:, 0])
    rmse_roughness_svr_rbf_train = np.sqrt(mse_roughness_svr_rbf_train)
    mae_roughness_svr_rbf_train = mean_absolute_error(y_train['roughness'], y_pred_svr_rbf_train[:, 0])
    roughness_pe_avg_svr_rbf_train = calculate_average_percentage_error(y_train['roughness'], y_pred_svr_rbf_train[:, 0])

    r2_tensile_strength_svr_rbf_train = r2_score(y_train['tension_strength'], y_pred_svr_rbf_train[:, 1])
    mse_tensile_strength_svr_rbf_train = mean_squared_error(y_train['tension_strength'], y_pred_svr_rbf_train[:, 1])
    rmse_tensile_strength_svr_rbf_train = np.sqrt(mse_tensile_strength_svr_rbf_train)
    mae_tensile_strength_svr_rbf_train = mean_absolute_error(y_train['tension_strength'], y_pred_svr_rbf_train[:, 1])
    tensile_strength_pe_avg_svr_rbf_train = calculate_average_percentage_error(y_train['tension_strength'], y_pred_svr_rbf_train[:, 1])

    r2_elongation_svr_rbf_train = r2_score(y_train['elongation'], y_pred_svr_rbf_train[:, 2])
    mse_elongation_svr_rbf_train = mean_squared_error(y_train['elongation'], y_pred_svr_rbf_train[:, 2])
    rmse_elongation_svr_rbf_train = np.sqrt(mse_elongation_svr_rbf_train)
    mae_elongation_svr_rbf_train = mean_absolute_error(y_train['elongation'], y_pred_svr_rbf_train[:, 2])
    elongation_pe_avg_svr_rbf_train = calculate_average_percentage_error(y_train['elongation'], y_pred_svr_rbf_train[:, 2])

    metrics = {
        "Module name": "Support Vector Regressor (RBF)",
        "R2 (Test)": r2_svr_rbf_test,
        "R2 (Train)": r2_svr_rbf_train,
        "MSE (Test)": mse_svr_rbf_test,
        "MSE (Train)": mse_svr_rbf_train,
        "RMSE (Test)": rmse_svr_rbf_test,
        "RMSE (Train)": rmse_svr_rbf_train,
        "MAE (Test)": mae_svr_rbf_test,
        "MAE (Train)": mae_svr_rbf_train,

        "R2 roughness (Test)": r2_roughness_svr_rbf_test,
        "R2 roughness (Train)": r2_roughness_svr_rbf_train,
        "MSE roughness (Test)": mse_roughness_svr_rbf_test,
        "MSE roughness (Train)": mse_roughness_svr_rbf_train,
        "RMSE roughness (Test)": rmse_roughness_svr_rbf_test,
        "RMSE roughness (Train)": rmse_roughness_svr_rbf_train,
        "MAE roughness (Test)": mae_roughness_svr_rbf_test,
        "MAE roughness (Train)": mae_roughness_svr_rbf_train,
        "roughness %E avg (Test)": roughness_pe_avg_svr_rbf_test,
        "roughness %E avg (Train)": roughness_pe_avg_svr_rbf_train,

        "R2 Tensile Strength (Test)": r2_tensile_strength_svr_rbf_test,
        "R2 Tensile Strength (Train)": r2_tensile_strength_svr_rbf_train,
        "MSE Tensile Strength (Test)": mse_tensile_strength_svr_rbf_test,
        "MSE Tensile Strength (Train)": mse_tensile_strength_svr_rbf_train,
        "RMSE Tensile Strength (Test)": rmse_tensile_strength_svr_rbf_test,
        "RMSE Tensile Strength (Train)": rmse_tensile_strength_svr_rbf_train,
        "MAE Tensile Strength (Test)": mae_tensile_strength_svr_rbf_test,
        "MAE Tensile Strength (Train)": mae_tensile_strength_svr_rbf_train,
        "Tensile Strength %E avg (Test)": tensile_strength_pe_avg_svr_rbf_test,
        "Tensile Strength %E avg (Train)": tensile_strength_pe_avg_svr_rbf_train,

        "R2 elongation (Test)": r2_elongation_svr_rbf_test,
        "R2 elongation (Train)": r2_elongation_svr_rbf_train,
        "MSE elongation (Test)": mse_elongation_svr_rbf_test,
        "MSE elongation (Train)": mse_elongation_svr_rbf_train,
        "RMSE elongation (Test)": rmse_elongation_svr_rbf_test,
        "RMSE elongation (Train)": rmse_elongation_svr_rbf_train,
        "MAE elongation (Test)": mae_elongation_svr_rbf_test,
        "MAE elongation (Train)": mae_elongation_svr_rbf_train,
        "elongation %E avg (Test)": elongation_pe_avg_svr_rbf_test,
        "elongation %E avg (Train)": elongation_pe_avg_svr_rbf_train,
    }
    return multi_output_svr_rbf, metrics

def train_svr_linear(X_train, y_train, X_test, y_test, C=100, epsilon=0.01):
    """Trains an SVR with linear kernel and specified hyperparameters."""
    svr_linear = SVR(kernel='linear', C=C, epsilon=epsilon)
    multi_output_svr_linear = MultiOutputRegressor(svr_linear)
    multi_output_svr_linear.fit(X_train, y_train)

    #predictions test and training sets
    y_pred_svr_linear_test = multi_output_svr_linear.predict(X_test)
    y_pred_svr_linear_train = multi_output_svr_linear.predict(X_train)

    #overall R2, MSE, RMSE, MAE
    r2_svr_linear_test = r2_score(y_test, y_pred_svr_linear_test)
    r2_svr_linear_train = r2_score(y_train, y_pred_svr_linear_train)

    mse_svr_linear_test = mean_squared_error(y_test, y_pred_svr_linear_test)
    mse_svr_linear_train = mean_squared_error(y_train, y_pred_svr_linear_train)

    rmse_svr_linear_test = np.sqrt(mse_svr_linear_test)
    rmse_svr_linear_train = np.sqrt(mse_svr_linear_train)

    mae_svr_linear_test = mean_absolute_error(y_test, y_pred_svr_linear_test)
    mae_svr_linear_train = mean_absolute_error(y_train, y_pred_svr_linear_train)

    #R2, MSE, RMSE, MAE, and %E avg for each target variable (Test set)
    r2_roughness_svr_linear_test = r2_score(y_test['roughness'], y_pred_svr_linear_test[:, 0])
    mse_roughness_svr_linear_test = mean_squared_error(y_test['roughness'], y_pred_svr_linear_test[:, 0])
    rmse_roughness_svr_linear_test = np.sqrt(mse_roughness_svr_linear_test)
    mae_roughness_svr_linear_test = mean_absolute_error(y_test['roughness'], y_pred_svr_linear_test[:, 0])
    roughness_pe_avg_svr_linear_test = calculate_average_percentage_error(y_test['roughness'], y_pred_svr_linear_test[:, 0])

    r2_tensile_strength_svr_linear_test = r2_score(y_test['tension_strength'], y_pred_svr_linear_test[:, 1])
    mse_tensile_strength_svr_linear_test = mean_squared_error(y_test['tension_strength'], y_pred_svr_linear_test[:, 1])
    rmse_tensile_strength_svr_linear_test = np.sqrt(mse_tensile_strength_svr_linear_test)
    mae_tensile_strength_svr_linear_test = mean_absolute_error(y_test['tension_strength'], y_pred_svr_linear_test[:, 1])
    tensile_strength_pe_avg_svr_linear_test = calculate_average_percentage_error(y_test['tension_strength'], y_pred_svr_linear_test[:, 1])

    r2_elongation_svr_linear_test = r2_score(y_test['elongation'], y_pred_svr_linear_test[:, 2])
    mse_elongation_svr_linear_test = mean_squared_error(y_test['elongation'], y_pred_svr_linear_test[:, 2])
    rmse_elongation_svr_linear_test = np.sqrt(mse_elongation_svr_linear_test)
    mae_elongation_svr_linear_test = mean_absolute_error(y_test['elongation'], y_pred_svr_linear_test[:, 2])
    elongation_pe_avg_svr_linear_test = calculate_average_percentage_error(y_test['elongation'], y_pred_svr_linear_test[:, 2])

    #R2, MSE, RMSE, MAE, and %E avg for each target variable (Train set)
    r2_roughness_svr_linear_train = r2_score(y_train['roughness'], y_pred_svr_linear_train[:, 0])
    mse_roughness_svr_linear_train = mean_squared_error(y_train['roughness'], y_pred_svr_linear_train[:, 0])
    rmse_roughness_svr_linear_train = np.sqrt(mse_roughness_svr_linear_train)
    mae_roughness_svr_linear_train = mean_absolute_error(y_train['roughness'], y_pred_svr_linear_train[:, 0])
    roughness_pe_avg_svr_linear_train = calculate_average_percentage_error(y_train['roughness'], y_pred_svr_linear_train[:, 0])

    r2_tensile_strength_svr_linear_train = r2_score(y_train['tension_strength'], y_pred_svr_linear_train[:, 1])
    mse_tensile_strength_svr_linear_train = mean_squared_error(y_train['tension_strength'], y_pred_svr_linear_train[:, 1])
    rmse_tensile_strength_svr_linear_train = np.sqrt(mse_tensile_strength_svr_linear_train)
    mae_tensile_strength_svr_linear_train = mean_absolute_error(y_train['tension_strength'], y_pred_svr_linear_train[:, 1])
    tensile_strength_pe_avg_svr_linear_train = calculate_average_percentage_error(y_train['tension_strength'], y_pred_svr_linear_train[:, 1])

    r2_elongation_svr_linear_train = r2_score(y_train['elongation'], y_pred_svr_linear_train[:, 2])
    mse_elongation_svr_linear_train = mean_squared_error(y_train['elongation'], y_pred_svr_linear_train[:, 2])
    rmse_elongation_svr_linear_train = np.sqrt(mse_elongation_svr_linear_train)
    mae_elongation_svr_linear_train = mean_absolute_error(y_train['elongation'], y_pred_svr_linear_train[:, 2])
    elongation_pe_avg_svr_linear_train = calculate_average_percentage_error(y_train['elongation'], y_pred_svr_linear_train[:, 2])

    metrics = {
        "Module name": "Support Vector Regressor (Linear)",
        "R2 (Test)": r2_svr_linear_test,
        "R2 (Train)": r2_svr_linear_train,
        "MSE (Test)": mse_svr_linear_test,
        "MSE (Train)": mse_svr_linear_train,
        "RMSE (Test)": rmse_svr_linear_test,
        "RMSE (Train)": rmse_svr_linear_test,
        "MAE (Test)": mae_svr_linear_test,
        "MAE (Train)": mae_svr_linear_train,

        "R2 roughness (Test)": r2_roughness_svr_linear_test,
        "R2 roughness (Train)": r2_roughness_svr_linear_train,
        "MSE roughness (Test)": mse_roughness_svr_linear_test,
        "MSE roughness (Train)": mse_roughness_svr_linear_train,
        "RMSE roughness (Test)": rmse_roughness_svr_linear_test,
        "RMSE roughness (Train)": rmse_roughness_svr_linear_train,
        "MAE roughness (Test)": mae_roughness_svr_linear_test,
        "MAE roughness (Train)": mae_roughness_svr_linear_train,
        "roughness %E avg (Test)": roughness_pe_avg_svr_linear_test,
        "roughness %E avg (Train)": roughness_pe_avg_svr_linear_train,

        "R2 Tensile Strength (Test)": r2_tensile_strength_svr_linear_test,
        "R2 Tensile Strength (Train)": r2_tensile_strength_svr_linear_train,
        "MSE Tensile Strength (Test)": mse_tensile_strength_svr_linear_test,
        "MSE Tensile Strength (Train)": mse_tensile_strength_svr_linear_train,
        "RMSE Tensile Strength (Test)": rmse_tensile_strength_svr_linear_test,
        "RMSE Tensile Strength (Train)": rmse_tensile_strength_svr_linear_train,
        "MAE Tensile Strength (Test)": mae_tensile_strength_svr_linear_test,
        "MAE Tensile Strength (Train)": mae_tensile_strength_svr_linear_train,
        "Tensile Strength %E avg (Test)": tensile_strength_pe_avg_svr_linear_test,
        "Tensile Strength %E avg (Train)": tensile_strength_pe_avg_svr_linear_train,

        "R2 elongation (Test)": r2_elongation_svr_linear_test,
        "R2 elongation (Train)": r2_elongation_svr_linear_train,
        "MSE elongation (Test)": mse_elongation_svr_linear_test,
        "MSE elongation (Train)": mse_elongation_svr_linear_train,
        "RMSE elongation (Test)": rmse_elongation_svr_linear_test,
        "RMSE elongation (Train)": rmse_elongation_svr_linear_train,
        "MAE elongation (Test)": mae_elongation_svr_linear_test,
        "MAE elongation (Train)": mae_elongation_svr_linear_train,
        "elongation %E avg (Test)": elongation_pe_avg_svr_linear_test,
        "elongation %E avg (Train)": elongation_pe_avg_svr_linear_train,
    }
    return multi_output_svr_linear, metrics
