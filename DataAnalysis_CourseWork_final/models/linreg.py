import pandas as pd
import sklearn
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from utils import calculate_average_percentage_error

from sklearn.linear_model import LinearRegression



def train_linear_regression(X_train, y_train, X_test, y_test):
    linear_reg_model = LinearRegression()
    linear_reg_model.fit(X_train, y_train)

    #predictions test and training sets
    y_pred_linear_reg_test = linear_reg_model.predict(X_test)
    y_pred_linear_reg_train = linear_reg_model.predict(X_train)

    #overall R2, MSE, RMSE, MAE
    r2_linear_reg_test = r2_score(y_test, y_pred_linear_reg_test)
    r2_linear_reg_train = r2_score(y_train, y_pred_linear_reg_train)

    mse_linear_reg_test = mean_squared_error(y_test, y_pred_linear_reg_test)
    mse_linear_reg_train = mean_squared_error(y_train, y_pred_linear_reg_train)

    rmse_linear_reg_test = np.sqrt(mse_linear_reg_test)
    rmse_linear_reg_train = np.sqrt(mse_linear_reg_train)

    mae_linear_reg_test = mean_absolute_error(y_test, y_pred_linear_reg_test)
    mae_linear_reg_train = mean_absolute_error(y_train, y_pred_linear_reg_train)

    #R2, MSE, RMSE, MAE, and %E avg for each target variable (Test set)
    r2_roughness_lr_test = r2_score(y_test['roughness'], y_pred_linear_reg_test[:, 0])
    mse_roughness_lr_test = mean_squared_error(y_test['roughness'], y_pred_linear_reg_test[:, 0])
    rmse_roughness_lr_test = np.sqrt(mse_roughness_lr_test)
    mae_roughness_lr_test = mean_absolute_error(y_test['roughness'], y_pred_linear_reg_test[:, 0])
    roughness_pe_avg_lr_test = calculate_average_percentage_error(y_test['roughness'], y_pred_linear_reg_test[:, 0])

    r2_tensile_strength_lr_test = r2_score(y_test['tension_strength'], y_pred_linear_reg_test[:, 1])
    mse_tensile_strength_lr_test = mean_squared_error(y_test['tension_strength'], y_pred_linear_reg_test[:, 1])
    rmse_tensile_strength_lr_test = np.sqrt(mse_tensile_strength_lr_test)
    mae_tensile_strength_lr_test = mean_absolute_error(y_test['tension_strength'], y_pred_linear_reg_test[:, 1])
    tensile_strength_pe_avg_lr_test = calculate_average_percentage_error(y_test['tension_strength'], y_pred_linear_reg_test[:, 1])

    r2_elongation_lr_test = r2_score(y_test['elongation'], y_pred_linear_reg_test[:, 2])
    mse_elongation_lr_test = mean_squared_error(y_test['elongation'], y_pred_linear_reg_test[:, 2])
    rmse_elongation_lr_test = np.sqrt(mse_elongation_lr_test)
    mae_elongation_lr_test = mean_absolute_error(y_test['elongation'], y_pred_linear_reg_test[:, 2])
    elongation_pe_avg_lr_test = calculate_average_percentage_error(y_test['elongation'], y_pred_linear_reg_test[:, 2])

    #R2, MSE, RMSE, MAE, and %E avg for each target variable (Train set)
    r2_roughness_lr_train = r2_score(y_train['roughness'], y_pred_linear_reg_train[:, 0])
    mse_roughness_lr_train = mean_squared_error(y_train['roughness'], y_pred_linear_reg_train[:, 0])
    rmse_roughness_lr_train = np.sqrt(mse_roughness_lr_train)
    mae_roughness_lr_train = mean_absolute_error(y_train['roughness'], y_pred_linear_reg_train[:, 0])
    roughness_pe_avg_lr_train = calculate_average_percentage_error(y_train['roughness'], y_pred_linear_reg_train[:, 0])

    r2_tensile_strength_lr_train = r2_score(y_train['tension_strength'], y_pred_linear_reg_train[:, 1])
    mse_tensile_strength_lr_train = mean_squared_error(y_train['tension_strength'], y_pred_linear_reg_train[:, 1])
    rmse_tensile_strength_lr_train = np.sqrt(mse_tensile_strength_lr_train)
    mae_tensile_strength_lr_train = mean_absolute_error(y_train['tension_strength'], y_pred_linear_reg_train[:, 1])
    tensile_strength_pe_avg_lr_train = calculate_average_percentage_error(y_train['tension_strength'], y_pred_linear_reg_train[:, 1])

    r2_elongation_lr_train = r2_score(y_train['elongation'], y_pred_linear_reg_train[:, 2])
    mse_elongation_lr_train = mean_squared_error(y_train['elongation'], y_pred_linear_reg_train[:, 2])
    rmse_elongation_lr_train = np.sqrt(mse_elongation_lr_train)
    mae_elongation_lr_train = mean_absolute_error(y_train['elongation'], y_pred_linear_reg_train[:, 2])
    elongation_pe_avg_lr_train = calculate_average_percentage_error(y_train['elongation'], y_pred_linear_reg_train[:, 2])

    metrics = {
        "Module name": "Linear Regression",
        "R2 (Test)": r2_linear_reg_test,
        "R2 (Train)": r2_linear_reg_train,
        "MSE (Test)": mse_linear_reg_test,
        "MSE (Train)": mse_linear_reg_train,
        "RMSE (Test)": rmse_linear_reg_test,
        "RMSE (Train)": rmse_linear_reg_train,
        "MAE (Test)": mae_linear_reg_test,
        "MAE (Train)": mae_linear_reg_train,

        "R2 roughness (Test)": r2_roughness_lr_test,
        "R2 roughness (Train)": r2_roughness_lr_train,
        "MSE roughness (Test)": mse_roughness_lr_test,
        "MSE roughness (Train)": mse_roughness_lr_train,
        "RMSE roughness (Test)": rmse_roughness_lr_test,
        "RMSE roughness (Train)": rmse_roughness_lr_train,
        "MAE roughness (Test)": mae_roughness_lr_test,
        "MAE roughness (Train)": mae_roughness_lr_train,
        "roughness %E avg (Test)": roughness_pe_avg_lr_test,
        "roughness %E avg (Train)": roughness_pe_avg_lr_train,

        "R2 Tensile Strength (Test)": r2_tensile_strength_lr_test,
        "R2 Tensile Strength (Train)": r2_tensile_strength_lr_train,
        "MSE Tensile Strength (Test)": mse_tensile_strength_lr_test,
        "MSE Tensile Strength (Train)": mse_tensile_strength_lr_train,
        "RMSE Tensile Strength (Test)": rmse_tensile_strength_lr_test,
        "RMSE Tensile Strength (Train)": rmse_tensile_strength_lr_train,
        "MAE Tensile Strength (Test)": mae_tensile_strength_lr_test,
        "MAE Tensile Strength (Train)": mae_tensile_strength_lr_train,
        "Tensile Strength %E avg (Test)": tensile_strength_pe_avg_lr_test,
        "Tensile Strength %E avg (Train)": tensile_strength_pe_avg_lr_train,

        "R2 elongation (Test)": r2_elongation_lr_test,
        "R2 elongation (Train)": r2_elongation_lr_train,
        "MSE elongation (Test)": mse_elongation_lr_test,
        "MSE elongation (Train)": mse_elongation_lr_train,
        "RMSE elongation (Test)": rmse_elongation_lr_test,
        "RMSE elongation (Train)": rmse_elongation_lr_train,
        "MAE elongation (Test)": mae_elongation_lr_test,
        "MAE elongation (Train)": mae_elongation_lr_train,
        "elongation %E avg (Test)": elongation_pe_avg_lr_test,
        "elongation %E avg (Train)": elongation_pe_avg_lr_train,
    }
    return linear_reg_model, metrics
