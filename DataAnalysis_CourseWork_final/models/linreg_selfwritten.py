import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from utils import calculate_average_percentage_error


class CustomLinearRegression:
    def __init__(self):
        self.coef_ = None

    def fit(self, X, y):
        X_bias = np.c_[np.ones(X.shape[0]), X]
        y_array = y.values  
        self.coef_ = np.linalg.pinv(X_bias.T @ X_bias) @ X_bias.T @ y_array

    def predict(self, X):
        X_bias = np.c_[np.ones(X.shape[0]), X]
        return X_bias @ self.coef_


def train_linear_regression(X_train, y_train, X_test, y_test):
    model = CustomLinearRegression()
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    #overall metrics
    r2_test = r2_score(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)

    mse_test = mean_squared_error(y_test, y_pred_test)
    mse_train = mean_squared_error(y_train, y_pred_train)

    rmse_test = np.sqrt(mse_test)
    rmse_train = np.sqrt(mse_train)

    mae_test = mean_absolute_error(y_test, y_pred_test)
    mae_train = mean_absolute_error(y_train, y_pred_train)

    #per-target metrics (assuming columns: 0=roughness, 1=tension_strength, 2=elongation)
    r2_roughness_test = r2_score(y_test['roughness'], np.array(y_pred_test[:, 0]))
    mse_roughness_test = mean_squared_error(y_test['roughness'], np.array(y_pred_test[:, 0]))
    rmse_roughness_test = np.sqrt(mse_roughness_test)
    mae_roughness_test = mean_absolute_error(y_test['roughness'], np.array(y_pred_test[:, 0]))
    roughness_pe_avg_test = calculate_average_percentage_error(y_test['roughness'], np.array(y_pred_test[:, 0]))

    r2_tension_test = r2_score(y_test['tension_strength'], np.array(y_pred_test[:, 1]))
    mse_tension_test = mean_squared_error(y_test['tension_strength'], np.array(y_pred_test[:, 1]))
    rmse_tension_test = np.sqrt(mse_tension_test)
    mae_tension_test = mean_absolute_error(y_test['tension_strength'], np.array(y_pred_test[:, 1]))
    tension_pe_avg_test = calculate_average_percentage_error(y_test['tension_strength'], np.array(y_pred_test[:, 1]))

    r2_elongation_test = r2_score(y_test['elongation'], np.array(y_pred_test[:, 2]))
    mse_elongation_test = mean_squared_error(y_test['elongation'], np.array(y_pred_test[:, 2]))
    rmse_elongation_test = np.sqrt(mse_elongation_test)
    mae_elongation_test = mean_absolute_error(y_test['elongation'], np.array(y_pred_test[:, 2]))
    elongation_pe_avg_test = calculate_average_percentage_error(y_test['elongation'], np.array(y_pred_test[:, 2]))

    #training set
    r2_roughness_train = r2_score(y_train['roughness'], np.array(y_pred_train[:, 0]))
    mse_roughness_train = mean_squared_error(y_train['roughness'], np.array(y_pred_train[:, 0]))
    rmse_roughness_train = np.sqrt(mse_roughness_train)
    mae_roughness_train = mean_absolute_error(y_train['roughness'], np.array(y_pred_train[:, 0]))
    roughness_pe_avg_train = calculate_average_percentage_error(y_train['roughness'], np.array(y_pred_train[:, 0]))

    r2_tension_train = r2_score(y_train['tension_strength'], np.array(y_pred_train[:, 1]))
    mse_tension_train = mean_squared_error(y_train['tension_strength'], np.array(y_pred_train[:, 1]))
    rmse_tension_train = np.sqrt(mse_tension_train)
    mae_tension_train = mean_absolute_error(y_train['tension_strength'], np.array(y_pred_train[:, 1]))
    tension_pe_avg_train = calculate_average_percentage_error(y_train['tension_strength'], np.array(y_pred_train[:, 1]))

    r2_elongation_train = r2_score(y_train['elongation'], np.array(y_pred_train[:, 2]))
    mse_elongation_train = mean_squared_error(y_train['elongation'], np.array(y_pred_train[:, 2]))
    rmse_elongation_train = np.sqrt(mse_elongation_train)
    mae_elongation_train = mean_absolute_error(y_train['elongation'], np.array(y_pred_train[:, 2]))
    elongation_pe_avg_train = calculate_average_percentage_error(y_train['elongation'], np.array(y_pred_train[:, 2]))

    metrics = {
        "Module name": "Linear Regression (Custom)",
        "R2 (Test)": r2_test,
        "R2 (Train)": r2_train,
        "MSE (Test)": mse_test,
        "MSE (Train)": mse_train,
        "RMSE (Test)": rmse_test,
        "RMSE (Train)": rmse_train,
        "MAE (Test)": mae_test,
        "MAE (Train)": mae_train,

        "R2 roughness (Test)": r2_roughness_test,
        "R2 roughness (Train)": r2_roughness_train,
        "MSE roughness (Test)": mse_roughness_test,
        "MSE roughness (Train)": mse_roughness_train,
        "RMSE roughness (Test)": rmse_roughness_test,
        "RMSE roughness (Train)": rmse_roughness_train,
        "MAE roughness (Test)": mae_roughness_test,
        "MAE roughness (Train)": mae_roughness_train,
        "roughness %E avg (Test)": roughness_pe_avg_test,
        "roughness %E avg (Train)": roughness_pe_avg_train,

        "R2 Tensile Strength (Test)": r2_tension_test,
        "R2 Tensile Strength (Train)": r2_tension_train,
        "MSE Tensile Strength (Test)": mse_tension_test,
        "MSE Tensile Strength (Train)": mse_tension_train,
        "RMSE Tensile Strength (Test)": rmse_tension_test,
        "RMSE Tensile Strength (Train)": rmse_tension_train,
        "MAE Tensile Strength (Test)": mae_tension_test,
        "MAE Tensile Strength (Train)": mae_tension_train,
        "Tensile Strength %E avg (Test)": tension_pe_avg_test,
        "Tensile Strength %E avg (Train)": tension_pe_avg_train,

        "R2 elongation (Test)": r2_elongation_test,
        "R2 elongation (Train)": r2_elongation_train,
        "MSE elongation (Test)": mse_elongation_test,
        "MSE elongation (Train)": mse_elongation_train,
        "RMSE elongation (Test)": rmse_elongation_test,
        "RMSE elongation (Train)": rmse_elongation_train,
        "MAE elongation (Test)": mae_elongation_test,
        "MAE elongation (Train)": mae_elongation_train,
        "elongation %E avg (Test)": elongation_pe_avg_test,
        "elongation %E avg (Train)": elongation_pe_avg_train,
    }

    return model, metrics