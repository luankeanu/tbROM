import pandas as pd
import sklearn
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from utils import calculate_average_percentage_error
from sklearn.linear_model import ElasticNet

def train_elastic_net(X_train, y_train, X_test, y_test, features_to_use=None, alpha=0.001, l1_ratio=0.3, max_iter=10000, tol=0.0001): # Add max_iter and tol
    if features_to_use is not None:
        X_train_subset = X_train[features_to_use]
        X_test_subset = X_test[features_to_use]
    else:
        X_train_subset = X_train
        X_test_subset = X_test

    elastic_net_model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42, max_iter=max_iter, tol=tol) # Use the parameters
    elastic_net_model.fit(X_train_subset, y_train)


    #predictions test and training sets
    y_pred_elastic_net_test = elastic_net_model.predict(X_test_subset)
    y_pred_elastic_net_train = elastic_net_model.predict(X_train_subset)

    #overall R2, MSE, RMSE, MAE
    r2_elastic_net_test = r2_score(y_test, y_pred_elastic_net_test)
    r2_elastic_net_train = r2_score(y_train, y_pred_elastic_net_train)

    mse_elastic_net_test = mean_squared_error(y_test, y_pred_elastic_net_test)
    mse_elastic_net_train = mean_squared_error(y_train, y_pred_elastic_net_train)

    rmse_elastic_net_test = np.sqrt(mse_elastic_net_test)
    rmse_elastic_net_train = np.sqrt(mse_elastic_net_train)

    mae_elastic_net_test = mean_absolute_error(y_test, y_pred_elastic_net_test)
    mae_elastic_net_train = mean_absolute_error(y_train, y_pred_elastic_net_train)

    #R2, MSE, RMSE, MAE, and %E avg for each target variable (Test set)
    r2_roughness_en_test = r2_score(y_test['roughness'], y_pred_elastic_net_test[:, 0])
    mse_roughness_en_test = mean_squared_error(y_test['roughness'], y_pred_elastic_net_test[:, 0])
    rmse_roughness_en_test = np.sqrt(mse_roughness_en_test)
    mae_roughness_en_test = mean_absolute_error(y_test['roughness'], y_pred_elastic_net_test[:, 0])
    roughness_pe_avg_en_test = calculate_average_percentage_error(y_test['roughness'], y_pred_elastic_net_test[:, 0])

    r2_tensile_strength_en_test = r2_score(y_test['tension_strength'], y_pred_elastic_net_test[:, 1])
    mse_tensile_strength_en_test = mean_squared_error(y_test['tension_strength'], y_pred_elastic_net_test[:, 1])
    rmse_tensile_strength_en_test = np.sqrt(mse_tensile_strength_en_test)
    mae_tensile_strength_en_test = mean_absolute_error(y_test['tension_strength'], y_pred_elastic_net_test[:, 1])
    tensile_strength_pe_avg_en_test = calculate_average_percentage_error(y_test['tension_strength'], y_pred_elastic_net_test[:, 1])

    r2_elongation_en_test = r2_score(y_test['elongation'], y_pred_elastic_net_test[:, 2])
    mse_elongation_en_test = mean_squared_error(y_test['elongation'], y_pred_elastic_net_test[:, 2])
    rmse_elongation_en_test = np.sqrt(mse_elongation_en_test)
    mae_elongation_en_test = mean_absolute_error(y_test['elongation'], y_pred_elastic_net_test[:, 2])
    elongation_pe_avg_en_test = calculate_average_percentage_error(y_test['elongation'], y_pred_elastic_net_test[:, 2])

    #R2, MSE, RMSE, MAE, and %E avg for each target variable (Train set)
    r2_roughness_en_train = r2_score(y_train['roughness'], y_pred_elastic_net_train[:, 0])
    mse_roughness_en_train = mean_squared_error(y_train['roughness'], y_pred_elastic_net_train[:, 0])
    rmse_roughness_en_train = np.sqrt(mse_roughness_en_train)
    mae_roughness_en_train = mean_absolute_error(y_train['roughness'], y_pred_elastic_net_train[:, 0])
    roughness_pe_avg_en_train = calculate_average_percentage_error(y_train['roughness'], y_pred_elastic_net_train[:, 0])

    r2_tensile_strength_en_train = r2_score(y_train['tension_strength'], y_pred_elastic_net_train[:, 1])
    mse_tensile_strength_en_train = mean_squared_error(y_train['tension_strength'], y_pred_elastic_net_train[:, 1])
    rmse_tensile_strength_en_train = np.sqrt(mse_tensile_strength_en_train)
    mae_tensile_strength_en_train = mean_absolute_error(y_train['tension_strength'], y_pred_elastic_net_train[:, 1])
    tensile_strength_pe_avg_en_train = calculate_average_percentage_error(y_train['tension_strength'], y_pred_elastic_net_train[:, 1])

    r2_elongation_en_train = r2_score(y_train['elongation'], y_pred_elastic_net_train[:, 2])
    mse_elongation_en_train = mean_squared_error(y_train['elongation'], y_pred_elastic_net_train[:, 2])
    rmse_elongation_en_train = np.sqrt(mse_elongation_en_train)
    mae_elongation_en_train = mean_absolute_error(y_train['elongation'], y_pred_elastic_net_train[:, 2])
    elongation_pe_avg_en_train = calculate_average_percentage_error(y_train['elongation'], y_pred_elastic_net_train[:, 2])

    metrics = {
        "Module name": "Elastic Net (Selected Features)",
        "R2 (Test)": r2_elastic_net_test,
        "R2 (Train)": r2_elastic_net_train,
        "MSE (Test)": mse_elastic_net_test,
        "MSE (Train)": mse_elastic_net_train,
        "RMSE (Test)": rmse_elastic_net_test,
        "RMSE (Train)": rmse_elastic_net_train,
        "MAE (Test)": mae_elastic_net_test,
        "MAE (Train)": mae_elastic_net_train,

        "R2 roughness (Test)": r2_roughness_en_test,
        "R2 roughness (Train)": r2_roughness_en_train,
        "MSE roughness (Test)": mse_roughness_en_test,
        "MSE roughness (Train)": mse_roughness_en_train,
        "RMSE roughness (Test)": rmse_roughness_en_test,
        "RMSE roughness (Train)": rmse_roughness_en_train,
        "MAE roughness (Test)": mae_roughness_en_test,
        "MAE roughness (Train)": mae_roughness_en_train,
        "roughness %E avg (Test)": roughness_pe_avg_en_test,
        "roughness %E avg (Train)": roughness_pe_avg_en_train,

        "R2 Tensile Strength (Test)": r2_tensile_strength_en_test,
        "R2 Tensile Strength (Train)": r2_tensile_strength_en_train,
        "MSE Tensile Strength (Test)": mse_tensile_strength_en_test,
        "MSE Tensile Strength (Train)": mse_tensile_strength_en_train,
        "RMSE Tensile Strength (Test)": rmse_tensile_strength_en_test,
        "RMSE Tensile Strength (Train)": rmse_tensile_strength_en_train,
        "MAE Tensile Strength (Test)": mae_tensile_strength_en_test,
        "MAE Tensile Strength (Train)": mae_tensile_strength_en_train,
        "Tensile Strength %E avg (Test)": tensile_strength_pe_avg_en_test,
        "Tensile Strength %E avg (Train)": tensile_strength_pe_avg_en_train,

        "R2 elongation (Test)": r2_elongation_en_test,
        "R2 elongation (Train)": r2_elongation_en_train,
        "MSE elongation (Test)": mse_elongation_en_test,
        "MSE elongation (Train)": mse_elongation_en_train,
        "RMSE elongation (Test)": rmse_elongation_en_test,
        "RMSE elongation (Train)": rmse_elongation_en_train,
        "MAE elongation (Test)": mae_elongation_en_test,
        "MAE elongation (Train)": mae_elongation_en_train,
        "elongation %E avg (Test)": elongation_pe_avg_en_test,
        "elongation %E avg (Train)": elongation_pe_avg_en_train,
    }
    return elastic_net_model, metrics