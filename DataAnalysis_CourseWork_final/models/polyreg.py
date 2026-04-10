import pandas as pd
import sklearn
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from utils import calculate_average_percentage_error 

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures



def train_polynomial_regression(X_train, y_train, X_test, y_test, degree=3):
    """Trains a Polynomial Regression model with a specified degree, matching linreg.py structure."""

    if isinstance(y_train, pd.DataFrame):
        y_train_df = y_train
        y_train_array = y_train.values
    elif isinstance(y_train, pd.Series):
        y_train_df = y_train 
        y_train_array = y_train.values.reshape(-1, 1) 
    else: 
        y_train_df = pd.DataFrame(y_train)
        y_train_array = y_train


    if isinstance(y_test, pd.DataFrame):
        y_test_df = y_test
        y_test_array = y_test.values
    elif isinstance(y_test, pd.Series):
        y_test_df = y_test 
        y_test_array = y_test.values.reshape(-1, 1) 
    else: 

        y_test_df = pd.DataFrame(y_test)
        y_test_array = y_test


    poly = PolynomialFeatures(degree=degree, include_bias=False) 

    X_train_poly = poly.fit_transform(X_train) 
    X_test_poly = poly.transform(X_test)       


    #instantiate the Linear Regression model
    linear_reg_model = LinearRegression()

    #train the Linear Regression model on the transformed training data
    linear_reg_model.fit(X_train_poly, y_train_array)

    #make predictions on the transformed test and training data
    y_pred_test = linear_reg_model.predict(X_test_poly)
    y_pred_train = linear_reg_model.predict(X_train_poly)

    if y_test_array.ndim > 1 and y_test_array.shape[1] > 1:
         if y_pred_test.ndim == 1:
              y_pred_test = y_pred_test.reshape(-1, y_test_array.shape[1])
         if y_pred_train.ndim == 1:
              y_pred_train = y_pred_train.reshape(-1, y_train_array.shape[1])

    elif y_test_array.ndim == 1 and y_pred_test.ndim > 1 and y_pred_test.shape[1] == 1:
         y_pred_test = y_pred_test.ravel()
         y_pred_train = y_pred_train.ravel()



    #overall R2, MSE, RMSE, MAE

    r2_poly_reg_test = r2_score(y_test_array, y_pred_test, multioutput='uniform_average')
    r2_poly_reg_train = r2_score(y_train_array, y_pred_train, multioutput='uniform_average')

    mse_poly_reg_test = mean_squared_error(y_test_array, y_pred_test, multioutput='uniform_average')
    mse_poly_reg_train = mean_squared_error(y_train_array, y_pred_train, multioutput='uniform_average')

    rmse_poly_reg_test = np.sqrt(mse_poly_reg_test)
    rmse_poly_reg_train = np.sqrt(mse_poly_reg_train)

    mae_poly_reg_test = mean_absolute_error(y_test_array, y_pred_test, multioutput='uniform_average')
    mae_poly_reg_train = mean_absolute_error(y_train_array, y_pred_train, multioutput='uniform_average')

    #R2, MSE, RMSE, MAE, and %E avg for each target variable (Test set)
    if y_test_df.shape[1] > 0: 
        try:
            r2_roughness_poly_test = r2_score(y_test_df['roughness'], y_pred_test[:, 0])
            mse_roughness_poly_test = mean_squared_error(y_test_df['roughness'], y_pred_test[:, 0])
            rmse_roughness_poly_test = np.sqrt(mse_roughness_poly_test)
            mae_roughness_poly_test = mean_absolute_error(y_test_df['roughness'], y_pred_test[:, 0])
            roughness_pe_avg_poly_test = calculate_average_percentage_error(y_test_df['roughness'], y_pred_test[:, 0])
        except KeyError:

             r2_roughness_poly_test, mse_roughness_poly_test, rmse_roughness_poly_test, mae_roughness_poly_test, roughness_pe_avg_poly_test = [np.nan] * 5


        try:
            r2_tensile_strength_poly_test = r2_score(y_test_df['tension_strength'], y_pred_test[:, 1])
            mse_tensile_strength_poly_test = mean_squared_error(y_test_df['tension_strength'], y_pred_test[:, 1])
            rmse_tensile_strength_poly_test = np.sqrt(mse_tensile_strength_poly_test)
            mae_tensile_strength_poly_test = mean_absolute_error(y_test_df['tension_strength'], y_pred_test[:, 1])
            tensile_strength_pe_avg_poly_test = calculate_average_percentage_error(y_test_df['tension_strength'], y_pred_test[:, 1])
        except (KeyError, IndexError): 

             r2_tensile_strength_poly_test, mse_tensile_strength_poly_test, rmse_tensile_strength_poly_test, mae_tensile_strength_poly_test, tensile_strength_pe_avg_poly_test = [np.nan] * 5

        try:
            r2_elongation_poly_test = r2_score(y_test_df['elongation'], y_pred_test[:, 2])
            mse_elongation_poly_test = mean_squared_error(y_test_df['elongation'], y_pred_test[:, 2])
            rmse_elongation_poly_test = np.sqrt(mse_elongation_poly_test)
            mae_elongation_poly_test = mean_absolute_error(y_test_df['elongation'], y_pred_test[:, 2])
            elongation_pe_avg_poly_test = calculate_average_percentage_error(y_test_df['elongation'], y_pred_test[:, 2])
        except (KeyError, IndexError): 

             r2_elongation_poly_test, mse_elongation_poly_test, rmse_elongation_poly_test, mae_elongation_poly_test, elongation_pe_avg_poly_test = [np.nan] * 5
    else: 
         r2_roughness_poly_test, mse_roughness_poly_test, rmse_roughness_poly_test, mae_roughness_poly_test, roughness_pe_avg_poly_test = [np.nan] * 5
         r2_tensile_strength_poly_test, mse_tensile_strength_poly_test, rmse_tensile_strength_poly_test, mae_tensile_strength_poly_test, tensile_strength_pe_avg_poly_test = [np.nan] * 5
         r2_elongation_poly_test, mse_elongation_poly_test, rmse_elongation_poly_test, mae_elongation_poly_test, elongation_pe_avg_poly_test = [np.nan] * 5

    if y_train_df.shape[1] > 0:
        try:
            r2_roughness_poly_train = r2_score(y_train_df['roughness'], y_pred_train[:, 0])
            mse_roughness_poly_train = mean_squared_error(y_train_df['roughness'], y_pred_train[:, 0])
            rmse_roughness_poly_train = np.sqrt(mse_roughness_poly_train)
            mae_roughness_poly_train = mean_absolute_error(y_train_df['roughness'], y_pred_train[:, 0])
            roughness_pe_avg_poly_train = calculate_average_percentage_error(y_train_df['roughness'], y_pred_train[:, 0])
        except KeyError:

            r2_roughness_poly_train, mse_roughness_poly_train, rmse_roughness_poly_train, mae_roughness_poly_train, roughness_pe_avg_poly_train = [np.nan] * 5

        try:
            r2_tensile_strength_poly_train = r2_score(y_train_df['tension_strength'], y_pred_train[:, 1])
            mse_tensile_strength_poly_train = mean_squared_error(y_train_df['tension_strength'], y_pred_train[:, 1])
            rmse_tensile_strength_poly_train = np.sqrt(mse_tensile_strength_poly_train)
            mae_tensile_strength_poly_train = mean_absolute_error(y_train_df['tension_strength'], y_pred_train[:, 1])
            tensile_strength_pe_avg_poly_train = calculate_average_percentage_error(y_train_df['tension_strength'], y_pred_train[:, 1])
        except (KeyError, IndexError): 

            r2_tensile_strength_poly_train, mse_tensile_strength_poly_train, rmse_tensile_strength_poly_train, mae_tensile_strength_poly_train, tensile_strength_pe_avg_poly_train = [np.nan] * 5

        try:
            r2_elongation_poly_train = r2_score(y_train_df['elongation'], y_pred_train[:, 2])
            mse_elongation_poly_train = mean_squared_error(y_train_df['elongation'], y_pred_train[:, 2])
            rmse_elongation_poly_train = np.sqrt(mse_elongation_poly_train)
            mae_elongation_poly_train = mean_absolute_error(y_train_df['elongation'], y_pred_train[:, 2])
            elongation_pe_avg_poly_train = calculate_average_percentage_error(y_train_df['elongation'], y_pred_train[:, 2])
        except (KeyError, IndexError): 

            r2_elongation_poly_train, mse_elongation_poly_train, rmse_elongation_poly_train, mae_elongation_poly_train, elongation_pe_avg_poly_train = [np.nan] * 5
    else: 
         r2_roughness_poly_train, mse_roughness_poly_train, rmse_roughness_poly_train, mae_roughness_poly_train, roughness_pe_avg_poly_train = [np.nan] * 5
         r2_tensile_strength_poly_train, mse_tensile_strength_poly_train, rmse_tensile_strength_poly_train, mae_tensile_strength_poly_train, tensile_strength_pe_avg_poly_train = [np.nan] * 5
         r2_elongation_poly_train, mse_elongation_poly_train, rmse_elongation_poly_train, mae_elongation_poly_train, elongation_pe_avg_poly_train = [np.nan] * 5


    metrics = {
        "Module name": f"Polynomial Regression (Degree {degree})",
        "R2 (Test)": r2_poly_reg_test,
        "R2 (Train)": r2_poly_reg_train,
        "MSE (Test)": mse_poly_reg_test,
        "MSE (Train)": mse_poly_reg_train,
        "RMSE (Test)": rmse_poly_reg_test,
        "RMSE (Train)": rmse_poly_reg_train,
        "MAE (Test)": mae_poly_reg_test,
        "MAE (Train)": mae_poly_reg_train,

        "R2 roughness (Test)": r2_roughness_poly_test,
        "R2 roughness (Train)": r2_roughness_poly_train,
        "MSE roughness (Test)": mse_roughness_poly_test,
        "MSE roughness (Train)": mse_roughness_poly_train,
        "RMSE roughness (Test)": rmse_roughness_poly_test,
        "RMSE roughness (Train)": rmse_roughness_poly_train,
        "MAE roughness (Test)": mae_roughness_poly_test,
        "MAE roughness (Train)": mae_roughness_poly_train,
        "roughness %E avg (Test)": roughness_pe_avg_poly_test,
        "roughness %E avg (Train)": roughness_pe_avg_poly_train,

        "R2 Tensile Strength (Test)": r2_tensile_strength_poly_test,
        "R2 Tensile Strength (Train)": r2_tensile_strength_poly_train,
        "MSE Tensile Strength (Test)": mse_tensile_strength_poly_test,
        "MSE Tensile Strength (Train)": mse_tensile_strength_poly_train,
        "RMSE Tensile Strength (Test)": rmse_tensile_strength_poly_test,
        "RMSE Tensile Strength (Train)": rmse_tensile_strength_poly_train,
        "MAE Tensile Strength (Test)": mae_tensile_strength_poly_test,
        "MAE Tensile Strength (Train)": mae_tensile_strength_poly_train,
        "Tensile Strength %E avg (Test)": tensile_strength_pe_avg_poly_test,
        "Tensile Strength %E avg (Train)": tensile_strength_pe_avg_poly_train, 

        "R2 elongation (Test)": r2_elongation_poly_test,
        "R2 elongation (Train)": r2_elongation_poly_train,
        "MSE elongation (Test)": mse_elongation_poly_test,
        "MSE elongation (Train)": mse_elongation_poly_train,
        "RMSE elongation (Test)": rmse_elongation_poly_test,
        "RMSE elongation (Train)": rmse_elongation_poly_train,
        "MAE elongation (Test)": mae_elongation_poly_test,
        "MAE elongation (Train)": mae_elongation_poly_train,
        "elongation %E avg (Test)": elongation_pe_avg_poly_test,
        "elongation %E avg (Train)": elongation_pe_avg_poly_train,
    }

    return linear_reg_model, metrics