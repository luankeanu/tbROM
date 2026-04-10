import pandas as pd
import sklearn
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from utils import calculate_average_percentage_error


class CustomKNNRegressor:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

        if self.y_train.ndim > 1 and self.y_train.shape[1] == 1:
             self.y_train = self.y_train.ravel()



    def predict(self, X_test):
        X_test = np.array(X_test)
        predictions = []
        for test_point in X_test:
            distances = self._calculate_distances(test_point)
            nearest_neighbors_indices = np.argsort(distances)[:self.n_neighbors]
            nearest_neighbors_values = self.y_train[nearest_neighbors_indices]

            if nearest_neighbors_values.ndim == 1: 
                predicted_value = np.mean(nearest_neighbors_values)
            elif nearest_neighbors_values.ndim > 1: 
                 predicted_value = np.mean(nearest_neighbors_values, axis=0)
            else:
                 predicted_value = np.mean(nearest_neighbors_values)

            predictions.append(predicted_value)


        predicted_output = np.array(predictions)
        if predicted_output.ndim == 1 and self.y_train.ndim > 1 and self.y_train.shape[1] > 1:

             pass 


        if predicted_output.ndim == 1:
             predicted_output = predicted_output.reshape(-1, 1)


        return predicted_output


    def _calculate_distances(self, test_point):
        return np.sqrt(np.sum((self.X_train - test_point)**2, axis=1))


def train_knn(X_train, y_train, X_test, y_test, n_neighbors=5):

    if isinstance(y_train, pd.DataFrame):
        y_train_df = y_train
        y_train_array = y_train.values
    elif isinstance(y_train, pd.Series):
        y_train_df = y_train 
        y_train_array = y_train.values.reshape(-1, 1) 
    else: 
        y_train_df = pd.DataFrame(y_train) 
        y_train_array = y_train
        print("Warning: y_train is not a DataFrame or Series. Per-target metrics might fail if structure is unexpected.")


    if isinstance(y_test, pd.DataFrame):
        y_test_df = y_test
        y_test_array = y_test.values
    elif isinstance(y_test, pd.Series):
        y_test_df = y_test 
        y_test_array = y_test.values.reshape(-1, 1) 
    else: 
        y_test_df = pd.DataFrame(y_test) 
        y_test_array = y_test
        print("Warning: y_test is not a DataFrame or Series. Per-target metrics might fail if structure is unexpected.")


    #instantiate Custom KNN Regressor
    knn_model = CustomKNNRegressor(n_neighbors=n_neighbors) 

    #train the model
    knn_model.fit(X_train, y_train_array) 

    #predictions test and training sets
    y_pred_knn_test = knn_model.predict(X_test)
    y_pred_knn_train = knn_model.predict(X_train)

    if y_pred_knn_test.ndim == 1:
        y_pred_knn_test = y_pred_knn_test.reshape(-1, 1)
    if y_pred_knn_train.ndim == 1:
        y_pred_knn_train = y_pred_knn_train.reshape(-1, 1)


    # overall R2, MSE, RMSE, MAE
    # Use the array versions for overall metrics
    r2_knn_test = r2_score(y_test_array, y_pred_knn_test, multioutput='uniform_average')
    r2_knn_train = r2_score(y_train_array, y_pred_knn_train, multioutput='uniform_average')

    mse_knn_test = mean_squared_error(y_test_array, y_pred_knn_test, multioutput='uniform_average')
    mse_knn_train = mean_squared_error(y_train_array, y_pred_knn_train, multioutput='uniform_average')

    rmse_knn_test = np.sqrt(mse_knn_test)
    rmse_knn_train = np.sqrt(mse_knn_train)

    mae_knn_test = mean_absolute_error(y_test_array, y_pred_knn_test, multioutput='uniform_average')
    mae_knn_train = mean_absolute_error(y_train_array, y_pred_knn_train, multioutput='uniform_average')

    #R2, MSE, RMSE, MAE, and %E avg for each target variable (Test set)
    target_variables = y_test_df.columns.tolist() 

    r2_roughness_knn_test = r2_score(y_test_df['roughness'], y_pred_knn_test[:, 0])
    mse_roughness_knn_test = mean_squared_error(y_test_df['roughness'], y_pred_knn_test[:, 0])
    rmse_roughness_knn_test = np.sqrt(mse_roughness_knn_test)
    mae_roughness_knn_test = mean_absolute_error(y_test_df['roughness'], y_pred_knn_test[:, 0])
    roughness_pe_avg_knn_test = calculate_average_percentage_error(y_test_df['roughness'], y_pred_knn_test[:, 0])

    r2_tensile_strength_knn_test = r2_score(y_test_df['tension_strength'], y_pred_knn_test[:, 1])
    mse_tensile_strength_knn_test = mean_squared_error(y_test_df['tension_strength'], y_pred_knn_test[:, 1])
    rmse_tensile_strength_knn_test = np.sqrt(mse_tensile_strength_knn_test)
    mae_tensile_strength_knn_test = mean_absolute_error(y_test_df['tension_strength'], y_pred_knn_test[:, 1])
    tensile_strength_pe_avg_knn_test = calculate_average_percentage_error(y_test_df['tension_strength'], y_pred_knn_test[:, 1])

    r2_elongation_knn_test = r2_score(y_test_df['elongation'], y_pred_knn_test[:, 2])
    mse_elongation_knn_test = mean_squared_error(y_test_df['elongation'], y_pred_knn_test[:, 2])
    rmse_elongation_knn_test = np.sqrt(mse_elongation_knn_test)
    mae_elongation_knn_test = mean_absolute_error(y_test_df['elongation'], y_pred_knn_test[:, 2])
    elongation_pe_avg_knn_test = calculate_average_percentage_error(y_test_df['elongation'], y_pred_knn_test[:, 2])

    #R2, MSE, RMSE, MAE, and %E avg for each target variable (Train set)
    r2_roughness_knn_train = r2_score(y_train_df['roughness'], y_pred_knn_train[:, 0])
    mse_roughness_knn_train = mean_squared_error(y_train_df['roughness'], y_pred_knn_train[:, 0])
    rmse_roughness_knn_train = np.sqrt(mse_roughness_knn_train)
    mae_roughness_knn_train = mean_absolute_error(y_train_df['roughness'], y_pred_knn_train[:, 0])
    roughness_pe_avg_knn_train = calculate_average_percentage_error(y_train_df['roughness'], y_pred_knn_train[:, 0])

    r2_tensile_strength_knn_train = r2_score(y_train_df['tension_strength'], y_pred_knn_train[:, 1])
    mse_tensile_strength_knn_train = mean_squared_error(y_train_df['tension_strength'], y_pred_knn_train[:, 1])
    rmse_tensile_strength_knn_train = np.sqrt(mse_tensile_strength_knn_train)
    mae_tensile_strength_knn_train = mean_absolute_error(y_train_df['tension_strength'], y_pred_knn_train[:, 1])
    tensile_strength_pe_avg_knn_train = calculate_average_percentage_error(y_train_df['tension_strength'], y_pred_knn_train[:, 1])

    r2_elongation_knn_train = r2_score(y_train_df['elongation'], y_pred_knn_train[:, 2])
    mse_elongation_knn_train = mean_squared_error(y_train_df['elongation'], y_pred_knn_train[:, 2])
    rmse_elongation_knn_train = np.sqrt(mse_elongation_knn_train)
    mae_elongation_knn_train = mean_absolute_error(y_train_df['elongation'], y_pred_knn_train[:, 2])
    elongation_pe_avg_knn_train = calculate_average_percentage_error(y_train_df['elongation'], y_pred_knn_train[:, 2])

    metrics = {
        "Module name": "KNN Regressor (Custom)", 
        "R2 (Test)": r2_knn_test,
        "R2 (Train)": r2_knn_train,
        "MSE (Test)": mse_knn_test,
        "MSE (Train)": mse_knn_train,
        "RMSE (Test)": rmse_knn_test,
        "RMSE (Train)": rmse_knn_train,
        "MAE (Test)": mae_knn_test,
        "MAE (Train)": mae_knn_train,

        "R2 roughness (Test)": r2_roughness_knn_test,
        "R2 roughness (Train)": r2_roughness_knn_train,
        "MSE roughness (Test)": mse_roughness_knn_test,
        "MSE roughness (Train)": mse_roughness_knn_train,
        "RMSE roughness (Test)": rmse_roughness_knn_test,
        "RMSE roughness (Train)": rmse_roughness_knn_train,
        "MAE roughness (Test)": mae_roughness_knn_test,
        "MAE roughness (Train)": mae_roughness_knn_train,
        "roughness %E avg (Test)": roughness_pe_avg_knn_test,
        "roughness %E avg (Train)": roughness_pe_avg_knn_train,

        "R2 Tensile Strength (Test)": r2_tensile_strength_knn_test,
        "R2 Tensile Strength (Train)": r2_tensile_strength_knn_train,
        "MSE Tensile Strength (Test)": mse_tensile_strength_knn_test,
        "MSE Tensile Strength (Train)": mse_tensile_strength_knn_train,
        "RMSE Tensile Strength (Test)": rmse_tensile_strength_knn_test,
        "RMSE Tensile Strength (Train)": rmse_tensile_strength_knn_train,
        "MAE Tensile Strength (Test)": mae_tensile_strength_knn_test,
        "MAE Tensile Strength (Train)": mae_tensile_strength_knn_train,
        "Tensile Strength %E avg (Test)": tensile_strength_pe_avg_knn_test,
        "Tensile Strength %E avg (Train)": tensile_strength_pe_avg_knn_train,

        "R2 elongation (Test)": r2_elongation_knn_test,
        "R2 elongation (Train)": r2_elongation_knn_train,
        "MSE elongation (Test)": mse_elongation_knn_test,
        "MSE elongation (Train)": mse_elongation_knn_train,
        "RMSE elongation (Test)": rmse_elongation_knn_test,
        "RMSE elongation (Train)": rmse_elongation_knn_train,
        "MAE elongation (Test)": mae_elongation_knn_test,
        "MAE elongation (Train)": mae_elongation_knn_train,
        "elongation %E avg (Test)": elongation_pe_avg_knn_test,
        "elongation %E avg (Train)": elongation_pe_avg_knn_train,
    }
    return knn_model, metrics 