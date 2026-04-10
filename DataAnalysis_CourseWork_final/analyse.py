"""
first we create a new dataframe to analyse the data from results_df by comparing the
    differente metrics we have collected so far (also export both dfs in .csv form)
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler

from read import *
from modules import *
from utils import *


### FEATURE SCALING ###
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)


### CORRELATION HEATMAP ###

#create correlation matrix for features
correlation_matrix = X.corr()

#create heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap of Features")
plt.xticks(fontsize=8) # Reduce x-axis label font size
plt.yticks(fontsize=8) # Reduce y-axis label font size
plt.tight_layout()
plt.savefig('files/feature_correlation_heatmap.png')

### FEATURE IMPORTANCE ANALYSIS ###

trained_models = train_models(X_train, y_train, X_test, y_test)
importance_data = {}

if 'Random Forest Regressor (Selected Features)' in trained_models:
    rf_model_selected = trained_models['Random Forest Regressor (Selected Features)']
    features_rf = ['layer_height', 'nozzle_temperature', 'print_speed'] #make sure this matches modules.py
    importance_data['Random Forest (Selected Features)'] = get_random_forest_feature_importance(rf_model_selected, features_rf)

if 'Decision Tree Regressor' in trained_models:
    dt_model = trained_models['Decision Tree Regressor']
    features_dt = ['layer_height', 'nozzle_temperature', 'print_speed'] #make sure this matches modules.py
    importance_data['Decision Tree'] = get_decision_tree_feature_importance(dt_model, features_dt)

if 'Linear Regression' in trained_models:
    lr_model = trained_models['Linear Regression']
    importance_data['Linear Regression'] = get_linear_regression_coefficients(lr_model, X_train.columns)

if 'Linear Regression (Custom)' in trained_models:
    lr_model_custom = trained_models['Linear Regression (Custom)']
    importance_data['Linear Regression (Custom)'] = get_custom_linear_regression_coefficients(lr_model_custom, X_train.columns)

if 'Ridge Regression' in trained_models:
    features_ridge = ['layer_height', 'nozzle_temperature', 'material_pla', 'wall_thickness', 'infill_pattern_honeycomb', 'bed_temperature'] #make sure this matches modules.py
    ridge_model = trained_models['Ridge Regression']
    importance_data['Ridge Regression'] = get_ridge_regression_coefficients(ridge_model, features_ridge)

if 'Lasso Regression' in trained_models:
    features_lasso = ['layer_height', 'nozzle_temperature', 'material_pla', 'wall_thickness', 'infill_pattern_honeycomb', 'bed_temperature'] #make sure this matches modules.py
    lasso_model = trained_models['Lasso Regression']
    importance_data['Lasso Regression'] = get_lasso_regression_coefficients(lasso_model, features_lasso)

if 'Elastic Net' in trained_models:
    features_en = ['layer_height', 'nozzle_temperature', 'material_pla', 'wall_thickness', 'infill_pattern_honeycomb', 'bed_temperature'] #make sure this matches modules.py
    elastic_net_model = trained_models['Elastic Net']
    importance_data['Elastic Net'] = get_elastic_net_coefficients(elastic_net_model, features_en)

if 'Polynomial Regression (Degree 3)' in trained_models:
    poly_reg_model = trained_models['Polynomial Regression (Degree 3)']
    original_features = X_train.columns
    fixed_degree_used = 2 
    importance_data['Polynomial Regression (Degree 3)'] = get_polynomial_regression_feature_importance(poly_reg_model, original_features, fixed_degree_used)

"""
SVR and KNN dont have native Feature Importance therefore we dont add them here.
"""
importance_df = pd.DataFrame(importance_data)


### ANALYSIS TABLE ###
"""
in modules.py we added a dataframe 'results_df' to store all values from the modules;
now we create a new dataframe 'analysis_df' to compare the metrics and store the comparisons.
"""

#initialize an empty DataFrame 'analysis_df' to store the results of the metrics analysis
analysis_df = pd.DataFrame(columns=[
    "Module name",
    "Overall R2 Train - Test Difference",
    "Overall MSE Test / Train Ratio",
    "Overall RMSE Train - Test Difference",
    "Overall MAE Test / Train Ratio",
    "Avg %E Difference (Test - Train)",

    "Roughness R2 Train - Test Difference",
    "Roughness MSE Test / Train Ratio",
    "Roughness MAE Difference (Test - Train)",
    "Roughness RMSE Test / Train Ratio",
    "Roughness %E Difference (Test - Train)",

    "Tensile Strength R2 Train - Test Difference",
    "Tensile Strength MSE Test / Train Ratio",
    "Tensile Strength MAE Difference (Test - Train)",
    "Tensile Strength RMSE Test / Train Ratio",
    "Tensile Strength %E Difference (Test - Train)",

    "Elongation R2 Train - Test Difference",
    "Elongation MSE Test / Train Ratio",
    "Elongation MAE Difference (Test - Train)",
    "Elongation RMSE Test / Train Ratio",
    "Elongation %E Difference (Test - Train)",
])


for index, row in results_df.iterrows():
    overall_r2_diff = row['R2 (Train)'] - row['R2 (Test)']
    overall_mse_ratio = row['MSE (Test)'] / row['MSE (Train)'] if row['MSE (Train)'] != 0 else float('inf')
    overall_rmse_diff = row['RMSE (Train)'] - row['RMSE (Test)']
    overall_mae_ratio = row['MAE (Test)'] / row['MAE (Train)'] if row['MAE (Train)'] != 0 else float('inf')
    avg_pe_diff = (row['roughness %E avg (Test)'] - row['roughness %E avg (Train)'] +
                   row['Tensile Strength %E avg (Test)'] - row['Tensile Strength %E avg (Train)'] +
                   row['elongation %E avg (Test)'] - row['elongation %E avg (Train)']) / 3


    roughness_r2_diff = row['R2 roughness (Train)'] - row['R2 roughness (Test)']
    roughness_mse_ratio = row['MSE roughness (Test)'] / row['MSE roughness (Train)'] if row['MSE roughness (Train)'] != 0 else float('inf')
    roughness_mae_diff = row['MAE roughness (Test)'] - row['MAE roughness (Train)']
    roughness_rmse_ratio = row['RMSE roughness (Test)'] / row['RMSE roughness (Train)'] if row['RMSE roughness (Train)'] != 0 else float('inf')
    roughness_pe_diff = row['roughness %E avg (Test)'] - row['roughness %E avg (Train)']

    tensile_strength_r2_diff = row['R2 Tensile Strength (Train)'] - row['R2 Tensile Strength (Test)']
    tensile_strength_mse_ratio = row['MSE Tensile Strength (Test)'] / row['MSE Tensile Strength (Train)'] if row['MSE Tensile Strength (Train)'] != 0 else float('inf')
    tensile_strength_mae_diff = row['MAE Tensile Strength (Test)'] - row['MAE Tensile Strength (Train)']
    tensile_strength_rmse_ratio = row['RMSE Tensile Strength (Test)'] / row['RMSE Tensile Strength (Train)'] if row['RMSE Tensile Strength (Train)'] != 0 else float('inf')
    tensile_strength_pe_diff = row['Tensile Strength %E avg (Test)'] - row['Tensile Strength %E avg (Train)']

    elongation_r2_diff = row['R2 elongation (Train)'] - row['R2 elongation (Test)']
    elongation_mse_ratio = row['MSE elongation (Test)'] / row['MSE elongation (Train)'] if row['MSE elongation (Train)'] != 0 else float('inf')
    elongation_mae_diff = row['MAE elongation (Test)'] - row['MAE elongation (Train)']
    elongation_rmse_ratio = row['RMSE elongation (Test)'] / row['RMSE elongation (Train)'] if row['RMSE elongation (Train)'] != 0 else float('inf')
    elongation_pe_diff = row['elongation %E avg (Test)'] - row['elongation %E avg (Train)']

    analysis_df = pd.concat([analysis_df, pd.Series({
        "Module name": row['Module name'],
        "Overall R2 Train - Test Difference": overall_r2_diff,
        "Overall MSE Test / Train Ratio": overall_mse_ratio,
        "Overall RMSE Train - Test Difference": overall_rmse_diff,
        "Overall MAE Test / Train Ratio": overall_mae_ratio,
        "Avg %E Difference (Test - Train)": avg_pe_diff,

        "Roughness R2 Train - Test Difference": roughness_r2_diff,
        "Roughness MSE Test / Train Ratio": roughness_mse_ratio,
        "Roughness MAE Difference (Test - Train)": roughness_mae_diff,
        "Roughness RMSE Test / Train Ratio": roughness_rmse_ratio,
        "Roughness %E Difference (Test - Train)": roughness_pe_diff,

        "Tensile Strength R2 Train - Test Difference": tensile_strength_r2_diff,
        "Tensile Strength MSE Test / Train Ratio": tensile_strength_mse_ratio,
        "Tensile Strength MAE Difference (Test - Train)": tensile_strength_mae_diff,
        "Tensile Strength RMSE Test / Train Ratio": tensile_strength_rmse_ratio,
        "Tensile Strength %E Difference (Test - Train)": tensile_strength_pe_diff,

        "Elongation R2 Train - Test Difference": elongation_r2_diff,
        "Elongation MSE Test / Train Ratio": elongation_mse_ratio,
        "Elongation MAE Difference (Test - Train)": elongation_mae_diff,
        "Elongation RMSE Test / Train Ratio": elongation_rmse_ratio,
        "Elongation %E Difference (Test - Train)": elongation_pe_diff,

    }).to_frame().T], ignore_index=True)




### EXPORT FILES ###
print("Heatmap has been exported to 'feature_coorelation_heatmap.png'")

results_df.to_csv('files/results.csv', index=False)
print("Model performance metrics have been exported to 'results.csv'")

analysis_df.to_csv('files/analysis.csv', index=False)
print("Model performance metrics have been exported to 'analysis.csv'")

importance_df.to_csv('files/feature_importance.csv', index=True, index_label='')
print("Feature importance analysis has been exported to 'feature_importance.csv'")



### GRAPHS ###

module_names = results_df['Module name'].tolist() #convert to list for Plotly

# Plot 1: R-squared Scores (Test Set)
fig1 = px.bar(results_df, x='Module name', y='R2 (Test)',
              title='P1 - Model Performance Comparison by R-squared (Test Set)',
              labels={'R2 (Test)': 'R-squared'})
fig1.update_layout(xaxis_tickangle=-45)
fig1.show()

# Plot 2: MSE Scores (Test Set)
fig2 = px.bar(results_df, x='Module name', y='MSE (Test)',
              title='P2 - Model Performance Comparison by Mean Squared Error (MSE) (Test Set)',
              labels={'MSE (Test)': 'Mean Squared Error'})
fig2.update_layout(xaxis_tickangle=-45)
fig2.show()

# Plot 3: RMSE Scores (Test Set)
fig3 = px.bar(results_df, x='Module name', y='RMSE (Test)',
              title='P3 - Model Performance Comparison by Root Mean Squared Error (RMSE) (Test Set)',
              labels={'RMSE (Test)': 'Root Mean Squared Error'})
fig3.update_layout(xaxis_tickangle=-45)
fig3.show()

# Plot 4: MAE Scores (Test Set)
fig4 = px.bar(results_df, x='Module name', y='MAE (Test)',
              title='P4 - Model Performance Comparison by Mean Absolute Error (MAE) (Test Set)',
              labels={'MAE (Test)': 'Mean Absolute Error'})
fig4.update_layout(xaxis_tickangle=-45)
fig4.show()

# Plot 5: Comparison of R-squared Scores for Each Target Variable (Test Set)
fig5 = go.Figure(data=[
    go.Bar(name='Roughness', x=module_names, y=results_df['R2 roughness (Test)'], marker_color='skyblue'),
    go.Bar(name='Tensile Strength', x=module_names, y=results_df['R2 Tensile Strength (Test)'], marker_color='lightcoral'),
    go.Bar(name='Elongation', x=module_names, y=results_df['R2 elongation (Test)'], marker_color='lightsteelblue')
])
fig5.update_layout(barmode='group', title='P5 - Comparison of R-squared Scores for Each Target Variable (Test Set)',
                  xaxis_tickangle=-45, yaxis_title='R-squared')
fig5.show()

# Plot 6: Test vs. Train Performance Comparison (Bar Chart of R-squared)
fig6 = go.Figure(data=[
    go.Bar(name='Test R2', x=module_names, y=results_df['R2 (Test)'], marker_color='skyblue'),
    go.Bar(name='Train R2', x=module_names, y=results_df['R2 (Train)'], marker_color='lightcoral')
])
fig6.update_layout(barmode='group', title='P6 - Test vs. Train Performance Comparison (R-squared)',
                  xaxis_tickangle=-45, yaxis_title='R-squared')
fig6.show()

# Plot 7: Feature Importance Analysis for Random Forest Regressor
if 'Random Forest (Selected Features)' in importance_df.columns:
    fig7 = px.bar(importance_df, x=importance_df.index, y='Random Forest (Selected Features)',
                  title='P7 - Feature Importance Analysis for Random Forest Regressor',
                  labels={'Random Forest (Selected Features)': 'Importance', 'index': 'Feature'})
    fig7.show()

# Plot 8: Residual Plots - Random Forest Regressor
if 'Random Forest Regressor (Selected Features)' in trained_models:
    rf_model_selected = trained_models['Random Forest Regressor (Selected Features)']
    y_pred_rf = rf_model_selected.predict(X_test[['layer_height', 'nozzle_temperature', 'print_speed']])
    target_variables = y_test.columns.tolist()

    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    fig8 = make_subplots(rows=1, cols=3, subplot_titles=target_variables)

    for i, target in enumerate(target_variables):
        residuals = y_test[target] - y_pred_rf[:, i]
        fig8.add_trace(go.Scatter(x=y_pred_rf[:, i], y=residuals, mode='markers',
                                  name=target,
                                  hovertemplate=f'Predicted {target}: %{{x}}<br>Residual: %{{y}}<extra></extra>'),
                          row=1, col=i + 1)

    fig8.update_layout(title_text='P8 - Residual Plots - Random Forest Regressor')
    fig8.update_xaxes(title_text='Predicted Value', row=1, col=1)
    fig8.update_yaxes(title_text='Residual', row=1, col=1)
    fig8.update_xaxes(title_text='Predicted Value', row=1, col=2)
    fig8.update_yaxes(title_text='Residual', row=1, col=2)
    fig8.update_xaxes(title_text='Predicted Value', row=1, col=3)
    fig8.update_yaxes(title_text='Residual', row=1, col=3)

    fig8.show()

# Plot 9: Predicted vs. Actual Values - Linear Regression (Sklearn)
if 'Linear Regression' in trained_models:
    lr_model_sklearn = trained_models['Linear Regression']
    y_pred_lr_sklearn = lr_model_sklearn.predict(X_test_scaled)

    target_variables = y_test.columns.tolist()

    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    fig9 = make_subplots(rows=1, cols=3, subplot_titles=target_variables)

    for i, target in enumerate(target_variables):
        #scatter plot of predicted vs. actual
        fig9.add_trace(go.Scatter(x=y_test[target], y=y_pred_lr_sklearn[:, i], mode='markers',
                                 name=target,
                                 hovertemplate=f'Actual {target}: %{{x}}<br>Predicted {target}: %{{y}}<extra></extra>'),
                          row=1, col=i + 1)
        #add ideal regression line (y=x)
        min_val = min(y_test[target])
        max_val = max(y_test[target])
        fig9.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines',
                                 name='Ideal Fit',
                                 line=dict(color='red', dash='dash')),
                          row=1, col=i + 1)

    fig9.update_layout(title_text='P9 - Predicted vs. Actual Values - Linear Regression (Sklearn)')
    fig9.update_xaxes(title_text='Actual Value', row=1, col=1)
    fig9.update_yaxes(title_text='Predicted Value', row=1, col=1)
    fig9.update_xaxes(title_text='Actual Value', row=1, col=2)
    fig9.update_yaxes(title_text='Predicted Value', row=1, col=2)
    fig9.update_xaxes(title_text='Actual Value', row=1, col=3)
    fig9.update_yaxes(title_text='Predicted Value', row=1, col=3)

    fig9.show()

# Plot 10: Predicted vs. Actual Values - Linear Regression (Custom)
if 'Linear Regression (Custom)' in trained_models:
    lr_model_custom = trained_models['Linear Regression (Custom)']
    y_pred_lr_custom = lr_model_custom.predict(X_test) 

    target_variables = y_test.columns.tolist()

    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    fig10 = make_subplots(rows=1, cols=3, subplot_titles=target_variables)

    for i, target in enumerate(target_variables):
        #scatter plot of predicted vs. actual
        fig10.add_trace(go.Scatter(x=y_test[target], y=y_pred_lr_custom[:, i], mode='markers',
                                  name=target,
                                  hovertemplate=f'Actual {target}: %{{x}}<br>Predicted {target}: %{{y}}<extra></extra>'),
                            row=1, col=i + 1)
        #add ideal regression line (y=x)
        min_val = min(y_test[target])
        max_val = max(y_test[target])
        fig10.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines',
                                  name='Ideal Fit',
                                  line=dict(color='red', dash='dash')),
                            row=1, col=i + 1)

    fig10.update_layout(title_text='P10 - Predicted vs. Actual Values - Linear Regression (Custom)')
    fig10.update_xaxes(title_text='Actual Value', row=1, col=1)
    fig10.update_yaxes(title_text='Predicted Value', row=1, col=1)
    fig10.update_xaxes(title_text='Actual Value', row=1, col=2)
    fig10.update_yaxes(title_text='Predicted Value', row=1, col=2)
    fig10.update_xaxes(title_text='Actual Value', row=1, col=3)
    fig10.update_yaxes(title_text='Predicted Value', row=1, col=3)

    fig10.show()

# Plot 11: Prediction Curve with Scatter - Support Vector Regressor (RBF)
if 'Support Vector Regressor (RBF)' in trained_models:
    svr_rbf_model = trained_models['Support Vector Regressor (RBF)']
    feature_to_vary = 'layer_height'
    target_variable = 'roughness'

    feature_values = np.linspace(X_train_scaled[feature_to_vary].min(), X_train_scaled[feature_to_vary].max(), 100)
    other_features = [col for col in X_train_scaled.columns if col != feature_to_vary]
    mean_values = X_train_scaled[other_features].mean().to_dict()

    X_plot = []
    for val in feature_values:
        sample = {feature_to_vary: val}
        sample.update(mean_values)
        X_plot.append(list(sample.values()))

    X_plot = np.array(X_plot)
    target_index = y_train.columns.get_loc(target_variable)
    y_pred_curve = svr_rbf_model.predict(X_plot)[:, target_index]

    #get actual and predicted values for the test set for the chosen feature and target
    test_feature_values = X_test_scaled[feature_to_vary]
    y_test_values = y_test[target_variable]
    y_pred_test = svr_rbf_model.predict(X_test_scaled)[:, target_index]

    import plotly.express as px
    import plotly.graph_objects as go

    fig11 = px.line(x=feature_values, y=y_pred_curve,
                    title=f'P11 - SVR (RBF) Prediction Curve with Scatter for {target_variable} vs. {feature_to_vary}',
                    labels={feature_to_vary: feature_to_vary, target_variable: f'Predicted {target_variable}'},
                    color_discrete_sequence=['blue']) 

    fig11.add_trace(go.Scatter(x=test_feature_values, y=y_test_values, mode='markers',
                               name='Actual Values',
                               marker=dict(color='green'),
                               hovertemplate=f'Actual {target_variable}: %{{y}}<br>{feature_to_vary}: %{{x}}<extra></extra>'))

    fig11.add_trace(go.Scatter(x=test_feature_values, y=y_pred_test, mode='markers',
                               name='Predicted Values (Test Set)',
                               marker=dict(color='red'),
                               hovertemplate=f'Predicted {target_variable}: %{{y}}<br>{feature_to_vary}: %{{x}}<extra></extra>'))

    fig11.update_layout(xaxis_title=feature_to_vary, yaxis_title=target_variable)

    fig11.show()

# Plot 12: Predicted vs. Actual Values - Support Vector Regressor (RBF)
if 'Support Vector Regressor (RBF)' in trained_models:
    svr_rbf_model = trained_models['Support Vector Regressor (RBF)']
    y_pred_svr_rbf = svr_rbf_model.predict(X_test_scaled)
    target_variables = y_test.columns.tolist()

    from plotly.subplots import make_subplots

    fig12 = make_subplots(rows=1, cols=3, subplot_titles=target_variables)

    for i, target in enumerate(target_variables):
        fig12.add_trace(go.Scatter(x=y_test[target], y=y_pred_svr_rbf[:, i], mode='markers',
                                  name=target,
                                  hovertemplate=f'Actual {target}: %{{x}}<br>Predicted {target}: %{{y}}<extra></extra>'),
                            row=1, col=i + 1)

    fig12.update_layout(title_text='P12 - Predicted vs. Actual Values - Support Vector Regressor (RBF)')
    fig12.update_xaxes(title_text='Actual Value', row=1, col=1)
    fig12.update_yaxes(title_text='Predicted Value', row=1, col=1)
    fig12.update_xaxes(title_text='Actual Value', row=1, col=2)
    fig12.update_yaxes(title_text='Predicted Value', row=1, col=2)
    fig12.update_xaxes(title_text='Actual Value', row=1, col=3)
    fig12.update_yaxes(title_text='Predicted Value', row=1, col=3)

    fig12.show()

# Plot 13: Predicted vs. Actual Values - KNN Regressor
if 'KNN Regressor' in trained_models:
    knn_model = trained_models['KNN Regressor']
    y_pred_knn = knn_model.predict(X_test_scaled)
    target_variables = y_test.columns.tolist()

    from plotly.subplots import make_subplots

    fig13 = make_subplots(rows=1, cols=3, subplot_titles=target_variables)

    for i, target in enumerate(target_variables):
        fig13.add_trace(go.Scatter(x=y_test[target], y=y_pred_knn[:, i], mode='markers',
                                  name=target,
                                  hovertemplate=f'Actual {target}: %{{x}}<br>Predicted {target}: %{{y}}<extra></extra>'),
                            row=1, col=i + 1)

    fig13.update_layout(title_text='P13 - Predicted vs. Actual Values - KNN Regressor')
    fig13.update_xaxes(title_text='Actual Value', row=1, col=1)
    fig13.update_yaxes(title_text='Predicted Value', row=1, col=1)
    fig13.update_xaxes(title_text='Actual Value', row=1, col=2)
    fig13.update_yaxes(title_text='Predicted Value', row=1, col=2)
    fig13.update_xaxes(title_text='Actual Value', row=1, col=3)
    fig13.update_yaxes(title_text='Predicted Value', row=1, col=3)

    fig13.show()

# Plot 14: Coefficient Plot - Ridge Regression
if 'Ridge Regression' in trained_models:
    ridge_model = trained_models['Ridge Regression']
    features_ridge = ['layer_height', 'nozzle_temperature', 'material_pla', 'wall_thickness', 'infill_pattern_honeycomb', 'bed_temperature']
    target_variables = y_train.columns.tolist()
    coefficients_data = []

    if hasattr(ridge_model, 'estimators_'):
        for i, estimator in enumerate(ridge_model.estimators_):
            if hasattr(estimator, 'coef_'):
                coeffs = estimator.coef_
                for j, coeff in enumerate(coeffs):
                    coefficients_data.append({
                        'Feature': features_ridge[j],
                        'Coefficient': coeff,
                        'Target': target_variables[i]
                    })
    elif hasattr(ridge_model, 'coef_'):
        coeffs = ridge_model.coef_
        if coeffs.ndim == 1: 
            for i, coeff in enumerate(coeffs):
                coefficients_data.append({
                    'Feature': features_ridge[i],
                    'Coefficient': coeff,
                    'Target': target_variables[0] if target_variables else 'Target 0'
                })
        elif coeffs.ndim == 2: 
            for i, target in enumerate(target_variables):
                for j, coeff in enumerate(coeffs[i]):
                    coefficients_data.append({
                        'Feature': features_ridge[j],
                        'Coefficient': coeff,
                        'Target': target
                    })

    coefficients_df = pd.DataFrame(coefficients_data)
    fig14 = px.bar(coefficients_df, x='Feature', y='Coefficient', color='Target',
                   title='P14 - Coefficient Plot - Ridge Regression',
                   labels={'y': 'Coefficient Value', 'x': 'Feature'})
    fig14.show()


# Plot 15: Coefficient Plot - Lasso Regression
if 'Lasso Regression' in trained_models:
    lasso_model = trained_models['Lasso Regression']
    features_lasso = ['layer_height', 'nozzle_temperature', 'material_pla', 'wall_thickness', 'infill_pattern_honeycomb', 'bed_temperature']
    target_variables = y_train.columns.tolist()
    coefficients_data = []

    if hasattr(lasso_model, 'estimators_'):
        for i, estimator in enumerate(lasso_model.estimators_):
            if hasattr(estimator, 'coef_'):
                coeffs = estimator.coef_
                for j, coeff in enumerate(coeffs):
                    coefficients_data.append({
                        'Feature': features_lasso[j],
                        'Coefficient': coeff,
                        'Target': target_variables[i]
                    })
    elif hasattr(lasso_model, 'coef_'):
        coeffs = lasso_model.coef_
        if coeffs.ndim == 1: 
            for i, coeff in enumerate(coeffs):
                coefficients_data.append({
                    'Feature': features_lasso[i],
                    'Coefficient': coeff,
                    'Target': target_variables[0] if target_variables else 'Target 0'
                })
        elif coeffs.ndim == 2: 
            for i, target in enumerate(target_variables):
                for j, coeff in enumerate(coeffs[i]):
                    coefficients_data.append({
                        'Feature': features_lasso[j],
                        'Coefficient': coeff,
                        'Target': target
                    })

    coefficients_df = pd.DataFrame(coefficients_data)
    fig15 = px.bar(coefficients_df, x='Feature', y='Coefficient', color='Target',
                   title='P15 - Coefficient Plot - Lasso Regression',
                   labels={'y': 'Coefficient Value', 'x': 'Feature'})
    fig15.show()

# Plot 16: Coefficient Plot - Elastic Net
if 'Elastic Net' in trained_models:
    elastic_net_model = trained_models['Elastic Net']
    features_en = ['layer_height', 'nozzle_temperature', 'material_pla', 'wall_thickness', 'infill_pattern_honeycomb', 'bed_temperature'] 
    target_variables = y_train.columns.tolist()
    coefficients_data = []

    if hasattr(elastic_net_model, 'estimators_'):
        for i, estimator in enumerate(elastic_net_model.estimators_):
            if hasattr(estimator, 'coef_'):
                coeffs = estimator.coef_
                for j, coeff in enumerate(coeffs):
                    coefficients_data.append({
                        'Feature': features_en[j],
                        'Coefficient': coeff,
                        'Target': target_variables[i]
                    })
    elif hasattr(elastic_net_model, 'coef_'):
        coeffs = elastic_net_model.coef_
        if coeffs.ndim == 1: 
            for i, coeff in enumerate(coeffs):
                coefficients_data.append({
                    'Feature': features_en[i],
                    'Coefficient': coeff,
                    'Target': target_variables[0] if target_variables else 'Target 0'
                })
        elif coeffs.ndim == 2: 
            for i, target in enumerate(target_variables):
                for j, coeff in enumerate(coeffs[i]):
                    coefficients_data.append({
                        'Feature': features_en[j],
                        'Coefficient': coeff,
                        'Target': target
                    })

    coefficients_df = pd.DataFrame(coefficients_data)
    fig16 = px.bar(coefficients_df, x='Feature', y='Coefficient', color='Target',
                   title='P16 - Coefficient Plot - Elastic Net',
                   labels={'y': 'Coefficient Value', 'x': 'Feature'})
    fig16.show()

# Plot 17: Predicted vs. Actual for Random Forest
if 'Random Forest Regressor (Selected Features)' in trained_models:
    rf_model_selected = trained_models['Random Forest Regressor (Selected Features)']
    y_pred_rf = rf_model_selected.predict(X_test[['layer_height', 'nozzle_temperature', 'print_speed']]) 
    target_variables = y_test.columns.tolist()

    from plotly.subplots import make_subplots

    fig17 = make_subplots(rows=1, cols=3, subplot_titles=target_variables)

    for i, target in enumerate(target_variables):
        fig17.add_trace(go.Scatter(x=y_test[target], y=y_pred_rf[:, i], mode='markers',
                                  name=target,
                                  hovertemplate=f'Actual {target}: %{{x}}<br>Predicted {target}: %{{y}}<extra></extra>'),
                          row=1, col=i + 1)

    fig17.update_layout(title_text='P17 - Predicted vs. Actual Values - Random Forest Regressor')
    fig17.update_xaxes(title_text='Actual Value', row=1, col=1)
    fig17.update_yaxes(title_text='Predicted Value', row=1, col=1)
    fig17.update_xaxes(title_text='Actual Value', row=1, col=2)
    fig17.update_yaxes(title_text='Predicted Value', row=1, col=2)
    fig17.update_xaxes(title_text='Actual Value', row=1, col=3)
    fig17.update_yaxes(title_text='Predicted Value', row=1, col=3)

    fig17.show()



# Plot 18: comparing R2 Before and After Tuning
"""
since we saved a frist version of the results before hyperparameter tuning we can use this data to
    plot a comparison between before and after hyperparameter tuning
"""

results_current_df = pd.read_csv('files/results.csv')
results_old_df = pd.read_csv('files_old/results_old.csv')

#models that went through hyperparameter tuning (adjusting for old names)
#add a placeholder for Polynomial Regression which will be handled by startswith filtering below
models_to_compare_current = [
    'Decision Tree Regressor (Selected Features)',
    'Random Forest Regressor (Selected Features)',
    'KNN Regressor (Selected Features)',
    'Support Vector Regressor (Linear)',
    'Support Vector Regressor (RBF)',
    'Ridge Regression (Selected Features)',
    'Lasso Regression (Selected Features)',
    'Elastic Net (Selected Features)',
]

models_to_compare_old = [
    'Decision Tree Regressor',
    'Random Forest Regressor',
    'KNN Regressor',
    'Support Vector Regressor (Linear)',
    'Support Vector Regressor (RBF)',
    'Ridge Regression',
    'Lasso Regression',
    'Elastic Net',
    'Polynomial Regression (Degree 2)' 
]

#filter DataFrames for the models to compare (and exclude Linear Regression)
#add .copy() to potentially avoid SettingWithCopyWarning
#modify filtering to explicitly include Polynomial Regression results based on name starting
results_current_filtered = results_current_df[
    (results_current_df['Module name'].isin(models_to_compare_current) |
     results_current_df['Module name'].str.startswith('Polynomial Regression')) & 
    (results_current_df['Module name'] != 'Linear Regression') 
].copy()

results_old_filtered = results_old_df[
    (results_old_df['Module name'].isin(models_to_compare_old) |
     results_old_df['Module name'].str.startswith('Polynomial Regression')) & 
    (results_old_df['Module name'] != 'Linear Regression') 
].copy()


#select R-squared metrics
r2_metrics = ['R2 (Test)', 'R2 (Train)']

#melt current results DataFrame
results_current_melted = results_current_filtered.melt(
    id_vars='Module name',
    value_vars=r2_metrics,
    var_name='Metric Type',
    value_name='R-squared Value'
)
results_current_melted['Tuning Status'] = 'New'

#melt old results DataFrame
results_old_melted = results_old_filtered.melt(
    id_vars='Module name',
    value_vars=r2_metrics,
    var_name='Metric Type',
    value_name='R-squared Value'
)
results_old_melted['Tuning Status'] = 'Old'

#create mapping dictionary for specific renames
name_mapping = {
    'Decision Tree Regressor': 'Decision Tree Regressor (Selected Features)',
    'Random Forest Regressor': 'Random Forest Regressor (Selected Features)',
    'KNN Regressor': 'KNN Regressor (Selected Features)',
    'Ridge Regression': 'Ridge Regression (Selected Features)',
    'Lasso Regression': 'Lasso Regression (Selected Features)',
    'Elastic Net': 'Elastic Net (Selected Features)',
}

#def function to apply consistent naming
def get_consistent_model_name(module_name):
    if module_name.startswith('Polynomial Regression'):
        return 'Polynomial Regression'
    return name_mapping.get(module_name, module_name) 

results_current_melted['Module name'] = results_current_melted['Module name'].apply(get_consistent_model_name)
results_old_melted['Module name'] = results_old_melted['Module name'].apply(get_consistent_model_name)

comparison_df = pd.concat([results_current_melted, results_old_melted])
comparison_df = comparison_df[comparison_df['Module name'] != 'Linear Regression'].copy() 


comparison_df = comparison_df.sort_values(by=['Module name', 'Tuning Status']).reset_index(drop=True)

#show plot 18
fig18 = px.bar(comparison_df,
             x='Module name',
             y='R-squared Value',
             color='Tuning Status',
             facet_col='Metric Type',
             barmode='group',
             title=f'P18 - Comparison of R-squared Before and After Tuning',
             labels={'R-squared Value': 'R-squared', 'Module name': 'Model', 'Tuning Status': 'Tuning Status', 'Metric Type': 'Dataset'})

# Optional: Improve layout
fig18.update_layout(xaxis={'categoryorder':'category ascending'}) # Center title

fig18.show()



# Plot 19: Comparison of Linear Regression and KNN Implementations (Sklearn vs Custom - R2 Only)
models_implementations_compare = [
    'Linear Regression', 'Linear Regression (Custom)',
    'KNN Regressor (Selected Features)', 'KNN Regressor (Custom)'
]
metrics_to_compare_r2_only = ['R2 (Test)', 'R2 (Train)']

data_implementations_compare = results_df[results_df['Module name'].isin(models_implementations_compare)].copy()

data_implementations_compare['Implementation Type'] = data_implementations_compare['Module name'].apply(
    lambda x: 'Linear Regression' if 'Linear Regression' in x else 'KNN Regressor'
)

data_implementations_compare['Implementation Detail'] = data_implementations_compare['Module name'].apply(
    lambda x: 'Custom' if '(Custom)' in x else 'Sklearn'
)

data_implementations_melted = data_implementations_compare.melt(
    id_vars=['Module name', 'Implementation Type', 'Implementation Detail'],
    value_vars=metrics_to_compare_r2_only,
    var_name='Metric Type',
    value_name='R-squared Value'
)

fig19 = make_subplots(rows=1, cols=2, subplot_titles=['R2 (Test)', 'R2 (Train)'])

data_r2_test = data_implementations_melted[data_implementations_melted['Metric Type'] == 'R2 (Test)'].copy() 

data_r2_test['Sort_Order'] = data_r2_test['Module name'].apply(
    lambda x: 0 if x == 'Linear Regression' else (1 if x == 'Linear Regression (Custom)' else (2 if x == 'KNN Regressor (Selected Features)' else 3))
)
data_r2_test = data_r2_test.sort_values('Sort_Order')


fig19.add_trace(go.Bar(
    x=[data_r2_test['Implementation Type'], data_r2_test['Implementation Detail']], 
    y=data_r2_test['R-squared Value'],
    name='R2 (Test)',
    marker_color=data_r2_test['Implementation Detail'].apply(lambda x: 'blue' if x == 'Sklearn' else 'red'), 
    showlegend=True 
), row=1, col=1)


data_r2_train = data_implementations_melted[data_implementations_melted['Metric Type'] == 'R2 (Train)'].copy() 

data_r2_train['Sort_Order'] = data_r2_train['Module name'].apply(
    lambda x: 0 if x == 'Linear Regression' else (1 if x == 'Linear Regression (Custom)' else (2 if x == 'KNN Regressor (Selected Features)' else 3)) 
)
data_r2_train = data_r2_train.sort_values('Sort_Order')

fig19.add_trace(go.Bar(
    x=[data_r2_train['Implementation Type'], data_r2_train['Implementation Detail']], 
    y=data_r2_train['R-squared Value'],
    name='R2 (Train)',
     marker_color=data_r2_train['Implementation Detail'].apply(lambda x: 'blue' if x == 'Sklearn' else 'red'), 
    showlegend=True 
), row=1, col=2)


fig19.update_layout(
    title_text='P19 - Comparison of Linear Regression and KNN Implementations (Sklearn vs Custom - R2 Only)',
    barmode='group', 
    legend_title_text='Implementation Type' 
)
fig19.update_yaxes(title_text='R-squared Value')

fig19.show()



# Plot 20: Comparison of All Four Implementations (R2 Only) - Alternative View

models_all_compare = [
    'Linear Regression', 'Linear Regression (Custom)',
    'KNN Regressor (Selected Features)', 'KNN Regressor (Custom)'
]
metrics_to_compare_r2_only = ['R2 (Test)', 'R2 (Train)']

data_all_compare = results_df[results_df['Module name'].isin(models_all_compare)].copy() 


data_all_melted = data_all_compare.melt(
    id_vars='Module name',
    value_vars=metrics_to_compare_r2_only,
    var_name='Metric Type',
    value_name='R-squared Value'
)

fig20 = px.bar(data_all_melted,
             x='Module name', 
             y='R-squared Value',
             color='Module name', 
             barmode='group',
             facet_col='Metric Type', 
             title='P20 - Comparison of All Four Implementations (R2 Only)',
             labels={'R-squared Value': 'R-squared Value', 'Module name': 'Model', 'Metric Type': 'Dataset'})


fig20.update_layout(showlegend=True)
fig20.update_xaxes(title_text='') 
fig20.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1])) 


fig20.show()
