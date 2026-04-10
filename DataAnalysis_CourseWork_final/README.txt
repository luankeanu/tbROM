GROUP L --> EG2002 Data Analysis Coursework Submission

Gabriela Nutsova
Luan Correia Gil
Niccolo Mandrendini Estense Paragues
Raela Cani
Rafael Santos

----##----##----

INTRODUCTION:
    This project focuses on predicting key performance indicators of a 3D printer using various machine
    learning regression models. The aim is to develop accurate and reliable models for predicting 'roughness', 
    'tension_strength', and 'elongation' based on different printer settings and material properties. 
    We explore a range of established regression techniques. The project encompasses data loading and preprocessing, 
    model training and hyperparameter optimization, performance evaluation using a comprehensive set of metrics, 
    and insightful data visualization to compare model performance and feature importance. We also investigate the 
    impact of the train-test split on model behavior by analyzing results across five different random states.

----##----##----

HOW TO USE:

    1. Ensure required libraries (seaborn, plotly, pandas, numpy, matplotlib, sklearn) are installed.
    2. Run 'python analyse.py' in your terminal to generate analysis outputs and plots in the 'files' folder.
    3. Open 'discussion.ipynb' in a Jupyter environment to view the detailed discussion, code examples, and visualizations.

    NOTE 1: running analyse.py will overwrite existing files in the 'files' folder.
    NOTE 2: everything printed in the terminal is for debugging. 
    NOTE 3: files_old contains a copy of the first version of the program when run, before tuning. 

----##----##----

MODULE DIVISION:

    1- read.py
        loads the 'data_3dprinter'.csv file;
        checks for missing data and treats it;
        split into training and testing sets (80%/20% split respectively)
            note: during trials 5 data sets were identified that gave a sufficiently accurate result
            (random states 454 670 343 595 and 930). these are used to show how the 80/20 random split
            can affect how the prediction models behave. 
            the models were optimised for random set rs=343.

    2- models folder
        inside the folder there are 8 .py modules that define 9 prediction models as functions:
            decisiontree.py         -> Desision Tree Model
            elasticnet.py           -> Elastic Net Model
            KNN.py                  -> K-Nearest Neighbours Model
            KNN_selfwritten.py      -> K-Nearest Neighbours Model (custom implementation w/o sklearn)
            lassoreg.py             -> Lasso Regressor Model
            linreg.py               -> Linear Regression Model
            linreg_selfwritten.py   -> Linear Regression Model (custom implementation w/o sklearn)
            polyreg.py              -> Polynomial Regression (degree 3)
            randomforest.py         -> Random Forest Regressor Model
            ridgereg.py             -> Ridge Regressor Model
            SVR.py (*2)             -> Support Vector Regression Model (rbf and linear kernels)
        each one trains the model with the 80% data and tests it with the 20%.
        all the metrics for both test and train sets are saved in a dataframe.

    3- modules.py
        initialises dataframe results_df for storing the metrics from each model
        runs the models and saves everything to the dataframe.
        also prints what random state was used (as the sets themselves)
    
    4- utils.py
        used for extra functions needed through the project
            %E avg calc measures the %error for each of the datapoints used (test v train)
            feature importance calculates the coeff. of importance for each feature for the model.        

    5- analyse.py
        prints a coorelation heatmat for the features;
        analyses feature importance (with the function set up in utils.py);
        initialises dataframe analyse_df for comparing the test v train of each
            metric in results_df;
        exports both dataframes and heatmap as .csv and .png files (respectively)
        plots data visualisation graphs

    6- hyperparameters.py
        used to test the accuracy of the models and compute the optimal hyperparameters
            for each one. the functions in this module test the models with every possible value
            for the hyperparameters for each model and outputs the optimal one.
            it was only necessary to run them each once, then update the models inside models folder.
        decision tree showed the biggest signs of overfitting (from the metrics in the dataframes)
            so that one was optimised first;
        then random forest with the same method;
        then lasso, ridge and elastic net regressors.
        then KNN, and both SVR's.

    7- discussion.ipynb
        created as a report to be able to run snippets of the code for analytic purposes
            1. Data Handling & Random States: Data loading, preprocessing, and impact of train-test split variability.
            2. Project Structure: Explanation of the modular code organization.
            3. Models & Metrics: Overview of implemented models and justification for evaluation metrics.
            4. Hyperparameter Tuning: Process of optimizing model performance.
            5. Performance Analysis: Detailed look at model results, comparison of models (including custom vs. sklearn), and key findings from graphs.
            6. Limitations & Future Work: Discussion of study limitations and potential next steps.

----##----##----

GRAPHS:

    Plot 1:     Model Performance Comparison by R-squared (Test Set) - Bar chart of overall R2 for all models.
    Plot 2:     Model Performance Comparison by Mean Squared Error (MSE) (Test Set) - Bar chart of overall MSE for all models.
    Plot 3:     Model Performance Comparison by Root Mean Squared Error (RMSE) (Test Set) - Bar chart of overall RMSE for all models.
    Plot 4:     Model Performance Comparison by Mean Absolute Error (MAE) (Test Set) - Bar chart of overall MAE for all models.
    Plot 5:     Comparison of R-squared Scores for Each Target Variable (Test Set) - Grouped bar chart comparing R2 for Roughness, Tensile Strength, and Elongation across all models.
    Plot 6:     Test vs. Train Performance Comparison (R-squared) - Grouped bar chart comparing overall Test R2 and Train R2 for all models.
    Plot 7:     Feature Importance Analysis for Random Forest Regressor - Bar chart showing feature importance for the Random Forest model.
    Plot 8:     Residual Plots - Random Forest Regressor - Scatter plots showing predicted vs. residuals for each target variable from the Random Forest model (in subplots).
    Plot 9:     Predicted vs. Actual Values - Linear Regression (Sklearn) - Scatter plots showing predicted vs. actual for each target variable from the Sklearn Linear Regression model, with the ideal fit line (in subplots).
    Plot 10:    Predicted vs. Actual Values - Linear Regression (Custom) - Scatter plots showing predicted vs. actual for each target variable from the Custom Linear Regression model, with the ideal fit line (in subplots).
    Plot 11:    Prediction Curve with Scatter - Support Vector Regressor (RBF) - Line plot showing the prediction curve of SVR (RBF) for 'roughness' vs. 'layer_height', along with scatter points of actual and predicted test data.
    Plot 12:    Predicted vs. Actual Values - Support Vector Regressor (RBF) - Scatter plots showing predicted vs. actual for each target variable from the SVR (RBF) model (in subplots).
    Plot 13:    Predicted vs. Actual Values - KNN Regressor - Scatter plots showing predicted vs. actual for each target variable from the KNN Regressor model (in subplots).
    Plot 14:    Coefficient Plot - Ridge Regression - Grouped bar chart showing coefficients for each feature and target variable from the Ridge Regression model.
    Plot 15:    Coefficient Plot - Lasso Regression - Grouped bar chart showing coefficients for each feature and target variable from the Lasso Regression model.
    Plot 16:    Coefficient Plot - Elastic Net - Grouped bar chart showing coefficients for each feature and target variable from the Elastic Net model.
    Plot 17:    Predicted vs. Actual Values - Random Forest Regressor - Scatter plots showing predicted vs. actual for each target variable from the Random Forest Regressor model (in subplots).
    Plot 18:    Comparison of R-squared Before and After Tuning - Grouped bar chart comparing R2 (Test) and R2 (Train) before and after tuning for the selected models that were tuned.
    Plot 19:    Comparison of Linear Regression and KNN Implementations (Sklearn vs Custom - R2 Only) - Combined grouped bar chart comparing R2 (Test) and R2 (Train) for Sklearn vs Custom implementations of both Linear Regression and KNN, using a hierarchical x-axis.
    Plot 20:    Comparison of All Four Implementations (R2 Only) - Grouped bar chart using facets to compare R2 (Test) and R2 (Train) across all four implementations: Sklearn Linear Regression, Custom Linear Regression, Sklearn KNN Regressor, and Custom KNN Regressor.