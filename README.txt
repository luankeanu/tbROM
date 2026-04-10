PySINDy Buffet Modelling Project

Luan Correia Gil

----##----##----

INTRODUCTION:
    This project models the buffet response of one airfoil from CFD simulation outputs.
    Each simulation file is treated as one time-series trajectory rather than one static sample.
    The program validates the files, trims all cases to the same useful region, computes pitch
    from the velocity components, and trains a PySINDy model to predict Cl and Cd in time.

----##----##----

HOW THE DATA IS TREATED:
    1. Read each file independently.
    2. Skip the 3 text header lines.
    3. Confirm that 8000 numeric rows are present.
    4. Discard the first 500 numeric rows.
    5. Keep the remaining 7500 rows.
    6. Compute pitch(t) from atan2(Vy, Vx).
    7. Compute pitch_rate(t) and pitch_accel(t).
    8. Combine all files into one database, but keep the source file label.
    9. Train one shared model across many trajectories.
    10. Validate the model by holding out complete files.

    The files should be combined as multiple labelled trajectories.
    They should not be combined as anonymous rows.

----##----##----

HOW TO USE:
    1. Ensure required libraries are installed:
        pandas
        numpy
        plotly
        pysindy

    2. Place all simulation files in the Data folder.
        Development files can be named B*, C*, D*, E*, etc.
        Final test files should begin with T, such as T1_OUTPUT.out.

    3. Edit config.py to change:
        plot windows
        hyperparameter ranges
        validation mode
        optional feature switches

    4. Run:
        python main.py

    NOTE 1: output files overwrite themselves on each run.
    NOTE 2: if no T* file is present, the program still runs using only development files.

----##----##----

MODULE DIVISION:
    1- config.py
        stores paths, trimming rules, validation settings, and hyperparameter ranges.

    2- pipeline.py
        reads and validates all .out files;
        trims the data;
        computes pitch features;
        exports the combined database and summary tables.

    3- model.py
        builds the PySINDy model;
        runs brute-force hyperparameter search;
        trains the final model;
        exports prediction tables and saves the model.

    4- analyse.py
        exports metric tables;
        plots the feature correlation heatmap;
        plots dataset overlays;
        plots actual vs predicted curves and residuals.

    5- main.py
        runs the whole workflow.

----##----##----

OUTPUT FILES:
    combined_database.csv
    file_validation.csv
    case_summary.csv
    feature_correlation.csv
    feature_correlation_heatmap.html
    hyperparameter_results.csv
    train_metrics.csv
    validation_metrics.csv
    validation_predictions.csv
    test_predictions.csv
    best_model_summary.csv
    best_model.pkl
    run_config.json

----##----##----

HYPERPARAMETER WORKFLOW:
    1. Start with a coarse range in config.py.
    2. Run python main.py.
    3. Inspect hyperparameter_results.csv.
    4. Narrow the best region.
    5. Run again with smaller steps.
