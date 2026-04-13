PySINDy Buffet Modelling Project - V2

Luan Correia Gil

----##----##----

INTRODUCTION:
    This project is the second iteration of the buffet-response modelling workflow.
    It uses CFD simulation outputs from Ansys Fluent to build and assess a PySINDy
    model for transonic buffet behaviour. The workflow is split into clear stages:
    reading and validating the data, defining and fitting the model, running
    confirmation-case predictions, and analysing both the available data and the
    model outputs.

----##----##----

HOW THE DATA IS TREATED:
    1. Read each Fluent file independently.
    2. Skip the 3 header lines at the top of each file.
    3. Confirm that 8000 numeric rows are present.
    4. Discard the first 500 numeric rows.
    5. Keep the remaining 7500 rows.
    6. Compute pitch(t) from atan2(Vy, Vx) in degrees.
    7. Keep Cl and Cd as the state variables of interest.
    8. Keep each file labelled as its own trajectory.
    9. Use B*, C*, D*, E*, and F* files as training cases.
    10. Use T* files as confirmation cases.

    The files are treated as multiple labelled trajectories.
    They are not combined as anonymous rows.

----##----##----

HOW TO USE:
    1. Ensure required libraries are installed:
        pandas
        numpy
        plotly
        pysindy

    2. Place all Fluent simulation files in the data folder.
        Training files should begin with letters such as B, C, D, E, or F.
        Confirmation files should begin with T, such as T1_OUTPUT.out.

    3. Run:
        python main.py

    4. Answer the four [y/n] prompts depending on which stages you want to run:
        read
        fit
        run
        analyse

    5. To run hyperparameter tuning separately, use:
        python assets/utils/hyperparameters.py

    NOTE 1: the main workflow is designed so that main.py is the single program entry point.
    NOTE 2: hyperparameters.py is separate from main.py and is only used during tuning.
    NOTE 3: latest output files overwrite themselves, while fit and run archives keep numbered history copies.

----##----##----

MODULE DIVISION:
    1- main.py
        asks the workflow [y/n] prompts;
        runs the read, fit, run, and analyse stages.

    2- assets/read.py
        reads all .out files from the data folder;
        validates the file structure;
        trims the first 500 numeric rows;
        computes pitch(t);
        exports prepared train and confirmation tables.

    3- assets/model.py
        defines the canonical PySINDy model only;
        sets the live default model parameters;
        does not fit, run, or analyse the model.

    4- assets/fit.py
        reuses prepared outputs when valid;
        rebuilds the per-case training trajectories;
        fits the model on training cases only;
        exports the latest fitted model and fit summaries.

    5- assets/run.py
        loads the latest fitted model;
        runs prediction on confirmation cases only;
        exports prediction tables and per-case error summaries.

    6- assets/analyse.py
        opens interactive Plotly figures;
        plots the available cl, cd, and pitch data;
        compares true and predicted confirmation results;
        provides the first analysis stage of the workflow.

    7- assets/utils/hyperparameters.py
        performs leave-one-training-case-out hyperparameter search;
        tests threshold, alpha, and polynomial degree;
        recommends the best values and can apply them to model.py.

    8- method.md
        records the methodology and implementation history of the project.

----##----##----

OUTPUT FILES:
    prepared_train_cases.csv
    prepared_confirmation_cases.csv
    file_validation.csv
    case_summary.csv
    latest_fitted_model.pkl
    fit_summary_latest.json
    equations_history.txt
    latest_confirmation_predictions.csv
    latest_confirmation_run_summary.csv
    model_archive/
    run_archive/

    Additional tuning outputs created only when hyperparameters.py is run:
        hyperparameter_results_latest.csv
        hyperparameter_best_summary_latest.json
        hyperparameter_equations_history.txt

----##----##----

ANALYSIS GRAPHS:
    1. All-case overlays for cl, cd, and pitch.
    2. Grouped-by-category overlays for cl, cd, and pitch.
    3. Zoomed plots over the first 0.25 seconds of each retained case.
    4. True-vs-predicted overlays for cl and cd.
    5. Residual-vs-time plots for cl and cd.
    6. Parity plots for cl and cd.
    7. Per-case RMSE and MAE bar charts.

    A tuned-vs-untuned comparison is planned for a later version of the analysis stage.

----##----##----

HYPERPARAMETER WORKFLOW:
    1. Fit and run the baseline model first.
    2. Run python assets/utils/hyperparameters.py.
    3. Inspect the tuning results and best-summary outputs.
    4. Apply the best values to model.py if appropriate.
    5. Refit the model using the updated defaults.
    6. Run the confirmation stage again and compare the results.
