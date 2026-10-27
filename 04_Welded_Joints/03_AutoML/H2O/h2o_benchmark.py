# =============================================================================
# H2O AUTOML BENCHMARK FOR NUMERICAL JOINT DATABASE
# Dataset: data/study_04_data.xlsx
# Target: moment resistance
# =============================================================================

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import h2o
from h2o.automl import H2OAutoML


# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

REPO_ROOT = BASE_DIR.parents[2]
DATA_FILE = REPO_ROOT / "data" / "study_04_data.xlsx"

RESULTS_DIR = BASE_DIR / "outputs" / "seed_42"
RESULTS_DIR.mkdir(exist_ok=True)

OUTPUT_COLUMN = "moment resistance"

INPUT_COLUMNS = [
    "hc",
    "bc",
    "twc",
    "tfc",
    "rc",
    "hb1",
    "bb1",
    "twb1",
    "tfb1",
    "rb1",
    "axial force percentage",
    "stiffening",
    "joint location",
    "moment ratio",
    "fy",
]

SEED = 42
TEST_SIZE = 0.20
TIME_LIMIT = 3600  # seconds = 1 hour

N_FOLDS = 5
MAX_MEM_SIZE = "4G"

LEADERBOARD_PATH = RESULTS_DIR / "h2o_leaderboard_seed42.csv"
METRICS_PATH = RESULTS_DIR / "h2o_best_model_metrics_seed42.csv"
PREDICTIONS_PATH = RESULTS_DIR / "h2o_predictions_seed42.csv"
PLOT_PATH = RESULTS_DIR / "h2o_true_vs_predicted_seed42.png"

MODEL_DIR = RESULTS_DIR / "H2OModels"
MODEL_DIR.mkdir(exist_ok=True)


def main():
    # =============================================================================
    # 1. START H2O
    # =============================================================================

    h2o.init(
        nthreads=-1,
        max_mem_size=MAX_MEM_SIZE,
    )


    # =============================================================================
    # 2. LOAD AND CLEAN DATA
    # =============================================================================

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_excel(DATA_FILE)

    missing_input_columns = [col for col in INPUT_COLUMNS if col not in df.columns]
    if missing_input_columns:
        raise ValueError(f"Missing input columns in Excel file: {missing_input_columns}")

    if OUTPUT_COLUMN not in df.columns:
        raise ValueError(f"Output column not found in Excel file: {OUTPUT_COLUMN}")

    df = df[INPUT_COLUMNS + [OUTPUT_COLUMN]].copy()

    # Exclude incomplete/non-finite records and nonpositive target values.
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[df[OUTPUT_COLUMN] > 0].copy()

    print("====================================================")
    print("H2O AUTOML BENCHMARK - NUMERICAL JOINT DATABASE")
    print("====================================================")
    print(f"Data file             = {DATA_FILE}")
    print(f"Total clean samples   = {len(df)}")
    print(f"Input columns         = {INPUT_COLUMNS}")
    print(f"Output column         = {OUTPUT_COLUMN}")
    print(f"Seed                  = {SEED}")
    print(f"Test size             = {TEST_SIZE}")
    print(f"Time limit            = {TIME_LIMIT} seconds")
    print(f"H2O CV folds          = {N_FOLDS}")
    print(f"H2O max memory        = {MAX_MEM_SIZE}")


    # =============================================================================
    # 3. TRAIN / TEST SPLIT
    # =============================================================================

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=SEED,
        shuffle=True,
    )

    print("\n====================================================")
    print("DATA SPLIT")
    print("====================================================")
    print(f"Train samples = {len(train_df)}")
    print(f"Test samples  = {len(test_df)}")


    # =============================================================================
    # 4. CONVERT TO H2O FRAMES
    # =============================================================================

    train_h2o = h2o.H2OFrame(train_df)
    test_h2o = h2o.H2OFrame(test_df)

    # Make sure target is numeric for regression
    train_h2o[OUTPUT_COLUMN] = train_h2o[OUTPUT_COLUMN].asnumeric()
    test_h2o[OUTPUT_COLUMN] = test_h2o[OUTPUT_COLUMN].asnumeric()


    # =============================================================================
    # 5. TRAIN H2O AUTOML
    # =============================================================================

    aml = H2OAutoML(
        max_runtime_secs=TIME_LIMIT,
        seed=SEED,
        nfolds=N_FOLDS,
        sort_metric="RMSE",
        stopping_metric="RMSE",
        project_name="H2O_Numerical_Joint_Benchmark_Seed42",
    )

    aml.train(
        x=INPUT_COLUMNS,
        y=OUTPUT_COLUMN,
        training_frame=train_h2o,
        leaderboard_frame=test_h2o,  # Original protocol: test frame contributes to leaderboard ranking.
    )

    leader = aml.leader

    print("\n====================================================")
    print("H2O BEST MODEL")
    print("====================================================")
    print(leader.model_id)


    # =============================================================================
    # 6. LEADERBOARD
    # =============================================================================

    leaderboard = aml.leaderboard.as_data_frame()
    leaderboard.to_csv(LEADERBOARD_PATH, index=False)

    print("\n====================================================")
    print("H2O LEADERBOARD")
    print("====================================================")
    print(leaderboard)


    # =============================================================================
    # 7. SAVE BEST MODEL
    # =============================================================================

    saved_model_path = h2o.save_model(
        model=leader,
        path=str(MODEL_DIR),
        force=True,
    )

    (RESULTS_DIR / "selected_model_path.txt").write_text(str(saved_model_path), encoding="utf-8")

    print(f"\nBest H2O model saved to: {saved_model_path}")


    # =============================================================================
    # 8. PREDICTIONS
    # =============================================================================

    train_pred_h2o = leader.predict(train_h2o)
    test_pred_h2o = leader.predict(test_h2o)

    y_pred_train = train_pred_h2o.as_data_frame()["predict"].to_numpy(dtype=float)
    y_pred_test = test_pred_h2o.as_data_frame()["predict"].to_numpy(dtype=float)

    y_train_true = train_df[OUTPUT_COLUMN].to_numpy(dtype=float)
    y_test_true = test_df[OUTPUT_COLUMN].to_numpy(dtype=float)


    # =============================================================================
    # 9. METRICS
    # =============================================================================

    def compute_metrics(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        mask = y_true != 0
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        ratios = y_pred / y_true

        bias = float(np.mean(ratios))
        cov = float(np.std(ratios) / abs(bias)) if abs(bias) > 1e-12 else float("inf")
        r2 = float(r2_score(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))

        return bias, cov, r2, rmse, mae


    train_bias, train_cov, train_r2, train_rmse, train_mae = compute_metrics(
        y_train_true,
        y_pred_train,
    )

    test_bias, test_cov, test_r2, test_rmse, test_mae = compute_metrics(
        y_test_true,
        y_pred_test,
    )

    metrics_df = pd.DataFrame([
        {
            "model": leader.model_id,
            "dataset": "train",
            "n": len(y_train_true),
            "bias": train_bias,
            "cov": train_cov,
            "r2": train_r2,
            "rmse": train_rmse,
            "mae": train_mae,
        },
        {
            "model": leader.model_id,
            "dataset": "test",
            "n": len(y_test_true),
            "bias": test_bias,
            "cov": test_cov,
            "r2": test_r2,
            "rmse": test_rmse,
            "mae": test_mae,
        },
    ])

    metrics_df.to_csv(METRICS_PATH, index=False)

    print("\n====================================================")
    print("H2O BEST MODEL METRICS")
    print("====================================================")
    print(metrics_df)


    # =============================================================================
    # 10. SAVE PREDICTIONS
    # =============================================================================

    train_predictions_df = train_df.copy()
    train_predictions_df["dataset"] = "train"
    train_predictions_df["true"] = y_train_true
    train_predictions_df["predicted"] = y_pred_train
    train_predictions_df["prediction_to_reference_ratio"] = y_pred_train / y_train_true

    test_predictions_df = test_df.copy()
    test_predictions_df["dataset"] = "test"
    test_predictions_df["true"] = y_test_true
    test_predictions_df["predicted"] = y_pred_test
    test_predictions_df["prediction_to_reference_ratio"] = y_pred_test / y_test_true

    predictions_df = pd.concat(
        [train_predictions_df, test_predictions_df],
        axis=0,
        ignore_index=True,
    )

    predictions_df.to_csv(PREDICTIONS_PATH, index=False)


    # =============================================================================
    # 11. TRUE VS PREDICTED PLOT
    # =============================================================================

    plt.figure(figsize=(7, 7))

    combined_values = np.concatenate(
        (
            y_pred_train,
            y_train_true,
            y_pred_test,
            y_test_true,
        )
    )

    min_val = float(np.min(combined_values))
    max_val = float(np.max(combined_values))
    line_x = np.linspace(min_val, max_val, 200)

    plt.scatter(
        y_pred_train,
        y_train_true,
        alpha=0.65,
        edgecolors="black",
        s=35,
        color="tab:blue",
        label=(
            f"Train data\n"
            f"n={len(y_train_true)}\n"
            f"b={train_bias:.4f}\n"
            f"CoV={train_cov:.4f}\n"
            f"R²={train_r2:.4f}"
        ),
    )

    plt.scatter(
        y_pred_test,
        y_test_true,
        alpha=0.85,
        edgecolors="black",
        s=50,
        color="tab:purple",
        label=(
            f"Test data\n"
            f"n={len(y_test_true)}\n"
            f"b={test_bias:.4f}\n"
            f"CoV={test_cov:.4f}\n"
            f"R²={test_r2:.4f}"
        ),
    )

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        "k--",
        linewidth=2,
        label="Ideal fit",
    )

    plt.plot(
        line_x,
        1.1 * line_x,
        "r--",
        linewidth=1.5,
        label="±10% error",
    )

    plt.plot(
        line_x,
        0.9 * line_x,
        "r--",
        linewidth=1.5,
    )

    plt.xlim(0, max_val * 1.05)
    plt.ylim(0, max_val * 1.05)

    plt.xlabel("Predicted moment resistance")
    plt.ylabel("True moment resistance")
    plt.title(f"H2O AutoML Best Model\n{leader.model_id}")

    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()

    plt.savefig(PLOT_PATH, dpi=300)
    plt.close()


    # =============================================================================
    # 12. VARIABLE IMPORTANCE, IF AVAILABLE
    # =============================================================================

    try:
        varimp = leader.varimp(use_pandas=True)

        if varimp is not None and len(varimp) > 0:
            varimp_path = RESULTS_DIR / "h2o_variable_importance_seed42.csv"
            varimp_plot_path = RESULTS_DIR / "h2o_variable_importance_seed42.png"

            varimp.to_csv(varimp_path, index=False)

            varimp_plot = varimp.sort_values("relative_importance", ascending=True)

            plt.figure(figsize=(8, 5))
            plt.barh(
                varimp_plot["variable"],
                varimp_plot["relative_importance"],
                edgecolor="black",
            )

            plt.xlabel("Relative Importance")
            plt.ylabel("Feature")
            plt.title(f"H2O Variable Importance\n{leader.model_id}")
            plt.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.6)
            plt.tight_layout()
            plt.savefig(varimp_plot_path, dpi=300)
            plt.close()

            print(f"Variable importance saved to: {varimp_path}")
            print(f"Variable importance plot saved to: {varimp_plot_path}")

        else:
            print("Variable importance not available for this leader model.")

    except Exception as e:
        print("Variable importance could not be extracted.")
        print(f"Reason: {e}")


    # =============================================================================
    # 13. OPTIONAL H2O EXPLAINABILITY REPORT
    # =============================================================================

    try:
        print("\nGenerating H2O explainability output...")
        explanation = h2o.explain(
            aml,
            test_h2o,
            include_explanations=[
                "leaderboard",
                "varimp",
                "shap_summary",
                "pdp",
            ],
        )
        print("H2O explainability output generated.")

    except Exception as e:
        print("H2O explainability report could not be generated.")
        print(f"Reason: {e}")


    # =============================================================================
    # 14. SAVE SUMMARY TXT
    # =============================================================================

    SUMMARY_PATH = RESULTS_DIR / "h2o_summary_seed42.txt"

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("H2O AUTOML BENCHMARK - NUMERICAL JOINT DATABASE\n")
        f.write("====================================================\n\n")

        f.write("DATA PROTOCOL\n")
        f.write("====================================================\n")
        f.write(f"Data file: {DATA_FILE.name}\n")
        f.write(f"Total clean samples: {len(df)}\n")
        f.write(f"Train/test split: {int((1 - TEST_SIZE) * 100)}:{int(TEST_SIZE * 100)}\n")
        f.write(f"Seed: {SEED}\n")
        f.write(f"Input columns: {INPUT_COLUMNS}\n")
        f.write(f"Output column: {OUTPUT_COLUMN}\n\n")

        f.write("H2O SETTINGS\n")
        f.write("====================================================\n")
        f.write(f"Time limit: {TIME_LIMIT} seconds\n")
        f.write(f"CV folds: {N_FOLDS}\n")
        f.write("Sort metric: RMSE\n")
        f.write("Stopping metric: RMSE\n\n")

        f.write("BEST MODEL\n")
        f.write("====================================================\n")
        f.write(f"Best model: {leader.model_id}\n")
        f.write(f"Saved model path: {saved_model_path}\n\n")

        f.write("METRICS\n")
        f.write("====================================================\n")
        f.write(metrics_df.to_string(index=False))
        f.write("\n\n")

        f.write("NOTES\n")
        f.write("====================================================\n")
        f.write(
            "H2O AutoML was used as an automated machine learning benchmark. "
            "The internal ranking metric was RMSE, while the final engineering "
            "evaluation was performed using bias, CoV, R2, RMSE, and MAE.\n"
        )


    # =============================================================================
    # 15. DONE
    # =============================================================================

    print("\n====================================================")
    print("DONE")
    print("====================================================")
    print(f"Leaderboard saved to:          {LEADERBOARD_PATH}")
    print(f"Metrics saved to:              {METRICS_PATH}")
    print(f"Predictions saved to:          {PREDICTIONS_PATH}")
    print(f"Prediction plot saved to:      {PLOT_PATH}")
    print(f"Summary saved to:              {SUMMARY_PATH}")
    print(f"Best H2O model saved to:       {saved_model_path}")

    h2o.shutdown(prompt=False)


if __name__ == "__main__":
    main()
