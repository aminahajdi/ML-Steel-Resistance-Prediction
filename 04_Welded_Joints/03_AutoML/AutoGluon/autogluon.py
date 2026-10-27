# =============================================================================
# STUDY 4 — AUTOGLUON BENCHMARK FOR WELDED-JOINT MOMENT RESISTANCE
# Independent 80/20 test partition and AutoGluon bagging/stacking
# =============================================================================

from pathlib import Path
import random
import pandas as pd
import numpy as np

from autogluon.tabular import TabularPredictor

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# =============================================================================
# 0. EXPERIMENT CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

REPO_ROOT = BASE_DIR.parents[2]  # Three folders above AutoGluon/
DATA_FILE = REPO_ROOT / "data" / "study_04_data.xlsx"

RESULTS_DIR = BASE_DIR / "outputs" / "seed_42"

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
TIME_LIMIT = 3600  # seconds

# AutoGluon settings
# Five-fold bagging with one stacking level; retained from the benchmark experiment.
NUM_BAG_FOLDS = 5
NUM_STACK_LEVELS = 1
PRESETS = "best_quality"

# Evaluation metric used internally by AutoGluon.
# Engineering metrics such as bias and CoV are calculated manually below.
AUTOGLUON_EVAL_METRIC = "root_mean_squared_error"


# =============================================================================
# 1. REPRODUCIBILITY
# =============================================================================

def main():
    """Train, evaluate, and export the AutoGluon benchmark."""
    # =============================================================================
    # 2. LOAD DATA
    # =============================================================================

    random.seed(SEED)
    np.random.seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_excel(DATA_FILE)

    missing_input_columns = [col for col in INPUT_COLUMNS if col not in df.columns]
    if missing_input_columns:
        raise ValueError(f"Missing input columns in Excel file: {missing_input_columns}")

    if OUTPUT_COLUMN not in df.columns:
        raise ValueError(f"Output column not found in Excel file: {OUTPUT_COLUMN}")

    df = df[INPUT_COLUMNS + [OUTPUT_COLUMN]].copy()

    # Basic cleaning
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[df[OUTPUT_COLUMN] > 0].copy()

    print("====================================================")
    print("AUTOGLUON BENCHMARK - NUMERICAL JOINT DATABASE")
    print("====================================================")
    print(f"Data file             = {DATA_FILE}")
    print(f"Total clean samples   = {len(df)}")
    print(f"Input columns         = {INPUT_COLUMNS}")
    print(f"Output column         = {OUTPUT_COLUMN}")
    print(f"Seed                  = {SEED}")
    print(f"Test size             = {TEST_SIZE}")
    print(f"Time limit            = {TIME_LIMIT} seconds")
    print(f"AutoGluon preset      = {PRESETS}")
    print(f"AutoGluon metric      = {AUTOGLUON_EVAL_METRIC}")
    print(f"Bag folds             = {NUM_BAG_FOLDS}")
    print(f"Stack levels          = {NUM_STACK_LEVELS}")


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
    # 4. TRAIN AUTOGLUON
    # =============================================================================

    predictor = TabularPredictor(
        label=OUTPUT_COLUMN,
        problem_type="regression",
        eval_metric=AUTOGLUON_EVAL_METRIC,
        path=str(RESULTS_DIR / "AutogluonModels"),
    )

    predictor.fit(
        train_data=train_df,
        presets=PRESETS,
        time_limit=TIME_LIMIT,
        num_bag_folds=NUM_BAG_FOLDS,
        num_stack_levels=NUM_STACK_LEVELS,
        ag_args_fit={"num_gpus": 0},
    )


    # =============================================================================
    # 5. LEADERBOARD
    # =============================================================================

    leaderboard = predictor.leaderboard(test_df, silent=True)

    leaderboard_path = RESULTS_DIR / "autogluon_leaderboard_seed42.csv"
    leaderboard.to_csv(leaderboard_path, index=False)

    print("\n====================================================")
    print("AUTOGLUON LEADERBOARD")
    print("====================================================")
    print(leaderboard)


    # =============================================================================
    # 6. METRICS FOR BEST MODEL
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


    best_model = predictor.model_best

    y_train_true = train_df[OUTPUT_COLUMN].to_numpy()
    y_test_true = test_df[OUTPUT_COLUMN].to_numpy()

    y_train_pred = predictor.predict(train_df, model=best_model).to_numpy()
    y_test_pred = predictor.predict(test_df, model=best_model).to_numpy()

    train_bias, train_cov, train_r2, train_rmse, train_mae = compute_metrics(
        y_train_true,
        y_train_pred,
    )

    test_bias, test_cov, test_r2, test_rmse, test_mae = compute_metrics(
        y_test_true,
        y_test_pred,
    )

    metrics_df = pd.DataFrame([
        {
            "model": best_model,
            "dataset": "train",
            "bias": train_bias,
            "cov": train_cov,
            "r2": train_r2,
            "rmse": train_rmse,
            "mae": train_mae,
        },
        {
            "model": best_model,
            "dataset": "test",
            "bias": test_bias,
            "cov": test_cov,
            "r2": test_r2,
            "rmse": test_rmse,
            "mae": test_mae,
        },
    ])

    metrics_path = RESULTS_DIR / "best_model_metrics_seed42.csv"
    metrics_df.to_csv(metrics_path, index=False)

    print("\n====================================================")
    print("BEST MODEL METRICS")
    print("====================================================")
    print(f"Best AutoGluon model = {best_model}")
    print(metrics_df)


    # =============================================================================
    # 7. PREDICTIONS TABLE
    # =============================================================================

    train_predictions_df = train_df.copy()
    train_predictions_df["prediction"] = y_train_pred
    train_predictions_df["prediction_to_reference_ratio"] = y_train_pred / y_train_true
    train_predictions_df["dataset"] = "train"

    test_predictions_df = test_df.copy()
    test_predictions_df["prediction"] = y_test_pred
    test_predictions_df["prediction_to_reference_ratio"] = y_test_pred / y_test_true
    test_predictions_df["dataset"] = "test"

    predictions_df = pd.concat(
        [train_predictions_df, test_predictions_df],
        axis=0,
        ignore_index=True,
    )

    predictions_path = RESULTS_DIR / "predictions_seed42.csv"
    predictions_df.to_csv(predictions_path, index=False)


    # =============================================================================
    # 8. FEATURE IMPORTANCE
    # =============================================================================

    # Permutation feature importance is evaluated on the held-out test partition.
    # The test partition is not used to fit the AutoGluon models.
    feature_importance = predictor.feature_importance(test_df)

    feature_importance_path = RESULTS_DIR / "feature_importance_seed42.csv"
    feature_importance.to_csv(feature_importance_path)

    print("\n====================================================")
    print("FEATURE IMPORTANCE")
    print("====================================================")
    print(feature_importance)


    # =============================================================================
    # 9. SAVE SUMMARY TXT
    # =============================================================================

    summary_path = RESULTS_DIR / "autogluon_summary_seed42.txt"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("AUTOGLUON BENCHMARK - NUMERICAL JOINT DATABASE\n")
        f.write("====================================================\n\n")

        f.write("DATA PROTOCOL\n")
        f.write("====================================================\n")
        f.write(f"Data file: {DATA_FILE.name}\n")
        f.write(f"Total clean samples: {len(df)}\n")
        f.write(f"Train/test split: {int((1 - TEST_SIZE) * 100)}:{int(TEST_SIZE * 100)}\n")
        f.write(f"Seed: {SEED}\n")
        f.write(f"Input columns: {INPUT_COLUMNS}\n")
        f.write(f"Output column: {OUTPUT_COLUMN}\n\n")

        f.write("AUTOGLUON SETTINGS\n")
        f.write("====================================================\n")
        f.write(f"Preset: {PRESETS}\n")
        f.write(f"Internal AutoGluon metric: {AUTOGLUON_EVAL_METRIC}\n")
        f.write(f"Time limit: {TIME_LIMIT} seconds\n")
        f.write(f"Bag folds: {NUM_BAG_FOLDS}\n")
        f.write(f"Stack levels: {NUM_STACK_LEVELS}\n\n")

        f.write("BEST MODEL\n")
        f.write("====================================================\n")
        f.write(f"Best model: {best_model}\n\n")

        f.write("METRICS\n")
        f.write("====================================================\n")
        f.write(metrics_df.to_string(index=False))
        f.write("\n\n")

        f.write("NOTES\n")
        f.write("====================================================\n")
        f.write(
            "AutoGluon was used as an automated machine learning benchmark. "
            "The internal optimization metric was RMSE, while the final engineering "
            "evaluation was performed using bias, CoV, R2, RMSE, and MAE.\n"
        )


    # =============================================================================
    # 10. DONE
    # =============================================================================

    print("\n====================================================")
    print("DONE")
    print("====================================================")
    print(f"Leaderboard saved to:        {leaderboard_path}")
    print(f"Metrics saved to:            {metrics_path}")
    print(f"Predictions saved to:        {predictions_path}")
    print(f"Feature importance saved to: {feature_importance_path}")
    print(f"Summary saved to:            {summary_path}")
    print(f"AutoGluon model folder:      {RESULTS_DIR / 'AutogluonModels'}")


if __name__ == "__main__":
    main()
