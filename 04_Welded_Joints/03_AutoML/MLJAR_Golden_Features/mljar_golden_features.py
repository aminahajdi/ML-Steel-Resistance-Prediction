# =============================================================================
# STUDY 4 — WELDED JOINTS: MLJAR AUTOML WITH GOLDEN FEATURES
# Fixed training/test protocol for numerical moment-resistance prediction
# =============================================================================

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from supervised.automl import AutoML

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# =============================================================================
# 0. EXPERIMENT CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DATA_FILE = REPO_ROOT / "data" / "study_04_data.xlsx"

RESULTS_DIR = SCRIPT_DIR / "outputs" / "seed_42"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MLJAR_MODELS_DIR = RESULTS_DIR / "MLJARModels"

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

METRICS_PATH = RESULTS_DIR / "mljar_golden_features_metrics_seed42.csv"
PREDICTIONS_PATH = RESULTS_DIR / "mljar_golden_features_predictions_seed42.csv"
PLOT_PATH = RESULTS_DIR / "mljar_golden_features_true_vs_predicted_seed42.png"
LEADERBOARD_PATH = RESULTS_DIR / "mljar_golden_features_leaderboard_seed42.csv"
GOLDEN_FEATURES_COPY_PATH = RESULTS_DIR / "golden_features_extracted_seed42.json"
SUMMARY_PATH = RESULTS_DIR / "mljar_golden_features_summary_seed42.txt"


def main():
    # =============================================================================
    # 1. LOAD DATA
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
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[df[OUTPUT_COLUMN] > 0].copy()

    print("====================================================")
    print("MLJAR AUTOML WITH GOLDEN FEATURES — WELDED JOINTS")
    print("====================================================")
    print(f"Dataset file        = {DATA_FILE}")
    print(f"Total clean samples = {len(df)}")
    print(f"Input columns       = {INPUT_COLUMNS}")
    print(f"Output column       = {OUTPUT_COLUMN}")
    print(f"Seed                = {SEED}")
    print(f"Golden features     = True")
    print(f"Time limit          = {TIME_LIMIT} seconds")


    # =============================================================================
    # 2. TRAIN / TEST SPLIT
    # =============================================================================

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=SEED,
        shuffle=True,
    )

    X_train = train_df[INPUT_COLUMNS].copy()
    y_train = train_df[OUTPUT_COLUMN].copy()

    X_test = test_df[INPUT_COLUMNS].copy()
    y_test = test_df[OUTPUT_COLUMN].copy()

    print(f"Train samples       = {len(train_df)}")
    print(f"Test samples        = {len(test_df)}")


    # =============================================================================
    # 3. TRAIN MLJAR WITH GOLDEN FEATURES
    # =============================================================================

    automl = AutoML(
        mode="Compete",
        ml_task="regression",
        eval_metric="rmse",
        total_time_limit=TIME_LIMIT,
        results_path=str(MLJAR_MODELS_DIR),
        random_state=SEED,
        n_jobs=-1,

        algorithms=[
            "Linear",
            "Random Forest",
            "Extra Trees",
            "LightGBM",
            "Xgboost",
            "CatBoost",
            "Neural Network",
        ],

        validation_strategy={
            "validation_type": "kfold",
            "k_folds": 5,
            "shuffle": True,
            "random_seed": SEED,
        },

        golden_features=True,
        features_selection=True,
        train_ensemble=True,
        stack_models=True,
        explain_level=2,
    )

    automl.fit(X_train, y_train)


    # =============================================================================
    # 4. PREDICTIONS
    # =============================================================================

    y_pred_train = np.asarray(automl.predict(X_train), dtype=float)
    y_pred_test = np.asarray(automl.predict(X_test), dtype=float)

    y_train_true = y_train.to_numpy(dtype=float)
    y_test_true = y_test.to_numpy(dtype=float)


    # =============================================================================
    # 5. METRICS
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

    best_model_name = getattr(automl, "_best_model", "MLJAR_best_model_with_golden_features")

    metrics_df = pd.DataFrame([
        {
            "model": best_model_name,
            "dataset": "train",
            "n": len(y_train_true),
            "bias": train_bias,
            "cov": train_cov,
            "r2": train_r2,
            "rmse": train_rmse,
            "mae": train_mae,
        },
        {
            "model": best_model_name,
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
    print("MLJAR GOLDEN FEATURES - BEST MODEL METRICS")
    print("====================================================")
    print(metrics_df)


    # =============================================================================
    # 6. SAVE PREDICTIONS
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
    # 7. TRUE VS PREDICTED PLOT
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

    plt.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=2, label="Ideal fit")
    plt.plot(line_x, 1.1 * line_x, "r--", linewidth=1.5, label="±10% error")
    plt.plot(line_x, 0.9 * line_x, "r--", linewidth=1.5)

    plt.xlim(0, max_val * 1.05)
    plt.ylim(0, max_val * 1.05)

    plt.xlabel("Predicted moment resistance")
    plt.ylabel("True moment resistance")
    plt.title(f"MLJAR AutoML with Golden Features\nSeed {SEED}")

    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()

    plt.savefig(PLOT_PATH, dpi=300)
    plt.close()


    # =============================================================================
    # 8. LEADERBOARD
    # =============================================================================

    try:
        leaderboard = automl.get_leaderboard()
        leaderboard.to_csv(LEADERBOARD_PATH, index=False)

        print("\n====================================================")
        print("MLJAR LEADERBOARD")
        print("====================================================")
        print(leaderboard)

    except Exception as e:
        print("Could not export MLJAR leaderboard directly.")
        print(f"Reason: {e}")
        print("See MLJARModels/ for MLJAR-generated model reports.")


    # =============================================================================
    # 9. EXTRACT GOLDEN FEATURES FILE
    # =============================================================================

    possible_golden_paths = list(MLJAR_MODELS_DIR.rglob("golden_features.json"))

    print("\n====================================================")
    print("GOLDEN FEATURES")
    print("====================================================")

    golden_features = None

    if possible_golden_paths:
        golden_path = possible_golden_paths[0]
        print(f"Golden features file found: {golden_path}")

        with open(golden_path, "r", encoding="utf-8") as f:
            golden_features = json.load(f)

        with open(GOLDEN_FEATURES_COPY_PATH, "w", encoding="utf-8") as f:
            json.dump(golden_features, f, indent=4)

        print("\nGolden features content:")
        print(json.dumps(golden_features, indent=4))

        print(f"\nCopied golden features to: {GOLDEN_FEATURES_COPY_PATH}")

    else:
        print("No golden_features.json file found.")
        print("Check manually inside:")
        print(MLJAR_MODELS_DIR)


    # =============================================================================
    # 10. SAVE SUMMARY TXT
    # =============================================================================

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("MLJAR AUTOML WITH GOLDEN FEATURES — WELDED JOINTS\n")
        f.write("====================================================\n\n")

        f.write("DATA PROTOCOL\n")
        f.write("====================================================\n")
        f.write(f"Data file: {DATA_FILE.name}\n")
        f.write(f"Total clean samples: {len(df)}\n")
        f.write(f"Train/test split: {int((1 - TEST_SIZE) * 100)}:{int(TEST_SIZE * 100)}\n")
        f.write(f"Seed: {SEED}\n")
        f.write(f"Input columns: {INPUT_COLUMNS}\n")
        f.write(f"Output column: {OUTPUT_COLUMN}\n\n")

        f.write("MLJAR SETTINGS\n")
        f.write("====================================================\n")
        f.write("Mode: Compete\n")
        f.write("Task: regression\n")
        f.write("Internal metric: RMSE\n")
        f.write(f"Time limit: {TIME_LIMIT} seconds\n")
        f.write("Validation: 5-fold CV\n")
        f.write("Golden features: True\n")
        f.write("Feature selection: True\n")
        f.write("Ensemble: True\n")
        f.write("Stacking: True\n")
        f.write("Explain level: 2\n\n")

        f.write("BEST MODEL METRICS\n")
        f.write("====================================================\n")
        f.write(metrics_df.to_string(index=False))
        f.write("\n\n")

        f.write("GOLDEN FEATURES\n")
        f.write("====================================================\n")
        if golden_features is not None:
            f.write(json.dumps(golden_features, indent=4))
            f.write("\n")
        else:
            f.write("No golden_features.json file found automatically.\n")

        f.write("\nNOTES\n")
        f.write("====================================================\n")
        f.write(
            "This run evaluates MLJAR AutoML with golden_features=True. "
            "The results can be compared with a baseline MLJAR run with "
            "golden_features=False, provided both runs use the same seed, inputs, target, "
            "train/test split, and time budget.\n"
        )


    # =============================================================================
    # 11. DONE
    # =============================================================================

    print("\n====================================================")
    print("DONE")
    print("====================================================")
    print(f"Results folder:        {RESULTS_DIR}")
    print(f"Metrics saved to:      {METRICS_PATH}")
    print(f"Predictions saved to:  {PREDICTIONS_PATH}")
    print(f"Plot saved to:         {PLOT_PATH}")
    print(f"Leaderboard export target: {LEADERBOARD_PATH} (if supported)")
    print(f"Summary saved to:      {SUMMARY_PATH}")
    print(f"Golden-features export target: {GOLDEN_FEATURES_COPY_PATH} (if found)")


if __name__ == "__main__":
    main()
