# =============================================================================
# MLJAR AUTOML BENCHMARK WITH GOLDEN FEATURES — EXPERIMENTAL CWC
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
REPO_ROOT = SCRIPT_DIR.parents[1]
DATA_FILE = REPO_ROOT / "data" / "study_03_data.xlsx"

RESULTS_DIR = SCRIPT_DIR / "outputs" / "seed_7"

MLJAR_MODELS_DIR = RESULTS_DIR / "MLJARModels"

OUTPUT_COLUMN = "F_exp"

INPUT_COLUMNS = [
    "h_c",
    "t_wc",
    "b_fc",
    "t_fc",
    "r_c",
    "t_fb",
    "N/Npl",
    "f_y",
]

SEED = 7
TEST_SIZE = 0.20
TIME_LIMIT = 3600  # Total MLJAR search budget in seconds

METRICS_PATH = RESULTS_DIR / "mljar_golden_features_metrics_seed7.csv"
PREDICTIONS_PATH = RESULTS_DIR / "mljar_golden_features_predictions_seed7.csv"
PLOT_PATH = RESULTS_DIR / "mljar_golden_features_true_vs_predicted_seed7.png"
LEADERBOARD_PATH = RESULTS_DIR / "mljar_golden_features_leaderboard_seed7.csv"
GOLDEN_FEATURES_COPY_PATH = RESULTS_DIR / "golden_features_extracted_seed7.json"




def main():
    """Train and evaluate MLJAR AutoML with Golden Features enabled."""
    if not DATA_FILE.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # =============================================================================
    # 1. LOAD DATA
    # =============================================================================

    df = pd.read_excel(DATA_FILE)

    df = df[INPUT_COLUMNS + [OUTPUT_COLUMN]].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[df[OUTPUT_COLUMN] > 0].copy()

    print("====================================================")
    print("MLJAR GOLDEN FEATURES - EXPERIMENTAL CWC DATABASE")
    print("====================================================")
    print(f"Dataset file          = {DATA_FILE}")
    print(f"Total clean specimens = {len(df)}")
    print(f"Input columns         = {INPUT_COLUMNS}")
    print(f"Output column         = {OUTPUT_COLUMN}")
    print(f"Golden features       = True")


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

    print(f"Train specimens       = {len(train_df)}")
    print(f"Test specimens        = {len(test_df)}")


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

    F_pred_train = np.asarray(automl.predict(X_train), dtype=float)
    F_pred_test = np.asarray(automl.predict(X_test), dtype=float)

    y_train_exp = y_train.to_numpy(dtype=float)
    y_test_exp = y_test.to_numpy(dtype=float)


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
        y_train_exp,
        F_pred_train,
    )

    test_bias, test_cov, test_r2, test_rmse, test_mae = compute_metrics(
        y_test_exp,
        F_pred_test,
    )

    best_model_name = getattr(automl, "_best_model", "MLJAR_best_model_with_golden_features")

    metrics_df = pd.DataFrame([
        {
            "model": best_model_name,
            "dataset": "train",
            "bias": train_bias,
            "cov": train_cov,
            "r2": train_r2,
            "rmse": train_rmse,
            "mae": train_mae,
        },
        {
            "model": best_model_name,
            "dataset": "test",
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

    predictions_df = pd.DataFrame({
        "dataset": ["train"] * len(y_train_exp) + ["test"] * len(y_test_exp),
        "F_true": np.concatenate([y_train_exp, y_test_exp]),
        "F_pred": np.concatenate([F_pred_train, F_pred_test]),
    })

    predictions_df["ratio_pred_true"] = predictions_df["F_pred"] / predictions_df["F_true"]
    predictions_df.to_csv(PREDICTIONS_PATH, index=False)


    # =============================================================================
    # 7. TRUE VS PREDICTED PLOT
    # =============================================================================

    plt.figure(figsize=(7, 7))

    combined_values = np.concatenate(
        (
            F_pred_train,
            y_train_exp,
            F_pred_test,
            y_test_exp,
        )
    )

    min_val = float(np.min(combined_values))
    max_val = float(np.max(combined_values))
    line_x = np.linspace(min_val, max_val, 200)

    plt.scatter(
        F_pred_train,
        y_train_exp,
        alpha=0.65,
        edgecolors="black",
        s=35,
        color="tab:blue",
        label=(
            f"Train data\n"
            f"n={len(y_train_exp)}\n"
            f"b={train_bias:.4f}\n"
            f"CoV={train_cov:.4f}\n"
            f"R²={train_r2:.4f}"
        ),
    )

    plt.scatter(
        F_pred_test,
        y_test_exp,
        alpha=0.85,
        edgecolors="black",
        s=50,
        color="tab:purple",
        label=(
            f"Test data\n"
            f"n={len(y_test_exp)}\n"
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

    plt.xlabel("Predicted experimental resistance")
    plt.ylabel("True experimental resistance")
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
        print("Check the MLJARModels folder. MLJAR automatically saves model reports there.")


    # =============================================================================
    # 9. EXTRACT GOLDEN FEATURES FILE
    # =============================================================================

    possible_golden_paths = list(MLJAR_MODELS_DIR.rglob("golden_features.json"))

    print("\n====================================================")
    print("GOLDEN FEATURES")
    print("====================================================")

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
    # 10. DONE
    # =============================================================================

    print("\n====================================================")
    print("DONE")
    print("====================================================")
    print(f"Results folder:        {RESULTS_DIR}")
    print(f"Metrics saved to:      {METRICS_PATH}")
    print(f"Predictions saved to:  {PREDICTIONS_PATH}")
    print(f"Plot saved to:         {PLOT_PATH}")
    print(f"Leaderboard path (if available): {LEADERBOARD_PATH}")
    print(f"Golden-features JSON path (if available): {GOLDEN_FEATURES_COPY_PATH}")

if __name__ == "__main__":
    main()
