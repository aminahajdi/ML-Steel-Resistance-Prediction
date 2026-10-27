# =============================================================================
# MLJAR AUTOML BENCHMARK FOR EXPERIMENTAL CWC DATABASE
# =============================================================================

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from supervised.automl import AutoML


# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[2]
DATA_FILE = REPO_ROOT / "data" / "study_03_data.xlsx"

RESULTS_DIR = BASE_DIR / "outputs" / "seed_7"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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
TIME_LIMIT = 3600  # Total MLJAR search budget, in seconds

LEADERBOARD_PATH = RESULTS_DIR / "mljar_leaderboard_seed7.csv"
METRICS_PATH = RESULTS_DIR / "mljar_best_model_metrics_seed7.csv"
PLOT_PATH = RESULTS_DIR / "mljar_true_vs_predicted_seed7.png"
PREDICTIONS_PATH = RESULTS_DIR / "mljar_predictions_seed7.csv"



def main():
    # =============================================================================
    # 1. LOAD AND CLEAN DATA
    # =============================================================================

    if not DATA_FILE.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_excel(DATA_FILE)

    df = df[INPUT_COLUMNS + [OUTPUT_COLUMN]].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[df[OUTPUT_COLUMN] > 0].copy()

    print("====================================================")
    print("MLJAR AUTOML BENCHMARK - EXPERIMENTAL CWC DATABASE")
    print("====================================================")
    print(f"Data file = {DATA_FILE}")
    print(f"Total clean specimens = {len(df)}")
    print(f"Input columns = {INPUT_COLUMNS}")
    print(f"Output column = {OUTPUT_COLUMN}")


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

    print(f"Train specimens = {len(train_df)}")
    print(f"Test specimens  = {len(test_df)}")


    # =============================================================================
    # 3. TRAIN MLJAR AUTOML
    # =============================================================================

    automl = AutoML(
        mode="Compete",
        ml_task="regression",
        eval_metric="rmse",
        total_time_limit=TIME_LIMIT,
        results_path=str(RESULTS_DIR / "MLJARModels"),
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

        golden_features=False,
        features_selection=True,
        train_ensemble=True,
        stack_models=True,
        explain_level=2,
    )

    automl.fit(X_train, y_train)


    # =============================================================================
    # 4. PREDICTIONS
    # =============================================================================

    F_pred_train = automl.predict(X_train)
    F_pred_test = automl.predict(X_test)

    F_pred_train = np.asarray(F_pred_train, dtype=float)
    F_pred_test = np.asarray(F_pred_test, dtype=float)

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

    best_model = getattr(automl, "_best_model", None)
    best_model_name = str(getattr(best_model, "get_name", lambda: best_model)()) if best_model is not None else "MLJAR_best_model"

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
    print("MLJAR BEST MODEL METRICS")
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

    plt.xlabel("Predicted experimental resistance")
    plt.ylabel("True experimental resistance")
    plt.title(f"MLJAR AutoML Best Model\nSeed {SEED}")

    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()

    plt.savefig(PLOT_PATH, dpi=300)
    plt.close()


    # =============================================================================
    # 8. LEADERBOARD / RESULTS
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
        print("See MLJARModels/ for automatically generated training reports and model rankings.")


    # =============================================================================
    # 9. DONE
    # =============================================================================

    print("\n====================================================")
    print("DONE")
    print("====================================================")
    print(f"MLJAR results folder: {RESULTS_DIR}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"Prediction plot saved to: {PLOT_PATH}")
    print(f"Predictions saved to: {PREDICTIONS_PATH}")

if __name__ == "__main__":
    main()
