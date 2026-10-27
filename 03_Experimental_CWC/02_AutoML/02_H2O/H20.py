"""H2O AutoML regression benchmark for experimental column-web compression.

The script evaluates H2O AutoML with a 3,600-second budget and an 80/20
split (seed 7), and saves a leaderboard, performance metrics, and figures.

Important: the original experiment supplies the test partition as H2O's
``leaderboard_frame``. This may influence leaderboard ranking/model choice;
therefore, the resulting test metrics must not be described as strictly
untouched independent-test estimates. The call is retained here to preserve
the original experiment's numerical procedure.
"""

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import h2o
from h2o.automl import H2OAutoML


# =============================================================================
# 0. EXPERIMENT CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
DATA_FILE = REPOSITORY_ROOT / "data" / "study_03_data.xlsx"

RESULTS_DIR = SCRIPT_DIR / "outputs" / "seed_7"


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
TIME_LIMIT = 3600  # H2O AutoML time budget, in seconds

LEADERBOARD_PATH = RESULTS_DIR / "h2o_leaderboard_seed7.csv"
METRICS_PATH = RESULTS_DIR / "h2o_best_model_metrics_seed7.csv"
PLOT_PATH = RESULTS_DIR / "h2o_true_vs_predicted_seed7.png"
MODEL_DIR = RESULTS_DIR / "H2OModels"



# =============================================================================
# 1. START H2O
# =============================================================================

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    h2o.init(
        nthreads=-1,
        max_mem_size="4G",
    )


    # =============================================================================
    # 2. LOAD AND CLEAN DATA
    # =============================================================================

    df = pd.read_excel(DATA_FILE)

    df = df[INPUT_COLUMNS + [OUTPUT_COLUMN]].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[df[OUTPUT_COLUMN] > 0].copy()

    print("====================================================")
    print("H2O AUTOML BENCHMARK - EXPERIMENTAL CWC DATABASE")
    print("====================================================")
    print(f"Data file = {DATA_FILE}")
    print(f"Total clean specimens = {len(df)}")
    print(f"Input columns = {INPUT_COLUMNS}")
    print(f"Output column = {OUTPUT_COLUMN}")


    # =============================================================================
    # 3. TRAIN / TEST SPLIT
    # =============================================================================

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=SEED,
        shuffle=True,
    )

    print(f"Train specimens = {len(train_df)}")
    print(f"Test specimens  = {len(test_df)}")


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
        nfolds=5,
        sort_metric="RMSE",
        stopping_metric="RMSE",
        project_name="H2O_CWC_Benchmark_Seed7",
    )

    aml.train(
        x=INPUT_COLUMNS,
        y=OUTPUT_COLUMN,
        training_frame=train_h2o,
        # Retained to match the original experiment. H2O can use this frame
        # for leaderboard scoring and model ranking; see README limitations.
        leaderboard_frame=test_h2o,
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

    print(f"\nBest H2O model saved to: {saved_model_path}")


    # =============================================================================
    # 8. PREDICTIONS
    # =============================================================================

    train_pred_h2o = leader.predict(train_h2o)
    test_pred_h2o = leader.predict(test_h2o)

    F_pred_train = train_pred_h2o.as_data_frame()["predict"].to_numpy(dtype=float)
    F_pred_test = test_pred_h2o.as_data_frame()["predict"].to_numpy(dtype=float)

    y_train_exp = train_df[OUTPUT_COLUMN].to_numpy(dtype=float)
    y_test_exp = test_df[OUTPUT_COLUMN].to_numpy(dtype=float)


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
        y_train_exp,
        F_pred_train,
    )

    test_bias, test_cov, test_r2, test_rmse, test_mae = compute_metrics(
        y_test_exp,
        F_pred_test,
    )

    metrics_df = pd.DataFrame([
        {
            "model": leader.model_id,
            "dataset": "train",
            "bias": train_bias,
            "cov": train_cov,
            "r2": train_r2,
            "rmse": train_rmse,
            "mae": train_mae,
        },
        {
            "model": leader.model_id,
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
    print("H2O BEST MODEL METRICS")
    print("====================================================")
    print(metrics_df)


    # =============================================================================
    # 10. TRUE VS PREDICTED PLOT
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
    plt.title(f"H2O AutoML Best Model\n{leader.model_id}")

    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()

    plt.savefig(PLOT_PATH, dpi=300)
    plt.close()


    # =============================================================================
    # 11. VARIABLE IMPORTANCE, IF AVAILABLE
    # =============================================================================

    try:
        varimp = leader.varimp(use_pandas=True)

        if varimp is not None and len(varimp) > 0:
            varimp_path = RESULTS_DIR / "h2o_variable_importance_seed7.csv"
            varimp_plot_path = RESULTS_DIR / "h2o_variable_importance_seed7.png"

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
    # 12. OPTIONAL H2O EXPLAINABILITY REPORT (NOT SAVED TO DISK)
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
    # 13. DONE
    # =============================================================================

    print("\n====================================================")
    print("DONE")
    print("====================================================")
    print(f"Leaderboard saved to: {LEADERBOARD_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"Prediction plot saved to: {PLOT_PATH}")
    print(f"Best H2O model saved to: {saved_model_path}")

    h2o.shutdown(prompt=False)

if __name__ == "__main__":
    main()
