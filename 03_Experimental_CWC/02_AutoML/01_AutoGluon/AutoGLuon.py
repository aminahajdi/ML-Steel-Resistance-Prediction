"""AutoGluon benchmark for experimental column-web transverse compression.

The script uses an independent 80/20 train/test split and a one-hour
AutoGluon training budget. AutoGluon selects its preferred model according
 to its configured validation metric (RMSE); resistance-based bias, CoV,
R², RMSE, and MAE are reported subsequently for that model.

The input dataset is distributed separately from the code repository.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 1. EXPERIMENT CONFIGURATION
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DATA_FILE = REPOSITORY_ROOT / "data" / "study_03_data.xlsx"
RESULTS_DIR = Path(__file__).resolve().parent / "outputs" / "seed_7"

INPUT_COLUMNS = [
    "h_c", "t_wc", "b_fc", "t_fc", "r_c", "t_fb", "N/Npl", "f_y"
]
TARGET_COLUMN = "F_exp"
SEED = 7
TEST_SIZE = 0.20
TIME_LIMIT_SECONDS = 3600


def compute_metrics(y_true, y_pred):
    """Return bias and population CoV of prediction/reference, R², RMSE, MAE."""
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


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_FILE.is_file():
        raise FileNotFoundError(f"Study 3 dataset not found: {DATA_FILE}")

    # 2. DATA LOADING AND COMPLETE-CASE SELECTION
    frame = pd.read_excel(DATA_FILE)
    required_columns = INPUT_COLUMNS + [TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in frame]
    if missing_columns:
        raise ValueError(f"Required dataset columns are missing: {missing_columns}")
    frame = frame[required_columns].copy()
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    frame = frame[frame[TARGET_COLUMN] > 0].copy()
    # No additional outlier exclusion is performed in this benchmark.

    print("AutoGluon benchmark — experimental CWC")
    print(f"Dataset: {DATA_FILE}")
    print(f"Complete, positive-target observations: {len(frame)}")
    print(f"Input variables: {INPUT_COLUMNS}")
    print(f"Target variable: {TARGET_COLUMN}")

    # 3. INDEPENDENT TRAIN/TEST PARTITION
    train_df, test_df = train_test_split(
        frame, test_size=TEST_SIZE, random_state=SEED, shuffle=True
    )
    print(f"Training observations: {len(train_df)}")
    print(f"Test observations: {len(test_df)}")

    # 4. AUTOMATED MODEL TRAINING
    predictor = TabularPredictor(
        label=TARGET_COLUMN,
        problem_type="regression",
        eval_metric="root_mean_squared_error",
        path=str(RESULTS_DIR / "AutoGluonModels"),
    )
    predictor.fit(
        train_data=train_df,
        presets="best_quality",
        time_limit=TIME_LIMIT_SECONDS,
        num_bag_folds=5,
        num_stack_levels=1,
        ag_args_fit={"num_gpus": 0},
    )

    # 5. HOLDOUT LEADERBOARD (DESCRIPTIVE; NOT USED TO SELECT THE MODEL)
    leaderboard = predictor.leaderboard(test_df, silent=True)
    leaderboard_path = RESULTS_DIR / "autogluon_leaderboard_seed7.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    print("\nAutoGluon leaderboard:\n", leaderboard)

    # 6. METRICS OF THE MODEL SELECTED BY AUTOGLUON
    best_model = predictor.model_best
    y_train_true = train_df[TARGET_COLUMN].to_numpy()
    y_test_true = test_df[TARGET_COLUMN].to_numpy()
    y_train_pred = predictor.predict(train_df, model=best_model).to_numpy()
    y_test_pred = predictor.predict(test_df, model=best_model).to_numpy()
    metrics_df = pd.DataFrame([
        dict(zip(("bias", "cov", "r2", "rmse", "mae"),
                 compute_metrics(y_train_true, y_train_pred)),
             model=best_model, dataset="train"),
        dict(zip(("bias", "cov", "r2", "rmse", "mae"),
                 compute_metrics(y_test_true, y_test_pred)),
             model=best_model, dataset="test"),
    ])
    metrics_df = metrics_df[["model", "dataset", "bias", "cov", "r2", "rmse", "mae"]]
    metrics_path = RESULTS_DIR / "best_model_metrics_seed7.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print("\nSelected-model metrics:\n", metrics_df)

    # 7. PERMUTATION FEATURE IMPORTANCE ON THE TEST SUBSET
    importance = predictor.feature_importance(test_df)
    importance_path = RESULTS_DIR / "feature_importance_seed7.csv"
    importance.to_csv(importance_path)
    print("\nFeature importance:\n", importance)

    print("\nResults saved to:", RESULTS_DIR)
    print("Leaderboard:", leaderboard_path)
    print("Metrics:", metrics_path)
    print("Feature importance:", importance_path)
    print("Trained models:", RESULTS_DIR / "AutoGluonModels")


if __name__ == "__main__":
    main()
