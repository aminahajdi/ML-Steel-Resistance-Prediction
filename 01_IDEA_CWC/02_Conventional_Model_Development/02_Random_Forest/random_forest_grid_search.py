# =============================================================================
# RANDOM FOREST REGRESSION - 5-FOLD CV GRID SEARCH
# - Manually specified focused grid of eight RF configurations
# - 80:20 train/test split
# - 5-fold CV on the 80% train/validation pool
# - Best model selected by lowest mean CV CoV, highest mean CV R² second
# - Final model retrained on full train/validation pool
# - Final evaluation on independent 20% test set
# =============================================================================

import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor


# =============================================================================
# 0. REPRODUCIBILITY
# =============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# =============================================================================
# 1. FILES AND FOLDERS
# =============================================================================
# Resolve paths from the repository layout, independent of the working directory.
# File: 01_IDEA_CWC/02_Conventional_Model_Development/02_Random_Forest/
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_FILE = REPO_ROOT / "data" / "study_01_data.xlsx"

RESULTS_DIR = Path(__file__).resolve().parent / "outputs"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV_PATH = RESULTS_DIR / "random_forest_5fold_cv_results.csv"
BEST_SUMMARY_PATH = RESULTS_DIR / "best_random_forest_5fold_summary.txt"

PLOT_CV_PATH = RESULTS_DIR / "A_ranked_mean_cv_cov_random_forest.png"
PLOT_FINAL_PATH = RESULTS_DIR / "B_best_random_forest_final_prediction_plot.png"
PLOT_IMPORTANCE_PATH = RESULTS_DIR / "C_random_forest_feature_importance.png"


# =============================================================================
# 2. USER SETTINGS
# =============================================================================
TEST_SIZE = 0.20
N_SPLITS = 5
RANDOM_STATE = 42

USE_DATA_FILTERING = True
USE_INPUT_SCALER = True  # Retained for consistency with the original experiment

# Filtering
NUM_BINS = 30
COUNT_THRESHOLD = 5
FILTER_COLUMN_1 = "tf [value]"
FILTER_COLUMN_2 = "tw [value]"

# Columns - same as XGBoost script
INPUT_COLUMNS = [
    "D [value]",
    "B [value]",
    "tf [value]",
    "tw [value]",
    "rw [value]",
    "x",
    "a",
    "fy",
    "epsilon_lim",
]

OUTPUT_COLUMN = "F_plRd_CBFEM"

# Random Forest search space
N_ESTIMATORS_LIST = [800, 1000, 2000, 4000]
MAX_DEPTH_LIST = [None]
MIN_SAMPLES_SPLIT_LIST = [2]
MIN_SAMPLES_LEAF_LIST = [1]
MAX_FEATURES_LIST = [1.0]
BOOTSTRAP_LIST = [True, False]

SCATTER_ZOOM = 1.0


# =============================================================================
# 3. HELPER FUNCTIONS
# =============================================================================
def remove_sparse_bins(df, col, bins=30, count_threshold=5):
    """
    Removes all rows with values higher than the first sparsely populated bin limit.
    """
    counts, bin_edges = np.histogram(df[col], bins=bins)
    low_count_bins = np.where(counts < count_threshold)[0]

    if len(low_count_bins) == 0:
        return df.copy(), None

    first_low_bin = low_count_bins[0]
    cutoff_value = bin_edges[first_low_bin + 1]

    filtered_df = df[df[col] <= cutoff_value].copy()
    return filtered_df, cutoff_value


def compute_bias_cov_r2(y_true, y_pred, epsilon=1e-12):
    """
    Computes Bias, CoV, and R².

    Bias and CoV are based on the prediction/reference ratio:
        ratio = y_pred / y_true

    bias = mean(ratio)
    CoV  = std(ratio) / mean(ratio)
    """
    y_true_safe = np.where(np.abs(y_true) < epsilon, epsilon, y_true)
    ratios = y_pred / y_true_safe

    bias = np.mean(ratios)

    if bias <= 0 or np.isnan(bias) or np.isinf(bias):
        return np.nan, np.nan, np.nan, ratios

    cov = np.std(ratios) / bias

    if np.isnan(cov) or np.isinf(cov):
        return bias, np.nan, np.nan, ratios

    r2 = r2_score(y_true, y_pred)
    return bias, cov, r2, ratios


def create_prediction_plot(
    y_pred_all,
    y_true_all,
    y_pred_test,
    y_true_test,
    bias_all,
    cov_all,
    r2_all,
    bias_test,
    cov_test,
    r2_test,
    filename,
    zoom=1.0
):
    """
    Creates final predicted vs reference resistance plot.
    """
    plt.figure(figsize=(7, 7))

    combined_values = np.concatenate((y_pred_all, y_true_all, y_pred_test, y_true_test))
    min_val = np.min(combined_values)
    max_val = np.max(combined_values)

    zoom_limit = max_val * zoom
    line_x = np.linspace(min_val, max_val, 200)

    plt.scatter(
        y_pred_all,
        y_true_all,
        alpha=0.5,
        color="blue",
        edgecolors="black",
        s=30,
        label=(
            f"All data\n"
            f"n={len(y_true_all)}\n"
            f"Bias={bias_all:.4f}\n"
            f"CoV={cov_all:.4f}\n"
            f"R²={r2_all:.4f}"
        )
    )

    plt.scatter(
        y_pred_test,
        y_true_test,
        alpha=0.7,
        color="orange",
        edgecolors="black",
        s=35,
        label=(
            f"Test data\n"
            f"n={len(y_true_test)}\n"
            f"Bias={bias_test:.4f}\n"
            f"CoV={cov_test:.4f}\n"
            f"R²={r2_test:.4f}"
        )
    )

    plt.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=2, label="Ideal fit")
    plt.plot(line_x, 1.1 * line_x, "r--", linewidth=2, label="+/-10% error")
    plt.plot(line_x, 0.9 * line_x, "r--", linewidth=2)

    plt.xlim(0, zoom_limit * 1.05)
    plt.ylim(0, zoom_limit * 1.05)

    plt.xlabel("Predicted resistance")
    plt.ylabel("Reference resistance")
    plt.title("Random Forest: Predicted vs Reference Resistance")

    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def build_random_forest_model(
    n_estimators,
    max_depth,
    min_samples_split,
    min_samples_leaf,
    max_features,
    bootstrap
):
    """
    Builds one Random Forest model.
    """
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        random_state=SEED,
        n_jobs=-1
    )


# =============================================================================
# 4. READ DATA
# =============================================================================
if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

df = pd.read_excel(DATA_FILE)

print("====================================================")
print("DATASET INFO")
print("====================================================")
print(df.info())

missing_input_columns = [col for col in INPUT_COLUMNS if col not in df.columns]
if missing_input_columns:
    raise ValueError(f"Missing input columns in dataset: {missing_input_columns}")

if OUTPUT_COLUMN not in df.columns:
    raise ValueError(f"Output column not found in dataset: {OUTPUT_COLUMN}")


# =============================================================================
# 5. FILTER DATA
# =============================================================================
if USE_DATA_FILTERING:
    print("\n====================================================")
    print("DATA FILTERING")
    print("====================================================")

    df_full_length = len(df)

    df_filtered_tf, tf_cutoff = remove_sparse_bins(
        df,
        FILTER_COLUMN_1,
        bins=NUM_BINS,
        count_threshold=COUNT_THRESHOLD
    )

    df_filtered_tw, tw_cutoff = remove_sparse_bins(
        df_filtered_tf,
        FILTER_COLUMN_2,
        bins=NUM_BINS,
        count_threshold=COUNT_THRESHOLD
    )

    df_filtered_output, output_cutoff = remove_sparse_bins(
        df_filtered_tw,
        OUTPUT_COLUMN,
        bins=NUM_BINS,
        count_threshold=COUNT_THRESHOLD
    )

    df = df_filtered_output.copy()

    print(f"Original data length: {df_full_length}")
    print(f"Filtered data length: {len(df)}")
    print(f"{FILTER_COLUMN_1} cutoff: {tf_cutoff}")
    print(f"{FILTER_COLUMN_2} cutoff: {tw_cutoff}")
    print(f"{OUTPUT_COLUMN} cutoff: {output_cutoff}")


# =============================================================================
# 6. DEFINE INPUTS AND OUTPUT
# =============================================================================
X_df = df[INPUT_COLUMNS].copy()
y_array = df[OUTPUT_COLUMN].to_numpy(dtype=float)

X_np = X_df.to_numpy(dtype=float)
y_np = y_array.copy()

print("\n====================================================")
print("INPUTS AND OUTPUT")
print("====================================================")
print(f"Input columns: {INPUT_COLUMNS}")
print(f"Output column: {OUTPUT_COLUMN}")
print(f"Number of samples: {len(y_np)}")
print(f"Number of input features: {X_np.shape[1]}")


# =============================================================================
# 7. 80:20 TRAIN/TEST SPLIT
# =============================================================================
X_trainval_np, X_test_np, y_trainval_np, y_test_np = train_test_split(
    X_np,
    y_np,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    shuffle=True
)

print("\n====================================================")
print("DATA SPLIT")
print("====================================================")
print(f"Train/validation pool: {len(X_trainval_np)}")
print(f"Independent test set:  {len(X_test_np)}")
print(f"Number of CV folds:    {N_SPLITS}")


# =============================================================================
# 8. RANDOM FOREST 5-FOLD CV GRID SEARCH
# =============================================================================
results = []
model_counter = 0

total_configurations = (
    len(N_ESTIMATORS_LIST)
    * len(MAX_DEPTH_LIST)
    * len(MIN_SAMPLES_SPLIT_LIST)
    * len(MIN_SAMPLES_LEAF_LIST)
    * len(MAX_FEATURES_LIST)
    * len(BOOTSTRAP_LIST)
)

kf = KFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=SEED
)

print("\n====================================================")
print("START RANDOM FOREST 5-FOLD CV GRID SEARCH")
print("====================================================")
print(f"Total configurations: {total_configurations}")

search_start_time = time.time()

for n_estimators in N_ESTIMATORS_LIST:
    for max_depth in MAX_DEPTH_LIST:
        for min_samples_split in MIN_SAMPLES_SPLIT_LIST:
            for min_samples_leaf in MIN_SAMPLES_LEAF_LIST:
                for max_features in MAX_FEATURES_LIST:
                    for bootstrap in BOOTSTRAP_LIST:
                        model_counter += 1
                        config_start_time = time.time()

                        print(
                            f"\nConfiguration {model_counter}/{total_configurations}: "
                            f"n_estimators={n_estimators}, "
                            f"max_depth={max_depth}, "
                            f"min_samples_split={min_samples_split}, "
                            f"min_samples_leaf={min_samples_leaf}, "
                            f"max_features={max_features}, "
                            f"bootstrap={bootstrap}"
                        )

                        fold_biases = []
                        fold_covs = []
                        fold_r2s = []

                        for fold_id, (train_idx, val_idx) in enumerate(kf.split(X_trainval_np), start=1):
                            X_fold_train = X_trainval_np[train_idx]
                            X_fold_val = X_trainval_np[val_idx]

                            y_fold_train = y_trainval_np[train_idx]
                            y_fold_val = y_trainval_np[val_idx]

                            if USE_INPUT_SCALER:
                                input_scaler = StandardScaler()
                                X_fold_train_scaled = input_scaler.fit_transform(X_fold_train)
                                X_fold_val_scaled = input_scaler.transform(X_fold_val)
                            else:
                                X_fold_train_scaled = X_fold_train.copy()
                                X_fold_val_scaled = X_fold_val.copy()

                            model = build_random_forest_model(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
                                max_features=max_features,
                                bootstrap=bootstrap
                            )

                            model.fit(X_fold_train_scaled, y_fold_train)
                            y_pred_val = model.predict(X_fold_val_scaled)

                            val_bias, val_cov, val_r2, _ = compute_bias_cov_r2(
                                y_fold_val,
                                y_pred_val
                            )

                            if np.isnan(val_cov) or np.isinf(val_cov):
                                print(f" -> Fold {fold_id}: invalid CoV. Skipping configuration.")
                                fold_covs = [np.nan]
                                break

                            fold_biases.append(val_bias)
                            fold_covs.append(val_cov)
                            fold_r2s.append(val_r2)

                            print(
                                f"   Fold {fold_id}: "
                                f"Bias={val_bias:.4f}, "
                                f"CoV={val_cov:.4f}, "
                                f"R²={val_r2:.6f}"
                            )

                        if np.any(np.isnan(fold_covs)):
                            continue

                        mean_cv_bias = float(np.mean(fold_biases))
                        std_cv_bias = float(np.std(fold_biases))

                        mean_cv_cov = float(np.mean(fold_covs))
                        std_cv_cov = float(np.std(fold_covs))

                        mean_cv_r2 = float(np.mean(fold_r2s))
                        std_cv_r2 = float(np.std(fold_r2s))

                        config_time = time.time() - config_start_time

                        print(
                            f" -> Mean CV Bias={mean_cv_bias:.4f} | "
                            f"Mean CV CoV={mean_cv_cov:.4f} | "
                            f"Std CV CoV={std_cv_cov:.4f} | "
                            f"Mean CV R²={mean_cv_r2:.6f} | "
                            f"Time={config_time:.2f} s"
                        )

                        results.append({
                            "model_id": model_counter,
                            "n_estimators": n_estimators,
                            "max_depth": max_depth,
                            "min_samples_split": min_samples_split,
                            "min_samples_leaf": min_samples_leaf,
                            "max_features": max_features,
                            "bootstrap": bootstrap,
                            "mean_cv_bias": mean_cv_bias,
                            "std_cv_bias": std_cv_bias,
                            "mean_cv_cov": mean_cv_cov,
                            "std_cv_cov": std_cv_cov,
                            "mean_cv_r2": mean_cv_r2,
                            "std_cv_r2": std_cv_r2,
                            "training_time_s": config_time
                        })

search_time = time.time() - search_start_time


# =============================================================================
# 9. RESULTS TABLE AND MODEL SELECTION
# =============================================================================
results_df = pd.DataFrame(results)

if results_df.empty:
    raise RuntimeError("No valid Random Forest configurations were found.")

# Selection rule: lowest mean CV CoV first, highest mean CV R² second
results_df = results_df.sort_values(
    by=["mean_cv_cov", "mean_cv_r2"],
    ascending=[True, False]
).reset_index(drop=True)

results_df["rank"] = np.arange(1, len(results_df) + 1)
results_df.to_csv(RESULTS_CSV_PATH, index=False)

best_row = results_df.iloc[0]

best_model_summary = {
    "model_id": int(best_row["model_id"]),
    "n_estimators": int(best_row["n_estimators"]),
    "max_depth": None if pd.isna(best_row["max_depth"]) else int(best_row["max_depth"]),
    "min_samples_split": int(best_row["min_samples_split"]),
    "min_samples_leaf": int(best_row["min_samples_leaf"]),
    "max_features": best_row["max_features"],
    "bootstrap": bool(best_row["bootstrap"]),
}

print("\n====================================================")
print("BEST RANDOM FOREST MODEL SELECTED BY 5-FOLD CV")
print("====================================================")
print(best_model_summary)
print(f"Mean CV Bias = {best_row['mean_cv_bias']:.6f}")
print(f"Mean CV CoV  = {best_row['mean_cv_cov']:.6f}")
print(f"Std CV CoV   = {best_row['std_cv_cov']:.6f}")
print(f"Mean CV R²   = {best_row['mean_cv_r2']:.6f}")


# =============================================================================
# 10. FINAL TRAINING ON FULL TRAIN/VALIDATION POOL
# =============================================================================
print("\n====================================================")
print("FINAL TRAINING ON FULL TRAIN/VALIDATION POOL")
print("====================================================")

if USE_INPUT_SCALER:
    final_scaler = StandardScaler()
    X_trainval_scaled = final_scaler.fit_transform(X_trainval_np)
    X_test_scaled = final_scaler.transform(X_test_np)
    X_all_scaled = final_scaler.transform(X_np)
else:
    final_scaler = None
    X_trainval_scaled = X_trainval_np.copy()
    X_test_scaled = X_test_np.copy()
    X_all_scaled = X_np.copy()

final_model = build_random_forest_model(
    n_estimators=best_model_summary["n_estimators"],
    max_depth=best_model_summary["max_depth"],
    min_samples_split=best_model_summary["min_samples_split"],
    min_samples_leaf=best_model_summary["min_samples_leaf"],
    max_features=best_model_summary["max_features"],
    bootstrap=best_model_summary["bootstrap"]
)

final_start_time = time.time()
final_model.fit(X_trainval_scaled, y_trainval_np)
final_training_time = time.time() - final_start_time

y_pred_test = final_model.predict(X_test_scaled)
y_pred_all = final_model.predict(X_all_scaled)

test_bias, test_cov, test_r2, test_ratios = compute_bias_cov_r2(y_test_np, y_pred_test)
all_bias, all_cov, all_r2, all_ratios = compute_bias_cov_r2(y_np, y_pred_all)

print("\n====================================================")
print("FINAL RANDOM FOREST METRICS")
print("====================================================")
print(f"All data:  Bias={all_bias:.6f}, CoV={all_cov:.6f}, R²={all_r2:.6f}")
print(f"Test data: Bias={test_bias:.6f}, CoV={test_cov:.6f}, R²={test_r2:.6f}")
print(f"Test ratio min={np.min(test_ratios):.6f}, max={np.max(test_ratios):.6f}")
print(f"All ratio min={np.min(all_ratios):.6f}, max={np.max(all_ratios):.6f}")
print(f"Final model training time: {final_training_time:.2f} s")
print(f"Total search time: {search_time:.2f} s")


# =============================================================================
# 11. ADD FINAL TEST METRICS TO RESULTS CSV FOR BEST ROW
# =============================================================================
results_df.loc[0, "test_bias"] = test_bias
results_df.loc[0, "test_cov"] = test_cov
results_df.loc[0, "test_r2"] = test_r2
results_df.loc[0, "test_ratio_min"] = np.min(test_ratios)
results_df.loc[0, "test_ratio_max"] = np.max(test_ratios)

results_df.loc[0, "all_bias"] = all_bias
results_df.loc[0, "all_cov"] = all_cov
results_df.loc[0, "all_r2"] = all_r2
results_df.loc[0, "all_ratio_min"] = np.min(all_ratios)
results_df.loc[0, "all_ratio_max"] = np.max(all_ratios)

results_df.loc[0, "final_training_time_s"] = final_training_time
results_df.loc[0, "total_search_time_s"] = search_time

results_df.to_csv(RESULTS_CSV_PATH, index=False)


# =============================================================================
# 12. PLOT A - RANKED MEAN CV CoV
# =============================================================================
plt.figure(figsize=(10, 5))

plt.scatter(
    results_df["rank"],
    results_df["mean_cv_cov"],
    s=30,
    alpha=0.7,
    label="Tested Random Forest configurations"
)

plt.scatter(
    best_row["rank"],
    best_row["mean_cv_cov"],
    s=120,
    marker="*",
    color="red",
    edgecolors="black",
    label=f"Best configuration\nMean CV CoV={best_row['mean_cv_cov']:.4f}"
)

plt.xlabel("Configuration rank")
plt.ylabel("Mean CV CoV")
plt.title("Random Forest 5-Fold CV Grid Search")
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_CV_PATH, dpi=300)
plt.close()


# =============================================================================
# 13. PLOT B - FINAL PREDICTION PLOT
# =============================================================================
create_prediction_plot(
    y_pred_all=y_pred_all,
    y_true_all=y_np,
    y_pred_test=y_pred_test,
    y_true_test=y_test_np,
    bias_all=all_bias,
    cov_all=all_cov,
    r2_all=all_r2,
    bias_test=test_bias,
    cov_test=test_cov,
    r2_test=test_r2,
    filename=PLOT_FINAL_PATH,
    zoom=SCATTER_ZOOM
)


# =============================================================================
# 14. PLOT C - FEATURE IMPORTANCE
# =============================================================================
feature_importances = final_model.feature_importances_

importance_df = pd.DataFrame({
    "feature": INPUT_COLUMNS,
    "importance": feature_importances
}).sort_values("importance", ascending=True)

plt.figure(figsize=(8, 5))
plt.barh(
    importance_df["feature"],
    importance_df["importance"]
)
plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance")
plt.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.6)
plt.tight_layout()
plt.savefig(PLOT_IMPORTANCE_PATH, dpi=300)
plt.close()

importance_df.sort_values("importance", ascending=False).to_csv(
    RESULTS_DIR / "random_forest_feature_importance.csv",
    index=False
)


# =============================================================================
# 15. SAVE SUMMARY TXT
# =============================================================================
with open(BEST_SUMMARY_PATH, "w", encoding="utf-8") as f:
    f.write("BEST RANDOM FOREST MODEL SUMMARY - 5-FOLD CV\n")
    f.write("============================================\n")
    f.write("Selection rule: lowest mean CV CoV first, highest mean CV R² second.\n\n")

    f.write("DATA PROTOCOL\n")
    f.write("============================================\n")
    f.write(f"Train/test split: {int((1 - TEST_SIZE) * 100)}:{int(TEST_SIZE * 100)}\n")
    f.write(f"CV folds: {N_SPLITS}\n")
    f.write(f"Input columns: {INPUT_COLUMNS}\n")
    f.write(f"Output column: {OUTPUT_COLUMN}\n\n")

    f.write("BEST PARAMETERS\n")
    f.write("============================================\n")
    for key, value in best_model_summary.items():
        f.write(f"{key}: {value}\n")

    f.write("\nCROSS-VALIDATION METRICS\n")
    f.write("============================================\n")
    f.write(f"Mean CV bias: {best_row['mean_cv_bias']:.6f}\n")
    f.write(f"Std CV bias:  {best_row['std_cv_bias']:.6f}\n")
    f.write(f"Mean CV CoV:  {best_row['mean_cv_cov']:.6f}\n")
    f.write(f"Std CV CoV:   {best_row['std_cv_cov']:.6f}\n")
    f.write(f"Mean CV R²:   {best_row['mean_cv_r2']:.6f}\n")
    f.write(f"Std CV R²:    {best_row['std_cv_r2']:.6f}\n")

    f.write("\nFINAL TEST METRICS\n")
    f.write("============================================\n")
    f.write(f"All data bias:     {all_bias:.6f}\n")
    f.write(f"All data CoV:      {all_cov:.6f}\n")
    f.write(f"All data R²:       {all_r2:.6f}\n")
    f.write(f"All ratio min:     {np.min(all_ratios):.6f}\n")
    f.write(f"All ratio max:     {np.max(all_ratios):.6f}\n")
    f.write(f"Test bias:         {test_bias:.6f}\n")
    f.write(f"Test CoV:          {test_cov:.6f}\n")
    f.write(f"Test R²:           {test_r2:.6f}\n")
    f.write(f"Test ratio min:    {np.min(test_ratios):.6f}\n")
    f.write(f"Test ratio max:    {np.max(test_ratios):.6f}\n")

    f.write("\nTIME\n")
    f.write("============================================\n")
    f.write(f"Total search time [s]: {search_time:.6f}\n")
    f.write(f"Final training time [s]: {final_training_time:.6f}\n")

    f.write("\nTOP 10 CONFIGURATIONS\n")
    f.write("============================================\n")
    top10_cols = [
        "rank",
        "model_id",
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
        "bootstrap",
        "mean_cv_bias",
        "mean_cv_cov",
        "std_cv_cov",
        "mean_cv_r2",
    ]
    f.write(results_df[top10_cols].head(10).to_string(index=False))


# =============================================================================
# 16. DONE
# =============================================================================
print("\n====================================================")
print("DONE")
print("====================================================")
print(f"Results table saved to: {RESULTS_CSV_PATH}")
print(f"Best summary saved to: {BEST_SUMMARY_PATH}")
print(f"Ranked CV plot saved to: {PLOT_CV_PATH}")
print(f"Final prediction plot saved to: {PLOT_FINAL_PATH}")
print(f"Feature importance plot saved to: {PLOT_IMPORTANCE_PATH}")
