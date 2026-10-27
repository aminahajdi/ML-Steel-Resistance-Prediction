"""Study 1: training-dataset-size sensitivity for RF, XGBoost and DNN.

Reproduces the archived experiment using nested training subsets and a fixed
80:20 training/test split. Hyperparameters are fixed; no tuning occurs here.

The source experiment applies sequential sparse-bin filters to two input
variables AND the target before splitting. It also varies the DNN random
initialization by training-size index. Both choices are intentionally retained
for traceability, pending manuscript/methodological review.

Run from any directory:
python 01_IDEA_CWC/03_Dataset_Size_Sensitivity/dataset_size_sensitivity.py
Data location: data/study_01_data.xlsx (distributed separately).
"""

# =============================================================================
# DATASET-SIZE SENSITIVITY ANALYSIS - RF, XGBoost, DNN
# =============================================================================
# Goal:
# - Evaluate how model performance changes with the number of training samples.
# - Keep the same filtered dataset, the same 80:20 train/test split, and the same
#   independent test set for every training size.
# - Use already selected final hyperparameters.
# - No hyperparameter tuning is performed.
#
# Output:
# - dataset_size_sensitivity_results.csv
# - dataset_size_test_cov_plot.png
# =============================================================================

import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

import torch
import torch.nn as nn
import torch.optim as optim


# =============================================================================
# 0. REPRODUCIBILITY
# =============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# 1. FILES AND FOLDERS
# =============================================================================
# Paths are based on this script, not the folder from which Python is launched.
STUDY_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = REPOSITORY_DIR / "data" / "study_01_data.xlsx"
RESULTS_DIR = STUDY_DIR / "outputs"
FIGURES_DIR = RESULTS_DIR
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV_PATH = RESULTS_DIR / "dataset_size_sensitivity_results.csv"
PLOT_COV_PATH = FIGURES_DIR / "dataset_size_test_cov_plot.png"
PLOT_R2_PATH = FIGURES_DIR / "dataset_size_test_r2_plot.png"
SUMMARY_TXT_PATH = RESULTS_DIR / "dataset_size_sensitivity_summary.txt"


# =============================================================================
# 2. SETTINGS
# =============================================================================
TEST_SIZE = 0.20
RANDOM_STATE = 42

USE_DATA_FILTERING = True
USE_INPUT_SCALER_FOR_TREES = True  # Mirrors the archived experiment; tree models do not require scaling.

# Training subset sizes are sampled from the 80% train/validation pool.
# "full" means all available samples in the train/validation pool.
TRAINING_SIZES = [1000, 2000, 3000, 4000, 5000, 6000, 7000, "full"]

# Filtering
NUM_BINS = 30
COUNT_THRESHOLD = 5
FILTER_COLUMN_1 = "tf [value]"
FILTER_COLUMN_2 = "tw [value]"

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


# =============================================================================
# 3. FINAL MODEL HYPERPARAMETERS
# =============================================================================
BEST_RF_PARAMS = {
    "n_estimators": 2000,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": 1.0,
    "bootstrap": True,
}

BEST_XGB_PARAMS = {
    "n_estimators": 8000,
    "max_depth": 7,
    "learning_rate": 0.01,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
}

# DNN hyperparameters selected previously
ARCHITECTURE = (128, 64, 32, 16)
ACTIVATION_NAME = "GELU"
OPTIMIZER_NAME = "AdamW"
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.01

# Use the final epochs selected from the previous DNN 5-fold diagnostic.
# Change this if your final DNN diagnostic selected a different value.
DNN_EPOCHS = 4500
DNN_EVAL_STEP = 100


# =============================================================================
# 4. HELPER FUNCTIONS
# =============================================================================
def remove_sparse_bins(df, col, bins=30, count_threshold=5):
    counts, bin_edges = np.histogram(df[col], bins=bins)
    low_count_bins = np.where(counts < count_threshold)[0]

    if len(low_count_bins) == 0:
        return df.copy(), None

    first_low_bin = low_count_bins[0]
    cutoff_value = bin_edges[first_low_bin + 1]
    filtered_df = df[df[col] <= cutoff_value].copy()
    return filtered_df, cutoff_value


def compute_bias_cov_r2(y_true, y_pred, epsilon=1e-12):
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


def build_random_forest_model():
    return RandomForestRegressor(
        n_estimators=BEST_RF_PARAMS["n_estimators"],
        max_depth=BEST_RF_PARAMS["max_depth"],
        min_samples_split=BEST_RF_PARAMS["min_samples_split"],
        min_samples_leaf=BEST_RF_PARAMS["min_samples_leaf"],
        max_features=BEST_RF_PARAMS["max_features"],
        bootstrap=BEST_RF_PARAMS["bootstrap"],
        random_state=SEED,
        n_jobs=-1,
    )


def build_xgboost_model():
    return XGBRegressor(
        n_estimators=BEST_XGB_PARAMS["n_estimators"],
        max_depth=BEST_XGB_PARAMS["max_depth"],
        learning_rate=BEST_XGB_PARAMS["learning_rate"],
        subsample=BEST_XGB_PARAMS["subsample"],
        colsample_bytree=BEST_XGB_PARAMS["colsample_bytree"],
        min_child_weight=BEST_XGB_PARAMS["min_child_weight"],
        reg_alpha=BEST_XGB_PARAMS["reg_alpha"],
        reg_lambda=BEST_XGB_PARAMS["reg_lambda"],
        objective="reg:squarederror",
        random_state=SEED,
        n_jobs=-1,
    )


def get_activation(name):
    if name == "ReLU":
        return nn.ReLU()
    if name == "LeakyReLU":
        return nn.LeakyReLU()
    if name == "ELU":
        return nn.ELU()
    if name == "GELU":
        return nn.GELU()
    if name == "SiLU":
        return nn.SiLU()
    if name == "Tanh":
        return nn.Tanh()
    raise ValueError(f"Unknown activation function: {name}")


def get_optimizer(name, model, learning_rate, weight_decay):
    if name == "SGD":
        return optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if name == "SGD_Momentum":
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    if name == "Adagrad":
        return optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if name == "RMSprop":
        return optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if name == "Adam":
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if name == "AdamW":
        return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


class RelativeSquaredError(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        relative_error = (y_true - y_pred) / (y_true + self.epsilon)
        return torch.mean(relative_error ** 2)


class DNNRegressor(nn.Module):
    def __init__(self, num_features, architecture, activation_name):
        super().__init__()
        layers = []
        in_features = num_features

        for hidden_size in architecture:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(get_activation(activation_name))
            in_features = hidden_size

        layers.append(nn.Linear(in_features, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def reset_dnn_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_dnn_fixed_epochs(X_train_np, y_train_real, epochs_to_train, seed_offset=0):
    """
    Trains one DNN with fixed hyperparameters and fixed number of epochs.
    Scaling is fitted only on the current training subset.
    """
    reset_dnn_seed(SEED + seed_offset)

    input_scaler = StandardScaler()
    X_train_scaled = input_scaler.fit_transform(X_train_np)

    max_y = np.max(y_train_real)
    y_train_scaled = (y_train_real / max_y) * 10.0

    X_train = torch.tensor(X_train_scaled, dtype=torch.float32).to(DEVICE)
    y_train = torch.tensor(y_train_scaled.reshape(-1, 1), dtype=torch.float32).to(DEVICE)

    model = DNNRegressor(
        num_features=X_train_np.shape[1],
        architecture=ARCHITECTURE,
        activation_name=ACTIVATION_NAME,
    ).to(DEVICE)

    criterion = RelativeSquaredError()
    optimizer = get_optimizer(OPTIMIZER_NAME, model, LEARNING_RATE, WEIGHT_DECAY)

    loss_history = []

    for epoch in range(1, epochs_to_train + 1):
        model.train()
        optimizer.zero_grad()
        y_pred_train_scaled = model(X_train)
        loss = criterion(y_pred_train_scaled, y_train)
        loss.backward()
        optimizer.step()

        if epoch % DNN_EVAL_STEP == 0 or epoch == 1:
            loss_history.append({"epoch": epoch, "train_loss": float(loss.item())})

    return model, input_scaler, max_y, loss_history


def predict_dnn(model, input_scaler, max_y, X_np):
    X_scaled = input_scaler.transform(X_np)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)

    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_tensor).cpu().numpy().reshape(-1)

    return (y_pred_scaled / 10.0) * max_y


def evaluate_tree_model(model_name, build_model_fn, X_train, y_train, X_test, y_test):
    if USE_INPUT_SCALER_FOR_TREES:
        scaler = StandardScaler()
        X_train_used = scaler.fit_transform(X_train)
        X_test_used = scaler.transform(X_test)
    else:
        X_train_used = X_train.copy()
        X_test_used = X_test.copy()

    model = build_model_fn()

    start_time = time.time()
    model.fit(X_train_used, y_train)
    training_time = time.time() - start_time

    y_pred_test = model.predict(X_test_used)
    test_bias, test_cov, test_r2, test_ratios = compute_bias_cov_r2(y_test, y_pred_test)

    return {
        "model": model_name,
        "test_bias": test_bias,
        "test_cov": test_cov,
        "test_r2": test_r2,
        "test_ratio_min": float(np.min(test_ratios)),
        "test_ratio_max": float(np.max(test_ratios)),
        "training_time_s": training_time,
    }


def evaluate_dnn_model(X_train, y_train, X_test, y_test, seed_offset=0):
    start_time = time.time()
    model, scaler, max_y, _ = train_dnn_fixed_epochs(
        X_train,
        y_train,
        epochs_to_train=DNN_EPOCHS,
        seed_offset=seed_offset,
    )
    training_time = time.time() - start_time

    y_pred_test = predict_dnn(model, scaler, max_y, X_test)
    test_bias, test_cov, test_r2, test_ratios = compute_bias_cov_r2(y_test, y_pred_test)

    return {
        "model": "DNN",
        "test_bias": test_bias,
        "test_cov": test_cov,
        "test_r2": test_r2,
        "test_ratio_min": float(np.min(test_ratios)),
        "test_ratio_max": float(np.max(test_ratios)),
        "training_time_s": training_time,
    }


def main():
    # =============================================================================
    # 5. LOAD AND FILTER THE DATASET
    # =============================================================================
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}. "
            "Download the Study 1 dataset from the approved data repository "
            "and save it as data/study_01_data.xlsx."
        )

    df = pd.read_excel(DATA_FILE)

    missing_input_columns = [col for col in INPUT_COLUMNS if col not in df.columns]
    if missing_input_columns:
        raise ValueError(f"Missing input columns in dataset: {missing_input_columns}")

    if OUTPUT_COLUMN not in df.columns:
        raise ValueError(f"Output column not found in dataset: {OUTPUT_COLUMN}")

    if USE_DATA_FILTERING:
        original_len = len(df)

        df, tf_cutoff = remove_sparse_bins(df, FILTER_COLUMN_1, bins=NUM_BINS, count_threshold=COUNT_THRESHOLD)
        df, tw_cutoff = remove_sparse_bins(df, FILTER_COLUMN_2, bins=NUM_BINS, count_threshold=COUNT_THRESHOLD)
        df, output_cutoff = remove_sparse_bins(df, OUTPUT_COLUMN, bins=NUM_BINS, count_threshold=COUNT_THRESHOLD)

        print("====================================================")
        print("DATA FILTERING")
        print("====================================================")
        print(f"Original data length: {original_len}")
        print(f"Filtered data length: {len(df)}")
        print(f"{FILTER_COLUMN_1} cutoff: {tf_cutoff}")
        print(f"{FILTER_COLUMN_2} cutoff: {tw_cutoff}")
        print(f"{OUTPUT_COLUMN} cutoff: {output_cutoff}")

    X_np = df[INPUT_COLUMNS].to_numpy(dtype=float)
    y_np = df[OUTPUT_COLUMN].to_numpy(dtype=float)


    # =============================================================================
    # 6. FIXED 80:20 TRAIN/TEST SPLIT
    # =============================================================================
    X_trainval_np, X_test_np, y_trainval_np, y_test_np = train_test_split(
        X_np,
        y_np,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    print("\n====================================================")
    print("DATASET-SIZE SENSITIVITY ANALYSIS")
    print("====================================================")
    print(f"Total filtered samples:      {len(y_np)}")
    print(f"Train/validation pool size:  {len(y_trainval_np)}")
    print(f"Independent test set size:   {len(y_test_np)}")
    print(f"Device for DNN:              {DEVICE}")
    print("No hyperparameter tuning is performed.")

    # Nested sampling: larger subsets contain the smaller subsets.
    subset_rng = np.random.default_rng(SEED)
    trainval_order = subset_rng.permutation(len(y_trainval_np))

    # Resolve requested sizes.
    resolved_training_sizes = []
    for size in TRAINING_SIZES:
        if isinstance(size, str) and size.lower() == "full":
            resolved_training_sizes.append(len(y_trainval_np))
        elif isinstance(size, int):
            if size <= len(y_trainval_np):
                resolved_training_sizes.append(size)
            else:
                print(f"Skipping training size {size}: larger than training pool ({len(y_trainval_np)}).")
        else:
            raise ValueError(f"Invalid training size: {size}")

    # Remove duplicates while preserving order.
    seen = set()
    resolved_training_sizes = [x for x in resolved_training_sizes if not (x in seen or seen.add(x))]


    # =============================================================================
    # 7. RUN ANALYSIS
    # =============================================================================
    all_results = []
    analysis_start = time.time()

    for size_idx, training_size in enumerate(resolved_training_sizes):
        print("\n====================================================")
        print(f"TRAINING SIZE: {training_size}")
        print("====================================================")

        subset_idx = trainval_order[:training_size]
        X_subset = X_trainval_np[subset_idx]
        y_subset = y_trainval_np[subset_idx]

        # Random Forest
        print("Running Random Forest...")
        rf_result = evaluate_tree_model(
            model_name="Random Forest",
            build_model_fn=build_random_forest_model,
            X_train=X_subset,
            y_train=y_subset,
            X_test=X_test_np,
            y_test=y_test_np,
        )
        rf_result["training_size"] = training_size
        all_results.append(rf_result)
        print(f"  RF Test CoV={rf_result['test_cov']:.6f}, R²={rf_result['test_r2']:.6f}")

        # XGBoost
        print("Running XGBoost...")
        xgb_result = evaluate_tree_model(
            model_name="XGBoost",
            build_model_fn=build_xgboost_model,
            X_train=X_subset,
            y_train=y_subset,
            X_test=X_test_np,
            y_test=y_test_np,
        )
        xgb_result["training_size"] = training_size
        all_results.append(xgb_result)
        print(f"  XGB Test CoV={xgb_result['test_cov']:.6f}, R²={xgb_result['test_r2']:.6f}")

        # DNN
        print("Running DNN...")
        dnn_result = evaluate_dnn_model(
            X_train=X_subset,
            y_train=y_subset,
            X_test=X_test_np,
            y_test=y_test_np,
            seed_offset=size_idx,  # Original experiment: initialization varies by subset size.
        )
        dnn_result["training_size"] = training_size
        all_results.append(dnn_result)
        print(f"  DNN Test CoV={dnn_result['test_cov']:.6f}, R²={dnn_result['test_r2']:.6f}")

    analysis_time = time.time() - analysis_start

    results_df = pd.DataFrame(all_results)
    results_df = results_df[
        [
            "training_size",
            "model",
            "test_bias",
            "test_cov",
            "test_r2",
            "test_ratio_min",
            "test_ratio_max",
            "training_time_s",
        ]
    ]
    results_df.to_csv(RESULTS_CSV_PATH, index=False)


    # =============================================================================
    # 8. PLOT TEST CoV VERSUS TRAINING SIZE
    # =============================================================================
    plt.figure(figsize=(8, 5))

    for model_name in ["Random Forest", "XGBoost", "DNN"]:
        model_df = results_df[results_df["model"] == model_name].sort_values("training_size")
        plt.plot(
            model_df["training_size"],
            model_df["test_cov"],
            marker="o",
            linewidth=2,
            label=model_name,
        )

    plt.xlabel("Training dataset size")
    plt.ylabel("Test CoV")
    plt.title("Influence of Training Dataset Size on Test CoV")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_COV_PATH, dpi=300)
    plt.close()


    # =============================================================================
    # 9. PLOT TEST R² VERSUS TRAINING SIZE
    # =============================================================================
    plt.figure(figsize=(8, 5))

    for model_name in ["Random Forest", "XGBoost", "DNN"]:
        model_df = results_df[results_df["model"] == model_name].sort_values("training_size")
        plt.plot(
            model_df["training_size"],
            model_df["test_r2"],
            marker="o",
            linewidth=2,
            label=model_name,
        )

    plt.xlabel("Training dataset size")
    plt.ylabel("Test R²")
    plt.title("Influence of Training Dataset Size on Test R²")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_R2_PATH, dpi=300)
    plt.close()


    # =============================================================================
    # 10. SAVE SUMMARY
    # =============================================================================
    with open(SUMMARY_TXT_PATH, "w", encoding="utf-8") as f:
        f.write("DATASET-SIZE SENSITIVITY ANALYSIS\n")
        f.write("=================================\n")
        f.write("This analysis uses fixed final hyperparameters for all models.\n")
        f.write("No hyperparameter tuning is performed.\n")
        f.write("The same independent test set is used for all training sizes and all models.\n")
        f.write("Training subsets are sampled from the same 80% train/validation pool.\n\n")

        f.write("DATA PROTOCOL\n")
        f.write("=================================\n")
        f.write(f"Total filtered samples: {len(y_np)}\n")
        f.write(f"Train/validation samples: {len(y_trainval_np)}\n")
        f.write(f"Test samples: {len(y_test_np)}\n")
        f.write(f"Training sizes: {resolved_training_sizes}\n")
        f.write(f"DNN epochs per size: {DNN_EPOCHS}\n")
        f.write(f"Total analysis time [s]: {analysis_time:.6f}\n\n")

        f.write("FINAL HYPERPARAMETERS\n")
        f.write("=================================\n")
        f.write(f"Random Forest: {BEST_RF_PARAMS}\n")
        f.write(f"XGBoost: {BEST_XGB_PARAMS}\n")
        f.write(
            "DNN: "
            f"architecture={ARCHITECTURE}, activation={ACTIVATION_NAME}, "
            f"optimizer={OPTIMIZER_NAME}, learning_rate={LEARNING_RATE}, "
            f"weight_decay={WEIGHT_DECAY}, epochs={DNN_EPOCHS}\n\n"
        )

        f.write("RESULTS\n")
        f.write("=================================\n")
        f.write(results_df.to_string(index=False))


    print("\n====================================================")
    print("DONE")
    print("====================================================")
    print(f"Results saved to: {RESULTS_CSV_PATH}")
    print(f"CoV plot saved to: {PLOT_COV_PATH}")
    print(f"R² plot saved to:  {PLOT_R2_PATH}")
    print(f"Summary saved to:  {SUMMARY_TXT_PATH}")


if __name__ == "__main__":
    main()
