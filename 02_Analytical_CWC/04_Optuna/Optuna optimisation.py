"""Study 2: Optuna-based selection of model families and hyperparameters.

Five-fold cross-validation is performed within an 80% training partition,
followed by evaluation of the selected model on a 20% holdout test set.
The original search space and training procedures are retained.
"""

# =============================================================================
# OPTUNA-BASED MODEL-FAMILY SELECTION AND HYPERPARAMETER OPTIMIZATION
# 5-FOLD CV + SEPARATE TEST SET
# Selection rule: lowest CoV first, highest R² second
# =============================================================================

import random
import ast
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import optuna

from optuna.visualization.matplotlib import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice,
    plot_edf,
)

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor,
    BaggingRegressor,
)

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


# =============================================================================
# 0. REPRODUCIBILITY
# =============================================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# =============================================================================
# 1. FILES AND FOLDERS
# =============================================================================

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = REPOSITORY_ROOT / "data" / "study_02_data.xlsx"

# Use separate output directories for each time budget.



# =============================================================================
# 2. EXPERIMENT CONFIGURATION
# =============================================================================

TIMEOUT_SECONDS = 900

# Output paths are tied to the run budget to avoid overwriting other runs.
RESULTS_DIR = Path(__file__).resolve().parent / "outputs" / f"{TIMEOUT_SECONDS}s"
RESULTS_DIR.mkdir(exist_ok=True)

RESULTS_CSV_PATH = RESULTS_DIR / "optuna_results.csv"
ALL_TRIALS_PATH = RESULTS_DIR / "all_optuna_trials.csv"
BEST_PARAMS_PATH = RESULTS_DIR / "best_params.txt"

FINAL_PLOT_PATH = RESULTS_DIR / "best_optuna_model_all_data_test_plot.png"
CONVERGENCE_PLOT_PATH = RESULTS_DIR / "optuna_convergence_cov_r2.png"

OPT_HISTORY_PATH = RESULTS_DIR / "optuna_optimization_history.png"
PARAM_IMPORTANCE_PATH = RESULTS_DIR / "optuna_param_importance.png"
PARALLEL_COORD_PATH = RESULTS_DIR / "optuna_parallel_coordinate.png"
SLICE_PLOT_PATH = RESULTS_DIR / "optuna_slice_plot.png"
EDF_PLOT_PATH = RESULTS_DIR / "optuna_edf_plot.png"

TEST_SIZE = 0.20
N_SPLITS = 5

MIN_EPOCHS = 1500
MAX_EPOCHS = 5000
EVAL_STEP = 100
START_EVAL_EPOCH = 500
EARLY_STOP_PATIENCE = 1000

USE_DATA_FILTERING = True

NUM_BINS = 30
COUNT_THRESHOLD = 5

FILTER_COLUMN_1 = "t_fc"
FILTER_COLUMN_2 = "t_wc"

OUTPUT_COLUMN = "F_c_wc_Rd_kN"

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


# =============================================================================
# 3. MODEL SEARCH SPACES
# =============================================================================

# -----------------------------
# DNN
# -----------------------------

ARCHITECTURES = [
    (64, 32),
    (128, 64),
    (256, 128),
    (64, 32, 16),
    (128, 64, 32),
    (256, 128, 64),
    (64, 32, 16, 8),
    (128, 64, 32, 16),
    (256, 128, 64, 32),
    (128, 64, 32, 16, 8),
    (256, 128, 64, 32, 16),
]

LEARNING_RATES = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]

ACTIVATION_FUNCTIONS = [
    "ReLU",
    "LeakyReLU",
    "ELU",
    "GELU",
    "SiLU",
    "Tanh",
]

OPTIMIZERS = [
    "SGD",
    "SGD_Momentum",
    "Adagrad",
    "RMSprop",
    "Adam",
    "AdamW",
]

WEIGHT_DECAYS = [0.0, 1e-5, 1e-4, 1e-3, 1e-2]


# -----------------------------
# Random Forest
# -----------------------------

RF_N_ESTIMATORS = [100, 200, 300, 400, 500]
RF_MAX_DEPTH = [4, 6, 8, 10, 12, 15, 20]
RF_MIN_SAMPLES_SPLIT = [2, 5, 10, 15, 20]
RF_MIN_SAMPLES_LEAF = [1, 2, 4, 6, 8, 10]


# -----------------------------
# Extra Trees
# -----------------------------

ET_N_ESTIMATORS = [100, 200, 300, 400, 500]
ET_MAX_DEPTH = [4, 6, 8, 10, 12, 15, 20]
ET_MIN_SAMPLES_SPLIT = [2, 5, 10, 15, 20]
ET_MIN_SAMPLES_LEAF = [1, 2, 4, 6, 8, 10]


# -----------------------------
# Gradient Boosting
# -----------------------------

GBR_N_ESTIMATORS = [100, 200, 300, 400, 500]
GBR_LEARNING_RATES = [0.01, 0.05, 0.1]
GBR_MAX_DEPTH = [2, 3, 4, 5, 6, 7, 8]
GBR_SUBSAMPLE = [0.7, 0.85, 1.0]


# -----------------------------
# AdaBoost
# -----------------------------

ADA_N_ESTIMATORS = [50, 100, 150, 200, 250, 300]
ADA_LEARNING_RATES = [0.01, 0.05, 0.1, 0.5]


# -----------------------------
# Bagging
# -----------------------------

BAG_N_ESTIMATORS = [50, 100, 150, 200, 250, 300]
BAG_MAX_SAMPLES = [0.6, 0.8, 1.0]
BAG_MAX_FEATURES = [0.6, 0.8, 1.0]


# -----------------------------
# XGBoost
# -----------------------------

XGB_N_ESTIMATORS = [100, 200, 300, 400, 500, 600]
XGB_MAX_DEPTH = [2, 3, 4, 5, 6, 8, 10]
XGB_LEARNING_RATES = [0.01, 0.05, 0.1]
XGB_SUBSAMPLE = [0.6, 0.8, 1.0]
XGB_COLSAMPLE = [0.6, 0.8, 1.0]
XGB_REG_ALPHA = [0.0, 1e-4, 1e-3, 1e-2]
XGB_REG_LAMBDA = [0.1, 1.0, 10.0]


# -----------------------------
# LightGBM
# -----------------------------

LGBM_N_ESTIMATORS = [100, 200, 300, 400, 500, 600]
LGBM_NUM_LEAVES = [7, 15, 31, 63]
LGBM_LEARNING_RATES = [0.01, 0.05, 0.1]
LGBM_SUBSAMPLE = [0.6, 0.8, 1.0]
LGBM_COLSAMPLE = [0.6, 0.8, 1.0]


# -----------------------------
# CatBoost
# -----------------------------

CAT_ITERATIONS = [100, 200, 300, 400, 500, 600]
CAT_DEPTH = [2, 3, 4, 5, 6, 7, 8]
CAT_LEARNING_RATES = [0.01, 0.05, 0.1]
CAT_L2_LEAF_REG = [1, 3, 5, 10]


# =============================================================================
# 4. HELPER FUNCTIONS
# =============================================================================

def compute_metrics(y_true, y_pred):
    ratios = y_pred / y_true
    bias = np.mean(ratios)
    cov = np.std(ratios) / abs(bias)
    r2 = r2_score(y_true, y_pred)
    return bias, cov, r2


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
    raise ValueError(f"Unknown activation: {name}")


def get_optimizer(name, model, learning_rate, weight_decay):
    if name == "SGD":
        return optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if name == "SGD_Momentum":
        return optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay
        )

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


class RegressionModel(nn.Module):
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


def train_dnn_fold(
    X_train_np,
    X_val_np,
    y_train_real,
    y_val_real,
    num_features,
    architecture,
    activation_name,
    optimizer_name,
    learning_rate,
    weight_decay,
):
    input_scaler = StandardScaler()

    X_train_scaled = input_scaler.fit_transform(X_train_np)
    X_val_scaled = input_scaler.transform(X_val_np)

    max_y = np.max(y_train_real)
    y_train_scaled = (y_train_real / max_y) * 10.0

    X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_val = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_train = torch.tensor(y_train_scaled.reshape(-1, 1), dtype=torch.float32)

    model = RegressionModel(num_features, architecture, activation_name)
    criterion = RelativeSquaredError()

    optimizer = get_optimizer(
        optimizer_name,
        model,
        learning_rate,
        weight_decay
    )

    best_cov = float("inf")
    best_bias = None
    best_r2 = None
    best_epoch = None
    early_stop_counter = 0

    for epoch in range(MAX_EPOCHS):
        model.train()

        optimizer.zero_grad()
        y_pred_train_scaled = model(X_train)
        loss = criterion(y_pred_train_scaled, y_train)
        loss.backward()
        optimizer.step()

        if epoch >= START_EVAL_EPOCH and epoch % EVAL_STEP == 0:
            model.eval()

            with torch.no_grad():
                y_pred_val_scaled = model(X_val).cpu().numpy().flatten()
                y_pred_val_real = (y_pred_val_scaled / 10.0) * max_y

            bias_val, cov_val, r2_val = compute_metrics(y_val_real, y_pred_val_real)

            if np.isnan(cov_val) or np.isinf(cov_val):
                continue

            if cov_val < best_cov:
                best_cov = cov_val
                best_bias = bias_val
                best_r2 = r2_val
                best_epoch = epoch
                early_stop_counter = 0
            else:
                early_stop_counter += EVAL_STEP

        if epoch >= MIN_EPOCHS and early_stop_counter >= EARLY_STOP_PATIENCE:
            break

    return best_cov, best_bias, best_r2, best_epoch


def build_sklearn_model(trial, model_type):
    if model_type == "RandomForest":
        return RandomForestRegressor(
            n_estimators=trial.suggest_categorical("rf_n_estimators", RF_N_ESTIMATORS),
            max_depth=trial.suggest_categorical("rf_max_depth", RF_MAX_DEPTH),
            min_samples_split=trial.suggest_categorical("rf_min_samples_split", RF_MIN_SAMPLES_SPLIT),
            min_samples_leaf=trial.suggest_categorical("rf_min_samples_leaf", RF_MIN_SAMPLES_LEAF),
            random_state=SEED,
            n_jobs=-1
        )

    if model_type == "ExtraTrees":
        return ExtraTreesRegressor(
            n_estimators=trial.suggest_categorical("et_n_estimators", ET_N_ESTIMATORS),
            max_depth=trial.suggest_categorical("et_max_depth", ET_MAX_DEPTH),
            min_samples_split=trial.suggest_categorical("et_min_samples_split", ET_MIN_SAMPLES_SPLIT),
            min_samples_leaf=trial.suggest_categorical("et_min_samples_leaf", ET_MIN_SAMPLES_LEAF),
            random_state=SEED,
            n_jobs=-1
        )

    if model_type == "GradientBoosting":
        return GradientBoostingRegressor(
            n_estimators=trial.suggest_categorical("gbr_n_estimators", GBR_N_ESTIMATORS),
            learning_rate=trial.suggest_categorical("gbr_learning_rate", GBR_LEARNING_RATES),
            max_depth=trial.suggest_categorical("gbr_max_depth", GBR_MAX_DEPTH),
            subsample=trial.suggest_categorical("gbr_subsample", GBR_SUBSAMPLE),
            random_state=SEED
        )

    if model_type == "AdaBoost":
        return AdaBoostRegressor(
            n_estimators=trial.suggest_categorical("ada_n_estimators", ADA_N_ESTIMATORS),
            learning_rate=trial.suggest_categorical("ada_learning_rate", ADA_LEARNING_RATES),
            random_state=SEED
        )

    if model_type == "Bagging":
        return BaggingRegressor(
            n_estimators=trial.suggest_categorical("bag_n_estimators", BAG_N_ESTIMATORS),
            max_samples=trial.suggest_categorical("bag_max_samples", BAG_MAX_SAMPLES),
            max_features=trial.suggest_categorical("bag_max_features", BAG_MAX_FEATURES),
            random_state=SEED,
            n_jobs=-1
        )

    if model_type == "XGBoost":
        return XGBRegressor(
            n_estimators=trial.suggest_categorical("xgb_n_estimators", XGB_N_ESTIMATORS),
            max_depth=trial.suggest_categorical("xgb_max_depth", XGB_MAX_DEPTH),
            learning_rate=trial.suggest_categorical("xgb_learning_rate", XGB_LEARNING_RATES),
            subsample=trial.suggest_categorical("xgb_subsample", XGB_SUBSAMPLE),
            colsample_bytree=trial.suggest_categorical("xgb_colsample", XGB_COLSAMPLE),
            reg_alpha=trial.suggest_categorical("xgb_reg_alpha", XGB_REG_ALPHA),
            reg_lambda=trial.suggest_categorical("xgb_reg_lambda", XGB_REG_LAMBDA),
            random_state=SEED,
            n_jobs=-1
        )

    if model_type == "LightGBM":
        return LGBMRegressor(
            n_estimators=trial.suggest_categorical("lgbm_n_estimators", LGBM_N_ESTIMATORS),
            num_leaves=trial.suggest_categorical("lgbm_num_leaves", LGBM_NUM_LEAVES),
            learning_rate=trial.suggest_categorical("lgbm_learning_rate", LGBM_LEARNING_RATES),
            subsample=trial.suggest_categorical("lgbm_subsample", LGBM_SUBSAMPLE),
            colsample_bytree=trial.suggest_categorical("lgbm_colsample", LGBM_COLSAMPLE),
            random_state=SEED,
            n_jobs=-1,
            verbose=-1
        )

    if model_type == "CatBoost":
        return CatBoostRegressor(
            iterations=trial.suggest_categorical("cat_iterations", CAT_ITERATIONS),
            depth=trial.suggest_categorical("cat_depth", CAT_DEPTH),
            learning_rate=trial.suggest_categorical("cat_learning_rate", CAT_LEARNING_RATES),
            l2_leaf_reg=trial.suggest_categorical("cat_l2_leaf_reg", CAT_L2_LEAF_REG),
            random_seed=SEED,
            verbose=False
        )

    raise ValueError(f"Unknown model type: {model_type}")


def save_optuna_plot(plot_function, study, path, plot_name):
    try:
        ax = plot_function(study)
        fig = ax.figure
        fig.tight_layout()
        fig.savefig(path, dpi=300)
        plt.close(fig)
        print(f"{plot_name} saved to: {path}")
    except Exception as e:
        print(f"Could not create {plot_name}: {e}")


# =============================================================================
# 5. READ AND FILTER DATA
# =============================================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

df = pd.read_excel(DATA_FILE)

missing_input_columns = [col for col in INPUT_COLUMNS if col not in df.columns]

if missing_input_columns:
    raise ValueError(f"Missing input columns in Excel file: {missing_input_columns}")

if OUTPUT_COLUMN not in df.columns:
    raise ValueError(f"Output column not found in Excel file: {OUTPUT_COLUMN}")

output_column = OUTPUT_COLUMN

if USE_DATA_FILTERING:
    df_full_length = len(df)

    counts_1, bin_edges_1 = np.histogram(df[FILTER_COLUMN_1], bins=NUM_BINS)
    low_count_bins_1 = np.where(counts_1 < COUNT_THRESHOLD)[0]

    if len(low_count_bins_1) > 0:
        cutoff_1 = bin_edges_1[low_count_bins_1[0] + 1]
        df = df[df[FILTER_COLUMN_1] <= cutoff_1].copy()
    else:
        cutoff_1 = None

    counts_2, bin_edges_2 = np.histogram(df[FILTER_COLUMN_2], bins=NUM_BINS)
    low_count_bins_2 = np.where(counts_2 < COUNT_THRESHOLD)[0]

    if len(low_count_bins_2) > 0:
        cutoff_2 = bin_edges_2[low_count_bins_2[0] + 1]
        df = df[df[FILTER_COLUMN_2] <= cutoff_2].copy()
    else:
        cutoff_2 = None

    counts_out, bin_edges_out = np.histogram(df[output_column], bins=NUM_BINS)
    low_count_bins_out = np.where(counts_out < COUNT_THRESHOLD)[0]

    if len(low_count_bins_out) > 0:
        output_cutoff = bin_edges_out[low_count_bins_out[0] + 1]
        df = df[df[output_column] <= output_cutoff].copy()
    else:
        output_cutoff = None

    print(f"Filtered data length: {len(df)} from {df_full_length}")
    print(f"{FILTER_COLUMN_1} cutoff: {cutoff_1}")
    print(f"{FILTER_COLUMN_2} cutoff: {cutoff_2}")
    print(f"{output_column} cutoff: {output_cutoff}")


X_df = df[INPUT_COLUMNS].copy()
y_array = df[output_column].to_numpy(dtype=float)

X_np = X_df.to_numpy(dtype=float)
y_np = y_array.copy()

X_trainval_np, X_test_np, y_trainval_np, y_test_np = train_test_split(
    X_np,
    y_np,
    test_size=TEST_SIZE,
    random_state=SEED,
    shuffle=True
)


# =============================================================================
# 6. DEFINE AVAILABLE MODELS
# =============================================================================

MODEL_TYPES = [
    "DNN",
    "RandomForest",
    "ExtraTrees",
    "GradientBoosting",
    "AdaBoost",
    "Bagging",
]

if HAS_XGBOOST:
    MODEL_TYPES.append("XGBoost")

if HAS_LIGHTGBM:
    MODEL_TYPES.append("LightGBM")

if HAS_CATBOOST:
    MODEL_TYPES.append("CatBoost")

print("====================================================")
print("DATA VARIABLES")
print("====================================================")
print(f"Input columns           = {INPUT_COLUMNS}")
print(f"Output column           = {OUTPUT_COLUMN}")
print(f"Input features          = {X_np.shape[1]}")
print(f"Train/validation pool   = {len(X_trainval_np)}")
print(f"Independent test set    = {len(X_test_np)}")
print(f"Cross-validation folds  = {N_SPLITS}")
print(f"Optuna timeout          = {TIMEOUT_SECONDS} seconds")

print("\nModels included:")
for model_name in MODEL_TYPES:
    print(f" - {model_name}")


# =============================================================================
# 7. OPTUNA OBJECTIVE WITH 5-FOLD CV
# =============================================================================

trial_records = []


def objective(trial):
    model_type = trial.suggest_categorical("model_type", MODEL_TYPES)

    kf = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED
    )

    fold_covs = []
    fold_biases = []
    fold_r2s = []
    fold_epochs = []

    start_time = time.time()

    for fold_id, (train_idx, val_idx) in enumerate(kf.split(X_trainval_np), start=1):
        X_fold_train = X_trainval_np[train_idx]
        X_fold_val = X_trainval_np[val_idx]

        y_fold_train = y_trainval_np[train_idx]
        y_fold_val = y_trainval_np[val_idx]

        if model_type == "DNN":
            architecture = trial.suggest_categorical(
                "architecture",
                [str(a) for a in ARCHITECTURES]
            )

            architecture = ast.literal_eval(architecture)

            learning_rate = trial.suggest_categorical(
                "learning_rate",
                LEARNING_RATES
            )

            activation_name = trial.suggest_categorical(
                "activation",
                ACTIVATION_FUNCTIONS
            )

            optimizer_name = trial.suggest_categorical(
                "optimizer",
                OPTIMIZERS
            )

            weight_decay = trial.suggest_categorical(
                "weight_decay",
                WEIGHT_DECAYS
            )

            cov_val, bias_val, r2_val, best_epoch = train_dnn_fold(
                X_train_np=X_fold_train,
                X_val_np=X_fold_val,
                y_train_real=y_fold_train,
                y_val_real=y_fold_val,
                num_features=X_np.shape[1],
                architecture=architecture,
                activation_name=activation_name,
                optimizer_name=optimizer_name,
                learning_rate=learning_rate,
                weight_decay=weight_decay
            )

            fold_epochs.append(best_epoch)

        else:
            input_scaler = StandardScaler()

            X_fold_train_scaled = input_scaler.fit_transform(X_fold_train)
            X_fold_val_scaled = input_scaler.transform(X_fold_val)

            model = build_sklearn_model(trial, model_type)
            model.fit(X_fold_train_scaled, y_fold_train)

            y_pred_val = model.predict(X_fold_val_scaled)

            bias_val, cov_val, r2_val = compute_metrics(y_fold_val, y_pred_val)

            fold_epochs.append(None)

        if np.isnan(cov_val) or np.isinf(cov_val):
            return float("inf")

        fold_covs.append(cov_val)
        fold_biases.append(bias_val)
        fold_r2s.append(r2_val)

        trial.report(float(np.mean(fold_covs)), fold_id)

        if trial.should_prune():
            raise optuna.TrialPruned()

    mean_cov = float(np.mean(fold_covs))
    mean_bias = float(np.mean(fold_biases))
    mean_r2 = float(np.mean(fold_r2s))
    std_cov = float(np.std(fold_covs))

    training_time = time.time() - start_time

    record = {
        "trial": trial.number,
        "model_type": model_type,
        "mean_cv_cov": mean_cov,
        "std_cv_cov": std_cov,
        "mean_cv_bias": mean_bias,
        "mean_cv_r2": mean_r2,
        "training_time_s": training_time,
        "best_epoch_mean": np.nanmean(fold_epochs)
        if any(epoch is not None for epoch in fold_epochs)
        else None,
    }

    for key, value in trial.params.items():
        record[key] = value

    trial_records.append(record)

    return mean_cov


# =============================================================================
# 8. RUN OPTUNA SEARCH
# =============================================================================

study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=SEED),
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=2
    )
)

study.optimize(
    objective,
    timeout=TIMEOUT_SECONDS
)


# =============================================================================
# 9. SAVE TRIAL RESULTS
# =============================================================================

results_df = pd.DataFrame(trial_records)

if results_df.empty:
    raise RuntimeError("No completed Optuna trials. Increase timeout or check model training.")

results_df = results_df.sort_values(
    by=["mean_cv_cov", "mean_cv_r2"],
    ascending=[True, False]
).reset_index(drop=True)

results_df.to_csv(RESULTS_CSV_PATH, index=False)

all_trials_df = study.trials_dataframe()
all_trials_df.to_csv(ALL_TRIALS_PATH, index=False)


# =============================================================================
# 10. SELECT BEST MODEL BY CoV FIRST, R² SECOND
# =============================================================================

best_selected = results_df.iloc[0]
best_trial_number = int(best_selected["trial"])

best_trial = None

for trial in study.trials:
    if trial.number == best_trial_number:
        best_trial = trial
        break

if best_trial is None:
    raise RuntimeError("Could not find selected Optuna trial.")

best_params = best_trial.params
best_model_type = best_params["model_type"]

print("\n====================================================")
print("BEST MODEL SELECTED BY CoV FIRST, R² SECOND")
print("====================================================")
print(f"Selected trial = {best_trial_number}")
print(f"Selected model = {best_model_type}")
print(f"Mean CV CoV = {best_selected['mean_cv_cov']:.6f}")
print(f"Mean CV R² = {best_selected['mean_cv_r2']:.6f}")
print(f"Mean CV bias = {best_selected['mean_cv_bias']:.6f}")

with open(BEST_PARAMS_PATH, "w") as f:
    f.write("BEST OPTUNA PARAMETERS\n")
    f.write("======================\n\n")
    f.write("Selection rule: lowest mean CV CoV first, highest mean CV R² second.\n\n")
    f.write(f"Selected trial = {best_trial_number}\n")
    f.write(f"Selected model = {best_model_type}\n")
    f.write(f"Mean CV CoV = {best_selected['mean_cv_cov']:.6f}\n")
    f.write(f"Mean CV R² = {best_selected['mean_cv_r2']:.6f}\n")
    f.write(f"Mean CV bias = {best_selected['mean_cv_bias']:.6f}\n\n")
    f.write("Input columns:\n")
    f.write(str(INPUT_COLUMNS))
    f.write("\n\n")
    f.write("Output column:\n")
    f.write(str(OUTPUT_COLUMN))
    f.write("\n\n")
    f.write("Best parameters:\n")
    f.write(str(best_params))
    f.write("\n")


# =============================================================================
# 11. CONVERGENCE PLOT: CoV AND R²
# =============================================================================

completed_df = results_df.copy()
completed_df = completed_df.sort_values("trial").reset_index(drop=True)

completed_df["best_cov_so_far"] = completed_df["mean_cv_cov"].cummin()
completed_df["best_r2_so_far"] = completed_df["mean_cv_r2"].cummax()

fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

axes[0].plot(
    completed_df["trial"],
    completed_df["best_cov_so_far"],
    marker="o",
    linewidth=2
)

axes[0].set_ylabel("Best CoV so far")
axes[0].set_title("Optuna Convergence - CoV")
axes[0].grid(True, linestyle="--", alpha=0.6)

axes[1].plot(
    completed_df["trial"],
    completed_df["best_r2_so_far"],
    marker="o",
    linewidth=2
)

axes[1].set_xlabel("Trial")
axes[1].set_ylabel("Highest observed mean CV R²")
axes[1].set_title("Optuna Trial History - Highest Mean CV R²")
axes[1].grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig(CONVERGENCE_PLOT_PATH, dpi=300)
plt.close()


# =============================================================================
# 12. OPTUNA BUILT-IN VISUALIZATION PLOTS
# =============================================================================

save_optuna_plot(
    plot_optimization_history,
    study,
    OPT_HISTORY_PATH,
    "Optuna optimization history plot"
)

save_optuna_plot(
    plot_param_importances,
    study,
    PARAM_IMPORTANCE_PATH,
    "Optuna parameter importance plot"
)

save_optuna_plot(
    plot_parallel_coordinate,
    study,
    PARALLEL_COORD_PATH,
    "Optuna parallel coordinate plot"
)

save_optuna_plot(
    plot_slice,
    study,
    SLICE_PLOT_PATH,
    "Optuna slice plot"
)

save_optuna_plot(
    plot_edf,
    study,
    EDF_PLOT_PATH,
    "Optuna EDF plot"
)


# =============================================================================
# 13. RETRAIN BEST MODEL ON FULL TRAIN/VALIDATION POOL
# =============================================================================

if best_model_type == "DNN":
    best_architecture = ast.literal_eval(best_params["architecture"])
    best_lr = best_params["learning_rate"]
    best_activation = best_params["activation"]
    best_optimizer_name = best_params["optimizer"]
    best_weight_decay = best_params["weight_decay"]

    X_internal_train, X_internal_val, y_internal_train, y_internal_val = train_test_split(
        X_trainval_np,
        y_trainval_np,
        test_size=0.15,
        random_state=SEED,
        shuffle=True
    )

    final_cov, final_bias, final_r2, final_epoch = train_dnn_fold(
        X_train_np=X_internal_train,
        X_val_np=X_internal_val,
        y_train_real=y_internal_train,
        y_val_real=y_internal_val,
        num_features=X_np.shape[1],
        architecture=best_architecture,
        activation_name=best_activation,
        optimizer_name=best_optimizer_name,
        learning_rate=best_lr,
        weight_decay=best_weight_decay
    )

    final_input_scaler = StandardScaler()

    X_trainval_scaled = final_input_scaler.fit_transform(X_trainval_np)
    X_test_scaled = final_input_scaler.transform(X_test_np)
    X_all_scaled = final_input_scaler.transform(X_np)

    final_max_y = np.max(y_trainval_np)

    y_trainval_scaled = (y_trainval_np / final_max_y) * 10.0

    X_trainval_tensor = torch.tensor(X_trainval_scaled, dtype=torch.float32)
    y_trainval_tensor = torch.tensor(y_trainval_scaled.reshape(-1, 1), dtype=torch.float32)

    final_model = RegressionModel(
        num_features=X_np.shape[1],
        architecture=best_architecture,
        activation_name=best_activation
    )

    criterion = RelativeSquaredError()

    optimizer = get_optimizer(
        best_optimizer_name,
        final_model,
        best_lr,
        best_weight_decay
    )

    epochs_to_train = final_epoch if final_epoch is not None else MIN_EPOCHS
    epochs_to_train = max(int(epochs_to_train), MIN_EPOCHS)

    for epoch in range(epochs_to_train):
        final_model.train()

        optimizer.zero_grad()
        y_pred_train_scaled = final_model(X_trainval_tensor)
        loss = criterion(y_pred_train_scaled, y_trainval_tensor)
        loss.backward()
        optimizer.step()

    final_model.eval()

    with torch.no_grad():
        y_pred_test_scaled = final_model(
            torch.tensor(X_test_scaled, dtype=torch.float32)
        ).cpu().numpy().flatten()

        y_pred_all_scaled = final_model(
            torch.tensor(X_all_scaled, dtype=torch.float32)
        ).cpu().numpy().flatten()

    y_pred_test = (y_pred_test_scaled / 10.0) * final_max_y
    y_pred_all = (y_pred_all_scaled / 10.0) * final_max_y

else:
    final_input_scaler = StandardScaler()

    X_trainval_scaled = final_input_scaler.fit_transform(X_trainval_np)
    X_test_scaled = final_input_scaler.transform(X_test_np)
    X_all_scaled = final_input_scaler.transform(X_np)

    fixed_trial = optuna.trial.FixedTrial(best_params)

    final_model = build_sklearn_model(
        fixed_trial,
        best_model_type
    )

    final_model.fit(X_trainval_scaled, y_trainval_np)

    y_pred_test = final_model.predict(X_test_scaled)
    y_pred_all = final_model.predict(X_all_scaled)


# =============================================================================
# 14. FINAL TEST EVALUATION
# =============================================================================

bias_test, cov_test, r2_test = compute_metrics(y_test_np, y_pred_test)
bias_all, cov_all, r2_all = compute_metrics(y_np, y_pred_all)

with open(BEST_PARAMS_PATH, "a") as f:
    f.write("\nFINAL TEST METRICS\n")
    f.write("==================\n")
    f.write(f"Test bias = {bias_test:.6f}\n")
    f.write(f"Test CoV = {cov_test:.6f}\n")
    f.write(f"Test R² = {r2_test:.6f}\n")

print("\n====================================================")
print("FINAL OPTUNA MODEL METRICS")
print("====================================================")
print(f"Best model type = {best_model_type}")
print(f"Best parameters = {best_params}")
print(f"All data:  bias = {bias_all:.4f}, CoV = {cov_all:.4f}, R² = {r2_all:.6f}")
print(f"Test data: bias = {bias_test:.4f}, CoV = {cov_test:.4f}, R² = {r2_test:.6f}")


# =============================================================================
# 15. FINAL PLOT: ALL DATA + TEST DATA
# =============================================================================

plt.figure(figsize=(7, 7))

combined_values = np.concatenate((y_pred_all, y_np, y_pred_test, y_test_np))

min_val = np.min(combined_values)
max_val = np.max(combined_values)

line_x = np.linspace(min_val, max_val, 200)

plt.scatter(
    y_pred_all,
    y_np,
    alpha=0.5,
    color="blue",
    edgecolors="black",
    s=30,
    label=f"All data\nn={len(y_np)}\nb={bias_all:.4f}\nCoV={cov_all:.4f}"
)

plt.scatter(
    y_pred_test,
    y_test_np,
    alpha=0.7,
    color="orange",
    edgecolors="black",
    s=35,
    label=f"Test data\nn={len(y_test_np)}\nb={bias_test:.4f}\nCoV={cov_test:.4f}"
)

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    "k--",
    linewidth=2,
    label="Ideal fit"
)

plt.plot(
    line_x,
    1.1 * line_x,
    "r--",
    linewidth=2,
    label="+/-10% error"
)

plt.plot(
    line_x,
    0.9 * line_x,
    "r--",
    linewidth=2
)

plt.xlim(0, max_val * 1.02)
plt.ylim(0, max_val * 1.02)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Best Optuna Model: {best_model_type}")

plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
plt.legend(loc="lower right")

plt.tight_layout()
plt.savefig(FINAL_PLOT_PATH, dpi=300)
plt.close()


# =============================================================================
# 16. DONE
# =============================================================================

print("\n====================================================")
print("DONE")
print("====================================================")

print(f"Optuna summarized results saved to: {RESULTS_CSV_PATH}")
print(f"Raw Optuna trials saved to: {ALL_TRIALS_PATH}")
print(f"Best parameters saved to: {BEST_PARAMS_PATH}")
print(f"Final prediction plot saved to: {FINAL_PLOT_PATH}")
print(f"CoV and R² convergence plot saved to: {CONVERGENCE_PLOT_PATH}")
print(f"Results folder: {RESULTS_DIR}")