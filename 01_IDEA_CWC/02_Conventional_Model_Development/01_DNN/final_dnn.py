"""Study 1: reproducible DNN experiment supporting the paper.

The dataset is expected at data/study_01_data.xlsx in the repository root.
The fixed DNN settings and original numerical evaluation protocol are retained.
"""

# =============================================================================
# FINAL DNN REGRESSION - 5-FOLD CV EVALUATION
# =============================================================================
# - Matches RF / XGBoost / Optuna-style evaluation protocol
# - 80:20 train/test split
# - 5-fold CV on the 80% train/validation pool
# - Fixed hyperparameters selected from manual sensitivity study
# - Final model retrained on full train/validation pool
# - Final evaluation on independent 20% test set
# =============================================================================

import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


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
# Find the repository root from this script's location, not the shell's working directory.
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
# Download the approved dataset separately and save it under this generic name.
DATA_FILE = REPOSITORY_ROOT / "data" / "study_01_data.xlsx"

RESULTS_DIR = Path(__file__).resolve().parent / "outputs" / "final_dnn"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV_PATH = RESULTS_DIR / "dnn_manual_5fold_cv_results.csv"
FOLD_RESULTS_CSV_PATH = RESULTS_DIR / "dnn_manual_fold_results.csv"
EPOCH_HISTORY_CSV_PATH = RESULTS_DIR / "dnn_manual_cv_epoch_history.csv"
BEST_SUMMARY_PATH = RESULTS_DIR / "best_dnn_manual_5fold_summary.txt"

PLOT_FINAL_PATH = RESULTS_DIR / "A_final_dnn_prediction_plot.png"
PLOT_CV_BAR_PATH = RESULTS_DIR / "B_fold_cov_bar_plot.png"
PLOT_FINAL_LOSS_PATH = RESULTS_DIR / "C_final_training_loss.png"


# =============================================================================
# 2. USER SETTINGS
# =============================================================================

# Same protocol as RF / XGBoost / Optuna
TEST_SIZE = 0.20
N_SPLITS = 5
RANDOM_STATE = 42

USE_DATA_FILTERING = True

# Filtering
NUM_BINS = 30
COUNT_THRESHOLD = 5
FILTER_COLUMN_1 = "tf [value]"
FILTER_COLUMN_2 = "tw [value]"

# Columns
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
# 3. FINAL MANUALLY SELECTED DNN HYPERPARAMETERS
# =============================================================================
ARCHITECTURE = (128, 64, 32, 16)
ACTIVATION_NAME = "GELU"
OPTIMIZER_NAME = "AdamW"
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.01


# =============================================================================
# 4. TRAINING SETTINGS
# =============================================================================
MAX_EPOCHS = 10000
MIN_EPOCHS = 1500
EVAL_STEP = 100
START_EVAL_EPOCH = 500
EARLY_STOP_PATIENCE = 1000

SCATTER_ZOOM = 1.0


# =============================================================================
# 5. HELPER FUNCTIONS
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

    Bias and CoV are based on:
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
        return optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    if name == "SGD_Momentum":
        return optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay
        )

    if name == "Adagrad":
        return optim.Adagrad(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    if name == "RMSprop":
        return optim.RMSprop(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    if name == "Adam":
        return optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    if name == "AdamW":
        return optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    raise ValueError(f"Unknown optimizer: {name}")


class RelativeSquaredError(nn.Module):
    """
    Relative squared error loss.

    Definition:
        relative_error = (y_true - y_pred) / y_true
    """
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

    plt.xlim(0, zoom_limit * 1.05)
    plt.ylim(0, zoom_limit * 1.05)

    plt.xlabel("Predicted resistance")
    plt.ylabel("Reference resistance")
    plt.title("DNN: Predicted vs Reference Resistance")

    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


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
    fold_id=None
):
    """
    Trains one DNN fold.

    Input scaling is fitted only on the training fold.
    Output scaling is local to the fold:
        y_scaled = (y_real / max_y_train) * 10

    Best epoch is selected by lowest validation CoV.
    """

    input_scaler = StandardScaler()

    X_train_scaled = input_scaler.fit_transform(X_train_np)
    X_val_scaled = input_scaler.transform(X_val_np)

    max_y = np.max(y_train_real)
    y_train_scaled = (y_train_real / max_y) * 10.0

    X_train = torch.tensor(X_train_scaled, dtype=torch.float32).to(DEVICE)
    X_val = torch.tensor(X_val_scaled, dtype=torch.float32).to(DEVICE)
    y_train = torch.tensor(y_train_scaled.reshape(-1, 1), dtype=torch.float32).to(DEVICE)

    model = DNNRegressor(
        num_features=num_features,
        architecture=architecture,
        activation_name=activation_name
    ).to(DEVICE)

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
    best_state_dict = None

    early_stop_counter = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        optimizer.zero_grad()
        y_pred_train_scaled = model(X_train)
        loss = criterion(y_pred_train_scaled, y_train)
        loss.backward()
        optimizer.step()

        if epoch >= START_EVAL_EPOCH and epoch % EVAL_STEP == 0:
            model.eval()

            with torch.no_grad():
                y_pred_val_scaled_np = model(X_val).cpu().numpy().reshape(-1)

            # Convert validation predictions back to real resistance values
            y_pred_val_real = (y_pred_val_scaled_np / 10.0) * max_y
            y_true_val_real = y_val_real.copy()

            val_bias, val_cov, val_r2, _ = compute_bias_cov_r2(
                y_true_val_real,
                y_pred_val_real
            )

            history.append({
                "fold": fold_id,
                "epoch": epoch,
                "train_loss": loss.item(),
                "val_bias": val_bias,
                "val_cov": val_cov,
                "val_r2": val_r2
            })

            if np.isnan(val_cov) or np.isinf(val_cov):
                continue

            if val_cov < best_cov:
                best_cov = val_cov
                best_bias = val_bias
                best_r2 = val_r2
                best_epoch = epoch
                best_state_dict = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                early_stop_counter = 0
            else:
                early_stop_counter += EVAL_STEP

        if epoch >= MIN_EPOCHS and early_stop_counter >= EARLY_STOP_PATIENCE:
            break

    if best_state_dict is None:
        raise RuntimeError(f"No valid DNN model found for fold {fold_id}.")

    return {
        "fold": fold_id,
        "best_bias": best_bias,
        "best_cov": best_cov,
        "best_r2": best_r2,
        "best_epoch": best_epoch,
        "history": history
    }


def train_final_dnn_model(
    X_train_np,
    y_train_real,
    num_features,
    architecture,
    activation_name,
    optimizer_name,
    learning_rate,
    weight_decay,
    epochs_to_train
):
    """
    Trains final DNN on the full train/validation pool.

    Returns:
        final_model
        final_scaler
        final_max_y
        loss_history
    """

    input_scaler = StandardScaler()
    X_train_scaled = input_scaler.fit_transform(X_train_np)

    max_y = np.max(y_train_real)
    y_train_scaled = (y_train_real / max_y) * 10.0

    X_train = torch.tensor(X_train_scaled, dtype=torch.float32).to(DEVICE)
    y_train = torch.tensor(y_train_scaled.reshape(-1, 1), dtype=torch.float32).to(DEVICE)

    model = DNNRegressor(
        num_features=num_features,
        architecture=architecture,
        activation_name=activation_name
    ).to(DEVICE)

    criterion = RelativeSquaredError()

    optimizer = get_optimizer(
        optimizer_name,
        model,
        learning_rate,
        weight_decay
    )

    loss_history = []

    for epoch in range(1, epochs_to_train + 1):
        model.train()

        optimizer.zero_grad()
        y_pred_train_scaled = model(X_train)
        loss = criterion(y_pred_train_scaled, y_train)
        loss.backward()
        optimizer.step()

        if epoch % EVAL_STEP == 0 or epoch == 1:
            loss_history.append({
                "epoch": epoch,
                "train_loss": loss.item()
            })

    return model, input_scaler, max_y, loss_history


def predict_final_dnn(model, input_scaler, max_y, X_np):
    """
    Predicts real resistance values using the final trained DNN.
    """

    X_scaled = input_scaler.transform(X_np)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)

    model.eval()

    with torch.no_grad():
        y_pred_scaled = model(X_tensor).cpu().numpy().reshape(-1)

    y_pred_real = (y_pred_scaled / 10.0) * max_y

    return y_pred_real


# =============================================================================
# 6. READ DATA
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
# 7. FILTER DATA
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
# 8. DEFINE INPUTS AND OUTPUT
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
# 9. 80:20 TRAIN/TEST SPLIT
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
# 10. DNN 5-FOLD CV EVALUATION
# =============================================================================
print("\n====================================================")
print("START DNN 5-FOLD CV EVALUATION")
print("====================================================")
print(f"Device: {DEVICE}")
print(f"Architecture: {ARCHITECTURE}")
print(f"Activation: {ACTIVATION_NAME}")
print(f"Optimizer: {OPTIMIZER_NAME}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Weight decay: {WEIGHT_DECAY}")

kf = KFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=SEED
)

fold_results = []
all_epoch_history = []

cv_start_time = time.time()

for fold_id, (train_idx, val_idx) in enumerate(kf.split(X_trainval_np), start=1):
    print(f"\n---------------- Fold {fold_id}/{N_SPLITS} ----------------")

    X_fold_train = X_trainval_np[train_idx]
    X_fold_val = X_trainval_np[val_idx]

    y_fold_train = y_trainval_np[train_idx]
    y_fold_val = y_trainval_np[val_idx]

    fold_start_time = time.time()

    fold_result = train_dnn_fold(
        X_train_np=X_fold_train,
        X_val_np=X_fold_val,
        y_train_real=y_fold_train,
        y_val_real=y_fold_val,
        num_features=X_np.shape[1],
        architecture=ARCHITECTURE,
        activation_name=ACTIVATION_NAME,
        optimizer_name=OPTIMIZER_NAME,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        fold_id=fold_id
    )

    fold_time = time.time() - fold_start_time

    fold_results.append({
        "fold": fold_id,
        "bias": fold_result["best_bias"],
        "cov": fold_result["best_cov"],
        "r2": fold_result["best_r2"],
        "best_epoch": fold_result["best_epoch"],
        "training_time_s": fold_time
    })

    all_epoch_history.extend(fold_result["history"])

    print(
        f"Fold {fold_id} best: "
        f"Bias={fold_result['best_bias']:.6f}, "
        f"CoV={fold_result['best_cov']:.6f}, "
        f"R²={fold_result['best_r2']:.6f}, "
        f"Best epoch={fold_result['best_epoch']}, "
        f"Time={fold_time:.2f} s"
    )

cv_time = time.time() - cv_start_time

fold_results_df = pd.DataFrame(fold_results)
fold_results_df.to_csv(FOLD_RESULTS_CSV_PATH, index=False)

epoch_history_df = pd.DataFrame(all_epoch_history)
epoch_history_df.to_csv(EPOCH_HISTORY_CSV_PATH, index=False)

mean_cv_bias = float(fold_results_df["bias"].mean())
std_cv_bias = float(fold_results_df["bias"].std(ddof=0))

mean_cv_cov = float(fold_results_df["cov"].mean())
std_cv_cov = float(fold_results_df["cov"].std(ddof=0))

mean_cv_r2 = float(fold_results_df["r2"].mean())
std_cv_r2 = float(fold_results_df["r2"].std(ddof=0))

mean_best_epoch = int(round(fold_results_df["best_epoch"].mean() / EVAL_STEP) * EVAL_STEP)
median_best_epoch = int(round(fold_results_df["best_epoch"].median() / EVAL_STEP) * EVAL_STEP)

FINAL_EPOCHS = max(mean_best_epoch, MIN_EPOCHS)

print("\n====================================================")
print("DNN 5-FOLD CV RESULTS")
print("====================================================")
print(f"Mean CV bias = {mean_cv_bias:.6f}")
print(f"Std CV bias  = {std_cv_bias:.6f}")
print(f"Mean CV CoV  = {mean_cv_cov:.6f}")
print(f"Std CV CoV   = {std_cv_cov:.6f}")
print(f"Mean CV R²   = {mean_cv_r2:.6f}")
print(f"Std CV R²    = {std_cv_r2:.6f}")
print(f"Mean best epoch   = {mean_best_epoch}")
print(f"Median best epoch = {median_best_epoch}")
print(f"Final epochs used = {FINAL_EPOCHS}")
print(f"Total CV time     = {cv_time:.2f} s")


# =============================================================================
# 11. FINAL TRAINING ON FULL TRAIN/VALIDATION POOL
# =============================================================================
print("\n====================================================")
print("FINAL DNN TRAINING ON FULL TRAIN/VALIDATION POOL")
print("====================================================")

final_start_time = time.time()

final_model, final_scaler, final_max_y, final_loss_history = train_final_dnn_model(
    X_train_np=X_trainval_np,
    y_train_real=y_trainval_np,
    num_features=X_np.shape[1],
    architecture=ARCHITECTURE,
    activation_name=ACTIVATION_NAME,
    optimizer_name=OPTIMIZER_NAME,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    epochs_to_train=FINAL_EPOCHS
)

final_training_time = time.time() - final_start_time

y_pred_all = predict_final_dnn(
    model=final_model,
    input_scaler=final_scaler,
    max_y=final_max_y,
    X_np=X_np
)

y_pred_test = predict_final_dnn(
    model=final_model,
    input_scaler=final_scaler,
    max_y=final_max_y,
    X_np=X_test_np
)

test_bias, test_cov, test_r2, test_ratios = compute_bias_cov_r2(
    y_test_np,
    y_pred_test
)

all_bias, all_cov, all_r2, all_ratios = compute_bias_cov_r2(
    y_np,
    y_pred_all
)

print("\n====================================================")
print("FINAL DNN METRICS")
print("====================================================")
print(f"All data:  Bias={all_bias:.6f}, CoV={all_cov:.6f}, R²={all_r2:.6f}")
print(f"Test data: Bias={test_bias:.6f}, CoV={test_cov:.6f}, R²={test_r2:.6f}")
print(f"Final model training time: {final_training_time:.2f} s")


# =============================================================================
# 12. SAVE RESULTS TABLE
# =============================================================================
results_df = pd.DataFrame([
    {
        "model": "DNN_manual_selected",
        "architecture": str(ARCHITECTURE),
        "activation": ACTIVATION_NAME,
        "optimizer": OPTIMIZER_NAME,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,

        "mean_cv_bias": mean_cv_bias,
        "std_cv_bias": std_cv_bias,
        "mean_cv_cov": mean_cv_cov,
        "std_cv_cov": std_cv_cov,
        "mean_cv_r2": mean_cv_r2,
        "std_cv_r2": std_cv_r2,

        "mean_best_epoch": mean_best_epoch,
        "median_best_epoch": median_best_epoch,
        "final_epochs_used": FINAL_EPOCHS,

        "test_bias": test_bias,
        "test_cov": test_cov,
        "test_r2": test_r2,

        "all_bias": all_bias,
        "all_cov": all_cov,
        "all_r2": all_r2,

        "cv_time_s": cv_time,
        "final_training_time_s": final_training_time
    }
])

results_df.to_csv(RESULTS_CSV_PATH, index=False)


# =============================================================================
# 13. PLOT A - FINAL PREDICTION PLOT
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
# 14. PLOT B - FOLD CoV BAR PLOT
# =============================================================================
plt.figure(figsize=(7, 5))
plt.bar(
    fold_results_df["fold"].astype(str),
    fold_results_df["cov"]
)

plt.axhline(
    mean_cv_cov,
    color="red",
    linestyle="--",
    linewidth=1.5,
    label=f"Mean CV CoV={mean_cv_cov:.4f}"
)

plt.xlabel("Fold")
plt.ylabel("Validation CoV")
plt.title("DNN 5-Fold CV Validation CoV")
plt.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_CV_BAR_PATH, dpi=300)
plt.close()


# =============================================================================
# 15. PLOT C - FINAL TRAINING LOSS
# =============================================================================
final_loss_df = pd.DataFrame(final_loss_history)

plt.figure(figsize=(8, 5))
plt.plot(
    final_loss_df["epoch"],
    final_loss_df["train_loss"],
    marker="o",
    markersize=3
)

plt.xlabel("Epoch")
plt.ylabel("Training loss")
plt.title("Final DNN Training Loss on Train/Validation Pool")
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
plt.tight_layout()
plt.savefig(PLOT_FINAL_LOSS_PATH, dpi=300)
plt.close()


# =============================================================================
# 16. SAVE SUMMARY TXT
# =============================================================================
with open(BEST_SUMMARY_PATH, "w", encoding="utf-8") as f:
    f.write("FINAL MANUALLY SELECTED DNN SUMMARY - 5-FOLD CV\n")
    f.write("============================================\n")
    f.write("This script evaluates the DNN hyperparameters selected from the manual sensitivity study.\n\n")

    f.write("DATA PROTOCOL\n")
    f.write("============================================\n")
    f.write(f"Train/test split: {int((1 - TEST_SIZE) * 100)}:{int(TEST_SIZE * 100)}\n")
    f.write(f"CV folds: {N_SPLITS}\n")
    f.write(f"Input columns: {INPUT_COLUMNS}\n")
    f.write(f"Output column: {OUTPUT_COLUMN}\n\n")

    f.write("SELECTED DNN HYPERPARAMETERS\n")
    f.write("============================================\n")
    f.write(f"Architecture: {ARCHITECTURE}\n")
    f.write(f"Activation: {ACTIVATION_NAME}\n")
    f.write(f"Optimizer: {OPTIMIZER_NAME}\n")
    f.write(f"Learning rate: {LEARNING_RATE}\n")
    f.write(f"Weight decay: {WEIGHT_DECAY}\n\n")

    f.write("CROSS-VALIDATION METRICS\n")
    f.write("============================================\n")
    f.write(f"Mean CV bias: {mean_cv_bias:.6f}\n")
    f.write(f"Std CV bias:  {std_cv_bias:.6f}\n")
    f.write(f"Mean CV CoV:  {mean_cv_cov:.6f}\n")
    f.write(f"Std CV CoV:   {std_cv_cov:.6f}\n")
    f.write(f"Mean CV R²:   {mean_cv_r2:.6f}\n")
    f.write(f"Std CV R²:    {std_cv_r2:.6f}\n")
    f.write(f"Mean best epoch:   {mean_best_epoch}\n")
    f.write(f"Median best epoch: {median_best_epoch}\n")
    f.write(f"Final epochs used: {FINAL_EPOCHS}\n\n")

    f.write("FINAL TEST METRICS\n")
    f.write("============================================\n")
    f.write(f"All data bias:  {all_bias:.6f}\n")
    f.write(f"All data CoV:   {all_cov:.6f}\n")
    f.write(f"All data R²:    {all_r2:.6f}\n")
    f.write(f"Test bias:      {test_bias:.6f}\n")
    f.write(f"Test CoV:       {test_cov:.6f}\n")
    f.write(f"Test R²:        {test_r2:.6f}\n\n")

    f.write("TIME\n")
    f.write("============================================\n")
    f.write(f"Total CV time [s]: {cv_time:.6f}\n")
    f.write(f"Final training time [s]: {final_training_time:.6f}\n\n")

    f.write("FOLD RESULTS\n")
    f.write("============================================\n")
    f.write(fold_results_df.to_string(index=False))


# =============================================================================
# 17. DONE
# =============================================================================
print("\n====================================================")
print("DONE")
print("====================================================")
print(f"Main results saved to: {RESULTS_CSV_PATH}")
print(f"Fold results saved to: {FOLD_RESULTS_CSV_PATH}")
print(f"Epoch history saved to: {EPOCH_HISTORY_CSV_PATH}")
print(f"Summary saved to: {BEST_SUMMARY_PATH}")
print(f"Final prediction plot saved to: {PLOT_FINAL_PATH}")
print(f"Fold CoV bar plot saved to: {PLOT_CV_BAR_PATH}")
print(f"Final training loss plot saved to: {PLOT_FINAL_LOSS_PATH}")