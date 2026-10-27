# =============================================================================
# FIXED DNN VALIDATION
# NO OUTPUT SCALING
# EXACT REPRODUCTION OF AUTOML DNN LOGIC
# =============================================================================

from pathlib import Path

import random
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
# SETTINGS
# =============================================================================

SEED = 42

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

OUTPUT_COLUMN = "F_c_wc_Rd_kN"


# =============================================================================
# FIXED DNN PARAMETERS FROM AUTOML
# =============================================================================

ARCHITECTURE = (128, 64, 32)
LEARNING_RATE = 0.0005
ACTIVATION = "ReLU"
OPTIMIZER_NAME = "Adam"
WEIGHT_DECAY = 0.0


# =============================================================================
# FULL REPRODUCIBILITY
# =============================================================================

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.use_deterministic_algorithms(True)


# =============================================================================
# FILES
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "Parametrized set.xlsx"

RESULTS_DIR = BASE_DIR / "Fixed_DNN_Validation_NO_OUTPUT_SCALING"
RESULTS_DIR.mkdir(exist_ok=True)

FINAL_PLOT_PATH = RESULTS_DIR / "fixed_dnn_validation_no_output_scaling.png"


# =============================================================================
# READ DATA
# =============================================================================

df = pd.read_excel(DATA_FILE)


# =============================================================================
# SAME FILTERING AS AUTOML
# =============================================================================

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

    counts_out, bin_edges_out = np.histogram(df[OUTPUT_COLUMN], bins=NUM_BINS)
    low_count_bins_out = np.where(counts_out < COUNT_THRESHOLD)[0]

    if len(low_count_bins_out) > 0:
        output_cutoff = bin_edges_out[low_count_bins_out[0] + 1]
        df = df[df[OUTPUT_COLUMN] <= output_cutoff].copy()
    else:
        output_cutoff = None

    print("====================================================")
    print("DATA FILTERING")
    print("====================================================")

    print(f"Filtered data length: {len(df)} from {df_full_length}")
    print(f"{FILTER_COLUMN_1} cutoff: {cutoff_1}")
    print(f"{FILTER_COLUMN_2} cutoff: {cutoff_2}")
    print(f"{OUTPUT_COLUMN} cutoff: {output_cutoff}")


# =============================================================================
# INPUTS / OUTPUTS
# =============================================================================

X = df[INPUT_COLUMNS].to_numpy(dtype=float)
y = df[OUTPUT_COLUMN].to_numpy(dtype=float)


# =============================================================================
# TRAIN / TEST SPLIT
# =============================================================================

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=SEED,
    shuffle=True
)


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(y_true, y_pred):

    ratios = y_pred / y_true

    bias = np.mean(ratios)
    cov = np.std(ratios) / abs(bias)
    r2 = r2_score(y_true, y_pred)

    return bias, cov, r2


# =============================================================================
# ACTIVATION
# =============================================================================

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


# =============================================================================
# OPTIMIZER
# =============================================================================

def get_optimizer(name, model, learning_rate, weight_decay):

    if name == "Adam":
        return optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    raise ValueError(f"Unknown optimizer: {name}")


# =============================================================================
# LOSS FUNCTION
# =============================================================================

class RelativeSquaredError(nn.Module):

    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):

        relative_error = (
            (y_true - y_pred) /
            (y_true + self.epsilon)
        )

        return torch.mean(relative_error ** 2)


# =============================================================================
# MODEL
# =============================================================================

class RegressionModel(nn.Module):

    def __init__(
        self,
        num_features,
        architecture,
        activation_name
    ):
        super().__init__()

        layers = []

        in_features = num_features

        for hidden_size in architecture:

            layers.append(
                nn.Linear(in_features, hidden_size)
            )

            layers.append(
                get_activation(activation_name)
            )

            in_features = hidden_size

        layers.append(
            nn.Linear(in_features, 1)
        )

        self.network = nn.Sequential(*layers)

    def forward(self, x):

        return self.network(x)


# =============================================================================
# TRAIN ONE DNN FOLD
# =============================================================================

def train_dnn_fold(
    X_train_np,
    X_val_np,
    y_train_real,
    y_val_real,
    num_features,
):

    input_scaler = StandardScaler()

    X_train_scaled = input_scaler.fit_transform(X_train_np)
    X_val_scaled = input_scaler.transform(X_val_np)

    # ============================================================
    # NO OUTPUT SCALING
    # ============================================================

    y_train_scaled = y_train_real.copy()

    X_train_tensor = torch.tensor(
        X_train_scaled,
        dtype=torch.float32
    )

    X_val_tensor = torch.tensor(
        X_val_scaled,
        dtype=torch.float32
    )

    y_train_tensor = torch.tensor(
        y_train_scaled.reshape(-1, 1),
        dtype=torch.float32
    )

    model = RegressionModel(
        num_features=num_features,
        architecture=ARCHITECTURE,
        activation_name=ACTIVATION
    )

    criterion = RelativeSquaredError()

    optimizer = get_optimizer(
        OPTIMIZER_NAME,
        model,
        LEARNING_RATE,
        WEIGHT_DECAY
    )

    best_cov = float("inf")
    best_bias = None
    best_r2 = None
    best_epoch = None

    early_stop_counter = 0

    for epoch in range(MAX_EPOCHS):

        model.train()

        optimizer.zero_grad()

        y_pred_train_scaled = model(X_train_tensor)

        loss = criterion(
            y_pred_train_scaled,
            y_train_tensor
        )

        loss.backward()

        optimizer.step()

        if epoch >= START_EVAL_EPOCH and epoch % EVAL_STEP == 0:

            model.eval()

            with torch.no_grad():

                y_pred_val_scaled = model(
                    X_val_tensor
                ).cpu().numpy().flatten()

                # ============================================================
                # NO OUTPUT SCALING
                # ============================================================

                y_pred_val_real = y_pred_val_scaled.copy()

            bias_val, cov_val, r2_val = compute_metrics(
                y_val_real,
                y_pred_val_real
            )

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

        if (
            epoch >= MIN_EPOCHS and
            early_stop_counter >= EARLY_STOP_PATIENCE
        ):
            break

    return (
        best_cov,
        best_bias,
        best_r2,
        best_epoch
    )


# =============================================================================
# 5-FOLD CROSS VALIDATION
# =============================================================================

kf = KFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=SEED
)

fold_covs = []
fold_biases = []
fold_r2s = []
fold_epochs = []

print("\n====================================================")
print("5-FOLD CROSS VALIDATION")
print("====================================================")

for fold_id, (train_idx, val_idx) in enumerate(
    kf.split(X_trainval),
    start=1
):

    X_fold_train = X_trainval[train_idx]
    X_fold_val = X_trainval[val_idx]

    y_fold_train = y_trainval[train_idx]
    y_fold_val = y_trainval[val_idx]

    cov, bias, r2, best_epoch = train_dnn_fold(
        X_train_np=X_fold_train,
        X_val_np=X_fold_val,
        y_train_real=y_fold_train,
        y_val_real=y_fold_val,
        num_features=X.shape[1]
    )

    fold_covs.append(cov)
    fold_biases.append(bias)
    fold_r2s.append(r2)
    fold_epochs.append(best_epoch)

    print(f"\nFold {fold_id}")
    print(f"Best epoch = {best_epoch}")
    print(f"Bias       = {bias:.6f}")
    print(f"CoV        = {cov:.6f}")
    print(f"R²         = {r2:.6f}")


# =============================================================================
# FINAL CV RESULTS
# =============================================================================

mean_bias = np.mean(fold_biases)
std_bias = np.std(fold_biases)

mean_cov = np.mean(fold_covs)
std_cov = np.std(fold_covs)

mean_r2 = np.mean(fold_r2s)
std_r2 = np.std(fold_r2s)

print("\n====================================================")
print("FINAL CROSS-VALIDATION RESULTS")
print("====================================================")

print(f"Mean CV Bias = {mean_bias:.6f} ± {std_bias:.6f}")
print(f"Mean CV CoV  = {mean_cov:.6f} ± {std_cov:.6f}")
print(f"Mean CV R²   = {mean_r2:.6f} ± {std_r2:.6f}")


# =============================================================================
# EXACT SAME FINAL RETRAINING AS AUTOML
# =============================================================================

X_internal_train, X_internal_val, y_internal_train, y_internal_val = train_test_split(
    X_trainval,
    y_trainval,
    test_size=0.15,
    random_state=SEED,
    shuffle=True
)

final_cov, final_bias, final_r2, final_epoch = train_dnn_fold(
    X_train_np=X_internal_train,
    X_val_np=X_internal_val,
    y_train_real=y_internal_train,
    y_val_real=y_internal_val,
    num_features=X.shape[1]
)

print("\n====================================================")
print("FINAL INTERNAL VALIDATION")
print("====================================================")

print(f"Selected epoch = {final_epoch}")
print(f"Validation CoV = {final_cov:.6f}")
print(f"Validation R²  = {final_r2:.6f}")


# =============================================================================
# FINAL TRAINING ON FULL TRAIN/VALIDATION SET
# =============================================================================

final_scaler = StandardScaler()

X_trainval_scaled = final_scaler.fit_transform(X_trainval)
X_test_scaled = final_scaler.transform(X_test)
X_all_scaled = final_scaler.transform(X)

# ============================================================
# NO OUTPUT SCALING
# ============================================================

y_trainval_scaled = y_trainval.copy()

X_trainval_tensor = torch.tensor(
    X_trainval_scaled,
    dtype=torch.float32
)

y_trainval_tensor = torch.tensor(
    y_trainval_scaled.reshape(-1, 1),
    dtype=torch.float32
)

final_model = RegressionModel(
    num_features=X.shape[1],
    architecture=ARCHITECTURE,
    activation_name=ACTIVATION
)

criterion = RelativeSquaredError()

optimizer = get_optimizer(
    OPTIMIZER_NAME,
    final_model,
    LEARNING_RATE,
    WEIGHT_DECAY
)

epochs_to_train = (
    final_epoch
    if final_epoch is not None
    else MIN_EPOCHS
)

epochs_to_train = max(
    int(epochs_to_train),
    MIN_EPOCHS
)

print(f"\nFinal training epochs = {epochs_to_train}")

for epoch in range(epochs_to_train):

    final_model.train()

    optimizer.zero_grad()

    y_pred_train_scaled = final_model(
        X_trainval_tensor
    )

    loss = criterion(
        y_pred_train_scaled,
        y_trainval_tensor
    )

    loss.backward()

    optimizer.step()


# =============================================================================
# PREDICTIONS
# =============================================================================

final_model.eval()

with torch.no_grad():

    y_pred_test_scaled = final_model(
        torch.tensor(
            X_test_scaled,
            dtype=torch.float32
        )
    ).cpu().numpy().flatten()

    y_pred_all_scaled = final_model(
        torch.tensor(
            X_all_scaled,
            dtype=torch.float32
        )
    ).cpu().numpy().flatten()

# ============================================================
# NO OUTPUT SCALING
# ============================================================

y_pred_test = y_pred_test_scaled.copy()
y_pred_all = y_pred_all_scaled.copy()


# =============================================================================
# FINAL METRICS
# =============================================================================

bias_test, cov_test, r2_test = compute_metrics(
    y_test,
    y_pred_test
)

bias_all, cov_all, r2_all = compute_metrics(
    y,
    y_pred_all
)

print("\n====================================================")
print("FINAL TEST RESULTS")
print("====================================================")

print(f"Bias = {bias_test:.6f}")
print(f"CoV  = {cov_test:.6f}")
print(f"R²   = {r2_test:.6f}")

print("\n====================================================")
print("ALL DATA RESULTS")
print("====================================================")

print(f"Bias = {bias_all:.6f}")
print(f"CoV  = {cov_all:.6f}")
print(f"R²   = {r2_all:.6f}")


# =============================================================================
# FINAL PLOT
# =============================================================================

plt.figure(figsize=(7, 7))

combined_values = np.concatenate(
    (
        y_pred_all,
        y,
        y_pred_test,
        y_test
    )
)

min_val = np.min(combined_values)
max_val = np.max(combined_values)

line_x = np.linspace(min_val, max_val, 200)

plt.scatter(
    y_pred_all,
    y,
    alpha=0.5,
    color="blue",
    edgecolors="black",
    s=30,
    label=f"All data\nn={len(y)}\nb={bias_all:.4f}\nCoV={cov_all:.4f}"
)

plt.scatter(
    y_pred_test,
    y_test,
    alpha=0.7,
    color="orange",
    edgecolors="black",
    s=35,
    label=f"Test data\nn={len(y_test)}\nb={bias_test:.4f}\nCoV={cov_test:.4f}"
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

plt.title("Fixed DNN Validation - No Output Scaling")

plt.grid(
    True,
    linestyle="--",
    linewidth=0.5,
    alpha=0.6
)

plt.legend(loc="lower right")

plt.tight_layout()

plt.savefig(
    FINAL_PLOT_PATH,
    dpi=300
)

plt.close()


# =============================================================================
# DONE
# =============================================================================

print("\n====================================================")
print("DONE")
print("====================================================")

print(f"Plot saved to: {FINAL_PLOT_PATH}")
print(f"Results folder: {RESULTS_DIR}")