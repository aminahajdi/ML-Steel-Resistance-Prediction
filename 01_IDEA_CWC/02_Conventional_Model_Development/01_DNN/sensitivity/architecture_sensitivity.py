"""Study 1: reproducible DNN experiment supporting the paper.

The supplied dataset is stored separately from this code repository.
The original model-selection protocol and numerical settings are retained.
"""

# =============================================================================
# ARCHITECTURE SEARCH
# =============================================================================

import copy
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


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
# Resolve the repository root from the installed sensitivity/ directory.
BASE_DIR = Path(__file__).resolve().parents[4]
# The dataset is expected at the repository-level data/ directory.
DATA_FILE = BASE_DIR / "data" / "study_01_data.xlsx"
RESULTS_DIR = Path(__file__).resolve().parent / "outputs" / "architecture_search"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV_PATH = RESULTS_DIR / "architecture_search_results.csv"
PLOT_A_PATH = RESULTS_DIR / "A_validation_cov_vs_architecture.png"
PLOT_B_PATH = RESULTS_DIR / "B_validation_bias_vs_architecture.png"
PLOT_C_PATH = RESULTS_DIR / "C_cov_over_epochs_top3.png"
PLOT_D_PATH = RESULTS_DIR / "D_best_model_all_data_plot.png"


# =============================================================================
# 2. USER SETTINGS
# =============================================================================
USE_DATA_FILTERING = True
USE_INPUT_SCALER = True
USE_OUTPUT_SCALER = True

RANDOM_STATE = 42

# split: 70 / 15 / 15
VAL_PLUS_TEST_SIZE = 0.30
VAL_WITHIN_TEMP_SIZE = 0.50

# training settings
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.0

MIN_EPOCHS = 3000
MAX_EPOCHS = 10000
EVAL_STEP = 100
START_EVAL_EPOCH = 500
EARLY_STOP_PATIENCE = 2000

# scheduler
LR_PATIENCE = 20
LR_FACTOR = 0.9
MIN_LR = 0.0001

# filtering
NUM_BINS = 30
COUNT_THRESHOLD = 5
FILTER_COLUMN_1 = "tf [value]"
FILTER_COLUMN_2 = "tw [value]"

# output
OUTPUT_COLUMN = None

# architecture list
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


# =============================================================================
# 3. LOSS FUNCTION
# =============================================================================
class RelativeSquaredError(nn.Module):
    def __init__(self, epsilon=0.0):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        relative_error = (y_true - y_pred) / (y_true + self.epsilon)
        return torch.mean(relative_error ** 2)


# =============================================================================
# 4. MODEL DEFINITION
# =============================================================================
class RegressionModel(nn.Module):
    def __init__(self, num_features, architecture):
        super().__init__()

        layers = []
        in_features = num_features

        for hidden_size in architecture:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.GELU())
            in_features = hidden_size

        layers.append(nn.Linear(in_features, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# =============================================================================
# 5. READ DATA
# =============================================================================
if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

df = pd.read_excel(DATA_FILE)

print("====================================================")
print("DATASET INFO")
print("====================================================")
print(df.info())

print("\n====================================================")
print("STATISTICAL SUMMARY")
print("====================================================")
print(df.describe())

if OUTPUT_COLUMN is None:
    output_column = df.columns[-1]
else:
    output_column = OUTPUT_COLUMN

print(f"\nSelected output column: {output_column}")


# =============================================================================
# 6. FILTER DATA
# =============================================================================
if USE_DATA_FILTERING:
    print("\n====================================================")
    print("DATA FILTERING")
    print("====================================================")

    df_full_length = len(df)

    if FILTER_COLUMN_1 not in df.columns:
        raise ValueError(f"Filtering column not found in dataset: {FILTER_COLUMN_1}")
    if FILTER_COLUMN_2 not in df.columns:
        raise ValueError(f"Filtering column not found in dataset: {FILTER_COLUMN_2}")
    if output_column not in df.columns:
        raise ValueError(f"Output column not found in dataset: {output_column}")

    # filter by tf
    counts_tf, bin_edges_tf = np.histogram(df[FILTER_COLUMN_1], bins=NUM_BINS)
    low_count_bins_tf = np.where(counts_tf < COUNT_THRESHOLD)[0]
    if len(low_count_bins_tf) > 0:
        tf_cutoff = bin_edges_tf[low_count_bins_tf[0] + 1]
        df = df[df[FILTER_COLUMN_1] <= tf_cutoff].copy()
    else:
        tf_cutoff = None
    print(f"{FILTER_COLUMN_1} cutoff at: {tf_cutoff}")

    # filter by tw
    counts_tw, bin_edges_tw = np.histogram(df[FILTER_COLUMN_2], bins=NUM_BINS)
    low_count_bins_tw = np.where(counts_tw < COUNT_THRESHOLD)[0]
    if len(low_count_bins_tw) > 0:
        tw_cutoff = bin_edges_tw[low_count_bins_tw[0] + 1]
        df = df[df[FILTER_COLUMN_2] <= tw_cutoff].copy()
    else:
        tw_cutoff = None
    print(f"{FILTER_COLUMN_2} cutoff at: {tw_cutoff}")

    # filter by output
    counts_out, bin_edges_out = np.histogram(df[output_column], bins=NUM_BINS)
    low_count_bins_out = np.where(counts_out < COUNT_THRESHOLD)[0]
    if len(low_count_bins_out) > 0:
        output_cutoff = bin_edges_out[low_count_bins_out[0] + 1]
        df = df[df[output_column] <= output_cutoff].copy()
    else:
        output_cutoff = None
    print(f"{output_column} cutoff at: {output_cutoff}")

    print(f"Filtered data length: {len(df)} (from {df_full_length})")

print("\n====================================================")
print("FEATURE RANGES AFTER FILTERING")
print("====================================================")
for col in df.columns:
    print(f"{col}: min = {df[col].min()}, max = {df[col].max()}")


# =============================================================================
# 7. DEFINE INPUTS AND OUTPUTS
# =============================================================================
X_df = df.drop(columns=[output_column]).copy()
y_array = df[output_column].to_numpy(dtype=float)
feature_names = list(X_df.columns)

print("\n====================================================")
print("INPUTS AND OUTPUT")
print("====================================================")
print("Input features:")
for name in feature_names:
    print(f" - {name}")
print(f"\nOutput feature: {output_column}")


# =============================================================================
# TRAIN / VALIDATION / TEST PARTITION (70% / 15% / 15%)
# =============================================================================
# Split the unscaled observations first. This ensures that neither validation
# nor test observations influence the fitted preprocessing parameters.
X_raw = X_df.to_numpy(dtype=float)
indices = np.arange(len(X_raw))

X_train_raw, X_temp_raw, y_train_real, y_temp_real, idx_train, idx_temp = train_test_split(
    X_raw, y_array, indices,
    test_size=VAL_PLUS_TEST_SIZE,
    random_state=RANDOM_STATE,
)
X_val_raw, X_test_raw, y_val_real, y_test_real, idx_val, idx_test = train_test_split(
    X_temp_raw, y_temp_real, idx_temp,
    test_size=VAL_WITHIN_TEMP_SIZE,
    random_state=RANDOM_STATE,
)

# =============================================================================
# PREPROCESSING — FIT ONLY ON THE TRAINING SUBSET
# =============================================================================
if USE_INPUT_SCALER:
    input_scaler = StandardScaler()
    X_train_np = input_scaler.fit_transform(X_train_raw)
    X_val_np = input_scaler.transform(X_val_raw)
    X_test_np = input_scaler.transform(X_test_raw)
    X_scaled = input_scaler.transform(X_raw)  # for descriptive all-data figure
else:
    input_scaler = None
    X_train_np = X_train_raw.copy()
    X_val_np = X_val_raw.copy()
    X_test_np = X_test_raw.copy()
    X_scaled = X_raw.copy()

if USE_OUTPUT_SCALER:
    max_y = np.max(y_train_real)  # training targets only
    if max_y <= 0:
        raise ValueError("Training target maximum must be positive for output scaling.")
    y_train_np = (y_train_real / max_y) * 10.0
    y_val_np = (y_val_real / max_y) * 10.0
    y_test_np = (y_test_real / max_y) * 10.0
    y_scaled = (y_array / max_y) * 10.0  # for descriptive all-data figure
    output_scaler = {"max_y": max_y}
else:
    y_train_np = y_train_real.copy()
    y_val_np = y_val_real.copy()
    y_test_np = y_test_real.copy()
    y_scaled = y_array.copy()
    output_scaler = None

# Preserve original engineering units for reporting metrics.
y_train_true_np = y_train_real.copy()
y_val_true_np = y_val_real.copy()
y_test_true_np = y_test_real.copy()

X_train = torch.tensor(X_train_np, dtype=torch.float32)
X_val = torch.tensor(X_val_np, dtype=torch.float32)
X_test = torch.tensor(X_test_np, dtype=torch.float32)
y_train = torch.tensor(y_train_np.reshape(-1, 1), dtype=torch.float32)
y_val = torch.tensor(y_val_np.reshape(-1, 1), dtype=torch.float32)
y_test = torch.tensor(y_test_np.reshape(-1, 1), dtype=torch.float32)
X_all = torch.tensor(X_scaled, dtype=torch.float32)
y_all_scaled_tensor = torch.tensor(y_scaled.reshape(-1, 1), dtype=torch.float32)
num_features = X_train.shape[1]

print("\n====================================================")
print("DATA VARIABLES")
print("====================================================")
print(f"Number of input features = {num_features}")
print(f"Training samples         = {len(X_train)}")
print(f"Validation samples       = {len(X_val)}")
print(f"Testing samples          = {len(X_test)}")


# =============================================================================
# 10. TRAIN ALL ARCHITECTURES
# =============================================================================
results = []
cov_histories = {}
epoch_histories = {}

print("\n====================================================")
print("START ARCHITECTURE SEARCH")
print("====================================================")

for architecture in ARCHITECTURES:
    print(f"\nTesting architecture: {architecture}")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    model = RegressionModel(num_features, architecture)

    criterion = RelativeSquaredError()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        min_lr=MIN_LR
    )

    best_cov = float("inf")
    best_bias = None
    best_r2 = None
    best_epoch = None
    best_state_dict = None
    early_stop_counter = 0

    architecture_cov_history = []
    architecture_epoch_history = []

    start_time = time.time()

    for epoch in range(MAX_EPOCHS):
        model.train()

        optimizer.zero_grad()
        y_pred_train_scaled = model(X_train)
        loss = criterion(y_pred_train_scaled, y_train)
        loss.backward()
        optimizer.step()

        scheduler.step(loss.item())

        # do not evaluate too early
        if epoch >= START_EVAL_EPOCH and epoch % EVAL_STEP == 0:
            model.eval()

            with torch.no_grad():
                y_pred_val_scaled = model(X_val).cpu().numpy().flatten()
                y_true_val_scaled = y_val.cpu().numpy().flatten()

                if USE_OUTPUT_SCALER:
                    y_pred_val_real = (y_pred_val_scaled / 10.0) * output_scaler["max_y"]
                    y_true_val_real = (y_true_val_scaled / 10.0) * output_scaler["max_y"]
                else:
                    y_pred_val_real = y_pred_val_scaled.copy()
                    y_true_val_real = y_true_val_scaled.copy()

                ratios_val = y_pred_val_real / y_true_val_real
                bias_val = np.mean(ratios_val)

                # skip invalid metric cases
                if bias_val <= 0:
                    continue

                cov_val = np.std(ratios_val) / bias_val

                if np.isnan(cov_val) or np.isinf(cov_val):
                    continue

                r2_val = r2_score(y_true_val_real, y_pred_val_real)

                architecture_epoch_history.append(epoch)
                architecture_cov_history.append(cov_val)

            if cov_val < best_cov:
                best_cov = cov_val
                best_bias = bias_val
                best_r2 = r2_val
                best_epoch = epoch
                best_state_dict = copy.deepcopy(model.state_dict())
                early_stop_counter = 0
            else:
                early_stop_counter += EVAL_STEP

            print(
                f"Epoch {epoch:5d} | "
                f"Val Bias = {bias_val:.4f} | "
                f"Val CoV = {cov_val:.4f} | "
                f"Val R² = {r2_val:.6f}"
            )

        # allow early stopping only after MIN_EPOCHS
        if epoch >= MIN_EPOCHS and early_stop_counter >= EARLY_STOP_PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    training_time = time.time() - start_time

    # if nothing valid was found, skip this architecture
    if best_state_dict is None:
        print(f"Architecture {architecture} produced no valid validation metrics. Skipping.")
        continue

    # restore best weights
    model.load_state_dict(best_state_dict)

    cov_histories[str(architecture)] = architecture_cov_history
    epoch_histories[str(architecture)] = architecture_epoch_history

    # final validation evaluation at best model
    model.eval()
    with torch.no_grad():
        y_pred_val_scaled = model(X_val).cpu().numpy().flatten()
        y_true_val_scaled = y_val.cpu().numpy().flatten()

        if USE_OUTPUT_SCALER:
            y_pred_val_real = (y_pred_val_scaled / 10.0) * output_scaler["max_y"]
            y_true_val_real = (y_true_val_scaled / 10.0) * output_scaler["max_y"]
        else:
            y_pred_val_real = y_pred_val_scaled.copy()
            y_true_val_real = y_true_val_scaled.copy()

        ratios_val = y_pred_val_real / y_true_val_real
        val_bias_final = np.mean(ratios_val)
        val_cov_final = np.std(ratios_val) / val_bias_final
        val_r2_final = r2_score(y_true_val_real, y_pred_val_real)

    # final test evaluation at best model
    with torch.no_grad():
        y_pred_test_scaled = model(X_test).cpu().numpy().flatten()
        y_true_test_scaled = y_test.cpu().numpy().flatten()

        if USE_OUTPUT_SCALER:
            y_pred_test_real = (y_pred_test_scaled / 10.0) * output_scaler["max_y"]
            y_true_test_real = (y_true_test_scaled / 10.0) * output_scaler["max_y"]
        else:
            y_pred_test_real = y_pred_test_scaled.copy()
            y_true_test_real = y_true_test_scaled.copy()

        ratios_test = y_pred_test_real / y_true_test_real
        test_bias = np.mean(ratios_test)
        test_cov = np.std(ratios_test) / test_bias
        test_r2 = r2_score(y_true_test_real, y_pred_test_real)

    results.append({
        "architecture": str(architecture),
        "depth": len(architecture),
        "total_neurons": int(np.sum(architecture)),
        "best_val_cov": val_cov_final,
        "best_val_bias": val_bias_final,
        "best_val_r2": val_r2_final,
        "best_epoch": best_epoch,
        "test_cov": test_cov,
        "test_bias": test_bias,
        "test_r2": test_r2,
        "training_time_s": training_time
    })


# =============================================================================
# 11. RESULTS TABLE
# =============================================================================
results_df = pd.DataFrame(results)

if results_df.empty:
    raise RuntimeError("No valid architectures were found. Check training and validation logic.")

results_df = results_df.sort_values("best_val_cov").reset_index(drop=True)
results_df.to_csv(RESULTS_CSV_PATH, index=False)

print("\n====================================================")
print("ARCHITECTURE SEARCH RESULTS")
print("====================================================")
print(results_df)

best_architecture_str = results_df.loc[0, "architecture"]
best_architecture = eval(best_architecture_str)
top3_architectures_str = list(results_df.head(3)["architecture"])

print("\n====================================================")
print("BEST ARCHITECTURE")
print("====================================================")
print(best_architecture)


# =============================================================================
# 12. PLOT A - VALIDATION CoV VS ARCHITECTURE
# =============================================================================
plt.figure(figsize=(10, 5))
plt.plot(results_df["architecture"], results_df["best_val_cov"], marker="o")
plt.xticks(rotation=45, ha="right")
plt.xlabel("Architecture")
plt.ylabel("Best Validation CoV")
plt.title("Validation CoV vs Architecture")
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
plt.tight_layout()
plt.savefig(PLOT_A_PATH, dpi=300)
plt.close()


# =============================================================================
# 13. PLOT B - VALIDATION BIAS VS ARCHITECTURE
# =============================================================================
plt.figure(figsize=(10, 5))
plt.plot(results_df["architecture"], results_df["best_val_bias"], marker="o")
plt.axhline(1.0, color="red", linestyle="--", linewidth=1.5, label="Bias = 1.0")
plt.xticks(rotation=45, ha="right")
plt.xlabel("Architecture")
plt.ylabel("Validation Bias")
plt.title("Validation Bias vs Architecture")
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_B_PATH, dpi=300)
plt.close()


# =============================================================================
# 14. PLOT C - CoV OVER EPOCHS FOR TOP 3 ARCHITECTURES
# =============================================================================
plt.figure(figsize=(10, 5))

for arch_str in top3_architectures_str:
    plt.plot(
        epoch_histories[arch_str],
        cov_histories[arch_str],
        label=arch_str
    )

plt.xlabel("Epoch")
plt.ylabel("Validation CoV")
plt.title("CoV over Epochs for Top 3 Architectures")
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_C_PATH, dpi=300)
plt.close()


# =============================================================================
# 15. RETRAIN BEST ARCHITECTURE AND RESTORE BEST EPOCH
# =============================================================================
print("\n====================================================")
print("RETRAINING BEST ARCHITECTURE FOR FINAL PLOT")
print("====================================================")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

best_model = RegressionModel(num_features, best_architecture)

criterion = RelativeSquaredError()
optimizer = optim.Adam(best_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=LR_FACTOR,
    patience=LR_PATIENCE,
    min_lr=MIN_LR
)

best_cov_retrain = float("inf")
best_state_dict_retrain = None
early_stop_counter = 0

for epoch in range(MAX_EPOCHS):
    best_model.train()

    optimizer.zero_grad()
    y_pred_train_scaled = best_model(X_train)
    loss = criterion(y_pred_train_scaled, y_train)
    loss.backward()
    optimizer.step()

    scheduler.step(loss.item())

    if epoch >= START_EVAL_EPOCH and epoch % EVAL_STEP == 0:
        best_model.eval()
        with torch.no_grad():
            y_pred_val_scaled = best_model(X_val).cpu().numpy().flatten()
            y_true_val_scaled = y_val.cpu().numpy().flatten()

            if USE_OUTPUT_SCALER:
                y_pred_val_real = (y_pred_val_scaled / 10.0) * output_scaler["max_y"]
                y_true_val_real = (y_true_val_scaled / 10.0) * output_scaler["max_y"]
            else:
                y_pred_val_real = y_pred_val_scaled.copy()
                y_true_val_real = y_true_val_scaled.copy()

            ratios_val = y_pred_val_real / y_true_val_real
            bias_val = np.mean(ratios_val)

            if bias_val <= 0:
                continue

            cov_val = np.std(ratios_val) / bias_val

            if np.isnan(cov_val) or np.isinf(cov_val):
                continue

        if cov_val < best_cov_retrain:
            best_cov_retrain = cov_val
            best_state_dict_retrain = copy.deepcopy(best_model.state_dict())
            early_stop_counter = 0
        else:
            early_stop_counter += EVAL_STEP

    if epoch >= MIN_EPOCHS and early_stop_counter >= EARLY_STOP_PATIENCE:
        break

if best_state_dict_retrain is None:
    raise RuntimeError("Best architecture retraining produced no valid checkpoint.")

best_model.load_state_dict(best_state_dict_retrain)


# =============================================================================
# 16. FINAL EVALUATION OF BEST MODEL
# =============================================================================
best_model.eval()

with torch.no_grad():
    y_pred_all_scaled = best_model(X_all).cpu().numpy().flatten()
    y_true_all_scaled = y_all_scaled_tensor.cpu().numpy().flatten()

    y_pred_test_scaled = best_model(X_test).cpu().numpy().flatten()
    y_true_test_scaled = y_test.cpu().numpy().flatten()

if USE_OUTPUT_SCALER:
    y_pred_all = (y_pred_all_scaled / 10.0) * output_scaler["max_y"]
    y_true_all = (y_true_all_scaled / 10.0) * output_scaler["max_y"]

    y_pred_test = (y_pred_test_scaled / 10.0) * output_scaler["max_y"]
    y_true_test = (y_true_test_scaled / 10.0) * output_scaler["max_y"]
else:
    y_pred_all = y_pred_all_scaled.copy()
    y_true_all = y_true_all_scaled.copy()

    y_pred_test = y_pred_test_scaled.copy()
    y_true_test = y_true_test_scaled.copy()

ratios_all = y_pred_all / y_true_all
bias_all = np.mean(ratios_all)
cov_all = np.std(ratios_all) / bias_all
r2_all = r2_score(y_true_all, y_pred_all)

ratios_test = y_pred_test / y_true_test
bias_test = np.mean(ratios_test)
cov_test = np.std(ratios_test) / bias_test
r2_test = r2_score(y_true_test, y_pred_test)

print("\n====================================================")
print("FINAL BEST MODEL METRICS")
print("====================================================")
print(f"Best architecture = {best_architecture}")
print(f"All data:  bias = {bias_all:.4f}, CoV = {cov_all:.4f}, R² = {r2_all:.6f}")
print(f"Test data: bias = {bias_test:.4f}, CoV = {cov_test:.4f}, R² = {r2_test:.6f}")


# =============================================================================
# 17. PLOT D - FINAL ALL DATA + TEST DATA PLOT FOR BEST MODEL
# =============================================================================
plt.figure(figsize=(7, 7))

combined_values = np.concatenate((y_pred_all, y_true_all, y_pred_test, y_true_test))
min_val = np.min(combined_values)
max_val = np.max(combined_values)
line_x = np.linspace(min_val, max_val, 200)

plt.scatter(
    y_pred_all, y_true_all,
    alpha=0.5,
    color="blue",
    edgecolors="black",
    s=30,
    label=f"All data\nn={len(y_true_all)}\nb={bias_all:.4f}\nCoV={cov_all:.4f}"
)

plt.scatter(
    y_pred_test, y_true_test,
    alpha=0.7,
    color="orange",
    edgecolors="black",
    s=35,
    label=f"Test data\nn={len(y_true_test)}\nb={bias_test:.4f}\nCoV={cov_test:.4f}"
)

plt.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=2, label="Ideal fit")
plt.plot(line_x, 1.1 * line_x, "r--", linewidth=2, label="+/-10% error")
plt.plot(line_x, 0.9 * line_x, "r--", linewidth=2)

plt.xlim(0, max_val * 1.02)
plt.ylim(0, max_val * 1.02)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Best Architecture: {best_architecture}\nR² all = {r2_all:.6f}, R² test = {r2_test:.6f}")
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(PLOT_D_PATH, dpi=300)
plt.close()


# =============================================================================
# 18. DONE
# =============================================================================
print("\n====================================================")
print("DONE")
print("====================================================")
print(f"Results table saved to: {RESULTS_CSV_PATH}")
print(f"Plot A saved to: {PLOT_A_PATH}")
print(f"Plot B saved to: {PLOT_B_PATH}")
print(f"Plot C saved to: {PLOT_C_PATH}")
print(f"Plot D saved to: {PLOT_D_PATH}")