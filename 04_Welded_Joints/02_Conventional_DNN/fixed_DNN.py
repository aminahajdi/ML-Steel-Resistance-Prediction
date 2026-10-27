# =============================================================================
# FIXED DNN FOR WELDED-JOINT MOMENT RESISTANCE
# Fixed hyperparameters selected by a prior Optuna search:
# architecture=(64, 32, 16), activation=GELU, optimizer=AdamW,
# learning_rate=0.005, weight_decay=0.0
#
# Workflow:
# 1) Read dataset
# 2) 80:20 train/test split
# 3) 5-fold CV on training pool using fixed DNN
# 4) Retrain final model on full training pool
# 5) Evaluate on independent test set and all data
# 6) Save metrics, predictions, model, scaler, and plots
# =============================================================================

import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# =============================================================================
# 0. REPRODUCIBILITY
# =============================================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# For CPU reproducibility
torch.set_num_threads(1)


# =============================================================================
# 1. FILES AND FOLDERS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

# The dataset is stored in the repository-level data/ directory.
REPO_ROOT = BASE_DIR.parent.parent
DATA_FILE = REPO_ROOT / "data" / "study_04_data.xlsx"
RESULTS_DIR = BASE_DIR / "outputs"

CV_RESULTS_PATH = RESULTS_DIR / "fixed_dnn_5fold_cv_results.csv"
METRICS_PATH = RESULTS_DIR / "fixed_dnn_metrics.txt"
PREDICTIONS_PATH = RESULTS_DIR / "fixed_dnn_predictions.csv"
FINAL_PLOT_PATH = RESULTS_DIR / "fixed_dnn_all_data_test_plot.png"
LOSS_PLOT_PATH = RESULTS_DIR / "fixed_dnn_training_loss.png"

MODEL_PATH = RESULTS_DIR / "fixed_dnn_model_state_dict.pt"
SCALER_PATH = RESULTS_DIR / "fixed_dnn_input_scaler.pkl"


# =============================================================================
# 2. EXPERIMENT CONFIGURATION
# =============================================================================

TEST_SIZE = 0.20
N_SPLITS = 5

# Training and epoch-selection settings used for the fixed DNN.
MIN_EPOCHS = 300
MAX_EPOCHS = 5000
EVAL_STEP = 25
START_EVAL_EPOCH = 100
EARLY_STOP_PATIENCE = 500

USE_DATA_FILTERING = False

NUM_BINS = 30
COUNT_THRESHOLD = 5

FILTER_COLUMN_1 = "tfc"
FILTER_COLUMN_2 = "twc"

INPUT_COLUMNS = [
    "hc",
    "bc",
    "twc",
    "tfc",
    "rc",
    "hb1",
    "bb1",
    "twb1",
    "tfb1",
    "rb1",
    "axial force percentage",
    "stiffening",
    "joint location",
    "moment ratio",
    "fy",
]

OUTPUT_COLUMN = "moment resistance"


# =============================================================================
# 3. FIXED BEST DNN PARAMETERS FROM OPTUNA
# =============================================================================

ARCHITECTURE = (64, 32, 16)
LEARNING_RATE = 0.005
ACTIVATION_NAME = "GELU"
OPTIMIZER_NAME = "AdamW"
WEIGHT_DECAY = 0.0


# =============================================================================
# 4. HELPER FUNCTIONS
# =============================================================================

def compute_metrics(y_true, y_pred):
    """Return bias, CoV, R2, RMSE, and MAE."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    ratios = y_pred / y_true
    bias = np.mean(ratios)
    cov = np.std(ratios) / abs(bias)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    return bias, cov, r2, rmse, mae


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
            weight_decay=weight_decay,
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
    """Mean squared relative error, with a small numerical stabilizer."""
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


def train_dnn_with_internal_validation(
    X_train_np,
    X_val_np,
    y_train_real,
    y_val_real,
    num_features,
    architecture=ARCHITECTURE,
    activation_name=ACTIVATION_NAME,
    optimizer_name=OPTIMIZER_NAME,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
):
    """
    Train one DNN fold and select the epoch with the lowest validation CoV.
    This follows the same logic as the Optuna script.
    """
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
        weight_decay,
    )

    best_cov = float("inf")
    best_bias = None
    best_r2 = None
    best_rmse = None
    best_mae = None
    best_epoch = None
    best_state_dict = None

    early_stop_counter = 0

    history = []

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

            bias_val, cov_val, r2_val, rmse_val, mae_val = compute_metrics(
                y_val_real,
                y_pred_val_real,
            )

            history.append(
                {
                    "epoch": epoch,
                    "loss": float(loss.item()),
                    "val_bias": bias_val,
                    "val_cov": cov_val,
                    "val_r2": r2_val,
                    "val_rmse": rmse_val,
                    "val_mae": mae_val,
                }
            )

            if np.isnan(cov_val) or np.isinf(cov_val):
                continue

            if cov_val < best_cov:
                best_cov = cov_val
                best_bias = bias_val
                best_r2 = r2_val
                best_rmse = rmse_val
                best_mae = mae_val
                best_epoch = epoch
                best_state_dict = {
                    key: value.detach().clone()
                    for key, value in model.state_dict().items()
                }
                early_stop_counter = 0
            else:
                early_stop_counter += EVAL_STEP

        if epoch >= MIN_EPOCHS and early_stop_counter >= EARLY_STOP_PATIENCE:
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return {
        "model": model,
        "scaler": input_scaler,
        "max_y": max_y,
        "best_epoch": best_epoch,
        "best_bias": best_bias,
        "best_cov": best_cov,
        "best_r2": best_r2,
        "best_rmse": best_rmse,
        "best_mae": best_mae,
        "history": history,
    }


def predict_dnn(model, input_scaler, max_y, X_np):
    X_scaled = input_scaler.transform(X_np)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    model.eval()

    with torch.no_grad():
        y_pred_scaled = model(X_tensor).cpu().numpy().flatten()

    y_pred_real = (y_pred_scaled / 10.0) * max_y

    return y_pred_real


def train_final_dnn_for_fixed_epochs(
    X_train_np,
    y_train_real,
    num_features,
    epochs_to_train,
    architecture=ARCHITECTURE,
    activation_name=ACTIVATION_NAME,
    optimizer_name=OPTIMIZER_NAME,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
):
    """Retrain final DNN on the full train/validation pool."""
    input_scaler = StandardScaler()

    X_train_scaled = input_scaler.fit_transform(X_train_np)

    max_y = np.max(y_train_real)
    y_train_scaled = (y_train_real / max_y) * 10.0

    X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train = torch.tensor(y_train_scaled.reshape(-1, 1), dtype=torch.float32)

    model = RegressionModel(num_features, architecture, activation_name)
    criterion = RelativeSquaredError()

    optimizer = get_optimizer(
        optimizer_name,
        model,
        learning_rate,
        weight_decay,
    )

    loss_history = []

    for epoch in range(epochs_to_train):
        model.train()

        optimizer.zero_grad()
        y_pred_train_scaled = model(X_train)
        loss = criterion(y_pred_train_scaled, y_train)
        loss.backward()
        optimizer.step()

        loss_history.append({"epoch": epoch, "loss": float(loss.item())})

    return model, input_scaler, max_y, loss_history


def main():
    """Run cross-validation, final training, and performance evaluation."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}. "
            "Place study_04_data.xlsx in the repository-level data/ directory."
        )

    df = pd.read_excel(DATA_FILE)

    missing_input_columns = [col for col in INPUT_COLUMNS if col not in df.columns]

    if missing_input_columns:
        raise ValueError(f"Missing input columns in dataset: {missing_input_columns}")

    if OUTPUT_COLUMN not in df.columns:
        raise ValueError(f"Output column not found in dataset: {OUTPUT_COLUMN}")

    df = df[INPUT_COLUMNS + [OUTPUT_COLUMN]].dropna().copy()

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

        print(f"Filtered data length: {len(df)} from {df_full_length}")
        print(f"{FILTER_COLUMN_1} cutoff: {cutoff_1}")
        print(f"{FILTER_COLUMN_2} cutoff: {cutoff_2}")
        print(f"{OUTPUT_COLUMN} cutoff: {output_cutoff}")


    X_df = df[INPUT_COLUMNS].copy()
    y_array = df[OUTPUT_COLUMN].to_numpy(dtype=float)

    X_np = X_df.to_numpy(dtype=float)
    y_np = y_array.copy()


    # =============================================================================
    # 6. 80:20 TRAIN/TEST SPLIT
    # =============================================================================

    X_trainval_np, X_test_np, y_trainval_np, y_test_np = train_test_split(
        X_np,
        y_np,
        test_size=TEST_SIZE,
        random_state=SEED,
        shuffle=True,
    )


    print("====================================================")
    print("FIXED DNN MODEL SETTINGS")
    print("====================================================")
    print(f"Dataset file            = {DATA_FILE}")
    print(f"Input columns           = {INPUT_COLUMNS}")
    print(f"Output column           = {OUTPUT_COLUMN}")
    print(f"Architecture            = {ARCHITECTURE}")
    print(f"Activation              = {ACTIVATION_NAME}")
    print(f"Optimizer               = {OPTIMIZER_NAME}")
    print(f"Learning rate           = {LEARNING_RATE}")
    print(f"Weight decay            = {WEIGHT_DECAY}")
    print(f"Input features          = {X_np.shape[1]}")
    print(f"Total data              = {len(X_np)}")
    print(f"Train/validation pool   = {len(X_trainval_np)}")
    print(f"Independent test set    = {len(X_test_np)}")
    print(f"Cross-validation folds  = {N_SPLITS}")


    # =============================================================================
    # 7. 5-FOLD CROSS-VALIDATION ON TRAINING POOL
    # =============================================================================

    kf = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED,
    )

    cv_records = []

    for fold_id, (train_idx, val_idx) in enumerate(kf.split(X_trainval_np), start=1):
        X_fold_train = X_trainval_np[train_idx]
        X_fold_val = X_trainval_np[val_idx]

        y_fold_train = y_trainval_np[train_idx]
        y_fold_val = y_trainval_np[val_idx]

        fold_result = train_dnn_with_internal_validation(
            X_train_np=X_fold_train,
            X_val_np=X_fold_val,
            y_train_real=y_fold_train,
            y_val_real=y_fold_val,
            num_features=X_np.shape[1],
        )

        record = {
            "fold": fold_id,
            "best_epoch": fold_result["best_epoch"],
            "val_bias": fold_result["best_bias"],
            "val_cov": fold_result["best_cov"],
            "val_r2": fold_result["best_r2"],
            "val_rmse": fold_result["best_rmse"],
            "val_mae": fold_result["best_mae"],
        }

        cv_records.append(record)

        print(
            f"Fold {fold_id}: "
            f"epoch={record['best_epoch']}, "
            f"bias={record['val_bias']:.6f}, "
            f"CoV={record['val_cov']:.6f}, "
            f"R2={record['val_r2']:.6f}"
        )


    cv_results_df = pd.DataFrame(cv_records)
    cv_results_df.to_csv(CV_RESULTS_PATH, index=False)

    mean_cv_bias = cv_results_df["val_bias"].mean()
    std_cv_bias = cv_results_df["val_bias"].std(ddof=0)

    mean_cv_cov = cv_results_df["val_cov"].mean()
    std_cv_cov = cv_results_df["val_cov"].std(ddof=0)

    mean_cv_r2 = cv_results_df["val_r2"].mean()
    std_cv_r2 = cv_results_df["val_r2"].std(ddof=0)

    mean_best_epoch = cv_results_df["best_epoch"].mean()

    print("\n====================================================")
    print("5-FOLD CV SUMMARY")
    print("====================================================")
    print(f"Mean CV bias = {mean_cv_bias:.6f} ± {std_cv_bias:.6f}")
    print(f"Mean CV CoV  = {mean_cv_cov:.6f} ± {std_cv_cov:.6f}")
    print(f"Mean CV R²   = {mean_cv_r2:.6f} ± {std_cv_r2:.6f}")
    print(f"Mean best epoch = {mean_best_epoch:.1f}")


    # =============================================================================
    # 8. FINAL MODEL TRAINING ON FULL TRAINING POOL
    # =============================================================================

    epochs_to_train = int(max(round(mean_best_epoch), MIN_EPOCHS))

    final_model, final_input_scaler, final_max_y, final_loss_history = train_final_dnn_for_fixed_epochs(
        X_train_np=X_trainval_np,
        y_train_real=y_trainval_np,
        num_features=X_np.shape[1],
        epochs_to_train=epochs_to_train,
    )

    y_pred_trainval = predict_dnn(final_model, final_input_scaler, final_max_y, X_trainval_np)
    y_pred_test = predict_dnn(final_model, final_input_scaler, final_max_y, X_test_np)
    y_pred_all = predict_dnn(final_model, final_input_scaler, final_max_y, X_np)


    # =============================================================================
    # 9. FINAL METRICS
    # =============================================================================

    bias_trainval, cov_trainval, r2_trainval, rmse_trainval, mae_trainval = compute_metrics(
        y_trainval_np,
        y_pred_trainval,
    )

    bias_test, cov_test, r2_test, rmse_test, mae_test = compute_metrics(
        y_test_np,
        y_pred_test,
    )

    bias_all, cov_all, r2_all, rmse_all, mae_all = compute_metrics(
        y_np,
        y_pred_all,
    )

    print("\n====================================================")
    print("FINAL FIXED DNN METRICS")
    print("====================================================")
    print(f"Train/validation data: bias = {bias_trainval:.4f}, CoV = {cov_trainval:.4f}, R² = {r2_trainval:.6f}, RMSE = {rmse_trainval:.4f}, MAE = {mae_trainval:.4f}")
    print(f"Independent test data: bias = {bias_test:.4f}, CoV = {cov_test:.4f}, R² = {r2_test:.6f}, RMSE = {rmse_test:.4f}, MAE = {mae_test:.4f}")
    print(f"All data:              bias = {bias_all:.4f}, CoV = {cov_all:.4f}, R² = {r2_all:.6f}, RMSE = {rmse_all:.4f}, MAE = {mae_all:.4f}")


    # =============================================================================
    # 10. SAVE MODEL, SCALER, PREDICTIONS, AND METRICS
    # =============================================================================

    torch.save(
        {
            "model_state_dict": final_model.state_dict(),
            "input_columns": INPUT_COLUMNS,
            "output_column": OUTPUT_COLUMN,
            "architecture": ARCHITECTURE,
            "activation": ACTIVATION_NAME,
            "optimizer": OPTIMIZER_NAME,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "max_y": final_max_y,
            "epochs_to_train": epochs_to_train,
            "seed": SEED,
        },
        MODEL_PATH,
    )

    joblib.dump(final_input_scaler, SCALER_PATH)

    predictions_df = df.copy()
    predictions_df["prediction"] = y_pred_all
    predictions_df["prediction_to_reference_ratio"] = y_pred_all / y_np
    predictions_df.to_csv(PREDICTIONS_PATH, index=False)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        f.write("FIXED DNN PARAMETERS\n")
        f.write("====================\n")
        f.write(f"Architecture: {ARCHITECTURE}\n")
        f.write(f"Activation: {ACTIVATION_NAME}\n")
        f.write(f"Optimizer: {OPTIMIZER_NAME}\n")
        f.write(f"Learning rate: {LEARNING_RATE}\n")
        f.write(f"Weight decay: {WEIGHT_DECAY}\n")
        f.write(f"Epochs trained: {epochs_to_train}\n")
        f.write(f"Seed: {SEED}\n\n")

        f.write("INPUT COLUMNS\n")
        f.write("=============\n")
        f.write(str(INPUT_COLUMNS))
        f.write("\n\n")

        f.write("OUTPUT COLUMN\n")
        f.write("=============\n")
        f.write(str(OUTPUT_COLUMN))
        f.write("\n\n")

        f.write("5-FOLD CV SUMMARY ON TRAINING POOL\n")
        f.write("==================================\n")
        f.write(f"Mean CV bias = {mean_cv_bias:.6f} ± {std_cv_bias:.6f}\n")
        f.write(f"Mean CV CoV  = {mean_cv_cov:.6f} ± {std_cv_cov:.6f}\n")
        f.write(f"Mean CV R²   = {mean_cv_r2:.6f} ± {std_cv_r2:.6f}\n")
        f.write(f"Mean best epoch = {mean_best_epoch:.1f}\n\n")

        f.write("FINAL METRICS\n")
        f.write("=============\n")
        f.write(
            f"Train/validation data: bias = {bias_trainval:.6f}, CoV = {cov_trainval:.6f}, "
            f"R² = {r2_trainval:.6f}, RMSE = {rmse_trainval:.6f}, MAE = {mae_trainval:.6f}\n"
        )
        f.write(
            f"Independent test data: bias = {bias_test:.6f}, CoV = {cov_test:.6f}, "
            f"R² = {r2_test:.6f}, RMSE = {rmse_test:.6f}, MAE = {mae_test:.6f}\n"
        )
        f.write(
            f"All data: bias = {bias_all:.6f}, CoV = {cov_all:.6f}, "
            f"R² = {r2_all:.6f}, RMSE = {rmse_all:.6f}, MAE = {mae_all:.6f}\n"
        )


    # =============================================================================
    # 11. FINAL PLOT: ALL DATA + TEST DATA
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
        label=f"All data\nn={len(y_np)}\nb={bias_all:.4f}\nCoV={cov_all:.4f}",
    )

    plt.scatter(
        y_pred_test,
        y_test_np,
        alpha=0.7,
        color="orange",
        edgecolors="black",
        s=35,
        label=f"Test data\nn={len(y_test_np)}\nb={bias_test:.4f}\nCoV={cov_test:.4f}",
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
        linewidth=2,
        label="+/-10% error",
    )

    plt.plot(
        line_x,
        0.9 * line_x,
        "r--",
        linewidth=2,
    )

    plt.xlim(0, max_val * 1.02)
    plt.ylim(0, max_val * 1.02)

    plt.xlabel("Predicted moment resistance")
    plt.ylabel("Reference moment resistance")
    plt.title("Fixed DNN: Moment resistance")

    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(FINAL_PLOT_PATH, dpi=300)
    plt.close()


    # =============================================================================
    # 12. TRAINING LOSS PLOT
    # =============================================================================

    loss_df = pd.DataFrame(final_loss_history)

    plt.figure(figsize=(8, 5))
    plt.plot(loss_df["epoch"], loss_df["loss"], linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Relative squared error loss")
    plt.title("Fixed DNN training loss")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_PATH, dpi=300)
    plt.close()


    # =============================================================================
    # 13. DONE
    # =============================================================================

    print("\n====================================================")
    print("DONE")
    print("====================================================")
    print(f"5-fold CV results saved to: {CV_RESULTS_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"Predictions saved to: {PREDICTIONS_PATH}")
    print(f"Final prediction plot saved to: {FINAL_PLOT_PATH}")
    print(f"Training loss plot saved to: {LOSS_PLOT_PATH}")
    print(f"Model state dict saved to: {MODEL_PATH}")
    print(f"Input scaler saved to: {SCALER_PATH}")
    print(f"Results folder: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
