# =============================================================================
# FEATURE ABLATION OF FIXED DNN - WELDED JOINT MOMENT RESISTANCE
# STANDARD SCALER + OUTPUT SCALING + SINGLE SEED 80:20
# =============================================================================
# Purpose:
# - Train the full fixed DNN model.
# - Then retrain the same DNN several times, each time removing ONE input feature.
# - Keep the model settings, random split seed, scaling, CV and training fixed.
# - Use one fixed seed:
#       SEED = 42
# - Save:
#   1) full ablation summary
#   2) compact paper-style table
#   3) per-case results for seed 42
#   4) ranking plots based on performance loss
# =============================================================================

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
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# =============================================================================
# 0. SETTINGS
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)

TEST_SIZE = 0.20
N_SPLITS = 5

# Fixed seed for this single-split experiment.
SEED = 42

BASE_DIR = Path(__file__).resolve().parent

# Repository layout: <root>/04_Welded_Joints/07_Ablation_Study/ablation_study.py
REPOSITORY_ROOT = BASE_DIR.parent.parent
DATA_FILE = REPOSITORY_ROOT / "data" / "study_04_data.xlsx"

# Generated outputs remain local; select aggregate results for publication separately.
RESULTS_DIR = BASE_DIR / "outputs"

ABLATION_SUMMARY_PATH = RESULTS_DIR / "feature_ablation_summary.csv"
ABLATION_COMPACT_TABLE_PATH = RESULTS_DIR / "feature_ablation_compact_table.csv"
ABLATION_CASE_RESULTS_PATH = RESULTS_DIR / "feature_ablation_case_results_seed42.csv"
ABLATION_ALL_CV_RESULTS_PATH = RESULTS_DIR / "feature_ablation_all_cv_results.csv"

PLOT_DELTA_COV_PATH = RESULTS_DIR / "feature_ablation_delta_test_cov.png"
PLOT_DELTA_R2_PATH = RESULTS_DIR / "feature_ablation_delta_test_r2.png"
PLOT_LOSS_SCORE_PATH = RESULTS_DIR / "feature_ablation_performance_loss_score.png"

SUMMARY_TXT_PATH = RESULTS_DIR / "feature_ablation_summary.txt"


# =============================================================================
# 1. DATA SETTINGS
# =============================================================================

FULL_INPUT_COLUMNS = [
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
# 2. FIXED BEST DNN HYPERPARAMETERS
# =============================================================================

ARCHITECTURE = (64, 32, 16)
LEARNING_RATE = 0.005
ACTIVATION_NAME = "GELU"
OPTIMIZER_NAME = "AdamW"
WEIGHT_DECAY = 0.0
LOSS_NAME = "relative_mse"

# Training control.
MIN_EPOCHS = 300
MAX_EPOCHS = 5000
EVAL_STEP = 25
START_EVAL_EPOCH = 100
EARLY_STOP_PATIENCE = 500


# =============================================================================
# 3. HELPER FUNCTIONS
# =============================================================================

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(y_true, y_pred):
    """Return bias, CoV, R2, RMSE, and MAE."""
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


class RelativeSquaredError(nn.Module):
    """Relative squared error loss."""
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


def get_loss_function(loss_name):
    if loss_name == "relative_mse":
        return RelativeSquaredError()

    raise ValueError(f"Unknown loss function: {loss_name}")


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


def scale_y(y_real, max_y):
    return (y_real / max_y) * 10.0


def unscale_y(y_scaled, max_y):
    return (y_scaled / 10.0) * max_y


def build_model(num_features):
    return RegressionModel(
        num_features=num_features,
        architecture=ARCHITECTURE,
        activation_name=ACTIVATION_NAME,
    ).to(DEVICE)


def predict_real(model, input_scaler, max_y, X_np):
    X_scaled = input_scaler.transform(X_np)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)

    model.eval()

    with torch.no_grad():
        y_pred_scaled = model(X_tensor).cpu().numpy().flatten()

    return unscale_y(y_pred_scaled, max_y)


def train_model_with_validation_selection(
    X_train_np,
    y_train_real,
    X_val_np,
    y_val_real,
    seed,
):
    """
    Train one CV fold and select best epoch by lowest validation CoV.
    """
    set_all_seeds(seed)

    input_scaler = StandardScaler()

    X_train_scaled = input_scaler.fit_transform(X_train_np)
    X_val_scaled = input_scaler.transform(X_val_np)

    max_y = float(np.max(y_train_real))
    y_train_scaled = scale_y(y_train_real, max_y)

    X_train = torch.tensor(X_train_scaled, dtype=torch.float32).to(DEVICE)
    X_val = torch.tensor(X_val_scaled, dtype=torch.float32).to(DEVICE)
    y_train = torch.tensor(y_train_scaled.reshape(-1, 1), dtype=torch.float32).to(DEVICE)

    model = build_model(num_features=X_train_np.shape[1])
    criterion = get_loss_function(LOSS_NAME)

    optimizer = get_optimizer(
        OPTIMIZER_NAME,
        model,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_state_dict = None
    best_epoch = None
    best_val_bias = None
    best_val_cov = float("inf")
    best_val_r2 = None
    best_val_rmse = None
    best_val_mae = None

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

            y_pred_val_real = unscale_y(y_pred_val_scaled, max_y)

            val_bias, val_cov, val_r2, val_rmse, val_mae = compute_metrics(
                y_val_real,
                y_pred_val_real,
            )

            if np.isnan(val_cov) or np.isinf(val_cov):
                continue

            if val_cov < best_val_cov:
                best_val_cov = val_cov
                best_val_bias = val_bias
                best_val_r2 = val_r2
                best_val_rmse = val_rmse
                best_val_mae = val_mae
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

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    if best_epoch is None:
        best_epoch = MAX_EPOCHS - 1

    return {
        "best_epoch": best_epoch,
        "best_val_bias": best_val_bias,
        "best_val_cov": best_val_cov,
        "best_val_r2": best_val_r2,
        "best_val_rmse": best_val_rmse,
        "best_val_mae": best_val_mae,
    }


def train_final_model_fixed_epochs(
    X_train_np,
    y_train_real,
    epochs_to_train,
    seed,
):
    """Retrain final DNN on full 80% training pool."""
    set_all_seeds(seed)

    input_scaler = StandardScaler()

    X_train_scaled = input_scaler.fit_transform(X_train_np)

    max_y = float(np.max(y_train_real))
    y_train_scaled = scale_y(y_train_real, max_y)

    X_train = torch.tensor(X_train_scaled, dtype=torch.float32).to(DEVICE)
    y_train = torch.tensor(y_train_scaled.reshape(-1, 1), dtype=torch.float32).to(DEVICE)

    model = build_model(num_features=X_train_np.shape[1])
    criterion = get_loss_function(LOSS_NAME)

    optimizer = get_optimizer(
        OPTIMIZER_NAME,
        model,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    for epoch in range(epochs_to_train):
        model.train()

        optimizer.zero_grad()
        y_pred_train_scaled = model(X_train)
        loss = criterion(y_pred_train_scaled, y_train)
        loss.backward()
        optimizer.step()

    return model, input_scaler, max_y


def train_evaluate_for_seed(X_np, y_np, seed):
    set_all_seeds(seed)

    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X_np,
        y_np,
        test_size=TEST_SIZE,
        random_state=seed,
        shuffle=True,
    )

    kf = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=seed,
    )

    cv_records = []

    for fold_id, (fold_train_idx, fold_val_idx) in enumerate(kf.split(X_train_np), start=1):
        X_fold_train = X_train_np[fold_train_idx]
        X_fold_val = X_train_np[fold_val_idx]

        y_fold_train = y_train_np[fold_train_idx]
        y_fold_val = y_train_np[fold_val_idx]

        fold_result = train_model_with_validation_selection(
            X_train_np=X_fold_train,
            y_train_real=y_fold_train,
            X_val_np=X_fold_val,
            y_val_real=y_fold_val,
            seed=seed + fold_id,
        )

        cv_records.append(
            {
                "seed": seed,
                "fold": fold_id,
                "best_epoch": fold_result["best_epoch"],
                "val_bias": fold_result["best_val_bias"],
                "val_cov": fold_result["best_val_cov"],
                "val_r2": fold_result["best_val_r2"],
                "val_rmse": fold_result["best_val_rmse"],
                "val_mae": fold_result["best_val_mae"],
            }
        )

    cv_df = pd.DataFrame(cv_records)

    epochs_to_train = int(round(cv_df["best_epoch"].mean()))
    epochs_to_train = max(epochs_to_train, MIN_EPOCHS)
    epochs_to_train = min(epochs_to_train, MAX_EPOCHS)

    final_model, final_input_scaler, final_max_y = train_final_model_fixed_epochs(
        X_train_np=X_train_np,
        y_train_real=y_train_np,
        epochs_to_train=epochs_to_train,
        seed=seed,
    )

    y_pred_train = predict_real(final_model, final_input_scaler, final_max_y, X_train_np)
    y_pred_test = predict_real(final_model, final_input_scaler, final_max_y, X_test_np)
    y_pred_all = predict_real(final_model, final_input_scaler, final_max_y, X_np)

    train_bias, train_cov, train_r2, train_rmse, train_mae = compute_metrics(y_train_np, y_pred_train)
    test_bias, test_cov, test_r2, test_rmse, test_mae = compute_metrics(y_test_np, y_pred_test)
    all_bias, all_cov, all_r2, all_rmse, all_mae = compute_metrics(y_np, y_pred_all)

    record = {
        "seed": seed,
        "epochs_to_train": epochs_to_train,

        "cv_val_bias": float(cv_df["val_bias"].mean()),
        "cv_val_cov": float(cv_df["val_cov"].mean()),
        "cv_val_r2": float(cv_df["val_r2"].mean()),
        "cv_val_rmse": float(cv_df["val_rmse"].mean()),
        "cv_val_mae": float(cv_df["val_mae"].mean()),

        "train_bias": train_bias,
        "train_cov": train_cov,
        "train_r2": train_r2,
        "train_rmse": train_rmse,
        "train_mae": train_mae,

        "test_bias": test_bias,
        "test_cov": test_cov,
        "test_r2": test_r2,
        "test_rmse": test_rmse,
        "test_mae": test_mae,

        "all_bias": all_bias,
        "all_cov": all_cov,
        "all_r2": all_r2,
        "all_rmse": all_rmse,
        "all_mae": all_mae,
    }

    return record, cv_df


def run_single_feature_set(df_original, feature_columns, case_name, removed_feature):
    # Original per-case complete-case filtering is retained. If missingness varies
    # by feature, cases may have different row counts and therefore different splits.
    df = df_original[feature_columns + [OUTPUT_COLUMN]].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna().copy()
    df = df[df[OUTPUT_COLUMN] > 0].copy()

    X_np = df[feature_columns].to_numpy(dtype=float)
    y_np = df[OUTPUT_COLUMN].to_numpy(dtype=float)

    print("\n====================================================")
    print(f"CASE: {case_name}")
    print("====================================================")
    print(f"Removed feature     = {removed_feature}")
    print(f"Number of features  = {len(feature_columns)}")
    print(f"Total clean samples = {len(X_np)}")

    record, cv_all_df = train_evaluate_for_seed(X_np, y_np, SEED)

    record["case_name"] = case_name
    record["removed_feature"] = removed_feature
    record["n_features"] = len(feature_columns)
    record["features_used"] = ", ".join(feature_columns)

    cv_all_df["case_name"] = case_name
    cv_all_df["removed_feature"] = removed_feature
    cv_all_df["n_features"] = len(feature_columns)

    print(
        f"Seed {SEED:>3} | "
        f"epochs={record['epochs_to_train']} | "
        f"Test bias={record['test_bias']:.4f}, "
        f"Test CoV={record['test_cov']:.4f}, "
        f"Test R²={record['test_r2']:.4f}"
    )

    summary_record = {
        "case_name": case_name,
        "removed_feature": removed_feature,
        "n_features": len(feature_columns),
        "features_used": ", ".join(feature_columns),

        "cv_val_bias": record["cv_val_bias"],
        "cv_val_cov": record["cv_val_cov"],
        "cv_val_r2": record["cv_val_r2"],
        "cv_val_rmse": record["cv_val_rmse"],
        "cv_val_mae": record["cv_val_mae"],

        "train_bias": record["train_bias"],
        "train_cov": record["train_cov"],
        "train_r2": record["train_r2"],
        "train_rmse": record["train_rmse"],
        "train_mae": record["train_mae"],

        "test_bias": record["test_bias"],
        "test_cov": record["test_cov"],
        "test_r2": record["test_r2"],
        "test_rmse": record["test_rmse"],
        "test_mae": record["test_mae"],

        "all_bias": record["all_bias"],
        "all_cov": record["all_cov"],
        "all_r2": record["all_r2"],
        "all_rmse": record["all_rmse"],
        "all_mae": record["all_mae"],

        "epochs_to_train": record["epochs_to_train"],
    }

    case_results_df = pd.DataFrame([record])

    return summary_record, case_results_df, cv_all_df


def save_bar_plot(df_plot, y_column, title, xlabel, output_path):
    plot_df = df_plot.copy()
    plot_df = plot_df[plot_df["removed_feature"] != "None"].copy()
    plot_df = plot_df.sort_values(y_column, ascending=True)

    plt.figure(figsize=(9, max(5, 0.38 * len(plot_df))))

    plt.barh(
        plot_df["removed_feature"],
        plot_df[y_column],
    )

    plt.axvline(0.0, linewidth=1)

    plt.xlabel(xlabel)
    plt.ylabel("Removed feature")
    plt.title(title)
    plt.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# =============================================================================
# 4. LOAD DATA
# =============================================================================


def main():

    if not DATA_FILE.is_file():
        raise FileNotFoundError(
            f"Study 4 dataset not found: {DATA_FILE}\n"
            "Place the authorized workbook at data/study_04_data.xlsx in the repository root."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_original = pd.read_excel(DATA_FILE)

    missing_input_columns = [col for col in FULL_INPUT_COLUMNS if col not in df_original.columns]

    if missing_input_columns:
        raise ValueError(f"Missing input columns in dataset: {missing_input_columns}")

    if OUTPUT_COLUMN not in df_original.columns:
        raise ValueError(f"Output column not found in dataset: {OUTPUT_COLUMN}")


    print("====================================================")
    print("FIXED DNN FEATURE ABLATION STUDY - JOINT DATASET")
    print("====================================================")
    print(f"Device              = {DEVICE}")
    print(f"Dataset file         = {DATA_FILE}")
    print(f"Output column        = {OUTPUT_COLUMN}")
    print(f"Full input columns   = {FULL_INPUT_COLUMNS}")
    print(f"Seed                 = {SEED}")
    print(f"Input scaler         = StandardScaler")
    print(f"Output scaling       = y_scaled = y / max_y_train * 10")
    print(f"Architecture         = {ARCHITECTURE}")
    print(f"Activation           = {ACTIVATION_NAME}")
    print(f"Optimizer            = {OPTIMIZER_NAME}")
    print(f"Learning rate        = {LEARNING_RATE}")
    print(f"Weight decay         = {WEIGHT_DECAY}")


    # =============================================================================
    # 5. FULL MODEL + ONE-BY-ONE FEATURE ABLATION
    # =============================================================================

    summary_records = []
    all_case_results = []
    all_cv_results = []

    # Full model reference.
    summary_record, seed_df, cv_df = run_single_feature_set(
        df_original=df_original,
        feature_columns=FULL_INPUT_COLUMNS,
        case_name="Full model",
        removed_feature="None",
    )

    summary_records.append(summary_record)
    all_case_results.append(seed_df)
    all_cv_results.append(cv_df)

    # Remove one feature at a time.
    for feature_to_remove in FULL_INPUT_COLUMNS:
        reduced_features = [
            feature for feature in FULL_INPUT_COLUMNS
            if feature != feature_to_remove
        ]

        summary_record, seed_df, cv_df = run_single_feature_set(
            df_original=df_original,
            feature_columns=reduced_features,
            case_name=f"Without {feature_to_remove}",
            removed_feature=feature_to_remove,
        )

        summary_records.append(summary_record)
        all_case_results.append(seed_df)
        all_cv_results.append(cv_df)


    # =============================================================================
    # 6. SUMMARY TABLES
    # =============================================================================

    ablation_df = pd.DataFrame(summary_records)

    full_row = ablation_df[ablation_df["removed_feature"] == "None"].iloc[0]

    ablation_df["delta_test_bias"] = ablation_df["test_bias"] - full_row["test_bias"]
    ablation_df["delta_test_cov"] = ablation_df["test_cov"] - full_row["test_cov"]
    ablation_df["delta_test_r2"] = ablation_df["test_r2"] - full_row["test_r2"]
    ablation_df["delta_test_rmse"] = ablation_df["test_rmse"] - full_row["test_rmse"]
    ablation_df["delta_test_mae"] = ablation_df["test_mae"] - full_row["test_mae"]

    ablation_df["delta_train_bias"] = ablation_df["train_bias"] - full_row["train_bias"]
    ablation_df["delta_train_cov"] = ablation_df["train_cov"] - full_row["train_cov"]
    ablation_df["delta_train_r2"] = ablation_df["train_r2"] - full_row["train_r2"]

    # Bigger score = removing this feature damaged performance more.
    # CoV increase is bad; R² decrease is bad.
    ablation_df["performance_loss_score"] = (
        ablation_df["delta_test_cov"] - ablation_df["delta_test_r2"]
    )

    # The full model has zero deltas by definition.
    ablation_df.loc[ablation_df["removed_feature"] == "None", "performance_loss_score"] = 0.0

    ablation_df = ablation_df.sort_values(
        by="performance_loss_score",
        ascending=False,
    )

    ablation_df.to_csv(ABLATION_SUMMARY_PATH, index=False)

    all_case_df = pd.concat(all_case_results, ignore_index=True)
    all_case_df.to_csv(ABLATION_CASE_RESULTS_PATH, index=False)

    all_cv_df = pd.concat(all_cv_results, ignore_index=True)
    all_cv_df.to_csv(ABLATION_ALL_CV_RESULTS_PATH, index=False)

    compact_columns = [
        "case_name",
        "removed_feature",
        "n_features",
        "test_bias",
        "test_cov",
        "test_r2",
        "delta_test_cov",
        "delta_test_r2",
        "performance_loss_score",
    ]

    compact_df = ablation_df[compact_columns].copy()
    compact_df.to_csv(ABLATION_COMPACT_TABLE_PATH, index=False)


    # =============================================================================
    # 7. PLOTS
    # =============================================================================

    save_bar_plot(
        df_plot=ablation_df,
        y_column="delta_test_cov",
        title="Feature ablation: change in test CoV",
        xlabel="Δ test CoV compared with full model",
        output_path=PLOT_DELTA_COV_PATH,
    )

    save_bar_plot(
        df_plot=ablation_df,
        y_column="delta_test_r2",
        title="Feature ablation: change in test R²",
        xlabel="Δ test R² compared with full model",
        output_path=PLOT_DELTA_R2_PATH,
    )

    save_bar_plot(
        df_plot=ablation_df,
        y_column="performance_loss_score",
        title="Feature ablation: performance loss score",
        xlabel="Performance loss score = ΔCoV - ΔR²",
        output_path=PLOT_LOSS_SCORE_PATH,
    )


    # =============================================================================
    # 8. SAVE TEXT SUMMARY
    # =============================================================================

    with open(SUMMARY_TXT_PATH, "w", encoding="utf-8") as f:
        f.write("FIXED DNN FEATURE ABLATION STUDY - JOINT DATASET\n")
        f.write("================================================\n\n")

        f.write("Methodology:\n")
        f.write("- Full model trained first.\n")
        f.write("- Then one input feature removed at a time.\n")
        f.write("- Same DNN hyperparameters, split logic, output scaling, and StandardScaler used in every case.\n")
        f.write("- For each case, one fixed seed was used: SEED = 42.\n")
        f.write("- Each seed used 5-fold CV inside the 80% training pool and one final independent test evaluation.\n\n")

        f.write("Fixed DNN settings:\n")
        f.write(f"Architecture = {ARCHITECTURE}\n")
        f.write(f"Activation = {ACTIVATION_NAME}\n")
        f.write(f"Optimizer = {OPTIMIZER_NAME}\n")
        f.write(f"Learning rate = {LEARNING_RATE}\n")
        f.write(f"Weight decay = {WEIGHT_DECAY}\n")
        f.write(f"Loss = {LOSS_NAME}\n")
        f.write("Input scaler = StandardScaler\n")
        f.write("Output scaling = y_scaled = y / max_y_train * 10\n\n")

        f.write("Compact ablation table:\n")
        f.write("=======================\n")
        f.write(compact_df.to_string(index=False))
        f.write("\n\n")

        f.write("Interpretation guide:\n")
        f.write("=====================\n")
        f.write("- Positive delta_test_cov means the model became worse after removing that feature.\n")
        f.write("- Negative delta_test_r2 means the model became worse after removing that feature.\n")
        f.write("- Larger performance_loss_score indicates a larger combined test-metric degradation for this split.\n")
        f.write("- Small changes do not establish that a feature is unimportant; correlated inputs may substitute for it.\n")


    print("\n====================================================")
    print("FEATURE ABLATION SUMMARY")
    print("====================================================")
    print(
        compact_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\n====================================================")
    print("DONE")
    print("====================================================")
    print(f"Ablation summary saved to: {ABLATION_SUMMARY_PATH}")
    print(f"Compact ablation table saved to: {ABLATION_COMPACT_TABLE_PATH}")
    print(f"Case results saved to: {ABLATION_CASE_RESULTS_PATH}")
    print(f"All CV results saved to: {ABLATION_ALL_CV_RESULTS_PATH}")
    print(f"Delta CoV plot saved to: {PLOT_DELTA_COV_PATH}")
    print(f"Delta R² plot saved to: {PLOT_DELTA_R2_PATH}")
    print(f"Performance loss plot saved to: {PLOT_LOSS_SCORE_PATH}")
    print(f"Text summary saved to: {SUMMARY_TXT_PATH}")
    print(f"Results folder: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
