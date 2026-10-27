# =============================================================================
# FIXED BEST DNN FOR EXPERIMENTAL CWC DATABASE - MULTI-SEED 80:20 VERSION
#Buckling subset 
# Purpose:
# - Use measured input variables only.
# - Use fixed DNN hyperparameters from Optuna.
# - Evaluate the same DNN over several random 80:20 train/test splits.
# - For each seed: 5-fold CV is performed inside the 80% training pool only.
# - The independent 20% test set is evaluated ONCE at the end for each seed.
# - Report mean ± std test performance across seeds.
# - Save one representative train/test plot using REPRESENTATIVE_SEED.
# =============================================================================

import random
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# =============================================================================
# 0. SETTINGS
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_SIZE = 0.20
N_SPLITS = 5

# Seeds used to evaluate split sensitivity.
SEEDS = [1,7,21,42,100]

# Seed used only for the representative final plot.
REPRESENTATIVE_SEED = 7

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR.parent / "outputs" / "CWC_EC_buckling.xlsx"

RESULTS_DIR = BASE_DIR / "outputs"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MULTI_SEED_RESULTS_PATH = RESULTS_DIR / "multi_seed_dnn_results.csv"
MULTI_SEED_SUMMARY_PATH = RESULTS_DIR / "multi_seed_dnn_summary.csv"
REPRESENTATIVE_PLOT_PATH = RESULTS_DIR / "representative_seed_dnn_train_test_plot.png"
REPRESENTATIVE_RESULTS_TXT_PATH = RESULTS_DIR / "representative_seed_dnn_results.txt"
REPRESENTATIVE_MODEL_PATH = RESULTS_DIR / "representative_seed_dnn_model.pth"
REPRESENTATIVE_INPUT_SCALER_PATH = RESULTS_DIR / "representative_seed_input_scaler.pkl"
REPRESENTATIVE_OUTPUT_SCALER_PATH = RESULTS_DIR / "representative_seed_output_scaler_info.pkl"

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

# Fixed best Optuna hyperparameters.
ARCHITECTURE = (16, 8)
LEARNING_RATE = 1e-3
ACTIVATION_NAME = "GELU"
OPTIMIZER_NAME = "AdamW"
WEIGHT_DECAY = 0.0
DROPOUT = 0.05
BATCH_SIZE = 4
LOSS_NAME = "relative_mse"

# Training control.
MIN_EPOCHS = 300
MAX_EPOCHS = 3000
EVAL_STEP = 25
START_EVAL_EPOCH = 100
EARLY_STOP_PATIENCE = 500


# =============================================================================
# 1. HELPER FUNCTIONS
# =============================================================================

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        relative_error = (y_true - y_pred) / (y_true + self.epsilon)
        return torch.mean(relative_error ** 2)


class RegressionModel(nn.Module):
    def __init__(self, num_features, architecture, activation_name, dropout):
        super().__init__()
        layers = []
        in_features = num_features

        for hidden_size in architecture:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(get_activation(activation_name))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
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
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]

    if name == "Adam":
        return optim.Adam(trainable_parameters, lr=learning_rate, weight_decay=weight_decay)
    if name == "AdamW":
        return optim.AdamW(trainable_parameters, lr=learning_rate, weight_decay=weight_decay)
    if name == "RMSprop":
        return optim.RMSprop(trainable_parameters, lr=learning_rate, weight_decay=weight_decay)
    if name == "SGD_Momentum":
        return optim.SGD(trainable_parameters, lr=learning_rate, momentum=0.9, weight_decay=weight_decay)

    raise ValueError(f"Unknown optimizer: {name}")


def scale_y(y_real, max_y):
    return (y_real / max_y) * 10.0


def unscale_y(y_scaled, max_y):
    return (y_scaled / 10.0) * max_y


def make_loader(X_scaled, y_scaled, batch_size, shuffle=True, seed=42):
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled.reshape(-1, 1), dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def predict_real(model, X_scaled, max_y):
    model.eval()
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        y_pred_scaled = model(X_tensor).cpu().numpy().flatten()

    return unscale_y(y_pred_scaled, max_y)


def build_model(num_features):
    return RegressionModel(
        num_features=num_features,
        architecture=ARCHITECTURE,
        activation_name=ACTIVATION_NAME,
        dropout=DROPOUT,
    ).to(DEVICE)


def train_model(
    model,
    train_loader,
    X_train_scaled,
    y_train_real,
    max_y,
    criterion,
    optimizer,
    max_epochs,
    X_val_scaled=None,
    y_val_real=None,
    use_validation_selection=False,
):
    """
    If validation data are provided, the best epoch is selected by highest validation R²,
    then lowest validation CoV. This is used only inside CV.

    For final training, validation is not used and the model is simply trained for
    the selected number of epochs.
    """
    best_state_dict = None
    best_epoch = None
    best_val_r2 = -float("inf")
    best_val_cov = float("inf")
    early_stop_counter = 0

    losses = []

    for epoch in range(max_epochs):
        model.train()
        epoch_losses = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            epoch_losses.append(float(loss.detach().cpu().item()))

        mean_epoch_loss = float(np.mean(epoch_losses))
        losses.append(mean_epoch_loss)

        should_evaluate = (epoch >= START_EVAL_EPOCH and epoch % EVAL_STEP == 0) or epoch == max_epochs - 1

        if should_evaluate and X_val_scaled is not None and y_val_real is not None:
            y_pred_val_epoch = predict_real(model, X_val_scaled, max_y)
            val_bias_ep, val_cov_ep, val_r2_ep, val_rmse_ep, val_mae_ep = compute_metrics(
                y_val_real,
                y_pred_val_epoch,
            )

            if not (np.isnan(val_r2_ep) or np.isnan(val_cov_ep) or np.isinf(val_cov_ep)):
                improved = False

                if val_r2_ep > best_val_r2:
                    improved = True
                elif np.isclose(val_r2_ep, best_val_r2, atol=1e-6) and val_cov_ep < best_val_cov:
                    improved = True

                if improved:
                    best_val_r2 = val_r2_ep
                    best_val_cov = val_cov_ep
                    best_epoch = epoch
                    best_state_dict = {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    }
                    early_stop_counter = 0
                else:
                    early_stop_counter += EVAL_STEP

        if (
            use_validation_selection
            and epoch >= MIN_EPOCHS
            and X_val_scaled is not None
            and early_stop_counter >= EARLY_STOP_PATIENCE
        ):
            break

    if use_validation_selection and best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    if best_epoch is None:
        best_epoch = max_epochs - 1

    return losses, best_epoch, best_val_r2, best_val_cov


def make_prediction_plot(y_train, y_pred_train, y_test, y_pred_test, title, save_path):
    bias_train, cov_train, r2_train, rmse_train, mae_train = compute_metrics(y_train, y_pred_train)
    bias_test, cov_test, r2_test, rmse_test, mae_test = compute_metrics(y_test, y_pred_test)

    plt.figure(figsize=(7, 7))

    combined_values = np.concatenate((y_train, y_pred_train, y_test, y_pred_test))
    min_val = float(np.min(combined_values))
    max_val = float(np.max(combined_values))
    line_x = np.linspace(min_val, max_val, 200)

    plt.scatter(
        y_pred_train,
        y_train,
        alpha=0.65,
        edgecolors="black",
        s=35,
        color="tab:blue",
        label=(
            f"Train data\n"
            f"n={len(y_train)}\n"
            f"b={bias_train:.4f}\n"
            f"CoV={cov_train:.4f}\n"
            f"R²={r2_train:.4f}"
        ),
    )

    plt.scatter(
        y_pred_test,
        y_test,
        alpha=0.85,
        edgecolors="black",
        s=50,
        color="tab:purple",
        label=(
            f"Test data\n"
            f"n={len(y_test)}\n"
            f"b={bias_test:.4f}\n"
            f"CoV={cov_test:.4f}\n"
            f"R²={r2_test:.4f}"
        ),
    )

    plt.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=2, label="Ideal fit")
    plt.plot(line_x, 1.1 * line_x, "r--", linewidth=1.5, label="±10% error")
    plt.plot(line_x, 0.9 * line_x, "r--", linewidth=1.5)

    plt.xlim(0, max_val * 1.05)
    plt.ylim(0, max_val * 1.05)
    plt.xlabel("Predicted experimental resistance")
    plt.ylabel("True experimental resistance")
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    return {
        "train_bias": bias_train,
        "train_cov": cov_train,
        "train_r2": r2_train,
        "train_rmse": rmse_train,
        "train_mae": mae_train,
        "test_bias": bias_test,
        "test_cov": cov_test,
        "test_r2": r2_test,
        "test_rmse": rmse_test,
        "test_mae": mae_test,
    }


def train_evaluate_for_seed(X_np, y_np, seed):
    set_all_seeds(seed)

    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X_np,
        y_np,
        test_size=TEST_SIZE,
        random_state=seed,
        shuffle=True,
    )

    criterion = get_loss_function(LOSS_NAME)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

    cv_records = []

    for fold_id, (fold_train_idx, fold_val_idx) in enumerate(kf.split(X_train_np), start=1):
        set_all_seeds(seed + fold_id)

        X_fold_train = X_train_np[fold_train_idx]
        X_fold_val = X_train_np[fold_val_idx]
        y_fold_train = y_train_np[fold_train_idx]
        y_fold_val = y_train_np[fold_val_idx]

        fold_scaler = StandardScaler()
        X_fold_train_scaled = fold_scaler.fit_transform(X_fold_train)
        X_fold_val_scaled = fold_scaler.transform(X_fold_val)

        fold_max_y = float(np.max(y_fold_train))
        y_fold_train_scaled = scale_y(y_fold_train, fold_max_y)

        fold_loader = make_loader(
            X_fold_train_scaled,
            y_fold_train_scaled,
            batch_size=BATCH_SIZE,
            shuffle=True,
            seed=seed + fold_id,
        )

        fold_model = build_model(num_features=X_np.shape[1])
        fold_optimizer = get_optimizer(
            name=OPTIMIZER_NAME,
            model=fold_model,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

        _, best_epoch, best_val_r2, best_val_cov = train_model(
            model=fold_model,
            train_loader=fold_loader,
            X_train_scaled=X_fold_train_scaled,
            y_train_real=y_fold_train,
            X_val_scaled=X_fold_val_scaled,
            y_val_real=y_fold_val,
            max_y=fold_max_y,
            criterion=criterion,
            optimizer=fold_optimizer,
            max_epochs=MAX_EPOCHS,
            use_validation_selection=True,
        )

        y_pred_val = predict_real(fold_model, X_fold_val_scaled, fold_max_y)
        val_bias, val_cov, val_r2, val_rmse, val_mae = compute_metrics(y_fold_val, y_pred_val)

        cv_records.append({
            "fold": fold_id,
            "best_epoch": best_epoch,
            "best_val_r2_during_training": best_val_r2,
            "best_val_cov_during_training": best_val_cov,
            "val_bias": val_bias,
            "val_cov": val_cov,
            "val_r2": val_r2,
            "val_rmse": val_rmse,
            "val_mae": val_mae,
        })

    cv_df = pd.DataFrame(cv_records)

    epochs_to_train = int(round(cv_df["best_epoch"].mean()))
    epochs_to_train = max(epochs_to_train, MIN_EPOCHS)
    epochs_to_train = min(epochs_to_train, MAX_EPOCHS)

    # Final model training on the full 80% training pool.
    set_all_seeds(seed)

    input_scaler = StandardScaler()
    X_train_scaled = input_scaler.fit_transform(X_train_np)
    X_test_scaled = input_scaler.transform(X_test_np)

    max_y = float(np.max(y_train_np))
    y_train_scaled = scale_y(y_train_np, max_y)

    train_loader = make_loader(
        X_train_scaled,
        y_train_scaled,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=seed,
    )

    model = build_model(num_features=X_np.shape[1])
    optimizer = get_optimizer(
        name=OPTIMIZER_NAME,
        model=model,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    training_losses, final_epoch, _, _ = train_model(
        model=model,
        train_loader=train_loader,
        X_train_scaled=X_train_scaled,
        y_train_real=y_train_np,
        max_y=max_y,
        criterion=criterion,
        optimizer=optimizer,
        max_epochs=epochs_to_train,
        use_validation_selection=False,
    )

    y_pred_train = predict_real(model, X_train_scaled, max_y)
    y_pred_test = predict_real(model, X_test_scaled, max_y)

    train_bias, train_cov, train_r2, train_rmse, train_mae = compute_metrics(y_train_np, y_pred_train)
    test_bias, test_cov, test_r2, test_rmse, test_mae = compute_metrics(y_test_np, y_pred_test)

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
    }

    data = {
        "model": model,
        "input_scaler": input_scaler,
        "max_y": max_y,
        "training_losses": training_losses,
        "cv_df": cv_df,
        "X_train": X_train_np,
        "X_test": X_test_np,
        "y_train": y_train_np,
        "y_test": y_test_np,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
        "record": record,
    }

    return record, data


# =============================================================================
# 2. READ DATA
# =============================================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

df = pd.read_excel(DATA_FILE)

missing_input_columns = [col for col in INPUT_COLUMNS if col not in df.columns]
if missing_input_columns:
    raise ValueError(f"Missing input columns in Excel file: {missing_input_columns}")

if OUTPUT_COLUMN not in df.columns:
    raise ValueError(f"Output column not found in Excel file: {OUTPUT_COLUMN}")

df = df[INPUT_COLUMNS + [OUTPUT_COLUMN]].copy()
df = df.replace([np.inf, -np.inf], np.nan).dropna().copy()
df = df[df[OUTPUT_COLUMN] > 0].copy()

X_np = df[INPUT_COLUMNS].to_numpy(dtype=float)
y_np = df[OUTPUT_COLUMN].to_numpy(dtype=float)

print("====================================================")
print("FIXED BEST DNN - MULTI-SEED 80:20 EXPERIMENTAL CWC DATABASE")
print("====================================================")
print(f"Device               = {DEVICE}")
print(f"Dataset file          = {DATA_FILE}")
print(f"Input columns         = {INPUT_COLUMNS}")
print(f"Output column         = {OUTPUT_COLUMN}")
print(f"Total clean specimens = {len(X_np)}")
print(f"Train/test split      = {int((1 - TEST_SIZE) * 100)}:{int(TEST_SIZE * 100)}")
print(f"Seeds                 = {SEEDS}")
print(f"Representative seed   = {REPRESENTATIVE_SEED}")
print(f"CV folds per seed     = {N_SPLITS}")
print("\nFixed hyperparameters:")
print(f"Architecture          = {ARCHITECTURE}")
print(f"Learning rate         = {LEARNING_RATE}")
print(f"Activation            = {ACTIVATION_NAME}")
print(f"Optimizer             = {OPTIMIZER_NAME}")
print(f"Weight decay          = {WEIGHT_DECAY}")
print(f"Dropout               = {DROPOUT}")
print(f"Batch size            = {BATCH_SIZE}")
print(f"Loss                  = {LOSS_NAME}")


# =============================================================================
# 3. MULTI-SEED EVALUATION
# =============================================================================

seed_records = []
representative_data = None

for seed in SEEDS:
    print("\n====================================================")
    print(f"SEED = {seed}")
    print("====================================================")

    record, data = train_evaluate_for_seed(X_np, y_np, seed)
    seed_records.append(record)

    print(
        f"Seed {seed:>3} | "
        f"epochs={record['epochs_to_train']} | "
        f"Train R²={record['train_r2']:.4f}, "
        f"Test R²={record['test_r2']:.4f}, "
        f"Test CoV={record['test_cov']:.4f}, "
        f"Test bias={record['test_bias']:.4f}"
    )

    if seed == REPRESENTATIVE_SEED:
        representative_data = data

seed_df = pd.DataFrame(seed_records)
seed_df.to_csv(MULTI_SEED_RESULTS_PATH, index=False)

summary_rows = []
for metric in ["bias", "cov", "r2", "rmse", "mae"]:
    summary_rows.append({
        "metric": metric,
        "train_mean": seed_df[f"train_{metric}"].mean(),
        "train_std": seed_df[f"train_{metric}"].std(),
        "test_mean": seed_df[f"test_{metric}"].mean(),
        "test_std": seed_df[f"test_{metric}"].std(),
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(MULTI_SEED_SUMMARY_PATH, index=False)

# Best seed here means highest test R². It is reported for reference only;
# use the multi-seed mean ± std for scientific performance reporting.
best_seed_row = seed_df.sort_values(
    by=["test_r2", "test_cov"],
    ascending=[False, True],
).iloc[0]

print("\n====================================================")
print("MULTI-SEED SUMMARY")
print("====================================================")
print(f"Number of seeds = {len(SEEDS)}")
print(f"Seeds = {SEEDS}")
print("\nTest-set performance across seeds:")
print(f"Mean Test R²   = {seed_df['test_r2'].mean():.4f} ± {seed_df['test_r2'].std():.4f}")
print(f"Mean Test CoV  = {seed_df['test_cov'].mean():.4f} ± {seed_df['test_cov'].std():.4f}")
print(f"Mean Test Bias = {seed_df['test_bias'].mean():.4f} ± {seed_df['test_bias'].std():.4f}")
print(f"Mean Test RMSE = {seed_df['test_rmse'].mean():.4f} ± {seed_df['test_rmse'].std():.4f}")
print(f"Mean Test MAE  = {seed_df['test_mae'].mean():.4f} ± {seed_df['test_mae'].std():.4f}")
print("\nBest seed by test R², for reference only:")
print(
    f"Seed {int(best_seed_row['seed'])} | "
    f"Test R²={best_seed_row['test_r2']:.4f}, "
    f"Test CoV={best_seed_row['test_cov']:.4f}, "
    f"Test bias={best_seed_row['test_bias']:.4f}"
)


# =============================================================================
# 4. REPRESENTATIVE FINAL PLOT AND MODEL SAVE
# =============================================================================

if representative_data is None:
    raise RuntimeError(f"Representative seed {REPRESENTATIVE_SEED} was not included in SEEDS.")

representative_metrics = make_prediction_plot(
    y_train=representative_data["y_train"],
    y_pred_train=representative_data["y_pred_train"],
    y_test=representative_data["y_test"],
    y_pred_test=representative_data["y_pred_test"],
    title=f"Buckling subset - Seed {REPRESENTATIVE_SEED}",
    save_path=REPRESENTATIVE_PLOT_PATH,
)

# Save representative model and scalers.
torch.save(
    {
        "model_state_dict": representative_data["model"].state_dict(),
        "input_columns": INPUT_COLUMNS,
        "output_column": OUTPUT_COLUMN,
        "architecture": ARCHITECTURE,
        "activation": ACTIVATION_NAME,
        "dropout": DROPOUT,
        "max_y": representative_data["max_y"],
        "loss": LOSS_NAME,
        "optimizer": OPTIMIZER_NAME,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "test_size": TEST_SIZE,
        "n_splits": N_SPLITS,
        "representative_seed": REPRESENTATIVE_SEED,
        "representative_metrics": representative_metrics,
        "multi_seed_summary": summary_df.to_dict(orient="records"),
    },
    REPRESENTATIVE_MODEL_PATH,
)

with open(REPRESENTATIVE_INPUT_SCALER_PATH, "wb") as f:
    pickle.dump(representative_data["input_scaler"], f)

with open(REPRESENTATIVE_OUTPUT_SCALER_PATH, "wb") as f:
    pickle.dump(
        {
            "max_y": representative_data["max_y"],
            "scaling": "y_scaled = y / max_y * 10",
        },
        f,
    )

with open(REPRESENTATIVE_RESULTS_TXT_PATH, "w", encoding="utf-8") as f:
    f.write("BUCKLING SUBSET - MULTI-SEED RESULTS FOR EXPERIMENTAL CWC DATABASE\n")
    f.write("===============================================================\n\n")
    f.write("Methodology:\n")
    f.write("- 80:20 train/test split repeated across multiple random seeds\n")
    f.write("- 5-fold CV inside the training pool for each seed\n")
    f.write("- final test set evaluated once per seed\n")
    f.write("- reported performance should be mean ± std across seeds\n\n")
    f.write(f"Seeds = {SEEDS}\n")
    f.write(f"Representative seed = {REPRESENTATIVE_SEED}\n\n")
    f.write("Fixed hyperparameters:\n")
    f.write(f"Architecture = {ARCHITECTURE}\n")
    f.write(f"Learning rate = {LEARNING_RATE}\n")
    f.write(f"Activation = {ACTIVATION_NAME}\n")
    f.write(f"Optimizer = {OPTIMIZER_NAME}\n")
    f.write(f"Weight decay = {WEIGHT_DECAY}\n")
    f.write(f"Dropout = {DROPOUT}\n")
    f.write(f"Batch size = {BATCH_SIZE}\n")
    f.write(f"Loss = {LOSS_NAME}\n\n")
    f.write("Multi-seed summary:\n")
    for _, row in summary_df.iterrows():
        f.write(
            f"{row['metric']}: "
            f"train = {row['train_mean']:.6f} ± {row['train_std']:.6f}, "
            f"test = {row['test_mean']:.6f} ± {row['test_std']:.6f}\n"
        )
    f.write("\nBest seed by test R², for reference only:\n")
    f.write(
        f"Seed {int(best_seed_row['seed'])}: "
        f"Test R²={best_seed_row['test_r2']:.6f}, "
        f"Test CoV={best_seed_row['test_cov']:.6f}, "
        f"Test bias={best_seed_row['test_bias']:.6f}\n"
    )

print("\n====================================================")
print(f"REPRESENTATIVE 80:20 TRAIN/TEST RESULTS, SEED = {REPRESENTATIVE_SEED}")
print("====================================================")
print(
    f"Train data: bias = {representative_metrics['train_bias']:.4f}, "
    f"CoV = {representative_metrics['train_cov']:.4f}, "
    f"R² = {representative_metrics['train_r2']:.6f}, "
    f"RMSE = {representative_metrics['train_rmse']:.4f}, "
    f"MAE = {representative_metrics['train_mae']:.4f}"
)
print(
    f"Test data:  bias = {representative_metrics['test_bias']:.4f}, "
    f"CoV = {representative_metrics['test_cov']:.4f}, "
    f"R² = {representative_metrics['test_r2']:.6f}, "
    f"RMSE = {representative_metrics['test_rmse']:.4f}, "
    f"MAE = {representative_metrics['test_mae']:.4f}"
)


# =============================================================================
# 5. DONE
# =============================================================================

print("\n====================================================")
print("DONE")
print("====================================================")
print(f"Multi-seed results saved to: {MULTI_SEED_RESULTS_PATH}")
print(f"Multi-seed summary saved to: {MULTI_SEED_SUMMARY_PATH}")
print(f"Representative plot saved to: {REPRESENTATIVE_PLOT_PATH}")
print(f"Representative model saved to: {REPRESENTATIVE_MODEL_PATH}")
print(f"Representative input scaler saved to: {REPRESENTATIVE_INPUT_SCALER_PATH}")
print(f"Representative output scaler saved to: {REPRESENTATIVE_OUTPUT_SCALER_PATH}")
print(f"Representative text results saved to: {REPRESENTATIVE_RESULTS_TXT_PATH}")
print(f"Results folder: {RESULTS_DIR}")
