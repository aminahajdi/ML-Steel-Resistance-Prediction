"""SHAP interpretation for an Optuna-selected DNN (3600-second run).

This script reads the saved Optuna parameters, retrains the selected DNN,
and calculates SHAP values. It does not repeat the Optuna search.
The saved parameters must identify a DNN, not LightGBM or another estimator.
"""

# =============================================================================
# SHAP ONLY FOR OPTUNA-SELECTED DNN
# PyTorch version
# DOES NOT RERUN OPTUNA
# =============================================================================

import ast
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# =============================================================================
# 1. REPRODUCIBILITY
# =============================================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# =============================================================================
# 2. FILES AND FOLDERS
# =============================================================================

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = REPOSITORY_ROOT / "data" / "study_02_data.xlsx"

# SHAP is defined for the Optuna-selected DNN from the 3600-second run.
RESULTS_DIR = Path(__file__).resolve().parent / "outputs" / "3600s"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BEST_PARAMS_PATH = RESULTS_DIR / "best_params.txt"

SHAP_PLOT_PATH = RESULTS_DIR / "SHAP_selected_DNN.png"


# =============================================================================
# 3. DATABASE SETTINGS
# SAME AS ORIGINAL OPTUNA SCRIPT
# =============================================================================

TEST_SIZE = 0.20

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

DISPLAY_NAMES = [
    r"$h_c$",
    r"$t_{wc}$",
    r"$b_{fc}$",
    r"$t_{fc}$",
    r"$r_c$",
    r"$t_{fb}$",
    r"$N/N_{pl}$",
    r"$f_y$",
]


# =============================================================================
# 4. DNN TRAINING SETTINGS
# SAME AS ORIGINAL SCRIPT
# =============================================================================

MIN_EPOCHS = 1500
MAX_EPOCHS = 5000

EVAL_STEP = 100
START_EVAL_EPOCH = 500
EARLY_STOP_PATIENCE = 1000


# =============================================================================
# 5. HELPER FUNCTIONS
# =============================================================================

def compute_metrics(y_true, y_pred):

    ratios = y_pred / y_true

    bias = np.mean(ratios)

    cov = np.std(ratios) / abs(bias)

    return bias, cov


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


def get_optimizer(
    name,
    model,
    learning_rate,
    weight_decay
):

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

    def __init__(self, epsilon=1e-8):

        super().__init__()

        self.epsilon = epsilon


    def forward(self, y_pred, y_true):

        relative_error = (
            (y_true - y_pred)
            /
            (y_true + self.epsilon)
        )

        return torch.mean(relative_error ** 2)


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
                nn.Linear(
                    in_features,
                    hidden_size
                )
            )

            layers.append(
                get_activation(
                    activation_name
                )
            )

            in_features = hidden_size

        layers.append(
            nn.Linear(
                in_features,
                1
            )
        )

        self.network = nn.Sequential(
            *layers
        )


    def forward(self, x):

        return self.network(x)


# =============================================================================
# 6. READ SAVED OPTUNA PARAMETERS
# =============================================================================

if not BEST_PARAMS_PATH.exists():

    raise FileNotFoundError(
        f"Cannot find:\n{BEST_PARAMS_PATH}"
    )


with open(
    BEST_PARAMS_PATH,
    "r"
) as f:

    lines = f.readlines()


best_params = None


for i, line in enumerate(lines):

    if line.strip() == "Best parameters:":

        best_params = ast.literal_eval(
            lines[i + 1].strip()
        )

        break


if best_params is None:

    raise RuntimeError(
        "Could not read Best parameters "
        "from best_params.txt"
    )


print("\nBEST PARAMETERS")
print("================")
print(best_params)


if best_params["model_type"] != "DNN":

    raise RuntimeError(
        "The selected Optuna model is "
        f"{best_params['model_type']}, not DNN."
    )


architecture = ast.literal_eval(
    best_params["architecture"]
)

learning_rate = float(
    best_params["learning_rate"]
)

activation_name = best_params["activation"]

optimizer_name = best_params["optimizer"]

weight_decay = float(
    best_params["weight_decay"]
)


print("\nSELECTED DNN")
print("============")
print("Architecture:", architecture)
print("Activation:", activation_name)
print("Optimizer:", optimizer_name)
print("Learning rate:", learning_rate)
print("Weight decay:", weight_decay)


# =============================================================================
# 7. READ DATABASE
# =============================================================================

if not DATA_FILE.exists():

    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_FILE}"
    )


df = pd.read_excel(
    DATA_FILE
)


# =============================================================================
# 8. APPLY SAME FILTERING AS ORIGINAL SCRIPT
# =============================================================================

if USE_DATA_FILTERING:

    # -----------------------------
    # Filter t_fc
    # -----------------------------

    counts_1, bin_edges_1 = np.histogram(
        df[FILTER_COLUMN_1],
        bins=NUM_BINS
    )

    low_count_bins_1 = np.where(
        counts_1 < COUNT_THRESHOLD
    )[0]

    if len(low_count_bins_1) > 0:

        cutoff_1 = (
            bin_edges_1[
                low_count_bins_1[0] + 1
            ]
        )

        df = df[
            df[FILTER_COLUMN_1] <= cutoff_1
        ].copy()


    # -----------------------------
    # Filter t_wc
    # -----------------------------

    counts_2, bin_edges_2 = np.histogram(
        df[FILTER_COLUMN_2],
        bins=NUM_BINS
    )

    low_count_bins_2 = np.where(
        counts_2 < COUNT_THRESHOLD
    )[0]

    if len(low_count_bins_2) > 0:

        cutoff_2 = (
            bin_edges_2[
                low_count_bins_2[0] + 1
            ]
        )

        df = df[
            df[FILTER_COLUMN_2] <= cutoff_2
        ].copy()


    # -----------------------------
    # Filter output
    # -----------------------------

    counts_out, bin_edges_out = np.histogram(
        df[OUTPUT_COLUMN],
        bins=NUM_BINS
    )

    low_count_bins_out = np.where(
        counts_out < COUNT_THRESHOLD
    )[0]

    if len(low_count_bins_out) > 0:

        output_cutoff = (
            bin_edges_out[
                low_count_bins_out[0] + 1
            ]
        )

        df = df[
            df[OUTPUT_COLUMN] <= output_cutoff
        ].copy()


print(
    "\nFiltered database size:",
    len(df)
)


# =============================================================================
# 9. INPUT AND OUTPUT ARRAYS
# =============================================================================

X_df = df[
    INPUT_COLUMNS
].copy()

X_np = X_df.to_numpy(
    dtype=float
)

y_np = df[
    OUTPUT_COLUMN
].to_numpy(
    dtype=float
)


# =============================================================================
# 10. RECREATE ORIGINAL 80:20 SPLIT
# =============================================================================

(
    X_trainval_np,
    X_test_np,
    y_trainval_np,
    y_test_np
) = train_test_split(

    X_np,
    y_np,

    test_size=TEST_SIZE,

    random_state=SEED,

    shuffle=True
)


print(
    "Train/validation pool:",
    len(X_trainval_np)
)

print(
    "Test set:",
    len(X_test_np)
)


# =============================================================================
# 11. INTERNAL SPLIT
# USED ONLY TO FIND FINAL NUMBER OF EPOCHS
# =============================================================================

(
    X_internal_train,
    X_internal_val,
    y_internal_train,
    y_internal_val
) = train_test_split(

    X_trainval_np,
    y_trainval_np,

    test_size=0.15,

    random_state=SEED,

    shuffle=True
)


internal_scaler = StandardScaler()


X_internal_train_scaled = (
    internal_scaler.fit_transform(
        X_internal_train
    )
)


X_internal_val_scaled = (
    internal_scaler.transform(
        X_internal_val
    )
)


max_y_internal = np.max(
    y_internal_train
)


y_internal_train_scaled = (
    y_internal_train
    /
    max_y_internal
) * 10.0


X_internal_train_tensor = torch.tensor(
    X_internal_train_scaled,
    dtype=torch.float32
)


X_internal_val_tensor = torch.tensor(
    X_internal_val_scaled,
    dtype=torch.float32
)


y_internal_train_tensor = torch.tensor(
    y_internal_train_scaled.reshape(-1, 1),
    dtype=torch.float32
)


# =============================================================================
# 12. TEMPORARY DNN TO DETERMINE BEST EPOCH
# =============================================================================

torch.manual_seed(SEED)


temp_model = RegressionModel(

    num_features=X_np.shape[1],

    architecture=architecture,

    activation_name=activation_name
)


criterion = RelativeSquaredError()


temp_optimizer = get_optimizer(

    optimizer_name,

    temp_model,

    learning_rate,

    weight_decay
)


best_cov = float("inf")

best_epoch = None

early_stop_counter = 0


for epoch in range(MAX_EPOCHS):

    temp_model.train()

    temp_optimizer.zero_grad()

    prediction = temp_model(
        X_internal_train_tensor
    )

    loss = criterion(
        prediction,
        y_internal_train_tensor
    )

    loss.backward()

    temp_optimizer.step()


    if (
        epoch >= START_EVAL_EPOCH
        and
        epoch % EVAL_STEP == 0
    ):

        temp_model.eval()

        with torch.no_grad():

            pred_scaled = temp_model(
                X_internal_val_tensor
            ).cpu().numpy().flatten()


        pred_real = (
            pred_scaled
            /
            10.0
        ) * max_y_internal


        bias_val, cov_val = compute_metrics(
            y_internal_val,
            pred_real
        )


        if cov_val < best_cov:

            best_cov = cov_val

            best_epoch = epoch

            early_stop_counter = 0

        else:

            early_stop_counter += EVAL_STEP


    if (
        epoch >= MIN_EPOCHS
        and
        early_stop_counter
        >= EARLY_STOP_PATIENCE
    ):

        break


if best_epoch is None:

    best_epoch = MIN_EPOCHS


epochs_to_train = max(
    int(best_epoch),
    MIN_EPOCHS
)


print(
    "\nSelected number of epochs:",
    epochs_to_train
)


# =============================================================================
# 13. SCALE FULL TRAIN/VALIDATION POOL
# =============================================================================

final_input_scaler = StandardScaler()


X_trainval_scaled = (
    final_input_scaler.fit_transform(
        X_trainval_np
    )
)


X_test_scaled = (
    final_input_scaler.transform(
        X_test_np
    )
)


final_max_y = np.max(
    y_trainval_np
)


y_trainval_scaled = (
    y_trainval_np
    /
    final_max_y
) * 10.0


X_trainval_tensor = torch.tensor(
    X_trainval_scaled,
    dtype=torch.float32
)


y_trainval_tensor = torch.tensor(
    y_trainval_scaled.reshape(-1, 1),
    dtype=torch.float32
)


# =============================================================================
# 14. RETRAIN ONLY THE SELECTED DNN
# =============================================================================

torch.manual_seed(SEED)


final_model = RegressionModel(

    num_features=X_np.shape[1],

    architecture=architecture,

    activation_name=activation_name
)


criterion = RelativeSquaredError()


final_optimizer = get_optimizer(

    optimizer_name,

    final_model,

    learning_rate,

    weight_decay
)


print(
    "\nRetraining selected DNN only..."
)


for epoch in range(
    epochs_to_train
):

    final_model.train()

    final_optimizer.zero_grad()

    prediction = final_model(
        X_trainval_tensor
    )

    loss = criterion(
        prediction,
        y_trainval_tensor
    )

    loss.backward()

    final_optimizer.step()


final_model.eval()


print(
    "Selected DNN retrained."
)


# =============================================================================
# 15. SHAP BACKGROUND + TEST SAMPLE
# =============================================================================

rng = np.random.default_rng(
    SEED
)


n_background = min(
    200,
    len(X_trainval_scaled)
)


background_idx = rng.choice(

    len(X_trainval_scaled),

    size=n_background,

    replace=False
)


background = torch.tensor(

    X_trainval_scaled[
        background_idx
    ],

    dtype=torch.float32
)


n_shap = min(
    1000,
    len(X_test_scaled)
)


shap_idx = rng.choice(

    len(X_test_scaled),

    size=n_shap,

    replace=False
)


X_shap_scaled = (
    X_test_scaled[
        shap_idx
    ]
)


X_shap_original = (
    X_test_np[
        shap_idx
    ]
)


X_shap_tensor = torch.tensor(

    X_shap_scaled,

    dtype=torch.float32
)


# =============================================================================
# 16. COMPUTE SHAP VALUES
# =============================================================================

print(
    "\nCalculating SHAP values..."
)


explainer = shap.DeepExplainer(

    final_model,

    background
)


shap_values = explainer.shap_values(

    X_shap_tensor
)


if isinstance(
    shap_values,
    list
):

    shap_values = shap_values[0]


shap_values = np.asarray(
    shap_values
)


if shap_values.ndim == 3:

    shap_values = (
        shap_values[:, :, 0]
    )


# Convert SHAP output
# from scaled target units
# back to resistance units

shap_values = (
    shap_values
    /
    10.0
) * final_max_y


print(
    "SHAP calculation completed."
)


# =============================================================================
# 17. SHAP BEESWARM PLOT
# =============================================================================

plt.figure(
    figsize=(8, 6)
)


shap.summary_plot(

    shap_values,

    X_shap_original,

    feature_names=DISPLAY_NAMES,

    plot_type="dot",

    show=False
)


plt.xlabel(
    "SHAP value (impact on predicted resistance)"
)


plt.tight_layout()


plt.savefig(

    SHAP_PLOT_PATH,

    dpi=600,

    bbox_inches="tight"
)


plt.show()


print(
    "\nDONE"
)

print(
    "SHAP plot saved to:"
)

print(
    SHAP_PLOT_PATH
)