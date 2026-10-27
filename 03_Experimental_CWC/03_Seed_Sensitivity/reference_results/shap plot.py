# =============================================================================
# SHAP BEESWARM PLOT FOR REPRESENTATIVE BEST DNN MODEL
# =============================================================================

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import shap


# =============================================================================
# 0. SETTINGS
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR.parent / "CWC database.xlsx"

RESULTS_DIR = BASE_DIR

MODEL_PATH = RESULTS_DIR / "representative_seed_dnn_model.pth"
INPUT_SCALER_PATH = RESULTS_DIR / "representative_seed_input_scaler.pkl"

SHAP_DIR = RESULTS_DIR / "SHAP_results"
SHAP_DIR.mkdir(exist_ok=True)

SHAP_BEESWARM_PATH = SHAP_DIR / "representative_seed_dnn_shap_beeswarm.png"
SHAP_BAR_PATH = SHAP_DIR / "representative_seed_dnn_shap_bar.png"

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

OUTPUT_COLUMN = "F_exp"

BACKGROUND_SIZE = 50
EXPLAIN_SIZE = 200

REPRESENTATIVE_SEED = 7


# =============================================================================
# 1. MODEL DEFINITION
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


# =============================================================================
# 2. LOAD DATA
# =============================================================================

df = pd.read_excel(DATA_FILE)
df = df[INPUT_COLUMNS + [OUTPUT_COLUMN]].copy()
df = df.replace([np.inf, -np.inf], np.nan).dropna()
df = df[df[OUTPUT_COLUMN] > 0].copy()

X_original = df[INPUT_COLUMNS].to_numpy(dtype=float)


# =============================================================================
# 3. LOAD MODEL AND SCALER
# =============================================================================

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

architecture = checkpoint["architecture"]
activation = checkpoint["activation"]
dropout = checkpoint["dropout"]
max_y = checkpoint["max_y"]

model = RegressionModel(
    num_features=len(INPUT_COLUMNS),
    architecture=architecture,
    activation_name=activation,
    dropout=dropout,
).to(DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

with open(INPUT_SCALER_PATH, "rb") as f:
    input_scaler = pickle.load(f)

X_scaled = input_scaler.transform(X_original)


# =============================================================================
# 4. DEFINE PREDICTION FUNCTION FOR SHAP
# =============================================================================

def predict_real_resistance(X_scaled_np):
    """
    SHAP sends scaled input values.
    The model output is scaled as:
        y_scaled = y_real / max_y * 10

    Therefore:
        y_real = y_scaled / 10 * max_y
    """
    X_tensor = torch.tensor(X_scaled_np, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        y_scaled_pred = model(X_tensor).cpu().numpy().flatten()

    y_real_pred = (y_scaled_pred / 10.0) * max_y
    return y_real_pred


# =============================================================================
# 5. SELECT BACKGROUND AND EXPLANATION DATA
# =============================================================================

rng = np.random.default_rng(REPRESENTATIVE_SEED)

n_samples = X_scaled.shape[0]

background_idx = rng.choice(
    n_samples,
    size=min(BACKGROUND_SIZE, n_samples),
    replace=False,
)

explain_idx = rng.choice(
    n_samples,
    size=min(EXPLAIN_SIZE, n_samples),
    replace=False,
)

X_background_scaled = X_scaled[background_idx]
X_explain_scaled = X_scaled[explain_idx]

X_explain_original = X_original[explain_idx]


# =============================================================================
# 6. COMPUTE SHAP VALUES
# =============================================================================

masker = shap.maskers.Independent(X_background_scaled)

explainer = shap.Explainer(
    predict_real_resistance,
    masker,
    feature_names=INPUT_COLUMNS,
)

shap_values = explainer(X_explain_scaled)

# Replace scaled feature values with original physical feature values
# so the beeswarm color bar shows actual feature magnitudes.
shap_values.data = X_explain_original


# =============================================================================
# 7. BEESWARM PLOT
# =============================================================================

plt.figure()
shap.plots.beeswarm(
    shap_values,
    max_display=len(INPUT_COLUMNS),
    show=False,
)

plt.title("SHAP Beeswarm Plot - Representative Best DNN")
plt.tight_layout()
plt.savefig(SHAP_BEESWARM_PATH, dpi=300, bbox_inches="tight")
plt.show()


# =============================================================================
# 8. OPTIONAL: SHAP BAR PLOT
# =============================================================================

plt.figure()
shap.plots.bar(
    shap_values,
    max_display=len(INPUT_COLUMNS),
    show=False,
)

plt.title("Mean Absolute SHAP Values - Representative Best DNN")
plt.tight_layout()
plt.savefig(SHAP_BAR_PATH, dpi=300, bbox_inches="tight")
plt.show()


print("DONE")
print(f"SHAP beeswarm saved to: {SHAP_BEESWARM_PATH}")
print(f"SHAP bar plot saved to: {SHAP_BAR_PATH}")


corr = df[INPUT_COLUMNS].corr()

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()