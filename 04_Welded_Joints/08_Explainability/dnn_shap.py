# =============================================================================
# SHAP EXPLAINABILITY FOR FIXED DNN - JOINT DATASET - MOMENT RESISTANCE
# =============================================================================
# Purpose:
# - Load the already trained fixed DNN model.
# - Load the saved input scaler.
# - Rebuild the same DNN architecture.
# - Compute SHAP values for the DNN using KernelExplainer.
# - Save SHAP beeswarm, bar plot, and mean absolute SHAP table.
#
# Important:
# - First run 04_Welded_Joints/02_Fixed_DNN/fixed_dnn.py.
# - This analysis loads that model and its fitted input scaler.
#
# - The model uses output scaling:
#     y_scaled = (y_real / max_y_train) * 10
#   Therefore predictions are converted back to real moment resistance units.
#
# - KernelExplainer can be computationally demanding. Settings:
#     BACKGROUND_SIZE = 100
#     EXPLAIN_SIZE = 500
# =============================================================================

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split

try:
    import shap
except ImportError as exc:
    raise ImportError(
        "The SHAP package is not installed. Install it with:\n\n"
        "pip install shap\n\n"
        "Then run this script again."
    ) from exc


# =============================================================================
# 1. REPOSITORY PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[2]  # 04_Welded_Joints/08_Explainability/SHAP
DATA_FILE = REPO_ROOT / "data" / "study_04_data.xlsx"

# The saved model and scaler are produced by 02_Fixed_DNN/fixed_dnn.py.
MODEL_RESULTS_DIR = BASE_DIR.parents[1] / "02_Fixed_DNN" / "outputs"
MODEL_PATH = MODEL_RESULTS_DIR / "fixed_dnn_model_state_dict.pt"
SCALER_PATH = MODEL_RESULTS_DIR / "fixed_dnn_input_scaler.pkl"

RESULTS_DIR = BASE_DIR / "outputs"
SHAP_VALUES_CSV_PATH = RESULTS_DIR / "shap_values.csv"
SHAP_MEAN_ABS_CSV_PATH = RESULTS_DIR / "shap_mean_absolute_importance.csv"
SHAP_EXPLAINED_DATA_CSV_PATH = RESULTS_DIR / "shap_explained_samples.csv"
SHAP_BEESWARM_PATH = RESULTS_DIR / "shap_beeswarm.png"
SHAP_BAR_PATH = RESULTS_DIR / "shap_bar_mean_abs.png"
SHAP_WATERFALL_PATH = RESULTS_DIR / "shap_waterfall_first_sample.png"
SUMMARY_TXT_PATH = RESULTS_DIR / "shap_summary.txt"

# =============================================================================
# 2. MODEL AND DATA SETTINGS
# =============================================================================

SEED = 42
TEST_SIZE = 0.20

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

ARCHITECTURE = (64, 32, 16)
ACTIVATION_NAME = "GELU"

# SHAP settings.
# Increase EXPLAIN_SIZE if you want more samples, but it will be slower.
BACKGROUND_SIZE = 100
EXPLAIN_SIZE = 500

# Number of samples used by KernelExplainer for each explained point.
# A larger value is more accurate but slower.
NSAMPLES = 300

DEVICE = torch.device("cpu")
torch.set_num_threads(1)


# =============================================================================
# 3. HELPER FUNCTIONS
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


def make_prediction_function(model, input_scaler, max_y):
    """Return a prediction function in physical moment-resistance units."""
    model.eval()

    def predict_real_from_unscaled_input(X_unscaled_np):
        X_scaled = input_scaler.transform(np.asarray(X_unscaled_np, dtype=float))
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            y_pred_scaled = model(X_tensor).cpu().numpy().reshape(-1)
        return (y_pred_scaled / 10.0) * max_y

    return predict_real_from_unscaled_input


def make_safe_array(shap_values):
    """
    SHAP can return either:
    - array with shape (n_samples, n_features)
    - list containing one array
    - array with an extra output dimension

    This function converts it to shape (n_samples, n_features).
    """
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]

    return shap_values


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # =============================================================================
    # 4. LOAD DATA
    # =============================================================================

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Study 4 dataset not found: {DATA_FILE}")

    df = pd.read_excel(DATA_FILE)

    missing_input_columns = [col for col in INPUT_COLUMNS if col not in df.columns]

    if missing_input_columns:
        raise ValueError(f"Missing input columns in dataset: {missing_input_columns}")

    if OUTPUT_COLUMN not in df.columns:
        raise ValueError(f"Output column not found in dataset: {OUTPUT_COLUMN}")

    df = df[INPUT_COLUMNS + [OUTPUT_COLUMN]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    df = df[df[OUTPUT_COLUMN] > 0].copy()

    X_df = df[INPUT_COLUMNS].copy()
    y_np = df[OUTPUT_COLUMN].to_numpy(dtype=float)
    X_np = X_df.to_numpy(dtype=float)


    # =============================================================================
    # 5. LOAD MODEL AND SCALER
    # =============================================================================

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Saved model was not found:\n{MODEL_PATH}\n\n"
            "Run the fixed DNN script first, or update MODEL_PATH in this SHAP script."
        )

    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Saved input scaler was not found:\n{SCALER_PATH}\n\n"
            "Run the fixed DNN script first, or update SCALER_PATH in this SHAP script."
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    # Rebuild the same DNN architecture used during training.
    model = RegressionModel(
        num_features=len(INPUT_COLUMNS),
        architecture=tuple(checkpoint.get("architecture", ARCHITECTURE)),
        activation_name=checkpoint.get("activation", ACTIVATION_NAME),
    ).to(DEVICE)

    # Load trained weights.
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load the fitted input scaler.
    input_scaler = joblib.load(SCALER_PATH)

    # Load output-scaling value used to convert predictions back to real units.
    if "max_y" not in checkpoint:
        raise KeyError(
            "The saved model checkpoint does not contain 'max_y'. "
            "The SHAP script needs max_y to convert scaled DNN predictions back to real moment resistance."
        )

    max_y = float(checkpoint["max_y"])

    # Validate the stored model's feature order before computing explanations.
    saved_input_columns = checkpoint.get("input_columns", INPUT_COLUMNS)
    if list(saved_input_columns) != INPUT_COLUMNS:
        raise ValueError("Input column order does not match the saved DNN checkpoint.")
    if checkpoint.get("output_column", OUTPUT_COLUMN) != OUTPUT_COLUMN:
        raise ValueError("The DNN checkpoint has a different prediction target.")

    predict_real_from_unscaled_input = make_prediction_function(
        model, input_scaler, max_y
    )

    # =============================================================================
    # 6. SELECT BACKGROUND AND EXPLAINED SAMPLES
    # =============================================================================

    # Use the same 80:20 split as the fixed DNN seed-42 script.
    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X_np,
        y_np,
        test_size=TEST_SIZE,
        random_state=SEED,
        shuffle=True,
    )

    background_size = min(BACKGROUND_SIZE, len(X_train_np))
    explain_size = min(EXPLAIN_SIZE, len(X_test_np))

    rng = np.random.default_rng(SEED)

    background_idx = rng.choice(len(X_train_np), size=background_size, replace=False)
    explain_idx = rng.choice(len(X_test_np), size=explain_size, replace=False)

    X_background = X_train_np[background_idx]
    X_explain = X_test_np[explain_idx]

    X_explain_df = pd.DataFrame(X_explain, columns=INPUT_COLUMNS)


    # =============================================================================
    # 7. COMPUTE SHAP VALUES
    # =============================================================================

    print("====================================================")
    print("SHAP EXPLAINABILITY FOR FIXED DNN")
    print("====================================================")
    print(f"Dataset file:         {DATA_FILE}")
    print(f"Model path:           {MODEL_PATH}")
    print(f"Scaler path:          {SCALER_PATH}")
    print(f"Total samples:        {len(X_np)}")
    print(f"Background samples:   {background_size}")
    print(f"Explained samples:    {explain_size}")
    print(f"SHAP nsamples:        {NSAMPLES}")
    print(f"Loaded max_y:         {max_y:.6f}")
    print("Model and scaler loaded successfully.")
    print("This may take some time...")


    explainer = shap.KernelExplainer(
        model=predict_real_from_unscaled_input,
        data=X_background,
    )

    shap_values_raw = explainer.shap_values(
        X_explain,
        nsamples=NSAMPLES,
    )

    shap_values = make_safe_array(shap_values_raw)

    expected_value = explainer.expected_value

    if isinstance(expected_value, (list, np.ndarray)):
        expected_value_scalar = float(np.asarray(expected_value).reshape(-1)[0])
    else:
        expected_value_scalar = float(expected_value)


    # =============================================================================
    # 8. SAVE SHAP TABLES
    # =============================================================================

    shap_values_df = pd.DataFrame(
        shap_values,
        columns=INPUT_COLUMNS,
    )

    shap_values_df.to_csv(SHAP_VALUES_CSV_PATH, index=False)

    X_explain_df.to_csv(SHAP_EXPLAINED_DATA_CSV_PATH, index=False)

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

    importance_df = pd.DataFrame(
        {
            "feature": INPUT_COLUMNS,
            "mean_abs_shap": mean_abs_shap,
        }
    )

    importance_df = importance_df.sort_values("mean_abs_shap", ascending=False)
    importance_df["rank"] = np.arange(1, len(importance_df) + 1)

    importance_df = importance_df[
        [
            "rank",
            "feature",
            "mean_abs_shap",
        ]
    ]

    importance_df.to_csv(SHAP_MEAN_ABS_CSV_PATH, index=False)


    # =============================================================================
    # 9. SAVE SHAP PLOTS
    # =============================================================================

    # Beeswarm plot.
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_explain_df,
        feature_names=INPUT_COLUMNS,
        show=False,
        max_display=len(INPUT_COLUMNS),
    )
    plt.tight_layout()
    plt.savefig(SHAP_BEESWARM_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    # Bar plot.
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_explain_df,
        feature_names=INPUT_COLUMNS,
        plot_type="bar",
        show=False,
        max_display=len(INPUT_COLUMNS),
    )
    plt.tight_layout()
    plt.savefig(SHAP_BAR_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    # Waterfall plot for first explained sample.
    try:
        explanation = shap.Explanation(
            values=shap_values[0],
            base_values=expected_value_scalar,
            data=X_explain_df.iloc[0].values,
            feature_names=INPUT_COLUMNS,
        )

        plt.figure()
        shap.plots.waterfall(
            explanation,
            max_display=len(INPUT_COLUMNS),
            show=False,
        )
        plt.tight_layout()
        plt.savefig(SHAP_WATERFALL_PATH, dpi=300, bbox_inches="tight")
        plt.close()

    except Exception as exc:
        print(f"Waterfall plot could not be saved: {exc}")


    # =============================================================================
    # 10. SAVE SUMMARY
    # =============================================================================

    with open(SUMMARY_TXT_PATH, "w", encoding="utf-8") as f:
        f.write("SHAP EXPLAINABILITY SUMMARY - FIXED DNN - JOINT DATASET\n")
        f.write("=======================================================\n\n")

        f.write("MODEL AND DATA\n")
        f.write("==============\n")
        f.write(f"Dataset file: {DATA_FILE}\n")
        f.write(f"Model path: {MODEL_PATH}\n")
        f.write(f"Scaler path: {SCALER_PATH}\n")
        f.write(f"Total samples: {len(X_np)}\n")
        f.write(f"Background samples: {background_size}\n")
        f.write(f"Explained samples: {explain_size}\n")
        f.write(f"SHAP nsamples: {NSAMPLES}\n")
        f.write(f"Expected value: {expected_value_scalar:.6f}\n\n")

        f.write("INPUT COLUMNS\n")
        f.write("=============\n")
        f.write(str(INPUT_COLUMNS))
        f.write("\n\n")

        f.write("FEATURE IMPORTANCE BY MEAN ABSOLUTE SHAP VALUE\n")
        f.write("==============================================\n")
        f.write(importance_df.to_string(index=False))
        f.write("\n\n")

        f.write("NOTES\n")
        f.write("=====\n")
        f.write("- SHAP values are computed using the DNN prediction function in real moment-resistance units.\n")
        f.write("- The bar plot ranks features by mean absolute SHAP magnitude.\n")
        f.write("- The beeswarm plot shows both magnitude and direction of feature influence.\n")
        f.write("- KernelExplainer is model-agnostic but can be slow; increase EXPLAIN_SIZE only if needed.\n")


    print("\n====================================================")
    print("DONE")
    print("====================================================")
    print(f"SHAP values saved to: {SHAP_VALUES_CSV_PATH}")
    print(f"Mean absolute SHAP table saved to: {SHAP_MEAN_ABS_CSV_PATH}")
    print(f"Explained samples saved to: {SHAP_EXPLAINED_DATA_CSV_PATH}")
    print(f"SHAP beeswarm saved to: {SHAP_BEESWARM_PATH}")
    print(f"SHAP bar plot saved to: {SHAP_BAR_PATH}")
    print(f"SHAP waterfall saved to: {SHAP_WATERFALL_PATH}")
    print(f"Summary saved to: {SUMMARY_TXT_PATH}")
    print(f"Results folder: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
