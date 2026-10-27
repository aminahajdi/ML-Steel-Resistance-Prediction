# =============================================================================
# SAVE SHAP SUMMARY FROM SAVED H2O MODEL
# For saved Stacked Ensemble model
# =============================================================================

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import h2o


# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

REPO_ROOT = BASE_DIR.parents[2]
DATA_FILE = REPO_ROOT / "data" / "study_04_data.xlsx"

RESULTS_DIR = BASE_DIR / "outputs" / "seed_42"

MODEL_DIR = RESULTS_DIR / "H2OModels"

OUTPUT_COLUMN = "moment resistance"

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

SEED = 42
TEST_SIZE = 0.20

SHAP_PLOT_PATH = RESULTS_DIR / "h2o_shap_summary_saved_model_seed42.png"


def main():
    # =============================================================================
    # 1. START H2O
    # =============================================================================

    h2o.init(
        nthreads=-1,
        max_mem_size="4G",
    )


    # =============================================================================
    # 2. LOAD AND CLEAN DATA
    # =============================================================================

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"H2O model folder not found: {MODEL_DIR}")

    df = pd.read_excel(DATA_FILE)

    df = df[INPUT_COLUMNS + [OUTPUT_COLUMN]].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[df[OUTPUT_COLUMN] > 0].copy()


    # =============================================================================
    # 3. RECREATE SAME TRAIN / TEST SPLIT
    # =============================================================================

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=SEED,
        shuffle=True,
    )

    train_h2o = h2o.H2OFrame(train_df)
    test_h2o = h2o.H2OFrame(test_df)

    train_h2o[OUTPUT_COLUMN] = train_h2o[OUTPUT_COLUMN].asnumeric()
    test_h2o[OUTPUT_COLUMN] = test_h2o[OUTPUT_COLUMN].asnumeric()


    # =============================================================================
    # 4. LOAD SAVED H2O MODEL
    # =============================================================================

    model_manifest = RESULTS_DIR / "selected_model_path.txt"
    if model_manifest.exists():
        model_path = Path(model_manifest.read_text(encoding="utf-8").strip())
        if not model_path.exists():
            model_path = MODEL_DIR / model_path.name
    else:
        model_files = [p for p in MODEL_DIR.iterdir() if p.is_file()]
        if len(model_files) != 1:
            raise RuntimeError(
                "Model identity is ambiguous. Provide selected_model_path.txt "
                "from the H2O benchmark run."
            )
        model_path = model_files[0]
    if not model_path.is_file():
        raise FileNotFoundError(f"Saved H2O model not found: {model_path}")

    model = h2o.load_model(str(model_path))

    print("====================================================")
    print("LOADED H2O MODEL")
    print("====================================================")
    print(f"Model path: {model_path}")
    print(f"Model ID:   {model.model_id}")


    # =============================================================================
    # 5. SAVE SHAP SUMMARY PLOT
    # =============================================================================

    try:
        print("\nGenerating SHAP summary plot with background frame...")

        model.shap_summary_plot(
            test_h2o,
            top_n_features=15,
            background_frame=train_h2o,
        )

        plt.tight_layout()
        plt.gcf().savefig(SHAP_PLOT_PATH, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"SHAP plot saved to: {SHAP_PLOT_PATH}")

    except TypeError as e:
        print("This H2O version may not accept background_frame in shap_summary_plot.")
        print(f"Reason: {e}")

    except Exception as e:
        print("SHAP plot could not be generated or saved.")
        print(f"Reason: {e}")


    # =============================================================================
    # 6. DONE
    # =============================================================================

    h2o.cluster().shutdown(prompt=False)

if __name__ == "__main__":
    main()
