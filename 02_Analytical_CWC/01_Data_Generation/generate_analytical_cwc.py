"""Generate the analytical CWC resistance database used in Study 2.

Sample steel sections, steel grades, axial-force ratios and beam-flange
thicknesses, then evaluate the resistance expressions implemented below.

The source cross-section workbook is supplied with the code. Model-training
datasets and later filtering are handled separately from this generator.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# 1. EXPERIMENT CONFIGURATION
# =============================================================================

N_CALCULATIONS = 10000

GAMMA_M0 = 1.0
GAMMA_M1 = 1.0

A_B = 0.0           # mm
E_MODULUS = 210000  # MPa

RANDOM_SEED = 42

# =============================================================================
# 2. AXIAL-FORCE RATIOS (N/Npl)
# =============================================================================

AXIAL_FORCE_LEVELS = [
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
]

# =============================================================================
# 3. BEAM-FLANGE THICKNESS SAMPLING RANGE
# =============================================================================

T_FB_MIN_RATIO = 0.70
T_FB_MAX_RATIO = 1.30

# =============================================================================
# 4. STEEL GRADES AND NOMINAL YIELD STRENGTHS (MPa)
# =============================================================================

STEEL_GRADES = {
    "S235": 235.0,
    "S275": 275.0,
    "S355": 355.0,
    "S420": 420.0,
    "S460": 460.0,
    "S500": 500.0,
    "S550": 550.0,
    "S620": 620.0,
    "S690": 690.0,
    "S700": 700.0,
}

# Paths are resolved relative to this script, not the terminal working directory.
script_dir = Path(__file__).resolve().parent
input_file = script_dir / "CWC_inputs_2.xlsx"
output_dir = script_dir / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "analytical_cwc_generated_10000.xlsx"


# =============================================================================
# 5. LOAD SOURCE CROSS-SECTIONS
# =============================================================================

df_sections = pd.read_excel(input_file)
df_sections.columns = df_sections.columns.str.strip()

for col in ["h_c", "t_wc", "b_fc", "t_fc", "r_c"]:
    df_sections[col] = pd.to_numeric(
        df_sections[col].astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )

df_sections = df_sections.dropna(
    subset=["h_c", "t_wc", "b_fc", "t_fc", "r_c"]
)


# =============================================================================
# 6. SAMPLE CROSS-SECTIONS WITH REPLACEMENT
# =============================================================================

rng = np.random.default_rng(RANDOM_SEED)

random_sections = df_sections.sample(
    n=N_CALCULATIONS,
    replace=True,
    random_state=RANDOM_SEED
).reset_index(drop=True)

# =============================================================================
# 7. ASSIGN STEEL GRADES
# =============================================================================

steel_names = np.array(list(STEEL_GRADES.keys()))

random_steel_names = rng.choice(
    steel_names,
    size=N_CALCULATIONS,
    replace=True
)

random_sections["steel_grade"] = random_steel_names

random_sections["f_y"] = random_sections["steel_grade"].map(
    STEEL_GRADES
)

df = random_sections.copy()

# =============================================================================
# 8. ASSIGN CONSTANTS
# =============================================================================

df["a_b"] = A_B
df["E"] = E_MODULUS

# =============================================================================
# 9. ASSIGN AXIAL-FORCE RATIOS
# =============================================================================
# Sample N/Npl independently from the predefined levels.

df["N/Npl"] = rng.choice(
    AXIAL_FORCE_LEVELS,
    size=N_CALCULATIONS,
    replace=True
)

# =============================================================================
# 10. SAMPLE BEAM-FLANGE THICKNESS t_fb
# =============================================================================
# Sample t_fb from available cross-section flange thicknesses within the
# specified ratio range relative to the chosen column flange thickness.

available_tfc_values = np.sort(
    df_sections["t_fc"].dropna().unique()
)

random_tfb_values = []

for t_fc_value in df["t_fc"]:

    lower_limit = T_FB_MIN_RATIO * t_fc_value
    upper_limit = T_FB_MAX_RATIO * t_fc_value

    possible_tfb_values = available_tfc_values[
        (available_tfc_values >= lower_limit) &
        (available_tfc_values <= upper_limit)
    ]

    if len(possible_tfb_values) == 0:

        closest_value = available_tfc_values[
            np.argmin(
                np.abs(
                    available_tfc_values - t_fc_value
                )
            )
        ]

        random_tfb_values.append(closest_value)

    else:

        random_tfb_values.append(
            rng.choice(possible_tfb_values)
        )

df["t_fb"] = random_tfb_values


# =============================================================================
# 11. ANALYTICAL CWC RESISTANCE PARAMETERS
# =============================================================================

df["s"] = df["r_c"]

df["b_eff_w"] = (
    df["t_fb"]
    + 2 * np.sqrt(2) * df["a_b"]
    + 5 * (df["t_fc"] + df["s"])
)

df["d_wc"] = (
    df["h_c"]
    - 2 * (df["t_fc"] + df["r_c"])
)

df["beta"] = 0.0
df["omega"] = 1.0

df["k_wc"] = np.minimum(
    1.0,
    1.7 - df["N/Npl"]
)

df["k_wc"] = df["k_wc"].clip(lower=0.0)

df["lambda_p"] = 0.932 * np.sqrt(
    (
        df["omega"]
        * df["k_wc"]
        * df["b_eff_w"]
        * df["d_wc"]
        * df["f_y"]
    )
    /
    (
        df["E"]
        * df["t_wc"]**2
    )
)

df["rho"] = np.where(
    df["lambda_p"] <= 0.673,
    1.0,
    (df["lambda_p"] - 0.22)
    / (df["lambda_p"]**2)
)

df["rho"] = df["rho"].clip(
    lower=0.0,
    upper=1.0
)

# =============================================================================
# 12. YIELDING AND BUCKLING RESISTANCES (N)
# =============================================================================

df["F_c_wc_Rd_yield_N"] = (
    df["omega"]
    * df["k_wc"]
    * df["b_eff_w"]
    * df["t_wc"]
    * df["f_y"]
    / GAMMA_M0
)

df["F_c_wc_Rd_buckling_N"] = (
    df["omega"]
    * df["k_wc"]
    * df["rho"]
    * df["b_eff_w"]
    * df["t_wc"]
    * df["f_y"]
    / GAMMA_M1
)

df["F_c_wc_Rd_N"] = np.minimum(
    df["F_c_wc_Rd_yield_N"],
    df["F_c_wc_Rd_buckling_N"]
)

# =============================================================================
# 13. CONVERT RESISTANCES TO kN
# =============================================================================

df["F_c_wc_Rd_yield_kN"] = (
    df["F_c_wc_Rd_yield_N"] / 1000
)

df["F_c_wc_Rd_buckling_kN"] = (
    df["F_c_wc_Rd_buckling_N"] / 1000
)

df["F_c_wc_Rd_kN"] = (
    df["F_c_wc_Rd_N"] / 1000
)

# =============================================================================
# 14. IDENTIFY GOVERNING RESISTANCE
# =============================================================================

df["governing_mode"] = np.where(
    df["F_c_wc_Rd_yield_N"]
    <=
    df["F_c_wc_Rd_buckling_N"],
    "yielding",
    "buckling"
)

# =============================================================================
# 15. EXPORT GENERATED DATABASE
# =============================================================================

df.insert(
    0,
    "sample_id",
    range(1, len(df) + 1)
)

columns_to_save = [
    "sample_id",
    "number",
    "Section",
    "steel_grade",
    "h_c",
    "t_wc",
    "b_fc",
    "t_fc",
    "r_c",
    "a_b",
    "t_fb",
    "E",
    "N/Npl",
    "f_y",
    "s",
    "b_eff_w",
    "d_wc",
    "beta",
    "omega",
    "k_wc",
    "lambda_p",
    "rho",
    "F_c_wc_Rd_yield_kN",
    "F_c_wc_Rd_buckling_kN",
    "F_c_wc_Rd_kN",
    "governing_mode"
]

df = df[columns_to_save]

df.to_excel(
    output_file,
    index=False
)

# =============================================================================
# 16. REPORT GENERATION SUMMARY
# =============================================================================

print("====================================================")
print("FINISHED")
print("====================================================")

print(f"Number of calculations: {N_CALCULATIONS}")
print(f"Results saved to: {output_file}")

print()
print("Random axial force levels:")
print(AXIAL_FORCE_LEVELS)

print()
print("t_fb generation:")
print(
    f"{T_FB_MIN_RATIO} * t_fc <= t_fb <= {T_FB_MAX_RATIO} * t_fc"
)

print()
print(df[[
    "Section",
    "steel_grade",
    "h_c",
    "t_wc",
    "t_fc",
    "r_c",
    "f_y",
    "N/Npl",
    "t_fb",
    "lambda_p",
    "rho",
    "F_c_wc_Rd_kN",
    "governing_mode"
]].head())