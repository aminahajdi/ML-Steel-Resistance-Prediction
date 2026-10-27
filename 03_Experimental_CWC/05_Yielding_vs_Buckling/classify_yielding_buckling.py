# Eurocode-based resistance-mode classification. Requires a locally supplied experimental dataset.
import pandas as pd
import numpy as np
from pathlib import Path

GAMMA_M0 = 1.0
GAMMA_M1 = 1.0

script_dir = Path(__file__).resolve().parent
repo_root = Path(__file__).resolve().parents[2]

input_file = repo_root / "data" / "study_03_data.xlsx"

output_dir = script_dir / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

output_all = output_dir / "CWC_database_with_EC.xlsx"
output_yielding = output_dir / "CWC_EC_yielding.xlsx"
output_buckling = output_dir / "CWC_EC_buckling.xlsx"

# =============================================================================
# READ EXCEL
# =============================================================================

df = pd.read_excel(input_file)
df.columns = df.columns.str.strip()

numeric_cols = [
    "h_c", "t_wc", "b_fc", "t_fc", "r_c",
    "a_b", "t_fb", "E", "f_y", "N/Npl", "F_exp"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(
        df[col].astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )

df = df.dropna(subset=[
    "h_c", "t_wc", "t_fc", "r_c",
    "a_b", "t_fb", "E", "f_y", "N/Npl"
]).copy()

# =============================================================================
# EUROCODE CALCULATION
# =============================================================================

df["s"] = df["r_c"]

df["b_eff_w"] = (
    df["t_fb"]
    + 2 * np.sqrt(2) * df["a_b"]
    + 5 * (df["t_fc"] + df["s"])
)

df["d_wc"] = df["h_c"] - 2 * (df["t_fc"] + df["r_c"])

df["omega"] = 1.0

df["k_wc"] = np.minimum(1.0, 1.7 - df["N/Npl"])
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
        df["E"] * df["t_wc"]**2
    )
)

df["rho"] = np.where(
    df["lambda_p"] <= 0.673,
    1.0,
    (df["lambda_p"] - 0.22) / df["lambda_p"]**2
)

df["rho"] = df["rho"].clip(lower=0.0, upper=1.0)

df["F_EC_yield_N"] = (
    df["omega"]
    * df["k_wc"]
    * df["b_eff_w"]
    * df["t_wc"]
    * df["f_y"]
    / GAMMA_M0
)

df["F_EC_buckling_N"] = (
    df["omega"]
    * df["k_wc"]
    * df["rho"]
    * df["b_eff_w"]
    * df["t_wc"]
    * df["f_y"]
    / GAMMA_M1
)

df["F_EC_N"] = np.minimum(
    df["F_EC_yield_N"],
    df["F_EC_buckling_N"]
)

df["F_EC_yield_kN"] = df["F_EC_yield_N"] / 1000
df["F_EC_buckling_kN"] = df["F_EC_buckling_N"] / 1000
df["F_EC_kN"] = df["F_EC_N"] / 1000

df["governing_mode"] = np.where(
    df["F_EC_yield_N"] <= df["F_EC_buckling_N"],
    "yielding",
    "buckling"
)

# Optional: EC / experimental ratio
df["F_EC_over_F_exp"] = df["F_EC_kN"] / df["F_exp"]

# =============================================================================
# SPLIT INTO TWO EXCEL FILES
# =============================================================================

df_yielding = df[df["governing_mode"] == "yielding"].copy()
df_buckling = df[df["governing_mode"] == "buckling"].copy()

df.to_excel(output_all, index=False)
df_yielding.to_excel(output_yielding, index=False)
df_buckling.to_excel(output_buckling, index=False)

print("Finished.")
print(f"Total rows: {len(df)}")
print(f"Yielding rows: {len(df_yielding)}")
print(f"Buckling rows: {len(df_buckling)}")

print(f"All results saved to: {output_all}")
print(f"Yielding saved to: {output_yielding}")
print(f"Buckling saved to: {output_buckling}")