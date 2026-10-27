# Experimental CWC — Pearson correlation matrix

`correlation_matrix.py` computes the Pearson correlation matrix for the eight input variables and experimental resistance (`F_exp`), and saves an annotated heatmap and correlation coefficients.

The dataset is loaded from `data/study_03_data.xlsx` in the repository root. The script first retains complete cases for the nine analysis columns and then removes the single observation with the largest `F_exp`, following the supplied original correlation analysis. **This data-selection rule must be considered when comparing with other Study 3 analyses.**

Run from the repository root:

```bash
python 03_Experimental_CWC/01_Database_Analysis/01_Correlation_Matrix/correlation_matrix.py
```

Results are written to `outputs/`. contains the original supplied figure, not a figure regenerated during code preparation.
