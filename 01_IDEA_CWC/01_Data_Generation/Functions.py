from pathlib import Path
from typing import List, Dict, Union, Tuple
from ideastatica_connection_api import ConMprlCrossSection, ConMember, MemberApi, MaterialApi, ConMprlElement, ConLoadEffect
from ideastatica_connection_api.connection_api_client import ConnectionApiClient
import uuid
from openpyxl import load_workbook
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json

"""
Reads a CSV file and retrieves the row matching the specified material name.

Args:
    file_name: Path to the CSV file
Returns:
    List of strings representing the matched row

Raises:
    ValueError: If material is not found
    FileNotFoundError: If CSV file doesn't exist
"""
def read_csv(file_name: Path) -> List[str]:
    import csv
    css_list:list[str] = []

    try:
        with open(file_name, mode='r', encoding='utf-8-sig') as file:
            reader = csv.reader(file, delimiter=';')

            # Skip first 4 rows (header + units)
            for _ in range(4):
                next(reader, None)

            # Search for material in remaining rows
            for row in reader:
                if not row:  # Skip empty rows
                    continue
                css_list.append(row[0])

            return css_list

    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {file_name}")
    
def change_cross_section(api_member:MemberApi, project_id:str, project_item:str, api_material:MaterialApi, cross_section:dict) -> ConMember:

    #get all members from the project item
    members:list[ConMember] = api_member.get_members(project_id, project_item)

    # define new cross-section
    chord_css_set:ConMprlCrossSection = ConMprlCrossSection(
        materialName=cross_section["new_cs_steel"],
        mprlName=cross_section["new_cs"]
    )

    # add new cross-section to the project
    cross_section__ = json.loads(api_material.add_cross_section(project_id, chord_css_set))
    ci_id = cross_section__['id']

    # get member id of the desired beam
    chord_member:list[ConMember] = [item for item in members if item.name == cross_section["beam_name"]]

    # set new cross-section for the desired beam
    chord_member[0].cross_section_id = ci_id

    # update cross-section of the desired beam
    update_css:ConMember = api_member.update_member(project_id, project_item, chord_member[0])

    return update_css


def read_xlsx(file_name: Union[str, Path]) -> List[str]:
    """
    Reads an XLSX file and retrieves the values from the first column (column A),
    skipping the first 4 rows (e.g., header + units).

    Args:
        file_name: Path or string pointing to the XLSX file

    Returns:
        List of strings from the first column (excluding the first 4 rows)

    Raises:
        ValueError: If there is an error reading the file
        FileNotFoundError: If the file does not exist
    """
    file_path = Path(file_name)  # Ensure it's a Path object

    if not file_path.exists():
        raise FileNotFoundError(f"XLSX file not found: {file_path}")

    try:
        wb = load_workbook(filename=file_path, data_only=True)
        sheet = wb.active  # Use the first (active) worksheet
        css_list: List[str] = []

        # Skip first 4 rows (1-based indexing)
        for row in sheet.iter_rows(min_row=5, min_col=1, max_col=1):
            cell = row[0]
            if cell.value is not None:
                css_list.append(str(cell.value))

        return css_list

    except Exception as e:
        raise ValueError(f"Error reading XLSX file: {e}")
    
def get_section_property(file_path: Union[str, Path], section_name: str, property_name: str) -> Tuple[float, str]:
    """
    Reads an XLSX file and retrieves a specified property value and its unit for a given section.

    Args:
        file_path: Path to the Excel file
        section_name: Name of the section (e.g., 'IPE80')
        property_name: Column header of the property (e.g., 'A', 'Iz')

    Returns:
        Tuple (value as float, unit as string)

    Raises:
        ValueError: If section or property not found

    Usage:
    value, unit = get_section_property("Cross-sections.xlsx", "IPE80", "A")
    print(f"A of IPE80: {value} {unit}")
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    wb = load_workbook(file_path, data_only=True)
    sheet = wb.active

    # Get headers from row 1
    headers = [cell.value for cell in sheet[1]]
    if property_name not in headers:
        raise ValueError(f"Property '{property_name}' not found in header row.")

    col_index = headers.index(property_name) + 1  # openpyxl is 1-based indexing

    # Get unit string from row 2
    unit = sheet.cell(row=2, column=col_index).value or ""

    # Iterate over data rows starting at row 4
    for row in sheet.iter_rows(min_row=4):
        section_cell = row[0].value
        if section_cell == section_name:
            value_cell = sheet.cell(row=row[0].row, column=col_index).value
            if value_cell is None:
                raise ValueError(f"No data found for section '{section_name}' and property '{property_name}'")
            return float(value_cell), unit

    raise ValueError(f"Section '{section_name}' not found in data.")


# Relative squared error loss function
# Usage: 
# criterion = RelativeSquaredError()

class RelativeSquaredError(nn.Module):
    def __init__(self, epsilon=0): #y_true must not be 0
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        relative_error = (y_true - y_pred) / (y_true + self.epsilon)
        return torch.mean(relative_error ** 2)


# Scaler for input and output data
# module-level scalers (will be set by scale_data)
def scale_data(X, y, use_input_scaler=False, use_output_scaler=False,
               use_log_output=False, use_similitude_scaling=False,
               scaling_factors=None, reference_length=200.0):
    """
    X: 2D numpy array (n_samples, n_features)
    y: 1D numpy array (n_samples,) or 2D (n_samples,1)
    use_similitude_scaling: apply scaling according to similitude theory
    scaling_factors: array-like of length n_features+1
                     (last element is for y, first n are for X)
    reference_length: reference cross-section length [mm] for scaling
    Returns: X_scaled (2D), y_scaled (1D), input_scaler, output_scaler
    """

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)  # ensure 1D

    # --- Similitude scaling ---
    if use_similitude_scaling:
        scaling_factors = np.asarray(scaling_factors, dtype=float)
        if len(scaling_factors) != X.shape[1] + 1:
            raise ValueError("scaling_factors length must be n_features + 1")

        scaling_factors = np.asarray(scaling_factors, dtype=float)
        X_factors = scaling_factors[:-1]
        y_factor = scaling_factors[-1]

        D_orig = X[:, 0].copy()
        L = D_orig / reference_length

        # Scale all columns according to factor (except D)
        for i in range(X.shape[1]):
            if i == 0:
                continue  # D will be forced later
            X[:, i] /= L ** X_factors[i]

        # Force D to reference length
        X[:, 0] = reference_length

        # Scale output
        y /= L ** y_factor

    # --- Log transform of output ---
    if use_log_output:
        y = np.log(y)

    # --- Standard input scaling ---
    if use_input_scaler:
        _input_scaler = StandardScaler()
        X = _input_scaler.fit_transform(X)
    else:
        _input_scaler = None


    # --- Custom output scaling ---
    if use_output_scaler:
        max_y = np.max(y)
        y = (y / max_y) * 10
        _output_scaler = {"max_y": max_y}
    else:
        _output_scaler = None

    return X, y, _input_scaler, _output_scaler


#The order of descale operations must be reverse to scale operations
def descale_data(_output_scaler, y_pred, use_output_scaler=False, use_log_output=False,
                 use_similitude_scaling=False, scaling_factors=None,
                 reference_length=200.0, X_original=None):

    y = np.asarray(y_pred).reshape(-1)

    # --- Undo custom output scaling ---
    if use_output_scaler:
        max_y = _output_scaler["max_y"]
        y = (y / 10) * max_y

    # --- Undo log transform ---
    if use_log_output:
        y = np.exp(y)

    # --- Undo similitude scaling ---
    if use_similitude_scaling:
        if scaling_factors is None:
            raise ValueError("scaling_factors must be provided for similitude descale")
        scaling_factors = np.asarray(scaling_factors, dtype=float)
        y_power = scaling_factors[-1]

        if X_original is not None:
            if isinstance(X_original, (pd.DataFrame, pd.Series)):
                X_arr = X_original.to_numpy()
            else:
                X_arr = np.asarray(X_original)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, 1)
            D_orig = X_arr[:, 0]
            L = D_orig / reference_length
        else:
            L = np.ones_like(y)

        y = y * (L ** y_power)

    return y



# Removes all data points with values higher than sparsely populated bin's range.
def remove_sparse_bins(df, col, bins=30, count_threshold=5):
    # Compute histogram for the column
    counts, bin_edges = np.histogram(df[col], bins=bins)
    # Find bins with counts below threshold
    low_count_bins = np.where(counts < count_threshold)[0]
    if len(low_count_bins) == 0:
        # No sparse bins
        return df, None
    
    # Get first sparse bin
    first_low_bin = low_count_bins[0]
    cutoff_value = bin_edges[first_low_bin + 1]
    
    # Filter rows where col <= cutoff
    filtered_df = df[df[col] <= cutoff_value].copy()
    
    return filtered_df, cutoff_value
"""
# Example usage with your DataFrame 'df'
df = pd.read_csv('your_data.csv')  # Replace with your actual DataFrame loading

# Remove sparse tf bins
df_filtered_tf, tf_cutoff = remove_sparse_bins(df, 'tf', bins=30, count_threshold=5)
print(f"tf cutoff at: {tf_cutoff}")

# Remove sparse tw bins (on the already filtered df)
df_filtered_tw, tw_cutoff = remove_sparse_bins(df_filtered_tf, 'tw', bins=30, count_threshold=5)
print(f"tw cutoff at: {tw_cutoff}")

print(f"Rows before filtering: {len(df)}")
print(f"Rows after filtering tf and tw sparsity: {len(df_filtered_tw)}")
"""


def create_zoomed_plot(zoom, y_pred_all, y_true_all, y_pred_test, y_true_test, bias, cov, bias_all, cov_all, n_all, is_debugging, filename):
    """
    Creates a zoomed-in scatter plot of predicted vs true values for all and test data.
    The plot focuses on the lower part of the data range on both axes.

    Args:
        zoom (float): Zoom factor to set the limit for the axes (e.g., 0.20 for 20%).
        y_pred_test (np.ndarray): Predicted values for the test dataset.
        y_true_test (np.ndarray): True values for the test dataset.
        bias (float): Bias for the test data.
        cov (float): CoV for the test data.
        is_debugging (bool): Flag to control plot display.
        filename (str): The name of the file to save the plot to.
        y_pred_all (np.ndarray, optional): Predicted values for the entire dataset. Defaults to None.
        y_true_all (np.ndarray, optional): True values for the entire dataset. Defaults to None.
        bias_all (float, optional): Bias for all data. Defaults to None.
        cov_all (float, optional): CoV for all data. Defaults to None.
        n_all (int, optional): Total number of data points. Defaults to None.
    """
    # Create the figure and axes
    plt.figure(figsize=(4.5, 4.5))

    # Determine the values to use for calculating plot limits
    if y_pred_all is not None and y_true_all is not None:
        combined_values = np.concatenate((y_pred_all, y_true_all, y_pred_test, y_true_test))
    else:
        combined_values = np.concatenate((y_pred_test, y_true_test))

    # Calculate the plot limits based on the zoom factor
    max_val_in_data = np.max(combined_values)
    zoom_limit = max_val_in_data * zoom

    # Determine the range for the diagonal lines
    min_val = np.min(combined_values)
    max_val = np.max(combined_values)
    line_x = np.linspace(min_val, max_val, 100)

    # Conditionally plot the 'All data' scatter plot and legend
    if y_pred_all is not None and y_true_all is not None:
        plt.scatter(
            y_pred_all, y_true_all, alpha=0.6,
            label=f'All data\n$n={n_all}$\n$b={bias_all:.4f}$\nCoV={cov_all:.4f}',
            color='blue'
        )

    # Always plot the 'Test data' scatter plot and legend
    plt.scatter(
        y_pred_test, y_true_test, alpha=0.6,
        label=f'Test data\n$n={y_true_test.size}$\n$b={bias:.4f}$\nCoV={cov:.4f}',
        color='orange'
    )
    
    # Plot the ideal fit and error lines
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Ideal fit')
    plt.plot(line_x, 1.1 * line_x, 'r--')
    plt.plot(line_x, 0.9 * line_x, 'r--', label='+/-10% error')
    
    # Set the zoomed-in axes limits
    plt.xlim(0, zoom_limit * 1.05)
    plt.ylim(0, zoom_limit * 1.05)

    # Add labels, legend, and grid
    plt.xlabel('Predicted [kN]'); plt.ylabel('True [kN]')
    #plt.title('Prediction vs Actual (Zoomed)')
    plt.legend(loc='lower right'); plt.grid(True); plt.tight_layout()
    
    # Construct the full file path by joining the folder and filename
    full_path = os.path.join("Plots", filename)
    plt.savefig(full_path, dpi=300)
    if is_debugging: plt.show()



def create_histogram_plot(zoom, y_1, y_2, bias_1, cov_1, bias_2, cov_2, n_1, n_2, is_debugging, filename):
    """
    
Creates side-by-side histograms of two datasets with zoomed-in x-axis and annotated bias and CoV.

    Args:
        zoom (float): Zoom factor to set the limit for the x-axis (e.g., 0.20 for 20%).
        y_1 (np.ndarray): First dataset.
        y_2 (np.ndarray): Second dataset.
        bias_1 (float): Bias for the first dataset.
        cov_1 (float): CoV for the first dataset.
        bias_2 (float): Bias for the second dataset.
        cov_2 (float): CoV for the second dataset.
        n_1 (int): Number of data points in the first dataset.
        n_2 (int): Number of data points in the second dataset.
        is_debugging (bool): Flag to control plot display.
        filename (str): The name of the file to save the plot to.
    """
    # Create the figure and subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # Determine the zoom limit based on the maximum value in both datasets
    max_val_1 = np.max(y_1)
    min_val_1 = np.min(y_1)
    max_val_2 = np.max(y_2)
    min_val_2 = np.min(y_2)
    zoom_limit_max_1 = max_val_1 * zoom
    zoom_limit_min_1 = min_val_1 * zoom
    zoom_limit_max_2 = max_val_2 * zoom
    zoom_limit_min_2 = min_val_2 * zoom

    # Plot histogram for y_1
    axs[0].hist(y_1, bins=30, color='blue', alpha=0.7)
    axs[0].set_xlim(zoom_limit_min_1, zoom_limit_max_1)
    axs[0].set_title(f'Dataset 1\nn={n_1}, b={bias_1:.4f}, CoV={cov_1:.4f}')
    axs[0].set_xlabel('Value')
    axs[0].set_ylabel('Frequency')
    axs[0].grid(True)

    # Plot histogram for y_2
    axs[1].hist(y_2, bins=30, color='orange', alpha=0.7)
    axs[1].set_xlim(zoom_limit_min_2, zoom_limit_max_2)
    axs[1].set_title(f'Dataset 2\nn={n_2}, b={bias_2:.4f}, CoV={cov_2:.4f}')
    axs[1].set_xlabel('Value')
    axs[1].set_ylabel('Frequency')
    axs[1].grid(True)

    # Adjust layout and save the plot
    plt.tight_layout()
    full_path = os.path.join("Plots", filename)
    plt.savefig(full_path, dpi=300)
    if is_debugging:
        plt.show()

def build_regression_model(num_features, widest_layer_size=128, num_layers=4):
    layers = []
    input_size = num_features
    current_size = widest_layer_size

    for i in range(num_layers):
        layers.append(nn.Linear(input_size, current_size))
        layers.append(nn.GELU())
        input_size = current_size
        current_size = max(1, current_size // 2)  # Reduce size each layer

    layers.append(nn.Linear(input_size, 1))  # Final output layer

    return nn.Sequential(*layers)

