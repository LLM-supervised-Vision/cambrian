# /data/austin/cambrian/eval/scripts/mmstar_tabulate.py
# NOTE: This script is named mmstar_tabulate.py as requested, but it processes
# data with category columns that match the MMBench format, based on the
# specific input data provided by the user.

import os
import json
import pandas as pd
import argparse
import ast  # For safely evaluating string representations of dicts
import numpy as np # For NaN handling

# --- Define Category Columns based on User's Provided Data Header ---
# These are the columns expected to contain dictionary strings with an 'accuracy' key
# (Taken directly from the user's example header)
CATEGORY_COLUMNS = [
    'image scene and topic',
    'image emotion',
    'image style & quality',
    'recognition',
    'object counting',
    'localization',
    'cross-instance attribute reasoning',
    'single-instance reasoning',
    'cross-instance relation reasoning',
    'common reasoning',
    'diagram reasoning',
    'code & sequence reasoning',
    'geometry',
    'numeric commonsense and calculation',
    'statistical reasoning',
    'biology & chemistry & physics',
    'electronics & energy & mechanical eng.',
    'geography & earth science & agriculture'
]

# --- Helper Function for Safe Parsing ---
def safe_literal_eval(val):
    """Safely evaluate a string literal (like a dict), returning None if it fails."""
    if pd.isna(val):
        return None
    try:
        # Prioritize ast.literal_eval for safety with Python dict/list strings
        return ast.literal_eval(str(val))
    except (ValueError, SyntaxError, TypeError):
        # Fallback for JSON-like strings with potential single quotes etc.
        try:
            # Basic replacements to handle potential variations if not strict JSON
            if isinstance(val, str):
                 # Replace single quotes used for strings, be careful not to replace quotes within strings
                 # A more robust regex might be needed for complex cases, but this handles simple dicts
                 cleaned_val = val.replace(": '", ': "').replace("',", '",').replace("'}", '"}')
                 cleaned_val = cleaned_val.replace("None", "null").replace("True", "true").replace("False", "false")
                 # Handle potential NaN string representation
                 cleaned_val = cleaned_val.replace("NaN", "null")
            else:
                cleaned_val = str(val) # Ensure it's a string for json.loads
            return json.loads(cleaned_val)
        except (json.JSONDecodeError, TypeError):
            # print(f"Warning: Could not parse value: {val}")
            return None # Return None if all parsing fails

def extract_accuracy(cell_value):
    """Extracts the 'accuracy' metric from a cell value (parsed dict)."""
    if pd.isna(cell_value):
        return np.nan

    data_dict = safe_literal_eval(cell_value)

    if isinstance(data_dict, dict):
        acc = data_dict.get('accuracy', np.nan)
        try:
            # Convert to float and handle potential non-numeric values
            return float(acc) * 100.0 if not pd.isna(acc) else np.nan
        except (ValueError, TypeError):
            # print(f"Warning: Could not convert accuracy '{acc}' to float.")
            return np.nan
    else:
        # print(f"Warning: Parsed value is not a dictionary: {data_dict}")
        return np.nan

# --- Main Tabulation Function ---
def tabulate_category_results(eval_dir, benchmark_name, results_csv_name, out_fname):
    """
    Loads results for a given benchmark, extracts category-specific accuracy scores
    from dictionary strings in specified columns, and aggregates them into a single table.
    """
    benchmark_dir = os.path.join(eval_dir, benchmark_name)
    results_path = os.path.join(benchmark_dir, results_csv_name)

    print(f"--- Processing Benchmark: {benchmark_name} ---")
    print(f"Reading results from: {results_path}")

    if not os.path.exists(results_path):
        print(f"Error: Results file not found: {results_path}. Exiting.")
        return

    try:
        # Read CSV, explicitly treating category columns as strings initially
        dtype_dict = {col: str for col in CATEGORY_COLUMNS}
        df = pd.read_csv(results_path, dtype=dtype_dict, keep_default_na=False, na_values=['']) # Handle empty strings as NA
    except Exception as e:
        print(f"Error reading {results_path}: {e}. Exiting.")
        return

    if df.empty:
        print(f"Warning: Results file {results_path} is empty. Exiting.")
        return

    # --- Data Cleaning and Selection ---
    print("Cleaning data: Sorting by time and selecting latest run per model...")
    df = df.sort_values("time")
    df = df.drop_duplicates("model", keep="last")
    df = df.set_index("model") # Use model name as index

    # --- Extract Category Accuracies ---
    print("Extracting category accuracies (from dict strings)...")
    extracted_data = {}

    # Process Overall Accuracy first
    if 'accuracy' in df.columns:
        # Assuming the overall accuracy in the input is already a percentage
         extracted_data['Overall_Accuracy'] = df['accuracy'].apply(lambda x: pd.to_numeric(x, errors='coerce'))
    else:
        print("Warning: 'accuracy' column not found for overall score.")


    # Process each category column
    for category_col in CATEGORY_COLUMNS:
        if category_col in df.columns:
            print(f"  Extracting from: '{category_col}'")
            # Apply extraction row by row using the helper function
            extracted_data[category_col] = df[category_col].apply(extract_accuracy)
        else:
            print(f"  Warning: Category column '{category_col}' not found in the CSV. Skipping.")
            # Add a column of NaNs if the category column is missing
            extracted_data[category_col] = pd.Series([np.nan] * len(df), index=df.index, name=category_col, dtype=float)

    # --- Create Final DataFrame ---
    final_df = pd.DataFrame(extracted_data)

    # --- Define Final Column Order ---
    # Start with Overall_Accuracy if it exists, then the category columns
    final_column_order = []
    if 'Overall_Accuracy' in final_df.columns:
        final_column_order.append('Overall_Accuracy')

    # Add category columns in the specified order, only if they exist in the final_df
    final_column_order.extend([col for col in CATEGORY_COLUMNS if col in final_df.columns])

    # Reorder the DataFrame
    final_df = final_df[final_column_order]

    # Sort by model name (index)
    final_df = final_df.sort_index()

    # --- Save Output ---
    print(f"Saving tabulated results to: {out_fname}")
    try:
        if out_fname.endswith(".xlsx"):
            final_df.to_excel(out_fname, float_format="%.2f") # Format as percentage with 2 decimals
            print(f"Successfully saved results to Excel: {out_fname}")
        else:
            if not out_fname.endswith(".csv"):
                out_fname += ".csv"
                print(f"Warning: Output filename did not end with .csv or .xlsx. Saving as CSV: {out_fname}")
            final_df.to_csv(out_fname, float_format="%.2f") # Format as percentage with 2 decimals
            print(f"Successfully saved results to CSV: {out_fname}")
    except Exception as e:
        print(f"\nError saving output file {out_fname}: {e}")


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tabulate category-specific accuracy results from a CSV file where categories are stored as dictionary strings.")
    parser.add_argument("--eval_dir", type=str, default="eval",
                        help="Base directory containing evaluation results (e.g., 'eval').")
    # Explicitly state we're targeting the 'mmstar' subdirectory as per user request
    parser.add_argument("--benchmark", type=str, default="mmstar",
                        help="Name of the benchmark subdirectory within eval_dir (default: 'mmstar').")
    parser.add_argument("--results_csv", type=str, default="experiments.csv",
                        help="Name of the CSV file containing raw experiment results within the benchmark directory.")
    parser.add_argument("--out_file", type=str, default="mmstar_category_results.xlsx",
                        help="Name of the output file (Excel or CSV) to save the tabulated category results.")

    args = parser.parse_args()

    # Check if the specified benchmark directory exists
    bench_path = os.path.join(args.eval_dir, args.benchmark)
    if not os.path.isdir(bench_path):
         print(f"Error: Benchmark directory '{bench_path}' does not exist.")
    else:
        tabulate_category_results(args.eval_dir, args.benchmark, args.results_csv, args.out_file)