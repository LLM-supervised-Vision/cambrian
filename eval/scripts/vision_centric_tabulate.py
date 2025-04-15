import os
import json
import pandas as pd
import argparse
import ast  # For safely evaluating string representations of dicts
import numpy as np # For NaN handling potentially

# Define the mapping from desired output columns to benchmark/source column/metric
# Format: (output_col_name, benchmark_name, source_csv_col, metric_key, multiplier)
# metric_key: None if the source column is already the final score
#             'score' for MME sub-scores
#             'accurcay' for SEED sub-scores (needs *100)
#             'circular_accuracy' for MMBench sub-scores (needs *100)
COLUMN_MAPPING = [
    # ################## Basic Perception & Recognition ##################
    ("SEED-Instance_Counting", "seed", "Instance Counting", 'accurcay', 100),
    ("MME-count", "mme", "count", 'score', 1),
    ("SEED-Instance_Identity", "seed", "Instance Identity", 'accurcay', 100),
    ("MMB-identity_reasoning", "mmbench_en", "identity_reasoning", 'circular_accuracy', 100),
    ("SEED-Instance_Attribute", "seed", "Instance Attribute", 'accurcay', 100),
    ("MMB-attribute_recognition", "mmbench_en", "attribute_recognition", 'circular_accuracy', 100),
    ("MMB-attribute_comparison", "mmbench_en", "attribute_comparison", 'circular_accuracy', 100),
    ("MMB-action_recognition", "mmbench_en", "action_recognition", 'circular_accuracy', 100),
    ("MME-color", "mme", "color", 'score', 1),
    ("MME-existence", "mme", "existence", 'score', 1),

    # ################## Scene Understanding ##################
    ("SEED-Scene_Understanding", "seed", "Scene Understanding", 'accurcay', 100),
    ("MME-scene", "mme", "scene", 'score', 1),
    ("MMB-image_scene", "mmbench_en", "image_scene", 'circular_accuracy', 100),
    ("MMB-image_topic", "mmbench_en", "image_topic", 'circular_accuracy', 100),
    ("MMB-image_style", "mmbench_en", "image_style", 'circular_accuracy', 100),
    ("MMB-image_quality", "mmbench_en", "image_quality", 'circular_accuracy', 100),
    ("MMB-image_emotion", "mmbench_en", "image_emotion", 'circular_accuracy', 100),

    # ################## Spatial Understanding ##################
    ("SEED-Instance_Location", "seed", "Instance Location", 'accurcay', 100),
    ("SEED-Spatial_Relation", "seed", "Spatial Relation", 'accurcay', 100),
    ("MMB-object_localization", "mmbench_en", "object_localization", 'circular_accuracy', 100),
    ("MMB-spatial_relationship", "mmbench_en", "spatial_relationship", 'circular_accuracy', 100),
    ("MME-position", "mme", "position", 'score', 1),

    # ################## OCR & Text Understanding ##################
    ("SEED-Text_Recognition", "seed", "Text Recognition", 'accurcay', 100),
    ("MME-OCR", "mme", "OCR", 'score', 1),
    ("MMB-ocr", "mmbench_en", "ocr", 'circular_accuracy', 100),

    # ################## Complex Visual Reasoning ##################
    ("SEED-Instance_Interaction", "seed", "Instance Interaction", 'accurcay', 100),
    ("SEED-Visual_Reasoning", "seed", "Visual Reasoning", 'accurcay', 100),
    ("MME-code_reasoning", "mme", "code_reasoning", 'score', 1),
    ("MME-numerical_calculation", "mme", "numerical_calculation", 'score', 1),
    ("MME-text_translation", "mme", "text_translation", 'score', 1),
    ("MME-commonsense_reasoning", "mme", "commonsense_reasoning", 'score', 1),
    ("MMB-structuralized_imagetext_understanding", "mmbench_en", "structuralized_imagetext_understanding", 'circular_accuracy', 100),
    ("MMB-future_prediction", "mmbench_en", "future_prediction", 'circular_accuracy', 100),
    ("MMB-physical_property_reasoning", "mmbench_en", "physical_property_reasoning", 'circular_accuracy', 100),
    ("MMB-function_reasoning", "mmbench_en", "function_reasoning", 'circular_accuracy', 100),
    ("MMB-nature_relation", "mmbench_en", "nature_relation", 'circular_accuracy', 100),
    ("MMB-physical_relation", "mmbench_en", "physical_relation", 'circular_accuracy', 100),
    ("MMB-social_relation", "mmbench_en", "social_relation", 'circular_accuracy', 100),
]

# Define the benchmarks we need to process
BENCHMARKS_INFO = {
    "mme": {"dir": "mme", "csv": "experiments.csv"},
    "mmbench_en": {"dir": "mmbench_en", "csv": "experiments.csv"},
    "seed": {"dir": "seed", "csv": "experiments.csv"},
}

# --- Define Category Groupings for Averaging ---
# Mapping from the desired average column name to the list of detailed columns
CATEGORY_GROUPS = {
    "AVG_Basic_Perception": [
        "SEED-Instance_Counting", "MME-count", "SEED-Instance_Identity",
        "MMB-identity_reasoning", "SEED-Instance_Attribute", "MMB-attribute_recognition",
        "MMB-attribute_comparison", "MMB-action_recognition", "MME-color", "MME-existence"
    ],
    "AVG_Scene_Understanding": [
        "SEED-Scene_Understanding", "MME-scene", "MMB-image_scene", "MMB-image_topic",
        "MMB-image_style", "MMB-image_quality", "MMB-image_emotion"
    ],
    "AVG_Spatial_Understanding": [
        "SEED-Instance_Location", "SEED-Spatial_Relation", "MMB-object_localization",
        "MMB-spatial_relationship", "MME-position"
    ],
    "AVG_OCR_Text": [
        "SEED-Text_Recognition", "MME-OCR", "MMB-ocr"
    ],
    "AVG_Complex_Reasoning": [
        "SEED-Instance_Interaction", "SEED-Visual_Reasoning", "MME-code_reasoning",
        "MME-numerical_calculation", "MME-text_translation", "MME-commonsense_reasoning",
        "MMB-structuralized_image-text_understanding", "MMB-future_prediction",
        "MMB-physical_property_reasoning", "MMB-function_reasoning",
        "MMB-nature_relation", "MMB-physical_relation", "MMB-social_relation"
    ]
}
# Create a reverse lookup: detailed column -> benchmark name
COL_TO_BENCHMARK = {item[0]: item[1] for item in COLUMN_MAPPING}


def safe_literal_eval(val):
    """Safely evaluate a string literal, returning None if it fails."""
    if pd.isna(val):
        return None
    try:
        # Attempt to handle potential single quotes within JSON-like strings
        if isinstance(val, str):
             val = val.replace("'", '"') # Basic replacement, might need refinement
             # Handle potential boolean literals if not already JSON standard
             val = val.replace("True", "true").replace("False", "false")
             val = val.replace("None", "null") # Handle None
        return ast.literal_eval(str(val)) # Ensure it's a string first
    except (ValueError, SyntaxError, TypeError):
         # If literal_eval fails, try json.loads as a fallback for complex cases
         try:
             return json.loads(val)
         except (json.JSONDecodeError, TypeError):
            # print(f"Warning: Could not parse value: {val}")
            return None # Return None if parsing fails


def extract_metric(cell_value, metric_key, multiplier):
    """Extracts the specific metric from a cell value, which might be a dict."""
    if pd.isna(cell_value):
        return pd.NA

    # Ensure cell_value is treated as a potential string first
    str_cell_value = str(cell_value)

    if metric_key is None: # The column itself is the score
        try:
            return float(str_cell_value) * multiplier
        except (ValueError, TypeError):
             # print(f"Warning: Could not convert value to float: {str_cell_value}")
             return pd.NA
    else:
        # Cell contains a dict (likely as a string)
        data_dict = safe_literal_eval(str_cell_value)
        if isinstance(data_dict, dict) and metric_key in data_dict:
            try:
                metric_val = data_dict[metric_key]
                if pd.isna(metric_val) or metric_val is None: # Check for None/NaN inside dict
                    return pd.NA
                return float(metric_val) * multiplier
            except (ValueError, TypeError):
                 # print(f"Warning: Could not convert metric '{metric_key}' to float in dict: {data_dict}")
                 return pd.NA
        else:
            # print(f"Warning: Could not find key '{metric_key}' in parsed data or data is not a dict: {data_dict} from cell {str_cell_value}")
            return pd.NA


def tabulate_vision_centric_results(eval_dir, out_fname):
    """
    Loads results from MME, MMBench, SEED, extracts specific category scores,
    calculates category averages, and aggregates them into a single table.
    """
    if not os.path.exists(eval_dir):
        raise ValueError(f"Evaluation directory {eval_dir} does not exist")

    print(f"Processing results from eval_dir: {eval_dir}")

    all_data = {} # Store dataframes per benchmark

    # --- Load and Process Data for Each Benchmark ---
    for bench_key, info in BENCHMARKS_INFO.items():
        results_path = os.path.join(eval_dir, info["dir"], info["csv"])
        print(f"\n--- Processing Benchmark: {bench_key} ---")
        if not os.path.exists(results_path):
            print(f"Warning: Results file not found for {bench_key} at {results_path}. Skipping.")
            continue

        try:
            df = pd.read_csv(results_path)
        except Exception as e:
            print(f"Error reading {results_path}: {e}. Skipping.")
            continue

        if df.empty:
            print(f"Warning: Results file {results_path} is empty. Skipping.")
            continue

        # Keep latest result per model
        df = df.sort_values("time")
        df = df.drop_duplicates("model", keep="last")
        df = df.set_index("model") # Use model name as index for merging

        processed_cols = {} # Store extracted columns for this benchmark
        # --- Extract Required Columns ---
        for out_col, bench_map, src_col, metric, mult in COLUMN_MAPPING:
            if bench_map == bench_key:
                if src_col in df.columns:
                    print(f"  Extracting: '{out_col}' from '{src_col}' (key: {metric}, mult: {mult})")
                    # Apply extraction row by row
                    extracted_series = df[src_col].apply(lambda x: extract_metric(x, metric, mult))
                    processed_cols[out_col] = extracted_series.astype(float) # Ensure numeric type
                else:
                    print(f"  Warning: Source column '{src_col}' not found in {results_path} for '{out_col}'.")
                    processed_cols[out_col] = pd.Series([np.nan] * len(df), index=df.index, name=out_col, dtype=float) # Use np.nan

        if processed_cols:
            all_data[bench_key] = pd.DataFrame(processed_cols)
        else:
            print(f"  No columns processed for {bench_key}.")


    # --- Merge DataFrames ---
    if not all_data:
        print("Error: No benchmark data could be processed. Exiting.")
        return

    final_df = None
    processed_keys = list(all_data.keys())
    if processed_keys:
        # Initialize with the first DataFrame
        first_key = processed_keys[0]
        final_df = all_data[first_key]

        # Outer merge with subsequent DataFrames
        for i in range(1, len(processed_keys)):
            key = processed_keys[i]
            # Ensure columns being merged don't already exist in final_df from a *different* benchmark source
            # (This shouldn't happen with the current setup but is good practice)
            cols_to_merge = [col for col in all_data[key].columns if col not in final_df.columns]
            if cols_to_merge:
                 final_df = pd.merge(final_df, all_data[key][cols_to_merge], left_index=True, right_index=True, how='outer')
            else:
                 print(f"  Skipping merge for {key} as its columns are already present (or none processed).")


    if final_df is None or final_df.empty:
         print("Error: Failed to merge any data. Final DataFrame is empty.")
         return

    # --- Calculate Category Averages ---
    print("\n--- Calculating Category Averages ---")
    average_scores_df = pd.DataFrame(index=final_df.index) # Ensure same index

    for avg_col_name, detail_cols in CATEGORY_GROUPS.items():
        print(f"  Calculating average for: {avg_col_name}")
        # Select only the detail columns that actually exist in final_df
        cols_present = [col for col in detail_cols if col in final_df.columns]
        if not cols_present:
            print(f"    Warning: No columns found in final_df for category {avg_col_name}. Skipping average.")
            average_scores_df[avg_col_name] = np.nan # Add NaN column
            continue

        # Create a copy to avoid modifying the original data during division
        category_data = final_df[cols_present].copy()

        # Apply MME division rule (divide MME scores by 2)
        mme_cols_in_category = []
        for col in cols_present:
            # Check if this column originates from the 'mme' benchmark
            if COL_TO_BENCHMARK.get(col) == 'mme':
                mme_cols_in_category.append(col)
                # Divide the column by 2
                category_data[col] = category_data[col] / 2.0

        if mme_cols_in_category:
            print(f"    Applied MME division rule (score/2) to columns: {mme_cols_in_category}")


        # Calculate the row-wise mean, ignoring NaNs
        average_scores_df[avg_col_name] = category_data.mean(axis=1, skipna=True)

    # --- Combine Average Scores and Detailed Scores ---
    # Concatenate the new average columns to the front of the detailed dataframe
    final_df = pd.concat([average_scores_df, final_df], axis=1)

    # --- Define Final Column Order ---
    avg_column_names = list(CATEGORY_GROUPS.keys())
    detail_column_order = [item[0] for item in COLUMN_MAPPING]

    # Ensure only columns actually present in the final_df are included
    final_avg_cols_present = [col for col in avg_column_names if col in final_df.columns]
    final_detail_cols_present = [col for col in detail_column_order if col in final_df.columns]

    # Combine the lists for the final order
    final_combined_order = final_avg_cols_present + final_detail_cols_present

    # Reorder the DataFrame
    final_df = final_df[final_combined_order]

    # Sort by model name (index)
    final_df = final_df.sort_index()

    # --- Save Output ---
    try:
        if out_fname.endswith(".xlsx"):
            final_df.to_excel(out_fname)
            print(f"\nSaved vision-centric category results (with averages) to Excel: {out_fname}")
        else:
            if not out_fname.endswith(".csv"):
                out_fname += ".csv"
                print(f"Warning: Output filename did not end with .csv or .xlsx. Saving as CSV: {out_fname}")
            final_df.to_csv(out_fname)
            print(f"\nSaved vision-centric category results (with averages) to CSV: {out_fname}")
    except Exception as e:
        print(f"\nError saving output file {out_fname}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tabulate vision-centric category results from MME, MMBench, and SEED, including category averages.")
    parser.add_argument("--eval_dir", type=str, default="eval", help="Directory containing evaluation results (e.g., 'eval/mme', 'eval/mmbench_en', 'eval/seed').")
    parser.add_argument("--out_file", type=str, default="vision_centric_results_with_avg.xlsx", help="Name of the output file (Excel or CSV).")

    args = parser.parse_args()

    tabulate_vision_centric_results(args.eval_dir, args.out_file)