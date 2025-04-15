import os
import json
import pandas as pd
import argparse
import ast  # For safely evaluating string representations of dicts

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

def safe_literal_eval(val):
    """Safely evaluate a string literal, returning None if it fails."""
    if pd.isna(val):
        return None
    try:
        return ast.literal_eval(str(val)) # Ensure it's a string first
    except (ValueError, SyntaxError, TypeError):
        # print(f"Warning: Could not parse value: {val}")
        return None # Return None if parsing fails

def extract_metric(cell_value, metric_key, multiplier):
    """Extracts the specific metric from a cell value, which might be a dict."""
    if pd.isna(cell_value):
        return pd.NA

    if metric_key is None: # The column itself is the score
        try:
            return float(cell_value) * multiplier
        except (ValueError, TypeError):
             # print(f"Warning: Could not convert value to float: {cell_value}")
             return pd.NA
    else:
        # Cell contains a dict (likely as a string)
        data_dict = safe_literal_eval(cell_value)
        if isinstance(data_dict, dict) and metric_key in data_dict:
            try:
                return float(data_dict[metric_key]) * multiplier
            except (ValueError, TypeError):
                 # print(f"Warning: Could not convert metric '{metric_key}' to float in dict: {data_dict}")
                 return pd.NA
        else:
            # print(f"Warning: Could not find key '{metric_key}' in parsed data or data is not a dict: {data_dict}")
            return pd.NA


def tabulate_vision_centric_results(eval_dir, out_fname):
    """
    Loads results from MME, MMBench, SEED, extracts specific category scores,
    and aggregates them into a single table.
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
                    extracted_series = df[src_col].apply(lambda x: extract_metric(x, metric, mult))
                    processed_cols[out_col] = extracted_series
                else:
                    print(f"  Warning: Source column '{src_col}' not found in {results_path} for '{out_col}'.")
                    # Create a series of NaNs to represent the missing data
                    processed_cols[out_col] = pd.Series([pd.NA] * len(df), index=df.index, name=out_col)

        if processed_cols:
            all_data[bench_key] = pd.DataFrame(processed_cols)
        else:
            print(f"  No columns processed for {bench_key}.")


    # --- Merge DataFrames ---
    if not all_data:
        print("Error: No benchmark data could be processed. Exiting.")
        return

    # Start with the first available benchmark data
    final_df = None
    processed_keys = list(all_data.keys())
    if processed_keys:
        final_df = all_data[processed_keys[0]]
        # Merge subsequent benchmarks
        for i in range(1, len(processed_keys)):
            key = processed_keys[i]
            final_df = pd.merge(final_df, all_data[key], left_index=True, right_index=True, how='outer')

    if final_df is None or final_df.empty:
         print("Error: Failed to merge any data. Final DataFrame is empty.")
         return

    # --- Reorder Columns based on Mapping ---
    final_column_order = [item[0] for item in COLUMN_MAPPING]
    # Ensure only columns that actually exist in the merged df are included
    final_column_order_present = [col for col in final_column_order if col in final_df.columns]

    # Add any columns present in final_df but not in the desired order (shouldn't happen with outer merge)
    # extra_cols = [col for col in final_df.columns if col not in final_column_order_present]
    # final_column_order_present.extend(extra_cols)

    final_df = final_df[final_column_order_present]

    # Sort by model name (index)
    final_df = final_df.sort_index()

    # --- Save Output ---
    try:
        if out_fname.endswith(".xlsx"):
            final_df.to_excel(out_fname)
            print(f"\nSaved vision-centric category results to Excel: {out_fname}")
        else:
            if not out_fname.endswith(".csv"):
                out_fname += ".csv"
                print(f"Warning: Output filename did not end with .csv or .xlsx. Saving as CSV: {out_fname}")
            final_df.to_csv(out_fname)
            print(f"\nSaved vision-centric category results to CSV: {out_fname}")
    except Exception as e:
        print(f"\nError saving output file {out_fname}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tabulate vision-centric category results from MME, MMBench, and SEED.")
    parser.add_argument("--eval_dir", type=str, default="eval", help="Directory containing evaluation results (e.g., 'eval/mme', 'eval/mmbench_en', 'eval/seed').")
    parser.add_argument("--out_file", type=str, default="vision_centric_results.xlsx", help="Name of the output file (Excel or CSV).")

    args = parser.parse_args()

    tabulate_vision_centric_results(args.eval_dir, args.out_file)