\
import os
import json
import csv
import random
import shutil
from pathlib import Path
import numpy as np
import yaml
from PIL import Image
import tifffile
from shapely.geometry import shape
from sklearn.model_selection import train_test_split
import argparse

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# These can be overridden by command-line arguments
DEFAULT_SOURCE_DATASET_DIR = Path("/Users/qberal/Developer/Cours/INSA_Rouen/S8/RL_Projet/Deep_learning_project/dataset")
DEFAULT_OUTPUT_DIR = Path("/Users/qberal/Developer/Cours/INSA_Rouen/S8/RL_Projet/Deep_learning_project/cellvit_dataset") # New directory for formatted data

# Split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1 # Proportion of the *original* dataset
TEST_RATIO = 0.1 # Proportion of the *original* dataset
# Ensure ratios sum to 1
if abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) > 1e-9:
    raise ValueError("Train, validation, and test ratios must sum to 1.0")

# --- Helper Functions ---
def create_dirs(base_path):
    """Creates the necessary directory structure for the CellViT dataset."""
    logging.info(f"Creating directory structure under: {base_path}")
    (base_path / "train" / "images").mkdir(parents=True, exist_ok=True)
    (base_path / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (base_path / "test" / "images").mkdir(parents=True, exist_ok=True)
    (base_path / "test" / "labels").mkdir(parents=True, exist_ok=True)
    (base_path / "splits" / "fold_0").mkdir(parents=True, exist_ok=True)

def get_geojson_path(tif_path, geojson_dir):
    """Derives the corresponding GeoJSON filename from a TIF filename."""
    # Assumes naming convention: roi_XXX.tif -> roi_XXX_nuclei.geojson
    base_name = tif_path.stem
    geojson_name = f"{base_name}_nuclei.geojson"
    return geojson_dir / geojson_name

def extract_nuclei_data(geojson_path, label_str_to_int, next_label_id_counter):
    """
    Reads a GeoJSON file, extracts nucleus centroids and classification labels.
    Updates the label mapping and assigns integer IDs.
    Returns a list of dictionaries {'x': ..., 'y': ..., 'label': ...} and the updated counter.
    """
    nuclei_list = []
    try:
        with open(geojson_path, 'r') as f:
            data = json.load(f)

        if not isinstance(data, dict) or 'features' not in data:
            logging.warning(f"Invalid GeoJSON structure in {geojson_path.name}: 'features' key missing.")
            return [], next_label_id_counter

        for feature in data.get('features', []):
            try:
                geom_data = feature.get('geometry')
                props = feature.get('properties', {})
                # --- !!! Adjust this part based on your actual GeoJSON structure !!! ---
                classification = props.get('classification', {})
                label_str = classification.get('name', 'Unknown') # Default if name is missing
                # --- End of adjustment section ---

                if not geom_data:
                    logging.warning(f"Skipping feature with missing geometry in {geojson_path.name}")
                    continue

                geom = shape(geom_data)
                centroid = geom.centroid

                if label_str not in label_str_to_int:
                    label_str_to_int[label_str] = next_label_id_counter[0]
                    logging.info(f"New label found: '{label_str}' -> ID {next_label_id_counter[0]}")
                    next_label_id_counter[0] += 1

                nuclei_list.append({
                    'x': int(round(centroid.x)), # Ensure coordinates are integers
                    'y': int(round(centroid.y)),
                    'label': label_str_to_int[label_str]
                })
            except Exception as e:
                logging.warning(f"Could not process feature in {geojson_path.name}: {e}. Feature: {str(feature)[:100]}...")
                continue

    except FileNotFoundError:
        logging.warning(f"GeoJSON file not found: {geojson_path}")
        return [], next_label_id_counter
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from {geojson_path}")
        return [], next_label_id_counter
    except ImportError:
         logging.error("The 'shapely' library is required but not installed. pip install shapely")
         raise
    except Exception as e:
        logging.error(f"An unexpected error occurred processing {geojson_path}: {e}")
        return [], next_label_id_counter

    return nuclei_list, next_label_id_counter

def convert_tif_to_png(tif_path, png_path):
    """Converts a TIF image to PNG format."""
    try:
        img_array = tifffile.imread(tif_path)
        # Handle different array shapes (e.g., remove alpha channel if present)
        if img_array.ndim == 3 and img_array.shape[-1] >= 3:
             img_array = img_array[..., :3] # Keep only RGB

        img = Image.fromarray(img_array)
        img.save(png_path, format='PNG')
    except FileNotFoundError:
        logging.error(f"TIF file not found: {tif_path}")
    except ImportError:
         logging.error("Libraries 'tifffile' and 'Pillow' are required. pip install tifffile Pillow")
         raise
    except Exception as e:
        logging.error(f"Failed to convert {tif_path.name} to PNG: {e}")


def write_label_csv(labels_data, csv_path):
    """Writes the extracted nuclei data to a CSV file."""
    try:
        with open(csv_path, 'w', newline='') as f:
            if not labels_data: # Write header even if empty
                 writer = csv.writer(f)
                 writer.writerow(['x', 'y', 'label'])
            else:
                 writer = csv.DictWriter(f, fieldnames=['x', 'y', 'label'])
                 writer.writeheader()
                 writer.writerows(labels_data)
    except IOError as e:
        logging.error(f"Failed to write CSV file {csv_path}: {e}")

def write_split_file(basenames, file_path):
    """Writes a list of image basenames to a CSV file (for train/val splits)."""
    try:
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['imagename']) # Header expected by some frameworks
            for name in basenames:
                writer.writerow([name])
    except IOError as e:
        logging.error(f"Failed to write split file {file_path}: {e}")

def write_label_map(label_map_int_to_str, yaml_path):
    """Writes the integer ID to string label mapping to a YAML file."""
    try:
        with open(yaml_path, 'w') as f:
            yaml.dump(label_map_int_to_str, f, default_flow_style=False)
    except IOError as e:
        logging.error(f"Failed to write label map {yaml_path}: {e}")
    except ImportError:
        logging.error("Library 'PyYAML' is required. pip install PyYAML")
        raise

# --- Main Script ---
def main(source_dir, output_dir):
    tif_roi_dir = source_dir / "01_training_dataset_tif_ROIs"
    geojson_nuclei_dir = source_dir / "01_training_dataset_geojson_nuclei"

    logging.info(f"Source TIF directory: {tif_roi_dir}")
    logging.info(f"Source GeoJSON directory: {geojson_nuclei_dir}")
    logging.info(f"Output directory: {output_dir}")

    if not tif_roi_dir.is_dir():
        logging.error(f"Error: Source TIF directory not found: {tif_roi_dir}")
        return
    if not geojson_nuclei_dir.is_dir():
        logging.error(f"Error: Source GeoJSON directory not found: {geojson_nuclei_dir}")
        return

    # Create output directories
    create_dirs(output_dir)

    # Get all TIF files
    tif_files = sorted(list(tif_roi_dir.glob("*.tif")))
    logging.info(f"Found {len(tif_files)} TIF files.")
    if not tif_files:
        logging.error("No TIF files found in source directory.")
        return

    # Shuffle and split files
    random.seed(42) # for reproducibility
    shuffled_files = list(tif_files) # Create a mutable copy
    random.shuffle(shuffled_files)

    if len(shuffled_files) < 3:
         logging.warning("Very few files found (<3), splitting might result in empty sets.")
         # Adjust split logic for small datasets if necessary, here we proceed but results might be skewed.

    try:
        train_val_files, test_files = train_test_split(shuffled_files, test_size=TEST_RATIO, random_state=42, shuffle=False) # Already shuffled
        # Adjust validation ratio relative to the remaining train_val set size
        if len(train_val_files) > 0:
             val_ratio_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
             train_files, val_files = train_test_split(train_val_files, test_size=val_ratio_adjusted, random_state=42, shuffle=False)
        else:
             train_files, val_files = [], []
             logging.warning("Train+Validation set is empty after initial split.")

    except ImportError:
        logging.error("Library 'scikit-learn' is required for splitting. pip install scikit-learn")
        raise
    except Exception as e:
        logging.error(f"Error during data splitting: {e}")
        return


    logging.info(f"Splitting into: {len(train_files)} train, {len(val_files)} validation, {len(test_files)} test files.")

    label_str_to_int = {}
    next_label_id_counter = [0] # Use list to pass by reference
    train_basenames = []
    val_basenames = []

    # --- Process Sets ---
    set_configs = [
        ("Training", train_files, output_dir / "train", train_basenames, True),
        ("Validation", val_files, output_dir / "train", val_basenames, False), # Val data goes into train folders
        ("Test", test_files, output_dir / "test", None, False) # No basename list needed for test?
    ]

    for set_name, file_list, target_base_dir, basename_list, update_label_map in set_configs:
        if not file_list:
            logging.warning(f"Skipping processing for empty {set_name} set.")
            continue

        logging.info(f"\nProcessing {set_name} set ({len(file_list)} files)...")
        output_img_dir = target_base_dir / "images"
        output_lbl_dir = target_base_dir / "labels"

        for tif_path in file_list:
            basename = tif_path.stem
            if basename_list is not None:
                basename_list.append(basename)

            png_path = output_img_dir / f"{basename}.png"
            csv_path = output_lbl_dir / f"{basename}.csv"
            geojson_path = get_geojson_path(tif_path, geojson_nuclei_dir)

            # Convert image first
            convert_tif_to_png(tif_path, png_path)

            # Process annotations
            if geojson_path.exists():
                 current_next_id = next_label_id_counter if update_label_map else list(next_label_id_counter) # Pass copy if not updating
                 nuclei_data, next_label_id_counter = extract_nuclei_data(geojson_path, label_str_to_int, current_next_id)
                 write_label_csv(nuclei_data, csv_path)
            else:
                 logging.warning(f"GeoJSON not found for {tif_path.name}, creating empty label file.")
                 write_label_csv([], csv_path) # Create empty CSV

    # --- Final Steps ---
    # Write split files
    logging.info("\nWriting split files...")
    split_dir = output_dir / "splits" / "fold_0"
    write_split_file(train_basenames, split_dir / "train.csv")
    write_split_file(val_basenames, split_dir / "val.csv")

    # Write label map
    if label_str_to_int:
        logging.info("\nWriting label map...")
        # Invert map for yaml (id: name)
        label_int_to_str = {v: k for k, v in label_str_to_int.items()}
        write_label_map(label_int_to_str, output_dir / "label_map.yaml")
        logging.info(f"Label map created: {label_int_to_str}")
    else:
        logging.warning("No labels were extracted, label_map.yaml not created.")


    logging.info("\nDataset preparation complete.")
    logging.info(f"Output dataset at: {output_dir}")


if __name__ == "__main__":
    print("--- CellViT Dataset Preparation Script ---")
    print("Note: This script requires libraries like Pillow, tifffile, shapely, PyYAML, scikit-learn, numpy.")
    print("Please ensure they are installed in your environment (e.g., `pip install Pillow tifffile shapely PyYAML scikit-learn numpy`).\n")

    parser = argparse.ArgumentParser(description="Prepare dataset for CellViT detection format.")
    parser.add_argument(
        "--source_dir",
        type=Path,
        default=DEFAULT_SOURCE_DATASET_DIR,
        help=f"Path to the root directory containing the original dataset folders (e.g., 01_training_dataset_tif_ROIs). Default: {DEFAULT_SOURCE_DATASET_DIR}"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Path to the directory where the formatted dataset will be created. Default: {DEFAULT_OUTPUT_DIR}"
    )
    args = parser.parse_args()

    main(args.source_dir, args.output_dir)
