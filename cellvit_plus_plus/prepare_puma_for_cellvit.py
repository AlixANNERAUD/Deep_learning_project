import argparse
import os
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from shapely.geometry import shape
from tqdm import tqdm

# Disable DecompressionBombWarning for large images
Image.MAX_IMAGE_PIXELS = None

# Define the mapping from PUMA class names to integer labels (0-based)
# Ensure all classes mentioned in the PUMA description are included.
PUMA_LABEL_MAP = {
}

# Inverse map for label_map.yaml
LABEL_MAP_YAML = {v: k for k, v in PUMA_LABEL_MAP.items()}

def get_centroid(geom):
    """Calculates the centroid of a GeoJSON geometry."""
    polygon = shape(geom)
    return polygon.centroid

def process_puma_roi(roi_path, annotation_path, output_image_dir, output_label_dir):
    """
    Processes a single PUMA ROI and its annotation.
    Converts image to PNG, extracts nuclei centroids and labels, saves label CSV.
    Returns the base filename (without extension).
    """
    base_filename = roi_path.stem
    output_image_path = output_image_dir / f"{base_filename}.png"
    output_label_path = output_label_dir / f"{base_filename}.csv"

    # 1. Convert image to PNG
    try:
        with Image.open(roi_path) as img:
            if img.mode == 'RGBA':
                img = img.convert('RGB') # Ensure 3 channels
            img.save(output_image_path, "PNG")
    except Exception as e:
        print(f"Error converting image {roi_path}: {e}")
        return None

    # 2. Process annotations
    labels_data = []
    try:
        with open(annotation_path, 'r') as f:
            geojson_data = json.load(f)

        for feature in geojson_data['features']:
            properties = feature.get('properties', {})
            classification = properties.get('classification', {})
            class_name = classification.get('name', '').lower().strip()
            geometry = feature.get('geometry')

            if not geometry:
                print(f"Warning: Skipping feature due to missing geometry in {annotation_path.name}")
                continue

            label_int = -1 # Initialize with invalid label
            if class_name in PUMA_LABEL_MAP:
                label_int = PUMA_LABEL_MAP[class_name]
            elif class_name: # If class_name is not empty and not in the map
                # Dynamically add the new class
                new_label = len(PUMA_LABEL_MAP)
                PUMA_LABEL_MAP[class_name] = new_label
                LABEL_MAP_YAML[new_label] = class_name # Update inverse map as well
                label_int = new_label
                print(f"Info: Discovered new class '{class_name}' in {annotation_path.name}, assigning label {label_int}.")
            else:
                print(f"Warning: Skipping feature with empty class name in {annotation_path.name}")
                continue # Skip if class name is empty

            # Proceed if we have a valid label
            if label_int != -1:
                centroid = get_centroid(geometry)
                labels_data.append({'x': int(round(centroid.x)), 'y': int(round(centroid.y)), 'label': label_int})

    except FileNotFoundError:
        print("Error: Annotation file not found:", annotation_path)
        # Clean up image if annotation failed
        if output_image_path.exists():
            output_image_path.unlink()
        return None
    except Exception as e:
        print("Error processing annotation", annotation_path, ":", e)
        # Clean up image if annotation failed
        if output_image_path.exists():
            output_image_path.unlink()
        return None

    # 3. Save labels CSV
    if labels_data:
        df = pd.DataFrame(labels_data)
        df.to_csv(output_label_path, index=False)
        return base_filename
    else:
        print(f"Warning: No valid nuclei found in {annotation_path.name}. Skipping this ROI.")
        # Clean up image if no labels found
        if output_image_path.exists():
            output_image_path.unlink()
        return None


def create_splits(all_files, output_split_dir, val_split_ratio=0.2):
    """Creates train/validation split files."""
    random.shuffle(all_files)
    split_idx = int(len(all_files) * (1 - val_split_ratio))
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]

    output_split_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({'basename': train_files}).to_csv(output_split_dir / 'train.csv', index=False)
    pd.DataFrame({'basename': val_files}).to_csv(output_split_dir / 'val.csv', index=False)
    print(f"Created splits: {len(train_files)} train, {len(val_files)} validation.")

def save_label_map(output_dir):
    """Saves the label map to label_map.yaml."""
    import yaml # Import here as it's only needed once
    label_map_path = output_dir / 'label_map.yaml'
    with open(label_map_path, 'w') as f:
        yaml.dump(LABEL_MAP_YAML, f, default_flow_style=False)
    print(f"Saved label map to {label_map_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare PUMA dataset for CellViT++ detection format.")
    parser.add_argument('--puma_roi_dir', type=str, required=True, help="Path to the PUMA ROI image directory (e.g., .../01_training_dataset_tif_ROIs)")
    parser.add_argument('--puma_annotation_dir', type=str, required=True, help="Path to the PUMA GeoJSON annotation directory (e.g., .../01_training_dataset_geojson_nuclei)")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to the output directory for the formatted dataset.")
    parser.add_argument('--val_split', type=float, default=0.2, help="Fraction of data to use for validation (default: 0.2)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for splitting (default: 42)")
    # Add argument for test set if needed later

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    puma_roi_path = Path(args.puma_roi_dir)
    puma_annotation_path = Path(args.puma_annotation_dir)
    output_path = Path(args.output_dir)

    if not puma_roi_path.is_dir():
        print("Error: PUMA ROI directory not found:", puma_roi_path)
        return
    if not puma_annotation_path.is_dir():
        print("Error: PUMA annotation directory not found:", puma_annotation_path)
        return

    # Create output structure
    # For simplicity, putting all processed data into a single 'train' set first,
    # then splitting based on the generated file list.
    # A separate 'test' set could be handled similarly if PUMA provides one.
    output_train_image_dir = output_path / 'train' / 'images'
    output_train_label_dir = output_path / 'train' / 'labels'
    output_split_dir = output_path / 'splits' / 'fold_0' # Assuming one fold for now

    output_train_image_dir.mkdir(parents=True, exist_ok=True)
    output_train_label_dir.mkdir(parents=True, exist_ok=True)
    # output_split_dir is created by create_splits

    print("Starting dataset preparation...")
    processed_files_basenames = []

    # Find matching ROI and annotation files
    roi_files = list(puma_roi_path.glob('*.tif')) # Adjust glob if needed (e.g., *.svs)
    print(f"Found {len(roi_files)} ROI files in {puma_roi_path}")

    for roi_file in tqdm(roi_files, desc="Processing ROIs"):
        # Construct expected annotation filename based on ROI filename
        # Assumes annotation filename matches ROI filename but with .geojson extension
        annotation_file = puma_annotation_path / f"{roi_file.stem}_nuclei.geojson" # Adjust if naming differs

        if annotation_file.exists():
            basename = process_puma_roi(roi_file, annotation_file, output_train_image_dir, output_train_label_dir)
            if basename:
                processed_files_basenames.append(basename)
        else:
            print(f"Warning: Annotation file not found for {roi_file.name}, expected at {annotation_file}")

    print(f"Successfully processed {len(processed_files_basenames)} ROIs.")

    if not processed_files_basenames:
        print("Error: No ROIs were successfully processed. Please check input paths and file formats.")
        return

    # Create train/validation splits
    create_splits(processed_files_basenames, output_split_dir, args.val_split)

    # Save label map
    save_label_map(output_path)

    print("Dataset preparation finished.")
    print(f"Formatted dataset saved to:", output_path)
    print("Structure:")
    print("  {output_path}/")
    print("  ├── train/")
    print("  │   ├── images/ (*.png)")
    print("  │   └── labels/ (*.csv)")
    print("  ├── splits/")
    print("  │   └── fold_0/")
    print("  │       ├── train.csv")
    print("  │       └── val.csv")
    print("  └── label_map.yaml")

if __name__ == "__main__":
    main()
