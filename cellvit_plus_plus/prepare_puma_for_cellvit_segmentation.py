import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from shapely.geometry import shape, mapping
from skimage.draw import polygon as sk_polygon # Use scikit-image for rasterization
from tqdm import tqdm
import yaml

# Disable DecompressionBombWarning for large images
Image.MAX_IMAGE_PIXELS = None

# Define the mapping from PUMA class names to integer labels (starting from 1 for segmentation)
PUMA_SEG_LABEL_MAP = {
}

# Inverse map for label_map.yaml
LABEL_MAP_YAML = {v: k for k, v in PUMA_SEG_LABEL_MAP.items()}

def process_puma_roi_segmentation(roi_path, annotation_path, output_image_dir, output_label_dir, image_size=(1024, 1024)):
    """
    Processes a single PUMA ROI and its annotation for segmentation.
    Converts image to PNG, creates instance and type maps, saves label NPY.
    Returns the base filename (without extension).
    """
    base_filename = roi_path.stem
    output_image_path = output_image_dir / f"{base_filename}.png"
    output_label_path = output_label_dir / f"{base_filename}.npy"

    # 1. Convert image to PNG
    try:
        with Image.open(roi_path) as img:
            if img.size != image_size:
                 print(f"Warning: Image {roi_path.name} has size {img.size}, expected {image_size}. Skipping.")
                 # Or resize if needed: img = img.resize(image_size)
                 return None
            if img.mode == 'RGBA':
                img = img.convert('RGB') # Ensure 3 channels
            img.save(output_image_path, "PNG")
    except Exception as e:
        print(f"Error converting image {roi_path}: {e}")
        return None

    # 2. Process annotations and create maps
    inst_map = np.zeros(image_size, dtype=np.int32)
    type_map = np.zeros(image_size, dtype=np.int32)
    instance_counter = 0

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
            if class_name in PUMA_SEG_LABEL_MAP:
                label_int = PUMA_SEG_LABEL_MAP[class_name]
            elif class_name: # If class_name is not empty and not in the map
                # Dynamically add the new class
                new_label = len(PUMA_SEG_LABEL_MAP) + 1 # Labels start from 1
                PUMA_SEG_LABEL_MAP[class_name] = new_label
                LABEL_MAP_YAML[new_label] = class_name # Update inverse map as well
                label_int = new_label
                print(f"Info: Discovered new class '{class_name}' in {annotation_path.name}, assigning label {label_int}.")
            else:
                print(f"Warning: Skipping feature with empty class name in {annotation_path.name}")
                continue # Skip if class name is empty

            # Proceed if we have a valid label
            if label_int != -1:
                try:
                    polygon = shape(geometry)

                    if polygon.geom_type == 'Polygon':
                        polygons_to_process = [polygon]
                    elif polygon.geom_type == 'MultiPolygon':
                        # Process each polygon within the MultiPolygon
                        polygons_to_process = list(polygon.geoms)
                    else:
                        print(f"Warning: Skipping feature with unsupported geometry type '{polygon.geom_type}' in {annotation_path.name}")
                        continue

                    for poly in polygons_to_process:
                        # Get exterior coordinates (ignore holes for simplicity for now)
                        coords_list = mapping(poly)['coordinates']
                        if not coords_list: # Check if coordinates list is empty
                            print(f"Warning: Skipping polygon with empty coordinates in {annotation_path.name}")
                            continue
                        
                        exterior_coords = coords_list[0]
                        if not exterior_coords or len(exterior_coords) < 3: # Check if exterior ring is valid
                             print(f"Warning: Skipping polygon with invalid exterior ring (less than 3 points) in {annotation_path.name}")
                             continue
                        
                        # Convert coordinates safely
                        try:
                            coords = np.array(exterior_coords, dtype=np.float64) # Specify dtype
                            if coords.ndim != 2 or coords.shape[1] != 2:
                                print(f"Warning: Skipping polygon with unexpected coordinate dimensions ({coords.shape}) in {annotation_path.name}")
                                continue
                        except Exception as coord_err:
                            print(f"Warning: Error converting coordinates for a polygon in {annotation_path.name}: {coord_err}. Skipping polygon.")
                            continue

                        instance_counter += 1 # Increment instance counter for each valid polygon part
                        rr, cc = sk_polygon(coords[:, 1], coords[:, 0], shape=image_size) # row, col (y, x)

                        # Clip coordinates to be within image bounds
                        valid_indices = (rr >= 0) & (rr < image_size[0]) & (cc >= 0) & (cc < image_size[1])
                        rr, cc = rr[valid_indices], cc[valid_indices]

                        # Fill maps
                        inst_map[rr, cc] = instance_counter
                        type_map[rr, cc] = label_int

                except Exception as geom_err:
                    # Catch errors during shape creation or general processing for this feature
                    print(f"Warning: Error processing a feature geometry in {annotation_path.name}: {geom_err}. Skipping feature.")
                    continue # Skip this feature

    except FileNotFoundError:
        print(f"Error: Annotation file not found: {annotation_path}")
        if output_image_path.exists():
            output_image_path.unlink()
        return None
    except Exception as e:
        print(f"Error processing annotation {annotation_path}: {e}")
        if output_image_path.exists():
            output_image_path.unlink()
        return None

    # 3. Save labels NPY
    if instance_counter > 0:
        label_dict = {'inst_map': inst_map, 'type_map': type_map}
        np.save(output_label_path, label_dict)
        return base_filename
    else:
        print(f"Warning: No valid nuclei found in {annotation_path.name}. Skipping this ROI.")
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
    label_map_path = output_dir / 'label_map.yaml'
    with open(label_map_path, 'w') as f:
        yaml.dump(LABEL_MAP_YAML, f, default_flow_style=False)
    print(f"Saved label map to {label_map_path}")

def main():
    parser = argparse.ArgumentParser(description="Prepare PUMA dataset for CellViT++ segmentation format.")
    parser.add_argument('--puma_roi_dir', type=str, required=True, help="Path to the PUMA ROI image directory (e.g., .../01_training_dataset_tif_ROIs)")
    parser.add_argument('--puma_annotation_dir', type=str, required=True, help="Path to the PUMA GeoJSON nuclei annotation directory (e.g., .../01_training_dataset_geojson_nuclei)")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to the output directory for the formatted dataset.")
    parser.add_argument('--image_height', type=int, default=1024, help="Expected height of ROI images (default: 1024)")
    parser.add_argument('--image_width', type=int, default=1024, help="Expected width of ROI images (default: 1024)")
    parser.add_argument('--val_split', type=float, default=0.2, help="Fraction of data to use for validation (default: 0.2)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for splitting (default: 42)")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    puma_roi_path = Path(args.puma_roi_dir)
    puma_annotation_path = Path(args.puma_annotation_dir)
    output_path = Path(args.output_dir)
    image_size = (args.image_height, args.image_width)

    if not puma_roi_path.is_dir():
        print(f"Error: PUMA ROI directory not found: {puma_roi_path}")
        return
    if not puma_annotation_path.is_dir():
        print(f"Error: PUMA annotation directory not found: {puma_annotation_path}")
        return

    # Create output structure
    output_train_image_dir = output_path / 'train' / 'images'
    output_train_label_dir = output_path / 'train' / 'labels'
    output_split_dir = output_path / 'splits' / 'fold_0'

    output_train_image_dir.mkdir(parents=True, exist_ok=True)
    output_train_label_dir.mkdir(parents=True, exist_ok=True)

    print("Starting dataset preparation for segmentation...")
    processed_files_basenames = []

    roi_files = list(puma_roi_path.glob('*.tif'))
    print(f"Found {len(roi_files)} ROI files in {puma_roi_path}")

    for roi_file in tqdm(roi_files, desc="Processing ROIs"):
        annotation_file = puma_annotation_path / f"{roi_file.stem}_nuclei.geojson"

        if annotation_file.exists():
            basename = process_puma_roi_segmentation(roi_file, annotation_file, output_train_image_dir, output_train_label_dir, image_size)
            if basename:
                processed_files_basenames.append(basename)
        else:
            print(f"Warning: Annotation file not found for {roi_file.name}, expected at {annotation_file}")

    print(f"Successfully processed {len(processed_files_basenames)} ROIs for segmentation.")

    if not processed_files_basenames:
        print("Error: No ROIs were successfully processed. Please check input paths, file formats, and image dimensions.")
        return

    create_splits(processed_files_basenames, output_split_dir, args.val_split)
    save_label_map(output_path)

    print("Dataset preparation finished.")
    print(f"Formatted dataset saved to: {output_path}")
    print("Structure:")
    print("  {output_path}/")
    print("  ├── train/")
    print("  │   ├── images/ (*.png)")
    print("  │   └── labels/ (*.npy containing 'inst_map' and 'type_map')")
    print("  ├── splits/")
    print("  │   └── fold_0/")
    print("  │       ├── train.csv")
    print("  │       └── val.csv")
    print("  └── label_map.yaml")

if __name__ == "__main__":
    main()
