#!/usr/bin/env python3
"""
Prepare dataset for nnU-Net from the existing GeoJSON and TIF files.
This script converts the GeoJSON tissue annotations to PNG image format required by nnU-Net
and sets up the proper directory structure.
"""

import os
import json
import shutil
import numpy as np
from pathlib import Path
import argparse
from PIL import Image, ImageDraw
import geojson
import re

def create_nnunet_directories(output_dir, dataset_id=1, dataset_name="PUMA"):
    """Create the nnU-Net directory structure."""
    dataset_folder = f"Dataset{dataset_id:03d}_{dataset_name}"
    
    # Create main directory
    dataset_path = os.path.join(output_dir, dataset_folder)
    os.makedirs(dataset_path, exist_ok=True)
    
    # Create imagesTr and labelsTr directories
    os.makedirs(os.path.join(dataset_path, "imagesTr"), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, "labelsTr"), exist_ok=True)
    
    return dataset_path

def convert_geojson_to_tissue_mask(geojson_path, image_path, output_path):
    """Convert GeoJSON tissue annotations to multi-class masks."""
    # Load the original image to get dimensions
    with Image.open(image_path) as img:
        # Ensure the source image is read as RGB if it has an alpha channel
        if img.mode != 'RGB':
            img = img.convert('RGB')
        width, height = img.size
        mask = Image.new('L', (width, height), 0)  # Background = 0
        draw = ImageDraw.Draw(mask)

    # Define tissue type to class mapping (matching GeoJSON classification names)
    tissue_class_map = {
        "tissue_tumor": 1,
        "tissue_stroma": 2,
        "tissue_epidermis": 3,
        "tissue_blood_vessel": 4,
        "tissue_necrosis": 5
    }

    # Load GeoJSON data
    with open(geojson_path) as f:
        gj_data = geojson.load(f)

    # Process tissue polygons and multipolygons
    for feature in gj_data["features"]:
        geom_type = feature["geometry"]["type"]
        coordinates = feature["geometry"]["coordinates"]
        tissue_type = feature["properties"].get("classification", {}).get("name", "").lower()
        class_value = tissue_class_map.get(tissue_type, 0)  # Default to background if unknown

        if class_value > 0: # Only draw if the tissue type is known
            if geom_type == "Polygon":
                # Structure: [ exterior_ring, hole_ring1, ... ]
                exterior_coords = coordinates[0]
                polygon = [(coord[0], coord[1]) for coord in exterior_coords]
                draw.polygon(polygon, fill=class_value)
                # Note: Ignoring potential holes in polygons for simplicity

            elif geom_type == "MultiPolygon":
                # Structure: [ polygon1, polygon2, ... ]
                # Each polygon: [ exterior_ring, hole_ring1, ... ]
                for poly_coords in coordinates:
                    exterior_coords = poly_coords[0]
                    polygon = [(coord[0], coord[1]) for coord in exterior_coords]
                    draw.polygon(polygon, fill=class_value)
                    # Note: Ignoring potential holes in polygons for simplicity

    # Save mask as PNG
    mask.save(output_path)

    return mask

def create_dataset_json(dataset_path, num_training_cases):
    """Create the dataset.json file required by nnU-Net."""
    dataset_json = {
        "channel_names": {
            "0": "R",
            "1": "G",
            "2": "B"
        },
        "labels": {
            "background": 0,
            "tissue_tumor": 1,
            "tissue_stroma": 2,
            "tissue_epidermis": 3,
            "tissue_blood_vessel": 4,
            "tissue_necrosis": 5
        },
        "numTraining": num_training_cases,
        "file_ending": ".png",
        "overwrite_image_reader_writer": "NaturalImage2DIO"
    }
    
    with open(os.path.join(dataset_path, "dataset.json"), 'w') as f:
        json.dump(dataset_json, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description='Convert dataset to nnU-Net format.')
    parser.add_argument('--input_dir', type=str, 
                        default="/Users/qberal/Developer/Cours/INSA_Rouen/S8/RL_Projet/Deep_learning_project/dataset",
                        help='Path to the input dataset directory')
    parser.add_argument('--output_dir', type=str, 
                        default="/Users/qberal/Developer/Cours/INSA_Rouen/S8/RL_Projet/Deep_learning_project/nnUNet_raw",
                        help='Path to the output nnUNet_raw directory')
    parser.add_argument('--dataset_id', type=int, default=1, help='Dataset ID (default: 1)')
    parser.add_argument('--dataset_name', type=str, default="PUMA", 
                        help='Dataset name (default: PUMA)')
    args = parser.parse_args()
    
    # Create nnU-Net directory structure
    dataset_path = create_nnunet_directories(args.output_dir, args.dataset_id, args.dataset_name)
    print(f"Created dataset directory: {dataset_path}")
    
    # Input directories
    images_dir = os.path.join(args.input_dir, "01_training_dataset_tif_ROIs")
    # Now using tissue annotations instead of nuclei
    geojson_dir = os.path.join(args.input_dir, "01_training_dataset_geojson_tissue")
    
    # Get all image files
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.tif') or f.endswith('.tiff')]
    print(f"Found {len(image_files)} image files")
    
    # Process each image
    count = 0
    for img_file in sorted(image_files):
        base_name = os.path.splitext(img_file)[0]
        
        # Source paths
        img_path = os.path.join(images_dir, img_file)
        geojson_path = os.path.join(geojson_dir, f"{base_name}_tissue.geojson")
        
        # Skip if segmentation doesn't exist
        if not os.path.exists(geojson_path):
            print(f"Warning: No tissue annotation found for {img_file}, skipping...")
            continue
        
        # For nnU-Net, we'll use a simplified case identifier by extracting the number
        case_id_match = re.search(r'(training_set_(?:metastatic|primary)_roi_(\d+))', base_name)
        if case_id_match:
            # Use the full name as case_id for clarity
            case_id = case_id_match.group(1)
            
            # Target paths for PNG format
            nnunet_img_path = os.path.join(dataset_path, "imagesTr", f"{case_id}_0000.png")
            nnunet_mask_path = os.path.join(dataset_path, "labelsTr", f"{case_id}.png")
            
            # Convert and copy the original image to PNG
            try:
                # Open and convert TIF image to PNG
                with Image.open(img_path) as img:
                    # Ensure image is in RGB format (3 channels) before saving
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img.save(nnunet_img_path)
                
                # Convert GeoJSON to multi-class tissue mask and save as PNG
                convert_geojson_to_tissue_mask(geojson_path, img_path, nnunet_mask_path)
                
                count += 1
                if count % 10 == 0:
                    print(f"Processed {count}/{len(image_files)} images")
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
    
    # Create dataset.json
    create_dataset_json(dataset_path, count)
    print(f"Created dataset.json with {count} training cases")
    print("Dataset preparation complete!")

if __name__ == "__main__":
    main()