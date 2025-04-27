#!/usr/bin/env python3
"""
Prepare the NSCLC dataset for nnU-Net from existing PNG images and masks.
This script copies the images and masks into the nnU-Net format directory structure
and generates the dataset.json file.
"""

import os
import json
import shutil
from pathlib import Path
import argparse
from PIL import Image

def create_nnunet_directories(output_dir, dataset_id=2, dataset_name="NSCLC"):
    """Create the nnU-Net directory structure."""
    dataset_folder = f"Dataset{dataset_id:03d}_{dataset_name}"
    
    # Create main directory
    dataset_path = Path(output_dir) / dataset_folder
    dataset_path.mkdir(parents=True, exist_ok=True)
    
    # Create imagesTr and labelsTr directories
    (dataset_path / "imagesTr").mkdir(exist_ok=True)
    (dataset_path / "labelsTr").mkdir(exist_ok=True)
    
    return str(dataset_path) # Return as string for os.path compatibility if needed

def create_dataset_json(dataset_path, num_training_cases):
    """Create the dataset.json file required by nnU-Net for NSCLC."""
    dataset_json = {
        "channel_names": {
            "0": "R",
            "1": "G",
            "2": "B"
        },
        "labels": {
            # Based on provided NSCLC classes
            "background": 0, # NON-ANNOTATED PIXELS
            "tumor": 1,
            "tumor_stroma": 2,
            "necrotic_debris": 3,
            "mucin": 4,
            "benign_lung": 5,
            "benign_connective_tissue": 6,
            "bleeding_blood": 7,
            "bronchial_mucosa": 8,
            "cartilage": 9,
            "peribronchial_glands": 10,
            "inflammatory_infiltrate": 11,
            "background_explicit": 12 # Explicit BACKGROUND class
        },
        "numTraining": num_training_cases,
        "file_ending": ".png",
        "overwrite_image_reader_writer": "NaturalImage2DIO" # For standard RGB images
    }
    
    output_file = Path(dataset_path) / "dataset.json"
    with open(output_file, 'w') as f:
        json.dump(dataset_json, f, indent=4)
    print(f"Created dataset.json at {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Convert NSCLC dataset to nnU-Net format.')
    parser.add_argument('--input_dir', type=str, 
                        default="../nsclc_data", # Relative path to nsclc_data from script location
                        help='Path to the input NSCLC dataset directory (containing image/ and mask/)')
    parser.add_argument('--output_dir', type=str, 
                        default="./nnUNet_raw", # Output within the nnunet folder
                        help='Path to the output nnUNet_raw directory')
    parser.add_argument('--dataset_id', type=int, default=2, 
                        help='Dataset ID for nnU-Net (default: 2)')
    parser.add_argument('--dataset_name', type=str, default="NSCLC", 
                        help='Dataset name for nnU-Net (default: NSCLC)')
    args = parser.parse_args()
    
    # Resolve absolute paths
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    # Create nnU-Net directory structure
    dataset_path = create_nnunet_directories(str(output_dir), args.dataset_id, args.dataset_name)
    print(f"Created dataset directory: {dataset_path}")
    
    # Input directories
    images_dir = input_dir / "image"
    masks_dir = input_dir / "mask"
    
    if not images_dir.is_dir():
        print(f"Error: Image directory not found at {images_dir}")
        return
    if not masks_dir.is_dir():
        print(f"Error: Mask directory not found at {masks_dir}")
        return

    # Get all image files (assuming PNG, TIF, TIFF)
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.tif', '.tiff','.jpg'))]
    print(f"Found {len(image_files)} image files in {images_dir}")
    
    # Process each image
    count = 0
    processed_files = []
    for img_file in sorted(image_files):
        base_name = Path(img_file).stem # Get filename without extension
        
        # Source paths
        img_path = images_dir / img_file
        # Assume mask has the same base name and is a PNG file
        mask_file = f"{base_name}.png" 
        mask_path = masks_dir / mask_file
        
        # Skip if mask doesn't exist
        if not mask_path.exists():
            print(f"Warning: No mask found for {img_file} at {mask_path}, skipping...")
            continue
        
        # Use base_name as case_id
        case_id = base_name
        
        # Target paths using pathlib
        nnunet_img_dir = Path(dataset_path) / "imagesTr"
        nnunet_mask_dir = Path(dataset_path) / "labelsTr"
        nnunet_img_path = nnunet_img_dir / f"{case_id}_0000.png"
        nnunet_mask_path = nnunet_mask_dir / f"{case_id}.png"
        
        try:
            # Copy/Convert the original image to PNG format for nnU-Net input
            with Image.open(img_path) as img:
                # Ensure image is RGB before saving as PNG
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(nnunet_img_path)
            
            # Verify and copy the mask file
            with Image.open(mask_path) as mask_img:
                # nnU-Net expects single-channel integer masks ('L' mode in PIL)
                if mask_img.mode != 'L':
                    print(f"Warning: Mask {mask_file} is not single channel ('L' mode). Mode was {mask_img.mode}. Attempting conversion...")
                    # This assumes the first channel contains the labels if multi-channel
                    mask_img = mask_img.convert('L') 
                
                # Check pixel values (optional, can be slow)
                # pixels = set(mask_img.getdata())
                # print(f"Mask {mask_file} pixel values: {pixels}")

                mask_img.save(nnunet_mask_path) # Save potentially converted mask
                # If conversion is not needed or risky, use shutil:
                # shutil.copyfile(mask_path, nnunet_mask_path)

            processed_files.append(case_id)
            count += 1
            if count % 20 == 0: # Adjust reporting frequency if needed
                print(f"Processed {count}/{len(image_files)} images")
        except Exception as e:
            print(f"Error processing {img_file} or its mask {mask_file}: {e}")
            # Optionally remove partially processed files
            if nnunet_img_path.exists(): nnunet_img_path.unlink()
            if nnunet_mask_path.exists(): nnunet_mask_path.unlink()

    
    # Create dataset.json
    if count > 0:
        create_dataset_json(dataset_path, count)
        print(f"\nSuccessfully processed {count} image/mask pairs.")
        # You might want to list the actual case IDs used in dataset.json as well
        # This can be done by modifying create_dataset_json or here
        # Example: Add "training": [{"image": f"./imagesTr/{case}_0000.png", "label": f"./labelsTr/{case}.png"} for case in processed_files] to dataset.json
    else:
        print("\nNo image/mask pairs were processed.")

    print("NSCLC Dataset preparation complete!")

if __name__ == "__main__":
    main()