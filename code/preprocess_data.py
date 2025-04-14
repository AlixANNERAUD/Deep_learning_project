#!/usr/bin/env python3
"""
HoVer-Net Data Preprocessing Script

This script converts TIFF images and GeoJSON annotations into the format required by HoVer-Net:
- For instance segmentation: [RGB, inst] arrays
- For instance segmentation + classification: [RGB, inst, type] arrays

Usage:
  python preprocess_data.py --images_dir path/to/tiff/images --nuclei_dir path/to/nuclei/geojson
                           [--tissue_dir path/to/tissue/geojson] [--output_dir path/to/output]
                           [--with_types] [--type_key classification]

Author: GitHub Copilot
"""

import os
import argparse
import json
import numpy as np
import tifffile
import cv2
import glob
from tqdm import tqdm
from typing import List, Tuple
from pydantic import field_validator, BaseModel, Field
from pathlib import Path
from typing import Literal, Union


class GeometryPolygon(BaseModel):
    """GeoJSON Polygon geometry."""

    type: Literal["Polygon"]
    coordinates: list[list[list[float]]]


class GeometryMultiPolygon(BaseModel):
    """GeoJSON MultiPolygon geometry."""

    type: Literal["MultiPolygon"]
    coordinates: list[list[list[list[float]]]]


class Classification(BaseModel):
    """Classification information for a nuclear type."""

    name: str
    color: list[int]


class Properties(BaseModel):
    """Properties of a GeoJSON feature."""

    object_type: str = Field("objectType")
    classification: Classification


class Feature(BaseModel):
    """GeoJSON Feature."""

    type: str
    id: str
    geometry: Union[GeometryPolygon, GeometryMultiPolygon] = Field(discriminator="type")
    properties: Properties

    @field_validator("type")
    @classmethod
    def validate_type(cls, v):
        if v != "Feature":
            raise ValueError(f'Expected feature type "Feature", got "{v}"')
        return v


class GeoJSONData(BaseModel):
    """Complete GeoJSON object."""

    type: str = Field("FeatureCollection")
    features: List[Feature]

    @field_validator("type")
    @classmethod
    def validate_type(cls, v):
        if v != "FeatureCollection":
            raise ValueError(f'Expected type "FeatureCollection", got "{v}"')
        return v


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess data for HoVer-Net training"
    )

    parser.add_argument(
        "--images", required=True, help="Directory containing TIFF images", type=Path
    )
    parser.add_argument(
        "--nuclei",
        required=True,
        help="Directory containing nuclei GeoJSON annotations",
        type=Path,
    )
    parser.add_argument(
        "--tissue",
        help="Directory containing tissue GeoJSON annotations (optional)",
        type=Path,
    )
    parser.add_argument(
        "--output",
        default="processed_data",
        help="Output directory for processed numpy arrays",
        type=Path,
    )
    parser.add_argument(
        "--with-types",
        action="store_true",
        help="Include type information (for classification)",
    )
    parser.add_argument(
        "--type-key",
        default="classification",
        help="Key in GeoJSON properties containing type information",
        type=str,
    )

    return parser.parse_args()


def load_geojson(path: Path) -> GeoJSONData:
    """Load and validate GeoJSON data using Pydantic models."""
    try:
        file = path.open("r")

        return GeoJSONData.model_validate_json(file.read())
    except Exception as e:
        raise ValueError(f"Could not load GeoJSON file {path}: {e}")


def create_type_mapping(nuclei_dir, type_key="classification"):
    """Create a mapping from type names to integers."""
    type_set = set()

    # Collect all unique types from GeoJSON files
    for geojson_file in glob.glob(os.path.join(nuclei_dir, "*.geojson")):
        data = load_geojson(Path(geojson_file))
        if data is None:
            continue

        for feature in data.features:
            # Try to get classification info using the provided key
            type_info = None
            if type_key == "classification" and hasattr(
                feature.properties, "classification"
            ):
                type_info = feature.properties.classification
            elif type_key == "nucleus_type" and hasattr(
                feature.properties, "nucleus_type"
            ):
                type_info = feature.properties.nucleus_type
            elif type_key == "type" and hasattr(feature.properties, "type"):
                type_info = feature.properties.type

            # Add type to set if found
            if type_info and hasattr(type_info, "name"):
                type_set.add(type_info.name)

    # Create mapping (background is 0)
    type_mapping = {typename: idx + 1 for idx, typename in enumerate(sorted(type_set))}

    # Print summary
    print(f"Found {len(type_mapping)} cell types:")
    for typename, idx in type_mapping.items():
        print(f"  {typename}: {idx}")

    # Save mapping to JSON
    type_info = {"type_info": {"0": {"name": "background", "color": [0, 0, 0]}}}

    for typename, idx in type_mapping.items():
        # Assign a random color for visualization
        color = np.random.randint(0, 256, 3).tolist()
        type_info["type_info"][str(idx)] = {"name": typename, "color": color}

    return type_mapping, type_info


def geojson_to_masks(
    geojson_path: Path,
    image_shape: Tuple[int, int, int],
    type_mapping: dict[str, int] | None = None,
    type_key: str = "classification",
):
    """Convert GeoJSON annotations to instance and type masks."""
    # Initialize masks
    inst_mask = np.zeros(image_shape[:2], dtype=np.int32)
    type_mask = np.zeros(image_shape[:2], dtype=np.int32) if type_mapping else None

    # Load GeoJSON with Pydantic validation
    data = load_geojson(geojson_path)
    if data is None:
        return inst_mask, type_mask

    # Process each annotation
    for i, feature in enumerate(data.features, 1):  # Start from 1, 0 is background
        try:
            # Handle different geometry types
            if feature.geometry.type == "Polygon":
                polygons = [feature.geometry.coordinates[0]]
            elif feature.geometry.type == "MultiPolygon":
                polygons = [poly[0] for poly in feature.geometry.coordinates]
            else:
                print(f"Unsupported geometry type: {feature.geometry.type}, skipping")
                continue

            # Process each polygon
            for polygon_coords in polygons:
                # Ensure coordinates are valid and properly formatted
                try:
                    # Convert to numpy array and ensure int32 type
                    points = np.array(polygon_coords, dtype=np.int32)

                    # Check if points have the right shape
                    if len(points.shape) != 2 or points.shape[1] != 2:
                        print(
                            f"Malformed polygon in feature {i}, skipping: {points.shape}"
                        )
                        continue

                    # Create instance mask
                    cv2.fillPoly(inst_mask, [points], i)

                    # Create type mask if needed
                    if type_mapping:
                        # Try to get classification info using the provided key
                        type_info = None
                        if type_key == "classification" and hasattr(
                            feature.properties, "classification"
                        ):
                            type_info = feature.properties.classification
                        elif type_key == "nucleus_type" and hasattr(
                            feature.properties, "nucleus_type"
                        ):
                            type_info = feature.properties.nucleus_type
                        elif type_key == "type" and hasattr(feature.properties, "type"):
                            type_info = feature.properties.type

                        if (
                            type_info
                            and hasattr(type_info, "name")
                            and type_info.name in type_mapping
                        ):
                            type_int = type_mapping[type_info.name]
                            cv2.fillPoly(type_mask, [points], type_int)
                except ValueError as ve:
                    print(f"Invalid polygon coordinates in feature {i}, skipping: {ve}")
                    continue
                except Exception as e:
                    print(f"Error processing polygon for feature {i}: {e}")
                    continue
        except Exception as e:
            print(f"Error processing feature {i}: {e}")
            continue

    return inst_mask, type_mask


def process_image(
    image_path, nuclei_dir, output_dir, type_mapping=None, type_key="classification"
):
    """Process a single image with its annotations."""
    try:
        print(f"Processing {image_path}")
        # Load image
        img = tifffile.imread(image_path)

        # Ensure image is RGB
        if len(img.shape) == 2:  # Grayscale
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[2] > 3:  # More than 3 channels
            img = img[:, :, :3]

        # Get corresponding GeoJSON path
        basename = os.path.basename(image_path).split(".")[0]
        geojson_path = nuclei_dir / f"{basename}_tissue.geojson"

        # Convert GeoJSON to masks
        inst_mask, type_mask = geojson_to_masks(
            geojson_path, img.shape, type_mapping, type_key
        )

        # Create output array
        if type_mapping is not None:
            # [RGB, inst, type] format for classification
            output = np.zeros((img.shape[0], img.shape[1], 5), dtype=np.uint8)
            output[:, :, :3] = img
            output[:, :, 3] = inst_mask
            output[:, :, 4] = type_mask
        else:
            # [RGB, inst] format for instance segmentation only
            output = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
            output[:, :, :3] = img
            output[:, :, 3] = inst_mask

        # Save as numpy array
        output_path = os.path.join(output_dir, f"{basename}.npy")
        np.save(output_path, output)

        return True
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False


def main():
    """Main function."""
    args = parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Create type mapping if needed
    type_mapping = None
    if args.with_types:
        print("Creating type mapping...")
        type_mapping, type_info = create_type_mapping(args.nuclei, args.type_key)
        with open(os.path.join(args.output, "type_info.json"), "w") as f:
            json.dump(type_info, f, indent=2)

    # Process all TIFF images
    image_files = glob.glob(os.path.join(args.images, "*.tif")) + glob.glob(
        os.path.join(args.images, "*.tiff")
    )

    success_count = 0
    for image_path in tqdm(image_files, desc="Processing images"):
        if process_image(
            image_path, args.nuclei, args.output, type_mapping, args.type_key
        ):
            success_count += 1

    print(f"Processed {success_count}/{len(image_files)} images successfully")
    print(f"Output saved to {args.output}")
    print(f"Next step: Use extract_patches.py to create training patches")


if __name__ == "__main__":
    main()
