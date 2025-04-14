#!/usr/bin/env python3
"""
HoVer-Net Data Preprocessing Script

This script converts TIFF images and GeoJSON annotations into the format required by HoVer-Net:
- For instance segmentation: [RGB, inst] arrays
- For instance segmentation + classification: [RGB, inst, type] arrays

Usage:
  python preprocess_data.py --images_dir path/to/tiff/images --nuclei_dir path/to/nuclei/geojson
                           [--tissue_dir path/to/tissue/geojson] [--output_dir path/to/output]
                           [--with_types] [--type_key classification] [--visualize]

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
import matplotlib.pyplot as plt
from skimage.color import label2rgb
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing


NUCLEI_MAP = {
    "tumor": 1,
    "lymphocyte": 2,
    "plasma_cell": 3,
    "histiocyte": 4,
    "melanophage": 5,
    "stroma": 7,
    "epithelium": 8,
    "endothelium": 9,
    "apoptosis": 10,
}

# Create tissue type mapping
TISSUE_MAP = {
    "tumor": 1,
    "stroma": 2,
    "epithelium": 3,
    "blood_vessel": 4,
    "necrotic": 5,
}


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
        "--nucleis",
        required=True,
        help="Directory containing nuclei GeoJSON annotations",
        type=Path,
    )
    parser.add_argument(
        "--tissues",
        required=True,
        help="Directory containing tissue GeoJSON annotations",
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
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization images of the masks for verification",
    )
    parser.add_argument(
        "--max-threads",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Maximum number of threads to use for parallel processing (default: all CPU cores)",
    )

    return parser.parse_args()


def load_geojson(path: Path) -> GeoJSONData:
    """Load and validate GeoJSON data using Pydantic models."""
    file = path.open("r")

    return GeoJSONData.model_validate_json(file.read())


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
    is_tissue: bool = False,
):
    """Convert GeoJSON annotations to instance and type masks.

    Args:
        geojson_path: Path to the GeoJSON file
        image_shape: Shape of the image
        type_mapping: Mapping from type names to integers
        type_key: Key in GeoJSON properties containing type information
        is_tissue: Whether this is a tissue annotation (for semantic segmentation)

    Returns:
        For nuclei: instance mask and type mask
        For tissue: semantic segmentation mask
    """
    # Initialize masks
    if is_tissue:
        # For tissue segmentation, we only need a semantic mask
        tissue_mask = np.zeros(image_shape[:2], dtype=np.int32)

        # Load GeoJSON with Pydantic validation
        data = load_geojson(geojson_path)
        if data is None:
            return tissue_mask

        # Process each tissue annotation
        for feature in data.features:
            # Get tissue type
            tissue_type = feature.properties.classification.name

            # Skip if no tissue type or not in mapping
            if not tissue_type or (type_mapping and tissue_type not in type_mapping):
                continue

            # Get tissue type value
            tissue_value = type_mapping.get(tissue_type, 0) if type_mapping else 1

            # Handle different geometry types
            if feature.geometry.type == "Polygon":
                polygons = [feature.geometry.coordinates[0]]
            elif feature.geometry.type == "MultiPolygon":
                polygons = [poly[0] for poly in feature.geometry.coordinates]
            else:
                print(f"Unsupported geometry type: {feature.geometry.type}, skipping")
                continue

            # Fill polygons in the mask
            for polygon_coords in polygons:
                # Convert to numpy array and ensure int32 type
                points = np.array(polygon_coords, dtype=np.int32)

                # Check if points have the right shape
                if len(points.shape) != 2 or points.shape[1] != 2:
                    print(
                        f"Malformed polygon in tissue feature, skipping: {points.shape}"
                    )
                    continue

                # Create tissue mask
                cv2.fillPoly(tissue_mask, [points], tissue_value)

        return tissue_mask
    else:
        # For nuclei: instance and type masks
        inst_mask = np.zeros(image_shape[:2], dtype=np.int32)
        type_mask = np.zeros(image_shape[:2], dtype=np.int32) if type_mapping else None

        # Load GeoJSON with Pydantic validation
        data = load_geojson(geojson_path)
        if data is None:
            return inst_mask, type_mask

        # Process each annotation
        for i, feature in enumerate(data.features, 1):  # Start from 1, 0 is background
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
                # Convert to numpy array and ensure int32 type
                points = np.array(polygon_coords, dtype=np.int32)

                # Check if points have the right shape
                if len(points.shape) != 2 or points.shape[1] != 2:
                    print(f"Malformed polygon in feature {i}, skipping: {points.shape}")
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

        return inst_mask, type_mask


def visualize_masks(img, inst_mask, type_mask, basename, output_dir):
    """Generate visualization of the masks for verification."""

    if img.shape[2] > 3:
        print(f"Image {basename} has more than 3 channels, skipping visualization")

    # Create visualization directory
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    # 1. Instance mask visualization (random colors)
    inst_overlay = label2rgb(inst_mask, bg_label=0, alpha=0.5)

    figure, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Original image
    axes[0, 0].imshow(img)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    # Instance mask only
    axes[0, 1].imshow(inst_overlay)
    axes[0, 1].set_title("Instance Mask")
    axes[0, 1].axis("off")

    # Overlay instance mask on image
    axes[1, 0].imshow(img)
    axes[1, 0].imshow(inst_overlay, alpha=0.5)
    axes[1, 0].set_title("Instance Mask Overlay")
    axes[1, 0].axis("off")

    # Type mask if available
    if type_mask is not None:
        type_overlay = label2rgb(type_mask, bg_label=0, alpha=0.5)
        axes[1, 1].imshow(img)
        axes[1, 1].imshow(type_overlay, alpha=0.5)
        axes[1, 1].set_title("Type Mask Overlay")
    else:
        # Contour visualization as alternative
        contour_img = img.copy()

        # Find contours for each instance
        unique_instances = np.unique(inst_mask)
        unique_instances = unique_instances[unique_instances > 0]  # Skip background

        # Draw contours
        contours = []
        for inst_id in unique_instances:
            binary_mask = (inst_mask == inst_id).astype(np.uint8)
            contour, _ = cv2.findContours(
                binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            contours.extend(contour)

        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
        axes[1, 1].imshow(contour_img)
        axes[1, 1].set_title("Contour Visualization")

    axes[1, 1].axis("off")

    # Save figure
    figure.tight_layout()
    figure.savefig(os.path.join(vis_dir, f"{basename}_visualization.png"), dpi=200)
    plt.close(figure)


def process_image(
    image_path,
    nuclei_dir,
    tissues_dir,
    output_dir,
    type_mapping=None,
    type_key="classification",
):
    """Process a single image with its annotations.

    Args:
        image_path: Path to the image file
        nuclei_dir: Directory containing nuclei annotations
        output_dir: Directory to save processed data
        type_mapping: Mapping from type names to integers (for cell classification)
        type_key: Key in GeoJSON properties containing type information
        tissues_dir: Directory containing tissue annotations (for semantic segmentation)
    """

    # Load image
    img = tifffile.imread(image_path)

    # Ensure image is RGB
    if len(img.shape) == 2:  # Grayscale
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[2] > 3:  # More than 3 channels
        img = img[:, :, :3]

    # Get base filename
    basename = os.path.basename(image_path).split(".")[0]

    # Process nuclei annotations
    nuclei_geojson_path = nuclei_dir / f"{basename}_nuclei.geojson"
    
    # Use the global NUCLEI_MAP instead of the passed type_mapping
    # Convert nuclei GeoJSON to masks
    inst_mask, type_mask = geojson_to_masks(
        nuclei_geojson_path, img.shape, NUCLEI_MAP, type_key, is_tissue=False
    )

    # Process tissue annotations (now required)
    tissue_mask = None
    tissue_geojson_path = tissues_dir / f"{basename}_tissue.geojson"

    # Use the global TISSUE_MAP instead of creating a new mapping
    # Convert tissue GeoJSON to semantic segmentation mask
    tissue_mask = geojson_to_masks(
        tissue_geojson_path,
        img.shape,
        TISSUE_MAP,
        type_key,
        is_tissue=True,
    )

    # [RGB, inst, type] format for classification
    output = np.zeros((img.shape[0], img.shape[1], 5), dtype=np.uint8)
    output[:, :, :3] = img
    output[:, :, 3] = type_mask
    output[:, :, 4] = tissue_mask

    # Save as numpy array
    output_path = os.path.join(output_dir, f"{basename}.npy")
    np.save(output_path, output)


def process_file(npy_file, output_dir):
    data = np.load(npy_file)
    basename = os.path.basename(npy_file).split(".")[0]

    img = data[:, :, :3]
    inst_mask = data[:, :, 3]

    type_mask = data[:, :, 4] if data.shape[2] > 4 else None

    visualize_masks(img, inst_mask, type_mask, basename, output_dir)


def visualize_processed_data(
    output_dir, nuclei_dir, type_key="classification", max_threads=None
):
    """Generate visualizations for already processed data."""
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    npy_files = glob.glob(os.path.join(output_dir, "*.npy"))

    if not npy_files:
        print(f"No processed data found in {output_dir}")
        return 0

    print(f"Generating visualizations for {len(npy_files)} processed files...")

    success_count = 0
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {
            executor.submit(process_file, file, output_dir): file for file in npy_files
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Generating visualizations"
        ):
            if future.result():
                success_count += 1

    return success_count


def main():
    """Main function."""
    args = parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    plt.rcParams.update({"figure.max_open_warning": 0})

    # Process all TIFF images
    image_files = glob.glob(os.path.join(args.images, "*.tif")) + glob.glob(
        os.path.join(args.images, "*.tiff")
    )

    success_count = 0

    # Process images in parallel with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.max_threads) as executor:
        futures = {
            executor.submit(
                process_image,
                image_path,
                args.nucleis,
                args.tissues,
                args.output,
                args.type_key
            ): image_path
            for image_path in image_files
        }

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing images"
        ):
            if future.result():
                success_count += 1

    print(f"Processed {success_count}/{len(image_files)} images successfully")
    print(f"Output saved to {args.output}")

    # Generate visualizations as a separate step if requested
    if args.visualize:
        print("Generating visualizations...")
        vis_count = visualize_processed_data(
            args.output, args.nucleis, args.type_key, args.max_threads
        )
        print(f"Generated {vis_count} visualizations")
        print(f"Visualizations saved to {os.path.join(args.output, 'visualizations')}")

    print(f"Next step: Use extract_patches.py to create training patches")


if __name__ == "__main__":
    main()
