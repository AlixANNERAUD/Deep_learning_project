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
import matplotlib
from matplotlib import colormaps

COLOR_MAP = [(0.0, 0.0, 0.0), *colormaps["gist_rainbow"](np.linspace(0, 1, 12))]

OUTPUT_SHAPE = [270, 270]

NUCLEI_MAP = {
    "background": 0,
    "nuclei_tumor": 1,
    "nuclei_stroma": 2,
    "nuclei_epithelium": 3,
    "nuclei_histiocyte": 4,
    "nuclei_melanophage": 5,
    "nuclei_neutrophil": 6,
    "nuclei_lymphocyte": 7,
    "nuclei_plasma_cell": 8,
    "nuclei_endothelium": 9,
    "nuclei_apoptosis": 10,
}

NUCLEI_COLOR_MAP = {
    i: [
        int(COLOR_MAP[i][0] * 255),
        int(COLOR_MAP[i][1] * 255),
        int(COLOR_MAP[i][2] * 255),
    ]
    for i in range(len(NUCLEI_MAP))
}


# Create tissue type mapping
TISSUE_MAP = {
    "background": 0,
    "tissue_tumor": 1,
    "tissue_stroma": 2,
    "tissue_epithelium": 3,
    "tissue_blood_vessel": 4,
    "tissue_necrosis": 5,
    "tissue_epidermis": 6,
    "tissue_white_background": 7,
}

TISSUE_COLOR_MAP = {
    i: [
        int(COLOR_MAP[i][0] * 255),
        int(COLOR_MAP[i][1] * 255),
        int(COLOR_MAP[i][2] * 255),
    ]
    for i in range(len(TISSUE_MAP))
}


RGB_LAYERS = [0, 1, 2]
TISSUE_LAYER = 3
NUCLEI_LAYER = 4

TRAIN_TEST_SPLIT = 0.8

RANDOM_SEED = 42


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


def geojson_to_mask(
    geojson_path: Path,
    image_shape: Tuple[int, int, int],
    mapping: dict[str, int],
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

    # For nuclei: instance and type masks
    mask = np.zeros(image_shape[:2], dtype=np.int8)

    # Load GeoJSON with Pydantic validation
    data = load_geojson(geojson_path)

    # Process each annotation
    for i, feature in enumerate(data.features):  # Start from 1, 0 is background
        # Handle different geometry types
        if feature.geometry.type == "Polygon":
            polygons = [feature.geometry.coordinates[0]]
        elif feature.geometry.type == "MultiPolygon":
            polygons = [poly[0] for poly in feature.geometry.coordinates]
        else:
            raise ValueError(f"Unsupported geometry type: {feature.geometry.type}")

        # Process each polygon
        for polygon_coords in polygons:
            # Convert to numpy array and ensure int32 type
            points = np.array(polygon_coords, dtype=np.int32)

            # Check if points have the right shape
            if len(points.shape) != 2 or points.shape[1] != 2:
                print(f"Malformed polygon in feature {i}, skipping: {points.shape}")
                continue

            # Try to get classification info using the provided key
            type_int = mapping[feature.properties.classification.name]

            cv2.fillPoly(mask, [points], type_int)

    return mask


def plot_overlay(image, mask, axe, color_map, mapping):
    """Plot overlay of the mask on the image."""

    # Normalize image to 0-1 range
    if image.dtype == np.uint8:
        img_normalized = image.astype(float) / 255.0
    else:
        img_normalized = (image - image.min()) / (image.max() - image.min() + 1e-8)
        img_normalized = np.clip(img_normalized, 0, 1)

    # Create color array for the mask
    mask_colors = np.zeros((*mask.shape, 3), dtype=float)

    # Apply colors based on the mask values
    for label, color in color_map.items():
        mask_region = mask == label
        if np.any(mask_region):
            for c in range(3):
                mask_colors[:, :, c][mask_region] = color[c] / 255.0

    # Create the overlay by blending
    alpha = 0.5
    overlay = np.zeros_like(img_normalized)

    # Only apply color where mask is not background (0)
    foreground = mask > 0

    for c in range(3):
        overlay[:, :, c] = img_normalized[:, :, c].copy()
        overlay[:, :, c][foreground] = (
            alpha * mask_colors[:, :, c][foreground]
            + (1 - alpha) * img_normalized[:, :, c][foreground]
        )

    # Display the overlay
    axe.imshow(overlay)
    axe.axis("off")

    # Create legend elements
    patches = []
    labels = []

    for label_id, color in color_map.items():
        if label_id == 0:  # Skip background
            continue

        # Get type name from mapping
        type_name = None
        for name, idx in mapping.items():
            if idx == label_id:
                type_name = name
                break

        if type_name:
            normalized_color = [c / 255.0 for c in color]
            patches.append(plt.Rectangle((0, 0), 1, 1, fc=normalized_color))
            labels.append(type_name)

    # Add legend if we have items
    if patches:
        # Place legend below the plot
        axe.legend(
            patches,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            fontsize="x-small",
            framealpha=0.7,
            ncol=2 if len(patches) > 5 else 1,
        )


def visualize_masks(img, tissue_mask, nuclei_mask, basename, output_dir):
    """Generate visualization of the masks for verification."""

    if img.shape[2] > 3:
        print(f"Image {basename} has more than 3 channels, skipping visualization")
        return

    # Create visualization directory
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Create a smaller figure with lower DPI for faster rendering
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=100)

    axes[0].imshow(img)
    axes[0].axis("off")
    axes[0].set_title(f"Original image ({img.shape[0]}x{img.shape[1]})")

    plot_overlay(img, tissue_mask, axes[1], TISSUE_COLOR_MAP, TISSUE_MAP)
    axes[1].set_title("Tissue overlay")

    plot_overlay(img, nuclei_mask, axes[2], NUCLEI_COLOR_MAP, NUCLEI_MAP)
    axes[2].set_title("Nuclei overlay")

    # Save figure with lower DPI and without tight_layout
    figure.savefig(vis_dir / f"{basename}_vis.png", dpi=100, bbox_inches="tight")
    plt.close(figure)


def extract_patches(image, patch_size):
    """Extract patches of specified size from an image.

    Args:
        image: Input image or mask
        patch_size: Size of patches [height, width]

    Returns:
        List of patches and their coordinates [(patch, (y, x)), ...]
    """
    height, width = image.shape[:2]
    patches = []

    # Extract patches with a sliding window approach
    for y in range(0, height - patch_size[0] + 1, patch_size[0]):
        for x in range(0, width - patch_size[1] + 1, patch_size[1]):
            if len(image.shape) == 3:  # RGB image
                patch = image[y : y + patch_size[0], x : x + patch_size[1], :]
            else:  # Mask (2D)
                patch = image[y : y + patch_size[0], x : x + patch_size[1]]

            patches.append((patch, (y, x)))

    return patches


def process_image(
    image_path,
    nuclei_dir,
    tissues_dir,
    output_dir,
):
    """Process a single image with its annotations.

    Args:
        image_path: Path to the image file
        nuclei_dir: Directory containing nuclei annotations
        output_dir: Directory to save processed data
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
    nuclei_mask = geojson_to_mask(nuclei_geojson_path, img.shape, NUCLEI_MAP)

    # Process tissue annotations (now required)
    tissue_geojson_path = tissues_dir / f"{basename}_tissue.geojson"

    # Use the global TISSUE_MAP instead of creating a new mapping
    # Convert tissue GeoJSON to semantic segmentation mask
    tissue_mask = geojson_to_mask(
        tissue_geojson_path,
        img.shape,
        TISSUE_MAP,
    )

    # Extract patches of size OUTPUT_SHAPE
    img_patches = extract_patches(img, OUTPUT_SHAPE)
    nuclei_mask_patches = extract_patches(nuclei_mask, OUTPUT_SHAPE)
    tissue_mask_patches = extract_patches(tissue_mask, OUTPUT_SHAPE)

    # Process and save each patch
    for i, ((img_patch, (x, y)), (nuclei_patch, _), (tissue_patch, _)) in enumerate(
        zip(img_patches, nuclei_mask_patches, tissue_mask_patches)
    ):
        # [RGB, inst, type] format for classification
        output = np.zeros((OUTPUT_SHAPE[0], OUTPUT_SHAPE[1], 5), dtype=np.uint8)
        output[:, :, RGB_LAYERS] = img_patch
        output[:, :, NUCLEI_LAYER] = nuclei_patch
        output[:, :, TISSUE_LAYER] = tissue_patch

        # Save as numpy array
        patch_output_path = os.path.join(output_dir, f"{basename}_y{y}_x{x}.npy")
        np.save(patch_output_path, output)


def process_file(npy_file, output_dir):
    data = np.load(npy_file)
    basename = os.path.basename(npy_file).split(".")[0]

    img = data[:, :, RGB_LAYERS]
    tissue_mask = data[:, :, TISSUE_LAYER]
    nuclei_mask = data[:, :, NUCLEI_LAYER]

    visualize_masks(img, tissue_mask, nuclei_mask, basename, output_dir)


def split_train_test(output_dir, train_ratio=TRAIN_TEST_SPLIT):
    """Split processed data into train and test sets.
    
    Args:
        output_dir: Directory containing processed data
        train_ratio: Ratio of data to use for training (default: 0.8)
    """
    # Get all numpy files
    npy_files = glob.glob(os.path.join(output_dir, "*.npy"))
    
    if not npy_files:
        print(f"No processed data found in {output_dir}")
        return

    # Shuffle files to ensure random split
    np.random.shuffle(npy_files)
    
    # Calculate split index
    split_idx = int(len(npy_files) * train_ratio)
    
    # Split into train and test sets
    train_files = npy_files[:split_idx]
    test_files = npy_files[split_idx:]
    
    # Create train and test directories
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    print(f"Moving {len(train_files)} files to train set...")
    for file in tqdm(train_files, desc="Moving to train set"):
        basename = os.path.basename(file)
        dst = os.path.join(train_dir, basename)
        # Move instead of copy
        os.rename(file, dst)
    
    print(f"Moving {len(test_files)} files to test set...")
    for file in tqdm(test_files, desc="Moving to test set"):
        basename = os.path.basename(file)
        dst = os.path.join(test_dir, basename)
        # Move instead of copy
        os.rename(file, dst)
    
    print(f"Data split complete: {len(train_files)} training samples, {len(test_files)} test samples")
    return len(train_files), len(test_files)


def visualize_processed_data(output_dir, nuclei_dir, max_threads=None):
    """Generate visualizations for already processed data."""
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    npy_files = glob.glob(os.path.join(output_dir, "*.npy"))

    if not npy_files:
        print(f"No processed data found in {output_dir}")
        return 0

    print(f"Generating visualizations for {len(npy_files)} processed files...")

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {
            executor.submit(process_file, file, output_dir): file for file in npy_files
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Generating visualizations"
        ):
            future.result()


def main():
    """Main function."""
    args = parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    np.random.seed(RANDOM_SEED)

    # Use Agg backend for faster non-interactive rendering
    matplotlib.use("Agg")
    # Disable figure max open warning and set autolayout to False for speed
    plt.rcParams.update({"figure.max_open_warning": 0, "figure.autolayout": False})

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
            ): image_path
            for image_path in image_files
        }

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing images"
        ):
            try:
                future.result()
                success_count += 1
            except Exception as e:
                print(f"Error processing {futures[future]}: {str(e)}")

    print(f"Processed {success_count}/{len(image_files)} images successfully")
    print(f"Output saved to {args.output}")

    # Generate visualizations as a separate step if requested
    if args.visualize:
        print("Generating visualizations...")
        visualize_processed_data(args.output, args.nucleis, args.max_threads)
        print(f"Visualizations saved to {os.path.join(args.output, 'visualizations')}")

    # Split data into train and test sets
    train_count, test_count = split_train_test(args.output)
    
    print(f"Data split complete: {train_count} training samples, {test_count} test samples")
    print(f"Train data saved to {os.path.join(args.output, 'train')}")
    print(f"Test data saved to {os.path.join(args.output, 'test')}")
    
    print("Preprocessing complete!")


if __name__ == "__main__":
    main()
