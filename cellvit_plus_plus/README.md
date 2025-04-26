```bash
uv run prepare_puma_for_cellvit_segmentation.py --puma_roi_dir ../dataset/01_training_dataset_tif_ROIs/01_training_dataset_tif_ROIs --puma_annotation_dir ../dataset/01_training_dataset_geojson_nuclei --val_split 0.2 --output_dir cellvit_data
```

```bash
pip install gdown
mkdir checkpoints
cd checkpoints
gdown 1Q38iiKgjtnggtzjOsBfqG9HhKvWYmHox
```

```bash
sbatch train_cellvit_classifier.slurm
```
