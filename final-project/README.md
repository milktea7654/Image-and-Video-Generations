# Final Project

3D reconstruction experiments using feature matching, COLMAP/hloc, NeRF, and point-cloud visualization.

## Folder Structure

```text
final-project/
├── README.md
├── requirements.txt
├── diffmatch_matcher.py          # DiffMatch feature extraction and pair matching helpers
├── run_diffmatch_pipeline.py     # DiffMatch + hloc + NeRF pipeline
├── lightglue_baseline.py         # LightGlue + hloc + NeRF baseline
├── vggt_baseline.py              # VGGT reconstruction baseline
├── export_pointcloud.py          # point-cloud export from NeRF outputs
├── export_pointcloud_simple.py   # simplified point-cloud export
├── visualize_pointclouds.py      # point-cloud visualization
├── visualize_reconstruction.py   # COLMAP reconstruction visualization
├── visualize_static.py           # static result visualization
├── dataset/                      # local input images, ignored by git
├── output/                       # generated reconstruction outputs, ignored by git
└── models/                       # external model repositories
```

## External Models

The `models/` directory is managed through git submodules:

- `models/ControlNet`
- `models/DiffMatch`
- `models/LightGlue`
- `models/nerfstudio`
- `models/vggt`

Initialize them with:

```bash
git submodule update --init --recursive
```

## Setup

```bash
pip install -r requirements.txt
```

Some pipelines also require the dependencies of the corresponding external model under `models/`.

## Example Commands

Run the DiffMatch reconstruction pipeline:

```bash
python run_diffmatch_pipeline.py \
  --image_dir dataset/dslr_images_undistorted \
  --output_dir output/diffmatch \
  --model_path models/DiffMatch/model_best_dped.pt
```

Run the LightGlue baseline:

```bash
python lightglue_baseline.py \
  --image_dir dataset/dslr_images_undistorted \
  --output_dir output/lightglue
```

Run the VGGT baseline:

```bash
python vggt_baseline.py \
  --image_dir dataset/dslr_images_undistorted \
  --output_dir output/vggt
```

## Notes

- Keep raw images in `dataset/`.
- Keep generated COLMAP, NeRF, point-cloud, and visualization files in `output/`.
- Do not commit large datasets, checkpoints, or generated outputs unless required for submission.
