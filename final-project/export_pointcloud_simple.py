#!/usr/bin/env python3
"""
Simple export using camera rendering to get depth maps and create point cloud
"""
import sys
import torch
from pathlib import Path
import numpy as np
import open3d as o3d
from tqdm import tqdm

# Patch torch.load for PyTorch 2.6
original_load = torch.load
def patched_load(f, *args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(f, *args, **kwargs)
torch.load = patched_load

# Add nerfstudio to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'nerfstudio'))

from nerfstudio.utils.eval_utils import eval_setup


def export_pointcloud_from_cameras(config_path: Path, output_dir: Path):
    """Export point cloud by rendering depth from training cameras"""
    
    print(f"Loading NeRF model from: {config_path}")
    
    # Load model
    _, pipeline, _, _ = eval_setup(config_path, test_mode='inference')
    
    # Get cameras
    datamanager = pipeline.datamanager
    cameras = datamanager.train_dataset.cameras
    
    print(f"Rendering depth from {len(cameras)} cameras...")
    
    all_points = []
    all_colors = []
    
    with torch.no_grad():
        for idx in tqdm(range(len(cameras))):
            # Get camera
            camera = cameras[idx:idx+1].to(pipeline.device)
            
            # Generate ray bundle
            ray_bundle = camera.generate_rays(camera_indices=0)
            
            # Render
            outputs = pipeline.model.get_outputs_for_camera_ray_bundle(ray_bundle)
            
            # Get depth and RGB
            depth = outputs['depth'].cpu().numpy().squeeze()
            rgb = outputs['rgb'].cpu().numpy().squeeze()
            
            # Get camera pose
            c2w = camera.camera_to_worlds[0].cpu().numpy()  # [4, 4]
            
            # Get intrinsics
            fx = camera.fx[0].item()
            fy = camera.fy[0].item()
            cx = camera.cx[0].item()
            cy = camera.cy[0].item()
            
            height, width = depth.shape
            
            # Create point cloud from depth
            y, x = np.mgrid[0:height, 0:width]
            
            # Filter valid depths
            valid_mask = (depth > 0) & (depth < 100)  # Filter outliers
            
            x_valid = x[valid_mask]
            y_valid = y[valid_mask]
            depth_valid = depth[valid_mask]
            rgb_valid = rgb[valid_mask]
            
            # Unproject to camera space
            x_cam = (x_valid - cx) * depth_valid / fx
            y_cam = (y_valid - cy) * depth_valid / fy
            z_cam = depth_valid
            
            # Stack to [N, 3]
            pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)
            
            # Transform to world space
            pts_world = pts_cam @ c2w[:3, :3].T + c2w[:3, 3]
            
            all_points.append(pts_world)
            all_colors.append(rgb_valid)
    
    # Combine all points
    all_points = np.concatenate(all_points, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)
    
    print(f"Total points before filtering: {len(all_points):,}")
    
    # Downsample to target count
    target_points = 1_000_000
    if len(all_points) > target_points:
        indices = np.random.choice(len(all_points), target_points, replace=False)
        all_points = all_points[indices]
        all_colors = all_colors[indices]
    
    print(f"Final point count: {len(all_points):,}")
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(all_colors, 0, 1).astype(np.float64))
    
    # Remove statistical outliers
    print("Removing outliers...")
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    # Estimate normals
    print("Estimating normals...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    
    # Save
    output_path = output_dir / "diffmatch_nerf_dense.ply"
    o3d.io.write_point_cloud(str(output_path), pcd)
    
    print(f"\n✓ Saved dense point cloud: {output_path}")
    print(f"  Points: {len(pcd.points):,}")
    
    return output_path


if __name__ == "__main__":
    config_path = Path("ours/output/top_k_noransac/nerf/unnamed/nerfacto/2025-12-24_210822/config.yml")
    output_dir = Path("ours/output/top_k_noransac")
    
    export_pointcloud_from_cameras(config_path, output_dir)
