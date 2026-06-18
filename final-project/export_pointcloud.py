#!/usr/bin/env python3
"""
Export NeRF point cloud manually (workaround for PyTorch 2.6 weights_only issue)
"""
import sys
import torch
from pathlib import Path

# Set weights_only=False for loading checkpoint
import torch.serialization
original_load = torch.load

def patched_load(f, *args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(f, *args, **kwargs)

torch.load = patched_load

# Add nerfstudio to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'nerfstudio'))

from nerfstudio.utils.eval_utils import eval_setup


def export_pointcloud(config_path: Path, output_dir: Path, num_points: int = 1000000):
    """Export point cloud from trained NeRF model"""
    import open3d as o3d
    import numpy as np
    from tqdm import tqdm
    
    print(f"Loading NeRF model from: {config_path}")
    
    # Load model
    _, pipeline, _, _ = eval_setup(config_path, test_mode='inference')
    
    print(f"Generating {num_points:,} points...")
    
    # Get scene bounds
    datamanager = pipeline.datamanager
    cameras = datamanager.train_dataset.cameras
    
    # Compute scene bounds from cameras
    camera_positions = []
    for idx in range(len(cameras)):
        cam = cameras[idx:idx+1]
        camera_positions.append(cam.camera_to_worlds[0, :3, 3].cpu().numpy())
    
    camera_positions = np.array(camera_positions)
    scene_center = camera_positions.mean(axis=0)
    scene_radius = np.linalg.norm(camera_positions - scene_center, axis=1).max()
    
    print(f"Scene center: {scene_center}")
    print(f"Scene radius: {scene_radius:.2f}")
    
    # Sample points in scene bounds
    bound_min = scene_center - scene_radius * 1.2
    bound_max = scene_center + scene_radius * 1.2
    
    # Generate random points
    points = np.random.uniform(bound_min, bound_max, size=(num_points * 2, 3))
    
    # Query density for each point using rendering
    batch_size = 4096
    points_tensor = torch.from_numpy(points).float().to(pipeline.device)
    
    densities = []
    colors = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(points_tensor), batch_size), desc="Querying NeRF"):
            batch_points = points_tensor[i:i+batch_size]
            
            # Create RaySamples for the points
            from nerfstudio.cameras.rays import RaySamples
            from nerfstudio.data.scene_box import SceneBox
            import torch
            
            # Create frustums
            from jaxtyping import Float
            from nerfstudio.cameras.rays import Frustums
            
            # Hack: create minimal frustums for point queries
            # We just need positions, so we create dummy frustums
            positions = batch_points.unsqueeze(0)  # [1, N, 3]
            directions = torch.zeros_like(positions)
            directions[..., 2] = 1  # point in z direction
            
            starts = torch.zeros((1, len(batch_points), 1), device=pipeline.device)
            ends = torch.ones((1, len(batch_points), 1), device=pipeline.device) * 0.001
            
            frustums = Frustums(
                origins=positions,
                directions=directions,
                starts=starts,
                ends=ends,
                pixel_area=torch.ones((1, len(batch_points), 1), device=pipeline.device),
            )
            
            ray_samples = RaySamples(
                frustums=frustums,
            )
            
            # Query model
            field_outputs = pipeline.model.field(ray_samples)
            density = field_outputs['density'].squeeze(0).cpu().numpy()
            rgb = field_outputs.get('rgb', torch.zeros(len(batch_points), 3)).squeeze(0).cpu().numpy()
            
            densities.append(density)
            colors.append(rgb)
    
    densities = np.concatenate(densities).squeeze()
    colors = np.concatenate(colors)
    
    # Filter by density threshold (keep high density points)
    density_threshold = np.percentile(densities, 50)  # Keep top 50%
    mask = densities > density_threshold
    
    filtered_points = points[mask]
    filtered_colors = colors[mask]
    
    # Subsample to target number
    if len(filtered_points) > num_points:
        indices = np.random.choice(len(filtered_points), num_points, replace=False)
        filtered_points = filtered_points[indices]
        filtered_colors = filtered_colors[indices]
    
    print(f"Filtered to {len(filtered_points):,} points")
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_points)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(filtered_colors, 0, 1))
    
    # Estimate normals
    print("Estimating normals...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    
    # Save point cloud
    output_path = output_dir / "diffmatch_nerf_dense.ply"
    o3d.io.write_point_cloud(str(output_path), pcd)
    print(f"✓ Saved dense point cloud: {output_path}")
    print(f"  Points: {len(filtered_points):,}")
    
    return output_path


if __name__ == "__main__":
    config_path = Path("ours/output/top_k_noransac/nerf/unnamed/nerfacto/2025-12-24_210822/config.yml")
    output_dir = Path("ours/output/top_k_noransac")
    
    export_pointcloud(config_path, output_dir, num_points=1000000)
