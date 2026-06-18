#!/usr/bin/env python3
"""
Visualize 3D reconstruction results from LightGlue baseline.
Uses matplotlib for visualization instead of Open3D for better compatibility.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import pycolmap
from pathlib import Path
from scipy.stats import gaussian_kde


class Arrow3D(FancyArrowPatch):
    """3D arrow for camera visualization."""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


def visualize_reconstruction(sparse_dir: Path, output_dir: Path = None):
    """Visualize 3D reconstruction with point cloud and camera positions."""
    
    # Load reconstruction
    reconstruction = pycolmap.Reconstruction(str(sparse_dir))
    
    print(f"Loaded reconstruction:")
    print(f"  - Images: {len(reconstruction.images)}")
    print(f"  - Points: {len(reconstruction.points3D)}")
    print(f"  - Cameras: {len(reconstruction.cameras)}")
    
    # Extract point cloud
    points = []
    colors = []
    for point3D in reconstruction.points3D.values():
        points.append(point3D.xyz)
        colors.append(point3D.color / 255.0)  # Normalize to [0, 1]
    
    points = np.array(points)
    colors = np.array(colors)
    
    # Extract camera positions
    camera_positions = []
    for image in reconstruction.images.values():
        pose = image.cam_from_world
        cam_pos = -pose.rotation.matrix().T @ pose.translation
        camera_positions.append(cam_pos)
    
    camera_positions = np.array(camera_positions)
    
    # Print statistics
    print(f"\nScene statistics:")
    print(f"  Point cloud bounds:")
    print(f"    X: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}] (range: {points[:, 0].max() - points[:, 0].min():.2f})")
    print(f"    Y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}] (range: {points[:, 1].max() - points[:, 1].min():.2f})")
    print(f"    Z: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}] (range: {points[:, 2].max() - points[:, 2].min():.2f})")
    print(f"  Camera positions:")
    print(f"    X: [{camera_positions[:, 0].min():.2f}, {camera_positions[:, 0].max():.2f}] (range: {camera_positions[:, 0].max() - camera_positions[:, 0].min():.2f})")
    print(f"    Y: [{camera_positions[:, 1].min():.2f}, {camera_positions[:, 1].max():.2f}] (range: {camera_positions[:, 1].max() - camera_positions[:, 1].min():.2f})")
    print(f"    Z: [{camera_positions[:, 2].min():.2f}, {camera_positions[:, 2].max():.2f}] (range: {camera_positions[:, 2].max() - camera_positions[:, 2].min():.2f})")
    
    scene_center = points.mean(axis=0)
    print(f"  Scene center: ({scene_center[0]:.2f}, {scene_center[1]:.2f}, {scene_center[2]:.2f})")
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate 1: Single main view (elev=15°, azim=45°)
        fig1 = plt.figure(figsize=(20, 16))
        ax1 = fig1.add_subplot(111, projection='3d')
        ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=4, alpha=0.8)
        ax1.view_init(elev=15, azim=45)
        # Hide axes
        ax1.set_axis_off()
        ax1.set_box_aspect([1, 1, 1])
        # Zoom in by setting tighter limits around the point cloud center
        center = points.mean(axis=0)
        range_factor = 0.4  # Show 40% of the full range (zoom in)
        x_range = (points[:, 0].max() - points[:, 0].min()) * range_factor
        y_range = (points[:, 1].max() - points[:, 1].min()) * range_factor
        z_range = (points[:, 2].max() - points[:, 2].min()) * range_factor
        ax1.set_xlim([center[0] - x_range/2, center[0] + x_range/2])
        ax1.set_ylim([center[1] - y_range/2, center[1] + y_range/2])
        ax1.set_zlim([center[2] - z_range/2, center[2] + z_range/2])
        plt.tight_layout()
        main_view_path = output_dir / "reconstruction_main_view.png"
        plt.savefig(main_view_path, dpi=200, bbox_inches='tight', pad_inches=0)
        print(f"\n✓ Saved: {main_view_path}")
        plt.close(fig1)
        
        # Generate 2: Multiple main views (6 different angles)
        fig2 = plt.figure(figsize=(20, 15))
        views = [
            (15, 45, 'View 1: elev=15°, azim=45°'),
            (20, 60, 'View 2: elev=20°, azim=60°'),
            (15, 90, 'View 3: elev=15°, azim=90°'),
            (20, 30, 'View 4: elev=20°, azim=30°'),
            (10, 45, 'View 5: elev=10°, azim=45°'),
            (25, 45, 'View 6: elev=25°, azim=45°'),
        ]
        for idx, (elev, azim, title) in enumerate(views, 1):
            ax = fig2.add_subplot(2, 3, idx, projection='3d')
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=1.5, alpha=0.8)
            ax.view_init(elev=elev, azim=azim)
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title(title, fontsize=10)
            ax.grid(True, alpha=0.2)
            ax.set_box_aspect([1, 1, 1])
        plt.tight_layout()
        main_views_path = output_dir / "reconstruction_main_views.png"
        plt.savefig(main_views_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {main_views_path}")
        plt.close(fig2)
        
        # Generate 3: High-quality main view
        fig3 = plt.figure(figsize=(18, 14))
        ax3 = fig3.add_subplot(111, projection='3d')
        # Generate 3: High-quality main view
        fig3 = plt.figure(figsize=(24, 18))
        ax3 = fig3.add_subplot(111, projection='3d')
        ax3.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=6, alpha=0.9)
        ax3.view_init(elev=18, azim=50)
        # Hide axes
        ax3.set_axis_off()
        ax3.set_box_aspect([1, 1, 1])
        # Zoom in by setting tighter limits around the point cloud center
        center = points.mean(axis=0)
        range_factor = 0.35  # Show 35% of the full range (more zoom in)
        x_range = (points[:, 0].max() - points[:, 0].min()) * range_factor
        y_range = (points[:, 1].max() - points[:, 1].min()) * range_factor
        z_range = (points[:, 2].max() - points[:, 2].min()) * range_factor
        ax3.set_xlim([center[0] - x_range/2, center[0] + x_range/2])
        ax3.set_ylim([center[1] - y_range/2, center[1] + y_range/2])
        ax3.set_zlim([center[2] - z_range/2, center[2] + z_range/2])
        plt.tight_layout()
        main_hq_path = output_dir / "reconstruction_main_hq.png"
        plt.savefig(main_hq_path, dpi=300, bbox_inches='tight', pad_inches=0, facecolor='white')
        print(f"✓ Saved: {main_hq_path}")
        plt.close(fig3)
        
        # Save camera poses
        camera_path = output_dir / "camera_poses.txt"
        with open(camera_path, 'w') as f:
            f.write("# image_id qw qx qy qz tx ty tz\n")
            for img_id, image in reconstruction.images.items():
                pose = image.cam_from_world
                q = pose.rotation.quat  # [qw, qx, qy, qz]
                t = pose.translation
                f.write(f"{img_id} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]}\n")
        print(f"✓ Saved: {camera_path}")
        
        print("\nVisualization complete!")
        print(f"Generated images:")
        print(f"  - {main_view_path.name} (single main view facing courtyard)")
        print(f"  - {main_views_path.name} (6 different main viewpoints)")
        print(f"  - {main_hq_path.name} (high-quality main view)")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize 3D reconstruction from COLMAP sparse reconstruction"
    )
    parser.add_argument(
        "--sparse_dir",
        type=str,
        required=True,
        help="Path to COLMAP sparse reconstruction directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Optional output directory to save visualizations",
    )
    
    args = parser.parse_args()
    
    visualize_reconstruction(
        sparse_dir=Path(args.sparse_dir),
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )


if __name__ == "__main__":
    main()
