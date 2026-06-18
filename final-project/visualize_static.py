#!/usr/bin/env python3
"""
Create static visualizations and screenshots of point clouds (no OpenGL required)
"""
import open3d as o3d
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_matplotlib_visualization(sparse_path, dense_path, output_dir):
    """Create matplotlib-based visualizations"""
    
    print("=" * 80)
    print("Creating Static Visualizations (matplotlib)")
    print("=" * 80)
    
    # Load point clouds
    print("\n[1] Loading point clouds...")
    sparse = o3d.io.read_point_cloud(str(sparse_path))
    dense = o3d.io.read_point_cloud(str(dense_path))
    
    sparse_points = np.asarray(sparse.points)
    sparse_colors = np.asarray(sparse.colors)
    
    dense_points = np.asarray(dense.points)
    dense_colors = np.asarray(dense.colors)
    
    print(f"    Sparse: {len(sparse_points):,} points")
    print(f"    Dense: {len(dense_points):,} points")
    
    # Subsample dense for visualization
    print("\n[2] Subsampling dense point cloud for visualization...")
    n_sample = min(50000, len(dense_points))
    indices = np.random.choice(len(dense_points), n_sample, replace=False)
    dense_points_sample = dense_points[indices]
    dense_colors_sample = dense_colors[indices]
    print(f"    Using {n_sample:,} points from dense cloud")
    
    # Create figure with subplots
    print("\n[3] Creating visualizations...")
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Sparse point cloud
    ax1 = fig.add_subplot(231, projection='3d')
    ax1.scatter(sparse_points[::1, 0], sparse_points[::1, 1], sparse_points[::1, 2],
                c=sparse_colors[::1], s=20, alpha=0.8)
    ax1.set_title(f'COLMAP Sparse\n({len(sparse_points):,} points)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # 2. Dense point cloud (subsampled)
    ax2 = fig.add_subplot(232, projection='3d')
    ax2.scatter(dense_points_sample[:, 0], dense_points_sample[:, 1], dense_points_sample[:, 2],
                c=dense_colors_sample, s=1, alpha=0.6)
    ax2.set_title(f'NeRF Dense\n({len(dense_points):,} points, showing {n_sample:,})', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    # 3. Comparison (different angle)
    ax3 = fig.add_subplot(233, projection='3d')
    ax3.scatter(sparse_points[::1, 0], sparse_points[::1, 1], sparse_points[::1, 2],
                c='red', s=30, alpha=1.0, label='Sparse')
    ax3.scatter(dense_points_sample[:, 0], dense_points_sample[:, 1], dense_points_sample[:, 2],
                c=dense_colors_sample, s=1, alpha=0.3, label='Dense')
    ax3.set_title(f'Combined View\n(Red: Sparse, Colors: Dense)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.legend()
    
    # 4. Top view (XY plane)
    ax4 = fig.add_subplot(234)
    ax4.scatter(sparse_points[:, 0], sparse_points[:, 1], c='red', s=10, alpha=0.8, label='Sparse')
    ax4.scatter(dense_points_sample[:, 0], dense_points_sample[:, 1], 
                c=dense_colors_sample, s=0.5, alpha=0.3, label='Dense')
    ax4.set_title('Top View (XY plane)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axis('equal')
    
    # 5. Side view (XZ plane)
    ax5 = fig.add_subplot(235)
    ax5.scatter(sparse_points[:, 0], sparse_points[:, 2], c='red', s=10, alpha=0.8, label='Sparse')
    ax5.scatter(dense_points_sample[:, 0], dense_points_sample[:, 2], 
                c=dense_colors_sample, s=0.5, alpha=0.3, label='Dense')
    ax5.set_title('Side View (XZ plane)', fontsize=12, fontweight='bold')
    ax5.set_xlabel('X')
    ax5.set_ylabel('Z')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.axis('equal')
    
    # 6. Front view (YZ plane)
    ax6 = fig.add_subplot(236)
    ax6.scatter(sparse_points[:, 1], sparse_points[:, 2], c='red', s=10, alpha=0.8, label='Sparse')
    ax6.scatter(dense_points_sample[:, 1], dense_points_sample[:, 2], 
                c=dense_colors_sample, s=0.5, alpha=0.3, label='Dense')
    ax6.set_title('Front View (YZ plane)', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Y')
    ax6.set_ylabel('Z')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.axis('equal')
    
    plt.suptitle('DiffMatch Point Cloud Visualization\nSparse vs Dense Comparison', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / "pointcloud_visualization.png"
    print(f"\n[4] Saving visualization to: {output_path}")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"    ✓ Saved: {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # Create density comparison plot
    print("\n[5] Creating density comparison plot...")
    fig2, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Point count comparison
    ax = axes[0]
    categories = ['COLMAP\nSparse', 'NeRF\nDense']
    counts = [len(sparse_points), len(dense_points)]
    colors_bar = ['#e74c3c', '#3498db']
    bars = ax.bar(categories, counts, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Number of Points', fontsize=12, fontweight='bold')
    ax.set_title('Point Count Comparison', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}\npoints',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Ratio visualization
    ax = axes[1]
    ratio = len(dense_points) / len(sparse_points)
    ax.barh(['Density\nIncrease'], [ratio], color='#2ecc71', alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_xlabel('Multiplier', fontsize=12, fontweight='bold')
    ax.set_title('Dense vs Sparse Ratio', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.text(ratio/2, 0, f'{ratio:.1f}x denser', 
            ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    
    plt.suptitle('Point Cloud Statistics', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    stats_path = output_dir / "pointcloud_statistics.png"
    plt.savefig(stats_path, dpi=150, bbox_inches='tight')
    print(f"    ✓ Saved: {stats_path}")
    
    print("\n" + "=" * 80)
    print("✓ Visualization Complete!")
    print("=" * 80)
    print(f"\nGenerated files:")
    print(f"  1. {output_path.name}")
    print(f"  2. {stats_path.name}")
    print(f"\nYou can view these images with:")
    print(f"  xdg-open {output_path}")
    print(f"  xdg-open {stats_path}")
    print("=" * 80)


if __name__ == "__main__":
    output_dir = Path("ours/output/top_k_noransac")
    
    sparse_path = output_dir / "diffmatch_colmap_sparse.ply"
    dense_path = output_dir / "diffmatch_nerf_dense.ply"
    
    create_matplotlib_visualization(sparse_path, dense_path, output_dir)
