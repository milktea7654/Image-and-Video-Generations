#!/usr/bin/env python3
"""
Visualize COLMAP sparse and NeRF dense point clouds
"""
import open3d as o3d
import numpy as np
from pathlib import Path
import sys

def visualize_pointclouds(sparse_path, dense_path):
    """Visualize both sparse and dense point clouds"""
    
    print("=" * 80)
    print("Loading Point Clouds")
    print("=" * 80)
    
    # Load sparse point cloud
    print(f"\n[1] Loading sparse: {sparse_path}")
    sparse = o3d.io.read_point_cloud(str(sparse_path))
    print(f"    Points: {len(sparse.points):,}")
    
    # Load dense point cloud
    print(f"\n[2] Loading dense: {dense_path}")
    dense = o3d.io.read_point_cloud(str(dense_path))
    print(f"    Points: {len(dense.points):,}")
    
    # Color sparse point cloud in red for distinction
    sparse.paint_uniform_color([1, 0, 0])  # Red
    
    print("\n" + "=" * 80)
    print("Visualization Options")
    print("=" * 80)
    print("\n[1] Sparse only (COLMAP - red)")
    print("[2] Dense only (NeRF - original colors)")
    print("[3] Both together")
    print("[4] Side-by-side comparison")
    print("[Q] Quit")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == "1":
        print("\n🔴 Visualizing COLMAP Sparse Point Cloud (red)...")
        o3d.visualization.draw_geometries(
            [sparse],
            window_name="COLMAP Sparse Point Cloud (6K points)",
            width=1200,
            height=800,
            left=50,
            top=50
        )
    
    elif choice == "2":
        print("\n🌈 Visualizing NeRF Dense Point Cloud (original colors)...")
        o3d.visualization.draw_geometries(
            [dense],
            window_name="NeRF Dense Point Cloud (970K points)",
            width=1200,
            height=800,
            left=50,
            top=50
        )
    
    elif choice == "3":
        print("\n🔴+🌈 Visualizing both together...")
        print("    Red: COLMAP sparse (6K)")
        print("    Colors: NeRF dense (970K)")
        o3d.visualization.draw_geometries(
            [sparse, dense],
            window_name="Combined: Sparse (red) + Dense (colors)",
            width=1400,
            height=900,
            left=50,
            top=50
        )
    
    elif choice == "4":
        print("\n📊 Side-by-side comparison...")
        
        # Show sparse first
        print("\n[Left] COLMAP Sparse (press Q to continue)")
        vis1 = o3d.visualization.Visualizer()
        vis1.create_window(window_name="COLMAP Sparse", width=800, height=600, left=50, top=50)
        vis1.add_geometry(sparse)
        vis1.run()
        vis1.destroy_window()
        
        # Show dense second
        print("\n[Right] NeRF Dense (press Q to close)")
        vis2 = o3d.visualization.Visualizer()
        vis2.create_window(window_name="NeRF Dense", width=800, height=600, left=900, top=50)
        vis2.add_geometry(dense)
        vis2.run()
        vis2.destroy_window()
    
    else:
        print("\nExiting...")
        return
    
    print("\n✓ Visualization complete!")


if __name__ == "__main__":
    output_dir = Path("ours/output/top_k_noransac")
    
    sparse_path = output_dir / "diffmatch_colmap_sparse.ply"
    dense_path = output_dir / "diffmatch_nerf_dense.ply"
    
    if not sparse_path.exists():
        print(f"❌ Sparse point cloud not found: {sparse_path}")
        sys.exit(1)
    
    if not dense_path.exists():
        print(f"❌ Dense point cloud not found: {dense_path}")
        sys.exit(1)
    
    visualize_pointclouds(sparse_path, dense_path)
