#!/usr/bin/env python3
"""
Run VGGT reconstruction on courtyard dataset.
Generates COLMAP format output for comparison with LightGlue baseline.
"""

import sys
import os
from pathlib import Path
import argparse

# Add VGGT submodule to path
VGGT_PATH = Path(__file__).parent / "models" / "vggt"
sys.path.insert(0, str(VGGT_PATH))


def patch_vggt_for_pycolmap():
    """Check pycolmap version - VGGT works with pycolmap 3.10.0"""
    import pycolmap
    version = pycolmap.__version__
    print(f"Using pycolmap version: {version}")
    if version.startswith("3.10"):
        print("✓ pycolmap 3.10.x detected - compatible with VGGT")
    else:
        print(f"⚠ Warning: VGGT expects pycolmap 3.10.x, but found {version}")
        print("  You may encounter compatibility issues.")


def main():
    parser = argparse.ArgumentParser(description="Run VGGT reconstruction")
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing input images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for reconstruction results",
    )
    parser.add_argument(
        "--use_ba",
        action="store_true",
        default=False,
        help="Use bundle adjustment (slower but more accurate)",
    )
    parser.add_argument(
        "--low_vram",
        action="store_true",
        default=False,
        help="Use low VRAM mode (reduces memory usage)",
    )
    
    args = parser.parse_args()
    
    # Patch VGGT for pycolmap compatibility
    patch_vggt_for_pycolmap()
    
    # Create output directory structure
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create scene directory structure required by VGGT
    scene_dir = output_dir / "scene"
    scene_dir.mkdir(exist_ok=True)
    
    images_dir = scene_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Create symlink to input images
    input_images = Path(args.image_dir)
    if not input_images.exists():
        raise ValueError(f"Input directory does not exist: {input_images}")
    
    # Copy or symlink images
    print(f"Linking images from {input_images} to {images_dir}")
    import shutil
    for img_file in sorted(input_images.glob("*")):
        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            target = images_dir / img_file.name
            if not target.exists():
                # Use hard copy instead of symlink for compatibility
                shutil.copy2(img_file, target)
                print(f"  Copied: {img_file.name}")
    
    # Count images
    num_images = len(list(images_dir.glob("*.jpg"))) + len(list(images_dir.glob("*.png"))) + len(list(images_dir.glob("*.jpeg")))
    print(f"Found {num_images} images")
    
    # Run VGGT demo_colmap.py
    print("\n" + "="*60)
    print("Running VGGT Reconstruction")
    print("="*60)
    
    import subprocess
    
    cmd = [
        "python",
        str(VGGT_PATH / "demo_colmap.py"),
        f"--scene_dir={scene_dir}",
    ]
    
    if args.use_ba:
        print("Using bundle adjustment (this will be slower)")
        cmd.append("--use_ba")
    
    if args.low_vram:
        print("Low VRAM mode: reducing all memory-intensive parameters")
        # Aggressively reduce parameters to minimize VRAM
        if args.use_ba:
            # BA mode needs even more aggressive reduction
            print("  BA mode: using minimal parameters (512 pts, 1 frame)")
            cmd.extend([
                "--max_query_pts=2048",        # Very low for BA
                "--query_frame_num=4",        # Only 1 query frame
            ])
            # Note: Cannot disable fine_tracking as it's default True
        else:
            cmd.extend([
                "--max_query_pts=1024",       # Moderate reduction
                "--query_frame_num=16",        # Use default for better coverage
            ])
    
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, cwd=str(VGGT_PATH))
    
    if result.returncode != 0:
        print(f"\n❌ VGGT reconstruction failed with exit code {result.returncode}")
        return
    
    # Check output
    sparse_dir = scene_dir / "sparse"
    if sparse_dir.exists():
        print("\n" + "="*60)
        print("✓ VGGT Reconstruction Complete")
        print("="*60)
        print(f"Output saved to: {sparse_dir}")
        print(f"  - cameras.bin")
        print(f"  - images.bin")
        print(f"  - points3D.bin")
        
        # Export to PLY and visualize
        print("\n" + "="*60)
        print("Exporting point cloud and creating visualizations")
        print("="*60)
        
        export_cmd = [
            "python",
            str(Path(__file__).parent / "visualize_reconstruction.py"),
            f"--sparse_dir={sparse_dir}",
            f"--output_dir={output_dir}",
        ]
        
        subprocess.run(export_cmd)
        
        print("\n✓ All done!")
    else:
        print(f"\n❌ Output directory not found: {sparse_dir}")


if __name__ == "__main__":
    main()
