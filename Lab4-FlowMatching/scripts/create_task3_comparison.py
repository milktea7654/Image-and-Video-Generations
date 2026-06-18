"""
創建 Task 3 樣本對比圖
"""

from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

def make_comparison_grid(model_dirs, model_labels, output_path, n_samples=8):
    """創建模型對比圖"""
    fig, axes = plt.subplots(len(model_dirs), n_samples, figsize=(20, 2.5*len(model_dirs)))
    
    for row_idx, (dir_path, label, fid) in enumerate(zip(model_dirs, model_labels, model_fids)):
        dir_path = Path(dir_path)
        images = sorted(list(dir_path.glob('sample_*.png')))[:n_samples]
        
        for col_idx, img_path in enumerate(images):
            img = Image.open(img_path)
            if len(model_dirs) == 1:
                axes[col_idx].imshow(img)
                axes[col_idx].axis('off')
            else:
                axes[row_idx, col_idx].imshow(img)
                axes[row_idx, col_idx].axis('off')
        
        if len(model_dirs) > 1:
            axes[row_idx, 0].set_ylabel(f'{label}\nFID={fid:.2f}', 
                                        fontsize=12, rotation=0, 
                                        labelpad=80, va='center',
                                        fontweight='bold')
    
    plt.suptitle('Rectified Flow: Sample Quality Comparison (5 Steps)', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"✅ Saved comparison grid to {output_path}")

# 對比圖配置
model_dirs = [
    'results/eval_task3/original_fm_steps5',
    'results/eval_task3/rf1_steps10',  # RF1 @ 5 steps 目錄可能不存在，用 10 steps
    'results/eval_task3/rf2_steps5'
]

model_labels = [
    'Original FM (5 steps)',
    'RF1 (10 steps)',  # 改為 10 steps
    'RF2 (5 steps)'
]

model_fids = [29.58, 23.32, 2.84]

output_path = 'results/eval_task3/samples_comparison.png'

# 檢查目錄是否存在
existing_dirs = []
existing_labels = []
existing_fids = []

for dir_path, label, fid in zip(model_dirs, model_labels, model_fids):
    if Path(dir_path).exists():
        existing_dirs.append(dir_path)
        existing_labels.append(label)
        existing_fids.append(fid)
    else:
        print(f"⚠️  Directory not found: {dir_path}")

if existing_dirs:
    print(f"\n📸 Creating comparison grid with {len(existing_dirs)} models...")
    make_comparison_grid(existing_dirs, existing_labels, output_path)
else:
    print("❌ No sample directories found!")

# 額外創建 RF1 vs RF2 @ 5 steps 對比（如果都存在）
rf1_5_dir = Path('results/rectified_5steps')
rf2_5_dir = Path('results/eval_task3/rf2_steps5')

if rf1_5_dir.exists() and rf2_5_dir.exists():
    print("\n📸 Creating RF1 vs RF2 comparison (5 steps)...")
    make_comparison_grid(
        [str(rf1_5_dir), str(rf2_5_dir)],
        ['RF1 (5 steps)', 'RF2 (5 steps)'],
        'results/eval_task3/rf1_vs_rf2_5steps.png'
    )

print("\n✅ Done!")
