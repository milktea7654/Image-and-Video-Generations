"""
Task 3 Ablation Study: Rectified Flow Analysis

Compare:
1. Original FM vs RF1 vs RF2 (trajectory straightness)
2. Different inference steps (1, 5, 10, 20, 50)
3. Quality vs Speed tradeoff
"""

import argparse
import json
import subprocess
import time
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def run_cmd(cmd):
    """執行命令並等待完成"""
    print(f"\n{'='*60}")
    print(f"Running: {cmd}")
    print('='*60)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return None
    return result.stdout

def sample_and_measure(ckpt_path, steps, cfg_scale, save_dir, model_name):
    """生成樣本並測量 FID"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 採樣
    print(f"\n🎨 Sampling {model_name} with {steps} steps...")
    samp_cmd = [
        'python -m image_common.sampling',
        '--use_cfg',
        f'--ckpt_path {ckpt_path}',
        f'--save_dir {save_dir}',
        f'--num_inference_steps {steps}',
        f'--cfg_scale {cfg_scale}'
    ]
    
    start_time = time.time()
    run_cmd(' '.join(samp_cmd))
    sampling_time = time.time() - start_time
    
    # 測量 FID
    print(f"\n📊 Measuring FID for {model_name}...")
    fid_cmd = f'python -m fid.measure_fid data/afhq/eval {save_dir}'
    fid_output = run_cmd(fid_cmd)
    
    if fid_output:
        for line in fid_output.split('\n'):
            if 'FID:' in line:
                fid = float(line.split('FID:')[1].strip())
                return {
                    'model': model_name,
                    'steps': steps,
                    'fid': fid,
                    'time': sampling_time,
                    'samples_per_sec': 500 / sampling_time,
                    'save_dir': str(save_dir)
                }
    
    return None

def make_comparison_grid(dirs, labels, output_path):
    """創建對比圖"""
    fig, axes = plt.subplots(len(dirs), 8, figsize=(20, 2.5*len(dirs)))
    
    for row_idx, (dir_path, label) in enumerate(zip(dirs, labels)):
        dir_path = Path(dir_path)
        images = sorted(list(dir_path.glob('*.png')))[:8]
        
        for col_idx, img_path in enumerate(images):
            img = Image.open(img_path)
            if len(dirs) == 1:
                axes[col_idx].imshow(img)
                axes[col_idx].axis('off')
            else:
                axes[row_idx, col_idx].imshow(img)
                axes[row_idx, col_idx].axis('off')
        
        if len(dirs) > 1:
            axes[row_idx, 0].set_ylabel(label, fontsize=12, rotation=0, 
                                        labelpad=60, va='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved comparison grid to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_ckpt', type=str, 
                       default='results/cfg_fm-11-17-125702/last.ckpt',
                       help='Original Flow Matching checkpoint')
    parser.add_argument('--rf1_ckpt', type=str,
                       default='results/rectified_fm_1-11-18-021910/last.ckpt',
                       help='Rectified Flow 1 checkpoint')
    parser.add_argument('--rf2_ckpt', type=str,
                       default='results/rectified_fm_2-11-19-124448/last.ckpt',
                       help='Rectified Flow 2 checkpoint')
    parser.add_argument('--output_dir', type=str, default='results/eval_task3')
    parser.add_argument('--cfg_scale', type=float, default=1.0,
                       help='CFG scale (use 1.0 for best quality based on Task 2)')
    parser.add_argument('--steps_list', type=str, default='1,5,10,20,50',
                       help='Comma-separated list of inference steps to test')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    steps_list = [int(s) for s in args.steps_list.split(',')]
    
    # 定義要測試的模型
    models = [
        ('Original FM', args.original_ckpt),
        ('RF1 (1-Rectified)', args.rf1_ckpt),
        ('RF2 (2-Rectified)', args.rf2_ckpt),
    ]
    
    results = {}
    
    print("\n" + "="*80)
    print("🔬 Task 3 Ablation Study: Rectified Flow Analysis")
    print("="*80)
    
    # 對每個模型測試不同步數
    for model_name, ckpt_path in models:
        print(f"\n\n{'#'*80}")
        print(f"# Testing: {model_name}")
        print(f"{'#'*80}")
        
        model_results = {}
        
        for steps in steps_list:
            save_dir = output_dir / f"{model_name.replace(' ', '_').lower()}_steps{steps}"
            
            result = sample_and_measure(
                ckpt_path=ckpt_path,
                steps=steps,
                cfg_scale=args.cfg_scale,
                save_dir=save_dir,
                model_name=f"{model_name} ({steps} steps)"
            )
            
            if result:
                model_results[f"{steps}_steps"] = result
                print(f"\n✅ {model_name} @ {steps} steps:")
                print(f"   FID: {result['fid']:.2f}")
                print(f"   Time: {result['time']:.2f}s")
                print(f"   Speed: {result['samples_per_sec']:.2f} samples/sec")
        
        results[model_name] = model_results
    
    # 保存結果
    results_file = output_dir / 'ablation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved results to {results_file}")
    
    # 生成對比圖表
    print("\n\n" + "="*80)
    print("📊 Generating Analysis Plots...")
    print("="*80)
    
    # Plot 1: FID vs Steps for all models
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Subplot 1: FID vs Steps
    for model_name in ['Original FM', 'RF1 (1-Rectified)', 'RF2 (2-Rectified)']:
        model_data = results[model_name]
        steps = []
        fids = []
        for key in sorted(model_data.keys(), key=lambda x: int(x.split('_')[0])):
            steps.append(model_data[key]['steps'])
            fids.append(model_data[key]['fid'])
        axes[0].plot(steps, fids, 'o-', linewidth=2, markersize=8, label=model_name)
    
    axes[0].axhline(y=30, color='r', linestyle='--', label='Target FID=30', alpha=0.5)
    axes[0].set_xlabel('Inference Steps', fontsize=12)
    axes[0].set_ylabel('FID Score (lower is better)', fontsize=12)
    axes[0].set_title('FID vs Inference Steps\n(CFG Scale = 1.0)', fontsize=13, fontweight='bold')
    axes[0].set_xscale('log')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Subplot 2: Speedup Analysis
    original_time_50 = results['Original FM']['50_steps']['time']
    
    for model_name in ['Original FM', 'RF1 (1-Rectified)', 'RF2 (2-Rectified)']:
        model_data = results[model_name]
        steps = []
        speedups = []
        for key in sorted(model_data.keys(), key=lambda x: int(x.split('_')[0])):
            steps.append(model_data[key]['steps'])
            speedup = original_time_50 / model_data[key]['time']
            speedups.append(speedup)
        axes[1].plot(steps, speedups, 'o-', linewidth=2, markersize=8, label=model_name)
    
    axes[1].set_xlabel('Inference Steps', fontsize=12)
    axes[1].set_ylabel('Speedup vs Original@50steps', fontsize=12)
    axes[1].set_title('Speed Improvement\n(Higher is better)', fontsize=13, fontweight='bold')
    axes[1].set_xscale('log')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Subplot 3: Quality-Speed Tradeoff
    for model_name in ['Original FM', 'RF1 (1-Rectified)', 'RF2 (2-Rectified)']:
        model_data = results[model_name]
        speedups = []
        fids = []
        for key in sorted(model_data.keys(), key=lambda x: int(x.split('_')[0])):
            speedup = original_time_50 / model_data[key]['time']
            speedups.append(speedup)
            fids.append(model_data[key]['fid'])
        axes[2].plot(speedups, fids, 'o-', linewidth=2, markersize=8, label=model_name)
        
        # 標註步數
        for i, key in enumerate(sorted(model_data.keys(), key=lambda x: int(x.split('_')[0]))):
            steps = model_data[key]['steps']
            axes[2].text(speedups[i], fids[i], f'{steps}', fontsize=8, ha='right')
    
    axes[2].axhline(y=30, color='r', linestyle='--', label='Target FID=30', alpha=0.5)
    axes[2].set_xlabel('Speedup (x faster)', fontsize=12)
    axes[2].set_ylabel('FID Score', fontsize=12)
    axes[2].set_title('Quality-Speed Tradeoff\n(Top-right is best)', fontsize=13, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    axes[2].invert_yaxis()  # Lower FID is better, so invert
    
    plt.tight_layout()
    plot_path = output_dir / 'ablation_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved analysis plot to {plot_path}")
    
    # 生成樣本對比圖（5 steps）
    print("\n📸 Creating sample comparison grid (5 steps)...")
    comparison_dirs = []
    comparison_labels = []
    
    for model_name in ['Original FM', 'RF1 (1-Rectified)', 'RF2 (2-Rectified)']:
        dir_name = f"{model_name.replace(' ', '_').lower()}_steps5"
        dir_path = output_dir / dir_name
        if dir_path.exists():
            comparison_dirs.append(dir_path)
            fid = results[model_name]['5_steps']['fid']
            comparison_labels.append(f"{model_name}\n(FID={fid:.2f})")
    
    if comparison_dirs:
        grid_path = output_dir / 'samples_comparison_5steps.png'
        make_comparison_grid(comparison_dirs, comparison_labels, grid_path)
    
    # 生成文字報告
    print("\n\n" + "="*80)
    print("📋 ABLATION STUDY SUMMARY")
    print("="*80)
    
    print("\n🎯 Key Findings:")
    
    # 找出最佳配置
    best_quality = None
    best_speed = None
    best_tradeoff = None
    
    for model_name in ['Original FM', 'RF1 (1-Rectified)', 'RF2 (2-Rectified)']:
        for key, data in results[model_name].items():
            if best_quality is None or data['fid'] < best_quality['fid']:
                best_quality = data
            if best_speed is None or data['samples_per_sec'] > best_speed['samples_per_sec']:
                best_speed = data
            
            # Tradeoff score: 低 FID + 高速度
            score = data['samples_per_sec'] / (data['fid'] + 1)
            if best_tradeoff is None or score > (best_tradeoff['samples_per_sec'] / (best_tradeoff['fid'] + 1)):
                best_tradeoff = data
    
    print(f"\n1. Best Quality: {best_quality['model']}")
    print(f"   - FID: {best_quality['fid']:.2f}")
    print(f"   - Time: {best_quality['time']:.2f}s")
    
    print(f"\n2. Best Speed: {best_speed['model']}")
    print(f"   - Speed: {best_speed['samples_per_sec']:.2f} samples/sec")
    print(f"   - FID: {best_speed['fid']:.2f}")
    
    print(f"\n3. Best Quality-Speed Tradeoff: {best_tradeoff['model']}")
    print(f"   - FID: {best_tradeoff['fid']:.2f}")
    print(f"   - Speed: {best_tradeoff['samples_per_sec']:.2f} samples/sec")
    print(f"   - Speedup: {original_time_50 / best_tradeoff['time']:.2f}x")
    
    # 對比表格
    print("\n\n📊 Detailed Results Table:")
    print("\n| Model | Steps | FID | Time (s) | Speedup | Samples/sec |")
    print("|-------|-------|-----|----------|---------|-------------|")
    
    for model_name in ['Original FM', 'RF1 (1-Rectified)', 'RF2 (2-Rectified)']:
        for key in sorted(results[model_name].keys(), key=lambda x: int(x.split('_')[0])):
            data = results[model_name][key]
            speedup = original_time_50 / data['time']
            print(f"| {model_name:20s} | {data['steps']:5d} | {data['fid']:5.2f} | "
                  f"{data['time']:8.2f} | {speedup:7.2f}x | {data['samples_per_sec']:11.2f} |")
    
    print("\n" + "="*80)
    print("✅ Ablation study complete!")
    print(f"📁 Results saved to: {output_dir}")
    print("="*80)

if __name__ == '__main__':
    main()
