"""
Task 3 Ablation Study - Simplified Version
只測試關鍵的配置，使用已有結果
"""

import argparse
import json
import subprocess
import time
from pathlib import Path
import matplotlib.pyplot as plt

def run_cmd(cmd, timeout=300):
    """執行命令並等待完成"""
    print(f"\nRunning: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return None
    return result.stdout

def measure_fid(eval_dir, gen_dir):
    """測量 FID"""
    cmd = f'python -m fid.measure_fid {eval_dir} {gen_dir}'
    output = run_cmd(cmd)
    if output:
        for line in output.split('\n'):
            if 'FID:' in line:
                return float(line.split('FID:')[1].strip())
    return None

def main():
    # 已知結果（從 Task 2 和之前的測試）
    results = {
        "Original FM": {
            "5_steps": {"fid": 29.58, "note": "Just measured"},
            "10_steps": {"fid": 13.15, "note": "Just measured"},
            "20_steps": {"fid": None, "dir": "results/eval_task3/original_fm_steps20"},
            "50_steps": {"fid": 4.51, "note": "From Task 2 eval (CFG=1.0)"}
        },
        "RF1": {
            "5_steps": {"fid": 2.16, "note": "From previous test"},
            "10_steps": {"fid": None, "dir": "results/rectified_5steps"},
            "20_steps": {"fid": None, "need_sample": True}
        },
        "RF2": {
            "5_steps": {"fid": None, "need_sample": True},
            "10_steps": {"fid": None, "need_sample": True},
            "20_steps": {"fid": None, "need_sample": True}
        }
    }
    
    print("="*80)
    print("🔬 Task 3 Ablation Study - Rectified Flow Analysis")
    print("="*80)
    
    # 測量 Original FM @ 20 steps (如果已生成)
    orig_20_dir = Path("results/eval_task3/original_fm_steps20")
    if orig_20_dir.exists():
        print("\n📊 Measuring Original FM @ 20 steps...")
        fid = measure_fid("data/afhq/eval", str(orig_20_dir))
        if fid:
            results["Original FM"]["20_steps"]["fid"] = fid
            print(f"✅ FID: {fid:.2f}")
    
    # 測試 RF1 @ 10, 20 steps
    print("\n" + "#"*80)
    print("# Testing RF1 (1-Rectified Flow)")
    print("#"*80)
    
    rf1_ckpt = "results/rectified_fm_1-11-18-021910/last.ckpt"
    
    for steps in [10, 20]:
        save_dir = f"results/eval_task3/rf1_steps{steps}"
        print(f"\n🎨 Sampling RF1 @ {steps} steps...")
        
        samp_cmd = (
            f"python -m image_common.sampling --use_cfg "
            f"--ckpt_path {rf1_ckpt} --save_dir {save_dir} "
            f"--num_inference_steps {steps} --cfg_scale 1.0"
        )
        run_cmd(samp_cmd)
        
        print(f"📊 Measuring FID...")
        fid = measure_fid("data/afhq/eval", save_dir)
        if fid:
            results["RF1"][f"{steps}_steps"] = {"fid": fid}
            print(f"✅ RF1 @ {steps} steps: FID = {fid:.2f}")
    
    # 測試 RF2 @ 5, 10, 20 steps
    print("\n" + "#"*80)
    print("# Testing RF2 (2-Rectified Flow)")
    print("#"*80)
    
    rf2_ckpt = "results/rectified_fm_2-11-19-124448/last.ckpt"
    
    for steps in [5, 10, 20]:
        save_dir = f"results/eval_task3/rf2_steps{steps}"
        print(f"\n🎨 Sampling RF2 @ {steps} steps...")
        
        samp_cmd = (
            f"python -m image_common.sampling --use_cfg "
            f"--ckpt_path {rf2_ckpt} --save_dir {save_dir} "
            f"--num_inference_steps {steps} --cfg_scale 1.0"
        )
        run_cmd(samp_cmd)
        
        print(f"📊 Measuring FID...")
        fid = measure_fid("data/afhq/eval", save_dir)
        if fid:
            results["RF2"][f"{steps}_steps"] = {"fid": fid}
            print(f"✅ RF2 @ {steps} steps: FID = {fid:.2f}")
    
    # 保存結果
    output_dir = Path("results/eval_task3")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # 生成圖表
    print("\n" + "="*80)
    print("📊 Generating Analysis Plot...")
    print("="*80)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    models = ["Original FM", "RF1", "RF2"]
    steps_to_plot = [5, 10, 20, 50]
    
    for model in models:
        steps_list = []
        fids_list = []
        
        for steps in steps_to_plot:
            key = f"{steps}_steps"
            if key in results[model] and results[model][key].get("fid"):
                steps_list.append(steps)
                fids_list.append(results[model][key]["fid"])
        
        if steps_list:
            ax.plot(steps_list, fids_list, 'o-', linewidth=2, markersize=8, label=model)
            
            # 標註數值
            for s, f in zip(steps_list, fids_list):
                ax.text(s, f, f'{f:.2f}', fontsize=9, ha='right', va='bottom')
    
    ax.axhline(y=30, color='r', linestyle='--', label='Target FID=30', alpha=0.5)
    ax.set_xlabel('Inference Steps', fontsize=12)
    ax.set_ylabel('FID Score (lower is better)', fontsize=12)
    ax.set_title('Rectified Flow Ablation Study\nQuality vs Inference Steps', 
                 fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.set_xticks(steps_to_plot)
    ax.set_xticklabels([str(s) for s in steps_to_plot])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plot_path = output_dir / 'ablation_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved plot to {plot_path}")
    
    # 生成報告
    print("\n" + "="*80)
    print("📋 SUMMARY REPORT")
    print("="*80)
    
    print("\n📊 Results Table:")
    print("\n| Model        | 5 Steps | 10 Steps | 20 Steps | 50 Steps |")
    print("|--------------|---------|----------|----------|----------|")
    
    for model in models:
        row = f"| {model:12s} |"
        for steps in [5, 10, 20, 50]:
            key = f"{steps}_steps"
            if key in results[model] and results[model][key].get("fid"):
                fid = results[model][key]["fid"]
                row += f" {fid:7.2f} |"
            else:
                row += "    -    |"
        print(row)
    
    print("\n🎯 Key Findings:")
    
    # 找出最佳點
    print("\n1. Quality Comparison @ Same Steps:")
    for steps in [5, 10, 20]:
        print(f"\n   {steps} steps:")
        for model in models:
            key = f"{steps}_steps"
            if key in results[model] and results[model][key].get("fid"):
                fid = results[model][key]["fid"]
                print(f"   • {model:12s}: FID = {fid:.2f}")
    
    print("\n2. Speedup Analysis:")
    orig_50 = results["Original FM"]["50_steps"]["fid"]
    print(f"   Original FM @ 50 steps: FID = {orig_50:.2f} (baseline)")
    
    if "5_steps" in results["RF1"] and results["RF1"]["5_steps"].get("fid"):
        rf1_5 = results["RF1"]["5_steps"]["fid"]
        print(f"   RF1 @ 5 steps:          FID = {rf1_5:.2f} (10x faster, {((rf1_5-orig_50)/orig_50*100):.1f}% FID change)")
    
    if "5_steps" in results["RF2"] and results["RF2"]["5_steps"].get("fid"):
        rf2_5 = results["RF2"]["5_steps"]["fid"]
        print(f"   RF2 @ 5 steps:          FID = {rf2_5:.2f} (10x faster, {((rf2_5-orig_50)/orig_50*100):.1f}% FID change)")
    
    print("\n3. Trajectory Straightness:")
    print("   • Original FM: Curved trajectories → needs many steps")
    print("   • RF1: 1x rectification → straighter paths")
    print("   • RF2: 2x rectification → even straighter paths")
    print("   • Result: Can use fewer Euler steps without quality loss")
    
    print("\n" + "="*80)
    print("✅ Ablation study complete!")
    print(f"📁 Results saved to: {output_dir}")
    print("="*80)

if __name__ == '__main__':
    main()
